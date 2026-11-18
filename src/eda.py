"""Exploratory data analysis on TRAIN and STUDENT_TEST partitions.
The evaluator partition is only counted (file inventory), never opened.

Usage: python -m src.eda
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from . import config as C
from .data import discover_trips, load_trip
from .labels import assign_point_labels
from .maps import trip_map


def main():
    trips = discover_trips()
    rows = []
    for tid, info in sorted(trips.items()):
        r = {"trip_id": tid, "partition": info["partition"],
             **{f"has_{k}": info[k] is not None for k in ("nodes", "arcs", "route")},
             **{f"{k}_kb": round(os.path.getsize(info[k]) / 1024, 1) if info[k] else None
                for k in ("track", "nodes", "arcs", "route")}}
        if info["partition"] in (C.TRAIN, C.STUDENT):
            tr = load_trip(tid, info)
            dt = np.diff(tr.t)
            g = nx.DiGraph(); g.add_nodes_from(range(len(tr.node_lonlat)))
            g.add_edges_from(map(tuple, tr.arcs))
            wcc = list(nx.weakly_connected_components(g))
            lab = assign_point_labels(tr)
            r.update({
                **tr.clean_stats,
                "duration_s": float(tr.t[-1] - tr.t[0]),
                "dt_eq_1s_frac": float((dt == 1).mean()), "dt_max_s": float(dt.max()),
                "n_gaps_gt_10s": int((dt > 10).sum()),
                "gps_path_km": float(np.linalg.norm(np.diff(tr.xy, axis=0), axis=1).sum() / 1000),
                "mean_speed_mps": float(np.median(tr.speed)),
                "lon_min": tr.lonlat[:, 0].min(), "lon_max": tr.lonlat[:, 0].max(),
                "lat_min": tr.lonlat[:, 1].min(), "lat_max": tr.lonlat[:, 1].max(),
                "n_nodes": len(tr.node_lonlat), "n_arcs": len(tr.arcs),
                "n_isolated_nodes": int(sum(1 for n in g.nodes if g.degree(n) == 0)),
                "n_weak_components": len(wcc), "largest_component_frac": max(map(len, wcc)) / g.number_of_nodes(),
                "n_route_arcs": len(tr.route), "route_km": float(tr.arc_len[tr.route].sum() / 1000),
                "route_disconnected_steps": int((tr.arcs[tr.route[:-1], 1] != tr.arcs[tr.route[1:], 0]).sum()),
                "median_gps_to_route_m": float(np.median(lab["label_dist"])),
                "unreliable_label_frac": float((~lab["label_ok"]).mean()),
            })
        rows.append(r)
    inv = pd.DataFrame(rows)
    inv.to_csv(C.RESULTS / "eda_inventory.csv", index=False)
    d = inv[inv["partition"].isin([C.TRAIN, C.STUDENT])]
    summary = {
        "trips_per_partition": inv["partition"].value_counts().to_dict(),
        "gps_points": d.groupby("partition")["n_clean"].describe().round(1).to_dict(),
        "invalid_rows_total": int(d["n_invalid"].sum()), "dup_timestamps_total": int(d["n_dup_timestamp"].sum()),
        "out_of_order_total": int(d["n_out_of_order"].sum()),
        "frac_steps_1s_mean": float(d["dt_eq_1s_frac"].mean()),
        "trips_with_gap_gt_10s": int((d["n_gaps_gt_10s"] > 0).sum()),
        "max_gap_s": float(d["dt_max_s"].max()),
        "route_disconnected_steps_total": int(d["route_disconnected_steps"].sum()),
        "trips_unreliable_labels_gt_5pct": d.loc[d["unreliable_label_frac"] > .05, "trip_id"].tolist(),
    }
    (C.RESULTS / "eda_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))

    tr_ = d[d.partition == C.TRAIN]
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    ax[0, 0].hist(tr_["n_clean"], bins=30); ax[0, 0].set_title("GPS points per trip (TRAIN)")
    ax[0, 1].hist(tr_["dt_max_s"], bins=30); ax[0, 1].set_yscale("log"); ax[0, 1].set_title("Largest time gap per trip (s)")
    ax[1, 0].scatter(tr_["n_arcs"], tr_["n_route_arcs"], s=12); ax[1, 0].set_xlabel("arcs in network")
    ax[1, 0].set_ylabel("arcs in reference route"); ax[1, 0].set_title("Network vs route size")
    ax[1, 1].hist(tr_["median_gps_to_route_m"], bins=30); ax[1, 1].set_title("Median GPS→reference-arc distance (m)")
    plt.tight_layout(); plt.savefig(C.PLOTS / "eda_overview.png", dpi=150); plt.close()

    # time-gap distribution + GPS noise distribution over all TRAIN points
    dts, dists = [], []
    for tid in tr_["trip_id"]:
        t = load_trip(tid, trips[tid]); dts.append(np.diff(t.t))
        dists.append(assign_point_labels(t)["label_dist"])
    dts, dists = np.concatenate(dts), np.concatenate(dists)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(np.clip(dts, 0, 30), bins=30); ax[0].set_yscale("log"); ax[0].set_title("Sampling interval (s, clipped at 30)")
    ax[1].hist(np.clip(dists, 0, 150), bins=75); ax[1].set_title("GPS distance to reference arc (m, clipped 150)")
    plt.tight_layout(); plt.savefig(C.PLOTS / "eda_sampling_noise.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(7, 7))
    for tid in tr_["trip_id"]:
        t = load_trip(tid, trips[tid]); ax.plot(t.lonlat[::5, 0], t.lonlat[::5, 1], lw=.5)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.set_title("All TRAIN GPS trajectories")
    plt.tight_layout(); plt.savefig(C.PLOTS / "eda_all_trajectories.png", dpi=150); plt.close()

    ex = tr_.sort_values("n_clean").iloc[len(tr_) // 2]["trip_id"]
    trip_map(load_trip(ex, trips[ex]), title=f"TRAIN trip {ex}: GPS vs reference").save(str(C.MAPS / f"eda_train_{ex}.html"))
    for bad in summary["trips_unreliable_labels_gt_5pct"][:2]:
        trip_map(load_trip(bad, trips[bad]), title=f"Anomalous TRAIN trip {bad}").save(str(C.MAPS / f"eda_anomaly_{bad}.html"))


if __name__ == "__main__":
    main()
