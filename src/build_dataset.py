"""Builds labelled candidate datasets for TRAIN and STUDENT partitions.

Steps: load + clean trips -> monotonic point labels -> candidate radius sweep (TRAIN only)
-> candidate generation + features -> parquet files in artifacts/datasets/.
The evaluator partition is never touched here.

Usage: python -m src.build_dataset
"""
import json
import time

import numpy as np
import pandas as pd

from . import config as C
from .candidates import build_pairs, generate_candidates
from .data import discover_trips, load_trip
from .labels import assign_point_labels


def load_labelled(partition, trips):
    out = []
    for tid in sorted(k for k, v in trips.items() if v["partition"] == partition):
        tr = load_trip(tid, trips[tid])
        lab = assign_point_labels(tr)
        if lab is None:
            print(f"  [skip] {tid}: no reference route")
            continue
        out.append((tr, lab))
    return out


def radius_sweep(train):
    rows = []
    for R in C.CAND_RADII_SWEEP:
        miss, tot, ncand = 0, 0, []
        for tr, lab in train:
            c = generate_candidates(tr, R)
            ok = lab["label_ok"]
            hit = c.assign(l=lab["label_arc"][c["pt"]]).query("arc == l")["pt"].unique()
            has = np.zeros(len(tr.xy), bool)
            has[hit] = True
            miss += int((~has & ok).sum())
            tot += int(ok.sum())
            ncand.append(c.groupby("pt").size().mean())
        rows.append({"radius_m": R, "miss_rate": miss / max(tot, 1), "mean_candidates": float(np.mean(ncand))})
        print(f"  radius {R:>4} m  miss_rate={rows[-1]['miss_rate']:.4f}  mean_cands={rows[-1]['mean_candidates']:.2f}")
    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS / "candidate_radius_sweep.csv", index=False)
    ok = df[df["miss_rate"] <= C.MAX_MISS_RATE]
    return float(ok["radius_m"].min() if len(ok) else df["radius_m"].max())


def build(partition, items, radius):
    pairs, points = [], []
    for tr, lab in items:
        p = build_pairs(tr, radius, lab)
        hit = np.zeros(len(tr.xy), bool)
        hit[p.loc[p["y"] == 1, "pt"].to_numpy()] = True
        points.append(pd.DataFrame({
            "trip_id": tr.trip_id, "pt": np.arange(len(tr.xy)),
            "label_arc": lab["label_arc"], "label_dist": lab["label_dist"],
            "label_ok": lab["label_ok"], "label_in_cands": hit,
        }))
        pairs.append(p)
    pairs = pd.concat(pairs, ignore_index=True)
    points = pd.concat(points, ignore_index=True)
    pairs.to_parquet(C.DATASETS / f"{partition}_pairs.parquet")
    points.to_parquet(C.DATASETS / f"{partition}_points.parquet")
    pl = points[points["label_ok"]]
    print(f"  {partition}: trips={pairs.trip_id.nunique()} points={len(points)} pairs={len(pairs)} "
          f"pos_rate={pairs.y.mean():.3f} unreliable_labels={(~points.label_ok).mean():.4f} "
          f"miss_rate={(~pl.label_in_cands).mean():.4f}")
    return pairs, points


def main():
    t0 = time.time()
    trips = discover_trips()
    print("Loading + labelling TRAIN ...")
    train = load_labelled(C.TRAIN, trips)
    print("Loading + labelling STUDENT_TEST ...")
    student = load_labelled(C.STUDENT, trips)
    print("Candidate radius sweep (TRAIN only):")
    radius = radius_sweep(train)
    print(f"Selected radius = {radius} m")
    (C.ARTIFACTS / "candidate_config.json").write_text(json.dumps({"radius": radius, "topk": C.CAND_TOPK}, indent=2))
    build(C.TRAIN, train, radius)
    build(C.STUDENT, student, radius)
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
