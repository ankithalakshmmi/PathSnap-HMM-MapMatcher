"""1) Tune HMM decoder params on TRAIN out-of-fold probabilities (never evaluator data).
2) Compare classifier-only vs classifier+HMM on STUDENT_TEST trips.
3) Generate Folium maps for the student-test trips.

Usage: python -m src.hmm_eval [--tune-trips 20]
"""
import argparse
import itertools
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from .data import discover_trips, load_trip
from .hmm import HMMParams, RoadGraph, viterbi
from .maps import trip_map
from .pipeline import score


def decode_top1(trips, probs, points, params, graphs):
    correct = total = 0
    for tid, trip in trips.items():
        pred = viterbi(trip, probs[probs["trip_id"] == tid], params, graph=graphs[tid])
        p = points[(points["trip_id"] == tid) & points["label_ok"]]
        correct += int((pred[p["pt"].to_numpy()] == p["label_arc"].to_numpy()).sum())
        total += len(p)
    return correct / max(total, 1)


def tune(n_trips):
    info = discover_trips()
    oof = pd.read_parquet(C.DATASETS / "train_oof.parquet")
    points = pd.read_parquet(C.DATASETS / "train_points.parquet")
    rng = np.random.default_rng(C.SEED)
    ids = sorted(oof["trip_id"].unique())
    ids = list(rng.choice(ids, min(n_trips, len(ids)), replace=False))
    trips = {t: load_trip(t, info[t], with_route=False) for t in ids}
    base = HMMParams()
    graphs = {t: RoadGraph(trips[t], base.max_route_m) for t in ids}
    oof = oof[oof["trip_id"].isin(ids)]
    rows = []

    def run(p, stage):
        t0 = time.perf_counter()
        acc = decode_top1(trips, oof, points, p, graphs)
        rows.append({**p.to_dict(), "stage": stage, "train_oof_top1": acc, "seconds": time.perf_counter() - t0})
        print(f"  {stage}: {p.to_dict()} -> top1={acc:.4f}")
        return acc

    # stage 1: emission vs transition balance
    best, best_acc = base, -1
    for beta, w_e in itertools.product([5, 10, 20, 40, 80], [0.5, 1.0, 2.0]):
        p = HMMParams(beta=beta, w_e=w_e, gap_s=base.gap_s, unreach_pen=base.unreach_pen)
        acc = run(p, "beta x w_e")
        if acc > best_acc:
            best, best_acc = p, acc
    # stage 2: gap + disconnection handling
    for gap_s, pen in itertools.product([10, 30, 60], [-5, -15, -30]):
        p = HMMParams(beta=best.beta, w_e=best.w_e, gap_s=gap_s, unreach_pen=pen)
        acc = run(p, "gap x penalty")
        if acc > best_acc:
            best, best_acc = p, acc
    # stage 3: stage-1/2 optimum sat on the grid boundary -> extend the grid beyond it
    for beta, w_e, pen in itertools.product([2, 3, 5, 8], [0.25, 0.5], [-30, -60, -100]):
        p = HMMParams(beta=beta, w_e=w_e, gap_s=best.gap_s, unreach_pen=pen)
        acc = run(p, "refine (extended grid)")
        if acc > best_acc:
            best, best_acc = p, acc
    pd.DataFrame(rows).to_csv(C.RESULTS / "hmm_param_search.csv", index=False)
    (C.MODELS / "hmm_params.json").write_text(json.dumps(best.to_dict(), indent=2))
    print("best HMM params:", best.to_dict(), "train OOF top1:", best_acc)
    return best


def evaluate(params):
    info = discover_trips()
    probs = pd.read_parquet(C.DATASETS / "student_probs.parquet")
    rows = []
    for tid in sorted(probs["trip_id"].unique()):
        trip = load_trip(tid, info[tid])                  # with route, for scoring + map only
        pr = probs[probs["trip_id"] == tid]
        n = len(trip.xy)
        cls = np.full(n, -1, dtype=np.int64)
        b = pr.loc[pr.groupby("pt")["prob"].idxmax()]
        cls[b["pt"].to_numpy()] = b["arc"].to_numpy()
        t0 = time.perf_counter()
        hmm = viterbi(trip.without_route(), pr, params)
        dec_s = time.perf_counter() - t0
        for tag, pred in (("classifier_only", cls), ("classifier_hmm", hmm)):
            rows.append({"trip_id": tid, "method": tag, "n_points": n, **score(trip, pred),
                         "decode_s": dec_s if tag == "classifier_hmm" else 0.0})
        trip_map(trip, {"Classifier only (argmax)": (cls, "#ff7f0e"),
                        "Classifier + HMM/Viterbi": (hmm, "#d62728")}).save(str(C.MAPS / f"student_{tid}.html"))
    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS / "hmm_vs_classifier_per_trip.csv", index=False)
    summ = df.groupby("method")[["top1_accuracy", "continuity", "route_f1", "route_precision",
                                 "route_recall", "sequence_lcs", "decode_s"]].mean()
    summ.to_csv(C.RESULTS / "hmm_vs_classifier_summary.csv")
    print(summ.round(4))

    piv = df.pivot(index="trip_id", columns="method", values="top1_accuracy")
    piv["delta"] = piv["classifier_hmm"] - piv["classifier_only"]
    piv.to_csv(C.RESULTS / "hmm_delta_per_trip.csv")
    metrics = ["top1_accuracy", "continuity", "route_f1", "sequence_lcs"]
    x = np.arange(len(metrics)); w = .38
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w / 2, summ.loc["classifier_only", metrics], w, label="classifier only")
    ax.bar(x + w / 2, summ.loc["classifier_hmm", metrics], w, label="classifier + HMM")
    ax.set_xticks(x, metrics); ax.set_ylim(0, 1.25); ax.legend(loc="upper center", ncol=2); ax.set_title("Student-test: classifier vs HMM")
    plt.tight_layout(); plt.savefig(C.PLOTS / "hmm_vs_classifier.png", dpi=150); plt.close()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(piv.index.astype(str), piv["delta"], color=np.where(piv["delta"] >= 0, "#2ca02c", "#d62728"))
    ax.axhline(0, c="k", lw=.8); ax.set_ylabel("Δ Top-1 (HMM − classifier)"); ax.set_title("Per-trip HMM effect")
    plt.xticks(rotation=45); plt.tight_layout(); plt.savefig(C.PLOTS / "hmm_delta_per_trip.png", dpi=150); plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune-trips", type=int, default=20)
    ap.add_argument("--skip-tune", action="store_true")
    a = ap.parse_args()
    if a.skip_tune:
        hp = C.MODELS / "hmm_params.json"
        params = HMMParams(**json.loads(hp.read_text())) if hp.exists() else HMMParams()
    else:
        params = tune(a.tune_trips)
    evaluate(params)
