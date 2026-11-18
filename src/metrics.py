"""Evaluation metrics: pair-level classification + observation-level + route-level."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score)


def pair_metrics(y, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1_pos": f1_score(y, pred, zero_division=0),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "log_loss": log_loss(y, np.clip(prob, 1e-7, 1 - 1e-7), labels=[0, 1]),
        "confusion": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def argmax_predictions(pairs: pd.DataFrame, score_col="prob") -> pd.DataFrame:
    """Highest-scoring candidate per (trip, point)."""
    idx = pairs.groupby(["trip_id", "pt"])[score_col].idxmax()
    return pairs.loc[idx, ["trip_id", "pt", "arc", "from_node", "to_node"]].rename(columns={"arc": "pred_arc"})


def top1_accuracy(pred_pts: pd.DataFrame, points: pd.DataFrame) -> float:
    """Observation-level Top-1: predicted arc == reference arc, over points with a reliable label.
    Points whose reference arc was not among candidates count as wrong."""
    m = points[points["label_ok"]].merge(pred_pts[["trip_id", "pt", "pred_arc"]], on=["trip_id", "pt"], how="left")
    return float((m["pred_arc"] == m["label_arc"]).mean()) if len(m) else float("nan")


def miss_rate(points: pd.DataFrame) -> float:
    p = points[points["label_ok"]]
    return float((~p["label_in_cands"]).mean()) if len(p) else float("nan")


def continuity(arc_seq, arcs_from, arcs_to):
    """Fraction of consecutive (deduplicated) predicted arcs that are connected in the
    directed road network (to_node of previous == from_node of next)."""
    s = [a for i, a in enumerate(arc_seq) if a >= 0 and (i == 0 or a != arc_seq[i - 1])]
    if len(s) < 2:
        return 1.0
    s = np.asarray(s)
    return float((arcs_to[s[:-1]] == arcs_from[s[1:]]).mean())


def continuity_from_preds(pred_pts: pd.DataFrame) -> float:
    vals = []
    for _, g in pred_pts.sort_values(["trip_id", "pt"]).groupby("trip_id"):
        a = g["pred_arc"].to_numpy()
        keep = np.r_[True, a[1:] != a[:-1]]
        fr, to = g["from_node"].to_numpy()[keep], g["to_node"].to_numpy()[keep]
        vals.append(1.0 if keep.sum() < 2 else float((to[:-1] == fr[1:]).mean()))
    return float(np.mean(vals)) if vals else float("nan")


def route_overlap(pred_arcs, ref_route, arc_len):
    """Length-weighted precision/recall/F1 between predicted arc set and reference arc set."""
    P = set(int(a) for a in pred_arcs if a >= 0)
    R = set(int(a) for a in ref_route)
    if not P or not R:
        return {"route_precision": 0.0, "route_recall": 0.0, "route_f1": 0.0}
    inter = sum(arc_len[a] for a in P & R)
    p = inter / max(sum(arc_len[a] for a in P), 1e-9)
    r = inter / max(sum(arc_len[a] for a in R), 1e-9)
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"route_precision": float(p), "route_recall": float(r), "route_f1": float(f)}
