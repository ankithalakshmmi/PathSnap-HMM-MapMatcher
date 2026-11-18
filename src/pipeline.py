"""End-to-end inference pipeline used by the evaluation scripts and the API.

predict() never sees the reference route: it is given a Trip with route=None.
score() is separate and is the only place reference labels are used.
"""
import json

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .candidates import FEATURES, build_pairs
from .data import Trip
from .hmm import HMMParams, RoadGraph, viterbi
from .labels import assign_point_labels
from .metrics import continuity, route_overlap


def dedup(seq):
    seq = [int(a) for a in seq if a >= 0]
    return [a for i, a in enumerate(seq) if i == 0 or a != seq[i - 1]]


def lcs_ratio(pred, ref):
    """Longest common subsequence of deduplicated arc sequences / reference length."""
    if not pred or not ref:
        return 0.0
    prev = [0] * (len(ref) + 1)
    for a in pred:
        cur = [0]
        for j, b in enumerate(ref):
            cur.append(prev[j] + 1 if a == b else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1] / len(ref)


class PathSnapModel:
    def __init__(self, model=None, radius=None, hmm_params=None):
        self.model = model or joblib.load(C.MODELS / "final_model.joblib")
        cc = json.loads((C.ARTIFACTS / "candidate_config.json").read_text())
        self.radius = radius or cc["radius"]
        hp = C.MODELS / "hmm_params.json"
        self.hmm = hmm_params or (HMMParams(**json.loads(hp.read_text())) if hp.exists() else HMMParams())

    def predict(self, trip: Trip):
        assert trip.route is None, "prediction must not receive reference labels"
        pairs = build_pairs(trip, self.radius)
        pairs["prob"] = self.model.predict_proba(pairs[FEATURES].to_numpy())[:, 1]
        n = len(trip.xy)
        cls = np.full(n, -1, dtype=np.int64)
        best = pairs.loc[pairs.groupby("pt")["prob"].idxmax()]
        cls[best["pt"].to_numpy()] = best["arc"].to_numpy()
        hmm = viterbi(trip, pairs, self.hmm)
        return {"classifier": cls, "hmm": hmm, "pairs": pairs}


def score(trip_with_route: Trip, pred_arcs: np.ndarray) -> dict:
    """Server-side evaluation against the reference route."""
    lab = assign_point_labels(trip_with_route)
    ok = lab["label_ok"]
    fr, to = trip_with_route.arcs[:, 0], trip_with_route.arcs[:, 1]
    d = dedup(pred_arcs)
    out = {
        "top1_accuracy": float((pred_arcs[ok] == lab["label_arc"][ok]).mean()),
        "n_points_scored": int(ok.sum()),
        "continuity": continuity(pred_arcs, fr, to),
        "sequence_lcs": lcs_ratio(d, dedup(trip_with_route.route)),
    }
    out.update(route_overlap(pred_arcs, trip_with_route.route, trip_with_route.arc_len))
    return out


def arcs_to_lonlat(trip: Trip, arc_seq):
    """List of [[lat,lon],[lat,lon]] segments for the deduplicated predicted route."""
    return [[[float(trip.node_lonlat[trip.arcs[a, 0], 1]), float(trip.node_lonlat[trip.arcs[a, 0], 0])],
             [float(trip.node_lonlat[trip.arcs[a, 1], 1]), float(trip.node_lonlat[trip.arcs[a, 1], 0])]]
            for a in dedup(arc_seq)]
