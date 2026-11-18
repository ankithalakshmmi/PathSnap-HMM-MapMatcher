"""Candidate road-segment generation + feature engineering for (GPS point, candidate arc) pairs."""
import numpy as np
import pandas as pd
import shapely

from . import config as C
from .data import Trip, angle_diff, point_seg_dist

FEATURES = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio",
    "win_dist_mean", "win_dist_max",
    "heading_diff", "cos_heading", "heading_valid",
    "heur_rank",
    "speed", "dt_prev",
    "arc_len", "proj_frac", "proj_end",
    "n_cands", "conn_prev", "conn_next",
]


def _tree(trip: Trip):
    if not hasattr(trip, "_strtree"):
        geoms = shapely.linestrings(np.stack([trip.A, trip.B], axis=1))
        trip._strtree = shapely.STRtree(geoms)
    return trip._strtree


def generate_candidates(trip: Trip, radius: float, topk: int = C.CAND_TOPK) -> pd.DataFrame:
    tree = _tree(trip)
    pts = shapely.points(trip.xy)
    pi, ai = tree.query(pts, predicate="dwithin", distance=radius)
    fallback = np.zeros(len(pi), bool)
    missing = np.setdiff1d(np.arange(len(trip.xy)), np.unique(pi))
    if len(missing):  # no arc within radius: take the nearest arc(s) instead
        mi, ma = tree.query_nearest(pts[missing], all_matches=True)
        pi = np.concatenate([pi, missing[mi]])
        ai = np.concatenate([ai, ma])
        fallback = np.concatenate([fallback, np.ones(len(mi), bool)])
    d, tt = point_seg_dist(trip.xy[pi], trip.A[ai], trip.B[ai])
    df = pd.DataFrame({"pt": pi, "arc": ai, "dist": d, "proj_frac": tt, "fallback": fallback})
    df = df.sort_values(["pt", "dist", "arc"]).groupby("pt", sort=False).head(topk)
    return df.reset_index(drop=True)


def add_features(trip: Trip, df: pd.DataFrame) -> pd.DataFrame:
    n = len(trip.xy)
    pt, arc = df["pt"].to_numpy(), df["arc"].to_numpy()
    A, B = trip.A[arc], trip.B[arc]

    g = df.groupby("pt")["dist"]
    df["dist_rank"] = g.rank(method="min")
    dmin = g.transform("min")
    df["dist_minus_min"] = df["dist"] - dmin
    df["dist_ratio"] = df["dist"] / (dmin + 1.0)
    df["n_cands"] = g.transform("size")

    # distance of neighbouring GPS points (window +-2) to the same arc: smooths GPS noise
    wd = [df["dist"].to_numpy()]
    for o in (-2, -1, 1, 2):
        P = trip.xy[np.clip(pt + o, 0, n - 1)]
        wd.append(point_seg_dist(P, A, B)[0])
    wd = np.vstack(wd)
    df["win_dist_mean"] = wd.mean(0)
    df["win_dist_max"] = wd.max(0)

    hv = trip.heading_valid[pt]
    hd = angle_diff(np.nan_to_num(trip.heading[pt]), trip.arc_heading[arc])
    df["heading_diff"] = np.where(hv, hd, 90.0)          # 90 = uninformative
    df["cos_heading"] = np.cos(np.radians(df["heading_diff"]))
    df["heading_valid"] = hv.astype(int)

    df["speed"] = trip.speed[pt]
    df["dt_prev"] = trip.dt_prev[pt]
    df["arc_len"] = trip.arc_len[arc]
    df["proj_frac"] = df["proj_frac"]
    df["proj_end"] = np.minimum(df["proj_frac"], 1 - df["proj_frac"]) * df["arc_len"]

    # simple label-free heuristic: distance + heading penalty -> best arc per point
    heur = df["dist"] + 30.0 * df["heading_diff"] / 180.0
    df["heur_rank"] = heur.groupby(df["pt"]).rank(method="min")
    best = np.full(n, -1)
    bi = heur.groupby(df["pt"]).idxmin()
    best[bi.index.to_numpy()] = df.loc[bi.to_numpy(), "arc"].to_numpy()
    fr, to = trip.arcs[:, 0], trip.arcs[:, 1]
    prev_best = best[np.clip(pt - 1, 0, n - 1)]
    next_best = best[np.clip(pt + 1, 0, n - 1)]
    pb_ok, nb_ok = prev_best >= 0, next_best >= 0
    df["conn_prev"] = (pb_ok & ((arc == prev_best) | (fr[arc] == to[np.maximum(prev_best, 0)]))).astype(int)
    df["conn_next"] = (nb_ok & ((arc == next_best) | (to[arc] == fr[np.maximum(next_best, 0)]))).astype(int)

    df["from_node"] = fr[arc]
    df["to_node"] = to[arc]
    return df


def build_pairs(trip: Trip, radius: float, labels: dict = None) -> pd.DataFrame:
    df = add_features(trip, generate_candidates(trip, radius))
    df.insert(0, "trip_id", trip.trip_id)
    if labels is not None:
        df["label_arc"] = labels["label_arc"][df["pt"]]
        df["label_ok"] = labels["label_ok"][df["pt"]]
        df["y"] = (df["arc"] == df["label_arc"]).astype(int)
    return df
