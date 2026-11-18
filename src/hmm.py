"""HMM + Viterbi route reconstruction.

Hidden states   : at GPS point i, the candidate arcs generated for i (same candidate set the
                  classifier scored, so candidate <-> arc correspondence is preserved).
Emission        : w_e * log P(arc is the reference arc | features) from the classifier.
Transition      : Newson & Krumm style. d_route = network distance from the projection of point
                  i on arc a to the projection of point i+1 on arc b (Dijkstra on the directed arc
                  graph); d_gc = straight-line distance between the two GPS points.
                  log P = -|d_route - d_gc| / beta.
Disconnected    : if b is unreachable from a within `max_route_m`, the transition gets a finite
                  penalty `unreach_pen` (not -inf), so decoding never dies; it just strongly
                  prefers connected paths.
GPS gaps        : if dt > gap_s the chain is "broken": transitions are uniform, i.e. the segment
                  after the gap is decoded independently of the one before it.
Missing cands   : candidate generation always falls back to the nearest arc, so every point has
                  >= 1 state.
"""
from dataclasses import asdict, dataclass

import networkx as nx
import numpy as np
import pandas as pd

from .data import Trip


@dataclass
class HMMParams:
    beta: float = 20.0
    w_e: float = 1.0
    gap_s: float = 30.0
    unreach_pen: float = -15.0
    max_route_m: float = 2000.0

    def to_dict(self):
        return asdict(self)


class RoadGraph:
    def __init__(self, trip: Trip, cutoff: float):
        g = nx.DiGraph()
        for (u, v), w in zip(trip.arcs, trip.arc_len):
            if u != v:
                g.add_edge(int(u), int(v), weight=float(w))
        self.g, self.cutoff, self.cache = g, cutoff, {}

    def dist_from(self, u):
        if u not in self.cache:
            self.cache[u] = (nx.single_source_dijkstra_path_length(self.g, u, cutoff=self.cutoff, weight="weight")
                             if u in self.g else {u: 0.0})
        return self.cache[u]


def viterbi(trip: Trip, cands: pd.DataFrame, params: HMMParams, prob_col="prob", graph: RoadGraph = None):
    """cands: rows (pt, arc, proj_frac, prob) for ONE trip. Returns predicted arc per GPS point (-1 if none)."""
    graph = graph or RoadGraph(trip, params.max_route_m)
    fr, to, L = trip.arcs[:, 0], trip.arcs[:, 1], trip.arc_len
    cands = cands.sort_values(["pt", prob_col], ascending=[True, False])
    groups = [(int(p), g["arc"].to_numpy(), g["proj_frac"].to_numpy(),
               params.w_e * np.log(np.clip(g[prob_col].to_numpy(), 1e-6, 1.0)))
              for p, g in cands.groupby("pt", sort=True)]
    n = len(trip.xy)
    out = np.full(n, -1, dtype=np.int64)
    if not groups:
        return out

    p0, a_prev, t_prev, V = groups[0]
    backs = []
    for p, a_cur, t_cur, e_cur in groups[1:]:
        dt = trip.t[p] - trip.t[p0]
        if dt > params.gap_s:
            trans = np.zeros((len(a_prev), len(a_cur)))
        else:
            d_gc = float(np.linalg.norm(trip.xy[p] - trip.xy[p0]))
            trans = np.full((len(a_prev), len(a_cur)), params.unreach_pen)
            for i, (a, ta) in enumerate(zip(a_prev, t_prev)):
                dist = graph.dist_from(int(to[a]))
                for j, (b, tb) in enumerate(zip(a_cur, t_cur)):
                    if a == b:
                        d_route = abs(tb - ta) * L[a]
                    else:
                        sp = dist.get(int(fr[b]))
                        if sp is None:
                            continue
                        d_route = (1 - ta) * L[a] + sp + tb * L[b]
                    trans[i, j] = max(-abs(d_route - d_gc) / params.beta, params.unreach_pen)
        scores = V[:, None] + trans
        best = scores.argmax(0)
        V = scores[best, np.arange(len(a_cur))] + e_cur
        V -= V.max()
        backs.append((best, a_prev))
        p0, a_prev, t_prev = p, a_cur, t_cur

    # backtrack
    k = int(V.argmax())
    out[groups[-1][0]] = a_prev[k]
    for (best, arcs_prev), (p, *_ ) in zip(reversed(backs), reversed(groups[:-1])):
        k = int(best[k])
        out[p] = arcs_prev[k]
    return out
