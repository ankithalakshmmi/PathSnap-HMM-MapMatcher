"""Per-GPS-point reference labels derived from the .route file.

The .route file is an ordered list of arcs, not one label per GPS point. We assign each
GPS point an index into the route with a monotonic (non-decreasing) constraint, minimising
the total point-to-arc distance (dynamic programming). This respects travel order, so
U-turns and revisited arcs are handled correctly, which plain nearest-arc snapping cannot do.
"""
import numpy as np

from . import config as C
from .data import Trip, point_seg_dist


def assign_point_labels(trip: Trip, chunk: int = 2000):
    """Returns dict with label_arc (n,), label_dist (n,), label_ok (n,) bool, route_idx (n,)."""
    n = len(trip.xy)
    r = trip.route
    if r is None or len(r) == 0 or n == 0:
        return None
    A, B = trip.A[r], trip.B[r]
    m = len(r)
    D = np.empty((n, m), dtype=np.float32)
    for s in range(0, n, chunk):
        P = trip.xy[s:s + chunk, None, :]
        D[s:s + chunk] = point_seg_dist(P, A[None], B[None])[0]

    back = np.empty((n, m), dtype=np.int32)
    idx = np.arange(m)
    cost = D[0].astype(np.float64)
    back[0] = idx
    for i in range(1, n):
        pm = np.minimum.accumulate(cost)
        arg = np.maximum.accumulate(np.where(cost == pm, idx, -1))
        back[i] = arg
        cost = D[i] + pm
    j = int(np.argmin(cost))
    ridx = np.empty(n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        ridx[i] = j
        j = back[i, j]
    label_dist = D[np.arange(n), ridx].astype(float)
    return {
        "route_idx": ridx,
        "label_arc": r[ridx],
        "label_dist": label_dist,
        "label_ok": label_dist <= C.LABEL_MAX_DIST,
    }
