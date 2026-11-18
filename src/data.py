"""Raw data discovery, loading, cleaning and projection.

File formats (from the dataset description):
  .track : lon lat timestamp(s)          (whitespace separated, no header)
  .nodes : lon lat                       (row index = node id, 0-based)
  .arcs  : from_node to_node             (row index = arc id, 0-based)
  .route : arc_id per line, chronological (reference route)
"""
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from pyproj import Transformer

from . import config as C


# --------------------------------------------------------------------------- discovery
def _infer_partition(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    for p in reversed(parts):
        if "eval" in p:
            return C.EVALUATOR
        if "student" in p or p in ("test", "testing", "student_test", "dev"):
            return C.STUDENT
        if "train" in p:
            return C.TRAIN
    return "unknown"


def _sibling(track_path: Path, ext: str) -> Optional[Path]:
    """Find the .nodes/.arcs/.route file belonging to a .track file."""
    same_stem = track_path.with_suffix(ext)
    if same_stem.exists():
        return same_stem
    files = sorted(track_path.parent.glob(f"*{ext}"))
    if len(files) == 1:                     # one-trip-per-folder layout
        return files[0]
    # shared prefix match e.g. trip_01_gps.track / trip_01.nodes
    stem = track_path.stem
    for f in files:
        if stem.startswith(f.stem) or f.stem.startswith(stem):
            return f
    return None


def discover_trips(root: Path = None) -> Dict[str, dict]:
    """Return {trip_id: {track, nodes, arcs, route, partition}}.

    Partition comes from DATA_ROOT/splits.json if present
    ({"train": [...], "student_test": [...], "evaluator_test": [...]}),
    otherwise it is inferred from folder names.
    """
    root = Path(root or C.DATA_ROOT)
    splits_file = root / "splits.json"
    override = {}
    if splits_file.exists():
        for part, ids in json.loads(splits_file.read_text()).items():
            for i in ids:
                override[str(i)] = part

    trips = {}
    for tp in sorted(root.rglob("*.track")):
        # trip id: folder name when one trip per folder, else file stem
        one_per_folder = len(list(tp.parent.glob("*.track"))) == 1 and tp.parent != root
        tid = tp.parent.name if one_per_folder else tp.stem
        if tid in trips:  # disambiguate duplicate names across partitions
            tid = f"{tp.parent.name}/{tid}"
        trips[tid] = {
            "track": tp,
            "nodes": _sibling(tp, ".nodes"),
            "arcs": _sibling(tp, ".arcs"),
            "route": _sibling(tp, ".route"),
            "partition": override.get(tid, _infer_partition(tp.relative_to(root))),
        }
    return trips


def trips_in(partition: str, trips: Dict[str, dict] = None):
    trips = trips or discover_trips()
    return sorted(k for k, v in trips.items() if v["partition"] == partition)


# --------------------------------------------------------------------------- geometry helpers
def utm_transformer(lon: float, lat: float) -> Transformer:
    zone = int((lon + 180) // 6) + 1
    epsg = (32600 if lat >= 0 else 32700) + zone
    return Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)


def point_seg_dist(P, A, B):
    """Distance from points P to segments AB (broadcastable, last dim = 2).
    Returns (distance, t) where t in [0,1] is the projection position along AB."""
    AB = B - A
    L2 = (AB ** 2).sum(-1)
    L2 = np.where(L2 == 0, 1e-12, L2)
    t = np.clip(((P - A) * AB).sum(-1) / L2, 0.0, 1.0)
    proj = A + t[..., None] * AB
    return np.linalg.norm(P - proj, axis=-1), t


def bearing_deg(dx, dy):
    """Compass bearing (0 = north, clockwise) from projected deltas."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def angle_diff(a, b):
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)


# --------------------------------------------------------------------------- Trip object
@dataclass
class Trip:
    trip_id: str
    partition: str
    lonlat: np.ndarray          # (n,2) GPS lon,lat (cleaned)
    t: np.ndarray               # (n,) seconds
    node_lonlat: np.ndarray     # (N,2)
    arcs: np.ndarray            # (E,2) from,to node ids
    route: Optional[np.ndarray] = None
    clean_stats: dict = field(default_factory=dict)

    def __post_init__(self):
        c = self.node_lonlat.mean(0) if len(self.node_lonlat) else self.lonlat.mean(0)
        self.tf = utm_transformer(*c)
        self.xy = np.column_stack(self.tf.transform(self.lonlat[:, 0], self.lonlat[:, 1]))
        self.node_xy = np.column_stack(self.tf.transform(self.node_lonlat[:, 0], self.node_lonlat[:, 1]))
        self.A = self.node_xy[self.arcs[:, 0]]
        self.B = self.node_xy[self.arcs[:, 1]]
        d = self.B - self.A
        self.arc_len = np.linalg.norm(d, axis=1)
        self.arc_heading = bearing_deg(d[:, 0], d[:, 1])
        self._gps_kinematics()

    def _gps_kinematics(self):
        n = len(self.xy)
        prev = np.clip(np.arange(n) - 1, 0, n - 1)
        nxt = np.clip(np.arange(n) + 1, 0, n - 1)
        d = self.xy[nxt] - self.xy[prev]
        dist = np.linalg.norm(d, axis=1)
        dt = np.maximum(self.t[nxt] - self.t[prev], 1e-6)
        self.speed = np.where(nxt != prev, dist / dt, 0.0)
        head = bearing_deg(d[:, 0], d[:, 1])
        valid = dist >= C.MIN_MOVE_FOR_HEADING
        # carry the last valid heading forward when the vehicle is (nearly) stationary
        last = np.nan
        out = np.empty(n)
        for i in range(n):
            if valid[i]:
                last = head[i]
            out[i] = last
        self.heading = out
        self.heading_valid = ~np.isnan(out)
        self.dt_prev = np.concatenate([[0.0], np.diff(self.t)])

    def without_route(self) -> "Trip":
        """Copy with the reference route removed (used by the prediction pipeline)."""
        tr = Trip.__new__(Trip)
        tr.__dict__.update(self.__dict__)
        tr.route = None
        return tr


def _read(path, dtype=float, ncols=None):
    if path is None or not Path(path).exists() or os.path.getsize(path) == 0:
        return np.empty((0, ncols or 1), dtype=dtype)
    arr = np.loadtxt(path, dtype=dtype, ndmin=2)
    return arr


def clean_track(raw: np.ndarray):
    stats = {"n_raw": int(len(raw))}
    ok = np.isfinite(raw).all(1)
    ok &= (np.abs(raw[:, 0]) <= 180) & (np.abs(raw[:, 1]) <= 90)
    ok &= ~((raw[:, 0] == 0) & (raw[:, 1] == 0))
    stats["n_invalid"] = int((~ok).sum())
    raw = raw[ok]
    order = np.argsort(raw[:, 2], kind="stable")
    stats["n_out_of_order"] = int((np.diff(raw[:, 2]) < 0).sum())
    raw = raw[order]
    _, first = np.unique(raw[:, 2], return_index=True)
    stats["n_dup_timestamp"] = int(len(raw) - len(first))
    raw = raw[np.sort(first)]
    stats["n_clean"] = int(len(raw))
    return raw, stats


def load_trip(trip_id: str, info: dict = None, with_route: bool = True) -> Trip:
    info = info or discover_trips()[trip_id]
    raw, stats = clean_track(_read(info["track"], float, 3))
    nodes = _read(info["nodes"], float, 2)[:, :2]
    arcs = _read(info["arcs"], np.int64, 2)[:, :2]
    bad_arc = (arcs < 0).any(1) | (arcs >= len(nodes)).any(1)
    stats["n_bad_arcs"] = int(bad_arc.sum())
    # keep arc ids aligned with row index: point bad arcs at a self-loop on node 0 (length 0)
    arcs = np.where(bad_arc[:, None], 0, arcs)
    route = None
    if with_route and info.get("route") is not None:
        r = _read(info["route"], np.int64, 1).ravel()
        r = r[(r >= 0) & (r < len(arcs))]
        route = r
    return Trip(trip_id, info["partition"], raw[:, :2], raw[:, 2], nodes, arcs, route, stats)


def load_route(trip_id: str, info: dict = None) -> Optional[np.ndarray]:
    """Reference route only (server-side scoring)."""
    info = info or discover_trips()[trip_id]
    if info.get("route") is None:
        return None
    return _read(info["route"], np.int64, 1).ravel()
