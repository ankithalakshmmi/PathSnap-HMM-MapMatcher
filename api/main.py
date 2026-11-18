"""PathSnap FastAPI backend (Laptop 1).

Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000
Env:  PATHSNAP_DATA        folder containing the trip folders (incl. evaluator trips)
      PATHSNAP_PARTITION   which partition to serve (default: evaluator_test)

Reference routes are loaded ONLY inside the scoring step and are never returned to the client.
"""
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src import config as C
from src.data import discover_trips, load_trip
from src.pipeline import PathSnapModel, arcs_to_lonlat, score

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pathsnap")

PARTITION = os.environ.get("PATHSNAP_PARTITION", C.EVALUATOR)
app = FastAPI(title="PathSnap map-matching API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

TRIPS = discover_trips()
MODEL = PathSnapModel()
log.info("Loaded model, candidate radius=%.0fm, HMM=%s", MODEL.radius, MODEL.hmm.to_dict())
log.info("Serving partition '%s' with %d trips from %s", PARTITION,
         sum(v["partition"] == PARTITION for v in TRIPS.values()), C.DATA_ROOT)


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/health")
def health():
    return {"status": "ok", "partition": PARTITION}


@app.get("/trips")
def trips():
    return {"partition": PARTITION,
            "trip_ids": sorted(k for k, v in TRIPS.items() if v["partition"] == PARTITION)}


@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    tid = req.trip_id
    log.info("REQUEST /predict trip_id=%s from %s", tid, request.client.host if request.client else "?")
    info = TRIPS.get(tid)
    if info is None or info["partition"] != PARTITION:
        log.warning("unknown trip_id=%s", tid)
        raise HTTPException(404, f"trip {tid} not available")

    t0 = time.perf_counter()
    trip = load_trip(tid, info, with_route=False)          # prediction input: NO labels
    out = MODEL.predict(trip)
    pred_s = time.perf_counter() - t0
    log.info("trip %s: %d GPS points, %d candidate pairs, predicted in %.2fs",
             tid, len(trip.xy), len(out["pairs"]), pred_s)

    metrics = {}
    if info.get("route") is not None:                         # server-side scoring only
        ref = load_trip(tid, info, with_route=True)
        metrics = {"classifier_only": score(ref, out["classifier"]), "classifier_hmm": score(ref, out["hmm"])}
        log.info("trip %s: Top-1 classifier=%.4f  HMM=%.4f  route F1 HMM=%.4f", tid,
                 metrics["classifier_only"]["top1_accuracy"], metrics["classifier_hmm"]["top1_accuracy"],
                 metrics["classifier_hmm"]["route_f1"])

    step = max(1, len(trip.lonlat) // 3000)
    return {
        "trip_id": tid,
        "n_gps_points": int(len(trip.xy)),
        "gps": trip.lonlat[::step, ::-1].round(6).tolist(),     # [lat, lon]
        "predicted_route": arcs_to_lonlat(trip, out["hmm"]),
        "classifier_route": arcs_to_lonlat(trip, out["classifier"]),
        "metrics": metrics,
        "timing_s": {"prediction": round(pred_s, 3), "total": round(time.perf_counter() - t0, 3)},
    }
