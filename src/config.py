"""Central configuration. Override the data location with the PATHSNAP_DATA env var."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("PATHSNAP_DATA", ROOT / "data"))

ARTIFACTS = ROOT / "artifacts"
DATASETS = ARTIFACTS / "datasets"
MODELS = ARTIFACTS / "models"
RESULTS = ROOT / "results"
PLOTS = RESULTS / "plots"
MAPS = ROOT / "maps"
for _d in (ARTIFACTS, DATASETS, MODELS, RESULTS, PLOTS, MAPS):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42

# Partition names used everywhere in the code
TRAIN, STUDENT, EVALUATOR = "train", "student_test", "evaluator_test"

# Candidate generation (radius is re-selected by build_dataset.py from a sweep on TRAIN only)
CAND_RADII_SWEEP = [20, 30, 50, 75, 100]
CAND_RADIUS_DEFAULT = 50.0       # metres
CAND_TOPK = 10                   # max candidates kept per GPS point
MAX_MISS_RATE = 0.02             # pick smallest radius whose train miss rate is <= this

# Labels: GPS points further than this from their assigned reference arc are flagged unreliable
LABEL_MAX_DIST = 100.0           # metres

# GPS heading is only trusted if the vehicle moved at least this far between i-1 and i+1
MIN_MOVE_FOR_HEADING = 2.0       # metres

N_FOLDS = 5
