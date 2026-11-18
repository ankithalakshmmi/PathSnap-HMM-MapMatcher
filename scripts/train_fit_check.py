"""In-sample (TRAIN) metrics for each saved model, to compare with CV scores for over/underfitting.
Usage: python scripts/train_fit_check.py"""
import sys, joblib, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C
from src.train_models import eval_split, load
pairs, points = load(C.TRAIN)
rows = []
for name in ["logreg", "gnb", "knn", "rf", "lgbm", "mlp"]:
    m, _ = eval_split(joblib.load(C.MODELS / f"{name}.joblib"), pairs, points)
    rows.append({"model": name, "train_top1": m["top1"], "train_macro_f1": m["macro_f1"], "train_log_loss": m["log_loss"]})
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(C.RESULTS / "train_fit_check.csv", index=False)
