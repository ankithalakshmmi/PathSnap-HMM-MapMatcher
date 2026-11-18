"""Combines result plots into report figures (side-by-side panels + confusion-matrix grid).
Usage: python scripts/report_figures.py"""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C
P = C.PLOTS
def side(a, b, out):
    A, B = Image.open(P / a).convert("RGB"), Image.open(P / b).convert("RGB")
    h = max(A.height, B.height)
    A = A.resize((int(A.width * h / A.height), h)); B = B.resize((int(B.width * h / B.height), h))
    im = Image.new("RGB", (A.width + B.width, h), "white"); im.paste(A, (0, 0)); im.paste(B, (A.width, 0))
    im.save(P / out)
side("cv_macro_f1.png", "student_top1_f1.png", "fig_model_scores.png")
side("log_loss.png", "accuracy_vs_time.png", "fig_logloss_time.png")
side("optuna_history.png", "tuned_vs_untuned.png", "fig_hpo.png")
side("hmm_vs_classifier.png", "hmm_delta_per_trip.png", "fig_hmm.png")
s = pd.read_csv(C.RESULTS / "model_comparison.csv")
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
for ax, (_, r) in zip(axes.ravel(), s.iterrows()):
    cm = np.array(json.loads(r["confusion"]))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["pred: not ref", "pred: ref"]); ax.set_yticks([0, 1], ["true: not ref", "true: ref"])
    ax.set_title(f"{r['model']} (student test)")
plt.suptitle("Confusion matrices, candidate-pair level, threshold 0.5"); plt.tight_layout()
plt.savefig(P / "fig_confusion_grid.png", dpi=120); plt.close()
print("done")
