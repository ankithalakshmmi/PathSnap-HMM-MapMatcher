"""Train + compare all models.

- 5-fold GroupKFold over the 80 TRAIN trips (groups = trip, so no trip leaks across folds)
- refit on all TRAIN, evaluate on the 10 STUDENT_TEST trips
- writes results/model_comparison.csv, results/cv_folds.csv, plots, and artifacts/models/*.joblib

Usage: python -m src.train_models [--models logreg gnb knn rf lgbm mlp]
"""
import argparse
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GroupKFold

from . import config as C
from .candidates import FEATURES
from .metrics import (argmax_predictions, continuity_from_preds, miss_rate, pair_metrics,
                      top1_accuracy)
from .models import MODEL_ZOO, EpochMLP, fit_model


def load(part):
    pairs = pd.read_parquet(C.DATASETS / f"{part}_pairs.parquet")
    points = pd.read_parquet(C.DATASETS / f"{part}_points.parquet")
    return pairs, points


def eval_split(model, pairs, points):
    t0 = time.perf_counter()
    prob = model.predict_proba(pairs[FEATURES].to_numpy())[:, 1]
    inf = time.perf_counter() - t0
    pairs = pairs.assign(prob=prob)
    lab = pairs[pairs["label_ok"]]
    m = pair_metrics(lab["y"].to_numpy(), lab["prob"].to_numpy())
    preds = argmax_predictions(pairs)
    m["top1"] = top1_accuracy(preds, points)
    m["continuity"] = continuity_from_preds(preds)
    m["inference_s"] = inf
    m["inference_ms_per_trip"] = 1000 * inf / max(pairs["trip_id"].nunique(), 1)
    return m, pairs


def main(names):
    tr_pairs, tr_points = load(C.TRAIN)
    st_pairs, st_points = load(C.STUDENT)
    tr_lab = tr_pairs[tr_pairs["label_ok"]].reset_index(drop=True)
    X, y, groups = tr_lab[FEATURES].to_numpy(), tr_lab["y"].to_numpy(), tr_lab["trip_id"].to_numpy()
    print(f"train pairs={len(X)}  student pairs={len(st_pairs)}  "
          f"student miss rate={miss_rate(st_points):.4f}")

    gkf = GroupKFold(n_splits=C.N_FOLDS)
    fold_rows, summary = [], []
    for name in names:
        family, _, cfg = MODEL_ZOO[name]
        print(f"\n=== {name} ({family}) ===")
        for k, (a, b) in enumerate(gkf.split(X, y, groups)):
            t0 = time.perf_counter()
            mdl = fit_model(name, X[a], y[a])
            fit_s = time.perf_counter() - t0
            val_trips = np.unique(groups[b])
            vp = tr_pairs[tr_pairs["trip_id"].isin(val_trips)]
            vpts = tr_points[tr_points["trip_id"].isin(val_trips)]
            m, _ = eval_split(mdl, vp, vpts)
            fold_rows.append({"model": name, "fold": k, "macro_f1": m["macro_f1"], "top1": m["top1"],
                              "log_loss": m["log_loss"], "fit_s": fit_s})
            print(f"  fold {k}: macroF1={m['macro_f1']:.4f} top1={m['top1']:.4f} logloss={m['log_loss']:.4f} fit={fit_s:.1f}s")

        # final fit on all TRAIN
        t0 = time.perf_counter()
        if name == "mlp":   # hold out ~10% of train trips only to record a validation-loss curve
            rng = np.random.default_rng(C.SEED)
            ut = np.unique(groups)
            val_t = rng.choice(ut, max(1, len(ut) // 10), replace=False)
            vm = np.isin(groups, val_t)
            mdl = EpochMLP().fit(X[~vm], y[~vm], X[vm], y[vm])
            hist = mdl.history_
            json.dump(hist, open(C.RESULTS / "mlp_history.json", "w"))
            plt.figure(figsize=(6, 4))
            plt.plot(hist["train_loss"], label="train (log loss)")
            plt.plot(hist["val_loss"], label="validation (held-out train trips)")
            plt.xlabel("epoch"); plt.ylabel("log loss"); plt.title("MLP training curve"); plt.legend()
            plt.tight_layout(); plt.savefig(C.PLOTS / "mlp_loss_curve.png", dpi=150); plt.close()
        else:
            mdl = fit_model(name, X, y)
        fit_s = time.perf_counter() - t0
        m, _ = eval_split(mdl, st_pairs, st_points)
        joblib.dump(mdl, C.MODELS / f"{name}.joblib")
        f = pd.DataFrame([r for r in fold_rows if r["model"] == name])
        summary.append({
            "model": name, "family": family, "config": cfg,
            "cv_macro_f1_mean": f["macro_f1"].mean(), "cv_macro_f1_std": f["macro_f1"].std(),
            "cv_top1_mean": f["top1"].mean(), "cv_top1_std": f["top1"].std(),
            "cv_log_loss": f["log_loss"].mean(),
            "student_top1": m["top1"], "student_macro_f1": m["macro_f1"],
            "student_precision": m["precision"], "student_recall": m["recall"],
            "student_accuracy": m["accuracy"], "student_log_loss": m["log_loss"],
            "student_continuity": m["continuity"],
            "train_time_s": fit_s, "inference_ms_per_trip": m["inference_ms_per_trip"],
            "confusion": json.dumps(m["confusion"]),
        })
        print(f"  STUDENT: top1={m['top1']:.4f} macroF1={m['macro_f1']:.4f} logloss={m['log_loss']:.4f} "
              f"continuity={m['continuity']:.4f}")
        ConfusionMatrixDisplay(np.array(m["confusion"]), display_labels=["not ref", "ref"]).plot(cmap="Blues")
        plt.title(f"{name} - student-test confusion matrix"); plt.tight_layout()
        plt.savefig(C.PLOTS / f"confusion_{name}.png", dpi=150); plt.close()

        save_results(pd.DataFrame([r for r in fold_rows if r["model"] == name]), pd.DataFrame([summary[-1]]), name)
    plot_comparison(pd.read_csv(C.RESULTS / "model_comparison.csv"))


def save_results(folds, summ, name):
    # merge with previous runs so models can be trained in separate invocations
    for df, fn in ((folds, "cv_folds.csv"), (summ, "model_comparison.csv")):
        p = C.RESULTS / fn
        if p.exists():
            old = pd.read_csv(p)
            df = pd.concat([old[old["model"] != name], df], ignore_index=True)
        df.to_csv(p, index=False)


def plot_comparison(s):
    s = s.sort_values("cv_macro_f1_mean", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(s["model"], s["cv_macro_f1_mean"], yerr=s["cv_macro_f1_std"], capsize=4, color="#4C72B0")
    ax.set_ylabel("CV macro F1 (mean ± std, 5 folds)"); ax.set_title("Cross-validation macro F1 (TRAIN, GroupKFold)")
    ax.set_ylim(max(0, s["cv_macro_f1_mean"].min() - 0.1), 1)
    plt.tight_layout(); plt.savefig(C.PLOTS / "cv_macro_f1.png", dpi=150); plt.close()

    x = np.arange(len(s)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w / 2, s["student_top1"], w, label="Top-1 accuracy")
    ax.bar(x + w / 2, s["student_macro_f1"], w, label="Macro F1")
    ax.set_xticks(x, s["model"]); ax.set_ylim(max(0, min(s["student_top1"].min(), s["student_macro_f1"].min()) - 0.1), 1)
    ax.set_title("Student-test performance"); ax.legend()
    plt.tight_layout(); plt.savefig(C.PLOTS / "student_top1_f1.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(s["model"], s["student_log_loss"], color="#DD8452")
    ax.set_ylabel("log loss (lower is better)"); ax.set_title("Student-test log loss")
    plt.tight_layout(); plt.savefig(C.PLOTS / "log_loss.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(s["inference_ms_per_trip"], s["student_top1"])
    for _, r in s.iterrows():
        ax.annotate(r["model"], (r["inference_ms_per_trip"], r["student_top1"]))
    ax.set_xscale("log"); ax.set_xlabel("inference time per trip (ms, log)"); ax.set_ylabel("student Top-1")
    ax.set_title("Accuracy vs inference cost")
    plt.tight_layout(); plt.savefig(C.PLOTS / "accuracy_vs_time.png", dpi=150); plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODEL_ZOO))
    main(ap.parse_args().models)
