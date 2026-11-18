"""Bayesian hyperparameter optimisation (Optuna TPE) for LightGBM.

Objective: mean observation-level Top-1 accuracy over GroupKFold folds on TRAIN trips
(group = trip). Top-1 is the quantity the route matcher actually consumes (best candidate per
GPS point), so it is a better target than pair-level F1 at a fixed 0.5 threshold.

To keep each trial affordable, trials use every HPO_STRIDE-th GPS point (1 Hz points are
near-duplicates of their neighbours). The final refit uses ALL training points.

Also produces out-of-fold (OOF) probabilities on TRAIN with the tuned model; the HMM decoder
parameters are tuned on these OOF probabilities (never on evaluator data).

Usage: python -m src.tune --trials 30
"""
import argparse
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold

from . import config as C
from .candidates import FEATURES
from .metrics import argmax_predictions, top1_accuracy
from .train_models import eval_split, load

HPO_STRIDE = 3
HPO_FOLDS = 3
UNTUNED = dict(n_estimators=400, learning_rate=0.05, num_leaves=31)


def make(params):
    return LGBMClassifier(random_state=C.SEED, verbose=-1, n_jobs=-1, **params)


def cv_top1(params, pairs, points, folds):
    lab = pairs[pairs["label_ok"]]
    scores = []
    for tr_trips, va_trips in folds:
        tr = lab[lab["trip_id"].isin(tr_trips)]
        m = make(params).fit(tr[FEATURES].to_numpy(), tr["y"].to_numpy())
        va = pairs[pairs["trip_id"].isin(va_trips)]
        va = va.assign(prob=m.predict_proba(va[FEATURES].to_numpy())[:, 1])
        scores.append(top1_accuracy(argmax_predictions(va), points[points["trip_id"].isin(va_trips)]))
    return float(np.mean(scores)), float(np.std(scores))


def main(n_trials):
    tr_pairs, tr_points = load(C.TRAIN)
    st_pairs, st_points = load(C.STUDENT)
    sub_pairs = tr_pairs[tr_pairs["pt"] % HPO_STRIDE == 0]
    sub_points = tr_points[tr_points["pt"] % HPO_STRIDE == 0]
    trips = np.array(sorted(tr_pairs["trip_id"].unique()))
    folds = [(trips[a], trips[b]) for a, b in GroupKFold(HPO_FOLDS).split(trips, groups=trips)]

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 300, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        }
        t0 = time.perf_counter()
        mean, std = cv_top1(params, sub_pairs, sub_points, folds)
        trial.set_user_attr("cv_top1_std", std)
        trial.set_user_attr("seconds", time.perf_counter() - t0)
        return mean

    t0 = time.perf_counter()
    base_mean, base_std = cv_top1(UNTUNED, sub_pairs, sub_points, folds)
    print(f"untuned CV top1 (same protocol) = {base_mean:.4f} ± {base_std:.4f}")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=C.SEED),
                                study_name="lgbm_top1")
    study.enqueue_trial({**UNTUNED, "min_child_samples": 20, "subsample": 1.0, "colsample_bytree": 1.0,
                         "reg_lambda": 1e-3, "reg_alpha": 1e-3})   # trial 0 = (approx.) defaults
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    hpo_s = time.perf_counter() - t0
    trials = study.trials_dataframe()
    trials.to_csv(C.RESULTS / "optuna_trials.csv", index=False)
    best = {**study.best_params, "subsample_freq": 1}
    json.dump({"best_params": best, "best_cv_top1": study.best_value, "n_trials": n_trials,
               "hpo_seconds": hpo_s, "folds": HPO_FOLDS, "point_stride": HPO_STRIDE},
              open(C.RESULTS / "best_params.json", "w"), indent=2)
    print("best:", study.best_value, best)

    vals = trials["value"].to_numpy()
    plt.figure(figsize=(7, 4))
    plt.plot(vals, "o", alpha=.5, label="trial")
    plt.plot(np.maximum.accumulate(vals), "-", label="best so far")
    plt.axhline(base_mean, ls="--", c="gray", label="untuned")
    plt.xlabel("trial"); plt.ylabel("CV Top-1 accuracy"); plt.title("Optuna TPE optimisation history")
    plt.legend(); plt.tight_layout(); plt.savefig(C.PLOTS / "optuna_history.png", dpi=150); plt.close()

    # ---- compare untuned vs tuned with identical protocol (full TRAIN, 5-fold CV + STUDENT)
    lab = tr_pairs[tr_pairs["label_ok"]]
    X, y = lab[FEATURES].to_numpy(), lab["y"].to_numpy()
    rows = []
    oof = None
    for tag, params in (("untuned", UNTUNED), ("tuned", best)):
        f1s, top1s, lls = [], [], []
        oof_parts = []
        for a, b in GroupKFold(C.N_FOLDS).split(trips, groups=trips):
            trt, vat = trips[a], trips[b]
            tr = lab[lab["trip_id"].isin(trt)]
            m = make(params).fit(tr[FEATURES].to_numpy(), tr["y"].to_numpy())
            mm, vp = eval_split(m, tr_pairs[tr_pairs["trip_id"].isin(vat)], tr_points[tr_points["trip_id"].isin(vat)])
            f1s.append(mm["macro_f1"]); top1s.append(mm["top1"]); lls.append(mm["log_loss"])
            oof_parts.append(vp)
        t0 = time.perf_counter()
        model = make(params).fit(X, y)
        fit_s = time.perf_counter() - t0
        sm, sp = eval_split(model, st_pairs, st_points)
        rows.append({"model": f"lgbm_{tag}", "cv_macro_f1": np.mean(f1s), "cv_macro_f1_std": np.std(f1s),
                     "cv_top1": np.mean(top1s), "cv_log_loss": np.mean(lls),
                     "student_top1": sm["top1"], "student_macro_f1": sm["macro_f1"],
                     "student_log_loss": sm["log_loss"], "student_continuity": sm["continuity"],
                     "train_time_s": fit_s, "inference_ms_per_trip": sm["inference_ms_per_trip"]})
        print(rows[-1])
        if tag == "tuned":
            joblib.dump(model, C.MODELS / "final_model.joblib")
            pd.concat(oof_parts).to_parquet(C.DATASETS / "train_oof.parquet")
            sp.to_parquet(C.DATASETS / "student_probs.parquet")
    comp = pd.DataFrame(rows)
    comp.to_csv(C.RESULTS / "tuned_vs_untuned.csv", index=False)
    json.dump({"final_model": "lgbm_tuned", "params": best, "features": FEATURES},
              open(C.MODELS / "final_model_config.json", "w"), indent=2)

    fig, ax = plt.subplots(figsize=(7, 4))
    cols = ["cv_macro_f1", "cv_top1", "student_top1", "student_macro_f1"]
    x = np.arange(len(cols)); w = .38
    ax.bar(x - w / 2, comp.loc[0, cols], w, label="untuned")
    ax.bar(x + w / 2, comp.loc[1, cols], w, label="tuned")
    ax.set_xticks(x, cols, rotation=15); ax.set_ylim(comp[cols].min().min() - 0.05, 1); ax.legend()
    ax.set_title("LightGBM: untuned vs Optuna-tuned"); plt.tight_layout()
    plt.savefig(C.PLOTS / "tuned_vs_untuned.png", dpi=150); plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    main(ap.parse_args().trials)
