"""Builds results/report_tables.md from the actual result CSVs (copy tables into the report).

Usage: python -m src.report_tables
"""
import json

import pandas as pd

from . import config as C


def fmt(df):
    return df.to_markdown(index=False, floatfmt=".4f")


def main():
    out = ["# PathSnap: auto-generated result tables\n",
           "_All numbers below are read directly from files in `results/`._\n"]
    p = C.RESULTS / "candidate_radius_sweep.csv"
    if p.exists():
        out += ["## Candidate radius sweep (TRAIN)\n", fmt(pd.read_csv(p)), ""]
    p = C.RESULTS / "model_comparison.csv"
    if p.exists():
        s = pd.read_csv(p)
        t = pd.DataFrame({
            "Model": s["model"], "Algorithm family": s["family"], "Key configuration": s["config"],
            "CV Macro F1": s["cv_macro_f1_mean"].map("{:.4f}".format) + " ± " + s["cv_macro_f1_std"].map("{:.4f}".format),
            "Top-1 Acc (student)": s["student_top1"], "Route continuity (student)": s["student_continuity"],
            "Student Macro F1": s["student_macro_f1"], "Log loss (student)": s["student_log_loss"],
            "Inference ms/trip": s["inference_ms_per_trip"],
        })
        out += ["## 3.1 Overall model comparison\n", fmt(t), "",
                "### Extra metrics\n", fmt(s[["model", "cv_top1_mean", "cv_top1_std", "student_accuracy",
                                             "student_precision", "student_recall", "train_time_s"]]), ""]
    p = C.RESULTS / "cv_folds.csv"
    if p.exists():
        out += ["## Per-fold CV results\n", fmt(pd.read_csv(p)), ""]
    p = C.RESULTS / "best_params.json"
    if p.exists():
        b = json.loads(p.read_text())
        out += ["## 4. Bayesian optimisation\n", f"- trials: {b['n_trials']}, folds: {b['folds']}, "
                f"point stride: {b['point_stride']}, wall time: {b['hpo_seconds']:.0f}s",
                f"- best CV Top-1 (HPO protocol): {b['best_cv_top1']:.4f}",
                "- best params: `" + json.dumps(b["best_params"]) + "`", ""]
    p = C.RESULTS / "tuned_vs_untuned.csv"
    if p.exists():
        out += ["### Untuned vs tuned\n", fmt(pd.read_csv(p)), ""]
    p = C.MODELS / "hmm_params.json"
    if p.exists():
        out += ["## 5. HMM decoder parameters (tuned on TRAIN OOF probabilities)\n", "`" + p.read_text().replace("\n", " ") + "`", ""]
    p = C.RESULTS / "hmm_vs_classifier_summary.csv"
    if p.exists():
        out += ["### Classifier-only vs classifier + HMM (student test, mean over trips)\n", fmt(pd.read_csv(p)), ""]
    p = C.RESULTS / "hmm_vs_classifier_per_trip.csv"
    if p.exists():
        out += ["### Per trip\n", fmt(pd.read_csv(p)), ""]
    (C.RESULTS / "report_tables.md").write_text("\n".join(out))
    print("wrote", C.RESULTS / "report_tables.md")


if __name__ == "__main__":
    main()
