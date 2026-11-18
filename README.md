# PathSnap: Distributed Map-Matching & Route Reconstruction

UE24CS352A Machine Learning hackathon, PES University.

PathSnap matches noisy GPS observations to a road network. A classifier scores each candidate
road segment for each GPS point. An HMM/Viterbi decoder then turns those scores into a connected route.
The system is served by a FastAPI backend (Laptop 1) and a Streamlit UI (Laptop 2).

## Repository layout

```
src/
  config.py          paths, partitions, constants
  data.py            trip discovery, loading, cleaning, UTM projection, geometry helpers
  labels.py          per-GPS-point reference labels from .route (monotonic DP alignment)
  candidates.py      candidate generation (STRtree radius query) + feature engineering
  build_dataset.py   radius sweep + labelled candidate datasets (TRAIN, STUDENT_TEST)
  eda.py             dataset inventory, data-quality checks, plots, Folium maps
  models.py          model zoo (6 models across 6 families)
  train_models.py    GroupKFold CV + student-test evaluation of every model
  tune.py            Optuna (TPE) Bayesian optimisation of LightGBM, tuned vs untuned
  hmm.py             HMM + Viterbi decoder
  hmm_eval.py        decoder parameter search (train OOF), classifier vs HMM comparison, maps
  pipeline.py        end-to-end inference + server-side scoring (used by the API)
  maps.py            Folium map builder
  report_tables.py   builds results/report_tables.md from result CSVs
api/main.py          FastAPI backend (Laptop 1)
ui/app.py            Streamlit evaluator UI (Laptop 2)
artifacts/           candidate config, trained models, HMM params (inference assets)
results/             CSV logs, JSON configs, plots (results/plots/)
maps/                interactive Folium HTML maps
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Point the code at the dataset folder. It should contain `train/`, `student_test/` and, on the
API laptop, the evaluator trips in a folder whose name contains `eval`. Each trip is a folder
with `.track/.nodes/.arcs/.route` files.

```bash
export PATHSNAP_DATA="/path/to/Student folder"        # Windows PowerShell: $env:PATHSNAP_DATA="C:\...\Student folder"
```

If the folder names differ, put a `splits.json` in the data folder:
`{"train": [...], "student_test": [...], "evaluator_test": [...]}`.

## Reproduce everything

```bash
python -m src.build_dataset          # labels, radius sweep, candidate datasets
python -m src.eda                    # EDA tables, plots, Folium maps
python -m src.train_models           # 6 models: 5-fold GroupKFold CV + student test
python -m src.tune --trials 30       # Optuna TPE on LightGBM, tuned vs untuned, final model
python -m src.hmm_eval               # HMM param search (train OOF), classifier vs HMM, maps
python -m src.report_tables          # results/report_tables.md
```

All random seeds are fixed (`SEED = 42` in `src/config.py`).

## Deployment (two laptops, same Wi-Fi)

**Laptop 1 (API)**

```bash
export PATHSNAP_DATA="/path/to/data/with/evaluator/trips"
export PATHSNAP_PARTITION=evaluator_test          # use student_test for rehearsal
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Find Laptop 1's IP with `ipconfig` (Windows) or `ifconfig` / `ip a` (macOS/Linux), and allow
port 8000 through the firewall. Every request is logged in this terminal.

**Laptop 2 (UI)**

```bash
pip install streamlit streamlit-folium folium requests
streamlit run ui/app.py
```

Enter `http://<laptop1-ip>:8000` in the sidebar, pick a trip and click **Run map-matching**.

Endpoints: `GET /health`, `GET /trips`, `POST /predict {"trip_id": "..."}`.

## Leakage safeguards

- The evaluator partition is never read by any training, tuning or EDA script.
- Cross-validation uses `GroupKFold` with trip as the group, so no trip appears in two folds.
- HMM decoder parameters are tuned on out-of-fold probabilities for TRAIN trips.
- The API loads each trip **without** its route for prediction (`load_trip(..., with_route=False)`,
  asserted in `PathSnapModel.predict`). The reference route is loaded separately and only for
  scoring, and it is never returned to the UI.
- Nothing is hardcoded: every response is computed live from the raw trip files.

## Notes

- `artifacts/models/rf.joblib` (~109 MB) is not committed because it exceeds GitHub's 100 MB
  file limit. Regenerate it with `python -m src.train_models --models rf`.
- The final deployed model is `artifacts/models/final_model.joblib` (Optuna-tuned LightGBM).
  Its configuration is in `final_model_config.json` and the decoder settings in `hmm_params.json`.
- Generated datasets (`artifacts/datasets/`) are not committed. `python -m src.build_dataset`
  recreates them.
