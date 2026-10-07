# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Machine-learning system that predicts NFL game winners and point spreads. It was run live through the 2025 season (Weeks 1–22, one notebook per week; that season is complete and graded, see `Final_Report.md`). Current work is improving the model in `nfl_predictor.py`, measured by the walk-forward scorecard `evaluate.py` on seasons 2010–2024.

Weekly notebooks (`Week{N}/Model.ipynb`, `Plot.ipynb`) are documented separately in [`docs/NOTEBOOKS.md`](docs/NOTEBOOKS.md). Weeks 1–21 are frozen historical snapshots that each carry their own inline copy of the pipeline; only Week 22 imports `nfl_predictor.py`.

## Experiment rules (model-improvement work)

- The only measure of a change is `./.venv/bin/python evaluate.py`.
  Current baseline: SCORE 0.6557, Brier 0.2319, accuracy 60.5%.
- Model changes go in `nfl_predictor.py` (`train_and_predict` and what it calls).
- Never edit `evaluate.py`, `fetch_data.py`, `loop.sh`, or anything in `tests/`.
- Never read `../NFL-Performance-Predictor-holdout-2025/` or use 2025 data in any form.
- Anything fitted (calibration, imputation, scaling, feature selection) uses only `train_df`.
- One idea per attempt. Afterward, run evaluate.py and pytest, and report SCORE, accuracy, Brier, and min/max probability.
- Never hardcode outcomes, special-case specific teams, seasons, or games, or weaken or skip tests or the leakage check.
- Don't run git or commit. The human or the loop script decides what to keep.

## Environment & Commands

Use the project-local env **`.venv/`** (gitignored): a conda env with Python 3.11 + `requirements.txt` (matching CI) plus `llvm-openmp` from conda-forge, which xgboost needs for `libomp.dylib`. Recreate with:

```bash
/opt/anaconda3/bin/conda create -y -p ./.venv python=3.11
./.venv/bin/python -m pip install -r requirements.txt
/opt/anaconda3/bin/conda install -y -p ./.venv -c conda-forge llvm-openmp
```

The Anaconda base env is broken (its scipy binary is rejected by dyld on this macOS version, so sklearn won't import there). Plain `python3` (Homebrew) has no packages. Always use `.venv/bin/python`.

```bash
# One-time: download 2010–2024 schedules + weekly stats into data/ (gitignored)
./.venv/bin/python fetch_data.py

# Scorecard: walk-forward log loss over 2011–2024, leakage-checked (~3 min). Last line is SCORE.
./.venv/bin/python evaluate.py

# Test suite (covers nfl_predictor.py only; a few seconds, no network needed). CI runs it on push/PR to main.
./.venv/bin/python -m pytest tests/ -v

# Experiment loop (human-run): N agent attempts at <focus>; keeps a change only if pytest passes
# and SCORE improves by >= 0.002. Needs a clean tree. Logs to loop_log.tsv, transcripts in loop_runs/.
./loop.sh "<focus>" <iterations>
```

## Data & the 2025 holdout

`data/` (2010–2024, written by `fetch_data.py`) is the only data inside the repo. `fetch_data.py` maps relocated teams' historical schedule codes (`OAK`→`LV`, `SD`→`LAC`, `STL`→`LA`) to the codes the weekly stats use; without that, those teams' games silently drop out of the features.

The 2025 season is the final holdout. The notebooks' `Week{N}/nfl_data/` caches (~25 GB, which include 2025) were moved out of the repo to `../NFL-Performance-Predictor-holdout-2025/`, and outputs were cleared from all tracked notebooks. `nfl_data/` and `data/` are gitignored; only `week*_predictions.csv` is tracked. Never `git add -f` a data CSV.

At the very end (human only), restore the caches with `for d in ../NFL-Performance-Predictor-holdout-2025/Week*/nfl_data; do mv "$d" "${d#../NFL-Performance-Predictor-holdout-2025/}"; done`.

## Scorecard (`evaluate.py`)

Frozen referee for model experiments. It loads `data/`, drops season ≥ 2025 (`HOLDOUT_SEASON`), and runs a leakage test: it rebuilds 7 checkpoint weeks with the future deleted and that week's scores scrambled, and any feature change means exit 1 with no SCORE. Then it runs walk-forward: it retrains `nfl_predictor.train_and_predict(train_df, test_df)` on strictly earlier games every `RETRAIN_EVERY = 4` weeks (weekly retraining would take ~13 min) and predicts the block. `test_df` has the label columns stripped. It prints `SCORE: <log loss>`; lower is better.

Current baseline: **SCORE 0.6557, Brier 0.2319, accuracy 60.5%** over 3,577 games. For reference: always predicting the training home-win rate scores 0.6860, a coin flip 0.6931, and always picking home is 56.1% accurate. History: 0.6838 with isotonic calibration, then 0.6557 with sigmoid (isotonic emitted exact 0/1 probabilities, and each lost "certain" game cost 34.5 log loss). Shrinkage toward the home-win rate was tried and dropped: it helped isotonic (0.6618) but not sigmoid.

## Architecture

### `nfl_predictor.py`

Holds `TEAM_MAPPING`, `NFLGamePredictor`, `NFLSpreadPredictor`, `predict_multiple_games_with_spreads()`, `run_validation_gate()`, and `train_and_predict()` (the scorecard's entry point). `Week22/Model.ipynb` also imports it. `tests/test_nfl_predictor.py` covers its pure/deterministic logic (15 tests, pytest, CI-enforced).

### Pipeline

`evaluate.py` uses `build_dataset` → `train_and_predict`, which calls `select_features` → `create_ensemble_model` → `final_model.predict_proba`.

1. `collect_data(start, end)`: downloads play-by-play, weekly, schedule, and team data via `nfl_data_py` (notebook path only; the scorecard uses `data/` from `fetch_data.py`).
2. `build_dataset(pbp, weekly, schedule)`: one row per scored game. Team features are season-to-date only (`week < current week`), cached per (season, week). `pbp` is unused.
3. `select_features(df, n_features=20)`: RFE with an LDA estimator, sweeping 2…20 features and keeping the best.
4. `create_ensemble_model(df)`: soft-voting `VotingClassifier` over Random Forest + Logistic Regression + Gradient Boosting + XGBoost, wrapped in `CalibratedClassifierCV` (sigmoid, cv=3), fit with temporal sample weights `np.exp(-0.15 * years_ago)` relative to the newest training season (a season 5 years older gets ~0.47× the weight).
5. `evaluate_model_with_calibration(df)`: `TimeSeriesSplit` CV reporting Brier score and log loss (notebook path).
6. `predict_games(games_df)`: win probabilities (notebook path; fills NaNs with test-batch means, unlike `train_and_predict`).

Supporting methods: `_calculate_defensive_stats()` (yards/points allowed), `_calculate_injury_percentage()` (an *estimate* derived from performance variance, not real injury reports), `create_team_features()` (season-to-date stats + last-3-game momentum), and `create_game_features()` (matchup deltas, `home_field_advantage` = 2.5 unless `is_neutral`). `build_dataset` passes neither `rest_days` nor `is_division_game`, so both are constants (7 and 0) in every row.

`NFLSpreadPredictor` trains four regressors (Random Forest, Linear, Gradient Boosting with quantile loss, XGBoost) and keeps the best by MAE. Spread confidence is a pure function of magnitude: `0.50 + min(0.45, abs(spread) * 0.025)`, capped at 0.95. The scorecard doesn't score spreads.

## Known Issues

- **`is_playoff` was inverted during training. Fixed in `nfl_predictor.py`.** `build_dataset` used `is_playoff=game.get('game_type', '') == 'REG'`, so regular-season games were labeled playoff and vice versa, while inference used `week >= 19`. It now reads `game_type in ('WC', 'DIV', 'CON', 'SB')`. Weeks 1–21's inline pipelines still have the bug.
- **`is_neutral` was hardcoded `False`. Fixed in `nfl_predictor.py`.** Training now reads the schedule's `location` column (`'Neutral'` for Super Bowls and international games). Weeks 1–21 still have the bug.
- **Validation gate is visibility only.** `run_validation_gate()` (PASS needs CV accuracy ≥ 0.55 and Brier ≤ 0.25) prints a banner in the Week 22 notebook but doesn't block anything. `evaluate.py` is the real measure.
- **Copy-forward drift.** The pipeline used to be duplicated per notebook, so a fix in one week didn't reach the others. Week 22 imports `nfl_predictor.py`, so new weeks inherit fixes. Weeks 1–21 keep their original, still-buggy pipelines on purpose: regenerating graded predictions with hindsight wouldn't be a real forecast.
- **Stale references.** `Week14/README_PDF_Conversion.md`, `MODEL_IMPROVEMENTS_SUMMARY.md`, `QUICK_START_IMPROVED_MODEL.md`, `Week10/Model_backup_20251106.ipynb`, and `.claude/agents/` were removed or never committed. Tracked markdown: `CLAUDE.md`, `docs/NOTEBOOKS.md`, `Week14/Project_Summary_Report.md`, `Final_Report.md`, `README.md`.

## Modifying the Model

All model changes go in `nfl_predictor.py`, through `train_and_predict` and what it calls, and are judged only by `evaluate.py` (see Experiment rules). High-leverage knobs:

- `train_and_predict()`: the whole fit/predict path. Calibration or post-processing goes here, fit on a temporal split of `train_df`.
- `select_features(df, n_features=...)`: RFE breadth.
- `create_ensemble_model()`: per-model hyperparameters, calibration method, and the `-0.15` temporal decay (more negative means stronger recency bias).
- `create_team_features()` / `create_game_features()`: where new features belong (rest days, division games, weather, QB availability). Features may use only information available before kickoff; the leakage test enforces this for `build_dataset`'s output.
