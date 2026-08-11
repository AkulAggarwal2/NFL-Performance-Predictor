# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Machine-learning system that predicts NFL game winners and point spreads for the **2025 season** (Weeks 1–22, including playoffs; Week 22 = Super Bowl, February 2026). The season is complete — all 22 weeks have predictions and results.

**Weeks 1–21** are standalone, self-contained notebooks — each `Week{N}/Model.ipynb` carries its own inline copy of the pipeline (~44 KB of class definitions in a single cell), exactly as it was run to produce that week's already-graded prediction. **Week 22** is the exception: its pipeline classes (`NFLGamePredictor`, `NFLSpreadPredictor`, `predict_multiple_games_with_spreads`, `run_validation_gate`) now live in `nfl_predictor.py` at the repo root, and `Week22/Model.ipynb` imports them rather than redefining them inline (see "Shared module" below). There is still no test suite and no CI. Understanding the *history* here is the important fact: a model change used to mean editing one notebook and copying it forward by hand, with the copy silently diverging from that point on — Weeks 1–21 are frozen artifacts of that pattern and are not being retrofitted.

## Environment & Commands

There is no venv or lockfile. The working interpreter is the **Anaconda base env**, which already has every dependency:

```bash
/opt/anaconda3/bin/python      # pandas 1.5.3, numpy 1.26.4, scikit-learn 1.5.1, xgboost 3.0.2, nfl_data_py
/opt/anaconda3/bin/jupyter lab # notebooks
```

Plain `python3` (Homebrew) has **no** packages installed — always use the Anaconda path for scripting.

```bash
# Run a week's notebook headlessly (writes outputs back into the .ipynb)
/opt/anaconda3/bin/jupyter nbconvert --to notebook --inplace --execute \
  --ExecutePreprocessor.timeout=3600 Week22/Model.ipynb

# Aggregate performance across all weeks
/opt/anaconda3/bin/jupyter nbconvert --to notebook --inplace --execute Plot.ipynb
```

Cell 3 of each notebook runs `%pip install xgboost nfl_data_py pillow`; it is a no-op in this env.

**Disk:** the repo is ~25 GB. Each `Week{N}/nfl_data/` holds a ~900 MB `pbp_data_2020_2025.csv` (21 near-identical copies). All `nfl_data/*.csv` are gitignored; only `week*_predictions.csv` is tracked. Never `git add -f` a data CSV.

## Architecture

### Shared module (`nfl_predictor.py`) — Week 22 only

`nfl_predictor.py` at the repo root holds `TEAM_MAPPING`, `NFLGamePredictor`, `NFLSpreadPredictor`, `predict_multiple_games_with_spreads()`, and `run_validation_gate()`. `Week22/Model.ipynb` imports from it (with a `sys.path` insert in cell 1, since the notebook's kernel cwd is `Week22/`, one level below the module). Weeks 1–21 do **not** import it and never will — each still carries its own inline, self-contained copy of the pipeline as originally run. Extracted via pure code motion from the (already bug-fixed, see Known Issues) Week22 notebook; no model logic changed in the move, only two implicit-global dependencies were closed: `NFLSpreadPredictor.train_spread_model()` now takes an explicit `reference_predictor` argument instead of reading a notebook-global `predictor`, and `predict_multiple_games_with_spreads()` now takes `weekly_data`/`schedule_data` as explicit parameters instead of reading notebook globals.

### Pipeline (class definitions live in `nfl_predictor.py` for Week 22; inline in one code cell — cell index 5 — for Weeks 1–21)

`NFLGamePredictor` owns the whole classification path:

1. `collect_data(2020, 2025)` — downloads play-by-play, weekly, schedule, and team data via `nfl_data_py`; caches to `nfl_data/`. **The call site uses 2020–2025**, not the `start_year=2010` default in the signature.
2. `build_dataset(pbp, weekly, schedule)` — joins to one row per game, writes `nfl_data/processed_game_features.csv`.
3. `select_features(df, n_features=20)` — RFE with an LDA estimator, sweeping 2…20 features and keeping the best.
4. `create_ensemble_model(df)` — soft-voting `VotingClassifier` over Random Forest + Logistic Regression + Gradient Boosting + XGBoost, each wrapped in `CalibratedClassifierCV` (isotonic, cv=3), fit with **temporal sample weights** `np.exp(-0.15 * years_ago)` (2024 ≈ 2× the weight of 2020 under the current 2020–2025 window).
5. `evaluate_model_with_calibration(df)` — `TimeSeriesSplit` CV reporting Brier score and log loss.
6. `predict_games(games_df)` — win probabilities.

Supporting private methods: `_calculate_defensive_stats()` (yards/points allowed), `_calculate_injury_percentage()` (an *estimate* derived from performance variance — not real injury reports), `create_team_features()` (season-to-date stats + last-3-game momentum), `create_game_features()` (matchup deltas, `rest_days`, division flag, `home_field_advantage` = 2.5 unless `is_neutral`).

`NFLSpreadPredictor` (separate cell) trains four regressors — Random Forest, Linear, Gradient Boosting (quantile loss), XGBoost — and keeps the best by MAE. Spread confidence is a pure function of magnitude: `0.50 + min(0.45, abs(spread) * 0.025)`, capped at 0.95.

### Two notebook generations

| Weeks | Layout | Notes |
|---|---|---|
| 1–13 | 7–9 code cells | Older pipeline. Weeks 1–5 winner-only; Weeks 6+ add spreads. |
| 14–22 | 11 code cells | Current pipeline, plus 10 markdown cells structured as an academic report (Intro → Data Cleaning → Modeling → Results → Discussion → Conclusions), EDA, and a model-performance/ROC cell. |

Week 15 is the same 11 code cells with the markdown stripped, and keeps a `Model_enhanced_backup.ipynb`.

**Weeks 16–22 are byte-identical in model code.** Only three code cells differ between consecutive weeks — 7, 8, and 9 in the 11-code-cell layout (cells 14, 15, 16 counting markdown). Weeks 14–15 differ additionally at code cell 6, where `is_playoff=False` was later changed to `is_playoff=(week >= 19)`.

Use **Week22/Model.ipynb as the template** for any new week. Because Week22 now imports its classes from `nfl_predictor.py` instead of redefining them inline, a `Week{N}` copied from it inherits that same import — meaning a fix made once in `nfl_predictor.py` after Week 22 will, for the first time in this repo's history, actually reach every subsequent week without a manual copy-paste. It does not reach Weeks 1–21 retroactively.

### Plot.ipynb

Aggregates every `Week{N}/week{N}_predictions.csv`, re-fetches actual results from `nfl_data_py`, and renders a 2×2 figure (weekly accuracy vs. 50% baseline, cumulative accuracy, correct/incorrect bars, high-confidence picks) plus a game-by-game table.

- `BASE_DIR` is **hardcoded** to `/Users/akulaggarwal/Documents/NFL-Performance-Predictor` (cell 1) — change it if the repo moves.
- `MAX_WEEK` (cell 1, currently `22`) bounds the sweep; raise it when adding a week.
- `get_predictions_for_week_smart()` looks in globals first (`week{N}_spread_results`, then `week{N}_results`), then falls back to the CSV.

## Adding a New Week

1. `cp Week22/Model.ipynb Week{N}/Model.ipynb` (don't copy `nfl_data/` — it re-downloads, and each copy is ~900 MB).
2. Edit exactly three code cells:
   - **"EXECUTE WEEK N SPREAD PREDICTIONS"** — the `week{N}_games` string, the `game_schedule` dict of kickoff times, `time_order`, and every `week{N-1}_spread_results` reference.
   - **"CONFIGURATION"** — `WEEK_NUMBER` and `SEASON`.
   - **"Save Week N predictions to CSV"** — variable and filename.
3. Run all cells. First run downloads data (slow, ~900 MB); later runs hit the cache.
4. After games are played, re-run the results-fetching cell.
5. Bump `MAX_WEEK` in Plot.ipynb.

**Game schedule format** — parsed by regex `\(Away\)\s+(.*?)\s+vs\.\s+\(Home\)\s+(.*)`, one game per line:

```
(Away) Seattle Seahawks vs. (Home) New England Patriots
```

Full team names resolve through an inline `team_mapping` dict; unmapped names fall back to the last whitespace token, so a typo silently produces a garbage abbreviation.

## Conventions

- **Result variables:** Weeks 1–5 `week{N}_results`; Weeks 6+ `week{N}_spread_results`. Plot.ipynb depends on both spellings.
- **Team abbreviations:** the Rams are `LA` in predictions but `LAR` in `nfl_data_py` schedules — normalize when joining, or the game silently drops.
- **CSV schema** (`week{N}_predictions.csv`, required by Plot.ipynb): `game_num, away_team, home_team, matchup, predicted_winner, confidence, home_win_prob, away_win_prob`. Weeks 6+ add `predicted_spread, spread_display, favored_team, spread_magnitude`. Week 1 instead carries `high_confidence`. `matchup` is `"AWAY @ HOME"`; `confidence` is 0–1; `predicted_spread` is signed home-minus-away.

## Actual 2025-Season Performance

Measured over all 283 scored games (the Week 4 GB@DAL 40–40 tie is excluded):

| Segment | Accuracy |
|---|---|
| **Full season (W1–22)** | **54.4%** (154/283) |
| Weeks 1–9 (older pipeline) | 60.4% (81/134) |
| Weeks 10–18 (post-"enhancement") | 50.0% (68/136) |
| Weeks 19–22 (playoffs) | 38.5% (5/13) |
| High-confidence picks (conf > 0.65) | 58.3% (49/84) |

Spread model, Weeks 6–22: **MAE 11.85, RMSE 14.55**, mean bias +1.93 toward the home team. Pooled correlation between predicted spread and actual margin is **0.11** — close to no signal.

Calibration is poor and non-monotonic — the highest-confidence bucket is the second-worst:

| Stated confidence | n | Actual accuracy |
|---|---|---|
| ≤0.55 | 76 | 38.2% |
| 0.55–0.60 | 68 | 67.6% |
| 0.60–0.65 | 55 | 54.5% |
| 0.65–0.70 | 47 | 63.8% |
| >0.70 | 37 | 51.4% |

Reproduce by joining the prediction CSVs against `Week22/nfl_data/schedule_data_2020_2025.csv` (filter `season == 2025`, map `LAR`→`LA`, skip ties), or by running Plot.ipynb.

**The Week-10 "enhanced model" did not deliver its targets.** Prior versions of this file recorded projected gains (66–68% accuracy, spread MAE 7.5–8.5) as if achieved. Measured results went the other way: accuracy fell from 60.4% to 50.0% after the change, and spread MAE was flat (11.72 → 11.90). Treat any performance claim in `Week14/Project_Summary_Report.md` (which states ~63%) as a projection, not a measurement.

## Known Issues

- **`is_playoff` was inverted during training — fixed in `nfl_predictor.py` (Week 22 only).** `build_dataset` used to build features with `is_playoff=game.get('game_type', '') == 'REG'` — `REG` is the *regular-season* code, so every regular-season game was labeled a playoff game and every real playoff game (`WC`/`DIV`/`CON`/`SB`) was labeled regular, while inference used `is_playoff=(week >= 19)` — training and prediction disagreed on this feature's meaning. Now reads `game.get('game_type', '') in ('WC', 'DIV', 'CON', 'SB')`. **Weeks 1–21 still have the original bug** — this was likely a contributor to the 38.5% playoff accuracy in those weeks, and that historical number has not been altered.
- **`is_neutral` was hardcoded `False` everywhere — fixed in `nfl_predictor.py` (Week 22 only).** Training now reads the real `location` column from `nfl_data_py` schedule data (`'Neutral'` for the Super Bowl and international games) instead of a hardcoded `False`; inference defaults to treating week 22 as neutral unless told otherwise, and Week22's own call site passes `is_neutral=True` explicitly. **Weeks 1–21 still have the original bug.**
- **No held-out validation gated a deploy — fixed in `nfl_predictor.py` (Week 22 only).** `evaluate_model_with_calibration()` existed but was never actually called anywhere in the pipeline. It's now invoked after `create_ensemble_model()` and checked by `run_validation_gate()` (PASS requires CV accuracy ≥ 0.55 and Brier ≤ 0.25; prints a clear PASS/FAIL banner either way). This doesn't retroactively gate anything already deployed, and doesn't auto-block execution on FAIL (it prints and continues) — it's a visibility fix, not a hard stop.
- **Copy-forward drift — fixed going forward only.** Because the pipeline used to be duplicated 22×, a fix applied to one week was invisible to the others. Week 22 now imports from `nfl_predictor.py` instead of carrying its own inline copy, so a future week built from the Week22 template will inherit fixes made to the shared module automatically. Weeks 1–21 still run their original, self-contained (and still-buggy) inline pipelines and always will — fixing them would mean regenerating already-graded historical predictions with hindsight of the real outcomes, which is not a legitimate prospective forecast.
- **Stale references.** `Week14/README_PDF_Conversion.md`, `MODEL_IMPROVEMENTS_SUMMARY.md`, `QUICK_START_IMPROVED_MODEL.md`, `Week10/Model_backup_20251106.ipynb`, and `.claude/agents/` (the `model-analyzer` / `model-optimizer` agents) were all removed or never committed. `CLAUDE.md`, `Week14/Project_Summary_Report.md`, `Final_Report.md`, and `README.md` are the tracked markdown files.

## Modifying the Model

For a new week built from the Week22 template, edits land in `nfl_predictor.py` directly (Week22/Model.ipynb imports from it — no more copying a class definition by hand). For Weeks 1–21, edits still land in that week's own inline class cell and go nowhere else; those weeks are historical snapshots, not live code. High-leverage knobs:

- `select_features(df, n_features=...)` — RFE breadth.
- `create_ensemble_model()` — per-model hyperparameters and the `-0.15` temporal decay (more negative = stronger recency bias).
- `create_game_features()` — where new features belong (weather, QB availability, real injury reports, a working neutral-site flag).
- `train_spread_model()` — the regressor candidates.

Given the measured results above, validate any change against a holdout **before** copying it forward: run `evaluate_model_with_calibration(df)`, and back-test on completed 2025 weeks by comparing against the schedule cache rather than trusting projected improvements.
