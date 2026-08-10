# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Machine-learning system that predicts NFL game winners and point spreads for the **2025 season** (Weeks 1–22, including playoffs; Week 22 = Super Bowl, February 2026). The season is complete — all 22 weeks have predictions and results.

There is **no application code outside the notebooks**. Every `Week{N}/Model.ipynb` is a standalone, self-contained copy of the entire pipeline (~44 KB of class definitions in a single cell). There is no shared module, no package, no test suite, and no CI. Understanding this is the single most important fact about the repo: a model change means editing one notebook and copying it forward.

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

### Pipeline (all inside one code cell — cell index 5 in Weeks 14–22)

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

Use **Week22/Model.ipynb as the template** for any new week.

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

- **`is_playoff` is inverted during training.** In `build_dataset`, features are built with `is_playoff=game.get('game_type', '') == 'REG'` — `REG` is the *regular-season* code, so every regular-season game is labeled a playoff game and every real playoff game (`WC`/`DIV`/`CON`/`SB`) is labeled regular. Inference uses `is_playoff=(week >= 19)`, so training and prediction disagree on this feature's meaning. Likely contributor to the 38.5% playoff accuracy.
- **`is_neutral` is hardcoded `False` everywhere**, including the Super Bowl, so Week 22 gives the nominal home team a 2.5-point advantage that does not exist.
- **No held-out validation gates a deploy.** `evaluate_model_with_calibration()` exists but nothing blocks shipping a worse model; regressions are only visible weeks later in Plot.ipynb.
- **Copy-forward drift.** Because the pipeline is duplicated 22×, a fix applied to one week is invisible to the others. Weeks 1–13 still run the older pipeline and will never pick up fixes.
- **Stale references.** `Week14/README_PDF_Conversion.md`, `MODEL_IMPROVEMENTS_SUMMARY.md`, `QUICK_START_IMPROVED_MODEL.md`, `Week10/Model_backup_20251106.ipynb`, and `.claude/agents/` (the `model-analyzer` / `model-optimizer` agents) were all removed or never committed. Only `CLAUDE.md` and `Week14/Project_Summary_Report.md` are tracked markdown.

## Modifying the Model

Edits land in the big class cell (code cell 1 in the 14–22 layout) of the newest week, then get copied forward. High-leverage knobs:

- `select_features(df, n_features=...)` — RFE breadth.
- `create_ensemble_model()` — per-model hyperparameters and the `-0.15` temporal decay (more negative = stronger recency bias).
- `create_game_features()` — where new features belong (weather, QB availability, real injury reports, a working neutral-site flag).
- `train_spread_model()` — the regressor candidates.

Given the measured results above, validate any change against a holdout **before** copying it forward: run `evaluate_model_with_calibration(df)`, and back-test on completed 2025 weeks by comparing against the schedule cache rather than trusting projected improvements.
