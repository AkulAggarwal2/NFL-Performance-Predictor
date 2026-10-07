# Weekly notebooks

Notes for the `Week{N}/Model.ipynb` notebooks and `Plot.ipynb` that produced and graded the live 2025-season predictions. Model-improvement work doesn't touch any of this; see `CLAUDE.md`.

## Commands

The notebooks need Jupyter, which `.venv/` doesn't have. Anaconda's `jupyter` still starts, but its kernel can't import sklearn (broken scipy; see `CLAUDE.md`), so executing a notebook currently fails until that env is fixed or Jupyter is added to `.venv/`.

```bash
# Run a week's notebook headlessly (writes outputs back into the .ipynb)
/opt/anaconda3/bin/jupyter nbconvert --to notebook --inplace --execute \
  --ExecutePreprocessor.timeout=3600 Week22/Model.ipynb

# Aggregate performance across all weeks
/opt/anaconda3/bin/jupyter nbconvert --to notebook --inplace --execute Plot.ipynb
```

Cell 3 of each notebook runs `%pip install xgboost nfl_data_py pillow`; it's a no-op once the packages are installed.

Each notebook caches its downloads in `Week{N}/nfl_data/` (~900 MB, mostly `pbp_data_2020_2025.csv`). Those caches currently live outside the repo in `../NFL-Performance-Predictor-holdout-2025/` (see "Data & the 2025 holdout" in `CLAUDE.md`), so running a notebook re-downloads. Outputs were cleared from every tracked notebook; re-run one to regenerate them.

## Notebook generations

Weeks 1–21 each carry their own inline copy of the pipeline (~44 KB of class definitions in code cell index 5), exactly as run to produce that week's graded prediction. Week 22 imports the classes from `nfl_predictor.py` instead (with a `sys.path` insert in cell 1, since its kernel cwd is `Week22/`).

| Weeks | Layout | Notes |
|---|---|---|
| 1–13 | 7–9 code cells | Older pipeline. Weeks 1–5 winner-only; Weeks 6+ add spreads. |
| 14–22 | 11 code cells | Current pipeline, plus 10 markdown cells structured as an academic report (Intro → Data Cleaning → Modeling → Results → Discussion → Conclusions), EDA, and a model-performance/ROC cell. |

Week 15 is the same 11 code cells with the markdown stripped, and keeps a `Model_enhanced_backup.ipynb`.

**Weeks 16–22 are byte-identical in model code.** Only three code cells differ between consecutive weeks: 7, 8, and 9 in the 11-code-cell layout (cells 14, 15, 16 counting markdown). Weeks 14–15 differ additionally at code cell 6, where `is_playoff=False` was later changed to `is_playoff=(week >= 19)`.

Use **Week22/Model.ipynb as the template** for any new week. Because it imports from `nfl_predictor.py`, a `Week{N}` copied from it inherits fixes made to the shared module, with no copy-paste. Those fixes don't reach Weeks 1–21.

## Adding a new week

1. `cp Week22/Model.ipynb Week{N}/Model.ipynb` (don't copy `nfl_data/`; it re-downloads).
2. Edit exactly three code cells:
   - **"EXECUTE WEEK N SPREAD PREDICTIONS"**: the `week{N}_games` string, the `game_schedule` dict of kickoff times, `time_order`, and every `week{N-1}_spread_results` reference.
   - **"CONFIGURATION"**: `WEEK_NUMBER` and `SEASON`.
   - **"Save Week N predictions to CSV"**: variable and filename.
3. Run all cells. First run downloads data (slow, ~900 MB); later runs hit the cache.
4. After games are played, re-run the results-fetching cell.
5. Bump `MAX_WEEK` in Plot.ipynb.

**Game schedule format.** Parsed by regex `\(Away\)\s+(.*?)\s+vs\.\s+\(Home\)\s+(.*)`, one game per line:

```
(Away) Seattle Seahawks vs. (Home) New England Patriots
```

Full team names resolve through an inline `team_mapping` dict; unmapped names fall back to the last whitespace token, so a typo silently produces a garbage abbreviation.

## Plot.ipynb

Aggregates every `Week{N}/week{N}_predictions.csv`, re-fetches actual results from `nfl_data_py`, and renders a 2×2 figure (weekly accuracy vs. 50% baseline, cumulative accuracy, correct/incorrect bars, high-confidence picks) plus a game-by-game table.

- `BASE_DIR` is **hardcoded** to `/Users/akulaggarwal/Documents/NFL-Performance-Predictor` (cell 1). Change it if the repo moves.
- `MAX_WEEK` (cell 1, currently `22`) bounds the sweep; raise it when adding a week.
- `get_predictions_for_week_smart()` looks in globals first (`week{N}_spread_results`, then `week{N}_results`), then falls back to the CSV.

## Conventions

- **Result variables:** Weeks 1–5 `week{N}_results`; Weeks 6+ `week{N}_spread_results`. Plot.ipynb depends on both spellings.
- **Team abbreviations:** the Rams are `LA` in predictions but `LAR` in `nfl_data_py` schedules. Normalize when joining, or the game silently drops.
- **CSV schema** (`week{N}_predictions.csv`, required by Plot.ipynb): `game_num, away_team, home_team, matchup, predicted_winner, confidence, home_win_prob, away_win_prob`. Weeks 6+ add `predicted_spread, spread_display, favored_team, spread_magnitude`. Week 1 instead carries `high_confidence`. `matchup` is `"AWAY @ HOME"`; `confidence` is 0–1; `predicted_spread` is signed home-minus-away.
