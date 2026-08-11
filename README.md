# NFL Game Outcome and Point Spread Prediction

CS5100 (Foundations of Artificial Intelligence) final project. A calibrated ensemble ML system that predicts NFL game winners and point spreads, deployed prospectively across all 22 weeks of the 2025 season (including playoffs and the Super Bowl).

**Full report:** [`Final_Report.pdf`](Final_Report.pdf) (source: [`Final_Report.md`](Final_Report.md)) — start there for problem statement, methodology, results, and discussion.

## Setup

```bash
pip install -r requirements.txt
```

The system was built and run against the specific package versions pinned in `requirements.txt`; `nfl_data_py` wraps official NFL data feeds and needs network access on first run per week.

## Running a week's pipeline

`Week22/Model.ipynb` imports its pipeline classes from [`nfl_predictor.py`](nfl_predictor.py) at the repo root (`NFLGamePredictor`, `NFLSpreadPredictor`, `predict_multiple_games_with_spreads`, `run_validation_gate`) rather than redefining them inline. **Weeks 1–21 remain self-contained** — each still carries its own inline copy of the pipeline as it was actually run to produce that week's already-graded, historical prediction, and none of them import the shared module. See "Known limitations" below for why those aren't being retroactively rewritten.

To run Week22 (or a future week built from it):

```bash
jupyter nbconvert --to notebook --inplace --execute \
  --ExecutePreprocessor.timeout=3600 Week22/Model.ipynb
```

The first run per week downloads and caches ~5 seasons of play-by-play/weekly/schedule data into `Week{N}/nfl_data/` (~900 MB, gitignored, 5–15 minutes); subsequent runs reuse the cache (3–5 minutes). Predictions are written to `Week{N}/week{N}_predictions.csv`.

To aggregate performance across every scored week and regenerate the report figures:

```bash
jupyter nbconvert --to notebook --inplace --execute Plot.ipynb
```

## Repository structure

This repo is organized **by NFL week** (`Week1/` … `Week22/`) rather than by artifact type (`data/`, `models/`, `scripts/`). That's a deliberate fit for how the project actually ran: the model retrains from scratch and is redeployed every week of a live season, so each `Week{N}/` folder is a self-contained unit holding that week's notebook, cached data, and output predictions — mirroring the real production timeline rather than a single offline experiment. See `Final_Report.md` §3.1 for the full rationale.

- `nfl_predictor.py` — shared pipeline module (`NFLGamePredictor`, `NFLSpreadPredictor`, `predict_multiple_games_with_spreads`, `run_validation_gate`); imported by Week22 only, see above
- `Week{N}/Model.ipynb` — full pipeline for week *N* (data collection → feature engineering → ensemble training → calibration → prediction)
- `Week{N}/week{N}_predictions.csv` — that week's graded predictions (tracked in git)
- `Week{N}/nfl_data/` — cached raw data (gitignored, regenerated on first run)
- `Plot.ipynb` — aggregates all weeks' predictions against actual results, produces the season-performance dashboard
- `figures/` — static exports of the report's visualizations (confusion matrix, calibration chart, ROC curve, pipeline diagram, season dashboard)
- `Final_Report.md` / `Final_Report.pdf` — the final report
- `CLAUDE.md` — detailed engineering notes: architecture, known bugs, and reproduction steps for the measured performance numbers

## Known limitations

See `Final_Report.md` §5.2 and `CLAUDE.md`'s "Known Issues" for the full list. Two items flagged there — an inverted `is_playoff` feature at training time and a hardcoded `is_neutral=False` (including for the Super Bowl) — are now fixed in `nfl_predictor.py` / `Week22/Model.ipynb`, along with a validation gate that actually calls the previously-orphaned `evaluate_model_with_calibration()`.

**These fixes were deliberately not used to regenerate any historical week's predictions.** `week{N}_predictions.csv` for every week, including Week 22, is left exactly as it was when originally generated — that's the record the whole report's "genuinely prospective" evaluation depends on. Regenerating a past week's prediction after the fact, with bugs fixed and the real outcome already known, would no longer be a legitimate prospective forecast. The fixes apply going forward only; Weeks 1–21 still run their original (buggy, self-contained) code and always will.
