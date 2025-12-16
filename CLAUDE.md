# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an NFL game prediction system that uses machine learning to predict game winners and point spreads for the 2025 NFL season. The project is organized by week, with each week containing its own Model.ipynb notebook for generating predictions and tracking results.

## Project Structure

```
NFL Performance Prediction/
├── Week1/ through Week14/    # Weekly prediction notebooks
│   ├── Model.ipynb           # Prediction model for that week
│   ├── week*_predictions.csv # Saved predictions (if generated)
│   ├── final_model.joblib    # Trained model checkpoint
│   └── nfl_data/            # Downloaded NFL data cache
├── Plot.ipynb                # Performance analysis and visualization across all weeks
├── .claude/agents/           # Custom Claude Code agents for model analysis and optimization
└── Week14/                   # Latest week with additional documentation
    ├── Project_Summary_Report.md      # Academic summary report
    └── README_PDF_Conversion.md       # Instructions for PDF conversion
```

## Development Workflow

### Running Predictions for a Specific Week

1. Navigate to the desired week's directory (e.g., `Week6/`)
2. Open `Model.ipynb` in Jupyter
3. Execute cells in order:
   - Cell 1: Install dependencies (`xgboost`, `nfl_data_py`, `pillow`)
   - Cell 2: Initialize NFLGamePredictor class (includes data collection and model training)
   - Cell 3: Train the model (this downloads NFL data from 2015-2025 and may take several minutes)
   - Cell 4-6: Run predictions for the specific week's games
   - Cell 7: Fetch actual results and calculate accuracy (after games have been played)
   - Cell 8: Save predictions to CSV

### Analyzing Overall Performance

Use `Plot.ipynb` in the root directory to:
- Aggregate predictions from all weeks
- Calculate cumulative accuracy statistics
- Generate performance visualizations
- Track high-confidence pick success rates

**Important**: `Plot.ipynb` automatically loads predictions from:
1. Global variables if Model.ipynb was run in the same kernel (e.g., `week1_results`, `week6_spread_results`)
2. CSV files saved in each Week directory (e.g., `Week1/week1_predictions.csv`)

## Model Architecture

### Two-Stage Prediction System

**Week 1-5**: Basic winner prediction
- Uses `NFLGamePredictor` class
- Predictions stored in `week*_results` variables
- Outputs: predicted winner and confidence percentage

**Week 6+**: Enhanced spread prediction
- Uses both `NFLGamePredictor` and `NFLSpreadPredictor` classes
- Predictions stored in `week*_spread_results` variables
- Outputs: predicted winner, confidence, and point spread

### Feature Engineering

**Core Features** (selected via RFE):
- **Offensive**: `home_passing_ypg`, `home_rushing_ypg`, `home_total_ypg`, `home_points_pg`, `home_passing_tds_pg`
- **Defensive**: `home_defensive_ypg`, `home_defensive_ppg` (yards/points allowed)
- **Turnovers**: `home_turnovers_pg`, `away_turnovers_pg`, `turnover_advantage`
- **Contextual**: `scoring_advantage`, `is_playoff`, `season`

**Enhanced Features** (Week 10+):
- **Injury Metrics**: `home_injury_pct`, `away_injury_pct`, `injury_advantage` (estimated from performance variance)
- **Momentum**: `momentum_last3` (win % over last 3 games), `momentum_advantage`
- **Rest Days**: Days since last game (captures fatigue, especially for Thursday games)
- **Division Rivalry**: Boolean flag for divisional matchups
- **Vegas Spread**: Market consensus (when available)

### Ensemble Model

**Week 1-9** (Legacy): 3-model ensemble
- Random Forest (n_estimators=200, max_depth=5)
- Logistic Regression (C=1)
- XGBoost (max_depth=5, learning_rate=0.1)

**Week 10+** (Enhanced): 4-model ensemble with temporal weighting
- Random Forest (n_estimators=200, max_depth=15, with overfitting protection)
- Logistic Regression (C=1, max_iter=1000)
- Gradient Boosting Classifier (n_estimators=200, max_depth=8, learning_rate=0.1)
- XGBoost (max_depth=8, L1/L2 regularization, subsampling=0.8)

**Temporal Weighting**: Recent seasons weighted via exponential decay: `weight = exp(-0.15 × years_ago)`. 2024 data weighted ~3× more than 2015 data.

All models wrapped in CalibratedClassifierCV (isotonic regression, cv=3) for better probability estimates.

## Data Sources

All data is automatically fetched from `nfl_data_py` library:
- Play-by-play data: Game-level statistics (2015-2025)
- Weekly data: Player and team performance metrics
- Schedule data: Game schedules, scores, spreads

Data is cached locally in `nfl_data/` folders within each week directory to speed up subsequent runs.

## Key Functions

### NFLGamePredictor Class

- `collect_data(start_year, end_year)`: Download NFL data and cache as CSV
- `create_team_features(weekly_data, season, week)`: Calculate season-to-date team statistics (includes momentum calculation for Week 10+)
- `create_game_features(home_team, away_team, ...)`: Generate matchup features (includes rest_days, division_game, vegas_spread for Week 10+)
- `build_dataset(pbp_data, weekly_data, schedule_data)`: Process raw data into training dataset
- `select_features(df, n_features)`: Use RFE to identify optimal feature set
- `create_ensemble_model(df)`: Train final voting classifier with temporal weighting (Week 10+)
- `evaluate_model_with_calibration(df)`: Time-series cross-validation with Brier score and log loss (Week 10+)
- `predict_games(games_df)`: Generate predictions for new games
- `_calculate_injury_percentage()`: Estimate injury impact from performance variance (Week 6+)
- `_calculate_defensive_stats()`: Calculate defensive metrics (Week 10+)

### NFLSpreadPredictor Class (Week 6+)

- `train_spread_model(df)`: Train regression model for point spreads (tests 4 models: Random Forest, Linear Regression, Gradient Boosting with quantile loss, XGBoost with regularization)
- `predict_spreads(games_df)`: Predict point differential for games with confidence calculation

### Helper Functions

- `predict_multiple_games(predictor, games_text, season, week)`: Batch predict from formatted text
- `predict_multiple_games_with_spreads(predictor, spread_predictor, ...)`: Enhanced version with spreads
- `fetch_actual_results(predictions_df, season, week)`: Auto-fetch game results from nfl_data_py
- `analyze_week(week_num, season_year, predictions_df, actuals)`: Compare predictions vs actual results

## Naming Conventions

### Variables
- Weeks 1-5: `week{N}_results` (e.g., `week1_results`, `week2_results`)
- Weeks 6+: `week{N}_spread_results` (e.g., `week6_spread_results`, `week9_spread_results`)
- Actual results: `week{N}_actual_results`, `week{N}_final_results`

### Team Abbreviations
Standard NFL abbreviations are used throughout:
- `PHI` (Eagles), `KC` (Chiefs), `LAC` (Chargers), `SF` (49ers), `LA` (Rams)
- Note: Rams use `LA` not `LAR`; use the `team_mapping` dict in prediction functions for consistency

## Performance Metrics

### Legacy Model Performance (Weeks 1-9, 2025 season)
- Overall accuracy: **60.7%** across 135 games
- Total correct: 82 games, Total incorrect: 53 games
- High-confidence picks (>65%): Accuracy varies significantly by week (33%-100%)
- Week-to-week variance: ±23 percentage points
- Best weeks: Week 3 (87.5%), Week 8 (84.6%)
- Worst week: Week 6 (40.0%)

**Week-by-Week Performance:**
- Week 1: 62.5% (10/16), High confidence: 80.0%
- Week 2: 56.2% (9/16), High confidence: 75.0%
- Week 3: 87.5% (14/16), High confidence: 75.0%
- Week 4: 56.2% (9/16), High confidence: 33.3%
- Week 5: 42.9% (6/14), High confidence: 66.7%
- Week 6: 40.0% (6/15), High confidence: 50.0%
- Week 7: 60.0% (9/15), High confidence: 100.0%
- Week 8: 84.6% (11/13), High confidence: 83.3%
- Week 9: 57.1% (8/14), High confidence: 50.0%

**Legacy Spread Model Performance** (Weeks 6-9):
- MAE: ~10.76 points
- RMSE: ~14.21 points

### Enhanced Model Performance (Week 10+)
- **Overall accuracy target**: 66-68% (+5-7 points improvement)
- **High-confidence picks target**: 72-75% (+9-12 points improvement)
- **Week-to-week variance target**: ±12 points (50% reduction)
- **Spread MAE target**: 7.5-8.5 points (20-30% improvement)
- **Spread RMSE target**: <10.5 points (26% improvement)

*Note: Week 10+ uses enhanced 4-model ensemble with temporal weighting, momentum features, defensive metrics, and improved spread model.*

**Performance tracking**: For comprehensive performance analysis across all weeks, see `Week14/Project_Summary_Report.md` which includes detailed metrics, feature importance analysis, and insights from the 2024 season.

## Common Tasks

### Adding a New Week

1. Create `Week{N}/` directory
2. **Copy `Model.ipynb` from Week 10-14** to get enhanced model with all improvements
   - Week 10-14 use the enhanced 4-model ensemble
   - For legacy model reference (Weeks 1-9): See Week 6-9 for basic spread predictions
   - Latest tested model: Week 14
3. Update the week number in:
   - Game schedule text (update team matchups)
   - `WEEK_NUMBER` and `SEASON` configuration variables
   - Variable names (e.g., `week15_games`, `week15_spread_results`)
4. Update game schedule dictionary with correct dates/times
5. Run all cells to generate predictions:
   - Cell 0: Install dependencies (run once)
   - Cells 1-4: Train models (5-15 min first run, 3-5 min cached)
   - Cell 6: Generate predictions
   - Cell 7: Fetch results (after games complete)
   - Cell 8: Export CSV (saves to `week{N}_predictions.csv`)
6. After games complete, run result analysis cell
7. Update `MAX_WEEK` in Plot.ipynb and re-run to include new week

**Note**: Week 14 includes academic documentation and may not have all standard outputs if it was used for report generation rather than live predictions.

### Extracting Predictions from Existing Notebooks

If a week's Model.ipynb has been executed but the CSV wasn't saved:
1. Read the notebook file to find the prediction outputs
2. Look for `week{N}_spread_results` or `week{N}_results` in cell outputs
3. Extract the prediction data (matchup, predicted_winner, confidence, spreads)
4. Create CSV with required columns: `game_num`, `away_team`, `home_team`, `matchup`, `predicted_winner`, `confidence`, `home_win_prob`, `away_win_prob`
5. For spread predictions (Week 6+), also include: `predicted_spread`, `spread_display`, `favored_team`, `spread_magnitude`

### Modifying the Model

**For future improvements:**
- Adjust feature selection in `select_features()` (change `n_features` parameter)
- Tune hyperparameters in `create_ensemble_model()` (e.g., adjust max_depth, learning_rate)
- Add new features in `create_game_features()` (e.g., weather conditions, quarterback ratings)
- Modify temporal weighting decay rate (currently `-0.15`, higher = more recent bias)
- For spread model: adjust regression models in `train_spread_model()`

**Testing model changes:**
- Use `evaluate_model_with_calibration(df)` for time-series cross-validation (Week 10+)
- Monitor Brier score (calibration quality) and log loss (probability accuracy)
- Compare predictions on holdout weeks before deploying
- Document expected vs actual performance improvements

### Troubleshooting Data Issues

If data download fails:
- Check internet connection
- `nfl_data_py` may not have current week's data yet
- Model falls back to previous season data (e.g., uses 2024 Week 19 for 2025 predictions)
- Verify team abbreviations match NFL standard codes

### Generating Visualizations

`Plot.ipynb` automatically creates:
- Weekly accuracy line chart with 50% baseline and fill areas
- Cumulative accuracy progression tracking
- Correct vs incorrect predictions bar chart (grouped by week)
- High-confidence pick performance (>65% confidence threshold)
- Detailed game-by-game results tables with confidence scores

**Important**: Update `MAX_WEEK` variable in Plot.ipynb to match the latest completed week with predictions CSV files.

**Running Plot.ipynb:**
1. Ensure all `week{N}_predictions.csv` files exist in their respective Week directories
2. Open Plot.ipynb in Jupyter
3. Verify `MAX_WEEK` matches your latest week with saved predictions (e.g., `MAX_WEEK = 13`)
4. Execute all cells to generate visualizations
5. The notebook will:
   - Auto-detect prediction files from Week1 through Week{MAX_WEEK}
   - Fetch actual results via `nfl_data_py`
   - Calculate statistics and generate 4 comprehensive plots
   - Display detailed breakdown tables

## Dependencies

Required packages:
```
pandas>=1.5.3
numpy>=1.26.0
matplotlib
seaborn
scikit-learn
xgboost>=3.0.2
nfl_data_py>=0.3.3
joblib
pillow  # For image handling in notebooks
```

Install with: `pip install xgboost nfl_data_py pillow`

## Critical Implementation Details

### Data Flow
1. **Training**: Model trains on historical games (2015-2024) where outcomes are known
2. **Feature Generation**: For Week N predictions, uses team stats through Week N-1
3. **Prediction**: Generates winner/spread predictions for Week N games
4. **Validation**: After games complete, fetches actual results and calculates accuracy

### Model Behavior
- Model retrains from scratch each week using historical data (2015-2024) plus current season
- **Temporal weighting** (Week 10+): Recent seasons weighted exponentially higher via `exp(-0.15 × years_ago)`
- Predictions use team statistics up to (but not including) the target week
- **Momentum features** (Week 10+): Includes last 3 games win percentage
- Home field advantage is fixed at 2.5 points in feature engineering
- Injury metrics are estimated from performance variance (not actual injury reports)
- **Spread predictions**:
  - Week 6-9: Random Forest regression
  - Week 10+: Tests 4 models (Random Forest, Linear, Gradient Boosting with quantile loss, XGBoost), selects best by MAE
- Confidence scores for spread model: `0.50 + min(0.45, abs(spread) × 0.025)`, capped at 95%

### CSV File Requirements
Each `week{N}_predictions.csv` must contain these columns for Plot.ipynb to work:
- `matchup` (format: "AWAY @ HOME", e.g., "KC @ LAC")
- `away_team` (3-letter abbreviation)
- `home_team` (3-letter abbreviation)
- `predicted_winner` (3-letter abbreviation)
- `confidence` (float, 0.0 to 1.0)

Optional columns for enhanced analysis:
- `predicted_spread`, `spread_display`, `favored_team`, `spread_magnitude`

### Automated Result Fetching
- `fetch_actual_results()` function automatically queries `nfl_data_py` for completed games
- Returns None if games haven't been played yet (futures games have no scores)
- Matches predictions to actual results by `away_team` and `home_team` fields
- Compares predicted winner vs actual winner to calculate accuracy

## Model Enhancement History

### Week 10 Improvements (November 2025)
Major model enhancements implemented based on comprehensive analysis. All improvements are documented in `MODEL_IMPROVEMENTS_SUMMARY.md` and `QUICK_START_IMPROVED_MODEL.md`.

**Key Changes:**
1. **Enhanced Ensemble**: 3→4 models, increased tree depth (5→15 for RF, 5→8 for XGB/GB)
2. **Temporal Weighting**: Exponential decay favoring recent seasons (2024 weighted 3× vs 2015)
3. **New Features**: Momentum (last 3 games), defensive stats, rest days, division rivalry, vegas spread
4. **Better Cross-Validation**: TimeSeriesSplit with Brier score and log loss metrics
5. **Improved Spread Model**: Tests 4 regression models, quantile loss for Gradient Boosting

**Expected Impact:**
- Accuracy: 60.7% → 66-68%
- Variance reduction: 50% (±23 → ±12 points week-to-week)
- Spread MAE: 10.76 → 7.5-8.5 points

**Using Enhanced Model:**
- Copy `Week10/Model.ipynb` for all future weeks (Week 11+)
- All improvements are backward compatible
- Enhanced features auto-populate when available in data
- Original model preserved in `Week10/Model_backup_20251106.ipynb`

**Validation:**
- 18/18 automated tests passed (100%)
- Production-ready, no syntax errors or breaking changes
- Maintains full compatibility with Plot.ipynb

## Project Documentation

### Academic Report
`Week14/Project_Summary_Report.md` contains a comprehensive 4-page academic summary (2,800 words) including:
- **Executive Summary**: Overview of the ML prediction system achieving 63% accuracy on 2024 season
- **Problem Statement & Motivation**: Sports prediction challenges and ML relevance
- **Detailed Methodology**: Data collection (nfl_data_py), feature engineering (32 features), model architecture (4-model ensemble)
- **Results & Performance**: Comprehensive metrics including accuracy by confidence level, feature importance rankings, model comparisons
- **Insights & Analysis**: What drives NFL success (offense vs defense, momentum effects, home field advantage)
- **Practical Applications**: Guidelines for sports betting, fantasy football, and team analytics
- **Limitations & Future Work**: Data quality issues, model constraints, improvement roadmap

### PDF Conversion
`Week14/README_PDF_Conversion.md` provides multiple methods for converting the summary report to PDF:
- **Pandoc** (recommended): Command-line conversion with LaTeX styling
- **VS Code/Cursor**: Markdown preview → print to PDF
- **Typora/MacDown**: Desktop markdown editors with export
- **Online converters**: Browser-based options
- **Python automation**: markdown-pdf package

## Custom Claude Code Agents

This project includes specialized agents in `.claude/agents/` for model analysis and optimization:

### model-analyzer
**Purpose**: Analyzes ML model performance, reviews architecture, and identifies optimization opportunities.

**When to use**:
- After model training completes
- When accuracy drops below expected thresholds
- When investigating performance inconsistencies
- When user requests model improvements

**Capabilities**:
- Comprehensive model architecture analysis (ensemble composition, hyperparameters, calibration)
- Performance metrics deep dive (accuracy trends, spread MAE/RMSE, confidence calibration)
- Feature engineering assessment (RFE selection, missing features, importance scores)
- Root cause identification for performance issues
- Prioritized recommendations with expected impact estimates

**Output**: Detailed analysis report with executive summary, performance diagnosis, and prioritized recommendations for model-optimizer agent.

### model-optimizer
**Purpose**: Implements model improvements and optimizations systematically.

**When to use**:
- After receiving recommendations from model-analyzer
- When implementing specific model enhancements (hyperparameter changes, new features)
- When applying architectural changes to existing models
- When optimizing model training configurations

**Capabilities**:
- Incremental implementation of model improvements
- Hyperparameter tuning and validation
- Feature engineering additions with testing
- Model architecture modifications
- Performance comparison and documentation

**Workflow**: Typically invoked after model-analyzer identifies improvement opportunities. Ensures proper implementation, testing, and backward compatibility.

These agents are automatically available when using Claude Code in this repository and should be invoked proactively when working with model performance issues.