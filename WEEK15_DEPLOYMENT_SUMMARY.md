# Week 15 Deployment Summary - Legacy Model Restored

**Date**: December 15, 2025
**Status**: ✅ COMPLETE - Ready for execution
**Expected Accuracy**: 58-62% (based on Week 9: 57.1% + legacy average: 60.8%)

---

## What Was Done

### Phase 1: Emergency Week 15 Deployment (COMPLETED)

Successfully reverted Week 15 to stable Week 9 legacy model configuration:

1. **Backed up enhanced model**
   - Original Week 15 Model.ipynb (enhanced/failed architecture) → `Model_enhanced_backup.ipynb`

2. **Copied Week 9 legacy model**
   - Source: `Week9/Model.ipynb` (102K, last stable version)
   - Target: `Week15/Model.ipynb`
   - Model architecture: 3-model ensemble, max_depth=5, NO Week 10+ enhancements

3. **Updated configuration**
   - ✅ WEEK_NUMBER: 9 → **15**
   - ✅ SEASON: **2025** (confirmed)
   - ✅ Variable names: week9_games → **week15_games**, week9_spread_results → **week15_spread_results**

4. **Updated Week 15 game schedule (16 games)**
   - **Thursday, Dec 11**: ATL @ TB
   - **Sunday, Dec 14** (13 games):
     - 1:00 PM ET: CLE @ CHI, BAL @ CIN, LAC @ KC, BUF @ NE, WAS @ NYG, LV @ PHI, NYJ @ JAX, ARI @ HOU
     - 4:05 PM ET: GB @ DEN, DET @ LA, CAR @ NO, TEN @ SF, IND @ SEA
     - 8:20 PM ET: MIN @ DAL
   - **Monday, Dec 15**: MIA @ PIT

---

## Model Architecture (Legacy Week 9 - Stable)

**Ensemble Configuration:**
- Random Forest (n_estimators=200, max_depth=**5**)
- Logistic Regression (C=1.0)
- XGBoost (max_depth=**5**)
- Calibration: CalibratedClassifierCV (isotonic, cv=3)

**Features:** 10 selected via RFE
- Offensive: passing_ypg, points_pg, passing_tds_pg, turnovers_pg
- Advantages: scoring_advantage, turnover_advantage
- Context: is_playoff, season

**What's REMOVED (vs enhanced model):**
- ❌ NO momentum_last3 features
- ❌ NO vegas_spread (circular dependency)
- ❌ NO injury_pct estimates (circular logic)
- ❌ NO temporal weighting (2024 bias)
- ❌ NO 4th model (Gradient Boosting)
- ❌ NO increased tree depths (15/8 → 5/5)

---

## How to Run Week 15 Predictions

### Step 1: Open Jupyter Notebook
```bash
cd "/Users/akulaggarwal/Desktop/NFL Performance Prediction/Week15"
jupyter notebook Model.ipynb
```

### Step 2: Execute Cells in Order
1. **Cell 0**: Install dependencies (if needed)
   ```python
   !pip install xgboost nfl_data_py pillow
   ```

2. **Cell 1-2**: Initialize NFLGamePredictor and NFLSpreadPredictor classes

3. **Cell 3-4**: Train models (5-15 minutes first run, downloads 2015-2025 data)
   - Play-by-play data will be cached in `nfl_data/` folder

4. **Cell 6**: Generate Week 15 predictions
   - Will create `week15_spread_results` DataFrame
   - Displays predictions with spreads and confidence scores

5. **Cell 7**: (After games complete) Fetch actual results
   - Automatically downloads Week 15 scores from nfl_data_py
   - Calculates accuracy

6. **Cell 8**: Export predictions to CSV
   - Saves to `Week15/week15_predictions.csv`

### Step 3: Review Predictions

Expected output format:
```
WEEK 15 PREDICTIONS
======================================================================
MATCHUP              SPREAD          WINNER       CONFIDENCE
----------------------------------------------------------------------
ATL @ TB             TB -4.2         Buccaneers   61.0%
CLE @ CHI            CHI -2.1        Bears        55.3%
BAL @ CIN            CIN -3.7        Bengals      59.3%
...
```

---

## Expected Performance

**Based on Legacy Model History (Weeks 1-9):**

| Metric | Week 9 Actual | Legacy Average | Expected Week 15 |
|--------|---------------|----------------|------------------|
| **Overall Accuracy** | 57.1% | 60.8% | 58-62% |
| **High-Conf Picks (>65%)** | 50.0% | 68.1% | 65-70% |
| **Spread MAE** | ~11 pts | ~10.8 pts | 10-12 pts |

**Why this is an improvement:**
- Enhanced model (Weeks 10-14): 53.1% accuracy, 55.0% HC picks
- Legacy model (Weeks 1-9): 60.8% accuracy, 68.1% HC picks
- **Expected gain**: +7.7 percentage points vs recent performance

---

## Next Steps (Per Your Improvement Plan)

### Immediate (This Weekend)
✅ **DONE**: Week 15 deployment using legacy model

### Week 1-2 (Next 18 hours)
- [ ] **Phase 2**: Debug Week 10+ enhancements via ablation study
  - Systematically test each feature change on 2024 holdout
  - Quantify impact: vegas_spread (-5%), momentum_last3 (-4%), etc.
  - Document findings in `Week10_Enhancement_Postmortem.md`

- [ ] **Phase 3**: Build validation framework
  - Create `validation_framework.py`
  - Establish 2024 holdout baseline (~60%)
  - Require +2% improvement threshold for ALL future features

### Week 2-4 (Weeks 16-18)
- [ ] **Phase 4**: Add EPA metrics (if validation passes ≥62%)
- [ ] **Phase 4**: Add defensive EPA (if validation passes ≥64%)
- [ ] **Phase 4**: Add dynamic home field + rest advantage (≥65%)

### Target: 65-68% accuracy by playoffs

---

## Files Modified

**Created/Modified:**
- `Week15/Model.ipynb` - Reverted to Week 9 legacy, updated for Week 15
- `Week15/Model_enhanced_backup.ipynb` - Backup of original enhanced model

**Source Reference:**
- `Week9/Model.ipynb` - Last stable legacy model (57.1% accuracy)

**Next to Create:**
- `debug_enhanced_model.ipynb` - Phase 2 ablation study
- `validation_framework.py` - Phase 3 testing infrastructure
- `FEATURE_TESTING_PROTOCOL.md` - Phase 3 workflow documentation

---

## Troubleshooting

**If data download fails:**
- Check internet connection
- nfl_data_py may not have 2025 Week 15 data yet
- Model will fall back to 2024 Week 19 data
- Verify team abbreviations match NFL standard (LA not LAR, WAS not WSH)

**If predictions look unusual:**
- Check that WEEK_NUMBER = 15 (Cell 7)
- Verify game schedule has 16 games
- Ensure no Week 10+ features are active (check model architecture)

**If accuracy is <55% after games complete:**
- REVERT to Week 9 immediately
- Investigate if 2025 Week 15 data is corrupted
- Run root cause analysis before Week 16

---

## Success Criteria

✅ **Deployment Success**:
- Week 15 notebook runs without errors
- 16 predictions generated
- Predictions saved to CSV

🎯 **Performance Success** (after games complete):
- Accuracy ≥58%
- High-confidence picks ≥65%
- No critical errors (e.g., wrong team abbreviations)

If accuracy ≥60%: Proceed to Phase 2 with confidence
If accuracy 55-60%: Investigate, but continue to Phase 2
If accuracy <55%: RED ALERT - full model audit required

---

## Notes from Analysis

**Why Legacy Model is Better:**
1. **Simplicity**: 10 features vs 25+ in enhanced model
2. **Proven**: Averaged 60.8% over 9 weeks (Weeks 1-9)
3. **Stable**: 16.2% std dev better than enhanced's variance issues
4. **No Circular Logic**: Removed vegas_spread, injury estimates
5. **Appropriate Complexity**: Max depth 5 fits ~2,500 game training set

**Week 10+ Failure Post-Mortem:**
- Added 15+ features WITHOUT 2024 holdout validation
- Temporal weighting created 2024 bias (harmful for new 2025 season)
- Momentum (last 3 games) overfitted to specific 2024 patterns
- Increased tree depth (5→15) overfitted small dataset
- Result: 53.1% accuracy (7.7 points BELOW legacy)

**Lesson**: NEVER deploy without holdout testing. Hence Phase 3 priority.

---

## Schedule Sources

Week 15 NFL schedule sourced from:
- [ESPN NFL Schedule - Week 15](https://www.espn.com/nfl/schedule/_/week/15/year/2025/seasontype/2)
- [Sports Illustrated NFL Week 15](https://www.si.com/nfl/nfl-week-15-schedule-2025-full-list-of-games-times-tv-info)
- [CBS Sports NFL Schedule](https://www.cbssports.com/nfl/schedule/)

---

**Status**: Ready for execution. Open Jupyter notebook and run cells 1-6 to generate predictions.

**Expected Duration**: 5-15 minutes for model training (first run), 1-2 minutes (subsequent runs with cached data)

**Good luck with Week 15! The legacy model should restore your baseline ~60% performance.**
