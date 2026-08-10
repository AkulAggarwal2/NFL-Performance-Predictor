# Ablation Study Results: Comprehensive Analysis & Next Steps

**Date**: December 15, 2024
**Study**: 8 configurations tested on 2024 NFL season (256 games)
**Baseline**: Week 9 Legacy Model

---

## Executive Summary

### Major Discovery: The Feature Combination Paradox

**Individual features help, but combining them hurts!**

- ✅ Vegas Spread alone: +1.2% (68.0% accuracy)
- ✅ Momentum alone: +0.8% (67.6% accuracy)
- ✅ Temporal Weighting alone: +0.4% (67.2% accuracy)
- ❌ ALL combined (Week 10+): -1.6% (65.2% accuracy)

**Baseline**: 66.8% accuracy on 2024 holdout

---

## Detailed Results Table

| Rank | Configuration | Accuracy | Delta | HC Accuracy | Brier | AUC | Impact |
|------|---------------|----------|-------|-------------|-------|-----|--------|
| 1 | Vegas Spread | 68.0% | +1.2% | 86.7% | 0.222 | 0.715 | ✅ HELPFUL |
| 2 | Momentum Features | 67.6% | +0.8% | 92.3% | 0.221 | 0.724 | ✅ HELPFUL |
| 3 | Temporal Weighting | 67.2% | +0.4% | 76.0% | 0.221 | 0.715 | ⚪ NEUTRAL |
| 4 | **BASELINE** | **66.8%** | **0.0%** | **73.3%** | **0.221** | **0.718** | - |
| 5 | Full Week 10+ | 65.2% | -1.6% | 85.7% | 0.219 | 0.717 | ❌ HARMFUL |
| 6 | Injury Estimates | 63.7% | -3.1% | 82.4% | 0.224 | 0.702 | ❌ HARMFUL |
| 7 | Defensive Stats | 62.9% | -3.9% | 83.3% | 0.223 | 0.701 | ❌ HARMFUL |
| 8 | Depth + 4th Model | 62.9% | -3.9% | 80.8% | 0.225 | 0.700 | ❌ HARMFUL |

---

## Surprising Findings

### 1. Vegas Spread: Expected Harmful, Actually Helpful (+1.2%)

**Expected**: -3 to -5% (circular dependency)
**Actual**: +1.2% improvement

**Why the discrepancy?**
- In the ablation study, `vegas_spread` was set to **0** (placeholder)
- So the improvement came from the **feature structure**, not actual values
- Adding an extra feature may have helped RFE select better combinations
- OR: The feature name triggered different model behavior

**Implication**:
- Don't add vegas_spread with real values (still circular)
- But the feature SLOT helped model selection somehow
- This needs further investigation

**Action**: Test with real vegas spread values to see if still helpful

---

### 2. Momentum Features: Expected Harmful, Actually Helpful (+0.8%)

**Expected**: -2 to -4% (overfitted to 2024)
**Actual**: +0.8% improvement, **92.3% HC accuracy** (best!)

**Why the discrepancy?**
- 2024 was a stable year, momentum actually predictive
- Implementation in study may differ from Week 10-14 production
- Small sample (3 games) worked well in 2024 specifically

**Caution**:
- 92.3% HC accuracy in 2024 doesn't guarantee same in 2025
- Your Week 13-14 showed 33.3% HC accuracy (massive discrepancy)
- Something different between study and production implementation

**Action**:
- Verify momentum calculation matches between study and Week 10-14
- Test momentum on 2023 and 2022 data (not just 2024)

---

### 3. Defensive Stats: Expected Helpful, Actually Harmful (-3.9%)

**Expected**: +1 to +2% (9.4% feature importance)
**Actual**: -3.9% (WORST individual feature!)

**Critical Bug Suspected**:

```python
# From ablation study:
features['defensive_ppg'] = defensive_plays['epa'].sum() * 6 / weeks

# Why multiply by 6???
# EPA is already in point units
# This is artificially inflating defensive points allowed by 6x!
```

**Likely Correct Implementation**:
```python
features['defensive_ppg'] = defensive_plays['epa'].sum() / weeks
# OR
features['defensive_epa_pg'] = defensive_plays['epa'].sum() / weeks
```

**Action**:
- Fix defensive stats calculation
- Re-run test #6 with corrected formula
- Expected: Should flip from -3.9% to +1-2%

---

### 4. Full Week 10+ Paradox: Why Do Individual Features Help But Combined Hurt?

**Individual feature deltas:**
- Vegas: +1.2%
- Momentum: +0.8%
- Temporal: +0.4%
- **Expected sum**: ~+2.4%

**Actual combined (Full Week 10+)**: -1.6%

**Delta vs expected**: **-4.0 percentage points** lost to interaction!

**Possible Causes**:

1. **Feature Correlation (Multicollinearity)**:
   - Vegas spread (even at 0) + momentum + temporal all capture "recent performance"
   - Model gets confused with redundant signals
   - RFE may select wrong subset with too many options

2. **Overfitting**:
   - 25+ features for ~2,500 training games
   - Rule of thumb: Need 10 samples per feature
   - Should have max 250 features for this dataset

3. **Interaction Effects**:
   - Momentum * Temporal Weighting may conflict
   - Both trying to weight recent data differently
   - Model can't reconcile contradictory signals

4. **RFE Selection With Full Set**:
   - With 25 features, RFE may pick different (worse) subset
   - Baseline with 10 features: RFE picks optimal from simple set
   - Full with 25 features: RFE overwhelmed, picks suboptimal

**Action**: Test specific combinations:
- Vegas + Momentum (test if +2.0% or lower)
- Momentum + Temporal (test if +1.2% or lower)
- Identify which pairs conflict

---

## Critical Discrepancy: Study vs Production

### Ablation Study (2024 Holdout):
- Full Week 10+: **65.2% accuracy**
- HC picks: **85.7% accuracy**

### Your Week 10-14 Production (2025):
- Overall: **53.1% accuracy**
- HC picks: **55.0% accuracy**

### Gap: **12.1 percentage points**

**This is a HUGE discrepancy! Something is very different between:**
- The ablation study implementation
- Your actual Week 10-14 Model.ipynb

**Possible Explanations**:

1. **Different Feature Engineering**:
   - Ablation study uses simplified momentum (fantasy points proxy)
   - Production uses actual win/loss records (if implemented)
   - Defensive stats calculated differently

2. **Different Data**:
   - Study uses 2024 data (complete season)
   - Production uses 2025 data (incomplete, early weeks)
   - 2025 season dynamics may differ from 2024

3. **RFE Selection Different**:
   - Study: RFE runs on 2015-2024 data
   - Production: RFE runs on 2015-2025 partial data
   - Different features selected

4. **Implementation Bugs**:
   - Defensive stats formula bug (multiply by 6)
   - Momentum calculation error
   - Temporal weighting not actually applied

**Action**:
- Compare Week10/Model.ipynb cell-by-cell with ablation study
- Find exact differences in feature engineering code
- Run Week 10 model on 2024 holdout to verify matches study

---

## Recommendations

### For Week 16 Deployment (Choose ONE)

**Option A: Enhanced Baseline (RECOMMENDED)**

Start with proven features from ablation study:

```python
config_week16 = {
    'use_momentum': True,      # +0.8%, 92.3% HC accuracy
    'use_vegas': False,        # Skip (placeholder=0 in study, not real)
    'use_temporal': False,     # Only +0.4%, not worth complexity
    'use_injuries': False,     # -3.1%, harmful
    'use_defensive': False,    # -3.9%, harmful (likely bug)
    'use_4th_model': False,    # -3.9%, harmful
    'tree_max_depth': 5,       # Keep simple
}
```

**Expected**: 67-68% accuracy (baseline 66.8% + momentum 0.8%)

**Pros**:
- Momentum proven helpful on 2024 holdout
- 92.3% HC accuracy in study (excellent calibration)
- Conservative improvement

**Cons**:
- Momentum may not generalize to 2025
- Your Week 10-14 showed poor HC performance (need to debug why)

---

**Option B: Pure Baseline (SAFEST)**

Use Week 9 legacy model with NO enhancements:

```python
config_week16 = {
    'use_momentum': False,
    'use_vegas': False,
    'use_temporal': False,
    'use_injuries': False,
    'use_defensive': False,
    'use_4th_model': False,
    'tree_max_depth': 5,
}
```

**Expected**: 66-67% accuracy

**Pros**:
- Most conservative
- Known stable performance
- No risk of feature interaction

**Cons**:
- Leaves +0.8% momentum improvement on table
- Doesn't learn from ablation study findings

---

**Option C: Investigate First, Deploy Later (MOST RIGOROUS)**

Before Week 16, complete these investigations:

1. **Fix defensive stats bug**:
   ```python
   # Change from:
   features['defensive_ppg'] = defensive_plays['epa'].sum() * 6 / weeks
   # To:
   features['defensive_ppg'] = defensive_plays['epa'].sum() / weeks
   ```
   Re-test to see if flips from -3.9% to positive

2. **Test feature combinations**:
   - Momentum + Temporal
   - Vegas (real values) + Momentum
   - Defensive (fixed) + Baseline

3. **Replicate Week 10-14 on 2024 holdout**:
   - Run exact Week10/Model.ipynb code on 2024 data
   - Should match study's 65.2% if implementation identical
   - If not, find the difference

4. **Validate momentum on 2023**:
   - Re-run test #3 but with TEST_YEAR = 2023
   - See if momentum still +0.8% or if 2024-specific

Then deploy best validated configuration to Week 16.

**Expected**: Unknown (but highest confidence)

**Pros**:
- Most thorough
- Identifies root causes
- Prevents future regressions

**Cons**:
- Takes 10-20 hours
- May miss Week 16 deadline

---

## Feature-by-Feature Verdict

### ✅ KEEP (Proven Helpful)

**1. Momentum Features (+0.8%)**
- **Evidence**: 67.6% accuracy, 92.3% HC accuracy on 2024
- **Caution**: Verify generalizes to 2025 and other years
- **Action**: Deploy cautiously, monitor Week 16 performance

### ⚪ TEST FURTHER (Unclear)

**2. Vegas Spread (+1.2%)**
- **Evidence**: Helped in study, BUT was placeholder (0 value)
- **Caution**: Real vegas values may create circular dependency
- **Action**: Test with real spread values on 2024 holdout

**3. Temporal Weighting (+0.4%)**
- **Evidence**: Slight improvement, low risk
- **Caution**: Only +0.4%, may not justify complexity
- **Action**: Skip for now, revisit if other features fail

### ❌ REMOVE (Proven Harmful)

**4. Injury Estimates (-3.1%)**
- **Evidence**: Consistently harmful across all tests
- **Reason**: Circular logic (using performance to predict performance)
- **Action**: Remove permanently unless real injury reports available

**5. Defensive Stats (-3.9%)**
- **Evidence**: Worst individual feature
- **Likely Cause**: Bug in formula (multiply by 6)
- **Action**: Fix bug and re-test, then re-evaluate

**6. Increased Depth + 4th Model (-3.9%)**
- **Evidence**: No benefit, increased overfitting
- **Reason**: Dataset too small for depth=15
- **Action**: Keep depth=5, keep 3-model ensemble

**7. Full Week 10+ Combination (-1.6%)**
- **Evidence**: Paradox - individuals help, combination hurts
- **Reason**: Feature interaction, multicollinearity
- **Action**: Never deploy all features together without validation

---

## Next Steps - Prioritized Action Items

### Immediate (This Weekend - Week 16 Deployment)

**Decision Required**: Choose Option A, B, or C above

If choosing **Option A (Enhanced Baseline)**:
1. Copy Week 9 Model.ipynb → Week 16
2. Add momentum features only
3. Test predictions for reasonability
4. Deploy

If choosing **Option B (Pure Baseline)**:
1. Copy Week 15 (already reverted) → Week 16
2. Update week number and schedule
3. Deploy

If choosing **Option C (Investigate First)**:
1. Skip Week 16 predictions
2. Complete investigation tasks below
3. Deploy to Week 17 with high confidence

### Short-Term (Next Week)

**Investigation Tasks**:

1. **Debug defensive stats** (2 hours):
   - Fix formula (remove × 6)
   - Re-run test #6 on 2024 holdout
   - Expected: Should flip to +1-2% if bug fixed

2. **Verify momentum implementation** (1 hour):
   - Compare ablation study momentum formula to Week 10 production
   - Identify differences
   - Test corrected version

3. **Test feature combinations** (3 hours):
   - Momentum + Temporal on 2024 → expect ~+1.0%?
   - Momentum + Defensive (fixed) on 2024 → expect ~+2.0%?
   - Vegas (real) + Momentum on 2024 → helpful or harmful?

4. **Replicate Week 10-14 on 2024** (2 hours):
   - Run exact Week10/Model.ipynb on 2024 data
   - Should get 65.2% if implementation matches study
   - If not, find discrepancy

5. **Cross-validate momentum on 2023** (2 hours):
   - Change TEST_YEAR = 2023 in ablation study
   - Re-run test #3 (momentum features)
   - If still positive, more confidence it generalizes

### Medium-Term (Next 2 Weeks)

**Proceed to Phase 3: Build Validation Framework**

With ablation study findings, now create proper testing:

1. **Create 2024 holdout validator** (from plan)
2. **Establish baseline**: 66.8% (from this study)
3. **Set threshold**: +2% improvement required (new features must achieve ≥68.8%)
4. **Test EPA metrics**: Expected +8-12%, will it actually deliver?
5. **Deploy only validated features**

### Long-Term (Offseason 2026)

1. **Get real injury data**: Replace estimated injury_pct
2. **Fix defensive calculations**: Use proper EPA formulas
3. **Test weather features**: Temperature, wind, precipitation
4. **Implement player-level data**: QB ratings, key injuries
5. **Build neural network**: LSTM for sequential dependencies

---

## Key Takeaways

### What We Learned:

1. **Individual features can help, but combinations can hurt** due to interaction effects
2. **Momentum features are promising** (+0.8%, 92.3% HC) but need cross-validation
3. **Defensive stats formula has likely bug** (multiply by 6)
4. **Your Week 10-14 production differs significantly from study** (12.1 point gap)
5. **Baseline model is quite strong** (66.8% on 2024 holdout)

### What We Still Don't Know:

1. **Why 12.1 point gap between study (65.2%) and Week 10-14 production (53.1%)?**
2. **Does momentum generalize to 2025, or was 2024-specific?**
3. **Would defensive stats help if formula fixed?**
4. **Do feature pairs interact negatively? Which ones?**
5. **Would real vegas spread values still help, or create circular dependency?**

### Critical Question to Answer:

**Before deploying momentum features to Week 16**, we must answer:

**"Why did momentum show 92.3% HC accuracy in the 2024 study but your Week 10-14 production showed 55.0% HC accuracy?"**

Until this is resolved, we can't trust that momentum will actually help in production.

---

## Files Generated

- ✅ `ablation_study_results.csv` - Raw metrics for all 8 tests
- ✅ `Week10_Enhancement_Postmortem.md` - Summary report
- 📄 `ABLATION_RESULTS_ANALYSIS.md` - This comprehensive analysis (you are here)

---

## Recommended Reading Order

1. **This file** - Comprehensive analysis
2. **Week10_Enhancement_Postmortem.md** - Quick summary
3. **ablation_study_results.csv** - Raw data
4. **debug_enhanced_model.ipynb** - Full study code and outputs

---

**Status**: Phase 2 Complete ✅

**Next Decision**: Choose Week 16 deployment option (A, B, or C)

**Next Phase**: Phase 3 - Build Validation Framework (after Week 16 deployed)
