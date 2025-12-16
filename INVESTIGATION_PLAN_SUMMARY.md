# Investigation Plan Summary

**Status**: Running investigations to solve ablation study mysteries
**Date**: December 16, 2024

---

## What We Know from Ablation Study

### The Paradox
- **Individual features help**, but **combining them hurts**!
- Vegas Spread alone: +1.2% (68.0% accuracy)
- Momentum alone: +0.8% (67.6% accuracy, 92.3% HC!)
- Temporal alone: +0.4% (67.2% accuracy)
- **Expected if combined**: ~+2.4%
- **Actual combined (Full Week 10+)**: -1.6% (65.2% accuracy)
- **Lost to interaction**: -4.0 percentage points

### Critical Findings

**1. Defensive Stats: WORST Individual Feature (-3.9%)**
- Baseline: 66.8%
- With Defensive Stats: 62.9%
- **Suspected Bug**: `defensive_ppg = epa.sum() * 6 / weeks` (Why × 6?)
- **Hypothesis**: Should be `defensive_epa_pg = epa.sum() / weeks`
- **Expected if fixed**: -3.9% → +1 to +2%

**2. Momentum: BEST HC Accuracy (92.3%)**
- Baseline HC: 73.3%
- Momentum HC: 92.3% (AMAZING!)
- Overall: +0.8% improvement
- **BUT**: Week 10-14 production showed only 55.0% HC accuracy
- **Mystery**: Why the 37.3 point discrepancy?

**3. The 12.1 Point Gap**
- Ablation study (2024 holdout): 65.2%
- Week 10-14 production (2025): 53.1%
- **Gap**: 12.1 percentage points
- **Possible causes**:
  - Different feature implementations
  - Different data (2024 vs 2025)
  - Different RFE feature selection
  - Implementation bugs in production

---

## Investigations Running

### Investigation 1: Find the 12.1 Point Gap
**Status**: Notebook created
**Purpose**: Compare Week10/Model.ipynb to ablation study code
**Hypotheses to test**:
1. Different momentum formula (wins vs fantasy points)
2. Defensive stats bug in production too
3. 2024 vs 2025 data differences
4. RFE selected different features
5. Temporal weighting not applied correctly

### Investigation 2: Fix Defensive Stats Bug
**Status**: Running initial test
**Purpose**: Validate if × 6 bug is root cause
**Test**:
- Compare buggy: `epa × 6 / weeks`
- Against fixed: `epa / weeks`
- Measure delta on 2024 holdout

**Expected**: If bug confirmed, -3.9% should flip to +1-2%

### Investigation 3: Validate Momentum on 2023
**Status**: Notebook created
**Purpose**: Test if momentum generalizes beyond 2024
**Test**:
- Run momentum test on 2023 data
- Run on 2024 for comparison
- Check if +0.8% and 92.3% HC persist

**Decision criteria**:
- If helps on both years → Deploy ✅
- If hurts on 2023 → Don't deploy ❌
- If mixed → Test on 2022 too ⚠️

### Investigation 4: Feature Pair Interactions
**Status**: Notebook created (basic structure)
**Purpose**: Identify synergistic vs conflicting pairs
**Pairs to test**:
1. Momentum + Temporal (both weight recent data - conflict?)
2. Momentum + Defensive (fixed)
3. Momentum + Vegas
4. Temporal + Defensive (fixed)
5. Vegas + Defensive (fixed)

**Interaction types**:
- **Synergistic**: A+B > delta_A + delta_B ✅
- **Additive**: A+B ≈ delta_A + delta_B ➡️
- **Conflicting**: A+B < delta_A + delta_B ❌

---

## Expected Outcomes

### If Investigations Confirm:

**1. Defensive Bug Fixed (+3-5 points)**
- Current: -3.9% harmful
- Fixed: +1 to +2% helpful
- **Total swing**: +4.9 to +5.9 points
- **Deploy**: Yes, with fixed formula

**2. Momentum Validated (+0.8 points)**
- If generalizes to 2023 and 2024
- 92.3% HC accuracy maintained
- **Deploy**: Yes, add to Week 16

**3. Safe Feature Pairs Identified**
- Avoid conflicting combinations
- Use only synergistic/additive pairs
- **Deploy**: Optimal non-conflicting set

**Combined Expected Improvement**:
- Baseline: 66.8%
- + Defensive (fixed): +1.5% → 68.3%
- + Momentum: +0.8% → 69.1%
- **Target**: 68-70% accuracy for Week 16

### If Investigations Reveal:

**1. Defensive Bug NOT Root Cause**
- Other implementation issue
- **Action**: Skip defensive stats entirely

**2. Momentum Doesn't Generalize**
- Year-specific overfitting to 2024
- **Action**: Don't deploy, stick with baseline

**3. All Pairs Conflict**
- Features inherently incompatible
- **Action**: Deploy best single feature only

---

## Quick Decision Matrix for Week 16

| Scenario | Defensive Fixed | Momentum Valid | Action |
|----------|----------------|----------------|--------|
| Best case | ✅ Yes | ✅ Yes | Deploy both (+2.3%) |
| Defensive only | ✅ Yes | ❌ No | Deploy defensive only (+1.5%) |
| Momentum only | ❌ No | ✅ Yes | Deploy momentum only (+0.8%) |
| Worst case | ❌ No | ❌ No | Stick with baseline (66.8%) |

---

## Current Status

**Running Now**: Loading NFL data (2015-2024) and creating investigation cache
**ETA**: 2-3 minutes
**Next**: Run defensive stats bug test
**Then**: Validate momentum on 2023
**Finally**: Test feature pair interactions

**Total Investigation Time**: 4-6 hours for all tests

---

## Files Created

1. ✅ `investigation_1_gap_analysis.ipynb` - Find 12.1 point gap
2. ✅ `investigation_2_fix_defensive_bug.ipynb` - Test defensive fix
3. ✅ `investigation_3_validate_momentum_2023.ipynb` - Cross-year validation
4. ✅ `investigation_4_feature_pairs.ipynb` - Interaction testing
5. ⏳ `ablation_cache/` - NFL data cache (creating now)

---

**Next Update**: After NFL data loaded and defensive bug test complete
