# Week 10+ Enhancement Postmortem: Ablation Study Results

**Test Date**: 2025-12-15
**Holdout Dataset**: 2024 NFL Season (Weeks 1-18)
**Baseline**: Week 9 Legacy Model (60.8% expected)

---

## Executive Summary

- **Baseline Accuracy**: 66.8%
- **Best Configuration**: 4. Baseline + Vegas Spread (68.0%)
- **Worst Configuration**: 7. Baseline + Increased Depth + 4th Model (62.9%)
- **Full Week 10 Performance**: 65.2%

## Detailed Results

| Configuration | Accuracy | Delta | HC Accuracy | Brier | AUC | Games |
|---------------|----------|-------|-------------|-------|-----|-------|
| 4. Baseline + Vegas Spread | 68.0% | +1.2% | 86.7% | 0.222 | 0.715 | 256 |
| 3. Baseline + Momentum Features | 67.6% | +0.8% | 92.3% | 0.221 | 0.724 | 256 |
| 2. Baseline + Temporal Weighting | 67.2% | +0.4% | 76.0% | 0.221 | 0.715 | 256 |
| 1. BASELINE (Week 9 Legacy) | 66.8% | +0.0% | 73.3% | 0.221 | 0.718 | 256 |
| 8. FULL WEEK 10+ (All Enhancements) | 65.2% | -1.6% | 85.7% | 0.219 | 0.717 | 256 |
| 5. Baseline + Injury Estimates | 63.7% | -3.1% | 82.4% | 0.224 | 0.702 | 256 |
| 6. Baseline + Defensive Stats | 62.9% | -3.9% | 83.3% | 0.223 | 0.701 | 256 |
| 7. Baseline + Increased Depth + 4th Model | 62.9% | -3.9% | 80.8% | 0.225 | 0.700 | 256 |

## Feature Impact Summary

- **Baseline + Momentum Features**: +0.8% - ⚪ NEUTRAL
- **Baseline + Temporal Weighting**: +0.4% - ⚪ NEUTRAL
- **BASELINE (Week 9 Legacy)**: +0.0% - ⚪ NEUTRAL
- **FULL WEEK 10+ (All Enhancements)**: -1.6% - ❌ HARMFUL
- **Baseline + Injury Estimates**: -3.1% - ❌ HARMFUL
- **Baseline + Defensive Stats**: -3.9% - ❌ HARMFUL
- **Baseline + Increased Depth + 4th Model**: -3.9% - ❌ HARMFUL

## Recommendations

Based on ablation study results:

### Remove These Features:
- FULL WEEK 10+ (All Enhancements) (-1.6%)
- Baseline + Injury Estimates (-3.1%)
- Baseline + Defensive Stats (-3.9%)
- Baseline + Increased Depth + 4th Model (-3.9%)

### Next Steps:
1. Revert to baseline for Week 16
2. Add ONLY validated helpful features one at a time
3. Test each on 2024 holdout before deploying to 2025
4. Build validation framework to prevent future regressions
