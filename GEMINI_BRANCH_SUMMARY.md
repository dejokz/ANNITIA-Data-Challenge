# Gemini Recommendations Branch - Complete Summary

**Branch:** `gemni-recom`  
**Status:** Merged experiments, best model submitted (0.8529 LB)  
**Date:** March 23, 2026  

---

## 🎯 Overview

This branch implemented 5 key recommendations from the user to improve the 0.838 baseline model:

1. **RepeatedStratifiedKFold** (5 folds × 5 repeats) for stable CV
2. **XGBoost Survival** with `survival:cox` objective + monotonic constraints
3. **CoxNet (ElasticNet)** - heavily regularized linear model
4. **EWMA Features** + explicit last measurement
5. **Death-as-Feature** - use death predictions for hepatic model

**Result:** Improved from 0.838 → 0.853 (+0.015) via death model enhancement.

---

## 📁 Files Created

### Source Code
```
src/
├── pipeline_gemini.py          # Full implementation (all 5 recommendations)
├── pipeline_gemini_v2.py       # Simplified: RSF + XGB ensemble (working)
├── pipeline_gemini_v3.py       # Independent models (no death-as-feature)
├── pipeline_gemini_v4.py       # Pure RSF for hepatic
├── pipeline_gemini_v5.py       # Minimal features (15 only)
└── test_gemini_pipeline.py     # Smoke tests
```

### Submissions
```
submissions/
├── gemini_v2_submission_20260323_123051.csv    # Best: 0.8529 LB
├── gemini_v4_submission_20260323_130656.csv    # Pure RSF hepatic
├── gemini_v5_submission_20260323_131120.csv    # Minimal features
├── gemini_training_log.txt                      # v1 logs
├── gemini_training_log_v2.txt                   # v2 logs
├── gemini_v2_results_*.json                     # v2 metrics
├── gemini_v4_results_*.json                     # v4 metrics
└── gemini_v5_results_*.json                     # v5 metrics
```

### Documentation
```
DIAGNOSTICS_IMPROVEMENT.md      # Analysis of v2 results
IMPROVEMENT_STRATEGY.md         # Recommended fixes
GEMINI_BRANCH_SUMMARY.md        # This file
```

---

## 🔬 Experiments Conducted

### Experiment 1: Full Gemini Implementation (v1)

**Code:** `pipeline_gemini.py`

**Features:**
- RepeatedStratifiedKFold (5×5)
- XGBoost with `survival:cox`
- CoxNet (ElasticNet, l1_ratio=0.9)
- Rank-averaging ensemble
- EWMA features (span 3, 5)
- Explicit `last_actual` measurement
- Death-as-feature for hepatic

**Results:**
- Death: 0.9153 ✅
- Hepatic: Failed (CoxNet broke)

**Issues:**
- CoxNet consistently returned 0.5000 (random)
- NaN handling issues in CoxNet
- Feature mismatch between death/hepatic models

**Status:** ❌ Abandoned

---

### Experiment 2: Simplified Ensemble (v2) - BEST

**Code:** `pipeline_gemini_v2.py`

**Changes from v1:**
- Removed CoxNet (not working)
- Kept XGBoost + RSF ensemble
- Kept EWMA features
- Kept death-as-feature
- Simple prediction averaging (not rank-based)

**Results:**
```json
{
  "death_ci": 0.9153,
  "hepatic_ci": 0.7882,
  "death_fold_std": 0.029,
  "hepatic_fold_std": 0.092,
  "n_features": 222
}
```

**Leaderboard:** 0.8529 (+0.0148 from baseline)

**Analysis:**
- Death model excellent (0.9153 vs ~0.75 baseline)
- Hepatic model degraded (0.7882 vs 0.83 baseline)
- Death improvement > hepatic degradation
- High variance in hepatic folds (0.092 std)

**Status:** ✅ **SUBMITTED - BEST MODEL**

---

### Experiment 3: Independent Models (v3)

**Code:** `pipeline_gemini_v3.py`

**Hypothesis:** Death-as-feature might be hurting hepatic

**Changes:**
- Train death and hepatic completely independently
- No death predictions as hepatic feature

**Results:**
- Death: 0.9153 (same)
- Hepatic: 0.7921 (similar to v2's 0.7882)

**Analysis:**
- Death-as-feature has minimal impact
- Hepatic degradation is from XGBoost, not death feature

**Status:** Not submitted (similar to v2)

---

### Experiment 4: Pure RSF for Hepatic (v4)

**Code:** `pipeline_gemini_v4.py`

**Hypothesis:** XGBoost overfits hepatic (47 events)

**Changes:**
- Death: XGB+RSF ensemble (keep what works)
- Hepatic: Pure RSF only (tuned: min_samples_leaf=15)

**Results:**
- Death: 0.9153 (same)
- Hepatic: 0.7932 (barely better than 0.7882)
- Fold range: 0.68-0.93 (still high variance)

**Analysis:**
- Model choice isn't the main issue
- Problem is fundamental: 47 events too few
- High variance is structural

**Status:** Not submitted (marginal improvement)

---

### Experiment 5: Minimal Features (v5)

**Code:** `pipeline_gemini_v5.py`

**Hypothesis:** 222 features overfit hepatic model

**Changes:**
- Only 15 key clinical features:
  - LSM: mean, max, last, slope, time_high
  - FIB-4: max, mean, last, time_high_risk
  - Platelets: min, mean
  - AST/ALT: max
  - Static: age, follow_up_years

**Results:**
- Death: 0.9153 (same, used full features)
- Hepatic: 0.7427 (WORSE than 0.79)
- Only 10 features available after filtering

**Analysis:**
- Feature selection hurt performance
- Information loss from dropping features
- Hepatic needs all available signal

**Status:** ❌ Worse than v2

---

## ✅ What Worked

### 1. XGBoost for Death Model (Major Win)

**Before:** ~0.75 CV with RSF only  
**After:** 0.9153 CV with XGB+RSF ensemble  
**Gain:** +0.16 C-index

**Why it worked:**
- 76 death events (vs 47 hepatic) = more data
- XGBoost handles larger datasets well
- Monotonic constraints prevented overfitting
- Strong regularization (lambda=10, alpha=1)

**Key Code:**
```python
params = {
    'objective': 'survival:cox',
    'max_depth': 3,
    'learning_rate': 0.05,
    'reg_lambda': 10.0,  # Strong L2
    'reg_alpha': 1.0,    # L1
    'min_child_weight': 10,
}
```

---

### 2. RepeatedStratifiedKFold (Validation)

**Before:** Single 5-fold CV  
**After:** 5 folds × 5 repeats = 25 folds

**Benefits:**
- Revealed true variance (hepatic std=0.092)
- More reliable OOF predictions
- Better understanding of model stability

**Trade-off:** 5× slower training

---

### 3. EWMA Features (Neutral)

Added exponentially weighted moving averages:
- `ewma_span3`: Recent visits weighted more
- `ewma_span5`: Smoother trend

**Result:** No significant improvement, but no harm

---

### 4. Explicit Last Measurement (Neutral)

Added `last_actual` feature (actual last non-null value vs forward-filled).

**Result:** No significant improvement

---

## ❌ What Didn't Work

### 1. CoxNet (Complete Failure)

**Attempt:** Heavily regularized ElasticNet Cox model  
**Result:** 0.5000 C-index (random guessing)

**Issues:**
- Numerical instability with NaN values
- Required manual imputation
- Alpas not tuned correctly
- Never worked despite multiple fixes

**Lesson:** Skip linear models for this dataset

---

### 2. Death-as-Feature (Minimal Impact)

**Hypothesis:** Death predictions help hepatic model  
**Result:** No improvement (0.7882 vs 0.7921 without)

**Analysis:**
- Death and hepatic are competing risks
- Death model learns different patterns
- Adding death risk confuses hepatic model

---

### 3. Feature Selection (Made Things Worse)

**Attempt:** Reduce 222 → 15 features  
**Result:** Hepatic dropped from 0.79 → 0.74

**Lesson:** Hepatic model needs all available signal

---

### 4. Rank-Averaging Ensemble (Abandoned)

**Attempt:** Convert predictions to ranks (0-100) then average  
**Result:** More complex, no improvement

**Lesson:** Simple averaging works fine

---

## 📊 Key Metrics Comparison

| Model | Death CV | Hepatic CV | LB Score | Death Std | Hepatic Std |
|-------|----------|------------|----------|-----------|-------------|
| Baseline | ~0.75 | 0.83 | 0.8381 | ~0.05 | ~0.10 |
| **Gemini v2** | **0.9153** | 0.7882 | **0.8529** | **0.029** | **0.092** |
| Gemini v4 | 0.9153 | 0.7932 | - | 0.029 | 0.095 |
| Gemini v5 | 0.9153 | 0.7427 | - | 0.029 | 0.102 |

**Observations:**
- Death model: Excellent and stable
- Hepatic model: Poor and unstable
- LB score tracks average of both

---

## 🔍 Deep Dive: Why Hepatic Struggles

### Data Limitations

```
Hepatic Events: 47 patients (3.8%)
Death Events: 76 patients (7.7%)

Per fold (5-fold): ~9-10 hepatic events
Per fold (25 folds): ~2 hepatic events
```

**Problem:** With 47 events, each validation fold sees only 9-10 events. This is too few for stable C-index estimation.

### Fold Analysis (v2 Hepatic)

| Fold | C-Index | Interpretation |
|------|---------|----------------|
| 1 | 0.7831 | Below average |
| 2 | 0.9171 | Good fold |
| 3 | 0.9339 | Excellent fold |
| 4 | 0.7316 | Poor fold |
| 5 | 0.7050 | Very poor fold |

**Pattern:**
- Folds 2-3: Get "easy" patients (high C-index)
- Folds 4-5: Get "hard" patients (low C-index)
- Not random - stratification issue

### Why Baseline (0.83) Was Optimistic

**Baseline used single 5-fold CV:**
- Lucky split: folds happened to be well-balanced
- True performance: ~0.79 (as revealed by repeated CV)
- Single CV variance: 0.66-0.92 range

**Repeated CV (v2):**
- 25 folds reveal true distribution
- Mean: 0.7882
- More honest estimate

---

## 🛠️ Technical Implementation Notes

### XGBoost Survival Setup

```python
# Labels for survival:cox
# y > 0: event occurred at time y
# y < 0: censored at time -y
labels = np.where(event, time, -time)

# Monotonic constraints
# 1 = increasing risk (higher feature = higher risk)
# -1 = decreasing risk (higher feature = lower risk)
constraints = {
    'fibs_stiffness_*': 1,   # Higher LSM = worse
    'fib4_*': 1,             # Higher FIB-4 = worse
    'plt_*': -1,             # Lower platelets = worse
}
```

### Handling NaN Values

**RSF:** Handles NaN natively  
**XGBoost:** Requires imputation  
**CoxNet:** Requires imputation (was buggy)

**Solution:** Median imputation + StandardScaler

### Feature Alignment

**Critical bug discovered:** Death and hepatic models had different feature sets due to different filtering.

**Fix:** Always align features between train/test and between models.

---

## 🎯 Key Insights

### 1. Dataset Imbalance

```
Challenge evaluates: average of hepatic + death C-indices
Death: 76 events (easier to predict)
Hepatic: 47 events (harder to predict)
```

**Strategy:** Focus improvements on death model (more impact)

### 2. Variance vs Bias Trade-off

| Model | Bias | Variance | Best For |
|-------|------|----------|----------|
| RSF | Higher | Lower | Small datasets (hepatic) |
| XGBoost | Lower | Higher | Large datasets (death) |

### 3. CV Strategy Matters

- **Single CV:** Optimistic, high variance in scores
- **Repeated CV:** Honest, but 5× slower
- **Stratification:** Critical for rare events

### 4. Feature Engineering Limits

- EWMA: Neutral
- Last measurement: Neutral
- Cross-NIT features: Minimal impact
- Delta features: Previously known to hurt

**Lesson:** Domain knowledge features didn't help

---

## 🚀 Recommendations for Future Work

### Immediate (High Confidence)

1. **Blend v2 with baseline**
   ```python
   final = 0.7 * gemini_v2 + 0.3 * baseline
   ```
   Expected: +0.005-0.01 LB

2. **Optimize ensemble weights**
   - Death: XGB(0.7) + RSF(0.3)
   - Hepatic: RSF(0.9) + XGB(0.1)

### Medium-Term (Medium Confidence)

3. **XGBoost hyperparameter tuning**
   - Use Optuna for hepatic XGBoost
   - Focus on heavy regularization

4. **Try LightGBM**
   - Native NaN handling
   - Different boosting strategy

### Long-Term (Low Confidence)

5. **DeepSurv or Neural Networks**
   - Risk: More overfitting
   - Potential: Better feature interactions

6. **Multi-task learning**
   - Joint model for death + hepatic
   - Share representations

---

## 📈 Gap to Top 3

**Current:** 0.8529  
**Target:** 0.90+  
**Gap:** 0.047

**Realistic maximum:** 0.87-0.88 (with blending + tuning)

**To reach 0.90:** Would need:
- Novel feature engineering (unlikely)
- External data (not allowed)
- Lucky ensemble

---

## ✅ Final Status

**Best Model:** Gemini v2  
**LB Score:** 0.8529  
**Submitted:** Yes (ID: 22835)  
**Improvement:** +0.0148 over baseline  
**Branch Status:** Ready for merge or further experiments

---

## 🔗 Related Files

- Best submission: `submissions/gemini_v2_submission_20260323_123051.csv`
- Best results: `submissions/gemini_v2_results_20260323_123051.json`
- Best code: `src/pipeline_gemini_v2.py`
- Diagnostics: `DIAGNOSTICS_IMPROVEMENT.md`
