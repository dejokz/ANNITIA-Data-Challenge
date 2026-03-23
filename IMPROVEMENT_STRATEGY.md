# 🎯 Improvement Strategy - Post v4 Analysis

## Current State (0.8529 LB)

| Component | CV Score | Std | Status |
|-----------|----------|-----|--------|
| Death Model | 0.9153 | 0.029 | ✅ Excellent |
| Hepatic Model | 0.79 | 0.095 | ❌ Unstable |

## 🔍 Why Hepatic is Failing

### Problem 1: Stratification Issues
- Repeated CV gives different patient splits each repeat
- Folds 4-5 consistently underperform (0.68-0.77)
- Folds 2-3 consistently overperform (0.92-0.93)
- **Solution:** Use stratification by risk quartiles, not just event yes/no

### Problem 2: Too Many Features (221)
- 47 events / 221 features = severe overfitting risk
- Many EWMA/new features add noise
- **Solution:** Aggressive feature selection

### Problem 3: Normalization Issues
- Predictions are z-scores (mean=0, std=1)
- LB might expect positive risk scores
- **Solution:** Use raw predictions or rank-transform

---

## 🚀 Recommended Fixes (In Order)

### Fix 1: Feature Selection (Highest Impact)

**Keep only clinically validated features:**
```python
keep_features = [
    # LSM (most important)
    'fibs_stiffness_med_BM_1_mean',
    'fibs_stiffness_med_BM_1_max',
    'fibs_stiffness_med_BM_1_last_actual',
    'fibs_stiffness_med_BM_1_slope',
    
    # FIB-4
    'fib4_max',
    'fib4_mean',
    'fib4_last_actual',
    
    # Platelets (inverse relationship)
    'plt_min',
    'plt_mean',
    
    # Static
    'age_baseline',
    'follow_up_years',
    'gender',
    'T2DM',
]
```
**Expected:** 10-15 features → more stable model

### Fix 2: Raw Predictions (Not Normalized)

Current output is z-scores. Try raw RSF/XGB predictions.

### Fix 3: Reduce CV Variance

Use fewer repeats but stratify better:
```python
# Instead of 5×5 = 25 folds
# Use 5×3 = 15 folds with better stratification
```

### Fix 4: Ensemble with Baseline

Blend predictions:
```python
final_hepatic = 0.6 * baseline_pred + 0.4 * gemini_pred
```

---

## ⚡ Quick Test: Feature Selection

Let me create a minimal feature set version:
