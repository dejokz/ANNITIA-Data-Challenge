# 🔍 Gemini v2 Diagnostic Analysis

## 📊 Current Performance (0.8529 LB Score)

| Model | Death CV | Hepatic CV | LB Score | Notes |
|-------|----------|------------|----------|-------|
| **Gemini v2** | **0.9153** | 0.7882 | **0.8529** | RSF + XGB ensemble |
| Baseline | ~0.75 | 0.83 | 0.8381 | RSF only |
| **Improvement** | **+0.16** | **-0.04** | **+0.015** | Death model dominates |

**Key Insight:** Death model improvement (+0.16) more than compensates for hepatic decline (-0.04).

---

## 🎯 Diagnostic Analysis

### 1. Death Model (Excellent)

```
OOF C-index: 0.9153 (+/- 0.029)
Fold Range: 0.859 - 0.959 (very stable)
```

**Why it works:**
- 76 events (more data than hepatic's 47)
- XGBoost excels with more samples
- Low variance (std=0.029) = reliable

### 2. Hepatic Model (Problematic)

```
OOF C-index: 0.7882 (+/- 0.092)
Fold Range: 0.705 - 0.934 (highly unstable!)
```

**Problems identified:**
- **High variance** (std=0.092 vs death's 0.029)
- **Only 47 events** - severely data-limited
- **Death-as-feature hurts** - correlation structure changes
- **XGBoost overfits** small datasets

### 3. Fold-by-Fold Breakdown (Hepatic)

| Repeat | Fold 1 CI | Issue |
|--------|-----------|-------|
| 1 | 0.7831 | Below average |
| 2 | 0.9171 | Good |
| 3 | 0.9339 | Excellent |
| 4 | 0.7316 | Poor - overfitting? |
| 5 | 0.7050 | Very poor |

**Pattern:** Folds 4-5 consistently underperform - suggests **stratification issues** or **data leakage**.

---

## 🔧 Improvement Opportunities

### Option 1: Revert Hepatic to Baseline RSF (Quick Win)

**Idea:** Use baseline RSF for hepatic (0.83 CV), keep XGBoost for death (0.915 CV)

**Expected:** ~0.87 LB score (+0.02 improvement)

```python
# Hepatic: Pure RSF (tuned)
RSF(n_estimators=500, min_samples_leaf=15, max_features='sqrt')

# Death: XGBoost + RSF ensemble
```

### Option 2: Remove Death-as-Feature (Already Tested - v3)

**v3 Results:** Hepatic 0.7921 (similar to v2's 0.7882)
- Death feature adds minimal value
- Independent models are cleaner

### Option 3: Optimize Ensemble Weights

**Current:** RSF(0.6) + XGB(0.4) for both models

**Try:**
- Death: RSF(0.3) + XGB(0.7) → XGB dominates
- Hepatic: RSF(0.8) + XGB(0.2) → RSF dominates (less overfitting)

### Option 4: Hyperparameter Tuning

**XGBoost for Hepatic (reduce overfitting):**
```python
{
    'max_depth': 2,          # Current: 3
    'learning_rate': 0.03,   # Current: 0.05
    'reg_lambda': 20.0,      # Current: 10.0
    'min_child_weight': 20,  # Current: 10
    'subsample': 0.7,        # Current: 0.8
}
```

### Option 5: Feature Selection for Hepatic

**Current:** 222 features (likely overfitting)

**Try:**
- Keep only top 50 features by importance
- Focus on LSM, FIB-4, platelets (clinical gold standards)
- Remove EWMA features (added noise)

### Option 6: Blending with Baseline

**Idea:** Weighted average of baseline (0.838) and Gemini v2 (0.853)

```python
final_pred = 0.7 * gemini_v2_pred + 0.3 * baseline_pred
```

**Expected:** Smooth predictions, reduce variance

---

## 🚀 Recommended Next Steps

### Quick Experiments (1-2 hours each):

1. **Pure RSF for Hepatic** (highest confidence)
   - Use baseline RSF config for hepatic
   - Keep XGB+RSF ensemble for death
   - Expected: 0.87 LB

2. **Heavy Regularization for Hepatic XGB**
   - Aggressive L2, lower learning rate
   - Expected: 0.80-0.82 hepatic CV

3. **Feature Pruning**
   - Select top 50 features only
   - Expected: More stable predictions

### Advanced (3-4 hours):

4. **Stacking Ensemble**
   - Train meta-learner on OOF predictions
   - Weight models by fold performance

5. **Target Encoding**
   - Encode categorical features with survival targets
   - Risk: leakage if not careful

---

## 📈 Gap to Top 3

**Current:** 0.8529  
**Target:** 0.90+  
**Gap:** 0.047

**Realistic improvements:**
- Fix hepatic model: +0.02
- Optimize ensemble: +0.01
- Better CV strategy: +0.01
- **Total realistic:** 0.88-0.89

**To reach 0.90:** May need novel features or different modeling approach

---

## 🎯 Action Items

| Priority | Action | Expected Gain | Time |
|----------|--------|---------------|------|
| 🔴 High | Pure RSF for hepatic | +0.02 | 30 min |
| 🟡 Med | Tune XGB regularization | +0.01 | 1 hour |
| 🟡 Med | Feature selection (top 50) | +0.01 | 1 hour |
| 🟢 Low | Blend with baseline | +0.005 | 30 min |

---

## 🔗 Files to Check

- `submissions/gemini_v2_results_*.json` - Detailed metrics
- `submissions/gemini_v2_log.txt` - Full training log
- `src/pipeline_gemini_v2.py` - Current implementation
- `src/pipeline.py` - Baseline (0.83) for comparison
