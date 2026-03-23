# 🏆 Best Model Summary - ANNITIA Challenge

**Current Best Score:** 0.83 (Public Leaderboard)  
**Submission ID:** 21788  
**Status:** ✅ VALIDATED

---

## 🎯 What Works

### Model: Random Survival Forest (RSF)

```python
RandomSurvivalForest(
    n_estimators=300,
    min_samples_leaf=20,      # Moderate regularization
    min_samples_split=40,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42
)
```

### Features (80 total):

1. **Trajectory features** (NO deltas - too noisy):
   - `fib4_max`, `fib4_mean`, `fib4_slope`, `fib4_time_high`
   - `lsm_max`, `lsm_mean`, `lsm_slope`, `lsm_time_high`
   - `plt_min`, `plt_declining`
   - `ast_max`, `ast_alt_ratio_max`

2. **Clinical scores** (calculated per visit):
   - FIB-4 at each visit
   - APRI at each visit
   - AST/ALT ratio at each visit

3. **Cross-NIT features**:
   - Agreement between LSM, FIB-4, FibroTest
   - Composite risk indicators

4. **Static features**:
   - gender, T2DM, Hypertension, Dyslipidaemia
   - age_baseline, follow_up_years
   - T2DM × fib4_max interaction

### Data Preprocessing:

```python
# Median imputation
imputer = SimpleImputer(strategy='median')

# Standard scaling
scaler = StandardScaler()

# NO feature selection (use all 80)
```

### Cross-Validation:

```python
5-fold StratifiedKFold (preserves event ratio)
Local CV: 0.79
Public LB: 0.83 (+0.04)
```

---

## ❌ What Doesn't Work

| Approach | Score | Why It Failed |
|----------|-------|---------------|
| LSTM | 0.76 | Overfitting (too complex for 47 events) |
| Delta features | 0.755 | Added noise, caused overfitting |
| Heavy regularization (min_samples=50) | 0.76 | Too restrictive |
| Feature selection (top 40) | 0.76 | Lost important signals |

---

## 🔬 Why Simple RSF Wins

1. **Bias-Variance Tradeoff**: 
   - 47 events → need low variance
   - RSF: ~300 trees × moderate depth = good balance
   - LSTM: millions of parameters = severe overfitting

2. **Feature Engineering vs Learning**:
   - Hand-crafted trajectory features capture domain knowledge
   - LSTM tries to learn from raw sequences but data is too sparse

3. **Regularization Sweet Spot**:
   - min_samples_leaf=20 is just right
   - Higher = underfitting, Lower = overfitting

---

## 🚀 Path to 0.90 (If You Want to Try)

Given that LSTM failed, here are realistic options:

### Option 1: Ensemble of Diverse RSFs (+0.02-0.04 expected)

```python
# Train 5 RSFs with different random seeds
models = []
for seed in [42, 123, 456, 789, 999]:
    model = RandomSurvivalForest(
        n_estimators=300,
        min_samples_leaf=20,
        random_state=seed
    )
    models.append(model)

# Average predictions
final_pred = np.mean([m.predict(X) for m in models], axis=0)
```

**Expected:** 0.85-0.87

### Option 2: Add Gradient Boosting to Ensemble (+0.01-0.03)

```python
models = [
    RSF(random_state=42),
    RSF(random_state=123),
    GradientBoostingSurvivalAnalysis(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3
    )
]
```

**Expected:** 0.86-0.88

### Option 3: Hyperparameter Optimization (+0.01-0.02)

```python
import optuna

# Optimize min_samples_leaf, max_depth, etc.
# But gains are limited with small dataset
```

**Expected:** 0.84-0.85

---

## 📊 Realistic Target Assessment

| Score | Difficulty | Approach Needed |
|-------|------------|-----------------|
| 0.83 | ✅ Achieved | Simple RSF |
| 0.85 | 🟡 Moderate | Ensemble (2-3 models) |
| 0.87 | 🟡 Moderate | Ensemble (5 models) + GBS |
| 0.90 | 🔴 Hard | Lucky ensemble + better features |
| 0.90+ | 🔴 Very Hard | Probably need external data or different approach |

**The gap from 0.83 to 0.90 is 0.07.**

Given that:
- LSTM failed (overfitting)
- Delta features failed (noise)
- Dataset has only 47 events

**Reaching 0.90 may not be possible** without:
1. Transfer learning from external liver datasets
2. Very lucky ensemble
3. Different feature engineering approach

---

## ✅ Recommendation

**Conservative approach (safest):**
1. ✅ Stick with current 0.83 model
2. ✅ Try simple ensemble (3 RSFs with different seeds)
3. ⚠️ Only submit if local CV shows improvement

**Aggressive approach (risky):**
1. Try ensemble of 5 diverse models
2. Add more sophisticated feature interactions
3. Risk: might drop below 0.83

**My recommendation:** Given the LSTM failure and the difficulty of reaching 0.90 with only 47 events, **stick with the 0.83 model** and make minor improvements (simple ensemble). Don't risk your current good standing.

---

## 📁 Files for 0.83 Model

```
src/pipeline_fast.py              ✅ Working code
submissions/optimized_submission.csv  ✅ Best submission
submissions/annitia_submission.ipynb  ✅ Notebook
```

To reproduce:
```bash
python src/pipeline_fast.py
```

---

## 🎯 Bottom Line

**Your 0.83 score is solid and validated.** The jump to 0.90 requires either:
1. Luck with ensemble
2. External data (not allowed?)
3. Very sophisticated feature engineering

The LSTM experiment proved that **complexity hurts** with this dataset size. Keep it simple.
