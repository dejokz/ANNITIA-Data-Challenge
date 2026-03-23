# 🎯 Roadmap to 0.90: Strategy for ANNITIA Challenge

**Current Score:** 0.82  
**Target:** 0.90+  
**Gap:** 0.08 (significant but achievable)

---

## 📊 Understanding the Gap

### Current Top Scores:
1. **0.902** - Likely using advanced deep learning + optimal feature engineering
2. **0.901** - Similar approach with ensemble
3. **0.880** - Strong ML approach
4. **Your 0.820** - Good baseline with trajectory features

**Gap Analysis:** You're 0.08 behind #1, but only 0.06 behind #3. This suggests you're close, but missing key optimizations.

---

## 🔍 What's Likely Missing (In Priority Order)

### 1. 🚨 PROPER SEQUENTIAL MODELING (Biggest Gap)

**What we did:** Treated visits as static features (slopes, max, min)

**What top competitors did:** 
- LSTM/GRU processing actual sequences
- Attention over visits (which visits matter most?)
- Transformer capturing long-range dependencies
- Time-aware embeddings (irregular intervals)

**Why it matters:**
- Our approach: "Patient X has max LSM of 12"
- Their approach: "Patient X went from 8→10→12 over 3 years with accelerating slope"
- The temporal pattern carries information we discarded

**Implementation:**
```python
# We need to:
1. Convert wide format → sequences: [(visit_1_features), (visit_2_features), ...]
2. Handle variable lengths (pad/pack sequences)
3. Use LSTM with attention or Transformer
4. Time-aware: incorporate actual visit dates (not just visit number)
```

---

### 2. 🚨 ADVANCED CLINICAL FEATURES

**Missing features that likely matter:**

#### A. Delta Features (Visit-to-Visit Changes)
```python
# Not just slope from first to last
# But: change between consecutive visits
delta_lsm_v2 = lsm_v2 - lsm_v1
delta_lsm_v3 = lsm_v3 - lsm_v2
# ...
# Acceleration: is the rate of change itself changing?
```

#### B. Clinical Rules as Features
```python
# "Any 2 of 3 NITs high"
# "FIB-4 increased by >0.5 in one year"
# "LSM crossing 8 kPa threshold"
# "Platelet count dropping >20% per year"
```

#### C. Time-Aware Features
```python
# We used visit numbers (1, 2, 3...)
# Should use actual time intervals
# Irregular sampling matters!
time_since_last_visit = age_v2 - age_v1
time_to_next_visit = age_v3 - age_v2
# Patient with 6-month follow-up vs 2-year follow-up are different
```

---

### 3. 🚨 BETTER ENSEMBLE & STACKING

**Current:** Single RSF model

**Top competitors likely use:**
- 5-10 diverse models
- Stacking (meta-learner on model predictions)
- Different feature subsets for each model
- Different validation strategies

**Stacking architecture:**
```
Level 0 (Base Models):
  - RSF on clinical features
  - RSF on trajectory features  
  - Gradient Boosting Survival
  - Cox Elastic Net
  - LSTM (when implemented)
  
Level 1 (Meta-Learner):
  - Ridge regression on base predictions
  - Or: weighted average optimized by CV
```

---

### 4. 🚨 ADVANCED REGULARIZATION

**Current issue:** Our models likely overfit
- 47 events / 200 features = severe overfitting risk
- We used min_samples_leaf=20, but probably need 40+
- No early stopping in RSF

**Solutions:**
```python
# Much stronger regularization
RandomSurvivalForest(
    n_estimators=1000,
    min_samples_leaf=50,      # Was 20 - much larger
    min_samples_split=100,    # Was 60
    max_depth=5,              # Limit tree depth
    max_features='sqrt',      # Feature subsampling
)

# Feature selection before modeling
# Keep only top 30 features by univariate C-index
# This forces model to focus on strongest signals
```

---

### 5. 🚨 DATA AUGMENTATION

**For survival data (rare events):**
```python
# Bootstrap resampling with stratification
# SMOTE for survival (SMOTE + time perturbation)
# Gaussian noise injection to lab values (realistic augmentation)
# Time-shift augmentation (small perturbations to visit times)
```

---

## 🛠️ Implementation Plan (Priority Order)

### Phase 1: Sequential Modeling (Expected +0.05-0.08)
**Effort:** High  
**Impact:** Very High

```python
# Create proper sequences
sequences = []
for patient in patients:
    visits = []
    for visit_num in range(1, 23):
        visit_features = [
            lsm_v[visit_num],
            fib4_v[visit_num],
            fibrotest_v[visit_num],
            plt_v[visit_num],
            ...
        ]
        visits.append(visit_features)
    sequences.append(visits)

# LSTM with attention
class LSTMSurvival(nn.Module):
    def __init__(self, input_size, hidden_size):
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, sequences, lengths):
        # Pack sequences
        packed = pack_padded_sequence(sequences, lengths)
        output, _ = self.lstm(packed)
        # Attention
        attn_weights = softmax(self.attention(output))
        # Weighted sum
        return risk_score
```

### Phase 2: Better Feature Engineering (+0.02-0.04)
**Effort:** Medium  
**Impact:** Medium

```python
# Add delta features
for var in ['lsm', 'fib4', 'plt']:
    for v in range(2, 23):
        df[f'{var}_delta_v{v}'] = df[f'{var}_v{v}'] - df[f'{var}_v{v-1}']

# Time-aware features
df['avg_visit_interval'] = (last_age - first_age) / n_visits
df['visit_regularity'] = std(visit_intervals)

# Clinical rule features
df['rapid_fib4_progression'] = (fib4_slope > 0.5).astype(int)
df['lsm_crossed_8'] = ((lsm_first < 8) & (lsm_max > 8)).astype(int)
```

### Phase 3: Model Stacking (+0.02-0.03)
**Effort:** Medium  
**Impact:** Medium

```python
# Train diverse models
models = {
    'rsf_clinical': RSF(features=clinical_only),
    'rsf_trajectory': RSF(features=trajectory_only),
    'rsf_all': RSF(features=all),
    'gbs': GradientBoostingSurvival(),
    'cox': CoxElasticNet(),
}

# Get OOF predictions
oof_preds = {}
for name, model in models.items():
    oof_preds[name] = cross_val_predict(model, X, y)

# Meta-learner (Ridge regression)
from sklearn.linear_model import Ridge
meta_model = Ridge(alpha=1.0)
meta_model.fit(oof_preds_df, y)
```

### Phase 4: Hyperparameter Optimization (+0.01-0.02)
**Effort:** Low  
**Impact:** Low-Medium

```python
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_est', 100, 1000),
        'min_samples_leaf': trial.suggest_int('min_leaf', 10, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
    }
    model = RandomSurvivalForest(**params)
    return cross_val_score(model, X, y).mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

---

## 🎯 Specific Implementation for LSTM

Since you have GPU, let's implement LSTM properly:

```python
# 1. Prepare sequences
class LongitudinalDataset(Dataset):
    def __getitem__(self, idx):
        patient = self.patients[idx]
        
        # Get actual visit times (not just visit numbers)
        visit_times = []
        visit_features = []
        
        for visit_num in range(1, 23):
            age = patient[f'Age_v{visit_num}']
            if pd.notna(age):
                features = [
                    patient[f'lsm_v{visit_num}'],
                    patient[f'fib4_v{visit_num}'],
                    patient[f'plt_v{visit_num}'],
                    # ... all NITs
                ]
                visit_times.append(age)
                visit_features.append(features)
        
        return {
            'sequences': torch.FloatTensor(visit_features),
            'times': torch.FloatTensor(visit_times),  # Actual ages!
            'length': len(visit_features),
            'event': patient['event'],
            'time_to_event': patient['time_to_event']
        }

# 2. LSTM with time-aware attention
class TimeAwareLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # Time embedding (irregular intervals)
        self.time_embed = nn.Linear(1, hidden_size)
        
        # Attention
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, sequences, visit_times, lengths):
        # LSTM processing
        lstm_out, _ = self.lstm(sequences)
        
        # Time embedding
        time_diffs = torch.diff(visit_times, prepend=visit_times[:, 0:1])
        time_emb = self.time_embed(time_diffs.unsqueeze(-1))
        
        # Combine LSTM + time
        combined = torch.cat([lstm_out, time_emb], dim=-1)
        
        # Attention
        attn = self.attention(combined)
        # ... rest of forward pass
```

---

## 📈 Expected Improvements

| Phase | Expected Gain | Cumulative Score |
|-------|---------------|------------------|
| Current | - | 0.82 |
| Phase 1: LSTM Sequential | +0.05-0.08 | 0.87-0.90 |
| Phase 2: Better Features | +0.02-0.04 | 0.89-0.94 |
| Phase 3: Stacking | +0.02-0.03 | 0.91-0.97 |
| Phase 4: Hyperopt | +0.01-0.02 | 0.92-0.99 |

**Realistic target:** 0.90-0.93  
**Aggressive target:** 0.95+ (if LSTM works well)

---

## ⚡ Quick Wins (Do These First)

### 1. Reduce Overfitting (30 minutes)
```python
# Much stronger regularization
RSF(min_samples_leaf=50, max_depth=5, n_estimators=500)

# Feature selection: keep only top 30 features
```

### 2. Add Delta Features (1 hour)
```python
# Visit-to-visit changes
for v in range(2, 23):
    df[f'fib4_delta_v{v}'] = df[f'fib4_v{v}'] - df[f'fib4_v{v-1}']
```

### 3. Simple Stacking (2 hours)
```python
# Train 3 different models, average predictions
```

**Expected quick win gain:** +0.02-0.04 (reach 0.84-0.86)

---

## 🚀 Recommendation

Given your 0.82 score and the top scores of 0.90, I recommend:

### Immediate (Next 2-4 hours):
1. **Stronger regularization** - Reduce features to top 30, increase min_samples_leaf to 50
2. **Add delta features** - Visit-to-visit changes
3. **Simple ensemble** - Average 3 different model types

**Expected:** 0.84-0.86

### Short-term (Next 1-2 days):
4. **Implement LSTM** with proper sequential data processing
5. **Time-aware features** - irregular visit intervals

**Expected:** 0.88-0.92

### Long-term (If time permits):
6. **Transformer model** for long-range dependencies
7. **Advanced stacking** with meta-learner
8. **Hyperparameter optimization** with Optuna

**Expected:** 0.90-0.95

---

## ❓ Questions to Consider

1. **Do you want to prioritize reaching 0.90 quickly, or maximize score?**
   - Quick: Focus on regularization + ensemble
   - Maximum: Implement LSTM properly

2. **How much time do you have?**
   - < 1 day: Do quick wins only
   - 2-3 days: Implement LSTM
   - > 3 days: Full optimization

3. **Do you have other datasets/external data?**
   - Top competitors might be using transfer learning
   - Pre-trained models on similar liver datasets

---

## 🎓 Why LSTM Should Work

**Current approach:**
```
Patient: fib4_max=3.5, fib4_slope=0.2
```

**LSTM approach:**
```
Patient: [1.2, 1.5, 1.8, 2.1, 2.5, 3.0, 3.5]  # Year by year progression
With attention: [0.1, 0.1, 0.1, 0.2, 0.3, 0.1, 0.1]  # Weight on year 5
```

The LSTM sees the **acceleration pattern** - the patient is getting worse faster over time, which is more predictive than just knowing the max is 3.5.

---

**Bottom line:** You need sequential modeling to reach 0.90. The static feature approach has a ceiling around 0.85.
