#!/usr/bin/env python3
"""
Quick test of Gemini pipeline with reduced repeats for validation.
"""

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, '.')
from pipeline_gemini import (
    TrajectoryFeatureEngineerV2,
    SurvivalModelEnsembleGemini,
    MAX_MISSING_RATE
)
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

print("="*80)
print("GEMINI PIPELINE QUICK TEST (2 folds × 2 repeats)")
print("="*80)

# Load data
print("\n1. Loading data...")
train_df = pd.read_csv('data/train-df.csv')
test_df = pd.read_csv('data/test-df.csv')
print(f"   Train: {train_df.shape}, Test: {test_df.shape}")

# Feature Engineering
print("\n2. Feature Engineering...")
engineer = TrajectoryFeatureEngineerV2()
X_train = engineer.transform(train_df)
X_test = engineer.transform(test_df)

# Align columns
common_cols = [c for c in X_train.columns if c in X_test.columns]
X_train = X_train[common_cols]
X_test = X_test[common_cols]
print(f"   Features: {X_train.shape[1]}")

# Check for new EWMA features
ewma_features = [c for c in X_train.columns if 'ewma' in c]
print(f"   EWMA features: {len(ewma_features)} (e.g., {ewma_features[:3]})")

# Check for last_actual features
last_features = [c for c in X_train.columns if 'last_actual' in c]
print(f"   Last_actual features: {len(last_features)} (e.g., {last_features[:3]})")

# Test Death Model First approach
print("\n3. Testing Death-First Strategy...")
ensemble = SurvivalModelEnsembleGemini(
    n_folds=2,  # Reduced for quick test
    n_repeats=2,
    random_state=42
)

# Death model
print("\n   Training Death Model...")
df_death, y_death = ensemble.prepare_survival_target(train_df, outcome='death')
X_death = X_train.loc[df_death.index]
print(f"   Death: {len(df_death)} patients, {y_death[y_death.dtype.names[0]].sum()} events")

# Quick CV test (just 1 fold for speed)
from sklearn.model_selection import StratifiedKFold

event_ind = y_death[y_death.dtype.names[0]]
skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
train_idx, val_idx = next(skf.split(X_death, event_ind))

X_tr, X_val = X_death.iloc[train_idx], X_death.iloc[val_idx]
y_tr, y_val = y_death[train_idx], y_death[val_idx]

# Preprocess
keep_cols = X_tr.columns[X_tr.isnull().mean() <= MAX_MISSING_RATE]
X_tr = X_tr[keep_cols]
X_val = X_val[keep_cols]

imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()
X_tr_proc = scaler.fit_transform(imputer.fit_transform(X_tr))
X_val_proc = scaler.transform(imputer.transform(X_val))

# Test XGBoost
print("\n4. Testing XGBoost Survival...")
from pipeline_gemini import XGBSurvivalModel
from sksurv.metrics import concordance_index_censored

xgb_model = XGBSurvivalModel(random_state=42)
xgb_model.fit(X_tr_proc, y_tr, feature_names=keep_cols.tolist())
xgb_preds = xgb_model.predict(X_val_proc)
xgb_ci = concordance_index_censored(
    y_val[y_val.dtype.names[0]],
    y_val[y_val.dtype.names[1]],
    xgb_preds
)[0]
print(f"   XGBoost C-index: {xgb_ci:.4f}")

# Test CoxNet
print("\n5. Testing CoxNet...")
from pipeline_gemini import CoxNetSurvivalModel

cox_model = CoxNetSurvivalModel(random_state=42)
cox_model.fit(pd.DataFrame(X_tr_proc, columns=keep_cols), y_tr)
cox_preds = cox_model.predict(pd.DataFrame(X_val_proc, columns=keep_cols))
cox_ci = concordance_index_censored(
    y_val[y_val.dtype.names[0]],
    y_val[y_val.dtype.names[1]],
    cox_preds
)[0]
print(f"   CoxNet C-index: {cox_ci:.4f}")

# Test Rank Ensemble
print("\n6. Testing Rank Ensemble...")
from pipeline_gemini import RankEnsemble

# Create dummy ensemble for testing
ensemble = RankEnsemble(weights={'rsf': 0.5, 'xgb': 0.3, 'coxnet': 0.2})

# Manually set models
from sksurv.ensemble import RandomSurvivalForest

rsf = RandomSurvivalForest(n_estimators=50, min_samples_leaf=20, random_state=42, n_jobs=-1)
rsf.fit(X_tr_proc, y_tr)

ensemble.models = {
    'rsf': rsf,
    'xgb': xgb_model,
    'coxnet': cox_model
}

rank_preds = ensemble.predict(pd.DataFrame(X_val_proc, columns=keep_cols))
rank_ci = concordance_index_censored(
    y_val[y_val.dtype.names[0]],
    y_val[y_val.dtype.names[1]],
    rank_preds
)[0]
print(f"   Rank Ensemble C-index: {rank_ci:.4f}")

print("\n" + "="*80)
print("✅ ALL TESTS PASSED")
print("="*80)
print(f"\nIndividual model performances on validation fold:")
print(f"  XGBoost:     {xgb_ci:.4f}")
print(f"  CoxNet:      {cox_ci:.4f}")
print(f"  RSF+XGB+Cox: {rank_ci:.4f}")
print(f"\nReady to run full pipeline!")
