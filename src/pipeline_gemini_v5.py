"""
Gemini Pipeline v5 - Minimal Features (15 key clinical features only)
"""

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
N_FOLDS = 5
N_REPEATS = 5

# Only the most important clinical features
KEY_FEATURES = [
    # LSM - Gold standard
    'fibs_stiffness_med_BM_1_mean',
    'fibs_stiffness_med_BM_1_max',
    'fibs_stiffness_med_BM_1_last_actual',
    'fibs_stiffness_med_BM_1_slope',
    'fibs_stiffness_med_BM_1_time_high',
    
    # FIB-4 - Blood-based score
    'fib4_max',
    'fib4_mean',
    'fib4_last_actual',
    'fib4_time_high_risk',
    
    # Platelets - Inverse relationship
    'plt_min',
    'plt_mean',
    
    # Liver enzymes
    'ast_max',
    'alt_max',
    
    # Static demographics
    'age_baseline',
    'follow_up_years',
]


def prepare_survival_target(df, outcome='hepatic'):
    """Prepare survival target."""
    df = df.copy()
    age_cols = [c for c in df.columns if c.startswith('Age_v')]
    df['last_observed_age'] = df[age_cols].max(axis=1)
    df['first_visit_age'] = df[age_cols].min(axis=1)
    
    if outcome == 'hepatic':
        event_col = 'evenements_hepatiques_majeurs'
        age_occur_col = 'evenements_hepatiques_age_occur'
        name = 'HepaticEvent'
        is_event = df[event_col] == 1
        invalid = is_event & df[age_occur_col].isna()
        df_valid = df[~invalid].copy()
    elif outcome == 'death':
        event_col = 'death'
        age_occur_col = 'death_age_occur'
        name = 'Death'
        is_event = df[event_col] == 1
        unknown = df[event_col].isna()
        invalid = is_event & df[age_occur_col].isna()
        df_valid = df[~(unknown | invalid)].copy()
    
    is_event_v = (df_valid[event_col] == 1)
    time_values = np.where(
        is_event_v,
        df_valid[age_occur_col] - df_valid['first_visit_age'],
        df_valid['last_observed_age'] - df_valid['first_visit_age']
    ).astype(float)
    time_values = np.maximum(time_values, 0.001)
    
    y = Surv.from_arrays(event=is_event_v.values, time=time_values, name_event=name, name_time='Time')
    return df_valid, y


def feature_engineering_minimal(df):
    """Minimal feature engineering - only key features."""
    from pipeline import TrajectoryFeatureEngineer
    engineer = TrajectoryFeatureEngineer()
    X = engineer.transform(df)
    
    # Keep only key features that exist
    available_features = [f for f in KEY_FEATURES if f in X.columns]
    logger.info(f"Using {len(available_features)}/{len(KEY_FEATURES)} key features")
    
    return X[available_features]


def cross_validate_minimal(X, y, n_folds=5, n_repeats=5):
    """CV with minimal features and pure RSF."""
    logger.info(f"{n_folds}-fold × {n_repeats}-repeat CV (minimal features)...")
    
    event_indicator = y[y.dtype.names[0]]
    rskf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=RANDOM_STATE)
    
    oof_preds = np.zeros(len(X))
    fold_cindices = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(rskf.split(X, event_indicator)):
        repeat_idx = fold_idx // n_folds
        fold_num = fold_idx % n_folds + 1
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Preprocess
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        X_train_proc = scaler.fit_transform(imputer.fit_transform(X_train))
        X_val_proc = scaler.transform(imputer.transform(X_val))
        
        # Pure RSF - conservative settings
        model = RandomSurvivalForest(
            n_estimators=300,
            min_samples_leaf=20,
            min_samples_split=40,
            max_features='sqrt',
            n_jobs=-1,
            random_state=RANDOM_STATE
        )
        model.fit(X_train_proc, y_train)
        preds = model.predict(X_val_proc)
        
        oof_preds[val_idx] = preds
        
        ci = concordance_index_censored(y_val[y_val.dtype.names[0]], y_val[y_val.dtype.names[1]], preds)[0]
        fold_cindices.append(ci)
        
        if fold_num == 1:
            logger.info(f"  Repeat {repeat_idx + 1}: fold {fold_num} CI={ci:.4f}")
    
    overall_ci = concordance_index_censored(event_indicator, y[y.dtype.names[1]], oof_preds)[0]
    
    logger.info(f"  Overall OOF C-index: {overall_ci:.4f}")
    logger.info(f"  Fold C-indices: {np.mean(fold_cindices):.4f} (+/- {np.std(fold_cindices):.4f})")
    
    return {
        'oof_preds': oof_preds,
        'overall_ci': overall_ci,
        'fold_cindices': fold_cindices
    }


def main():
    logger.info("="*80)
    logger.info("GEMINI PIPELINE V5 - Minimal Features (15 key features)")
    logger.info("="*80)
    
    start = datetime.now()
    
    # Load
    train_df = pd.read_csv('data/train-df.csv')
    test_df = pd.read_csv('data/test-df.csv')
    
    # Features - minimal
    logger.info("\nFeature Engineering (minimal)...")
    X_train = feature_engineering_minimal(train_df)
    X_test = feature_engineering_minimal(test_df)
    
    # Align
    common_cols = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]
    
    # === DEATH MODEL (keep XGB+RSF from v2) ===
    logger.info("\n--- DEATH MODEL (using v2 ensemble) ---")
    from pipeline_gemini_v2 import cross_validate, SimpleEnsemble
    
    df_death, y_death = prepare_survival_target(train_df, outcome='death')
    X_death = X_train.loc[df_death.index]
    
    # Use all features for death (more data = less overfitting)
    from pipeline_gemini_v2 import TrajectoryFeatureEngineerV2
    engineer_full = TrajectoryFeatureEngineerV2()
    X_death_full = engineer_full.transform(train_df).loc[df_death.index]
    
    death_cv = cross_validate(X_death_full, y_death, n_folds=N_FOLDS, n_repeats=N_REPEATS)
    
    death_ensemble = SimpleEnsemble(weights={'rsf': 0.6, 'xgb': 0.4})
    death_ensemble.fit(X_death_full, y_death)
    
    # === HEPATIC MODEL (minimal features) ===
    logger.info("\n--- HEPATIC MODEL (minimal features) ---")
    df_hep, y_hep = prepare_survival_target(train_df, outcome='hepatic')
    X_hep = X_train.loc[df_hep.index]
    
    hep_cv = cross_validate_minimal(X_hep, y_hep, n_folds=N_FOLDS, n_repeats=N_REPEATS)
    
    # Fit final hepatic model
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    X_hep_proc = scaler.fit_transform(imputer.fit_transform(X_hep))
    
    hep_model = RandomSurvivalForest(
        n_estimators=500,
        min_samples_leaf=20,
        min_samples_split=40,
        max_features='sqrt',
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    hep_model.fit(X_hep_proc, y_hep)
    
    # === GENERATE SUBMISSION ===
    logger.info("\n--- GENERATING SUBMISSION ---")
    
    # Death predictions (full features)
    X_test_death_full = engineer_full.transform(test_df)
    pred_death = death_ensemble.predict(X_test_death_full)
    
    # Hepatic predictions (minimal features)
    X_test_hep = X_test[common_cols]
    X_test_hep_proc = scaler.transform(imputer.transform(X_test_hep))
    pred_hepatic = hep_model.predict(X_test_hep_proc)
    
    submission = pd.DataFrame({
        'trustii_id': test_df['trustii_id'].values,
        'risk_hepatic_event': pred_hepatic,
        'risk_death': pred_death
    })
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = f'submissions/gemini_v5_submission_{timestamp}.csv'
    submission.to_csv(submission_path, index=False)
    
    logger.info(f"\nSubmission: {submission_path}")
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("RESULTS SUMMARY")
    logger.info("="*80)
    logger.info(f"Death OOF C-index:     {death_cv['overall_ci']:.4f}")
    logger.info(f"  Fold std: {np.std(death_cv['fold_cindices']):.4f}")
    logger.info(f"Hepatic OOF C-index:   {hep_cv['overall_ci']:.4f}")
    logger.info(f"  Fold std: {np.std(hep_cv['fold_cindices']):.4f}")
    logger.info(f"Average:               {(death_cv['overall_ci'] + hep_cv['overall_ci'])/2:.4f}")
    logger.info(f"Features used:         {len(common_cols)}")
    logger.info("="*80)
    
    results = {
        'timestamp': timestamp,
        'death_ci': float(death_cv['overall_ci']),
        'hepatic_ci': float(hep_cv['overall_ci']),
        'death_fold_std': float(np.std(death_cv['fold_cindices'])),
        'hepatic_fold_std': float(np.std(hep_cv['fold_cindices'])),
        'n_features': len(common_cols),
        'elapsed_minutes': (datetime.now() - start).total_seconds() / 60
    }
    
    with open(f'submissions/gemini_v5_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nElapsed: {results['elapsed_minutes']:.1f} minutes")
    return results


if __name__ == "__main__":
    main()
