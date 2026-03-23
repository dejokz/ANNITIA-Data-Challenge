"""Gemini Pipeline Final - Baseline RSF + Repeated CV only"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from sksurv.ensemble import RandomSurvivalForest
from tqdm import tqdm
import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
N_FOLDS = 5
N_REPEATS = 5


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


def feature_engineering_baseline(df):
    """Baseline feature engineering (from original 0.83 model)."""
    import sys
    sys.path.insert(0, '.')
    from pipeline import TrajectoryFeatureEngineer
    engineer = TrajectoryFeatureEngineer()
    return engineer.transform(df)


def cross_validate_rsf(X, y, n_folds=5, n_repeats=5):
    """Repeated CV with RSF only."""
    logger.info(f"{n_folds}-fold × {n_repeats}-repeat CV with RSF...")
    
    # Filter features
    keep_cols = X.columns[X.isnull().mean() <= 0.60].tolist()
    X = X[keep_cols].copy()
    
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
        
        # RSF with baseline params
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
        'fold_cindices': fold_cindices,
        'feature_cols': keep_cols
    }


def main():
    logger.info("="*80)
    logger.info("GEMINI PIPELINE FINAL - Baseline RSF + Repeated CV")
    logger.info("="*80)
    
    start = datetime.now()
    
    # Load
    train_df = pd.read_csv('data/train-df.csv')
    test_df = pd.read_csv('data/test-df.csv')
    
    # Features
    logger.info("\nFeature Engineering...")
    X_train = feature_engineering_baseline(train_df)
    X_test = feature_engineering_baseline(test_df)
    
    common_cols = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]
    logger.info(f"Features: {len(common_cols)}")
    
    # Death Model
    logger.info("\n--- DEATH MODEL ---")
    df_death, y_death = prepare_survival_target(train_df, outcome='death')
    death_cv = cross_validate_rsf(X_train.loc[df_death.index], y_death, N_FOLDS, N_REPEATS)
    
    # Hepatic Model
    logger.info("\n--- HEPATIC MODEL ---")
    df_hep, y_hep = prepare_survival_target(train_df, outcome='hepatic')
    hep_cv = cross_validate_rsf(X_train.loc[df_hep.index], y_hep, N_FOLDS, N_REPEATS)
    
    # Generate submission
    logger.info("\n--- GENERATING SUBMISSION ---")
    
    # Fit final models
    from sklearn.pipeline import Pipeline
    
    # Death
    X_death = X_train.loc[df_death.index]
    death_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    X_death_proc = death_pipe.fit_transform(X_death)
    death_model = RandomSurvivalForest(n_estimators=500, min_samples_leaf=20, min_samples_split=40, 
                                       max_features='sqrt', n_jobs=-1, random_state=RANDOM_STATE)
    death_model.fit(X_death_proc, y_death)
    
    # Hepatic
    X_hep = X_train.loc[df_hep.index]
    hep_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    X_hep_proc = hep_pipe.fit_transform(X_hep)
    hep_model = RandomSurvivalForest(n_estimators=500, min_samples_leaf=20, min_samples_split=40,
                                     max_features='sqrt', n_jobs=-1, random_state=RANDOM_STATE)
    hep_model.fit(X_hep_proc, y_hep)
    
    # Predict
    pred_death = death_model.predict(death_pipe.transform(X_test))
    pred_hepatic = hep_model.predict(hep_pipe.transform(X_test))
    
    submission = pd.DataFrame({
        'trustii_id': test_df['trustii_id'].values,
        'risk_hepatic_event': pred_hepatic,
        'risk_death': pred_death
    })
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = f'submissions/gemini_final_submission_{timestamp}.csv'
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
    logger.info("="*80)
    
    results = {
        'timestamp': timestamp,
        'death_ci': float(death_cv['overall_ci']),
        'hepatic_ci': float(hep_cv['overall_ci']),
        'death_fold_std': float(np.std(death_cv['fold_cindices'])),
        'hepatic_fold_std': float(np.std(hep_cv['fold_cindices'])),
        'elapsed_minutes': (datetime.now() - start).total_seconds() / 60
    }
    
    with open(f'submissions/gemini_final_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nElapsed: {results['elapsed_minutes']:.1f} minutes")
    return results


if __name__ == "__main__":
    main()
