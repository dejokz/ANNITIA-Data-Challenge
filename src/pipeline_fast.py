"""
ANNITIA MASLD Survival Analysis - Fast Optimized Pipeline
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
N_FOLDS = 5


def calculate_clinical_scores(df):
    """Calculate FIB-4 and APRI for each visit."""
    result = df.copy()
    max_visits = 22
    
    logger.info("Calculating FIB-4 and APRI scores...")
    
    for visit in range(1, max_visits + 1):
        age_col = f'Age_v{visit}'
        ast_col = f'ast_v{visit}'
        alt_col = f'alt_v{visit}'
        plt_col = f'plt_v{visit}'
        
        required_cols = [age_col, ast_col, alt_col, plt_col]
        if all(col in result.columns for col in required_cols):
            # FIB-4 = (Age × AST) / (Platelets × √ALT)
            result[f'fib4_v{visit}'] = (
                result[age_col] * result[ast_col] / 
                (result[plt_col] * np.sqrt(result[alt_col] + 0.001) + 0.001)
            )
            
            # APRI = ((AST / ULN_AST) × 100) / Platelets
            ULN_AST = 40
            result[f'apri_v{visit}'] = (
                (result[ast_col] / ULN_AST * 100) / 
                (result[plt_col] + 0.001)
            )
    
    return result


def extract_trajectory_features(df):
    """Extract trajectory features efficiently."""
    vars_to_process = [
        'fibs_stiffness_med_BM_1', 'fibrotest_BM_2', 'aixp_aix_result_BM_3',
        'alt', 'ast', 'plt', 'bilirubin', 'ggt', 'fib4', 'apri'
    ]
    
    features = pd.DataFrame(index=df.index)
    
    logger.info("Extracting trajectory features...")
    
    for var in vars_to_process:
        visit_cols = [c for c in df.columns if c.startswith(f'{var}_v') and c.split('_v')[-1].isdigit()]
        
        if not visit_cols:
            continue
        
        # Sort columns
        visit_nums = [int(c.split('_v')[-1]) for c in visit_cols]
        sorted_pairs = sorted(zip(visit_nums, visit_cols))
        sorted_cols = [col for _, col in sorted_pairs]
        
        values = df[sorted_cols]
        
        # Basic stats
        features[f'{var}_max'] = values.max(axis=1)
        features[f'{var}_min'] = values.min(axis=1)
        features[f'{var}_mean'] = values.mean(axis=1)
        features[f'{var}_std'] = values.std(axis=1)
        features[f'{var}_last'] = values.ffill(axis=1).iloc[:, -1]
        features[f'{var}_first'] = values.iloc[:, 0]
        
        # Rate of change
        features[f'{var}_roc'] = (features[f'{var}_last'] - features[f'{var}_first'])
        
        # Clinical thresholds
        if var == 'fib4':
            features[f'{var}_time_high'] = (values > 2.67).sum(axis=1) / values.notna().sum(axis=1)
            features[f'{var}_ever_high'] = (values > 2.67).any(axis=1).astype(int)
        elif var == 'fibs_stiffness_med_BM_1':
            features[f'{var}_time_high'] = (values > 8.0).sum(axis=1) / values.notna().sum(axis=1)
            features[f'{var}_ever_high'] = (values > 8.0).any(axis=1).astype(int)
        elif var == 'fibrotest_BM_2':
            features[f'{var}_time_high'] = (values > 0.72).sum(axis=1) / values.notna().sum(axis=1)
            features[f'{var}_ever_high'] = (values > 0.72).any(axis=1).astype(int)
    
    # Add static features
    static_cols = ['gender', 'T2DM', 'Hypertension', 'Dyslipidaemia', 'bariatric_surgery']
    for col in static_cols:
        if col in df.columns:
            features[col] = df[col]
    
    return features


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
    
    y = Surv.from_arrays(
        event=is_event_v.values,
        time=time_values,
        name_event=name,
        name_time='Time'
    )
    
    return df_valid, y


def cross_validate(X, y, outcome='hepatic'):
    """Stratified CV for survival."""
    logger.info(f"\nStarting {N_FOLDS}-fold CV for {outcome}...")
    
    # Filter features
    missing_rates = X.isna().mean()
    keep_cols = missing_rates[missing_rates <= 0.70].index.tolist()
    X = X[keep_cols]
    logger.info(f"Features: {len(keep_cols)}/{len(missing_rates)}")
    
    event_indicator = y[y.dtype.names[0]]
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    oof_preds = np.zeros(len(X))
    fold_cindices = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, event_indicator)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Preprocess
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        X_train_proc = scaler.fit_transform(imputer.fit_transform(X_train))
        X_val_proc = scaler.transform(imputer.transform(X_val))
        
        # Train RSF (faster, good performance)
        model = RandomSurvivalForest(
            n_estimators=200,
            min_samples_leaf=20,
            max_features='sqrt',
            n_jobs=-1,
            random_state=RANDOM_STATE
        )
        model.fit(X_train_proc, y_train)
        
        preds = model.predict(X_val_proc)
        oof_preds[val_idx] = preds
        
        ci = concordance_index_censored(
            y_val[y_val.dtype.names[0]], y_val[y_val.dtype.names[1]], preds
        )[0]
        fold_cindices.append(ci)
        logger.info(f"  Fold {fold+1}: C-index = {ci:.4f}")
    
    overall_ci = concordance_index_censored(
        event_indicator, y[y.dtype.names[1]], oof_preds
    )[0]
    
    logger.info(f"\n{outcome.upper()} RESULTS:")
    logger.info(f"  Mean fold C-index: {np.mean(fold_cindices):.4f} (+/- {np.std(fold_cindices):.4f})")
    logger.info(f"  Overall OOF C-index: {overall_ci:.4f}")
    
    return {'oof_preds': oof_preds, 'overall_ci': overall_ci, 'fold_cis': fold_cindices}


def train_final(X, y):
    """Train final model on full data."""
    missing_rates = X.isna().mean()
    keep_cols = missing_rates[missing_rates <= 0.70].index.tolist()
    X = X[keep_cols]
    
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    X_proc = scaler.fit_transform(imputer.fit_transform(X))
    
    model = RandomSurvivalForest(
        n_estimators=500,
        min_samples_leaf=20,
        max_features='sqrt',
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    model.fit(X_proc, y)
    
    return model, imputer, scaler, keep_cols


def main():
    logger.info("="*70)
    logger.info("ANNITIA MASLD SURVIVAL ANALYSIS - OPTIMIZED PIPELINE")
    logger.info("="*70)
    
    # Load data
    logger.info("\nLoading data...")
    train_df = pd.read_csv('data/train-df.csv')
    test_df = pd.read_csv('data/test-df.csv')
    logger.info(f"Train: {train_df.shape}, Test: {test_df.shape}")
    
    # Feature engineering
    logger.info("\n" + "="*70)
    logger.info("FEATURE ENGINEERING")
    logger.info("="*70)
    
    train_df = calculate_clinical_scores(train_df)
    test_df = calculate_clinical_scores(test_df)
    
    X_train = extract_trajectory_features(train_df)
    X_test = extract_trajectory_features(test_df)
    
    common_cols = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]
    logger.info(f"Features: {len(common_cols)}")
    
    # Medical corroboration
    logger.info("\n" + "="*70)
    logger.info("MEDICAL CORROBORATION")
    logger.info("="*70)
    
    y_event = train_df['evenements_hepatiques_majeurs'].values
    key_features = ['fib4_max', 'fib4_time_high', 'fib4_ever_high', 
                   'fibs_stiffness_med_BM_1_max', 'fibs_stiffness_med_BM_1_time_high',
                   'fibrotest_BM_2_max', 'plt_min', 'ast_max']
    
    validated = 0
    for feat in key_features:
        if feat in X_train.columns:
            corr, _ = pearsonr(X_train[feat].fillna(X_train[feat].median()), y_event)
            status = "PASS" if abs(corr) > 0.05 else "FLAG"
            logger.info(f"[{status}] {feat:40s} | r={corr:6.3f}")
            if abs(corr) > 0.05:
                validated += 1
    
    logger.info(f"\nValidated: {validated}/{len(key_features)} features")
    
    # Model training
    logger.info("\n" + "="*70)
    logger.info("MODEL TRAINING")
    logger.info("="*70)
    
    # Hepatic model
    logger.info("\n--- HEPATIC EVENTS ---")
    df_hep, y_hep = prepare_survival_target(train_df, 'hepatic')
    cv_hep = cross_validate(X_train.loc[df_hep.index], y_hep, 'hepatic')
    model_hep, imp_hep, scl_hep, cols_hep = train_final(X_train.loc[df_hep.index], y_hep)
    
    # Death model
    logger.info("\n--- DEATH ---")
    df_death, y_death = prepare_survival_target(train_df, 'death')
    cv_death = cross_validate(X_train.loc[df_death.index], y_death, 'death')
    model_death, imp_death, scl_death, cols_death = train_final(X_train.loc[df_death.index], y_death)
    
    # Predictions
    logger.info("\n" + "="*70)
    logger.info("GENERATING SUBMISSION")
    logger.info("="*70)
    
    X_test_hep = X_test[cols_hep]
    X_test_death = X_test[cols_death]
    
    pred_hep = model_hep.predict(scl_hep.transform(imp_hep.transform(X_test_hep)))
    pred_death = model_death.predict(scl_death.transform(imp_death.transform(X_test_death)))
    
    submission = pd.DataFrame({
        'trustii_id': test_df['trustii_id'].values,
        'risk_hepatic_event': pred_hep,
        'risk_death': pred_death
    })
    
    submission.to_csv('submissions/optimized_submission.csv', index=False)
    
    logger.info(f"\nSubmission saved: {submission.shape}")
    logger.info(f"\nPreview:")
    logger.info(submission.head().to_string())
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("FINAL RESULTS")
    logger.info("="*70)
    logger.info(f"Hepatic C-index: {cv_hep['overall_ci']:.4f}")
    logger.info(f"Death C-index:   {cv_death['overall_ci']:.4f}")
    logger.info(f"Average:         {(cv_hep['overall_ci'] + cv_death['overall_ci'])/2:.4f}")
    logger.info("="*70)
    
    results = {
        'hepatic_ci': float(cv_hep['overall_ci']),
        'death_ci': float(cv_death['overall_ci']),
        'hepatic_folds': [float(x) for x in cv_hep['fold_cis']],
        'death_folds': [float(x) for x in cv_death['fold_cis']]
    }
    
    with open('submissions/results_optimized.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == '__main__':
    results = main()
