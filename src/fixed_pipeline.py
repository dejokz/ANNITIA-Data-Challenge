"""
FIXED Pipeline v3.0

Fixes issues from v2.0:
1. Proper CV without feature selection leakage
2. Remove problematic delta features  
3. Simpler, more robust models
4. Better regularization
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
N_FOLDS = 5


class RobustFeatureEngineer:
    """Feature engineering WITHOUT problematic delta features."""
    
    def __init__(self):
        self.visit_vars = [
            'fibs_stiffness_med_BM_1', 'fibrotest_BM_2', 'aixp_aix_result_BM_3',
            'alt', 'ast', 'plt', 'bilirubin', 'ggt',
            'gluc_fast', 'chol', 'triglyc', 'BMI'
        ]
        
    def calculate_clinical_scores(self, df):
        """Calculate FIB-4 and APRI for each visit."""
        result = df.copy()
        max_visits = 22
        
        for visit in range(1, max_visits + 1):
            age_col = f'Age_v{visit}'
            ast_col = f'ast_v{visit}'
            alt_col = f'alt_v{visit}'
            plt_col = f'plt_v{visit}'
            
            required = [age_col, ast_col, alt_col, plt_col]
            if all(col in result.columns for col in required):
                # FIB-4
                result[f'fib4_v{visit}'] = (
                    result[age_col] * result[ast_col] / 
                    (result[plt_col] * np.sqrt(result[alt_col].clip(lower=1)) + 0.001)
                )
                
                # APRI
                ULN_AST = 40
                result[f'apri_v{visit}'] = (
                    (result[ast_col] / ULN_AST * 100) / 
                    (result[plt_col] + 0.001)
                )
                
                # AST/ALT ratio
                result[f'ast_alt_ratio_v{visit}'] = (
                    result[ast_col] / (result[alt_col] + 0.001)
                )
        
        self.visit_vars.extend(['fib4', 'apri', 'ast_alt_ratio'])
        return result
    
    def extract_trajectory_features(self, df):
        """Extract features WITHOUT delta features (too noisy)."""
        features = pd.DataFrame(index=df.index)
        
        logger.info("Extracting trajectory features (NO deltas)...")
        
        for var in self.visit_vars:
            visit_cols = [c for c in df.columns if c.startswith(f'{var}_v') and c.split('_v')[-1].isdigit()]
            
            if not visit_cols:
                continue
            
            # Sort
            visit_nums = [int(c.split('_v')[-1]) for c in visit_cols]
            sorted_pairs = sorted(zip(visit_nums, visit_cols))
            sorted_cols = [col for _, col in sorted_pairs]
            
            values = df[sorted_cols]
            
            # Basic stats only (no deltas to avoid overfitting)
            features[f'{var}_max'] = values.max(axis=1)
            features[f'{var}_min'] = values.min(axis=1)
            features[f'{var}_mean'] = values.mean(axis=1)
            features[f'{var}_median'] = values.median(axis=1)
            features[f'{var}_std'] = values.std(axis=1)
            features[f'{var}_first'] = values.iloc[:, 0]
            features[f'{var}_last'] = values.ffill(axis=1).iloc[:, -1]
            features[f'{var}_range'] = features[f'{var}_max'] - features[f'{var}_min']
            
            # Simple slope only
            def calc_slope(row):
                valid = row.dropna()
                if len(valid) < 2:
                    return 0.0
                x = np.arange(len(valid))
                slope, _, _, _, _ = stats.linregress(x, valid.values)
                return slope
            
            features[f'{var}_slope'] = values.apply(calc_slope, axis=1)
            
            # Rate of change (first to last)
            age_cols = [c for c in df.columns if c.startswith('Age_v')]
            time_span = df[age_cols].max(axis=1) - df[age_cols].min(axis=1)
            features[f'{var}_roc'] = (features[f'{var}_last'] - features[f'{var}_first']) / (time_span + 0.001)
            
            # Clinical thresholds
            if var == 'fib4':
                features[f'{var}_time_high'] = (values > 2.67).sum(axis=1) / values.notna().sum(axis=1)
                features[f'{var}_ever_high'] = (values > 2.67).any(axis=1).astype(int)
                features[f'{var}_worsening'] = (features[f'{var}_slope'] > 0.1).astype(int)
                
            elif var == 'fibs_stiffness_med_BM_1':
                features[f'{var}_time_high'] = (values > 8.0).sum(axis=1) / values.notna().sum(axis=1)
                features[f'{var}_ever_high'] = (values > 8.0).any(axis=1).astype(int)
                features[f'{var}_worsening'] = (features[f'{var}_slope'] > 0.5).astype(int)
                
            elif var == 'fibrotest_BM_2':
                features[f'{var}_time_high'] = (values > 0.72).sum(axis=1) / values.notna().sum(axis=1)
                features[f'{var}_ever_high'] = (values > 0.72).any(axis=1).astype(int)
                
            elif var == 'plt':
                features[f'{var}_declining'] = (features[f'{var}_slope'] < -5).astype(int)
        
        return features
    
    def add_static_features(self, df, features):
        """Add static features."""
        static_cols = ['gender', 'T2DM', 'Hypertension', 'Dyslipidaemia', 'bariatric_surgery']
        
        for col in static_cols:
            if col in df.columns:
                features[col] = df[col]
        
        # Key interaction only
        if 'T2DM' in features.columns and 'fib4_max' in features.columns:
            features['T2DM_x_fib4_max'] = features['T2DM'] * features['fib4_max']
        
        # Time features
        age_cols = [c for c in df.columns if c.startswith('Age_v')]
        if age_cols:
            features['age_baseline'] = df[age_cols].min(axis=1)
            features['follow_up_years'] = df[age_cols].max(axis=1) - df[age_cols].min(axis=1)
        
        return features
    
    def transform(self, df):
        """Full pipeline."""
        df = self.calculate_clinical_scores(df)
        features = self.extract_trajectory_features(df)
        features = self.add_static_features(df, features)
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
        is_event = df[event_col] == 1
        invalid = is_event & df[age_occur_col].isna()
        df_valid = df[~invalid].copy()
    else:
        event_col = 'death'
        age_occur_col = 'death_age_occur'
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
        name_event='Event',
        name_time='Time'
    )
    
    return df_valid, y


def cross_validate_with_feature_selection(X, y, n_folds=5, max_features=30):
    """
    PROPER CV: Feature selection inside each fold to prevent leakage.
    """
    from scipy.stats import pearsonr
    
    event_indicator = y[y.dtype.names[0]]
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    
    oof_preds = np.zeros(len(X))
    fold_cis = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, event_indicator)):
        logger.info(f"\nFold {fold + 1}/{n_folds}")
        
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Feature selection ONLY on training data (NO LEAKAGE)
        train_events = y_train_fold[y_train_fold.dtype.names[0]]
        feature_scores = []
        
        for col in X_train_fold.columns:
            if X_train_fold[col].isna().mean() > 0.5:  # Skip if >50% missing
                continue
            values = X_train_fold[col].fillna(X_train_fold[col].median())
            corr, _ = pearsonr(values, train_events)
            feature_scores.append((col, abs(corr)))
        
        # Select top features
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        selected_features = [f[0] for f in feature_scores[:max_features]]
        
        logger.info(f"  Selected {len(selected_features)} features")
        
        # Preprocess
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        
        X_train_proc = scaler.fit_transform(imputer.fit_transform(X_train_fold[selected_features]))
        X_val_proc = scaler.transform(imputer.transform(X_val_fold[selected_features]))
        
        # Train with HEAVY regularization
        model = RandomSurvivalForest(
            n_estimators=300,
            min_samples_leaf=50,  # Very strong regularization
            min_samples_split=100,
            max_features='sqrt',
            max_depth=5,  # Limit tree depth
            n_jobs=-1,
            random_state=RANDOM_STATE
        )
        
        model.fit(X_train_proc, y_train_fold)
        preds = model.predict(X_val_proc)
        oof_preds[val_idx] = preds
        
        ci = concordance_index_censored(
            y_val_fold[y_val_fold.dtype.names[0]],
            y_val_fold[y_val_fold.dtype.names[1]],
            preds
        )[0]
        fold_cis.append(ci)
        logger.info(f"  C-index: {ci:.4f}")
    
    overall_ci = concordance_index_censored(
        event_indicator.astype(bool),
        y[y.dtype.names[1]],
        oof_preds
    )[0]
    
    return overall_ci, oof_preds, fold_cis


def main():
    """Main pipeline with FIXED methodology."""
    logger.info("="*70)
    logger.info("FIXED PIPELINE v3.0 - NO LEAKAGE, HEAVY REGULARIZATION")
    logger.info("="*70)
    
    # Load data
    train_df = pd.read_csv('data/train-df.csv')
    test_df = pd.read_csv('data/test-df.csv')
    
    logger.info(f"\nData loaded: train={train_df.shape}, test={test_df.shape}")
    
    # Feature engineering
    logger.info("\n" + "="*70)
    logger.info("FEATURE ENGINEERING (No Deltas)")
    logger.info("="*70)
    
    engineer = RobustFeatureEngineer()
    X_train = engineer.transform(train_df)
    X_test = engineer.transform(test_df)
    
    common_cols = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]
    
    logger.info(f"\nTotal features: {len(common_cols)}")
    
    # Prepare targets
    logger.info("\n" + "="*70)
    logger.info("MODEL TRAINING (Proper CV with Feature Selection)")
    logger.info("="*70)
    
    df_hep, y_hep = prepare_survival_target(train_df, 'hepatic')
    df_death, y_death = prepare_survival_target(train_df, 'death')
    
    X_hep = X_train.loc[df_hep.index]
    X_death = X_train.loc[df_death.index]
    
    logger.info(f"\nHepatic: {len(df_hep)} patients, {y_hep['Event'].sum()} events")
    logger.info(f"Death: {len(df_death)} patients, {y_death['Event'].sum()} events")
    
    # Train hepatic model
    logger.info("\n" + "="*70)
    logger.info("HEPATIC EVENTS MODEL")
    logger.info("="*70)
    
    ci_hep, oof_hep, fold_cis_hep = cross_validate_with_feature_selection(
        X_hep, y_hep, n_folds=N_FOLDS, max_features=25
    )
    
    logger.info(f"\nOverall OOF C-index: {ci_hep:.4f}")
    logger.info(f"Fold CIs: {[f'{c:.4f}' for c in fold_cis_hep]}")
    
    # Train death model
    logger.info("\n" + "="*70)
    logger.info("DEATH MODEL")
    logger.info("="*70)
    
    ci_death, oof_death, fold_cis_death = cross_validate_with_feature_selection(
        X_death, y_death, n_folds=N_FOLDS, max_features=25
    )
    
    logger.info(f"\nOverall OOF C-index: {ci_death:.4f}")
    logger.info(f"Fold CIs: {[f'{c:.4f}' for c in fold_cis_death]}")
    
    # Generate test predictions (using all data)
    logger.info("\n" + "="*70)
    logger.info("GENERATING SUBMISSION")
    logger.info("="*70)
    
    # Hepatic predictions
    event_indicator = y_hep['Event']
    feature_scores = []
    for col in X_hep.columns:
        if X_hep[col].isna().mean() > 0.5:
            continue
        values = X_hep[col].fillna(X_hep[col].median())
        corr = np.corrcoef(values, event_indicator)[0, 1]
        feature_scores.append((col, abs(corr)))
    feature_scores.sort(key=lambda x: x[1], reverse=True)
    selected_hep = [f[0] for f in feature_scores[:25]]
    
    imputer_hep = SimpleImputer(strategy='median')
    scaler_hep = StandardScaler()
    X_hep_proc = scaler_hep.fit_transform(imputer_hep.fit_transform(X_hep[selected_hep]))
    X_test_hep = scaler_hep.transform(imputer_hep.transform(X_test[selected_hep]))
    
    model_hep = RandomSurvivalForest(
        n_estimators=300, min_samples_leaf=50, min_samples_split=100,
        max_features='sqrt', max_depth=5, n_jobs=-1, random_state=RANDOM_STATE
    )
    model_hep.fit(X_hep_proc, y_hep)
    pred_hep = model_hep.predict(X_test_hep)
    
    # Death predictions
    event_indicator = y_death['Event']
    feature_scores = []
    for col in X_death.columns:
        if X_death[col].isna().mean() > 0.5:
            continue
        values = X_death[col].fillna(X_death[col].median())
        corr = np.corrcoef(values, event_indicator)[0, 1]
        feature_scores.append((col, abs(corr)))
    feature_scores.sort(key=lambda x: x[1], reverse=True)
    selected_death = [f[0] for f in feature_scores[:25]]
    
    imputer_death = SimpleImputer(strategy='median')
    scaler_death = StandardScaler()
    X_death_proc = scaler_death.fit_transform(imputer_death.fit_transform(X_death[selected_death]))
    X_test_death = scaler_death.transform(imputer_death.transform(X_test[selected_death]))
    
    model_death = RandomSurvivalForest(
        n_estimators=300, min_samples_leaf=50, min_samples_split=100,
        max_features='sqrt', max_depth=5, n_jobs=-1, random_state=RANDOM_STATE
    )
    model_death.fit(X_death_proc, y_death)
    pred_death = model_death.predict(X_test_death)
    
    # Save submission
    submission = pd.DataFrame({
        'trustii_id': test_df['trustii_id'].values,
        'risk_hepatic_event': pred_hep,
        'risk_death': pred_death
    })
    
    submission.to_csv('submissions/fixed_v3_submission.csv', index=False)
    
    logger.info(f"\n✅ Submission saved!")
    logger.info(submission.head().to_string())
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("FINAL RESULTS")
    logger.info("="*70)
    logger.info(f"\n🎯 HEPATIC C-index: {ci_hep:.4f}")
    logger.info(f"🎯 DEATH C-index: {ci_death:.4f}")
    logger.info(f"🎯 AVERAGE: {(ci_hep + ci_death)/2:.4f}")
    logger.info("="*70)
    
    results = {
        'hepatic_ci': float(ci_hep),
        'death_ci': float(ci_death),
        'average_ci': float((ci_hep + ci_death)/2),
    }
    
    with open('submissions/fixed_v3_results.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
