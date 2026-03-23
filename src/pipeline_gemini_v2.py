"""
ANNITIA MASLD Survival Analysis - Gemini Recommendations (Fixed Version)
========================================================================

Simplified approach:
1. RepeatedStratifiedKFold (5 folds × 5 repeats)
2. XGBoost with survival:cox + monotonic constraints
3. RSF ensemble (no CoxNet - was failing)
4. EWMA features + explicit last measurement
5. Death model first, then use predictions for hepatic model

Fixes from v1:
- Remove CoxNet (consistently returning 0.5)
- Fix feature alignment between death and hepatic models
- Better handling of NaN values
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from sksurv.ensemble import RandomSurvivalForest
import xgboost as xgb
from tqdm import tqdm
import logging
from typing import Tuple, Dict, List, Optional
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
N_FOLDS = 5
N_REPEATS = 5
MAX_MISSING_RATE = 0.60


class TrajectoryFeatureEngineerV2:
    """Enhanced feature engineering with EWMA and explicit last measurement."""
    
    def __init__(self):
        self.visit_vars = [
            'fibs_stiffness_med_BM_1', 'fibrotest_BM_2', 'aixp_aix_result_BM_3',
            'alt', 'ast', 'plt', 'bilirubin', 'ggt', 'gluc_fast', 'chol', 'triglyc', 'BMI',
        ]
        
    def calculate_clinical_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate FIB-4 and APRI scores for each visit."""
        result = df.copy()
        max_visits = 22
        
        for visit in range(1, max_visits + 1):
            age_col = f'Age_v{visit}'
            ast_col = f'ast_v{visit}'
            alt_col = f'alt_v{visit}'
            plt_col = f'plt_v{visit}'
            
            required_cols = [age_col, ast_col, alt_col, plt_col]
            if all(col in result.columns for col in required_cols):
                result[f'fib4_v{visit}'] = (
                    result[age_col] * result[ast_col] / 
                    (result[plt_col] * np.sqrt(result[alt_col] + 0.001) + 0.001)
                )
                result[f'apri_v{visit}'] = (
                    (result[ast_col] / 40.0) * 100 / (result[plt_col] + 0.001)
                )
                result[f'ast_alt_ratio_v{visit}'] = result[ast_col] / (result[alt_col] + 0.001)
        
        return result
    
    def extract_trajectory_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract enhanced trajectory features."""
        features = pd.DataFrame(index=df.index)
        
        for var in tqdm(self.visit_vars + ['fib4', 'apri', 'ast_alt_ratio'], desc="Features"):
            visit_cols = [c for c in df.columns if c.startswith(f'{var}_v')]
            if len(visit_cols) < 2:
                continue
            
            visit_cols = sorted(visit_cols, key=lambda x: int(x.split('_v')[-1]))
            values = df[visit_cols]
            
            # Basic stats
            features[f'{var}_mean'] = values.mean(axis=1, skipna=True)
            features[f'{var}_median'] = values.median(axis=1, skipna=True)
            features[f'{var}_max'] = values.max(axis=1, skipna=True)
            features[f'{var}_min'] = values.min(axis=1, skipna=True)
            features[f'{var}_std'] = values.std(axis=1, skipna=True)
            
            # Explicit last non-null measurement
            def get_last_non_null(row):
                non_null = row.dropna()
                return non_null.iloc[-1] if len(non_null) > 0 else np.nan
            features[f'{var}_last_actual'] = values.apply(get_last_non_null, axis=1)
            
            # First measurement
            features[f'{var}_first'] = values.iloc[:, 0]
            
            # EWMA (span=3, span=5)
            for span in [3, 5]:
                ewma_values = values.T.ewm(span=span, min_periods=1).mean().T
                features[f'{var}_ewma_span{span}'] = ewma_values.apply(
                    lambda row: row.dropna().iloc[-1] if len(row.dropna()) > 0 else np.nan, axis=1
                )
                features[f'{var}_ewma_span{span}_max'] = ewma_values.max(axis=1, skipna=True)
            
            # Slope and ROC
            def calc_slope(row):
                valid_mask = row.notna()
                if valid_mask.sum() < 2:
                    return np.nan
                y = row[valid_mask].values.astype(float)
                x = np.arange(len(y))
                if np.std(y) < 1e-10:
                    return 0.0
                slope, _, _, _, _ = stats.linregress(x, y)
                return slope
            
            features[f'{var}_slope'] = values.apply(calc_slope, axis=1)
            time_span = values.notna().sum(axis=1).replace(0, 1)
            features[f'{var}_roc'] = (features[f'{var}_last_actual'] - features[f'{var}_first']) / time_span
            features[f'{var}_cv'] = features[f'{var}_std'] / (features[f'{var}_mean'].abs() + 0.001)
            
            # Medical thresholds
            if var == 'fib4':
                high_visits = (values > 2.67).sum(axis=1)
                total_visits = values.notna().sum(axis=1)
                features[f'{var}_time_high_risk'] = high_visits / (total_visits + 0.001)
                features[f'{var}_ever_high'] = (values > 2.67).any(axis=1).astype(int)
            elif var == 'fibs_stiffness_med_BM_1':
                high_visits = (values > 8.0).sum(axis=1)
                very_high_visits = (values > 12.0).sum(axis=1)
                total_visits = values.notna().sum(axis=1)
                features[f'{var}_time_high'] = high_visits / (total_visits + 0.001)
                features[f'{var}_time_very_high'] = very_high_visits / (total_visits + 0.001)
                features[f'{var}_ever_high'] = (values > 8.0).any(axis=1).astype(int)
        
        return features
    
    def extract_static_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract static features."""
        features = pd.DataFrame(index=df.index)
        
        static_cols = ['gender', 'T2DM', 'Hypertension', 'Dyslipidaemia']
        for col in static_cols:
            if col in df.columns:
                features[col] = df[col]
        
        age_cols = [c for c in df.columns if c.startswith('Age_v')]
        if age_cols:
            features['age_baseline'] = df[age_cols].min(axis=1)
            features['follow_up_years'] = df[age_cols].max(axis=1) - df[age_cols].min(axis=1)
        
        return features
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full feature engineering."""
        df_with_scores = self.calculate_clinical_scores(df)
        trajectory_features = self.extract_trajectory_features(df_with_scores)
        static_features = self.extract_static_features(df)
        
        all_features = pd.concat([trajectory_features, static_features], axis=1)
        logger.info(f"Features: {all_features.shape[1]}")
        return all_features


class XGBSurvivalModel:
    """XGBoost with survival:cox objective."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        
    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else None
        
        event = y[y.dtype.names[0]].astype(int)
        time = y[y.dtype.names[1]]
        labels = np.where(event, time, -time)
        
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        dtrain = xgb.DMatrix(X_array, label=labels, feature_names=self.feature_names)
        
        params = {
            'objective': 'survival:cox',
            'max_depth': 3,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_lambda': 10.0,
            'reg_alpha': 1.0,
            'min_child_weight': 10,
            'random_state': self.random_state,
            'verbosity': 0,
        }
        
        self.model = xgb.train(params, dtrain, num_boost_round=500, verbose_eval=False)
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        dtest = xgb.DMatrix(X_array, feature_names=self.feature_names)
        return self.model.predict(dtest)


class SimpleEnsemble:
    """Simple RSF + XGB ensemble (no CoxNet)."""
    
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {'rsf': 0.6, 'xgb': 0.4}
        self.models = {}
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        self.feature_cols = None
        
    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """Fit ensemble models."""
        self.feature_cols = X.columns.tolist()
        
        # Preprocess
        X_array = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_array)
        
        # RSF
        self.models['rsf'] = RandomSurvivalForest(
            n_estimators=500,
            min_samples_leaf=20,
            min_samples_split=40,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42
        )
        self.models['rsf'].fit(X_scaled, y)
        
        # XGB
        self.models['xgb'] = XGBSurvivalModel(random_state=42)
        self.models['xgb'].fit(pd.DataFrame(X_array, columns=self.feature_cols), y)
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Average predictions."""
        X_array = self.imputer.transform(X[self.feature_cols])
        X_scaled = self.scaler.transform(X_array)
        
        rsf_preds = self.models['rsf'].predict(X_scaled)
        xgb_preds = self.models['xgb'].predict(pd.DataFrame(X_array, columns=self.feature_cols))
        
        # Normalize to same scale before averaging
        rsf_norm = (rsf_preds - rsf_preds.mean()) / (rsf_preds.std() + 1e-8)
        xgb_norm = (xgb_preds - xgb_preds.mean()) / (xgb_preds.std() + 1e-8)
        
        return self.weights['rsf'] * rsf_norm + self.weights['xgb'] * xgb_norm


def prepare_survival_target(df: pd.DataFrame, outcome: str = 'hepatic') -> Tuple[pd.DataFrame, np.ndarray]:
    """Convert raw targets to sksurv format."""
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
    else:
        raise ValueError(f"Unknown outcome: {outcome}")
    
    is_event_v = (df_valid[event_col] == 1)
    time_values = np.where(
        is_event_v,
        df_valid[age_occur_col] - df_valid['first_visit_age'],
        df_valid['last_observed_age'] - df_valid['first_visit_age']
    ).astype(float)
    time_values = np.maximum(time_values, 0.001)
    
    y = Surv.from_arrays(event=is_event_v.values, time=time_values, name_event=name, name_time='Time')
    
    logger.info(f"{outcome.upper()}: {len(df_valid)} patients, {is_event_v.sum()} events ({100*is_event_v.mean():.1f}%)")
    return df_valid, y


def cross_validate(X: pd.DataFrame, y: np.ndarray, n_folds: int = 5, n_repeats: int = 5) -> Dict:
    """Repeated stratified CV."""
    logger.info(f"\n{n_folds}-fold × {n_repeats}-repeat CV...")
    
    # Filter features
    keep_cols = X.columns[X.isnull().mean() <= MAX_MISSING_RATE].tolist()
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
        
        # Fit ensemble
        ensemble = SimpleEnsemble(weights={'rsf': 0.6, 'xgb': 0.4})
        ensemble.fit(X_train, y_train)
        preds = ensemble.predict(X_val)
        
        oof_preds[val_idx] = preds
        
        ci = concordance_index_censored(y_val[y_val.dtype.names[0]], y_val[y_val.dtype.names[1]], preds)[0]
        fold_cindices.append(ci)
        
        if fold_num == 1:
            logger.info(f"  Repeat {repeat_idx + 1}: fold {fold_num} CI={ci:.4f}")
    
    overall_ci = concordance_index_censored(event_indicator, y[y.dtype.names[1]], oof_preds)[0]
    
    logger.info(f"\n  Overall OOF C-index: {overall_ci:.4f}")
    logger.info(f"  Fold C-indices: {np.mean(fold_cindices):.4f} (+/- {np.std(fold_cindices):.4f})")
    
    return {
        'oof_preds': oof_preds,
        'overall_ci': overall_ci,
        'fold_cindices': fold_cindices,
        'feature_cols': keep_cols
    }


def main():
    """Main execution."""
    logger.info("="*80)
    logger.info("GEMINI PIPELINE V2 - Simplified (RSF + XGB only)")
    logger.info("="*80)
    
    start_time = datetime.now()
    
    # Load data
    logger.info("\nLoading data...")
    train_df = pd.read_csv('data/train-df.csv')
    test_df = pd.read_csv('data/test-df.csv')
    
    # Feature Engineering
    logger.info("\nFeature Engineering...")
    engineer = TrajectoryFeatureEngineerV2()
    X_train = engineer.transform(train_df)
    X_test = engineer.transform(test_df)
    
    # Align
    common_cols = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]
    
    # === DEATH MODEL FIRST ===
    logger.info("\n--- DEATH MODEL ---")
    df_death, y_death = prepare_survival_target(train_df, outcome='death')
    X_death = X_train.loc[df_death.index]
    
    death_cv = cross_validate(X_death, y_death, n_folds=N_FOLDS, n_repeats=N_REPEATS)
    
    # Fit final death model
    death_ensemble = SimpleEnsemble(weights={'rsf': 0.6, 'xgb': 0.4})
    death_ensemble.fit(X_death, y_death)
    
    # === HEPATIC MODEL WITH DEATH FEATURE ===
    logger.info("\n--- HEPATIC MODEL (with death predictions) ---")
    df_hep, y_hep = prepare_survival_target(train_df, outcome='hepatic')
    X_hep = X_train.loc[df_hep.index].copy()
    
    # Add death OOF predictions as feature
    X_hep['death_risk_pred'] = np.nan
    common_idx = X_hep.index.intersection(df_death.index)
    X_hep.loc[common_idx, 'death_risk_pred'] = death_cv['oof_preds'][df_death.index.get_indexer(common_idx)]
    X_hep['death_risk_pred'] = X_hep['death_risk_pred'].fillna(X_hep['death_risk_pred'].median())
    
    hep_cv = cross_validate(X_hep, y_hep, n_folds=N_FOLDS, n_repeats=N_REPEATS)
    
    # Fit final hepatic model
    # Get death predictions for all hepatic patients
    death_preds_for_hep = death_ensemble.predict(X_hep)
    X_hep_final = X_train.loc[df_hep.index].copy()
    X_hep_final['death_risk_pred'] = death_preds_for_hep
    
    hep_ensemble = SimpleEnsemble(weights={'rsf': 0.6, 'xgb': 0.4})
    hep_ensemble.fit(X_hep_final, y_hep)
    
    # === GENERATE SUBMISSION ===
    logger.info("\n--- GENERATING SUBMISSION ---")
    
    # Death predictions
    pred_death = death_ensemble.predict(X_test)
    
    # Hepatic predictions (with death feature)
    X_test_hep = X_test.copy()
    X_test_hep['death_risk_pred'] = pred_death
    pred_hepatic = hep_ensemble.predict(X_test_hep)
    
    submission = pd.DataFrame({
        'trustii_id': test_df['trustii_id'].values,
        'risk_hepatic_event': pred_hepatic,
        'risk_death': pred_death
    })
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = f'submissions/gemini_v2_submission_{timestamp}.csv'
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
    
    # Save results
    results = {
        'timestamp': timestamp,
        'death_ci': float(death_cv['overall_ci']),
        'hepatic_ci': float(hep_cv['overall_ci']),
        'death_fold_std': float(np.std(death_cv['fold_cindices'])),
        'hepatic_fold_std': float(np.std(hep_cv['fold_cindices'])),
        'n_features': len(common_cols) + 1,  # +1 for death_risk_pred
        'elapsed_minutes': (datetime.now() - start_time).total_seconds() / 60
    }
    
    with open(f'submissions/gemini_v2_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nElapsed: {results['elapsed_minutes']:.1f} minutes")
    
    return results


if __name__ == "__main__":
    main()
