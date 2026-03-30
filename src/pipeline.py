"""
ANNITIA MASLD Survival Analysis - Winning Pipeline
Elite Kaggle Grandmaster Implementation

This pipeline implements:
1. Advanced trajectory feature engineering for longitudinal NIT data
2. Medical corroboration to ensure biological plausibility
3. Ensemble survival models (Gradient Boosting + Random Survival Forest)
4. Stratified cross-validation for rare event handling
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
from sksurv.linear_model import CoxnetSurvivalAnalysis
import lightgbm as lgb
from tqdm import tqdm
import logging
from typing import Tuple, Dict, List, Optional
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
RANDOM_STATE = 42
N_FOLDS = 5
MAX_MISSING_RATE = 0.60

# Medical thresholds
FIB4_HIGH_THRESHOLD = 2.67
FIB4_INTERMEDIATE_THRESHOLD = 1.30
LSM_HIGH_THRESHOLD = 8.0
LSM_VERY_HIGH_THRESHOLD = 12.0
FIBROTEST_HIGH_THRESHOLD = 0.72


class TrajectoryFeatureEngineer:
    """
    Extract trajectory features from longitudinal NIT measurements.
    Converts wide-format time series into clinically meaningful summaries.
    """
    
    def __init__(self):
        self.visit_vars = [
            'fibs_stiffness_med_BM_1',  # LSM (kPa)
            'fibrotest_BM_2',            # FibroTest score
            'aixp_aix_result_BM_3',      # AIX
            'alt',                       # Alanine aminotransferase
            'ast',                       # Aspartate aminotransferase
            'plt',                       # Platelets
            'bilirubin',                 # Total bilirubin
            'ggt',                       # Gamma-glutamyl transferase
            'gluc_fast',                 # Fasting glucose
            'chol',                      # Cholesterol
            'triglyc',                   # Triglycerides
            'BMI',                       # Body mass index
        ]
        
    def calculate_clinical_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate FIB-4 and APRI scores for each visit.
        These are the most validated non-invasive fibrosis scores.
        """
        result = df.copy()
        max_visits = 22
        
        logger.info("Calculating clinical scores (FIB-4, APRI) for each visit...")
        
        for visit in range(1, max_visits + 1):
            age_col = f'Age_v{visit}'
            ast_col = f'ast_v{visit}'
            alt_col = f'alt_v{visit}'
            plt_col = f'plt_v{visit}'
            
            # Only calculate if all required columns exist
            required_cols = [age_col, ast_col, alt_col, plt_col]
            if all(col in result.columns for col in required_cols):
                
                # FIB-4 = (Age x AST) / (Platelets x sqrt(ALT))
                result[f'fib4_v{visit}'] = (
                    result[age_col] * result[ast_col] / 
                    (result[plt_col] * np.sqrt(result[alt_col] + 0.001) + 0.001)
                )
                
                # APRI = ((AST / ULN_AST) x 100) / Platelets
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
        
        logger.info("Clinical scores calculated. New variables: fib4, apri, ast_alt_ratio")
        return result
    
    def extract_trajectory_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract trajectory features for each longitudinal variable.
        """
        features = pd.DataFrame(index=df.index)
        
        logger.info("Extracting trajectory features...")
        
        for var in tqdm(self.visit_vars, desc="Processing variables"):
            visit_cols = [c for c in df.columns if c.startswith(f'{var}_v') and c.split('_v')[-1].isdigit()]
            
            if not visit_cols:
                continue
                
            visit_nums = [int(c.split('_v')[-1]) for c in visit_cols]
            sorted_pairs = sorted(zip(visit_nums, visit_cols))
            sorted_cols = [col for _, col in sorted_pairs]
            
            values = df[sorted_cols]
            
            # Extreme values
            features[f'{var}_max'] = values.max(axis=1)
            features[f'{var}_min'] = values.min(axis=1)
            features[f'{var}_mean'] = values.mean(axis=1)
            features[f'{var}_median'] = values.median(axis=1)
            features[f'{var}_std'] = values.std(axis=1)
            
            features[f'{var}_first'] = values.iloc[:, 0]
            features[f'{var}_last'] = values.ffill(axis=1).iloc[:, -1]
            features[f'{var}_range'] = features[f'{var}_max'] - features[f'{var}_min']
            
            # Volatility
            features[f'{var}_cv'] = features[f'{var}_std'] / (features[f'{var}_mean'].abs() + 0.001)
            
            # Trajectory slopes
            def calc_slope(row):
                valid_mask = row.notna()
                if valid_mask.sum() < 2:
                    return 0.0
                
                x = np.arange(len(row))[valid_mask.values]
                y = row[valid_mask].values.astype(float)
                
                if len(y) < 2:
                    return 0.0
                
                slope, _, _, _, _ = stats.linregress(x, y)
                return slope
            
            features[f'{var}_slope'] = values.apply(calc_slope, axis=1)
            
            # Rate of change
            time_span = (df[[c for c in df.columns if c.startswith('Age_v')]].max(axis=1) - 
                        df[[c for c in df.columns if c.startswith('Age_v')]].min(axis=1))
            features[f'{var}_roc'] = (features[f'{var}_last'] - features[f'{var}_first']) / (time_span + 0.001)
            
            # Acceleration
            def calc_acceleration(row):
                valid_mask = row.notna()
                if valid_mask.sum() < 3:
                    return 0.0
                
                y = row[valid_mask].values.astype(float)
                x = np.arange(len(y))
                
                slopes = np.diff(y) / np.diff(x)
                if len(slopes) < 2:
                    return 0.0
                
                return np.mean(np.diff(slopes))
            
            features[f'{var}_accel'] = values.apply(calc_acceleration, axis=1)
            
            # Clinical thresholds
            if var == 'fib4':
                high_risk_visits = (values > FIB4_HIGH_THRESHOLD).sum(axis=1)
                total_visits = values.notna().sum(axis=1)
                features[f'{var}_time_high_risk'] = high_risk_visits / (total_visits + 0.001)
                features[f'{var}_ever_high'] = (values > FIB4_HIGH_THRESHOLD).any(axis=1).astype(int)
                features[f'{var}_ever_intermediate'] = (values > FIB4_INTERMEDIATE_THRESHOLD).any(axis=1).astype(int)
                
                def get_max_visit_num(row):
                    if row.isna().all():
                        return 0
                    max_col = row.idxmax()
                    try:
                        return int(max_col.split('_v')[-1]) if '_v' in str(max_col) else 0
                    except:
                        return 0
                
                features[f'{var}_max_visit_num'] = values.apply(get_max_visit_num, axis=1)
                
            elif var == 'fibs_stiffness_med_BM_1':
                high_risk_visits = (values > LSM_HIGH_THRESHOLD).sum(axis=1)
                total_visits = values.notna().sum(axis=1)
                features[f'{var}_time_high_risk'] = high_risk_visits / (total_visits + 0.001)
                features[f'{var}_ever_high'] = (values > LSM_HIGH_THRESHOLD).any(axis=1).astype(int)
                features[f'{var}_ever_very_high'] = (values > LSM_VERY_HIGH_THRESHOLD).any(axis=1).astype(int)
                
            elif var == 'fibrotest_BM_2':
                high_risk_visits = (values > FIBROTEST_HIGH_THRESHOLD).sum(axis=1)
                total_visits = values.notna().sum(axis=1)
                features[f'{var}_time_high_risk'] = high_risk_visits / (total_visits + 0.001)
                features[f'{var}_ever_high'] = (values > FIBROTEST_HIGH_THRESHOLD).any(axis=1).astype(int)
        
        return features
    
    def extract_cross_nit_concordance(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Extract features measuring agreement between different NITs."""
        logger.info("Extracting cross-NIT concordance features...")
        
        nit_pairs = [
            ('fibs_stiffness_med_BM_1', 'fib4'),
            ('fibs_stiffness_med_BM_1', 'fibrotest_BM_2'),
            ('fib4', 'fibrotest_BM_2'),
        ]
        
        for nit1, nit2 in nit_pairs:
            slope1_col = f'{nit1}_slope'
            slope2_col = f'{nit2}_slope'
            
            if slope1_col in features.columns and slope2_col in features.columns:
                features[f'{nit1}_{nit2}_slope_agree'] = (
                    (features[slope1_col] * features[slope2_col]) > 0
                ).astype(int)
                
                features[f'{nit1}_{nit2}_slope_diff'] = (
                    features[slope1_col] - features[slope2_col]
                ).abs()
                
                ever_high1 = f'{nit1}_ever_high'
                ever_high2 = f'{nit2}_ever_high'
                if ever_high1 in features.columns and ever_high2 in features.columns:
                    features[f'{nit1}_{nit2}_risk_agree'] = (
                        (features[ever_high1] == features[ever_high2])
                    ).astype(int)
        
        slope_cols = [c for c in features.columns if c.endswith('_slope') and 
                     any(nit in c for nit in ['fibs_stiffness', 'fib4', 'fibrotest'])]
        
        if len(slope_cols) >= 2:
            features['all_nits_worsening'] = (
                features[slope_cols].gt(0).all(axis=1)
            ).astype(int)
            features['any_nit_worsening'] = (
                features[slope_cols].gt(0).any(axis=1)
            ).astype(int)
            features['nits_worsening_count'] = features[slope_cols].gt(0).sum(axis=1)
        
        return features
    
    def add_static_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add static demographic and clinical features."""
        static_cols = [
            'gender', 'T2DM', 'Hypertension', 'Dyslipidaemia',
            'bariatric_surgery', 'bariatric_surgery_age'
        ]
        
        for col in static_cols:
            if col in df.columns:
                features[col] = df[col]
        
        if 'T2DM' in features.columns and 'fib4_max' in features.columns:
            features['T2DM_x_fib4_max'] = features['T2DM'] * features['fib4_max']
        
        if 'gender' in features.columns and 'fibs_stiffness_med_BM_1_max' in features.columns:
            features['gender_x_lsm_max'] = features['gender'] * features['fibs_stiffness_med_BM_1_max']
        
        return features
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full feature engineering pipeline."""
        logger.info("Starting feature engineering...")
        
        df = self.calculate_clinical_scores(df)
        features = self.extract_trajectory_features(df)
        features = self.extract_cross_nit_concordance(df, features)
        features = self.add_static_features(df, features)
        
        logger.info(f"Feature engineering complete. Total features: {features.shape[1]}")
        return features


class MedicalCorroboration:
    """Validate that engineered features align with established medical science."""
    
    def __init__(self, min_correlation_threshold: float = 0.05):
        self.min_correlation_threshold = min_correlation_threshold
        self.validation_results = {}
        
    def validate_feature(self, feature_name: str, correlation: float, 
                        expected_direction: str, medical_rationale: str) -> bool:
        is_valid = True
        messages = []
        
        if abs(correlation) < self.min_correlation_threshold:
            messages.append(f"WEAK: |r|={abs(correlation):.3f} < {self.min_correlation_threshold}")
            is_valid = False
        
        if expected_direction == 'positive' and correlation < -0.1:
            messages.append(f"WRONG DIRECTION: expected positive, got r={correlation:.3f}")
            is_valid = False
        elif expected_direction == 'negative' and correlation > 0.1:
            messages.append(f"WRONG DIRECTION: expected negative, got r={correlation:.3f}")
            is_valid = False
        
        status = "PASS" if is_valid else "FLAG"
        logger.info(f"[{status}] {feature_name:40s} | r={correlation:7.3f} | {medical_rationale}")
        if messages:
            logger.info(f"         Notes: {'; '.join(messages)}")
        
        self.validation_results[feature_name] = {
            'correlation': correlation,
            'valid': is_valid,
            'rationale': medical_rationale
        }
        
        return is_valid
    
    def corroborate_features(self, X: pd.DataFrame, y_event: np.ndarray) -> List[str]:
        logger.info("\n" + "="*80)
        logger.info("MEDICAL CORROBORATION ANALYSIS")
        logger.info("="*80)
        logger.info("Validating that features align with established hepatology science...\n")
        
        valid_features = []
        
        medical_expectations = {
            'fib4_max': ('positive', 'Higher maximum FIB-4 indicates worse fibrosis'),
            'fib4_slope': ('positive', 'Increasing FIB-4 suggests disease progression'),
            'fib4_time_high_risk': ('positive', 'More time in high-risk zone = worse outcome'),
            'fib4_ever_high': ('positive', 'Ever having high FIB-4 indicates risk'),
            'fib4_last': ('positive', 'Recent high FIB-4 indicates current risk'),
            'fibs_stiffness_med_BM_1_max': ('positive', 'Higher maximum LSM indicates cirrhosis'),
            'fibs_stiffness_med_BM_1_slope': ('positive', 'Increasing stiffness indicates progression'),
            'fibs_stiffness_med_BM_1_time_high_risk': ('positive', 'Time with elevated LSM indicates risk'),
            'fibs_stiffness_med_BM_1_ever_high': ('positive', 'Ever having high LSM indicates damage'),
            'fibrotest_BM_2_max': ('positive', 'Higher FibroTest indicates more fibrosis'),
            'fibrotest_BM_2_slope': ('positive', 'Worsening FibroTest indicates progression'),
            'plt_min': ('negative', 'Low platelets indicate portal hypertension/advanced fibrosis'),
            'plt_last': ('negative', 'Low recent platelets indicate ongoing liver dysfunction'),
            'plt_slope': ('negative', 'Declining platelets indicate worsening disease'),
            'ast_max': ('positive', 'High AST indicates hepatocyte damage'),
            'ast_alt_ratio_max': ('positive', 'AST/ALT > 1 suggests cirrhosis'),
            'T2DM': ('positive', 'Diabetes accelerates MASLD progression'),
            'age_first': ('positive', 'Older age is major risk factor for fibrosis'),
        }
        
        for feature, (expected_dir, rationale) in medical_expectations.items():
            if feature in X.columns:
                valid_mask = X[feature].notna()
                if valid_mask.sum() < 10:
                    logger.warning(f"[SKIP] {feature}: insufficient non-null values")
                    continue
                
                corr, pvalue = pearsonr(X[feature][valid_mask], y_event[valid_mask])
                is_valid = self.validate_feature(feature, corr, expected_dir, rationale)
                
                if is_valid:
                    valid_features.append(feature)
            else:
                logger.debug(f"[MISSING] {feature}: not in feature set")
        
        logger.info("\n" + "="*80)
        logger.info(f"CORROBORATION SUMMARY: {len(valid_features)}/{len(medical_expectations)} key features validated")
        logger.info("="*80 + "\n")
        
        return valid_features


class SurvivalModelEnsemble:
    """Ensemble of survival models with stratified cross-validation."""
    
    def __init__(self, n_folds: int = 5, random_state: int = 42):
        self.n_folds = n_folds
        self.random_state = random_state
        self.models_hepatic = []
        self.models_death = []
        self.oof_preds_hepatic = None
        self.oof_preds_death = None
        
    def prepare_survival_target(self, df: pd.DataFrame, outcome: str = 'hepatic') -> Tuple[pd.DataFrame, np.ndarray]:
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
        
        y = Surv.from_arrays(
            event=is_event_v.values,
            time=time_values,
            name_event=name,
            name_time='Time'
        )
        
        logger.info(f"{outcome.upper()} target prepared:")
        logger.info(f"  Patients: {len(df_valid)}")
        logger.info(f"  Events: {is_event_v.sum()} ({100*is_event_v.mean():.1f}%)")
        logger.info(f"  Time range: {time_values.min():.2f} - {time_values.max():.2f} years")
        
        return df_valid, y
    
    def create_preprocessing_pipeline(self, max_missing_rate: float = 0.60):
        return Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ])
    
    def filter_features_by_missing_rate(self, X: pd.DataFrame, max_missing_rate: float = 0.60) -> List[str]:
        missing_rates = X.isna().mean()
        keep_cols = missing_rates[missing_rates <= max_missing_rate].index.tolist()
        logger.info(f"Feature filtering: {len(keep_cols)}/{len(X.columns)} retained (missing <= {max_missing_rate})")
        return keep_cols
    
    def fit_fold(self, X_train: pd.DataFrame, y_train: np.ndarray, 
                 X_val: pd.DataFrame, model_type: str = 'rsf'):
        if model_type == 'rsf':
            model = RandomSurvivalForest(
                n_estimators=500,
                min_samples_leaf=30,
                min_samples_split=60,
                max_features='sqrt',
                n_jobs=-1,
                random_state=self.random_state
            )
        elif model_type == 'gbs':
            model = GradientBoostingSurvivalAnalysis(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.8,
                random_state=self.random_state
            )
        elif model_type == 'cox':
            model = CoxnetSurvivalAnalysis(
                l1_ratio=0.9,
                fit_baseline_model=True
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        preprocessor = self.create_preprocessing_pipeline()
        X_train_processed = preprocessor.fit_transform(X_train)
        X_val_processed = preprocessor.transform(X_val)
        
        model.fit(X_train_processed, y_train)
        risk_scores = model.predict(X_val_processed)
        
        return model, preprocessor, risk_scores
    
    def cross_validate(self, X: pd.DataFrame, y: np.ndarray, 
                      outcome: str = 'hepatic') -> Dict[str, float]:
        logger.info(f"\nStarting {self.n_folds}-fold stratified CV for {outcome}...")
        
        keep_cols = self.filter_features_by_missing_rate(X, MAX_MISSING_RATE)
        X = X[keep_cols]
        
        event_indicator = y[y.dtype.names[0]]
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, 
                             random_state=self.random_state)
        
        oof_preds = np.zeros(len(X))
        fold_cindices = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, event_indicator)):
            logger.info(f"\nFold {fold + 1}/{self.n_folds}")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            models_this_fold = []
            preds_this_fold = []
            
            for model_type in ['rsf', 'gbs']:
                try:
                    model, preprocessor, preds = self.fit_fold(
                        X_train, y_train, X_val, model_type
                    )
                    models_this_fold.append((model_type, model, preprocessor))
                    preds_this_fold.append(preds)
                    
                    ci = concordance_index_censored(
                        y_val[y_val.dtype.names[0]],
                        y_val[y_val.dtype.names[1]],
                        preds
                    )[0]
                    logger.info(f"  {model_type.upper()} C-index: {ci:.4f}")
                    
                except Exception as e:
                    logger.warning(f"  {model_type.upper()} failed: {e}")
            
            if preds_this_fold:
                ensemble_preds = np.mean(preds_this_fold, axis=0)
                oof_preds[val_idx] = ensemble_preds
                
                ensemble_ci = concordance_index_censored(
                    y_val[y_val.dtype.names[0]],
                    y_val[y_val.dtype.names[1]],
                    ensemble_preds
                )[0]
                logger.info(f"  ENSEMBLE C-index: {ensemble_ci:.4f}")
                fold_cindices.append(ensemble_ci)
        
        overall_ci = concordance_index_censored(
            event_indicator,
            y[y.dtype.names[1]],
            oof_preds
        )[0]
        
        logger.info(f"\n{'='*50}")
        logger.info(f"{outcome.upper()} CV RESULTS:")
        logger.info(f"  Mean fold C-index: {np.mean(fold_cindices):.4f} (+/- {np.std(fold_cindices):.4f})")
        logger.info(f"  Overall OOF C-index: {overall_ci:.4f}")
        logger.info(f"{'='*50}\n")
        
        return {
            'oof_preds': oof_preds,
            'overall_ci': overall_ci,
            'fold_cindices': fold_cindices
        }
    
    def fit_final_models(self, X: pd.DataFrame, y: np.ndarray, 
                        outcome: str = 'hepatic') -> List[Tuple]:
        logger.info(f"Fitting final models for {outcome}...")
        
        keep_cols = self.filter_features_by_missing_rate(X, MAX_MISSING_RATE)
        X = X[keep_cols]
        
        final_models = []
        
        for model_type in ['rsf', 'gbs']:
            preprocessor = self.create_preprocessing_pipeline()
            X_processed = preprocessor.fit_transform(X)
            
            if model_type == 'rsf':
                model = RandomSurvivalForest(
                    n_estimators=1000,
                    min_samples_leaf=30,
                    min_samples_split=60,
                    max_features='sqrt',
                    n_jobs=-1,
                    random_state=self.random_state
                )
            else:
                model = GradientBoostingSurvivalAnalysis(
                    n_estimators=1000,
                    learning_rate=0.05,
                    max_depth=3,
                    subsample=0.8,
                    random_state=self.random_state
                )
            
            model.fit(X_processed, y)
            final_models.append((model_type, model, preprocessor, keep_cols))
            logger.info(f"  {model_type.upper()} fitted")
        
        return final_models
    
    def predict(self, X: pd.DataFrame, models: List[Tuple]) -> np.ndarray:
        preds = []
        
        for model_type, model, preprocessor, feature_cols in models:
            X_subset = X[feature_cols] if all(c in X.columns for c in feature_cols) else X
            X_processed = preprocessor.transform(X_subset)
            pred = model.predict(X_processed)
            preds.append(pred)
        
        return np.mean(preds, axis=0)


def main():
    logger.info("="*80)
    logger.info("ANNITIA MASLD SURVIVAL ANALYSIS - WINNING PIPELINE")
    logger.info("Elite Kaggle Grandmaster Implementation")
    logger.info("="*80)
    
    logger.info("\nLoading data...")
    train_df = pd.read_csv('data/train-df.csv')
    test_df = pd.read_csv('data/test-df.csv')
    
    logger.info(f"Train shape: {train_df.shape}")
    logger.info(f"Test shape: {test_df.shape}")
    
    logger.info("\n" + "="*80)
    logger.info("PHASE 1: ADVANCED FEATURE ENGINEERING")
    logger.info("="*80)
    
    engineer = TrajectoryFeatureEngineer()
    X_train = engineer.transform(train_df)
    X_test = engineer.transform(test_df)
    
    common_cols = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]
    
    logger.info(f"Final feature matrix: {X_train.shape}")
    
    logger.info("\n" + "="*80)
    logger.info("PHASE 2: MEDICAL CORROBORATION")
    logger.info("="*80)
    
    corroborator = MedicalCorroboration()
    y_hepatic_event = train_df['evenements_hepatiques_majeurs'].values
    validated_features = corroborator.corroborate_features(X_train, y_hepatic_event)
    
    logger.info("\n" + "="*80)
    logger.info("PHASE 3: SURVIVAL MODEL TRAINING")
    logger.info("="*80)
    
    ensemble = SurvivalModelEnsemble(n_folds=N_FOLDS, random_state=RANDOM_STATE)
    
    logger.info("\n--- HEPATIC EVENTS MODEL ---")
    df_hep, y_hep = ensemble.prepare_survival_target(train_df, outcome='hepatic')
    X_hep = X_train.loc[df_hep.index]
    
    cv_results_hep = ensemble.cross_validate(X_hep, y_hep, outcome='hepatic')
    final_models_hep = ensemble.fit_final_models(X_hep, y_hep, outcome='hepatic')
    
    logger.info("\n--- DEATH MODEL ---")
    df_death, y_death = ensemble.prepare_survival_target(train_df, outcome='death')
    X_death = X_train.loc[df_death.index]
    
    cv_results_death = ensemble.cross_validate(X_death, y_death, outcome='death')
    final_models_death = ensemble.fit_final_models(X_death, y_death, outcome='death')
    
    logger.info("\n" + "="*80)
    logger.info("PHASE 4: GENERATING SUBMISSION")
    logger.info("="*80)
    
    pred_hepatic = ensemble.predict(X_test, final_models_hep)
    pred_death = ensemble.predict(X_test, final_models_death)
    
    submission = pd.DataFrame({
        'trustii_id': test_df['trustii_id'].values,
        'risk_hepatic_event': pred_hepatic,
        'risk_death': pred_death
    })
    
    submission_path = 'submissions/survival_ensemble_submission.csv'
    submission.to_csv(submission_path, index=False)
    
    logger.info(f"\nSubmission saved to: {submission_path}")
    logger.info(f"Submission shape: {submission.shape}")
    logger.info(f"\nSubmission preview:")
    logger.info(submission.head(10).to_string())
    
    logger.info("\n" + "="*80)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("="*80)
    logger.info(f"Hepatic Events OOF C-index: {cv_results_hep['overall_ci']:.4f}")
    logger.info(f"Death OOF C-index: {cv_results_death['overall_ci']:.4f}")
    logger.info(f"Average C-index: {(cv_results_hep['overall_ci'] + cv_results_death['overall_ci'])/2:.4f}")
    logger.info("="*80)
    
    results = {
        'hepatic_ci': float(cv_results_hep['overall_ci']),
        'death_ci': float(cv_results_death['overall_ci']),
        'hepatic_fold_cis': [float(x) for x in cv_results_hep['fold_cindices']],
        'death_fold_cis': [float(x) for x in cv_results_death['fold_cindices']],
        'n_features': X_train.shape[1],
        'n_validated_features': len(validated_features)
    }
    
    with open('submissions/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == '__main__':
    results = main()
