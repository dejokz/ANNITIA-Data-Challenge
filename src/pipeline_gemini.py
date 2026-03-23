"""
ANNITIA MASLD Survival Analysis - Gemini Recommendations Implementation
======================================================================

Key Improvements:
1. RepeatedStratifiedKFold (5 folds × 5 repeats) for stable CV
2. XGBoost with survival:cox objective + monotonic constraints
3. Heavily regularized CoxNet with rank-averaging ensemble
4. EWMA features + explicit last measurement
5. Death model OOF predictions as feature for Hepatic model
6. Comprehensive training diagnostics

Author: Enhanced Pipeline
Branch: gemni-recom
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, rankdata
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis
import xgboost as xgb
from tqdm import tqdm
import logging
from typing import Tuple, Dict, List, Optional
import json
from dataclasses import dataclass
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
RANDOM_STATE = 42
N_FOLDS = 5
N_REPEATS = 5
MAX_MISSING_RATE = 0.60

# Medical thresholds
FIB4_HIGH_THRESHOLD = 2.67
FIB4_INTERMEDIATE_THRESHOLD = 1.30
LSM_HIGH_THRESHOLD = 8.0
LSM_VERY_HIGH_THRESHOLD = 12.0
FIBROTEST_HIGH_THRESHOLD = 0.72


@dataclass
class CVResult:
    """Store CV results with diagnostics."""
    oof_preds: np.ndarray
    overall_ci: float
    fold_cindices: List[float]
    repeat_cindices: List[float]
    feature_importance: Optional[pd.DataFrame] = None
    model_disagreement: Optional[Dict] = None


class TrajectoryFeatureEngineerV2:
    """
    Enhanced feature engineering with EWMA and explicit last measurement.
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
        """Calculate FIB-4 and APRI scores for each visit."""
        result = df.copy()
        max_visits = 22
        
        logger.info("Calculating clinical scores (FIB-4, APRI) for each visit...")
        
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
                # ULN_AST typically 40 U/L
                result[f'apri_v{visit}'] = (
                    (result[ast_col] / 40.0) * 100 / 
                    (result[plt_col] + 0.001)
                )
                
                # AST/ALT ratio
                result[f'ast_alt_ratio_v{visit}'] = (
                    result[ast_col] / (result[alt_col] + 0.001)
                )
        
        new_cols = [c for c in result.columns if c not in df.columns]
        logger.info(f"  Clinical scores calculated. New variables: {', '.join([c.split('_')[0] for c in new_cols[:3]])}")
        
        return result
    
    def extract_trajectory_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract enhanced trajectory features including EWMA and explicit last measurement.
        """
        features = pd.DataFrame(index=df.index)
        
        logger.info("Extracting trajectory features (including EWMA and last measurement)...")
        
        for var in tqdm(self.visit_vars + ['fib4', 'apri', 'ast_alt_ratio'], desc="Processing variables"):
            # Get all visit columns for this variable
            visit_cols = [c for c in df.columns if c.startswith(f'{var}_v')]
            if len(visit_cols) < 2:
                continue
            
            visit_cols = sorted(visit_cols, key=lambda x: int(x.split('_v')[-1]))
            values = df[visit_cols]
            
            # === BASIC STATS ===
            features[f'{var}_mean'] = values.mean(axis=1, skipna=True)
            features[f'{var}_median'] = values.median(axis=1, skipna=True)
            features[f'{var}_max'] = values.max(axis=1, skipna=True)
            features[f'{var}_min'] = values.min(axis=1, skipna=True)
            features[f'{var}_std'] = values.std(axis=1, skipna=True)
            
            # === EXPLICIT LAST NON-NULL MEASUREMENT ===
            # Get the actual last available measurement (not forward-filled)
            def get_last_non_null(row):
                non_null = row.dropna()
                return non_null.iloc[-1] if len(non_null) > 0 else np.nan
            
            features[f'{var}_last_actual'] = values.apply(get_last_non_null, axis=1)
            
            # First measurement
            features[f'{var}_first'] = values.iloc[:, 0]
            
            # Forward-fill last (for comparison with last_actual)
            features[f'{var}_last_ffill'] = values.ffill(axis=1).iloc[:, -1]
            
            # === EWMA (Exponentially Weighted Moving Average) ===
            # Span=3: more weight to recent visits
            for span in [3, 5]:
                ewma_values = values.T.ewm(span=span, min_periods=1).mean().T
                # Last EWMA value
                features[f'{var}_ewma_span{span}'] = ewma_values.apply(
                    lambda row: row.dropna().iloc[-1] if len(row.dropna()) > 0 else np.nan, 
                    axis=1
                )
                # Max EWMA (peak risk)
                features[f'{var}_ewma_span{span}_max'] = ewma_values.max(axis=1, skipna=True)
            
            # === TRAJECTORY SLOPES ===
            def calc_slope(row):
                """Linear regression slope over time."""
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
            
            # === RATE OF CHANGE (ROC) ===
            time_span = values.notna().sum(axis=1).replace(0, 1)
            features[f'{var}_roc'] = (features[f'{var}_last_actual'] - features[f'{var}_first']) / time_span
            
            # === VOLATILITY ===
            features[f'{var}_cv'] = features[f'{var}_std'] / (features[f'{var}_mean'].abs() + 0.001)
            
            # === MEDICAL THRESHOLD FEATURES ===
            if var == 'fib4':
                # Time in high-risk zone
                high_visits = (values > FIB4_HIGH_THRESHOLD).sum(axis=1)
                total_visits = values.notna().sum(axis=1)
                features[f'{var}_time_high_risk'] = high_visits / (total_visits + 0.001)
                features[f'{var}_ever_high'] = (values > FIB4_HIGH_THRESHOLD).any(axis=1).astype(int)
                
            elif var == 'fibs_stiffness_med_BM_1':
                # LSM thresholds
                high_visits = (values > LSM_HIGH_THRESHOLD).sum(axis=1)
                very_high_visits = (values > LSM_VERY_HIGH_THRESHOLD).sum(axis=1)
                total_visits = values.notna().sum(axis=1)
                
                features[f'{var}_time_high'] = high_visits / (total_visits + 0.001)
                features[f'{var}_time_very_high'] = very_high_visits / (total_visits + 0.001)
                features[f'{var}_ever_high'] = (values > LSM_HIGH_THRESHOLD).any(axis=1).astype(int)
                features[f'{var}_ever_very_high'] = (values > LSM_VERY_HIGH_THRESHOLD).any(axis=1).astype(int)
                
            elif var == 'fibrotest_BM_2':
                high_visits = (values > FIBROTEST_HIGH_THRESHOLD).sum(axis=1)
                total_visits = values.notna().sum(axis=1)
                features[f'{var}_time_high'] = high_visits / (total_visits + 0.001)
                features[f'{var}_ever_high'] = (values > FIBROTEST_HIGH_THRESHOLD).any(axis=1).astype(int)
        
        return features
    
    def extract_cross_nit_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract cross-NIT concordance features."""
        features = pd.DataFrame(index=df.index)
        
        logger.info("Extracting cross-NIT concordance features...")
        
        nit_pairs = [
            ('fibs_stiffness_med_BM_1', 'fib4'),
            ('fibs_stiffness_med_BM_1', 'fibrotest_BM_2'),
            ('fib4', 'fibrotest_BM_2'),
        ]
        
        for nit1, nit2 in nit_pairs:
            # Check if they move in same direction (agreement)
            worsening_1 = df.get(f'{nit1}_worsening', pd.Series(0, index=df.index))
            worsening_2 = df.get(f'{nit2}_worsening', pd.Series(0, index=df.index))
            
            features[f'{nit1}_{nit2}_agreement'] = (worsening_1 == worsening_2).astype(int)
        
        # Composite risk indicator
        fibrosis_markers = [
            'fibs_stiffness_med_BM_1_worsening',
            'fib4_worsening',
            'fibrotest_BM_2_worsening'
        ]
        
        available_markers = [m for m in fibrosis_markers if m in df.columns]
        if available_markers:
            features['n_worsening_markers'] = df[available_markers].sum(axis=1)
            features['all_worsening'] = (features['n_worsening_markers'] == len(available_markers)).astype(int)
        
        return features
    
    def extract_static_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract static demographic and clinical features."""
        features = pd.DataFrame(index=df.index)
        
        # Basic demographics
        static_cols = ['gender', 'T2DM', 'Hypertension', 'Dyslipidaemia']
        for col in static_cols:
            if col in df.columns:
                features[col] = df[col]
        
        # Age at baseline
        age_cols = [c for c in df.columns if c.startswith('Age_v')]
        if age_cols:
            features['age_baseline'] = df[age_cols].min(axis=1)
            features['age_last'] = df[age_cols].max(axis=1)
            features['follow_up_years'] = (df[age_cols].max(axis=1) - df[age_cols].min(axis=1))
        
        # Key interactions
        if 'T2DM' in df.columns and 'fib4_max' in df.columns:
            features['T2DM_x_fib4_max'] = df['T2DM'] * df.get('fib4_max', 0)
        
        if 'gender' in df.columns and 'fibs_stiffness_med_BM_1_max' in df.columns:
            features['gender_x_lsm_max'] = df['gender'] * df.get('fibs_stiffness_med_BM_1_max', 0)
        
        return features
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full feature engineering pipeline."""
        # Calculate clinical scores
        df_with_scores = self.calculate_clinical_scores(df)
        
        # Extract trajectory features
        trajectory_features = self.extract_trajectory_features(df_with_scores)
        
        # Extract cross-NIT features
        cross_nit_features = self.extract_cross_nit_features(trajectory_features)
        
        # Extract static features
        static_features = self.extract_static_features(df)
        
        # Combine all features
        all_features = pd.concat([
            trajectory_features,
            cross_nit_features,
            static_features
        ], axis=1)
        
        logger.info(f"Feature engineering complete. Total features: {all_features.shape[1]}")
        
        return all_features


class MedicalCorroboration:
    """Validate that features have medically plausible correlations with outcome."""
    
    def __init__(self):
        self.expected_correlations = {
            'fib4_max': 'positive',
            'fib4_time_high_risk': 'positive',
            'fibs_stiffness_med_BM_1_max': 'positive',
            'fibs_stiffness_med_BM_1_time_high': 'positive',
            'plt_min': 'negative',
        }
    
    def corroborate_features(self, X: pd.DataFrame, y_event: np.ndarray,
                            p_threshold: float = 0.05) -> List[str]:
        """Validate features and return list of corroborated features."""
        logger.info("\nValidating medical plausibility...")
        
        validated = []
        
        for feature, expected_dir in self.expected_correlations.items():
            if feature not in X.columns:
                continue
            
            valid_mask = X[feature].notna() & ~np.isnan(y_event)
            if valid_mask.sum() < 10:
                continue
            
            x_vals = X.loc[valid_mask, feature].values
            y_vals = y_event[valid_mask]
            
            # Pearson correlation
            r, p = pearsonr(x_vals, y_vals)
            
            # Check direction
            actual_dir = 'positive' if r > 0 else 'negative'
            direction_match = actual_dir == expected_dir
            significant = p < p_threshold
            
            if direction_match and significant:
                status = "[PASS]"
                validated.append(feature)
            else:
                status = "[WARN]"
            
            logger.info(f"  {status} {feature:40s} | r={r:6.3f} | p={p:.3f}")
        
        logger.info(f"\nValidated {len(validated)}/{len(self.expected_correlations)} key features")
        return validated


class XGBSurvivalModel:
    """XGBoost with survival:cox objective and monotonic constraints."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.monotone_constraints = None
        self.feature_names = None
        
    def fit(self, X: pd.DataFrame, y: np.ndarray, feature_names: List[str] = None):
        """Fit XGBoost survival model with monotonic constraints."""
        
        # Store feature names
        if feature_names is not None:
            self.feature_names = feature_names
        elif isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        # Set monotonic constraints based on feature names
        if self.feature_names:
            self.monotone_constraints = self._get_monotone_constraints(self.feature_names)
        
        # Prepare data for XGBoost
        # y is sksurv structured array with 'event' and 'time'
        event = y[y.dtype.names[0]].astype(int)
        time = y[y.dtype.names[1]]
        
        # XGBoost survival:cox expects: label = time if event else -time
        labels = np.where(event, time, -time)
        
        # Convert X to numpy if DataFrame
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        dtrain = xgb.DMatrix(X_array, label=labels, feature_names=self.feature_names)
        
        params = {
            'objective': 'survival:cox',
            'eval_metric': 'cox-nloglik',
            'max_depth': 3,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_lambda': 10.0,  # Strong L2 regularization
            'reg_alpha': 1.0,    # L1 regularization
            'min_child_weight': 10,
            'random_state': self.random_state,
            'verbosity': 0,
        }
        
        if self.monotone_constraints:
            params['monotone_constraints'] = self.monotone_constraints
        
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            verbose_eval=False
        )
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict risk scores (higher = more risk)."""
        # Convert X to numpy if DataFrame
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        dtest = xgb.DMatrix(X_array, feature_names=self.feature_names)
        # XGBoost survival:cox outputs log hazard ratio
        return self.model.predict(dtest)
    
    def _get_monotone_constraints(self, feature_names: List[str]) -> str:
        """
        Generate monotonic constraint string.
        1 = increasing (higher feature value → higher risk)
        -1 = decreasing (higher feature value → lower risk)
        0 = no constraint
        """
        constraints = []
        for name in feature_names:
            name_lower = name.lower()
            
            # Increasing risk features (higher = worse)
            if any(x in name_lower for x in ['lsm', 'fib4', 'fibrotest', 'stiffness', 'ast', 'alt', 'bili', 'ggt']):
                constraints.append('1')
            # Decreasing risk features (higher = better)
            elif any(x in name_lower for x in ['plt', 'platelet']):
                constraints.append('-1')
            # No constraint for others
            else:
                constraints.append('0')
        
        return '(' + ','.join(constraints) + ')'


class CoxNetSurvivalModel:
    """Heavily regularized CoxNet with feature selection."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        
    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """Fit heavily regularized CoxNet."""
        # Scale features (required for regularized Cox)
        X_scaled = self.scaler.fit_transform(X)
        
        # Heavily regularized CoxNet
        self.model = CoxnetSurvivalAnalysis(
            l1_ratio=0.9,  # Heavy L1 for sparsity
            alphas=[0.01, 0.1, 1.0, 10.0],  # Wide range of penalties
            fit_baseline_model=True,
            max_iter=1000
        )
        
        self.model.fit(X_scaled, y)
        
        # Log selected features
        if hasattr(self.model, 'coef_'):
            n_selected = np.sum(np.abs(self.model.coef_) > 1e-6)
            logger.info(f"  CoxNet selected {n_selected}/{X.shape[1]} features")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict risk scores."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class RankEnsemble:
    """Rank-averaging ensemble for C-index optimization."""
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        Args:
            weights: Dict mapping model name to weight (should sum to 1)
        """
        self.weights = weights or {
            'rsf': 0.5,
            'xgb': 0.3,
            'coxnet': 0.2
        }
        self.models = {}
        
    def fit_models(self, X: pd.DataFrame, y: np.ndarray, 
                   feature_names: List[str] = None):
        """Fit all ensemble models."""
        logger.info("Fitting ensemble models...")
        
        # RSF
        logger.info("  Training RSF...")
        self.models['rsf'] = RandomSurvivalForest(
            n_estimators=500,
            min_samples_leaf=20,
            min_samples_split=40,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42
        )
        self.models['rsf'].fit(X, y)
        
        # XGBoost
        logger.info("  Training XGBoost...")
        self.models['xgb'] = XGBSurvivalModel(random_state=42)
        self.models['xgb'].fit(X, y, feature_names)
        
        # CoxNet
        logger.info("  Training CoxNet...")
        self.models['coxnet'] = CoxNetSurvivalModel(random_state=42)
        self.models['coxnet'].fit(X, y)
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using rank averaging.
        
        1. Get raw predictions from each model
        2. Convert to ranks (0-100)
        3. Weighted average of ranks
        """
        rank_preds = {}
        
        for name, model in self.models.items():
            raw_preds = model.predict(X)
            # Convert to percentiles (0-100)
            ranks = rankdata(raw_preds, method='average')
            percentiles = 100 * (ranks - 1) / (len(ranks) - 1)
            rank_preds[name] = percentiles
        
        # Weighted average of ranks
        final_preds = np.zeros(len(X))
        for name, preds in rank_preds.items():
            final_preds += self.weights.get(name, 0) * preds
        
        return final_preds
    
    def get_individual_predictions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Get individual model predictions for analysis."""
        return {name: model.predict(X) for name, model in self.models.items()}


class SurvivalModelEnsembleGemini:
    """
    Enhanced ensemble with Repeated CV, XGBoost, CoxNet, and rank averaging.
    """
    
    def __init__(self, n_folds: int = 5, n_repeats: int = 5, random_state: int = 42):
        self.n_folds = n_folds
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.cv_results = {}
        
    def prepare_survival_target(self, df: pd.DataFrame, outcome: str = 'hepatic') -> Tuple[pd.DataFrame, np.ndarray]:
        """Convert raw targets to sksurv structured array format."""
        df = df.copy()
        
        # Calculate age-based time variables
        age_cols = [c for c in df.columns if c.startswith('Age_v')]
        df['last_observed_age'] = df[age_cols].max(axis=1)
        df['first_visit_age'] = df[age_cols].min(axis=1)
        
        if outcome == 'hepatic':
            event_col = 'evenements_hepatiques_majeurs'
            age_occur_col = 'evenements_hepatiques_age_occur'
            name = 'HepaticEvent'
            
            # Filter: drop if event=1 but age_occur is NaN
            is_event = df[event_col] == 1
            invalid = is_event & df[age_occur_col].isna()
            df_valid = df[~invalid].copy()
            
        elif outcome == 'death':
            event_col = 'death'
            age_occur_col = 'death_age_occur'
            name = 'Death'
            
            # Filter: drop if death is NaN (unknown) or event=1 but no age
            is_event = df[event_col] == 1
            unknown = df[event_col].isna()
            invalid = is_event & df[age_occur_col].isna()
            df_valid = df[~(unknown | invalid)].copy()
        else:
            raise ValueError(f"Unknown outcome: {outcome}")
        
        # Calculate survival time
        is_event_v = (df_valid[event_col] == 1)
        time_values = np.where(
            is_event_v,
            df_valid[age_occur_col] - df_valid['first_visit_age'],
            df_valid['last_observed_age'] - df_valid['first_visit_age']
        ).astype(float)
        
        # Ensure positive times
        time_values = np.maximum(time_values, 0.001)
        
        # Create structured array
        y = Surv.from_arrays(
            event=is_event_v.values,
            time=time_values,
            name_event=name,
            name_time='Time'
        )
        
        logger.info(f"{outcome.upper()} target prepared:")
        logger.info(f"  Patients: {len(df_valid)}")
        logger.info(f"  Events: {is_event_v.sum()} ({100*is_event_v.mean():.1f}%)")
        logger.info(f"  Mean follow-up: {time_values.mean():.1f} years")
        
        return df_valid, y
    
    def create_preprocessing_pipeline(self) -> Pipeline:
        """Create preprocessing pipeline."""
        return Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ])
    
    def filter_features_by_missing_rate(self, X: pd.DataFrame, max_missing_rate: float = 0.60) -> List[str]:
        """Remove features with too many missing values."""
        missing_rates = X.isnull().mean()
        keep_cols = missing_rates[missing_rates <= max_missing_rate].index.tolist()
        logger.info(f"Feature filtering: {len(keep_cols)}/{len(X.columns)} retained (missing <= {max_missing_rate})")
        return keep_cols
    
    def cross_validate(self, X: pd.DataFrame, y: np.ndarray, 
                      outcome: str = 'hepatic',
                      death_oof_preds: Optional[np.ndarray] = None,
                      death_indices: Optional[pd.Index] = None) -> CVResult:
        """
        Repeated stratified cross-validation with rank ensemble.
        
        Args:
            X: Feature matrix
            y: Survival target
            outcome: 'hepatic' or 'death'
            death_oof_preds: OOF death predictions (for hepatic model)
            death_indices: Indices for death predictions alignment
        """
        logger.info(f"\nStarting {self.n_folds}-fold × {self.n_repeats}-repeat stratified CV for {outcome}...")
        
        # Add death predictions as feature if provided
        if death_oof_preds is not None and outcome == 'hepatic':
            X = X.copy()
            X['predicted_death_risk'] = np.nan
            # Align indices
            common_indices = X.index.intersection(death_indices)
            if len(common_indices) > 0:
                X.loc[common_indices, 'predicted_death_risk'] = death_oof_preds[
                    death_indices.get_indexer(common_indices)
                ]
                logger.info(f"  Added death predictions as feature ({len(common_indices)} aligned)")
        
        # Filter features
        keep_cols = self.filter_features_by_missing_rate(X, MAX_MISSING_RATE)
        X = X[keep_cols]
        
        # Stratify by event indicator
        event_indicator = y[y.dtype.names[0]]
        
        # Repeated Stratified K-Fold
        rskf = RepeatedStratifiedKFold(
            n_splits=self.n_folds, 
            n_repeats=self.n_repeats,
            random_state=self.random_state
        )
        
        oof_preds = np.zeros(len(X))
        fold_cindices = []
        repeat_cindices = [[] for _ in range(self.n_repeats)]
        
        n_total_folds = self.n_folds * self.n_repeats
        
        for fold_idx, (train_idx, val_idx) in enumerate(rskf.split(X, event_indicator)):
            repeat_idx = fold_idx // self.n_folds
            fold_num = fold_idx % self.n_folds + 1
            
            logger.info(f"\nRepeat {repeat_idx + 1}/{self.n_repeats}, Fold {fold_num}/{self.n_folds}")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Fit rank ensemble
            ensemble = RankEnsemble(weights={'rsf': 0.5, 'xgb': 0.3, 'coxnet': 0.2})
            
            try:
                ensemble.fit_models(X_train, y_train, feature_names=X.columns.tolist())
                ensemble_preds = ensemble.predict(X_val)
                
                # Store OOF predictions
                oof_preds[val_idx] = ensemble_preds
                
                # Calculate C-index
                ci = concordance_index_censored(
                    y_val[y_val.dtype.names[0]],
                    y_val[y_val.dtype.names[1]],
                    ensemble_preds
                )[0]
                
                fold_cindices.append(ci)
                repeat_cindices[repeat_idx].append(ci)
                
                logger.info(f"  Ensemble C-index: {ci:.4f}")
                
                # Individual model performance
                individual_preds = ensemble.get_individual_predictions(X_val)
                for name, preds in individual_preds.items():
                    ind_ci = concordance_index_censored(
                        y_val[y_val.dtype.names[0]],
                        y_val[y_val.dtype.names[1]],
                        preds
                    )[0]
                    logger.info(f"    {name.upper()}: {ind_ci:.4f}")
                
            except Exception as e:
                logger.warning(f"  Fold failed: {e}")
                import traceback
                logger.warning(traceback.format_exc())
        
        # Overall OOF C-index
        overall_ci = concordance_index_censored(
            event_indicator,
            y[y.dtype.names[1]],
            oof_preds
        )[0]
        
        # Calculate per-repeat means
        repeat_means = [np.mean(cis) if cis else np.nan for cis in repeat_cindices]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"{outcome.upper()} CV RESULTS:")
        logger.info(f"{'='*60}")
        logger.info(f"  Total folds: {len(fold_cindices)}")
        logger.info(f"  Fold C-indices: {np.mean(fold_cindices):.4f} (+/- {np.std(fold_cindices):.4f})")
        logger.info(f"  Overall OOF C-index: {overall_ci:.4f}")
        logger.info(f"  Per-repeat means: {[f'{m:.4f}' for m in repeat_means]}")
        logger.info(f"{'='*60}\n")
        
        return CVResult(
            oof_preds=oof_preds,
            overall_ci=overall_ci,
            fold_cindices=fold_cindices,
            repeat_cindices=repeat_means
        )
    
    def fit_final_models(self, X: pd.DataFrame, y: np.ndarray,
                        death_oof_preds: Optional[np.ndarray] = None,
                        death_indices: Optional[pd.Index] = None) -> RankEnsemble:
        """Fit final ensemble on full training data."""
        logger.info("Fitting final ensemble on full training data...")
        
        # Add death predictions as feature if provided
        if death_oof_preds is not None:
            X = X.copy()
            X['predicted_death_risk'] = np.nan
            common_indices = X.index.intersection(death_indices)
            if len(common_indices) > 0:
                X.loc[common_indices, 'predicted_death_risk'] = death_oof_preds[
                    death_indices.get_indexer(common_indices)
                ]
        
        # Filter features
        keep_cols = self.filter_features_by_missing_rate(X, MAX_MISSING_RATE)
        X = X[keep_cols]
        
        # Fit ensemble
        ensemble = RankEnsemble(weights={'rsf': 0.5, 'xgb': 0.3, 'coxnet': 0.2})
        ensemble.fit_models(X, y, feature_names=X.columns.tolist())
        
        logger.info("Final ensemble fitted successfully")
        
        return ensemble


def main():
    """Main execution pipeline."""
    logger.info("="*80)
    logger.info("ANNITIA MASLD SURVIVAL ANALYSIS - GEMINI RECOMMENDATIONS")
    logger.info("Repeated CV | XGBoost | CoxNet | Rank Ensemble | Death-as-Feature")
    logger.info("="*80)
    
    start_time = datetime.now()
    
    # Load data
    logger.info("\nLoading data...")
    train_df = pd.read_csv('data/train-df.csv')
    test_df = pd.read_csv('data/test-df.csv')
    
    logger.info(f"Train shape: {train_df.shape}")
    logger.info(f"Test shape: {test_df.shape}")
    
    # Feature Engineering
    logger.info("\n" + "="*80)
    logger.info("PHASE 1: ADVANCED FEATURE ENGINEERING (V2)")
    logger.info("="*80)
    
    engineer = TrajectoryFeatureEngineerV2()
    X_train = engineer.transform(train_df)
    X_test = engineer.transform(test_df)
    
    # Align columns
    common_cols = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]
    
    logger.info(f"Final feature matrix: {X_train.shape}")
    
    # Model Training
    logger.info("\n" + "="*80)
    logger.info("PHASE 2: SURVIVAL MODEL TRAINING")
    logger.info("="*80)
    
    ensemble = SurvivalModelEnsembleGemini(
        n_folds=N_FOLDS, 
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE
    )
    
    # === DEATH MODEL FIRST ===
    logger.info("\n--- STAGE 1: DEATH MODEL ---")
    df_death, y_death = ensemble.prepare_survival_target(train_df, outcome='death')
    X_death = X_train.loc[df_death.index]
    
    # Cross-validate death model
    cv_results_death = ensemble.cross_validate(X_death, y_death, outcome='death')
    
    # Fit final death model
    final_death_model = ensemble.fit_final_models(X_death, y_death)
    
    # === HEPATIC EVENTS MODEL (WITH DEATH PREDICTIONS) ===
    logger.info("\n--- STAGE 2: HEPATIC EVENTS MODEL (with Death-as-Feature) ---")
    df_hep, y_hep = ensemble.prepare_survival_target(train_df, outcome='hepatic')
    X_hep = X_train.loc[df_hep.index]
    
    # Cross-validate hepatic model with death OOF predictions
    cv_results_hep = ensemble.cross_validate(
        X_hep, y_hep, 
        outcome='hepatic',
        death_oof_preds=cv_results_death.oof_preds,
        death_indices=df_death.index
    )
    
    # Fit final hepatic model with death predictions
    # First get death predictions for all hepatic patients
    death_preds_for_hep = final_death_model.predict(X_hep)
    
    final_hepatic_model = ensemble.fit_final_models(
        X_hep, y_hep,
        death_oof_preds=death_preds_for_hep,
        death_indices=X_hep.index
    )
    
    # Generate Test Predictions
    logger.info("\n" + "="*80)
    logger.info("PHASE 3: GENERATING SUBMISSION")
    logger.info("="*80)
    
    # Death predictions for test set
    pred_death = final_death_model.predict(X_test)
    
    # Add death predictions to test features for hepatic model
    X_test_with_death = X_test.copy()
    X_test_with_death['predicted_death_risk'] = pred_death
    
    # Hepatic predictions
    pred_hepatic = final_hepatic_model.predict(X_test_with_death)
    
    submission = pd.DataFrame({
        'trustii_id': test_df['trustii_id'].values,
        'risk_hepatic_event': pred_hepatic,
        'risk_death': pred_death
    })
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = f'submissions/gemini_submission_{timestamp}.csv'
    submission.to_csv(submission_path, index=False)
    
    logger.info(f"\nSubmission saved to: {submission_path}")
    logger.info(f"Submission shape: {submission.shape}")
    logger.info(f"\nSubmission preview:")
    logger.info(submission.head(10).to_string())
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("="*80)
    logger.info(f"Hepatic Events OOF C-index: {cv_results_hep.overall_ci:.4f}")
    logger.info(f"  Fold std: {np.std(cv_results_hep.fold_cindices):.4f}")
    logger.info(f"Death OOF C-index: {cv_results_death.overall_ci:.4f}")
    logger.info(f"  Fold std: {np.std(cv_results_death.fold_cindices):.4f}")
    logger.info(f"Average C-index: {(cv_results_hep.overall_ci + cv_results_death.overall_ci)/2:.4f}")
    logger.info("="*80)
    
    # Save results
    results = {
        'timestamp': timestamp,
        'hepatic_ci': float(cv_results_hep.overall_ci),
        'hepatic_fold_std': float(np.std(cv_results_hep.fold_cindices)),
        'hepatic_fold_cis': [float(x) for x in cv_results_hep.fold_cindices],
        'hepatic_repeat_cis': [float(x) for x in cv_results_hep.repeat_cindices],
        'death_ci': float(cv_results_death.overall_ci),
        'death_fold_std': float(np.std(cv_results_death.fold_cindices)),
        'death_fold_cis': [float(x) for x in cv_results_death.fold_cindices],
        'death_repeat_cis': [float(x) for x in cv_results_death.repeat_cindices],
        'n_features': X_train.shape[1],
        'n_folds': N_FOLDS,
        'n_repeats': N_REPEATS,
        'elapsed_time_minutes': (datetime.now() - start_time).total_seconds() / 60
    }
    
    results_path = f'submissions/gemini_results_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to: {results_path}")
    logger.info(f"Elapsed time: {results['elapsed_time_minutes']:.1f} minutes")
    
    return results


if __name__ == "__main__":
    main()
