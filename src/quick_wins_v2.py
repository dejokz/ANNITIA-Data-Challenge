"""
Quick Wins v2.0 - Targeting 0.85+ C-index

Changes from v1.0:
1. Stronger regularization (reduce overfitting)
2. Delta features (visit-to-visit changes)
3. Better feature selection
4. Simple ensemble of 3 models
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
from sksurv.linear_model import CoxnetSurvivalAnalysis
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
N_FOLDS = 5


class ImprovedTrajectoryEngineer:
    """Feature engineering with delta features and better clinical logic."""
    
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
        """Extract features with delta (visit-to-visit changes)."""
        features = pd.DataFrame(index=df.index)
        
        logger.info("Extracting trajectory features with deltas...")
        
        for var in self.visit_vars:
            visit_cols = [c for c in df.columns if c.startswith(f'{var}_v') and c.split('_v')[-1].isdigit()]
            
            if not visit_cols:
                continue
            
            # Sort
            visit_nums = [int(c.split('_v')[-1]) for c in visit_cols]
            sorted_pairs = sorted(zip(visit_nums, visit_cols))
            sorted_cols = [col for _, col in sorted_pairs]
            
            values = df[sorted_cols]
            
            # === BASIC STATS ===
            features[f'{var}_max'] = values.max(axis=1)
            features[f'{var}_min'] = values.min(axis=1)
            features[f'{var}_mean'] = values.mean(axis=1)
            features[f'{var}_median'] = values.median(axis=1)
            features[f'{var}_std'] = values.std(axis=1)
            features[f'{var}_first'] = values.iloc[:, 0]
            features[f'{var}_last'] = values.ffill(axis=1).iloc[:, -1]
            features[f'{var}_range'] = features[f'{var}_max'] - features[f'{var}_min']
            
            # === DELTA FEATURES (Visit-to-visit changes) ===
            for i in range(1, len(sorted_cols)):
                curr_col = sorted_cols[i]
                prev_col = sorted_cols[i-1]
                delta_col = f'{var}_delta_v{i+1}'
                features[delta_col] = df[curr_col] - df[prev_col]
            
            # Delta statistics
            delta_cols = [c for c in features.columns if c.startswith(f'{var}_delta_')]
            if delta_cols:
                features[f'{var}_delta_mean'] = features[delta_cols].mean(axis=1)
                features[f'{var}_delta_std'] = features[delta_cols].std(axis=1)
                features[f'{var}_delta_max'] = features[delta_cols].max(axis=1)
                features[f'{var}_delta_min'] = features[delta_cols].min(axis=1)
                # Is it accelerating?
                features[f'{var}_accel'] = features[delta_cols].apply(
                    lambda x: stats.linregress(range(len(x)), x.fillna(0))[0] if x.notna().sum() > 1 else 0, 
                    axis=1
                )
            
            # === SLOPE & RATE ===
            def calc_slope(row):
                valid = row.dropna()
                if len(valid) < 2:
                    return 0.0
                x = np.arange(len(valid))
                slope, _, _, _, _ = stats.linregress(x, valid.values)
                return slope
            
            features[f'{var}_slope'] = values.apply(calc_slope, axis=1)
            
            # Rate of change (per year)
            age_cols = [c for c in df.columns if c.startswith('Age_v')]
            time_span = df[age_cols].max(axis=1) - df[age_cols].min(axis=1)
            features[f'{var}_roc'] = (features[f'{var}_last'] - features[f'{var}_first']) / (time_span + 0.001)
            
            # Recent trend (last 3 vs first 3)
            if len(sorted_cols) >= 6:
                first_3 = values.iloc[:, :3].mean(axis=1)
                last_3 = values.iloc[:, -3:].mean(axis=1)
                features[f'{var}_recent_change'] = last_3 - first_3
                features[f'{var}_recent_change_rate'] = features[f'{var}_recent_change'] / (df[age_cols].max(axis=1) - df[age_cols].min(axis=1) + 0.001)
            
            # Volatility in recent period
            if len(sorted_cols) >= 5:
                features[f'{var}_recent_volatility'] = values.iloc[:, -5:].std(axis=1)
            
            # === CLINICAL THRESHOLDS ===
            if var == 'fib4':
                high_visits = (values > 2.67).sum(axis=1)
                int_visits = ((values > 1.30) & (values <= 2.67)).sum(axis=1)
                total_visits = values.notna().sum(axis=1)
                
                features[f'{var}_time_high'] = high_visits / (total_visits + 0.001)
                features[f'{var}_time_intermediate'] = int_visits / (total_visits + 0.001)
                features[f'{var}_ever_high'] = (values > 2.67).any(axis=1).astype(int)
                features[f'{var}_ever_intermediate'] = (values > 1.30).any(axis=1).astype(int)
                features[f'{var}_worsening'] = (features[f'{var}_slope'] > 0.1).astype(int)
                features[f'{var}_rapid_rise'] = (features[f'{var}_roc'] > 0.5).astype(int)  # >0.5/year
                
            elif var == 'fibs_stiffness_med_BM_1':
                high_visits = (values > 8.0).sum(axis=1)
                very_high_visits = (values > 12.0).sum(axis=1)
                total_visits = values.notna().sum(axis=1)
                
                features[f'{var}_time_high'] = high_visits / (total_visits + 0.001)
                features[f'{var}_time_very_high'] = very_high_visits / (total_visits + 0.001)
                features[f'{var}_ever_high'] = (values > 8.0).any(axis=1).astype(int)
                features[f'{var}_ever_very_high'] = (values > 12.0).any(axis=1).astype(int)
                features[f'{var}_worsening'] = (features[f'{var}_slope'] > 0.5).astype(int)
                features[f'{var}_rapid_rise'] = (features[f'{var}_roc'] > 2.0).astype(int)  # >2 kPa/year
                
            elif var == 'fibrotest_BM_2':
                high_visits = (values > 0.72).sum(axis=1)
                total_visits = values.notna().sum(axis=1)
                
                features[f'{var}_time_high'] = high_visits / (total_visits + 0.001)
                features[f'{var}_ever_high'] = (values > 0.72).any(axis=1).astype(int)
                features[f'{var}_worsening'] = (features[f'{var}_slope'] > 0.05).astype(int)
                
            elif var == 'plt':
                low_visits = (values < 150).sum(axis=1)
                total_visits = values.notna().sum(axis=1)
                
                features[f'{var}_time_low'] = low_visits / (total_visits + 0.001)
                features[f'{var}_ever_low'] = (values < 150).any(axis=1).astype(int)
                features[f'{var}_declining'] = (features[f'{var}_slope'] < -5).astype(int)
                features[f'{var}_rapid_decline'] = (features[f'{var}_roc'] < -20).astype(int)  # >20k drop/year
        
        return features
    
    def extract_cross_nit_features(self, df, features):
        """Cross-NIT concordance."""
        nit_pairs = [
            ('fibs_stiffness_med_BM_1', 'fib4'),
            ('fibs_stiffness_med_BM_1', 'fibrotest_BM_2'),
            ('fib4', 'fibrotest_BM_2'),
        ]
        
        for nit1, nit2 in nit_pairs:
            # Slope agreement
            slope1 = f'{nit1}_slope'
            slope2 = f'{nit2}_slope'
            
            if slope1 in features.columns and slope2 in features.columns:
                features[f'{nit1}_{nit2}_slope_agree'] = (
                    (features[slope1] * features[slope2]) > 0
                ).astype(int)
                features[f'{nit1}_{nit2}_slope_diff'] = (
                    features[slope1] - features[slope2]
                ).abs()
        
        # Composite indicators
        worsening_cols = [c for c in features.columns if c.endswith('_worsening')]
        if worsening_cols:
            features['any_fibrosis_worsening'] = features[worsening_cols].any(axis=1).astype(int)
            features['n_fibrosis_worsening'] = features[worsening_cols].sum(axis=1)
            features['all_fibrosis_worsening'] = features[worsening_cols].all(axis=1).astype(int)
        
        high_cols = [c for c in features.columns if c.endswith('_ever_high')]
        if high_cols:
            features['any_nit_high'] = features[high_cols].any(axis=1).astype(int)
            features['n_nits_high'] = features[high_cols].sum(axis=1)
            features['all_nits_high'] = features[high_cols].all(axis=1).astype(int)
        
        rapid_cols = [c for c in features.columns if c.startswith('fib') and 'rapid' in c]
        if rapid_cols:
            features['any_rapid_progression'] = features[rapid_cols].any(axis=1).astype(int)
        
        return features
    
    def add_static_features(self, df, features):
        """Add static and time-aware features."""
        static_cols = ['gender', 'T2DM', 'Hypertension', 'Dyslipidaemia', 'bariatric_surgery']
        
        for col in static_cols:
            if col in df.columns:
                features[col] = df[col]
        
        # Interactions
        if 'T2DM' in features.columns:
            if 'fib4_max' in features.columns:
                features['T2DM_x_fib4_max'] = features['T2DM'] * features['fib4_max']
            if 'fibs_stiffness_med_BM_1_max' in features.columns:
                features['T2DM_x_lsm_max'] = features['T2DM'] * features['fibs_stiffness_med_BM_1_max']
            if 'plt_min' in features.columns:
                features['T2DM_x_low_plt'] = features['T2DM'] * (features['plt_min'] < 150).astype(int)
        
        # Time features
        age_cols = [c for c in df.columns if c.startswith('Age_v')]
        if age_cols:
            features['age_baseline'] = df[age_cols].min(axis=1)
            features['age_last'] = df[age_cols].max(axis=1)
            features['follow_up_years'] = features['age_last'] - features['age_baseline']
            features['n_visits'] = df[age_cols].notna().sum(axis=1)
            features['visit_frequency'] = features['n_visits'] / (features['follow_up_years'] + 0.001)
        
        return features
    
    def transform(self, df):
        """Full pipeline."""
        df = self.calculate_clinical_scores(df)
        features = self.extract_trajectory_features(df)
        features = self.extract_cross_nit_features(df, features)
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
        name = 'HepaticEvent'
        is_event = df[event_col] == 1
        invalid = is_event & df[age_occur_col].isna()
        df_valid = df[~invalid].copy()
    else:
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


def rank_features_by_cindex(X, y, feature_list):
    """Rank features by univariate C-index."""
    from scipy.stats import pearsonr
    
    scores = []
    event_indicator = y[y.dtype.names[0]]
    time_to_event = y[y.dtype.names[1]]
    
    for feature in feature_list:
        if feature not in X.columns:
            continue
        
        values = X[feature].fillna(X[feature].median())
        corr, _ = pearsonr(values, event_indicator)
        
        # Flip if negative correlation
        if corr < 0:
            values = -values
        
        try:
            ci = concordance_index_censored(
                event_indicator.astype(bool),
                time_to_event,
                values.values
            )[0]
            scores.append({'feature': feature, 'c_index': ci, 'corr': corr})
        except:
            pass
    
    return pd.DataFrame(scores).sort_values('c_index', ascending=False)


class SurvivalEnsembleV2:
    """Ensemble with stronger regularization."""
    
    def __init__(self, n_folds=5, random_state=42):
        self.n_folds = n_folds
        self.random_state = random_state
    
    def cross_validate(self, X, y, feature_cols, model_configs):
        """CV with multiple models."""
        X = X[feature_cols]
        event_indicator = y[y.dtype.names[0]]
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        oof_preds = {}
        fold_cis = {}
        
        for model_name, ModelClass, model_params in model_configs:
            logger.info(f"\\nTraining {model_name}...")
            oof_preds[model_name] = np.zeros(len(X))
            fold_cis[model_name] = []
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(X, event_indicator)):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                
                # Preprocess
                imputer = SimpleImputer(strategy='median')
                scaler = StandardScaler()
                X_train_proc = scaler.fit_transform(imputer.fit_transform(X_train_fold))
                X_val_proc = scaler.transform(imputer.transform(X_val_fold))
                
                # Train
                model = ModelClass(**model_params, random_state=self.random_state)
                if hasattr(model, 'n_jobs'):
                    model.set_params(n_jobs=-1)
                
                model.fit(X_train_proc, y_train_fold)
                preds = model.predict(X_val_proc)
                oof_preds[model_name][val_idx] = preds
                
                ci = concordance_index_censored(
                    y_val_fold[y_val_fold.dtype.names[0]],
                    y_val_fold[y_val_fold.dtype.names[1]],
                    preds
                )[0]
                fold_cis[model_name].append(ci)
            
            logger.info(f"  {model_name}: {np.mean(fold_cis[model_name]):.4f} (+/- {np.std(fold_cis[model_name]):.4f})")
        
        # Ensemble (average)
        ensemble_preds = np.mean(list(oof_preds.values()), axis=0)
        
        overall_ci = concordance_index_censored(
            event_indicator.astype(bool),
            y[y.dtype.names[1]],
            ensemble_preds
        )[0]
        
        logger.info(f"\\nEnsemble OOF C-index: {overall_ci:.4f}")
        
        return overall_ci, oof_preds, fold_cis
    
    def fit_final(self, X, y, feature_cols, model_configs):
        """Fit final models."""
        X = X[feature_cols]
        
        final_models = []
        for model_name, ModelClass, model_params in model_configs:
            imputer = SimpleImputer(strategy='median')
            scaler = StandardScaler()
            X_proc = scaler.fit_transform(imputer.fit_transform(X))
            
            model = ModelClass(**model_params, random_state=self.random_state)
            if hasattr(model, 'n_jobs'):
                model.set_params(n_jobs=-1)
            
            model.fit(X_proc, y)
            final_models.append((model_name, model, imputer, scaler))
        
        return final_models
    
    def predict(self, X, feature_cols, models):
        """Generate predictions."""
        X = X[feature_cols]
        preds = []
        
        for model_name, model, imputer, scaler in models:
            X_proc = scaler.transform(imputer.transform(X))
            pred = model.predict(X_proc)
            preds.append(pred)
        
        return np.mean(preds, axis=0)


def main():
    """Main pipeline."""
    logger.info("="*70)
    logger.info("QUICK WINS v2.0 - TARGETING 0.85+ C-INDEX")
    logger.info("="*70)
    
    # Load data
    train_df = pd.read_csv('data/train-df.csv')
    test_df = pd.read_csv('data/test-df.csv')
    
    logger.info(f"\\nData loaded: train={train_df.shape}, test={test_df.shape}")
    
    # Feature engineering
    logger.info("\\n" + "="*70)
    logger.info("FEATURE ENGINEERING")
    logger.info("="*70)
    
    engineer = ImprovedTrajectoryEngineer()
    X_train = engineer.transform(train_df)
    X_test = engineer.transform(test_df)
    
    common_cols = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]
    
    logger.info(f"\\nTotal features: {len(common_cols)}")
    
    # Prepare targets
    logger.info("\\n" + "="*70)
    logger.info("MODEL TRAINING")
    logger.info("="*70)
    
    df_hep, y_hep = prepare_survival_target(train_df, 'hepatic')
    df_death, y_death = prepare_survival_target(train_df, 'death')
    
    X_hep = X_train.loc[df_hep.index]
    X_death = X_train.loc[df_death.index]
    
    logger.info(f"\\nHepatic: {len(df_hep)} patients, {y_hep['HepaticEvent'].sum()} events")
    logger.info(f"Death: {len(df_death)} patients, {y_death['Death'].sum()} events")
    
    # Feature selection (keep top 40 by univariate C-index)
    logger.info("\\n" + "-"*70)
    logger.info("Feature Selection (Top 40 by C-index)")
    logger.info("-"*70)
    
    scores_hep = rank_features_by_cindex(X_hep, y_hep, common_cols)
    top_features_hep = scores_hep.head(40)['feature'].tolist()
    
    logger.info(f"\\nTop 10 hepatic features:")
    logger.info(scores_hep.head(10).to_string(index=False))
    
    scores_death = rank_features_by_cindex(X_death, y_death, common_cols)
    top_features_death = scores_death.head(40)['feature'].tolist()
    
    # Model configs (stronger regularization)
    model_configs = [
        ('RSF_v1', RandomSurvivalForest, {
            'n_estimators': 500,
            'min_samples_leaf': 40,  # Much stronger
            'min_samples_split': 80,
            'max_features': 'sqrt',
            'max_depth': 6,  # Limit depth
        }),
        ('RSF_v2', RandomSurvivalForest, {
            'n_estimators': 500,
            'min_samples_leaf': 30,
            'min_samples_split': 60,
            'max_features': 'log2',  # Different feature selection
            'max_depth': 8,
        }),
        ('GBS', GradientBoostingSurvivalAnalysis, {
            'n_estimators': 300,
            'learning_rate': 0.05,
            'max_depth': 3,
            'subsample': 0.8,
        }),
    ]
    
    # Train hepatic model
    logger.info("\\n" + "="*70)
    logger.info("HEPATIC EVENTS MODEL")
    logger.info("="*70)
    
    ensemble = SurvivalEnsembleV2(n_folds=5, random_state=42)
    ci_hep, oof_hep, fold_cis_hep = ensemble.cross_validate(X_hep, y_hep, top_features_hep, model_configs)
    final_models_hep = ensemble.fit_final(X_hep, y_hep, top_features_hep, model_configs)
    
    # Train death model
    logger.info("\\n" + "="*70)
    logger.info("DEATH MODEL")
    logger.info("="*70)
    
    ci_death, oof_death, fold_cis_death = ensemble.cross_validate(X_death, y_death, top_features_death, model_configs)
    final_models_death = ensemble.fit_final(X_death, y_death, top_features_death, model_configs)
    
    # Generate submission
    logger.info("\\n" + "="*70)
    logger.info("GENERATING SUBMISSION")
    logger.info("="*70)
    
    pred_hep = ensemble.predict(X_test, top_features_hep, final_models_hep)
    pred_death = ensemble.predict(X_test, top_features_death, final_models_death)
    
    submission = pd.DataFrame({
        'trustii_id': test_df['trustii_id'].values,
        'risk_hepatic_event': pred_hep,
        'risk_death': pred_death
    })
    
    submission.to_csv('submissions/quick_wins_v2_submission.csv', index=False)
    
    logger.info(f"\\n✅ Submission saved!")
    logger.info(f"📊 Shape: {submission.shape}")
    logger.info("\\nFirst 10 rows:")
    logger.info(submission.head(10).to_string(index=False))
    
    # Summary
    logger.info("\\n" + "="*70)
    logger.info("FINAL RESULTS")
    logger.info("="*70)
    logger.info(f"\\n🎯 HEPATIC C-index: {ci_hep:.4f}")
    logger.info(f"🎯 DEATH C-index: {ci_death:.4f}")
    logger.info(f"🎯 AVERAGE: {(ci_hep + ci_death)/2:.4f}")
    logger.info("="*70)
    
    results = {
        'hepatic_ci': float(ci_hep),
        'death_ci': float(ci_death),
        'average_ci': float((ci_hep + ci_death)/2),
        'n_features': len(common_cols),
        'n_selected_hepatic': len(top_features_hep),
        'n_selected_death': len(top_features_death),
    }
    
    with open('submissions/quick_wins_v2_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("\\n✅ Results saved to quick_wins_v2_results.json")


if __name__ == '__main__':
    main()
