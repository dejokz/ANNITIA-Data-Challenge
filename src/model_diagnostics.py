#!/usr/bin/env python3
"""
Model Diagnostics & Interpretability Toolkit

Analyzes the winning 0.83 model to understand:
1. Feature importance (what drives predictions)
2. Error analysis (where model fails)
3. Risk stratification (how well it separates patients)
4. Clinical plausibility (does it make medical sense)
5. Improvement opportunities (what's missing)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
import logging
from typing import Dict, List, Tuple, Optional
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

RANDOM_STATE = 42


class ModelDiagnostics:
    """Comprehensive diagnostic toolkit for survival models."""
    
    def __init__(self, model, X_train, y_train, X_test=None, feature_names=None):
        """
        Initialize diagnostics.
        
        Args:
            model: Trained survival model
            X_train: Training features
            y_train: Training survival target
            X_test: Test features (optional)
            feature_names: List of feature names
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.feature_names = feature_names or X_train.columns.tolist()
        
        self.event_indicator = y_train[y_train.dtype.names[0]]
        self.survival_time = y_train[y_train.dtype.names[1]]
        
        # Get predictions
        self.train_preds = model.predict(X_train)
        
        logger.info("="*70)
        logger.info("MODEL DIAGNOSTICS INITALIZED")
        logger.info("="*70)
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Events: {self.event_indicator.sum()} ({100*self.event_indicator.mean():.1f}%)")
        logger.info(f"Features: {len(self.feature_names)}")
        
    def analyze_feature_importance(self, n_top=20) -> pd.DataFrame:
        """
        Analyze feature importance using multiple methods.
        """
        logger.info("\n" + "="*70)
        logger.info("1. FEATURE IMPORTANCE ANALYSIS")
        logger.info("="*70)
        
        # Method 1: Univariate correlation with predictions (always works)
        logger.info("Calculating feature-prediction correlations...")
        correlations = []
        for col in self.X_train.columns:
            try:
                corr, _ = pearsonr(self.X_train[col].fillna(self.X_train[col].median()), self.train_preds)
                correlations.append(abs(corr))
            except:
                correlations.append(0.0)
        
        # Method 2: Correlation with target (events)
        target_correlations = []
        for col in self.X_train.columns:
            try:
                corr, _ = pearsonr(self.X_train[col].fillna(self.X_train[col].median()), self.event_indicator)
                target_correlations.append(abs(corr))
            except:
                target_correlations.append(0.0)
        
        # Combine into DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'pred_correlation': correlations,
            'target_correlation': target_correlations,
            'combined_score': np.array(correlations) + np.array(target_correlations)
        })
        
        # Rank by combined score
        importance_df = importance_df.sort_values('combined_score', ascending=False)
        
        logger.info(f"\nTop {n_top} Most Important Features:")
        logger.info(importance_df.head(n_top).to_string(index=False))
        
        # Categorize features
        logger.info("\n" + "-"*70)
        logger.info("FEATURE CATEGORY BREAKDOWN:")
        logger.info("-"*70)
        
        categories = {
            'LSM (Stiffness)': [f for f in importance_df['feature'] if 'fibs_stiffness' in f][:5],
            'FIB-4': [f for f in importance_df['feature'] if 'fib4_' in f and 'delta' not in f][:5],
            'FibroTest': [f for f in importance_df['feature'] if 'fibrotest' in f][:5],
            'Platelets': [f for f in importance_df['feature'] if 'plt_' in f][:5],
            'Liver Enzymes': [f for f in importance_df['feature'] if any(x in f for x in ['alt_', 'ast_'])][:5],
            'Cross-NIT': [f for f in importance_df['feature'] if any(x in f for x in ['agree', 'worsening', 'n_'])][:5],
        }
        
        for cat, feats in categories.items():
            if feats:
                avg_imp = importance_df[importance_df['feature'].isin(feats)]['combined_score'].mean()
                logger.info(f"  {cat:20s}: Avg Importance = {avg_imp:.4f}")
        
        return importance_df
    
    def _c_index_scorer(self, estimator, X, y):
        """Custom scorer for permutation importance."""
        preds = estimator.predict(X)
        try:
            ci = concordance_index_censored(
                y[y.dtype.names[0]], y[y.dtype.names[1]], preds
            )[0]
        except:
            ci = 0.5
        return ci
    
    def error_analysis(self) -> Dict:
        """
        Analyze where the model makes errors.
        """
        logger.info("\n" + "="*70)
        logger.info("2. ERROR ANALYSIS")
        logger.info("="*70)
        
        # Calculate residuals (prediction errors)
        # For survival, we look at concordance pairs
        
        # Identify misclassified pairs
        n_samples = len(self.X_train)
        concordant = 0
        discordant = 0
        tied = 0
        
        error_cases = []
        
        # Sample pairs for analysis (full O(n²) is too slow)
        np.random.seed(RANDOM_STATE)
        n_pairs = min(10000, n_samples * (n_samples - 1) // 2)
        
        event_mask = self.event_indicator == 1
        event_indices = np.where(event_mask)[0]
        
        logger.info(f"Analyzing prediction errors on {len(event_indices)} event patients...")
        
        for idx in event_indices:
            # Compare event patient with all others who survived longer
            event_time = self.survival_time[idx]
            event_risk = self.train_preds[idx]
            
            longer_survival = np.where(self.survival_time > event_time)[0]
            
            for other_idx in longer_survival[:100]:  # Sample to speed up
                other_risk = self.train_preds[other_idx]
                
                if event_risk > other_risk:
                    concordant += 1
                elif event_risk < other_risk:
                    discordant += 1
                    # Record error case
                    error_cases.append({
                        'event_patient': idx,
                        'event_risk': event_risk,
                        'other_patient': other_idx,
                        'other_risk': other_risk,
                        'other_survival_time': self.survival_time[other_idx]
                    })
                else:
                    tied += 1
        
        total_pairs = concordant + discordant + tied
        
        logger.info(f"\nPairwise Concordance Analysis:")
        logger.info(f"  Concordant pairs:  {concordant} ({100*concordant/total_pairs:.1f}%)")
        logger.info(f"  Discordant pairs:  {discordant} ({100*discordant/total_pairs:.1f}%)")
        logger.info(f"  Tied pairs:        {tied} ({100*tied/total_pairs:.1f}%)")
        
        # Analyze characteristics of misclassified patients
        if error_cases:
            error_df = pd.DataFrame(error_cases)
            
            logger.info(f"\nAnalyzing {len(error_df)} discordant pairs...")
            logger.info(f"  Event patients had LOWER risk scores than they should")
            logger.info(f"  Average risk difference: {(error_df['other_risk'] - error_df['event_risk']).mean():.3f}")
        
        return {
            'concordant': concordant,
            'discordant': discordant,
            'tied': tied,
            'error_cases': error_cases
        }
    
    def risk_stratification_analysis(self) -> Dict:
        """
        Analyze how well the model stratifies patients by risk.
        """
        logger.info("\n" + "="*70)
        logger.info("3. RISK STRATIFICATION ANALYSIS")
        logger.info("="*70)
        
        # Divide into risk quartiles
        risk_quartiles = pd.qcut(self.train_preds, q=4, labels=['Low', 'Med-Low', 'Med-High', 'High'])
        
        analysis = []
        for quartile in ['Low', 'Med-Low', 'Med-High', 'High']:
            mask = risk_quartiles == quartile
            n_patients = mask.sum()
            n_events = self.event_indicator[mask].sum()
            event_rate = n_events / n_patients if n_patients > 0 else 0
            mean_risk = self.train_preds[mask].mean()
            
            analysis.append({
                'quartile': quartile,
                'n_patients': n_patients,
                'n_events': n_events,
                'event_rate': event_rate,
                'mean_risk': mean_risk
            })
            
            logger.info(f"  {quartile:10s}: {n_patients:4d} patients, {n_events:3d} events ({100*event_rate:.1f}%), Mean Risk: {mean_risk:.3f}")
        
        # Calculate separation metrics
        low_risk = self.train_preds[risk_quartiles == 'Low']
        high_risk = self.train_preds[risk_quartiles == 'High']
        
        separation = high_risk.mean() - low_risk.mean()
        logger.info(f"\n  Risk Separation (High - Low): {separation:.3f}")
        
        # T-test for separation
        t_stat, p_val = stats.ttest_ind(high_risk, low_risk)
        logger.info(f"  T-test p-value: {p_val:.2e} ({'Significant' if p_val < 0.001 else 'Not significant'})")
        
        return {
            'quartile_analysis': analysis,
            'separation': separation,
            'p_value': p_val
        }
    
    def clinical_plausibility_check(self) -> Dict:
        """
        Verify predictions align with clinical knowledge.
        """
        logger.info("\n" + "="*70)
        logger.info("4. CLINICAL PLAUSIBILITY CHECK")
        logger.info("="*70)
        
        checks = []
        
        # Check 1: High FIB-4 should correlate with high risk
        if 'fib4_max' in self.X_train.columns:
            corr = pearsonr(self.X_train['fib4_max'].fillna(0), self.train_preds)[0]
            passed = corr > 0.1
            checks.append(('FIB-4 max vs Risk', corr, passed, 'Higher FIB-4 should = Higher risk'))
        
        # Check 2: Low platelets should correlate with high risk
        if 'plt_min' in self.X_train.columns:
            corr = pearsonr(self.X_train['plt_min'].fillna(200), self.train_preds)[0]
            passed = corr < -0.1
            checks.append(('Platelet min vs Risk', corr, passed, 'Lower platelets should = Higher risk'))
        
        # Check 3: High LSM should correlate with high risk
        if 'fibs_stiffness_med_BM_1_max' in self.X_train.columns:
            corr = pearsonr(self.X_train['fibs_stiffness_med_BM_1_max'].fillna(0), self.train_preds)[0]
            passed = corr > 0.1
            checks.append(('LSM max vs Risk', corr, passed, 'Higher LSM should = Higher risk'))
        
        # Check 4: Diabetes should increase risk
        if 'T2DM' in self.X_train.columns:
            diabetes_risk = self.train_preds[self.X_train['T2DM'] == 1].mean()
            no_diabetes_risk = self.train_preds[self.X_train['T2DM'] == 0].mean()
            diff = diabetes_risk - no_diabetes_risk
            passed = diff > 0
            checks.append(('Diabetes vs Risk', diff, passed, 'Diabetes should increase risk'))
        
        # Check 5: Age should increase risk
        if 'age_baseline' in self.X_train.columns:
            corr = pearsonr(self.X_train['age_baseline'].fillna(50), self.train_preds)[0]
            passed = corr > 0
            checks.append(('Age vs Risk', corr, passed, 'Older age should = Higher risk'))
        
        # Print results
        passed_count = sum(1 for _, _, passed, _ in checks)
        logger.info(f"\nPassed {passed_count}/{len(checks)} clinical plausibility checks:\n")
        
        for name, value, passed, rationale in checks:
            status = "✅ PASS" if passed else "❌ FAIL"
            logger.info(f"  {status} {name:30s} (value={value:6.3f}) - {rationale}")
        
        return {
            'checks': checks,
            'passed': passed_count,
            'total': len(checks)
        }
    
    def identify_improvement_opportunities(self, importance_df: pd.DataFrame) -> List[str]:
        """
        Identify specific areas for improvement.
        """
        logger.info("\n" + "="*70)
        logger.info("5. IMPROVEMENT OPPORTUNITIES")
        logger.info("="*70)
        
        opportunities = []
        
        # Opportunity 1: Missing important features
        top_features = importance_df.head(10)['feature'].tolist()
        
        if not any('delta' in f for f in top_features):
            opportunities.append("Consider delta features (visit-to-visit changes) - not in top 10")
        
        if not any('slope' in f for f in top_features):
            opportunities.append("Trajectory slopes may be underutilized")
        
        # Opportunity 2: Feature interactions
        if 'T2DM_x_fib4_max' not in top_features:
            opportunities.append("T2DM × FIB-4 interaction not highly ranked - may need better interaction features")
        
        # Opportunity 3: Cross-NIT features
        cross_nit_in_top = sum(1 for f in top_features if any(x in f for x in ['agree', 'worsening', 'n_fibrosis']))
        if cross_nit_in_top < 2:
            opportunities.append("Cross-NIT concordance features could be more predictive")
        
        # Opportunity 4: Time-aware features
        if 'follow_up_years' not in top_features and 'visit_frequency' not in top_features:
            opportunities.append("Visit pattern features (frequency, regularity) not highly ranked")
        
        # Opportunity 5: Rare event handling
        event_rate = self.event_indicator.mean()
        if event_rate < 0.05:
            opportunities.append(f"Very rare events ({100*event_rate:.1f}%) - consider SMOTE or class weighting")
        
        logger.info("\nIdentified Opportunities:\n")
        for i, opp in enumerate(opportunities, 1):
            logger.info(f"  {i}. {opp}")
        
        if not opportunities:
            logger.info("  No obvious improvement opportunities detected - model is well-optimized!")
        
        return opportunities
    
    def generate_report(self, output_file: str = 'submissions/model_diagnostics_report.json'):
        """
        Generate comprehensive diagnostic report.
        """
        logger.info("\n" + "="*70)
        logger.info("GENERATING COMPREHENSIVE DIAGNOSTIC REPORT")
        logger.info("="*70)
        
        # Run all analyses
        importance_df = self.analyze_feature_importance(n_top=20)
        error_analysis = self.error_analysis()
        risk_analysis = self.risk_stratification_analysis()
        clinical_checks = self.clinical_plausibility_check()
        opportunities = self.identify_improvement_opportunities(importance_df)
        
        # Compile report
        report = {
            'model_performance': {
                'training_c_index': float(concordance_index_censored(
                    self.event_indicator, self.survival_time, self.train_preds
                )[0]),
                'n_samples': len(self.X_train),
                'n_events': int(self.event_indicator.sum()),
                'event_rate': float(self.event_indicator.mean())
            },
            'top_features': importance_df.head(10).to_dict('records'),
            'risk_stratification': risk_analysis,
            'clinical_plausibility': {
                'passed': clinical_checks['passed'],
                'total': clinical_checks['total'],
                'checks': clinical_checks['checks']
            },
            'improvement_opportunities': opportunities,
            'error_analysis': {
                'concordant_pairs': error_analysis['concordant'],
                'discordant_pairs': error_analysis['discordant'],
                'tied_pairs': error_analysis['tied']
            }
        }
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\n✅ Report saved to: {output_file}")
        
        return report


def main():
    """Run diagnostics on the best model."""
    logger.info("="*70)
    logger.info("MODEL DIAGNOSTICS - UNDERSTANDING THE 0.83 MODEL")
    logger.info("="*70)
    
    # Load data
    logger.info("\nLoading data...")
    train_df = pd.read_csv('data/train-df.csv')
    
    # Recreate the winning model (from pipeline.py)
    from pipeline import TrajectoryFeatureEngineer
    
    engineer = TrajectoryFeatureEngineer()
    X_train = engineer.transform(train_df)
    
    # Prepare target
    age_cols = [c for c in train_df.columns if c.startswith('Age_v')]
    train_df['last_observed_age'] = train_df[age_cols].max(axis=1)
    train_df['first_visit_age'] = train_df[age_cols].min(axis=1)
    
    event_col = 'evenements_hepatiques_majeurs'
    age_occur_col = 'evenements_hepatiques_age_occur'
    
    is_event = train_df[event_col] == 1
    invalid = is_event & train_df[age_occur_col].isna()
    train_df_filtered = train_df[~invalid].copy()
    
    X_train = X_train.loc[train_df_filtered.index]
    
    is_event_v = (train_df_filtered[event_col] == 1)
    time_values = np.where(
        is_event_v,
        train_df_filtered[age_occur_col] - train_df_filtered['first_visit_age'],
        train_df_filtered['last_observed_age'] - train_df_filtered['first_visit_age']
    ).astype(float)
    time_values = np.maximum(time_values, 0.001)
    
    y_train = Surv.from_arrays(
        event=is_event_v.values,
        time=time_values,
        name_event='HepaticEvent',
        name_time='Time'
    )
    
    # Preprocess
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    X_train_processed = scaler.fit_transform(imputer.fit_transform(X_train))
    
    # Train model
    logger.info("\nTraining RSF model...")
    model = RandomSurvivalForest(
        n_estimators=300,
        min_samples_leaf=20,
        min_samples_split=40,
        max_features='sqrt',
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    model.fit(X_train_processed, y_train)
    
    # Create diagnostics
    X_train_df = pd.DataFrame(X_train_processed, columns=X_train.columns, index=X_train.index)
    
    diagnostics = ModelDiagnostics(model, X_train_df, y_train, feature_names=X_train.columns.tolist())
    
    # Generate report
    report = diagnostics.generate_report()
    
    logger.info("\n" + "="*70)
    logger.info("DIAGNOSTICS COMPLETE")
    logger.info("="*70)
    
    return report


if __name__ == '__main__':
    report = main()
