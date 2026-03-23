"""Gemini Pipeline v3 - Train Death and Hepatic independently (no death-as-feature)"""

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, '.')
from pipeline_gemini_v2 import (
    TrajectoryFeatureEngineerV2, SimpleEnsemble,
    prepare_survival_target, cross_validate, 
    RANDOM_STATE, N_FOLDS, N_REPEATS, MAX_MISSING_RATE,
    logger, pd, np, json, datetime
)

def main():
    logger.info("="*80)
    logger.info("GEMINI PIPELINE V3 - Independent Models (NO death-as-feature)")
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
    
    # === DEATH MODEL ===
    logger.info("\n--- DEATH MODEL ---")
    df_death, y_death = prepare_survival_target(train_df, outcome='death')
    X_death = X_train.loc[df_death.index]
    
    death_cv = cross_validate(X_death, y_death, n_folds=N_FOLDS, n_repeats=N_REPEATS)
    
    death_ensemble = SimpleEnsemble(weights={'rsf': 0.6, 'xgb': 0.4})
    death_ensemble.fit(X_death, y_death)
    
    # === HEPATIC MODEL (NO death feature) ===
    logger.info("\n--- HEPATIC MODEL (independent) ---")
    df_hep, y_hep = prepare_survival_target(train_df, outcome='hepatic')
    X_hep = X_train.loc[df_hep.index]
    
    hep_cv = cross_validate(X_hep, y_hep, n_folds=N_FOLDS, n_repeats=N_REPEATS)
    
    hep_ensemble = SimpleEnsemble(weights={'rsf': 0.6, 'xgb': 0.4})
    hep_ensemble.fit(X_hep, y_hep)
    
    # === GENERATE SUBMISSION ===
    logger.info("\n--- GENERATING SUBMISSION ---")
    pred_death = death_ensemble.predict(X_test)
    pred_hepatic = hep_ensemble.predict(X_test)
    
    submission = pd.DataFrame({
        'trustii_id': test_df['trustii_id'].values,
        'risk_hepatic_event': pred_hepatic,
        'risk_death': pred_death
    })
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = f'submissions/gemini_v3_submission_{timestamp}.csv'
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
        'n_features': len(common_cols),
        'elapsed_minutes': (datetime.now() - start_time).total_seconds() / 60
    }
    
    with open(f'submissions/gemini_v3_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nElapsed: {results['elapsed_minutes']:.1f} minutes")
    return results

if __name__ == "__main__":
    main()
