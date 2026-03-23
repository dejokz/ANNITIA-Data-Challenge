"""
Gemini Pipeline v4 - Optimized: XGB+RSF for Death, Pure RSF for Hepatic
"""

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, '.')
from pipeline_gemini_v2 import (
    TrajectoryFeatureEngineerV2, XGBSurvivalModel,
    prepare_survival_target, RANDOM_STATE, N_FOLDS, N_REPEATS,
    logger, pd, np, json, datetime, SimpleImputer, StandardScaler
)
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import RepeatedStratifiedKFold
from sksurv.metrics import concordance_index_censored


def cross_validate_rsf_only(X, y, n_folds=5, n_repeats=5):
    """CV with pure RSF (optimized for hepatic)."""
    logger.info(f"{n_folds}-fold × {n_repeats}-repeat CV with RSF only...")
    
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
        
        # Pure RSF - tuned for small dataset
        model = RandomSurvivalForest(
            n_estimators=500,        # More trees
            min_samples_leaf=15,     # Slightly lower (was 20)
            min_samples_split=30,    # Lower split
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
    logger.info("GEMINI PIPELINE V4 - XGB+RSF (Death), Pure RSF (Hepatic)")
    logger.info("="*80)
    
    start = datetime.now()
    
    # Load
    train_df = pd.read_csv('data/train-df.csv')
    test_df = pd.read_csv('data/test-df.csv')
    
    # Features
    logger.info("\nFeature Engineering...")
    engineer = TrajectoryFeatureEngineerV2()
    X_train = engineer.transform(train_df)
    X_test = engineer.transform(test_df)
    
    common_cols = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]
    
    # === DEATH MODEL: XGB+RSF Ensemble ===
    logger.info("\n--- DEATH MODEL (XGB+RSF) ---")
    from pipeline_gemini_v2 import cross_validate
    df_death, y_death = prepare_survival_target(train_df, outcome='death')
    X_death = X_train.loc[df_death.index]
    
    death_cv = cross_validate(X_death, y_death, n_folds=N_FOLDS, n_repeats=N_REPEATS)
    
    # Fit final death ensemble
    from pipeline_gemini_v2 import SimpleEnsemble
    death_ensemble = SimpleEnsemble(weights={'rsf': 0.6, 'xgb': 0.4})
    death_ensemble.fit(X_death, y_death)
    
    # === HEPATIC MODEL: Pure RSF ===
    logger.info("\n--- HEPATIC MODEL (Pure RSF) ---")
    df_hep, y_hep = prepare_survival_target(train_df, outcome='hepatic')
    X_hep = X_train.loc[df_hep.index]
    
    hep_cv = cross_validate_rsf_only(X_hep, y_hep, n_folds=N_FOLDS, n_repeats=N_REPEATS)
    
    # Fit final hepatic RSF
    keep_cols = hep_cv['feature_cols']
    X_hep_final = X_hep[keep_cols]
    
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    X_hep_proc = scaler.fit_transform(imputer.fit_transform(X_hep_final))
    
    hep_model = RandomSurvivalForest(
        n_estimators=1000,
        min_samples_leaf=15,
        min_samples_split=30,
        max_features='sqrt',
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    hep_model.fit(X_hep_proc, y_hep)
    
    # === GENERATE SUBMISSION ===
    logger.info("\n--- GENERATING SUBMISSION ---")
    pred_death = death_ensemble.predict(X_test)
    
    # Hepatic prediction
    X_test_hep = X_test[keep_cols]
    X_test_hep_proc = scaler.transform(imputer.transform(X_test_hep))
    pred_hepatic = hep_model.predict(X_test_hep_proc)
    
    submission = pd.DataFrame({
        'trustii_id': test_df['trustii_id'].values,
        'risk_hepatic_event': pred_hepatic,
        'risk_death': pred_death
    })
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = f'submissions/gemini_v4_submission_{timestamp}.csv'
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
        'elapsed_minutes': (datetime.now() - start).total_seconds() / 60
    }
    
    with open(f'submissions/gemini_v4_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nElapsed: {results['elapsed_minutes']:.1f} minutes")
    return results


if __name__ == "__main__":
    main()
