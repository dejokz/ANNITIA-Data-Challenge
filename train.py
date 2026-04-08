"""
ANNITIA Autoresearch — train.py
This is the ONLY file the agent should edit.
Run: python train.py

Experiment #4b: Hepatic 2-model ensemble (RSF + XGB Cox) on full features.
Keep proven 3-model death ensemble.
"""

import warnings
warnings.filterwarnings('ignore')

import json
import time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
import xgboost as xgb

from prepare import load_data, prepare_survival_target, score_cv, format_submission

RANDOM_STATE = 42
N_FOLDS = 5

# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def calculate_clinical_scores(df):
    result = df.copy()
    max_visits = 22
    for visit in range(1, max_visits + 1):
        age_col = f'Age_v{visit}'
        ast_col = f'ast_v{visit}'
        alt_col = f'alt_v{visit}'
        plt_col = f'plt_v{visit}'
        required = [age_col, ast_col, alt_col, plt_col]
        if all(col in result.columns for col in required):
            result[f'fib4_v{visit}'] = (
                result[age_col] * result[ast_col] /
                (result[plt_col] * np.sqrt(result[alt_col].clip(lower=1)) + 0.001)
            )
            ULN_AST = 40
            result[f'apri_v{visit}'] = (
                (result[ast_col] / ULN_AST * 100) / (result[plt_col] + 0.001)
            )
            result[f'ast_alt_ratio_v{visit}'] = result[ast_col] / (result[alt_col] + 0.001)
    return result


def _calculate_slope(values):
    valid = values.dropna()
    if len(valid) < 2:
        return 0.0
    x = np.arange(len(valid))
    slope, _, _, _, _ = stats.linregress(x, valid.values)
    return slope


def extract_trajectory_features(df):
    visit_vars = [
        'fibs_stiffness_med_BM_1', 'fibrotest_BM_2', 'aixp_aix_result_BM_3',
        'alt', 'ast', 'plt', 'bilirubin', 'ggt',
        'gluc_fast', 'chol', 'triglyc', 'BMI',
        'fib4', 'apri', 'ast_alt_ratio'
    ]
    features = pd.DataFrame(index=df.index)

    for var in visit_vars:
        visit_cols = [c for c in df.columns
                      if c.startswith(f'{var}_v') and c.split('_v')[-1].isdigit()]
        if not visit_cols:
            continue
        visit_nums = [int(c.split('_v')[-1]) for c in visit_cols]
        sorted_pairs = sorted(zip(visit_nums, visit_cols))
        sorted_cols = [col for _, col in sorted_pairs]
        values = df[sorted_cols]

        features[f'{var}_max'] = values.max(axis=1)
        features[f'{var}_min'] = values.min(axis=1)
        features[f'{var}_mean'] = values.mean(axis=1)
        features[f'{var}_std'] = values.std(axis=1)
        features[f'{var}_first'] = values.iloc[:, 0]
        features[f'{var}_last'] = values.ffill(axis=1).iloc[:, -1]
        features[f'{var}_range'] = features[f'{var}_max'] - features[f'{var}_min']
        features[f'{var}_slope'] = values.apply(_calculate_slope, axis=1)

        age_cols = [c for c in df.columns if c.startswith('Age_v')]
        time_span = df[age_cols].max(axis=1) - df[age_cols].min(axis=1)
        features[f'{var}_roc'] = (features[f'{var}_last'] - features[f'{var}_first']) / (time_span + 0.001)

        # Variability
        features[f'{var}_cv'] = features[f'{var}_std'] / (features[f'{var}_mean'].abs() + 0.001)

        # Visit-to-visit deltas
        features[f'{var}_max_delta'] = values.diff(axis=1).max(axis=1)
        features[f'{var}_min_delta'] = values.diff(axis=1).min(axis=1)

        if var == 'fib4':
            features[f'{var}_time_high'] = (values > 2.67).sum(axis=1) / (values.notna().sum(axis=1) + 0.001)
            features[f'{var}_ever_high'] = (values > 2.67).any(axis=1).astype(int)
            features[f'{var}_worsening'] = (features[f'{var}_slope'] > 0.1).astype(int)
        elif var == 'fibs_stiffness_med_BM_1':
            features[f'{var}_time_high'] = (values > 8.0).sum(axis=1) / (values.notna().sum(axis=1) + 0.001)
            features[f'{var}_ever_high'] = (values > 8.0).any(axis=1).astype(int)
            features[f'{var}_worsening'] = (features[f'{var}_slope'] > 0.5).astype(int)
        elif var == 'fibrotest_BM_2':
            features[f'{var}_time_high'] = (values > 0.72).sum(axis=1) / (values.notna().sum(axis=1) + 0.001)
            features[f'{var}_ever_high'] = (values > 0.72).any(axis=1).astype(int)
        elif var == 'plt':
            features[f'{var}_declining'] = (features[f'{var}_slope'] < -5).astype(int)

    # Cross-NIT concordance
    nit_pairs = [
        ('fibs_stiffness_med_BM_1', 'fib4'),
        ('fibs_stiffness_med_BM_1', 'fibrotest_BM_2'),
        ('fib4', 'fibrotest_BM_2'),
    ]
    for nit1, nit2 in nit_pairs:
        slope1_col = f'{nit1}_slope'
        slope2_col = f'{nit2}_slope'
        if slope1_col in features.columns and slope2_col in features.columns:
            features[f'{nit1}_{nit2}_slope_agree'] = ((features[slope1_col] * features[slope2_col]) > 0).astype(int)

    fibrosis_markers = ['fibs_stiffness_med_BM_1_worsening', 'fib4_worsening', 'fibrotest_BM_2_worsening']
    available_markers = [c for c in fibrosis_markers if c in features.columns]
    if len(available_markers) >= 2:
        features['any_fibrosis_worsening'] = features[available_markers].any(axis=1).astype(int)
        features['n_fibrosis_worsening'] = features[available_markers].sum(axis=1)

    # Static features + interactions
    static_cols = ['gender', 'T2DM', 'Hypertension', 'Dyslipidaemia', 'bariatric_surgery']
    for col in static_cols:
        if col in df.columns:
            features[col] = df[col]
    if 'T2DM' in features.columns and 'fib4_max' in features.columns:
        features['T2DM_x_fib4_max'] = features['T2DM'] * features['fib4_max']
    if 'Hypertension' in features.columns and 'fibs_stiffness_med_BM_1_max' in features.columns:
        features['Hypertension_x_lsm_max'] = features['Hypertension'] * features['fibs_stiffness_med_BM_1_max']
    if 'age_last' in features.columns and 'fib4_mean' in features.columns:
        features['age_x_fib4_mean'] = features['age_last'] * features['fib4_mean']

    age_cols = [c for c in df.columns if c.startswith('Age_v')]
    if age_cols:
        features['age_baseline'] = df[age_cols].min(axis=1)
        features['age_last'] = df[age_cols].max(axis=1)
        features['followup_years'] = features['age_last'] - features['age_baseline']

    return features


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class RSFModel:
    """Simple RSF wrapper."""
    def __init__(self, n_estimators=300, min_samples_leaf=20):
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.model = None
        self.imputer = None
        self.scaler = None

    def fit(self, X, y):
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        X_proc = self.scaler.fit_transform(self.imputer.fit_transform(X))
        self.model = RandomSurvivalForest(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            max_features='sqrt',
            n_jobs=-1,
            random_state=RANDOM_STATE
        )
        self.model.fit(X_proc, y)
        return self

    def predict(self, X):
        X_proc = self.scaler.transform(self.imputer.transform(X))
        return self.model.predict(X_proc)


class XGBCoxModel:
    """XGBoost survival:cox wrapper."""
    def __init__(self, max_depth=4, learning_rate=0.05, num_boost_round=100,
                 subsample=0.8, colsample_bytree=0.8, min_child_weight=5):
        self.params = {
            'objective': 'survival:cox',
            'eval_metric': 'cox-nloglik',
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'min_child_weight': min_child_weight,
            'random_state': RANDOM_STATE,
        }
        self.num_boost_round = num_boost_round
        self.model = None
        self.imputer = None
        self.scaler = None

    def fit(self, X, y):
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        X_proc = self.scaler.fit_transform(self.imputer.fit_transform(X))
        time_ = y[y.dtype.names[1]]
        event = y[y.dtype.names[0]]
        y_lower = time_
        y_upper = np.where(event, time_, float('inf'))
        dtrain = xgb.DMatrix(X_proc, label=y_lower)
        dtrain.set_float_info('label_lower_bound', y_lower)
        dtrain.set_float_info('label_upper_bound', y_upper)
        self.model = xgb.train(self.params, dtrain, num_boost_round=self.num_boost_round)
        return self

    def predict(self, X):
        X_proc = self.scaler.transform(self.imputer.transform(X))
        return self.model.predict(xgb.DMatrix(X_proc))


class XGBRegressorSurvivalModel:
    """XGBRegressor on log(time) with event weights. Risk = -prediction."""
    def __init__(self, max_depth=6, learning_rate=0.05, n_estimators=300,
                 subsample=0.8, colsample_bytree=0.8, min_child_weight=3):
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        self.imputer = None
        self.scaler = None

    def fit(self, X, y):
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        X_proc = self.scaler.fit_transform(self.imputer.fit_transform(X))
        event = y[y.dtype.names[0]]
        time_ = y[y.dtype.names[1]]
        target = np.log1p(time_.astype(float))
        weight = event.astype(float)
        self.model.fit(X_proc, target, sample_weight=weight)
        return self

    def predict(self, X):
        X_proc = self.scaler.transform(self.imputer.transform(X))
        return -self.model.predict(X_proc)


class HepaticEnsemble:
    """2-model ensemble: RSF + XGB Cox for hepatic."""
    def __init__(self, rsf_weight=0.6):
        self.rsf_weight = rsf_weight
        self.xgb_weight = 1.0 - rsf_weight
        self.rsf = RSFModel(n_estimators=300, min_samples_leaf=20)
        self.xgb_cox = XGBCoxModel(max_depth=5, learning_rate=0.05, num_boost_round=150)

    def fit(self, X, y):
        self.rsf.fit(X, y)
        self.xgb_cox.fit(X, y)
        return self

    def predict(self, X):
        p_rsf = self.rsf.predict(X)
        p_cox = self.xgb_cox.predict(X)
        for p in [p_rsf, p_cox]:
            if np.std(p) > 0:
                p -= np.mean(p)
                p /= (np.std(p) + 1e-8)
        return self.rsf_weight * p_rsf + self.xgb_weight * p_cox


class DeathEnsemble:
    """3-model ensemble: RSF + XGB Cox + XGBRegressor."""
    def __init__(self, weights=None):
        self.weights = weights or {'rsf': 0.4, 'xgb_cox': 0.3, 'xgb_reg': 0.3}
        self.rsf = RSFModel(n_estimators=400, min_samples_leaf=20)
        self.xgb_cox = XGBCoxModel(max_depth=5, learning_rate=0.05, num_boost_round=150)
        self.xgb_reg = XGBRegressorSurvivalModel(max_depth=6, learning_rate=0.05, n_estimators=300)

    def fit(self, X, y):
        self.rsf.fit(X, y)
        self.xgb_cox.fit(X, y)
        self.xgb_reg.fit(X, y)
        return self

    def predict(self, X):
        p_rsf = self.rsf.predict(X)
        p_cox = self.xgb_cox.predict(X)
        p_reg = self.xgb_reg.predict(X)
        for p in [p_rsf, p_cox, p_reg]:
            if np.std(p) > 0:
                p -= np.mean(p)
                p /= (np.std(p) + 1e-8)
        return (self.weights['rsf'] * p_rsf +
                self.weights['xgb_cox'] * p_cox +
                self.weights['xgb_reg'] * p_reg)


# ---------------------------------------------------------------------------
# CV Harness
# ---------------------------------------------------------------------------

def cross_validate(X, y, model_class, n_folds=N_FOLDS, **model_kwargs):
    event_indicator = y[y.dtype.names[0]]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(X))
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, event_indicator)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr = y[tr_idx]
        model = model_class(**model_kwargs).fit(X_tr, y_tr)
        oof[val_idx] = model.predict(X_val)
    return oof


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    start = time.time()

    train_df, test_df = load_data()

    # Feature engineering
    train_df = calculate_clinical_scores(train_df)
    test_df = calculate_clinical_scores(test_df)
    X_train = extract_trajectory_features(train_df)
    X_test = extract_trajectory_features(test_df)

    common_cols = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    # Prepare targets
    df_hep, y_hep = prepare_survival_target(train_df, 'hepatic')
    df_death, y_death = prepare_survival_target(train_df, 'death')

    X_hep = X_train.loc[df_hep.index]
    X_death = X_train.loc[df_death.index]
    X_test_hep = X_test.loc[test_df.index]
    X_test_death = X_test.loc[test_df.index]

    # -----------------------------------------------------------------------
    # AGENT EDIT ZONE
    # -----------------------------------------------------------------------

    # Hepatic: 2-model ensemble (RSF + XGB Cox)
    oof_hep = cross_validate(X_hep, y_hep, HepaticEnsemble, n_folds=N_FOLDS, rsf_weight=0.6)
    final_hep = HepaticEnsemble(rsf_weight=0.6).fit(X_hep, y_hep)
    pred_hep = final_hep.predict(X_test_hep)

    # Death: proven 3-model ensemble
    oof_death = cross_validate(X_death, y_death, DeathEnsemble, n_folds=N_FOLDS)
    final_death = DeathEnsemble().fit(X_death, y_death)
    pred_death = final_death.predict(X_test_death)

    # -----------------------------------------------------------------------

    scores = score_cv(oof_hep, y_hep, oof_death, y_death)
    elapsed = time.time() - start

    # Save submission
    sub = format_submission(test_df, pred_hep, pred_death)
    sub.to_csv('submissions/latest_submission.csv', index=False)

    # Print standardized output
    print("---")
    print(f"hepatic_ci:       {scores['hepatic_ci']:.6f}")
    print(f"death_ci:         {scores['death_ci']:.6f}")
    print(f"average_ci:       {scores['average_ci']:.6f}")
    print(f"elapsed_seconds:  {elapsed:.1f}")


if __name__ == '__main__':
    main()
