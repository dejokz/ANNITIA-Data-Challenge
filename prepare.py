"""
ANNITIA Autoresearch — Fixed utilities.
Do NOT modify this file. It provides data loading, CV harness, and scoring.
"""

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis

RANDOM_STATE = 42
N_FOLDS = 5
TIME_BUDGET_SECONDS = 600  # 10 minutes hard limit for agent runs

TRAIN_PATH = os.path.join('..', 'data', 'train-df.csv')
TEST_PATH = os.path.join('..', 'data', 'test-df.csv')


def load_data():
    """Load train and test dataframes."""
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, test_df


def prepare_survival_target(df, outcome='hepatic'):
    """Prepare sksurv-compatible structured array."""
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


def score_cv(oof_hep, y_hep, oof_death, y_death):
    """Compute C-indices from out-of-fold predictions."""
    hep_ci = concordance_index_censored(
        y_hep['HepaticEvent'], y_hep['Time'], oof_hep
    )[0]
    death_ci = concordance_index_censored(
        y_death['Death'], y_death['Time'], oof_death
    )[0]
    return {
        'hepatic_ci': float(hep_ci),
        'death_ci': float(death_ci),
        'average_ci': float((hep_ci + death_ci) / 2)
    }


def format_submission(test_df, pred_hep, pred_death):
    """Create submission dataframe."""
    return pd.DataFrame({
        'trustii_id': test_df['trustii_id'].values,
        'risk_hepatic_event': pred_hep,
        'risk_death': pred_death
    })


def run_stratified_cv(X, y, model_builder, n_folds=N_FOLDS):
    """
    Generic stratified CV runner.
    model_builder(X_train, y_train) -> fitted model
    model.predict(X_test) -> risk scores
    Returns oof predictions aligned with X index.
    """
    event_indicator = y[y.dtype.names[0]]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(X))
    for tr_idx, val_idx in skf.split(X, event_indicator):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr = y[tr_idx]
        model = model_builder(X_tr, y_tr)
        oof[val_idx] = model.predict(X_val)
    return oof
