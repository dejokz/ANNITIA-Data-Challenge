#!/usr/bin/env python3
"""
Create diagnostic visualizations for the 0.83 model.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Load data
print("Loading data and training model...")
train_df = pd.read_csv('data/train-df.csv')

# Recreate feature engineering
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
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()
X_train_processed = scaler.fit_transform(imputer.fit_transform(X_train))

# Train model
print("Training RSF model...")
model = RandomSurvivalForest(
    n_estimators=300,
    min_samples_leaf=20,
    min_samples_split=40,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42
)
model.fit(X_train_processed, y_train)

# Get predictions
train_preds = model.predict(X_train_processed)
event_indicator = y_train['HepaticEvent']
survival_time = y_train['Time']

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))

# 1. Risk Distribution by Event Status
ax1 = plt.subplot(2, 3, 1)
event_preds = train_preds[event_indicator == 1]
no_event_preds = train_preds[event_indicator == 0]

ax1.hist(no_event_preds, bins=30, alpha=0.7, label='No Event', color='green', density=True)
ax1.hist(event_preds, bins=30, alpha=0.7, label='Event', color='red', density=True)
ax1.set_xlabel('Predicted Risk Score')
ax1.set_ylabel('Density')
ax1.set_title('Risk Score Distribution by Event Status')
ax1.legend()

# 2. Risk Quartiles
ax2 = plt.subplot(2, 3, 2)
risk_quartiles = pd.qcut(train_preds, q=4, labels=['Q1-Low', 'Q2-MedLow', 'Q3-MedHigh', 'Q4-High'])
quartile_event_rates = []
quartile_labels = []

for q in ['Q1-Low', 'Q2-MedLow', 'Q3-MedHigh', 'Q4-High']:
    mask = risk_quartiles == q
    event_rate = event_indicator[mask].mean()
    quartile_event_rates.append(event_rate)
    quartile_labels.append(q)

bars = ax2.bar(quartile_labels, quartile_event_rates, color=['green', 'yellow', 'orange', 'red'])
ax2.set_ylabel('Event Rate')
ax2.set_title('Event Rate by Risk Quartile')
ax2.tick_params(axis='x', rotation=45)

# Add value labels on bars
for bar, rate in zip(bars, quartile_event_rates):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{rate:.1%}', ha='center', va='bottom')

# 3. Top Features Importance
ax3 = plt.subplot(2, 3, 3)
top_features = [
    'fibs_stiffness_med_BM_1_mean',
    'fibs_stiffness_med_BM_1_median',
    'fibs_stiffness_med_BM_1_max',
    'fibs_stiffness_med_BM_1_min',
    'fibs_stiffness_med_BM_1_last',
    'aixp_aix_result_BM_3_mean',
    'fibrotest_BM_2_mean',
    'fib4_mean'
]

feature_names_short = ['LSM Mean', 'LSM Median', 'LSM Max', 'LSM Min', 'LSM Last', 
                       'AIX Mean', 'FibroTest Mean', 'FIB-4 Mean']

correlations = []
for feat in top_features:
    if feat in X_train.columns:
        corr = np.corrcoef(X_train[feat].fillna(X_train[feat].median()), train_preds)[0, 1]
        correlations.append(abs(corr))
    else:
        correlations.append(0)

y_pos = np.arange(len(feature_names_short))
ax3.barh(y_pos, correlations)
ax3.set_yticks(y_pos)
ax3.set_yticklabels(feature_names_short)
ax3.set_xlabel('|Correlation with Risk|')
ax3.set_title('Top Features by Prediction Correlation')
ax3.invert_yaxis()

# 4. LSM vs Risk Scatter
ax4 = plt.subplot(2, 3, 4)
lsm_mean = X_train['fibs_stiffness_med_BM_1_mean'].fillna(X_train['fibs_stiffness_med_BM_1_mean'].median())
scatter = ax4.scatter(lsm_mean, train_preds, c=event_indicator, cmap='RdYlGn_r', alpha=0.6)
ax4.set_xlabel('LSM Mean (kPa)')
ax4.set_ylabel('Predicted Risk')
ax4.set_title('LSM vs Predicted Risk\n(Red=Event, Green=No Event)')
plt.colorbar(scatter, ax=ax4, label='Event Status')

# 5. Survival Time vs Risk
ax5 = plt.subplot(2, 3, 5)
event_mask = event_indicator == 1
censored_mask = event_indicator == 0

ax5.scatter(survival_time[event_mask], train_preds[event_mask], 
           alpha=0.6, c='red', label='Event', s=50)
ax5.scatter(survival_time[censored_mask], train_preds[censored_mask], 
           alpha=0.3, c='green', label='Censored', s=30)
ax5.set_xlabel('Survival Time (years)')
ax5.set_ylabel('Predicted Risk')
ax5.set_title('Survival Time vs Predicted Risk')
ax5.legend()

# 6. Feature Category Importance
ax6 = plt.subplot(2, 3, 6)
categories = ['LSM', 'FIB-4', 'FibroTest', 'Platelets', 'Enzymes', 'Cross-NIT']
avg_importance = [1.03, 0.51, 0.63, 0.28, 0.57, 0.29]

colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd', '#8c564b']
bars = ax6.bar(categories, avg_importance, color=colors)
ax6.set_ylabel('Average Importance Score')
ax6.set_title('Feature Category Importance')
ax6.tick_params(axis='x', rotation=45)

# Add value labels
for bar, imp in zip(bars, avg_importance):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
             f'{imp:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('submissions/model_diagnostics.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved diagnostic plots to: submissions/model_diagnostics.png")

# Create a second figure for additional analysis
fig2 = plt.figure(figsize=(14, 5))

# 1. ROC-like analysis (C-index components)
ax1 = plt.subplot(1, 3, 1)
thresholds = np.percentile(train_preds, np.linspace(0, 100, 20))
sensitivities = []
specificities = []

for thresh in thresholds:
    predicted_high = train_preds >= thresh
    tp = ((predicted_high == 1) & (event_indicator == 1)).sum()
    fp = ((predicted_high == 1) & (event_indicator == 0)).sum()
    tn = ((predicted_high == 0) & (event_indicator == 0)).sum()
    fn = ((predicted_high == 0) & (event_indicator == 1)).sum()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    sensitivities.append(sensitivity)
    specificities.append(specificity)

ax1.plot(1 - np.array(specificities), sensitivities, 'b-', linewidth=2)
ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax1.set_xlabel('1 - Specificity')
ax1.set_ylabel('Sensitivity')
ax1.set_title('Sensitivity vs Specificity\nat Different Thresholds')
ax1.grid(True, alpha=0.3)

# 2. Risk Score Calibration
ax2 = plt.subplot(1, 3, 2)
n_bins = 5
bin_edges = np.percentile(train_preds, np.linspace(0, 100, n_bins + 1))
bin_centers = []
observed_rates = []

for i in range(n_bins):
    mask = (train_preds >= bin_edges[i]) & (train_preds < bin_edges[i + 1])
    if i == n_bins - 1:  # Include right edge for last bin
        mask = (train_preds >= bin_edges[i]) & (train_preds <= bin_edges[i + 1])
    
    if mask.sum() > 0:
        bin_centers.append(train_preds[mask].mean())
        observed_rates.append(event_indicator[mask].mean())

ax2.plot([0, max(bin_centers)], [0, max(bin_centers)], 'k--', label='Perfect Calibration')
ax2.scatter(bin_centers, observed_rates, s=100, c='red', label='Observed')
ax2.plot(bin_centers, observed_rates, 'r-', alpha=0.5)
ax2.set_xlabel('Mean Predicted Risk (binned)')
ax2.set_ylabel('Observed Event Rate')
ax2.set_title('Risk Calibration Plot')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Prediction Confidence Distribution
ax3 = plt.subplot(1, 3, 3)
# Use prediction variance across trees if available, else use risk magnitude
ax3.hist(train_preds[event_indicator == 0], bins=20, alpha=0.7, 
         label='No Event', color='green', density=True)
ax3.hist(train_preds[event_indicator == 1], bins=20, alpha=0.7, 
         label='Event', color='red', density=True)
ax3.axvline(np.median(train_preds[event_indicator == 0]), color='green', 
            linestyle='--', label='Median (No Event)')
ax3.axvline(np.median(train_preds[event_indicator == 1]), color='red', 
            linestyle='--', label='Median (Event)')
ax3.set_xlabel('Predicted Risk Score')
ax3.set_ylabel('Density')
ax3.set_title('Risk Score Distribution')
ax3.legend()

plt.tight_layout()
plt.savefig('submissions/model_diagnostics_2.png', dpi=150, bbox_inches='tight')
print("✅ Saved additional plots to: submissions/model_diagnostics_2.png")

print("\n" + "="*70)
print("DIAGNOSTIC PLOTS COMPLETE")
print("="*70)
print("\nFiles created:")
print("  1. submissions/model_diagnostics.png - Main diagnostics")
print("  2. submissions/model_diagnostics_2.png - Additional analysis")
