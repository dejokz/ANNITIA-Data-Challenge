# ANNITIA Data Challenge - MASLD Survival Analysis

My solution to the [ANNITIA Data Challenge](https://app.trustii.io/datasets/1551) for MASLD (Metabolic dysfunction-associated steatotic liver disease) survival analysis.

## 🏆 Results

| Metric | Score |
|--------|-------|
| **Public Leaderboard** | **0.8529** |
| Baseline | 0.8381 |
| Improvement | +0.0148 |

## 📊 Challenge Overview

- **Task:** Predict hepatic events and death from longitudinal patient data
- **Train:** 1,253 patients
- **Test:** 423 patients
- **Hepatic Events:** 47 (3.8%)
- **Death Events:** 76 (7.7%)
- **Visits per patient:** ~22 longitudinal measurements

## 🚀 Key Improvements

### What Worked
1. **XGBoost Survival Model for Death Prediction** - Improved death C-index from ~0.75 → **0.9153**
2. **RepeatedStratifiedKFold (5×5)** - More stable cross-validation estimates
3. **Ensemble of RSF + XGBoost** - Combined tree-based models

### What Didn't Work
- CoxNet (ElasticNet) - consistently returned random predictions (0.5000)
- Feature selection (minimal features) - lost predictive signal
- Death-as-feature for hepatic model - no improvement
- EWMA features - neutral impact

## 📁 Repository Structure

```
.
├── src/
│   ├── pipeline.py                 # Baseline model (0.838 LB)
│   ├── pipeline_gemini_v2.py       # Best model (0.853 LB)
│   ├── pipeline_gemini_v*.py       # Experimental variants
│   ├── submit.py                   # Trustii submission script
│   └── ...
├── submissions/
│   ├── gemini_v2_submission_*.csv  # Best submission file
│   └── ...
├── notebooks/
│   └── annitia_submission.ipynb    # Challenge notebook
├── requirements.txt
└── README.md
```

## 🔧 Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run best model
python src/pipeline_gemini_v2.py

# Submit to leaderboard
python src/submit.py
```

## 📝 Dependencies

- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-survival >= 0.22.0
- xgboost >= 2.0.0
- lightgbm >= 4.3.0
- lifelines >= 0.27.0
- scikit-learn >= 1.3.0

## 📄 License

This repository contains my personal solution to the ANNITIA Data Challenge.
