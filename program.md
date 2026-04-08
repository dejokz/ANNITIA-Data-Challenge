# ANNITIA Autoresearch Program

## Mission
Autonomously improve the survival analysis pipeline for the ANNITIA MASLD Data Challenge.

**Target metric:** `average_ci = (hepatic_ci + death_ci) / 2` — higher is better.

**Current baseline (in train.py):**
- Hepatic C-index: ~0.79
- Death C-index: ~0.91
- Average C-index: ~0.85

The hepatic model is the bottleneck. Your primary goal is to push hepatic_ci above 0.80 without crashing death_ci below 0.90.

---

## Setup

1. **Read the in-scope files**:
   - `prepare.py` — fixed data loading, CV scoring, submission formatting. **Do not modify.**
   - `train.py` — the file you edit. Contains feature engineering, models, and the experiment loop.
   - `program.md` — this file.

2. **Verify data exists**: `../data/train-df.csv` and `../data/test-df.csv` should exist.

3. **Baseline run**: Your first run should always be the unmodified `train.py` to establish the baseline score.

---

## Experimentation

**What you CAN do:**
- Modify ONLY `train.py`. Everything inside is fair game:
  - Feature engineering (add new trajectory features, ratios, interactions, EWMA, deltas)
  - Model architecture (change RSF hyperparameters, XGBoost params, ensemble weights)
  - Feature selection (univariate ranking, missing-rate filters, correlation pruning)
  - Target engineering (different time calculations, stratification schemes)
  - Add new models or remove underperforming ones

**What you CANNOT do:**
- Modify `prepare.py`.
- Install new packages not already in the environment (xgboost, sksurv, pandas, numpy, scipy, sklearn are available).
- Change the evaluation metric or CV harness.
- Use CoxnetSurvivalAnalysis (proven to crash/return 0.5000 due to numerical instability).

**Time budget:** Each experiment should finish within ~10 minutes. If it exceeds 15 minutes, kill it and treat as a failure.

---

## Domain Knowledge

- **Hepatic events are extremely rare**: 47 events out of 1253 patients (3.8%).
- **Death events**: 76 out of 1253 (6.1%).
- The key predictors are longitudinal NIT trajectories:
  - `fibs_stiffness_med_BM_1` — liver stiffness measurement (LSM)
  - `fibrotest_BM_2` — FibroTest score
  - `alt`, `ast`, `plt` — standard liver labs
  - Clinical scores derived from visits: FIB-4, APRI, AST/ALT ratio
- Trajectory dynamics (slopes, max values, time-above-threshold) matter more than single snapshots.
- RepeatedStratifiedKFold with more repeats stabilizes variance for the hepatic model.

---

## What Has Already Been Tried

From the parent repo history:
- Simple RSF baseline → average ~0.838 LB
- XGB+RSF ensemble for death → death 0.9153 (major win)
- RSF+XGB for hepatic with death predictions as feature → hepatic ~0.788 (plateau)
- CoxNet → 0.5000 (abandon)
- Minimal features → hepatic drops to ~0.74
- RepeatedStratifiedKFold (5×5) improved stability

---

## High-Value Hypotheses to Test

1. **Advanced trajectory features**: rate-of-change acceleration, visit-to-visit deltas, coefficient of variation, time-since-last-normal.
2. **Last-N-visits only**: Restrict features to the last 3-5 visits instead of all 22 visits.
3. **Pure XGBoost for hepatic**: The death model improved dramatically with XGBoost. Hepatic has not been tried with XGBoost-only yet.
4. **Deeper ensembles**: 3-model ensembles (RSF + XGB + GradientBoostingSurvivalAnalysis) with learned weights.
5. **Feature selection by univariate C-index**: Rank all features individually and keep only top 30-60.
6. **Interaction terms**: T2DM × fibrosis progression, age × LSM slope, etc.
7. **Patient clustering**: Cluster by trajectory shape and train separate small models.
8. **Different imputation**: Use forward-fill then median, or visit-specific medians.

---

## Output Format

After running `python train.py`, the script prints:

```
---
hepatic_ci:       0.789123
death_ci:         0.915300
average_ci:       0.852212
elapsed_seconds:  245.3
```

Extract the key metrics with:
```bash
grep "^hepatic_ci:\|^death_ci:\|^average_ci:" run.log
```

---

## Logging Results

The `loop.py` script logs experiments to `experiments.json`. Do not commit this file to git. It tracks:
- `commit` — git commit hash
- `average_ci` — primary metric
- `hepatic_ci` / `death_ci` — component metrics
- `status` — `keep`, `discard`, or `crash`
- `description` — what you tried

---

## Experiment Loop

LOOP FOREVER:
1. Inspect current `train.py` and git state.
2. Propose a hypothesis based on `experiments.json` history and this `program.md`.
3. Edit `train.py` directly.
4. `git commit -am "experiment: <description>"`
5. Run: `python train.py > run.log 2>&1`
6. Read metrics: `grep "^average_ci:" run.log`
7. If empty → crash. Read `tail -n 50 run.log`, attempt a fix. If unfixable, log crash and move on.
8. If `average_ci` improved → keep the commit.
9. If `average_ci` equal or worse → `git reset --hard HEAD~1` to discard.

**NEVER STOP**. Continue experimenting autonomously. If you run out of ideas, re-read this file, combine previous near-misses, or try more radical architectural changes.

**Simplicity criterion**: A tiny improvement that adds massive complexity is questionable. A simplification that maintains score is a win. A big hepatic improvement (>0.01) justifies moderate complexity.
