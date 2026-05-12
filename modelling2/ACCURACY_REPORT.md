# Accuracy report — clustering V2 (greenfield wash projection)

All content below refers only to:

**`daily_data/daily-data-modelling/clustering_v2/`**

Figures are from  
**`daily_data/daily-data-modelling/clustering_v2/results/monthly_level_holdout_eval.json`**  
(re-run **`python eval_accuracy_report.py`** from that directory for fresh numbers).

---

## Holdout level models (Ridge vs RandomForest)

**What is evaluated:** models are fit on **train** only; scores are on **held-out** rows. **>2y:** each row = one **site × calendar month** in test; actual = sum of **daily** washes; prediction = sum of **daily** level-model outputs for that month. **<2y:** each row = one **site × month** (already monthly); train = first-year months, test = second-year months.

### Main metrics

| Cohort | Model | n | WAPE | R² | MAE (washes / month) |
|--------|--------|---:|-----:|---:|---------------------:|
| **>2y** monthly aggregate | Ridge | 2,888 | 0.329 | 0.679 | 3,004 |
| **>2y** monthly aggregate | RandomForest | 2,888 | **0.142** | **0.884** | **1,293** |
| **<2y** monthly panel | Ridge | 4,689 | 0.140 | 0.778 | 1,357 |
| **<2y** monthly panel | RandomForest | 4,689 | **0.137** | 0.773 | **1,333** |

### “How close in percent?” (same rows as above)

| Cohort | Model | Within ±10% of actual | Within ±15% | Median abs % error |
|--------|--------|------------------------:|--------------:|-------------------:|
| >2y | Ridge | 18.0% | 27.7% | 29.3% |
| >2y | RandomForest | **45.8%** | **61.9%** | **11.1%** |
| <2y | Ridge | 50.0% | 64.3% | 10.2% |
| <2y | RandomForest | **52.1%** | **67.8%** | **9.4%** |

### What “% within ±10% of actual” means

For **each** evaluation row: **absolute percent error**  
`|prediction − actual| ÷ actual` (only where `actual > 0`).  
**“Within ±10%”** means that error is **≤ 0.10** (prediction between 90% and 110% of actual). The table value is the **share of rows** that pass (higher = better).

### Pred vs peer train range (“inside cluster min–max %”)

Same **field names** in JSON for both cohorts; the **band** is defined at **monthly wash total** grain.

| Cohort | Model | Pred inside peer train min–max | Actual inside same band |
|--------|--------|--------------------------------:|--------------------------:|
| **>2y** (monthly sum from daily) | Ridge | **88.2%** | **95.0%** |
| **>2y** | RandomForest | **99.5%** | **95.0%** |
| **<2y** (one row = one month) | Ridge | **97.8%** | **97.0%** |
| **<2y** | RandomForest | **98.6%** | **97.0%** |

**>2y band:** train-only, per cluster: min/max of peer **site × calendar month** wash totals (sum daily `wash_count_total`), then compare each test site-month pred/actual (`eval_accuracy_report.py`; see `peer_band_definition` in JSON).

**<2y band:** `ctx_wash_count_total_min` / `max` from cluster context on monthly train rows.

### Short review

**RandomForest** beats **Ridge** on **>2y** monthly aggregates (WAPE **0.33 → 0.14**, many more site-months within ±10%). **<2y** is strong for both. Clustering + TS are not separately scored here—the headline is **level-model** holdout accuracy.

More detail: **`daily_data/daily-data-modelling/clustering_v2/APPROACH_AND_EVAL_BRIEF.md`**

---

## Method and accuracy metrics (shareable)

### Method (brief)

1. Split data into **train** and **holdout test** (unseen rows).
2. Train **Ridge** and **RandomForest** level models on train only.
3. Evaluate on holdout monthly site-level outcomes:
   - **>2y cohort:** aggregate daily predictions and actuals to site-month totals.
   - **<2y cohort:** evaluate directly on monthly panel rows.
4. Compare models using error, fit, and closeness-to-actual metrics.

### Accuracy metrics table

| Cohort | Model | WAPE | R² | MAE (washes/month) | Within ±10% | Within ±15% | Median abs % error |
|--------|-------|-----:|---:|-------------------:|------------:|------------:|-------------------:|
| >2y | Ridge | 0.329 | 0.679 | 3,004 | 18.0% | 27.7% | 29.3% |
| >2y | RandomForest | **0.142** | **0.884** | **1,293** | **45.8%** | **61.9%** | **11.1%** |
| <2y | Ridge | 0.140 | **0.778** | 1,357 | 50.0% | 64.3% | 10.2% |
| <2y | RandomForest | **0.137** | 0.773 | **1,333** | **52.1%** | **67.8%** | **9.4%** |

### Metric meanings

| Metric | Meaning | Better direction |
|--------|---------|------------------|
| WAPE | Weighted Absolute Percentage Error = `sum(|prediction - actual|) / sum(actual)`. Interprets total absolute error as a percentage of total actuals. | Lower |
| R² | Variance explained by model predictions on holdout rows. | Higher |
| MAE | Mean absolute error in washes/month (average miss in original units). | Lower |
| Within ±10% | Share of rows with absolute percent error `<= 10%`. | Higher |
| Within ±15% | Share of rows with absolute percent error `<= 15%`. | Higher |
| Median abs % error | Median absolute percentage error across rows (typical percentage miss, robust to outliers). | Lower |

---

## Files (clustering_v2 only)

| Path | Contents |
|------|----------|
| `daily_data/daily-data-modelling/clustering_v2/results/monthly_level_holdout_eval.json` | Ridge + RF holdout metrics |
| `daily_data/daily-data-modelling/clustering_v2/results/monthly_ridge_holdout_eval.json` | Same JSON (legacy mirror filename) |
| `daily_data/daily-data-modelling/clustering_v2/eval_accuracy_report.py` | Regenerates the eval JSON |
| `daily_data/daily-data-modelling/clustering_v2/build_v2.py` | Trains Ridge + RF artifacts under `.../clustering_v2/models/` |

---

## Zeta modelling accuracy (phase2 + phase3)

### Brief approach

1. Build monthly features from historical data (`phase1_final_monthly_2024_2025.csv`).
2. Train LightGBM models on a time split (train before 2025-01-01, test from 2025-01-01).
3. Select the better core model (`lightgbm_with_lag`).
4. For new sites, infer nearest clusters from location, then generate 5-year forecasts with uncertainty calibration.
5. Backtest end-to-end forecast quality.

### How each cluster's 60-month curve is generated

1. Create 60 future monthly rows (`month=1..60`) with date/age/seasonality features.
2. For a chosen `cluster_id`, attach cluster baselines (`cluster_month_avg`, `cluster_age_avg`, and related variance/growth features).
3. Predict month-wise multipliers using trained models (early-life and main-stage blocks).
4. Convert to volume month-wise: `pred_volume[m] = pred_multiplier[m] * cluster_month_avg[m]`.
5. Repeat for all 60 months to get that cluster's curve, then blend top-3 cluster curves by distance-based weights.

### Accuracy metrics

| Stage | Model / evaluation | MAE | RMSE | Interpretation |
|------|---------------------|----:|-----:|----------------|
| Phase 2 (time-split holdout) | lightgbm_no_lag | 2,236.31 | 3,716.71 | Baseline LightGBM variant |
| Phase 2 (time-split holdout) | lightgbm_with_lag | **1,466.60** | **2,679.94** | Best core predictive model |
| Phase 3 (advanced pipeline backtest) | End-to-end forecast pipeline | 2,089.77 | 3,604.58 | Full pipeline error after clustering + forecasting |

### Uncertainty interval quality (phase3 backtest)

| Metric | Value | Meaning |
|--------|------:|---------|
| p10-p90 coverage | 0.454 | Only 45.4% actuals landed inside predicted 10-90% band |
| target coverage | 0.800 | Desired interval coverage used for calibration |

### Metric meanings

| Metric | Meaning | Better direction |
|--------|---------|------------------|
| MAE | Mean absolute error in washes/month; average miss in original units. | Lower |
| RMSE | Root mean squared error in washes/month; penalizes large misses more than MAE. | Lower |
| p10-p90 coverage | Share of actuals lying within predicted 10th-90th percentile range. | Closer to target |
