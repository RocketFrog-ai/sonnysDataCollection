# Model 2 (`data_2`) - Cohort-Separated 60-Month Forecasting

This README documents the approach implemented in:

- `zeta_modelling/model_2/cohort_forecast_pipeline.py`

using data in:

- `zeta_modelling/data_2/less_than-2yrs.csv`
- `zeta_modelling/data_2/more_than-2yrs.csv`

The objective is to forecast monthly wash volume for a new site over 60 months, while preserving lifecycle behavior differences between early-life (`<2y`) and mature (`>2y`) cohorts.

## Why this approach

The two cohorts have different behavior and signal quality:

- `<2y` captures ramp-up and high early-stage variability.
- `>2y` captures mature seasonality/stability and long-run baseline patterns.

Instead of forcing one model to learn both regimes, Model 2 trains and forecasts them separately, then stitches them in time:

- months 1-24: early cohort model
- months 25-60: mature cohort model blended with mature cluster time-series forecast

---

## Pipeline overview

## 1) Data preparation (separate by cohort)

### `<2y` input (`less_than-2yrs.csv`)
- Builds `site_id` from `site_client_id` (with normalization), fallback to lat/lon.
- Converts `month_number` into dates:
  - 1-12 -> 2024-01..2024-12
  - 13-24 -> 2025-01..2025-12
- Aggregates to monthly site-level volume.
- Uses `period_index` as `age_in_months` (fallback to per-site cumulative index if missing).

### `>2y` input (`more_than-2yrs.csv`)
- Builds `site_id` similarly.
- Aggregates daily `calendar_day` into monthly timestamps.
- Filters to 2024-01 through 2025-12.
- Creates `age_in_months = per-site cumulative month index + 24` to align mature lifecycle.

Both cohorts remain separate through feature engineering, clustering, modeling, and evaluation.

## 2) Spatial clustering (separate DBSCAN runs)

For each cohort independently:

- DBSCAN with haversine distance (`eps_km=12`, `min_samples=5`).
- Produces `cluster_id` per site.
- Saves separate centroid files.

This lets local market structure differ between early and mature cohorts.

## 3) Feature engineering

Shared foundations:
- month seasonality (`month`, `month_sin`, `month_cos`)
- lifecycle shape (`age_sq`, `log_age`, stage flags, `age_saturation`)
- geo interaction (`lat_lon_interaction`)
- cluster growth profile (`cluster_growth_rate`, `cluster_age_avg`, `growth_velocity`)

`<2y`-specific cluster features include:
- `cluster_month_avg`, `cluster_month_std`
- `cluster_month_mean`, `cluster_month_median`, `cluster_month_p25`, `cluster_month_p75`

`>2y`-specific cluster features include:
- `cluster_daily_avg`, `cluster_std`
- `cluster_month_avg`, `cluster_month_median`, `cluster_month_p25`, `cluster_month_p75`

Target for both cohorts:
- `target_multiplier = monthly_volume / cluster_month_avg`

## 4) Site-type modeling

- Site behavior stats (`site_avg_volume`, `site_peak`, `site_std`) are clustered with KMeans to derive latent site types.
- A LightGBM classifier predicts site-type probabilities from geography + cluster context.
- For `<2y` prediction, top-k probable site types are used for probabilistic blending during inference (instead of hard single-type assignment).

## 5) Cohort models

### `<2y` model
- Two LightGBM multiplier models:
  - early model (`age_in_months <= 6`)
  - main model (`age_in_months > 6`)
- If split subsets are too small, fallback uses broader training data.

### `>2y` model
- One LightGBM multiplier model trained on mature cohort.

## 6) Mature cluster time-series models

- For each mature cluster, builds monthly median volume series from training period.
- Fits Holt-Winters (Exponential Smoothing) per cluster (with median fallback when fitting fails or history is short).
- Produces 36-step forecasts used for months 25-60.

## 7) 60-month new-site forecast stitching

Given `(lat, lon)`:

1. Finds nearest `<2y` and `>2y` clusters from their separate centroids.
2. Builds 60 monthly rows starting from `start_date` (default `2026-01-01`).
3. Predicts months 1-24 from `<2y` model path.
4. Predicts months 25-60 from mature LightGBM baseline and blends with mature Holt-Winters forecast:
   - Holt-Winters branch is anchor-scaled to connect smoothly from month 24.
   - blend weight transitions over horizon (higher model weight near month 25, lower near month 60).
5. Generates uncertainty bands:
   - base width from cohort-specific cluster std
   - `p10/p90 = p50 +/- 1.28 * std * interval_width_scale`

## 8) Evaluation and calibration

Time split:
- train: before `2025-01-01`
- test: from `2025-01-01`

Reported metrics:
- MAE and WMAPE for `<2y` and `>2y`
- interval coverage before/after calibration
- cold-start holdout MAE (held-out `<2y` sites, first 6 months)

Interval calibration:
- starts from raw std-based `p10/p90`
- iteratively rescales interval width toward target coverage (0.80), with guardrails to avoid over-adjustment.

---

## Current outputs (`data_2/model_2_outputs`)

Generated files:

- `model2_metrics.json`
- `model2_sample_forecast_60m.csv`
- `model2_less_cluster_centroids.csv`
- `model2_more_cluster_centroids.csv`

Based on current `model2_metrics.json`:

- `<2y` test MAE: `3240.24` (WMAPE `0.2295`, rows `3228`)
- `>2y` test MAE: `1505.73` (WMAPE `0.1592`, rows `5756`)
- raw P10-P90 coverage: `0.9794`
- calibrated P10-P90 coverage: `0.8362`
- interval width scale: `0.3807`
- cold-start first-6-month MAE: `4501.48` (rows `558`)

---

## Run

From repo root:

```bash
python zeta_modelling/model_2/cohort_forecast_pipeline.py
```

Optional arguments:

```bash
python zeta_modelling/model_2/cohort_forecast_pipeline.py \
  --data-dir zeta_modelling/data_2 \
  --out-dir zeta_modelling/data_2/model_2_outputs
```

## Notes

- This pipeline is intentionally cohort-separated end-to-end.
- Cluster IDs are cohort-local (a cluster id in `<2y` does not correspond to same geography as same id in `>2y`).
- Forecast uncertainty here is pragmatic (std-based + calibration), not full quantile modeling.
