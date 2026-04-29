# Model 1 Detailed Documentation (Data 1 + Model 1)

This document is a detailed, implementation-level explanation of the full Model 1 stack under:

- `zeta_modelling/model_1/`
- `zeta_modelling/data_1/`

It covers data inventory, assumptions, feature engineering, modeling stages, uncertainty calibration, business outputs, and limitations.

---

## 1) Scope and objective

Model 1 is a cold-start friendly forecasting system for new car-wash sites. It is designed to:

- forecast monthly wash volume over 3-5 years,
- provide uncertainty ranges (P10/P50/P90),
- derive business metrics (monthly/cumulative profit, break-even month),
- run without requiring historical lag features from the new site itself.

Current production-facing forecaster is implemented in:

- `zeta_modelling/model_1/phase3_advanced_forecast.py`

and persisted via:

- `zeta_modelling/model_1/build_phase3_artifacts.py`
- `zeta_modelling/model_1/phase3_artifacts.joblib`

---

## 2) Data inventory (`zeta_modelling/data_1`)

### 2.1 Raw cohort files

- `less_than-2yrs.csv`
  - ~12,907 rows, 37 columns.
  - Monthly-style lifecycle data for younger sites.
- `more_than-2yrs.csv`
  - ~345,804 rows, 33 columns.
  - Daily records for mature sites (later aggregated to month).

### 2.2 Core prepared panel

- `phase1_final_monthly_2024_2025.csv`
  - ~18,865 rows, 9 columns.
  - Canonical modeling table.
  - Columns:
    - `site_id`, `date`, `age_in_months`, `real_age_months`, `monthly_volume`,
      `latitude`, `longitude`, `cluster_id`, `maturity_bucket`.

### 2.3 Feature/experiment outputs

- `phase2_features_2024_2025.csv` (baseline phase-2 feature table)
- `phase2_features_no_lag_upgrade_2024_2025.csv` (upgraded no-lag features)
- `phase2_metrics_2024_2025.json`
- `phase2_no_lag_upgrade_metrics.json`
- `phase2_deployable_multiplier_metrics.json`
- `phase2_site_profile_multiplier_metrics.json`
- diagnostics:
  - `phase2_error_by_site.csv`
  - `phase2_error_by_cluster.csv`
  - `phase2_error_by_age_phase.csv`
  - `phase2_feature_importance.csv`
  - `phase2_site_profile_multiplier_error_by_age_phase.csv`

### 2.4 Phase-3 outputs

- `phase3_advanced_forecast.csv`
- `phase3_advanced_report.json`
- `phase3_backtest_plots/*`
- legacy/simple phase-3 outputs:
  - `phase3_new_site_forecast_5y.csv`
  - `phase3_forecast_summary.json`

### 2.5 Benchmark alignment output

- `model1_benchmark_volume_sample.csv`
  - Aligned 2025+ test predictions from multiple model variants vs actual.

---

## 3) End-to-end pipeline map (top to bottom)

### Stage A: Dataset standardization

Script:

- `phase1_final_dataset_prep.py`

What it does:

1. Creates stable `site_id`:
   - prefers normalized `site_client_id`,
   - fallback is `latitude_longitude`.
2. Maps `month_number` to strict 2024-2025 dates:
   - `1..12 -> 2024-01..2024-12`
   - `13..24 -> 2025-01..2025-12`
3. `<2y`:
   - uses monthly rows directly,
   - computes `age_in_months` by per-site order.
4. `>2y`:
   - aggregates to monthly `(site_id, date)` totals,
   - sets `age_in_months = cumulative_index + 24`.
5. Harmonizes `real_age_months`, `cluster_id`, maturity bucket.
6. Applies quality gates:
   - no rows after 2025,
   - no duplicate `(site_id, date)`,
   - `real_age_months >= age_in_months`,
   - drops rows missing lat/lon.

Output:

- `phase1_final_monthly_2024_2025.csv`

---

### Stage B: Baseline phase-2 feature and model experiments

Script:

- `phase2_feature_engineering_and_model.py`

Feature blocks:

- time: `month`, `year`, `month_sin`, `month_cos`
- lifecycle: `age_sq`, `log_age`, stage flags
- cluster context: `cluster_month_avg`, `cluster_age_avg`
- global seasonality factor by month
- geography: `latitude`, `longitude`, `lat_lon_interaction`
- optional true lags: `lag_1`, `lag_3`, `rolling_mean_3`

Models:

- `lightgbm_no_lag`
- `lightgbm_with_lag` (benchmark only; not cold-start deployable)

Split:

- train `< 2025-01-01`
- test `>= 2025-01-01`

Key metrics from `phase2_metrics_2024_2025.json`:

- no-lag MAE: `2236.31`, RMSE: `3716.71`
- with-lag MAE: `1466.60`, RMSE: `2679.94`

---

### Stage C: No-lag upgrade with pseudo-lags/trend

Script:

- `phase2_no_lag_upgrade.py`

Adds deployable no-leakage proxies:

- cluster calendar pseudo-lags: `cluster_lag_1`, `cluster_lag_3`
- cluster trend: `cluster_growth_rate`, `cluster_rolling_mean`
- uncertainty proxy: `cluster_std`
- interactions: `age_cluster_interaction`, `age_seasonality_interaction`

Also trains:

- baseline no-lag,
- upgraded no-lag,
- warm model with true lags (benchmark ceiling).

Inner time-respecting tuning in train period:

- train-sub `< 2024-10-01`
- val-sub `>= 2024-10-01 and < 2025-01-01`

Key metrics from `phase2_no_lag_upgrade_metrics.json`:

- baseline no-lag MAE: `2252.40`
- upgraded no-lag MAE: `2146.26`
- warm (true lags) MAE: `1409.79`
- no-lag MAE improvement: `106.14`

---

### Stage D: Site-profile + multiplier-target experiment

Script:

- `phase2_site_profile_multiplier.py`

Core idea:

- derive site behavior profiles from train-only data:
  - `site_avg_volume`, `site_peak`, `site_volatility`, early slope `site_growth`
- cluster these into latent `site_type` (KMeans),
- compare absolute-target vs multiplier-target formulations.

Important details:

- `target_multiplier = monthly_volume / cluster_month_avg`
- evaluates no-lag upgraded multiplier path vs warm lag absolute path.

Key metrics from `phase2_site_profile_multiplier_metrics.json`:

- baseline no-lag absolute MAE: `2391.02`
- upgraded no-lag multiplier MAE: `1882.39`
- warm lag absolute MAE: `2002.40`

This experiment showed a strong gain from multiplier formulation in a no-lag setup.

---

### Stage E: Deployable no-lag multiplier model with inferred site type

Script:

- `phase2_deployable_multiplier.py`

Deployability goals:

- no true site lags at inference,
- infer `site_type` from geospatial context only.

Method:

1. build train-only site labels (`low/mid/high`) by site average train volume,
2. train LightGBM classifier:
   - inputs: `cluster_id`, `latitude`, `longitude`,
   - output: inferred `site_type`,
3. train LightGBM multiplier regressor on final no-lag feature set.

Cold-start simulation:

- hold out entire sites,
- evaluate first 6 months only.

Key metrics from `phase2_deployable_multiplier_metrics.json`:

- full test MAE: `2049.88`, RMSE: `3465.98`
- cold-start early-month MAE: `3125.20` on 441 rows

---

### Stage F: Production forecaster (Phase 3 advanced)

Script:

- `phase3_advanced_forecast.py`

This is the most complete model and current production base for zeta forecasting.

#### F1. Training artifacts

`train_artifacts(df)` trains and stores:

- site-type classifier (KMeans-labeled + LightGBM classifier),
- two quantile model sets (early and main):
  - `p10`, `p50`, `p90`,
- cluster tables:
  - `cluster_month_avg`, `cluster_age_avg`,
  - `cluster_month_std`, `cluster_age_std`,
- cluster centroids.

#### F2. Inference mechanics for a new site

For input `(lat, lon)`:

1. Build synthetic monthly rows for `months` horizon.
2. Find top-3 nearest clusters by centroid (inverse-distance weights).
3. For each candidate cluster:
   - attach cluster features,
   - infer probabilistic site type,
   - score quantile multipliers with early/main models.
4. Blend cluster predictions by top-3 weights.

Final monthly outputs before calibration:

- `p10`, `p50`, `p90`, plus `volume = p50`.

#### F2.1b Mature-year YoY guard (conditional)

After blending, optional **mature-year YoY** adjustment runs only when forecast years from `mature_yoy_start_year` onward have **strictly decreasing** annual P50 totals (each year lower than the previous). If the late tail is flat, up, or mixed, the forecast is left unchanged. When triggered, each affected year’s monthly quantiles are scaled so the annual total sits in `[prior_year_total × (1+min_yoy), prior_year_total × (1+max_yoy)]`. See `apply_mature_yoy_control` in `phase3_advanced_forecast.py`.

#### F2.1 Growth-control layer (new)

To reduce unrealistic long-horizon decline, a control layer can be applied after cluster blending:

- Build blended cluster lifecycle curve from top-3 cluster weights using `cluster_age_avg`.
- Normalize by anchor month (`growth_anchor_month`, default 12).
- Smooth (`growth_smoothing_window`, default 3).
- Enforce business shape:
  - non-decreasing through ramp end (`growth_ramp_end_month`, default 30),
  - plateau afterward.
- Convert to correction factor and apply to forecast quantiles (`p10/p50/p90`).

This is a post-model structural control (not a model replacement) and is configurable via:

- `enable_growth_control`
- `growth_anchor_month`
- `growth_ramp_end_month`
- `growth_smoothing_window`

#### F3. Uncertainty calibration

Function:

- `apply_global_uncertainty_calibration(...)`

Approach:

- read backtest coverage (`current_coverage`) from report,
- compute single global scale:
  - `scale = target_coverage / current_coverage`,
- widen/narrow intervals:
  - `low = p50 - scale * (p50 - p10)`
  - `high = p50 + scale * (p90 - p50)`

Current report values (`phase3_advanced_report.json`):

- raw backtest coverage: `0.454`
- target coverage: `0.80`
- applied scale: `1.7621`

#### F4. Business outputs

Function:

- `break_even_from_costs(...)`

Calculations:

- `monthly_profit = volume * margin_per_wash - fixed_monthly_cost`
- month 1 includes extra `ramp_up_cost`
- `cumulative_profit = cumsum(monthly_profit)`
- `break_even_month` is first month where cumulative profit becomes positive.

#### F5. Reported advanced backtest

From `phase3_advanced_report.json`:

- MAE: `2089.77`
- RMSE: `3604.58`
- P10-P90 coverage (pre-calibration): `0.454`

---

## 4) Feature catalog (consolidated)

### 4.1 Temporal and seasonal

- `month`, `year`, `month_sin`, `month_cos`
- global seasonal factor (`seasonality_factor` in phase-2 variants)

### 4.2 Lifecycle

- `age_in_months`, `real_age_months`
- `age_sq`, `log_age`, `age_saturation`
- stage flags: `is_early`, `is_growth`, `is_mature`

### 4.3 Cluster context

- `cluster_month_avg`
- `cluster_age_avg`
- `cluster_std` / `cluster_month_std` / `cluster_age_std`
- `cluster_growth_curve`, `cluster_growth_rate`, `cluster_rolling_mean`
- pseudo-lag features (`cluster_lag_1`, `cluster_lag_3`) in upgraded phase-2

### 4.4 Site behavior and type

- `site_avg_volume`, `site_peak`, `site_volatility`, `site_growth`
- `site_type` (KMeans-derived label; inferred by classifier for deployable paths)

### 4.5 Geography and interactions

- `latitude`, `longitude`
- `lat_lon_interaction`
- other interactions in no-lag upgrade:
  - `age_cluster_interaction`,
  - `age_seasonality_interaction`

### 4.6 Warm-start-only (not deployable for brand new sites)

- `lag_1`, `lag_3`, `rolling_mean_3`

---

## 5) Explicit modeling assumptions

1. **Date window assumption**
   - modeling/evaluation window constrained to 2024-2025.

2. **Cohort alignment assumption**
   - mature (`>2y`) monthly age index starts at `+24`.

3. **Spatial transfer assumption**
   - new-site behavior can be inferred from nearest cluster centroids.

4. **Multiplier stability assumption**
   - ratio to cluster baseline (`monthly_volume / cluster_month_avg`) is easier to generalize than absolute volume.

5. **Site-type latent structure assumption**
   - site behavior can be represented via latent classes and inferred from cluster+geo context.

6. **Global uncertainty calibration assumption**
   - one global scaling factor is sufficient to move interval coverage toward target.

7. **No monotonic growth assumption**
   - model does not enforce strictly increasing year-over-year volume; mature decline/plateau is allowed if cluster context suggests it.

---

## 6) Train/test and leakage posture

- Primary split is time-based:
  - train `< 2025-01-01`,
  - test `>= 2025-01-01`.
- Several scripts include warm lag benchmarks; these are useful upper-bound references but not deployable for true cold-start.
- Site behavior profiling and site-type labeling are designed to use train-period information.

---

## 7) Artifact and file-level guide

### 7.1 Scripts (`model_1`)

- `phase1_final_dataset_prep.py`: canonical monthly panel build + validation.
- `phase2_feature_engineering_and_model.py`: baseline no-lag/with-lag experiments.
- `phase2_no_lag_upgrade.py`: pseudo-lag no-leak upgrade + tuning + diagnostics.
- `phase2_site_profile_multiplier.py`: site-profile + multiplier experiments.
- `phase2_deployable_multiplier.py`: deployable no-lag multiplier with inferred site type.
- `phase3_forecast_engine.py`: earlier phase-3 engine (kept for historical comparison).
- `phase3_advanced_forecast.py`: current advanced forecaster (quantile + calibration + business outputs).
- `build_phase3_artifacts.py`: trains/saves `phase3_artifacts.joblib`.
- `benchmarking.py`: aggregates benchmark metrics across phase artifacts.
- `export_benchmark_volume_sample.py`: aligned test prediction export for model comparison.

### 7.2 Key data files (`data_1`)

- canonical train/test panel: `phase1_final_monthly_2024_2025.csv`
- phase-2 metrics:
  - `phase2_metrics_2024_2025.json`
  - `phase2_no_lag_upgrade_metrics.json`
  - `phase2_site_profile_multiplier_metrics.json`
  - `phase2_deployable_multiplier_metrics.json`
- phase-3 report: `phase3_advanced_report.json`
- benchmark alignment: `model1_benchmark_volume_sample.csv`

---

## 8) Known limitations and practical cautions

1. **Centroid distance metric**
   - nearest-cluster logic in some model-1 paths uses Euclidean distance in lat/lon degree space, not haversine km.

2. **Single global interval scaling**
   - calibration is global; not cluster-conditional or horizon-conditional.

3. **Potential long-horizon behavior drift**
   - year 4/5 can decline if cluster mature trend implies normalization; no business-rule floor is imposed.

4. **Static feature regime**
   - macro changes (new competition, demographics shifts) are not explicitly modeled.

5. **Train window scope**
   - only 2024-2025 behavior directly informs fit; extrapolation beyond that relies on learned structure.

---

## 9) Suggested next hardening steps

1. Add optional business constraints at post-processing:
   - monotone floor after year N,
   - grow-to-plateau policy curves.
2. Use geodesic/haversine nearest-cluster assignment consistently.
3. Move from global to conditional calibration (by age bucket and/or cluster density).
4. Add rolling-origin backtests beyond single split.
5. Track data drift and per-cluster error over time.

---

## 10) Current headline metrics snapshot

From checked `data_1` artifacts:

- phase2 no-lag baseline MAE: `2236.31`
- phase2 with-lag MAE: `1466.60` (benchmark upper bound)
- phase2 upgraded no-lag MAE: `2146.26`
- deployable multiplier full-test MAE: `2049.88`
- deployable cold-start early-month MAE: `3125.20`
- phase3 advanced P50 backtest MAE: `2089.77`
- phase3 raw P10-P90 coverage: `0.454` (scaled to target 0.80 at inference)

This is the top-to-bottom state of Model 1 using current `model_1` code and `data_1` artifacts.
