# Zeta Modelling - End-to-End Approach and Final Results

This document explains the full modeling approach implemented in `zeta_modelling/model_1/`, step by step, and summarizes the final metrics/evaluation results.

---

## 1) Objective

Build a deployable forecasting system for new car wash locations:

- Predict monthly wash volume over a long horizon (3 to 5 years)
- Provide uncertainty ranges (not just point estimates)
- Convert forecast into business outputs (profit, break-even)
- Keep the system usable for cold-start sites (no site history)

---

## 2) Data Preparation (Phase 1)

Implemented in:

- `phase1_final_dataset_prep.py`

### 2.1 Source datasets

- `<2yrs` dataset: already monthly panel
- `>2yrs` dataset: daily records

### 2.2 Site identity

- Primary key: `site_client_id`
- Fallback: `latitude_longitude` string
- Standardized ID formatting to avoid `15` vs `15.0` duplicate issues

### 2.3 Date logic (strict 2024-2025)

Date created from `month_number` with explicit mapping:

- `1..12` -> `2024-01..2024-12`
- `13..24` -> `2025-01..2025-12`

This prevents leakage to fake 2026 dates.

### 2.4 Aggregation and lifecycle

- `>2yrs`: daily -> monthly by `(site_id, date)` sum of `wash_count_total`
- `<2yrs`: already monthly
- `age_in_months`:
  - `<2yrs`: cumulative month index per site
  - `>2yrs`: cumulative index + 24 offset
- `real_age_months`:
  - `>2yrs`: from `age_on_30_sep_25 * 12`
  - `<2yrs`: approximated by `age_in_months`

### 2.5 Hard quality fixes

- Removed duplicates on `(site_id, date)`
- Enforced `real_age_months >= age_in_months`
- Dropped rows with missing lat/lon
- Validated:
  - no duplicates
  - no post-2025 leakage
  - age consistency

Final consolidated dataset used downstream:

- `zeta_modelling/data/phase1_final_monthly_2024_2025.csv`

---

## 3) Feature Engineering and Baselines (Phase 2)

Implemented in:

- `phase2_feature_engineering_and_model.py`
- `phase2_no_lag_upgrade.py`
- `phase2_site_profile_multiplier.py`
- `phase2_deployable_multiplier.py`

### 3.1 Core feature groups

- Time: `month`, `year`, `month_sin`, `month_cos`
- Lifecycle: `age_in_months`, `real_age_months`, `age_sq`, `log_age`, phase flags
- Cluster context:
  - `cluster_month_avg`
  - `cluster_age_avg`
  - cluster trend/variance features
- Location: `latitude`, `longitude`, interaction term
- Optional warm-start features: `lag_1`, `lag_3`, rolling mean

### 3.2 Modeling strategy evolution

1. Baseline no-lag LightGBM
2. Upgraded no-lag with pseudo-lags and cluster trend features
3. Warm model with true lags (benchmark only)
4. Site-profile + multiplier-target experiments
5. Deployable no-leakage model with inferred `site_type`

---

## 4) Final Forecast Engine (Phase 3)

Implemented in:

- `phase3_advanced_forecast.py`
- Artifacts build script: `build_phase3_artifacts.py`
- UI app: `zeta_modelling/streamlit_app.py`

### 4.1 Key design decisions

- **Target formulation:** multiplier target
  - `target_multiplier = monthly_volume / cluster_month_avg`
- **Uncertainty:** quantile LightGBM (`P10`, `P50`, `P90`)
- **Early-stage specialization:** separate models for `age <= 6` and `age > 6`
- **Cluster assignment:** top-3 nearest clusters with inverse-distance weights
- **Site-type uncertainty:** probabilistic site-type blending
- **Calibration:** global one-shot interval calibration
  - `scale = 0.80 / current_coverage`
  - output adjusted as:
    - `low = p50 - scale * (p50 - p10)`
    - `high = p50 + scale * (p90 - p50)`
- **Business outputs:** monthly profit, cumulative profit, break-even month

### 4.2 Final inference output

Per month output includes:

- `volume` (P50)
- `low`, `high` (calibrated interval)
- cumulative volume/profit
- confidence label from interval width

---

## 5) Final Metrics and Evaluation

Numbers below are from:

- `zeta_modelling/data/phase3_advanced_report.json`
- `zeta_modelling/data/phase2_no_lag_upgrade_metrics.json`

### 5.1 Phase 2 benchmark metrics (2025 test split)

- **Baseline no-lag:** MAE `2252.40`, RMSE `3772.10`
- **Upgraded no-lag:** MAE `2146.26`, RMSE `3670.88`
- **Warm (true lags):** MAE `1409.79`, RMSE `2558.81`
- **Gap reduction (baseline -> upgraded no-lag):** MAE improvement `106.14`

### 5.2 Final advanced backtest metrics

- MAE: `2089.77`
- RMSE: `3604.58`
- P10-P90 empirical coverage (before calibration): `0.454`
- Global calibration scale applied: `1.7621` (to target 0.80 coverage)

### 5.3 Example 5-year forecast output (sample site)

- Expected volume (5y): `779,864.84`
- Range (5y): `552,177.02 - 1,107,479.08`
- Total profit: `-30,540.63` (with margin=4, fixed/ramp assumptions)
- Break-even: not reached
- Confidence: low
- Risk: moderate variability cluster

---

## 6) Artifacts Produced

- Data:
  - `zeta_modelling/data/phase1_final_monthly_2024_2025.csv`
  - `zeta_modelling/data/phase3_advanced_forecast.csv`
  - `zeta_modelling/data/phase3_advanced_report.json`
- Model artifact:
  - `zeta_modelling/model_1/phase3_artifacts.joblib`
- Backtest visuals:
  - `zeta_modelling/data/phase3_backtest_plots/`

---

## 7) How to Run

### 7.1 Build artifacts (one-time / after retrain)

```bash
python zeta_modelling/model_1/build_phase3_artifacts.py
```

### 7.2 Run Streamlit decision app

```bash
streamlit run zeta_modelling/streamlit_app.py
```

---

## 8) Current Status

`model_1` is now a full forecasting + decision pipeline:

- data preparation
- deployable cold-start modeling
- calibrated uncertainty
- business interpretation
- usable UI for planning decisions
