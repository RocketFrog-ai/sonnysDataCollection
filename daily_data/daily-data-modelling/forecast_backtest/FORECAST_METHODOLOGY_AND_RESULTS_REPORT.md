# Car Wash Forecasting: One Complete Methodology and Results Report

## 1) What this report is

This is a single combined report of everything we ran:
- the **18-month train -> next 6-month test** backtest
- the **walk-forward time-series test** (`6->6`, `12->6`, `18->6`)
- model details, parameters, and why each method was used
- clear interpretation in simple terms (with examples)

Main goal: help forecast wash counts at the **site level**.

---

## 2) Data used

- File: `daily_data/daily-data-modelling/master_daily_with_site_metadata.csv`
- Time range in file: `2024-01-01` to `2025-12-31` (24 months)
- Rows used after cleaning: `344,348`
- Sites: `480`
- Target to predict: `wash_count_total`

Each row is one site on one day (`site_client_id` x `calendar_day`).

---

## 3) What "site / city / zip / region model" means (simple)

You asked about this, so here is the plain explanation:

- **Site model**: forecast each site directly from that site's own history.
- **City model**: forecast total demand for a city first, then split to sites in that city.
- **Zip model**: forecast total demand for a zip code first, then split to sites in that zip.
- **Region model**: forecast total demand for a region first, then split to sites in that region.

How split is done:
- We use each site's historical share inside city/zip/region.
- We use day-of-week shares first (for weekly behavior).
- If not available, we fall back to overall share in that group.

Simple example:
- Suppose zip 12345 forecast for next month is 100,000 washes.
- Site A historically contributes 30% of that zip, Site B 70%.
- Then zip-level allocation gives:
  - Site A: 30,000
  - Site B: 70,000

This is why `zip_arima_site_alloc` means:
- ARIMA forecast at zip level,
- then allocated back down to each site.

---

## 4) Methodologies we ran

## A) Forecast backtest (single fixed split)

This was your original request pattern:
- Train on first 18 months: `2024-01-01` to `2025-06-30`
- Predict next 6 months: `2025-07-01` to `2025-12-31`

Also used a validation window for stacking/calibration:
- Train on 2024 -> predict `2025-01-01` to `2025-06-30`

Purpose:
- simulate "train now, forecast next 6 months" for one realistic test window

## B) Walk-forward (expanding windows)

This was your later suggestion (good time-series practice):
- `6m train -> next 6m test` (Jan-Jun 2024 -> Jul-Dec 2024)
- `12m train -> next 6m test` (Jan-Dec 2024 -> Jan-Jun 2025)
- `18m train -> next 6m test` (Jan 2024-Jun 2025 -> Jul-Dec 2025)

Purpose:
- check performance across multiple periods, not just one split
- understand stability and drift

---

## 5) Models and exact parameters used

## 5.1 ARIMA
- Library: `statsmodels.tsa.arima.model.ARIMA`
- Order: `(1,0,1)`
- `enforce_stationarity=False`
- `enforce_invertibility=False`
- fit iterations cap: `maxiter=80`

Used at:
- site, city, zip, region levels

Why:
- strong baseline for trend + autocorrelation

## 5.2 Holt-Winters (Exponential Smoothing)
- Library: `statsmodels.tsa.api.ExponentialSmoothing`
- trend: `add`
- seasonal: `add`
- seasonal period: `7` (weekly seasonality)
- initialization: `estimated`

Used at:
- site, city, zip, region levels

Why:
- demand has weekly pattern (weekday/weekend behavior)

## 5.3 Ensembles

### simple average ensemble
- `simple_avg_ensemble = average of 8 model signals`
- 8 signals:
  - site ARIMA, site Holt-Winters
  - city ARIMA/HW (site allocated)
  - zip ARIMA/HW (site allocated)
  - region ARIMA/HW (site allocated)

Why:
- simple robust combination; low overfitting risk

### linear stacked ensemble
- model: `LinearRegression(positive=True)`
- trained on validation predictions

Why:
- learn weights automatically from validation data

### boosted stacked ensemble with exogenous inputs
- model: `GradientBoostingRegressor`
- parameters:
  - `loss="squared_error"`
  - `max_depth=6`
  - `learning_rate=0.05`
  - `n_estimators=300`
  - `min_samples_leaf=50`
  - `subsample=0.8`
  - `random_state=42`
- features used:
  - base forecast signals
  - calendar features (month, day-of-week, weekend, month start/end)
  - weather/competition/retail/site metadata features

Why:
- test if richer non-linear model improves accuracy

### post-calibration model
- model: simple linear regression mapping
- fit on validation: `actual ~ simple_avg_ensemble`
- applied to test as `simple_avg_calibrated`

Why:
- test whether bias-correction improves accuracy

---

## 6) Metrics (simple meaning)

- **MAE**: average absolute mistake (in washes).  
  Example: MAE = 100 means on average off by ~100 washes.

- **RMSE**: like MAE but punishes big misses more.  
  If RMSE is much higher than MAE, some big-error days exist.

- **MAPE**: average percentage error.  
  Can look too large when actual values are small.

- **WAPE**: total absolute error divided by total actual volume.  
  Best business metric here for volume forecasting.

---

## 7) Results from fixed 18m -> 6m backtest

## 7.1 Site-day (daily per site)
Best model:
- `simple_avg_ensemble`
- WAPE: `30.42%`
- MAE: `93.40`
- RMSE: `170.93`

Meaning:
- daily site prediction is usable but noisy

## 7.2 Site-month (monthly total per site, using simple_avg_ensemble)
- WAPE: `18.51%`
- mean abs error per site-month: `1,691.69` washes
- median abs error per site-month: `839.41` washes
- median abs percent error per site-month: `12.83%`

Meaning:
- monthly per-site forecasts are more stable than daily per-site

## 7.3 Site-6month total (6-month total per site, using simple_avg_ensemble)
- WAPE: `14.38%`
- mean abs error per site over full 6 months: `7,875.70` washes
- median abs error per site over full 6 months: `3,845.05` washes
- median abs percent error per site over full 6 months: `9.81%`

Error spread across sites (6-month totals):
- `26.25%` of sites within `<=5%`
- `50.63%` of sites within `<=10%`
- `73.75%` of sites within `<=20%`
- `26.25%` of sites above `20%`

Meaning:
- for many sites, 6-month totals are reasonably close
- some sites still have large misses (usually harder, irregular sites)

## 7.4 Overall 6-month total bias (all sites combined, simple_avg_ensemble)
- Actual: `26,291,840`
- Predicted: `26,647,661.71`
- Difference: `+355,821.71` (`+1.35%`)

Meaning:
- total 6-month business volume is quite close
- aggregation cancels part of site-level errors

## 7.5 Monthly direction (all sites combined, simple_avg_ensemble)
- Jul: `+1.76%`
- Aug: `+1.25%`
- Sep: `-0.53%`
- Oct: `-5.00%`
- Nov: `+3.40%`
- Dec: `+8.28%`

Meaning:
- October was underpredicted
- December was overpredicted

---

## 8) Results from walk-forward (6->6, 12->6, 18->6)

## 8.1 Site-day WAPE (`simple_avg_ensemble`)
- 6->6: `29.62%`
- 12->6: `35.74%`
- 18->6: `30.42%`

Best model per fold:
- 6->6: `simple_avg_ensemble`
- 12->6: `region_arima_site_alloc`
- 18->6: `simple_avg_ensemble`

## 8.2 Site-month WAPE (best model each fold)
- 6->6: `18.41%` (`zip_arima_site_alloc`)
- 12->6: `17.76%` (`zip_arima_site_alloc`)
- 18->6: `18.46%` (`zip_arima_site_alloc`)

## 8.3 Site-6month WAPE (best model each fold)
- 6->6: `12.27%` (`zip_arima_site_alloc`)
- 12->6: `13.53%` (`zip_arima_site_alloc`)
- 18->6: `14.06%` (`zip_arima_site_alloc`)

Meaning:
- walk-forward gives a more realistic "across-time" performance view
- more training history does not always reduce error in every future period
- for monthly and 6-month site totals, `zip_arima_site_alloc` is consistently strongest

---

## 9) Which model should you use?

If business question is:

- **Daily per-site next 6 months** -> use `simple_avg_ensemble`
- **Monthly per-site totals** -> use `zip_arima_site_alloc`
- **6-month per-site total** -> use `zip_arima_site_alloc`

Practical recommendation:
- use walk-forward as regular evaluation protocol
- retrain with latest actuals each cycle
- track WAPE trend by fold/month, not just one split result

---

## 10) Easy end-to-end example (non-technical)

Suppose site X actual monthly totals are:
- Jul: 10,000
- Aug: 11,000
- Sep: 9,500
- Oct: 12,000
- Nov: 10,500
- Dec: 9,000
- 6-month actual total: 62,000

Model predicts:
- Jul: 10,300
- Aug: 10,900
- Sep: 9,400
- Oct: 11,200
- Nov: 10,700
- Dec: 9,800
- 6-month predicted total: 62,300

Interpretation:
- Monthly errors exist (some up, some down)
- 6-month total is close (+300 only)
- This is exactly why monthly/daily can look noisy while 6-month total can still be good

---

## 11) Files to keep and use

Primary report:
- `daily_data/daily-data-modelling/FORECAST_METHODOLOGY_AND_RESULTS_REPORT.md`

Core result files (each folder holds its script plus outputs):
- `daily_data/daily-data-modelling/forecast_backtest/run_forecast_backtest.py`
- `daily_data/daily-data-modelling/forecast_backtest/metrics_summary.csv`
- `daily_data/daily-data-modelling/forecast_backtest/monthly_actual_vs_predicted.csv`
- `daily_data/daily-data-modelling/forecast_backtest/site_level_summary_simple_avg.json`
- `daily_data/daily-data-modelling/walkforward/run_walkforward_6m_backtest.py`
- `daily_data/daily-data-modelling/walkforward/walkforward_best_model_by_fold_scope.csv`
- `daily_data/daily-data-modelling/walkforward/walkforward_simple_avg_wape_trend.csv`
- `daily_data/daily-data-modelling/walkforward/walkforward_summary.json`

