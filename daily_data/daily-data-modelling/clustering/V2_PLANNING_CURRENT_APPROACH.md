# Version 2 Planning: Current (V1) Modelling Approach

This document summarizes the **current V1 pipeline** used for both:
- `more_than_2yrs` (daily history, clustered dataset)
- `less_than_2yrs` (monthly-style history, prepared then clustered)

It explains the flow from raw data to forecasting/projection.

---

## 1) Goal

For a **new input site** (address or lat/lon), return:
- nearest cluster assignment (`12km` or `18km`)
- cluster historical operating range (min/p10/median/p90/max)
- model-based wash prediction (Ridge portable model)
- forecast projection for next **6, 12, 18, 24 months**
- chart-ready bars (cumulative projection at each horizon)

---

## 2) Data Inputs (Current V1)

### A) More than 2 years cohort
- File: `daily_data/daily-data-modelling/more_than-2yrs.csv`
- Granularity: daily site rows
- Already includes geo + cluster columns:
  - `latitude`, `longitude`
  - `dbscan_cluster_12km`, `dbscan_cluster_18km`

### B) Less than 2 years cohort
- Raw file: `daily_data/daily-data-modelling/less_than-2yrs.csv`
- Prepared file used by pipeline:
  - `daily_data/daily-data-modelling/less_than-2yrs-clustering-ready.csv`

Preparation includes:
1. geocode missing site coordinates
2. monthly-aware time features
3. lag/rolling features
4. DBSCAN cluster assignment (`12km`, `18km`)

Script: `daily_data/daily-data-modelling/clustering/prepare_less_than_2yrs.py`

---

## 3) Cluster + Model Training (V1)

### A) Cluster quality and range checks
- `cluster_range_check.py`
- `cluster_assignment_quality.py`

These produce:
- expected cluster ranges (train-only percentiles)
- assignment reliability stats (centroid distance + radius gate)

### B) Ridge model training (portable inference format)
- Script: `cluster_model_eval.py`
- Features: numeric weather/site/time/geo + cluster id
- Train/Test split: time-based 80/20
- Output portable models:
  - `clustering/models/wash_count_model_12km.portable.json`
  - `clustering/models/wash_count_model_18km.portable.json`
  - plus `<2y` versions under `clustering/models/less_than_2yrs/`

Inference helper:
- `app/server/cluster_portable_model.py`

---

## 4) Runtime API Flow for New Site (V1)

Endpoint:
- `/v1/cluster/standalone/projection`

Request fields:
- `address` OR (`latitude`, `longitude`)
- `radius`: `12km` or `18km`
- `method`: `holt_winters` or `arima` or `blend`

Runtime steps per cohort:
1. Load cohort assets (portable model + train centroids + train range stats)
2. Geocode address if needed
3. Assign nearest cluster via haversine distance to centroid
4. Apply radius gate (unassigned if too far)
5. Run ridge prediction for assigned cluster
6. Build cluster monthly historical series
   - uses cluster-month site-level median behavior
7. Forecast next 24 months using selected method
8. Return cumulative projections at 6/12/18/24 and bar-ready payload

---

## 5) Forecasting Method in V1

Current methods:
- `holt_winters`
- `arima`
- `blend` (average of both)

Default currently used in demo: `blend`.

Backtest notes used for method checks:
- overall ARIMA is strong in many scenarios
- for some cohort/radius combinations, blend performs better
- therefore method remains configurable in request

---

## 6) Example from Current Output (12km + blend)

Reference response:
- `clustering/results/projection_demo/projection_response_12km_blend_20260420-033655.json`

Input:
- Address: `5360 Laurel Springs Pkwy, Suwanee, GA 30024`
- Radius: `12km`
- Method: `blend`

### A) `more_than_2yrs`
- assigned cluster: `0`
- distance to centroid: `3.70 km`
- ridge prediction: `275.41`
- cumulative projection:
  - 6m: `51,523`
  - 12m: `99,457`
  - 18m: `145,756`
  - 24m: `188,475`

### B) `less_than_2yrs`
- assigned cluster: `2`
- distance to centroid: `6.24 km`
- ridge prediction: `7,756.96`
- cumulative projection:
  - 6m: `34,512`
  - 12m: `69,460`
  - 18m: `104,797`
  - 24m: `140,522`

Chart:
- `clustering/results/projection_demo/projection_bars_12km_blend_20260420-033655.png`

---

## 7) V1 Practical Summary

Current V1 is:
- dual-cohort (`>2y` and `<2y`) capable
- cluster-aware (12km / 18km)
- trained-range aware (historical percentile block)
- portable-model inference safe
- forecast-method configurable
- chart-ready for quick business review

This is the baseline to evolve in Version 2.
