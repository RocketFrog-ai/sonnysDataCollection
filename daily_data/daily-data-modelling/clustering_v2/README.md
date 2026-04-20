# Clustering V2 — daily + monthly cohorts, localisation-aware Ridge, multi-horizon projection

This folder is the **self-contained V2** of the site-level clustering /
Ridge / time-series projection pipeline. It supersedes everything under
`clustering/` for the projection use-case while keeping the same
on-disk, portable-JSON model contract so the FastAPI serving layer can
be pointed at it with minimal changes.

## What V2 fixes vs V1

V1 issues found during review:

1. **No daily weather** in the Ridge features for the `>2y` cohort,
   even though `master_more_than-2yrs.csv` carries 19 `daily_weather_*`
   columns. V1 only used annual climate aggregates.
2. **No localisation signal in Ridge** — we clustered sites to localise
   them, but then Ridge scored each site in isolation without any
   feature describing "what do the OTHER sites in this cluster look
   like?". That defeats the purpose of clustering.
3. **Serving was fragile** — it re-read full CSVs on every request and
   depended on joblib/sklearn version pinning. Artifacts weren't fully
   portable.
4. **Time/lag features were being fed with calendar values at serve
   time** while training encoded them as site-relative ages. This
   produced runaway Ridge predictions for new sites. V2's serve path
   leaves these NaN and uses the imputer's train medians.
5. **`<2y` cohort wasn't even part of the serving pipeline** until we
   wired it in; V2 trains a dedicated model for it and the two cohorts
   are returned side-by-side.

V2 additions:

- `daily_weather_*` block added to the `>2y` feature set.
- Per-cluster, **train-only** aggregates (`ctx_wash_count_total_median`,
  `_mean`, `_std`, and medians of competition / retail / gas / weather
  features) joined into every row. This is the "localisation"
  signal V1 was missing.
- Train-only DBSCAN **centroids** saved separately for serve-time
  nearest-cluster assignment (no CSV scan at request time).
- Train-only per-cluster **monthly median wash-count series** saved
  separately and used as the input to Holt-Winters / ARIMA / blend.
- A single `project_site.py` script that takes an address or lat/lon
  and writes `projection_<method>_<tag>.json` + a bar-plot PNG.

## Data used

| file | role | granularity | rows | sites |
|---|---|---|---|---|
| `daily_data/daily-data-modelling/master_more_than-2yrs.csv` | >2y source with all feature engineering (daily weather, competition, retail, gas, annual weather) | daily | 345,804 | 482 |
| `daily_data/daily-data-modelling/more_than-2yrs.csv` | >2y source with DBSCAN 12km labels + lag features | daily | 345,804 | 482 |
| `daily_data/daily-data-modelling/less_than-2yrs-clustering-ready.csv` | <2y source (geocoded + DBSCAN + monthly lags, from `prepare_less_than_2yrs.py`) | monthly panel | 12,907 | 491 |

For `>2y` the two source files are joined on `(site_client_id,
calendar_day)` so we get daily_weather + DBSCAN + lag features in one
frame.

## Splits (strict, no leakage)

| cohort | train | test |
|---|---|---|
| `>2y` (daily) | `calendar_day < 2025-07-01` (≈ 260k rows) | `calendar_day >= 2025-07-01` (≈ 86k rows) |
| `<2y` (monthly panel, `period_index` = month-in-site-operation) | `period_index <= 12` (site year 1, 8.2k rows) | `period_index >= 25` (site year 2, 4.7k rows) |

The `<2y` panel has a gap at `period_index` 13–24 by construction, so
the split is clean site-year-1 vs site-year-2.

## V2 backtest vs V1 (see `results/backtest_summary.json`)

`>2y` (hold-out = H2 2025):

| model | features | MAE | RMSE | R² | **WAPE** |
|---|---|---|---|---|---|
| V1 baseline (wash only) | 31 | 130.7 | 184.8 | 0.554 | 0.426 |
| V1 + daily weather | 41 | 123.5 | 178.5 | 0.584 | 0.402 |
| **V2 final** (+ cluster context) | **59** | **118.2** | **171.4** | **0.617** | **0.385** |

→ **WAPE improves 4.1 points absolute (9.5% relative)** from V1 to V2.

`<2y` (hold-out = site-year 2):

| model | features | MAE | RMSE | R² | **WAPE** |
|---|---|---|---|---|---|
| **V1 baseline** | 12 | 1,278 | 2,802 | 0.783 | **0.132** |
| V2 with ctx (deliberately OFF for <2y) | 12 | same | same | same | same |

The `<2y` cohort has a thin feature schema and its hold-out is
site-year-2 of the SAME sites, so cluster-aggregate ctx features become
near-duplicates of the target. V1 baseline already gets to WAPE 0.13
(R² 0.78) on this cohort, so V2 matches it rather than adding noise
(ctx aggregates are skipped; cluster centroids + monthly series are
still persisted for a unified projection API).

## Artifacts per cohort

```
models/
├── more_than/
│   ├── wash_count_model_12km.portable.json   # feature_order + imputer + scaler + ridge
│   ├── cluster_centroids_12km.json           # train-only centroids for serving
│   ├── cluster_context_12km.json             # train-only per-cluster ctx_* lookup
│   ├── cluster_monthly_series_12km.json      # per-cluster monthly median wash series
│   ├── feature_spec_12km.json                # what to feed at serve time
│   └── training_metrics_12km.json
└── less_than/
    └── (same structure, monthly model)
```

All files are plain JSON — no sklearn or joblib needed at serve time.

## Projection algorithm (in `project_site.py`)

For a NEW site at `(lat, lon)`:

1. Geocode the address if lat/lon not provided (via
   `app.utils.common.get_lat_long`).
2. **Nearest cluster by haversine** against the train-only centroids.
3. Build a Ridge feature vector:
   - `latitude`, `longitude`, `dbscan_cluster_12km` = known
   - `ctx_*` = pulled from the assigned cluster's context row
   - everything else left NaN so the train-median imputer fills in
     "typical site at median age" values
4. Score the portable Ridge → expected wash-count level.
   - `>2y`: per-day → multiplied by 30 to get monthly ballpark
   - `<2y`: already monthly
5. Pull the cluster's monthly median wash-count series (train-only)
   and forecast 24 months via Holt-Winters (seasonal if ≥ 24 months
   of history, non-seasonal trend otherwise), ARIMA(1,1,1), or the
   average of the two (`blend`).
6. **Anchor** the forecast: scale it by
   `ridge_monthly_level / last-6-month cluster mean` so the trajectory
   reflects THIS site's features instead of the cluster average.
7. Aggregate cumulative totals at 6 / 12 / 18 / 24 months.

Both cohorts run in parallel and return in the same payload.

## Scripts

| script | purpose |
|---|---|
| `build_v2.py` | End-to-end: load both cohorts, split, fit train-only centroids + context + monthly series + Ridge, export all portable artifacts. Re-run to refresh models. |
| `backtest_v2.py` | Time-based backtest comparing V1-baseline vs V1+daily-weather vs V2-final on the same hold-out. Writes `results/backtest_summary.json`. |
| `project_site.py` | Serving-ready demo. Reads only the portable JSON artifacts, takes an address or lat/lon, writes JSON + bar-plot PNG to `results/projection_demo/`. |

## Quick usage

```bash
# 1. Build / refresh artifacts for both cohorts
python daily_data/daily-data-modelling/clustering_v2/build_v2.py

# 2. Validate accuracy vs V1 baseline
python daily_data/daily-data-modelling/clustering_v2/backtest_v2.py

# 3. Project a new site (by lat/lon)
python daily_data/daily-data-modelling/clustering_v2/project_site.py \
    --lat 39.99 --lon -75.16 --method blend --out-name demo_philly

# Or by address (requires TOMTOM_API_KEY)
python daily_data/daily-data-modelling/clustering_v2/project_site.py \
    --address "1234 Market St, Philadelphia, PA" --method blend
```

Outputs land in `results/projection_demo/projection_<method>_<tag>.{json,png}`.
