# Car wash count prediction (modelling2)

- **Rows:** 482 sites
- **Target:** `current_count` (car wash volume)
- **Features:** 25 numeric columns; **`tunnel_count` excluded**
- **Best model (highest hold-out R², with overfit gap as tie-breaker):** `extra_trees`
- **Alternative:** `random_forest` — nearly the same R² (0.846) with **lower MAPE** (17.65% vs 18.21%).

## Hold-out test performance (20%, random_state=42)

- **R² (variance explained):** 0.8485 → **84.8%** of variance explained
- **MAPE:** 18.21% mean absolute percentage error
- **Accuracy-style (100 − MAPE):** 81.8%
- **MAE:** 16383 (on a mean count of 123213)

## Bootstrap 95% confidence intervals (test set, resampled rows)

- **R²:** [0.6942, 0.9687]
- **MAPE %:** [13.92, 23.39]

## All models (sorted by test R²)

| model | CV R² mean±std | train R² | test R² | test MAPE % | overfit gap (train−CV R²) |
|---|---:|---:|---:|---:|---:|
| extra_trees | 0.945±0.008 | 0.982 | 0.848 | 18.21 | 0.037 |
| random_forest | 0.941±0.007 | 0.980 | 0.846 | 17.65 | 0.039 |
| knn_k5_distance | 0.626±0.038 | 1.000 | 0.505 | 74.84 | 0.374 |
| knn_k5_uniform | 0.626±0.040 | 0.746 | 0.500 | 75.23 | 0.121 |
| knn_k3_distance | 0.554±0.039 | 1.000 | 0.497 | 72.64 | 0.446 |
| knn_k3_uniform | 0.554±0.038 | 0.773 | 0.493 | 73.04 | 0.218 |
| knn_k7_distance | 0.633±0.038 | 1.000 | 0.480 | 80.74 | 0.367 |
| knn_k7_uniform | 0.628±0.042 | 0.728 | 0.472 | 81.50 | 0.100 |
| knn_k11_distance | 0.612±0.036 | 1.000 | 0.463 | 85.32 | 0.388 |
| knn_k11_uniform | 0.602±0.038 | 0.702 | 0.454 | 86.24 | 0.100 |

## Column reference (all `finale.csv` columns)

| column | dtype | missing | role |
|---|---|---:|---|
| `_match_type` | object | 0 | excluded (id/text) |
| `age_on_30_sep_25` | float64 | 0 | feature (numeric) |
| `carwash_type_encoded` | int64 | 0 | feature (numeric) |
| `city` | object | 0 | excluded (id/text) |
| `client_id` | object | 0 | excluded (id/text) |
| `competitor_1_distance_miles` | float64 | 16 | feature (numeric) |
| `competitor_1_google_rating` | float64 | 66 | feature (numeric) |
| `competitor_1_rating_count` | float64 | 66 | feature (numeric) |
| `competitors_count_4miles` | float64 | 5 | feature (numeric) |
| `costco_enc` | float64 | 0 | feature (numeric) |
| `count_food_joints_0_5miles (0.5 mile)` | float64 | 5 | feature (numeric) |
| `current_count` | float64 | 0 | target |
| `distance_nearest_costco(5 mile)` | float64 | 312 | feature (numeric) |
| `distance_nearest_target (5 mile)` | float64 | 154 | feature (numeric) |
| `distance_nearest_walmart(5 mile)` | float64 | 47 | feature (numeric) |
| `location_id` | int64 | 0 | excluded (id/text) |
| `nearest_gas_station_distance_miles` | float64 | 0 | feature (numeric) |
| `nearest_gas_station_rating` | float64 | 0 | feature (numeric) |
| `nearest_gas_station_rating_count` | int64 | 0 | feature (numeric) |
| `official_website` | object | 12 | excluded (id/text) |
| `other_grocery_count_1mile` | float64 | 99 | feature (numeric) |
| `previous_count` | float64 | 0 | feature (numeric) |
| `primary_carwash_type` | object | 0 | excluded (id/text) |
| `region` | object | 2 | excluded (id/text) |
| `region_enc` | int64 | 0 | feature (numeric) |
| `site_client_id` | int64 | 0 | excluded (id/text) |
| `state` | object | 0 | excluded (id/text) |
| `state_enc` | int64 | 0 | feature (numeric) |
| `street` | object | 0 | excluded (id/text) |
| `tunnel_count` | int64 | 0 | excluded (tunnel) |
| `weather_avg_daily_max_windspeed_ms` | float64 | 5 | feature (numeric) |
| `weather_days_below_freezing` | float64 | 5 | feature (numeric) |
| `weather_days_pleasant_temp` | float64 | 5 | feature (numeric) |
| `weather_rainy_days` | float64 | 5 | feature (numeric) |
| `weather_total_precipitation_mm` | float64 | 5 | feature (numeric) |
| `weather_total_snowfall_cm` | float64 | 5 | feature (numeric) |
| `weather_total_sunshine_hours` | float64 | 5 | feature (numeric) |
| `zip` | int64 | 0 | excluded (id/text) |

EDA figures: `eda_output/`

Run: `python carwash_count_prediction.py` from this folder (uses `MPLCONFIGDIR` under `modelling2/.mplconfig`).