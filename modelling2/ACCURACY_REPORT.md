# Accuracy report (modelling2) — plain language

**Dataset:** `finale.csv` — **482** sites.

## Why this felt different from `app/modelling`

| What you saw | What it measures |
|---|---|
| **`app/modelling`** (`ds.prediction`) | **Classification:** each site gets a **tier** Q1–Q4 from `current_count`. **Accuracy** = % of sites where the **predicted tier equals the true tier** (5-fold CV). Docs cite ~**63.5%** exact and ~**98%** within one tier when **tunnel_count + effective_capacity** are in the model. |
| **Earlier modelling2 regression** | **Regression:** predict the **number** of washes. **R²** = variance explained (not “% correct”). **MAPE** = average **percent** error. There is no single “accuracy %” unless we define one (e.g. % predictions within 20% of true). |

So: **production “accuracy” is tier hit rate; regression “accuracy” must be defined** (we use median APE + “within 20% of actual” below).

---

## 1) Quantile / tier classification (like production)

Tiers use the same style as production default: percentile splits **[19, 31, 31, 19]** on `current_count`.

### 1a) **Modelling2 policy — NO `tunnel_count`, NO `effective_capacity`**

| Model | Exact tier % | Within ±1 tier % |
|---|---:|---:|
| knn_classifier | 34.02 | 80.91 |
| random_forest_classifier | 37.55 | 81.95 |
| extra_trees_classifier | 39.0 | 78.63 |

**1a-note — KNN “top 5 similar sites” only** (same features as 1a; no `previous_count`): find the **5 nearest** sites in imputed + scaled feature space; predicted tier = **vote** among those 5 sites’ true tiers (`uniform` = simple majority; `distance` = closer neighbors count more).

| Model | Exact tier % | Within ±1 tier % |
|---|---:|---:|
| knn_k5_uniform | 33.61 | 80.08 |
| knn_k5_distance | 32.99 | 80.08 |

### 1b) **Reference only — WITH `tunnel_count` + `effective_capacity`** (matches production feature idea)

| Model | Exact tier % | Within ±1 tier % |
|---|---:|---:|
| knn_classifier | 54.98 | 94.19 |
| random_forest_classifier | 69.29 | 99.38 |
| extra_trees_classifier | 68.26 | 98.96 |

### 1c) **No tunnel, but you know last year** — feature `log1p(previous_count)` added

| Model | Exact tier % | Within ±1 tier % |
|---|---:|---:|
| knn_classifier | 44.4 | 91.08 |
| random_forest_classifier | 84.85 | 100.0 |
| extra_trees_classifier | 65.35 | 93.57 |

---

## 2) Regression — predict **current** wash count (number, not tier)

### 2a) Site features only (no tunnel / EC)

| Model | R² OOF (×100) | Median APE % | MAPE trim worst 5% % | Within 20% of actual % |
|---|---:|---:|---:|---:|
| knn_regressor | 1.75 | 48.34 | 80.37 | 23.86 |
| random_forest_regressor | 19.45 | 43.21 | 70.55 | 22.2 |
| extra_trees_regressor | 16.07 | 44.32 | 68.84 | 22.61 |

### 2b) Same + `log1p(previous_count)` (previous year known)

| Model | R² OOF (×100) | Median APE % | MAPE trim worst 5% % | Within 20% of actual % |
|---|---:|---:|---:|---:|
| knn_regressor | 35.23 | 42.8 | 63.24 | 26.76 |
| random_forest_regressor | 92.53 | 10.12 | 11.54 | 78.84 |
| extra_trees_regressor | 92.21 | 10.24 | 12.95 | 75.73 |

**Naive baseline:** always guess **current = previous** → MAPE **12.69%**, R² **91.61**, within 20% of actual **83.61%**.

---

## 3) YoY-style targets (predict **ratio** `current ÷ previous` from site features, no tunnel)

| Model | R² OOF (×100) | Median APE % | Within 20% of actual % |
|---|---:|---:|---:|
| knn_regressor | -4.8 | 10.03 | 82.57 |
| random_forest_regressor | 6.94 | 8.44 | 85.27 |
| extra_trees_regressor | 9.21 | 8.57 | 86.31 |

*(Delta `current − previous` is in `benchmark_results.json`; percentage error on raw deltas is misleading without scaling.)*

On raw counts, **mean** MAPE is often inflated by a few catastrophic OOF rows; prefer **median APE** or **trimmed MAPE** (table in §2).

---

## 4) Which features mattered most? (tree importances)

RandomForest / ExtraTrees only (KNN has no importances). Trained on all rows; importances are Gini-based split contributions — useful for ranking drivers, not causal effects.

### 4a) **Quantile (Q1–Q4)** — with `log1p_previous_count`

**Random forest classifier — top features**

| Rank | Feature | Importance % (of model total) |
|---:|---|---:|
| 1 | `log1p_previous_count` | 45.929 |
| 2 | `age_on_30_sep_25` | 4.773 |
| 3 | `competitor_1_rating_count` | 2.76 |
| 4 | `weather_days_pleasant_temp` | 2.688 |
| 5 | `is_express` | 2.582 |
| 6 | `weather_total_precipitation_mm` | 2.557 |
| 7 | `distance_nearest_walmart(5 mile)` | 2.426 |
| 8 | `weather_drive_score` | 2.382 |
| 9 | `weather_avg_daily_max_windspeed_ms` | 2.382 |
| 10 | `nearest_gas_station_distance_miles` | 2.083 |
| 11 | `competition_quality` | 2.078 |
| 12 | `distance_nearest_target (5 mile)` | 2.063 |

**Extra trees classifier — top features**

| Rank | Feature | Importance % (of model total) |
|---:|---|---:|
| 1 | `log1p_previous_count` | 35.706 |
| 2 | `is_express` | 12.708 |
| 3 | `age_on_30_sep_25` | 8.333 |
| 4 | `distance_nearest_walmart(5 mile)` | 2.828 |
| 5 | `state_enc` | 2.505 |
| 6 | `weather_total_precipitation_mm` | 2.388 |
| 7 | `weather_days_pleasant_temp` | 2.152 |
| 8 | `region_enc` | 2.07 |
| 9 | `competitors_count_4miles` | 2.069 |
| 10 | `costco_enc` | 2.06 |
| 11 | `weather_days_below_freezing` | 2.056 |
| 12 | `weather_drive_score` | 2.039 |

### 4b) **Quantile** — site features only (no previous year)

**Random forest**

| Rank | Feature | Importance % (of model total) |
|---:|---|---:|
| 1 | `age_on_30_sep_25` | 7.766 |
| 2 | `distance_nearest_walmart(5 mile)` | 4.916 |
| 3 | `competitor_1_rating_count` | 4.823 |
| 4 | `weather_total_precipitation_mm` | 4.815 |
| 5 | `weather_days_pleasant_temp` | 4.663 |
| 6 | `weather_avg_daily_max_windspeed_ms` | 4.433 |
| 7 | `is_express` | 4.405 |
| 8 | `competition_quality` | 4.002 |
| 9 | `nearest_gas_station_distance_miles` | 3.974 |
| 10 | `gas_station_draw` | 3.917 |
| 11 | `retail_proximity` | 3.849 |
| 12 | `distance_nearest_target (5 mile)` | 3.822 |

### 4c) **Regression** (predict `current_count`) — with `log1p_previous_count`

**Random forest regressor**

| Rank | Feature | Importance % (of model total) |
|---:|---|---:|
| 1 | `log1p_previous_count` | 96.122 |
| 2 | `weather_avg_daily_max_windspeed_ms` | 0.511 |
| 3 | `distance_nearest_walmart(5 mile)` | 0.403 |
| 4 | `weather_total_snowfall_cm` | 0.326 |
| 5 | `age_on_30_sep_25` | 0.269 |
| 6 | `competitor_1_rating_count` | 0.205 |
| 7 | `weather_rainy_days` | 0.202 |
| 8 | `competitor_1_distance_miles` | 0.2 |
| 9 | `count_food_joints_0_5miles (0.5 mile)` | 0.166 |
| 10 | `nearest_gas_station_distance_miles` | 0.158 |
| 11 | `weather_total_sunshine_hours` | 0.133 |
| 12 | `weather_total_precipitation_mm` | 0.123 |

**Extra trees regressor**

| Rank | Feature | Importance % (of model total) |
|---:|---|---:|
| 1 | `log1p_previous_count` | 84.295 |
| 2 | `is_express` | 7.176 |
| 3 | `age_on_30_sep_25` | 2.596 |
| 4 | `state_enc` | 0.41 |
| 5 | `count_food_joints_0_5miles (0.5 mile)` | 0.408 |
| 6 | `nearest_gas_station_rating` | 0.407 |
| 7 | `distance_nearest_walmart(5 mile)` | 0.318 |
| 8 | `competitors_count_4miles` | 0.304 |
| 9 | `weather_avg_daily_max_windspeed_ms` | 0.3 |
| 10 | `nearest_gas_station_rating_count` | 0.266 |
| 11 | `distance_nearest_target (5 mile)` | 0.264 |
| 12 | `weather_total_precipitation_mm` | 0.262 |

### 4d) **Regression** — site features only (no previous year)

**Random forest regressor**

| Rank | Feature | Importance % (of model total) |
|---:|---|---:|
| 1 | `age_on_30_sep_25` | 14.291 |
| 2 | `is_express` | 8.122 |
| 3 | `gas_station_draw` | 6.425 |
| 4 | `weather_avg_daily_max_windspeed_ms` | 5.586 |
| 5 | `weather_total_precipitation_mm` | 5.198 |
| 6 | `competitor_1_rating_count` | 4.874 |
| 7 | `nearest_gas_station_rating_count` | 3.94 |
| 8 | `distance_nearest_walmart(5 mile)` | 3.935 |
| 9 | `nearest_gas_station_distance_miles` | 3.898 |
| 10 | `distance_nearest_target (5 mile)` | 3.774 |
| 11 | `competitor_1_distance_miles` | 3.69 |
| 12 | `retail_proximity` | 3.223 |

Full ranked lists: key `feature_importances` in `benchmark_results.json`.

---

## Bottom line (copy-paste)

- **Tier “accuracy” without tunnel:** best exact tier match ≈ **39.0%** (ExtraTrees in this run).
- **Tier “accuracy” with tunnel+EC (reference):** best exact ≈ **69.29%** (aligns with production ~63–69% depending on exact pipeline).
- **Count prediction (median APE, site features only):** best ≈ **43.21%**; **within 20% of true count** on ≈ **23.86%** of sites.
- **Count prediction with last year in the model:** best median APE ≈ **10.12%**; **within 20%** on ≈ **78.84%** of sites.

Full numbers: `benchmark_results.json`.
