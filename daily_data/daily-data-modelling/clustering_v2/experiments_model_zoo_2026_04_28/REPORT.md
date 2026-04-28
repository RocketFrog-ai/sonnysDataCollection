
## 2) Mature sites: level model only, train 2024 → test 2025

**Runner:** `misc_runners/run_time_split_gt_2024_2025.py`  
**Output root:** `outputs_mature_level_train2024_test2025/<UTC_run_id>/`

**Approach:** **>2y** daily rows: calendar **2024** train, **2025** test. Refit clusters on 2024 train sites; predict **daily** wash with Ridge / RF / XGB; aggregate to **monthly site totals** in 2025 and score (WAPE, R², etc.). This is **not** the full anchor+cluster-TS projection—only the **level** head.

**Latest leaderboard (monthly totals, 2025):**

| Rank | Model | WAPE | R² | % within 20% rel. error |
|-----:|-------|-----:|---:|--------------------------:|
| 1 | **rf** | **0.1419** | 0.8906 | 71.9 |
| 2 | xgb | 0.1928 | 0.8737 | 52.46 |
| 3 | ridge | 0.3122 | 0.6835 | 35.07 |

---

## 3) End-to-end temporal split, train 2024 → test 2025 (greenfield-like features)

**Runner:** `misc_runners/run_time_split_end_to_end_2024_2025.py`  
**Output root:** `outputs_end_to_end_train2024_test2025/<UTC_run_id>/`

**Approach:** **>2y:** 2024 daily train → 2025 monthly test; **\<2y:** 2024 monthly train → 2025 monthly test (second-year `calendar_day` in the CSV shifted back one year so months align). Features: geo + cluster id + **2024-only** `ctx_*` peer context. Pipeline: nearest train centroid → XGB/RF/Ridge anchor → cluster monthly median series → **ARIMA / Holt / blend** → scale to anchor vs cluster recent level. Score = **WAPE(\<2y 2025) + WAPE(>2y 2025)** (lower better).

**Latest leaderboard (full table, sorted by score):**

| rank | variant | score | lt_wape | gt_wape | lt_r² | gt_r² |
|---:|---|---:|---:|---:|---:|---:|
| 1 | xgb_arima | **0.8343** | 0.4785 | 0.3558 | -0.146 | 0.4971 |
| 2 | xgb_blend | 0.8444 | 0.469 | 0.3754 | -0.1286 | 0.3891 |
| 3 | xgb_holt | 0.8704 | 0.4722 | 0.3982 | -0.145 | 0.3006 |
| 4 | rf_arima | 0.905 | 0.5036 | 0.4014 | -0.2774 | 0.3521 |
| 5 | rf_blend | 0.9083 | 0.4982 | 0.4101 | -0.2868 | 0.2695 |
| 6 | rf_holt | 0.9333 | 0.5047 | 0.4286 | -0.3333 | 0.1952 |
| 7 | ridge_arima | 0.9993 | 0.5163 | 0.483 | -0.269 | 0.1832 |
| 8 | ridge_blend | 1.0005 | 0.5081 | 0.4924 | -0.2502 | 0.1119 |
| 9 | ridge_holt | 1.0196 | 0.5099 | 0.5097 | -0.2621 | 0.0437 |

**Prophet vs ARIMA (same setup, XGB level head):** With **Prophet installed** in `conda` and **shortlisted** variants only (`--only xgb_arima,xgb_prophet`), run `20260428T015437Z` under `outputs_end_to_end_train2024_test2025/`. On this split, **cluster-scale Prophet forecasts (12 months of training history per cluster, then scaled to each site’s anchor) are much worse than ARIMA**:

| variant | score (lt_wape + gt_wape) | gt_wape | lt_wape | gt_r² | lt_r² |
|---:|---|---:|---:|---:|---:|
| **xgb_arima** | **0.8348** | 0.3558 | 0.479 | 0.4972 | -0.148 |
| xgb_prophet | 2.7435 | 1.6697 | 1.0738 | −11.0 | −8.69 |

So **Prophet is not a win** here (short, noisy cluster medians; one-year history; then global scale). **`meta`** (pick ARIMA / Holt / blend / Prophet by rolling MAE) is implemented in the runner but **very slow**—use `--only xgb_meta` (or include it in a comma list) and expect a long job or run overnight.

**Conda env:** ensure `pip install xgboost` in `sonnysDataCollection` so the script does not fail after the Prophet precompute phase.

---


## 5) Temporal TS: per-cluster vs mean curves (XGB + ARIMA, 2024→2025)

**Runner:** `misc_runners/run_temporal_avg_ts_2024_2025_and_4y_outlook.py`  
**Output root:** `outputs_temporal_ts_mean_curves/<UTC_run_id>/`

**Approach:** Same data split as (3). Fixes level model to **XGB + ARIMA** and compares where the monthly **shape** comes from: per-cluster medians vs **mean across clusters** per cohort vs **mean of the two cohort mean curves** (shared shape). Also writes an **illustrative** 48-month ARIMA extension on those mean curves (cluster-median units, **not** validated against future site data).

**2025 test (combined WAPE = lt + gt):**

| TS shape | >2y WAPE | \<2y WAPE | Sum |
|----------|---------:|----------:|----:|
| Per-cluster | 0.3558 | 0.4785 | **0.8343** |
| Cohort mean (separate curves) | 0.3505 | 0.4435 | 0.7940 |
| Cross-cohort mean (one shared curve) | 0.3210 | 0.4714 | 0.7924 |

---

## How to rerun

```bash
cd daily_data/daily-data-modelling/clustering_v2/experiments_model_zoo_2026_04_28/misc_runners
python run_site_holdout_zoo.py
# or: run_time_split_gt_2024_2025.py, run_time_split_end_to_end_2024_2025.py,
#     run_lt_time_split_2024_6m6m.py, run_temporal_avg_ts_2024_2025_and_4y_outlook.py
```

Artifacts land under the parent folder in the `outputs_*` directories listed above. Update **this** `REPORT.md` if you change code or need to record a new best run.
