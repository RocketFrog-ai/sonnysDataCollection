# Clustering V2 — approach

**Production pipeline:** everything below lives in **`daily_data/daily-data-modelling/clustering_v2/`** (train, backtest, projection, monthly eval). Do not use the legacy sibling folder **`clustering/`** for V2 train or serve—only for an optional **one-off** rebuild of `less_than-2yrs-clustering-ready.csv` if that input file is missing or stale.

Self-contained pipeline: **DBSCAN cluster assignment → level model (Ridge or quantiles) using train-only cluster context → monthly forecast anchored to that level → optional use of opening-site forecast to inform mature-site continuation.**

---

**Mature vs newbie:** `>2y` and `<2y` are **separate** models and **separate** DBSCAN fits on purpose (different variance / life stage). Cluster ids are **not** comparable across cohorts.

**Cluster ids within a cohort:** Training rows use the same **train-only refit** 12 km DBSCAN labels as the **saved centroids** used at projection time (CSV labels are only a fallback if a site was missing from the refit map).

## Two cohorts (do not mix)

| | **>2y (`more_than`)** | **<2y (`less_than`)** |
|---|------------------------|-------------------------|
| **Grain** | Daily rows | Monthly rows |
| **Sources** | `master_more_than-2yrs.csv` + `more_than-2yrs.csv` (join on site + day) | `less_than-2yrs-clustering-ready.csv` in **`daily-data-modelling/`** (regenerate only if needed via legacy **`../clustering/prepare_less_than_2yrs.py`** — not part of the v2 train path) |
| **Site / market features** | Static site, gas, competitors, retail, annual + **daily** weather, wash lags | Lat/lon, region/state encodings, month indices, **wash-only** lags (`prev_wash_count`, etc.) — no weather or retail on the panel |
| **Cluster `ctx_*`** | Peer **medians** for gas / retail / selected daily weather columns, **plus** peer `wash_count_total` **median, mean, std, min, max** | **Only** peer `wash_count_total` **median, mean, std, min, max** (no fake weather/gas columns) |
| **DBSCAN centroids** | Fit on **>2y train** sites | Fit on **<2y train** sites — **cluster ids are not the same** between cohorts for the same address |
| **Level model** | Ridge + optional quantile GBMs (`models_quantile/more_than/`) | Ridge + optional quantile GBMs (`models_quantile/less_than/`) |
| **Ridge daily → monthly** | Expected day-level wash × 30 for anchoring | Already monthly |

---

## Splits (no leakage)

| Cohort | Train | Test |
|--------|--------|------|
| >2y | `calendar_day < 2025-07-01` | `calendar_day >= 2025-07-01` |
| <2y | `period_index <= 12` | `period_index >= 25` |

**`<2y` — `month_number`, `year_number`, and `period_index` (encoding in the clustering-ready CSV):** The source panel uses **`month_number` 1–12** for **`year_number` 1** (first operational year, e.g. 2024) and **`month_number` 13–24** for **`year_number` 2** (second year, e.g. 2025). Then

`period_index = (year_number - 1) * 12 + month_number`

so **year 1** gives **1–12** and **year 2** gives **25–36** only. Values **13–24 never appear** — not because rows are “missing,” but because **that is how the index is defined** once `month_number` runs 13–24 in the second year. **`calendar_day` in the CSV is synthetic** (anchored from `2024-01` + `(period_index - 1)` months) for modeling; it is not claiming every row is literal wall-calendar 2024/2025. The backtest train/test split (`<= 12` vs `>= 25`) just separates **first-year** vs **second-year** bands under this encoding.

---

## Artifacts

**Ridge / serving JSON** — `models/more_than/` and `models/less_than/`:

- `wash_count_model_12km.portable.json` — feature order, imputer, scaler, Ridge (serve without sklearn).
- `cluster_centroids_12km.json` — train centroids for **nearest-cluster** assignment.
- `cluster_context_12km.json` — per-cluster `ctx_*` lookup (composition differs by cohort, see table above).
- `cluster_monthly_series_12km.json` — train-only monthly series per cluster (forecast input).
- `feature_spec_12km.json`, `training_metrics_12km.json`.

**Quantile models** — `models_quantile/<more_than|less_than>/`:

- `feature_order.json`, `q10.joblib`, `q50.joblib`, `q90.joblib`, `quantile_metrics.json` — requires sklearn at predict time.

**Summaries** — `results/build_summary.json`, `results/backtest_summary.json`, `results/quantile_backtest_summary.json`.

---

## Scripts

| Script | Role |
|--------|------|
| `build_v2.py` | Train both cohorts; write `models/<cohort>/` JSON. |
| `build_quantile_v2.py` | Train q10/q50/q90 pipelines; write `models_quantile/`. |
| `backtest_v2.py` | Ridge backtests → `results/backtest_summary.json`. |
| `project_site.py` | Address or lat/lon → JSON + PNG under `results/projection_demo/`. |
| `project_site_quantile.py` | Same flow using quantile models + band outputs. |
| `eval_accuracy_report.py` | Monthly holdout metrics (Ridge + RF) → `results/monthly_level_holdout_eval.json` (also `monthly_ridge_holdout_eval.json` copy). |
| `eval_ts_arima_vs_holt.py` | One-step TS on cluster median monthly series → `results/ts_arima_vs_holt_pick.json` (**ARIMA** beats Holt-Winters on MAE/WAPE for both cohorts; default `--method` in projection CLIs is **arima**). |

Rebuild after changing upstream CSVs or feature logic (from **`clustering_v2/`**):

```bash
cd daily_data/daily-data-modelling/clustering_v2
python build_v2.py
python build_quantile_v2.py
python backtest_v2.py
python eval_accuracy_report.py
python eval_ts_arima_vs_holt.py
```

**Single command — three-way Ridge PNG + JSON** (default TS = **ARIMA**; nearest cluster even if &gt;20 km):

```bash
python project_site.py --address "7021 Executive Center Dr, Brentwood, TN 37027" --plot-two-way --allow-distant-nearest-cluster --out-name mysite

**Ridge vs RF (2×2)** — after `build_v2.py` (creates `wash_count_model_12km.rf.joblib` per cohort):

`python project_site.py --address "7021 Executive Center Dr, Brentwood, TN 37027" --plot-ridge-rf-four-way --allow-distant-nearest-cluster --out-name mysite`
```

Quantile (same defaults): `python project_site_quantile.py --address "7021 Executive Center Dr, Brentwood, TN 37027" --allow-distant-nearest-cluster --out-name mysite_q`  
Override TS: `--method holt_winters` or `--method blend` if you want to compare.

---

## Greenfield projection flow (`project_site.py`)

1. **Geocode** (or use `--lat` / `--lon`).
2. **Nearest cluster** — Haversine distance to each cohort’s train centroids; pick closest (same logic, **separate** centroid sets per cohort).
3. **Feature vector** — `latitude`, `longitude`, `dbscan_cluster_12km`, and the cluster **context** row: `ctx_*` peer aggregates plus **`local_feature_medians`**: train-only **median of each non-ctx feature within that cluster** (e.g. `prev_wash_count`, statics, weather columns). Missing slots use those first; anything still missing uses the portable model’s **global** training medians.
4. **Level** — Ridge scores the vector → monthly ballpark (`>2y`: daily prediction × 30). Quantile path yields q10 / q50 / q90.
5. **Forecast** — Cluster’s train **monthly** series → **ARIMA (default)**, Holt-Winters, or average **blend** for **24 months ahead**. Default is **ARIMA** after one-step backtest on cluster median tracks (`eval_ts_arima_vs_holt.py` → `results/ts_arima_vs_holt_pick.json`).
6. **Anchor** — Scale forecast so its level matches the site’s predicted monthly level vs recent cluster mean.
7. **Horizons** — Disjoint 6-month sums (6m, 12m, 18m, 24m) on the anchored monthly track.

### Opening → mature handoff (years 1–2 → 3–4)

By default (`use_opening_prefix_for_mature_forecast=True`; disable with `--no-opening-prefix` on CLI):

1. Run **<2y** first → 24 monthly washes (opening phase).
2. For **>2y**, build the TS input as **up to last 72 months of mature cluster history** concatenated with those **24 months**, then forecast the **next** 24 months (continuation). Ridge anchor for >2y still uses the mature cluster tail vs mature Ridge level as before.
3. **`_bridge_mature_monthly_to_opening_last_month`** (default **on** when `<2y` prefix is used) scales the **>2y** monthly track so month **25** lines up with **<2y** month **24** — this fixes the old “Year 4 cliff” when prefix was on. **`--legacy-prefix-no-bridge`** on a **single** run reproduces the old behavior for debugging only. **`--plot-two-way`** compares **no prefix** vs **prefix + bridge** (two PNG panels only; no legacy middle panel).

So the **<2y** trajectory is explicitly reused as **context** for extrapolating the **>2y** view, not ignored.

### Nearest cluster distance cap (`project_site.py` / `project_site_quantile.py`)

Assignment is always the **geographically nearest train centroid**. By default, if that centroid is **more than 20 km** away, the tool returns an **error** instead of numbers (peers are too far for the 12 km DBSCAN design). Pass **`--allow-distant-nearest-cluster`** to force projection anyway; the JSON includes **`distance_cap_relaxed`: true** on each cohort block when used.

---

## Why two cohort bars can look similar

Same address can land in clusters with similar recent wash levels and anchor scales. The **models and `ctx_*` composition still differ** (daily rich vs monthly wash-only).

---

## KPI usage

- Use **q50** (quantile) or Ridge **monthly level** as the primary planning number; **q10–q90** (or Ridge + judgment) as a risk band.
- Training quality: see **`results/backtest_summary.json`** and **`results/quantile_backtest_summary.json`** after running backtests / quantile build.
