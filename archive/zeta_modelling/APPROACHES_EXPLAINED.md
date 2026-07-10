# Zeta modelling: two approaches

**Model 1** merges young and mature sites into **one monthly panel** and ships the quantile + calibration stack used for PnL-style projections. **Model 2** keeps **`<2y` and `>2y` data separate** end-to-end, trains separate models, then **stitches** a 60-month new-site curve.

---

## Terms (what the models are)

| Idea | Plain meaning | Why it shows up here |
|------|----------------|----------------------|
| **Ridge regression** | Linear prediction from features, with an **L2 penalty** so correlated features do not blow up coefficients. | **Not** the main learner in this zeta stack. Other pipelines in the wider repo sometimes use Ridge for simple daily level models; here the production path is **LightGBM**. |
| **Random forest** | Many trees on random row/feature subsamples; prediction = **average** over trees. | Only as an optional **Streamlit benchmark** next to linear regression and LightGBM—not the saved Phase 3 forecaster. |
| **LightGBM** | Gradient-boosted trees; good for **nonlinear** effects, **categorical** columns (e.g. cluster), and **quantile** objectives. | Core regressor for volumes/multipliers and, in places, **site-type** classification. |
| **Lag vs no lag** | **True lags**: this site’s own prior-month volumes. **Pseudo-lags**: lagged **cluster-average** volume over time (peers, not the target site). | New sites have **no** own history → true lags are invalid or leaky for greenfield. Deployed models use **no true lags**; pseudo-lags add time structure safely. |
| **Quantile (P10–P50–P90)** | Separate models trained so predictions approximate chosen **quantiles** of the target. | Model 1 uses them for **uncertainty bands** around the central track. |
| **Holt–Winters** | Smoothing with **trend + seasonal** components on a monthly series. | Model 2: per **mature** cluster median volume series, to stabilize the **late** segment of the 60-month forecast when blended with LightGBM. |

---

## Seasonality and calendar features (how time enters the model)

These recur across phases:

- **`month`** (1–12): discrete calendar month.
- **`month_sin`, `month_cos`**: \(\sin(2\pi m/12)\), \(\cos(2\pi m/12)\) from month \(m\). They map months onto a **circle** so the learner sees January next to December (no false “December ≫ January” ordering that raw integers imply).
- **`seasonality_factor`** (Phase 2 only): for each calendar month, **(global mean volume in that month across all training rows) / (global mean volume overall)**. It is a **multiplicative** “typical uplift vs annual average” for that month.

Model 1 Phase 3 also passes **`month`** together with **`month_sin` / `month_cos`**, so the model gets both a smooth cycle and a direct month index where useful.

Model 2’s multiplier feature lists use **only** `month_sin` and `month_cos` (no raw `month` column in the trained list), so seasonality there is **purely cyclic**.

---

## Approach 1 — Model 1 (one combined monthly panel)

### Goal

Single table: early-life and mature cohorts **combined**, with **`age_in_months`** aligned (mature source gets a **+24** month offset). Forecast monthly volume with **quantile uncertainty** and calibration suitable for planning.

### Data prep (high level)

Monthly grain per site; stable site IDs; calendar anchored to 2024–2025; dedupe and age checks; bad coordinates dropped.

### Phase 2 — feature sets used in the benchmark comparison

**Baseline no-lag** (direct volume target, LightGBM):

- **Lifecycle:** `age_in_months`, `real_age_months`, `age_sq`, `log_age`, `is_early`, `is_growth`, `is_mature`
- **Cluster:** `cluster_id`, `cluster_month_avg`, `cluster_age_avg`
- **Seasonality / calendar:** `month`, `month_sin`, `month_cos`, `seasonality_factor`
- **Location:** `latitude`, `longitude`, `lat_lon_interaction`
- **Cohort encoding:** `maturity_bucket`

**Upgraded no-lag** (everything above **plus** peer time structure—still **no** this-site lag):

- `cluster_lag_1`, `cluster_lag_3` — lagged **cluster** mean volume (peer trajectory, not the site)
- `cluster_rolling_mean` — short rolling mean of that cluster mean series
- `cluster_growth_rate` — month-to-month **pct change** of cluster mean volume
- `cluster_std` — dispersion of volumes within **cluster × calendar month**
- `age_cluster_interaction` — `age_in_months` × numeric cluster id
- `age_seasonality_interaction` — `age_in_months` × `seasonality_factor`

**Warm / leakage benchmark** (not for greenfield deploy):

- All upgraded features **plus** true site lags: `lag_1`, `lag_3`, `rolling_mean_3` (prior **this site** volume only).

### Phase 3 — features in the shipped multiplier / quantile models

Categoricals (`cluster_id`, `maturity_bucket`, `site_type`) are **integer-encoded** for training; the column names in the matrix include **`cluster_id_code`**, **`maturity_bucket_code`**, **`site_type_code`**.

**Numeric / calendar:**

- `age_in_months`, `real_age_months`
- `month`, `month_sin`, `month_cos`
- `latitude`, `longitude`, `lat_lon_interaction`

**Lifecycle shape:**

- `age_sq`, `log_age`
- `is_early`, `is_growth`, `is_mature` (≤6 mo, 7–18, >18)
- `age_saturation` — \(\tanh(\text{age}/12)\): curve **saturates** as age grows
- `growth_velocity` — `cluster_age_avg / (age_in_months + 1)` (peer curve vs age)

**Cluster context (from training panel groupbys):**

- `cluster_month_avg` — mean volume for **(cluster, calendar month)**
- `cluster_age_avg` — mean volume for **(cluster, age_in_months)**
- `cluster_growth_curve` — month-on-month change along the **cluster age curve**
- `distance_from_peak` — `cluster_age_avg - cluster_month_avg` (lifecycle vs seasonal peer norm)

**Target for training:** `monthly_volume / cluster_month_avg` (multiplier), then re-scale to washes.

**Other Phase 3 mechanics (not extra columns in that list):** separate LightGBM bundles for **age ≤ 6** vs **> 6**; quantiles including P10/P50/P90; **top-3 clusters** by distance at inference; **site-type** blending from a classifier trained on cluster + geo + site stats.

### Results (simple reading)

| Setup | MAE | RMSE | Meaning |
|-------|-----|------|---------|
| No lag (baseline) | **2252** | **3772** | Cold-start-safe; rougher. |
| No lag (upgraded + cluster pseudo-lags / interactions) | **2146** | **3671** | Better without this-site history. |
| With **true** site lags | **1410** | **2559** | Upper bound on accuracy—**not** valid for new sites. |

**Phase 3 backtest (held-out 2025):** MAE **~2090**, RMSE **~3605**; raw P10–P90 coverage **~0.45** (too narrow); one-shot calibration widens bands toward **~80%** coverage (scale **~1.76**).

**Illustrative single-site 5y example** (not a global metric): central total **~780k** washes, band roughly **~552k–1.1M** depending on calibration and scenario.

**Train / test:** before vs from **2025-01-01**.

---

## Approach 2 — Model 2 (`<2y` vs `>2y` separated, then stitched)

### Goal

**60 months** for a greenfield site: months **1–24** from the **early** cohort model path; **25–60** from the **mature** cohort model **plus** Holt–Winters on mature cluster medians, **blended** so the handoff at month 24 is smooth.

### Data (high level)

- **Early cohort:** monthly panel; ages from period / cumulative index.
- **Mature cohort:** daily aggregated to monthly; age index **+24** so it continues after the two-year window.

**DBSCAN** (12 km, 5 min samples) is fit **separately** per cohort—cluster labels are **not** comparable across cohorts.

### Features built in code (before column filter)

**Shared constructs** (both cohorts): `month`, `month_sin`, `month_cos`; `age_sq`, `log_age`; `is_early`, `is_growth`, `is_mature`; `cluster_age_avg`; `cluster_growth_rate` (pct change of cluster monthly mean along time); `lat_lon_interaction`; `age_saturation` (\(\tanh(\text{age}/12)\)); `growth_velocity`; `target_multiplier` = volume / `cluster_month_avg`.

**`<2y` cluster × month block:** `cluster_month_avg`, `cluster_month_std`, `cluster_month_mean`, `cluster_month_median`, `cluster_month_p25`, `cluster_month_p75` — distribution of peer volumes for **that cluster and calendar month**.

**`>2y` cluster × month block:** `cluster_daily_avg`, `cluster_std`, `cluster_month_avg`, `cluster_month_median`, `cluster_month_p25`, `cluster_month_p75` (mature panel uses the naming from daily-aggregated monthly stats).

**`site_type`:** from KMeans on each site’s mean / max / std volume (train period), then filled; a **LightGBM classifier** predicts type probabilities from cluster, lat/lon, and cluster density / average volume for **inference blending** (top‑k types) on the early path.

### Columns actually fed to LightGBM (multiplier models)

**`<2y`:** `age_in_months`, `age_sq`, `log_age`, `age_saturation`, `growth_velocity`, `is_early`, `is_growth`, `is_mature`, `cluster_id`, `cluster_month_avg`, `cluster_month_std`, `cluster_month_mean`, `cluster_month_median`, `cluster_month_p25`, `cluster_month_p75`, `cluster_age_avg`, `cluster_growth_rate`, `latitude`, `longitude`, `lat_lon_interaction`, `month_sin`, `month_cos`, `site_type`.

**`>2y`:** same core + lifecycle + `month_sin` / `month_cos` / `site_type`, but swaps in mature-specific cluster stats: `cluster_daily_avg`, `cluster_std`, `cluster_month_avg`, `cluster_month_median`, `cluster_month_p25`, `cluster_month_p75`, `cluster_age_avg`, `cluster_growth_rate`, etc. (no `cluster_month_mean` in this list—only columns present after the join survive).

**Models:** two LightGBMs on **`<2y`** (age ≤6 vs >6), one on **`>2y`**; categorical columns **`cluster_id`** and **`site_type`**.

### Results

| Slice | MAE | WMAPE | Plain English |
|-------|-----|-------|----------------|
| `<2y` test | **3240** | **~0.23** | Noisier ramp phase. |
| `>2y` test | **1506** | **~0.16** | More stable. |
| Cold holdout (new sites, **first 6 months**) | **4501** | — | Hardest. |
| Intervals | raw P10–P90 **~98%** → calibrated **~84%** | | Bands tightened (width scale **~0.38**) toward an **80%** target. |

**Train / test:** same **2025-01-01** cut.

---

## One-glance comparison

| | **Model 1** | **Model 2** |
|--|-------------|-------------|
| Cohorts | **Combined** monthly panel | **Split** `<2y` / `>2y` + stitch |
| Seasonality in final learner | `month` + `month_sin` / `month_cos` | `month_sin` / `month_cos` only (in multiplier inputs) |
| Uncertainty | **Quantile** LightGBM + global calibration | **Std-based** P10/P90 + calibration |
| Horizon | Multi-year forecast + PnL integration | Explicit **60-month** design + Holt–Winters on mature clusters |

---

## Feature dictionary (one line each)

Names appear in different stages; meaning is the same unless noted.

- **`age_cluster_interaction`** — Product of site age in months and a numeric encoding of cluster id, letting the model learn **age effects that differ by market cluster** (Phase 2 upgraded only).
- **`age_in_months`** — How many **months since open** this row represents (mature cohorts may use an offset so lifelines align with young sites).
- **`age_saturation`** — \(\tanh(\text{age}/12)\): a **bounded** transform of age so extra months matter less once the site is older.
- **`age_seasonality_interaction`** — Product of age and **`seasonality_factor`**, so **ramp shape can differ by calendar month** (Phase 2 upgraded only).
- **`age_sq`** — Age in months **squared**, allowing a **curved** (nonlinear) lifecycle instead of a straight line in age.
- **`cluster_age_avg`** — For this row’s **cluster** and **age_in_months**, the **mean monthly volume** among peer sites (typical lifecycle level for “sites like yours at this age”).
- **`cluster_avg_volume`** — Used for the **site-type classifier**: typically **mean monthly volume across rows in that cluster** (training joins on actual volumes; inference can use the mean of **`cluster_month_avg`** in-cluster).
- **`cluster_daily_avg`** — In the **mature (`>2y`)** panel, the **mean** of the monthly series at **(cluster, calendar month)** under a name that reflects daily-originated aggregation (Model 2).
- **`cluster_density`** — **Count of distinct sites** (or row count where coded that way) in the **same spatial cluster**—a coarse “how crowded is this market” signal for **site-type** prediction.
- **`cluster_growth_curve`** — **Month-to-month change** along the cluster’s **age curve** (`cluster_age_avg`), capturing whether peers are on an up- or down-sloped part of the ramp (Model 1 Phase 3).
- **`cluster_growth_rate`** — **Percent change** from one month to the next in the cluster’s **average volume series over time** (peer **momentum**, not the target site).
- **`cluster_id`** — **Spatial / peer group** label from DBSCAN (sites within ~12 km grouped together); **not comparable** across Model 2’s separate `<2y` vs `>2y` fits.
- **`cluster_id_code`**, **`maturity_bucket_code`**, **`site_type_code`** — **Integer encodings** of the corresponding categorical columns so gradient-boosted trees consume them as numeric inputs (Model 1 Phase 3).
- **`cluster_lag_1`** — **Previous month’s** cluster-mean volume (lag on the **peer** aggregate time series, not this site’s own lag).
- **`cluster_lag_3`** — **Three-month-smoothed** lag variant of the cluster mean series (still **peer-only**).
- **`cluster_month_avg`** — **Mean** site volume in this **cluster** for this **calendar month** (seasonal peer norm).
- **`cluster_month_mean`** — Same as **`cluster_month_avg`** where both exist (Model 2 `<2y` block keeps both names from the aggregation dict).
- **`cluster_month_median`** — **Median** peer volume for **(cluster, month)**—robust to outliers vs the mean.
- **`cluster_month_p25`**, **`cluster_month_p75`** — **25th and 75th percentiles** of peer volumes for **(cluster, month)**—spread of the local pack.
- **`cluster_month_std`** — **Standard deviation** of peer volumes for **(cluster, month)** in the **early** cohort (Model 2); used for uncertainty scaling as well as features.
- **`cluster_rolling_mean`** — Short **rolling average** of the cluster’s **mean volume over time** (smoothed peer baseline).
- **`cluster_std`** — **Dispersion** of volumes in **(cluster, month)**; in Phase 2 upgraded it comes from the early-life feature builder; in Model 2 **`>2y`** it is the monthly **`std`** from the same groupby as other cluster-month stats.
- **`distance_from_peak`** — **`cluster_age_avg` minus `cluster_month_avg`** for the row: compares **lifecycle peer level** to **seasonal peer level** (are peers “hot” for the age but “cold” for the month, or the reverse).
- **`growth_velocity`** — **`cluster_age_avg / (age_in_months + 1)`**: peer typical volume **per unit age**, emphasizing **early steepness** of the ramp.
- **`is_early`** — **1** if age ≤ 6 months, else **0** (brand-new ramp phase).
- **`is_growth`** — **1** if age between **7 and 18** months, else **0** (middle ramp).
- **`is_mature`** — **1** if age **> 18** months, else **0** (post-ramp operation).
- **`lag_1`**, **`lag_3`**, **`rolling_mean_3`** — **This site’s** own prior-month volume, three months back, and a **3-month rolling mean** of past own volumes (**benchmark / warm model only**—not cold-start safe).
- **`lat_lon_interaction`** — **Latitude × longitude**, a single nonlinear **geo fingerprint** (interaction is not “physical distance” but helps trees carve space).
- **`latitude`**, **`longitude`** — Site coordinates (WGS84-style as in the panel).
- **`log_age`** — **log(1 + age_in_months)**, compressing large ages so the model treats **early months** more finely.
- **`maturity_bucket`** — **Categorical** label for which **source cohort / life segment** the row came from after merging datasets (Model 1 Phase 2 / encoding in Phase 3).
- **`month`** — Calendar **month index 1–12** (discrete seasonality).
- **`month_cos`**, **`month_sin`** — **Cosine and sine** of \(2\pi \times \text{month}/12\) so December and January are **neighbors** on a yearly cycle.
- **`real_age_months`** — **Calendar-based** site age where available (e.g. from reported age), else aligned to **`age_in_months`** in the prep rules (Model 1 Phase 2).
- **`seasonality_factor`** — **Ratio** of the **global average volume in this calendar month** to the **global overall average**—how “big” that month tends to be vs the year (multiplicative seasonality).
- **`site_type`** — **Discrete bucket** (string) summarizing **typical size/volatility** of the site from KMeans on historical **mean / max / std** volume; fed as categorical to Model 2 multipliers.
- **`site_avg_volume`**, **`site_peak`**, **`site_std`** — Per-site **mean, max, and std** of monthly volume over training—used to **define** KMeans **site_type** labels, not always fed as regressor inputs in Phase 3.

**Targets (not input features in the usual sense):**

- **`monthly_volume`** — **Actual** washes in that month (what models try to predict, or the numerator of the multiplier).
- **`target_multiplier`** — **`monthly_volume / cluster_month_avg`** (small epsilon may be added in code): “**this site vs typical peer in the same cluster and month**.”
