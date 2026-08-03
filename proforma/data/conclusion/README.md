# n70 proforma backtest — master dataset

`n70_backtest_dataset.csv` — **one row per site**, the 70 "mature, matched" car-wash sites from
`experiments/old-proforma-analysis/n70-final-considered/`. This is the single common table for the
proforma-vs-model backtest: every **input variable**, all three **forecasters**, and the **actual**
washes, side by side. 70 rows × 80 columns.

## Provenance (how each block was built)

| block | source | notes |
|---|---|---|
| inputs (factors, demographics, traffic) | `old-proforma-combined.csv` | extracted from the 70 proforma Excel files |
| `plateau_loo_location`, `m5in_*`, `coldstart_v15_y*` | `ensemble/results/ensemble_features.csv` | cold-start location model |
| `model5_y*`, `model5_mature` | `ensemble/results/model5_loso.csv` | leave-one-site-out |
| `actual_y*`, `actual_mature_wash`, `actual_nobs_y*` | `proforma/data/panel/main-data-v2-stitched.csv` | **recomputed live** — ground truth |
| `operational_start` | the panel's own `operational_start` | true open date |

**Actuals are the ground truth.** `actual_yN` = mean of `mem_wash_count + ret_wash_count` over the
observed (non-imputed) months in operating-year N, aligned to each site's **true** open date. Partial
years use the observed monthly rate; a year with zero observed months is left blank. Reconciled cell-for-cell
against the panel (0 mismatches).

## Column groups

1. **Keys / context** — `source_file, client_name, client_id, site_id, state, lat, lon, proforma_type,
   operational_start, proforma_assumed_open, open_year`.
   The site key is **`client_id + site_id`** (`site_id` alone collides). `operational_start` is the true
   (panel) open; `proforma_assumed_open` is what the proforma assumed — they differ for 2 sites
   (Seaford, Jupiter Galaxy), where the proforma date was a wrong placeholder.
2. **Proforma factor inputs** — `factor_<name>_choice` (raw, e.g. "2 pay stations") + `factor_<name>_score`
   (weighted) for all 10 factors, plus `cumulative_site_score`.
3. **Demographics** — `demog_*_value` / `demog_*_score`, `cumulative_demographic_score`.
4. **Capacity / traffic / hours** — `traffic_count`, `traffic_incr_y3..5`, `avg_daily_wash_hours`,
   `weekly_hours_operation`.
5. **Model-5 / cold-start inputs** — `plateau_loo_location` (the location-only plateau, Model 5's `lp`
   feature), `m5in_pay/vac/tos` (the 3 proven proforma factor scores Model 5 consumes), `m5in_era_anchor`,
   `m5in_n_era`.
6. **Forecasts (all monthly washes)** —
   `proforma_y1..5` (Excel projection) · `coldstart_v15_y1..5` (location-only) ·
   `model5_y1..5` + `model5_mature` (super-ensemble).
7. **Actuals (ground truth)** — `actual_y1..5`, `actual_mature_wash`.
8. **Coverage** — `actual_nobs_y1..5` = observed months behind each `actual_yN` (12 = full year;
   `<12` = partial, rate-extrapolated; `0`/blank = not yet that old).
9. **Year-5 capacity + tunnel length** —
   `proforma_y5_monthly` (= `proforma_y5`), `proforma_y5_yearly` (×12), `proforma_y5_daily` (×12/365);
   `year5_max_hourly` (the proforma's year-5 **max hourly** volume, joined from
   `proforma/data/ref/old-excel-proforma-data-enriched.csv`);
   **`tunnel_length_ft` = `year5_max_hourly + 20`**, **`tunnel_length_m` = `(year5_max_hourly + 20) × 0.3048`**.
   The metre column reproduces the enriched file's own `tunnel_length_predicted` exactly (0/70 mismatch).
   Note the formula's base is max **hourly**, not yearly.
   (Actual built tunnel length is **not** included — to be added manually. For reference, the enriched file
   has a `tunnel_length_actual` for only 12 of the 70, and it does not track the formula.)
10. **`actuals_suspect`** — source flag (none set among the 70).

## Data-quality caveat — collapsed actuals

Two sites' mature washes collapsed far below their early-year peak (real panel values, not imputed —
a closure / stopped reporting):
- **Splash Car Wash** — peak ~6,968 → mature **41** (proforma ratio ×205). Degenerate; exclude from mature scoring.
- **Rock N Roll Stuart** — peak ~5,888 → mature 2,229 (×5.3).

Filter with e.g. `actual_mature_wash >= 0.4 * d[["actual_y1","actual_y2","actual_y3"]].max(axis=1)` before
scoring if you want to drop collapse cases.

## Quick backtest recipe

```python
import pandas as pd, numpy as np
d = pd.read_csv("proforma/data/conclusion/n70_backtest_dataset.csv")
def mdape(pred, act):
    m = (act>0) & pred.notna(); return np.median(np.abs(pred[m]-act[m])/act[m])*100
for f in ["proforma_y5", "coldstart_v15_y5", "model5_mature"]:
    print(f, round(mdape(d[f], d.actual_mature_wash),1), "% MdAPE")
# proforma_y5 ~60% · coldstart_v15_y5 ~ mid · model5_mature ~30%
```
