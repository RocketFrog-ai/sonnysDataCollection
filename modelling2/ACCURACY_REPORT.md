# Accuracy report — clustering V2 (greenfield wash projection)

All content below refers only to:

**`daily_data/daily-data-modelling/clustering_v2/`**

Figures are from  
**`daily_data/daily-data-modelling/clustering_v2/results/monthly_level_holdout_eval.json`**  
(re-run **`python eval_accuracy_report.py`** from that directory for fresh numbers).

---

## Holdout level models (Ridge vs RandomForest)

**What is evaluated:** models are fit on **train** only; scores are on **held-out** rows. **>2y:** each row = one **site × calendar month** in test; actual = sum of **daily** washes; prediction = sum of **daily** level-model outputs for that month. **<2y:** each row = one **site × month** (already monthly); train = first-year months, test = second-year months.

### Main metrics

| Cohort | Model | n | WAPE | R² | MAE (washes / month) |
|--------|--------|---:|-----:|---:|---------------------:|
| **>2y** monthly aggregate | Ridge | 2,888 | 0.329 | 0.679 | 3,004 |
| **>2y** monthly aggregate | RandomForest | 2,888 | **0.142** | **0.884** | **1,293** |
| **<2y** monthly panel | Ridge | 4,689 | 0.140 | 0.778 | 1,357 |
| **<2y** monthly panel | RandomForest | 4,689 | **0.137** | 0.773 | **1,333** |

### “How close in percent?” (same rows as above)

| Cohort | Model | Within ±10% of actual | Within ±15% | Median abs % error |
|--------|--------|------------------------:|--------------:|-------------------:|
| >2y | Ridge | 18.0% | 27.7% | 29.3% |
| >2y | RandomForest | **45.8%** | **61.9%** | **11.1%** |
| <2y | Ridge | 50.0% | 64.3% | 10.2% |
| <2y | RandomForest | **52.1%** | **67.8%** | **9.4%** |

### What “% within ±10% of actual” means

For **each** evaluation row: **absolute percent error**  
`|prediction − actual| ÷ actual` (only where `actual > 0`).  
**“Within ±10%”** means that error is **≤ 0.10** (prediction between 90% and 110% of actual). The table value is the **share of rows** that pass (higher = better).

### Pred vs peer train range (“inside cluster min–max %”)

Same **field names** in JSON for both cohorts; the **band** is defined at **monthly wash total** grain.

| Cohort | Model | Pred inside peer train min–max | Actual inside same band |
|--------|--------|--------------------------------:|--------------------------:|
| **>2y** (monthly sum from daily) | Ridge | **88.2%** | **95.0%** |
| **>2y** | RandomForest | **99.5%** | **95.0%** |
| **<2y** (one row = one month) | Ridge | **97.8%** | **97.0%** |
| **<2y** | RandomForest | **98.6%** | **97.0%** |

**>2y band:** train-only, per cluster: min/max of peer **site × calendar month** wash totals (sum daily `wash_count_total`), then compare each test site-month pred/actual (`eval_accuracy_report.py`; see `peer_band_definition` in JSON).

**<2y band:** `ctx_wash_count_total_min` / `max` from cluster context on monthly train rows.

### Short review

**RandomForest** beats **Ridge** on **>2y** monthly aggregates (WAPE **0.33 → 0.14**, many more site-months within ±10%). **<2y** is strong for both. Clustering + TS are not separately scored here—the headline is **level-model** holdout accuracy.

More detail: **`daily_data/daily-data-modelling/clustering_v2/APPROACH_AND_EVAL_BRIEF.md`**

---

## Files (clustering_v2 only)

| Path | Contents |
|------|----------|
| `daily_data/daily-data-modelling/clustering_v2/results/monthly_level_holdout_eval.json` | Ridge + RF holdout metrics |
| `daily_data/daily-data-modelling/clustering_v2/results/monthly_ridge_holdout_eval.json` | Same JSON (legacy mirror filename) |
| `daily_data/daily-data-modelling/clustering_v2/eval_accuracy_report.py` | Regenerates the eval JSON |
| `daily_data/daily-data-modelling/clustering_v2/build_v2.py` | Trains Ridge + RF artifacts under `.../clustering_v2/models/` |
