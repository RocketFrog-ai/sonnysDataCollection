# Monthly Ridge holdout — figures only

**Source:** `monthly_level_holdout_eval.json` (same content mirrored as `monthly_ridge_holdout_eval.json`). JSON is nested as `more_than_2yrs.ridge` / `more_than_2yrs.random_forest` (and likewise for \<2y).  
**Generated (UTC):** Re-run `python eval_accuracy_report.py` for a fresh timestamp.  
**Method:** V2 level models (Ridge + RandomForest, same preprocessing as `build_v2.py`), train-only DBSCAN refit + cluster context; splits match `backtest_v2.py`. Percent errors = \|pred − actual\| / actual × 100 on each **month-level** unit below.

**Summary:** Use **`ridge`** blocks for the original headline numbers. **RandomForest** is typically much stronger on **\>2y** monthly aggregates from daily preds; **\<2y** Ridge vs RF are close. **Quantile** path remains separate (`build_quantile_v2.py` / `project_site_quantile.py`).

---

## \>2y — monthly (aggregated from daily)

*Each evaluation row = one **site × calendar month** in the test window. Actual = sum of daily `wash_count_total`. Pred = sum of daily Ridge predictions (same model as row-level backtest).*

| Metric | Value |
|--------|------:|
| Test site-months | **2,888** |
| MAE (washes / month) | **3,003.7** |
| RMSE | **4,273.0** |
| R² | **0.679** |
| WAPE | **0.329** |

**% of site-months with error vs actual month total at most:**

| Threshold | All months | Subset: actual month ≥ 3,000 washes (n = 2,405) |
|-----------|------------|---------------------------------------------------|
| ≤10% | 18.0% | 20.8% |
| ≤15% | 27.7% | 32.1% |
| ≤20% | 35.8% | 41.1% |
| ≤30% | 51.0% | 58.3% |

| Dispersion (all months, n = 2,888) | Value |
|--------------------------------------|------:|
| Median abs % error | 29.3% |
| Mean abs % error | 207.8% *(inflated by small-volume months in denominator)* |

---

## \<2y — monthly (panel already monthly)

*Each evaluation row = one **monthly** panel row (`wash_count_total`). Ridge predicts that month’s total directly.*

| Metric | Value |
|--------|------:|
| Test rows | **4,689** |
| MAE (washes / month) | **1,356.6** |
| RMSE | **2,834.6** |
| R² | **0.778** |
| WAPE | **0.140** |
| Pred inside cluster train min–max | **97.82%** |
| Actual inside same min–max | **96.95%** |

**% of months with error vs actual at most:**

| Threshold | All months | Subset: actual ≥ 2,000 (n = 4,603) |
|-----------|------------|-------------------------------------|
| ≤10% | 49.6% | 50.4% |
| ≤15% | 64.3% | 65.3% |
| ≤20% | 73.9% | 75.1% |
| ≤30% | 86.2% | 87.5% |

| Dispersion (all months) | Value |
|-------------------------|------:|
| Median abs % error | 10.2% |
| Mean abs % error | 18.9% |

---

*Regenerate: `python eval_accuracy_report.py` from `clustering_v2/`.*
