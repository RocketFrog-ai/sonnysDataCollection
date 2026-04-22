# V2 clustering projection — short guide

Plain-language summary of **what the model does** and **how we measure accuracy**. For full detail see `APPROACH.md` and `results/SYSTEM_ACCURACY_REPORT.md`.

---

## Part 1 — The approach (what happens end to end)

**Goal:** For a **new address** (no history yet), estimate how many car washes you might see in **6‑month windows** and **calendar years** after open, using sites that already trained in the data.

**Steps (simple):**

1. **Find the site on a map** (geocode the address, or pass `--lat` / `--lon`).
2. **Pick “peer” sites** — Among existing train sites, find the **closest cluster** (by distance to that cluster’s center). Those peers share similar geography and context.
3. **Build one feature row** — Things like location, cluster id, and **summary stats from peers** (for example: typical wash levels in that cluster). Missing values are filled with sensible defaults (peer medians, then global medians).
4. **Level model** — A regression model (by default **Ridge**, optionally **RandomForest** after `build_v2.py`) outputs a **ballpark monthly wash level** for this site. For mature sites this starts from a **daily** notion then scales to “per month.”
5. **Time series on the cluster** — We take the cluster’s **historical monthly wash track**, optionally **glue on** the first 24 months from the “new site” (<2y) forecast as extra context, then run a simple forecast (**ARIMA** by default) for the **next 24 months**.
6. **Anchor** — The forecast curve is **scaled** so its overall level lines up with the level from step 4 (site vs cluster).
7. **Bridge (optional but default when using prefix)** — When we use the <2y forecast as context for >2y, we can **scale** the mature track so month 25 matches the end of the opening phase (fixes a visible “cliff” between year 2 and year 3).

**What you get:** JSON with horizons, monthly series, and optional **calendar‑year totals** (sum of 12 forecast months for years 1–4). Plots bundle “no opening prefix” vs “prefix + bridge” and can show **Ridge vs RandomForest** side by side.

### Example (one address)

Address used in docs:

`7021 Executive Center Dr, Brentwood, TN 37027`

**Train models (whenever data or code changes):**

```bash
cd daily_data/daily-data-modelling/clustering_v2
python build_v2.py
```

**Compare Ridge vs RF in one 2×2 chart** (needs `build_v2.py` so RF joblib files exist):

```bash
python project_site.py \
  --address "7021 Executive Center Dr, Brentwood, TN 37027" \
  --method arima \
  --plot-ridge-rf-four-way \
  --allow-distant-nearest-cluster \
  --out-name mysite
```

Outputs go under `results/projection_demo/`, e.g. `projection_arima_mysite_ridge_rf_four.png` and matching `.json`.

---

## Part 2 — How accuracy is evaluated (short)

We do **not** evaluate on the same rows used to train. We **hold out** future (or later-life) data, fit on the past, then score on the held-out piece.

### Two cohorts, two shapes of “one prediction”

| Cohort | Idea of one test row | Train vs test split (plain words) |
|--------|----------------------|-----------------------------------|
| **More than 2 years old (`>2y`)** | One **site × calendar month** in the test window | Train on all **daily** rows with date **before 2025‑07‑01**. Test on daily rows **on or after 2025‑07‑01**. We predict **each day**, then **sum** predictions and actuals **within each month** and compare those monthly totals. |
| **Less than 2 years old (`<2y`)** | One **site × month** row that is already monthly | Train on **first‑year** months only (`period_index` 1–12). Test on **second‑year** months (`period_index` 25–36). The model predicts that month’s **total washes** directly. |

### Metrics (what the numbers mean)

- **WAPE** — “Weighted absolute percent error”: sum of absolute errors divided by sum of actuals. **Lower is better.** Good for comparing models when volumes differ a lot.
- **MAE / RMSE** — Average size of error in **washes** (same units as the target).
- **R²** — How much of the variation in actuals the predictions explain (0–1; higher is better).
- **% within ±10%** — Share of rows where prediction is within 10% of actual (higher is better).
- **Peer min–max check:** **% of predictions inside the cluster’s train min–max wash range** — **<2y** uses context columns on monthly train rows; **>2y** uses the same idea at **monthly** grain (train peer site×month sums, then per-cluster min/max). See `eval_accuracy_report.py` and `peer_band_definition` in the JSON.

### Where the numbers live

Running:

```bash
cd daily_data/daily-data-modelling/clustering_v2
python eval_accuracy_report.py
```

writes **`results/monthly_level_holdout_eval.json`** (same content also copied to `monthly_ridge_holdout_eval.json`). Inside the file, metrics are grouped as:

- `more_than_2yrs.ridge` and `more_than_2yrs.random_forest`
- `less_than_2yrs.ridge` and `less_than_2yrs.random_forest`

### Example numbers (from one eval run on this repo)

These are **illustrative** — re-run the script after fresh data for up-to-date values.

**>2y (monthly totals built from daily predictions), test from July 2025 onward**

- **Ridge:** WAPE about **0.33** (roughly **33%** weighted error), R² about **0.68**, about **18%** of site‑months within ±10% of actual.
- **RandomForest (same split):** WAPE about **0.14**, R² about **0.88**, about **46%** within ±10%.

**<2y (one row per site‑month, second year held out)**

- **Ridge vs RF** are **close** (WAPE about **0.14** vs **0.14** on a typical run); Ridge is still fine here; RF’s big win in this pipeline is mainly on the **>2y** side.

**Takeaway in one line:** Evaluation is **honest out-of-sample** by date or by life stage; **RandomForest** usually looks much stronger for **>2y** monthly aggregates, while **<2y** stays tight for both.

---

## Related files

| File | Role |
|------|------|
| `APPROACH.md` | Full technical description |
| `build_v2.py` | Train Ridge + RF artifacts |
| `project_site.py` | Score a single address + plots |
| `eval_accuracy_report.py` | Monthly holdout JSON above |
| `results/SYSTEM_ACCURACY_REPORT.md` | Human-readable summary tied to figures |
