# Car Wash Projection Approach (Simple)

This is a short, non-technical explanation of how the V2 system works.

## 1) What goes in

You give:
- an `address` (or `lat/lon`)
- projection method (`blend`, `holt_winters`, or `arima`)

The system returns projections for both cohorts:
- `more_than_2yrs` (daily-history model)
- `less_than_2yrs` (monthly-history model)

## 2) Step-by-step flow

1. **Find location**
   - Convert address to latitude/longitude (or use provided lat/lon).

2. **Assign nearest 12km cluster**
   - Compare your point with train-time DBSCAN cluster centroids.
   - Pick the nearest cluster for each cohort.

3. **Build feature vector for new site**
   - Use known fields: lat, lon, cluster id.
   - Add cluster context stats (local market signals from nearby sites) where useful.
   - Missing fields are filled by train medians.

4. **Predict base wash-count level**
   - Point model: Ridge (single value).
   - Quantile model: q10 / q50 / q90 (low / median / high).

5. **Forecast next 24 months from cluster history**
   - Use cluster monthly median history.
   - Forecast using selected method (`blend` = Holt-Winters + ARIMA average).

6. **Anchor forecast to site-specific level**
   - Scale cluster forecast so it matches your site prediction level.

7. **Aggregate horizons**
   - Return cumulative values for 6m, 12m, 18m, 24m.
   - For quantile mode, return ranges (q10/q50/q90).

## 3) Why two plots can look similar

Sometimes `>2y` and `<2y` bars look close because:
- both assigned clusters have similar recent monthly levels,
- both get similar anchor scales,
- cumulative horizons smooth differences.

That does **not** mean models are identical; it means this location has similar expected volume in both cohort views.

## 4) Dry run example (Suwanee)

Input:
- Address: `5360 Laurel Springs Pkwy, Suwanee, GA 30024`
- Resolved lat/lon: `34.085454, -84.16069`
- Method: `blend`

### A) Quantile output (from `projection_quantile_blend_demo_suwanee.json`)

`more_than_2yrs`:
- Cluster: `0` (distance `3.70 km`, size `3`)
- 24m cumulative: q10 `101,063`, q50 `150,129`, q90 `208,559`

`less_than_2yrs`:
- Cluster: `2` (distance `6.23 km`, size `4`)
- 24m cumulative: q10 `113,850`, q50 `155,065`, q90 `221,266`

Interpretation:
- Use q50 as planning number.
- Use q10/q90 as risk band.

### B) Point output (older blue style)

For same location (previous point run):
- `more_than_2yrs` 24m: `188,475`
- `less_than_2yrs` 24m: `140,522`

Interpretation:
- One number only, no uncertainty range.
- Often feels more intuitive for a single KPI view.

## 5) Which output to use for KPI

If KPI is **car wash volume prediction**:
- Use **q50** (or point forecast) as primary KPI number.
- Track **WAPE** for model quality.
- Keep q10/q90 as optional risk guardrails.

