# Car-Wash Site KPI Predictor

Predict a **new** car-wash site's monthly KPIs from its **geographic cluster
neighbours**, building on the validated finding (see
`notebooks/cluster_pattern.ipynb`) that sites within ~20 km co-move month-to-month
significantly more than random groups.

A user drops a pin on a USA map; the app finds neighbour sites within the 20 km
radius and predicts 5 KPI trajectories as a distance-weighted blend of those
neighbours.

## The 5 predicted metrics
`membership_purchased_count`, `membership_wash_count`, `asp_per_membership`,
`retail_wash_count`, `asp_per_retail_wash`.

## Method
- **Neighbours** = sites within `BUFFER_KM` (20 km) by haversine distance; if
  fewer than 3, fall back to the 5 nearest.
- **Prediction** = inverse-distance-weighted (IDW) mean of neighbour
  trajectories per month (toggle to simple mean). A new site has no history, so
  only distance can weight neighbours. Per-month weights renormalise over the
  neighbours that have data that month.
- **Cleaning** = winsorize each KPI to its 1st/99th percentile (tames impossible
  spikes like a $5,734 retail wash); drop all-zero flatline sites. Toggle in
  `src/config.py`.

## Evaluation (`artifacts/eval_results.csv`)
Leave-one-out spatial CV over the ≥12-month cohort: hide each real site, predict
from its neighbours, compare to actual. Median Pearson r ("how synced they
move"), MAE, sMAPE, and uplift vs a global-mean baseline. Wash counts sync
strongly (retail r≈0.74, membership r≈0.56); retail ASP barely moves with
neighbours (negative uplift) — reported honestly.

## Layout
```
src/config.py        constants (paths, BUFFER_KM, KPIs, winsorize flags)
src/data_loader.py   load + clean + winsorize -> sites table + PANELS
src/neighbours.py    haversine, find_neighbours, cluster bounding boxes
src/predict.py       predict_site() -> PredictionResult
src/evaluate.py      LOOCV spatial cross-validation
app.py               Streamlit app
artifacts/           cluster_bboxes.json, eval_results.csv (generated)
```

## Run
```bash
cd final_modelling
pip install -r requirements.txt

# regenerate artifacts (optional; already committed)
python3 -m src.neighbours      # -> artifacts/cluster_bboxes.json
python3 -m src.evaluate        # -> artifacts/eval_results.csv

streamlit run app.py
```

## Notes
- ASP metrics are ratios and internally consistent (asp = sales / count); they
  are IDW-averaged directly.
- Data is filtered to `package_name == "Monthly Recurring Plan"`;
  `site_uid = client_id + "__" + site_id` (a brand owns 1–82 sites).
