# Local Market Explorer + Drop-a-Pin Forecaster (Streamlit)

Interactive companion to the notebooks. Two modes (top of the sidebar):

```bash
cd earnest-proforma-2.0/streamlits
streamlit run app.py            # http://localhost:8501
```

Runs in the repo's main `venv` (streamlit, plotly, folium, streamlit-folium, lightgbm, scikit-learn).
Reads `../data/main-ds.csv` directly (cleans + density-aware clustering on first load, cached) — no notebook run required.

## 🗺️ Mode 1 — Explore markets
Pick a site (a **pin**), see its neighbours within a radius (the **local market**), and watch what happens when a
**new site** enters: the newest entrant is drawn in **red** with its opening marked, so you can see it ramp up while
the incumbents' series react.
- 🎲 Random pin, cluster/site pickers, or **click a pin on the map**.
- Metric (total / membership / retail washes, revenue, mem-share), neighbour radius, smoothing, max-sites.
- X-axis = *Calendar date* (entry event + neighbour response) or *Months since open* (compare ramp shapes).
- Bottom: the new site's **ramp** + the incumbents' **before/after** retail change.

## 📍 Mode 2 — Drop-a-pin forecast (cold-start)
For a site that **doesn't exist yet**: **click a location** (or type lat/lon), pick the **operator**, and get the
**expected monthly car-wash count for the next 5 years** (membership + retail, with P10–P90 bands) **plus the impact
on existing neighbours**. Powered by [`coldstart_model.py`](coldstart_model.py) (built/evaluated in
[`../notebooks/coldstart_forecast.ipynb`](../notebooks/coldstart_forecast.ipynb)):

- **Plateau level** ← LightGBM quantile regression on location + local-market features + **operator/brand** (the
  dominant driver). Honest accuracy: leave-one-site-out **R²≈0.33, ~27% median error** with a known operator
  (≈40–46% for naive baselines); a fresh area with no neighbours leans on a weak region prior → **wider bands**.
- **Ramp shape** ← learned **region-pooled** curves (membership ~7 mo to 80%, retail ~flat from month 1); data-driven to ~3.5 yr then a flat asymptote.
- **Neighbours** ← learned **cannibalization-by-distance** (≈ −23% retail within 5 km → −9% at 10–20 km).
- Controls: operator (or *unknown*), **plateau override**, and year-3–5 growth sliders (the long run is a scenario, not a precise prediction).

## Density-aware clustering (the "local market" definition)
A single 20 km radius **chains** dense metros into 100 km+ blobs. We use an **adaptive** scheme — a site links to a
neighbour only within `min(rᵢ, rⱼ)`, where rᵢ = **10 km if dense (≥5 sites within 10 km) else 20 km**; components wider
than 25 km are re-split; isolated sites are **standalone (grey on the map)**. It **won the bake-off** vs fixed-20 km
DBSCAN and HDBSCAN (higher within-cluster co-movement, no chaining, finer markets — see the cold-start notebook §1).
Implemented once in `coldstart_model.assign_clusters(...)` and used by both the app and the model.

## Files
- `app.py` — the two-mode Streamlit app (MOIRAI-free).
- `coldstart_model.py` — adaptive clustering, cold-start plateau/ramp/neighbour model, `predict_site` / `predict_neighbours`.

> MOIRAI / time-series forecasting was explored in `../notebooks/` and found **not** to help the cold-start goal (it can't forecast a no-history site, and on existing sites it only ties a seasonal-naive baseline). That code and its isolated venv were removed; the analysis and evidence remain in the notebooks.
