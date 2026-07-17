# Cold-start plateau × ramp model — model card

The drop-pin wash-count model: for a lat/lon with no operating history, predict a 5-year
monthly trajectory `total/mem/ret` with P10–P90 bands.

**Artifact:** `coldstart_artifacts.joblib` in this folder (~46 MB; `coldstart.MODEL_PATH`
points here). Prior versions in `backups/` (`.bak-pre-v2`, `.bak-pre-cleanramp`,
`.bak-pre24mo`) — kept for rollback, loaded by nothing.

## How it works (see `proforma/MODELLING.md` for the full write-up)

`level (plateau) × ramp (life-cycle curve) × cannibalization`, fitted over the panel
(`proforma/data/panel/main-data-v2-stitched.csv`):

- **Plateau**: LightGBM quantile models (q10/q50/q90) + ExtraTrees level on neighbour
  features (counts/levels at 5/10/20 km, nearest site, region/state) with `brand_loo`
  (operator mean mature level) as the strongest feature. With no operator given, the
  **local-mature anchor** fills it: median mature level of ≥24-month neighbours ≤20 km
  (CoV-guarded), then a w=0.50 log-space calibration blend.
- **Ramp**: normalized life-cycle curves (median across sites, cluster → region → global
  fallback; 2020 COVID cohort excluded from the shape pool).
- **Cannibalization**: learned distance-decay retail impact on neighbours
  (`predict_neighbours`).

## The UI's "Models 1–4" are flag combos of this one artifact

| Label | Flags | LOO backtest (1,223 sites) |
|---|---|---|
| Model 1 | no local anchor | DROPPED — bias 0.72, badly under-predicts |
| Model 2 | `model_kind="lgb"` + anchor | WAPE 43.6% |
| Model 3 | `model_kind="et"` + anchor | WAPE 40.2% — production default |
| Model 4 | Model 3 + `brand=` operator | WAPE ≈34% when the operator is known |
| Model 5 | Model 3 + SUPER level layer | see `../ensemble_super/README.md` |

Independent panel-scale validation (862 sites opened 2021–24, leave-one-out with artifact
surgery): mature MdAPE 32.9%, ratio 0.99 (unbiased), ρ 0.46; Y1–Y5 MdAPE 28–34%.

## Ops

- **Loaded by** `proforma/models/coldstart.py` (`load()`), consumed by the Streamlit UI
  and the FastAPI backend via `proforma/pnl/data.py`.
- **Refit** with `cm.fit(save=True)` in conda `sonnys` after a panel rebuild
  (`proforma/scripts/process_main_data_v2.py`). The pickle holds no module references
  (rename-safe) but **is** welded to library versions — refit in the env that loads it
  (`docs/ENVIRONMENTS.md`).
- Run `./scripts/smoke.sh` after any refit — it captures this model over three fixed pins.
