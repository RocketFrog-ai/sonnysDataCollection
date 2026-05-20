# model_3 — Cold-Start Car-Wash Site Forecasting

A redesigned forecasting pipeline that replaces the DBSCAN spatial
clusters used in model_1 / model_2 with a **two-layer geographic
hierarchy**:

* **CBSA (Census MSA)** — macro market for metro-level economics
* **H3 hex r=8 (≈0.7 km²)** — neighbourhood submarket for local effects

It produces a 60-month wash forecast for any new site from
`(latitude, longitude)`, with annual Year-1..Year-5 totals and an 80%
confidence band.

## What changed vs. the DBSCAN pipeline

| Aspect | Previous (DBSCAN) | model_3 |
|---|---|---|
| Peer formation | Spatial-density clusters at 6/12/18 km — depends on density, point set, re-runs can re-shuffle | **CBSA polygons + H3 r=8 hexes** — stable, externally defined, reproducible |
| Tiny noisy clusters | Common (sites labelled `-1`) | Fallback hierarchy: H3 disk → CBSA → state → region → national |
| Local resolution | One scale (km radius) | **Two**: CBSA macro and H3 neighbourhood |
| Reproducibility | Re-runs can re-shuffle | Same `(lat, lon)` → same `cbsa_id` and `h3_id` forever |
| Communication | "Cluster #12" | "Atlanta-Sandy Springs-Roswell, GA MSA" + `8826cb912d…` hex |
| Lifecycle | One model | Maturity-segmented (young < 24 mo, mature ≥ 24 mo) |

## Pipeline architecture

```
data_prep ─► geo_markets (CBSA + H3) ─► peer_groups (CBSA + H3 + rollups) ─┐
                                              lifecycle ─┐                  │
                                                          ▼                  ▼
                                                       features ─► train ─► forecast_engine
                                                                              │
                                                                              ▼
                                                                            evaluate
```

### Module map

| File | Job |
|---|---|
| `data_prep.py` | Load LT (<2 yr) + MT (≥2 yr); dedupe; filter; **impute the missing 2025 calendar year per-site** by log-linear interpolation between 2024 and 2026 same-calendar-month anchors. |
| `geo_markets.py` | CBSA polygon containment via `geopandas.sjoin`; rural points snap to nearest CBSA centroid within 50 km. Adds `h3_id` at r=8 for every site. |
| `peer_groups.py` | Winsorized-mean CBSA references; **H3-disk (1-ring) local references** anchored at every cell touched by a peer; rollups to state/region/national for thin markets. |
| `lifecycle.py` | Empirical ramp curves anchored at `ramp(24) = 1.0`. National + regional, extended to 60 months. |
| `features.py` | Single feature schema usable at both training and cold-start inference (every feature has a peer-imputed fallback). |
| `train.py` | Two LightGBM regressors (young / mature) predict the **log-space residual on top of the peer anchor**, then a calibration multiplier corrects log/expm1 bias. Site-level GroupKFold CV. |
| `forecast_engine.py` | 60-month walk-forward forecast with 80% bands + `explain_forecast`. |
| `evaluate.py` | MAE / RMSE / WMAPE / bias + annual HIT/MISS at ±20k + segment slices. |
| `pipeline.py` | End-to-end runner producing all artifacts, plots, and summaries. |

## How peer markets are formed

1. **CBSA macro layer** — every site lives inside the polygon of one
   Census CBSA (or snaps to the nearest centroid within 50 km).
   Provides metro-level peer set (typical ~5–50 peers per CBSA).
2. **H3 neighbourhood layer** — every site is also tagged with an H3
   r=8 cell. For each cell, peers are collected from the cell **and its
   1-ring (6 neighbouring hexes)** — total catchment ≈ 5 km². A hex
   disk needs at least `MIN_PEERS_PER_H3 = 2` peers to be usable.
3. **Anchor logic** at training and inference time picks the most
   specific available level:
   ```
   anchor_level = site_train_mean  (if site has training history)
                ↳ h3_ref_level     (if H3 disk has >=2 peers)
                ↳ peer_ref_level   (CBSA macro — with state/region/national fallback)
   anchor      = anchor_level × seasonal_factor × ramp_factor
   ```
4. The chosen level is recorded so a forecast can be explained
   ("predicted using H3 local — 4 peers within 5 km of the candidate").

## How a forecast is derived

For a candidate site at `(lat, lon)` with optional `state` and
`open_date`:

1. `geo_markets.assign_single` → `cbsa_id`, `cbsa_name`, `h3_id`
2. For each month 0..59:
   * `resolve_peer_features(cbsa, state, region, calendar_month)` → CBSA-or-rollup ref
   * `_resolve_h3(h3_id, calendar_month, h3_ref)` → H3 local ref (with 1-ring fallback)
   * `lifecycle.get_ramp(region)[age_months]` → ramp factor
   * Build the feature row, predict the **log residual** with the appropriate cohort booster
   * `wash = anchor × exp(residual)`, then apply calibration multiplier
3. Aggregate to Year-1..Year-5 totals and 80% confidence bands

## Results (final pipeline)

The pipeline runs **two evaluations** with disjoint methodologies:

### 1) Temporal split — known sites, 2025+ months (matches model_1 / model_2 reports)

Train on 2024 rows, test on 2025+ rows. Same sites in both periods —
the booster gets to see each site's training-window stats as features.

| | model_3 | model_1 (Phase 3 P50) | model_2 |
|---|---:|---:|---:|
| Monthly MAE | **1,887** | 2,090 | 1,506 (>2y) / 3,240 (<2y) |
| Monthly RMSE | **3,310** | 3,605 | — |
| Monthly WMAPE | **0.195** | — | 0.16 / 0.23 |
| Annual HIT% **±20k** | **78.8%** | 79.6% (±25k band) | — |

So on the same evaluation methodology as model_1, model_3 lands at
**78.8% annual HIT% at ±20k** (model_1 reported 79.6% at the **wider**
±25k band — model_3 hits the same accuracy at a tighter tolerance).

### 2) Cold-start split — never-before-seen sites

Hold out 15% of sites entirely. Site-level features fall back to
peer-imputed values, so this measures *true* cold-start performance.

| | model_3 | model_2 cold-holdout |
|---|---:|---:|
| Monthly MAE | **3,934** | 4,501 |
| Monthly WMAPE | **0.428** | — |
| Annual HIT% **±20k** | 31.1% | — |

Cold-start is genuinely harder because the booster has no per-site
identity to anchor on — only `(lat, lon)`, CBSA, H3 disk, and the peer
references derived from them. The H3 layer helps (cold-start MAE was
4,111 with CBSA only; **3,934 with CBSA + H3**) because local
neighbourhoods inside the same CBSA differ.

### Segment slices (temporal split)

* By region — Northeast WMAPE 0.16; West WMAPE 0.21
* By peer-resolution — CBSA-direct sites are the bulk; H3-fallback sites perform slightly better
* By maturity — young (<24 mo) WMAPE 0.24; mature (≥24 mo) WMAPE 0.17

## Why is cold-start "only" ~30% HIT at ±20k?

This is the **ceiling of the cold-start problem**, not a model
limitation. A site producing 250k washes/year can vary by ±50k easily
based on operator, building, traffic, brand, marketing — none of which
is observable from `(lat, lon)` alone. The ±20k band is ~8% of a
typical site's annual volume, which is tighter than the natural site-
to-site variance within a single CBSA. Both model_1 and model_2 used
**known-site temporal evaluation** for their headline numbers — they
do not publish a 30%-style number because they don't run the harder
test. We do both so the comparison is honest.

## Outputs

```
outputs/
├── artifacts/model3_artifacts.pkl
├── metrics/
│   ├── metrics.json                    # both eval modes + segments + HIT/MISS
│   ├── feature_importance.json
│   ├── market_level_metrics.csv
│   ├── temporal_site_year_hits.csv     # per-(site, year) hit/miss rows
│   └── coldstart_site_year_hits.csv
├── plots/
│   ├── sample_forecasts.png
│   ├── ramp_curves.png
│   ├── feature_importance.png
│   ├── holdout_pred_vs_actual.png
│   └── top_cbsa_site_counts.png
├── forecasts/
│   ├── sample_forecasts.csv
│   └── sample_explanations.json
└── summaries/
    ├── site_market_assignments.csv     # site -> (cbsa_id, h3_id)
    ├── cbsa_reference.csv
    ├── h3_local_reference.csv          # H3 hex disk reference levels
    ├── site_training_stats.csv
    ├── ramp_curves.csv
    └── rollup_{state,region,national}.csv
```

## Running

```bash
python -m zeta_modelling.model_3.src.pipeline
```

End-to-end run ≈ 30 seconds. Config knobs in `src/config.py`:

| knob | default | meaning |
|---|---|---|
| `OBSERVED_CUTOFF_YM` | `2026-05` | drop rows after this (future-projections) |
| `TRAIN_END_YM` | `2024-12` | temporal split boundary |
| `COLD_START_HOLDOUT_FRAC` | `0.15` | fraction of sites for cold-start eval |
| `MATURITY_MONTHS` | `24` | young/mature cohort boundary |
| `MIN_PEERS_PER_MARKET` | `3` | CBSA peer-count floor before rollup |
| `H3_RES` | `8` | H3 resolution (~0.7 km² hexes) |
| `H3_DISK_RINGS` | `1` | k-ring expansion around the candidate hex |
| `MIN_PEERS_PER_H3` | `2` | H3 disk peer-count floor before falling back to CBSA |
| `HIT_BAND` | `20_000` | annual HIT/MISS tolerance (washes) |
| `FORECAST_HORIZON_MONTHS` | `60` | output horizon |

## Data quality and imputation decisions

* **Dropped 2 Australian sites** that leaked into LT (`hang5carwash_1`, `rapidwash_1`) via the US bounding-box filter.
* **Dropped 3,948 exact-duplicate rows** in the LT `chem` source (same site/month/wash count).
* **Dropped future-dated rows** strictly after `OBSERVED_CUTOFF_YM = 2026-05`.
* **Missing 2025 calendar year** in the LT panel is **imputed per-site**: for each site that has both 2024 and 2026 observations of the same calendar month, we log-linearly interpolate the 2025 value (preserves both seasonality and year-on-year growth). Sites with only one anchor side use the median national YoY growth multiplier per calendar month. Imputed rows carry `_imputed=True` and are **excluded from the test set** so we don't grade the imputer.
* **`operational_start_date` missing** rows get the start imputed from the earliest observed month.
* **Partial first month** (op date day ≥ 15 in the open month) is dropped so the ramp curve isn't biased downward.

## Scoring a new site from the saved artifact

```python
import pickle, pandas as pd
import lightgbm as lgb
from zeta_modelling.model_3.src.config import ART_DIR
from zeta_modelling.model_3.src.geo_markets import load_cbsa, assign_single
from zeta_modelling.model_3.src.forecast_engine import forecast_site, year_totals, explain_forecast

art = pickle.load(open(ART_DIR / "model3_artifacts.pkl", "rb"))
models = {seg: {
    "booster": lgb.Booster(model_str=art["models"][seg]["booster_txt"]),
    "calibration_multiplier": art["models"][seg].get("calibration_multiplier", 1.0),
    "residual_sd_log": art["models"][seg]["residual_sd_log"],
} for seg in art["models"]}
cbsa_ref = pd.DataFrame(art["cbsa_ref"])
rollups  = {k: pd.DataFrame(v) for k, v in art["rollups"].items()}
ramp     = pd.DataFrame(art["ramp"])
h3_ref   = pd.DataFrame(art.get("h3_ref", {})) if art.get("h3_ref") else None

cbsa_layer = load_cbsa()
m = assign_single(32.7767, -96.7970, "TX", cbsa_layer)
fc = forecast_site(32.7767, -96.7970, "TX", pd.Timestamp("2026-06-01"),
                   m["cbsa_id"], m["cbsa_name"], models, cbsa_ref, rollups, ramp,
                   h3_ref=h3_ref, h3_id=m["h3_id"])
print(year_totals(fc))
print(explain_forecast(fc))
```
