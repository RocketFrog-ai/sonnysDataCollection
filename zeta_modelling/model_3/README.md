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
| `data_prep.py` | Load LT (<2 yr) + MT (≥2 yr); **drop the LT `chem`-source cohort** (`EXCLUDE_CHEM` — those rows have no open date, are duplicated, and the chemical program confounds wash volume); dedupe; filter; relabel mislabelled LT year-2 months. |
| `geo_markets.py` | CBSA polygon containment via `geopandas.sjoin`; rural points snap to nearest CBSA centroid within 50 km. Adds `h3_id` at r=8 for every site. |
| `peer_groups.py` | Winsorized-mean CBSA references; **H3-disk (1-ring) local references** anchored at every cell touched by a peer; rollups to state/region/national for thin markets. |
| `lifecycle.py` | Empirical ramp curves anchored at `ramp(24) = 1.0`. National + regional, extended to 60 months. |
| `features.py` | Single feature schema usable at both training and cold-start inference (every feature has a peer-imputed fallback). |
| `train.py` | Two LightGBM regressors (young / mature) predict the **log-space residual on top of the peer anchor**, then a calibration multiplier corrects log/expm1 bias. Site-level GroupKFold CV. |
| `forecast_engine.py` | 60-month walk-forward forecast with 80% bands + `explain_forecast`. |
| `evaluate.py` | MAE / RMSE / WMAPE / bias + **`annual_hit_miss`** — annual HIT/MISS on either calendar or **tenure** years, with an absolute (±20k) and a percentage (±20%) band. |
| `pipeline.py` | End-to-end runner. Trains the production booster, runs the temporal eval, and runs the **k-fold cold-start cross-validation** (`coldstart_kfold_eval`). |

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

The pipeline runs **two evaluations** with disjoint methodologies. Both
exclude the LT `chem`-source cohort (`EXCLUDE_CHEM`).

### 1) Temporal split — known sites, 2025 months

Train on 2024 rows, test on 2025 rows. Same sites in both periods — the
booster has each site's 2024 history as features. Graded on calendar-year
annual totals (a calendar split by construction).

| metric | model_3 |
|---|---:|
| Monthly MAE | **1,857** |
| Monthly RMSE | **3,347** |
| Monthly WMAPE | **0.189** |
| Annual HIT% **±20%** | **78.4%** |
| Annual HIT% **±20k** | **76.0%** |
| Median annual abs % error | **10.2%** |

### 2) Cold-start — k-fold cross-validation, never-before-seen sites

`COLD_START_KFOLDS`-fold CV: **every site is held out exactly once** and
scored as brand-new (full-coverage metric, not a single 15% slice). For
each fold the held-out sites' CBSA / H3 references are rebuilt from the
*other* folds only (**leave-fold-out** — a site never sees itself as a
peer) and from the **2024 training window only** — i.e. a new site is
predicted purely from the *temporal trend of its neighbours*. A booster is
retrained per fold on the other folds. Site-level features are imputed from
peers and `cbsa_code = 0`, matching how the production engine scores a
brand-new market.

Graded on **tenure years** (Year-1 = site age 0-11 months) — **not**
calendar years. The sites open across all 12 months of 2024, so a calendar
year mixes life-stages and a partial first calendar year would be scored as
a full-year miss. Tenure grading aligns Year-1 with the actual first 12
months of operation.

| metric | model_3 cold-start |
|---|---:|
| Monthly MAE | 4,810 |
| Monthly WMAPE | 0.509 |
| Annual HIT% ±20% | 24.8% |
| Annual HIT% ±20k | 25.3% |
| Median annual abs % error | 42.8% |

Tenure-year breakdown (833 full site-years across all 673 sites):

| tenure year | n | HIT ±20% | median abs % error |
|---|---:|---:|---:|
| Year 1 | 129 | 17.1% | 39.6% |
| Year 2 | 176 | 25.0% | 42.2% |
| Year 3 | 165 | 27.3% | 42.8% |
| Year 4 | 86 | 26.7% | 41.4% |
| Year 5 | 56 | 35.7% | 40.3% |

### Segment slices

* Temporal split, by region — all regions WMAPE ≈ 0.18–0.21
* Temporal split, by maturity — young (<24 mo) WMAPE 0.24; mature 0.17
* Cold-start, by region — Northeast WMAPE 0.41 (best), West 0.66 (worst)

## Why is cold-start "only" ~25% HIT and ~40% median error?

This is close to the **ceiling of the cold-start problem**. A new site has
no own history, so it is predicted from `(lat, lon)`, CBSA, H3 disk and the
peer references derived from them. A site producing 200k washes/year can
swing ±40% on operator, building, traffic, brand and marketing — none of
which is observable from location alone. The booster barely beats the raw
peer anchor at cold-start (MAE 4,810 vs 4,903) because every site-level
feature collapses to a peer-imputed value.

Two things that were *not* a true ceiling and have been fixed:

1. **Calendar-year grading was wrong.** The earlier report graded
   cold-start "Year 1" as calendar-year 2024. The sites are a 2024–2025
   monthly snapshot and open across *every* month of 2024 — so calendar
   2024 is a partial, pre-stabilisation slice for most sites and calendar
   2025 blends two life-years. Cold-start is now graded on **tenure years**
   (Year-1 = site age 0-11 months).
2. **The old single 15% holdout leaked.** Peer references were built from
   the full panel, so a held-out site appeared in its own CBSA/H3 peer
   pool (badly so for 2-peer H3 disks). The k-fold CV rebuilds every
   reference leave-fold-out and from the neighbours' 2024 window only.

What *remains* a real, fixable error source — **carwash-format mix**.
The CBSA peer reference pools all formats, but the MT panel is a mix of
high-throughput Express Tunnels and low-volume legacy formats (Full
Service, Self-Serve, Hand Wash). New builds are almost all Express
Tunnels, so the pooled reference is dragged *down* and new sites are
**systematically under-predicted** (Year-1 cold bias is negative); very
old legacy sites are over-predicted (tenure Year-12+ shows >100% error).
A **format-aware peer reference** (restrict the mature anchor to Express
Tunnel peers) is the highest-value next step for cold-start accuracy.

## Outputs

```
outputs/
├── artifacts/model3_artifacts.pkl
├── metrics/
│   ├── metrics.json                    # both eval modes + segments + HIT/MISS
│   ├── feature_importance.json
│   ├── market_level_metrics.csv
│   ├── temporal_site_year_hits.csv     # per-(site, calendar-year) hit/miss rows
│   └── coldstart_site_year_hits.csv    # per-(site, TENURE-year) cold-start hit/miss
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

End-to-end run ≈ 2-3 minutes (the cold-start k-fold retrains a booster per
fold). Config knobs in `src/config.py`:

| knob | default | meaning |
|---|---|---|
| `EXCLUDE_CHEM` | `True` | drop the LT `chem`-source cohort (no open date, duplicated, confounded) |
| `OBSERVED_CUTOFF_YM` | `2025-12` | drop rows after this (future-projections) |
| `TRAIN_END_YM` | `2024-12` | temporal split boundary |
| `COLD_START_KFOLDS` | `5` | folds for the cold-start cross-validation (every site held out once) |
| `COLD_REF_LEAVE_SITE_OUT` | `True` | cold-start refs exclude the held-out fold |
| `COLD_REF_TRAIN_WINDOW_ONLY` | `True` | cold-start ref *levels* use the neighbours' 2024 window only |
| `MATURITY_MONTHS` | `24` | young/mature cohort boundary |
| `MIN_PEERS_PER_MARKET` | `3` | CBSA peer-count floor before rollup |
| `H3_RES` | `8` | H3 resolution (~0.7 km² hexes) |
| `H3_DISK_RINGS` | `1` | k-ring expansion around the candidate hex |
| `MIN_PEERS_PER_H3` | `2` | H3 disk peer-count floor before falling back to CBSA |
| `HIT_BAND` | `20_000` | annual HIT/MISS absolute tolerance (washes) |
| `ANNUAL_PCT_BAND` | `0.20` | annual HIT/MISS percentage tolerance |
| `FORECAST_HORIZON_MONTHS` | `60` | output horizon |

## Data quality and imputation decisions

* **Dropped the LT `chem`-source cohort** (`EXCLUDE_CHEM=True`) — 270 of 491
  LT sites. Those rows have **no `operational_start_date`** (so site age,
  and therefore the ramp, cannot be trusted), are exact-duplicated, and
  belong to a chemical-program cohort whose wash volumes are confounded by
  the program. The 221 surviving LT `control` sites all carry a real,
  dated open month in 2024.
* **The LT file is a calendar snapshot**, not an open-aligned panel:
  `month_number=1` is Jan-2024 for *every* site regardless of when it
  opened. LT year-2 months are mislabelled `2026` in the raw file and are
  relabelled to `2025` from `month_number`.
* **US-only** — points outside the US bounding box (e.g. Australian sites
  that leaked in) are dropped.
* **Dropped future-dated rows** strictly after `OBSERVED_CUTOFF_YM = 2025-12`.
* **`operational_start_date`** — MT missing dates are back-calculated from
  `age_on_30_sep_25`; LT missing dates fall back to the first observed
  month (should not fire once chem is excluded).
* **Partial first month** (op date day ≥ 15 in the open month) is dropped
  so the ramp curve isn't biased downward.
* **Floor** the bottom 0.5% of wash counts; **drop sites with < 3 months**;
  short interior gaps (≤ 2 months) are linearly patched.

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
