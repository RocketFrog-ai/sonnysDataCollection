# Model 4 — Cold-start + Warm-start approach

Predicting `wash_count_total` for the first 60 months of a car wash site.

Two cohorts modeled separately:
- **LT2Y** — sites ≤ 2 years old → predicts months 0–23
- **GT2Y** — sites > 2 years old → predicts mature months 25+

## Problem

The launch-decision question: *given a planned site's location, demographics, and local peers, what monthly wash counts will it do?* No operational data exists yet for the new site.

## Approach overview

Three layered prediction paths, each more demanding of data:

| Path | Data needed | Annual hit ±25k (best) |
|---|---|---|
| **Cold-start** | location + demographics + peer history | lt2y 35% · gt2y 31% |
| **Cold + peer-warmup-proxy** | + peers' first 6 months observations | lt2y 36% · gt2y 31% |
| **Warm-start** | + site's own first 6 months actuals | lt2y 52% · gt2y 87% |

Warm-start is only usable *after* a site has been open 6 months. For pre-launch underwriting, cold-start (with or without proxy) is the truthful number.

## Peer grouping (how `dg_peer_ids` and the tiers were built)

We iterated through several clustering ideas before landing on a hybrid that respects local density. Short version of each:

1. **DBSCAN at 6/12/18 km radii** — first attempt. Too sparse: 47% of sites had no peer within 18 km. Singletons everywhere.
2. **Pure k-NN connected components** — opposite problem: nearest-neighbor chains hop coast-to-coast, collapsing 700 sites into 1–2 giant components.
3. **Hybrid k-NN with max-distance cap** — better, but still chain-y. Florida and Texas merged into one blob.
4. **CBSA + H3 hex bins** — CBSA gives 69–75% coverage but many singleton CBSAs (only one Sonny's site in that metro). H3 r=4 (~52 km) is more uniform.
5. **Per-site adaptive k-NN** — for each site, take ALL peers within 50 km, expand outward if fewer than 3. Reflects local density honestly: dense Jacksonville gets ~22 peers, rural Wyoming gets 3 far ones.

We kept three layered peer sets in the CSV, each useful for a different purpose:

### A. Tiered hyperlocal (`tier_*`)
Concentric rings around each site. First tier with ≥ 3 peers is the **recommended_tier**.
- `tier_neighbor` — ≤ 8 km (truly local micro-market)
- `tier_district` — ≤ 20 km (sub-metro)
- `tier_metro` — ≤ 50 km (metro-wide)

Used in the model as density features (`tier_*_count`, `tier_*_max_km`). Each tier has counts and the actual farthest peer distance.

### B. Behavioral peer set (`fg_*`) — built but not used in production
Geographic candidate pool within 100 km, ranked by **behavioral z-distance** (log avg wash count, membership share). Top-5 kept. Discarded for modeling because (a) it leaks the target via wash-count similarity and (b) `dg_*` was more honest.

### C. Demographic + geographic peer set (`dg_*`) — the modeling peer set
**This is what the model actually uses.** For each site:
1. Get the **geographic candidate pool**: all CONUS sites within 100 km (excludes AUS / PR sites)
2. Build a 6-dim demographic feature vector from ACS zip-level data: median household income, per-capita income, population, median age, median home value, % owner-occupied. Log-scale the dollar values, then z-standardize.
3. Rank candidates by **Euclidean distance in the standardized demographic space**.
4. Keep the **top 5 most similar peers** (`dg_peer_ids`).
5. Record peer-set quality: `dg_peer_count`, `dg_peer_avg_km`, `dg_peer_max_km`, `dg_peer_demo_dist`.

Why this works: ranks peers by "same kind of market" (income, density, age) while constraining to ≤ 100 km so they share weather, regional trends, and labor market. Two sites in similar demographics but on opposite coasts are NOT peers under this definition; that's intentional.

**Fallbacks for sparse areas:**
- If fewer than 5 sites are within 100 km, take whatever exists (`dg_tier = "geo_short"`)
- If 0 sites are within 100 km, fall back to the 5 nearest by pure distance (`dg_tier = "geo_expanded"`)

For LT2Y: 45% of sites get a clean `demo_geo` match, 40% `geo_short`, 15% `geo_expanded`. For GT2Y: 67% / 23% / 9%.

## Data hygiene

Before modeling, two filters drop noise:
1. `(msl == 0) AND (op_date.day > 15)` — partial-month launch rows where wash counts only reflect a few days
2. `wash_count_total < 500` — operational anomalies / soft-opens that don't reflect real operating volume

Dropping these takes monthly MAPE from 1500%+ down to 75–90%.

## Feature engineering

### Lifecycle (computed from `msl`, `calendar_month`)
- `age_months`, `log_age`, `age_sq`, `age_saturation = tanh(msl/12)`
- `is_early`, `is_growth`, `is_mature` — stage flags
- `month_sin`, `month_cos` — cyclic seasonality

### Geographic
- `latitude`, `longitude`
- `h3_r4` (~52 km hex), `h3_r5` (~25 km hex) — computed in-notebook from lat/lon

### Demographics (from ACS via zip)
- `median_household_income`, `per_capita_income`
- `population`, `median_age`, `median_home_value`
- `pct_owner_occupied`

### Operational
- `client_id_freq` — frequency encoding of the parent brand
- `client_type_freq` — frequency encoding of business type (lt2y only)
- `tier_neighbor_count` / `_district_count` / `_metro_count` — peer density at 8 / 20 / 50 km
- `tier_neighbor_max_km` / `_district_max_km` / `_metro_max_km` — peer reach at each tier
- `nearest_peer_km`
- `dg_peer_count`, `dg_peer_avg_km`, `dg_peer_max_km`, `dg_peer_demo_dist` — quality of the demographic+geo peer set

### Peer aggregates (computed train-only per fold)
For each (site, msl) row, look up peers at the same `msl` from training history and aggregate:
- `dg_*` from `dg_peer_ids` (top-5 demographic+geo similar peers): mean, median, std, p25, p75, count
- `h3_*` from same H3-r4 hex: mean, median, std, count
- `h5_*` from same H3-r5 hex: mean, median, std, count
- `st_*` from same state: mean, median, std, count

### Anchor (hierarchical seasonal baseline)
- `h3_seas_mean` = mean wash count for (h3_r4, calendar_month) in training data
- Fallback to `state_seas_mean`, then `nat_seas_mean` (national mean per calendar month)
- `anchor` = first non-null of the three
- `log_anchor = log1p(anchor)` — used to define the residual target

## Modeling

**Target transformation:** `y_residual = log1p(wash_count_total) − log_anchor`
- Trains on the multiplicative ratio between actuals and the seasonal anchor
- Inversion: `pred_wash = expm1(model.predict(X) + log_anchor)`

**Calibration:** post-training multiplier `calibration = sum(train_actual) / sum(train_predicted)` debiases the log → linear conversion.

**Cross-validation:** 5-fold `GroupKFold` by site (cold-start protocol). Each test site never appears in training; peer aggregates are computed from train-only history (no leakage).

**Models tested:** Baseline_anchor (anchor alone), Baseline_dg_mean (peer mean alone), Ridge, ElasticNet, RandomForest, ExtraTrees, HistGradBoost, LightGBM, XGBoost.

## Warm-start

For each held-out site, give the model its **first 6 months of actual wash counts**. Compute:
- `site_warm_mean`, `site_warm_median`, `site_warm_max`, `site_warm_std`
- `site_warm_recent3` — avg of months 4–6 (late-warmup level)
- `site_warm_growth` = (recent3 − month1) / month1
- `site_vs_anchor` = site_warm_mean / anchor

Train on full feature set + warm features. Predict months 7–60.

**Headline:** `Baseline_site_warm` (predict every future month = warmup mean) wins gt2y warm-start at **86.9% annual hit ±25k**. Mature sites are stable; once observed, the level is the prediction.

## Peer-warmup-proxy (truly cold-start)

For a planned new site we don't have its first 6 months. But we do know its 5 peers from `dg_peer_ids`, and those peers' first 6 months are observable in lt2y data. Compute proxy features from peers' launch trajectories:
- `peer_warm_mean`, `peer_warm_median`, `peer_warm_p25`, `peer_warm_p75`
- `peer_warm_max`, `peer_warm_recent3`, `peer_warm_n`

Adds ~1 point to lt2y annual hit ±25k (35.0% → 36.1%). Smaller lift than the standalone prototype because model 4 already has multi-scope peer aggregates that capture much of the same signal.

## Results

### Cold-start (no operational data needed)
| Cohort | Best monthly model | hit ±20% | Annual hit ±25k |
|---|---|---|---|
| LT2Y | ExtraTrees | 27.5% | 35.0% (RandomForest) |
| GT2Y | ExtraTrees | 26.6% | 31.1% (XGBoost) |

### Warm-start (requires 6 months of site data)
| Cohort | Best monthly model | hit ±20% | Annual hit ±25k |
|---|---|---|---|
| LT2Y | ElasticNet | 42.6% | 51.6% (ElasticNet/Ridge) |
| GT2Y | Baseline_site_warm | 59.0% | 86.9% (Baseline_site_warm) |

## Honest interpretation

- **Cold-start ceiling is ~35% annual hit ±25k** for first-2-years prediction. Complex models barely beat simple baselines (anchor + peer mean).
- **The bottleneck is site-specific variance** that demographics + geo can't explain. Two sites in identical zips can perform very differently.
- **Warm-start unlocks dramatic accuracy** because the site's own data reveals its level. Use for any forecasting *after* launch.
- **To push cold-start above 40% you need new data**: traffic counts, competition density (Google Places), lease/parcel attributes, pre-launch marketing spend. The dataset has been wrung of its available signal.

---

# Column dictionary — `less_than-2yrs-nochem.csv` (49 cols) and `more_than-2yrs_monthly-nochem.csv` (47 cols)

Notation: ✅ used in model · ⚠️ leaky (must guard) · ❌ dead/redundant

## Core identifiers and time
| Column | Description | Status |
|---|---|---|
| `client_id_location_id` | Unique site key | ✅ key |
| `client_id` | Parent brand / chain (e.g. `gateexpress`, `magnolia`) | ✅ frequency-encoded |
| `client_type` (lt2y only) | Business type label | ✅ frequency-encoded |
| `operational_start_date` | Date site began operating | ✅ → `msl` |
| `calendar_year`, `calendar_month` | Row month | ✅ → seasonality, `msl` |
| `age_on_30_sep_25` | Site age in months at a fixed snapshot date | ❌ redundant with `msl` |

## Target and derived (leaky)
| Column | Description | Status |
|---|---|---|
| `wash_count_total` | Total monthly washes (retail + membership) | ✅ **target** |
| `wash_count_retail` | Retail wash count | ⚠️ component of target — leak |
| `wash_count_membership` | Membership wash count | ⚠️ component of target — leak |
| `prev_wash_count` (lt2y only) | Previous month's total | ⚠️ autoregressive lag — leak for cold-start |
| `avg_monthly_total` | Site's all-time average monthly wash count | ⚠️ aggregate of target — leak |
| `membership_share` | Lifetime membership / total ratio | ⚠️ derived from target components — leak |

## Address / geography
| Column | Description | Status |
|---|---|---|
| `street` | Street address text | ❌ raw, never used |
| `city` | City name | ❌ not encoded |
| `state` | US state code | ✅ anchor fallback |
| `zip` | ZIP code | ✅ joins ACS demographics |
| `region` | Coarse US region (e.g. Southeast) | ❌ not encoded |
| `latitude`, `longitude` | Site coordinates | ✅ geo features + H3 derivation |

## Peer set: demographic + geographic (`dg_*`)
Top-5 most demographically + geographically similar peers within 100 km.
| Column | Description | Status |
|---|---|---|
| `dg_peer_ids` | Comma-separated peer site IDs | ✅ source for peer aggregates + warmup proxy |
| `dg_peer_count` | Number of peers in the set | ✅ feature |
| `dg_peer_avg_km` | Average geographic distance to peers | ✅ feature |
| `dg_peer_max_km` | Farthest peer distance | ✅ feature |
| `dg_peer_demo_dist` | Average demographic z-distance to peers (lower = more similar) | ✅ feature |
| `dg_tier` | Quality flag (`demo_geo` / `geo_short` / `geo_expanded`) | ❌ not encoded |

## Peer set: feature-aware behavioral (`fg_*`)
Top-5 most behaviorally similar peers (volume + membership share). Never picked over `dg_*` in modeling.
| Column | Description | Status |
|---|---|---|
| `fg_peer_ids` | Comma-separated peer IDs | ❌ unused (`dg_peer_ids` preferred) |
| `fg_peer_count` | Number of peers | ❌ unused |
| `fg_peer_avg_km` | Avg distance | ❌ unused |
| `fg_peer_max_km` | Max distance | ❌ unused |
| `fg_peer_feat_dist` | Avg behavioral z-distance | ⚠️ leaky — behavioral similarity derives from wash counts |
| `fg_tier` | Quality flag | ❌ unused |

## Peer set: tiered hyperlocal (`tier_*`)
Concentric local peer rings.
| Column | Description | Status |
|---|---|---|
| `tier_neighbor_count`, `tier_neighbor_max_km` | Peers within 8 km | ✅ features |
| `tier_neighbor_ids` | Peer IDs in the 8 km ring | ❌ redundant (count + max_km used instead) |
| `tier_district_count`, `tier_district_max_km` | Peers within 20 km | ✅ features |
| `tier_district_ids` | Peer IDs in the 20 km ring | ❌ redundant |
| `tier_metro_count`, `tier_metro_max_km` | Peers within 50 km | ✅ features |
| `tier_metro_ids` | Peer IDs in the 50 km ring | ❌ redundant |
| `recommended_tier` | Which tier first hits ≥3 peers (`neighbor` / `district` / `metro` / `sparse`) | ❌ not encoded |
| `nearest_peer_km` | Distance to closest peer regardless of tier | ✅ feature |

## ACS demographics (zip-level, joined externally)
| Column | Description | Status |
|---|---|---|
| `median_household_income` | $ median household income | ✅ feature |
| `per_capita_income` | $ per capita income | ✅ feature |
| `population` | Population in the zip | ✅ feature |
| `median_age` | Median age of residents | ✅ feature |
| `median_home_value` | $ median home value | ✅ feature |
| `pct_owner_occupied` | Share of owner-occupied housing | ✅ feature |

---

## Recommended cleanup

Drop these **20 columns** for a leaner dataset (no information lost from the model's perspective):

```
Leaky (6):    wash_count_retail, wash_count_membership, prev_wash_count,
              avg_monthly_total, membership_share, fg_peer_feat_dist
Redundant/dead (14):  age_on_30_sep_25, street, city, region,
                      tier_neighbor_ids, tier_district_ids, tier_metro_ids,
                      recommended_tier, fg_peer_ids, fg_peer_count,
                      fg_peer_avg_km, fg_peer_max_km, fg_tier, dg_tier
```

Remaining 29 columns are the lean working set.
