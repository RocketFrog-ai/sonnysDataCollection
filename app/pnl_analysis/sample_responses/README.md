# Proforma backend — API reference & sample responses

Plot-ready sample payloads + the request/response contract for the **Earnest Proforma** backend, which
serves the three Streamlit modes (`earnest-proforma-2.0/streamlits/app.py`):

| Streamlit mode | Endpoints | Module |
|---|---|---|
| 🗺️ **Explore markets** | `/explore-market`, `/explore-market/kpis`, `/operators` | [market.py](../modelling/market.py) |
| 📍 **Forecast** | `/pinpoint-forecast`, `/pnl-forecast`, `/campaign/*`, `/brands` | [market.py](../modelling/market.py) · [pnl.py](../modelling/pnl.py) · [campaign.py](../modelling/campaign.py) |
| 🔎 **Site analysis** | `/site-context` (sync) · `/analyze-site` (async) | [site_context.py](../../site_analysis/modelling/site_context.py) |

All pnl_analysis routes are mounted at **`/v1/pnl_analysis/...`**; site analysis at **`/v1/...`**.
Run the API in conda env `sonnysDataCollection`:
`uvicorn app.site_analysis.server.main:app --host 0.0.0.0 --port 8002` (from the repo root).

**Conventions.** Every pin endpoint accepts `latitude`+`longitude` **OR** `address` (geocoded server-side).
All series are **parallel arrays** (`x[i]` pairs with `y[i]`, `months[i]` with `total_med[i]`, …). Missing
months come through as `null` (skip/gap them, don't plot as 0). Numbers are plain JSON (never `NaN`).

---

## Lookups

### `GET /pnl_analysis/operators`
Operator/brand **names** for the Explore-markets "highlight operator" dropdown. → `{n_operators, operators:[name…]}`.

### `GET /pnl_analysis/brands` — [brands.json](brands.json)
Operator/brand list for the Forecast tab. Each: `client_id` (the value `pinpoint-forecast`/`pnl-forecast` key
on — the strongest plateau predictor), `client_name`, `n_sites`, `model_known` (does the cold-start model have
a learned prior for it). Sorted by site count.

---

## 1. Explore markets

### `POST /pnl_analysis/explore-market` — [explore_market.json](explore_market.json)
The local-market **MAP + 4 header metric cards** (Pin · Sites in market · New entrants · Local market). No
time series — those are the 8 KPI panels below. Pair with `GET /operators` to populate the highlight dropdown.

```json
{ "latitude": 28.929, "longitude": -82.038, "radius_km": 20,
  "max_sites": 10, "min_months": 36, "operator": null, "demo": false }
```
`min_months` keeps rich-history sites (`36` = the Explore view, ≥3 yrs — that's why the count is small).
`operator` (a `client_name` from `GET /operators`) plots that brand's whole footprint; `demo` anonymizes
(hides names, shades cluster regions instead of exact dots).

**Response** — header counts `n_sites_in_market` / `n_entrants` / `n_shown`, plus `focal_site_key` (the newest
entrant / nearest site) and `map`:
- `markers[]` — the in-market sites ("cluster points"), each `role` ∈ `focal|entrant|incumbent` with `color`,
  `is_entrant`, `lat`/`lon`, `dist_km`, `op_start`.
- `reference_dots[]` — geographic reference: other rich-history sites ≤50 km of the pin (outside the market).
- `operator_footprint[]` — every site of the highlighted `operator` (empty unless one is passed).
- `cluster_regions[]` — demo mode only: shaded local-market footprints.

### `POST /pnl_analysis/explore-market/kpis` — [explore_market_kpis.json](explore_market_kpis.json)
The 6 grouped **per-site KPI series** for the whole local market — the "Local-market KPIs over time" panels.

```json
{ "latitude": 33.7489, "longitude": -84.3879, "radius_km": 20, "smoothing": 1, "min_months": 36, "demo": false }
```
**Response** — `n_sites`, `focal_site_key`, `sites[]` (legend order, focal last), and `groups[]`:
`Washes` (total/retail/membership), `Revenue` (total/retail/membership), `ASPs` (retail/membership per wash).
Each group → `panels[]` → `{col, label, unit, series:[{site_key, name, is_focal, x, y}]}`.

---

## 2. Forecast

Both forecast endpoints take the **same body** (`latitude`+`longitude` or `address`, `brand`, `plateau_override`,
`mem_growth_pct`, `ret_growth_pct`, `horizon_months`). See [PINPOINT_FORECAST.md](../../../earnest-proforma-2.0/streamlits/PINPOINT_FORECAST.md) for the model.

### `POST /pnl_analysis/pinpoint-forecast` — [pinpoint_forecast.json](pinpoint_forecast.json)
The **new site's own** 5-year trajectory + the summary KPI cards (the "Predicted 5-year trajectory" chart).

```json
{ "latitude": 33.7489, "longitude": -84.3879, "brand": "812",
  "plateau_override": null, "mem_growth_pct": 0, "ret_growth_pct": 0, "horizon_months": 60 }
```
**Response** — `summary` (`plateau_med/lo/hi`, `mem_share`, `n_neighbours_20km`, `brand_known`, `ramp_source`,
`state`/`region`, applied `mem_growth`/`ret_growth`) + `trajectory` (x=`months` 0..H, `total_med` with
`total_lo`/`hi` band, plus `mem_med`/`ret_med` split).

### `POST /pnl_analysis/market-forecast` — [market_forecast.json](market_forecast.json) · [market_forecast_no_neighbours.json](market_forecast_no_neighbours.json)
The **total local-market** wash count: actual history + 5-year growth forecast (the "Total local-market wash
count" plot). Same body as `pinpoint-forecast`.

**Response** — `open_date`, `has_neighbours`, `history.dates`/`values` (actuals; may be `null`), and
`forecast.{dates, with_new_site, without_new_site, band_lo, band_hi, new_entrant_journey}` + `net_change_year5`.
(Cannibalization is applied internally to `with_new_site`; its params are not surfaced.) For an empty market:
`has_neighbours:false`, `history` empty, `without_new_site` all 0, `with_new_site == new_entrant_journey`.

### `POST /pnl_analysis/pnl-forecast` — [pnl_forecast.json](pnl_forecast.json) · [pnl_forecast_campaign.json](pnl_forecast_campaign.json)
The 💰 monthly **revenue vs operating expense vs net** chart, with an optional campaign overlay.

```json
{ "latitude": 33.7489, "longitude": -84.3879, "brand": null,
  "plateau_override": null, "mem_growth_pct": 0, "ret_growth_pct": 0,
  "asp_override": null, "opex_growth_pct": 0,
  "campaign_on": false, "campaign_launch": 13, "campaign_intensity": 1.0, "window": 6, "horizon_months": 60 }
```
`asp_override` = blended $/wash (null → cluster average). `opex_growth_pct` escalates opex on top of the learned
new-site ramp (default flat). **Response** — `months` 0..60; `asp{mem,ret,blend,used,scope,refs}`; `opex{mature_opex,
ramp_scope, opw_scope, ramp_hage, hist_yoy}`; `series{revenue_base, revenue, opex_base, opex, net, net_base}`
(base = no campaign); `campaign{applied, launch, intensity, window, conv_pct, mem_share_settled}`;
`summary{plateau_med, mem_share, state, region, total_revenue_5yr, total_opex_5yr, net_5yr, breakeven_month}`.

### `POST /pnl_analysis/campaign/verdict` — [campaign_verdict.json](campaign_verdict.json)
The 🎯 "should this site run a promotion?" recommendation. → `{ok, verdict_level (`recommended|marginal|
not_recommended`), verdict_text, neighbours_mem_share, n_established_incumbents, this_site_mem_share, conv_pct}`.

### `POST /pnl_analysis/campaign/eating-the-market` — [eating_the_market.json](eating_the_market.json)
📈 Your site vs each incumbent forecast forward 5 yr; with a campaign the incumbents drift down as your promo
steals share. Body adds `campaign_on/launch/intensity/window/max_incumbents`. → `{months, your_site{base,
with_campaign}, incumbents:[{site_key, name, expected[], with_campaign[]}], n_incumbents, n_shown, steal_peak}`.

### `GET /pnl_analysis/campaign/snapshot` — [campaign_snapshot.json](campaign_snapshot.json)
What a campaign does to OPEX/Revenue/Profit/Membership from the real P&L event study (book_v4). Not pin-specific.
→ `{buckets:[{bucket, title, n_campaigns, camp_months, mfs[], opex[], revenue[], profit[], mem_purchases[]}]}`
for 1-month / 2-month / 3+-month campaigns (medians by month-offset −3..6).

### `POST /pnl_analysis/campaign/local-evidence` — [local_campaign_evidence.json](local_campaign_evidence.json)
Real in-market sites' monthly series with detected promo-OPEX-spike months marked (the evidence). Body:
`metric` ∈ `mem_share_wash | mem_wash_count | ret_wash_count`, `max_sites`, `demo`. → `{metric, sites:[{site_key,
name, x[], y[], campaign_months[] (ISO dates)}]}`.

---

## 3. Site analysis

### `POST /v1/site-context`
Synchronous "what surrounds this location" for the shared lat/lon pin — weather, competing car washes, retail
anchors and gas stations + map markers + rule-based insights, all in **one** response (no task polling).

```json
{ "latitude": 33.7489, "longitude": -84.3879, "include_ai": false, "demo": false }
```
Needs `GOOGLE_MAPS_API_KEY` (competitor/retail/gas) — degrades to empty lists without it; weather (Open-Meteo)
needs no key. `include_ai` adds an internal-LLM write-up per dimension when reachable (2 s socket pre-check).
**Response** — `has_api_key`; `metrics{competitors_4mi, nearest_competitor_mi, retail_anchors_3mi, gas_stations_3mi,
comfortable_days}`; `markers[]` (origin + gas + car_wash + retail, each lat/lon/category/distance); `dimensions{
competition, retail, gas, weather }` — each carries its raw rows + a rule-based `insight{insight,pro,con,conclusion}`
(and `ai_insight` when AI was requested and available).

> The **async** pipeline (`POST /v1/analyze-site` → poll `GET /v1/task/{id}` → read `/{dimension}/data-by-task/{id}`
> + `/map/data-by-task/{id}`) keys on an `address` and is unchanged — use it for the batch/queued flow.
