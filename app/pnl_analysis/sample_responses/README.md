# P&L analysis — sample API responses

Plot-ready sample payloads for the two `pnl_analysis` endpoints, for frontend wiring.
Shapes match exactly what [market.py](../modelling/market.py) returns (numbers are synthetic).

All series are **parallel arrays** — `x[i]` pairs with `y[i]`, `months[i]` with `total_med[i]`,
`dates[i]` with `values[i]`, etc. — so the frontend can feed them straight into a line chart.
Missing months come through as `null` (skip/gap them, don't plot as 0).

---

## 1. `POST /pnl_analysis/explore-market` — local-market explorer (Tab 1)

**Request** (provide `latitude`+`longitude` OR `address`):
```json
{ "latitude": 33.7489, "longitude": -84.3879,
  "metric": "tot_wash_count", "radius_km": 20, "smoothing": 1,
  "max_sites": 10, "x_axis": "date" }
```
`metric` ∈ `tot_wash_count | mem_wash_count | ret_wash_count | tot_revenue | mem_share_wash`.
`x_axis` ∈ `date | months_since_open`.

**Samples**
- [explore_market.json](explore_market.json) — `x_axis: "date"` (x = `"YYYY-MM-DD"` strings).
- [explore_market_months_since_open.json](explore_market_months_since_open.json) — `x_axis: "months_since_open"` (x = integer month offsets from each site's open; aligns ramps).

**Response fields**
- top-level `metric_label`, `x_axis_label`, `y_axis_label` → ready-to-use chart axis labels.
- `n_sites_in_market` / `n_shown` / `n_entrants` → header counts (`max_sites` caps lines drawn; every entrant is always kept).
- `series[]` → one line per site:
  - `role`: `"entrant"` (opened after the market's first site) vs `"incumbent"` — style differently.
  - `dist_km`, `op_start` ("YYYY-MM"), `name`, `site_key`.
  - `x` / `y` → the plotted points (`y` may contain `null`).
- `entry_markers[]` → vertical markers (`op_start` = "YYYY-MM-DD") to annotate when each entrant opened.

---

## 2. `POST /pnl_analysis/pinpoint-forecast` — drop-a-pin 5-yr forecast (Tab 2)

**Request**:
```json
{ "latitude": 33.7489, "longitude": -84.3879, "brand": "812",
  "mem_growth_pct": 0, "ret_growth_pct": 0, "horizon_months": 60 }
```

**Samples**
- [pinpoint_forecast.json](pinpoint_forecast.json) — normal case, neighbours present.
- [pinpoint_forecast_no_neighbours.json](pinpoint_forecast_no_neighbours.json) — empty market (`has_neighbours: false`, empty history, `without_new_site` all 0).

**Response fields**
- `summary` → KPI cards: `plateau_med/lo/hi` (mature washes/mo), `mem_share`, `n_neighbours_20km`,
  `brand_known`, `ramp_source`, applied `mem_growth`/`ret_growth` (decimal, e.g. 0.031 = +3.1%/yr).
- `trajectory` → **the new site's own** monthly curve (chart A), x = `months` (0..horizon):
  - `total_med` with `total_lo`/`total_hi` (P10–P90 band), plus `mem_med` / `ret_med` split.
- `market_forecast` → **the whole local market** total wash count, history + forecast (chart B):
  - `history.dates` / `history.values` → actuals (values may be `null`).
  - `forecast.dates` → future months; then four parallel series:
    - `without_new_site` — market trend if nothing opens (baseline).
    - `with_new_site` — market incl. the entrant, net of cannibalization.
    - `band_lo` / `band_hi` — CI band around `with_new_site`.
    - `new_entrant_journey` — the entrant's contribution alone (= `trajectory.total_med`).
  - `cannib` `{a, L, fallback}` — learned retail cannibalization `a·exp(-d/L)` for the region.
  - `net_change_year5` — headline: net wash delta at horizon (`with − without`).
