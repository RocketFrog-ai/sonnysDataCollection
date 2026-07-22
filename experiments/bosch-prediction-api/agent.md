# Bosch Prediction API — implementation plan (IMPLEMENTED — see §8 for what shipped and what was found)

Source: impromptu meeting, 2026-07-22 (Dhruv Sood + Lakshya Tomar). Fathom recording:
https://fathom.video/calls/756359481

## 1. Objective

Add a new API endpoint that reproduces Rafal's "Bosch" proforma car-wash-volume prediction —
currently a manual Excel formula — so the front end (Amit) can render it as a bar graph
(Year 1–5 wash-count estimate) alongside the existing pin-based 5-year forecast.

Inputs are two blocks Amit's UI will collect from the user per pin:
- **Site-specific factors** — 10 categorical choices (e.g. "Area Profile: Shopping"), each with a
  fixed numeric "site score" weight baked into the Excel.
- **Demographic components** — 4 numeric inputs (avg household size, % population 25–65, % HH
  income > $35k, base car-wash price), each scored against a fixed target via formula.

The API combines these into a Cumulative Site Score + Cumulative Demographic Score, derives
Year 1/2/Mature "target scores," and multiplies by a traffic-count input to produce Year 1–5
wash-volume estimates (yearly and/or monthly). No ML model — it's a deterministic formula
port, unlike the existing coldstart-based `/pinpoint-forecast`.

## 2. Source of truth — the Excel formulas

Location: `experiments/old-proforma-analysis/old-proforma-data/*.xlsx` (179 files, two layouts).
`experiments/old-proforma-analysis/code/extract_proformas.py` already parses the same 10 site
factors + 4 demographic components (values only, not formulas) into
`old-proforma-combined.csv` — reuse its `SITE_FACTORS` / `DEMOG` label lists as the canonical
field names.

**I opened the workbooks directly (`openpyxl`, `data_only=False`) to pull the live formulas.**
Verified on two files:
- Legacy "Sheet1" layout (7/179 files) — `Car Wash Cafe_ SONNY'S Express Proforma_069f2000004sKgFAAU.xlsx`
- Current "Input Form" layout (112/179 files, the dominant one) —
  `Sonny's Express Exterior Proforma 35952 US Hwy 19 N, Palm Harbor, FL 34684_0695w00000Y5LyUAAV.xlsx`

Both encode the **same math**, just laid out differently. Use the Input Form layout below as
canonical (it's the majority layout and the values referenced by name, not scattered cells).

### 2a. Site-selection factors (10, sum → Cumulative Site Score)

Each factor is a 4-way categorical choice; the chosen option's weight is picked by a nested
`IF`. Weights are **not uniform across factors** — don't assume a generic 0.15/0.10/0.05/0
pattern, use the table below (pulled from actual `IF()` formulas, sheet cells F35–F53):

| # | Factor | Option 1 | Option 2 | Option 3 | Option 4 |
|---|--------|----------|----------|----------|----------|
| 1 | Area Profile | Shopping = 0.15 | Business = 0.1 | Residential = 0.05 | Industrial = −0.25 |
| 2 | Nearest Competition | One in 4mi = 0.15 | Multiple in 4mi = 0.125 | One in 2mi = 0.075 | Multiple in 2mi = −0.025 |
| 3 | Weekly Hours of Operation | >70hrs = 0.15 | 65–70hrs = 0.1 | 60–64hrs = 0.025 | <60hrs = 0 |
| 4 | Type of Site | Corner+Light = 0.15 | Corner noLight = 0.125 | Inside nearLight = 0.075 | Inside noLight = 0.05 |
| 5 | Site Accessibility | Easy In&Out = 0.15 | Easy w/DividedHwy = 0.1 | Easy one-way = 0.05 | Difficult = 0 |
| 6 | Entrance Stack Up | >20veh = 0.15 | 20–15veh = 0.125 | 14–10veh = 0.075 | <10veh = 0.05 |
| 7 | Free Vacuum Slots | >20veh = 0.15 | 12–20veh = 0.1 | <12veh = 0.05 | Coin/None = −0.25 |
| 8 | Pay Stations | ≥3 = 0.15 | 2 = 0.1 | 1 = 0.05 | Live person = 0 |
| 9 | Visibility | >500ft = 0.15 | 400–500ft = 0.1 | 300–400ft = 0.05 | <300ft = 0 |
| 10 | Traffic Speed | <30mph = 0.15 | 30–40mph = 0.1 | 40–50mph = 0.05 | >50mph = 0 |

`Cumulative Site Score = sum of all 10 chosen weights` (cell F54).

### 2b. Demographic components (4, average → Cumulative Demographic Score)

Numeric inputs scored against a fixed target (cells F57–F60):

```
avg_household_size (target 2.1):
    score = -0.1*(2.1 - value)         if value < 2.101
          = -0.5*(value - 3)           if value > 3
          = 0                          otherwise

pct_pop_25_65 (target 55%):
    score = -0.3*(0.55 - value)

pct_hh_income_over_35k (target 50%):
    score = -0.75*(0.5 - value)

base_price_carwash (baseline $5):
    score = 0.5*(5 - value)            if value < 5.01
          = -0.075*(value - 5)         otherwise
```

`Cumulative Demographic Score = sum(4 scores) / 4` (cell F61).

### 2c. Target scores

```
first_year_target  = (CumSiteScore * 0.7) * (1 + CumDemogScore) / 85
second_year_target =  CumSiteScore        * (1 + CumDemogScore) / 92
mature_target       =  CumSiteScore        * (1 + CumDemogScore) / 76
```

### 2d. Traffic count input

`base_traffic` — 24-hour bidirectional traffic count at the site (site-specific input, like the
existing site-factors' StreetLight traffic — may or may not come from the same source; confirm
with Amit whether this is manually entered or sourced from an existing dataset).

`year3_growth_pct, year4_growth_pct, year5_growth_pct` — user-entered YoY traffic growth
estimates.

**Corrected after validating against real workbooks' cached outputs (§8) — this was wrong in the
first draft.** The Input Form layout's own formula is `(1 + rate)`, so these are **additive rates**
(0.0 = flat, 0.02 = +2%), not multipliers — the "102.00%" example I'd pulled from the legacy Sheet1
layout's instructions does not apply to the dominant Input Form layout used here. Also: each year's
effective traffic is `base_traffic * (1 + that year's own rate)`, independently — years do **not**
compound off each other (`E74`'s formula multiplies `$A$70` directly, not `D74`'s result), contrary
to the legacy layout, which does chain.

```
effective_traffic_y3 = base_traffic * (1 + year3_growth_pct)
effective_traffic_y4 = base_traffic * (1 + year4_growth_pct)   # NOT chained off y3
effective_traffic_y5 = base_traffic * (1 + year5_growth_pct)   # NOT chained off y4
```

### 2e. Yearly / monthly wash volume — the actual API output

```
year1 = first_year_target  * base_traffic       * 300
year2 = second_year_target * base_traffic       * 300
year3 = (CumSiteScore*(1+CumDemogScore))/85 * effective_traffic_y3 * 300
year4 = (CumSiteScore*(1+CumDemogScore))/82 * effective_traffic_y4 * 300
year5 = mature_target * effective_traffic_y5 * 300

monthly[y] = yearly[y] / 12
```

Note the divisor ramp 85 → 92 → 85 → 82 → 76 across years 1–5 — looks like an intentional
startup→mature ramp (year 1 and year 3 both land on 85 despite different numerators; year 5's
divisor matches `mature_target`'s own 76). **This is not a typo to "fix" — replicate exactly**,
but flag it to Rafal/Dhruv as a sanity check before shipping, since it wasn't explicitly walked
through in the meeting.

`300` = assumed operating days/year (not confirmed with Rafal — verify).

**Also present in the Input Form sheet but NOT mentioned in the meeting** (rows 76–77):
`Max Daily = monthly/4.5/7*3`, `Max Hourly = max_daily / avg_daily_wash_hours * 1.3`. Probably
out of scope (transcript only asked for the Year 1–5 bar graph), but flag to Amit/Dhruv in case
the front end wants them too — cheap to include in the same response.

### 2f. Before implementing — verify formula consistency across proforma types

I only inspected 2 of 179 files (`extract_proformas.py`'s `parse_type()` distinguishes
Flexserve / Flex / Express Exterior / Express / Other). **First step of implementation should be
a quick script pass** confirming the Year 1–5 formula (§2c–2e) is identical across a sample of
each `proforma_type`, not just the Express/Express Exterior files checked so far. If a type
diverges, the API needs a `proforma_type` input to pick the right formula variant.

## 3. Existing repo pattern to mirror

Per Dhruv: build this as a new module in the same `app/pnl_analysis/` region as the existing
5-year forecast API, following its path/shape. Confirmed by reading the code:

- **`app/pnl_analysis/modelling/site_factors.py`** — closest existing analog. Same shape needed
  here: pin/request in, dict out, wired through `router.py` + `cached()`. Note it already surfaces
  demographics + competitors for a pin from `experiments/council/data/Council--site-wise-data.csv`
  — **check whether that source can supply this API's demographic/traffic inputs automatically**,
  instead of requiring Amit to collect them all manually. Worth raising with Amit/Dhruv: which of
  the 10 site factors + 4 demographics + traffic count are user-entered on the front end vs.
  auto-fillable from data we already have.
- **`app/pnl_analysis/modelling/market.py`** (`pinpoint_forecast`) — the existing "5-year monthly
  trajectory" API. Same output shape family (Year 1–5 / monthly arrays) — model the new response
  schema similarly so Amit's charting code can reuse patterns.
- **`app/server/router.py`** — add a new route under `router = APIRouter(prefix="/pnl_analysis")`,
  e.g. `POST /pnl_analysis/bosch-forecast`, decorated with `@cached("bosch-forecast")` like every
  other endpoint here. Delegate to the new engine module; keep the handler a thin delegator per
  this file's own docstring convention.
- **`app/server/schemas.py`** — add a `BoschForecastRequest(BaseModel)` (does NOT need to inherit
  `_PinRequest` — this endpoint's inputs are the manual site-factor selections + demographic
  values + traffic count, not a lat/lon lookup, unless we decide to auto-fill from
  `site_factors.py`'s pin lookup per the point above).
- New engine module: `app/pnl_analysis/modelling/bosch_forecast.py` (or similar name — confirm
  with team; "Bosch" is Rafal's internal name, may not be what should ship in code/API naming).
  Pure function(s) implementing §2a–2e, no I/O beyond the request payload.

## 4. Proposed request/response contract (draft, later superseded — see §9 for the confirmed shipped contract)

```jsonc
POST /pnl_analysis/bosch-forecast
{
  "site_factors": {
    "area_profile": "shopping",              // one of the 4 options per factor, see §2a
    "nearest_competition": "one_in_4mi",
    "weekly_hours": "more_than_70",
    "type_of_site": "corner_with_light",
    "site_accessibility": "easy_in_out",
    "entrance_stack_up": "more_than_20",
    "free_vacuum_slots": "more_than_20",
    "pay_stations": "3_or_more",
    "visibility": "more_than_500ft",
    "traffic_speed": "less_than_30mph"
  },
  "demographics": {
    "avg_household_size": 2.4,
    "pct_pop_25_65": 0.55,
    "pct_hh_income_over_35k": 0.5,
    "base_price_carwash": 6.0
  },
  "traffic": {
    "base_traffic": 29250,
    "year3_growth_pct": 1.0,
    "year4_growth_pct": 1.0,
    "year5_growth_pct": 1.0
  }
}
```

```jsonc
{
  "cumulative_site_score": 0.475,
  "cumulative_demographic_score": 0.021,
  "yearly": {"year1": 51234, "year2": 58021, "year3": 60112, "year4": 61890, "year5": 65210},
  "monthly": {"year1": 4269.5, "year2": 4835.1, "...": "..."}
}
```

Open decision (Dhruv left it to us): **return both yearly and monthly** rather than picking one —
cheap given monthly is just yearly/12, and avoids a second round-trip with Amit if the front end
wants one and Amit assumed the other.

## 5. Validation plan

Before treating the port as correct: pick 3–5 proforma files spanning different `proforma_type`s,
extract their actual input values + their Excel-computed Year 1–5 outputs (via `data_only=True`,
which reads Excel's last-cached values), feed the same inputs through the new Python
implementation, and diff. This repo has a strict "prove you changed no numbers" culture
(`scripts/smoke.sh`) — hold this port to the same bar: exact match against Excel-computed values,
not just "looks about right."

## 6. Open items to confirm before/while building

1. **Yearly vs monthly** — recommend shipping both (see §4); confirm Amit's bar graph only needs
   one, in which case drop the other to simplify the contract.
2. **Formula consistency across proforma types** — §2f, do this first.
3. **Which inputs are manual vs auto-filled** — can `site_factors.py` / `experiments/council`
   data supply any of the 10 site factors, 4 demographics, or the traffic count automatically, or
   is Amit collecting all 14 inputs as a form? Changes whether this is a pure-compute endpoint or
   also does a pin lookup like `site_factors.py`.
4. **The 300 constant and the divisor ramp (85/92/85/82/76)** — replicate as-is (§2e), but flag to
   Rafal/Dhruv since neither was explained in the meeting.
5. **Naming** — "Bosch" is Rafal's/Dhruv's shorthand from the meeting; confirm actual endpoint/
   module naming before merging (e.g. `bosch_forecast` vs. `proforma_volume_forecast` vs. other).
6. **Where this lives relative to `proforma/pnl/`** — per `CLAUDE.md`, math shared between the
   Streamlit UI and the API belongs in `proforma/pnl/`; math only the API needs belongs in
   `app/pnl_analysis/modelling/`. Since this is a new, API-only feature with no Streamlit
   consumer yet, `app/pnl_analysis/modelling/` is the right home — revisit if the UI later wants
   this too.

## 7. Out of scope for this pass

- Max Daily / Max Hourly derived metrics (§2e) — flag, don't build unless requested.
- Any ML/coldstart-model involvement — this is a deterministic formula port, not a model.
- Changing/"fixing" the divisor ramp or the 300 constant — replicate faithfully; raise questions
  separately, don't silently correct.

## 8. Implementation notes — what shipped, and what §2/§5/§6 got wrong before validation

**Shipped:**
- `app/pnl_analysis/modelling/bosch_forecast.py` — pure formula engine (§2a-2e as corrected below).
- `app/server/schemas.py` — `SiteFactorsInput` + `BoschForecastRequest`.
- `app/server/router.py` — `POST /pnl_analysis/bosch-forecast`, **not wrapped in `@cached(...)`**:
  that decorator (`app/server/cache.py`) unconditionally calls `service.resolve_lat_lon` on the
  request and 400s if it finds no `latitude`/`longitude`/`address` — it assumes every route is
  pin-driven. This request isn't, so caching was dropped rather than faked; the computation is
  pure and cheap enough not to need it.

**§2f resolved** — swept all 179 files by `proforma_type`. The site-score/demographic/target-score/
divisor-ramp formula (§2a-2c, 2e) is **identical** across Express Exterior, Flexserve, Express and
Other. Only the "300" constant varies per file (300×88, 280×24, 330×6, 310×1) — confirmed it's a
per-site editable assumption, not a fixed constant, so it shipped as `operating_days_per_year`
(default 300) rather than hardcoded.

**§2d was wrong — corrected by validating against real cached Excel outputs.** I initially assumed
(copying the legacy Sheet1 layout, which the dominant Input Form layout does NOT match):
- `year{3,4,5}_growth_pct` are multipliers (1.0 = flat) — **wrong**. The Input Form layout's own
  formula is `(1 + rate)`, so the input is an **additive rate** (0.0 = flat, 0.02 = +2%). Shipped
  with corrected default `0.0`.
- effective traffic compounds year-over-year (`y4 = y3_effective * y4_rate`) — **wrong**. Each
  year's formula (`D74`/`E74`/`F74`) independently multiplies the same base `A70`, e.g.
  `E74 = (...)/82*($A$70*(1+C$70))*300` uses `$A$70` directly, not `D74`'s traffic figure. Years do
  **not** compound. Shipped corrected.

Both were caught by writing a throwaway validation script (not committed) that read a real
workbook's own chosen site-factor options + demographic/traffic inputs, ran them through the new
Python engine, and diffed against that same workbook's Excel-cached Year 1-5 values — exactly the
check §5 called for. Recommend re-running an equivalent check (openpyxl, `data_only=True` for
expected outputs vs. `data_only=False` for inputs) after any future change to this formula.

**Validation result:** ran this diff against every "Input Form"-layout file with complete inputs
(111 of 119 Express-Exterior-type files qualified — others lacked one of the 14 inputs, e.g. no
site-factor chosen or a blank traffic count). **106/111 matched exactly** (site score, demographic
score, and all 5 yearly figures within 1e-4 relative tolerance). The 5 mismatches were traced to
individual workbooks where the preparer had hand-edited the formula itself, not a code bug:
- 4 files changed the "Base Price of Car Wash" target from $5 to $10 directly in the `F60` `IF()`
  formula (`IF(E60<10.01,...)` instead of `IF(E60<5.01,...)`) — a per-site formula edit the
  standard port can't detect from a values-only input.
- 1 file had cell `F60` overwritten with a literal `0` instead of its formula.

This is a real, if narrow, limitation: **the API assumes the standard, un-edited formula.** If
Rafal/Dhruv confirm some sites intentionally use a different base-price target (or other hand-edited
constants), that should become an explicit request parameter rather than a silent per-file
divergence — flag to them before this ships broadly.

**Still open** (unchanged from §6, not resolved by this pass): which of the 14 inputs are meant to
be manual (Amit's form) vs. auto-filled from `site_factors.py`/council data; the exact
yearly-vs-monthly decision; the "Bosch" naming; and the unexplained 300-constant / divisor-ramp
semantics (replicated faithfully, still not explained by anyone on the call).

## 9. Field/option naming — aligned to the real front-end contract (resolved)

§4's draft used made-up shorthand key strings (`one_in_4mi`, `hours_65_70`, `veh_20_15`, …) since
no front-end contract existed yet. Amit's actual `FACTORS` config (a `FactorConfig[]` array with
`name`/`value`/`score` per factor/option) has since been shared. Cross-checked it against §2a's
Excel-derived weight table: **every one of the 40 weights matches exactly** — good independent
confirmation the Excel reverse-engineering was right.

The **option value strings and 4 of the 10 top-level factor names did not match** the draft
(they're literal wire values checked against `Literal[...]` types — a mismatch 422s any real
request). `SITE_FACTOR_WEIGHTS` in `bosch_forecast.py` and `SiteFactorsInput` in `schemas.py` were
updated to match Amit's config verbatim, snake_cased:

| Old (draft) field | Now (matches front end) | Old option values (draft) | Now (matches front end) |
|---|---|---|---|
| `nearest_competition` | *(same)* | `one_in_4mi`, `multiple_in_4mi`, `one_in_2mi`, `multiple_in_2mi` | `one_in_4_miles`, `multiple_in_4_miles`, `one_in_2_miles`, `multiple_in_2_miles` |
| `weekly_hours` | `weekly_hours_category` | `hours_65_70`, `hours_60_64` | `65_to_70`, `60_to_64` |
| `type_of_site` | *(same)* | `corner_with_light`, `corner_no_light`, `inside_near_light`, `inside_no_light` | `corner_lot_with_light`, `corner_lot_without_light`, `inside_lot_near_light`, `inside_lot_no_light` |
| `site_accessibility` | *(same)* | `easy_in_out`, `easy_in_out_divided_hwy`, `easy_one_way`, `difficult` | `easy_in_easy_out`, `easy_in_out_divided_highway`, `easy_in_or_out_one_way`, `difficult_in_and_out` |
| `entrance_stack_up` | `entrance_stack_up_area` | `more_than_20`, `veh_20_15`, `veh_14_10`, `less_than_10` | `more_than_20_vehicles`, `20_to_15_vehicles`, `14_to_10_vehicles`, `less_than_10_vehicles` |
| `free_vacuum_slots` | `number_of_free_vacuum_slots` | `more_than_20`, `veh_12_20`, `less_than_12` | `more_than_20`, `12_to_20`, `less_than_12` |
| `pay_stations` | `number_of_pay_stations` | `three_or_more`, `two`, `one` | `3_or_more`, `2`, `1` |
| `visibility` | *(same)* | `more_than_500ft`, `ft_400_500`, `ft_300_400`, `less_than_300ft` | `more_than_500_ft`, `400_to_500_ft`, `300_to_400_ft`, `less_than_300_ft` |
| `traffic_speed` | *(same)* | `less_than_30mph`, `mph_30_40`, `mph_40_50`, `more_than_50mph` | `less_than_30_mph`, `30_to_40_mph`, `40_to_50_mph`, `more_than_50_mph` |
| `area_profile`, `visibility`, `traffic_speed`, `type_of_site`, `site_accessibility`, `nearest_competition` | *(names unchanged — already matched)* | | |

Re-validated after the rename: same real workbook (South Charleston, WV) through the pure engine
AND through the live HTTP `TestClient` round-trip — both still match that file's Excel-computed
Year 1-5 output exactly. Router/response shape is unaffected (`router.py` just calls
`req.site_factors.model_dump()`, so it doesn't hardcode any of these field names).

**Still unconfirmed**: whether the actual JSON the front end sends uses these snake_case field
names as-is, or whether Amit's `FactorConfig.name` values (`areaProfile`, `weeklyHoursCategory`, …)
get sent as camelCase and need a translation layer. This repo's other request schemas are all
snake_case natively (no casing middleware observed), so snake_case was kept for consistency — but
this should be confirmed against Amit's actual fetch/request code before this ships. If the wire
format turns out to be camelCase, each field needs a pydantic `alias` (small change, not a redesign).
