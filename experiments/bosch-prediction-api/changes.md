# Changes — Bosch Prediction API

Implements the plan in [agent.md](agent.md). For the formula derivation, validation methodology,
and full rationale, read agent.md §2, §8, §9 — this file is just the change log / file index.

## Update: field/option names realigned to the real front-end contract

`SiteFactorsInput` (schemas.py) and `SITE_FACTOR_WEIGHTS` (bosch_forecast.py) were rewritten to
match Amit's actual `FACTORS` config verbatim (snake_cased) — the original draft used made-up
shorthand strings (`one_in_4mi`, `veh_20_15`, …) that don't match the real UI's option values, which
would have 422'd every real request. All 40 weights cross-checked identical to the draft (good
confirmation the Excel-derived math was right); only the key *strings* changed. 4 of 10 top-level
factor names also changed: `weekly_hours`→`weekly_hours_category`,
`entrance_stack_up`→`entrance_stack_up_area`, `free_vacuum_slots`→`number_of_free_vacuum_slots`,
`pay_stations`→`number_of_pay_stations`. Full before/after table in agent.md §9. Re-validated
against the same real workbook (pure engine + live HTTP `TestClient`) post-rename — still exact.

**Open**: whether the wire format is snake_case (kept, matching this repo's other schemas) or
camelCase (matching `FactorConfig.name` literally) — needs confirming against Amit's actual request
code before shipping.

## Files touched

| File | Type | What |
|---|---|---|
| `app/pnl_analysis/modelling/bosch_forecast.py` | **new** | Pure formula engine: `SITE_FACTOR_WEIGHTS`, `cumulative_site_score()`, `cumulative_demographic_score()`, `target_scores()`, `bosch_forecast()`. No I/O, no FastAPI/pydantic dependency. |
| `app/server/schemas.py` | modified | Added `SiteFactorsInput` (10 `Literal`-typed factor choices) and `BoschForecastRequest`, appended at the end of the file. |
| `app/server/router.py` | modified | Added `POST /pnl_analysis/bosch-forecast` (full path `/v1/pnl_analysis/bosch-forecast`, per `app/main.py`'s `/v1` prefix), appended at the end of the file, plus the `bosch_forecast as bosch_engine` import and `BoschForecastRequest` import at the top. |

No other files were modified. Nothing in `proforma/`, `app/db/`, `app/core/`, or any existing
endpoint changed.

## Key implementation decisions

- **Not wrapped in `@cached(...)`.** That decorator (`app/server/cache.py`) assumes every request
  is pin-driven (calls `resolve_lat_lon` unconditionally, 400s without lat/lon/address).
  `BoschForecastRequest` has none of those fields, so caching was left off rather than faked.
- **`operating_days_per_year` is a request field (default `300.0`)**, not a hardcoded constant — a
  sweep of all 119 source workbooks found this value varies per file (300/280/330/310).
- **`year{3,4,5}_growth_pct` are additive rates** (`0.0` = flat, `0.02` = +2%), applied
  independently to `base_traffic` per year — **not** compounded year-over-year, and **not**
  multipliers. Both of those were wrong in the first draft; caught by validating against real
  workbooks' own cached Excel outputs (see agent.md §8 for the full story).

## Testing performed

1. **Pure-function validation** — ran the engine's output against 111 real proforma workbooks'
   own Excel-cached Year 1–5 values (`openpyxl`, `data_only=True`). 106/111 exact matches; the 5
   misses were individual workbooks with hand-edited formula cells (not a port bug) — see agent.md
   §8 for detail.
2. **Live HTTP test** — mounted the actual `schemas.BoschForecastRequest` +
   `bosch_forecast.bosch_forecast()` on a bare FastAPI app (scratch venv: `fastapi`, `pydantic`,
   `httpx`; this machine has neither the project's `sonnys` conda env nor `numpy`/`pandas`
   installed) and hit it with Starlette's `TestClient`:
   - Real workbook inputs → `200`, output matches that workbook's ground truth exactly.
   - Missing required field → `422` (not `500`).
   - Invalid site-factor enum value → `422` (boundary validation via `Literal` types works).
   - Omitted optional fields → `200`, defaults produce identical output to the explicit case.
3. **Not tested**: booting the real `app.main` / full `app.server.router` module. Importing it
   pulls in an unrelated LLM/insights dependency chain (Azure OpenAI clients, `dotenv`, DB
   drivers) that this feature doesn't use and isn't installed here. `py_compile` confirms all
   three files are syntactically valid and the router's new import resolves.

## Open items (unchanged from agent.md §6/§8 — need a person, not more code)

- Confirm "Bosch" as the actual endpoint/module naming with Dhruv/Rafal.
- Decide with Amit which of the 14 inputs are manual (front-end form) vs. auto-fillable from
  `site_factors.py` / the council dataset.
- Yearly vs. monthly — currently returns both; confirm whether the front end needs one or both.
- Flag the 5 hand-edited-formula source files to Rafal (base-price target changed from $5→$10 in
  4 of them) in case that reflects an intentional per-market variant worth exposing as a parameter.
