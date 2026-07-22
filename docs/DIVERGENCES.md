# Divergences and known defects

Things that are wrong, duplicated, or surprising in this repo, found during the 2026-07
restructure. Most were **deliberately left alone**: the restructure was behavior-preserving, and
fixing them would have moved a number or flipped broken-to-working, each a separate decision with a
separate review. Entries that have since been *resolved* say so and are kept for the record.
Each says what it is, how it was verified, and what the fix would be.

Nothing here is a regression introduced by the restructure.

---

## 1. The P&L helper math is now shared; only the orchestration is still twice

**Largely RESOLVED (2026-07).** The forecasting/P&L math used to be implemented **twice** — once
in-process in the Streamlit UI, once as a port under `app/pnl_analysis/modelling/`. A drift map (every
one of the 21 twin helper functions run through both sides on the golden pins) found them numerically
identical on every reachable path — the differences were docstrings, a `state_to_region` refactor,
caching, and two unreachable fallbacks. So the twins were extracted into one Streamlit-free package:

```
proforma/pnl/    data.py  trend.py  opex.py  campaign.py   — ONE copy of the shared math
   ▲                                    ▲
   │ import                             │ import
app/pnl_analysis/modelling/        proforma/ui/panels/
   (JSON endpoints)                    (Streamlit render)
```

Both sides now import the same `data` loaders, `trend` (market growth / seasonal forecast),
`opex`/`asp` helpers, and `campaign` helpers. The extraction was behaviour-preserving to 1e-9
(`model.json` / `api.json` / `ui_render.json` all unchanged).

**What is still separate, on purpose:** the *orchestration* — the API's `pnl_forecast` / `expense_plan`
/ `explore_market` (which return JSON) versus the UI's `render()` (which draws widgets). These have the
same math underneath (now the shared helpers) but different I/O, so they cannot be one function. §1a
below is the one place their glue still differs (the ASP *override*).

Also genuinely shared, as before: `proforma/models/coldstart.py` (`predict_site`,
`predict_neighbours`, `assign_clusters`, the cannibalization fit) — plateau × ramp × cannibalization
is computed once.

**These are agreed** (checked, identical): `PNL_EXCLUDE = {"alpinecarwash_000087"}`; the ASP>200
revenue-nulling in the panel loader; `ASP_MIN_WASH = 200`, `ASP_FLOOR_MEM = 4.0`,
`ASP_FLOOR_RET = 5.0`; and the row predicate inside `_drop_corrupt_asp_rows` — these now live once in
`proforma/pnl/opex.py`.

### 1a. Default revenue is priced per-leg on both sides; only the ASP *override* diverges

This entry has been wrong twice, in opposite directions, so it is worth stating carefully with the
code pinned.

**The default revenue is identical, and it is per-leg.** Both sides price the membership and retail
legs each with the cluster's own ASP, every month:

| | |
|---|---|
| UI (`_pinpoint_forecast.py:1244`) | `asp_mem_pp = per_year(mem_asp_in, cl_mem_pp)`, `asp_ret = per_year(ret_asp_in, cl_ret)`; `revenue = mem_purch × asp_mem_pp + ret × asp_ret` = `mem × cl_mem + ret × cl_ret` at default |
| API (`pnl.py:316–318,350`) | `asp = asp_override if given else asp_blend`; `k_asp = asp / asp_blend`; `asp_mem, asp_ret = cl_mem × k_asp, cl_ret × k_asp`; `rev_base = mem × asp_mem + ret × asp_ret` |

At the default (`asp_override=None`) the API's `asp = asp_blend`, so `k_asp = asp_blend/asp_blend = 1`
and `rev_base = mem × cl_mem + ret × cl_ret` — the same per-leg formula as the UI. **`asp_blend`
never prices the default revenue; it cancels.** It is only the *recommended* single number shown as
the slider default (schema calls `asp_override` "Blended $/wash") and is reported back as `asp.blend`.
So `mem_share` / `asp_blend` being a mature scalar has no effect on the forecast unless the user
overrides the ASP. (An earlier version of this entry claimed the API computed
`revenue = (mem+ret) × asp_blend` and tabulated a month-by-month gap; that was against an older
`pnl.py` and is false today.)

The cluster ASPs themselves match: the UI's `global_healthy_asp` returns `(cl_mem_pp, ppw, cl_ret)`
in dollars per membership *purchase*, the API's returns `(cl_mem, cl_ret)` in dollars per *wash*, and
`cl_mem_pp × ppw = Σmem_revenue/Σmem_purchase × Σmem_purchase/Σmem_wash = Σmem_revenue/Σmem_wash =
cl_mem` (measured `3.6e-15` apart). Both derive the cluster ASP from `Σrevenue/Σwash` over the
≤20 km neighbours' last 12 months — **not** from the panel's direct `ASP_mem`/`ASP_ret` columns
(those are used only for the per-site median reference table, `data.py:173` / `asp_refs`).

**The genuine divergence is the ASP *override*.**

| | |
|---|---|
| UI | two independent per-year sliders: membership in **$/membership purchase**, retail in **$/wash**. Each reprices its own leg; the mem/retail ratio can be changed. |
| API | one **blended $/wash** (`asp_override` scalar, or `asp` per-year map). It scales *both* legs by `k_asp = override/asp_blend`, holding the mem/retail ratio fixed. |

So the API cannot express an independent membership-vs-retail reprice, and the number a user reads
off the UI's membership slider (`$/purchase`) is not the same unit as the API's `asp_override`
(`$/wash`). A client forwarding the Streamlit slider value into `asp_override` will get the wrong
revenue. That is the only ASP behaviour that actually differs between the two implementations.

**Would change:** forecast revenue, hence the P&L, breakeven, and the campaign ROI. Not fixed here:
picking a winner moves numbers people have signed off on, and the Streamlit side has no golden
baseline to verify the move against (§6).

### 1b. ~~`express_only` exists only in the UI~~ — RESOLVED (2026-07)

The API now accepts `express_only` on the 8 pin-based modelling requests
(ExploreMarket, ExploreKpis, PinpointForecast, PnlForecast, ExpensePlan, CampaignVerdict,
EatingMarket, LocalCampaigns; not on `/insights/*`, which annotate rather than model). It applies
the UI's two filters in the UI's order — `primary_carwash_type == "Express Tunnel"`, then
`n_obs >= 30` — re-clusters the subset, and passes `anchor_keys` to
`coldstart.predict_site` so only the level anchor is scoped while the LightGBM neighbour features
stay on the full site set.

Verified against the UI in both conda envs: **742 sites, 35,778 rows, identical `site_key` sets and
identical cluster labels.** At `false` every number is bit-for-bit unchanged —
`api.json` and `model.json` still match the frozen baselines.

**Defaults diverge by tab (2026-07):** the two Explore-markets requests (ExploreMarket, ExploreKpis)
now default `express_only` to **`true`** — market exploration is express-first — while the 6
Forecast-tab requests keep `false`. The golden capture pins `express_only=false` on the explore
cases (`scripts/_golden/capture_api.py`), so the frozen all-sites numbers still verify. Relatedly,
the `/insights/*` prompts are scoped to the express/tunnel segment, and two insight requests grew
their own `express_only: true` default — on `/insights` it grounds Key Insights on the express-only
panel, on `/insights/competition` it counts only express sites as the client portfolio; neither
touches a modelled number. `/insights/competition` analyses express/conveyor-tunnel rivals only
(the all-types count survives as one labelled context line), states the trade-area + Places radii in
its summary, and is grounded on a Google Places nearby fetch (any wash type, express-keyword-tagged;
see `app/core/places/nearby_competitors.py`).

### Why this is not fixed here

Unifying means picking a winner at each disagreement, and each choice moves a number someone has
already signed off on. It is the *next* project, and it needs a golden baseline over the Streamlit
P&L first. Today `scripts/smoke.sh` freezes the **API** numbers exactly and the UI only at its
first render (§6), so a unification done now would be unverifiable on the side that matters most.

---

## 2. ~~The Celery worker cannot start~~ — RESOLVED by removing Celery (2026-07)

*Kept for the record; the defect and the subsystem are both gone.*

`app/tasks/celery_app.py` declared `include=[..., "app.pnl_analysis.modelling.zeta_pnl", ...]`.
That module did not exist, and did not exist at the `pre-refactor` tag either — it was deleted in
`814fa37` ("cleaning dead code") without updating the list. Verified by doing exactly what a worker
does at startup:

```
>>> celery_app.loader.import_default_modules()
ModuleNotFoundError: No module named 'app.pnl_analysis.modelling.zeta_pnl'
```

So `celery -A app.tasks.celery_app worker` could not boot. `POST /v1/analyze-site` enqueued work
nothing would ever run, and the caller polled forever. Nobody noticed because the synchronous
`POST /v1/site-context` already did the same job in one call.

**Resolution.** Celery was removed rather than repaired, on the reasoning that a subsystem broken
for months with a working synchronous replacement is not a subsystem anyone depends on. Deleted:
`app/tasks/`, `app/site_analysis/modelling/site_analysis.py`, `scripts/start_celery_worker.sh`,
`scripts/stop_celery_worker.sh`, `scripts/test_weather_api.py`, the `celery` and `redis` deps, and
these 12 endpoints:

```
POST /v1/analyze-site
GET  /v1/task/{task_id}
GET  /v1/result/{task_id}
GET  /v1/{weather,competition,retail,gas}/data-by-task/{task_id}
GET  /v1/{weather,competition,retail,gas}/summary-by-task/{task_id}
GET  /v1/map/data-by-task/{task_id}
```

Removing Celery and (later) the whole `site_analysis` subsystem left the forecast API untouched:
`app.main` now serves 18 OpenAPI paths, 17 under `/v1/pnl_analysis/*` (see `docs/ARCHITECTURE.md`),
and `model.json` / `api.json` were unchanged, so no forecast number moved.

**Residue:** the `GET /v1/cache/site-analysis/all` cache route that the async pipeline served was
removed with the subsystem — no route reads Postgres today. The `REDIS_*` / `CELERY_*` keys may
still sit in your `.env`; nothing reads them.

---

## 3. Two different files named `site_carwash_types.csv`

| path | bytes | what it is |
|---|---:|---|
| `proforma/data/ref/site_carwash_types.csv` | 1,306,923 | the **resolved** table the model reads |
| `libs/carwash_type/data/site_carwash_types.csv` | 1,302,501 | the classifier's **raw output** |

They are not copies of each other (different sha256, different size). The `proforma/data/ref/`
copy has been further resolved — note the sibling `site_carwash_types.csv.bak-pre-resolved` and
`unknownsites_resolved.csv`. They were **not** merged during the restructure. Two files that share
a name are not the same file.

By contrast, the three panel CSVs that *were* duplicated across the old `earnest-proforma-2.0/data`
and `earnest-proforma-final-1.6/data` trees (`main-data-v2.csv`, `-processed`, `-stitched`) were
verified byte-identical by sha256 before the duplicate was removed. See `docs/DATA.md`.

---

## 4. `predict_site` does not return what its docstring says

`proforma/models/coldstart.py::predict_site` is documented as:

> Return DataFrame[month, total_med/lo/hi, mem, ret]

It actually returns a **2-tuple** `(DataFrame, meta_dict)`, where `meta_dict` carries the anchor
internals (`anchor_level`, `n_local_mature`, `calib_applied`, `proxy_used`, `local_cov`,
`plateau_med/lo/hi`, `model_kind`, …). Callers already unpack the tuple, so only the docstring is
wrong. Left as-is to keep the diff to path changes; worth a one-line fix.

---

## 5. Model 3 (`model_kind="et"`) is not bit-reproducible

The pickled ExtraTreesRegressor carries `n_jobs=-1`. scikit-learn's forest `predict` accumulates
each tree's contribution into a shared buffer from a joblib thread pool, so the **summation order
varies between runs** and the float64 last bits move. Measured spread across runs:

```
max relative deviation 1.944e-15   (~9x float64 epsilon), confined to model_kind="et" cases
```

`OMP_NUM_THREADS` does not fix this — it governs BLAS/OpenMP, not joblib's thread pool.

This is why the golden harness pins `n_jobs=1` **at capture time only**
(`scripts/_golden/capture_model.py`), and why the acceptance tolerance is `1e-9` rather than
bit-exactness. Production is untouched and still runs `n_jobs=-1`. If you ever need bit-exact
forecasts, set `n_jobs=1` on the estimator before `predict`, at a real throughput cost.

---

## 6. Coverage gaps in the golden harness — written down, not papered over

`scripts/smoke.sh` proves a lot, and these things it does **not** prove:

- **The Streamlit UI is only covered on its first render.** `scripts/_golden/capture_ui.py` uses
  `streamlit.testing.v1.AppTest`, which genuinely *executes* the script body (imports, data load,
  model load, widget construction) and freezes the rendered widget surface. But widgets that appear
  only after a user picks a mode, drops a pin, or clicks a button are never exercised, and nothing
  compares pixels. A layout or interaction regression will not be caught.
- **`/v1/pnl_analysis/insights/*` are excluded** — they call an LLM and are non-deterministic by
  construction. They are documented as annotating, never altering, modelled numbers.
- **`proforma/backtests/**` is `ast`-parsed but never imported.** Those scripts read data and fit
  models at module scope.
- **The five `/insights/*` endpoints have no golden outputs** and are the only remaining
  third-party surface. They were exercised live once, by hand, and all returned 200.

---

## 7. ~~Six P&L functions are dead in the UI and live in the API~~ — RESOLVED (2026-07)

`regional_opex`, `opex_per_wash`, `opex_ramp`, `opex_pct_fit`, `opex_trend_hist`, and `asp_refs`
were defined in the UI's `_pinpoint_forecast.py` and called **nowhere** in `proforma/`, while their
API namesakes were live. When §1's helpers were extracted into `proforma/pnl/`, these were resolved
together: the UI now imports only the three it actually calls (`_drop_corrupt_asp_rows`,
`global_healthy_asp`, `opex_pct_curve_fit`) from the shared package, and the dead defs were deleted
(the UI panel dropped ~430 lines). The single canonical copy of each lives in `proforma/pnl/opex.py`.

Related: `proforma/ui/site_analysis_page.py` was **not reachable** from `app.py`'s `MODES`
dispatch. It was deleted in 2026-07, and the whole `app/site_analysis` subsystem with it.

---

## 8. Sundry

- **`test_endpoint.py` (repo root) was broken for years and has been deleted.** It imported
  `get_competitors_dynamics_endpoint` and `CompetitorsDynamicsRequest`; `git grep` at the
  `pre-refactor` tag shows neither symbol ever existed. It went with the `site_analysis` subsystem
  it referenced.
- **`app/pnl_analysis/modelling/data.py`'s docstring** used to claim `load_panel()` reads
  `main-ds.csv`. It reads `main-data-v2-stitched.csv`; `main-ds.csv` is the superseded legacy
  schema. Docstring corrected during the restructure (prose only, no behavior change).
- **33 of 2,103 `client_id+site_id` keys have implausible coordinates**, including several at
  exactly `lat=0.0, lon=0.0` and one at `lat=90.0, lon=180.0`. They sit outside any plausible US
  bounding box. Pre-existing; see `docs/DATA.md`.
- **`app/tasks/tasks.py`** is reachable as `app.tasks.tasks`. The stutter is a consequence of
  renaming the package `app/celery` → `app/tasks` (which was necessary: a local package named
  `celery` can shadow the real distribution). Renaming the module too would have been churn beyond
  code motion.
- **~~The startup scripts put a nonexistent directory on `PYTHONPATH`~~ — FIXED (2026-07).**
  `scripts/start_uvicorn_fast_api.sh` exported
  `PYTHONPATH=".../app/site_analysis/features/competitors:.../app/site_analysis/features"`. There was
  never a `features/competitors` (it was `features/active/competitors`, since deleted as dead code),
  and `git ls-tree pre-refactor` confirms the path never existed. Removing it is a no-op: Python
  silently ignores a nonexistent `sys.path` entry, so imports resolved via the `features/` entry the
  whole time. `start_celery_worker.sh` carried the same line and was deleted with Celery.
- **`.gitattributes` had 11 dead git-LFS patterns.** `git lfs ls-files` reported zero LFS-tracked
  files at HEAD both before and after. Replaced with an explanation. See `docs/DATA.md`.
- **`.env` is present in the repo's earliest git history.** It was committed in the first two
  commits and removed later; the 249-byte blob remains reachable from history. It is gitignored and
  untracked today, and this restructure never touched it. Nothing here can fix that — it needs a
  history rewrite plus rotation of whatever keys it held, which is a separate, deliberate operation.
- **Five `sys.path.insert` calls remain**: two in `proforma/ui/` (`app.py`, `site_visual_page.py` —
  the Streamlit entrypoints), one in `app/` (`insights/tests/test_metrics.py`), and two in
  `experiments/datafetching/competition/`. `streamlit/web/bootstrap.py:59` puts only
  `dirname(main_script_path)` on `sys.path`, never the repo root, so an entrypoint cannot reach
  `proforma.*` without one — and packaging (`pyproject.toml`) was explicitly out of scope. Every
  remaining call sits in a script that is invoked directly; no *library* module has one. The one that
  mattered, in `app/pnl_analysis/modelling/data.py`, is gone.

- **Moving a tree disarmed the guard that watched it.** `check_imports_resolve.py` exists because the
  `app/utils` → `app/core` rename broke five modules in `datafetching/` and nobody noticed for two
  commits. Then `datafetching/` was quarantined into `experiments/`, and the checker's `SKIP_DIRS`
  contained `experiments` while its `FIRST_PARTY` tuple still said `datafetching`. So it stopped
  scanning the exact tree it was written to protect, and `experiments.council` was never first-party
  at all. It reported `ok 64` on a tree with eight broken imports. Fixed 2026-07: `experiments` is
  first-party and no longer skipped, verified with two negative controls. **If you quarantine a
  directory, check what stopped looking at it.**

---

## 9. `expense_plan` is reproducible only against a pinned `scipy`

**Status: accepted and pinned.** This is the one place where "the same code on the same data" does
not mean "the same number."

`opex_pct_curve_fit` (`app/pnl_analysis/modelling/pnl.py`) fits the opex%-of-revenue decay with
`scipy.optimize.curve_fit`, a bounded non-linear least-squares. It stops when `ftol`/`xtol`/`gtol`
(all `1e-8` by default) are satisfied — **near** the minimum, not **at** it. Its fitted parameters
are consequently only defined to about `1e-8`, and different scipy builds take different steps and
halt at different points.

Consolidating onto one Python 3.11 environment moved scipy 1.13.1 → 1.17.1. The fit's inputs are
bit-identical across both (`age`, `y`, `w` sha256-matched; `p0` equal to the last digit) — pandas
2.2→3.0 and numpy 2.0→2.4 change nothing. Only the output moves:

```
tau  2.4609460487255252  ->  2.4609460460784720     (1.1e-9 relative)
```

`opex = shape × revenue` inherits it; `net = revenue − expenses` amplifies it by cancellation to
`1.4e-9`, past the harness's `1e-9`. Blast radius: **209 floats, all inside `cases.expense_plan`**,
worst absolute move `1.128e-05` on `combined_expenses[0] ≈ 86,410`, and exactly **two** values
(`net[5]`, `net[6]`) exceeding `1e-9` on both the absolute and relative test. In money: half a
millionth of a dollar on a monthly net of −$3,733.52. The other 14 API cases, all 24 coldstart cases
and the whole Streamlit render surface are bit-identical.

The upgrade was therefore landed as its own commit, re-baselining `api.json` alone, so the diff is
attributable. `environment.yml` pins `scipy==1.17.1` — the version the baseline was captured under —
and `scripts/smoke.sh` asserts it before running any golden.

**What this really says:** a `1e-9` contract over an iterative optimizer's output was never pinning
*behavior*; it was pinning a *build*. It passed before the upgrade only because both captures used
the same scipy. The honest fix, if `expense_plan` ever needs cross-solver reproducibility, is to
tighten `curve_fit`'s tolerances (`ftol=xtol=gtol=1e-14`) so it converges to the true minimum rather
than stopping near it. That is a modelling change, not a restructure, and it would move the numbers
again — so it was not done here.
