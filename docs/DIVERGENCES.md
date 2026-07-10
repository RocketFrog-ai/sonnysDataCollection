# Divergences and known defects

Things that are wrong, duplicated, or surprising in this repo, found during the 2026-07
restructure. Most were **deliberately left alone**: the restructure was behavior-preserving, and
fixing them would have moved a number or flipped broken-to-working, each a separate decision with a
separate review. Entries that have since been *resolved* say so and are kept for the record.
Each says what it is, how it was verified, and what the fix would be.

Nothing here is a regression introduced by the restructure.

---

## 1. The P&L math exists twice, and the copies have drifted

The forecasting/P&L math is implemented **twice**:

| | |
|---|---|
| `proforma/ui/app.py` | the original, in-process with the Streamlit UI |
| `app/pnl_analysis/modelling/{market,pnl,campaign,trend}.py` | a port of the same math, exposed as API endpoints |

`app/pnl_analysis/modelling/data.py` says its loaders "mirror the loaders in
`proforma/ui/app.py`". *Mirror* is doing real work in that sentence — it is a second
implementation, not a shared one.

Only the shared parts are genuinely shared: `proforma/models/coldstart.py`
(`predict_site`, `predict_neighbours`, `assign_clusters`, the cannibalization fit) is imported by
both, so plateau × ramp × cannibalization is computed once.

**These are agreed** (checked, identical): `PNL_EXCLUDE = {"alpinecarwash_000087"}`; the ASP>200
revenue-nulling in the panel loader; `ASP_MIN_WASH = 200`, `ASP_FLOOR_MEM = 4.0`,
`ASP_FLOOR_RET = 5.0`; and the row predicate inside `_drop_corrupt_asp_rows`.

### 1a. Revenue is priced per-leg in the UI and by one blended ASP in the API

An earlier version of this entry claimed the two sides compute membership ASP as *different
quantities*. That was wrong, and the correction is worth keeping because the wrong version is the
plausible one.

**Not a divergence.** The UI's `global_healthy_asp` returns `(cl_mem_pp, ppw, cl_ret)` — dollars per
membership *purchase*, times purchases-per-wash — while the API's returns `(cl_mem, cl_ret)` in
dollars per *wash*. But over the same row subset,

```
cl_mem_pp * ppw = Σmem_revenue/Σmem_purchase_count * Σmem_purchase_count/Σmem_wash_count
                = Σmem_revenue/Σmem_wash_count = cl_mem
```

Measured on the real panel: `59.18531174297621 × 0.3220765744212813 = 19.062202462233410` vs the
API's `19.062202462233408` — a difference of `3.6e-15`. The UI merely keeps the factorization,
because its slider shows the user `$/membership purchase`, which is a number an operator recognises.

**The real divergence is one level up: how the legs are combined.**

| | |
|---|---|
| UI (`_pinpoint_forecast.py:1129,1134`) | `mem_purch = mem × ppw`; `revenue = mem_purch × cl_mem_pp + ret × cl_ret` — i.e. `mem × cl_mem + ret × cl_ret`, each leg priced with its own ASP, **every month** |
| API (`pnl.py:309,316`) | `asp_blend = mem_share × cl_mem + (1−mem_share) × cl_ret`, then `revenue = (mem+ret) × asp_blend` — one blended `$/wash`, and `mem_share` is the **mature** share, a scalar |

These agree only where the month's membership share equals the mature share. It does not during the
ramp: membership share climbs from ~0.36 at open to ~0.73 by month 60, while `mem_share` is fixed
at 0.6656. Measured at the dense Houston pin (`cl_mem = $17.70`, `cl_ret = $19.36`):

```
month  0:  UI $59,639   API $58,044   -2.67%
month  6:  UI $131,423  API $130,541  -0.67%
month 24:  UI $155,871  API $155,944  +0.05%
month 60:  UI $150,445  API $151,303  +0.57%
5-yr total: UI $8,890,118  API $8,897,825  +0.09%
```

The API under-prices the early ramp and over-prices maturity. The error scales with
`|cl_mem − cl_ret|`, which is only $1.65 here — in a market where membership and retail ASPs sit
further apart it will be materially larger.

Consequently `asp_override` means different things on each side: the API documents it as
"Blended $/wash" and applies it to total washes; the UI's slider is `$/membership purchase` applied
to the membership leg only. A client sending the number a user read off the Streamlit slider will
get the wrong revenue.

**Would change:** forecast revenue, hence the P&L, breakeven, and the campaign ROI. Not fixed here:
picking a winner moves numbers people have signed off on, and the Streamlit side has no golden
baseline to verify the move against (§6).

### 1b. ~~`express_only` exists only in the UI~~ — RESOLVED (2026-07)

The API now accepts `express_only: bool = False` on the 8 pin-based modelling requests
(ExploreMarket, ExploreKpis, PinpointForecast, PnlForecast, ExpensePlan, CampaignVerdict,
EatingMarket, LocalCampaigns; not on `/insights/*`, which annotate rather than model). It applies
the UI's two filters in the UI's order — `primary_carwash_type == "Express Tunnel"`, then
`n_obs >= 30` — re-clusters the subset, and passes `anchor_keys` to
`coldstart.predict_site` so only the level anchor is scoped while the LightGBM neighbour features
stay on the full site set.

Verified against the UI in both conda envs: **742 sites, 35,778 rows, identical `site_key` sets and
identical cluster labels.** The default (`false`) leaves every number bit-for-bit unchanged —
`api.json` and `model.json` still match the frozen baselines.

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

Route count on `app.main` went 40 → 28; `/v1/pnl_analysis/*` was untouched (22 routes, identical),
and `model.json` / `api.json` were unchanged, so no forecast number moved.

**Residue, on purpose:** `GET /v1/cache/site-analysis/all` still serves the Postgres cache, but its
only writer was the Celery task, so the table is now read-only and will go stale. The `REDIS_*` /
`CELERY_*` keys may still sit in your `.env`; nothing reads them.

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

## 7. Six P&L functions are dead in the UI and live in the API

`load_pnl`, `regional_opex`, `opex_per_wash`, `opex_ramp`, `opex_trend_hist`, and `asp_refs` are
defined in `proforma/ui/panels/_pinpoint_forecast.py` and called **nowhere** in `proforma/`.
Their namesakes in `app/pnl_analysis/modelling/pnl.py` *are* called (`pnl.py:282`, `pnl.py:311`, …).

So the API port kept them live while the Streamlit side stopped using them. They were relocated
verbatim during the split rather than deleted, because deleting code is not code motion and because
they are the closest thing to a specification of what the API's copies are supposed to do. Delete
them only together with a decision about §1.

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
- **Three `sys.path.insert` calls remain in `proforma/ui/`** (the Streamlit entrypoints),
  three in `app/` (two feature scripts, one test), and five in `experiments/datafetching/` (one per standalone
  fetcher). `streamlit/web/bootstrap.py:59` puts only `dirname(main_script_path)` on `sys.path`,
  never the repo root, so an entrypoint cannot reach `proforma.*` without one — and packaging
  (`pyproject.toml`) was explicitly out of scope. Every remaining call sits in a script that is
  invoked directly; no *library* module has one. The one that mattered, in
  `app/pnl_analysis/modelling/data.py`, is gone.
