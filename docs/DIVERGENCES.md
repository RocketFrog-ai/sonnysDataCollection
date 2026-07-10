# Divergences and known defects

Things that are wrong, duplicated, or surprising in this repo, found during the 2026-07
restructure and **deliberately left alone**. The restructure was behavior-preserving: fixing any
of these would have changed a number or changed broken-to-working, which is a separate decision
with a separate review. Each entry says what it is, how it was verified, and what the fix would be.

Nothing here is a regression introduced by the restructure.

---

## 1. The P&L math exists twice, and the copies have drifted

The forecasting/P&L math is implemented **twice**:

| | |
|---|---|
| `proforma/v1_5/ui/app.py` | the original, in-process with the Streamlit UI |
| `app/pnl_analysis/modelling/{market,pnl,campaign,trend}.py` | a port of the same math, exposed as API endpoints |

`app/pnl_analysis/modelling/data.py` says its loaders "mirror the loaders in
`proforma/v1_5/ui/app.py`". *Mirror* is doing real work in that sentence — it is a second
implementation, not a shared one. `pnl.py`'s own docstring claims it "is ported verbatim from the
Streamlit reference … the only [difference] …", which is an admission that a difference exists.

**Do not unify these as a drive-by.** Unifying them means picking a winner for every place they
disagree, and each choice moves a number that someone has already looked at. That is the *next*
project, and it needs its own golden baseline covering the Streamlit side (which today has none —
see §6).

Only the shared parts are genuinely shared: `proforma/v1_5/models/coldstart.py`
(`predict_site`, `predict_neighbours`, `assign_clusters`, the cannibalization fit) is imported by
both, so plateau × ramp × cannibalization is computed once.

---

## 2. The Celery worker cannot start (pre-existing, breaks `POST /v1/analyze-site`)

`app/tasks/celery_app.py` declares:

```python
include=[
    "app.site_analysis.modelling.site_analysis",
    "app.pnl_analysis.modelling.zeta_pnl",   # <-- does not exist
    "app.tasks.tasks",
]
```

`app/pnl_analysis/modelling/zeta_pnl.py` does not exist and does not exist at the `pre-refactor`
tag either. It was deleted in `814fa37` ("cleaning dead code") and the `include` list was never
updated.

Verified by doing exactly what a worker does at startup:

```
>>> celery_app.loader.import_default_modules()
ModuleNotFoundError: No module named 'app.pnl_analysis.modelling.zeta_pnl'
```

**Consequence:** `celery -A app.tasks.celery_app worker` fails to boot, so the async
`POST /v1/analyze-site` → `GET /v1/task/{id}` pipeline is dead. The synchronous
`POST /v1/site-context` variant and the entire `/v1/pnl_analysis/*` surface are unaffected, which
is presumably why nobody noticed.

**Fix:** delete that one line. Not done here because it flips behavior from broken to working.

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

`proforma/v1_5/models/coldstart.py::predict_site` is documented as:

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
- **`app/site_analysis/features/**` is `ast`-parsed but never imported.** Those modules run live
  HTTP/LLM calls at module scope: `app/site_analysis/features/inactive/experimental_features/typeOfSite/test_o4_mini.py` fires an API request and prints a
  `403` on import, and `.../typeOfSite/o4mini_images_classification.py` calls a vision model at module level.
  Importing that tree costs money.
- **The Celery pipeline is not exercised** (needs Redis — and see §2).
- **`app/site_analysis/*` endpoints have no golden outputs**, because they need Redis/Celery. Their
  route paths and their full OpenAPI schema (every field name, type, default, and validation bound)
  were diffed against the `pre-refactor` tag once, by hand, and matched byte-for-byte. That is a
  one-time check, not a standing test.

---

## 7. Sundry

- **`test_endpoint.py` (repo root) has been broken for some time.** It does
  `from app.site_analysis.server.routes import get_competitors_dynamics_endpoint` and
  `from app.site_analysis.server.models import CompetitorsDynamicsRequest`. Neither symbol exists in
  `app/` today, and `git grep` at the `pre-refactor` tag shows neither existed then. It is an ad-hoc
  manual script, not a pytest. During the restructure `routes.py`/`models.py` were split into
  `router.py` / `schemas.py` / `service.py`; keeping them alive as re-export shims would not have
  helped, because the *symbols* are what is missing, not the modules. Left broken.
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
- **The startup scripts put a nonexistent directory on `PYTHONPATH`.** Both
  `scripts/start_uvicorn_fast_api.sh` and `scripts/start_celery_worker.sh` export
  `PYTHONPATH=".../app/site_analysis/features/competitors:.../app/site_analysis/features"`. There is
  no `features/competitors` — it is `features/active/competitors`. `git ls-tree pre-refactor` shows
  the path never existed. A nonexistent `sys.path` entry is silently ignored, so this is harmless
  today, but it means the intra-feature bare imports resolve via the `features/` entry alone, not
  the one someone intended. Left alone: correcting it *adds* a directory to `sys.path`, which can
  change import resolution, and that is a behavior change.
- **`.gitattributes` had 11 dead git-LFS patterns.** `git lfs ls-files` reported zero LFS-tracked
  files at HEAD both before and after. Replaced with an explanation. See `docs/DATA.md`.
- **`.env` is present in the repo's earliest git history.** It was committed in the first two
  commits and removed later; the 249-byte blob remains reachable from history. It is gitignored and
  untracked today, and this restructure never touched it. Nothing here can fix that — it needs a
  history rewrite plus rotation of whatever keys it held, which is a separate, deliberate operation.
- **Three `sys.path.insert` calls remain in `proforma/v1_5/ui/`** (the Streamlit entrypoints) and
  three in `app/` (two feature scripts, one test). `streamlit/web/bootstrap.py:59` puts only
  `dirname(main_script_path)` on `sys.path`, never the repo root, so an entrypoint cannot reach
  `proforma.*` without one — and packaging (`pyproject.toml`) was explicitly out of scope. No
  *library* module has one. The one that mattered, in
  `app/pnl_analysis/modelling/data.py`, is gone.
