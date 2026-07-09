# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A car-wash **site-selection / proforma forecasting** research platform for Sonny's. The central
product: drop a pin on a US map for a site that doesn't exist yet and get a 5-year monthly wash-count
forecast + P&L, plus what it does to existing neighbours. Everything orbits one panel dataset of ~2,000
real sites (`client_id + site_id`, monthly, 2020→2027).

The repo is **research-heavy**: many parallel modelling iterations coexist. Know which are live vs. legacy
(see [Directory orientation](#directory-orientation)) before touching anything — most top-level folders are
superseded experiments kept for reference.

## The one thing to understand first: the cold-start model has two consumers

The forecasting math lives in **`earnest-proforma-2.0/streamlits/coldstart_model.py`** (plateau × ramp ×
cannibalization; see `earnest-proforma-2.0/MODELLING.md` for the full method). It is loaded two ways:

1. **In-process by the Streamlit app** (`earnest-proforma-2.0/streamlits/app.py`) — `import coldstart_model`,
   pure Python, no HTTP. This is the original UI.
2. **Ported into a FastAPI backend** (`app/pnl_analysis/`) whose modelling modules re-implement the
   Streamlit P&L/market math as API endpoints. It imports `coldstart_model` via a **`sys.path` hack** in
   `app/pnl_analysis/modelling/data.py` (inserts `earnest-proforma-2.0/streamlits/` onto the path). This
   requires the repo layout `<root>/app` + `<root>/earnest-proforma-2.0` to be intact.

So the same logic exists in both a notebook-style Streamlit and a service. When you change forecasting
behavior, check whether the change belongs in `coldstart_model.py` (shared), in `streamlits/app.py` (UI P&L),
or in `app/pnl_analysis/modelling/*` (the API port) — the last two duplicate math and drift apart if you edit
only one.

## Environments (three of them — this is a common footgun)

There is no single interpreter. Match the tool to its env:

| Env | Python | Defined in | Runs |
|-----|--------|-----------|------|
| conda `sonnysDataCollection` | 3.9 | `environment.yml` | the **FastAPI backend + Celery** (`serve_pnl.py`, `app.site_analysis.server.main`) |
| conda `proforma311` | 3.11 | `environment-proforma311.yml` | the **Streamlit app** (streamlit 1.58, lightgbm, geopandas) |
| `venv/` | 3.13 | (checked-in venv) | dev/streamlit per `MODELLING.md` (`venv/bin/streamlit ...`) |

**Critical unpickle gotcha:** `coldstart_model.py` saves/loads a `joblib` artifact
(`earnest-proforma-2.0/notebooks/artifacts/coldstart_artifacts.joblib`). scikit-learn/lightgbm pickles are
version-sensitive — **refit the artifact in the same environment that will load it.** For the FastAPI backend
that means the conda `sonnysDataCollection` env; refitting in the py3.13 `venv` produces an artifact the
backend cannot unpickle. Inference-time logic (anchor calibration, ASP corruption filter, breakeven) needs no
refit.

## Common commands

```bash
# --- Streamlit app (the drop-a-pin explorer + forecaster) ---
# runs coldstart_model in-process; run from repo root so streamlit puts the script dir on sys.path
streamlit run earnest-proforma-2.0/streamlits/app.py            # http://localhost:8501
#   or: cd earnest-proforma-2.0/streamlits && streamlit run app.py

# --- FastAPI backend ---
# full backend (site_analysis + pnl_analysis); needs Redis + Celery for async site_analysis
python -m app.site_analysis.server.main                          # port from FAST_API_PORT (.env), default 8002
scripts/start_uvicorn_fast_api.sh                                # nohup launcher (verifies conda env, sets PYTHONPATH)

# pnl-only backend (no celery/openai heavy deps) — explore-markets + forecast endpoints only
uvicorn serve_pnl:app --host 127.0.0.1 --port 8010              # routes under /v1/pnl_analysis/...

# --- Celery worker (required for POST /v1/analyze-site async pipeline) ---
scripts/start_celery_worker.sh                                   # or: celery -A app.celery.celery_app worker --loglevel=info

# --- Data: rebuild the panel after upstream CSV changes ---
python earnest-proforma-2.0/scripts/process_main_data_v2.py     # main-data-v2.csv -> *-processed.csv, syncs to final-1.6/data

# --- Council: retrospective backtest (isolated experiment) ---
python -m council.harness --limit 8                              # cheap smoke over 8 sites (run FROM REPO ROOT)
python -m council.harness                                        # full N≈420 — many LLM calls, slow/$$
streamlit run council/streamlit_view.py                          # council UI over the backtest outputs
```

There is **no test suite / linter**. `test_*.py` at root (`test_endpoint.py`, `test_db_ssl.py`) are ad-hoc
manual scripts that call backend functions directly, not a pytest harness.

## Backend layout (`app/`)

FastAPI, two routers both mounted under `/v1` by `app/site_analysis/server/main.py`
(`serve_pnl.py` mounts only the second):

- **`app/site_analysis/`** — async external-data enrichment for a lat/lon (weather, competing washes, retail
  anchors, gas). `POST /v1/analyze-site` enqueues a Celery task → poll `GET /v1/task/{id}` →
  `GET /v1/{dimension}/data-by-task/{id}`. `POST /v1/site-context` is the **synchronous** single-call variant.
- **`app/pnl_analysis/`** — router prefix `/pnl_analysis` (so full paths are `/v1/pnl_analysis/...`).
  `modelling/` ports the Streamlit market/P&L/campaign math; `insights/` is the LLM layer.
  Endpoints: `explore-market`, `pinpoint-forecast`, `market-forecast`, `pnl-forecast`, `expense-plan`,
  `campaign/*`, and `insights/*`.
- **`app/pnl_analysis/insights/`** — LLM "context, not the model" layer. `graph.py` is a 2-node LangGraph
  (compute metrics → generate narrative) with a sequential fallback if langgraph is absent. `llm.py` toggles
  backend via `INSIGHTS_LLM_BACKEND=azure|local` and cascades on failure. `location_poc.py` is a separate
  location-only (no internal data) research pipeline. **Never let these alter modelled numbers** — they annotate.
- **`app/celery/celery_app.py`** — Redis broker/backend from env; `app/utils/common.py` is the central config
  hub (loads `.env` from repo root regardless of CWD; exposes API keys + geocoding helpers).

**Import gotchas:** the `sys.path` insert in `pnl_analysis/modelling/data.py` (above); lazy `__getattr__` in
`insights/__init__.py` (avoids importing langgraph unless `market_insights` is used); the startup scripts add
`app/site_analysis/features/...` to `PYTHONPATH` so intra-feature bare imports resolve.

## Data — source of truth

The canonical panel is **`earnest-proforma-final-1.6/data/main-data-v2-stitched.csv`** (2020-01→2027-01,
~2,103 sites, handoffs stitched, `imputed=0`). `earnest-proforma-2.0/data/` holds a byte-identical mirror plus
derived files. The Streamlit app + `coldstart_model.py` read the mirror's `main-data-v2-stitched.csv` for the
panel, `opex-data.csv` for P&L, and `site_carwash_types.csv` for wash types (`main-ds.csv` is a legacy
schema). The council reads only the final-1.6 copy. Keep the two mirrors in sync via
`process_main_data_v2.py`. The real site key is `client_id + site_id` (site_id alone is a within-brand index).

Several large CSVs are **git-LFS** tracked (see `.gitattributes`). `carwash_reviews*.py/csv` are gitignored
(contain a live API key) — do not commit them.

## Directory orientation (live vs. legacy)

- **Live:** `earnest-proforma-2.0/` (production modelling + Streamlit), `app/` (backend services),
  `earnest-proforma-final-1.6/data/` (canonical data), `council/` (active isolated experiment),
  `datafetching/` (live ingestion feeding `app/site_analysis`).
- **Superseded / reference only:** `earnest-proforma-1.5/`, `zeta_modelling/` (two-track LightGBM forecaster),
  `final_modelling/` (IDW neighbour baseline), `hypothesis-testing/`, `notebooks/`. Read for method history;
  don't build on them.
- **Standalone subprojects:** `customer-churn/`, `type_car_wash/` (car-wash-type classifier that resolves
  `site_carwash_types.csv`), `carwash_reviews_sentiment.py`.

## Deeper docs (read before non-trivial modelling changes)

- `earnest-proforma-2.0/MODELLING.md` — the authoritative forecasting reference: plateau × ramp, the 4
  dashboard models, anchor-weight calibration (`ANCHOR_CALIB_W=0.50`), the ASP-corruption filter, cash-payback
  breakeven, and the facts each was fixed to address.
- `earnest-proforma-2.0/streamlits/README.md` and `PINPOINT_FORECAST.md` — the two Streamlit modes and the
  density-aware adaptive clustering that defines a "local market" (`assign_clusters`, used by both app and model).
- `council/README.md` + `council/COUNCIL_MEETING_NOTES.md` — the council design, leakage controls, and the key
  finding: greenfield mature-*level* is ~unpredictable here; the only leakage-clean edge is a small go/no-go
  signal (out-of-fold AUC 0.57) that beats "always build."
