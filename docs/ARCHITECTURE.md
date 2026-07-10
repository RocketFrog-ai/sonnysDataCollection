# Architecture

## The shape of it

```
proforma/          ALL modelling, versioned. v1_5 is live; v1_6 is an experiment.
  data/            every dataset, once. Shared across versions, versioned by filename.
  v1_5/models/     coldstart.py — THE model. plateau x ramp x cannibalization.
  v1_5/ui/         Streamlit. app.py is a thin entry; panels/ holds the modes.
  v1_5/artifacts/  the fitted joblib. Belongs to the version, not the repo.
app/               FastAPI backend. Two entrypoints (below).
libs/carwash_type/ importable utility: classify a wash from its website.
datafetching/      live ingestion feeding app/site_analysis.
archive/           frozen prior work. Read for method history; do not build on it.
experiments/       standalone, not on the import path.
```

## The one thing to understand first

The forecasting math lives in **`proforma/v1_5/models/coldstart.py`** and has **two consumers**:

1. **In-process by the Streamlit app** — `proforma/v1_5/ui/` imports it directly, no HTTP.
2. **By the FastAPI backend** — `app/pnl_analysis/modelling/data.py` does
   `from proforma.v1_5.models import coldstart as cm`.

Historically (2) reached (1) through a `sys.path.insert` that pointed into the Streamlit directory.
That is gone; both now import the same package off the repo root.

**The model is shared. The P&L and market math around it is not.** It is implemented twice — once
inside `proforma/v1_5/ui/`, once in `app/pnl_analysis/modelling/*` — and the two have drifted. When
you change forecasting behaviour, work out which of the three places it belongs in *before* you
edit. Read `docs/DIVERGENCES.md` §1. Unifying them is a separate project, and it needs a golden
baseline covering the Streamlit side first.

## Two backend entrypoints, on purpose

```bash
python -m app.main                    # site_analysis + pnl_analysis. Needs openai + the live fetchers.
uvicorn app.pnl_only:app --port 8010  # pnl_analysis ONLY. No openai, no live fetchers.
```

This is **not** redundant code — it is two mount configurations with different dependency
footprints. `app/pnl_only.py` exists so the forecasting API can be served without dragging in the
live external-data fetchers or the OpenAI client. `serve_pnl.py` at the root is a one-line
back-compat shim re-exporting the same FastAPI object (`serve_pnl.app is app.pnl_only.app`), kept
because `serve_pnl:app` may be baked into deploy scripts.

Both mount their routers under `/v1`. `app/pnl_analysis` additionally carries the prefix
`/pnl_analysis`, so its full paths are `/v1/pnl_analysis/...`.

### There is no Celery, and no async pipeline

There used to be. `app/tasks/` (once `app/celery/`) backed an async enrichment pipeline:
`POST /v1/analyze-site` enqueued a task, you polled `GET /v1/task/{id}`, then read
`GET /v1/{dimension}/data-by-task/{id}`. **All of it was removed in 2026-07**, along with Celery,
Redis, the two worker scripts, and `app/site_analysis/modelling/site_analysis.py`.

Why: the worker had been unable to start for months — `celery_app.conf.include` named a module
deleted in `814fa37` — so every `POST /v1/analyze-site` enqueued work nothing would ever run and the
caller polled forever. The capability had already migrated to a synchronous endpoint.

**`POST /v1/site-context` is the replacement.** One call, one response: weather, competing car
washes, retail anchors, gas stations, map markers, and rule-based insights. It fetches the same
sources in parallel with a thread pool (`site_context.py`) instead of a task queue.

If you need the async shape back, reintroduce it deliberately: it is a queue, a worker, and a result
store, not a refactor.

## Inside `app/`

- **`app/core/`** (was `app/utils/`) — the config hub. Loads `.env` from the repo root regardless of
  CWD; exposes API keys and geocoding helpers. Nearly everything imports it.
- **`app/site_analysis/`** — async external-data enrichment. `server/` splits into `router.py`
  (parse, delegate, serialize), `schemas.py` (pydantic), `service.py` (the logic).
  `features/` holds one module per data source.
- **`app/pnl_analysis/`** — the P&L/market API. Same `router` / `schemas` / `service` split.
  `modelling/` ports the Streamlit math; `insights/` is the LLM layer.
- **`app/pnl_analysis/insights/`** — "context, not the model". `graph.py` is a 2-node LangGraph
  (compute metrics → generate narrative) with a sequential fallback if langgraph is absent.
  `llm.py` switches backend via `INSIGHTS_LLM_BACKEND=azure|local`.
  **These annotate; they must never alter a modelled number.**

### Import gotchas that survive

- `app/site_analysis/features/**` are **scripts, not libraries**. Several run live HTTP/LLM calls at
  module import time. The startup scripts put their directories on `PYTHONPATH` so their bare
  intra-feature imports resolve. Never `import` that tree to test it.
- The Streamlit **entrypoints** under `proforma/v1_5/ui/` each put the repo root on `sys.path`.
  `streamlit run` only adds the script's own directory (`streamlit/web/bootstrap.py:59`), and there
  is deliberately no packaging. No library module does this.
- `proforma`, `proforma.v1_5`, `libs` are implicit namespace packages (no `__init__.py`).
  `app` and `proforma.v1_6` are regular packages.

## Verifying you changed nothing

```bash
./scripts/smoke.sh
```

Captures the cold-start model over three fixed pins, every deterministic
`/v1/pnl_analysis/*` endpoint, the Streamlit app's rendered widget surface, and an import sweep —
then diffs all of it against `scripts/_golden/baseline/` at `1e-9`. What it does **not** cover is
listed in its own header comment, and in `docs/DIVERGENCES.md` §6. There is no other test suite;
`test_*.py` at the root are ad-hoc manual scripts, not pytest (and `test_endpoint.py` has been
broken for some time — `docs/DIVERGENCES.md` §8).
