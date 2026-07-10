# Architecture

## The shape of it

```
proforma/          ALL modelling. One tree; versions are git tags, not folders.
  data/            every dataset, once.
  models/          coldstart.py — THE model. plateau x ramp x cannibalization.
  ui/              Streamlit. app.py is a thin entry; panels/ holds the modes.
  artifacts/       the fitted joblib.
app/               FastAPI backend. Two entrypoints (below).
libs/carwash_type/ importable utility: classify a wash from its website.
archive/           frozen prior work. Read for method history; do not build on it.
experiments/       standalone, not on the import path (customer-churn, datafetching CLIs).
```

## Versioning: git, not folders

There used to be `proforma/v1_5/` and `proforma/v1_6/`. That layout implied `v1_6` succeeded `v1_5`.
It did not: `v1_6` (the council) imported **nothing** from `v1_5` — no `coldstart`, no shared code.
It was an orthogonal experiment that happened to read the same panel. Meanwhile a directory cannot
express "v1.6 is v1.5 plus a delta"; only git can.

So: **one `proforma/` tree, versioned by git tag.** The council moved to `experiments/council/`.

```bash
git tag -l                                   # proforma-v1.5, council-v1.6, pre-refactor
git checkout proforma-v1.5 -- proforma       # recover the v1.5 tree exactly
```

To ship a new model version: branch, change `proforma/models/`, refit `proforma/artifacts/`,
run `./scripts/smoke.sh` to see exactly which numbers moved, then tag on merge.

## The one thing to understand first

The forecasting math lives in **`proforma/models/coldstart.py`** and has **two consumers**:

1. **In-process by the Streamlit app** — `proforma/ui/` imports it directly, no HTTP.
2. **By the FastAPI backend** — `app/pnl_analysis/modelling/data.py` does
   `from proforma.models import coldstart as cm`.

Historically (2) reached (1) through a `sys.path.insert` that pointed into the Streamlit directory.
That is gone; both now import the same package off the repo root.

**The model is shared. The P&L and market math around it is not.** It is implemented twice — once
inside `proforma/ui/`, once in `app/pnl_analysis/modelling/*` — and the two have drifted. When
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
live external-data fetchers or the OpenAI client. There is no `serve_pnl.py` shim: if a deploy
script still says `serve_pnl:app`, point it at `app.pnl_only:app`.

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
- **`app/site_analysis/`** — synchronous external-data enrichment. `server/` splits into `router.py`
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
- The Streamlit **entrypoints** under `proforma/ui/` each put the repo root on `sys.path`.
  `streamlit run` only adds the script's own directory (`streamlit/web/bootstrap.py:59`), and there
  is deliberately no packaging. No library module does this.
- `proforma`, `proforma`, `libs` are implicit namespace packages (no `__init__.py`).
  `app` and `experiments.council` are regular packages.

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
