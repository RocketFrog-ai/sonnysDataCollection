# Architecture

## The shape of it

```
proforma/          ALL modelling. One tree; versions are git tags, not folders.
  data/            every dataset, once.
  models/          coldstart.py — THE model. plateau x ramp x cannibalization.
  pnl/             shared P&L/market helper math (data/trend/opex/campaign), imported by UI + API.
  ui/              Streamlit. app.py is a thin entry; panels/ holds the modes.
  artifacts/       the fitted joblib.
app/               FastAPI backend. main.py + server/ (routes) + core/ + pnl_analysis/.
libs/carwash_type/ importable utility: classify a wash from its website.
archive/           frozen prior work. Read for method history; do not build on it.
experiments/       council/ (imported by the Explore-markets panel), datafetching/ CLIs, customer-churn/.
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

## One backend, one entrypoint

```bash
python -m app.main                       # host/port from FAST_API_HOST / FAST_API_PORT
uvicorn app.main:app --port 8010
```

Everything is mounted under `/v1`, and `app/server` carries the prefix `/pnl_analysis`, so the paths
are `/v1/pnl_analysis/...`. 17 paths under `/v1/pnl_analysis` (plus `GET /`) — the market / forecast /
campaign endpoints plus five `insights/*` (`/insights`, `/insights/location`, `/insights/competition`,
`/insights/pollinated`, `/insights/independent-research`).

### What used to be here, and why it isn't

- **Celery + an async pipeline.** `POST /v1/analyze-site` enqueued a task; you polled
  `GET /v1/task/{id}`. The worker had been unable to boot for months (`celery_app.conf.include`
  named a module deleted in `814fa37`), so callers polled forever. Removed 2026-07 with Redis and
  the worker scripts.
- **The whole `site_analysis` subsystem** — `server/`, `modelling/site_context.py`, `config.py`, and
  feature fetchers for weather, gas, retail anchors, stores and traffic lights. It backed
  `POST /v1/site-context`, `/site-features`, `/traffic-lights`, `/nearby-stores`. **Nothing rendered
  any of it**: the Streamlit page that did (`site_analysis_page.py`) was never wired into the mode
  dispatch, and Sitewise calls Google Places directly. Removed 2026-07.
  Recover with `git checkout site-analysis-api -- app/site_analysis app/main.py`.
- **The second entrypoint.** `app/pnl_only.py` existed only because `main` also mounted the
  site_analysis router. With that gone the two apps were identical, so they were collapsed.

Two modules survived that removal, because Explore-markets still needs them: `app/core/places/`
(`nearby_competitors`, `search_nearby`) anchors the **Competition Coverage** insight on real Google
Places car washes rather than LLM guesswork.

## Inside `app/`

- **`app/main.py`** — the FastAPI app. Mounts one router.
- **`app/server/`** — the HTTP layer, global rather than nested under a feature: `router.py`
  (parse, delegate, serialize), `schemas.py` (pydantic), `service.py` (shared handler logic).
- **`app/core/`** (was `app/utils/`) — the config hub. Loads `.env` from the repo root regardless of
  CWD; exposes API keys and geocoding helpers. `core/llm/` is the LLM transport, `core/places/` the
  Google Places helpers.
- **`app/pnl_analysis/`** — `modelling/` ports the Streamlit math; `insights/` is the LLM layer.
- **`app/pnl_analysis/insights/`** — "context, not the model". `graph.py` is a 2-node LangGraph
  (compute metrics → generate narrative) with a sequential fallback if langgraph is absent.
  `llm.py` switches backend via `INSIGHTS_LLM_BACKEND=azure|local`.
  **These annotate; they must never alter a modelled number.**

### Import gotchas that survive

- `proforma/backtests/**` are **scripts, not libraries**: they read data and fit models at module
  scope. Never `import` that tree to test it.
- The Streamlit **entrypoints** under `proforma/ui/` each put the repo root on `sys.path`.
  `streamlit run` only adds the script's own directory (`streamlit/web/bootstrap.py:59`), and there
  is deliberately no packaging. No library module does this.
- `proforma`, `proforma.models`, `libs` are implicit namespace packages (no `__init__.py`).
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
