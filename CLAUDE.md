# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

## What this repo is

A car-wash **site-selection / proforma forecasting** research platform for Sonny's. The product:
drop a pin on a US map for a site that doesn't exist yet, get a 5-year monthly wash-count forecast
+ P&L, plus what it does to existing neighbours. Everything orbits one panel of ~2,100 real sites
(`client_id + site_id`, monthly, 2020→2027).

Start with `README.md`, then `docs/ARCHITECTURE.md`. **Read `docs/DIVERGENCES.md` before "fixing"
anything that looks wrong** — several things are knowingly broken, and one of them (the Celery
worker) has been broken for months.

## Layout

```
proforma/          ALL modelling, versioned.  v1_5 = LIVE.  v1_6 = experiment (council).
  data/            every dataset, exactly once. Shared; versioned by FILENAME, not folder.
  v1_5/models/     coldstart.py — the model. tunnel_capex.py.
  v1_5/artifacts/  the fitted joblib (~46 MB). Owned by the version.
  v1_5/ui/         Streamlit only. app.py is a thin entry; pages/ holds the modes.
  v1_5/backtests/  scripts (side effects at import). Run them; never import them.
app/               FastAPI. app/main.py (full) and app/pnl_only.py (P&L only) are the entrypoints.
  core/            was app/utils/ — config, env, geocoding. Almost everything imports it.
  tasks/           was app/celery/ — renamed so it can't shadow the real `celery` package.
libs/carwash_type/ was type_car_wash/ — a real importable utility.
archive/           frozen prior work. Read for history; do not build on it.
experiments/       standalone; not on the import path.
```

## The one thing to understand first

Forecasting math lives in **`proforma/v1_5/models/coldstart.py`** (plateau × ramp ×
cannibalization; see `proforma/v1_5/MODELLING.md`). It has **two consumers**:

1. the Streamlit app, in-process;
2. the FastAPI backend, via `from proforma.v1_5.models import coldstart as cm` in
   `app/pnl_analysis/modelling/data.py`.

The **model** is shared. The **P&L / market math around it is implemented twice** — once in
`proforma/v1_5/ui/`, once in `app/pnl_analysis/modelling/*` — and the two have drifted. Decide which
of the three places a change belongs in *before* editing. Do not unify them as a drive-by; see
`docs/DIVERGENCES.md` §1.

## Environments (three of them — the classic footgun)

| Env | Python | Defined in | Runs |
|-----|--------|-----------|------|
| conda `sonnysDataCollection` | 3.9 | `environment.yml` | FastAPI backend + Celery |
| conda `proforma311` | 3.11 | `environment-proforma311.yml` | the Streamlit app |
| `venv/` | 3.13 | not checked in | ad-hoc dev |

**Unpickle rule:** `proforma/v1_5/artifacts/coldstart_artifacts.joblib` is a plain pickle of
lightgbm/sklearn/numpy objects — it holds **no reference to the module that wrote it**, so the model
module can be renamed freely. It *is* coupled to library versions: **refit it in the env that will
load it** (conda `sonnysDataCollection`). Inference-time logic needs no refit. Details, including
the benign sklearn 1.6.1-vs-1.8.0 mismatch the Streamlit env produces, are in `docs/ENVIRONMENTS.md`.

## Common commands

```bash
# Streamlit explorer (conda proforma311) — run from the repo root
streamlit run proforma/v1_5/ui/app.py                       # http://localhost:8501

# Backends (conda sonnysDataCollection)
python -m app.main                                          # full: site_analysis + pnl_analysis
uvicorn app.pnl_only:app --host 127.0.0.1 --port 8010       # pnl only; no celery/openai deps
scripts/start_uvicorn_fast_api.sh                           # nohup launcher (sets PYTHONPATH)

# Celery worker — NOTE: currently cannot start, see docs/DIVERGENCES.md §2
scripts/start_celery_worker.sh                              # celery -A app.tasks.celery_app worker

# Rebuild the panel after upstream CSV changes
python proforma/v1_5/scripts/process_main_data_v2.py

# Council backtest (experiment, isolated)
python -m proforma.v1_6.harness --limit 8                   # cheap smoke; full run is ~2000 LLM calls
streamlit run proforma/v1_6/streamlit_view.py

# Prove you changed no numbers
./scripts/smoke.sh
```

## Before you commit a modelling change

`./scripts/smoke.sh` captures the cold-start model over three fixed pins, every deterministic
`/v1/pnl_analysis/*` endpoint, the Streamlit app's rendered widget surface, and an import sweep,
then diffs against `scripts/_golden/baseline/` at `1e-9`. If you *intend* to change numbers, it will
tell you exactly which ones moved — read that diff, don't silence it by re-baselining.

Its coverage gaps are stated in its own header comment and in `docs/DIVERGENCES.md` §6: the UI is
only first-render, `/insights/*` are LLM and excluded, `app/site_analysis/features/**` is never
imported (module-scope live HTTP/LLM calls), and Celery is not exercised.

There is **no test suite and no linter**. `test_*.py` at the root are ad-hoc manual scripts;
`test_endpoint.py` is itself broken (`docs/DIVERGENCES.md` §7).

## Rules of the road

- **The site key is `client_id + site_id`.** `site_id` alone is a within-brand index and collides.
- **Data lives once**, under `proforma/data/`. Never copy a dataset into a model version. Artifacts
  are the opposite: they belong to a version, because they're welded to the code that fitted them.
- **`app/site_analysis/features/**` and `proforma/v1_5/backtests/**` are scripts, not libraries.**
  They do real work — including live API calls — at module import. Never import them to test.
- **`insights/` annotates; it must never alter a modelled number.**
- **No packaging.** No `pyproject.toml`, no `pip install -e .`. Imports resolve off the repo root.
  The two conda envs and the version-sensitive joblib make packaging a separate, riskier project.
- Large CSVs are ordinary git blobs. **git-LFS is not in use** (`.gitattributes` explains why).
- `carwash_reviews*.py/csv` are gitignored and contain a live API key — never `git add` them. They
  do not currently exist on disk.
