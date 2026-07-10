# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

## What this repo is

A car-wash **site-selection / proforma forecasting** research platform for Sonny's. The product:
drop a pin on a US map for a site that doesn't exist yet, get a 5-year monthly wash-count forecast
+ P&L, plus what it does to existing neighbours. Everything orbits one panel of ~2,100 real sites
(`client_id + site_id`, monthly, 2020→2027).

Start with `README.md`, then `docs/ARCHITECTURE.md`. **Read `docs/DIVERGENCES.md` before "fixing"
anything that looks wrong** — several things are knowingly broken.

## Layout

```
proforma/          ALL modelling. ONE tree. Versions are git tags, not folders.
  data/            every dataset, exactly once.
  models/          coldstart.py — the model. tunnel_capex.py.
  artifacts/       the fitted joblib (~46 MB).
  ui/              Streamlit only. app.py is a thin entry; panels/ holds the modes.
  backtests/       scripts (side effects at import). Run them; never import them.
app/               FastAPI. app/main.py is the single entrypoint; app/server/ is the router layer.
  core/            was app/utils/ — config, env, geocoding. Almost everything imports it.
libs/carwash_type/ was type_car_wash/ — a real importable utility.
archive/           frozen prior work. Read for history; do not build on it.
experiments/       standalone; not on the import path.
```

## The one thing to understand first

Forecasting math lives in **`proforma/models/coldstart.py`** (plateau × ramp ×
cannibalization; see `proforma/MODELLING.md`). It has **two consumers**:

1. the Streamlit app, in-process;
2. the FastAPI backend, via `from proforma.models import coldstart as cm` in
   `app/pnl_analysis/modelling/data.py`.

The **model** is shared. The **P&L / market math around it is implemented twice** — once in
`proforma/ui/`, once in `app/pnl_analysis/modelling/*` — and the two have drifted. Decide which
of the three places a change belongs in *before* editing. Do not unify them as a drive-by; see
`docs/DIVERGENCES.md` §1.

## Environment (one of them)

| Env | Python | Defined in | Runs |
|-----|--------|-----------|------|
| conda `sonnys` | 3.11 | `environment.yml` | backend, Streamlit, model, council |

`conda env create -f environment.yml && conda activate sonnys`. There used to be three (a py3.9
backend env, a py3.11 Streamlit env, a py3.13 `venv/`); nothing needed py3.9, so they collapsed.

**`scipy` and `numpy` are pinned on purpose.** `opex_pct_curve_fit` uses `scipy.optimize.curve_fit`,
which stops on a `1e-8` tolerance rather than exactly, so a different scipy build moves
`expense_plan` by ~1e-9 relative. `smoke.sh` asserts the pinned version. To upgrade: bump, re-capture
the baseline, and commit that diff **alone**. See `docs/ENVIRONMENTS.md`.

**Unpickle rule:** `proforma/artifacts/coldstart_artifacts.joblib` is a plain pickle of
lightgbm/sklearn/numpy objects — it holds **no reference to the module that wrote it**, so the model
module can be renamed freely. It *is* coupled to library versions: **refit it in the env that will
load it** (conda `sonnys`). Inference-time logic needs no refit. Details, including the benign
sklearn 1.6.1-vs-1.8.0 mismatch, are in `docs/ENVIRONMENTS.md`.

## Common commands

```bash
# everything runs in `conda activate sonnys`, from the repo root
streamlit run proforma/ui/app.py                            # the explorer, http://localhost:8501
python -m app.main                                          # the API (only /v1/pnl_analysis/*)
scripts/start_uvicorn_fast_api.sh                           # nohup launcher (needs CONDA_ENV_NAME=sonnys)

# Rebuild the panel after upstream CSV changes
python proforma/scripts/process_main_data_v2.py

# Council backtest (experiment, isolated)
python -m experiments.council.harness --limit 8                   # cheap smoke; full run is ~2000 LLM calls
streamlit run experiments/council/streamlit_view.py

# Prove you changed no numbers
./scripts/smoke.sh
```

## Before you commit a modelling change

`./scripts/smoke.sh` captures the cold-start model over three fixed pins, every deterministic
`/v1/pnl_analysis/*` endpoint, the Streamlit app's rendered widget surface, and an import sweep,
then diffs against `scripts/_golden/baseline/` at `1e-9`. If you *intend* to change numbers, it will
tell you exactly which ones moved — read that diff, don't silence it by re-baselining.

Its coverage gaps are stated in its own header comment and in `docs/DIVERGENCES.md` §6: the UI is
only first-render, and `/insights/*` are LLM and excluded.

There is **no test suite and no linter**. `test_*.py` at the root are ad-hoc manual scripts;
`test_endpoint.py` is itself broken (`docs/DIVERGENCES.md` §8).

## Rules of the road

- **The site key is `client_id + site_id`.** `site_id` alone is a within-brand index and collides.
- **Data lives once**, under `proforma/data/`. Never copy a dataset into a model version. Artifacts
  are the opposite: they belong to a version, because they're welded to the code that fitted them.
- **`proforma/backtests/**` are scripts, not libraries.** They fit models at module import.
  Never import them to test.
- **`insights/` annotates; it must never alter a modelled number.**
- **No packaging.** No `pyproject.toml`, no `pip install -e .`. Imports resolve off the repo root.
  The two conda envs and the version-sensitive joblib make packaging a separate, riskier project.
- Large CSVs are ordinary git blobs. **git-LFS is not in use** (`.gitattributes` explains why).
- `carwash_reviews*.py/csv` are gitignored and contain a live API key — never `git add` them. They
  do not currently exist on disk.
