# `proforma/` — all the modelling

The cold-start forecaster. Drop a pin on a US map for a site that does not exist yet; get a 5-year
monthly wash-count forecast, a P&L, and what it does to the existing neighbours.

**One tree. Versions are git tags, not folders.**

```
proforma/
├── MODELLING.md      the authoritative method reference — read before changing math
├── models/           coldstart.py (the model), tunnel_capex.py
├── artifacts/        coldstart_artifacts.joblib — fitted; do not refit casually
├── data/             every dataset, exactly once. Shared, immutable.
├── ui/               Streamlit only. app.py is a thin entry; panels/ holds the modes
├── backtests/        the evidence behind the model's design choices. Scripts, not libraries
├── notebooks/        exploration + the figures in MODELLING.md
└── scripts/          process_main_data_v2.py — rebuilds the panel
```

## Why there is no `v1_5/` directory

There was, briefly, alongside a `v1_6/`. That implied `v1_6` superseded `v1_5`. It did not — `v1_6`
(the council) imported nothing from `v1_5`, not even `coldstart`. It was an orthogonal experiment,
and it now lives at `../experiments/council/`.

A directory cannot express "v1.6 = v1.5 + a delta". Git can. So versions are tags:

```bash
git tag -l                                  # proforma-v1.5, council-v1.6, pre-refactor
git checkout proforma-v1.5 -- proforma      # recover the v1.5 tree exactly
```

**To ship a new model version:** branch, change `models/`, refit `artifacts/` in the env that will
load it, run `./scripts/smoke.sh` to see exactly which numbers moved, tag on merge.

## Datasets it reads

From `data/`, never a private copy. See `data/README.md` and `../docs/DATA.md`.

| file | used for |
|---|---|
| `data/panel/main-data-v2-stitched.csv` | the site panel: levels, ramps, neighbours, clusters |
| `data/opex/opex-data.csv` | the P&L |
| `data/ref/site_carwash_types.csv` | wash types (express vs not) |
| `data/ref/old-excel-proforma-data-enriched.csv` | tunnel length → build CAPEX |
| `data/ref/merged_all_sites.csv` | site coordinate lookup (UI + backtests) |

## Running it

```bash
# the Streamlit explorer — conda proforma311, from the repo root
streamlit run proforma/ui/app.py                         # http://localhost:8501

# the model, directly — conda sonnysDataCollection
python -c "from proforma.models import coldstart as cm; \
           traj, meta = cm.predict_site(29.798555, -95.719688); print(meta)"
```

`predict_site` returns a **2-tuple** `(DataFrame, meta)` — its docstring says otherwise and the
docstring is wrong (`../docs/DIVERGENCES.md` §4).

## Two consumers, one model

`models/coldstart.py` is imported two ways, and they must not drift:

1. **In-process by the Streamlit app** (`ui/app.py`).
2. **By the FastAPI backend** — `app/pnl_analysis/modelling/data.py` does
   `from proforma.models import coldstart as cm`.

The *model* is shared. The **P&L and market math around it is implemented twice** — once in `ui/`,
once in `app/pnl_analysis/modelling/*` — and the two have drifted. Read `../docs/DIVERGENCES.md` §1
before changing forecast behaviour.

## The artifact

`artifacts/coldstart_artifacts.joblib` (~46 MB) is a plain pickle of numpy arrays plus lightgbm /
sklearn estimators. It carries **no reference to the module that wrote it**, so the module can be
renamed freely — but it *is* coupled to library versions.

> Refit it in the environment that will load it: conda `sonnysDataCollection`.
> Refitting in the 3.13 `venv` produces an artifact the backend cannot unpickle.

Refit with `cm.fit(save=True)`. Inference-time changes (anchor calibration, the ASP-corruption
filter, breakeven) need no refit. After any refit, run `./scripts/smoke.sh`.

## Backtests

`backtests/*.py` are **scripts with module-level side effects** — they read data and fit models on
import. Run them, never import them. They are excluded from the import smoke test for that reason
(`scripts/_golden/import_smoke.py:AST_ONLY_PREFIXES`).

```bash
python proforma/backtests/backtest_anchor.py    # from the repo root
```
