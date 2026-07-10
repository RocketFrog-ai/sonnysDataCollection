# `proforma/v1_5` — ── LIVE ──

The cold-start forecaster. Drop a pin on a US map for a site that does not exist yet; get a 5-year
monthly wash-count forecast, a P&L, and what it does to the existing neighbours.

**This is the version in production.** The FastAPI backend imports it; the Streamlit app runs it.

## Layout

```
v1_5/
├── MODELLING.md      the authoritative method reference — read this before changing math
├── models/           coldstart.py (the model), tunnel_capex.py
├── artifacts/        coldstart_artifacts.joblib — fitted; DO NOT refit casually
├── ui/               Streamlit only. app.py is a thin entry; panels/ holds the modes
├── backtests/        the evidence behind the model's design choices. Scripts, not libraries.
├── notebooks/        exploration + the figures in MODELLING.md
└── scripts/          process_main_data_v2.py — rebuilds the panel
```

## Datasets it reads

From the shared store at `proforma/data/` — never a private copy:

| file | used for |
|---|---|
| `../data/panel/main-data-v2-stitched.csv` | the site panel: levels, ramps, neighbours, clusters |
| `../data/opex/opex-data.csv` | the P&L |
| `../data/ref/site_carwash_types.csv` | wash types (express vs not) |
| `../data/ref/old-excel-proforma-data-enriched.csv` | tunnel length → build CAPEX |
| `../data/ref/merged_all_sites.csv` | site coordinate lookup (UI + backtests) |

## Running it

```bash
# the Streamlit explorer — conda proforma311, from the repo root
streamlit run proforma/v1_5/ui/app.py                    # http://localhost:8501

# the model, directly — conda sonnysDataCollection
python -c "from proforma.v1_5.models import coldstart as cm; \
           traj, meta = cm.predict_site(29.798555, -95.719688); print(meta)"
```

`predict_site` returns a **2-tuple** `(DataFrame, meta)` — its docstring says otherwise and the
docstring is wrong (`docs/DIVERGENCES.md` §4).

## Two consumers, one model

`models/coldstart.py` is imported two ways, and they must not drift:

1. **In-process by the Streamlit app** (`ui/app.py`).
2. **By the FastAPI backend** — `app/pnl_analysis/modelling/data.py` does
   `from proforma.v1_5.models import coldstart as cm`. (This used to be a `sys.path.insert` hack
   pointing into the Streamlit directory. It is gone.)

The *model* is shared. The **P&L and market math around it is implemented twice** — once in
`ui/app.py`, once in `app/pnl_analysis/modelling/*` — and the two have drifted. Changing forecast
behaviour means deciding which of the three places it belongs in. Read `docs/DIVERGENCES.md` §1
first.

## The artifact

`artifacts/coldstart_artifacts.joblib` (~46 MB) is a plain pickle of numpy arrays plus lightgbm /
sklearn estimators. It carries **no reference to the module that wrote it**, so the module can be
renamed freely — but it *is* coupled to library versions.

> Refit it in the environment that will load it: conda `sonnysDataCollection`.
> Refitting in the 3.13 `venv` produces an artifact the backend cannot unpickle.

Refit with `cm.fit(save=True)`. Inference-time changes (anchor calibration, the ASP-corruption
filter, breakeven) need no refit. After any refit, run `scripts/smoke.sh` — it will tell you
exactly which numbers moved.

## Backtests

`backtests/*.py` are **scripts with module-level side effects** — they read data and fit models on
import. Run them, never import them. They are excluded from the import smoke test for that reason
(`scripts/_golden/import_smoke.py:AST_ONLY_PREFIXES`).

```bash
python proforma/v1_5/backtests/backtest_anchor.py    # from the repo root
```
