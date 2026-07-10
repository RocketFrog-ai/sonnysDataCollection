# Environments

There are **three** Python environments and they are not interchangeable. Picking the wrong one is
the most common way to waste an afternoon here.

| Env | Python | Defined in | Runs |
|---|---|---|---|
| conda `sonnysDataCollection` | 3.9 | `environment.yml` | FastAPI backend (`app.main`, `app.pnl_only`) + Celery |
| conda `proforma311` | 3.11 | `environment-proforma311.yml` | the Streamlit app (`proforma/v1_5/ui/app.py`) |
| `venv/` | 3.13 | not checked in (gitignored, ~1.2 GB) | ad-hoc dev |

```bash
conda env create -f environment.yml               # backend
conda env create -f environment-proforma311.yml   # streamlit  (needs --solver=libmamba on slow links)
```

## The joblib rule

`proforma/v1_5/artifacts/coldstart_artifacts.joblib` is a **plain pickle of library objects** — a
dict of numpy arrays plus `lightgbm.Booster` / `LGBMRegressor` / `sklearn` `ExtraTreesRegressor`
estimators. It contains **no reference to the module that wrote it**, verified by walking every
`GLOBAL`/`STACK_GLOBAL` opcode in the pickle. That is why `coldstart_model.py` could be renamed to
`proforma/v1_5/models/coldstart.py` without a compat shim, and why it loads fine from a directory
where `coldstart_model` is not importable.

What the artifact *is* coupled to is **library versions**:

> **Refit the artifact in the environment that will load it.** For the backend that means conda
> `sonnysDataCollection`. Refitting in the 3.13 `venv` produces an artifact the backend cannot
> unpickle.

Inference-time logic (anchor calibration, the ASP-corruption filter, breakeven) needs no refit.

### The mismatch you will see, and why it is currently benign

The artifact was fitted under **scikit-learn 1.6.1** (the backend env). `environment-proforma311.yml`
pins **scikit-learn 1.8.0**. So the Streamlit app loads a pickle written by a different sklearn and
emits:

```
InconsistentVersionWarning: Trying to unpickle estimator ExtraTreeRegressor
from version 1.6.1 when using version 1.8.0
```

This was checked rather than assumed: the full 24-case golden capture was run under **both** envs
and the outputs were **bit-identical**. So today the warning is cosmetic.

sklearn does not *guarantee* that across versions, and the warning exists precisely because a
future version could silently change a prediction. Treat it as a real risk that happens not to have
bitten yet. If you refit, refit for the backend (1.6.1) — that is the env that must unpickle it —
and re-run `scripts/smoke.sh` under both envs.

## Verifying an environment

```bash
./scripts/smoke.sh          # uses the right interpreter for each component automatically
```

`smoke.sh` resolves both conda envs from `conda info --base` and fails loudly if either is missing.
It checks the artifact unpickles in the **backend** env specifically, since that is the one where a
version mismatch would be fatal rather than cosmetic.

## Import resolution — there is no packaging

By deliberate choice there is no `pyproject.toml` and nothing is `pip install -e .`'d. Imports
resolve off the **repo root**, which must be the CWD (or on `PYTHONPATH`). `proforma`, `proforma.v1_5`,
`libs`, and `libs.carwash_type` are implicit namespace packages; `app` and `proforma.v1_6` are
regular packages.

The one wrinkle: `streamlit run` puts only the *script's own directory* on `sys.path`
(`streamlit/web/bootstrap.py:59`), never the repo root. So the three Streamlit **entrypoints** under
`proforma/v1_5/ui/` each bootstrap the repo root onto `sys.path` before importing `app.*` or
`proforma.*`. No library module does this. Do not remove those lines without replacing them with a
`PYTHONPATH` launcher — see `docs/DIVERGENCES.md` §7.

The startup scripts additionally put `app/site_analysis/features/...` on `PYTHONPATH` so that the
feature modules' bare intra-feature imports resolve. That is why those modules cannot simply be
imported from the repo root.
