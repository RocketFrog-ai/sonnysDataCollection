# Environments

There is **one** Python environment. Everything — the FastAPI backend, the Streamlit app, the model,
the council — runs in it.

| Env | Python | Defined in | Runs |
|---|---|---|---|
| conda `sonnys` | 3.11 | `environment.yml` | all of it |

```bash
conda env create -f environment.yml     # add --solver=libmamba on slow links
conda activate sonnys

uvicorn app.main:app --port 8010        # the API
streamlit run proforma/ui/app.py        # the UI
scripts/smoke.sh                        # prove the numbers still match
```

There used to be three: a py3.9 conda env `sonnysDataCollection` for the backend, a py3.11
`proforma311` for Streamlit (py3.9 caps Streamlit at 1.50), and a py3.13 `venv/` for ad-hoc dev.
Nothing actually needed py3.9. Remove the old ones once you have `sonnys`:

```bash
conda env remove -n sonnysDataCollection && conda env remove -n proforma311
```

If you use `scripts/start_uvicorn_fast_api.sh`, set `CONDA_ENV_NAME=sonnys` in `.env` — the script
refuses to launch unless the active env matches.

## Why `scipy` is pinned to an exact version

`environment.yml` pins `scipy==1.17.1`. Not because 1.17.1 is special, but because it is **the
version `scripts/_golden/baseline/api.json` was captured under**. Bumping scipy moves a number.

`app/pnl_analysis/modelling/pnl.py::opex_pct_curve_fit` fits the opex%-of-revenue decay curve with
`scipy.optimize.curve_fit` — a bounded non-linear least-squares that terminates on a **tolerance**
(`ftol`/`xtol`/`gtol` all default to `1e-8`), not on exactness. Its fitted parameters are therefore
only defined to about `1e-8`, and different scipy builds walk different step paths and stop at
different points. Measured across scipy 1.13.1 → 1.17.1, on inputs verified bit-identical by sha256
(`age`, `y`, `w` all matching; `p0` equal to the last digit):

```
hot  0.8743861484901816  ->  0.8743861486257372     (1.5e-10 relative)
mat  0.38664364199564244 ->  0.3866436420304434     (9.0e-11 relative)
tau  2.4609460487255252  ->  2.4609460460784720     (1.1e-9  relative)
```

`opex = shape × revenue` inherits that, and `net = revenue − expenses` amplifies it by cancellation
to `1.4e-9` — past the `1e-9` the golden harness enforces. In dollars it is **$5.3e-06 on a monthly
net of −$3,733.52**. Everything else in the repo — all 24 coldstart cases, the other 14 API cases,
the whole Streamlit render surface — is bit-identical across scipy versions, and pandas 2.2→3.0 /
numpy 2.0→2.4 change *nothing at all*.

So the pin exists to keep a real (if microscopic) numeric change from being smuggled in under an
unrelated commit. `scripts/smoke.sh` asserts the scipy version before running anything, so a stray
`pip install -U scipy` fails loudly instead of surfacing as a mystery `1.4e-9` in the final diff.

**To upgrade scipy:** bump the pin, run `scripts/smoke.sh --capture-baseline`, confirm that *only*
`cases.expense_plan` moved, and commit the re-baselined `api.json` **on its own** so the diff is
attributable to the upgrade and to nothing else. That is exactly how 1.13.1 → 1.17.1 landed; see
`docs/DIVERGENCES.md` §9.

The deeper lesson: a `1e-9` contract over an iterative optimizer's output was never really pinning
*behavior* — it was pinning a build. It only ever passed because both captures used the same scipy.
If `expense_plan` needs to be genuinely reproducible across solver versions, tighten `curve_fit`'s
tolerances so it converges to the true minimum instead of stopping near it.

## The joblib rule

`proforma/artifacts/coldstart_artifacts.joblib` is a **plain pickle of library objects** — a
dict of numpy arrays plus `lightgbm.Booster` / `LGBMRegressor` / `sklearn` `ExtraTreesRegressor`
estimators. It contains **no reference to the module that wrote it**, verified by walking every
`GLOBAL`/`STACK_GLOBAL` opcode in the pickle. That is why `coldstart_model.py` could be renamed to
`proforma/models/coldstart.py` without a compat shim, and why it loads fine from a directory
where `coldstart_model` is not importable.

What the artifact *is* coupled to is **library versions**:

> **Refit the artifact in the environment that will load it** — now unambiguously conda `sonnys`,
> since there is only one. Refitting in some other interpreter can produce an artifact this one
> cannot unpickle.

Inference-time logic (anchor calibration, the ASP-corruption filter, breakeven) needs no refit.

### The mismatch you will see, and why it is currently benign

The artifact was fitted under **scikit-learn 1.6.1**. `environment.yml` pins **1.8.0**. So loading it
emits:

```
InconsistentVersionWarning: Trying to unpickle estimator ExtraTreeRegressor
from version 1.6.1 when using version 1.8.0
```

This was checked rather than assumed: the full 24-case golden capture was run under both sklearn
versions and the outputs were **bit-identical**. So today the warning is cosmetic.

sklearn does not *guarantee* that across versions, and the warning exists precisely because a future
version could silently change a prediction. Treat it as a real risk that happens not to have bitten
yet. If you refit, refit under `sonnys` and re-run `scripts/smoke.sh`.

## Verifying an environment

```bash
./scripts/smoke.sh
```

It resolves `sonnys` from `conda info --base`, fails loudly if it is missing, asserts the pinned
scipy, then checks the artifact unpickles before running any golden.

## Import resolution — there is no packaging

By deliberate choice there is no `pyproject.toml` and nothing is `pip install -e .`'d. Imports
resolve off the **repo root**, which must be the CWD (or on `PYTHONPATH`). `proforma` and `libs` are
implicit namespace packages; `app` and `experiments.council` are regular packages.

The one wrinkle: `streamlit run` puts only the *script's own directory* on `sys.path`
(`streamlit/web/bootstrap.py:59`), never the repo root. So the Streamlit **entrypoints** under
`proforma/ui/` each bootstrap the repo root onto `sys.path` before importing `app.*` or
`proforma.*`. No library module does this. Do not remove those lines without replacing them with a
`PYTHONPATH` launcher — see `docs/DIVERGENCES.md` §8.

(The startup scripts used to put `app/site_analysis/features/...` on `PYTHONPATH` for that tree's
bare intra-feature imports. The tree is gone; nothing needs `PYTHONPATH` any more.)
