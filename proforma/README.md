# `proforma/` — all modelling, versioned

Everything that forecasts lives here. Nothing else in the repo does modelling.

```
proforma/
├── data/     shared, immutable, named datasets — ONE copy, read by every version
├── v1_5/     ── LIVE ──        the cold-start forecaster + the Streamlit app
└── v1_6/     ── IN PROGRESS ── the "council" retrospective backtest
```

## The version map

| version | status | what it is | reads |
|---|---|---|---|
| `v1_5` | **live** | plateau × ramp × cannibalization cold-start model, its artifacts, the Streamlit explorer, and the backtests behind it | `data/panel`, `data/opex`, `data/ref` |
| `v1_6` | in progress | a council of LLM "seats" + deterministic adjudication, graded against a leakage-clean retrospective backtest | `data/panel` |

`v1_5` is what the FastAPI backend imports (`from proforma.v1_5.models import coldstart`) and what
the Streamlit app runs. `v1_6` is an isolated experiment: nothing in `app/` imports it, and it does
not affect any forecast. Its headline finding is that greenfield mature-*level* is close to
unpredictable on this data, and the only leakage-clean edge is a small go/no-go signal
(out-of-fold AUC 0.57) that beats "always build".

Older work is not here. The old `earnest-proforma-1.5` tree (now `archive/proforma-v1_0/`), the
two-track LightGBM forecaster, and the IDW neighbour baseline are frozen under `archive/` — read
them for method history, don't build on them.

## Data lives once

There is a real tension between "the panel is shared across model versions" and "the old tree named
it after a version". It is resolved by **versioning the dataset by filename, not by folder**:
`data/panel/main-data-v2-stitched.csv` is one file, and each version's README declares which
datasets it consumes. The two byte-identical mirrors that used to exist were verified equal and
collapsed. See `data/README.md` and `../docs/DATA.md`.

Artifacts are the opposite: they belong to a **version**, because they are welded to the code that
fitted them. `v1_5/artifacts/coldstart_artifacts.joblib` is `v1_5`'s, and refitting it is a `v1_5`
operation. See `../docs/ENVIRONMENTS.md` for the refit rule.

## Adding a `v1_7`

1. `mkdir -p proforma/v1_7/{models,artifacts}` — a `README.md` stating status, inputs, and how to
   run it is not optional.
2. Import shared data by path from `proforma/data/…`. **Do not copy a dataset into your version.**
   If you need a new dataset, add it under `proforma/data/` and document its provenance in
   `data/README.md`.
3. Give it its own `artifacts/`. Never load another version's artifact — the pickle is coupled to
   the library versions and the feature list of the code that wrote it.
4. Leave `v1_5` alone until `v1_7` is proven against the golden baseline (`scripts/smoke.sh`).
   Two live versions is fine; a silently-swapped one is not.
5. Point `app/pnl_analysis/modelling/data.py` at the new version only as a deliberate, separate
   commit, and re-run `scripts/smoke.sh` — it will tell you exactly which numbers moved.
