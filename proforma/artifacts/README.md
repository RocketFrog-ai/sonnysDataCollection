# proforma/artifacts — fitted model artifacts

**One folder per model** — the folder name carries the UI model number in brackets; each
folder holds the model card (`README.md`) and that model's joblib(s). Artifacts are
**welded to the environment that fitted them** (conda `sonnys`; scipy/sklearn pinned — see
`docs/ENVIRONMENTS.md`): refit in the env that will load them.

| Folder | Model | Loaded by | Fitted by |
|---|---|---|---|
| `coldstart (models 1-4)/` | **Cold-start plateau×ramp** — the drop-pin wash-count model; UI "Models 1–4" are flag combos of the one artifact (`coldstart_artifacts.joblib`, ~46 MB; prior versions in `backups/`) | `proforma/models/coldstart.py` (`load()`); Streamlit UI + FastAPI via `proforma/pnl/data.py` | `coldstart.fit()` over the panel (`main-data-v2-stitched.csv`) |
| `ensemble_super (model 5)/` | **SUPER ensemble v1** — calibrated level layer on top of Model 3 (UI "Model 5"); beats the old proformas and raw Model 3 in audited backtests | `proforma/models/super_ensemble.py` (`predict_site_super()`, `apply_super()`) | `experiments/old-proforma-analysis/ensemble/fit_super_artifact.py` |
| `tunnel_capex (capex model)/` | **Tunnel-length → CAPEX** — build-CAPEX line in the forecast P&L; no joblib (fits from CSV at call time) | `proforma/models/tunnel_capex.py` | in-module fit over `proforma/data/ref/old-excel-proforma-data-enriched.csv` |

### Loose files at the root (caches & notebook outputs — NOT model artifacts)

- `panel.parquet`, `site.parquet` — coldstart's cached panel (read by live code in
  `coldstart._build_from_csv`; regenerated from the CSV). **Do not move.**
- `events.parquet`, `mo_*.parquet`, `mo_*_examples.json`, `seedroll_*.json`,
  `ft_eval_*.json` — outputs of `proforma/notebooks/` (MOIRAI backtests, coldstart eval).
- `coldstart_features.parquet` — legacy, unreferenced.

Rule of thumb: data lives once under `proforma/data/`; artifacts belong to the model
version that fitted them and live here.
