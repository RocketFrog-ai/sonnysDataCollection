# SUPER ensemble v1 — model card

Calibrated wash-count **level** layer on top of cold-start Model 3 (plateau × ramp). Built
2026-07 from the old-proforma backtest programme (`experiments/old-proforma-analysis/`,
full context in that folder's `context.md`; charts:
https://claude.ai/code/artifact/f75cc9a5-8755-468f-82c7-9d450ffd8911).

## What it is

Model 3 predicts a new site's trajectory from location alone. Its dominant error is the
**mature level** (oracle test: given the true level, Y2 forecast error falls 39%→13%).
This artifact adds two small ridge layers that fix the level, then rescales Model 3's
ramp and P10–P90 bands and applies per-operating-year debias constants:

- **level_A — "with user inputs"**: ridge on `[log plateau, pay-stations score,
  vacuum score, lot-type score, log traffic]`. The choice→score maps are embedded in the
  artifact (they are the proforma Excel's own weights). Trained on the 70 mature
  proforma-matched sites — the only population where these inputs exist today.
- **level_B — "pin only" fallback**: ridge calibration of the plateau from
  `predict_site`'s own info (model vs anchor plateau, anchor size, region, open cohort).
  Trained on 862 panel-eligible sites (opened 2021–24, ≥24 clean months).
- **year_debias**: per-op-year median log-residual constants (cross-fitted on the 862).

## Runtime inputs

Required: **lat, lon, planned open year**. Optional (activates level_A, each proven
signal): **pay stations** ("1"/"2"/"3 or more"/"live person"), **free vacuum slots**
("less than 12"/"12 - 20"/"more than 20"/"coin or none"), **lot type** ("corner lot
with light"/"corner lot without light"/"inside lot near light"/"inside lot no light"),
**daily traffic count**. Everything else (neighbours, anchors, ramps, trends) is fetched
by the cold-start model itself.

```python
from proforma.models import super_ensemble as se
traj, info = se.predict_site_super(lat, lon, open_year=2026,
    pay_stations="2", vacuum_slots="12 - 20",
    lot_type="corner lot with light", traffic_count=25000)
```

## Accuracy (all leakage-audited; LOSO = leave-one-site-out)

| Path | Population | MdAPE (mature) | within ±20% | rank ρ |
|---|---|---|---|---|
| raw Model 3 (reference) | 862 panel | 32.9% | 33% | 0.46 |
| level_B (pin only) | 862 panel, 10-fold CV | 31.8% | 30% | 0.47 |
| level_A (4 user inputs) | 70 matched, nested LOSO | **29.6%** | **38.6%** | 0.58 |
| old proforma (reference) | 70 matched | 58.5% | 21% | 0.31 |

Site-year forecasts (level × ramp × year debias) beat the old proforma with MdAPE
37.0% vs 40.9%, winning 62.6% of 227 paired site-years (sign test p=0.0002; survives
×5 multiple-comparison correction). Negative controls: with shuffled targets the fitted
skill collapses (ρ→0.02) — signal, not leakage.

## Limitations / honest caveats

- level_A is trained on n=70 old-proforma sites; its gain (−4 MdAPE pts) is measured on
  that cohort and its transfer panel-wide is unverified until capacity attributes are
  collected for panel sites.
- Ex-ante site-level error is structurally ~±30% (panel year-over-year noise is ~7%,
  persistence Y2→Y3 is ~9% — the missing information is the site itself). Quote P10–P90,
  not the point. After opening, re-forecast from the site's own months (updater in
  `experiments/old-proforma-analysis/ensemble/`).
- Refit path: `experiments/old-proforma-analysis/ensemble/fit_super_artifact.py`
  (reads that folder's `results/` CSVs; rerun the pipeline there after a panel refresh).
- joblib welded to conda `sonnys` sklearn — same unpickle rule as coldstart.
