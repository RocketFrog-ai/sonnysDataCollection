# Tunnel-length → build-CAPEX model — model card

Sets the build CAPEX line in the forecast-tab P&L from the chosen tunnel length.

**Artifact:** none — this model is small enough to fit from CSV at call time
(`proforma/models/tunnel_capex.py`; data:
`proforma/data/ref/old-excel-proforma-data-enriched.csv`). This folder holds the card only.

## How it works

Robust linear fit of total project cost vs tunnel length over ~187 historical proforma
builds. Slope ≈ **$98k per metre of tunnel**; correlation ≈ 0.53 — tunnel length explains
roughly a quarter of build-cost variance, the rest is land/site-work spread, so the P&L
treats the output as a central estimate, not a quote.

## Consumers

- Streamlit forecast tab (`proforma/ui/panels/_pinpoint_forecast.py`) — CAPEX selector.
- Used in breakeven/payback math in the P&L panels.

## Caveats

- Fitted on historical (pre-2023) proforma budgets; construction-cost inflation since is
  not modelled — treat levels as relative, refresh the CSV to recalibrate.
- No artifact/versioning: behavior changes only when the CSV or the fitting code changes
  (covered by `./scripts/smoke.sh` via the UI first-render capture).
