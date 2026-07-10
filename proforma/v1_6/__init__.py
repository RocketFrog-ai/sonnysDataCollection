"""
Council of Agents + Retrospective Backtesting — an isolated experiment.

This package assembles the car-wash insight modes that already live in
`app.pnl_analysis.insights` into a *council* of analysts that reason about one
build/no-build decision and can disagree, then adjudicates their verdicts with a
DETERMINISTIC rulebook (not a free-form LLM fusion). It also ships a retrospective
backtest harness that freezes the council at each site's opening month T, feeds it
only the local market as it looked strictly before T, and grades the go/no-go call
against what actually happened after T.

Design constraints (see plan / README):
  • Single data source: proforma/data/panel/main-data-v2-stitched.csv.
  • Build date T = each site's `operational_start` (first-data-entry month).
  • Reuses the existing insight functions UNCHANGED — nothing under app/ is edited.
  • Run from the repo root: `python -m council.harness`.
"""
