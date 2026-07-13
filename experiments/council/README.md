# `experiments/council` — AI Site-Selection Committee

A **committee of five domain experts that talks its way to a build/no-build decision** for a candidate
car-wash site. Each expert fetches real data, publishes it to a shared board, then the seats
**question · challenge · request · revise · endorse · vote** across bounded rounds until they converge —
and the run emits an in-depth report plus an **animated "council chamber"** you can watch.

**Status:** experiment. Fully **self-contained** under `experiments/council/` — makes **no `app.*` /
`proforma.*` imports**, edits nothing in production. The only touch-point is the already-`try/except` hook
in `proforma/ui/panels/_explore_markets.py`, which auto-wires the Streamlit view.

## Run (repo root, conda `sonnys`)
```bash
streamlit run experiments/council/streamlit_view.py        # the animated chamber, standalone
python -m experiments.council.harness                      # honest, no-LLM signal backtest (N≈420)
python -m experiments.council.harness --committee --limit 6   # qualitative committee sample (LLM) + transcripts
python -m experiments.council.harness --committee --light     # committee sample, ZERO LLM (regression guard)
```
Live use in code:
```python
from experiments.council.committee import run_committee_pin
res = run_committee_pin(33.75, -84.39)     # LIVE Azure discussion
print(res.decision.verdict, res.decision.confidence)   # e.g. "Pass" 0.65
print(res.report)                          # the in-depth markdown report
```

## Architecture (no framework lock-in)
A **blackboard system** + a **typed message protocol**, orchestrated as a **LangGraph `StateGraph`**
(this repo's idiom; falls back to an identical hand-rolled loop if langgraph is absent). The LLM only
*proposes* typed messages; deterministic Python routes them, weights the votes, and writes the verdict.

| Layer | File(s) |
|---|---|
| Blackboard + message protocol | `workspace.py`, `protocol.py` (Evidence · Message[7 verbs] · BeliefState) |
| Orchestration | `coordinator.py` (LangGraph graph: investigate → publish → **discuss ⟲ until converged** → finance) + `committee.py` |
| The five experts | `experts/{historical,competition,local_market,capacity,finance}.py` on `experts/base.py` |
| The one LLM touchpoint | `llm_react.py` → `llm.py` (**Azure only**) |
| Decision rule | `anchor.py` — `decide_final` (the committee's weighted-MAJORITY lean; a Build must be earned) |
| Tools (self-contained) | `forecast.py` (panel projection) · `datasets.py` (site-wise + old-proforma) · `places.py` (Google) |
| Output | `report.py` (in-depth report) · `chamber.py` (animated boardroom HTML) · `streamlit_view.py` |
| Backtest | `harness.py` · `decider.py`/`features.py`/`scorer.py`/`snapshot.py`/`data_1_6.py` (substrate) |

## The five experts (each fetches real data, then argues)
- **Historical** — 12-mile neighbour cluster (washcount/revenue/membership/ASP + ramp); each comparable
  carries its opened date, lat/lon and distance from the pin.
- **Competition** — live Google Places rival counts at 3 & 5 mi (the panel has only *our* sites).
- **Local-Market** — site-wise demographics/income + an LLM demographics/seasonality read (world-knowledge, down-weighted).
- **Capacity** — tunnel length from the peak-month proxy (`peak_hour + 20 ft`).
- **Finance** — revenue (washes × ASP), CAPEX (nearest old-proforma builds), opex (fixed pattern) → net / breakeven.

## Data & method
- **One data root:** the council's own copies under `data/` — `Council--historical-data.csv` (the panel),
  `Council--site-wise-data.csv` (demographics), `Council--old-proforma-data.csv` (CAPEX). Eligibility mirrors
  the modelling code: **≥30 months of history**, **≥24 to trust a mature level**, the **COVID-2020 cohort dropped**.
- **The committee decides — alone.** The verdict is the seats' weighted-MAJORITY lean; a Build must be
  earned (a real majority AND no challenge left standing). The leakage-clean signal decider survives only
  as the **offline backtest** (`python -m experiments.council.harness`, N≈420, out-of-fold: +10 pt build
  precision, AUC 0.572) that grades the method honestly — it never appears in a live meeting.

## The hard lesson, encoded
The prior council was one-pass and had no predictive edge. This one is a real deliberation (challenge →
defend → revise → converge), but the **data still governs**: data-grounded seats out-weigh world-knowledge
ones in the tally, and uncited claims are dropped. Deliberation adds *explainability and a traceable,
watchable report* — the honest measure of edge stays in the offline harness backtest.
