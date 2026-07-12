"""
AI Site-Selection Committee — a talking, reasoning council (isolated experiment).

Five domain experts (Historical · Competition · Local-Market · Capacity · Finance) investigate a
candidate car-wash site, publish evidence to a shared blackboard, then QUESTION / CHALLENGE / REQUEST /
REVISE / ENDORSE / VOTE across bounded discussion rounds run by a facilitator, until they reach a
consensus recommendation — and emit an in-depth report + a Streamlit "council chamber" you can watch.

Design constraints (see the approved plan and README):
  • Data: the council's OWN copies under experiments/council/data/ only
    (Council--historical-data.csv = the stitched panel, Council--site-wise-data.csv, Council--old-proforma-data.csv)
    plus named insight APIs (competition-coverage, local-market-analysis, washcount forecast) and the
    fixed opex pattern. LLM narration is world-knowledge, badged and down-weighted.
  • LLM: the Azure endpoint only (app.pnl_analysis.insights.llm.complete backend="azure").
  • Deterministic Python routes messages / weights votes / writes the verdict; the LLM only proposes text.
  • The leakage-clean signal decider is computed and put on the board as evidence + a divergence flag;
    the committee decides (MODE="deliberative"); MODE="anchored" makes the signal govern the binary.
  • Isolated: imports app.* / proforma.* one-way; nothing under app/ is edited.

Run from the repo root (conda `sonnys`):
  python -m experiments.council.harness            # honest, no-LLM signal backtest (N≈420)
  python -m experiments.council.harness --llm --limit 6   # qualitative committee sample
  streamlit run experiments/council/streamlit_view.py     # the animated chamber, standalone
"""
