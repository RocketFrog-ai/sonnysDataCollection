"""
Tunable knobs for the committee — one place, everything reads from here.

Groups: trade-area radii · data eligibility (min-history / COVID guards, mirrored from the modelling code) ·
committee mechanics (rounds, message caps, convergence epsilons, LLM budget) · vote weighting · capacity /
opex constants · display metadata (verdict labels, message + expert colours for the chamber and transcript).
Pure literals; no imports beyond the mile→km constant.
"""
from __future__ import annotations

MI_TO_KM = 1.60934

# ── trade-area radii ──
HISTORICAL_CLUSTER_MI = 12.0                 # Dhruv: Historical reads the 12-mile neighbour cluster
HISTORICAL_CLUSTER_KM = HISTORICAL_CLUSTER_MI * MI_TO_KM   # ≈ 19.31 km
RADIUS_KM = 20.0                             # signal decider / snapshot neighbourhood (as the decider was featurised)
GOOGLE_PLACES_RADII_MI = (3.0, 5.0)          # Competition: real rival counts at 3 & 5 miles
COMPETITION_RADIUS_MI = 5.0
INDEPENDENT_RADIUS_MI = 3.0

# ── data eligibility (mirror proforma/pnl/data.py + proforma/models/coldstart.py) ──
MIN_HISTORY_MONTHS = 30      # a site needs ≥30 monthly records to be a reliable comparable / ramp donor (EXPRESS_MIN_MONTHS=30)
MODEL_MIN_MONTHS = 24        # ≥24 months to trust a mature level / anchor (coldstart MODEL_MIN_MONTHS)
MAT_MIN_MONTHS = 4           # ≥4 months in the 18–30 window for a realized mature level (mat_n≥4)
MIN_NBR_MONTHS = 6           # a neighbour needs ≥ this many pre-T months to count in the snapshot
DROP_OPEN_YEARS = (2020,)    # exclude the COVID-2020 cohort from the ramp / comparable pool (coldstart drop_open_years)
FIRST_OPEN_YEAR = 2021       # focal-sample openings start here (post-COVID)

# ── committee mechanics ──
MAX_ROUNDS = 3               # discussion rounds after the publish round

# The meeting AGENDA — one topic per deliberation round, so the debate covers the whole case instead of
# fixating on the single most quotable disagreement. Each entry: (topic, what to debate, owner seats —
# always woken that round even if settled, because their evidence carries the topic).
ROUND_AGENDA = {
    1: ("the demand case",
        "the projected mature washes vs the cluster's OBSERVED washes/revenue; EACH considered site's own "
        "numbers (washes, revenue, mem-buys, ASP, distance, opened date — only sites meeting the ≥30mo/"
        "matured/non-COVID bar are considered); the ramp pattern; and whether demographics (projected "
        "forward growth, income, vehicles) support that demand",
        ("historical", "local_market")),
    2: ("competition & pricing",
        "TRUE express-tunnel competition (the benchmark score, effective rivals at 3mi); mass merchants and "
        "grocery anchors — traffic generators vs share-of-wallet competition; and what the cluster's observed "
        "membership/retail ASPs say about pricing power here",
        ("competition", "local_market", "historical")),
    3: ("economics & capacity",
        "Finance's 5-yr revenue and net, the CAPEX, the breakeven month, membership share; whether the tunnel "
        "length actually handles the projected peak month; and what breaks the payback case",
        ("finance", "capacity", "historical")),
}
MAX_MSGS_PER_EXPERT = 3      # messages an expert may emit per react
MAX_FOLLOWUPS_PER_EXPERT = 2 # dynamic REQUEST investigations an expert will service
EPS_CONF = 0.05              # convergence: |Δconfidence| this small across two rounds ⇒ stable
EPS_NUM = 0.05               # convergence: relative |Δkey_number| this small ⇒ stable
PER_SITE_LLM_BUDGET = 30     # hard cap on LLM calls per site (budget exhaustion forces convergence)

# ── vote weighting (the anti-bullish-drift guardrail) ──
DATA_EXPERT_WEIGHT = 1.0     # a data-grounded lean counts this much
WORLD_EXPERT_WEIGHT = 0.4    # a world-knowledge (🌐) lean counts less

# ── capacity (tunnel length) — confirmed formula: peak_hour + 20 ft ──
OPERATING_DAYS = 25          # repo tunnel-proxy convention
OPERATING_HOURS = 10
TUNNEL_BUFFER_FT = 20        # Dhruv: peak-hour throughput + 20 = tunnel length in feet

# ── opex fixed pattern (inlined copy of proforma/pnl/opex._synthetic_opex_pct) ──
OPEX_MATURE = 0.45           # opex settles to ~45% of revenue
OPEX_HOT = 0.17              # + up to 17 pts early (hot ~62%), decaying with age
OPEX_TAU = 6.0               # months decay constant

# ── LLM ──
LLM_TEMPERATURE = 0.2
REACT_MAX_TOKENS = 700
RESEARCH_MAX_TOKENS = 900

# ── display: verdict vocabulary (investment lingo → plain language) ──
VERDICT_LABELS = {
    "Build": "Build",
    "Pass": "Don't build",
    "Conditional": "Conditional",
    "Insufficient": "Not enough signal",
}

# expert order (Finance last — it consolidates the others' numbers)
EXPERT_ORDER = ["historical", "competition", "local_market", "capacity", "finance"]

# per-expert display metadata (name · emoji figurine · colour) for the chamber + transcript
EXPERT_META = {
    "historical":   {"name": "Historical",   "emoji": "📊", "color": "#2563eb"},
    "competition":  {"name": "Competition",  "emoji": "🛰️", "color": "#dc2626"},
    "local_market": {"name": "Local-Market", "emoji": "🏙️", "color": "#16a34a"},
    "capacity":     {"name": "Capacity",     "emoji": "🏗️", "color": "#9333ea"},
    "finance":      {"name": "Finance",      "emoji": "💰", "color": "#ea580c"},
    "facilitator":  {"name": "Facilitator",  "emoji": "🧭", "color": "#0f172a"},
}

# message-type colours (chamber speech bubbles + transcript chips)
MSG_COLORS = {
    "PUBLISH":   "#64748b",   # slate — assert a finding
    "QUESTION":  "#3b82f6",   # blue  — clarify
    "CHALLENGE": "#ef4444",   # red   — dispute
    "REQUEST":   "#8b5cf6",   # violet— ask for a new investigation
    "REVISE":    "#f59e0b",   # amber — update my belief
    "ENDORSE":   "#22c55e",   # green — agree
    "VOTE":      "#0f172a",   # ink   — final lean
}

# verdict → colour (chamber halo + verdict badge)
VERDICT_COLORS = {"Build": "#16a34a", "Pass": "#dc2626", "Conditional": "#f59e0b", "Insufficient": "#64748b"}

# which experts are "world-knowledge" (get WORLD_EXPERT_WEIGHT) vs data-grounded (DATA_EXPERT_WEIGHT).
# Local-Market leans on world-knowledge narration, so it is down-weighted per the hard lesson.
WORLD_EXPERTS = {"local_market"}
