"""
Tunable knobs for the council — change here, everything reads from here.

Per-seat trade-area radii are intentionally different: the internal grounded read and the qualitative
location read use the wider local market, while the competition seat uses a TIGHT trade area (express
washes draw from ~3–5 mi, so a wide radius over-counts rivals), and the independent seat sizes the
market at its own radius.
"""
from __future__ import annotations

MI_TO_KM = 1.60934

# ── trade-area radii ──
RADIUS_KM = 20.0                 # local market: internal grounded seat + neighbourhood + location read
COMPETITION_RADIUS_MI = 5.0      # competition seat — tight trade area (was implicitly 12 mi; over-counted)
INDEPENDENT_RADIUS_MI = 3.0      # independent seat — radius it sizes the market at

# ── neighbourhood / weighting ──
MIN_NBR_MONTHS = 6               # a neighbour needs ≥ this many pre-T months to count
W_INTERNAL = None                # None → internal seat weight ≈ sum of external weights (dominant)

# ── display ──
# the internal Build/Pass/Conditional vocabulary is investment lingo ("I'll pass" = decline); map it to
# plain language for humans in the UI and notes.
VERDICT_LABELS = {
    "Build": "Build",
    "Pass": "Don't build",
    "Conditional": "Conditional",
    "Insufficient": "Not enough signal",
}
