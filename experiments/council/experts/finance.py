"""
Finance Analyst — consolidates the cluster forecast into a 5-year P&L (revenue, opex, CAPEX, net,
breakeven). Runs LAST via `consolidate()`, once the other seats' numbers are on the board; `investigate`
is a no-op by contract (the base class already returns `[]`; kept explicit here for clarity).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from experiments.council import config as C
from experiments.council import datasets
from experiments.council import forecast
from experiments.council.experts.base import Expert
from experiments.council.protocol import BeliefState, Evidence

DEFAULT_CAPEX = 1.5e6   # fallback when the nearest-build CAPEX lookup has no comparables


def _opex_pct(m: int) -> float:
    """Fixed opex-as-%-of-revenue curve (inlined copy of proforma/pnl/opex._synthetic_opex_pct): settles
    near OPEX_MATURE, hot in the first few months, decaying with age `m` (months since opening)."""
    pct = C.OPEX_MATURE + C.OPEX_HOT * np.exp(-m / C.OPEX_TAU)
    return float(np.clip(pct, 0.30, 1.5))


def _breakeven_month(revenue: List[float], opex: List[float], capex: float) -> Optional[int]:
    """First month index where cumulative (revenue - opex) reaches CAPEX, else None (never breaks even
    within the horizon)."""
    cum = 0.0
    for m, (r, o) in enumerate(zip(revenue, opex)):
        cum += r - o
        if cum >= capex:
            return m
    return None


class FinanceExpert(Expert):
    name = "finance"
    role = "Finance Analyst"
    persona = ("You turn the cluster forecast into a 5-year P&L — revenue less opex less CAPEX — and call "
               "whether the build pays for itself, and by when.")
    is_world = False

    def investigate(self, ws) -> List[Evidence]:
        return []

    def consolidate(self, ws) -> List[Evidence]:
        proj = forecast.project_site(ws.lat, ws.lon)
        revenue = proj["revenue"]
        opex = [revenue[m] * _opex_pct(m) for m in range(len(revenue))]

        capex_info = datasets.capex_for_pin(ws.lat, ws.lon)
        looked_up_capex = capex_info.get("capex")
        capex = float(looked_up_capex) if looked_up_capex is not None else DEFAULT_CAPEX

        revenue_5yr = float(sum(revenue))
        opex_5yr = float(sum(opex))
        net_5yr = revenue_5yr - opex_5yr - capex
        breakeven = _breakeven_month(revenue, opex, capex)

        return [
            self.ev("fin.revenue_5yr", "5-yr revenue", revenue_5yr, unit="$",
                    source="forecast.project_site", confidence=0.6),
            self.ev("fin.opex", "5-yr opex", opex_5yr, unit="$",
                    source="OPEX_MATURE/OPEX_HOT decay curve", confidence=0.5),
            self.ev("fin.capex", "CAPEX", capex, unit="$",
                    source="datasets.capex_for_pin" if looked_up_capex is not None
                           else "default (no nearby comparables)",
                    confidence=0.6 if looked_up_capex is not None else 0.3),
            self.ev("fin.net_5yr", "5-yr net (revenue - opex - CAPEX)", net_5yr, unit="$",
                    source="fin.revenue_5yr - fin.opex - fin.capex", confidence=0.55),
            self.ev("fin.breakeven", "breakeven month", breakeven, unit="mo",
                    source="first month cumulative(revenue-opex) >= CAPEX", confidence=0.55),
            self.ev("fin.membership", "membership share", proj["mem_share"],
                    source="forecast.project_site", confidence=0.6),
        ]

    def initial_belief(self, ws) -> BeliefState:
        net_ev = ws.evidence.get("fin.net_5yr")
        be_ev = ws.evidence.get("fin.breakeven")
        net_5yr = net_ev.value if net_ev is not None else None
        breakeven = be_ev.value if be_ev is not None else None

        if net_5yr is None:
            lean, confidence = None, 0.3
        elif net_5yr > 0 and breakeven is not None:
            lean, confidence = "Build", 0.7
        elif net_5yr < 0:
            lean, confidence = "Pass", 0.6
        else:
            lean, confidence = "Conditional", 0.5

        return BeliefState(expert=self.name, lean=lean, confidence=confidence, key_number=net_5yr,
                           key_number_label="5-yr net $",
                           supporting=[e.eid for e in ws.evidence_of(self.name)])
