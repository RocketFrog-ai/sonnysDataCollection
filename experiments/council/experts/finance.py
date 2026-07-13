"""
Finance Analyst — consolidates the cluster forecast into a 5-year P&L (revenue, opex, CAPEX, net,
breakeven). Runs LAST via `consolidate()`, once the other seats' numbers are on the board.

  • Revenue  = the Historical/forecast wash trajectory × the local cluster ASP (from `forecast.project_site`).
  • Opex     = a fixed opex-%-of-revenue curve (inlined `_synthetic_opex_pct`: hot early, ~45% at maturity).
  • CAPEX    = **demand-driven** — matched to the Capacity seat's tunnel length (bigger projected volume →
               longer tunnel → bigger build → more CAPEX), via `datasets.capex_for_pin(tunnel_ft=…)`.
  • Net / breakeven = revenue − opex − CAPEX, and the first month cumulative net clears CAPEX.
  • Web benchmarks (full mode) = a live web read of current express-tunnel build cost, membership pricing and
    operating margins, so the ROI argument is grounded in real numbers, not just the model.
"""
from __future__ import annotations

import json
from typing import List, Optional

import numpy as np

from experiments.council import config as C
from experiments.council import datasets
from experiments.council import forecast
from experiments.council.experts.base import Expert
from experiments.council.protocol import BeliefState, Evidence

DEFAULT_CAPEX = 1.5e6   # fallback when the CAPEX lookup has no comparables

_FIN_SYS = (
    "You are a Finance Analyst for a NEW EXPRESS-TUNNEL car wash. From the LIVE WEB SOURCES, extract concrete "
    "BENCHMARKS a committee can argue with: typical express-tunnel build / CAPEX cost, unlimited-membership "
    "monthly price, and operating-margin norms — for this area if the sources say so.\n"
    'Return STRICT JSON only, no prose: {"capex_benchmark": "...", "pricing_benchmark": "...", '
    '"margin_benchmark": "..."}. Each 1-2 sentences WITH numbers; use "n/a" if the sources do not cover it.'
)


def _opex_pct(m: int) -> float:
    """Fixed opex-as-%-of-revenue curve (inlined copy of proforma/pnl/opex._synthetic_opex_pct): settles
    near OPEX_MATURE, hot in the first few months, decaying with age `m` (months since opening)."""
    return float(np.clip(C.OPEX_MATURE + C.OPEX_HOT * np.exp(-m / C.OPEX_TAU), 0.30, 1.5))


def _breakeven_month(revenue: List[float], opex: List[float], capex: float) -> Optional[int]:
    """First month index where cumulative (revenue - opex) reaches CAPEX, else None (never within horizon)."""
    cum = 0.0
    for m, (r, o) in enumerate(zip(revenue, opex)):
        cum += r - o
        if cum >= capex:
            return m
    return None


class FinanceExpert(Expert):
    name = "finance"
    role = "Finance Analyst"
    persona = ("You turn the cluster forecast into a 5-year P&L — revenue less opex less a demand-sized CAPEX "
               "— and call whether the build pays for itself, and by when, checking it against live cost and "
               "membership-pricing benchmarks from the web.")
    is_world = False

    def investigate(self, ws) -> List[Evidence]:
        return []

    def consolidate(self, ws) -> List[Evidence]:
        proj = forecast.project_site(ws.lat, ws.lon)
        revenue = proj["revenue"]
        opex = [revenue[m] * _opex_pct(m) for m in range(len(revenue))]

        # CAPEX scales with the demand-driven tunnel length (from the Capacity seat), else nearest by location
        tun_ev = ws.evidence.get("cap.tunnel_ft")
        tunnel_ft = float(tun_ev.value) if (tun_ev is not None and isinstance(tun_ev.value, (int, float))) else None
        capex_info = datasets.capex_for_pin(ws.lat, ws.lon, tunnel_ft=tunnel_ft)
        looked_up = capex_info.get("capex")
        capex = float(looked_up) if looked_up is not None else DEFAULT_CAPEX
        capex_basis = capex_info.get("basis") if looked_up is not None else "default (no comparables)"

        revenue_5yr, opex_5yr = float(sum(revenue)), float(sum(opex))
        net_5yr = revenue_5yr - opex_5yr - capex
        breakeven = _breakeven_month(revenue, opex, capex)

        out = [
            self.ev("fin.revenue_5yr", "5-yr revenue", revenue_5yr, unit="$",
                    source="forecast.project_site (washes × cluster ASP)", confidence=0.6),
            self.ev("fin.opex", "5-yr opex", opex_5yr, unit="$",
                    source="OPEX_MATURE/OPEX_HOT decay curve", confidence=0.5),
            self.ev("fin.capex", f"CAPEX — {capex_basis}", capex, unit="$",
                    source="datasets.capex_for_pin (tunnel-scaled)" if tunnel_ft is not None else "datasets.capex_for_pin (nearest)",
                    confidence=0.6 if looked_up is not None else 0.3),
            self.ev("fin.net_5yr", "5-yr net (revenue - opex - CAPEX)", net_5yr, unit="$",
                    source="fin.revenue_5yr - fin.opex - fin.capex", confidence=0.55),
            self.ev("fin.breakeven", "breakeven month", breakeven, unit="mo",
                    source="first month cumulative(revenue-opex) >= CAPEX", confidence=0.55),
            self.ev("fin.membership", "membership share", proj["mem_share"],
                    source="forecast.project_site", confidence=0.6),
        ]
        bench = self._web_benchmarks(ws)
        if bench:
            out.append(self.ev("fin.web_benchmarks", "live web cost / pricing / margin benchmarks", bench,
                               kind="text", source="websearch (live)", confidence=0.45, leakage_safe=False))
        return out

    def _web_benchmarks(self, ws) -> Optional[str]:
        """Best-effort live web read of express-tunnel CAPEX, membership pricing and margins → one benchmark
        line for the ROI debate. Skipped in light mode; None on any failure (no key / off / bad JSON)."""
        if getattr(ws, "light", False):
            return None
        try:
            from experiments.council import llm, websearch
            place = websearch.reverse_geocode(ws.lat, ws.lon) or f"{ws.lat:.3f},{ws.lon:.3f}"
            srcs = websearch.web_search(
                f"express tunnel car wash construction cost CAPEX and unlimited membership monthly price {place}",
                max_results=6)
            if not srcs:
                return None
            user = {"place": place, "web_sources": [{"title": s.get("title"), "url": s.get("url"),
                    "snippet": (s.get("content") or "")[:300]} for s in srcs[:6]]}
            text = llm.complete([{"role": "system", "content": _FIN_SYS},
                                 {"role": "user", "content": json.dumps(user, ensure_ascii=False)[:8000]}],
                                json_mode=True, temperature=C.LLM_TEMPERATURE, max_tokens=500)
            j = llm.parse_json_lax(text)
            parts = [str(j.get(k)) for k in ("capex_benchmark", "pricing_benchmark", "margin_benchmark")
                     if j.get(k) and str(j.get(k)).strip().lower() != "n/a"]
            return " · ".join(p[:200] for p in parts) or None
        except Exception:
            return None

    def initial_belief(self, ws) -> BeliefState:
        net_ev = ws.evidence.get("fin.net_5yr")
        be_ev = ws.evidence.get("fin.breakeven")
        net_5yr = net_ev.value if net_ev is not None else None
        breakeven = be_ev.value if be_ev is not None else None

        # A P&L is only as real as its demand anchor: if the projection is a global fallback (no local
        # comparables), every downstream dollar is fabricated — abstain rather than bless it.
        proj_ev = ws.evidence.get("hist.projected_mature")
        if proj_ev is not None and proj_ev.confidence <= 0.25:
            return BeliefState(expert=self.name, lean=None, confidence=0.25, key_number=None,
                               key_number_label="P&L rests on a global fallback — not underwritable",
                               open_concerns=["revenue/net/breakeven all derive from a fallback demand anchor, "
                                              "not local evidence"],
                               supporting=[e.eid for e in ws.evidence_of(self.name)])

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
