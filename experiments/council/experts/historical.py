"""
Historical Analyst — the 12-mile neighbour-cluster forecast, plus the leakage-clean signal exhibit.

`investigate` runs the council-local panel forecast (`forecast.project_site`) for the pin and posts its
headline cluster numbers (mature level, 5-yr revenue, membership mix, ASP, ramp shape). It then computes
the leakage-clean signal decider's independent Build/Pass call (`anchor.compute_anchor`) — the one
component with measured out-of-fold edge — and posts it as `signal.decider`, wrapped so a failure there
never costs the rest of this seat's evidence. `initial_belief` defers to that signal when it has one,
else falls back to a level-vs-floor read.
"""
from __future__ import annotations

from typing import List

from experiments.council import anchor as A
from experiments.council import data_1_6 as D
from experiments.council import forecast
from experiments.council.experts.base import Expert
from experiments.council.protocol import BeliefState, Evidence


class HistoricalExpert(Expert):
    name = "historical"
    role = "Historical Analyst"
    persona = ("You read the local 12-mile neighbour cluster's track record — the mature level comparable "
               "sites hold, how they ramped from opening, and what they charge — to project what a new "
               "build here would likely mature into.")
    is_world = False

    def investigate(self, ws) -> List[Evidence]:
        proj = forecast.project_site(ws.lat, ws.lon)
        out = [
            self.ev("hist.cluster_wash", "cluster mature-anchor washes/mo", proj["mature_anchor"],
                    unit="washes/mo", source="forecast.project_site", confidence=0.7),
            self.ev("hist.cluster_rev", "cluster 5-yr revenue", proj["revenue_5yr"],
                    unit="$", source="forecast.project_site", confidence=0.6),
            self.ev("hist.membership", "cluster membership share", proj["mem_share"],
                    source="forecast.project_site", confidence=0.6),
            self.ev("hist.cluster_asp", "cluster ASP (membership / retail)",
                    {"asp_mem": proj["asp_mem"], "asp_ret": proj["asp_ret"]},
                    kind="table", unit="$/wash", source="forecast.project_site", confidence=0.6),
            self.ev("hist.ramp_pattern", "ramp pattern",
                    f"ramps to 90% in {proj['ramp_to_90pct_months']} mo over {proj['n_donors']} donors",
                    kind="text", source="forecast.project_site", confidence=0.6),
        ]
        try:
            out.append(A.compute_anchor(ws.snap).as_evidence())     # eid "signal.decider"
        except Exception:
            pass
        return out

    def initial_belief(self, ws) -> BeliefState:
        wash_ev = ws.evidence.get("hist.cluster_wash")
        mature_anchor = wash_ev.value if wash_ev is not None else None

        # defer to the signal decider when it has an opinion (mirrors Anchor.abstains); else level-vs-floor
        sig_ev = ws.evidence.get("signal.decider")
        lean, confidence = None, 0.6
        if sig_ev is not None and isinstance(sig_ev.value, dict):
            sig_lean = sig_ev.value.get("lean")
            n_matured = int(sig_ev.value.get("n_matured") or 0)
            if sig_lean is not None and n_matured >= 2:
                lean, confidence = sig_lean, float(sig_ev.confidence)
        if lean is None:
            lean = self.lean_from_level(mature_anchor, D.mature_floor())

        return BeliefState(expert=self.name, lean=lean, confidence=confidence, key_number=mature_anchor,
                           key_number_label="mature washes/mo",
                           supporting=[e.eid for e in ws.evidence_of(self.name)])
