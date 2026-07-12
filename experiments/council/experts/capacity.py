"""
Capacity / Tunnel Analyst — sizes the physical tunnel from projected peak-month throughput. No LLM; this
seat informs build sizing/CAPEX, it does not weigh in on go/no-go (lean is always None).
"""
from __future__ import annotations

from typing import List

from experiments.council import config as C
from experiments.council import forecast
from experiments.council.experts.base import Expert
from experiments.council.protocol import BeliefState, Evidence


class CapacityExpert(Expert):
    name = "capacity"
    role = "Capacity / Tunnel Analyst"
    persona = ("You size the physical tunnel a candidate site needs from its projected peak-month "
               "throughput — peak-hour washes plus a fixed operating buffer.")
    is_world = False

    def investigate(self, ws) -> List[Evidence]:
        proj = forecast.project_site(ws.lat, ws.lon)
        peak = proj["peak_month_washes"]
        # confirmed formula: peak-hour throughput + a fixed buffer = tunnel length in feet
        tunnel_ft = round(peak / (C.OPERATING_DAYS * C.OPERATING_HOURS) + C.TUNNEL_BUFFER_FT, 1)
        return [
            self.ev("cap.peak_month_washes", "projected peak-month washes", peak,
                    unit="washes/mo", source="forecast.project_site", confidence=0.7),
            self.ev("cap.tunnel_ft", "sized tunnel length", tunnel_ft, unit="ft",
                    source="peak / (OPERATING_DAYS*OPERATING_HOURS) + TUNNEL_BUFFER_FT", confidence=0.7),
        ]

    def initial_belief(self, ws) -> BeliefState:
        ev = ws.evidence.get("cap.tunnel_ft")
        return BeliefState(expert=self.name, lean=None, confidence=0.7,
                           key_number=(ev.value if ev is not None else None),
                           key_number_label="tunnel ft",
                           supporting=[e.eid for e in ws.evidence_of(self.name)])
