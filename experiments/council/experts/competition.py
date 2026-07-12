"""
Competition Analyst — live nearby-rival counts (Google Places, present-day) plus one LLM saturation read.

`investigate` pulls real car washes within 3 and 5 driving miles — a present-day snapshot, so **not**
leakage-safe even when read for a backtest pin — and posts the counts and the closest rivals. It then
makes ONE optional Azure call to grade competitive saturation from those counts (or, when Google/Azure is
unavailable, from the pin's location alone via world knowledge); the call is wrapped so a down backend
degrades to `comp.saturation = "unknown"` rather than dropping the seat's evidence.
"""
from __future__ import annotations

import json
from typing import List, Tuple

from experiments.council import config as C
from experiments.council import llm
from experiments.council import places
from experiments.council.experts.base import Expert
from experiments.council.protocol import BeliefState, Evidence

_SATURATION_SYS = (
    "You are a Competition Analyst on a car-wash site-selection committee. Grade the local competitive "
    "saturation for a candidate car-wash site from nearby-rival counts (or, if none are given, your own "
    "knowledge of car-wash density near that latitude/longitude).\n"
    'Return STRICT JSON only, no prose, no code fences: {"saturation": "low|medium|high", '
    '"headroom": "<=200 chars, one clause on remaining headroom or risk"}.'
)


class CompetitionExpert(Expert):
    name = "competition"
    role = "Competition Analyst"
    persona = ("You size up the live competitive field around a candidate site — how many rivals sit "
               "within 3 and 5 driving miles, who they are, and whether the market still has headroom.")
    is_world = False

    def investigate(self, ws) -> List[Evidence]:
        r3, r5 = C.GOOGLE_PLACES_RADII_MI
        near3 = places.nearby_competitors(ws.lat, ws.lon, r3)
        near5 = places.nearby_competitors(ws.lat, ws.lon, r5)
        rivals = near5["competitors"] or near3["competitors"]

        out = [
            self.ev("comp.count_3mi", f"car washes within {r3:g} driving mi (present-day)", near3["count"],
                    unit="count", source="places.nearby_competitors", confidence=0.8, leakage_safe=False),
            self.ev("comp.count_5mi", f"car washes within {r5:g} driving mi (present-day)", near5["count"],
                    unit="count", source="places.nearby_competitors", confidence=0.8, leakage_safe=False),
            self.ev("comp.rivals", "closest rivals",
                    [{"name": c["name"], "distance_miles": c["distance_miles"]} for c in rivals[:8]],
                    kind="table", source="places.nearby_competitors", confidence=0.7, leakage_safe=False),
        ]
        saturation, headroom = self._llm_saturation(ws, near3["count"], near5["count"], rivals)
        out.append(self.ev("comp.saturation", "competitive saturation", saturation, kind="text",
                           source="llm" if saturation != "unknown" else "no-op",
                           confidence=0.55 if saturation != "unknown" else 0.2, leakage_safe=False))
        out.append(self.ev("comp.headroom", "headroom note", headroom, kind="text",
                           source="llm" if headroom else "no-op",
                           confidence=0.5 if headroom else 0.2, leakage_safe=False))
        return out

    def _llm_saturation(self, ws, n3: int, n5: int, rivals: list) -> Tuple[str, str]:
        """ONE wrapped Azure call. Falls back to ("unknown", "") on any failure (no key, timeout, bad JSON) —
        e.g. no Google key means n3=n5=0/rivals=[], and the prompt tells the model to fall back to its own
        world knowledge of the lat/lon rather than treat an empty count as a literally-zero-rival market."""
        try:
            user = {"lat": ws.lat, "lon": ws.lon, "count_3mi": n3, "count_5mi": n5,
                   "top_rivals": [f"{c.get('name')} ({c.get('distance_miles')} mi)" for c in rivals[:5]]}
            text = llm.complete(
                [{"role": "system", "content": _SATURATION_SYS},
                 {"role": "user", "content": json.dumps(user, ensure_ascii=False)}],
                json_mode=True, temperature=C.LLM_TEMPERATURE, max_tokens=C.RESEARCH_MAX_TOKENS)
            j = llm.parse_json_lax(text)
            sat = str(j.get("saturation") or "").strip().lower()
            if sat not in ("low", "medium", "high"):
                sat = "unknown"
            return sat, str(j.get("headroom") or "")[:200]
        except Exception:
            return "unknown", ""

    def initial_belief(self, ws) -> BeliefState:
        sat_ev = ws.evidence.get("comp.saturation")
        n3_ev = ws.evidence.get("comp.count_3mi")
        saturation = sat_ev.value if sat_ev is not None else "unknown"
        lean = {"low": "Build", "high": "Pass"}.get(saturation, "Conditional")
        confidence = 0.65 if saturation in ("low", "high") else 0.4
        return BeliefState(expert=self.name, lean=lean, confidence=confidence,
                           key_number=(n3_ev.value if n3_ev is not None else None),
                           key_number_label="rivals within 3mi",
                           supporting=[e.eid for e in ws.evidence_of(self.name)])
