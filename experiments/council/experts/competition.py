"""
Competition Analyst — TRUE express-tunnel competition within 3 driving miles.

Google Places returns every "car wash" nearby — but most are detailers, hand washes, mobile detailers and
self-serve bays that do NOT compete with an express tunnel. So `investigate` pulls the 3-mile rivals, then
one Azure call CLASSIFIES them: it keeps only conveyor/express/tunnel/exterior-automatic washes, counts the
true express competitors, and grades saturation on THAT count (not the raw total). With no Google key it
falls back to world-knowledge of express density near the lat/lon. Present-day snapshot → not leakage-safe.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from experiments.council import config as C
from experiments.council import llm
from experiments.council import places
from experiments.council.experts.base import Expert
from experiments.council.protocol import BeliefState, Evidence

_COMP_SYS = (
    "You are a Competition Analyst for a NEW EXPRESS-TUNNEL car wash. You are given nearby car-wash "
    "businesses from Google within 3 driving miles. CLASSIFY each as a TRUE express-tunnel competitor or "
    "NOT: detailing shops, hand washes, mobile detailers, self-serve bays and full-detail-only shops are "
    "NOT direct competitors for an express tunnel — only conveyor / express / tunnel / exterior-automatic "
    "washes are. Count the TRUE express competitors and grade saturation on THAT count, not the raw total. "
    "If the rival list is EMPTY, estimate express-tunnel competition near the given lat/lon from your own "
    "world knowledge and say so.\n"
    'Return STRICT JSON only, no prose, no code fences: {"express_competitors": ["name", ...], '
    '"express_count": <int>, "excluded_non_competitors": ["name", ...], "saturation": "low|medium|high", '
    '"headroom": "<=240 chars on the express-tunnel headroom or risk"}.'
)


class CompetitionExpert(Expert):
    name = "competition"
    role = "Competition Analyst"
    persona = ("You size up the TRUE express-tunnel competition within 3 driving miles — filtering out "
               "detailers, hand washes and mobile detailers, which don't compete with a tunnel — and judge "
               "whether the market has real headroom for another express wash.")
    is_world = False

    def investigate(self, ws) -> List[Evidence]:
        r3 = C.GOOGLE_PLACES_RADII_MI[0]                      # 3 miles — the express trade area
        near3 = places.nearby_competitors(ws.lat, ws.lon, r3)
        rivals = near3["competitors"]
        cls = self._classify(ws, r3, rivals)

        express = cls.get("express_competitors") or []
        express_count = cls.get("express_count")
        if express_count is None:
            express_count = len(express)
        saturation = cls.get("saturation") or "unknown"
        graded = saturation != "unknown"

        return [
            self.ev("comp.washes_3mi_all", f"all car-wash listings within {r3:g} driving mi (Google, all types)",
                    near3["count"], unit="count", source="places.nearby_competitors",
                    confidence=0.8, leakage_safe=False),
            self.ev("comp.express_3mi", f"TRUE express-tunnel competitors within {r3:g} mi (detailers/hand/mobile excluded)",
                    express_count, unit="count", source="places+llm classify",
                    confidence=0.7 if graded else 0.3, leakage_safe=False),
            self.ev("comp.rivals", "the express-tunnel rivals that actually count", express[:8],
                    kind="table", source="places+llm classify", confidence=0.65, leakage_safe=False),
            self.ev("comp.saturation", "express-tunnel saturation (graded on the true count)", saturation,
                    kind="text", source="llm" if graded else "no-op",
                    confidence=0.6 if graded else 0.2, leakage_safe=False),
            self.ev("comp.headroom", "headroom note", str(cls.get("headroom") or "")[:240] or "unavailable",
                    kind="text", source="llm" if graded else "no-op",
                    confidence=0.55 if graded else 0.2, leakage_safe=False),
        ]

    def _classify(self, ws, r3: float, rivals: list) -> Dict[str, Any]:
        """ONE wrapped Azure call: filter the Google list to true express-tunnel competitors + grade
        saturation on that count. Falls back to {} (→ 'unknown') on any failure (no key / timeout / bad JSON)."""
        try:
            payload = {"lat": ws.lat, "lon": ws.lon, "radius_miles": r3,
                       "rivals_3mi": [{"name": c.get("name"), "dist_mi": c.get("distance_miles"),
                                       "type": c.get("primary_type")} for c in rivals]}
            text = llm.complete(
                [{"role": "system", "content": _COMP_SYS},
                 {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                json_mode=True, temperature=C.LLM_TEMPERATURE, max_tokens=C.RESEARCH_MAX_TOKENS)
            j = llm.parse_json_lax(text)
            if not isinstance(j, dict):
                return {}
            sat = str(j.get("saturation") or "").strip().lower()
            if sat not in ("low", "medium", "high"):
                j["saturation"] = "unknown"
            return j
        except Exception:
            return {}

    def initial_belief(self, ws) -> BeliefState:
        sat_ev = ws.evidence.get("comp.saturation")
        exp = ws.evidence.get("comp.express_3mi")
        sat = sat_ev.value if sat_ev is not None else "unknown"
        # low express-saturation → room to Build; high → Pass; else Conditional
        lean = {"low": "Build", "high": "Pass"}.get(sat, "Conditional")
        confidence = 0.65 if sat in ("low", "high") else 0.4
        return BeliefState(expert=self.name, lean=lean, confidence=confidence,
                           key_number=(exp.value if exp is not None else None),
                           key_number_label="express rivals within 3mi",
                           supporting=[e.eid for e in ws.evidence_of(self.name)])
