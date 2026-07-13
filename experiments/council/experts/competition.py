"""
Competition Analyst — TRUE express-tunnel competition within 3 driving miles, scored on an industry benchmark.

Google Places returns every "car wash" nearby — but most are detailers, hand washes, mobile detailers and
self-serve bays that do NOT compete with an express tunnel. So `investigate` pulls the 3-mile rivals and makes
ONE Azure call to CLASSIFY them (keep only conveyor/express/tunnel/exterior-automatic washes). The
saturation is then **deterministic**, not an LLM guess: a competition SCORE (0-100) from the true express
count on the standard feasibility-study benchmark (0 rivals → 100 "ideal" … 7+ → 10 "oversaturated"), plus a
distance-weighted "effective competitors" refinement (a rival at 0.5 mi bites harder than one at 2.8 mi).
With no Google key it falls back to a world-knowledge estimate. Present-day snapshot → not leakage-safe.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from experiments.council import config as C
from experiments.council import llm
from experiments.council import places
from experiments.council.experts.base import Expert
from experiments.council.protocol import BeliefState, Evidence

# industry benchmark: competition score by # of TRUE express-tunnel competitors within 3 mi
# (0 ideal · 1-2 good · 3-4 acceptable w/ strong demo+traffic · 5+ needs differentiation · 7+ oversaturated)
_SCORE_TABLE = {0: 100, 1: 90, 2: 80, 3: 70, 4: 60, 5: 45, 6: 30}


def _competition_score(express_count: Optional[float]) -> int:
    n = int(round(express_count or 0))
    return 10 if n >= 7 else _SCORE_TABLE.get(n, 10)


def _score_to_saturation(score: int) -> str:
    return "low" if score >= 70 else ("medium" if score >= 45 else "high")   # 0-3 → low, 4-5 → medium, 6+ → high


def _dist_weight(dist_mi: Optional[float]) -> float:
    """A closer express rival competes harder: ~1.0 within 1 mi, ~0.6 at 2 mi, ~0.3 at 3 mi."""
    if dist_mi is None:
        return 0.6
    return float(max(0.25, min(1.0, 1.3 - float(dist_mi) / 3.0)))


_COMP_SYS = (
    "You are a Competition Analyst for a NEW EXPRESS-TUNNEL car wash. You are given nearby car-wash "
    "businesses from Google within 3 driving miles. CLASSIFY each as a TRUE express-tunnel competitor or "
    "NOT: detailing shops, hand washes, mobile detailers, self-serve bays and full-detail-only shops are "
    "NOT direct competitors for an express tunnel — only conveyor / express / tunnel / exterior-automatic "
    "washes are. Count the TRUE express competitors. (Feasibility-study benchmark for a 3-mi radius: 0 "
    "ideal, 1-2 good, 3-4 acceptable with strong demographics/traffic, 5+ needs differentiation, 7+ "
    "oversaturated.) If the rival list is EMPTY, estimate express-tunnel competition near the given lat/lon "
    "from your own world knowledge and say so.\n"
    'Return STRICT JSON only, no prose, no code fences: {"express_competitors": ["name", ...], '
    '"express_count": <int>, "excluded_non_competitors": ["name", ...], '
    '"headroom": "<=240 chars on the express-tunnel headroom or risk vs the benchmark"}.'
)


class CompetitionExpert(Expert):
    name = "competition"
    role = "Competition Analyst"
    persona = ("You size up the TRUE express-tunnel competition within 3 driving miles — filtering out "
               "detailers, hand washes and mobile detailers — and score saturation on the standard "
               "feasibility benchmark (0 rivals ideal … 7+ oversaturated), distance-weighted.")
    is_world = False

    def investigate(self, ws) -> List[Evidence]:
        r3 = C.GOOGLE_PLACES_RADII_MI[0]                      # 3 miles — the express trade area
        near3 = places.nearby_competitors(ws.lat, ws.lon, r3)
        rivals = near3["competitors"]
        cls = self._classify(ws, r3, rivals)

        express_names = cls.get("express_competitors") or []
        express_count = cls.get("express_count")
        if express_count is None:
            express_count = len(express_names)

        # match the LLM's express names back to the Google rivals to recover distances → distance-weighted index
        express_rivals = [r for r in rivals if any(str(n).lower() in str(r.get("name", "")).lower()
                                                   or str(r.get("name", "")).lower() in str(n).lower()
                                                   for n in express_names)] if express_names else []
        effective = round(sum(_dist_weight(r.get("distance_miles")) for r in express_rivals), 1) if express_rivals else None

        score = _competition_score(express_count)
        saturation = _score_to_saturation(score)             # DETERMINISTIC from the benchmark, not the LLM
        graded = bool(cls) and express_count is not None

        return [
            self.ev("comp.washes_3mi_all", f"all car-wash listings within {r3:g} driving mi (Google, all types)",
                    near3["count"], unit="count", source="places.nearby_competitors",
                    confidence=0.8, leakage_safe=False),
            self.ev("comp.express_3mi", f"TRUE express-tunnel competitors within {r3:g} mi (detailers/hand/mobile excluded)",
                    express_count, unit="count", source="places+llm classify",
                    confidence=0.75 if graded else 0.3, leakage_safe=False),
            self.ev("comp.effective_3mi", "distance-weighted effective express competitors (closer = heavier)",
                    effective, unit="index", source="places+distance weight", confidence=0.6, leakage_safe=False),
            self.ev("comp.score", "competition score 0-100 (industry benchmark: 0 rivals=100 ideal … 7+=10 oversaturated)",
                    score, unit="/100", source="feasibility benchmark table", confidence=0.75, leakage_safe=False),
            self.ev("comp.saturation", "express-tunnel saturation (from the benchmark score)", saturation,
                    kind="text", source="benchmark score", confidence=0.7, leakage_safe=False),
            self.ev("comp.rivals", "the express-tunnel rivals that actually count", express_names[:8],
                    kind="table", source="places+llm classify", confidence=0.65, leakage_safe=False),
            self.ev("comp.headroom", "headroom note vs the benchmark", str(cls.get("headroom") or "")[:240] or "unavailable",
                    kind="text", source="llm" if graded else "no-op",
                    confidence=0.55 if graded else 0.2, leakage_safe=False),
        ]

    def _classify(self, ws, r3: float, rivals: list) -> Dict[str, Any]:
        """ONE wrapped Azure call: filter the Google list to true express-tunnel competitors. Falls back to {}
        on any failure (no key / timeout / bad JSON). Light mode skips it (the score is still deterministic)."""
        if getattr(ws, "light", False):
            return {}
        try:
            payload = {"lat": ws.lat, "lon": ws.lon, "radius_miles": r3,
                       "rivals_3mi": [{"name": c.get("name"), "dist_mi": c.get("distance_miles"),
                                       "type": c.get("primary_type")} for c in rivals]}
            text = llm.complete(
                [{"role": "system", "content": _COMP_SYS},
                 {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                json_mode=True, temperature=C.LLM_TEMPERATURE, max_tokens=C.RESEARCH_MAX_TOKENS)
            j = llm.parse_json_lax(text)
            return j if isinstance(j, dict) else {}
        except Exception:
            return {}

    def initial_belief(self, ws) -> BeliefState:
        sat_ev = ws.evidence.get("comp.saturation")
        score_ev = ws.evidence.get("comp.express_3mi")
        sat = sat_ev.value if sat_ev is not None else "unknown"
        # low saturation → room to Build; high → Pass; medium → Conditional
        lean = {"low": "Build", "high": "Pass"}.get(sat, "Conditional")
        confidence = 0.68 if sat in ("low", "high") else 0.45
        return BeliefState(expert=self.name, lean=lean, confidence=confidence,
                           key_number=(score_ev.value if score_ev is not None else None),
                           key_number_label="express rivals within 3mi",
                           supporting=[e.eid for e in ws.evidence_of(self.name)])
