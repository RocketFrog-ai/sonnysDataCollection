"""
Local-Market Analyst — demographic/income context from the council's sitewise CSV, plus one LLM
seasonality read. `is_world=True`: this seat leans on world-knowledge narration, so the committee tally
down-weights it (`config.WORLD_EXPERTS`) — the old council's bullish-drift lesson.
"""
from __future__ import annotations

import json
from typing import List

from experiments.council import config as C
from experiments.council import datasets
from experiments.council import llm
from experiments.council.experts.base import Expert
from experiments.council.protocol import BeliefState, Evidence

_SEASONALITY_SYS = (
    "You are a Local-Market Analyst on a car-wash site-selection committee. Given a site's location and "
    "demographic snapshot, describe the local market's demographic character and its car-wash "
    "seasonality (weather, snow/salt season, tourism, pollen, etc).\n"
    'Return STRICT JSON only, no prose, no code fences: {"seasonality": "<=400 chars narrative"}.'
)


class LocalMarketExpert(Expert):
    name = "local_market"
    role = "Local-Market Analyst"
    persona = ("You describe the neighbourhood around a candidate site — who lives there, what they earn, "
               "and how weather/seasonality shapes car-wash demand through the year. Context, not a vote.")
    is_world = True

    def investigate(self, ws) -> List[Evidence]:
        prof = datasets.sitewise_for_pin(ws.lat, ws.lon)
        out = [
            self.ev("mkt.demographics", "population / growth / age",
                    {"population_2025": prof.get("population_2025"),
                     "growth_2020_2025": prof.get("growth_2020_2025"),
                     "avg_age": prof.get("avg_age")},
                    kind="table", source="datasets.sitewise_for_pin", confidence=0.7),
            self.ev("mkt.income", "household income",
                    {"median_household_income": prof.get("median_household_income"),
                     "avg_household_income": prof.get("avg_household_income"),
                     "pct_hh_income_50k_plus": prof.get("pct_hh_income_50k_plus")},
                    kind="table", unit="$", source="datasets.sitewise_for_pin", confidence=0.7),
        ]
        out.append(self._llm_seasonality(ws, prof))
        return out

    def _llm_seasonality(self, ws, prof: dict) -> Evidence:
        """ONE wrapped Azure research call. Falls back to a plain 'unavailable' text on any failure."""
        try:
            user = {"lat": ws.lat, "lon": ws.lon, "population_2025": prof.get("population_2025"),
                   "median_household_income": prof.get("median_household_income"),
                   "avg_age": prof.get("avg_age")}
            text = llm.complete(
                [{"role": "system", "content": _SEASONALITY_SYS},
                 {"role": "user", "content": json.dumps(user, ensure_ascii=False)}],
                json_mode=True, temperature=C.LLM_TEMPERATURE, max_tokens=C.RESEARCH_MAX_TOKENS)
            j = llm.parse_json_lax(text)
            narrative = str(j.get("seasonality") or "").strip()[:400]
        except Exception:
            narrative = ""
        return self.ev("mkt.seasonality", "demographic & seasonality narrative", narrative or "unavailable",
                       kind="text", source="llm" if narrative else "no-op",
                       confidence=0.5 if narrative else 0.2, leakage_safe=False)

    def initial_belief(self, ws) -> BeliefState:
        dem_ev = ws.evidence.get("mkt.demographics")
        population = (dem_ev.value or {}).get("population_2025") if dem_ev is not None else None
        # context seat, not a vote: a weak Conditional carries through the tally at WORLD_EXPERT_WEIGHT
        return BeliefState(expert=self.name, lean="Conditional", confidence=0.3, key_number=population,
                           key_number_label="population (2025 est.)",
                           supporting=[e.eid for e in ws.evidence_of(self.name)])
