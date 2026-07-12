"""
Local-Market Analyst — grounded sitewise demographics + a RICH, multi-section market read.

`investigate` pulls the council's sitewise columns (population, income, vehicles, retail anchors) AND makes
one Azure call for a full local-market analysis — demand drivers, traffic/access & visibility, weather
seasonality, the demographic read, and the qualitative competitive landscape — so the seat has substantive,
specific material to argue with (not a one-line blurb). `is_world=True`: it leans on world-knowledge
narration, so the committee tally down-weights it (`config.WORLD_EXPERTS`) — the old bullish-drift lesson.
"""
from __future__ import annotations

import json
from typing import Dict, List

from experiments.council import config as C
from experiments.council import datasets
from experiments.council import llm
from experiments.council.experts.base import Expert
from experiments.council.protocol import BeliefState, Evidence

_LM_SYS = (
    "You are a Local-Market Analyst evaluating a site for a NEW EXPRESS-TUNNEL car wash. Using the "
    "coordinates and the demographic snapshot, write a RICH, specific market read that another analyst on "
    "the committee could argue with — quote the numbers you are given and reason about what they imply for "
    "express-wash demand, pricing, membership potential and throughput.\n"
    'Return STRICT JSON only, no prose, no code fences: {"demand_drivers": "...", "traffic_access": "...", '
    '"seasonality": "...", "demographic_read": "...", "competitive_context": "..."}. '
    "Each value = 1-3 substantive sentences with specifics."
)


class LocalMarketExpert(Expert):
    name = "local_market"
    role = "Local-Market Analyst"
    persona = ("You describe the neighbourhood around a candidate site — who lives there, what they earn, "
               "their vehicles, nearby retail anchors, road access, and how weather shapes car-wash demand "
               "through the year. You provide rich context; the data seats carry the go/no-go weight.")
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
            self.ev("mkt.vehicles", "vehicles in the market (demand base)",
                    {"avg_vehicles_per_hh": prof.get("avg_vehicles"),
                     "total_vehicles": prof.get("total_vehicles")},
                    kind="table", source="datasets.sitewise_for_pin", confidence=0.7),
            self.ev("mkt.retail_anchors", "nearby retail anchors (traffic generators)",
                    {"mass_merchants": prof.get("mass_merchant_count"), "grocery": prof.get("grocery_count")},
                    kind="table", source="datasets.sitewise_for_pin", confidence=0.6),
        ]
        out.extend(self._llm_analysis(ws, prof))
        return out

    def _llm_analysis(self, ws, prof: dict) -> List[Evidence]:
        """ONE wrapped Azure call → the rich, structured market read posted as several substantive evidence
        items. Falls back to a single 'unavailable' item on any failure (no key / timeout / bad JSON)."""
        try:
            user = {"lat": ws.lat, "lon": ws.lon,
                    "population": prof.get("population_2025"),
                    "median_income": prof.get("median_household_income"),
                    "avg_household_income": prof.get("avg_household_income"),
                    "avg_age": prof.get("avg_age"),
                    "avg_vehicles": prof.get("avg_vehicles"),
                    "pct_hh_income_50k_plus": prof.get("pct_hh_income_50k_plus"),
                    "mass_merchants": prof.get("mass_merchant_count")}
            text = llm.complete(
                [{"role": "system", "content": _LM_SYS},
                 {"role": "user", "content": json.dumps(user, ensure_ascii=False)}],
                json_mode=True, temperature=C.LLM_TEMPERATURE, max_tokens=1100)
            j = llm.parse_json_lax(text)
        except Exception:
            j = {}
        fields = [("mkt.demand_drivers", "demand drivers"), ("mkt.traffic", "traffic / access / visibility"),
                  ("mkt.seasonality", "weather seasonality"), ("mkt.competitive_context", "competitive landscape (world-knowledge)")]
        keys = {"mkt.demand_drivers": "demand_drivers", "mkt.traffic": "traffic_access",
                "mkt.seasonality": "seasonality", "mkt.competitive_context": "competitive_context"}
        out: List[Evidence] = []
        got = False
        for eid, label in fields:
            val = str((j or {}).get(keys[eid]) or "").strip()[:600]
            if val:
                got = True
            out.append(self.ev(eid, label, val or "unavailable", kind="text",
                               source="llm" if val else "no-op", confidence=0.5 if val else 0.2,
                               leakage_safe=False))
        # fold the demographic read into an extra item when present
        drd = str((j or {}).get("demographic_read") or "").strip()[:600]
        if drd:
            out.append(self.ev("mkt.demographic_read", "demographic read (world-knowledge)", drd, kind="text",
                               source="llm", confidence=0.5, leakage_safe=False))
        return out if got else out[:1]

    def initial_belief(self, ws) -> BeliefState:
        dem_ev = ws.evidence.get("mkt.demographics")
        population = (dem_ev.value or {}).get("population_2025") if dem_ev is not None else None
        # context seat, not a vote: a weak Conditional carries through the tally at WORLD_EXPERT_WEIGHT
        return BeliefState(expert=self.name, lean="Conditional", confidence=0.3, key_number=population,
                           key_number_label="population (2025 est.)",
                           supporting=[e.eid for e in ws.evidence_of(self.name)])
