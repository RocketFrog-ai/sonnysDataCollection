"""
The `Expert` base class — the contract every committee seat implements.

Investigation is **pure Python** (tools → `Evidence`, no LLM). Reaction is the **single LLM call** via
`llm_react.react`, which updates this expert's `BeliefState` (kept in `ws.beliefs`) and returns the typed
`Message`s it proposes (the coordinator routes them). Finance overrides `consolidate()` to run last from the
others' settled numbers. Subclasses set `name / role / persona / is_world` and override `investigate()` +
`initial_belief()`.
"""
from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from experiments.council import config as C
from experiments.council import llm_react
from experiments.council.protocol import BeliefState, Evidence, Message


class Expert:
    name: str = "expert"
    role: str = "Analyst"
    persona: str = ""
    is_world: bool = False          # True → world-knowledge weighted (down-weighted in the tally)

    def __init__(self) -> None:
        self.followups_used = 0

    # ── pure-Python investigation (override) ──
    def investigate(self, ws) -> List[Evidence]:
        """Run this expert's tools on ws.snap / ws.lat,lon and return the Evidence to post. No LLM."""
        return []

    def investigate_followup(self, ws, request: Message) -> List[Evidence]:
        """Optional deeper dig when another expert REQUESTs it (dynamic task). No LLM by default."""
        return []

    def consolidate(self, ws) -> List[Evidence]:
        """Finance only: run last, once the others' numbers are on the board."""
        return []

    # ── initial belief from own evidence (override to set lean/key_number) ──
    def initial_belief(self, ws) -> BeliefState:
        return BeliefState(expert=self.name, lean=None, confidence=0.3)

    # ── the one LLM react (shared) ──
    def react(self, ws, delta, log) -> List[Message]:
        belief = ws.beliefs.get(self.name) or self.initial_belief(ws)
        peers = {k: v for k, v in ws.beliefs.items() if k != self.name}   # so a seat can debate PEERS' positions
        msgs, belief = llm_react.react(
            self.name, self.role, self.persona,
            my_evidence=ws.evidence_of(self.name),
            board=ws.all_evidence(),
            recent=log.recent(12),
            peers=peers,
            belief=belief, rnd=ws.round, max_msgs=C.MAX_MSGS_PER_EXPERT)
        ws.set_belief(belief)
        return msgs

    # ── helpers subclasses reuse ──
    def ev(self, eid: str, label: str, value: Any, *, kind: str = "number", unit: Optional[str] = None,
           source: str = "", confidence: float = 0.6, leakage_safe: bool = True) -> Evidence:
        return Evidence(eid=eid, expert=self.name, label=label, value=value, kind=kind, unit=unit,
                        source=source, confidence=confidence, leakage_safe=leakage_safe)

    @staticmethod
    def _finite(x: Any) -> Optional[float]:
        try:
            v = float(x)
            return v if np.isfinite(v) else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def lean_from_level(projected: Optional[float], floor: float) -> Optional[str]:
        """Turn a projected mature level into a Build/Conditional/Pass lean vs the healthy-site floor."""
        p = Expert._finite(projected)
        if p is None or floor <= 0:
            return None
        ratio = p / floor
        return "Build" if ratio >= 1.0 else ("Pass" if ratio < 0.5 else "Conditional")
