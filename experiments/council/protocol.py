"""
The committee's data types — pure dataclasses, no I/O.

`MsgType` is the collaboration language (PUBLISH / QUESTION / CHALLENGE / REQUEST / REVISE / ENDORSE / VOTE).
`Evidence` is one item on the shared board (namespaced id, value, a `leakage_safe` flag → 🔒/🌐 badge).
`Message` is one typed utterance an expert proposes (the LLM proposes; the coordinator routes deterministically).
`BeliefState` is an expert's evolving position — its lean/confidence/key-number plus a running `memory`
(what it argued/conceded) fed back into its next react, and a per-round `history` for the belief-evolution chart.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class MsgType(str, Enum):
    PUBLISH = "PUBLISH"       # assert a finding (initial or updated); MUST cite ≥1 evidence
    QUESTION = "QUESTION"     # ask an expert to clarify (non-adversarial); names `to`
    CHALLENGE = "CHALLENGE"   # dispute a claim; MUST cite the conflicting evidence; names `to`
    REQUEST = "REQUEST"       # ask an expert to run a NEW investigation (dynamic task); names `to`
    REVISE = "REVISE"         # update MY OWN belief in response to the board
    ENDORSE = "ENDORSE"       # agree with a claim (raises consensus); cites the endorsed evidence
    VOTE = "VOTE"             # final lean + confidence


def _mid() -> str:
    return uuid.uuid4().hex[:8]


@dataclass
class Evidence:
    eid: str                          # stable, namespaced: "historical.mature_anchor", "comp.saturation"
    expert: str
    label: str                        # human label, e.g. "local mature-neighbour anchor"
    value: Any = None                 # float | dict | list | str
    kind: str = "number"              # "number" | "series" | "text" | "table" | "flag"
    unit: Optional[str] = None        # "washes/mo" | "$" | "mi" | "ft"
    source: str = ""                  # producing tool / dataset
    confidence: float = 0.5
    leakage_safe: bool = True         # False = LLM world-knowledge / live present-day snapshot → 🌐 badge
    round_added: int = 0

    def badge(self) -> str:
        return "🔒" if self.leakage_safe else "🌐"

    def to_dict(self) -> Dict[str, Any]:
        return {"eid": self.eid, "expert": self.expert, "label": self.label, "value": self.value,
                "kind": self.kind, "unit": self.unit, "source": self.source,
                "confidence": round(float(self.confidence), 3), "leakage_safe": self.leakage_safe,
                "round_added": self.round_added}


@dataclass
class Message:
    mtype: MsgType
    sender: str                       # expert name or "facilitator"
    text: str                         # one clause of natural language
    round: int = 0
    cites: List[str] = field(default_factory=list)     # Evidence.eid; required for PUBLISH/CHALLENGE/ENDORSE
    to: Optional[str] = None          # addressed expert; None = broadcast
    payload: Dict[str, Any] = field(default_factory=dict)  # {lean, confidence, key_number, ask, condition}
    mid: str = field(default_factory=_mid)
    answered_by: Optional[str] = None  # mid of the message that resolved this one

    def to_dict(self) -> Dict[str, Any]:
        return {"mid": self.mid, "type": self.mtype.value, "sender": self.sender, "to": self.to,
                "text": self.text, "round": self.round, "cites": list(self.cites),
                "payload": self.payload, "answered_by": self.answered_by}


@dataclass
class BeliefState:
    expert: str
    lean: Optional[str] = None                   # "Build" | "Pass" | "Conditional" | None
    confidence: float = 0.3
    key_number: Optional[float] = None           # headline datum (mature washes/mo, CAPEX$, net_5yr, …)
    key_number_label: str = ""
    supporting: List[str] = field(default_factory=list)     # Evidence.eid backing the lean
    open_concerns: List[str] = field(default_factory=list)
    memory: str = ""                             # the expert's running notes — persists across rounds
    history: List[Dict[str, Any]] = field(default_factory=list)  # [{round, lean, confidence, key_number}]

    def snapshot(self, rnd: int) -> None:
        self.history.append({"round": rnd, "lean": self.lean, "confidence": round(float(self.confidence), 3),
                             "key_number": self.key_number})

    def to_dict(self) -> Dict[str, Any]:
        return {"expert": self.expert, "lean": self.lean, "confidence": round(float(self.confidence), 3),
                "key_number": self.key_number, "key_number_label": self.key_number_label,
                "supporting": list(self.supporting), "open_concerns": list(self.open_concerns),
                "memory": self.memory, "history": list(self.history)}
