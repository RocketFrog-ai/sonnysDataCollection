"""
The shared committee blackboard (`Workspace`) + the running `DiscussionLog`.

`Workspace` holds every posted `Evidence`, each expert's `BeliefState`, the open REQUEST / CHALLENGE
queues, the round counter, and an `llm_calls` budget meter. The
coordinator posts evidence, reads the per-round delta, and snapshots beliefs each round. `DiscussionLog`
is the ordered list of typed `Message`s with helpers for the transcript, convergence test, and CSV export.

No LLM, no I/O. Imports only `protocol` (so it never pulls the coordinator/anchor → no import cycle).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from experiments.council.protocol import BeliefState, Evidence, Message, MsgType


@dataclass
class WorkspaceDelta:
    """What changed since a given round — the react() context."""
    evidence: List[Evidence] = field(default_factory=list)
    messages: List[Message] = field(default_factory=list)


@dataclass
class DiscussionLog:
    messages: List[Message] = field(default_factory=list)

    def add(self, m: Message) -> None:
        self.messages.append(m)

    def by_round(self) -> Dict[int, List[Message]]:
        out: Dict[int, List[Message]] = {}
        for m in self.messages:
            out.setdefault(m.round, []).append(m)
        return out

    def in_round(self, rnd: int) -> List[Message]:
        return [m for m in self.messages if m.round == rnd]

    def recent(self, n: int = 12) -> List[Message]:
        return self.messages[-n:]

    def unanswered(self, mtype: MsgType) -> List[Message]:
        return [m for m in self.messages if m.mtype == mtype and m.answered_by is None]

    def resolve(self, target_mid: str, by_mid: str) -> None:
        for m in self.messages:
            if m.mid == target_mid and m.answered_by is None:
                m.answered_by = by_mid

    def to_records(self) -> List[Dict[str, Any]]:
        return [m.to_dict() for m in self.messages]


@dataclass
class Workspace:
    lat: float
    lon: float
    radius_km: float
    snap: Any = None                                  # the leakage-controlled Snapshot (from snapshot.py)
    focal_key: str = "__live__::0"
    evidence: Dict[str, Evidence] = field(default_factory=dict)
    beliefs: Dict[str, BeliefState] = field(default_factory=dict)
    open_requests: List[Message] = field(default_factory=list)
    open_challenges: List[Message] = field(default_factory=list)
    plan: List[Dict[str, str]] = field(default_factory=list)   # facilitator seed questions per expert
    agenda: Optional[str] = None                      # this round's agenda topic (set by the coordinator)
    round: int = 0
    llm_calls: int = 0
    light: bool = False                               # light mode → experts skip their investigate-phase LLM/web calls

    # ── evidence ──
    def post(self, ev: Optional[Evidence]) -> None:
        if ev is not None and ev.eid:
            ev.round_added = self.round
            self.evidence[ev.eid] = ev

    def has(self, eid: str) -> bool:
        return eid in self.evidence

    def evidence_of(self, expert: str) -> List[Evidence]:
        return [e for e in self.evidence.values() if e.expert == expert]

    def all_evidence(self) -> List[Evidence]:
        return list(self.evidence.values())

    def valid_cites(self, cites: List[str]) -> List[str]:
        """Resolve cited ids to real evidence eids, tolerant of the 🔒/🌐 badge or quotes the LLM may prepend
        (it reads board lines like '🔒 hist.cluster_wash …', so it sometimes cites the bare id and sometimes
        the badged form). A known eid contained anywhere in the cited token resolves to that eid."""
        out: List[str] = []
        for c in (cites or []):
            c = str(c).strip()
            if c in self.evidence:
                if c not in out:
                    out.append(c)
                continue
            hit = next((eid for eid in self.evidence if eid in c), None)
            if hit and hit not in out:
                out.append(hit)
        return out

    # ── beliefs ──
    def set_belief(self, b: BeliefState) -> None:
        self.beliefs[b.expert] = b

    def snapshot_beliefs(self) -> None:
        for b in self.beliefs.values():
            b.snapshot(self.round)

    # ── delta for react() ──
    def delta_since(self, rnd: int, log: DiscussionLog) -> WorkspaceDelta:
        return WorkspaceDelta(
            evidence=[e for e in self.evidence.values() if e.round_added > rnd],
            messages=[m for m in log.messages if m.round > rnd],
        )
