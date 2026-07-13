"""
The Facilitator + the committee's orchestration graph.

The committee is driven as a **LangGraph `StateGraph`** (this repo's idiom — see
`app/pnl_analysis/insights/graph.py`): nodes `investigate → publish → discuss ⟲ → finance`, where the
`discuss` self-loop is gated by a conditional edge running the deterministic `converged()` predicate — a
genuine agentic cycle. When langgraph isn't importable it falls back to the SAME phases in a plain loop, so
behaviour is identical either way (`decide_final` + the report run in `committee.run_committee` after this).

Design invariant kept from the start: the LLM only *proposes* typed messages inside `Expert.react`; `route()`
here decides their effect (drops uncited assertions, queues requests/challenges, resolves answered ones).
No LLM lives in this file. Self-contained: council imports only.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from experiments.council import config as C
from experiments.council.anchor import compute_anchor
from experiments.council.protocol import Message, MsgType
from experiments.council.workspace import DiscussionLog, Workspace

_MOVING = {MsgType.CHALLENGE, MsgType.REQUEST, MsgType.REVISE, MsgType.QUESTION}


def _rel_change(a, b) -> float:
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return float("inf")
    return abs(float(a) - float(b)) / max(abs(float(b)), 1e-9)


class Facilitator:
    """Holds the shared board + seats and exposes the committee phases as methods. The graph (or the
    fallback loop) sequences these; the phase logic is identical for both drivers."""

    def __init__(self, ws: Workspace, log: DiscussionLog, experts: List, *,
                 light: bool = False, max_rounds: Optional[int] = None) -> None:
        self.ws = ws
        self.log = log
        self.experts = experts
        self.light = light
        self.max_rounds = max_rounds if max_rounds is not None else C.MAX_ROUNDS
        self.names = {e.name for e in experts}

    def _by_name(self, name: Optional[str]):
        return next((e for e in self.experts if e.name == name), None)

    # ── phase: plan (deterministic seed questions) ──
    def plan(self) -> List[Dict[str, str]]:
        seeds = {
            "historical": "What do comparable sites in the 12-mile cluster mature to, and how did new sites ramp?",
            "competition": "How many real car-wash rivals sit within 3–5 miles, and is the market saturated?",
            "local_market": "What do local demographics, income and seasonality say about demand?",
            "capacity": "What tunnel length does the projected peak-month volume call for?",
            "finance": "What are the 5-yr revenue, opex, CAPEX, net and breakeven for this build?",
        }
        self.ws.plan = [{"expert": e.name, "question": seeds.get(e.name, "Investigate your domain.")}
                        for e in self.experts]
        return self.ws.plan

    # ── phase: investigate (pure tools, threaded) ──
    def investigate(self) -> None:
        self.ws.round = 0

        def _one(e):
            if e.name == "finance":
                return e, []
            try:
                return e, (e.investigate(self.ws) or [])
            except Exception:
                return e, []

        with ThreadPoolExecutor(max_workers=min(8, max(1, len(self.experts)))) as ex:
            gathered = list(ex.map(_one, self.experts))
        for _e, evs in gathered:                     # post sequentially (no concurrent dict writes)
            for ev in evs:
                self.ws.post(ev)

    def set_anchor(self, snap) -> None:
        self.ws.anchor = compute_anchor(snap)        # the leakage-clean signal exhibit
        self.ws.post(self.ws.anchor.as_evidence())

    def set_initial_beliefs(self) -> None:
        for e in self.experts:
            if e.name == "finance":
                continue
            try:
                self.ws.set_belief(e.initial_belief(self.ws))
            except Exception:
                pass

    # ── phase: publish round ──
    def publish_round(self) -> None:
        if self.light:
            return
        self.ws.round = 0
        delta = self.ws.delta_since(-1, self.log)
        for e in self.experts:
            if e.name == "finance":
                continue
            if self.ws.llm_calls >= C.PER_SITE_LLM_BUDGET:
                break
            self._react_and_route(e, delta)
        self.ws.snapshot_beliefs()

    # ── phase: ONE discussion round (the graph self-loops on this) ──
    def discussion_round(self, r: int) -> None:
        if self.light:
            return
        self.ws.round = r
        delta = self.ws.delta_since(r - 1, self.log)
        for e in self.experts:
            if e.name == "finance":
                continue
            if self.ws.llm_calls >= C.PER_SITE_LLM_BUDGET:
                break
            self._react_and_route(e, delta)
        self._service_requests()
        self.ws.snapshot_beliefs()

    def _react_and_route(self, e, delta) -> None:
        try:
            msgs = e.react(self.ws, delta, self.log)
        except Exception:
            msgs = []
        self.ws.llm_calls += 1
        for m in msgs:
            self.route(m)

    def _service_requests(self) -> None:
        for req in list(self.ws.open_requests):
            tgt = self._by_name(req.to)
            if tgt is not None and tgt.followups_used < C.MAX_FOLLOWUPS_PER_EXPERT:
                try:
                    for ev in (tgt.investigate_followup(self.ws, req) or []):
                        self.ws.post(ev)
                except Exception:
                    pass
                tgt.followups_used += 1
            self.log.resolve(req.mid, req.mid)       # mark serviced either way
        self.ws.open_requests.clear()

    # ── phase: Finance consolidates last ──
    def finance_consolidate(self) -> None:
        self.close_moot_challenges()                 # discussion is over: retire challenges between seats that now agree
        fin = self._by_name("finance")
        if fin is None:
            return
        try:
            for ev in (fin.consolidate(self.ws) or []):
                self.ws.post(ev)
            self.ws.set_belief(fin.initial_belief(self.ws))
        except Exception:
            pass
        self.ws.snapshot_beliefs()

    # ── deterministic routing (LLM proposes; Python decides) ──
    def route(self, m: Message) -> None:
        if m.mtype in (MsgType.PUBLISH, MsgType.CHALLENGE, MsgType.ENDORSE):
            m.cites = self.ws.valid_cites(m.cites)
            if not m.cites:
                return                               # drop uncited assertion (anti-hallucination)
        if m.mtype in (MsgType.QUESTION, MsgType.CHALLENGE, MsgType.REQUEST):
            if m.to not in self.names:
                m.to = None                          # broadcast if the addressee is invalid
        if m.mtype in (MsgType.CHALLENGE, MsgType.QUESTION) and m.to and self._is_relitigation(m):
            return                                   # drop a repeat (open OR already answered) — a won argument retires
        self.log.add(m)
        if m.mtype == MsgType.REQUEST:
            self.ws.open_requests.append(m)
        elif m.mtype == MsgType.CHALLENGE:
            self.ws.open_challenges.append(m)
        elif m.mtype == MsgType.ENDORSE:
            for eid in m.cites:                      # a cited endorsement firms the evidence a touch
                ev = self.ws.evidence.get(eid)
                if ev is not None:
                    ev.confidence = min(1.0, ev.confidence + 0.05)
        # REVISE / VOTE: the sender's belief was already updated inside llm_react.
        self._resolve_open(m)

    def _is_relitigation(self, m: Message) -> bool:
        """Same sender → same target, same type, arguing from (mostly) the same evidence = the same point
        again — whether or not the earlier one was answered. Once a point is made (and possibly conceded),
        it doesn't get raised a third time."""
        new = set(m.cites or [])
        for om in self.log.messages:
            if om.sender != m.sender or om.to != m.to or om.mtype != m.mtype:
                continue
            old = set(om.cites or [])
            if not new and not old:
                return True                          # two uncited questions to the same target — one is enough
            inter, union = len(new & old), len(new | old)
            if union and inter / union >= 0.5:
                return True
        return False

    def _resolve_open(self, m: Message) -> None:
        """A PUBLISH/REVISE by X answers any open CHALLENGE/QUESTION addressed to X."""
        if m.mtype not in (MsgType.PUBLISH, MsgType.REVISE):
            return
        for om in self.log.messages:
            if om.answered_by is None and om.to == m.sender and om.mtype in (MsgType.CHALLENGE, MsgType.QUESTION):
                om.answered_by = m.mid

    def close_moot_challenges(self) -> None:
        """Closing sweep (deterministic, before the verdict): a still-open challenge whose sender and target
        now hold the SAME lean is moot — the two seats converged, so there is no standing disagreement left.
        Challenges between seats that still disagree stay open and are reported honestly."""
        for om in self.log.unanswered(MsgType.CHALLENGE):
            s = self.ws.beliefs.get(om.sender)
            t = self.ws.beliefs.get(om.to) if om.to else None
            if s is not None and t is not None and s.lean is not None and s.lean == t.lean:
                om.answered_by = f"moot:{om.mid}"    # resolved by agreement, not by a reply

    # ── the exact convergence predicate (the conditional-edge router uses this) ──
    def converged(self, r: int) -> bool:
        if r >= self.max_rounds:
            return True                              # (H) hard cap — always terminates
        if self.ws.llm_calls >= C.PER_SITE_LLM_BUDGET:
            return True                              # budget exhausted → stop
        if self.log.unanswered(MsgType.CHALLENGE):
            return False                             # (a) no open challenges
        if self.ws.open_requests:
            return False                             # (b) no pending dynamic investigations
        if self.log.unanswered(MsgType.QUESTION):
            return False                             # (c) no open questions
        for b in self.ws.beliefs.values():           # (d) confidence AND key number stable over 2 rounds
            if len(b.history) < 2:
                return False
            if abs(b.history[-1]["confidence"] - b.history[-2]["confidence"]) > C.EPS_CONF:
                return False
            if _rel_change(b.history[-1]["key_number"], b.history[-2]["key_number"]) > C.EPS_NUM:
                return False
        if [m for m in self.log.in_round(r) if m.mtype in _MOVING]:   # (e) nothing moving this round
            return False
        return True


# ─────────────────────────── LangGraph orchestration (this repo's idiom) ───────────────────────────
def build_committee_graph(fac: "Facilitator", snap):
    """Compile the committee as a LangGraph StateGraph: investigate → publish → discuss ⟲ → finance.
    The `discuss` self-loop is gated by `fac.converged()` (a real agentic cycle). Returns None if langgraph
    is not importable, so the caller falls back to the identical hand-rolled loop."""
    try:
        from langgraph.graph import END, START, StateGraph
    except Exception:
        return None

    def _investigate(state):
        fac.plan(); fac.investigate(); fac.set_anchor(snap); fac.set_initial_beliefs()
        return {"round": 0}

    def _publish(state):
        fac.publish_round()
        return {"round": 0}

    def _discuss(state):
        r = int(state.get("round", 0)) + 1
        fac.discussion_round(r)
        return {"round": r}

    def _finance(state):
        fac.finance_consolidate()
        return {}

    def _after_discuss(state) -> str:                # the convergence conditional edge
        return "finance" if (fac.light or fac.converged(int(state.get("round", 0)))) else "discuss"

    g = StateGraph(dict)
    for name, fn in (("investigate", _investigate), ("publish", _publish),
                     ("discuss", _discuss), ("finance", _finance)):
        g.add_node(name, fn)
    g.add_edge(START, "investigate")
    g.add_edge("investigate", "publish")
    g.add_edge("publish", "discuss")
    g.add_conditional_edges("discuss", _after_discuss, {"discuss": "discuss", "finance": "finance"})
    g.add_edge("finance", END)
    return g.compile()


def deliberate(fac: "Facilitator", snap) -> str:
    """Run the deliberation phases via LangGraph if available, else the identical hand-rolled loop.
    Mutates fac.ws / fac.log in place. Returns the driver used ("langgraph" | "fallback")."""
    graph = build_committee_graph(fac, snap)
    if graph is not None:
        try:
            graph.invoke({"round": 0}, config={"recursion_limit": max(10, fac.max_rounds * 4 + 8)})
            return "langgraph"
        except Exception:
            pass                                     # graph failed → fall through to the same phases
    # hand-rolled fallback (same order, same predicate)
    fac.plan(); fac.investigate(); fac.set_anchor(snap); fac.set_initial_beliefs()
    fac.publish_round()
    for r in range(1, fac.max_rounds + 1):
        fac.discussion_round(r)
        if fac.light or fac.converged(r):
            break
    fac.finance_consolidate()
    return "fallback"
