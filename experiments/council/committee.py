"""
`run_committee(snap) -> CommitteeResult` — the one entry point the Streamlit view and the harness share.

It seats the experts, runs the Facilitator phases (plan → investigate → publish → discussion → resolution
→ Finance consolidate), makes the deterministic `decide_final` call, and synthesizes the report. In
`light=True` mode it skips every LLM (no publish/discussion reacts, no report LLM) — the regression guard that
proves the discussion can't silently move the verdict off the data.

`CommitteeResult.chamber_data()` flattens the run into the plain dict `chamber.build_chamber_html` renders.
Self-contained: council imports only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from experiments.council import config as C
from experiments.council.anchor import Decision, decide_final
from experiments.council.coordinator import Facilitator, deliberate
from experiments.council.experts import build_experts
from experiments.council.protocol import MsgType
from experiments.council.snapshot import build_live_snapshot
from experiments.council.workspace import DiscussionLog, Workspace

_YES = {"Build", "Conditional"}


def run_committee(snap, *, mode: Optional[str] = None, backend: Optional[str] = None, light: bool = False,
                  max_rounds: Optional[int] = None, radius_km: Optional[float] = None) -> "CommitteeResult":
    """Convene the committee on a leakage-controlled `snap`. `backend` is ignored (Azure-only)."""
    radius_km = radius_km if radius_km is not None else C.RADIUS_KM
    ws = Workspace(lat=float(snap.lat), lon=float(snap.lon), radius_km=float(radius_km),
                   snap=snap, focal_key=getattr(snap, "focal_key", "__live__::0"), light=light)
    log = DiscussionLog()
    experts = build_experts()
    fac = Facilitator(ws, log, experts, light=light, max_rounds=max_rounds)

    deliberate(fac, snap)                                # LangGraph: investigate → publish → discuss ⟲ → finance
    plan = ws.plan

    decision = decide_final(ws, mode=mode)
    report = ""
    if not light:
        try:
            from experiments.council.report import synthesize_report
            report = synthesize_report(ws, log, decision)
        except Exception as exc:                         # report must never sink the run
            report = f"_(report unavailable: {exc})_"

    return CommitteeResult(decision=decision, workspace=ws, log=log, report=report, plan=plan,
                           lat=ws.lat, lon=ws.lon, radius_km=ws.radius_km)


def run_committee_pin(lat: float, lon: float, *, radius_km: Optional[float] = None, **kw) -> "CommitteeResult":
    """Convenience for a live drop-pin (builds a present-day snapshot, then runs the committee)."""
    r = radius_km if radius_km is not None else C.RADIUS_KM
    snap = build_live_snapshot(lat, lon, radius_km=r)
    return run_committee(snap, radius_km=r, **kw)


def run_committee_json(lat: float, lon: float, *, radius_km: Optional[float] = None, mode: Optional[str] = None,
                       light: bool = False, max_rounds: Optional[int] = None) -> Dict[str, Any]:
    """Run the committee and return a fully JSON-serialisable payload for the UI. The Streamlit view runs
    THIS in a subprocess: the committee's pandas/numpy work segfaults inside Streamlit's ScriptRunner
    thread on some (bleeding-edge numpy/pyarrow) stacks, but is rock-solid in a fresh process."""
    res = run_committee_pin(lat, lon, radius_km=radius_km, mode=mode, light=light, max_rounds=max_rounds)
    ws, dec = res.workspace, res.decision
    evidence = [{"expert": e.expert, "eid": e.eid, "label": e.label, "value": e.value,
                 "unit": e.unit, "badge": e.badge()} for e in ws.all_evidence()]
    chamber = res.chamber_data()
    round_digest: Dict[str, Any] = {"rounds": [], "takeaway": ""}
    if not light:
        try:                                             # the Facilitator's per-round recap (never sinks the run)
            from experiments.council.debate_digest import summarize_debate
            round_digest = summarize_debate(chamber.get("messages", []), chamber.get("belief_history", {}),
                                            dec.verdict, dec.condition)
        except Exception:
            pass
    return {"chamber": chamber, "report": res.report, "evidence": evidence, "plan": res.plan,
            "round_digest": round_digest, "verdict": dec.verdict, "confidence": dec.confidence,
            "basis": dec.basis, "condition": dec.condition, "lat": res.lat, "lon": res.lon}


@dataclass
class CommitteeResult:
    decision: Decision
    workspace: Workspace
    log: DiscussionLog
    report: str
    plan: List[Dict[str, str]] = field(default_factory=list)
    lat: float = 0.0
    lon: float = 0.0
    radius_km: float = 0.0

    # ── flatten to the plain dict the chamber renders ──
    def chamber_data(self) -> Dict[str, Any]:
        ws, log, dec = self.workspace, self.log, self.decision
        meta = C.EXPERT_META

        def _seat_meta(key: str) -> Dict[str, str]:
            m = meta.get(key, {"name": key.title(), "emoji": "•", "color": "#64748b"})
            return {"name": m["name"], "emoji": m["emoji"], "color": m["color"]}

        experts = []
        leans_yes = 0
        n_leaned = 0
        for key in C.EXPERT_ORDER:
            b = ws.beliefs.get(key)
            sm = _seat_meta(key)
            lean = b.lean if b else None
            if lean:
                n_leaned += 1
                leans_yes += 1 if lean in _YES else 0
            experts.append({
                "key": key, "name": sm["name"], "emoji": sm["emoji"], "color": sm["color"],
                "lean": lean, "lean_color": C.VERDICT_COLORS.get(lean, "#94a3b8"),
                "confidence": round(float(b.confidence), 2) if b else 0.0,
                "key_number": (b.key_number if b else None),
                "key_number_label": (b.key_number_label if b else ""),
                "weight": (C.WORLD_EXPERT_WEIGHT if key in C.WORLD_EXPERTS else C.DATA_EXPERT_WEIGHT),
                "is_world": key in C.WORLD_EXPERTS,
            })

        messages = []
        for m in log.messages:
            sm = _seat_meta(m.sender) if m.sender in meta else {"name": m.sender.title(), "emoji": "🧭", "color": "#0f172a"}
            tm = _seat_meta(m.to) if (m.to and m.to in meta) else None
            messages.append({
                "mid": m.mid, "type": m.mtype.value, "type_color": C.MSG_COLORS.get(m.mtype.value, "#64748b"),
                "round": m.round, "sender": m.sender, "sender_name": sm["name"], "sender_emoji": sm["emoji"],
                "sender_color": sm["color"], "to": m.to, "to_name": (tm["name"] if tm else None),
                "text": m.text, "cites": list(m.cites),
            })

        belief_history = {k: (b.history if b else []) for k, b in ws.beliefs.items()}
        rounds = max([m.round for m in log.messages], default=0)
        consensus_pct = round(leans_yes / n_leaned, 2) if (n_leaned and dec.verdict in _YES) else (
            round((n_leaned - leans_yes) / n_leaned, 2) if n_leaned else 0.0)

        return {
            "verdict": dec.verdict, "verdict_label": C.VERDICT_LABELS.get(dec.verdict, dec.verdict),
            "verdict_color": C.VERDICT_COLORS.get(dec.verdict, "#64748b"),
            "confidence": round(float(dec.confidence), 2), "basis": dec.basis,
            "condition": dec.condition, "site": {"lat": self.lat, "lon": self.lon},
            "rounds": rounds, "consensus_pct": consensus_pct,
            "open_challenges": len(log.unanswered(MsgType.CHALLENGE)),
            "experts": experts, "facilitator": _seat_meta("facilitator"),
            "numbers": dec.numbers, "messages": messages, "belief_history": belief_history,
        }


# ─────────────────────────── subprocess entrypoint (the Streamlit view calls this) ───────────────────────────
def _coerce(o):
    try:
        return float(o)
    except (TypeError, ValueError):
        return str(o)


if __name__ == "__main__":
    import json
    import sys
    _lat, _lon, _radius = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
    _light = len(sys.argv) > 4 and sys.argv[4] == "1"
    _payload = run_committee_json(_lat, _lon, radius_km=_radius, light=_light)
    sys.stdout.write("__COUNCIL_JSON__" + json.dumps(_payload, default=_coerce, ensure_ascii=False))
    sys.stdout.flush()
