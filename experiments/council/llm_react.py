"""
The committee's single LLM touchpoint — every expert "reacts" here, **Azure only** (council `llm.py`).

`react(...)` builds a system persona + the collaboration rules + a compact JSON context (the expert's own
evidence, its running belief/memory, the board, the recent discussion) and asks Azure for STRICT JSON:
  {"belief": {"lean","confidence","key_number","open_concerns","memory"},
   "messages": [{"type","to","text","cites","payload"}]}
It parses that into typed `Message`s + an updated `BeliefState`. On ANY LLM/parse failure it returns
`([], belief-unchanged)` — a safe no-op, so a bad completion never crashes a round. The coordinator routes
the returned messages deterministically (drops uncited assertions, applies effects).
"""
from __future__ import annotations

import json
from typing import List, Optional, Tuple

from experiments.council import config as C
from experiments.council import llm
from experiments.council.protocol import BeliefState, Evidence, Message, MsgType

_VALID_TYPES = {t.value for t in MsgType}

REACT_SYS = """You are the {role} on a FIVE-SEAT car-wash site-selection COMMITTEE that argues its way to a
decision. You are ONE independent seat; the other seats are peers with their own evidence. Reason ONLY from the
EVIDENCE on the shared board (ids like 'historical.mature_anchor') and the discussion so far; never invent
numbers — every factual claim cites a board evidence id that exists.

{phase}

General rules:
- Data-grounded evidence (🔒) outranks world-knowledge (🌐): if your world-knowledge conflicts with a 🔒
  number, challenge it with a cited reason or concede toward Conditional — never hand-wave.
- REVISE your lean or key_number whenever cited evidence genuinely moves you, and say WHAT changed in 'memory'
  (e.g. "cut mature washes 12.6k→11.2k after Competition's rival-density point").
- PUBLISH / CHALLENGE / ENDORSE MUST cite ≥1 board id — the BARE id like 'hist.cluster_wash', WITHOUT the
  🔒/🌐 badge. QUESTION / CHALLENGE / REQUEST set 'to' to a seat name.
- Emit ≤{max_msgs} messages, address seats by name, prefer a concrete CHALLENGE (with your number) over a
  vague question, and never re-ask a question already in the discussion.

Return STRICT JSON only, no prose, no code fences:
{{"belief": {{"lean": "Build|Pass|Conditional|null", "confidence": 0.0-1.0, "key_number": <number|null>,
             "open_concerns": ["..."], "memory": "1-2 sentences: what you argued, conceded, and still hold"}},
 "messages": [{{"type": "PUBLISH|QUESTION|CHALLENGE|REQUEST|REVISE|ENDORSE|VOTE",
               "to": "<seat name|null>", "text": "one pointed clause",
               "cites": ["<evidence id>", ...],
               "payload": {{"ask": "...", "condition": "...", "key_number": <number|null>}}}}]}}
Your remit: {persona}"""

_PHASE_OPEN = ("THIS IS ROUND 0 — OPENING STATEMENTS. Emit a PUBLISH stating your headline finding (cite your "
               "own evidence) and set your lean + confidence + key_number in 'belief'. Then read "
               "'committee_positions': if ANY peer's lean OPPOSES yours (e.g. they lean Build while you lean "
               "Pass), you MUST CHALLENGE that seat by name to defend its number against your evidence. That "
               "is how the debate starts — do not just state your own view and stop.")


def _phase_deliberate(rnd: int) -> str:
    return (f"THIS IS ROUND {rnd} — DELIBERATION. FIRST resolve EVERY item in 'you_must_respond_to': answer a "
            "QUESTION with a PUBLISH that states the answer, REVISE if a CHALLENGE convinced you, or DEFEND with "
            "a cited PUBLISH — never leave an item aimed at you unanswered, and never merely VOTE while one is "
            "open. THEN look at 'committee_positions' and CHALLENGE any peer whose lean or number still conflicts "
            "with your 🔒 evidence. Only if you are settled AND nothing is aimed at you AND no peer conflicts with "
            "you: emit a single VOTE or stay silent.")


def _fmt_val(v) -> str:
    if isinstance(v, float):
        return f"{v:,.1f}"
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False)[:200]
    return str(v)[:200]


def _evidence_line(e: Evidence) -> str:
    # id FIRST so the model cites the bare id (route()'s valid_cites is also badge-tolerant as a backstop)
    return f"{e.eid} {e.badge()} [{e.expert}] {e.label}: {_fmt_val(e.value)}{(' ' + e.unit) if e.unit else ''}"


def react(expert: str, role: str, persona: str, *, my_evidence: List[Evidence], board: List[Evidence],
          recent: List[Message], peers: Optional[dict] = None, belief: BeliefState, rnd: int,
          max_msgs: Optional[int] = None) -> Tuple[List[Message], BeliefState]:
    """One Azure react. Returns (typed messages proposed by this expert, updated belief). Safe no-op on failure."""
    max_msgs = max_msgs or C.MAX_MSGS_PER_EXPERT
    try:
        # challenges/questions/requests aimed at ME and not yet answered — I must respond to these this turn
        directed = [m for m in recent if m.to == expert and m.answered_by is None
                    and m.mtype in (MsgType.CHALLENGE, MsgType.QUESTION, MsgType.REQUEST)]
        ctx = {
            "my_current_belief": {"lean": belief.lean, "confidence": round(belief.confidence, 2),
                                  "key_number": belief.key_number, "memory": belief.memory,
                                  "open_concerns": belief.open_concerns},
            "committee_positions": [
                (f"{name} leans {b.lean or 'undecided'} (confidence {round(b.confidence, 2)}"
                 + (f", {b.key_number_label or 'key#'}={b.key_number:,.0f}" if isinstance(b.key_number, (int, float)) else "")
                 + ")") for name, b in (peers or {}).items()],
            "you_must_respond_to": [f"{m.mtype.value} from {m.sender}: {m.text}"
                                    + (f" (cites {', '.join(m.cites)})" if m.cites else "") for m in directed],
            "my_evidence": [_evidence_line(e) for e in my_evidence],
            "board": [_evidence_line(e) for e in board],
            "recent_discussion": [f"{m.mtype.value} {m.sender}" + (f"→{m.to}" if m.to else "")
                                  + f": {m.text}" for m in recent],
        }
        phase = _PHASE_OPEN if rnd == 0 else _phase_deliberate(rnd)
        sys = REACT_SYS.format(role=role, persona=persona, max_msgs=max_msgs, phase=phase)
        text = llm.complete([{"role": "system", "content": sys},
                             {"role": "user", "content": json.dumps(ctx, ensure_ascii=False)[:8000]}],
                            json_mode=True, temperature=C.LLM_TEMPERATURE, max_tokens=C.REACT_MAX_TOKENS)
        j = llm.parse_json_lax(text)
    except Exception:
        return [], belief
    if not isinstance(j, dict):
        return [], belief

    # ── update this expert's belief (memory persists across rounds) ──
    bj = j.get("belief") or {}
    lean = str(bj.get("lean") or "").strip().title()
    if lean in {"Build", "Pass", "Conditional"}:
        belief.lean = lean
    if isinstance(bj.get("confidence"), (int, float)):
        belief.confidence = float(min(max(float(bj["confidence"]), 0.0), 1.0))
    if isinstance(bj.get("key_number"), (int, float)):
        belief.key_number = float(bj["key_number"])
    if isinstance(bj.get("open_concerns"), list):
        belief.open_concerns = [str(x)[:160] for x in bj["open_concerns"]][:5]
    if bj.get("memory"):
        belief.memory = str(bj["memory"])[:600]

    # ── parse proposed messages (coordinator validates cites + applies effects) ──
    out: List[Message] = []
    for mj in (j.get("messages") or [])[:max_msgs]:
        if not isinstance(mj, dict):
            continue
        t = str(mj.get("type") or "").upper()
        if t not in _VALID_TYPES:
            continue
        to = mj.get("to")
        to = str(to) if (to and str(to).strip().lower() not in ("null", "none", "")) else None
        out.append(Message(mtype=MsgType(t), sender=expert, text=str(mj.get("text") or "")[:400], round=rnd,
                           cites=[str(c) for c in (mj.get("cites") or []) if c][:6], to=to,
                           payload=mj.get("payload") if isinstance(mj.get("payload"), dict) else {}))
    return out, belief
