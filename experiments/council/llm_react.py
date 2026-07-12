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

REACT_SYS = """You are the {role} on a car-wash site-selection committee.
Reason ONLY from the EVIDENCE on the shared board (each item has an id like 'historical.mature_anchor').
Never invent numbers; every factual claim cites an evidence id that already exists on the board.
Data-grounded evidence (🔒) outranks world-knowledge (🌐): if your world-knowledge conflicts with a 🔒
number, defer or CHALLENGE it with a cited reason.

Return STRICT JSON only, no prose, no code fences:
{{"belief": {{"lean": "Build|Pass|Conditional|null", "confidence": 0.0-1.0, "key_number": <number|null>,
             "open_concerns": ["..."], "memory": "one or two sentences: what you argued/conceded and still hold"}},
 "messages": [{{"type": "PUBLISH|QUESTION|CHALLENGE|REQUEST|REVISE|ENDORSE|VOTE",
               "to": "<expert name|null>", "text": "one clause",
               "cites": ["<evidence id>", ...],
               "payload": {{"ask": "...", "condition": "...", "key_number": <number|null>}}}}]}}
Rules: PUBLISH/CHALLENGE/ENDORSE MUST cite ≥1 board evidence id. QUESTION/CHALLENGE/REQUEST set 'to'.
Emit ≤{max_msgs} messages. Be specific and cite.
Your remit: {persona}"""


def _fmt_val(v) -> str:
    if isinstance(v, float):
        return f"{v:,.1f}"
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False)[:200]
    return str(v)[:200]


def _evidence_line(e: Evidence) -> str:
    return f"{e.badge()} {e.eid} [{e.expert}] {e.label}: {_fmt_val(e.value)}{(' ' + e.unit) if e.unit else ''}"


def react(expert: str, role: str, persona: str, *, my_evidence: List[Evidence], board: List[Evidence],
          recent: List[Message], belief: BeliefState, rnd: int,
          max_msgs: Optional[int] = None) -> Tuple[List[Message], BeliefState]:
    """One Azure react. Returns (typed messages proposed by this expert, updated belief). Safe no-op on failure."""
    max_msgs = max_msgs or C.MAX_MSGS_PER_EXPERT
    try:
        ctx = {
            "my_current_belief": {"lean": belief.lean, "confidence": round(belief.confidence, 2),
                                  "key_number": belief.key_number, "memory": belief.memory,
                                  "open_concerns": belief.open_concerns},
            "my_evidence": [_evidence_line(e) for e in my_evidence],
            "board": [_evidence_line(e) for e in board],
            "recent_discussion": [f"{m.mtype.value} {m.sender}" + (f"→{m.to}" if m.to else "")
                                  + f": {m.text}" for m in recent],
        }
        sys = REACT_SYS.format(role=role, persona=persona, max_msgs=max_msgs)
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
