"""
The Facilitator's plain-language recap — the "moderator/summarizer" role both the blackboard-architecture
and information-asymmetry papers recommend a committee should have.

`summarize_debate(messages, belief_history, verdict, condition)` makes ONE Azure pass that turns the raw
typed messages into a per-ROUND recap: a short title, one sentence of what happened (plain English, no
evidence-ids, no ellipsis), and a one-sentence decision-relevant insight — plus a final takeaway. On any
LLM/parse failure it falls back to a deterministic, clean per-round summary, so the digest is never empty
and never blocks the run. Azure-only; self-contained (council imports only).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from experiments.council import config as C
from experiments.council import llm

_SYS = (
    "You are the neutral FACILITATOR who chaired a five-expert car-wash site-selection committee "
    "(Historical, Competition, Local-Market, Capacity, Finance). Write a crisp recap for a busy executive "
    "who will not read the raw transcript. For EACH round you are given, return:\n"
    "  • title  — 3 to 6 words naming what the round was about\n"
    "  • recap  — ONE sentence: who argued or conceded what, in plain English\n"
    "  • insight — ONE sentence: what it means for the build/no-build decision\n"
    "Then a 'synthesis' object that decomposes the debate at the CLAIM level (not a vote tally):\n"
    "  • consensus    — the strongest point (nearly) all seats ended up agreeing on\n"
    "  • disagreement — the sharpest tension that was NOT fully resolved (or 'The room aligned.' if none)\n"
    "  • unique       — one important point only ONE seat raised that the others never engaged with "
    "(empty string if there was none) — the easily-missed insight\n"
    "Rules: plain English only — NEVER write an evidence id like 'hist.projected_mature' or a raw code token; "
    "name the number in words ('the ~12.6k mature-wash projection'). Use ONLY facts present in the transcript; "
    "never invent a number. No ellipsis ('...'), no markdown, no code fences. Keep every sentence complete.\n"
    'Return STRICT JSON: {"rounds":[{"round":<int>,"title":"...","recap":"...","insight":"..."}],'
    '"synthesis":{"consensus":"...","disagreement":"...","unique":"..."}}'
)


def _moves_by_round(belief_history: Dict[str, List[dict]]) -> Dict[int, List[str]]:
    """Lean changes per round (factual, from the belief history) — 'Historical moved Build → Conditional'."""
    out: Dict[int, List[str]] = {}
    for ekey, seq in (belief_history or {}).items():
        name = C.EXPERT_META.get(ekey, {}).get("name", ekey.title())
        for prev, cur in zip(seq, seq[1:]):
            if cur.get("lean") != prev.get("lean"):
                out.setdefault(int(cur.get("round", 0)), []).append(
                    f"{name} moved {prev.get('lean') or '—'} → {cur.get('lean') or '—'}")
    return out


def _rounds_payload(messages: List[dict], moves: Dict[int, List[str]]) -> List[dict]:
    """Group the flattened chamber messages by round, attaching the round's lean changes."""
    by_round: Dict[int, List[dict]] = {}
    for m in messages:
        r = int(m.get("round", 0))
        line = {"from": m.get("sender_name"), "type": m.get("type"), "text": m.get("text")}
        if m.get("to_name"):
            line["to"] = m["to_name"]
        by_round.setdefault(r, []).append(line)
    payload = []
    for r in sorted(by_round):
        payload.append({"round": r, "messages": by_round[r], "position_changes": moves.get(r, [])})
    return payload


def _final_leans(belief_history: Dict[str, List[dict]]) -> Dict[str, Optional[str]]:
    return {k: (seq[-1].get("lean") if seq else None) for k, seq in (belief_history or {}).items()}


def _fallback(payload: List[dict], moves: Dict[int, List[str]],
              belief_history: Dict[str, List[dict]]) -> Dict[str, Any]:
    """Deterministic, clean per-round summary + claim synthesis (no LLM, no ellipsis) — used if Azure is down."""
    _V = {"PUBLISH": "opened", "CHALLENGE": "challenged", "QUESTION": "questioned", "REVISE": "revised",
          "ENDORSE": "endorsed", "VOTE": "voted", "REQUEST": "asked for more"}
    rounds = []
    for p in payload:
        r = p["round"]
        n_ch = sum(1 for m in p["messages"] if m["type"] == "CHALLENGE")
        n_rv = sum(1 for m in p["messages"] if m["type"] == "REVISE")
        title = "Opening positions" if r == 0 else (f"Round {r} deliberation" if n_ch or n_rv else f"Round {r}")
        acts = []
        for m in p["messages"][:4]:
            tgt = f" {m['to']}" if m.get("to") else ""
            acts.append(f"{m['from']} {_V.get(m['type'], m['type'].lower())}{tgt}")
        recap = "; ".join(acts) + "." if acts else "No messages this round."
        insight = (", ".join(p["position_changes"]) + ".") if p["position_changes"] else \
                  (f"{n_ch} challenge(s) raised." if n_ch else "Positions held.")
        rounds.append({"round": r, "title": title, "recap": recap, "insight": insight})
    leans = [l for l in _final_leans(belief_history).values() if l]
    if leans:
        from collections import Counter
        top, cnt = Counter(leans).most_common(1)[0]
        consensus = f"{cnt} of {len(leans)} seats ended on {top}."
        disagreement = (f"Seats split across {', '.join(sorted(set(leans)))}." if len(set(leans)) > 1
                        else "The room aligned.")
    else:
        consensus, disagreement = "No seat produced a lean.", "The room aligned."
    return {"rounds": rounds, "synthesis": {"consensus": consensus, "disagreement": disagreement, "unique": ""}}


def summarize_debate(messages: List[dict], belief_history: Dict[str, List[dict]], verdict: str,
                     condition: Optional[str]) -> Dict[str, Any]:
    """One Azure pass → {rounds:[{round,title,recap,insight}], takeaway}. Clean deterministic fallback."""
    moves = _moves_by_round(belief_history)
    payload = _rounds_payload(messages or [], moves)
    if not payload:
        return {"rounds": [], "synthesis": {}}
    try:
        user = {"verdict": verdict, "condition": condition, "rounds": payload}
        raw = llm.complete([{"role": "system", "content": _SYS},
                            {"role": "user", "content": json.dumps(user, ensure_ascii=False)[:11000]}],
                           json_mode=True, temperature=0.3, max_tokens=1000)
        j = llm.parse_json_lax(raw)
        rounds = j.get("rounds") if isinstance(j, dict) else None
        if not isinstance(rounds, list) or not rounds:
            return _fallback(payload, moves, belief_history)
        clean = []
        for r in rounds:
            if not isinstance(r, dict):
                continue
            clean.append({"round": int(r.get("round", 0)),
                          "title": str(r.get("title") or "").strip()[:80],
                          "recap": str(r.get("recap") or "").strip()[:400],
                          "insight": str(r.get("insight") or "").strip()[:400]})
        if not clean:
            return _fallback(payload, moves, belief_history)
        syn = j.get("synthesis") if isinstance(j.get("synthesis"), dict) else {}
        synthesis = {k: str(syn.get(k) or "").strip()[:400] for k in ("consensus", "disagreement", "unique")}
        if not any(synthesis.values()):                  # model skipped it → deterministic synthesis
            synthesis = _fallback(payload, moves, belief_history)["synthesis"]
        return {"rounds": clean, "synthesis": synthesis}
    except Exception:
        return _fallback(payload, moves, belief_history)
