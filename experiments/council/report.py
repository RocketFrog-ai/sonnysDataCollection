"""
The committee's final in-depth report — one deterministic markdown synthesis of the whole
deliberation (`ws` + `log` + the settled `decision`), with an OPTIONAL one-shot Azure executive
summary bolted on top.

`synthesize_report(ws, log, decision) -> str` is the only public entry point. Everything below the
executive-summary line is plain Python string-building off the documented dataclass shapes in
`protocol.py` / `workspace.py` / `anchor.py` — no LLM, so the report never goes blank if Azure is
down. The one optional LLM call (an Azure `llm.complete`, temperature 0.3, ~350 tokens) writes a
2-3 sentence lead paragraph from the *already-computed* facts; on any failure (Azure not configured,
network error, bad JSON, anything) it is swapped for a deterministic one-liner built from the same
facts, so the report is fully deterministic whenever Azure is down.

Self-contained: imports only `experiments.council.{config,llm,protocol}` + stdlib (and `anchor` /
`workspace` only under `TYPE_CHECKING`, for hints — never at runtime). Never imports `app.*` /
`proforma.*` / `streamlit`. `ws`, `log`, and `decision` are read only through the attributes
documented for this module (see the task's INPUTS contract) so this file stays decoupled from
exactly how those dataclasses evolve.

Sections, in order (kept deliberately SHORT — the debate story lives in the UI digest + transcript):
header + verdict → executive summary → the numbers → where the committee stands → strengths & risks
(top 3 each) → recommendations (operating spec) → what-if sensitivity.
"""
from __future__ import annotations

import json
import math
import re
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from experiments.council import config as C
from experiments.council import llm
from experiments.council.protocol import MsgType

if TYPE_CHECKING:
    from experiments.council.anchor import Decision
    from experiments.council.workspace import DiscussionLog, Workspace


# ─────────────────────────── tiny formatters (deterministic, None-safe) ───────────────────────────
def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _clean(s: Any) -> str:
    """Collapse whitespace/newlines so free text (LLM memory, concerns, message text) stays one line."""
    return " ".join(str(s if s is not None else "").split())


def _esc_cell(s: Any) -> str:
    """`_clean` + escape `|` so free text can never break a markdown table row."""
    return _clean(s).replace("|", "\\|")


def _pct0(x: Any) -> str:
    """A 0..1 fraction (confidence / prob / yes_frac — all contractually 0..1) → 'NN%'."""
    v = _safe_float(x)
    return "—" if v is None else f"{v * 100:.0f}%"


def _pct(x: Any) -> str:
    """A share whose scale isn't pinned by contract (membership_share) → 'NN%', tolerant of 0-100 input."""
    v = _safe_float(x)
    if v is None:
        return "—"
    p = v * 100 if abs(v) <= 1.5 else v
    return f"{p:.0f}%"


def _money(x: Any) -> str:
    """Thousands get comma-separated raw dollars; ≥$1M switches to an 'M' suffix."""
    v = _safe_float(x)
    if v is None:
        return "—"
    sign, av = ("-" if v < 0 else ""), abs(v)
    if av >= 1_000_000:
        return f"{sign}${av / 1_000_000:,.2f}M"
    return f"{sign}${av:,.0f}"


def _month(x: Any) -> str:
    v = _safe_float(x)
    return "—" if v is None else f"Month {int(round(v))}"


def _washes(x: Any) -> str:
    v = _safe_float(x)
    return "—" if v is None else f"{v:,.0f} washes/mo"


def _feet(x: Any) -> str:
    v = _safe_float(x)
    return "—" if v is None else f"{v:,.0f} ft"


def _num(x: Any) -> str:
    v = _safe_float(x)
    if v is None:
        return "—"
    return f"{v:,.0f}" if float(v).is_integer() else f"{v:,.2f}"


def _fmt_key_number(value: Any, label: Optional[str]) -> str:
    """An expert's headline `key_number` + its `key_number_label`, e.g. '12,588 mature washes/mo' or
    '$3.10M 5-yr net' (a label ending in '$' renders the value as money and drops the trailing '$')."""
    v = _safe_float(value)
    if v is None:
        return "—"
    lbl = _clean(label or "")
    is_money = lbl.endswith("$")
    lbl = lbl.rstrip("$").strip()
    val_s = _money(v) if is_money else _num(v)
    return f"{val_s} {lbl}".strip() if lbl else val_s


# the 7 headline rows for "## The numbers", also reused as the LLM fact sheet
_NUMBER_ROWS: List[Tuple[str, str, Callable[[Any], str]]] = [
    ("revenue_5yr", "Expected 5-yr revenue", _money),
    ("net_5yr", "Expected 5-yr net", _money),
    ("breakeven_month", "Breakeven", _month),
    ("membership_share", "Membership share", _pct),
    ("mature_washes", "Mature wash count", _washes),
    ("tunnel_ft", "Recommended tunnel", _feet),
    ("capex", "Est. CAPEX", _money),
]


# ─────────────────────────── section 1: header + verdict ───────────────────────────
def _section_header(decision: "Decision") -> str:
    label = C.VERDICT_LABELS.get(decision.verdict, decision.verdict or "Unknown")
    meta = [f"confidence **{_pct0(decision.confidence)}**", f"basis: _{_clean(decision.basis)}_"]
    lines = [f"# 🧭 Committee recommendation — {label}", "", " · ".join(meta)]
    if decision.condition:
        lines += ["", f"**Condition:** {_clean(decision.condition)}"]
    return "\n".join(lines)


# ─────────────────────────── section 2: the numbers ───────────────────────────
def _numbers_table(numbers: Dict[str, Any]) -> str:
    rows = ["| Metric | Value |", "|---|---|"]
    for key, label, fmt in _NUMBER_ROWS:
        rows.append(f"| {label} | {fmt(numbers.get(key))} |")
    return "\n".join(rows)


# ─────────────────────────── section 3: where the committee stands ───────────────────────────
def _committee_table(ws: "Workspace") -> str:
    rows = ["| Seat | Lean | Confidence | Key number | Concern |", "|---|---|---|---|---|"]
    for name in C.EXPERT_ORDER:
        meta = C.EXPERT_META.get(name, {})
        seat = f"{meta.get('emoji', '')} {meta.get('name', name.title())}".strip()
        if name in C.WORLD_EXPERTS:
            seat += " 🌐"
        b = ws.beliefs.get(name)
        if b is None:
            rows.append(f"| {seat} | — | — | — | _no belief recorded_ |")
            continue
        lean = C.VERDICT_LABELS.get(b.lean, b.lean) if b.lean else "—"
        concern = _esc_cell(b.open_concerns[0]) if b.open_concerns else "—"
        kn = _esc_cell(_fmt_key_number(b.key_number, b.key_number_label))
        rows.append(f"| {seat} | {lean} | {_pct0(b.confidence)} | {kn} | {concern} |")
    table = "\n".join(rows)
    if C.WORLD_EXPERTS:
        table += ("\n\n_🌐 = world-knowledge seat (down-weighted in the committee vote); "
                 "unmarked seats are data-grounded._")
    return table


# ─────────────────────────── section 4: key strengths / key risks ───────────────────────────
def _key_strengths(ws: "Workspace") -> List[str]:
    out: List[str] = []
    for name in C.EXPERT_ORDER:
        b = ws.beliefs.get(name)
        if b and b.lean == "Build":
            seat = C.EXPERT_META.get(name, {}).get("name", name.title())
            kn = _fmt_key_number(b.key_number, b.key_number_label)
            tail = f" — {kn}." if kn != "—" else "."
            out.append(f"**{seat}** leans Build ({_pct0(b.confidence)} confidence){tail}")
    if not out:
        out.append("_No seat leans Build — the case to build is thin._")
    return out


def _key_risks(ws: "Workspace", log: "DiscussionLog", decision: "Decision") -> List[str]:
    out: List[str] = []
    for name in C.EXPERT_ORDER:
        b = ws.beliefs.get(name)
        if not b:
            continue
        seat = C.EXPERT_META.get(name, {}).get("name", name.title())
        if b.lean == "Pass":
            kn = _fmt_key_number(b.key_number, b.key_number_label)
            tail = f" — {kn}." if kn != "—" else "."
            out.append(f"**{seat}** leans Pass ({_pct0(b.confidence)} confidence){tail}")
        for concern in (b.open_concerns or [])[:2]:
            out.append(f"**{seat}** flags: {_clean(concern)}.")
    for m in log.unanswered(MsgType.CHALLENGE):
        frm = C.EXPERT_META.get(m.sender, {}).get("name", m.sender.title())
        to = C.EXPERT_META.get(m.to, {}).get("name", m.to.title()) if m.to else "the committee"
        out.append(f"Unresolved challenge — **{frm}** → **{to}**: {_clean(m.text)}")
    if not out:
        out.append("_No material open risks were raised._")
    return out


# ─────────────────────────── section 5: recommendations (operating spec) ───────────────────────────
def _ramp_months(ws: "Workspace") -> Optional[int]:
    """The learned ramp-to-90% months, parsed from Historical's own ramp evidence text."""
    ev = ws.evidence.get("hist.ramp_pattern")
    m = re.search(r"~?90% of mature in (\d+)\s*mo", str(ev.value)) if ev is not None else None
    return int(m.group(1)) if m else None


def _recommendations(ws: "Workspace", decision: "Decision") -> List[str]:
    """The committee's operating spec — not just go/no-go but HOW to build it: tunnel, pricing,
    membership mix, and the maturation timeline. Deterministic, from the settled board."""
    n = decision.numbers or {}
    out: List[str] = []
    tunnel, peak = _f(n.get("tunnel_ft")), _f(n.get("peak_month_washes"))
    if tunnel is not None:
        spec = int(math.ceil(tunnel / 5.0) * 5)          # build to the next 5-ft increment
        peak_s = f" for the {peak:,.0f}-wash peak month" if peak is not None else ""
        out.append(f"**Tunnel:** build **~{spec} ft**{peak_s} (computed {tunnel:,.0f} ft, incl. 20 ft buffer).")
    asp_ev = ws.evidence.get("hist.cluster_asp")
    asp = asp_ev.value if (asp_ev is not None and isinstance(asp_ev.value, dict)) else {}
    am, ar = _f(asp.get("asp_mem")), _f(asp.get("asp_ret"))
    if am is not None or ar is not None:
        prices = " · ".join(filter(None, [f"membership **${am:,.0f}**" if am is not None else None,
                                          f"retail **${ar:,.0f}**" if ar is not None else None]))
        out.append(f"**Pricing:** {prices} per wash — the cluster's observed ASPs.")
    ms = _f(n.get("membership_share"))
    if ms is not None:
        out.append(f"**Membership mix:** target **{ms:.0%}** of washes on plans (cluster-observed).")
    mature, ramp_mo = _f(n.get("mature_washes")), _ramp_months(ws)
    if mature is not None:
        ramp_s = f"; ~90% by month {ramp_mo}" if ramp_mo else ""
        out.append(f"**Maturation:** ≈ **{mature:,.0f} washes/mo** at maturity, over **24–30 months**{ramp_s} "
                   "— don't judge year 1.")
    capex, be = _f(n.get("capex")), _f(n.get("breakeven_month"))
    if capex is not None and be is not None:
        out.append(f"**Capital:** ≈ **{_money(capex)}**, paid back around **month {be:,.0f}**.")
    return out or ["_Not enough settled numbers on the board to write an operating spec._"]


# ─────────────────────────── section 7: what-if (deterministic sensitivity) ───────────────────────────
_f = _safe_float                                         # the shared None/NaN-safe float coercion


def _what_if(decision: "Decision") -> str:
    """Pure-math sensitivity on the committee's own numbers: how the 5-yr net and breakeven move if demand
    or ASP disappoints. Linear on the realized operating margin; opex ratio and ramp shape held fixed."""
    n = decision.numbers or {}
    rev, net, capex = _f(n.get("revenue_5yr")), _f(n.get("net_5yr")), _f(n.get("capex"))
    washes = _f(n.get("mature_washes"))
    if not rev or net is None or capex is None or rev <= 0:
        return "_Insufficient settled numbers for a sensitivity read._"
    margin = (net + capex) / rev                         # operating margin over 5yr, gross of build cost
    rows = ["| Scenario | 5-yr revenue | 5-yr net | ~Breakeven |", "|---|---|---|---|"]

    def _row(label: str, rev2: float) -> None:
        net2 = rev2 * margin - capex
        be2 = (capex / (rev2 * margin / 60.0)) if rev2 * margin > 0 else None
        be_s = f"mo {be2:,.0f}" if (be2 is not None and be2 <= 60) else ("> 60 mo" if be2 else "—")
        rows.append(f"| {label} | {_money(rev2)} | {_money(net2)} | {be_s} |")

    _row("Base (committee)", rev)
    for d in (-0.20, -0.10, +0.10):
        _row(f"Demand {d:+.0%}", rev * (1 + d))
    if washes:
        _row("ASP −$1/wash", rev - washes * 60.0)        # every wash a dollar cheaper across 5 yrs (approx.)
    return "\n".join(rows) + (
        f"\n\n_Linear on the realized {margin:.0%} operating margin; opex ratio and ramp held fixed._")


# ─────────────────────────── optional Azure exec-summary lead-in ───────────────────────────
_EXEC_SUMMARY_SYS = (
    "You write a 2-3 sentence executive-summary lead-in for a car-wash site-selection committee "
    "report, for a real-estate/ops decision-maker who will skim only this paragraph. Use ONLY the "
    "JSON facts given below — never invent or restate a number that isn't in the JSON. Plain English, "
    "no jargon, no markdown headers or bullet points, no repeating the JSON back verbatim. Cover: the "
    "verdict, the one or two headline numbers that matter most, and the single biggest risk if one "
    "is present. Return plain prose only."
)


def _fallback_summary(decision: "Decision") -> str:
    label = C.VERDICT_LABELS.get(decision.verdict, decision.verdict)
    return (f"The committee's recommendation is **{label}** (confidence {_pct0(decision.confidence)}), "
            f"based on {_clean(decision.basis)}.")


def _exec_summary_facts(ws: "Workspace", decision: "Decision", strengths: List[str],
                        risks: List[str]) -> Dict[str, Any]:
    numbers = decision.numbers or {}
    return {
        "verdict": C.VERDICT_LABELS.get(decision.verdict, decision.verdict),
        "confidence_pct": _pct0(decision.confidence),
        "basis": decision.basis,
        "condition": decision.condition,
        "numbers": {label: fmt(numbers.get(key)) for key, label, fmt in _NUMBER_ROWS},
        "top_strengths": strengths[:3],
        "top_risks": risks[:3],
        "committee_yes_frac_pct": _pct0(decision.yes_frac) if decision.yes_frac is not None else None,
    }


def _executive_summary(ws: "Workspace", decision: "Decision", strengths: List[str], risks: List[str]) -> str:
    """ONE Azure call for a 2-3 sentence lead-in; any failure (Azure down, bad JSON, network) falls back
    to a deterministic one-liner built from the same facts — the report is never empty, never blocked."""
    facts = _exec_summary_facts(ws, decision, strengths, risks)
    text: Optional[str] = None
    try:
        raw = llm.complete(
            [{"role": "system", "content": _EXEC_SUMMARY_SYS},
             {"role": "user", "content": json.dumps(facts, ensure_ascii=False, default=str)[:6000]}],
            temperature=0.3, max_tokens=350)
        raw = _clean(raw)
        if raw:
            text = raw
    except Exception:
        text = None
    return f"_{text or _fallback_summary(decision)}_"


# ─────────────────────────── public entry point ───────────────────────────
def synthesize_report(ws, log, decision) -> str:
    """Return the committee report — short and to the point: verdict, a 2-3 sentence summary, the numbers,
    the seats, top-3 strengths/risks, the operating spec, and the what-if table. The debate story lives in
    the UI digest + transcript, not here. One optional Azure exec-summary pass (deterministic fallback)."""
    strengths = _key_strengths(ws)[:3]
    risks = _key_risks(ws, log, decision)[:3]

    sections = [
        _section_header(decision),
        _executive_summary(ws, decision, strengths, risks),
        "## The numbers\n\n" + _numbers_table(decision.numbers or {}),
        "## Where the committee stands\n\n" + _committee_table(ws),
        "## Strengths & risks\n\n" + "\n".join(f"- ✅ {s}" for s in strengths)
            + "\n" + "\n".join(f"- ⚠️ {r}" for r in risks),
        "## Recommendations — the operating spec\n\n" + "\n".join(f"- {r}" for r in _recommendations(ws, decision)),
        "## What-if (sensitivity)\n\n" + _what_if(decision),
    ]
    return "\n\n".join(sections)
