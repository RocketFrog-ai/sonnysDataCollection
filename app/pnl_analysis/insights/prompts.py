"""
Node 2 plumbing — prompt construction and response parsing.

  • build_combined_messages(metrics)  — investor persona + a MARKET CONTEXT block (start dates,
                                        per-site reporting spans, coverage caveat) + the three grounded
                                        fact blocks (levels, peak-vs-current, momentum, yearly data
                                        points, entrant effect, cannibalization, mix, ASP) + the required
                                        output shape, as one chat request.
  • parse_group_sections(text)        — pull each [Washes]/[Revenue]/[ASPs] section back into rendered
                                        markdown (headline + bullets + a colour-coded signal).

Grounding discipline mirrors app/site_analysis/.../summaries.py: feed ONLY the computed numbers and
forbid invention. There is NO fixed-string fallback narrative — the graph cascades Azure↔local and
surfaces an honest error if neither answers.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

_SIGNAL_EMOJI = {"positive": "🟢", "neutral": "⚪", "cautionary": "🟠", "negative": "🔴"}

SYSTEM_PROMPT = (
     "You are a private-equity investor running a site-selection model for car washes, deciding whether to BUILD a NEW site in this local market. "
    "You read the market's actual monthly operating data (every site here has 30+ months of history) and write a tight, decision-useful read-outs "
   "\n"
    "What you are deciding: is this an attractive market to enter? Your best analog for a new build is how the LAST entrant ramped here. "
    "Weigh market size & headroom, membership demand quality, pricing power (ASP), competitive density, per-site economics, and cannibalization risk.\n"
    "Rules:\n"
    "- Use ONLY the numbers provided. Never invent, extrapolate beyond a stated trend, or cite a number not given. If something is absent, omit it.\n"
    "- Be concrete and quantitative: cite absolute levels, dates, peak-vs-current, recent momentum (last N months) vs year-over-year, and the new entrant's ramp & effect.\n"
    "- LEVELS ARE 3-MONTH AVERAGED: every 'peak', 'current' and YoY here is a ~3-month average, not a single month — so describe a peak as a period (e.g. 'around 2024-Q2'), never as a one-month spike.\n"
    "- SAME-STORE (a.k.a. like-for-like) = compares ONLY the sites reporting in BOTH periods. If sites stopped reporting (see MARKET CONTEXT) a falling market SUM is a coverage artifact, not real demand loss: lead with the same-store YoY, call it 'same-store', and never call a coverage drop a market decline.\n"
    "- CONSISTENCY: if same-store demand is flat or only modestly down, do not later describe the market as 'declining' anywhere else in the read-out, including the verdict — use 'maturing', 'saturated', or 'plateaued' instead. Every later section must stay consistent with what the Market Demand section established.\n"
    "- Membership penetration and its growth are your most important demand-quality signal — but a high membership share is only a GOOD signal if that base isn't already locked into a competitor. If the dominant share belongs to an incumbent, say so explicitly as a lock-in risk, not just a demand-quality positive.\n"
    "- Competitive density is measured by distance to the nearest existing site, not by volume — always cite the nearest-site distance and state plainly whether there is or isn't an open trade area.\n"
    "- ASP and the membership premium are your pricing-power signal, judged independently of volume — strong, stable ASPs can coexist with a bad location decision.\n"
    "- The verdict should state that (a) states in one sentence whether the economics are sound and whether the location is sound, as two separate judgments, and (b) gives the recommendation plus the one condition that would change your mind (e.g., a different site becoming available, or an incumbent's exit)."
    )


# ─────────────────────────── number formatting ───────────────────────────
def _pct(x: Optional[float]) -> Optional[str]:
    return None if x is None else f"{x:+.0%}"


def _pp(x: Optional[float]) -> Optional[str]:
    return None if x is None else f"{x:+.1f} pp"


def _per_yr(x: Optional[float]) -> Optional[str]:
    return None if x is None else f"{x:+.0%}/yr"


def _count(x: Optional[float]) -> Optional[str]:
    return None if x is None else f"{x:,.0f}"


def _money0(x: Optional[float]) -> Optional[str]:
    return None if x is None else f"${x:,.0f}"


def _money2(x: Optional[float]) -> Optional[str]:
    return None if x is None else f"${x:,.2f}"


def _share(x: Optional[float]) -> Optional[str]:
    return None if x is None else f"{x:.0%}"


def _level_line(label: str, b: Dict[str, Any], money: bool = False, n: int = 12) -> Optional[str]:
    """A rich one-liner for a market series: now / peak / drawdown / YoY (+ same-store) / last-N / trend / since."""
    fmt = _money0 if money else _count
    parts: List[str] = []
    if b.get("current") is not None:
        parts.append(f"now {fmt(b['current'])}/mo")
    if b.get("peak") is not None and b.get("peak_date"):
        parts.append(f"peak ~{fmt(b['peak'])}/mo around {b['peak_date']}")
    if b.get("current_vs_peak") is not None:
        parts.append(f"{_pct(b['current_vs_peak'])} vs peak")
    if b.get("yoy") is not None:
        y = f"YoY {_pct(b['yoy'])}"
        if b.get("yoy_same_store") is not None:
            y += f" (same-store {_pct(b['yoy_same_store'])})"
        parts.append(y)
    if b.get("change_last_n") is not None:
        parts.append(f"last {n}mo {_pct(b['change_last_n'])}")
    if b.get("trend_annual") is not None:
        parts.append(f"trend {_per_yr(b['trend_annual'])}")
    if b.get("start"):
        parts.append(f"data since {b['start']}")
    return f"- {label}: " + "; ".join(parts) + "." if parts else None


def _traj(label: str, pts: List[Dict[str, Any]], money: bool = False) -> Optional[str]:
    if not pts:
        return None
    fmt = _money0 if money else _count
    body = ", ".join(f"{p['year']}: {fmt(p['avg'])}" for p in pts if p.get("avg") is not None)
    return f"- {label} by year (avg/mo): {body}." if body else None


# ─────────────────────────── shared market-context block ───────────────────────────
def _context_block(m: Dict[str, Any]) -> str:
    cov = m["coverage"]
    lines = [
        f"- Local market of {cov['n_sites']} site(s); data spans {cov['market_start']} → {cov['market_end']}.",
        f"- Sites reporting recently: {cov['active_now']} of {cov['n_sites']} (a year ago: {cov['active_year_ago']}).",
    ]
    for s in cov["sites"]:
        tags = []
        if s.get("is_entrant"):
            tags.append("NEW entrant")
        if not s.get("active_recent"):
            tags.append("STOPPED reporting")
        tag = f" [{', '.join(tags)}]" if tags else ""
        recent = _count(s.get("avg_washes_last12")) or "n/a"
        lines.append(f"  - {s['name']}: opened {s.get('op_start') or 'n/a'}, reports {s.get('first_obs')}"
                     f"→{s.get('last_obs')} ({s.get('months')} mo), ~{recent}/mo recently{tag}.")
    if cov.get("note"):
        lines.append(f"- COVERAGE NOTE: {cov['note']}")
    return "\n".join(lines)


def _site_selection_block(m: Dict[str, Any]) -> str:
    ss = m.get("site_selection") or {}
    lines: List[str] = []
    dens = []
    if ss.get("sites_in_market") is not None:
        dens.append(f"{ss['sites_in_market']} existing site(s)")
    if ss.get("nearest_site_km") is not None:
        dens.append(f"nearest {ss['nearest_site_km']:.1f} km")
    if ss.get("median_site_km") is not None:
        dens.append(f"median {ss['median_site_km']:.1f} km")
    if dens:
        lines.append("- Competitive density: " + ", ".join(dens) + ".")
    eco = []
    wps, rps = _count(ss.get("washes_per_active_site")), _money0(ss.get("revenue_per_active_site"))
    if wps:
        eco.append(f"{wps}/mo washes per active site")
    if rps:
        eco.append(f"{rps}/mo revenue per active site")
    if eco:
        lines.append("- Per-site economics (proxy for a new build): " + ", ".join(eco) + ".")
    pb = ss.get("last_entrant_playbook")
    if pb:
        mult = f", {pb['ramp_multiple']:.1f}x" if pb.get("ramp_multiple") is not None else ""
        lines.append(f"- Last-entrant analog ({pb.get('name')}): ramped {_count(pb.get('first3_per_month'))} → "
                     f"{_count(pb.get('current_per_month'))}/mo{mult} over {pb.get('months_open')} mo; peaked "
                     f"{_count(pb.get('peak_per_month'))}/mo at month {pb.get('months_to_peak')}; now "
                     f"{_share(pb.get('share_of_market_washes'))} of market washes.")
    return "\n".join(lines) if lines else "- (no site-selection context available)"


def _data_points_block(m: Dict[str, Any]) -> str:
    """The actual quarterly market series — gives the model the shape it would read off the charts."""
    dp = m.get("data_points_quarterly") or {}
    spec = [
        ("total_washes", "Total washes", _count), ("membership_washes", "Membership washes", _count),
        ("retail_washes", "Retail washes", _count), ("total_revenue", "Total revenue", _money0),
        ("membership_share_washes", "Membership share", _share), ("asp_retail", "Retail ASP", _money2),
        ("asp_membership", "Membership ASP", _money2),
    ]
    lines = []
    for key, label, fmt in spec:
        pts = dp.get(key) or []
        body = ", ".join(f"{p['q']} {fmt(p['avg'])}" for p in pts if p.get("avg") is not None)
        if body:
            lines.append(f"- {label} (quarterly avg/mo): {body}.")
    return "\n".join(lines) if lines else "- (no quarterly points)"


# ─────────────────────────── per-group fact blocks ───────────────────────────
def _washes_facts(m: Dict[str, Any]) -> List[str]:
    w = m["washes"]
    n = m["meta"]["last_n_months"]
    out = [
        _level_line("Total washes", w["total"], n=n),
        _level_line("Retail washes", w["retail"], n=n),
        _level_line("Membership washes", w["membership"], n=n),
    ]
    ms = w.get("membership_share") or {}
    if ms.get("current") is not None:
        extra = f"; peak ~{_share(ms['peak'])} around {ms['peak_date']}" if ms.get("peak") is not None else ""
        out.append(f"- Membership share of washes: now {_share(ms['current'])} (YoY {_pp(ms.get('yoy_delta_pp'))}){extra}.")
    out.append(_traj("Total washes", (w.get("trajectory_yearly") or {}).get("total", [])))
    out.append(_traj("Membership washes", (w.get("trajectory_yearly") or {}).get("membership", [])))
    ee = w.get("entry_effect")
    if ee:
        out.append(f"- Around the new entrant ({ee['entry_date']}): market total {_count(ee['pre_per_month'])}/mo → "
                   f"{_count(ee['post_per_month'])}/mo ({_pct(ee['change'])}).")
    fr = w.get("focal_ramp")
    if fr:
        thin = ", thin history" if fr.get("short_history") else ""
        peak = (f", peaked ~{_count(fr['peak_per_month'])}/mo around {fr['peak_date']} ({_pct(fr.get('current_vs_peak'))} since)"
                if fr.get("peak") is not None else "")
        out.append(f"- New site ramp: {_count(fr['first3_per_month'])}/mo → {_count(fr['current_per_month'])}/mo over "
                   f"{fr['months_open']} mo ({_pct(fr['ramp'])}){peak}{thin}.")
    cb = w.get("cannibalization")
    if cb:
        out.append(f"- Incumbents after entry (retail): {_pct(cb.get('retail_change'))} "
                   f"(deseasonalized {_pct(cb.get('retail_change_deseason_median'))}), across {cb['n_incumbents']} sites.")
    return [o for o in out if o]


def _revenue_facts(m: Dict[str, Any]) -> List[str]:
    r = m["revenue"]
    n = m["meta"]["last_n_months"]
    out = [_level_line("Total revenue", r["total"], money=True, n=n)]
    sh = r.get("mem_share") or {}
    if sh.get("current") is not None:
        slope = f"; {sh['slope_pp_per_yr']:+.1f} pp/yr" if sh.get("slope_pp_per_yr") is not None else ""
        out.append(f"- Revenue mix: membership {_share(sh['current'])} / retail {_share(sh.get('retail_share_current'))} "
                   f"(YoY {_pp(sh.get('yoy_delta_pp'))}{slope}).")
    pw = r.get("per_wash") or {}
    if pw.get("current") is not None:
        out.append(f"- Blended revenue per wash: {_money2(pw['current'])} (YoY {_pct(pw.get('yoy'))}).")
    mv = r.get("mem_vs_ret_yoy") or {}
    if mv.get("membership") is not None or mv.get("retail") is not None:
        out.append(f"- Revenue growth by stream (YoY): membership {_pct(mv.get('membership'))}, retail {_pct(mv.get('retail'))}.")
    out.append(_traj("Total revenue", (r.get("trajectory_yearly") or {}).get("total", []), money=True))
    out.append(_traj("Membership revenue", (r.get("trajectory_yearly") or {}).get("membership", []), money=True))
    fc = r.get("focal_contribution")
    if fc:
        l12 = f" ({_share(fc['share_of_market_last12'])} over last 12 mo)" if fc.get("share_of_market_last12") is not None else ""
        out.append(f"- New site is {_share(fc.get('share_of_market'))} of latest-month market revenue{l12} "
                   f"({_money0(fc.get('focal_revenue_per_month'))}/mo).")
    return [o for o in out if o]


def _asp_facts(m: Dict[str, Any]) -> List[str]:
    a = m["asps"]
    n = m["meta"]["last_n_months"]
    out = []
    for key, label in (("retail", "Retail ASP/wash"), ("membership", "Membership ASP")):
        b = a.get(key) or {}
        if b.get("current") is None:
            continue
        peak = f"; peak ~{_money2(b['peak'])} around {b['peak_date']}" if b.get("peak") is not None else ""
        mom = b.get("mom_smoothed") if b.get("mom_smoothed") is not None else b.get("mom")
        out.append(f"- {label}: now {_money2(b['current'])} (MoM {_pct(mom)}, YoY {_pct(b.get('yoy'))}, "
                   f"last {n}mo {_pct(b.get('change_last_n'))}){peak}.")
    prem = a.get("membership_premium") or {}
    if prem.get("abs") is not None:
        ratio = f" ({prem['ratio']:.1f}x retail)" if prem.get("ratio") is not None else ""
        out.append(f"- Membership ASP premium over retail: {_money2(prem['abs'])}{ratio}.")
    out.append(_traj("Retail ASP", (a.get("trajectory_yearly") or {}).get("retail", []), money=True))
    out.append(_traj("Membership ASP", (a.get("trajectory_yearly") or {}).get("membership", []), money=True))
    g = a.get("focal_gap")
    if g:
        out.append(f"- New site vs incumbents: retail ASP {_money2(g.get('retail_focal'))} vs "
                   f"{_money2(g.get('retail_incumbent'))} ({_pct(g.get('retail_gap_pct'))}); membership ASP "
                   f"{_money2(g.get('membership_focal'))} vs {_money2(g.get('membership_incumbent'))}.")
    return [o for o in out if o]


_FACT_BUILDERS = {"Washes": _washes_facts, "Revenue": _revenue_facts, "ASPs": _asp_facts}


# ─────────────────────────── request construction (JSON output) ───────────────────────────
# def build_combined_messages(metrics: Dict[str, Any]) -> List[dict]:
#     blocks = []
#     for group in ("Washes", "Revenue", "ASPs"):
#         facts = "\n".join(_FACT_BUILDERS[group](metrics)) or "- (no data)"
#         blocks.append(f"[{group}]\n{facts}")
#     user = (
#         "MARKET CONTEXT (read first — explains data coverage):\n"
#         f"{_context_block(metrics)}\n\n"
#         "SITE-SELECTION CONTEXT (you are deciding whether to BUILD a new site here):\n"
#         f"{_site_selection_block(metrics)}\n\n"
#         "QUARTERLY DATA POINTS (the actual market series — read the shape; prefer same-store comparisons "
#         "when coverage dropped):\n"
#         f"{_data_points_block(metrics)}\n\n"
#         "FACTS by group:\n"
#         f"{chr(10).join(blocks)}\n\n"
#         "Return a single JSON object (and NOTHING else) with exactly these three keys: \"washes\", "
#         "\"revenue\", \"asps\". Each maps to an object:\n"
#         '  {"headline": "<one decision-useful sentence, no markdown>",\n'
#         '   "bullets": ["4 to 6 specific sentences, each grounded in the numbers above"],\n'
#         '   "signal": "positive" | "neutral" | "cautionary"}\n\n'
#         "Each section should help the reader interpret the plotted lines AND judge a new build. Cover:\n"
#         "- washes: current level vs peak (and % off peak), recent momentum (last-N) vs YoY (prefer same-store "
#         "YoY when coverage dropped), absolute total volume, the new entrant's ramp (use the last-entrant analog as "
#         "the expected ramp for a new build), its effect on incumbents, and a coverage caveat if any site stopped "
#         "reporting.\n"
#         "- revenue: total revenue level/peak/trend, the membership-vs-retail split & which stream drives growth, "
#         "blended revenue-per-wash, revenue per site, and the new site's revenue contribution.\n"
#         "- asps: retail & membership ASP level/peak, MoM/YoY/last-N momentum, the membership premium over retail "
#         "(pricing power), and how the new entrant prices vs incumbents — i.e. the pricing headroom for a new site.\n"
#         "End each section's last bullet with the implication for building a new site here. "
#         "Do not use markdown inside any string. signal = the investor's view of building here (positive = "
#         "attractive, cautionary = a real risk)."
#     )
#     return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
def build_combined_messages(metrics: Dict[str, Any]) -> List[dict]:
    blocks = []
    for group in ("Washes", "Revenue", "ASPs"):
        facts = "\n".join(_FACT_BUILDERS[group](metrics)) or "- (no data)"
        blocks.append(f"[{group}]\n{facts}")
    user = (
        "MARKET CONTEXT (read first — explains data coverage):\n"
        f"{_context_block(metrics)}\n\n"
        "SITE-SELECTION CONTEXT (you are deciding whether to BUILD a new site here):\n"
        f"{_site_selection_block(metrics)}\n\n"
        "QUARTERLY DATA POINTS (the actual market series — read the shape; prefer same-store comparisons "
        "when coverage dropped):\n"
        f"{_data_points_block(metrics)}\n\n"
        "FACTS by group:\n"
        f"{chr(10).join(blocks)}\n\n"
        "Return a single JSON object (and NOTHING else) with exactly these three keys: \"washes\", \"revenue\", \"asps\".\n\n"
    "\"revenue\" and \"asps\" must always be present as empty stubs:\n"
"  {\"headline\": \"\", \"bullets\": [], \"signal\": \"neutral\"}\n\n"    "\"washes\" must have this exact shape:\n"
    "  {\"headline\": \"one crisp sentence that separates the economics judgment from the location "
    "judgment, e.g. 'Sound membership economics, wrong location.'>\",\n"
    "   \"bullets\": [<array of plain-prose strings — one string per section, in the order listed below>],\n"
    "   \"signal\": \"<positive|neutral|cautionary|negative> — <sentence 1: elaborate the headline judgment in full>. <sentence 2: the recommendation plus the one condition that would change your mind.>\"}\n\n"

    "The \"bullets\" array must contain exactly these sections as individual plain-prose strings, in this exact order. "
    "Each string must begin with the section label followed by ' — ' and then 2-4 sentences of analysis:\n\n"
    "\"⚠️ Data Note\" — include ONLY if MARKET CONTEXT shows sites stopped reporting; state plainly that the raw "
    "market-level trend is a coverage artifact and name which figure (same-store) the reader should trust instead. "
    "If coverage didn't change, omit this string from the array entirely.\n\n"
    "\"Market Demand\" — same-store YoY vs raw YoY; current level vs peak (with date and % off peak); AND whether "
    "the most relevant comparable (the most recent entrant, if one exists) is itself above or below its own peak — "
    "use that to judge whether the growth window is still open or has likely closed.\n\n"
    "\"Business Model\" — which revenue model (membership vs. retail) is actually carrying the market, based on share "
    "level and its trend; and if that share is already concentrated in a single existing competitor, name that "
    "explicitly as a lock-in risk rather than just a positive signal.\n\n"
    "\"Competitive Headroom\" — lead with the distance to the nearest existing site; conclude plainly whether there is "
    "open trade-area separation or whether this is a head-to-head fight with an entrenched operator.\n\n"
    "\"Market Timing\" — the period the market peaked, and where the most relevant comparable now sits in its own ramp "
    "(still ramping / plateaued / past its own peak); conclude whether the entry window is open or closed. Use "
    "'maturing' / 'saturated' / 'plateaued' if demand is flat same-store — never 'declining'.\n\n"
    "\"Revenue & Pricing\" — ASP level/trend for retail and membership as the pricing-power signal, separate from "
    "volume; state plainly what would actually have to happen for a new site's revenue to ramp (e.g., pulling "
    "subscribers from an incumbent vs. organic growth).\n\n"
    
    "Do not use markdown, nested JSON, or bullet points inside any string value. "
    "Do not invent a section label not listed above. Data Note is the only optional section."

)
    
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


# ─────────────────────────── response parsing (JSON, with a loose fallback) ───────────────────────────
def _esc_dollars(s: str) -> str:
    """Escape $ so Streamlit/KaTeX doesn't render $...$ pairs as LaTeX math."""
    return s.replace("$", "\\$")


def _render_group(obj: Any) -> Optional[str]:
    if not isinstance(obj, dict):
        return None
    head = str(obj.get("headline") or "").strip().strip("*").strip().rstrip(".")
    bullets = obj.get("bullets") or []
    if isinstance(bullets, str):
        bullets = [bullets]

    # Keep raw for display, lowercase only for keyword lookup
    sig_raw = str(obj.get("signal") or "").strip()

    parts: List[str] = []
    if head:
        parts.append(f"**{_esc_dollars(head)}**")
    clean = [_esc_dollars(str(b).strip().lstrip("-•*").strip()) for b in bullets if str(b).strip()]
    if clean:
        parts.append("\n".join(f"- {b}" for b in clean))

    # Split "positive — verdict line 1. verdict line 2." if present
    if " — " in sig_raw:
        sig_word, verdict_text = sig_raw.split(" — ", 1)
        sig_word = sig_word.strip().lower()
        verdict_text = verdict_text.strip()
    else:
        sig_word = sig_raw.strip().lower()
        verdict_text = ""

    if sig_word in _SIGNAL_EMOJI:
        verdict_display = f"\n\n{_esc_dollars(verdict_text)}" if verdict_text else ""
        parts.append(f"{_SIGNAL_EMOJI[sig_word]} _{sig_word}_{verdict_display}")

    return "\n\n".join(parts).strip() or None


def _extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    t = re.sub(r"```(?:json)?|```", "", text).strip()  # strip code fences
    try:
        return json.loads(t)
    except Exception:
        pass
    m = re.search(r"\{.*\}", t, re.DOTALL)              # first {...} block
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


# loose fallback if a backend ignored JSON mode: tolerate [Washes] / **Washes** / ### Washes / Washes: headers
_LOOSE_RE = re.compile(
    r"(?:^|\n)[\[\*#\s]*\b(Washes|Revenue|ASPs)\b[\]\*:]*\s*\n(.+?)"
    r"(?=\n[\[\*#\s]*\b(?:Washes|Revenue|ASPs)\b[\]\*:]*\s*\n|\Z)",
    re.DOTALL | re.IGNORECASE)


def _render_loose(body: str) -> Optional[str]:
    head = re.search(r"(?:Headline:)?\s*(.+)", body)
    sig = re.search(r"Signal:\s*(positive|neutral|cautionary|negative)", body, re.IGNORECASE)
    bullets = re.findall(r"^\s*[-*]\s+(.+)$", body, re.MULTILINE)
    headline = (re.sub(r"^Headline:\s*", "", head.group(1), flags=re.IGNORECASE) if head else "")
    return _render_group({"headline": headline, "bullets": bullets,
                          "signal": sig.group(1) if sig else ""})


def parse_group_sections(text: str) -> Dict[str, Optional[str]]:
    """Map LLM output to {'Washes','Revenue','ASPs'} rendered-markdown (None if a group is missing)."""
    out: Dict[str, Optional[str]] = {"Washes": None, "Revenue": None, "ASPs": None}
    data = _extract_json(text)
    if isinstance(data, dict):
        low = {(k.lower() if isinstance(k, str) else k): v for k, v in data.items()}
        for disp, key in (("Washes", "washes"), ("Revenue", "revenue"), ("ASPs", "asps")):
            out[disp] = _render_group(low.get(key))
        if any(out.values()):
            return out
    for mtch in _LOOSE_RE.finditer(text or ""):       # fallback for non-JSON output
        grp = mtch.group(1)
        out["ASPs" if grp.lower() == "asps" else grp.capitalize()] = _render_loose(mtch.group(2))
    return out
