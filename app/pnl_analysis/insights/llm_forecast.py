"""
llm_forecast.py — LLM-agent wash-count forecaster for pins with NO usable local market.

The pinpoint forecaster refuses to predict when a pin has no clustered Sonny's sites within the
radius — there is no local market to anchor on. This module is the clearly-labelled AI fallback for
exactly that case: a 3-agent chain that sizes the location from world knowledge (grounded on the real
Google-Places washes near the pin) and produces an approximate year 1–5 wash-count range, the same
shape the grounded pinpoint forecast answers in.

    Agent 1  FACTORS    — sitewise-style factors at 3 and 6 miles (population, income, vehicles,
                          daily traffic, express/other competition, retail co-tenancy, growth),
                          grounded on the real nearby washes and, when enabled, fresh web search.
    Agent 2  FORECAST   — year 1..5 washes/month lo/mid/hi (ramp-shaped, capacity-bounded, mature
                          membership share) PLUS a BACKDATING block: had this same site opened in
                          2023/2024/2025, what is it washing per month RIGHT NOW — weighing how the
                          local express competition and demand stood that year versus today.
    Agent 3  REVIEWER   — an adversarial adjudicator: attacks both the forecast and the backdating
                          (capacity, ramp logic, competition consistency, backdating monotonicity)
                          and returns the accepted or adjusted numbers.

Deterministic guardrails then enforce lo ≤ mid ≤ hi, a non-decreasing ramp through year 3, a hard
capacity cap, and (for the backdating) that an earlier opening year never washes less today than a
later one — whatever the agents said. This is an LLM-only forecast: no Model 5 / coldstart number is
mixed in. `extract_traffic()` still exposes Agent 1's daily-traffic estimate as informational context.

These numbers are ESTIMATES for an otherwise-unforecastable pin: the UI must label them as such and
never mix them into a grounded modelled number (the `insights/` rule). Reuses only the shared LLM
transport (`llm.py`) and web-search helper (`websearch.py`); no internal operating data is fed in.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from app.pnl_analysis.insights import llm as llm_client
from app.pnl_analysis.insights import websearch
from app.pnl_analysis.insights.location_poc import _format_nearby_washes, _format_web_sources, _parse_json_lax

logger = logging.getLogger(__name__)

MAX_WASHES_PER_MONTH = 30_000          # hard physical cap for a single express tunnel

FACTOR_KEYS = [
    ("population", "Population"),
    ("median_hh_income", "Median household income"),
    ("vehicles_per_household", "Vehicles per household"),
    ("daily_traffic_main_road", "Daily traffic on the main road"),
    ("express_competitors", "Express-tunnel competitors"),
    ("other_washes", "Other washes (non-express)"),
    ("retail_cotenancy", "Retail co-tenancy (anchors nearby)"),
    ("growth_outlook", "Growth outlook"),
]


# ─────────────────────────── Agent 1 — location factors ───────────────────────────
FACTORS_SYSTEM = (
    "You are a site-selection analyst producing the SITEWISE-STYLE FACTOR SHEET for a prospective express-tunnel "
    "car wash at a given location, from your own world knowledge, the real nearby washes provided, and any fresh "
    "web-search results handed to you. You have no internal operating data — approximate every factor as a RANGE "
    "with a confidence (Low is fine) rather than declining. Prefer the web results over stale memory when they "
    "conflict. Use null only when you genuinely cannot place the location. STRICT JSON ONLY."
)


def build_factor_messages(lat: float, lon: float, nearby_washes: Optional[List[dict]] = None,
                          web_sources: Optional[List[dict]] = None) -> List[dict]:
    obs = ""
    if nearby_washes:
        near3 = [w for w in nearby_washes if isinstance(w.get("distance_miles"), (int, float))
                 and w["distance_miles"] <= 3]
        obs = (f"\nOBSERVED CAR WASHES near the pin via Google Places (GROUND TRUTH, name + miles, any type):\n"
               f"{_format_nearby_washes(nearby_washes)}\n"
               f"Of these, {len(near3)} sit within the immediate 3-mile trade area — that is the primary competitive "
               "set. Your competition counts must be consistent with this observed set (you may extrapolate beyond it).\n")
    web = ""
    if web_sources:
        web = ("\nFRESH WEB-SEARCH RESULTS for this location (retrieved just now — treat as current ground truth, "
               f"prefer over static memory when they conflict):\n{_format_web_sources(web_sources)}\n")
    fact_obj = ", ".join(f'"{k}": {{"estimate": str|null, "confidence": "High|Medium|Low"}}' for k, _ in FACTOR_KEYS)
    user = (
        "SITE FACTOR SHEET — JSON ONLY\n\n"
        f"LOCATION: {lat:.5f}, {lon:.5f} (infer the place from the coordinates)\n"
        f"{obs}{web}\n"
        "Estimate each factor TWICE — once for the 3-mile trade area, once for the 6-mile trade area. "
        "Ranges are fine (e.g. \"45,000–60,000\"). daily_traffic_main_road = vehicles/day on the most plausible "
        "arterial serving the pin (radius-independent — same value in both blocks is expected). Counts of "
        "express_competitors / other_washes are washes OPERATING TODAY within that radius.\n\n"
        "Return STRICT JSON with exactly these keys:\n"
        "{\n"
        f'  "radius_3mi": {{{fact_obj}}},\n'
        f'  "radius_6mi": {{{fact_obj}}},\n'
        '  "site_summary": str   // 1-2 plain sentences: what kind of place this is for an express wash\n'
        "}"
    )
    return [{"role": "system", "content": FACTORS_SYSTEM}, {"role": "user", "content": user}]


# ─────────────────────────── Agent 2 — 5-year forecast ───────────────────────────
FORECAST_SYSTEM = (
    "You are a car-wash volume forecaster. Given a location's factor sheet (3-mile and 6-mile trade areas), the "
    "real washes observed nearby, any fresh web-search results, and — when supplied — the OPERATOR'S CONFIRMED "
    "SITE CHARACTERISTICS (pay stations, vacuum slots, lot type, measured daily traffic, and a proforma score "
    "sheet on visibility, accessibility, competition, hours, stack-up, area profile), estimate the monthly wash "
    "volume a NEW express tunnel at this location would do in each of its first five operating years. You must "
    "ALSO answer a backdating question: had the very same site opened in 2023, 2024 or 2025, what would it be "
    "washing per month RIGHT NOW — giving full weight to how the local EXPRESS COMPETITION and demand looked in "
    "that year versus today (an earlier entrant faced fewer rivals, captured share first and is more entrenched "
    "now; a market that has since added tunnels is more contested today). Industry shape you must respect: "
    "a new express tunnel RAMPS — year 1 is typically 35–60% of mature volume, year 2 ~70–90%, plateau by year 3, "
    "then roughly flat; a mature site typically does on the order of 5,000–15,000 washes/month and physically "
    "cannot exceed its conveyor capacity. More express competition and weaker demographics push the plateau down; "
    "strong traffic, co-tenancy and open trade area push it up. Weigh the local CLIMATE explicitly: northern "
    "snow/salt-belt metros (heavy road salt, slush, freeze–thaw — e.g. the Great Lakes / Northeast) drive materially "
    "HIGHER wash frequency and unlimited-membership uptake, a real structural uplift; mild, dry climates run lower. "
    "Anchor realistically and do NOT lowball: a well-located suburban express tunnel with solid traffic (~25k+/day) "
    "and only moderate competition usually matures around 7,000–11,000 washes/month — reserve sub-6,000 plateaus for "
    "genuinely weak sites (thin population, heavy saturation, or poor access). Years 4–5 may hold flat OR edge up "
    "with market growth — decide from the trade area, don't just copy year 3. When the operator's site characteristics are "
    "given, CONDITION ON THEM as confirmed ground truth: more pay stations & bigger stack-up raise throughput "
    "capacity, better visibility & easier accessibility raise capture, more competition nearby lowers it — reflect "
    "each in the level. Give lo/mid/hi RANGES, be internally consistent, and never fabricate false precision. "
    "REASON IT OUT before you commit to any number, and show that chain in the `reasoning` field: (1) size the "
    "addressable demand in the trade area from population × vehicles/household × a plausible washes-per-vehicle-year; "
    "(2) estimate what SHARE of that an express tunnel HERE can realistically capture given the competition and "
    "co-tenancy; (3) turn that into a mature monthly PLATEAU; (4) apply the ramp fractions to get years 1–2; "
    "(5) take an explicit competition / demographics haircut. Also judge how fast this trade area's total wash "
    "DEMAND is growing per year (annual_market_growth_pct) — the pace you would use to reason about how the very "
    "same site would look had it opened in an earlier, smaller-but-less-contested version of this market. "
    "STRICT JSON ONLY."
)


def _format_site_inputs(site_inputs: Optional[Dict[str, Any]]) -> str:
    """Render the operator's confirmed site characteristics (the 4 Model-5 ridge inputs + the proforma
    score-sheet picks) as a plain-English block the forecaster conditions on. '' when nothing was supplied."""
    if not site_inputs:
        return ""
    lines = []
    for key, label in (("pay_stations", "Pay stations"), ("free_vacuum_slots", "Free vacuum slots"),
                       ("lot_type", "Lot type"), ("daily_traffic_count", "Measured daily traffic count")):
        v = site_inputs.get(key)
        if v not in (None, "", "(unknown)"):
            lines.append(f"- {label}: {v}")
    for label, choice in (site_inputs.get("score_sheet") or {}).items():
        if choice and choice != "(not set)":
            lines.append(f"- {label}: {choice}")
    if not lines:
        return ""
    return ("\nOPERATOR'S CONFIRMED SITE CHARACTERISTICS (ground truth from the operator — condition the forecast "
            "on these, they are not estimates):\n" + "\n".join(lines) + "\n")


def build_forecast_messages(lat: float, lon: float, factors: Dict[str, Any],
                            nearby_washes: Optional[List[dict]] = None,
                            site_inputs: Optional[Dict[str, Any]] = None,
                            web_sources: Optional[List[dict]] = None,
                            today_year: int = 2026) -> List[dict]:
    import json as _json
    obs = ""
    if nearby_washes:
        obs = (f"\nOBSERVED WASHES near the pin (Google Places ground truth, name + miles — the ≤3-mile ones are the "
               f"immediate competitive set):\n{_format_nearby_washes(nearby_washes)}\n")
    web = ""
    if web_sources:
        web = ("\nFRESH WEB-SEARCH RESULTS (retrieved just now — current ground truth on this market; prefer over "
               f"stale memory):\n{_format_web_sources(web_sources)}\n")
    user = (
        "5-YEAR WASH-VOLUME FORECAST + BACKDATING — JSON ONLY\n\n"
        f"LOCATION: {lat:.5f}, {lon:.5f}  ·  TODAY = mid-{today_year}\n"
        f"FACTOR SHEET (from the site-factor agent):\n{_json.dumps(factors, indent=1)[:4000]}\n"
        f"{obs}{web}{_format_site_inputs(site_inputs)}\n"
        "PART A — Estimate the NEW site's own washes/month for each operating year 1..5 (year averages, not month 1 "
        "vs month 12). lo/hi should bracket a realistic ~80% band; mid is your best guess. Respect the ramp shape "
        "and the capacity bound, and size the plateau from the demand and the observed competition — do NOT default "
        "to the low end.\n"
        f"PART B (BACKDATING) — for each hypothetical OPENING YEAR 2023, 2024, 2025, estimate this SAME site's "
        f"washes/month RIGHT NOW (mid-{today_year}). It would have ~{today_year}-open_year operating years behind it "
        "now (2023≈3 yrs → at/near plateau; 2024≈2 yrs; 2025≈1 yr → still ramping), so it sits further up the ramp. "
        "Give explicit weight to the COMPETITION AND MARKET AS THEY WERE that opening year versus now: fewer express "
        "rivals and a smaller-but-growing trade area earlier usually mean the site captured share first and holds a "
        "larger, stickier membership base today; a field that has since added tunnels is more contested now.\n"
        "Return STRICT JSON:\n"
        "{\n"
        '  "reasoning": str,                 // FIRST, before the numbers: the explicit step-by-step chain — addressable demand → capture share → mature plateau → ramp fractions → competition/demographics haircut\n'
        '  "years": [ {"year": 1, "washes_per_month_lo": int, "washes_per_month_mid": int, "washes_per_month_hi": int},\n'
        '             ... exactly 5 entries, years 1..5 ],\n'
        '  "backdated": [ {"open_year": 2023, "operating_years_now": int, "washes_per_month_now_lo": int, "washes_per_month_now_mid": int, "washes_per_month_now_hi": int, "competition_then_vs_now": str, "rationale": str},\n'
        '                 ... exactly 3 entries, open years 2023, 2024, 2025 ],  // this same site\'s CURRENT monthly volume had it opened that year\n'
        '  "membership_share_mature": str,   // e.g. "55–65%" — share of washes from unlimited members at maturity\n'
        '  "annual_market_growth_pct": str,  // e.g. "3–6%/yr" — how fast TOTAL wash demand in this trade area is growing per year\n'
        '  "plateau_washes_per_month": str,  // the mature level as a range, consistent with years 3-5\n'
        '  "basis": str                      // 2-3 short clauses: the main drivers of the level\n'
        "}"
    )
    return [{"role": "system", "content": FORECAST_SYSTEM}, {"role": "user", "content": user}]


# ─────────────────────────── Agent 3 — adversarial reviewer ───────────────────────────
REVIEW_SYSTEM = (
    "You are the adversarial reviewer of a car-wash volume forecast. Your job is to ATTACK it: check that the ramp "
    "shape is sane (year1 < year2 <= year3 ≈ plateau; no decline invented without a reason), that the plateau "
    "respects single-tunnel capacity (mature express sites do ~5,000–15,000 washes/month; >20,000 is extraordinary "
    "and needs extraordinary justification), that the level is consistent with the competition and demographics in "
    "the factor sheet (heavy express saturation should depress it, but an under-built market should NOT be lowballed "
    "either), and that lo <= mid <= hi in every year. ALSO attack the BACKDATING block: an earlier opening year must "
    "not wash LESS today than a later one (more operating time ⇒ at least as mature), the current-volume numbers must "
    "respect capacity, and the competition-then-vs-now story must be coherent. If it all survives, accept unchanged. "
    "If not, return the ADJUSTED five-year table and/or adjusted backdating fixing ONLY what is wrong, and say what "
    "you changed. STRICT JSON ONLY."
)


def build_review_messages(factors: Dict[str, Any], forecast: Dict[str, Any],
                          site_inputs: Optional[Dict[str, Any]] = None) -> List[dict]:
    import json as _json
    user = (
        "REVIEW THIS FORECAST — JSON ONLY\n\n"
        f"FACTOR SHEET:\n{_json.dumps(factors, indent=1)[:3000]}\n"
        f"{_format_site_inputs(site_inputs)}\n"
        f"FORECAST UNDER REVIEW:\n{_json.dumps(forecast, indent=1)[:2000]}\n\n"
        "Return STRICT JSON:\n"
        "{\n"
        '  "reasoning": str,                 // FIRST: your step-by-step attack — ramp sanity, capacity bound, competition-consistency, lo<=mid<=hi, and backdating monotonicity — and what you concluded\n'
        '  "verdict": "accept" | "adjust",\n'
        '  "issues": [str],                  // what you attacked and what failed ([] if clean)\n'
        '  "adjusted_years": [ ...same 5-entry shape as the forecast\'s "years"... ] | null,   // null when nothing to fix there\n'
        '  "adjusted_backdated": [ ...same 3-entry shape as the forecast\'s "backdated"... ] | null,   // null when nothing to fix there\n'
        '  "confidence": "High|Medium|Low"   // your confidence in the FINAL (accepted or adjusted) numbers\n'
        "}"
    )
    return [{"role": "system", "content": REVIEW_SYSTEM}, {"role": "user", "content": user}]


# ─────────────────────────── deterministic guardrails ───────────────────────────
def _guardrail_years(years: Any) -> List[Dict[str, int]]:
    """Whatever the agents said: exactly 5 rows, ints, 0 <= lo <= mid <= hi <= cap, and a
    non-decreasing mid through year 3 (the ramp — a 'decline' before maturity is a parse artifact)."""
    rows = {int(r.get("year")): r for r in (years or []) if isinstance(r, dict) and r.get("year")}
    out: List[Dict[str, int]] = []
    prev_mid = 0
    for y in (1, 2, 3, 4, 5):
        r = rows.get(y) or {}

        def _num(k):
            v = r.get(k)
            try:
                return max(0.0, float(str(v).replace(",", "")))
            except (TypeError, ValueError):
                return None

        lo, mid, hi = _num("washes_per_month_lo"), _num("washes_per_month_mid"), _num("washes_per_month_hi")
        mid = mid if mid is not None else (lo if lo is not None else hi) or 0.0
        lo = lo if lo is not None else mid * 0.7
        hi = hi if hi is not None else mid * 1.3
        lo, hi = min(lo, mid), max(hi, mid)
        if y <= 3:
            mid = max(mid, prev_mid)                       # ramp never goes backwards pre-plateau
            hi = max(hi, mid)
        prev_mid = mid
        out.append({"year": y,
                    "washes_per_month_lo": int(round(min(lo, MAX_WASHES_PER_MONTH))),
                    "washes_per_month_mid": int(round(min(mid, MAX_WASHES_PER_MONTH))),
                    "washes_per_month_hi": int(round(min(hi, MAX_WASHES_PER_MONTH)))})
    return out


def _guardrail_backdated(backdated: Any, years: Any = None, today_year: int = 2026) -> List[Dict[str, Any]]:
    """The backdating block, cleaned: exactly 3 rows for open years 2023/2024/2025, ints, 0 <= lo <= mid <= hi <= cap,
    operating_years_now recomputed deterministically, and a non-increasing mid from the earliest to the latest open
    year (more operating time ⇒ at least as much volume today). When the LLM omits or zeros a row, it FALLS BACK to
    the forward 5-year curve read at the age the site would be (opened N years ago ⇒ ~operating year N) so the table
    is never empty; the LLM's market-then-vs-now numbers win when present. Free-text notes pass through."""
    yrs = {int(y.get("year")): y for y in (years or []) if isinstance(y, dict) and y.get("year")}

    def _fwd(op_year: int):                                # forward-curve (lo, mid, hi) for a given operating year
        y = yrs.get(max(1, min(op_year, 5))) or {}

        def _v(k):
            try:
                return max(0.0, float(str(y.get(k)).replace(",", "")))
            except (TypeError, ValueError):
                return 0.0
        return _v("washes_per_month_lo"), _v("washes_per_month_mid"), _v("washes_per_month_hi")

    rows = {int(r.get("open_year")): r for r in (backdated or []) if isinstance(r, dict) and r.get("open_year")}
    open_years = [2023, 2024, 2025]
    parsed: Dict[int, dict] = {}
    for oy in open_years:
        r = rows.get(oy) or {}
        elapsed = max(0, today_year - oy)                  # opened N yrs ago ⇒ ~N operating years in today

        def _num(k):
            v = r.get(k)
            try:
                return max(0.0, float(str(v).replace(",", "")))
            except (TypeError, ValueError):
                return None

        lo, mid, hi = _num("washes_per_month_now_lo"), _num("washes_per_month_now_mid"), _num("washes_per_month_now_hi")
        if not mid:                                        # LLM gave nothing usable → forward-curve fallback at age N
            flo, fmid, fhi = _fwd(elapsed)
            lo, mid, hi = (lo or flo), (mid or fmid), (hi or fhi)
        mid = mid or 0.0
        lo = lo if lo else mid * 0.8
        hi = hi if hi else mid * 1.2
        lo, hi = min(lo, mid), max(hi, mid)
        parsed[oy] = dict(elapsed=elapsed, lo=lo, mid=mid, hi=hi,
                          competition_then_vs_now=(str(r.get("competition_then_vs_now")).strip()
                                                   if r.get("competition_then_vs_now") else None),
                          rationale=(str(r.get("rationale")).strip() if r.get("rationale") else None))
    floor = 0.0                                            # earlier open year washes >= later one → walk 2025→2023
    for oy in (2025, 2024, 2023):
        parsed[oy]["mid"] = max(parsed[oy]["mid"], floor)
        parsed[oy]["hi"] = max(parsed[oy]["hi"], parsed[oy]["mid"])
        floor = parsed[oy]["mid"]
    out = []
    for oy in open_years:
        p = parsed[oy]
        out.append({"open_year": oy, "operating_years_now": p["elapsed"],
                    "washes_per_month_now_lo": int(round(min(p["lo"], MAX_WASHES_PER_MONTH))),
                    "washes_per_month_now_mid": int(round(min(p["mid"], MAX_WASHES_PER_MONTH))),
                    "washes_per_month_now_hi": int(round(min(p["hi"], MAX_WASHES_PER_MONTH))),
                    "competition_then_vs_now": p["competition_then_vs_now"], "rationale": p["rationale"]})
    return out


_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def extract_traffic(factors: Dict[str, Any]) -> Optional[float]:
    """Midpoint of Agent 1's daily-traffic estimate (a string like '25,000–35,000'), for feeding Model 5
    SUPER's input-calibrated level when the user hasn't supplied a traffic count. None if unparseable."""
    for blk in ("radius_3mi", "radius_6mi"):
        est = (((factors or {}).get(blk) or {}).get("daily_traffic_main_road") or {}).get("estimate")
        nums = [float(m.group().replace(",", "")) for m in _NUM_RE.finditer(str(est or ""))]
        nums = [n for n in nums if 100 <= n <= 500_000]    # a plausible vehicles/day figure
        if nums:
            return float(sum(nums[:2]) / len(nums[:2]))
    return None


# ─────────────────────────── orchestrator ───────────────────────────
def llm_pin_forecast(lat: float, lon: float, *, nearby_washes: Optional[List[dict]] = None,
                     site_inputs: Optional[Dict[str, Any]] = None, use_web_search: bool = False,
                     today_year: int = 2026, address: Optional[str] = None,
                     backend: Optional[str] = None, max_tokens: int = 3000,
                     temperature: float = 0.3) -> Dict[str, Any]:
    """Run the LLM-only 3-agent chain for a pin with no local market: site factors (web-grounded) → a 5-year
    forecast PLUS a market-then-vs-now backdating (had it opened 2023/24/25, what is it washing now) → adversarial
    review. When `use_web_search` is on and a provider is configured, fresh web results for this location ground
    every agent. When `site_inputs` (pay stations, vacuum slots, lot type, measured traffic, proforma score-sheet
    picks) are supplied, the forecaster and reviewer condition on them as ground truth. Returns
    {factors, forecast, review, years, backdated, membership_share_mature, basis, reasoning,
     annual_market_growth_pct, web_sources, place, traffic_estimate_daily, backend}.
    Raises llm.LLMUnavailable if no backend answers."""
    web_sources: List[dict] = []
    place = (address or "").strip()
    if use_web_search:
        try:
            web_sources, place = websearch.gather_location_sources(lat, lon, address=address, radius_km=16)
        except Exception as e:                             # web search must never break the forecast
            logger.warning("Web-search grounding failed for (%.4f, %.4f), continuing without it: %s", lat, lon, e)

    f_text, used = llm_client.complete_cascade(build_factor_messages(lat, lon, nearby_washes, web_sources),
                                               backend=backend, max_tokens=max_tokens,
                                               temperature=temperature, json_mode=True)
    factors = _parse_json_lax(f_text)

    fc_text, _ = llm_client.complete_cascade(
        build_forecast_messages(lat, lon, factors, nearby_washes, site_inputs, web_sources, today_year),
        backend=backend, max_tokens=max_tokens, temperature=temperature, json_mode=True)
    forecast = _parse_json_lax(fc_text)

    review: Dict[str, Any] = {}
    try:                                                   # the reviewer is best-effort — a dead third call
        rv_text, _ = llm_client.complete_cascade(build_review_messages(factors, forecast, site_inputs),
                                                 backend=backend, max_tokens=max_tokens,
                                                 temperature=temperature, json_mode=True)
        review = _parse_json_lax(rv_text)
    except Exception as e:                                 # never loses the first two agents' work
        logger.warning("LLM forecast reviewer failed, keeping unreviewed forecast: %s", e)

    adjusting = str(review.get("verdict", "")).lower() == "adjust"
    years = review.get("adjusted_years") if adjusting else None
    back = review.get("adjusted_backdated") if adjusting else None
    final_years = _guardrail_years(years or forecast.get("years"))
    final_backdated = _guardrail_backdated(back or forecast.get("backdated"), final_years, today_year)
    logger.info("LLM pin forecast via %s backend (lat=%.4f lon=%.4f, reviewer=%s, web_sources=%d).",
                used, lat, lon, review.get("verdict") or "unavailable", len(web_sources))
    return {
        "factors": factors,
        "forecast": forecast,
        "review": review,
        "years": final_years,
        "backdated": final_backdated,
        "membership_share_mature": forecast.get("membership_share_mature"),
        "basis": forecast.get("basis"),
        "reasoning": forecast.get("reasoning"),
        "annual_market_growth_pct": forecast.get("annual_market_growth_pct"),
        "review_reasoning": review.get("reasoning"),
        "traffic_estimate_daily": extract_traffic(factors),
        "web_sources": web_sources,
        "place": place,
        "backend": used,
        "caveat": "LLM world-knowledge estimate (web-grounded) for a pin with no local Sonny's market — "
                  "not a grounded model forecast.",
    }
