"""
POC — raw-GPT market analysis from LOCATION ALONE (no operating data).

A deliberately standalone proof-of-concept, kept separate from the grounded Key-Insights pipeline
(graph.py / prompts.py / metrics.py). That pipeline feeds the model THIS market's actual monthly
numbers and forbids invention. This module does the opposite experiment: it hands the model NOTHING
but the goal, the context, the location (lat/lon + an optional address) and the trade-area radius,
then asks what market analysis it can produce for a NEW car-wash build purely from its own world
knowledge of that place.

The point is to see the ceiling of "data-free" analysis — what GPT already knows about a coordinate:
the metro, roads, demographics, competing brands, climate, growth. We give it context + goal + place,
not data.

Reuses ONLY the Azure transport in `llm.py` (the same call "Generate Key Insights" uses). Its prompt,
its request shape and its output are entirely its own — nothing here touches the grounded pipeline.

Public entry point:
    location_market_analysis(lat, lon, *, address=None, radius_km=20, backend=None)
        -> {"text": <markdown>, "backend": <backend used>, "prompt": <the user message sent>}
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from app.pnl_analysis.insights import llm as llm_client

logger = logging.getLogger(__name__)


# ─────────────────────────── prompt (this module's own — NOT shared) ───────────────────────────
SYSTEM_PROMPT = (
    "You are a senior site-selection analyst for EXPRESS TUNNEL car washes, advising an operator who is "
    "deciding whether to BUILD a NEW car wash at a specific location.\n\n"
    "You are given ONLY a location (latitude/longitude, and sometimes a rough address) plus a search radius "
    "that defines the local trade area. You have NO operating data — no revenue, no wash counts, no traffic "
    "counts — only your own general knowledge of the real world.\n\n"
    "Your job: produce the most useful market analysis you can for this site FROM LOCATION ALONE. First work "
    "out where this actually is, then reason from what you know about that place. Be specific and name real "
    "things you know — neighborhoods, major roads, nearby retail anchors, car-wash brands that operate in that "
    "metro, the typical climate. \n\n"
    "Discipline:\n"
    "- Clearly SEPARATE what you genuinely KNOW about this location from what you are INFERRING or assuming.\n"
    "- Never fabricate precise statistics. Give ranges and label them as estimates (e.g. '~25k–40k households, est.').\n"
    "- Mark a confidence level (High / Medium / Low) on each section.\n"
    "- If you cannot place the coordinates with confidence, say so plainly rather than guessing a city.\n"
    "- This is an exploratory read from knowledge only — be useful and concrete, but honest about uncertainty."
)


def build_location_messages(lat: float, lon: float, *, address: Optional[str] = None,
                            radius_km: float = 20) -> List[dict]:
    """Construct the chat request — context + goal + location + radius, and NO market data."""
    addr = (address or "").strip() or "(not provided — infer the place from the coordinates)"
    user = (
        "NEW CAR-WASH SITE — MARKET ANALYSIS REQUEST (location only, no operating data supplied)\n\n"
        "GOAL: Assess whether this is an attractive place to build a new express-tunnel car wash, and describe "
        "what the local market looks like — entirely from your knowledge of this location.\n\n"
        "LOCATION:\n"
        f"- Latitude, Longitude: {lat:.5f}, {lon:.5f}\n"
        f"- Approx address / description: {addr}\n"
        f"- Local trade-area radius: {radius_km:g} km (≈ {radius_km * 0.621:.0f} miles)\n\n"
        "Write a markdown report. Cover as many of the following as your knowledge of this place supports — and "
        "add anything else relevant. For each section, LEAD with your conclusion, then give the reasoning, then "
        "mark confidence (High/Medium/Low):\n\n"
        "1. **Location read** — what and where is this? Metro / city / neighborhood, the road context, and whether "
        "it reads as urban, suburban, exurban or rural.\n"
        "2. **Demographics & income** — population density, household income band, vehicle ownership, household "
        "growth trend in the trade area.\n"
        "3. **Traffic & access** — major roads/highways, commuter flow, visibility and ingress/egress, and the "
        "retail co-tenancy nearby (grocery anchors, big-box, QSR) that a car wash feeds off.\n"
        "4. **Competition** — car-wash brands and express tunnels likely operating within the radius, and how "
        "saturated the trade area feels for express washing.\n"
        "5. **Climate & seasonality** — weather/precipitation/road-salt patterns that drive or dampen wash "
        "frequency through the year.\n"
        "6. **Demand drivers & risks** — anything specific to this location that helps or hurts a new car wash.\n"
        "7. **Verdict** — is this an attractive build? Give a clear lean, your overall confidence, and the top 3 "
        "pieces of REAL data you'd pull next to confirm or kill the site.\n"
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


# ─────────────────────────── entry point ───────────────────────────
def location_market_analysis(lat: float, lon: float, *, address: Optional[str] = None,
                             radius_km: float = 20, backend: Optional[str] = None,
                             max_tokens: int = 2200, temperature: float = 0.5) -> Dict[str, Any]:
    """Ask the LLM for a location-only market read. Azure-first (cascades to local), like the
    Key-Insights button. Free-form markdown — no JSON mode — so we see the full range of what it can do.

    Raises `llm_client.LLMUnavailable` if no backend answers; the caller surfaces that honestly.
    """
    messages = build_location_messages(lat, lon, address=address, radius_km=radius_km)
    text, used = llm_client.complete_cascade(messages, backend=backend, max_tokens=max_tokens,
                                             temperature=temperature, json_mode=False)
    logger.info("Location POC analysis generated via %s backend.", used)
    return {"text": (text or "").strip(), "backend": used, "prompt": messages[-1]["content"]}


# ─────────────────────────── pollination (qualitative × quantitative) ───────────────────────────
# A THIRD, separate LLM call. It takes (A) the location-only qualitative commentary from the call above
# and (B) the grounded quantitative Key-Insights (from the LangGraph pipeline, computed off THIS market's
# real numbers), and fuses them: world-knowledge "why" against data "what's actually happening", with the
# DATA as the tie-breaker. Its own prompt, separate from both build_location_messages and the graph prompts.
POLLINATE_SYSTEM_PROMPT = (
    "You are the lead market analyst producing ONE combined investment read on whether to BUILD a new "
    "express-tunnel car wash at a location. You are handed two INDEPENDENT analyses of the same local market:\n"
    " (A) QUALITATIVE — a location / world-knowledge read written with NO access to operating data "
    "(demographics, roads, retail co-tenancy, competitors, climate, growth). These are CONTEXT and HYPOTHESES.\n"
    " (B) QUANTITATIVE — a grounded read computed ONLY from this market's ACTUAL monthly operating data "
    "(wash volumes, revenue, membership mix, ASPs, the most recent entrant's ramp). These are FACTS.\n"
    " (C) COMPETITIVE LANDSCAPE — an estimate, from world knowledge, of the TOTAL set of car washes (and express "
    "tunnels) operating in the trade area versus the client's own footprint. Use it to gauge SATURATION and the "
    "client's local share — how outnumbered their site(s) are. (C) is an estimate, like (A); never quote it as hard "
    "fact, and never describe the client's own data as incomplete — the other washes are competitors, not missing data.\n\n"
    "Your job is to POLLINATE them — fuse the 'why' of (A) with the 'what is actually happening' of (B):\n"
    "- Where (A) and (B) AGREE, say so plainly — that is your highest-confidence conclusion.\n"
    "- Where (B) CONTRADICTS or fails to support an (A) assumption, THE DATA WINS — call the surprise out "
    "explicitly (e.g. 'the trade area looks affluent and growing, but same-store membership is flat and ASP "
    "is below peak — demand is not converting the way the demographics suggest').\n"
    "- Use (A) to EXPLAIN (B)'s numbers where it can (e.g. a competitor named from world knowledge explains a "
    "cannibalization signal; a known commuter corridor explains strong retail volume).\n"
    "- Let (C) MODULATE the build verdict: high saturation / a large competitors-per-site multiple is a HEADWIND "
    "even when (A) and (B) look strong; low saturation is a TAILWIND / whitespace signal. (C) sets the competitive "
    "context the (B) numbers play out in, but it never overrides a fact from (B).\n"
    "- NEVER invent numbers. Quote hard figures only from (B). Treat (A) and (C) as context/estimates, never as fact. "
    "If (A) asserts something the data can neither confirm nor deny, label it as an untested assumption.\n"
)


def _format_quant_insights(quantitative: Any) -> str:
    """Render the grounded insights (a {'Washes','Revenue','ASPs'} markdown dict, or any text) into a block."""
    if isinstance(quantitative, dict):
        parts = [f"### {g}\n{quantitative[g]}" for g in ("Washes", "Revenue", "ASPs")
                 if quantitative.get(g)]
        return "\n\n".join(parts) if parts else "(no grounded quantitative insights available)"
    return str(quantitative or "(no grounded quantitative insights available)")


def _format_competition(competition: Any) -> str:
    """Render a compact (C) block from the dict competition_scale_analysis returns — phrased as competition,
    not data coverage. Returns a placeholder when no competition read is available."""
    if not isinstance(competition, dict) or not competition:
        return "(no competition-scale read available)"
    data = competition.get("data") or {}
    n_known = competition.get("known_count") or 0
    tot = data.get("estimated_total_carwashes") or {}
    exp = data.get("estimated_express_tunnels") or {}
    se = competition.get("scale_express") or {}
    parts = []
    parts.append(f"Client runs {n_known} express site(s) of their own here.")
    if exp:
        line = f"Estimated {exp.get('low', '?')}–{exp.get('high', '?')} express tunnels operate in the radius"
        if se:
            line += f" -> client faces ~{se.get('low', '?')}x–{se.get('high', '?')}x its own count in express rivals"
        parts.append(line + ".")
    if tot:
        parts.append(f"Estimated {tot.get('low', '?')}–{tot.get('high', '?')} car washes of all types in the radius.")
    share = data.get("estimated_client_share") or {}
    if share:
        parts.append(f"Est. client share of express tunnels: ~{share.get('low', '?')}–{share.get('high', '?')}%.")
    for label, key in [("Saturation", "saturation"), ("Competitive intensity", "competitive_intensity"),
                       ("Headroom", "headroom"), ("Client position", "client_position")]:
        if data.get(key):
            parts.append(f"{label}: {data[key]}.")
    rivals = [c.get("name") for c in (data.get("competitors") or []) if isinstance(c, dict) and c.get("name")]
    if rivals:
        parts.append("Named rivals: " + ", ".join(str(r) for r in rivals[:8]) + ".")
    return " ".join(parts)


def build_pollination_messages(qualitative_text: str, quantitative: Any, *, lat: float, lon: float,
                               radius_km: float = 20, competition: Any = None) -> List[dict]:
    """Construct the fusion request — the analyses verbatim plus the required combined-read shape. The optional
    competition read (C) folds in the competitive-saturation dimension."""
    user = (
        "NEW CAR-WASH SITE — POLLINATED MARKET READ (fuse the world-knowledge commentary with the real data)\n\n"
        f"LOCATION: {lat:.5f}, {lon:.5f} · local trade-area radius {radius_km:g} km.\n\n"
        "You are given the independent analyses below. Fuse them into a single decision-useful read.\n\n"
        "══════ (A) QUALITATIVE — location / world-knowledge read (NO operating data) ══════\n"
        f"{(qualitative_text or '').strip() or '(no qualitative commentary available)'}\n\n"
        "══════ (B) QUANTITATIVE — grounded in THIS market's actual operating data ══════\n"
        f"{_format_quant_insights(quantitative)}\n\n"
        "══════ (C) COMPETITIVE SATURATION — client footprint vs total trade-area competition (estimate) ══════\n"
        f"{_format_competition(competition)}\n\n"
        "Now write the combined read as markdown with EXACTLY these sections (lead each with the conclusion):\n"
        "## Combined market read\n"
        "**Where the story and the data agree** — the points (A) and (B) corroborate; highest confidence.\n\n"
        "**Where the data overrides the location story** — every place (B) contradicts or fails to support (A); "
        "name the figure from (B) that settles it.\n\n"
        "**What the location context adds** — the 'why' (A) supplies that the numbers alone can't explain.\n\n"
        "**Competitive saturation** — how crowded the trade area is for express tunnels and how outnumbered the "
        "client's site(s) are (cite (C)'s scale multiple and named rivals); say whether competition is a headwind "
        "or whitespace.\n\n"
        "**Combined verdict** — build / pass / conditional, the overall confidence, and the single condition that "
        "would change the call. Cite (B)'s numbers; weigh (C)'s saturation alongside (A)/(B); keep (A) as context."
    )
    return [{"role": "system", "content": POLLINATE_SYSTEM_PROMPT}, {"role": "user", "content": user}]


def pollinate_analysis(qualitative_text: str, quantitative: Any, *, lat: float, lon: float,
                       radius_km: float = 20, competition: Any = None, backend: Optional[str] = None,
                       max_tokens: int = 2000, temperature: float = 0.4) -> Dict[str, Any]:
    """Fuse a location-only qualitative commentary (A) with the grounded quantitative insights (B) — and, when
    supplied, the competitive-saturation read (C) — into one combined read, via a fresh LLM call. Azure-first
    (cascades to local). Raises LLMUnavailable on failure."""
    messages = build_pollination_messages(qualitative_text, quantitative, lat=lat, lon=lon, radius_km=radius_km,
                                          competition=competition)
    text, used = llm_client.complete_cascade(messages, backend=backend, max_tokens=max_tokens,
                                             temperature=temperature, json_mode=False)
    logger.info("Pollinated analysis generated via %s backend.", used)
    return {"text": (text or "").strip(), "backend": used, "prompt": messages[-1]["content"]}


# ─────────────────── competition scale (client footprint vs total competitive landscape — saturation) ───────────────────
# A FOURTH, separate call. The operator's site list is their OWN PORTFOLIO — the sites THIS client actually owns and
# runs — and it is complete for what it represents (an operator only tracks the sites they operate). Every OTHER car
# wash in the trade area is simply a COMPETITOR the client has no operating data for, which is the normal state of the
# world, NOT a gap. This asks the LLM, from world knowledge, how many car washes ACTUALLY operate within the radius
# (total + express tunnels), names the rival brands it expects, and recognises which of the CLIENT'S own sites it
# knows. The UI turns that into a SATURATION MULTIPLE (LLM total ÷ the client's own count) so we can say "the client
# faces ~Nx its own count in competitors". Estimates, not ground truth — labelled as such. STRICT JSON output.
COMPETITION_SYSTEM_PROMPT = (
    "You are a car-wash market analyst measuring COMPETITIVE SATURATION around a location for an express-tunnel "
    "operator. You are given a location, a trade-area radius, and the client's OWN PORTFOLIO of car washes inside "
    "that radius — i.e. the site(s) THIS operator actually owns and runs. This is the client's footprint, and it is "
    "COMPLETE for what it represents: an operator only tracks and considers the sites they themselves operate. Every "
    "OTHER car wash in the trade area is a COMPETITOR the client does not (and would not) have operating data for — "
    "that is the normal, expected state of the world, NOT a gap in the data.\n\n"
    "Your job, from your own real-world knowledge, is to size the TOTAL competitive landscape in the radius so the "
    "client can see how outnumbered their footprint is. Specifically: estimate how many car washes ACTUALLY operate "
    "within the radius (all types, and express conveyor TUNNELS specifically — the client's segment), name the rival "
    "brands/operators you would expect to find there, identify which of the client's own listed sites you recognize, "
    "and characterise how crowded the trade area is for express washing. The headline measure is SATURATION / local "
    "market share: the client owns a known handful of sites; the total competitive set is some larger number; the "
    "ratio tells the client how many competitors they face per site they run.\n\n"
    "Framing rules — be strict about these:\n"
    "- Treat the client's listed sites as their deliberate, complete portfolio, never as an incomplete or partial dataset.\n"
    "- Do NOT use the words 'missing', 'incomplete', 'coverage gap', 'not seeing', or 'what we don't have'. The other "
    "washes are not missing data — they are competitors.\n"
    "- Frame the count difference as competitive pressure / saturation / share-of-the-trade-area, e.g. 'the client "
    "runs N of an estimated M express tunnels here, so they face roughly (M-N) express rivals' or 'the client's site "
    "is outnumbered ~Kx by competing tunnels'.\n\n"
    "Estimation discipline: give counts as low–high RANGES and treat them as estimates; name only real brands you "
    "genuinely associate with that metro; never invent exact street addresses; mark your confidence in placing the "
    "location. Respond with STRICT JSON ONLY — no prose, no markdown fences."
)


def build_competition_messages(lat: float, lon: float, *, known_sites: Optional[List[str]] = None,
                               address: Optional[str] = None, radius_km: float = 20) -> List[dict]:
    """Construct the competitive-saturation JSON request — location, radius, and the client's OWN sites."""
    addr = (address or "").strip() or "(not provided — infer the place from the coordinates)"
    known = [str(s) for s in (known_sites or []) if str(s).strip()]
    known_block = ("; ".join(known)) if known else "(the client has no sites of their own in this radius)"
    user = (
        "COMPETITIVE SATURATION ESTIMATE — JSON ONLY\n\n"
        "LOCATION:\n"
        f"- Latitude, Longitude: {lat:.5f}, {lon:.5f}\n"
        f"- Approx address / description: {addr}\n"
        f"- Trade-area radius: {radius_km:g} km (≈ {radius_km * 0.621:.0f} miles)\n\n"
        f"THE CLIENT'S OWN CAR WASHES in this radius — the site(s) this operator runs ({len(known)} owned): {known_block}\n\n"
        "From your knowledge of this place, estimate the TOTAL competitive set in the radius — every car wash "
        "competing with the client, plus the express-tunnel segment specifically — and give a full competitive read "
        "so we can size how outnumbered the client's footprint is and what kind of competition it is. Return STRICT "
        "JSON with exactly these keys:\n"
        "{\n"
        '  "estimated_total_carwashes": {"low": int, "high": int},   // every car wash: tunnels, in-bay automatics, self-serve\n'
        '  "estimated_express_tunnels": {"low": int, "high": int},   // express conveyor tunnels only (the client\'s segment) — total in the trade area, the client\'s own count included\n'
        '  "competitors": [ {"name": str, "type": str, "scale": str, "threat": str, "note": str} ],  // every rival you expect. type = "Express tunnel"|"In-bay automatic"|"Self-serve"|"Other"; scale = "National"|"Regional"|"Local/Independent"; threat = "High"|"Medium"|"Low" to the client\n'
        '  "client_sites_recognized": [str],   // which of the CLIENT\'S OWN listed sites you recognize (names), [] if none\n'
        '  "client_position": str,             // 1-2 sentences: the client\'s competitive standing here vs these rivals\n'
        '  "estimated_client_share": {"low": int, "high": int},  // client\'s share of the EXPRESS tunnels in the radius, as a PERCENT (their count ÷ total express)\n'
        '  "competitive_intensity": str,       // "Low"|"Medium"|"High" + why (price wars, aggressive promos, membership churn vs sleepy market)\n'
        '  "pricing_positioning": str,         // typical unlimited-plan $/month and per-wash retail price norms you expect in this market\n'
        '  "expansion_signals": str,           // chains expanding here / new builds or M&A expected nearby; "" if none known\n'
        '  "headroom": str,                    // is there room for another express tunnel here? whitespace vs saturated, and why\n'
        '  "saturation": str,                  // "Low"|"Medium"|"High" for express SUPPLY density + one-line why\n'
        '  "confidence": str,                  // "High"|"Medium"|"Low" for placing the location & the counts\n'
        '  "reasoning": str                    // 2-3 sentences on how you sized the competitive set; describe rivals as competition, never as missing data\n'
        "}\n"
    )
    return [{"role": "system", "content": COMPETITION_SYSTEM_PROMPT}, {"role": "user", "content": user}]


def _parse_json_lax(text: str) -> Dict[str, Any]:
    """Parse the model's JSON, tolerating stray markdown fences or leading/trailing prose."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", t).strip()
    try:
        return json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}", t, re.DOTALL)        # first {...} block
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {}


def _mid(rng: Any) -> Optional[float]:
    """Midpoint of a {'low','high'} estimate (or a bare number); None if unparseable."""
    if isinstance(rng, dict):
        lo, hi = rng.get("low"), rng.get("high")
        vals = [float(v) for v in (lo, hi) if isinstance(v, (int, float))]
        return sum(vals) / len(vals) if vals else None
    return float(rng) if isinstance(rng, (int, float)) else None


def competition_scale_analysis(lat: float, lon: float, *, known_sites: Optional[List[str]] = None,
                               address: Optional[str] = None, radius_km: float = 20,
                               backend: Optional[str] = None, max_tokens: int = 1900,
                               temperature: float = 0.3) -> Dict[str, Any]:
    """Ask the LLM how many car washes really operate near this pin and size the client's competitive saturation.

    Returns the parsed estimate, the client's own site count, and the implied SATURATION MULTIPLE (LLM total ÷ the
    client's own count = competitors per client site) for both the express-tunnel segment and all car washes.
    Estimates only — labelled as such in the UI. Raises `llm_client.LLMUnavailable` if no backend answers."""
    known = [str(s) for s in (known_sites or []) if str(s).strip()]
    messages = build_competition_messages(lat, lon, known_sites=known, address=address, radius_km=radius_km)
    text, used = llm_client.complete_cascade(messages, backend=backend, max_tokens=max_tokens,
                                             temperature=temperature, json_mode=True)
    data = _parse_json_lax(text)
    n_known = len(known)
    exp_mid = _mid(data.get("estimated_express_tunnels"))
    tot_mid = _mid(data.get("estimated_total_carwashes"))

    def _scale(rng):                                   # multiple vs our count, using the estimate's low & high
        if not isinstance(rng, dict) or n_known <= 0:
            return None
        out = {}
        for k in ("low", "high"):
            v = rng.get(k)
            if isinstance(v, (int, float)):
                out[k] = round(float(v) / n_known, 1)
        return out or None

    logger.info("Competition-scale estimate via %s backend (known=%d).", used, n_known)
    return {
        "data": data,
        "known_count": n_known,
        "known_sites": known,
        "express_mid": exp_mid,
        "total_mid": tot_mid,
        "scale_express": _scale(data.get("estimated_express_tunnels")),
        "scale_total": _scale(data.get("estimated_total_carwashes")),
        "backend": used,
        "prompt": messages[-1]["content"],
        "raw": (text or "").strip(),
    }
