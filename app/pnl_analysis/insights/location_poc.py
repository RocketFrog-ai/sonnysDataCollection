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
from app.pnl_analysis.insights import websearch

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
    "things you know — neighborhoods, major roads, nearby retail anchors, the typical climate. Do NOT analyse "
    "competitors or car-wash saturation — that is handled in a separate Competition Coverage summary.\n\n"
    "Discipline:\n"
    "- DATA-FIRST, FEW ADJECTIVES: lead each point with the concrete fact — a place, road, brand, or number — not a descriptor. Cut hype adjectives ('great', 'strong', 'excellent', 'prime'); let the fact carry the point. One fact plus a short reason per bullet; no long paragraphs. Do NOT repeat the same point across sections.\n"
    "- Clearly SEPARATE what you genuinely KNOW about this location from what you are INFERRING or assuming.\n"
    "- Never fabricate precise statistics. Give ranges and label them as estimates (e.g. '~25k–40k households, est.').\n"
    "- Mark a confidence level (High / Medium / Low) on each section.\n"
    "- If you cannot place the coordinates with confidence, say so plainly rather than guessing a city.\n"
    "- This is an exploratory read from knowledge only — be useful and concrete, but honest about uncertainty."
)


def _format_web_sources(sources: List[dict]) -> str:
    """Number the retrieved web sources for the prompt: '[n] Title — URL' + a trimmed snippet."""
    out = []
    for i, s in enumerate(sources, 1):
        title = (s.get("title") or s.get("url") or "source").strip()
        url = (s.get("url") or "").strip()
        snippet = (s.get("content") or "").strip().replace("\n", " ")
        if len(snippet) > 320:
            snippet = snippet[:320] + "…"
        out.append(f"[{i}] {title} — {url}" + (f"\n    {snippet}" if snippet else ""))
    return "\n".join(out)


def build_location_messages(lat: float, lon: float, *, address: Optional[str] = None,
                            radius_km: float = 20, web_sources: Optional[List[dict]] = None) -> List[dict]:
    """Construct the chat request — context + goal + location + radius, and NO market data. When `web_sources`
    are supplied (fresh web-search results), they are injected as citable ground truth and the model is told to
    cite them by number and list its sources."""
    addr = (address or "").strip() or "(not provided — infer the place from the coordinates)"
    system = SYSTEM_PROMPT
    web_section, cite_rule = "", ""
    if web_sources:
        web_section = (
            "\nFRESH WEB SEARCH RESULTS (retrieved just now for this exact location — treat as current, "
            "authoritative context; prefer them over static memory when they conflict):\n"
            f"{_format_web_sources(web_sources)}\n"
        )
        cite_rule = (
            "\n- Ground concrete claims (demographics, income, roads, named competitors, growth, new builds) in the "
            "WEB SEARCH RESULTS above and cite the source inline as [n] using its number. End the report with a "
            "'## Sources' section listing every [n] you cited as '[n] Title — URL'. Never cite a source you did not use.\n"
        )
        system = SYSTEM_PROMPT + (
            "\n\nYou have ALSO been handed fresh web-search results for this exact location. Use them as current "
            "ground truth, prefer them over static memory when they conflict, and cite them by number."
        )
    user = (
        "NEW CAR-WASH SITE — MARKET ANALYSIS REQUEST (location only, no operating data supplied)\n\n"
        "GOAL: Assess what the local market looks like for a new express-tunnel car wash — from your knowledge of "
        "this location and any web results provided.\n\n"
        "LOCATION:\n"
        f"- Latitude, Longitude: {lat:.5f}, {lon:.5f}\n"
        f"- Approx address / description: {addr}\n"
        f"- Local trade-area radius: {radius_km:g} km (≈ {radius_km * 0.621:.0f} miles)\n"
        f"{web_section}\n"
        "FORMATTING — return clean, react-markdown-compatible Markdown and nothing else:\n"
        "- Start with a single `# Local Market Analysis` title.\n"
        "- Use a `## ` heading for each numbered section below (drop the number, keep the name).\n"
        "- Under each heading, LEAD with a one-line **bold** takeaway, then supporting detail as `- ` bullet points.\n"
        "- **Bold** the key facts inside bullets (numbers, road names, brands, confidence). End each section's takeaway "
        "line with its confidence in bold, e.g. **(Confidence: Medium)**.\n"
        "- Do NOT use raw HTML or Markdown tables. Keep it scannable — short, number-led bullets, no long paragraphs; "
        "do not repeat a fact you already stated in another section.\n\n"
        "Cover as many of the following sections as your knowledge of this place supports (add anything else relevant):\n\n"
        "1. **Location read** — what and where is this? Metro / city / neighborhood, the road context, and whether "
        "it reads as urban, suburban, exurban or rural.\n"
        "2. **Demographics & income** — population density, household income band, vehicle ownership, household "
        "growth trend in the trade area.\n"
        "3. **Traffic & access** — major roads/highways, commuter flow, visibility and ingress/egress, and the "
        "retail co-tenancy nearby (grocery anchors, big-box, QSR) that a car wash feeds off.\n"
        "4. **Climate & seasonality** — weather/precipitation/road-salt patterns that drive or dampen wash "
        "frequency through the year.\n"
        "5. **Demand drivers & risks** — anything specific to this location that helps or hurts a new car wash.\n\n"
        "IMPORTANT: This is the market-context read ONLY. Do NOT cover COMPETITION or car-wash saturation — that is "
        "produced separately in a dedicated Competition Coverage summary, so leave competitors out entirely here. Do NOT "
        "give a build/pass verdict, recommendation, overall score, or a 'next steps / questions to confirm' section — the "
        "verdict is produced separately once this is combined with the operating data and the competitive coverage. End "
        "after the sections above; add nothing further.\n"
        f"{cite_rule}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ─────────────────────────── entry point ───────────────────────────
def location_market_analysis(lat: float, lon: float, *, address: Optional[str] = None,
                             radius_km: float = 20, backend: Optional[str] = None,
                             max_tokens: int = 2200, temperature: float = 0.5,
                             use_web_search: bool = False) -> Dict[str, Any]:
    """Ask the LLM for a location-only market read. Azure-first (cascades to local), like the
    Key-Insights button. Free-form markdown — no JSON mode — so we see the full range of what it can do.

    When `use_web_search` is True and a search provider is configured, fresh web results for this location
    are retrieved and fed in as citable context; the returned dict carries the `sources` (title/url) so the
    caller can render clickable links backing the commentary. Falls back to un-grounded output if search is
    unavailable or fails.

    Raises `llm_client.LLMUnavailable` if no backend answers; the caller surfaces that honestly.
    """
    sources: List[dict] = []
    place = (address or "").strip()
    if use_web_search:
        try:
            sources, place = websearch.gather_location_sources(lat, lon, address=address, radius_km=radius_km)
        except Exception as e:                                    # search must never break the analysis
            logger.warning("Web-search grounding failed, continuing without it: %s", e)
            sources = []
    messages = build_location_messages(lat, lon, address=address, radius_km=radius_km, web_sources=sources)
    text, used = llm_client.complete_cascade(messages, backend=backend, max_tokens=max_tokens,
                                             temperature=temperature, json_mode=False)
    logger.info("Location POC analysis generated via %s backend (web_sources=%d).", used, len(sources))
    return {"text": (text or "").strip(), "backend": used, "prompt": messages[-1]["content"],
            "sources": sources, "web_used": bool(sources), "place": place}


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
    " (C) COMPETITIVE LANDSCAPE — an estimate, from world knowledge, of the EXPRESS/CONVEYOR-TUNNEL car washes "
    "operating in the trade area (the client's segment — other wash types are out of scope) versus the client's own "
    "footprint. Use it to gauge SATURATION and the "
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


_NO_COMPETITION = "(no competition-scale read available)"


def _format_competition(competition: Any) -> str:
    """Render a compact (C) block for the pollination prompt. Accepts, in order of preference:
      • the `{"table","summary"}` response the /insights/competition route returns → use its markdown `summary`;
      • a JSON string of either that response or the bare model JSON → parsed and re-dispatched;
      • an already-formatted markdown/prose string → used verbatim;
      • the rich dict `competition_scale_analysis` returns (has a `data` key) → rendered here;
    so it works whether the caller passes the pre-computed API response (the new flow) or the raw result dict
    (the Streamlit flow). Returns a placeholder when no competition read is available."""
    if not competition:
        return _NO_COMPETITION
    if isinstance(competition, str):
        s = competition.strip()
        if not s:
            return _NO_COMPETITION
        try:                                          # a JSON string? parse and re-dispatch
            return _format_competition(json.loads(s))
        except Exception:
            return s                                  # already a markdown/prose summary
    if not isinstance(competition, dict) or not competition:
        return _NO_COMPETITION
    if competition.get("summary"):                    # the {"table","summary"} route response
        return str(competition["summary"])
    if "data" not in competition and ("estimated_total_carwashes" in competition or "competitors" in competition):
        competition = {"data": competition,           # bare model JSON — wrap it as a rich result
                       "known_count": len(competition.get("client_sites_recognized") or [])}
    data = competition.get("data") or {}
    n_known = competition.get("known_count") or 0
    exp = data.get("estimated_express_tunnels") or {}
    se = competition.get("scale_express") or {}
    parts = []                                        # express/tunnel segment only — the all-types count stays out
    parts.append(f"Client runs {n_known} express site(s) of their own here.")
    if exp:
        line = f"Estimated {exp.get('low', '?')}–{exp.get('high', '?')} express tunnels operate in the radius"
        if se:
            line += f" -> client faces ~{se.get('low', '?')}x–{se.get('high', '?')}x its own count in express rivals"
        parts.append(line + ".")
    share = data.get("estimated_client_share") or {}
    if share:
        parts.append(f"Est. client share of express tunnels: ~{share.get('low', '?')}–{share.get('high', '?')}%.")
    for label, key in [("Saturation", "saturation"), ("Competitive intensity", "competitive_intensity"),
                       ("Headroom", "headroom"), ("Client position", "client_position")]:
        if data.get(key):
            parts.append(f"{label}: {data[key]}.")
    rivals = [c.get("name") for c in (data.get("competitors") or [])
              if isinstance(c, dict) and c.get("name") and not _non_express(c)]
    if rivals:
        parts.append("Named express rivals: " + ", ".join(str(r) for r in rivals[:8]) + ".")
    nearby = competition.get("nearby_washes") or []
    if nearby:
        exp_names = [str(w.get("name")) for w in nearby if w.get("express_likely") and w.get("name")]
        tag = (", incl. express-tagged " + ", ".join(exp_names[:6])) if exp_names else ""
        parts.append(f"Grounded on {len(nearby)} real washes of all types observed via Google Places nearby{tag}.")
    return " ".join(parts)


def build_pollination_messages(qualitative_text: str, quantitative: Any, *, lat: float, lon: float,
                               radius_km: float = 20, competition: Any = None) -> List[dict]:
    """Construct the fusion request — the analyses verbatim plus the required combined-read shape. The optional
    competition read (C) folds in the competitive-saturation dimension."""
    user = (
        "NEW CAR-WASH SITE — FINAL CONSOLIDATED SUMMARY (fuse the three analyses below into ONE short read)\n\n"
        f"LOCATION: {lat:.5f}, {lon:.5f} · local trade-area radius {radius_km:g} km.\n\n"
        "You are given three analyses of the same local market: the Key Insights from the operating data (B), the "
        "Local Market Analysis (A), and the Competition Coverage (C). Distil ALL THREE into ONE brief, decision-"
        "useful summary that ends in a clear verdict and recommendation.\n\n"
        "══════ (A) LOCAL MARKET ANALYSIS — location / world-knowledge read (NO operating data) ══════\n"
        f"{(qualitative_text or '').strip() or '(no local market analysis available)'}\n\n"
        "══════ (B) KEY INSIGHTS — grounded in THIS market's actual operating data / plots ══════\n"
        f"{_format_quant_insights(quantitative)}\n\n"
        "══════ (C) COMPETITION COVERAGE — client footprint vs total trade-area competition (estimate) ══════\n"
        f"{_format_competition(competition)}\n\n"
        "Now write the FINAL summary as clean, react-markdown-compatible Markdown, and NOTHING else.\n\n"
        "RULES:\n"
        "- Write in ONE unified analyst voice. Do NOT label statements with (A)/(B)/(C) or mention 'the location "
        "analysis says' — just state the conclusion, weaving the three sources together seamlessly.\n"
        "- Be CONCISE — the whole thing must fit in roughly 90–130 words. No long paragraphs; short, number-led bullets only.\n"
        "- DATA-FIRST, FEW ADJECTIVES: lead each bullet with a figure or fact, not a descriptor; cut hype adjectives "
        "('strong', 'healthy', 'attractive'). One point per bullet, each adding something new — never repeat a figure.\n"
        "- When the data (B) and the location story (A) disagree, trust the data. Quote only hard numbers that "
        "appear in (B); treat (A) and (C) as context/estimates. Never invent figures.\n"
        "- **Bold** the few key figures and the verdict call itself.\n\n"
        "USE EXACTLY THIS STRUCTURE:\n"
        "## Summary\n"
        "One **bold** sentence with the overall takeaway, then 3–4 short `- ` bullets covering demand & the market's "
        "trajectory, the business/pricing model, and competitive saturation. Keep each bullet to one line.\n\n"
        "## Verdict\n"
        "One line: **Build**, **Pass**, or **Conditional** + a half-sentence why.\n"
        "**Recommendation:** one concise line — the action to take and the single condition that would change the call."
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
# runs — and it is complete for what it represents (an operator only tracks the sites they operate). Every OTHER
# express tunnel in the trade area is simply a COMPETITOR the client has no operating data for, which is the normal
# state of the world, NOT a gap. This asks the LLM, from world knowledge, how many EXPRESS/CONVEYOR-TUNNEL washes
# ACTUALLY operate within the radius (an all-types count is kept as background calibration only), names the rival
# express brands it expects, and recognises which of the CLIENT'S own sites it knows. The UI turns that into a
# SATURATION MULTIPLE (LLM total ÷ the client's own count) so we can say "the client faces ~Nx its own count in
# competitors". Estimates, not ground truth — labelled as such. STRICT JSON output.
COMPETITION_SYSTEM_PROMPT = (
    "You are a car-wash market analyst measuring COMPETITIVE SATURATION around a location for an EXPRESS-TUNNEL "
    "operator. Your entire read is scoped to the EXPRESS / CONVEYOR-TUNNEL segment — the client's segment. You are "
    "given a location, a trade-area radius, and the client's OWN PORTFOLIO of car washes inside "
    "that radius — i.e. the site(s) THIS operator actually owns and runs. This is the client's footprint, and it is "
    "COMPLETE for what it represents: an operator only tracks and considers the sites they themselves operate. Every "
    "OTHER express tunnel in the trade area is a COMPETITOR the client does not (and would not) have operating data "
    "for — that is the normal, expected state of the world, NOT a gap in the data.\n\n"
    "Your job, from your own real-world knowledge, is to size the EXPRESS-TUNNEL competitive landscape in the radius "
    "so the client can see how outnumbered their footprint is. Specifically: estimate how many express conveyor "
    "tunnels ACTUALLY operate within the radius, name the rival express brands/operators you would expect to find "
    "there, identify which of the client's own listed sites you recognize, and characterise how crowded the trade "
    "area is for express washing. The headline measure is SATURATION / local market share: the client owns a known "
    "handful of sites; the express competitive set is some larger number; the ratio tells the client how many express "
    "rivals they face per site they run.\n\n"
    "SEGMENT SCOPE — the express/tunnel rule, be strict about it:\n"
    "- Everything you present — the named `competitors`, saturation, competitive intensity, pricing, headroom and the "
    "client's position — covers EXPRESS / CONVEYOR-TUNNEL car washes ONLY. Self-serve bays, in-bay automatics, hand "
    "washes, detailing shops and gas-station washes are NOT the client's segment: never name them as competitors and "
    "never let them drive the read. They may be counted ONLY inside `estimated_total_carwashes` (background context).\n"
    "- Observed nearby washes handed to you can be of ANY type: classify each from its name, its tag and your brand "
    "knowledge; keep the express/tunnel ones in the analysis and set the rest aside (they still count toward the "
    "all-types total, nothing else).\n\n"
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


def _format_nearby_washes(nearby_washes: List[dict]) -> str:
    """Render the real Google-Places washes as a name + distance ground-truth list for the prompt.
    Washes whose name/type carries an express/tunnel keyword are tagged so the model can scope its
    express-only read faster — a hint, not a verdict (untagged washes may still be express tunnels)."""
    lines = []
    for w in nearby_washes[:25]:
        nm = (w.get("name") or "?").strip()
        d = w.get("distance_miles")
        line = f"- {nm}" + (f" — {float(d):.1f} mi" if isinstance(d, (int, float)) else "")
        if w.get("express_likely"):
            line += " · likely express/tunnel"
        lines.append(line)
    return "\n".join(lines)


def build_competition_messages(lat: float, lon: float, *, known_sites: Optional[List[str]] = None,
                               address: Optional[str] = None, radius_km: float = 20,
                               nearby_washes: Optional[List[dict]] = None) -> List[dict]:
    """Construct the competitive-saturation JSON request — location, radius, the client's OWN sites, and
    (when supplied) the REAL nearby car washes observed via Google Places (name + distance + express tag) as
    ground truth the estimate must be anchored to. The read is scoped to the EXPRESS/TUNNEL segment: the
    observed list may hold any wash type, but only express/conveyor tunnels may be named/analysed."""
    addr = (address or "").strip() or "(not provided — infer the place from the coordinates)"
    known = [str(s) for s in (known_sites or []) if str(s).strip()]
    known_block = ("; ".join(known)) if known else "(the client has no sites of their own in this radius)"
    observed_block = ""
    if nearby_washes:
        observed_block = (
            f"OBSERVED CAR WASHES near this pin from Google Places (GROUND TRUTH — real washes actually operating "
            f"within ~11 miles, name + distance; {len(nearby_washes)} found, nearest first; the list can contain ANY "
            f"wash type — entries whose name suggests the client's segment are tagged '· likely express/tunnel'):\n"
            f"{_format_nearby_washes(nearby_washes)}\n"
            "Anchor your estimate to this observed set: classify each observed wash as express/conveyor tunnel or not "
            "from its name, its tag and your brand knowledge. Your express-tunnel count must be AT LEAST the number "
            "of observed washes you classify as express, and `competitors` must include those express washes by name. "
            "NEVER list an observed wash you classify as self-serve / in-bay / hand-wash / detailing / gas-station in "
            "`competitors` — it counts only toward the all-types total. You may extrapolate beyond the Places "
            "returns, but never contradict or fall below this observed set.\n\n"
        )
    know_src = "your knowledge of this place and the observed washes below" if nearby_washes else "your knowledge of this place"
    user = (
        "COMPETITIVE SATURATION ESTIMATE — EXPRESS/TUNNEL SEGMENT ONLY — JSON ONLY\n\n"
        "LOCATION:\n"
        f"- Latitude, Longitude: {lat:.5f}, {lon:.5f}\n"
        f"- Approx address / description: {addr}\n"
        f"- Trade-area radius: {radius_km:g} km (≈ {radius_km * 0.621:.0f} miles)\n\n"
        f"THE CLIENT'S OWN CAR WASHES in this radius — the site(s) this operator runs ({len(known)} owned): {known_block}\n\n"
        f"{observed_block}"
        f"From {know_src}, estimate the EXPRESS/CONVEYOR-TUNNEL competitive set in the radius — the segment the "
        "client competes in — and give a full competitive read so we can size how outnumbered the client's footprint "
        "is and what kind of express competition it is. Every named competitor and every qualitative field covers "
        "express tunnels ONLY; the all-types count is background context and nothing more. Return STRICT "
        "JSON with exactly these keys:\n"
        "{\n"
        '  "estimated_total_carwashes": {"low": int, "high": int},   // background context ONLY: every wash of any type (tunnels, in-bay, self-serve) — never presented, never a competitor source\n'
        '  "estimated_express_tunnels": {"low": int, "high": int},   // express conveyor tunnels only (the client\'s segment) — total in the trade area, the client\'s own count included\n'
        '  "competitors": [ {"name": str, "type": "Express tunnel", "scale": str, "threat": str, "note": str} ],  // EXPRESS/TUNNEL rivals ONLY — never list a self-serve, in-bay automatic, hand-wash, detailing or gas-station wash. scale = "National"|"Regional"|"Local/Independent"; threat = "High"|"Medium"|"Low" to the client\n'
        '  "client_sites_recognized": [str],   // which of the CLIENT\'S OWN listed sites you recognize (names), [] if none\n'
        '  "client_position": str,             // 1-2 sentences: the client\'s competitive standing here vs these express rivals\n'
        '  "estimated_client_share": {"low": int, "high": int},  // client\'s share of the EXPRESS tunnels in the radius, as a PERCENT (their count ÷ total express)\n'
        '  "competitive_intensity": str,       // "Low"|"Medium"|"High" + why, among express tunnels (price wars, aggressive promos, membership churn vs sleepy market)\n'
        '  "pricing_positioning": str,         // typical express unlimited-plan $/month and per-wash retail price norms you expect in this market\n'
        '  "expansion_signals": str,           // express chains expanding here / new tunnel builds or M&A expected nearby; "" if none known\n'
        '  "headroom": str,                    // is there room for another express tunnel here? whitespace vs saturated, and why\n'
        '  "saturation": str,                  // "Low"|"Medium"|"High" for express SUPPLY density + one-line why\n'
        '  "confidence": str,                  // "High"|"Medium"|"Low" for placing the location & the counts\n'
        '  "reasoning": str                    // 2-3 sentences on how you sized the express competitive set; describe rivals as competition, never as missing data\n'
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
                               temperature: float = 0.3,
                               nearby_washes: Optional[List[dict]] = None) -> Dict[str, Any]:
    """Ask the LLM how many EXPRESS/CONVEYOR-TUNNEL washes really operate near this pin and size the client's
    competitive saturation in that segment. Other wash types never enter the presented read — the model keeps an
    all-types count as background calibration only.

    When `nearby_washes` (real Google-Places washes: name + distance within ~11 mi, any type, with an
    `express_likely` name-keyword tag) are supplied, they are fed in as GROUND TRUTH the estimate is anchored to,
    and echoed back in the result so the UI can show the source of truth behind the numbers.

    Returns the parsed estimate, the client's own site count, and the implied SATURATION MULTIPLE (LLM total ÷ the
    client's own count = competitors per client site) for both the express-tunnel segment and all car washes.
    Estimates only — labelled as such in the UI. Raises `llm_client.LLMUnavailable` if no backend answers."""
    known = [str(s) for s in (known_sites or []) if str(s).strip()]
    nearby = list(nearby_washes or [])
    messages = build_competition_messages(lat, lon, known_sites=known, address=address, radius_km=radius_km,
                                          nearby_washes=nearby)
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
        "nearby_washes": nearby,            # real Google-Places washes fed as ground truth (name + distance)
        "backend": used,
        "prompt": messages[-1]["content"],
        "raw": (text or "").strip(),
    }


def _rng(rng: Any) -> str:
    """'low–high' from a {'low','high'} estimate, '?' when unknown."""
    if isinstance(rng, dict) and (rng.get("low") is not None or rng.get("high") is not None):
        return f"{rng.get('low', '?')}–{rng.get('high', '?')}"
    return "?"


_NON_EXPRESS_TYPE_HINTS = ("in-bay", "in bay", "self-serve", "self serve", "self-service",
                           "hand wash", "handwash", "detail", "gas")


def _non_express(c: dict) -> bool:
    """True when a competitor row is clearly typed OUTSIDE the express/tunnel segment. The prompt already
    demands express-only rows; this keeps the rendered read honest if the model slips one through. Blank or
    unrecognised types are kept — the model may omit the type on a genuine express rival."""
    return any(h in str(c.get("type") or "").lower() for h in _NON_EXPRESS_TYPE_HINTS)


def _competition_summary_md(result: Dict[str, Any]) -> str:
    """Build the Competition Coverage markdown SUMMARY from a `competition_scale_analysis` result — deterministic,
    no extra LLM call. React-markdown compatible (headings / **bold** / `- ` bullets). EXPRESS/TUNNEL washes only:
    the all-types estimate stays out of the read, and competitor rows typed outside the segment are dropped."""
    data = result.get("data") or {}
    n = result.get("known_count") or 0
    exp = data.get("estimated_express_tunnels") or {}
    se = result.get("scale_express") or {}
    share = data.get("estimated_client_share") or {}
    out: List[str] = ["## Nearby Carwash Analysis",
                      "_Express / conveyor-tunnel car washes only — other wash types are out of scope for this read._",
                      ""]

    # headline
    head = f"The client runs **{n}** express site(s) of their own here"
    if exp:
        head += f" against an estimated **{_rng(exp)} express tunnels** in the trade area"
    if se:
        head += f" — outnumbered roughly **{se.get('low', '?')}×–{se.get('high', '?')}×**"
    out += [head + ".", ""]

    out.append("### Saturation & headroom")
    sat_bullets = []
    if exp:
        sat_bullets.append(f"- **Express tunnels (client's segment):** ~**{_rng(exp)}** in the radius.")
    if share:
        sat_bullets.append(f"- **Client share of express tunnels:** ~**{_rng(share)}%**.")
    for label, key in (("Saturation", "saturation"), ("Competitive intensity", "competitive_intensity"),
                       ("Headroom", "headroom")):
        if data.get(key):
            sat_bullets.append(f"- **{label}:** {data[key]}")
    out += (sat_bullets or ["- _No saturation estimate available._"]) + [""]

    rivals = [c for c in (data.get("competitors") or [])
              if isinstance(c, dict) and c.get("name") and not _non_express(c)]
    if rivals:
        out.append("### Named competitors — express tunnels")
        out += ["| Competitor | Type | Scale | Threat | Notes |", "| --- | --- | --- | --- | --- |"]
        for c in rivals[:12]:
            out.append(f"| **{_md_cell(c.get('name'))}** | {_md_cell(c.get('type')) or '—'} | "
                       f"{_md_cell(c.get('scale')) or '—'} | {_md_cell(c.get('threat')) or '—'} | "
                       f"{_md_cell(c.get('note')) or '—'} |")
        out.append("")

    out.append("### Positioning")
    pos = []
    if data.get("client_position"):
        pos.append(f"- **Client position:** {data['client_position']}")
    if data.get("pricing_positioning"):
        pos.append(f"- **Pricing:** {data['pricing_positioning']}")
    if data.get("expansion_signals"):
        pos.append(f"- **Expansion signals:** {data['expansion_signals']}")
    out += (pos or ["- _No positioning detail available._"]) + [""]

    nearby = result.get("nearby_washes") or []
    if data.get("reasoning") or nearby:
        out.append("### How this was sized")
        if data.get("reasoning"):
            out.append(f"- {data['reasoning']}")
        if nearby:
            exp_names = [str(w.get("name")) for w in nearby if w.get("express_likely") and w.get("name")]
            tag = ("; " + f"{len(exp_names)} name-tagged express/tunnel: " + ", ".join(exp_names[:6])) if exp_names else ""
            out.append(f"- Anchored to **{len(nearby)}** real washes of all types observed nearby via "
                       f"Google Places{tag}.")
        if data.get("confidence"):
            out.append(f"- **Confidence:** {data['confidence']}")
    return "\n".join(out).strip()


def build_competition_response(result: Dict[str, Any]) -> Dict[str, Any]:
    """Shape a `competition_scale_analysis` result into the Competition Coverage API response. For now this returns
    only the react-markdown `summary` (which already embeds the competitors table + saturation/positioning); the
    structured `table` is intentionally omitted."""
    return {"summary": _competition_summary_md(result)}


# ─────────────── market-scale multiplier (ONE small JSON call — powers the 🌐 true-market blue-line scaling) ───────────────
def build_market_multiplier_messages(lat: float, lon: float, *, n_client_sites: int, radius_km: float = 20,
                                     nearby_washes: Optional[List[dict]] = None,
                                     address: Optional[str] = None) -> List[dict]:
    """A deliberately SMALL competitive request: the client's market chart sums only their OWN sites' washes —
    ask for the single MULTIPLIER that scales that client-only volume up to the full competitive market.
    EXPRESS CONVEYOR TUNNELS ONLY — the client's segment; hand washes / detailing / self-serve never count."""
    addr = (address or "").strip() or "(not provided — infer the place from the coordinates)"
    observed = ""
    if nearby_washes:
        observed = (f"OBSERVED washes near the pin via Google Places (GROUND TRUTH, name + miles; only covers ~5 mi "
                    f"and includes ALL wash types — count only the express tunnels among and beyond them):\n"
                    f"{_format_nearby_washes(nearby_washes)}\n"
                    "Your express-tunnel total must be consistent with this observed set, extrapolated to the full radius.\n\n")
    user = (
        "MARKET SCALE MULTIPLIER — JSON ONLY\n\n"
        f"LOCATION: {lat:.5f}, {lon:.5f} · {addr}\n"
        f"TRADE-AREA RADIUS: {radius_km:g} km (≈ {radius_km * 0.621:.0f} miles)\n"
        f"THE CLIENT OPERATES {n_client_sites} EXPRESS TUNNEL site(s) in this radius; their chart of 'total market "
        f"washes' sums ONLY those sites.\n\n"
        f"{observed}"
        "Estimate the TOTAL number of EXPRESS CONVEYOR TUNNEL car washes actually operating in this radius — the "
        "client's own included. STRICTLY EXCLUDE hand washes, detailing shops, self-serve bays, in-bay automatics "
        "and gas-station washes: they are NOT the client's segment and must not inflate the count. Then give the "
        "multiplier that scales the client-only wash volume up to the full express-tunnel market "
        "(express total ÷ the client's count, assuming a typical express tunnel does broadly similar volume). "
        "Return STRICT JSON with exactly these keys:\n"
        "{\n"
        '  "total_competing_washes": {"low": int, "high": int},  // EXPRESS TUNNELS ONLY in the radius\n'
        '  "multiplier": float,     // midpoint express total ÷ the client\'s count — the factor to scale the market line by\n'
        '  "confidence": str,       // "High"|"Medium"|"Low"\n'
        '  "reasoning": str         // 1-2 sentences on how you sized the express set\n'
        "}\n"
    )
    return [{"role": "system", "content": COMPETITION_SYSTEM_PROMPT}, {"role": "user", "content": user}]


def market_scale_multiplier(lat: float, lon: float, *, n_client_sites: int, radius_km: float = 20,
                            nearby_washes: Optional[List[dict]] = None,
                            address: Optional[str] = None, backend: Optional[str] = None,
                            max_tokens: int = 400, temperature: float = 0.2) -> Dict[str, Any]:
    """One small LLM call → {"multiplier": float ≥ 1, "total": {low, high}, "confidence", "reasoning", "backend"}.

    EXPRESS TUNNELS ONLY: the multiplier is what the client-only market-total line should be multiplied by to
    represent the full EXPRESS-TUNNEL competitive set (e.g. the client shows 5 sites, ~10 more express tunnels
    compete → ×3) — hand washes / detailing / self-serve are excluded by the prompt. The LLM's own `multiplier`
    is cross-checked against `total_competing_washes` ÷ n_client_sites and sanity-clamped to [1, 25].
    Raises `llm_client.LLMUnavailable` if no backend answers."""
    messages = build_market_multiplier_messages(lat, lon, n_client_sites=max(int(n_client_sites), 1),
                                                radius_km=radius_km, nearby_washes=list(nearby_washes or []),
                                                address=address)
    text, used = llm_client.complete_cascade(messages, backend=backend, max_tokens=max_tokens,
                                             temperature=temperature, json_mode=True)
    data = _parse_json_lax(text)
    tot = data.get("total_competing_washes")
    mult = data.get("multiplier")
    mult = float(mult) if isinstance(mult, (int, float)) else None
    tot_mid = _mid(tot)
    if mult is None and tot_mid is not None:                       # derive from the total if the model skipped it
        mult = tot_mid / max(int(n_client_sites), 1)
    if mult is not None:
        mult = float(min(max(mult, 1.0), 25.0))                    # a market line never shrinks; cap runaway estimates
    logger.info("Market-scale multiplier via %s backend: %.2f (n_client=%d).", used, mult or -1, n_client_sites)
    return {
        "multiplier": mult,
        "total": tot if isinstance(tot, dict) else None,
        "confidence": data.get("confidence"),
        "reasoning": data.get("reasoning"),
        "backend": used,
        "prompt": messages[-1]["content"],
        "raw": (text or "").strip(),
    }


# ─────────────── independent market research (external LLM knowledge only, per radius — no internal data) ───────────────
# A separate experiment answering: can a capable external model size a NEW car-wash market from PUBLIC knowledge alone
# (plus optional web search), with NO access to the repository's operating data? It is run INDEPENDENTLY per radius
# (3 / 6 / 9 miles) — one blind LLM call each — and each returns the requested business metrics as strict JSON, with a
# hard rule to say "cannot estimate" (estimate=null + why) rather than fabricate. Its own prompt; touches no internal data.
INDEPENDENT_RESEARCH_SYSTEM_PROMPT = (
    "You are an independent market-research analyst estimating the viability of a NEW express-tunnel car wash at a "
    "given location, using ONLY your own general world knowledge (and any web-search results explicitly provided). "
    "You have NO access to the operator's internal data, no proprietary numbers, and no site-level history — this is "
    "a test of how far a capable external model can size a market from public knowledge alone.\n\n"
    "Principles:\n"
    "- Reason from what you genuinely know about this location: metro, population density, household income, vehicle "
    "ownership, traffic corridors, existing car-wash competition, and climate/seasonality.\n"
    "- MAKE A BEST-EFFORT ESTIMATE wherever you have any reasonable basis. For a recognisable US location you CAN "
    "approximate population/vehicle counts in a radius, typical express-wash pricing ($/wash and unlimited plans), "
    "and the volume/revenue a mature express tunnel tends to do. Give these as RANGES with an explicit confidence "
    "(Low confidence is perfectly acceptable) and state your reasoning in \"basis\". Do not decline a metric merely "
    "because you lack precise figures — approximate it and mark the confidence.\n"
    "- Reserve \"estimate\": null STRICTLY for metrics you genuinely cannot approximate even roughly (e.g. you cannot "
    "place the coordinates at all); when you do, say WHY in \"basis\". NEVER fabricate false precision, but do not "
    "hide behind null either — most of these metrics can be reasonably approximated for a known place.\n"
    "- Respond with STRICT JSON ONLY — no prose, no markdown fences."
)


def build_independent_research_messages(lat: float, lon: float, radius_miles: float, *, address: Optional[str] = None,
                                        web_sources: Optional[List[dict]] = None) -> List[dict]:
    """One blind, per-radius request: estimate the car-wash market metrics for this location within `radius_miles`,
    from world knowledge only (plus optional web results). Strict JSON; null-with-reason when a metric can't be sized."""
    addr = (address or "").strip() or "(not provided — infer the place from the coordinates)"
    km = radius_miles * 1.60934
    web_section = ""
    if web_sources:
        web_section = (
            "\nFRESH WEB SEARCH RESULTS (retrieved just now for this location — treat as current ground truth and prefer "
            f"over static memory when they conflict):\n{_format_web_sources(web_sources)}\n"
        )
    know_src = "your world knowledge and the web results below" if web_sources else "your world knowledge"
    user = (
        "INDEPENDENT CAR-WASH MARKET RESEARCH — JSON ONLY (NO internal/operator data supplied)\n\n"
        "LOCATION:\n"
        f"- Latitude, Longitude: {lat:.5f}, {lon:.5f}\n"
        f"- Approx address / description: {addr}\n"
        f"- Trade-area radius for THIS analysis: {radius_miles:g} miles (≈ {km:.1f} km)\n"
        f"{web_section}\n"
        f"From {know_src}, estimate the following for a NEW express-tunnel car wash at this location, considering the "
        f"{radius_miles:g}-mile trade area. Base every figure on what you actually know about this place. "
        "ESTIMATE, DON'T DECLINE: give a best-effort RANGE for every metric you can reasonably approximate (with an "
        "explicit confidence — Low is fine). Use \"estimate\": null ONLY for a metric you genuinely cannot approximate "
        "even roughly, and then say why in \"basis\". Do not fabricate false precision, but do not over-use null — for "
        "a recognisable place, demand, volume, revenue and ASP can all be approximated.\n\n"
        "CRITICAL — SITE CAPACITY vs MARKET SIZE (read carefully): the metrics below describe ONE new express tunnel, "
        "not the whole market. A single express tunnel has a FINITE monthly throughput — a mature site typically does "
        "on the order of ~5,000–15,000 washes/month and physically cannot exceed its capacity. The trade-area RADIUS "
        "changes the addressable MARKET (customer_demand) and the competitive set, but it does NOT multiply a single "
        "site's own wash volume or revenue: almost all express-wash customers come from within ~3–5 miles, so a 9-mile "
        "radius does not mean the one site washes 3x the cars it would at 3 miles. Keep the SITE-LEVEL metrics "
        "(wash counts, revenues, ASPs) essentially consistent across radii — do NOT scale them up with the radius; if "
        "they barely change, say so in the basis. Only customer_demand / market_opportunity should grow with radius.\n"
        "Keep the numbers internally consistent: total washes = retail + membership washes; total revenue = retail + "
        "membership revenue; each revenue ≈ its wash count × its ASP.\n\n"
        "Return STRICT JSON with EXACTLY these keys, each an object {\"estimate\", \"unit\", \"confidence\", \"basis\"}:\n"
        "{\n"
        '  "market_opportunity":     {"estimate": str, "unit": "", "confidence": "High|Medium|Low", "basis": str},   // overall qualitative read; MAY grow with radius\n'
        '  "customer_demand":        {"estimate": str|null, "unit": "households / vehicles in radius", "confidence": "...", "basis": "..."},   // addressable MARKET size; grows with radius\n'
        '  "wash_volume":            {"estimate": str|null, "unit": "washes/month", "confidence": "...", "basis": "..."},   // the NEW SITE\'s OWN total washes/mo at maturity (capacity-bound; ~radius-independent)\n'
        '  "retail_wash_count":      {"estimate": str|null, "unit": "washes/month", "confidence": "...", "basis": "..."},   // of that total, the retail (pay-per-wash) portion\n'
        '  "membership_wash_count":  {"estimate": str|null, "unit": "washes/month", "confidence": "...", "basis": "..."},   // of that total, the unlimited-membership portion\n'
        '  "total_revenue":          {"estimate": str|null, "unit": "$/month", "confidence": "...", "basis": "..."},   // the site\'s total monthly revenue at maturity\n'
        '  "retail_revenue":         {"estimate": str|null, "unit": "$/month", "confidence": "...", "basis": "..."},   // retail portion of monthly revenue\n'
        '  "membership_revenue":     {"estimate": str|null, "unit": "$/month", "confidence": "...", "basis": "..."},   // membership portion of monthly revenue\n'
        '  "revenue_potential":      {"estimate": str|null, "unit": "$/year", "confidence": "...", "basis": "..."},   // the site\'s annual revenue potential / upside ceiling\n'
        '  "asp_retail":             {"estimate": str|null, "unit": "$/retail wash", "confidence": "...", "basis": "..."},   // typical retail average selling price per wash\n'
        '  "asp_membership":         {"estimate": str|null, "unit": "$/membership wash", "confidence": "...", "basis": "..."}   // EFFECTIVE revenue per membership wash (monthly plan price ÷ washes/member/month)\n'
        "}\n"
        "Give ranges (e.g. \"8,000–12,000\") where appropriate. \"estimate\" must be null (not 0, not a guess) whenever "
        "you cannot responsibly size it from knowledge.\n"
        "KEEP IT SHORT: every \"basis\" is ONE short clause (about 8–14 words) in plain English — the single key reason only, "
        "fact/number-led, no hype adjectives, and do NOT restate the estimate inside it. The three radii are shown side by "
        "side in ONE table, so keep the \"basis\" for a given metric consistent across radii (the reasoning does not change "
        "with radius for site-level figures). This report is read at a glance, so brevity matters."
    )
    return [{"role": "system", "content": INDEPENDENT_RESEARCH_SYSTEM_PROMPT}, {"role": "user", "content": user}]


# Plain-English labels; unit lives in the label so the number cells stay bare (no "/mo" clutter repeated per radius).
_INDEP_METRICS = [
    ("market_opportunity", "Market opportunity (overall read)"),
    ("customer_demand", "Nearby market size (households & vehicles)"),
    ("wash_volume", "Total washes per month (new site)"),
    ("retail_wash_count", "Pay-per-wash washes per month"),
    ("membership_wash_count", "Member washes per month"),
    ("total_revenue", "Total revenue per month"),
    ("retail_revenue", "Revenue from pay-per-wash per month"),
    ("membership_revenue", "Revenue from memberships per month"),
    ("revenue_potential", "Revenue per year (at maturity)"),
    ("asp_retail", "Price of one pay-per-wash"),
    ("asp_membership", "Revenue per member wash"),
]
_INDEP_NULLISH = {None, "", "null", "none", "n/a", "na", "unknown", "unavailable"}


def _md_cell(s: Any) -> str:
    """Make a value safe for a GFM markdown table cell: single line, escaped pipes."""
    return str(s if s is not None else "").replace("\r", " ").replace("\n", " ").replace("|", "\\|").strip()


def _short(s: str, limit: int = 140) -> str:
    """Backstop the 'keep it short' rule: trim over-long text to one clause at a word boundary."""
    s = (s or "").strip()
    if len(s) <= limit:
        return s
    cut = s[:limit].rsplit(" ", 1)[0].rstrip(" ,.;:—-")
    return cut + "…"


def _fmt_est_cell(v: Dict[str, Any]) -> str:
    """One estimate → a compact cell. Nullish → '—'; prepend '$' for money metrics that lack it; trim long
    qualitative reads so a single row stays scannable across all three radius columns."""
    est = v.get("estimate")
    nullish = (est if not isinstance(est, str) else str(est).strip().lower()) in _INDEP_NULLISH
    if nullish:
        return "—"
    est_s = str(est).strip()
    unit = v.get("unit") or ""
    if "$" in unit and "$" not in est_s and any(ch.isdigit() for ch in est_s):
        est_s = "$" + est_s
    return _md_cell(_short(est_s, 90))


def _independent_summary_md(results: List[Dict[str, Any]]) -> str:
    """Render the external-knowledge estimates as ONE compact react-markdown table: each metric is a row, the
    trade-area radii (3 / 6 / 9 mile) are columns, plus Confidence and a short plain-English 'Why'. Site-level
    figures barely move across radii (only nearby market size grows), so a single table reads far easier than
    three. '—' = the model could not responsibly size it (reason shown in 'Why'), never a fabricated value."""
    res = [r for r in (results or []) if r]
    res.sort(key=lambda d: d.get("radius_miles") or 0)
    out: List[str] = [
        "# Independent Market Research — external LLM knowledge only",
        "",
        "_World-knowledge estimate for this location (no internal data). Site figures barely change with radius — "
        "only nearby market size grows. '—' = not sized rather than guessed._",
        "",
    ]
    if not res or not any((r.get("metrics") or {}) for r in res):
        out.append("_The model returned no parseable estimate._")
        return "\n".join(out).strip()

    # Qualitative overall read is radius-independent — show it once as a takeaway, not thrice in a table row.
    overall = next((str(mo.get("estimate")).strip() for r in res
                    for mo in [((r.get("metrics") or {}).get("market_opportunity") or {})]
                    if mo.get("estimate") and str(mo.get("estimate")).strip().lower() not in _INDEP_NULLISH), "")
    if overall:
        out += [f"**Overall:** {_md_cell(_short(overall, 220))}", ""]

    radii = [r.get("radius_miles") or 0 for r in res]
    out += [
        "| Metric | " + " | ".join(f"{rm:g} mi" for rm in radii) + " | Confidence | Why |",
        "| --- | " + " | ".join("---" for _ in radii) + " | --- | --- |",
    ]
    for key, label in _INDEP_METRICS:
        if key == "market_opportunity":                    # shown above as the one-line takeaway
            continue
        cells, rep_conf, rep_basis = [], "", ""
        for r in res:
            m = r.get("metrics") or {}
            v = m.get(key) if isinstance(m.get(key), dict) else {}
            cells.append(_fmt_est_cell(v))
            if not rep_basis:                              # first available reasoning (radius-independent for site metrics)
                b = (v.get("basis") or "").strip()
                if b:
                    rep_conf, rep_basis = (v.get("confidence") or "").strip(), b
        out.append(f"| **{label}** | " + " | ".join(cells) +
                   f" | {_md_cell(rep_conf) or '—'} | {_md_cell(_short(rep_basis, 110)) or '—'} |")
    return "\n".join(out).strip()


def independent_market_research(lat: float, lon: float, *, address: Optional[str] = None,
                                radii_miles: Any = (3, 6, 9), backend: Optional[str] = None,
                                max_tokens: int = 1500, temperature: float = 0.3,
                                use_web_search: bool = False) -> Dict[str, Any]:
    """Can an external LLM size a NEW car-wash market from PUBLIC knowledge alone (no internal data)? Runs one blind
    LLM call PER radius (default 3/6/9 mi), each estimating the requested business metrics as strict JSON with a hard
    'say-null-if-you-cannot' rule. Radius calls run concurrently. When `use_web_search` is on and a provider is
    configured, fresh web results for the location are fed in as citable ground truth. Raises LLMUnavailable if no
    backend answers any radius."""
    radii = [float(x) for x in (radii_miles or (3, 6, 9)) if _is_pos(x)][:6] or [3.0, 6.0, 9.0]
    sources: List[dict] = []
    place = (address or "").strip()
    if use_web_search:
        try:
            sources, place = websearch.gather_location_sources(lat, lon, address=address,
                                                               radius_km=max(radii) * 1.60934)
        except Exception as e:                                     # search must never break the analysis
            logger.warning("Independent-research web grounding failed, continuing without it: %s", e)
            sources = []

    def _one(r: float) -> Dict[str, Any]:
        messages = build_independent_research_messages(lat, lon, r, address=address, web_sources=sources)
        text, used = llm_client.complete_cascade(messages, backend=backend, max_tokens=max_tokens,
                                                 temperature=temperature, json_mode=True)
        return {"radius_miles": r, "metrics": _parse_json_lax(text), "backend": used}

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(radii)) as ex:         # radii are independent → size them concurrently
        results = list(ex.map(_one, radii))
    results.sort(key=lambda d: d["radius_miles"])
    used = next((r.get("backend") for r in results if r.get("backend")), None)
    logger.info("Independent market research via %s backend (radii=%s, web_sources=%d).",
                used, [r["radius_miles"] for r in results], len(sources))
    return {"radii": results, "summary": _independent_summary_md(results),
            "sources": sources, "web_used": bool(sources), "place": place, "backend": used}


def _is_pos(x: Any) -> bool:
    try:
        return float(x) > 0
    except (TypeError, ValueError):
        return False


def build_independent_research_response(result: Dict[str, Any]) -> Dict[str, Any]:
    """Shape the independent-research result into the API response: per-radius `radii` (radius + metrics), the
    react-markdown `summary`, and any web `sources` (title/url) used. Required keys only."""
    return {
        "radii": [{"radius_miles": r.get("radius_miles"), "metrics": r.get("metrics") or {}}
                  for r in (result.get("radii") or [])],
        "summary": result.get("summary") or "",
        "sources": [{"title": s.get("title"), "url": s.get("url")} for s in (result.get("sources") or [])],
    }
