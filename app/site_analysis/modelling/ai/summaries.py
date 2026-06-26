"""
Plain-English AI summaries grounded ONLY on the raw fetched location data.

One LLM call per dimension (weather / competition / retail / gas). Each function takes the raw
fetched payload for that dimension and returns {insight, pro, con, conclusion}. No percentiles,
quartiles, categories, or model output — the summary describes the real values that were fetched.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from app.site_analysis.modelling.ai.common import get_llm_text
from app.site_analysis.server.config import (
    GAS_RADIUS_FAR_MILES,
    RETAIL_RADIUS_FAR_MILES,
    WEATHER_METRIC_CONFIG,
    WEATHER_METRIC_DISPLAY,
    get_weather_metric_value_from_climate,
    is_high_traffic_gas_brand,
)

logger = logging.getLogger(__name__)

_EMPTY = {"insight": None, "pro": None, "con": None, "conclusion": None}


def _build_prompt(dimension_title: str, facts: str, guidance: str) -> str:
    """Shared prompt shell: feed the raw facts, ask for Insight/Pro/Con/Conclusion in plain English."""
    return f"""You are a car wash site analyst. Using ONLY the facts below about {dimension_title} near this site,
write a short, plain-English read for a non-technical reader. Refer to it as "this site".

Facts (use only these numbers; do not invent any):
{facts}

What matters: {guidance}

Strict rules:
- No jargon, scores, percentiles, quartiles, or model talk.
- Keep sentences short and conversational, cause-and-effect (X → wash demand).
- Do not repeat the same idea.

Output EXACTLY this format:
Insight: <1-2 sentences on what the {dimension_title} picture means for wash demand>
Pro: <1 sentence on what helps wash demand>
Con: <1 sentence on what limits wash demand>
Conclusion: <1 short takeaway sentence>"""


def _run(prompt: str) -> Dict[str, Optional[str]]:
    """Call the LLM and parse the Insight/Pro/Con/Conclusion sections."""
    try:
        text = get_llm_text(prompt, max_new_tokens=512)
    except Exception as e:
        logger.warning("Summary LLM call failed: %s", e)
        return dict(_EMPTY)
    if not text:
        return dict(_EMPTY)

    def _grab(label: str) -> Optional[str]:
        m = re.search(
            rf"{label}:\s*(.+?)(?=\s*(?:Insight|Pro|Con|Conclusion):|$)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        return m.group(1).strip().replace("**", "") if m else None

    return {
        "insight": _grab("Insight"),
        "pro": _grab("Pro"),
        "con": _grab("Con"),
        "conclusion": _grab("Conclusion"),
    }


# ─────────────────────────── weather ───────────────────────────
def summarize_weather(climate: Optional[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    """Summary from the raw climate dict (rainy days, snowfall, comfortable days, freezing days)."""
    climate = climate or {}
    if not climate or climate.get("error"):
        return dict(_EMPTY)
    lines: List[str] = []
    for metric_key in WEATHER_METRIC_CONFIG:
        value, unit = get_weather_metric_value_from_climate(climate, metric_key)
        if value is None:
            continue
        display_name, _ = WEATHER_METRIC_DISPLAY.get(metric_key, (metric_key, ""))
        lines.append(f"- {display_name}: {value:.0f} {unit}")
    if not lines:
        return dict(_EMPTY)
    facts = "\n".join(lines)
    guidance = (
        "more rain and snow create dirt (more wash demand), but freezing days and harsh weather "
        "shut washing down; comfortable-temperature days are prime washing days"
    )
    return _run(_build_prompt("the local weather", facts, guidance))


# ─────────────────────────── competition ───────────────────────────
def summarize_competition(competitors_data: Optional[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    """Summary from nearby same-format car washes (within 4 miles)."""
    competitors_data = competitors_data or {}
    competitors = competitors_data.get("competitors") or []
    count = competitors_data.get("count")
    if count is None:
        count = len(competitors)
    nearest = competitors[0] if competitors else {}

    lines = [f"- Same-format car washes within 4 miles: {count}"]
    if nearest:
        d = nearest.get("distance_miles")
        r = nearest.get("rating")
        rc = nearest.get("user_rating_count") or nearest.get("rating_count")
        nm = nearest.get("name")
        if nm:
            lines.append(f"- Nearest competitor: {nm}")
        if d is not None:
            lines.append(f"- Distance to nearest competitor: {float(d):.2f} miles")
        if r is not None:
            lines.append(f"- Nearest competitor Google rating: {float(r):.1f} stars")
        if rc is not None:
            lines.append(f"- Nearest competitor review count: {int(rc)}")
    facts = "\n".join(lines)
    guidance = (
        "fewer and farther competitors mean less rivalry for this site; a close, highly-rated, "
        "heavily-reviewed competitor is strong competition that pulls demand away"
    )
    return _run(_build_prompt("nearby competing car washes", facts, guidance))


# ─────────────────────────── retail ───────────────────────────
def summarize_retail(retail_anchors: Optional[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    """Summary from nearby retail anchors (warehouse clubs, big box, grocery, food)."""
    retail_anchors = retail_anchors or {}
    anchors = retail_anchors.get("anchors") or []
    costco = retail_anchors.get("costco_dist")
    walmart = retail_anchors.get("walmart_dist")
    target = retail_anchors.get("target_dist")
    grocery = retail_anchors.get("grocery_count_1mile")
    food = retail_anchors.get("food_count_0_5miles")

    lines: List[str] = []
    if costco is not None:
        lines.append(f"- Nearest warehouse club (Costco/Sam's): {float(costco):.2f} miles")
    if walmart is not None:
        lines.append(f"- Nearest Walmart: {float(walmart):.2f} miles")
    if target is not None:
        lines.append(f"- Nearest Target: {float(target):.2f} miles")
    if grocery is not None:
        lines.append(f"- Grocery stores within 1 mile: {int(grocery)}")
    if food is not None:
        lines.append(f"- Food & beverage spots within 0.5 mile: {int(food)}")
    nearest = anchors[0] if anchors else {}
    if nearest and nearest.get("name"):
        nd = nearest.get("distance_miles")
        nd_str = f" ({float(nd):.2f} mi)" if nd is not None else ""
        lines.append(f"- Nearest retail anchor: {nearest['name']}{nd_str}")
    if not lines:
        return dict(_EMPTY)
    facts = "\n".join(lines)
    guidance = (
        "close, busy retail anchors and lots of nearby grocery/food traffic feed errand trips that "
        "drive impulse car washes; far or sparse retail means weaker passing traffic"
    )
    return _run(_build_prompt("nearby shopping and food activity", facts, guidance))


# ─────────────────────────── gas ───────────────────────────
def summarize_gas(gas_stations: Optional[List[Dict[str, Any]]]) -> Dict[str, Optional[str]]:
    """Summary from nearby gas stations (nearest distance/rating/reviews + high-traffic brand)."""
    stations = gas_stations or []
    stations = [s for s in stations if s.get("distance_miles") is not None]
    stations.sort(key=lambda s: s["distance_miles"])
    within = [s for s in stations if s["distance_miles"] <= GAS_RADIUS_FAR_MILES]
    nearest = stations[0] if stations else {}

    lines = [f"- Gas stations within {GAS_RADIUS_FAR_MILES:.0f} miles: {len(within)}"]
    if nearest:
        nm = nearest.get("name")
        d = nearest.get("distance_miles")
        r = nearest.get("rating")
        rc = nearest.get("user_rating_count") or nearest.get("rating_count")
        if nm:
            lines.append(f"- Nearest gas station: {nm}")
        if d is not None:
            lines.append(f"- Distance to nearest gas station: {float(d):.2f} miles")
        if r is not None:
            lines.append(f"- Nearest gas station rating: {float(r):.1f} stars")
        if rc is not None:
            lines.append(f"- Nearest gas station review count: {int(rc)}")
        if is_high_traffic_gas_brand(nm):
            lines.append("- The nearest station is a high-traffic fuel brand")
    facts = "\n".join(lines)
    guidance = (
        "a close, busy, well-known gas station means lots of passing drivers and impulse stops that "
        "lift wash demand; a far or quiet station means weaker fuel-stop traffic"
    )
    return _run(_build_prompt("nearby gas stations", facts, guidance))
