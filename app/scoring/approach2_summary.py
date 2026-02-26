"""
Approach 2 Dimension Summaries
==============================

Generates human-readable summaries and rationale for each dimension (Weather,
Gas Stations, Competition, etc.) using Approach 2 categories (Excellent, Very Good,
Good, Fair, Poor, Very Poor) and percentile ranks.

Methodology (from output.txt / output2.txt):
  - Step 1: Calculate percentile rank of your value vs Proforma dataset
  - Step 2: Apply direction (invert if lower is better)
  - Step 3: Category from interpretation (Excellent, Very Good, Good, Fair, Poor, Very Poor)
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple

from app.scoring.approach2_scorer import (
    WEATHER_API_TO_PROFILER,
    GAS_API_TO_PROFILER,
    COMPETITORS_API_TO_PROFILER,
    score_feature_with_config,
    compute_dimension_score,
)


# Dimension -> (task feature key -> profiler feature key) for features the task collects
DIMENSION_FEATURE_MAP: Dict[str, Dict[str, str]] = {
    "Weather": WEATHER_API_TO_PROFILER,
    "Gas": {
        "distance_from_nearest_gas_station": "nearest_gas_station_distance_miles",
        "nearest_gas_station_rating": "nearest_gas_station_rating",
        "nearest_gas_station_rating_count": "nearest_gas_station_rating_count",
    },
    "Retail Proximity": {
        "distance_from_nearest_costco": "distance_nearest_costco(5 mile)",
        "distance_from_nearest_walmart": "distance_nearest_walmart(5 mile)",
        "distance_from_nearest_target": "distance_nearest_target (5 mile)",
        "other_grocery_count_1mile": "other_grocery_count_1mile",
        "count_food_joints_0_5miles": "count_food_joints_0_5miles (0.5 mile)",
    },
    "Competition": {
        "competitors_count": "competitors_count_4miles",
        "competitor_1_distance_miles": "competitor_1_distance_miles",
        "competitor_1_google_rating": "competitor_1_google_rating",
        "competitor_1_google_user_rating_count": "competitor_1_rating_count",
    },
}

# Human-readable feature labels
FEATURE_LABELS: Dict[str, str] = {
    "weather_total_precipitation_mm": "Total precipitation",
    "weather_rainy_days": "Rainy days per year",
    "weather_total_snowfall_cm": "Total snowfall",
    "weather_days_below_freezing": "Days below freezing",
    "weather_total_sunshine_hours": "Total sunshine hours",
    "weather_days_pleasant_temp": "Pleasant temperature days",
    "weather_avg_daily_max_windspeed_ms": "Average daily max windspeed",
    "nearest_gas_station_distance_miles": "Distance to nearest gas station",
    "nearest_gas_station_rating": "Nearest gas station rating",
    "nearest_gas_station_rating_count": "Nearest gas station review count",
    "competitors_count_4miles": "Competitors within 4 miles",
    "competitor_1_distance_miles": "Distance to nearest competitor",
    "competitor_1_google_rating": "Nearest competitor rating",
    "competitor_1_rating_count": "Nearest competitor review count",
    "distance_nearest_costco(5 mile)": "Distance to nearest Costco (5 mi)",
    "distance_nearest_walmart(5 mile)": "Distance to nearest Walmart (5 mi)",
    "distance_nearest_target (5 mile)": "Distance to nearest Target (5 mi)",
    "other_grocery_count_1mile": "Other grocery stores within 1 mile",
    "count_food_joints_0_5miles (0.5 mile)": "Food joints within 0.5 mile",
}

# Portfolio size for summary copy (mention once per dimension)
PORTFOLIO_N_SITES = 1267

# Set False to revert: summary will not include LLM-generated driving/negative factors and score summary tail
INCLUDE_LLM_SUMMARY_TAIL = True

# Direction explanations for rationale
DIRECTION_EXPLAIN: Dict[str, str] = {
    "higher_is_better": "Higher values are better",
    "lower_is_better": "Lower values are better",
}

# Short business context per feature (car wash / site selection)
FEATURE_CAR_WASH_CONTEXT: Dict[str, str] = {
    "weather_total_precipitation_mm": "Moderate rain supports need without compressing demand into few dry days.",
    "weather_rainy_days": "More rainy days can mean more wash need; portfolio comparison sets seasonality expectations.",
    "weather_total_snowfall_cm": "Snow drives winter need but can suppress visits; factor in labor and seasonality.",
    "weather_days_below_freezing": "Moderate freeze days balance equipment and demand; extremes need stronger convenience.",
    "weather_total_sunshine_hours": "More sunshine supports impulse volume; less sunshine favors necessity-driven capture.",
    "weather_days_pleasant_temp": "Pleasant days drive discretionary volume; below median suggests loyalty and convenience focus.",
    "weather_avg_daily_max_windspeed_ms": "Lower wind helps drying and drive-through experience.",
    "nearest_gas_station_distance_miles": "Closer gas supports impulse traffic and capture potential.",
    "nearest_gas_station_rating": "Higher-rated nearby gas often indicates stronger trade area.",
    "nearest_gas_station_rating_count": "More reviews suggest busier station and more nearby demand.",
    "distance_nearest_costco(5 mile)": "Closer Costco supports co-visits and trade-area quality.",
    "distance_nearest_walmart(5 mile)": "Distance to Walmart helps size the addressable demand pool.",
    "distance_nearest_target (5 mile)": "Closer Target supports trip consolidation and volume.",
    "other_grocery_count_1mile": "More grocery nearby supports convenience trips and demand base.",
    "count_food_joints_0_5miles (0.5 mile)": "More food options extend dwell time and capture.",
    "competitors_count_4miles": "Moderate count balances visibility and pricing; extremes need differentiation.",
    "competitor_1_distance_miles": "Farther competitor can mean less overlap; closer needs convenience edge.",
    "competitor_1_google_rating": "Weaker nearest competitor can ease differentiation pressure.",
    "competitor_1_rating_count": "Fewer reviews can mean less established competitor.",
}


def _extract_flat_for_dimension(dimension: str, feature_values: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and map task feature_values to profiler keys for the dimension."""
    mapping = DIMENSION_FEATURE_MAP.get(dimension, {})
    if not mapping:
        return {}

    flat = {}
    for task_key, profiler_key in mapping.items():
        val = feature_values.get(task_key)
        if val is not None and not (isinstance(val, float) and str(val) == "nan"):
            flat[profiler_key] = val
    return flat


def _score_features_full(flat: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Score each feature using config direction; return value, percentile, score, category."""
    results = []
    for profiler_key, value in flat.items():
        info = score_feature_with_config(profiler_key, float(value))
        if info is None:
            continue

        interp = info.get("interpretation", "")
        category = interp.split(" (")[0].strip() if interp else "N/A"

        results.append({
            "feature": profiler_key,
            "value": value,
            "raw_percentile": info.get("raw_percentile"),
            "final_score": info.get("final_score"),
            "category": category,
            "direction": info.get("direction", ""),
        })
    return results


def _llm_summary_tail(
    dimension: str,
    scored: List[Dict[str, Any]],
    overall_category: str,
) -> str:
    """
    Use local LLM to generate a short tail: driving factors, negative factors, and one-line score summary.
    Returns empty string on failure so caller can keep summary unchanged.
    """
    if not scored:
        return ""
    profiler_scores = {s["feature"]: float(s.get("final_score", 0)) for s in scored}
    dim_score = compute_dimension_score(profiler_scores, dimension)
    score_str = f"{dim_score:.1f}" if dim_score is not None else "N/A"

    lines = [
        f"Dimension: {dimension}. Overall category: {overall_category}. Dimension score (0–100): {score_str}.",
        "Per-feature (label, value, score 0–100, category):",
    ]
    for s in scored:
        label = FEATURE_LABELS.get(s["feature"], s["feature"].replace("_", " ").title())
        val = s.get("value", "")
        if isinstance(val, float):
            val = f"{val:,.1f}" if abs(val) >= 1 else f"{val:.2f}"
        fs = s.get("final_score")
        fs_str = f"{fs:.0f}" if fs is not None else "N/A"
        cat = s.get("category", "N/A")
        lines.append(f"  - {label}: value {val}, score {fs_str}, category {cat}")

    prompt = "\n".join(lines) + """

Using only the feature labels above (no variable names), write 2–3 short sentences total:
1) Which factors are driving the score or are strengths?
2) Which are negative or weaknesses?
3) One-line concise summary of what this score means for this dimension.
Be concise and use plain English only."""

    try:
        from app.utils.llm import local_llm as llm
        response = llm.get_llm_response(prompt, reasoning_effort="low", temperature=0.2)
        text = (response or {}).get("generated_text", "").strip()
        return text if text else ""
    except Exception:
        return ""


def get_dimension_summary_approach2(
    dimension: str,
    feature_values: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Get Approach 2 summary for a dimension.

    Returns:
        {
            "features_scored": int,
            "feature_scores": [ { feature, value, raw_percentile, final_score, category } ],
            "overall_category": str (dominant category or average),
            "summary": str (human-readable rationale),
        }
    """
    flat = _extract_flat_for_dimension(dimension, feature_values)
    if not flat:
        return {
            "features_scored": 0,
            "feature_scores": [],
            "overall_category": "Insufficient Data",
            "summary": f"No Approach 2 features available for {dimension}. Ensure the analysis task collected the required data.",
        }

    scored = _score_features_full(flat)
    if not scored:
        return {
            "features_scored": 0,
            "feature_scores": [],
            "overall_category": "Insufficient Data",
            "summary": f"Could not score any {dimension} features against the Proforma dataset.",
        }

    # Overall: use median category or most common
    cat_order = ["Excellent", "Very Good", "Good", "Fair", "Poor", "Very Poor"]
    scores_numeric = [cat_order.index(s["category"]) if s["category"] in cat_order else 3 for s in scored]
    avg_idx = sum(scores_numeric) / len(scores_numeric)
    overall_category = cat_order[min(int(round(avg_idx)), len(cat_order) - 1)]

    summary = _build_rationale(dimension, scored)

    if INCLUDE_LLM_SUMMARY_TAIL and scored:
        try:
            tail = _llm_summary_tail(dimension, scored, overall_category)
            if tail:
                summary = summary + "\n\n" + tail
        except Exception:
            pass

    return {
        "features_scored": len(scored),
        "feature_scores": scored,
        "overall_category": overall_category,
        "summary": summary,
    }


def _favorable_pct(pct: float, direction: str) -> float:
    """Convert raw percentile to 'favorable' share (higher = better for the site)."""
    if direction in ("lower_is_better",):
        return 100 - pct
    if direction == "moderate_is_best":
        dist = abs(pct - 50)
        return max(0, min(100, 100 - 2 * dist))
    return pct


def _pct_context(pct: float, direction: str, n_sites: int | None = None) -> str:
    """
    Plain-English comparison. If n_sites given, include portfolio size (use once per block).
    Otherwise short form for use after an intro that already stated the portfolio.
    """
    favorable = _favorable_pct(pct, direction)
    if n_sites is not None:
        return f"better than {favorable:.0f}% of the {n_sites:,} car wash sites analysed"
    return f"better than {favorable:.0f}% of sites"


def _fmt_val(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:,.1f}" if abs(value) >= 1 else f"{value:.2f}"
    return str(value)


def _build_rationale(dimension: str, scored: List[Dict[str, Any]]) -> str:
    """
    Short, human-readable paragraph using actual values and plain English.
    No jargon — explains what the numbers mean for this location.
    """
    if not scored:
        return f"No data available for {dimension}."

    def fv(feat_key: str):
        return next((s for s in scored if s["feature"] == feat_key), None)

    if dimension == "Weather":
        lines = [f"🌤️ Weather summary based on historical data for this location, compared to {PORTFOLIO_N_SITES:,} car wash sites:"]
        for s in scored:
            pct = s.get("raw_percentile")
            direction = s.get("direction", "")
            lbl = FEATURE_LABELS.get(s["feature"], s["feature"].replace("_", " ").title())
            val = _fmt_val(s["value"])
            ctx = _pct_context(pct, direction, n_sites=None) if pct is not None else None
            line = f"• {lbl} is {val}"
            if ctx:
                line += f", {ctx}"
            lines.append(line)
        return "\n".join(lines)

    if dimension == "Gas":
        dist_s = fv("nearest_gas_station_distance_miles")
        rating_s = fv("nearest_gas_station_rating")
        count_s = fv("nearest_gas_station_rating_count")
        parts = []
        if dist_s:
            pct = dist_s.get("raw_percentile")
            ctx = _pct_context(pct, dist_s.get("direction", ""), PORTFOLIO_N_SITES) if pct is not None else None
            txt = f"The nearest gas station is {_fmt_val(dist_s['value'])} miles away"
            if ctx:
                txt += f" — {ctx}"
            parts.append(txt)
        if rating_s and rating_s.get("value") is not None:
            pct = rating_s.get("raw_percentile")
            ctx = _pct_context(pct, rating_s.get("direction", ""), n_sites=None) if pct is not None else None
            txt = f"It has a rating of {_fmt_val(rating_s['value'])}/5"
            if ctx:
                txt += f" ({ctx})"
            parts.append(txt)
        if count_s and count_s.get("value") is not None:
            pct = count_s.get("raw_percentile")
            ctx = _pct_context(pct, count_s.get("direction", ""), n_sites=None) if pct is not None else None
            txt = f"It has {int(count_s['value'])} reviews on Google"
            if ctx:
                txt += f" ({ctx})"
            parts.append(txt)
        return "⛽ " + ". ".join(parts) + "." if parts else "⛽ No gas station data available."

    if dimension == "Retail Proximity":
        lines = [f"🛒 Retail anchors near this site, compared to {PORTFOLIO_N_SITES:,} car wash sites:"]
        anchor_map = {
            "distance_nearest_costco(5 mile)": "Nearest Costco",
            "distance_nearest_walmart(5 mile)": "Nearest Walmart",
            "distance_nearest_target (5 mile)": "Nearest Target",
            "other_grocery_count_1mile": "Other grocery stores within 1 mile",
            "count_food_joints_0_5miles (0.5 mile)": "Food joints within 0.5 miles",
        }
        for feat, lbl in anchor_map.items():
            s = fv(feat)
            if s is None:
                continue
            pct = s.get("raw_percentile")
            direction = s.get("direction", "")
            ctx = _pct_context(pct, direction, n_sites=None) if pct is not None else None
            is_dist = "distance" in feat
            val = _fmt_val(s["value"])
            if is_dist:
                line = f"• {lbl}: {val} miles away"
            else:
                line = f"• {lbl}: {val}"
            if ctx:
                line += f" ({ctx})"
            lines.append(line)
        return "\n".join(lines)

    if dimension == "Competition":
        count_s = fv("competitors_count_4miles")
        dist_s = fv("competitor_1_distance_miles")
        rating_s = fv("competitor_1_google_rating")
        review_s = fv("competitor_1_rating_count")
        parts = []
        if count_s:
            pct = count_s.get("raw_percentile")
            ctx = _pct_context(pct, count_s.get("direction", ""), PORTFOLIO_N_SITES) if pct is not None else None
            txt = f"There are {int(count_s['value'])} car wash competitors within 4 miles"
            if ctx:
                txt += f" — {ctx}"
            parts.append(txt)
        if dist_s and dist_s.get("value") is not None:
            pct = dist_s.get("raw_percentile")
            ctx = _pct_context(pct, dist_s.get("direction", ""), n_sites=None) if pct is not None else None
            txt = f"The nearest competitor is {_fmt_val(dist_s['value'])} miles away"
            if ctx:
                txt += f" ({ctx})"
            parts.append(txt)
        if rating_s and rating_s.get("value") is not None:
            txt = f"It has a Google rating of {_fmt_val(rating_s['value'])}/5"
            if review_s and review_s.get("value") is not None:
                txt += f" with {int(review_s['value'])} reviews"
            parts.append(txt)
        return "🏪 " + ". ".join(parts) + "." if parts else "🏪 No competition data available."

    # Generic fallback
    lines = [f"📊 {dimension} summary:"]
    for s in scored:
        pct = s.get("raw_percentile")
        lbl = FEATURE_LABELS.get(s["feature"], s["feature"].replace("_", " ").title())
        ctx = _pct_context(pct, s.get("direction", "")) if pct is not None else None
        line = f"• {lbl}: {_fmt_val(s['value'])}"
        if ctx:
            line += f" ({ctx})"
        lines.append(line)
    return "\n".join(lines)


def get_feature_performance_map(scored: List[Dict[str, Any]]) -> Dict[str, str]:
    """Map feature name -> category for backward compatibility with feature_performance."""
    return {s["feature"]: s.get("category", "N/A") for s in scored}
