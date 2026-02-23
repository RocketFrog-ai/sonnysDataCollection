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
)


# Dimension -> (task feature key -> profiler feature key) for features the task collects
DIMENSION_FEATURE_MAP: Dict[str, Dict[str, str]] = {
    "Weather": WEATHER_API_TO_PROFILER,
    "Gas": {
        "distance_from_nearest_gas_station": "nearest_gas_station_distance_miles",
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

    return {
        "features_scored": len(scored),
        "feature_scores": scored,
        "overall_category": overall_category,
        "summary": summary,
    }


def _score_to_signal(final_score: float) -> str:
    """Map final_score to a short human signal word."""
    if final_score >= 80:
        return "strong"
    if final_score >= 60:
        return "above average"
    if final_score >= 40:
        return "average"
    if final_score >= 20:
        return "below average"
    return "weak"


def _build_rationale(dimension: str, scored: List[Dict[str, Any]]) -> str:
    """
    Single short paragraph summary. No per-feature breakdown — just a human-readable
    interpretation of the overall picture for this dimension.
    """
    if not scored:
        return f"No data available for {dimension}."

    cat_order = ["Excellent", "Very Good", "Good", "Fair", "Poor", "Very Poor"]

    strengths = [s for s in scored if s.get("final_score", 50) >= 60]
    weaknesses = [s for s in scored if s.get("final_score", 50) < 40]
    avg_score = sum(s.get("final_score", 50) for s in scored) / len(scored)

    def label(s):
        return FEATURE_LABELS.get(s["feature"], s["feature"].replace("_", " ").title())

    if dimension == "Weather":
        strength_names = [label(s) for s in strengths]
        weak_names = [label(s) for s in weaknesses]
        parts = []
        if strength_names:
            parts.append(f"Favorable on {', '.join(strength_names[:2])}")
        if weak_names:
            parts.append(f"challenging on {', '.join(weak_names[:2])}")
        note = ". ".join(parts) if parts else "Mixed weather profile"
        signal = _score_to_signal(avg_score)
        return f"🌤️ Weather is {signal} vs portfolio. {note}. Seasonality and demand planning are key for this location."

    if dimension == "Gas":
        s0 = scored[0]
        pct = s0.get("raw_percentile")
        dist_val = s0.get("value")
        signal = _score_to_signal(s0.get("final_score", 50))
        pct_note = f"{pct:.0f}th percentile" if pct is not None else ""
        dist_note = f"{dist_val:.2f} mi away" if isinstance(dist_val, float) else ""
        detail = f" ({dist_note}, {pct_note})" if dist_note or pct_note else ""
        return f"⛽ Nearest gas station{detail} is a {signal} traffic co-location opportunity vs portfolio."

    if dimension == "Retail Proximity":
        strength_names = [label(s) for s in strengths]
        weak_names = [label(s) for s in weaknesses]
        signal = _score_to_signal(avg_score)
        parts = []
        if strength_names:
            parts.append(f"strong anchor proximity ({', '.join(strength_names[:2])})")
        if weak_names:
            parts.append(f"limited on {', '.join(weak_names[:2])}")
        note = "; ".join(parts) if parts else "mixed anchor mix"
        return f"🛒 Retail proximity is {signal} vs portfolio — {note}. Anchor traffic supports co-visit and convenience demand."

    if dimension == "Competition":
        count_s = next((s for s in scored if "count" in s["feature"]), None)
        dist_s = next((s for s in scored if "distance" in s["feature"]), None)
        signal = _score_to_signal(avg_score)
        parts = []
        if count_s:
            pct = count_s.get("raw_percentile")
            parts.append(f"{int(count_s['value'])} competitors within 4 mi ({pct:.0f}th pct)" if pct is not None else f"{int(count_s['value'])} competitors within 4 mi")
        if dist_s:
            parts.append(f"nearest is {dist_s['value']:.2f} mi away")
        note = ", ".join(parts) if parts else "competition data available"
        return f"🏪 Competitive landscape is {signal} vs portfolio — {note}. Differentiation on convenience and service is key."

    # Generic fallback
    signal = _score_to_signal(avg_score)
    top = sorted(scored, key=lambda s: s.get("final_score", 50), reverse=True)
    top_label = label(top[0]) if top else ""
    return f"📊 {dimension} scores {signal} vs portfolio. Strongest metric: {top_label}."


def get_feature_performance_map(scored: List[Dict[str, Any]]) -> Dict[str, str]:
    """Map feature name -> category for backward compatibility with feature_performance."""
    return {s["feature"]: s.get("category", "N/A") for s in scored}
