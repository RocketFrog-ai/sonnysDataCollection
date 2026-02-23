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
    _get_profiler,
    WEATHER_API_TO_PROFILER,
    GAS_API_TO_PROFILER,
    COMPETITORS_API_TO_PROFILER,
)


# Dimension -> (task feature key -> profiler feature key) for features the task collects
DIMENSION_FEATURE_MAP: Dict[str, Dict[str, str]] = {
    "Weather": WEATHER_API_TO_PROFILER,
    "Retail Proximity": {
        "distance_from_nearest_gas_station": "nearest_gas_station_distance_miles",
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


def _score_features_full(
    flat: Dict[str, Any],
    profiler,
) -> List[Dict[str, Any]]:
    """Score each feature and return full details (value, percentile, score, category)."""
    results = []
    for profiler_key, value in flat.items():
        try:
            info = profiler.score_feature_cto_method(profiler_key, float(value))
        except (KeyError, TypeError, ValueError):
            continue
        if "error" in info:
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

    profiler = _get_profiler()
    scored = _score_features_full(flat, profiler)
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


def _build_rationale(dimension: str, scored: List[Dict[str, Any]]) -> str:
    """
    Build human-readable rationale from Approach 2 methodology.

    Explains: your value, percentile rank, direction, category, and what it means.
    """
    lines: List[str] = []

    dim_intro = {
        "Weather": "**Weather** conditions affect car wash demand. Each metric is compared against 1,200+ sites in our portfolio. Lower precipitation and wind are better; more sunshine and pleasant days are better.",
        "Retail Proximity": "**Retail Proximity** covers gas stations, Costco, Walmart, Target, other grocery, and food joints. For distances (gas, Costco, Walmart, Target), closer is better. For counts (other grocery within 1 mi, food joints within 0.5 mi), higher is better. All are compared to our portfolio.",
        "Competition": "**Competition** reflects the local car wash landscape. Fewer competitors and greater distance to the nearest one are generally better; competitor rating and review count provide context.",
    }
    intro = dim_intro.get(dimension, f"**{dimension}**")
    lines.append(intro)
    lines.append("")

    for s in scored:
        feat = s["feature"]
        label = FEATURE_LABELS.get(feat, feat.replace("_", " ").title())
        value = s["value"]
        pct = s.get("raw_percentile")
        cat = s.get("category", "N/A")
        direction = s.get("direction", "")

        # Format value
        if isinstance(value, float):
            val_str = f"{value:,.2f}" if abs(value) >= 1 else f"{value:.2f}"
        else:
            val_str = str(value)

        line = f"**{label}** — Your value: {val_str}. "
        if pct is not None:
            line += f"This ranks at the {pct:.1f}th percentile of all sites. "
        line += f"Category: **{cat}**. "

        if "higher" in direction:
            if pct is not None:
                if pct >= 75:
                    line += f"You outperform {pct:.0f}% of sites—this is a strength."
                elif pct >= 50:
                    line += f"You are above the median, which is favorable."
                elif pct >= 25:
                    line += f"You are below the median; room for improvement."
                else:
                    line += f"You rank in the bottom {100-pct:.0f}%—consider this a weakness."
        else:
            if pct is not None:
                if pct <= 25:
                    line += f"You rank in the best {pct:.0f}% (low values)—this is a strength."
                elif pct <= 50:
                    line += f"You are better than the median; favorable."
                elif pct <= 75:
                    line += f"You are above the median; room for improvement."
                else:
                    line += f"You rank in the top {100-pct:.0f}% highest values—this is a weakness."

        lines.append(line)

    return "\n\n".join(lines)


def get_feature_performance_map(scored: List[Dict[str, Any]]) -> Dict[str, str]:
    """Map feature name -> category for backward compatibility with feature_performance."""
    return {s["feature"]: s.get("category", "N/A") for s in scored}
