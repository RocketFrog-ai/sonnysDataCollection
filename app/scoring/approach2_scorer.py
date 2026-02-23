"""
Approach 2 scorer for API responses.
Uses CTOExactProfiler (percentile-based) to assign category (Excellent, Very Good, Good, Fair, Poor, Very Poor)
to each feature value for display in the UI.
"""

import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Dict, Any, Optional, List

import pandas as pd

from app.scoring.feature_weights_config import (
    FEATURE_WEIGHTS,
    DIMENSION_FEATURES,
    DIMENSION_WEIGHTS_FOR_OVERALL,
)

# Load CTOExactProfiler from scoringmetric/approach2
_project_root = Path(__file__).resolve().parent.parent.parent
_approach2_dir = _project_root / "scoringmetric" / "approach2"
if str(_approach2_dir) not in sys.path:
    sys.path.insert(0, str(_approach2_dir))
from main import CTOExactProfiler  # noqa: E402

# Singleton profiler (lazy load)
_profiler: Optional[CTOExactProfiler] = None

# API feature name -> Profiler feature name (Proforma: Proforma-v2-data-final.xlsx columns)
WEATHER_API_TO_PROFILER: Dict[str, str] = {
    "total_precipitation_mm": "weather_total_precipitation_mm",
    "rainy_days": "weather_rainy_days",
    "total_snowfall_cm": "weather_total_snowfall_cm",
    "days_below_freezing": "weather_days_below_freezing",
    "total_sunshine_hours": "weather_total_sunshine_hours",
    "days_pleasant_temp": "weather_days_pleasant_temp",
    "avg_daily_max_windspeed_ms": "weather_avg_daily_max_windspeed_ms",
}

# Gas: API keys (from nearest station) -> Profiler keys
GAS_API_TO_PROFILER: Dict[str, str] = {
    "distance_miles": "nearest_gas_station_distance_miles",
    "rating": "nearest_gas_station_rating",
    "rating_count": "nearest_gas_station_rating_count",
}

# Competitors: API keys -> Profiler keys
COMPETITORS_API_TO_PROFILER: Dict[str, str] = {
    "count": "competitors_count_4miles",
    "competitor_1_distance_miles": "competitor_1_distance_miles",
    "competitor_1_google_rating": "competitor_1_google_rating",
    "competitor_1_rating_count": "competitor_1_rating_count",
}

# Retailers (Costco, Walmart, Target, other grocery, food joints) - API/task key -> Proforma column
RETAILER_API_TO_PROFILER: Dict[str, str] = {
    "distance_from_nearest_costco": "distance_nearest_costco(5 mile)",
    "distance_from_nearest_walmart": "distance_nearest_walmart(5 mile)",
    "distance_from_nearest_target": "distance_nearest_target (5 mile)",
    "other_grocery_count_1mile": "other_grocery_count_1mile",
    "count_food_joints_0_5miles": "count_food_joints_0_5miles (0.5 mile)",
}


def _get_profiler() -> CTOExactProfiler:
    global _profiler
    if _profiler is None:
        data_path = _project_root / "scoringmetric" / "approach2" / "Proforma-v2-data-final (1).xlsx"
        if not data_path.exists():
            data_path = _project_root / "scoringmetric" / "approach2" / "Proforma-v2-data-final.xlsx"
        if not data_path.exists():
            raise FileNotFoundError(
                f"Scoring dataset not found: {data_path}. "
                "Run approach2 with Proforma-v2-data-final.xlsx first."
            )
        kwargs = {"engine": "openpyxl"}
        if "Proforma-v2-data" in str(data_path):
            kwargs["header"] = 1
        df = pd.read_excel(data_path, **kwargs)
        with redirect_stdout(StringIO()):
            _profiler = CTOExactProfiler(df)
    return _profiler


def _interpretation_to_category(interpretation: str) -> str:
    """Extract short category from interpretation string.
    e.g. 'Excellent (top 10%)' -> 'Excellent'
         'Very Good (top 25% - low values)' -> 'Very Good'
    """
    if not interpretation:
        return "N/A"
    return interpretation.split(" (")[0].strip()


def enrich_features_with_categories(
    flat_data: Dict[str, Any],
    api_to_profiler_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Transform flat feature dict into { feature: { value, category } } using Approach 2 scoring.

    Args:
        flat_data: e.g. {"total_precipitation_mm": 716.0, "rainy_days": 48, ...}
        api_to_profiler_map: optional mapping; defaults to WEATHER_API_TO_PROFILER for weather

    Returns:
        e.g. {"total_precipitation_mm": {"value": 716.0, "category": "Good"}, ...}
    """
    if api_to_profiler_map is None:
        api_to_profiler_map = WEATHER_API_TO_PROFILER

    profiler = _get_profiler()
    result = {}

    for api_key, value in flat_data.items():
        if value is None or (isinstance(value, float) and str(value) == "nan"):
            result[api_key] = {"value": value, "category": "N/A"}
            continue

        profiler_key = api_to_profiler_map.get(api_key, api_key)
        try:
            score_info = profiler.score_feature_cto_method(profiler_key, float(value))
        except (KeyError, TypeError, ValueError):
            result[api_key] = {"value": value, "category": "N/A"}
            continue

        if "error" in score_info:
            result[api_key] = {"value": value, "category": "N/A"}
            continue

        category = _interpretation_to_category(score_info.get("interpretation", ""))
        result[api_key] = {"value": value, "category": category}

    return result


def get_feature_final_scores(
    flat_data: Dict[str, Any],
    api_to_profiler_map: Dict[str, str],
) -> Dict[str, float]:
    """
    Get Approach 2 final_score (0-100) per profiler feature for weighting.
    Returns { profiler_feature_name: final_score }.
    """
    profiler = _get_profiler()
    result = {}
    for api_key, value in flat_data.items():
        if value is None or (isinstance(value, float) and str(value) == "nan"):
            continue
        profiler_key = api_to_profiler_map.get(api_key, api_key)
        try:
            score_info = profiler.score_feature_cto_method(profiler_key, float(value))
        except (KeyError, TypeError, ValueError):
            continue
        if "error" in score_info or "final_score" not in score_info:
            continue
        result[profiler_key] = float(score_info["final_score"])
    return result


def compute_dimension_score(
    profiler_scores: Dict[str, float],
    dimension_name: str,
) -> Optional[float]:
    """
    Weighted average of feature scores for the given dimension.
    Uses FEATURE_WEIGHTS and DIMENSION_FEATURES from config.
    Returns None if no scored features in this dimension.
    """
    features = DIMENSION_FEATURES.get(dimension_name, [])
    if not features:
        return None
    total_w = 0.0
    weighted_sum = 0.0
    for f in features:
        if f not in profiler_scores:
            continue
        w = FEATURE_WEIGHTS.get(f, 1.0)
        total_w += w
        weighted_sum += w * profiler_scores[f]
    if total_w <= 0:
        return None
    return round(weighted_sum / total_w, 2)


# Task feature_values key -> profiler key (for gas; task only has distance_from_nearest_gas_station)
TASK_GAS_TO_PROFILER: Dict[str, str] = {
    "distance_from_nearest_gas_station": "nearest_gas_station_distance_miles",
}


def get_all_profiler_scores_from_task_feature_values(
    feature_values: Dict[str, Any],
) -> Dict[str, float]:
    """
    Map task feature_values to profiler keys and return { profiler_key: final_score } for all
    dimensions (Weather, Gas, Retail Proximity, Competition). Used for overall-score endpoint.
    """
    all_scores: Dict[str, float] = {}
    # Weather
    flat_w = {k: feature_values[k] for k in WEATHER_API_TO_PROFILER if feature_values.get(k) is not None}
    if flat_w:
        all_scores.update(get_feature_final_scores(flat_w, WEATHER_API_TO_PROFILER))
    # Gas (task only has distance_from_nearest_gas_station)
    flat_g = {k: feature_values[k] for k in TASK_GAS_TO_PROFILER if feature_values.get(k) is not None}
    if flat_g:
        all_scores.update(get_feature_final_scores(flat_g, TASK_GAS_TO_PROFILER))
    # Retail Proximity
    flat_r = {k: feature_values[k] for k in RETAILER_API_TO_PROFILER if feature_values.get(k) is not None}
    if flat_r:
        all_scores.update(get_feature_final_scores(flat_r, RETAILER_API_TO_PROFILER))
    # Competition
    flat_c = {k: feature_values[k] for k in COMPETITORS_API_TO_PROFILER if feature_values.get(k) is not None}
    if flat_c:
        all_scores.update(get_feature_final_scores(flat_c, COMPETITORS_API_TO_PROFILER))
    return all_scores


def compute_overall_score(profiler_scores: Dict[str, float]) -> Optional[float]:
    """
    Feature-weight-based overall score (0-100).
    If DIMENSION_WEIGHTS_FOR_OVERALL is set, overall = weighted avg of dimension scores.
    Else overall = weighted avg of all feature scores using FEATURE_WEIGHTS.
    """
    if not profiler_scores:
        return None
    if DIMENSION_WEIGHTS_FOR_OVERALL:
        dim_scores = {}
        for dim in DIMENSION_WEIGHTS_FOR_OVERALL:
            s = compute_dimension_score(profiler_scores, dim)
            if s is not None:
                dim_scores[dim] = s
        if not dim_scores:
            return None
        total_w = sum(DIMENSION_WEIGHTS_FOR_OVERALL.get(d, 0) for d in dim_scores)
        if total_w <= 0:
            return None
        return round(
            sum(dim_scores[d] * DIMENSION_WEIGHTS_FOR_OVERALL[d] for d in dim_scores) / total_w,
            2,
        )
    total_w = 0.0
    weighted_sum = 0.0
    for f, score in profiler_scores.items():
        w = FEATURE_WEIGHTS.get(f, 1.0)
        total_w += w
        weighted_sum += w * score
    if total_w <= 0:
        return None
    return round(weighted_sum / total_w, 2)


def enrich_gas_features_with_categories(
    nearest_station: Optional[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Score gas station features from nearest station.
    Returns { profiler_feature_name: { value, category } } for scorable Proforma features.
    """
    if not nearest_station:
        return {}
    flat = {
        "distance_miles": nearest_station.get("distance_miles"),
        "rating": nearest_station.get("rating"),
        "rating_count": nearest_station.get("rating_count"),
    }
    raw = enrich_features_with_categories(flat, api_to_profiler_map=GAS_API_TO_PROFILER)
    return {GAS_API_TO_PROFILER.get(k, k): v for k, v in raw.items()}


def enrich_competitors_features_with_categories(
    competitors: list,
    count: int,
) -> Dict[str, Dict[str, Any]]:
    """
    Score competitor features from count + nearest (first) competitor.
    Returns { profiler_feature_name: { value, category } }.
    """
    flat: Dict[str, Any] = {"count": count}
    if competitors:
        c1 = competitors[0]
        flat["competitor_1_distance_miles"] = c1.get("distance_miles")
        flat["competitor_1_google_rating"] = c1.get("rating")
        flat["competitor_1_rating_count"] = c1.get("user_rating_count")
    raw = enrich_features_with_categories(flat, api_to_profiler_map=COMPETITORS_API_TO_PROFILER)
    return {COMPETITORS_API_TO_PROFILER.get(k, k): v for k, v in raw.items()}


def enrich_retailers_features_with_categories(
    flat: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """
    Score retailer features (Costco, Walmart, Target, other grocery, food joints).
    flat should have keys like distance_from_nearest_costco, distance_from_nearest_walmart,
    distance_from_nearest_target, other_grocery_count_1mile, count_food_joints_0_5miles.
    Returns { profiler_feature_name: { value, category } }.
    """
    raw = enrich_features_with_categories(flat, api_to_profiler_map=RETAILER_API_TO_PROFILER)
    return {RETAILER_API_TO_PROFILER.get(k, k): v for k, v in raw.items()}
