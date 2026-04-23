"""Maps task features through PercentileProfiler for UI categories and weighted dimension/overall scores."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from app.site_analysis.modelling.ds.profiler import PercentileProfiler
from app.site_analysis.modelling.ds.feature_weights_config import (
    FEATURE_WEIGHTS,
    DIMENSION_FEATURES,
    DIMENSION_FEATURE_WEIGHTS,
    DIMENSION_WEIGHTS_FOR_OVERALL,
    FEATURE_DIRECTION,
)

# ── Singleton profiler (lazy-loaded) ─────────────────────────────────────────

_profiler: Optional[PercentileProfiler] = None
_ds_dir = Path(__file__).resolve().parent


# ── API key → Profiler key mappings ──────────────────────────────────────────

WEATHER_API_TO_PROFILER: Dict[str, str] = {
    "total_precipitation_mm": "weather_total_precipitation_mm",
    "rainy_days": "weather_rainy_days",
    "total_snowfall_cm": "weather_total_snowfall_cm",
    "days_below_freezing": "weather_days_below_freezing",
    "total_sunshine_hours": "weather_total_sunshine_hours",
    "days_pleasant_temp": "weather_days_pleasant_temp",
    "avg_daily_max_windspeed_ms": "weather_avg_daily_max_windspeed_ms",
}

GAS_API_TO_PROFILER: Dict[str, str] = {
    "distance_miles": "nearest_gas_station_distance_miles",
    "rating": "nearest_gas_station_rating",
    "rating_count": "nearest_gas_station_rating_count",
}

COMPETITORS_API_TO_PROFILER: Dict[str, str] = {
    "count": "competitors_count_4miles",
    "competitor_1_distance_miles": "competitor_1_distance_miles",
    "competitor_1_google_rating": "competitor_1_google_rating",
    "competitor_1_rating_count": "competitor_1_rating_count",
}

RETAILER_API_TO_PROFILER: Dict[str, str] = {
    "distance_from_nearest_costco": "distance_nearest_costco(5 mile)",
    "distance_from_nearest_walmart": "distance_nearest_walmart(5 mile)",
    "distance_from_nearest_target": "distance_nearest_target (5 mile)",
    "other_grocery_count_1mile": "other_grocery_count_1mile",
    "count_food_joints_0_5miles": "count_food_joints_0_5miles (0.5 mile)",
}

# Task feature_values key → profiler key (gas, used by overall-score endpoint)
TASK_GAS_TO_PROFILER: Dict[str, str] = {
    "distance_from_nearest_gas_station": "nearest_gas_station_distance_miles",
    "nearest_gas_station_rating": "nearest_gas_station_rating",
    "nearest_gas_station_rating_count": "nearest_gas_station_rating_count",
}


# ── Profiler singleton ────────────────────────────────────────────────────────

def _get_profiler() -> PercentileProfiler:
    global _profiler
    if _profiler is None:
        data_path = _ds_dir / "datasets" / "Proforma-v2-data-final (1).xlsx"
        if not data_path.exists():
            data_path = _ds_dir / "datasets" / "Proforma-v2-data-final.xlsx"
        if not data_path.exists():
            raise FileNotFoundError(
                f"Scoring dataset not found at {data_path}. "
                "Place Proforma-v2-data-final.xlsx under app/site_analysis/modelling/ds/datasets/."
            )
        kwargs: Dict[str, Any] = {"engine": "openpyxl"}
        if "Proforma-v2-data" in str(data_path):
            kwargs["header"] = 1
        df = pd.read_excel(data_path, **kwargs)
        with redirect_stdout(StringIO()):
            _profiler = PercentileProfiler(df)
    return _profiler


# ── Helpers ───────────────────────────────────────────────────────────────────

def _interpretation_to_category(interpretation: str) -> str:
    if not interpretation:
        return "N/A"
    return interpretation.split(" (")[0].strip()


def _interpret_from_final_score(final_score: float) -> str:
    if final_score >= 90:
        return "Excellent (top 10%)"
    if final_score >= 75:
        return "Very Good (top 25%)"
    if final_score >= 50:
        return "Good (above median)"
    if final_score >= 25:
        return "Fair (below median)"
    if final_score >= 10:
        return "Poor (bottom 25%)"
    return "Very Poor (bottom 10%)"


def _apply_direction_override(profiler_key: str, score_info: Dict[str, Any]) -> Dict[str, Any]:
    """Override direction and final_score using FEATURE_DIRECTION config."""
    direction_override = FEATURE_DIRECTION.get(profiler_key)
    if not direction_override:
        return score_info
    raw = score_info.get("raw_percentile")
    if raw is None:
        return score_info

    if direction_override == "moderate_is_best":
        dist = abs(raw - 50)
        final_score = max(0.0, 100.0 - 2.0 * dist)
        if dist <= 10:
            interpretation = "Excellent (near portfolio middle)"
        elif dist <= 25:
            interpretation = "Very Good (close to middle)"
        elif dist <= 40:
            interpretation = "Good (moderate)"
        elif dist <= 50:
            interpretation = "Fair (away from middle)"
        else:
            interpretation = "Poor (extreme)"
        return {**score_info, "final_score": round(final_score, 2),
                "direction": "moderate_is_best", "interpretation": interpretation}

    if direction_override == "lower_is_better":
        final_score = 100.0 - raw
        return {**score_info, "final_score": round(final_score, 2),
                "direction": "lower_is_better",
                "interpretation": _interpret_from_final_score(final_score)}

    if direction_override == "higher_is_better":
        final_score = raw
        return {**score_info, "final_score": round(final_score, 2),
                "direction": "higher_is_better",
                "interpretation": _interpret_from_final_score(final_score)}

    return score_info


# ── Public scoring API ────────────────────────────────────────────────────────

def score_feature_with_config(profiler_key: str, value: float) -> Optional[Dict[str, Any]]:
    """Profiler score + FEATURE_DIRECTION override; None if unscorable."""
    profiler = _get_profiler()
    try:
        info = profiler.score_feature_percentile(profiler_key, float(value))
    except (KeyError, TypeError, ValueError):
        return None
    if "error" in info:
        return None
    return _apply_direction_override(profiler_key, info)


def enrich_features_with_categories(
    flat_data: Dict[str, Any],
    api_to_profiler_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """{api_key: {value, category}}; default map WEATHER_API_TO_PROFILER."""
    if api_to_profiler_map is None:
        api_to_profiler_map = WEATHER_API_TO_PROFILER

    result: Dict[str, Dict[str, Any]] = {}
    for api_key, value in flat_data.items():
        if value is None or (isinstance(value, float) and str(value) == "nan"):
            result[api_key] = {"value": value, "category": "N/A"}
            continue
        profiler_key = api_to_profiler_map.get(api_key, api_key)
        score_info = score_feature_with_config(profiler_key, float(value))
        if score_info is None:
            result[api_key] = {"value": value, "category": "N/A"}
            continue
        category = _interpretation_to_category(score_info.get("interpretation", ""))
        result[api_key] = {"value": value, "category": category}
    return result


def get_feature_final_scores(
    flat_data: Dict[str, Any],
    api_to_profiler_map: Dict[str, str],
) -> Dict[str, float]:
    """Return { profiler_feature_name: final_score (0-100) } for all scorable features."""
    result: Dict[str, float] = {}
    for api_key, value in flat_data.items():
        if value is None or (isinstance(value, float) and str(value) == "nan"):
            continue
        profiler_key = api_to_profiler_map.get(api_key, api_key)
        score_info = score_feature_with_config(profiler_key, float(value))
        if score_info is None or "final_score" not in score_info:
            continue
        result[profiler_key] = float(score_info["final_score"])
    return result


def compute_dimension_score(
    profiler_scores: Dict[str, float],
    dimension_name: str,
) -> Optional[float]:
    """Weighted dimension score from DIMENSION_FEATURE_WEIGHTS; None if empty."""
    dim_weights = DIMENSION_FEATURE_WEIGHTS.get(dimension_name, {})
    features = DIMENSION_FEATURES.get(dimension_name, [])
    if not features:
        return None
    total_w = 0.0
    weighted_sum = 0.0
    for f in features:
        if f not in profiler_scores:
            continue
        w = dim_weights.get(f) or FEATURE_WEIGHTS.get(f, 1.0)
        total_w += w
        weighted_sum += w * profiler_scores[f]
    if total_w <= 0:
        return None
    return round(weighted_sum / total_w, 2)


def compute_overall_score(profiler_scores: Dict[str, float]) -> Optional[float]:
    """0–100 overall; uses DIMENSION_WEIGHTS_FOR_OVERALL or all-feature weights."""
    if not profiler_scores:
        return None
    if DIMENSION_WEIGHTS_FOR_OVERALL:
        dim_scores: Dict[str, float] = {}
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
            sum(dim_scores[d] * DIMENSION_WEIGHTS_FOR_OVERALL[d] for d in dim_scores) / total_w, 2
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


def get_all_profiler_scores_from_task_feature_values(
    feature_values: Dict[str, Any],
) -> Dict[str, float]:
    """Task feature_values → {profiler_key: final_score} for all dimensions present."""
    all_scores: Dict[str, float] = {}
    flat_w = {k: feature_values[k] for k in WEATHER_API_TO_PROFILER if feature_values.get(k) is not None}
    if flat_w:
        all_scores.update(get_feature_final_scores(flat_w, WEATHER_API_TO_PROFILER))
    flat_g = {k: feature_values[k] for k in TASK_GAS_TO_PROFILER if feature_values.get(k) is not None}
    if flat_g:
        all_scores.update(get_feature_final_scores(flat_g, TASK_GAS_TO_PROFILER))
    flat_r = {k: feature_values[k] for k in RETAILER_API_TO_PROFILER if feature_values.get(k) is not None}
    if flat_r:
        all_scores.update(get_feature_final_scores(flat_r, RETAILER_API_TO_PROFILER))
    flat_c = {k: feature_values[k] for k in COMPETITORS_API_TO_PROFILER if feature_values.get(k) is not None}
    if flat_c:
        all_scores.update(get_feature_final_scores(flat_c, COMPETITORS_API_TO_PROFILER))
    return all_scores


# ── Per-dimension enrichment helpers (used by route handlers) ─────────────────

def enrich_gas_features_with_categories(
    nearest_station: Optional[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
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
    competitors: List,
    count: int,
) -> Dict[str, Dict[str, Any]]:
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
    raw = enrich_features_with_categories(flat, api_to_profiler_map=RETAILER_API_TO_PROFILER)
    return {RETAILER_API_TO_PROFILER.get(k, k): v for k, v in raw.items()}
