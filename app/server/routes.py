import json
import logging
import redis
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from celery.result import AsyncResult
from fastapi import APIRouter, HTTPException

from app.utils import common as calib
from app.server.config import (
    REDIS_HOST,
    REDIS_PORT,
    REDIS_DB,
    REDIS_PASSWORD,
    CELERY_BROKER_URL,
    CELERY_RESULT_BACKEND,
    DIMENSIONS,
    WEATHER_METRIC_CONFIG,
    WEATHER_METRIC_DISPLAY,
    WEATHER_METRIC_TO_V3_FEATURE,
    get_weather_metric_value_from_climate,
    COMPETITION_METRIC_DISPLAY,
    COMPETITION_METRIC_TO_V3_FEATURE,
    COMPETITION_RADIUS_MILES,
    nearest_brand_strength_from_quantile,
    RETAIL_METRIC_TO_V3_FEATURE,
    RETAIL_RADIUS_NEAR_MILES,
    RETAIL_RADIUS_FAR_MILES,
    RETAIL_SCORE_V3_KEYS,
    GAS_METRIC_TO_V3_FEATURE,
    GAS_RADIUS_NEAR_MILES,
    GAS_RADIUS_FAR_MILES,
    GAS_SCORE_V3_KEYS,
    is_high_traffic_gas_brand,
    SITE_SCORE_WEIGHTS,
    SITE_SCORE_CATEGORY_WEIGHTS,
    SITE_SCORE_FEATURE_CATEGORY,
)
from app.server.models import (
    AnalyseRequest,
    TaskResponse,
    TaskStatus,
    TaskStatusResponse,
    DimensionSummaryResponse,
    QuantileSummaryResponse,
)
from app.server.app import (
    get_traffic_lights,
    get_nearby_stores,
)
from app.celery.tasks import analyse_site
from app.celery.celery_app import celery_app
from app.modelling.ds.scorer import (
    enrich_features_with_categories,
    get_feature_final_scores,
    get_all_profiler_scores_from_task_feature_values,
    compute_dimension_score,
    compute_overall_score,
    WEATHER_API_TO_PROFILER,
)
from app.modelling.ds.dimension_summary import (
    get_dimension_summary_approach2,
    build_full_profiling_rationale,
    _overall_score_to_category,
    DIMENSION_FEATURE_MAP,
)
from app.modelling.ds.quantile_display import get_category_for_quantile
from app.features.active.nearbyCompetitors.classify_competitor_types import classify_competitors


# -----------------------------------------------------------------------------
# Router
# -----------------------------------------------------------------------------

logger = logging.getLogger(__name__)
router = APIRouter()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _geocode(address: str) -> Tuple[float, float]:
    if not address or not address.strip():
        raise HTTPException(status_code=400, detail="Address is required")
    geo = calib.get_lat_long(address)
    if not geo:
        raise HTTPException(status_code=400, detail="Could not geocode address (no results or API error)")
    lat = geo.get("lat")
    lon = geo.get("lon")
    if lat is None or lon is None:
        raise HTTPException(status_code=400, detail="Could not geocode address")
    return lat, lon


def get_redis_client():
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        password=REDIS_PASSWORD,
        decode_responses=True,
    )


# Redis key for partial result (written by Celery task after fetch / quantile / narratives)
RESULT_CACHE_KEY = "site_analysis:{task_id}"


def _get_partial_result_from_redis(task_id: str) -> Optional[Dict[str, Any]]:
    """Return partial or full result from Redis when task is still running (or just finished)."""
    try:
        client = get_redis_client()
        key = RESULT_CACHE_KEY.format(task_id=task_id)
        raw = client.get(key)
        if not raw:
            return None
        return json.loads(raw)
    except Exception as e:
        logger.warning("Redis partial result read failed: %s", e)
        return None


def _get_task_result_or_raise(task_id: str):
    """Return full task result; raises if task not SUCCESS or failed."""
    task_result = AsyncResult(task_id, app=celery_app)
    if task_result.state != TaskStatus.SUCCESS.value:
        if task_result.state == TaskStatus.FAILURE.value:
            raise HTTPException(
                status_code=422,
                detail=f"Task {task_id} failed: {str(task_result.result) if task_result.result else 'Task failed'}",
            )
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not completed yet (status={task_result.state}). Poll GET /result/{task_id} for partial or full result.",
        )
    result = task_result.result
    if result is None:
        raise HTTPException(status_code=422, detail=f"Task {task_id} completed but result is empty.")
    return result


def _get_result_or_partial(task_id: str):
    """Return result (partial from Redis or full from Celery). Raises if task failed or not found."""
    task_result = AsyncResult(task_id, app=celery_app)
    if task_result.state == TaskStatus.FAILURE.value:
        raise HTTPException(
            status_code=422,
            detail=f"Task {task_id} failed: {str(task_result.result) if task_result.result else 'Task failed'}",
        )
    if task_result.state == TaskStatus.SUCCESS.value and task_result.result is not None:
        return task_result.result
    partial = _get_partial_result_from_redis(task_id)
    if partial is not None:
        return partial
    raise HTTPException(
        status_code=404,
        detail=f"Task {task_id} not started or no result yet. Poll GET /result/{task_id}.",
    )


def _dimension_summary(task_id: str, dimension: str) -> dict:
    result = _get_task_result_or_raise(task_id)
    feature_values = result.get("feature_values") or {}

    if dimension not in DIMENSION_FEATURE_MAP:
        return DimensionSummaryResponse(
            task_id=task_id,
            dimension=dimension,
            predicted_tier="Insufficient Data",
            fit_score=0.0,
            features_scored=0,
            feature_breakdown={},
            discriminatory_power={},
            summary=f"No Approach 2 data for dimension {dimension}.",
            feature_values_slice={},
            feature_performance={},
        ).model_dump()

    try:
        a2 = get_dimension_summary_approach2(dimension, feature_values)
    except Exception as e:
        logger.warning(f"Approach 2 summary failed for {dimension}: {e}")
        a2 = None

    if not a2 or a2.get("features_scored", 0) == 0:
        return DimensionSummaryResponse(
            task_id=task_id,
            dimension=dimension,
            predicted_tier="Insufficient Data",
            fit_score=0.0,
            features_scored=0,
            feature_breakdown={},
            discriminatory_power={},
            summary=a2.get("summary", f"No scorable features for {dimension}.") if a2 else f"No data for {dimension}.",
            feature_values_slice={},
            feature_performance={},
        ).model_dump()

    scored = a2.get("feature_scores", [])
    fit_avg = sum(s.get("final_score", 0) for s in scored) / len(scored) if scored else 0
    feat_breakdown = {
        s["feature"]: {
            "value": s["value"],
            "raw_percentile": s.get("raw_percentile"),
            "final_score": s.get("final_score"),
            "category": s.get("category"),
        }
        for s in scored
    }
    feat_perf = {s["feature"]: s.get("category", "N/A") for s in scored}
    mapping = DIMENSION_FEATURE_MAP[dimension]
    fv_slice = {tk: feature_values.get(tk) for tk in mapping if feature_values.get(tk) is not None}

    return DimensionSummaryResponse(
        task_id=task_id,
        dimension=dimension,
        predicted_tier=a2.get("overall_category", "Insufficient Data"),
        fit_score=round(fit_avg, 1),
        features_scored=a2.get("features_scored", 0),
        feature_breakdown=feat_breakdown,
        discriminatory_power={},
        summary=a2.get("summary", ""),
        feature_values_slice=fv_slice,
        feature_performance=feat_perf,
    ).model_dump()


# -----------------------------------------------------------------------------
# Analysis (single-fetch + quantile via app.modelling.site_analysis)
# -----------------------------------------------------------------------------

@router.post("/analyze-site")
def analyze_site_endpoint(features: AnalyseRequest):
    """
    Enqueue site analysis: geocode → single fetch (features/active) → quantile (v3).
    Returns task_id; use GET /task/{task_id} for status and result (feature_values, quantile_result).
    tunnel_count (1–4): optional number of wash tunnels — strongly improves prediction accuracy.
    carwash_type_encoded (1–3): optional car wash type — improves quantile prediction.
    """
    if not features.address:
        raise HTTPException(status_code=400, detail="No site address provided")
    try:
        result = analyse_site.delay(
            features.address,
            tunnel_count=features.tunnel_count,
            carwash_type_encoded=features.carwash_type_encoded,
        )
        return TaskResponse(
            task_id=result.id,
            status=TaskStatus.PENDING,
            message="Site successfully submitted for Analysis",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing site: {str(e)}")


# -----------------------------------------------------------------------------
# Weather — only from task result (no separate fetch; use GET with task_id)
# -----------------------------------------------------------------------------
# Flow: POST /analyze-site → GET /result/{task_id} or GET /weather/data-by-task/{task_id}.
# Metric keys: dirt-trigger-days, dirt-deposit-severity, comfortable-washing-days, shutdown-risk-days.
# -----------------------------------------------------------------------------

@router.get("/weather/summary/{task_id}")
def get_weather_summary_by_task(task_id: str):
    """Dimension summary for Weather from task result (feature_values)."""
    return _dimension_summary(task_id, "Weather")


@router.get("/weather/data-by-task/{task_id}")
def get_weather_data_by_task(task_id: str):
    """
    Return all 4 weather metrics from the task result: value, unit, quantile_score
    (percentile e.g. 50.1), quantile (Q1–Q4), category (Poor/Fair/Good/Strong), min, max,
    and narrative summary. Category from v3: Q1→Poor, Q2→Fair, Q3→Good, Q4→Strong.
    Only available once task is fully complete (fetch + quantile + narratives).
    """
    result = _get_task_result_or_raise(task_id)
    climate = (result.get("fetched") or {}).get("climate") or {}
    feature_values = result.get("feature_values") or {}
    if not climate and not feature_values:
        raise HTTPException(
            status_code=422,
            detail=f"Task {task_id} has no climate or feature_values.",
        )
    if not climate or climate.get("error"):
        climate = {
            "rainy_days": feature_values.get("rainy_days"),
            "total_snowfall_cm": feature_values.get("total_snowfall_cm"),
            "days_pleasant_temp": feature_values.get("days_pleasant_temp"),
            "days_below_freezing": feature_values.get("days_below_freezing"),
        }
    quantile_result = result.get("quantile_result") or {}
    feature_analysis = quantile_result.get("feature_analysis") or {}
    narratives_feature = (result.get("narratives") or {}).get("feature") or []
    narrative_by_v3_key = {n["feature_key"]: n for n in narratives_feature if isinstance(n, dict) and "feature_key" in n}

    metrics = []
    percentiles_for_score = []  # 0–100 per metric for weather_score (25% weight each)
    for metric_key in WEATHER_METRIC_CONFIG:
        value, unit = get_weather_metric_value_from_climate(climate, metric_key)
        if value is None:
            continue
        v3_key = WEATHER_METRIC_TO_V3_FEATURE.get(metric_key)
        fa = feature_analysis.get(v3_key, {}) if v3_key else {}
        narrative = narrative_by_v3_key.get(v3_key, {}) if v3_key else {}
        
        pct = narrative.get("percentile")
        if pct is None:
            pct = fa.get("adjusted_percentile")
            
        if pct is not None:
            percentiles_for_score.append(float(pct))
            
        boundaries = fa.get("quantile_boundaries") or []
        dist_min = fa.get("dist_min")
        dist_max = fa.get("dist_max")
        
        wash_q = narrative.get("wash_q")
        if wash_q is None:
            wash_q = fa.get("wash_correlated_q")
            
        feature_q = fa.get("feature_quantile_adj")
        quantile_str = f"Q{int(wash_q)}" if wash_q is not None else (f"Q{int(feature_q)}" if feature_q is not None else None)
        
        category = narrative.get("category")
        if category is None:
            category = get_category_for_quantile(wash_q) or get_category_for_quantile(feature_q)
            
        summary = narrative.get("summary")
        impact_classification = narrative.get("impact_classification")
        min_val = float(dist_min) if dist_min is not None else (float(boundaries[0]) if len(boundaries) > 0 else None)
        max_val = float(dist_max) if dist_max is not None else (float(boundaries[-1]) if len(boundaries) > 0 else None)
        display_name, subtitle = WEATHER_METRIC_DISPLAY.get(metric_key, (metric_key, ""))
        metrics.append({
            "metric_key": metric_key,
            "display_name": display_name,
            "subtitle": subtitle,
            "value": float(value),
            "unit": unit,
            "min": min_val,
            "max": max_val,
            "quantile_score": float(pct) if pct is not None else None,
            "quantile": quantile_str,
            "category": category,
            "summary": summary,
            "impact_classification": impact_classification,
        })
    # weather_score: 25% of each metric's percentile (0–100) → mean of 4 percentiles, range 0–100
    weather_score = None
    if percentiles_for_score:
        weather_score = round(sum(percentiles_for_score) / len(percentiles_for_score), 1)
    narratives_overall = (result.get("narratives") or {}).get("overall") or {}
    has_overall = any(narratives_overall.get(k) for k in ("insight", "observation", "conclusion"))
    complete = bool(quantile_result and has_overall)
    return {
        "task_id": task_id,
        "address": result.get("address"),
        "complete": complete,
        "success": complete,
        "weather_score": weather_score,
        "metrics": metrics,
        "overall": {
            "insight": narratives_overall.get("insight"),
            "observation": narratives_overall.get("observation"),
            "conclusion": narratives_overall.get("conclusion"),
        },
    }


# -----------------------------------------------------------------------------
# Competition (nearby same-format car washes, 4-mile radius)
# -----------------------------------------------------------------------------

@router.get("/competition/data-by-task/{task_id}")
def get_competition_data_by_task(task_id: str):
    result = _get_task_result_or_raise(task_id)
    fetched = result.get("fetched") or {}
    competitors_data = fetched.get("competitors_data") or {}
    feature_values = result.get("feature_values") or {}
    quantile_result = result.get("quantile_result") or {}
    feature_analysis = quantile_result.get("feature_analysis") or {}
    narratives_comp = (result.get("narratives") or {}).get("competition") or {}

    competitors_list = competitors_data.get("competitors") or []
    count = competitors_data.get("count") or len(competitors_list) or feature_values.get("count", 0)
    nearest = competitors_list[0] if competitors_list else {}
    distance_to_nearest = nearest.get("distance_miles") if nearest else feature_values.get("competitor_1_distance_miles")
    nearest_rating = nearest.get("rating") if nearest else feature_values.get("competitor_1_google_rating")
    nearest_review_count = nearest.get("user_rating_count") or nearest.get("rating_count") if nearest else feature_values.get("competitor_1_rating_count")
    fa_quality = feature_analysis.get("competition_quality") or {}
    nearest_brand_strength = nearest_brand_strength_from_quantile(
        category=fa_quality.get("category"),
        wash_q=fa_quality.get("wash_correlated_q") or fa_quality.get("feature_quantile_adj"),
    )

    percentiles_for_score = []
    for v3_key in COMPETITION_METRIC_TO_V3_FEATURE.values():
        fa = feature_analysis.get(v3_key, {})
        pct = fa.get("adjusted_percentile")
        if pct is not None:
            percentiles_for_score.append(float(pct))
    competition_score = None
    if percentiles_for_score:
        competition_score = round(sum(percentiles_for_score) / len(percentiles_for_score), 1)

    # Run AI classification pipeline (scrape → Gemini AI) on each competitor.
    # classify_competitors handles: DB cache read, website scraping, AI call, DB cache write.
    # Each competitor dict already has "website" from Place Details (get_nearby_competitors).
    classified_list = classify_competitors(competitors_list) if competitors_list else []

    nearby_list = []
    for c in classified_list:
        classification = c.get("classification") or {}
        entry = {
            "name": c.get("name"),
            "rating": float(c["rating"]) if c.get("rating") is not None else None,
            "user_rating_count": c.get("user_rating_count") or c.get("rating_count"),
            "address": c.get("address"),
            "distance_miles": float(c["distance_miles"]) if c.get("distance_miles") is not None else None,
            # official_website: the URL that was actually used for classification
            # (Place Details websiteUri, or fallback found by find_official_website)
            "official_website": c.get("website"),
            # primary_carwash_type: AI-classified type (e.g. "Express Tunnel", "Full Service")
            "primary_carwash_type": classification.get("primary_type") if classification else None,
            # Full classification payload for richer downstream use
            "classification": classification if classification else None,
        }
        nearby_list.append(entry)

    has_comp_narrative = any(narratives_comp.get(k) for k in ("insight", "observation"))
    complete = bool(quantile_result and has_comp_narrative)

    return {
        "task_id": task_id,
        "address": result.get("address"),
        "complete": complete,
        "success": complete,
        "competition_score": competition_score,
        "nearby_car_washes": {
            "count": count,
            "list": nearby_list,
        },
        "nearest": {
            "distance_miles": float(distance_to_nearest) if distance_to_nearest is not None else None,
            "brand_strength": nearest_brand_strength,
            "rating": float(nearest_rating) if nearest_rating is not None else None,
            "user_rating_count": nearest_review_count,
        },
        "overall": {
            "insight": narratives_comp.get("insight"),
            "observation": narratives_comp.get("observation"),
        },
    }


# -----------------------------------------------------------------------------
# Retail — data-by-task
# -----------------------------------------------------------------------------

@router.get("/retail/data-by-task/{task_id}")
def get_retail_data_by_task(task_id: str):
    result = _get_task_result_or_raise(task_id)

    fetched = result.get("fetched") or {}
    retail_anchors_data = fetched.get("retail_anchors") or {}
    quantile_result = result.get("quantile_result") or {}
    feature_analysis = quantile_result.get("feature_analysis") or {}
    narratives_retail = (result.get("narratives") or {}).get("retail") or {}

    anchors = retail_anchors_data.get("anchors") or []
    within_1 = [a for a in anchors if a.get("distance_miles") is not None and a["distance_miles"] <= RETAIL_RADIUS_NEAR_MILES]
    within_3 = [a for a in anchors if a.get("distance_miles") is not None and RETAIL_RADIUS_NEAR_MILES < a["distance_miles"] <= RETAIL_RADIUS_FAR_MILES]
    nearest = anchors[0] if anchors else {}

    # Named anchor lookup: nearest per class from fetched v3 values (pre-computed)
    costco_dist = retail_anchors_data.get("costco_dist")
    walmart_dist = retail_anchors_data.get("walmart_dist")
    target_dist = retail_anchors_data.get("target_dist")

    def _nearest_of_type(types: List[str]) -> Optional[Dict[str, Any]]:
        for a in anchors:
            if a.get("type") in types:
                return {"name": a["name"], "type": a["type"], "distance_miles": a["distance_miles"]}
        return None

    key_anchors = {
        "warehouse_club": _nearest_of_type(["Warehouse Club"])
            or ({"name": None, "type": "Warehouse Club", "distance_miles": costco_dist} if costco_dist else None),
        "big_box": _nearest_of_type(["Supercenter", "Big Box / Discount", "Big Box"])
            or ({"name": None, "type": "Big Box", "distance_miles": walmart_dist or target_dist} if (walmart_dist or target_dist) else None),
        "grocery": _nearest_of_type(["Grocery Anchor"]),
        "food_beverage": _nearest_of_type(["Food & Beverage"]),
    }

    # retail_score: mean of key v3 percentiles
    percentiles_for_score = []
    for v3_key in RETAIL_SCORE_V3_KEYS:
        fa = feature_analysis.get(v3_key, {})
        pct = fa.get("adjusted_percentile")
        if pct is not None:
            percentiles_for_score.append(float(pct))
    retail_score = round(sum(percentiles_for_score) / len(percentiles_for_score), 1) if percentiles_for_score else None

    has_narrative = any(narratives_retail.get(k) for k in ("insight", "observation", "conclusion"))
    complete = bool(quantile_result and has_narrative)

    return {
        "task_id": task_id,
        "address": result.get("address"),
        "complete": complete,
        "success": complete,
        "retail_score": retail_score,
        "nearest_anchor": {
            "name": nearest.get("name"),
            "type": nearest.get("type"),
            "distance_miles": nearest.get("distance_miles"),
        },
        "key_anchors": key_anchors,
        "retail_anchors": {
            "within_1_mile": {
                "count": len(within_1),
                "list": [{"name": a["name"], "type": a["type"], "distance_miles": a["distance_miles"]} for a in within_1],
            },
            "within_3_miles": {
                "count": len(within_3),
                "list": [{"name": a["name"], "type": a["type"], "distance_miles": a["distance_miles"]} for a in within_3],
            },
        },
        "narratives": {
            "insight": narratives_retail.get("insight"),
            "observation": narratives_retail.get("observation"),
            "conclusion": narratives_retail.get("conclusion"),
        },
    }


# -----------------------------------------------------------------------------
# Gas — data-by-task
# -----------------------------------------------------------------------------

@router.get("/gas/data-by-task/{task_id}")
def get_gas_data_by_task(task_id: str):
    result = _get_task_result_or_raise(task_id)

    fetched = result.get("fetched") or {}
    quantile_result = result.get("quantile_result") or {}
    feature_analysis = quantile_result.get("feature_analysis") or {}
    narratives_gas = (result.get("narratives") or {}).get("gas") or {}

    gas_list_raw = fetched.get("gas_stations") or []

    # Normalise each station (distance_miles, rating, user_rating_count/rating_count, name)
    stations: list = []
    for s in gas_list_raw:
        d = s.get("distance_miles")
        stations.append({
            "name": s.get("name"),
            "distance_miles": float(d) if d is not None else None,
            "rating": float(s["rating"]) if s.get("rating") is not None else None,
            "user_rating_count": s.get("user_rating_count") or s.get("rating_count"),
            "high_traffic_brand": is_high_traffic_gas_brand(s.get("name")),
        })
    stations.sort(key=lambda s: (s.get("distance_miles") is None, s.get("distance_miles") or float("inf")))

    within_1 = [s for s in stations if s.get("distance_miles") is not None and s["distance_miles"] <= GAS_RADIUS_NEAR_MILES]
    #within_3 = [s for s in stations if s.get("distance_miles") is not None and s["distance_miles"] <= GAS_RADIUS_FAR_MILES]
    within_3 = [s for s in stations if s.get("distance_miles") is not None and GAS_RADIUS_NEAR_MILES < s["distance_miles"] <= GAS_RADIUS_FAR_MILES]
    nearest = stations[0] if stations else {}

    # Fallback nearest details from feature_values if no stations in fetched
    if not nearest:
        fv = result.get("feature_values") or {}
        nearest = {
            "name": None,
            "distance_miles": fv.get("distance_from_nearest_gas_station"),
            "rating": fv.get("nearest_gas_station_rating"),
            "user_rating_count": fv.get("nearest_gas_station_rating_count"),
            "high_traffic_brand": False,
        }

    # ---- gas_score: mean of key v3 percentiles ----
    percentiles_for_score = []
    for v3_key in GAS_SCORE_V3_KEYS:
        fa = feature_analysis.get(v3_key, {})
        pct = fa.get("adjusted_percentile")
        if pct is not None:
            percentiles_for_score.append(float(pct))
    gas_score = round(sum(percentiles_for_score) / len(percentiles_for_score), 1) if percentiles_for_score else None

    has_narrative = any(narratives_gas.get(k) for k in ("insight", "observation", "conclusion"))
    complete = bool(quantile_result and has_narrative)

    return {
        "task_id": task_id,
        "address": result.get("address"),
        "complete": complete,
        "success": complete,
        "gas_score": gas_score,
        "nearest": {
            "name": nearest.get("name"),
            "distance_miles": nearest.get("distance_miles"),
            "high_traffic_brand": nearest.get("high_traffic_brand", False),
        },
        "gas_stations": {
            "within_1_mile": {
                "count": len(within_1),
                "list": within_1,
            },
            "within_3_miles": {
                "count": len(within_3),
                "list": within_3,
            },
        },
        "overall": {
            "insight": narratives_gas.get("insight"),
            "observation": narratives_gas.get("observation"),
            "conclusion": narratives_gas.get("conclusion"),
        },
    }


# -----------------------------------------------------------------------------
# Traffic lights
# -----------------------------------------------------------------------------

@router.post("/traffic-lights")
def get_traffic_lights_endpoint(features: AnalyseRequest):
    try:
        lat, lon = _geocode(features.address)
        data = get_traffic_lights(lat, lon)
        return {"address": features.address, "lat": lat, "lon": lon, "data": data}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Traffic lights fetch failed")
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------------------------
# Nearby stores
# -----------------------------------------------------------------------------

@router.post("/nearby-stores")
def get_nearby_stores_endpoint(features: AnalyseRequest):
    try:
        lat, lon = _geocode(features.address)
        data = get_nearby_stores(lat, lon)
        return {"address": features.address, "lat": lat, "lon": lon, "data": data}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Nearby stores fetch failed")
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------------------------
# Task status & profiling
# -----------------------------------------------------------------------------

@router.get("/task/{task_id}", response_model=TaskStatusResponse)
def get_task_status(task_id: str):
    """
    Task status and result. While task is running, result may contain partial data
    (fetched first, then quantile, then full) from Redis so clients can show progress.
    """
    task_result = AsyncResult(task_id, app=celery_app)
    status_str = task_result.state
    try:
        status = TaskStatus(status_str)
    except ValueError:
        status = TaskStatus.PENDING

    response = TaskStatusResponse(
        task_id=task_id,
        status=status,
        result=None,
        error=None,
        created_at=None,
        completed_at=None,
    )
    if status == TaskStatus.SUCCESS:
        response.result = task_result.result
    elif status == TaskStatus.FAILURE:
        response.error = str(task_result.result) if task_result.result else "Task failed"
    return response


@router.get("/result/{task_id}")
def get_result_by_task(task_id: str):
    """
    Get analyse-site result by task_id. Returns full result only once all stages
    (fetch + quantile + narratives) are complete. Poll until status = success.
    """
    task_result = AsyncResult(task_id, app=celery_app)
    status_str = task_result.state
    if status_str == TaskStatus.SUCCESS.value:
        return {"task_id": task_id, "status": "success", "result": task_result.result}
    if status_str == TaskStatus.FAILURE.value:
        return {
            "task_id": task_id,
            "status": "failure",
            "error": str(task_result.result) if task_result.result else "Task failed",
            "result": None,
        }
    return {"task_id": task_id, "status": task_result.state.lower(), "result": None}


@router.get("/quantile/{task_id}")
def get_quantile_result(task_id: str):
    """Return v3 quantile prediction result from the task (single-fetch run)."""
    result = _get_task_result_or_raise(task_id)
    quantile_result = result.get("quantile_result")
    if quantile_result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} has no quantile_result (task may predate quantile run).",
        )
    return {"task_id": task_id, "quantile_result": quantile_result}


@router.get("/narratives/{task_id}")
def get_narratives_by_task(task_id: str):
    """Return per-feature and overall narratives from the task when present (run_narratives=True)."""
    result = _get_task_result_or_raise(task_id)
    narratives = result.get("narratives")
    return {"task_id": task_id, "narratives": narratives}


@router.get("/overall-score/{task_id}")
def get_overall_score(task_id: str):
    result = _get_task_result_or_raise(task_id)
    feature_values = result.get("feature_values") or {}
    if not feature_values:
        raise HTTPException(status_code=422, detail=f"Task {task_id} has no feature_values.")
    profiler_scores = get_all_profiler_scores_from_task_feature_values(feature_values)
    if not profiler_scores:
        raise HTTPException(status_code=422, detail="Could not score any features for this task.")
    overall = compute_overall_score(profiler_scores)
    dimension_scores = {}
    for dim in DIMENSIONS:
        s = compute_dimension_score(profiler_scores, dim)
        if s is not None:
            dimension_scores[dim] = s
    return {
        "task_id": task_id,
        "overall_score": overall,
        "dimension_scores": dimension_scores,
    }


@router.get("/profiling/summary/{task_id}")
def get_full_profiling_summary(task_id: str):
    """
    Full profiling summary from task result (feature_values + optional quantile_result).
    When the task was run with quantile prediction, expected_volume and optional
    quantile_result/narratives are included.
    """
    result = _get_task_result_or_raise(task_id)
    feature_values = result.get("feature_values") or {}
    profiler_scores = get_all_profiler_scores_from_task_feature_values(feature_values)
    if not profiler_scores:
        raise HTTPException(status_code=422, detail=f"Task {task_id}: no scorable features.")
    overall_score = compute_overall_score(profiler_scores)
    dimension_results = {}
    for dim in DIMENSIONS:
        dimension_results[dim] = get_dimension_summary_approach2(dim, feature_values)
    rationale = build_full_profiling_rationale(overall_score, dimension_results)
    dim_scores = {}
    for dim in DIMENSIONS:
        s = compute_dimension_score(profiler_scores, dim)
        res = dimension_results.get(dim, {})
        fs_list = res.get("feature_scores") or []
        fit_avg = sum(f.get("final_score", 0) for f in fs_list) / len(fs_list) if fs_list else 0
        dim_scores[dim] = {
            "predicted": res.get("overall_category", "Insufficient Data"),
            "fit_score": round(fit_avg, 1),
            "features_scored": res.get("features_scored", 0),
        }
    expected_volume = {}
    quantile_result = result.get("quantile_result")
    if quantile_result and isinstance(quantile_result, dict):
        wr = quantile_result.get("predicted_wash_range") or {}
        if wr:
            expected_volume = {
                "min": wr.get("min"),
                "max": wr.get("max"),
                "label": wr.get("label"),
            }
    response = QuantileSummaryResponse(
        task_id=task_id,
        overall_tier=_overall_score_to_category(overall_score),
        overall_fit_score=round(overall_score, 1),
        expected_volume=expected_volume,
        dimensions=dim_scores,
        vote={},
        strengths=[],
        weaknesses=[],
        rationale=rationale,
    ).model_dump()
    if quantile_result is not None:
        response["quantile_result"] = quantile_result
    if result.get("narratives") is not None:
        response["narratives"] = result["narratives"]
    return response


# -----------------------------------------------------------------------------
# Overall site score (business-logic weights × v3 percentiles) + quantile
# -----------------------------------------------------------------------------

@router.get("/overall/{task_id}")
def get_overall(task_id: str):
    """
    Returns:
    - site_score (0–100): weighted composite using business-logic feature weights.
      Formula: Σ (adjusted_percentile_i × weight_i) for each feature in SITE_SCORE_WEIGHTS.
    - category_scores: per-category weighted scores (Weather, Competition, Retail, Gas).
    - feature_scores: per-feature adjusted_percentile, weight, and weighted contribution.
    - predicted_quantile: Q1–Q4 tier from v3 model.
    - predicted_tier: label (High Performer, etc.)
    - expected_annual_volume: min / max / label (cars/year).
    - quantile_probabilities: model confidence per tier.
    """
    result = _get_task_result_or_raise(task_id)
    quantile_result = result.get("quantile_result") or {}
    feature_analysis = quantile_result.get("feature_analysis") or {}

    from typing import Optional, List, Any

    def _overall_site_analysis_verdict(
        *,
        predicted_tier: Optional[str],
        expected_volume_label: Optional[str],
        site_score: Optional[float],
        strengths: Optional[List[Any]],
        weaknesses: Optional[List[Any]],
    ) -> str:
        """
        Plain-English verdict for the overall site analysis.
        Keep it short and business-readable; avoid technical language.
        """
        tier_part = f"This site is projected as **{predicted_tier}**" if predicted_tier else "This site has no quantile prediction yet"
        vol_part = f"with an expected annual volume of **{expected_volume_label}**" if expected_volume_label else ""
        score_part = f"Overall site score is **{site_score:.1f}/100**." if site_score is not None else ""

        def _top_labels(items: Optional[List[Any]], n: int = 2) -> str:
            if not items:
                return ""
            labels = []
            for it in items:
                if isinstance(it, dict):
                    lbl = it.get("label") or it.get("feature") or it.get("name")
                    if lbl:
                        # Clean up technical labels for the plain English verdict
                        lbl = str(lbl)
                        lbl = lbl.replace("Costco Distance (mi, 99=none)", "Distance to Costco")
                        lbl = lbl.replace("distance_nearest_walmart(5 mile)", "Distance to Walmart")
                        lbl = lbl.replace("distance_nearest_target (5 mile)", "Distance to Target")
                        lbl = lbl.replace("Nearest Competitor (miles)", "Distance to Nearest Competitor")
                        # Strip any other parentheticals that leaked through
                        import re
                        lbl = re.sub(r'\s*\(.*?\)', '', lbl).strip()
                        labels.append(lbl)
                if len(labels) >= n:
                    break
            return ", ".join(labels)

        s_lbls = _top_labels(strengths, 2)
        w_lbls = _top_labels(weaknesses, 2)

        sw_part = ""
        if s_lbls and w_lbls:
            sw_part = f" Biggest strengths are **{s_lbls}**, while key weaknesses are **{w_lbls}**."
        elif s_lbls:
            sw_part = f" Biggest strengths are **{s_lbls}**."
        elif w_lbls:
            sw_part = f" Key weaknesses are **{w_lbls}**."

        # Keep to 1–2 sentences.
        base = f"{tier_part} {vol_part}".strip()
        if base.endswith(".") is False:
            base = base + "."
        return " ".join([p for p in [base, score_part, sw_part.strip()] if p]).strip()


    if not quantile_result:
        feature_values = result.get("feature_values") or {}
        site_score = None
        category_scores: dict = {}
        feature_scores: dict = {}
        if feature_values:
            profiler_scores = get_all_profiler_scores_from_task_feature_values(feature_values)
            if profiler_scores:
                overall = compute_overall_score(profiler_scores)
                site_score = round(overall, 1)
        return {
            "task_id": task_id,
            "status": "no_quantile_result",
            "message": f"Task {task_id} has no v3 quantile_result (task may predate v3 or quantile was not run).",
            "site_score": site_score,
            # category_scores / feature_scores intentionally omitted for now
            "predicted_quantile": None,
            "predicted_tier": None,
            "expected_annual_volume": None,
            "quantile_probabilities": {},
            "tunnel_count": (result.get("feature_values") or {}).get("tunnel_count"),
            "carwash_type_encoded": (result.get("feature_values") or {}).get("carwash_type_encoded"),
            "overall_site_analysis_verdict": _overall_site_analysis_verdict(
                predicted_tier=None,
                expected_volume_label=None,
                site_score=site_score,
                strengths=None,
                weaknesses=None,
            ),
        }

    # Compute site score
    weighted_sum = 0.0
    total_weight_used = 0.0
    feature_scores: dict = {}
    category_accum: dict = {cat: {"weighted_sum": 0.0, "weight_used": 0.0} for cat in SITE_SCORE_CATEGORY_WEIGHTS}

    for v3_key, weight in SITE_SCORE_WEIGHTS.items():
        fa = feature_analysis.get(v3_key) or {}
        pct = fa.get("adjusted_percentile")
        if pct is None:
            continue
        contribution = float(pct) * weight
        weighted_sum += contribution
        total_weight_used += weight
        category = SITE_SCORE_FEATURE_CATEGORY.get(v3_key, "Other")
        feature_scores[v3_key] = {
            "label": fa.get("label", v3_key),
            "category": category,
            "adjusted_percentile": round(float(pct), 1),
            "weight": weight,
            "weighted_contribution": round(contribution, 3),
            "imputed": fa.get("imputed", False),
        }
        if category in category_accum:
            category_accum[category]["weighted_sum"] += contribution
            category_accum[category]["weight_used"] += weight

    # Normalise: if some features were missing, scale score to 0–100 range
    site_score = round(weighted_sum / total_weight_used, 1) if total_weight_used > 0 else None

    # Small business rule boost: Express + tunnel_count > 1 tends to perform better.
    # Apply a slight uplift to the overall site score (2–3 points) for scoring display.
    fv = result.get("feature_values") or {}
    try:
        tc_val = fv.get("tunnel_count")
        cw_val = fv.get("carwash_type_encoded")
        if site_score is not None and cw_val is not None and int(cw_val) == 1 and tc_val is not None and int(tc_val) > 1:
            site_score = min(100.0, round(site_score + 2.5, 1))
    except Exception:
        pass

    # Per-category scores (0–100)
    category_scores: dict = {}
    for cat, cat_weight in SITE_SCORE_CATEGORY_WEIGHTS.items():
        acc = category_accum.get(cat, {})
        w_used = acc.get("weight_used", 0)
        w_sum = acc.get("weighted_sum", 0.0)
        category_scores[cat] = {
            "score": round(w_sum / w_used, 1) if w_used > 0 else None,
            "category_weight": cat_weight,
            "features_scored": sum(1 for f in feature_scores.values() if f["category"] == cat),
        }

    # Quantile fields
    predicted_q = quantile_result.get("predicted_wash_quantile")
    wash_range = quantile_result.get("predicted_wash_range") or {}
    proba = quantile_result.get("quantile_probabilities") or {}

    strengths = quantile_result.get("strengths") or []
    weaknesses = quantile_result.get("weaknesses") or []

    response = {
        "task_id": task_id,
        "address": result.get("address"),
        "site_score": site_score,
        # category_scores / feature_scores intentionally omitted for now
        "predicted_quantile": f"Q{predicted_q}" if predicted_q else None,
        "predicted_tier": quantile_result.get("predicted_wash_tier"),
        "expected_annual_volume": {
            "min": wash_range.get("min"),
            "max": wash_range.get("max"),
            "label": wash_range.get("label"),
        },
        "quantile_probabilities": {
            f"Q{k}": round(v * 100, 1) for k, v in proba.items()
        },
        "tunnel_count": (result.get("feature_values") or {}).get("tunnel_count"),
        "carwash_type_encoded": (result.get("feature_values") or {}).get("carwash_type_encoded"),
        "overall_site_analysis_verdict": _overall_site_analysis_verdict(
            predicted_tier=quantile_result.get("predicted_wash_tier"),
            expected_volume_label=(wash_range.get("label") if wash_range else None),
            site_score=site_score,
            strengths=strengths,
            weaknesses=weaknesses,
        ),
    }
    return response


# -----------------------------------------------------------------------------
# Health
# -----------------------------------------------------------------------------

@router.get("/health")
def health_check():
    return {"status": "healthy", "service": "site-analysis-pipeline"}
