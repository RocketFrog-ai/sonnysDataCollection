"""
Tape Cracker API routes.

Analysis flow (single fetch, async): POST /analyze-site enqueues a task that runs
geocode → fetch all features once (features/active) → quantile (v3) → optional narratives.
No second fetch: weather and all data come from that run only. Use GET with task_id to
get result (e.g. GET /result/{task_id}, GET /weather/data-by-task/{task_id}).

Progressive result: the task writes partial result to Redis after fetch (data available
quickly), then after quantile, then after narratives. Poll GET /result/{task_id} or
GET /task/{task_id} to get partial or full result without waiting for the full run.

Scoring / dimension summaries: app.modelling.ds. Quantile: app.modelling.ds.quantile_predictor (v3).
"""

import json
import logging
import redis
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

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
)
from app.server.models import (
    AnalyseRequest,
    GasStationRequest,
    DataFetchNearestGasRequest,
    RetailersRequest,
    CompetitorsDynamicsRequest,
    TaskResponse,
    TaskStatus,
    TaskStatusResponse,
    DimensionSummaryResponse,
    QuantileSummaryResponse,
)
from app.server.app import (
    get_competitors,
    get_traffic_lights,
    get_nearby_stores,
)
from app.features.active.nearbyStores.nearby_stores import get_nearby_stores_data
from app.features.active.nearbyGasStations.get_nearby_gas_stations import (
    get_nearby_gas_stations,
    get_nearest_gas_station_only,
)
from app.features.active.nearbyRetailers.get_nearby_retailers import get_nearby_retailers
from app.features.active.nearbyCompetitors.get_nearby_competitors import get_nearby_competitors
from app.celery.tasks import analyse_site
from app.celery.celery_app import celery_app
from app.modelling.ds.scorer import (
    enrich_features_with_categories,
    enrich_gas_features_with_categories,
    enrich_competitors_features_with_categories,
    enrich_retailers_features_with_categories,
    get_feature_final_scores,
    get_all_profiler_scores_from_task_feature_values,
    compute_dimension_score,
    compute_overall_score,
    WEATHER_API_TO_PROFILER,
    GAS_API_TO_PROFILER,
    COMPETITORS_API_TO_PROFILER,
    RETAILER_API_TO_PROFILER,
)
from app.modelling.ds.dimension_summary import (
    get_dimension_summary_approach2,
    build_full_profiling_rationale,
    _overall_score_to_category,
    DIMENSION_FEATURE_MAP,
)
from app.modelling.ds.quantile_display import get_category_for_quantile


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
    """
    if not features.address:
        raise HTTPException(status_code=400, detail="No site address provided")
    try:
        result = analyse_site.delay(features.address)
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
    Uses fetched data + quantile_result from the same analyse-site run (no extra fetch).
    """
    result = _get_result_or_partial(task_id)
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
    for metric_key in WEATHER_METRIC_CONFIG:
        value, unit = get_weather_metric_value_from_climate(climate, metric_key)
        if value is None:
            continue
        v3_key = WEATHER_METRIC_TO_V3_FEATURE.get(metric_key)
        fa = feature_analysis.get(v3_key, {}) if v3_key else {}
        pct = fa.get("adjusted_percentile")
        boundaries = fa.get("quantile_boundaries") or []
        dist_min = fa.get("dist_min")
        dist_max = fa.get("dist_max")
        wash_q = fa.get("wash_correlated_q")
        feature_q = fa.get("feature_quantile_adj")
        quantile_str = f"Q{int(wash_q)}" if wash_q is not None else (f"Q{int(feature_q)}" if feature_q is not None else None)
        category = get_category_for_quantile(wash_q) or get_category_for_quantile(feature_q)
        narrative = narrative_by_v3_key.get(v3_key, {}) if v3_key else {}
        summary = narrative.get("summary")
        business_impact = narrative.get("business_impact")
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
            "business_impact": business_impact,
            "impact_classification": impact_classification,
        })
    narratives_overall = (result.get("narratives") or {}).get("overall") or {}
    return {
        "task_id": task_id,
        "address": result.get("address"),
        "metrics": metrics,
        "overall": {
            "insight": narratives_overall.get("insight"),
            "observation": narratives_overall.get("observation"),
            "conclusion": narratives_overall.get("conclusion"),
        },
    }


# -----------------------------------------------------------------------------
# Gas stations
# -----------------------------------------------------------------------------

@router.post("/gas_station")
def get_gas_stations_endpoint(req: GasStationRequest):
    try:
        lat, lon = _geocode(req.address)
        api_key = calib.GOOGLE_MAPS_API_KEY
        if not api_key:
            raise HTTPException(status_code=503, detail="Google Maps API key not configured")
        radius = req.radius_miles if req.radius_miles is not None else 2.0
        max_results = min(req.max_results if req.max_results is not None else 6, 6)
        fetch_details = req.fetch_place_details if req.fetch_place_details is not None else True
        stations = get_nearby_gas_stations(
            api_key, lat, lon,
            radius_miles=radius,
            max_results=max_results,
            fetch_place_details=fetch_details,
        )

        feature_scores = {}
        dimension_score = None
        if stations:
            s0 = stations[0]
            feature_scores = enrich_gas_features_with_categories(s0)
            flat = {
                k: v for k, v in {
                    "distance_miles": s0.get("distance_miles"),
                    "rating": s0.get("rating"),
                    "rating_count": s0.get("rating_count"),
                }.items()
                if v is not None
            }
            if flat:
                scores = get_feature_final_scores(flat, GAS_API_TO_PROFILER)
                dimension_score = compute_dimension_score(scores, "Gas")

        return {
            "address": req.address,
            "lat": lat,
            "lon": lon,
            "radius_miles": radius,
            "stations": stations,
            "count": len(stations),
            "nearest_station_feature_scores": feature_scores,
            "dimension_score": {"Gas": dimension_score} if dimension_score is not None else {},
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Gas stations fetch failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/gas_station/summary/{task_id}")
def get_gas_station_summary_by_task(task_id: str):
    return _dimension_summary(task_id, "Gas")


@router.post("/datafetch/nearbygasstations")
def datafetch_nearby_gas_stations(req: DataFetchNearestGasRequest):
    try:
        lat, lon = _geocode(req.address)
        api_key = calib.GOOGLE_MAPS_API_KEY
        if not api_key:
            raise HTTPException(status_code=503, detail="Google Maps API key not configured")
        fetch_details = req.fetch_place_details if req.fetch_place_details is not None else True
        nearest = get_nearest_gas_station_only(api_key, lat, lon, fetch_place_details=fetch_details)
        feature_scores = enrich_gas_features_with_categories(nearest)
        dimension_score = None
        if nearest:
            flat = {
                k: nearest.get(k)
                for k in ("distance_miles", "rating", "rating_count")
                if nearest.get(k) is not None
            }
            if flat:
                scores = get_feature_final_scores(flat, GAS_API_TO_PROFILER)
                dimension_score = compute_dimension_score(scores, "Gas")
        return {
            "address": req.address,
            "lat": lat,
            "lon": lon,
            "nearest_gas_station": nearest,
            "feature_scores": feature_scores,
            "dimension_score": {"Gas": dimension_score} if dimension_score is not None else {},
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Datafetch nearest gas station failed")
        raise HTTPException(status_code=500, detail=str(e))


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
# Retailers
# -----------------------------------------------------------------------------

@router.post("/retailers")
def get_retailers_endpoint(req: RetailersRequest):
    try:
        lat, lon = _geocode(req.address)
        api_key = calib.GOOGLE_MAPS_API_KEY
        if not api_key:
            raise HTTPException(status_code=503, detail="Google Maps API key not configured")
        radius = req.radius_miles if req.radius_miles is not None else 0.5
        data = get_nearby_retailers(
            api_key, lat, lon,
            radius_miles=radius,
            fetch_place_details=False,
        )
        stores = get_nearby_stores_data(lat, lon)

        raw_list = data.get("retailers") or []
        grocery_within_1mi = [r for r in raw_list if r.get("category") == "Grocery" and (r.get("distance_miles") or 0) <= 1.0]
        food_within_0_5mi = [r for r in raw_list if r.get("category") == "Food Joint" and (r.get("distance_miles") or 0) <= 0.5]
        other_grocery_count = len(grocery_within_1mi)
        food_joint_count = len(food_within_0_5mi)

        def _avg(items, key):
            vals = [x[key] for x in items if x.get(key) is not None]
            return round(sum(vals) / len(vals), 2) if vals else None

        flat_retailer = {
            k: v
            for k, v in {
                "distance_from_nearest_costco": stores.get("distance_from_nearest_costco"),
                "distance_from_nearest_walmart": stores.get("distance_from_nearest_walmart"),
                "distance_from_nearest_target": stores.get("distance_from_nearest_target"),
                "other_grocery_count_1mile": other_grocery_count,
                "count_food_joints_0_5miles": food_joint_count,
            }.items()
            if v is not None
        }
        feature_scores = enrich_retailers_features_with_categories(flat_retailer) if flat_retailer else {}
        dimension_score = None
        if flat_retailer:
            scores = get_feature_final_scores(flat_retailer, RETAILER_API_TO_PROFILER)
            dimension_score = compute_dimension_score(scores, "Retail Proximity")

        def _anchor_item(display_name: str, details):
            if details is None:
                return {"name": display_name, "found": False, "message": "No store within 5 mile radius"}
            rest = {k: v for k, v in details.items() if k != "name"}
            return {"name": display_name, "found": True, "store_name": details.get("name"), **rest}

        retailers = [
            _anchor_item("Costco", stores.get("nearest_costco")),
            _anchor_item("Walmart", stores.get("nearest_walmart")),
            _anchor_item("Target", stores.get("nearest_target")),
            {
                "name": "Other groceries",
                "count_within_1_mile": other_grocery_count,
                "avg_rating": _avg(grocery_within_1mi, "rating"),
                "avg_distance_miles": _avg(grocery_within_1mi, "distance_miles"),
            },
            {
                "name": "Food joints",
                "count_within_0_5_miles": food_joint_count,
                "avg_rating": _avg(food_within_0_5mi, "rating"),
                "avg_distance_miles": _avg(food_within_0_5mi, "distance_miles"),
            },
        ]

        return {
            "address": req.address,
            "lat": lat,
            "lon": lon,
            "retailers": retailers,
            "feature_scores": feature_scores,
            "dimension_score": {"Retail Proximity": dimension_score} if dimension_score is not None else {},
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Retailers fetch failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/retailers/summary/{task_id}")
def get_retailers_summary_by_task(task_id: str):
    return _dimension_summary(task_id, "Retail Proximity")


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
# Competitors
# -----------------------------------------------------------------------------

@router.post("/competitors/dynamics")
def get_competitors_dynamics_endpoint(req: CompetitorsDynamicsRequest):
    try:
        lat, lon = _geocode(req.address)
        api_key = calib.GOOGLE_MAPS_API_KEY
        if not api_key:
            raise HTTPException(status_code=503, detail="Google Maps API key not configured")
        radius = req.radius_miles if req.radius_miles is not None else 4.0
        fetch_details = req.fetch_place_details if req.fetch_place_details is not None else True
        data = get_nearby_competitors(
            api_key, lat, lon,
            radius_miles=radius,
            fetch_place_details=fetch_details,
        )
        competitors = data.get("competitors") or []
        if competitors:
            try:
                from app.features.active.nearbyCompetitors.classify_competitor_types import classify_competitors
                competitors = classify_competitors(competitors)
            except Exception as e:
                logger.warning("Competitor classification skipped: %s", e)
            data["competitors"] = competitors

        count = data.get("count", 0)
        feature_scores = enrich_competitors_features_with_categories(competitors, count)
        flat_comp = {"count": count}
        if competitors:
            c1 = competitors[0]
            flat_comp["competitor_1_distance_miles"] = c1.get("distance_miles")
            flat_comp["competitor_1_google_rating"] = c1.get("rating")
            flat_comp["competitor_1_rating_count"] = c1.get("user_rating_count")
        flat_comp = {k: v for k, v in flat_comp.items() if v is not None}
        dimension_score = None
        if flat_comp:
            scores = get_feature_final_scores(flat_comp, COMPETITORS_API_TO_PROFILER)
            dimension_score = compute_dimension_score(scores, "Competition")
        return {
            "address": req.address,
            "lat": lat,
            "lon": lon,
            "radius_miles": radius,
            **data,
            "feature_scores": feature_scores,
            "dimension_score": {"Competition": dimension_score} if dimension_score is not None else {},
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Competitors dynamics fetch failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/competitors/dynamics/summary/{task_id}")
def get_competitors_dynamics_summary_by_task(task_id: str):
    return _dimension_summary(task_id, "Competition")


@router.post("/competitors")
def get_competitors_endpoint(features: AnalyseRequest):
    try:
        lat, lon = _geocode(features.address)
        data = get_competitors(lat, lon)
        return {"address": features.address, "lat": lat, "lon": lon, "data": data}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Competitors fetch failed")
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
    elif status in (TaskStatus.PENDING, TaskStatus.STARTED):
        partial = _get_partial_result_from_redis(task_id)
        if partial is not None:
            response.result = partial
    return response


@router.get("/result/{task_id}")
def get_result_by_task(task_id: str):
    """
    Get analyse-site result by task_id (GET, no extra fetch).
    Returns partial result as soon as fetch is done (data quickly), then quantile, then full.
    Poll this after POST /analyze-site to get fetched data, quantile result, and narratives.
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
    partial = _get_partial_result_from_redis(task_id)
    if partial is not None:
        return {"task_id": task_id, "status": "running", "result": partial}
    return {"task_id": task_id, "status": "pending", "result": None}


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
# Health
# -----------------------------------------------------------------------------

@router.get("/health")
def health_check():
    return {"status": "healthy", "service": "site-analysis-pipeline"}
