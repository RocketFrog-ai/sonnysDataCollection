import logging
import requests
from typing import Any, Dict, List, Optional, Tuple

from celery.result import AsyncResult
from fastapi import APIRouter, HTTPException

from app.utils import common as calib
from app.site_analysis.server.config import (
    WEATHER_METRIC_CONFIG,
    WEATHER_METRIC_DISPLAY,
    get_weather_metric_value_from_climate,
    RETAIL_RADIUS_NEAR_MILES,
    RETAIL_RADIUS_FAR_MILES,
    GAS_RADIUS_NEAR_MILES,
    GAS_RADIUS_FAR_MILES,
    is_high_traffic_gas_brand,
)
from app.site_analysis.server.models import (
    AnalyseRequest,
    SiteContextRequest,
    TaskResponse,
    TaskStatus,
    TaskStatusResponse,
)
from app.site_analysis.features.active.trafficLights.nearby_traffic_lights import get_traffic_lights_summary
from app.site_analysis.features.active.nearbyStores.nearby_stores import get_nearby_stores_data
from app.site_analysis.features.active.nearbyCompetitors.classify_competitor_types import classify_competitors
from app.site_analysis.server.db_cache import get_all_site_analysis_cache
from app.site_analysis.modelling.site_analysis import run_site_analysis
from app.site_analysis.modelling.ai import (
    summarize_weather,
    summarize_competition,
    summarize_retail,
    summarize_gas,
)
from app.celery.celery_app import celery_app

logger = logging.getLogger(__name__)
router = APIRouter()
PLACE_DETAILS_URL = "https://places.googleapis.com/v1/places/"

# Retail / map: cap how many anchors we show per type.
_ANCHOR_DISPLAY_CAP = 5


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _cap_anchors_by_type(anchor_list: List[Dict[str, Any]], cap: int) -> List[Dict[str, Any]]:
    seen: Dict[str, int] = {}
    out: List[Dict[str, Any]] = []
    for a in anchor_list:
        t = a.get("type", "Other")
        n = seen.get(t, 0)
        if n < cap:
            out.append(a)
            seen[t] = n + 1
    return out


def _lat_lon_from_address_or_400(address: str) -> Tuple[float, float]:
    """Geocode for HTTP handlers: map geocode failures to 400."""
    try:
        return calib.resolve_lat_lon(address)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


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
            detail=(
                f"Task {task_id} not completed yet (status={task_result.state}). "
                "Poll GET /task/{task_id} or GET /result/{task_id} until status is success."
            ),
        )
    result = task_result.result
    if result is None:
        raise HTTPException(status_code=422, detail=f"Task {task_id} completed but result is empty.")
    return result


def _retail_anchor_category(anchor_type: Optional[str], anchor_name: Optional[str]) -> str:
    at = (anchor_type or "").strip().lower()
    an = (anchor_name or "").strip().lower()
    if "warehouse club" in at or "costco" in an or "sam's club" in an or "bj's" in an:
        return "costco"
    if "target" in an:
        return "target"
    if "supercenter" in at or "walmart" in an:
        return "walmart"
    if "big box" in at:
        return "big_box"
    if "grocery" in at:
        return "grocery_anchor"
    if "food" in at:
        return "food_beverage"
    return "retail_anchor"


def _resolve_marker_coordinates(raw: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """Resolve marker coords from embedded lat/lon, Place Details by place_id, or address geocode."""
    lat = raw.get("latitude")
    lon = raw.get("longitude")
    if lat is not None and lon is not None:
        return float(lat), float(lon)

    place_id = raw.get("place_id")
    api_key = calib.GOOGLE_MAPS_API_KEY or ""
    if place_id and api_key:
        try:
            resp = requests.get(
                f"{PLACE_DETAILS_URL}{place_id}",
                headers={
                    "Content-Type": "application/json",
                    "X-Goog-Api-Key": api_key,
                    "X-Goog-FieldMask": "location",
                },
                timeout=8,
            )
            resp.raise_for_status()
            loc = (resp.json() or {}).get("location") or {}
            dlat = loc.get("latitude")
            dlon = loc.get("longitude")
            if dlat is not None and dlon is not None:
                return float(dlat), float(dlon)
        except Exception:
            pass

    address = raw.get("address")
    if address:
        try:
            geo = calib.get_lat_long(address)
            if geo and geo.get("lat") is not None and geo.get("lon") is not None:
                return float(geo["lat"]), float(geo["lon"])
        except Exception:
            pass

    return None, None


# -----------------------------------------------------------------------------
# Analyse-site: kick off the external-data fetch pipeline
# -----------------------------------------------------------------------------

@router.post("/analyze-site")
def analyze_site_endpoint(features: AnalyseRequest):
    """
    Enqueue the fetch pipeline: geocode → fetch weather / competitors / gas / retail (in parallel).
    Returns task_id; poll GET /task/{task_id} until success, then read the per-dimension
    /{dimension}/data-by-task endpoints (fast) and /{dimension}/summary-by-task endpoints (LLM).
    """
    if not features.address:
        raise HTTPException(status_code=400, detail="No site address provided")
    try:
        result = run_site_analysis.delay(features.address)
        return TaskResponse(
            task_id=result.id,
            status=TaskStatus.PENDING,
            message="Site successfully submitted for analysis",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing site: {str(e)}")


# -----------------------------------------------------------------------------
# Per-dimension raw DATA (fast — no LLM). Reads only the stored `fetched` payload.
# -----------------------------------------------------------------------------

@router.get("/weather/data-by-task/{task_id}")
def get_weather_data_by_task(task_id: str):
    """Raw weather metrics (rainy days, snowfall, comfortable days, freezing days) from the fetch."""
    result = _get_task_result_or_raise(task_id)
    climate = (result.get("fetched") or {}).get("climate") or {}

    metrics = []
    for metric_key in WEATHER_METRIC_CONFIG:
        value, unit = get_weather_metric_value_from_climate(climate, metric_key)
        if value is None:
            continue
        display_name, subtitle = WEATHER_METRIC_DISPLAY.get(metric_key, (metric_key, ""))
        metrics.append({
            "metric_key": metric_key,
            "display_name": display_name,
            "subtitle": subtitle,
            "value": float(value),
            "unit": unit,
        })

    return {
        "task_id": task_id,
        "address": result.get("address"),
        "complete": True,
        "metrics": metrics,
    }


@router.get("/competition/data-by-task/{task_id}")
def get_competition_data_by_task(task_id: str):
    """Raw nearby same-format car washes (within 4 miles) from the fetch."""
    result = _get_task_result_or_raise(task_id)
    competitors_data = (result.get("fetched") or {}).get("competitors_data") or {}
    competitors_list = competitors_data.get("competitors") or []
    count = competitors_data.get("count") or len(competitors_list)

    # Classify each competitor's car-wash type (DB cache → website scrape → AI). No ds dependency.
    classified_list = classify_competitors(competitors_list) if competitors_list else []

    nearby_list = []
    for c in classified_list:
        classification = c.get("classification") or {}
        nearby_list.append({
            "name": c.get("name"),
            "rating": float(c["rating"]) if c.get("rating") is not None else None,
            "user_rating_count": c.get("user_rating_count") or c.get("rating_count"),
            "address": c.get("address"),
            "distance_miles": float(c["distance_miles"]) if c.get("distance_miles") is not None else None,
            # official_website: the URL actually used for classification (Place Details or found fallback)
            "official_website": c.get("website"),
            # primary_carwash_type: AI-classified type (e.g. "Express Tunnel", "Full Service")
            "primary_carwash_type": classification.get("primary_type") if classification else None,
        })

    nearest = nearby_list[0] if nearby_list else {}
    return {
        "task_id": task_id,
        "address": result.get("address"),
        "complete": True,
        "nearby_car_washes": {"count": count, "list": nearby_list},
        "nearest": {
            "name": nearest.get("name"),
            "distance_miles": nearest.get("distance_miles"),
            "rating": nearest.get("rating"),
            "user_rating_count": nearest.get("user_rating_count"),
        },
    }


@router.get("/retail/data-by-task/{task_id}")
def get_retail_data_by_task(task_id: str):
    """Raw nearby retail anchors (within 1 and 3 miles) from the fetch."""
    result = _get_task_result_or_raise(task_id)
    retail_anchors_data = (result.get("fetched") or {}).get("retail_anchors") or {}
    anchors = retail_anchors_data.get("anchors") or []

    within_1 = [a for a in anchors if a.get("distance_miles") is not None and a["distance_miles"] <= RETAIL_RADIUS_NEAR_MILES]
    within_3 = [a for a in anchors if a.get("distance_miles") is not None and RETAIL_RADIUS_NEAR_MILES < a["distance_miles"] <= RETAIL_RADIUS_FAR_MILES]
    within_1 = _cap_anchors_by_type(within_1, _ANCHOR_DISPLAY_CAP)
    within_3 = _cap_anchors_by_type(within_3, _ANCHOR_DISPLAY_CAP)
    nearest = anchors[0] if anchors else {}

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

    return {
        "task_id": task_id,
        "address": result.get("address"),
        "complete": True,
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
    }


@router.get("/gas/data-by-task/{task_id}")
def get_gas_data_by_task(task_id: str):
    """Raw nearby gas stations (within 1 and 3 miles) from the fetch."""
    result = _get_task_result_or_raise(task_id)
    gas_list_raw = (result.get("fetched") or {}).get("gas_stations") or []

    stations = []
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
    within_3 = [s for s in stations if s.get("distance_miles") is not None and GAS_RADIUS_NEAR_MILES < s["distance_miles"] <= GAS_RADIUS_FAR_MILES]
    nearest = stations[0] if stations else {}

    return {
        "task_id": task_id,
        "address": result.get("address"),
        "complete": True,
        "nearest": {
            "name": nearest.get("name"),
            "distance_miles": nearest.get("distance_miles"),
            "high_traffic_brand": nearest.get("high_traffic_brand", False),
        },
        "gas_stations": {
            "within_1_mile": {"count": len(within_1), "list": within_1},
            "within_3_miles": {"count": len(within_3), "list": within_3},
        },
    }


# -----------------------------------------------------------------------------
# Per-dimension AI SUMMARIES (on-demand — one LLM call, grounded on raw fetched data).
# -----------------------------------------------------------------------------

@router.get("/weather/summary-by-task/{task_id}")
def get_weather_summary_by_task(task_id: str):
    result = _get_task_result_or_raise(task_id)
    climate = (result.get("fetched") or {}).get("climate") or {}
    return {"task_id": task_id, "summary": summarize_weather(climate)}


@router.get("/competition/summary-by-task/{task_id}")
def get_competition_summary_by_task(task_id: str):
    result = _get_task_result_or_raise(task_id)
    competitors_data = (result.get("fetched") or {}).get("competitors_data") or {}
    return {"task_id": task_id, "summary": summarize_competition(competitors_data)}


@router.get("/retail/summary-by-task/{task_id}")
def get_retail_summary_by_task(task_id: str):
    result = _get_task_result_or_raise(task_id)
    retail_anchors = (result.get("fetched") or {}).get("retail_anchors") or {}
    return {"task_id": task_id, "summary": summarize_retail(retail_anchors)}


@router.get("/gas/summary-by-task/{task_id}")
def get_gas_summary_by_task(task_id: str):
    result = _get_task_result_or_raise(task_id)
    gas_stations = (result.get("fetched") or {}).get("gas_stations") or []
    return {"task_id": task_id, "summary": summarize_gas(gas_stations)}


# -----------------------------------------------------------------------------
# Map markers
# -----------------------------------------------------------------------------

@router.get("/map/data-by-task/{task_id}")
def get_map_data_by_task(task_id: str):
    """Map-ready markers: origin site + nearby gas stations, competitors, and retail anchors."""
    result = _get_task_result_or_raise(task_id)
    fetched = result.get("fetched") or {}

    lat = result.get("lat")
    lon = result.get("lon")
    if lat is None or lon is None:
        raise HTTPException(status_code=422, detail=f"Task {task_id} has no geocoded site coordinates yet.")

    markers: List[Dict[str, Any]] = [
        {
            "id": "origin",
            "name": "Input Site",
            "category": "origin",
            "latitude": float(lat),
            "longitude": float(lon),
            "distance_miles": 0.0,
            "address": result.get("address"),
        }
    ]

    def _add_marker(raw: Dict[str, Any], category: str, fallback_id: str) -> None:
        mlat, mlon = _resolve_marker_coordinates(raw)
        if mlat is None or mlon is None:
            return
        distance = raw.get("distance_miles")
        rating_count = raw.get("user_rating_count")
        if rating_count is None:
            rating_count = raw.get("rating_count")
        markers.append({
            "id": raw.get("place_id") or fallback_id,
            "name": raw.get("name"),
            "category": category,
            "latitude": float(mlat),
            "longitude": float(mlon),
            "distance_miles": float(distance) if distance is not None else None,
            "rating": float(raw["rating"]) if raw.get("rating") is not None else None,
            "user_rating_count": int(rating_count) if rating_count is not None else None,
            "address": raw.get("address"),
        })

    # gas: within_1 + within_3
    gas_stations = []
    for s in (fetched.get("gas_stations") or []):
        d = s.get("distance_miles")
        gas_stations.append({**s, "distance_miles": float(d) if d is not None else None})
    gas_stations.sort(key=lambda s: (s.get("distance_miles") is None, s.get("distance_miles") or float("inf")))
    gas_for_map = [s for s in gas_stations if s.get("distance_miles") is not None and s["distance_miles"] <= GAS_RADIUS_FAR_MILES]
    for idx, station in enumerate(gas_for_map, start=1):
        _add_marker(station, "gas_station", f"gas_{idx}")

    competitors = (fetched.get("competitors_data") or {}).get("competitors") or []
    for idx, comp in enumerate(competitors, start=1):
        _add_marker(comp, "car_wash", f"competitor_{idx}")

    retail_anchors_all = (fetched.get("retail_anchors") or {}).get("anchors") or []
    retail_within_1 = [a for a in retail_anchors_all if a.get("distance_miles") is not None and a["distance_miles"] <= RETAIL_RADIUS_NEAR_MILES]
    retail_within_3 = [a for a in retail_anchors_all if a.get("distance_miles") is not None and RETAIL_RADIUS_NEAR_MILES < a["distance_miles"] <= RETAIL_RADIUS_FAR_MILES]
    retail_for_map = _cap_anchors_by_type(retail_within_1, _ANCHOR_DISPLAY_CAP) + _cap_anchors_by_type(retail_within_3, _ANCHOR_DISPLAY_CAP)
    for idx, anchor in enumerate(retail_for_map, start=1):
        category = _retail_anchor_category(anchor.get("type"), anchor.get("name"))
        _add_marker(anchor, category, f"retail_{idx}")

    return {
        "task_id": task_id,
        "address": result.get("address"),
        "lat": float(lat),
        "lon": float(lon),
        "complete": True,
        "counts": {
            "markers_total": len(markers),
            "gas_stations": len([m for m in markers if m["category"] == "gas_station"]),
            "competitors": len([m for m in markers if m["category"] == "car_wash"]),
            "retail_anchors": len([m for m in markers if m["category"] not in {"origin", "gas_station", "car_wash"}]),
        },
        "markers": markers,
    }


# -----------------------------------------------------------------------------
# Synchronous lat/lon site analysis (the shared map pin) — one call, no task polling
# -----------------------------------------------------------------------------

@router.post("/site-context")
def get_site_context(req: SiteContextRequest):
    """
    Synchronous "what surrounds this location" for a lat/lon pin (or address): weather, competing car washes,
    retail anchors and gas stations + map markers + rule-based insights (optionally LLM-rewritten), all in ONE
    response. The lat/lon counterpart to the async /analyze-site pipeline; mirrors the Streamlit Site-analysis page.
    """
    from app.site_analysis.modelling.site_context import analyze_site_context

    if req.latitude is not None and req.longitude is not None:
        lat, lon = float(req.latitude), float(req.longitude)
        address = req.address
    elif req.address:
        lat, lon = _lat_lon_from_address_or_400(req.address)
        address = req.address
    else:
        raise HTTPException(status_code=400, detail="Provide either latitude/longitude or address.")
    try:
        return analyze_site_context(lat, lon, address=address, include_ai=req.include_ai, demo=req.demo)
    except Exception as e:
        logger.exception("Site context fetch failed")
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------------------------
# Single-fetch utilities (synchronous, address in / data out)
# -----------------------------------------------------------------------------

@router.post("/traffic-lights")
def get_traffic_lights_endpoint(features: AnalyseRequest):
    try:
        lat, lon = _lat_lon_from_address_or_400(features.address)
        data = get_traffic_lights_summary(lat, lon)
        return {"address": features.address, "lat": lat, "lon": lon, "data": data}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Traffic lights fetch failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/nearby-stores")
def get_nearby_stores_endpoint(features: AnalyseRequest):
    try:
        lat, lon = _lat_lon_from_address_or_400(features.address)
        try:
            data = get_nearby_stores_data(lat, lon)
        except Exception:
            logger.exception("Nearby stores fetch failed")
            data = {"error": "Could not retrieve nearby stores data."}
        return {"address": features.address, "lat": lat, "lon": lon, "data": data}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Nearby stores fetch failed")
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------------------------
# Task status
# -----------------------------------------------------------------------------

@router.get("/task/{task_id}", response_model=TaskStatusResponse)
def get_task_status(task_id: str):
    """Task status and result from Celery. Full `result` is present only when status is success."""
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
    """Get analyse-site result by task_id. Poll until status = success."""
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


# -----------------------------------------------------------------------------
# Health & cache
# -----------------------------------------------------------------------------

@router.get("/health")
def health_check():
    return {"status": "healthy", "service": "site-analysis-pipeline"}


@router.get("/cache/site-analysis/all")
def get_site_analysis_cache_all():
    return get_all_site_analysis_cache(page=1, page_size=50, include_response=True)
