import redis
import logging
from app.utils import common as calib
from datetime import datetime, timezone
from celery.result import AsyncResult
from fastapi import APIRouter, HTTPException
from app.server.models import *
from app.server.app import get_climate, get_competitors, get_traffic_lights, get_nearby_stores
from app.ai.analysis import analyze_site_from_dict
from app.celery.tasks import analyse_site
from app.celery.celery_app import celery_app


REDIS_HOST = calib.REDIS_HOST
REDIS_PORT = calib.REDIS_PORT
REDIS_DB = calib.REDIS_DB
REDIS_PASSWORD = calib.REDIS_PASSWORD
CELERY_BROKER_URL = calib.CELERY_BROKER_URL
CELERY_RESULT_BACKEND = calib.CELERY_RESULT_BACKEND


logger = logging.getLogger(__name__)
router = APIRouter()


def _geocode(address: str):
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
        decode_responses=True
    )


@router.post("/analyze-site")
def analyze_site(features: AnalyseRequest):
    if not features.address:
        raise HTTPException(status_code=400, detail="No site address provided")
    try:
        result = analyse_site.delay(features.address)
        return TaskResponse(
            task_id=result.id,
            status=TaskStatus.PENDING,
            message="Site succesfully submitted for Analysis"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing site: {str(e)}")


@router.post("/analyze-site/direct")
def analyze_site_direct(features: AnalyseRequest):
    if not features.address:
        raise HTTPException(status_code=400, detail="No site address provided")
    try:
        result = analyze_site_from_dict(features.address)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Direct analyze failed")
        raise HTTPException(status_code=500, detail=f"Error analyzing site: {str(e)}")


def _resolve_weather_dates(start_date: str = None, end_date: str = None):
    from app.features.weather.open_meteo import get_default_weather_range
    if start_date and end_date:
        return str(start_date).strip(), str(end_date).strip()
    default_start, default_end = get_default_weather_range()
    return default_start, default_end


@router.post("/weather")
def get_weather(req: WeatherRequest):
    try:
        lat, lon = _geocode(req.address)
        start_date, end_date = _resolve_weather_dates(req.start_date, req.end_date)
        data = get_climate(lat, lon, start_date=start_date, end_date=end_date)
        return {
            "address": req.address,
            "lat": lat,
            "lon": lon,
            "start_date": start_date,
            "end_date": end_date,
            "data": data,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Weather fetch failed")
        raise HTTPException(status_code=500, detail=str(e))


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


@router.get("/task/{task_id}", response_model=TaskStatusResponse)
def get_task_status(task_id: str):
    """Get the status and result of an analyze-site task by its task_id."""
    task_result = AsyncResult(task_id, app=celery_app)
    status_str = task_result.state
    # Map Celery state to our TaskStatus enum (Celery uses same names)
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


@router.get("/health")
def health_check():
    return {"status": "healthy", "service": "site-analysis-pipeline"}