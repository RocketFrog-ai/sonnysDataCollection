"""Route decorators ONLY for the site_analysis endpoints mounted under /v1: parse the request, delegate
to service.py, serialize the response. Split out of the former monolithic routes.py; see routes.py in
this package for why that module still exists, and schemas.py / service.py for the request models and
the extracted helper/handler logic respectively."""
from fastapi import APIRouter

from app.site_analysis.server import service
from app.site_analysis.server.db_cache import get_all_site_analysis_cache
from app.site_analysis.server.schemas import (
    AnalyseRequest,
    SiteContextRequest,
    SiteFeaturesRequest,
    TaskStatusResponse,
)

router = APIRouter()


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
    return service.analyze_site(features)


# -----------------------------------------------------------------------------
# Per-dimension raw DATA (fast — no LLM). Reads only the stored `fetched` payload.
# -----------------------------------------------------------------------------

@router.get("/weather/data-by-task/{task_id}")
def get_weather_data_by_task(task_id: str):
    """Raw weather metrics (rainy days, snowfall, comfortable days, freezing days) from the fetch."""
    return service.get_weather_data(task_id)


@router.get("/competition/data-by-task/{task_id}")
def get_competition_data_by_task(task_id: str):
    """Raw nearby same-format car washes (within 4 miles) from the fetch."""
    return service.get_competition_data(task_id)


@router.get("/retail/data-by-task/{task_id}")
def get_retail_data_by_task(task_id: str):
    """Raw nearby retail anchors (within 1 and 3 miles) from the fetch."""
    return service.get_retail_data(task_id)


@router.get("/gas/data-by-task/{task_id}")
def get_gas_data_by_task(task_id: str):
    """Raw nearby gas stations (within 1 and 3 miles) from the fetch."""
    return service.get_gas_data(task_id)


# -----------------------------------------------------------------------------
# Per-dimension AI SUMMARIES (on-demand — one LLM call, grounded on raw fetched data).
# -----------------------------------------------------------------------------

@router.get("/weather/summary-by-task/{task_id}")
def get_weather_summary_by_task(task_id: str):
    return service.get_weather_summary(task_id)


@router.get("/competition/summary-by-task/{task_id}")
def get_competition_summary_by_task(task_id: str):
    return service.get_competition_summary(task_id)


@router.get("/retail/summary-by-task/{task_id}")
def get_retail_summary_by_task(task_id: str):
    return service.get_retail_summary(task_id)


@router.get("/gas/summary-by-task/{task_id}")
def get_gas_summary_by_task(task_id: str):
    return service.get_gas_summary(task_id)


# -----------------------------------------------------------------------------
# Map markers
# -----------------------------------------------------------------------------

@router.get("/map/data-by-task/{task_id}")
def get_map_data_by_task(task_id: str):
    """Map-ready markers: origin site + nearby gas stations, competitors, and retail anchors."""
    return service.get_map_data(task_id)


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
    return service.get_site_context(req)


# -----------------------------------------------------------------------------
# Nearest-site features (offline) — lat/lon in, precomputed dataset features out
# -----------------------------------------------------------------------------

@router.post("/site-features")
def get_site_features(req: SiteFeaturesRequest):
    """
    Look up the single closest site in the precomputed dataset (merged_all_sites.csv) by haversine
    distance to the given lat/lon, and return its features grouped by theme (demographics, income,
    vehicles, housing, mass-merchants, retail, traffic). No external calls; competitor info excluded.
    """
    return service.get_site_features(req)


# -----------------------------------------------------------------------------
# Single-fetch utilities (synchronous, address in / data out)
# -----------------------------------------------------------------------------

@router.post("/traffic-lights")
def get_traffic_lights_endpoint(features: AnalyseRequest):
    return service.get_traffic_lights(features)


@router.post("/nearby-stores")
def get_nearby_stores_endpoint(features: AnalyseRequest):
    return service.get_nearby_stores(features)


# -----------------------------------------------------------------------------
# Task status
# -----------------------------------------------------------------------------

@router.get("/task/{task_id}", response_model=TaskStatusResponse)
def get_task_status(task_id: str):
    """Task status and result from Celery. Full `result` is present only when status is success."""
    return service.get_task_status(task_id)


@router.get("/result/{task_id}")
def get_result_by_task(task_id: str):
    """Get analyse-site result by task_id. Poll until status = success."""
    return service.get_result_by_task(task_id)


# -----------------------------------------------------------------------------
# Health & cache
# -----------------------------------------------------------------------------

@router.get("/health")
def health_check():
    return {"status": "healthy", "service": "site-analysis-pipeline"}


@router.get("/cache/site-analysis/all")
def get_site_analysis_cache_all():
    return get_all_site_analysis_cache(page=1, page_size=50, include_response=True)
