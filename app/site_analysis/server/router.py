"""Route decorators ONLY for the site_analysis endpoints mounted under /v1: parse the request, delegate
to service.py, serialize the response. See schemas.py / service.py for the request models and the
extracted helper/handler logic respectively.

Celery and the async task pipeline were removed in 2026-07. The endpoints that existed only to
enqueue and poll a Celery task are gone:

    POST /analyze-site
    GET  /task/{task_id}
    GET  /result/{task_id}
    GET  /{weather,competition,retail,gas}/data-by-task/{task_id}
    GET  /{weather,competition,retail,gas}/summary-by-task/{task_id}
    GET  /map/data-by-task/{task_id}

`POST /site-context` is their synchronous replacement: it returns the same weather / competing-wash /
retail-anchor / gas data, plus map markers and insights, in a single call. See docs/ARCHITECTURE.md.
"""
from fastapi import APIRouter

from app.site_analysis.server import service
from app.site_analysis.server.db_cache import get_all_site_analysis_cache
from app.site_analysis.server.schemas import (
    AddressRequest,
    SiteContextRequest,
    SiteFeaturesRequest,
)

router = APIRouter()


# -----------------------------------------------------------------------------
# Synchronous lat/lon site analysis (the shared map pin) — one call, no task polling
# -----------------------------------------------------------------------------

@router.post("/site-context")
def get_site_context(req: SiteContextRequest):
    """
    Synchronous "what surrounds this location" for a lat/lon pin (or address): weather, competing car washes,
    retail anchors and gas stations + map markers + rule-based insights (optionally LLM-rewritten), all in ONE
    response. Mirrors the Streamlit Site-analysis page.
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
def get_traffic_lights_endpoint(features: AddressRequest):
    return service.get_traffic_lights(features)


@router.post("/nearby-stores")
def get_nearby_stores_endpoint(features: AddressRequest):
    return service.get_nearby_stores(features)


# -----------------------------------------------------------------------------
# Health & cache
# -----------------------------------------------------------------------------

@router.get("/health")
def health_check():
    return {"status": "healthy", "service": "site-analysis-pipeline"}


@router.get("/cache/site-analysis/all")
def get_site_analysis_cache_all():
    """Read-only view of the Postgres site-analysis cache.

    NOTE: nothing writes this cache any more. Its only writer was the Celery task
    (`run_site_analysis`), removed with the async pipeline. Rows already in the table are still
    served; no new rows appear. Kept because the data is real and the route is harmless.
    """
    return get_all_site_analysis_cache(page=1, page_size=50, include_response=True)
