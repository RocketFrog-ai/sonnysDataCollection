"""Logic the /v1/* site_analysis route handlers used to inline: the synchronous site-context /
site-features / traffic-lights / nearby-stores handlers. router.py's handlers stay thin delegators.

Celery and the async task pipeline were removed in 2026-07. Everything that fetched or shaped a
Celery task result went with it: `analyze_site`, `get_task_status`, `get_result_by_task`, the
per-dimension `get_*_data` / `get_*_summary` readers, `get_map_data`, and the three private helpers
(`_get_task_result_or_raise`, `_cap_anchors_by_type`, `_retail_anchor_category`,
`_resolve_marker_coordinates`) that only they reached. `get_site_context` is the replacement: it
returns the same weather / competing-wash / retail-anchor / gas data, plus markers and insights, in
one synchronous call. See docs/ARCHITECTURE.md.

Dropping those handlers also dropped their module-level imports of
`app.site_analysis.features.nearbyCompetitors.classify_competitor_types` and
`app.site_analysis.modelling.ai`, so importing this module no longer pulls that tree (and its
import-time HTTP/LLM calls) into the process. `site_context.py` imports the feature fetchers it
needs itself, lazily, inside `get_site_context`.
"""
from __future__ import annotations

import logging
from typing import Tuple

from fastapi import HTTPException

from app.core import common as calib
from app.site_analysis.server.schemas import (
    AddressRequest,
    SiteContextRequest,
    SiteFeaturesRequest,
)
from app.site_analysis.server.site_features import nearest_site_features
from app.site_analysis.features.trafficLights.nearby_traffic_lights import get_traffic_lights_summary
from app.site_analysis.features.nearbyStores.nearby_stores import get_nearby_stores_data

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _lat_lon_from_address_or_400(address: str) -> Tuple[float, float]:
    """Geocode for HTTP handlers: map geocode failures to 400."""
    try:
        return calib.resolve_lat_lon(address)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# -----------------------------------------------------------------------------
# Synchronous lat/lon site analysis (the shared map pin) — one call, no task polling
# -----------------------------------------------------------------------------

def get_site_context(req: SiteContextRequest):
    """
    Synchronous "what surrounds this location" for a lat/lon pin (or address): weather, competing car washes,
    retail anchors and gas stations + map markers + rule-based insights (optionally LLM-rewritten), all in ONE
    response. Mirrors the Streamlit Site-analysis page.
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
# Nearest-site features (offline) — lat/lon in, precomputed dataset features out
# -----------------------------------------------------------------------------

def get_site_features(req: SiteFeaturesRequest):
    """
    Look up the single closest site in the precomputed dataset (merged_all_sites.csv) by haversine
    distance to the given lat/lon, and return its features grouped by theme (demographics, income,
    vehicles, housing, mass-merchants, retail, traffic). No external calls; competitor info excluded.
    """
    try:
        return nearest_site_features(float(req.latitude), float(req.longitude))
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Sites dataset not found: {e}")
    except Exception as e:
        logger.exception("Nearest-site feature lookup failed")
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------------------------
# Single-fetch utilities (synchronous, address in / data out)
# -----------------------------------------------------------------------------

def get_traffic_lights(features: AddressRequest):
    try:
        lat, lon = _lat_lon_from_address_or_400(features.address)
        data = get_traffic_lights_summary(lat, lon)
        return {"address": features.address, "lat": lat, "lon": lon, "data": data}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Traffic lights fetch failed")
        raise HTTPException(status_code=500, detail=str(e))


def get_nearby_stores(features: AddressRequest):
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
