from __future__ import annotations

import logging
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict

from app.utils import common as calib
from app.site_analysis.features.active.nearbyGasStations.get_nearby_gas_stations import get_nearby_gas_stations
from app.site_analysis.features.active.nearbyRetailers.get_nearby_retail_anchors import get_nearby_retail_anchors
from app.site_analysis.features.active.nearbyCompetitors.get_nearby_competitors import get_nearby_competitors
from app.site_analysis.features.active.weather.open_meteo import fetch_climate_for_site, get_default_weather_range
from app.site_analysis.server.db_cache import (
    get_cached_site_fetch,
    save_site_fetch_cache,
    get_cached_site_analysis_by_latlon,
    save_site_analysis_response,
)
from app.celery.celery_app import celery_app

logger = logging.getLogger(__name__)

SITE_FETCH_CACHE_VERSION = "v1"
SITE_FETCH_CACHE_TTL_DAYS = int(os.getenv("SITE_FETCH_CACHE_TTL_DAYS", "30"))
SITE_RESPONSE_CACHE_TOLERANCE = float(os.getenv("SITE_RESPONSE_CACHE_TOLERANCE", "0.5"))


def fetch_all_features(lat: float, lon: float) -> Dict[str, Any]:
    """
    Fetch all external location data for (lat, lon) in parallel. Called once per analyse-site run.
    Google Places (searchNearby / Distance Matrix / Place Details) and climate (Open-Meteo) are
    invoked only here; the data-by-task / summary-by-task endpoints read the stored result.
    Returns: {climate, gas_stations, retail_anchors, competitors_data}.
    """
    start_date, end_date = get_default_weather_range()
    api_key = calib.GOOGLE_MAPS_API_KEY or ""

    def _fetch_climate() -> Dict[str, Any]:
        out = fetch_climate_for_site(lat, lon, start_date=start_date, end_date=end_date)
        return out if out and not out.get("error") else {}

    def _fetch_gas() -> list:
        if not api_key:
            return []
        try:
            return get_nearby_gas_stations(
                api_key, lat, lon, radius_miles=3.0, max_results=20, fetch_place_details=False
            ) or []
        except Exception as e:
            logger.warning("Gas stations fetch failed: %s", e)
            return []

    def _fetch_retail_anchors() -> Dict[str, Any]:
        if not api_key:
            return {}
        try:
            return get_nearby_retail_anchors(api_key, lat, lon, radius_miles=3.0) or {}
        except Exception as e:
            logger.warning("Retail anchors fetch failed: %s", e)
            return {}

    def _fetch_competitors() -> Dict[str, Any]:
        if not api_key:
            return {}
        try:
            return get_nearby_competitors(
                api_key, lat, lon, radius_miles=4.0, fetch_place_details=True
            ) or {}
        except Exception as e:
            logger.warning("Competitors fetch failed: %s", e)
            return {}

    result: Dict[str, Any] = {
        "climate": {},
        "gas_stations": [],
        "retail_anchors": {},
        "competitors_data": {},
    }
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_key = {
            executor.submit(_fetch_climate): "climate",
            executor.submit(_fetch_gas): "gas_stations",
            executor.submit(_fetch_retail_anchors): "retail_anchors",
            executor.submit(_fetch_competitors): "competitors_data",
        }
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                result[key] = future.result()
            except Exception as e:
                logger.warning("Feature fetch %s failed: %s", key, e)

    return result


@celery_app.task(bind=True, name="analyse_site")
def run_site_analysis(self, address: str) -> Dict[str, Any]:
    """
    Celery task: geocode → fetch external location data (climate, gas, retail, competitors).
    Call via run_site_analysis.delay(address). Registered Celery name: analyse_site.
    Result: {address, lat, lon, fetch_cache, fetched}. AI summaries are produced on demand by the
    /{dimension}/summary-by-task endpoints from `fetched`.
    """
    task_id = self.request.id
    logger.info("analyse_site task started: task_id=%s address=%s", task_id, address)

    normalized_address = " ".join((address or "").strip().lower().split())
    lat, lon = calib.resolve_lat_lon(address)

    cached_response = get_cached_site_analysis_by_latlon(lat, lon, tolerance=SITE_RESPONSE_CACHE_TOLERANCE)
    if cached_response and cached_response.get("response"):
        cached_result = cached_response["response"]
        cached_result["fetch_cache"] = {
            "hit": True,
            "cache_version": cached_response.get("cache_version"),
            "match_address": normalized_address,
            "match_lat": cached_response.get("lat"),
            "match_lon": cached_response.get("lon"),
            "match_type": "lat_lon_closest",
            "tolerance": SITE_RESPONSE_CACHE_TOLERANCE,
        }
        logger.info("analyse_site task finished (lat/lon cache hit): task_id=%s", task_id)
        return cached_result

    cache_key = hashlib.sha256(
        f"{SITE_FETCH_CACHE_VERSION}|{normalized_address}".encode("utf-8")
    ).hexdigest()

    cached = get_cached_site_fetch(cache_key)
    cached_fetched = (cached or {}).get("fetched") or {}
    cache_hit = bool(
        cached
        and cached.get("lat") is not None
        and cached.get("lon") is not None
        and cached_fetched
        and "climate" in cached_fetched
        and "gas_stations" in cached_fetched
        and "retail_anchors" in cached_fetched
        and "competitors_data" in cached_fetched
    )

    if cache_hit:
        lat = float(cached["lat"])
        lon = float(cached["lon"])
        fetched = cached["fetched"] or {}
        logger.info("run_site_analysis: cache hit for address=%s", address)
    else:
        logger.info("run_site_analysis: cache miss, geocoded lat=%.4f lon=%.4f, fetching features", lat, lon)
        fetched = fetch_all_features(lat, lon)
        climate = fetched.get("climate") or {}
        has_any_data = bool(
            (climate and not climate.get("error"))
            or (fetched.get("gas_stations") or [])
            or (fetched.get("retail_anchors") or {})
            or (fetched.get("competitors_data") or {})
        )
        if has_any_data:
            save_site_fetch_cache(
                address_key=cache_key,
                address_input=address,
                normalized_address=normalized_address,
                lat=lat,
                lon=lon,
                fetched=fetched,
                cache_version=SITE_FETCH_CACHE_VERSION,
                ttl_days=SITE_FETCH_CACHE_TTL_DAYS,
            )

    result: Dict[str, Any] = {
        "address": address,
        "lat": lat,
        "lon": lon,
        "fetch_cache": {
            "hit": cache_hit,
            "cache_version": SITE_FETCH_CACHE_VERSION,
            "match_address": normalized_address,
        },
        "fetched": fetched,
    }

    save_site_analysis_response(
        address_key=cache_key,
        address_input=address,
        normalized_address=normalized_address,
        lat=lat,
        lon=lon,
        response=result,
        cache_version=SITE_FETCH_CACHE_VERSION,
        ttl_days=SITE_FETCH_CACHE_TTL_DAYS,
    )

    logger.info("analyse_site task finished: task_id=%s", task_id)
    return result
