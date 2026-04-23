from __future__ import annotations

import logging
import math
import hashlib
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional

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

# When True, only fetch weather; use random placeholders for gas/retail/competitors.
FETCH_WEATHER_ONLY = True
# When True and FETCH_WEATHER_ONLY, also fetch real competitors (4-mile radius).
FETCH_COMPETITION_WITH_WEATHER = True
# When True and FETCH_WEATHER_ONLY, also fetch real gas stations (3-mile radius).
FETCH_GAS_WITH_WEATHER = True
# When True and FETCH_WEATHER_ONLY, also fetch real retail anchors (unified 3-mile radius fetch).
FETCH_RETAIL_WITH_WEATHER = True

SITE_FETCH_CACHE_VERSION = "v1"
SITE_FETCH_CACHE_TTL_DAYS = int(os.getenv("SITE_FETCH_CACHE_TTL_DAYS", "30"))
SITE_RESPONSE_CACHE_TOLERANCE = float(os.getenv("SITE_RESPONSE_CACHE_TOLERANCE", "0.5"))


def _random_placeholder_fetched(lat: float, lon: float) -> Dict[str, Any]:
    """Return placeholder data for gas, retail anchors, and competitors (for quantile when FETCH_WEATHER_ONLY)."""
    # Seed by lat/lon so same address gives consistent placeholders
    rng = random.Random(int(lat * 1e6) + int(lon * 1e6))
    return {
        "gas_stations": [
            {
                "distance_miles": round(rng.uniform(0.5, 2.5), 2),
                "rating": round(rng.uniform(3.0, 4.5), 1),
                "user_rating_count": rng.randint(20, 300),
                "rating_count": rng.randint(20, 300),
            }
        ],
        "retail_anchors": {
            "anchors": [],
            "costco_dist": round(rng.uniform(0.5, 4.0), 2),
            "walmart_dist": round(rng.uniform(1.0, 3.0), 2),
            "target_dist": round(rng.uniform(1.0, 3.0), 2),
            "grocery_count_1mile": rng.randint(0, 3),
            "food_count_0_5miles": rng.randint(0, 5),
        },
        "competitors_data": {
            "count": rng.randint(2, 10),
            "competitors": [
                {
                    "distance_miles": round(rng.uniform(0.3, 2.0), 2),
                    "rating": round(rng.uniform(3.2, 4.8), 1),
                    "user_rating_count": rng.randint(10, 400),
                    "rating_count": rng.randint(10, 400),
                }
            ],
        },
    }


def fetch_all_features(lat: float, lon: float) -> Dict[str, Any]:
    """
    Fetch all feature data for (lat, lon) using features/active. Called exactly once per
    analyse-site run. Google API (Places searchNearby, Distance Matrix, Place Details) and
    climate (Open-Meteo) are invoked only here — GET /weather/data-by-task and
    GET /competition/data-by-task never re-fetch; they read from the stored task result.
    Returns a dict with keys: climate, gas_stations, stores, retailers_data, competitors_data.
    When FETCH_WEATHER_ONLY is True, only climate (+ optionally competitors) is fetched.
    """
    start_date, end_date = get_default_weather_range()
    api_key = calib.GOOGLE_MAPS_API_KEY or ""

    def _fetch_climate() -> Dict[str, Any]:
        out = fetch_climate_for_site(lat, lon, start_date=start_date, end_date=end_date)
        return out if out and not out.get("error") else {}

    if FETCH_WEATHER_ONLY:
        placeholders = _random_placeholder_fetched(lat, lon)

        result_partial: Dict[str, Any] = {
            "climate": {},
            "gas_stations": placeholders["gas_stations"],
            "retail_anchors": placeholders["retail_anchors"],
            "competitors_data": placeholders["competitors_data"],
        }

        def _partial_competitors() -> Dict[str, Any]:
            if not api_key:
                return placeholders["competitors_data"]
            try:
                data = get_nearby_competitors(
                    api_key, lat, lon,
                    radius_miles=4.0,
                    fetch_place_details=True,
                )
                return data or placeholders["competitors_data"]
            except Exception as e:
                logger.warning("Competitors fetch failed (using placeholder): %s", e)
                return placeholders["competitors_data"]

        def _partial_gas() -> list:
            if not api_key:
                return placeholders["gas_stations"]
            try:
                data = get_nearby_gas_stations(
                    api_key, lat, lon,
                    radius_miles=3.0,
                    max_results=20,
                    fetch_place_details=False,
                )
                return data or placeholders["gas_stations"]
            except Exception as e:
                logger.warning("Gas stations fetch failed (using placeholder): %s", e)
                return placeholders["gas_stations"]

        def _partial_retail() -> Dict[str, Any]:
            if not api_key:
                return placeholders["retail_anchors"]
            try:
                data = get_nearby_retail_anchors(api_key, lat, lon, radius_miles=3.0)
                return data or placeholders["retail_anchors"]
            except Exception as e:
                logger.warning("Retail anchors fetch failed (using placeholder): %s", e)
                return placeholders["retail_anchors"]

        futures_map: Dict[Any, str] = {}
        with ThreadPoolExecutor(max_workers=5) as ex:
            futures_map[ex.submit(_fetch_climate)] = "climate"
            if FETCH_COMPETITION_WITH_WEATHER:
                futures_map[ex.submit(_partial_competitors)] = "competitors_data"
            if FETCH_GAS_WITH_WEATHER:
                futures_map[ex.submit(_partial_gas)] = "gas_stations"
            if FETCH_RETAIL_WITH_WEATHER:
                futures_map[ex.submit(_partial_retail)] = "retail_anchors"
            for future in as_completed(futures_map):
                key = futures_map[future]
                try:
                    result_partial[key] = future.result()
                except Exception as e:
                    logger.warning("Partial fetch %s failed: %s", key, e)

        return result_partial

    def _fetch_gas() -> list:
        if not api_key:
            return []
        try:
            return get_nearby_gas_stations(
                api_key, lat, lon,
                radius_miles=3.0,
                max_results=20,
                fetch_place_details=False,
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
                api_key, lat, lon,
                radius_miles=4.0,
                fetch_place_details=True,
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

    return {
        "climate": result["climate"],
        "gas_stations": result["gas_stations"],
        "retail_anchors": result["retail_anchors"],
        "competitors_data": result["competitors_data"],
    }


def build_feature_values_and_v3_input(
    fetched: Dict[str, Any],
    tunnel_count: Optional[int] = None,
    carwash_type_encoded: Optional[int] = None,
) -> tuple[Dict[str, Any], Dict[str, float]]:
    """
    From fetch_all_features output, build:
    1) feature_values: task-style keys (for existing routes / profiler).
    2) location_features: predictor keys (for QuantilePredictorV4.analyze).
    """
    feature_values: Dict[str, Any] = {}
    location_features: Dict[str, float] = {}

    climate = fetched.get("climate") or {}
    if climate and "error" not in climate:
        for api_key, profiler_key in [
            ("total_precipitation_mm", "weather_total_precipitation_mm"),
            ("rainy_days", "weather_rainy_days"),
            ("total_snowfall_cm", "weather_total_snowfall_cm"),
            ("days_below_freezing", "weather_days_below_freezing"),
            ("total_sunshine_hours", "weather_total_sunshine_hours"),
            ("days_pleasant_temp", "weather_days_pleasant_temp"),
            ("avg_daily_max_windspeed_ms", "weather_avg_daily_max_windspeed_ms"),
        ]:
            v = climate.get(api_key)
            if v is not None:
                feature_values[api_key] = v
                location_features[profiler_key] = float(v)

    gas_stations = fetched.get("gas_stations") or []
    if gas_stations:
        s0 = gas_stations[0]
        d = s0.get("distance_miles")
        r = s0.get("rating")
        rc = s0.get("user_rating_count") or s0.get("rating_count")
        if d is not None:
            feature_values["distance_from_nearest_gas_station"] = d
            location_features["nearest_gas_station_distance_miles"] = float(d)
        if r is not None:
            feature_values["nearest_gas_station_rating"] = r
            location_features["nearest_gas_station_rating"] = float(r)
        if rc is not None:
            feature_values["nearest_gas_station_rating_count"] = rc
            location_features["nearest_gas_station_rating_count"] = float(rc)

    retail_anchors = fetched.get("retail_anchors") or {}
    costco_dist = retail_anchors.get("costco_dist")
    walmart_dist = retail_anchors.get("walmart_dist")
    target_dist = retail_anchors.get("target_dist")
    other_grocery = retail_anchors.get("grocery_count_1mile", 0)
    food_joints = retail_anchors.get("food_count_0_5miles", 0)

    if costco_dist is not None:
        feature_values["distance_from_nearest_costco"] = costco_dist
        v = float(costco_dist) if isinstance(costco_dist, (int, float)) else None
        location_features["costco_enc"] = 99.0 if v is None or v > 5 else v
    else:
        location_features["costco_enc"] = 99.0
    if walmart_dist is not None:
        feature_values["distance_from_nearest_walmart"] = walmart_dist
        location_features["distance_nearest_walmart(5 mile)"] = float(walmart_dist)
    if target_dist is not None:
        feature_values["distance_from_nearest_target"] = target_dist
        location_features["distance_nearest_target (5 mile)"] = float(target_dist)
    if "costco_enc" not in location_features:
        location_features["costco_enc"] = 99.0

    feature_values["other_grocery_count_1mile"] = other_grocery
    feature_values["count_food_joints_0_5miles"] = food_joints
    location_features["other_grocery_count_1mile"] = float(other_grocery)
    location_features["count_food_joints_0_5miles (0.5 mile)"] = float(food_joints)

    comp = fetched.get("competitors_data") or {}
    count = comp.get("count", 0)
    feature_values["count"] = count
    location_features["competitors_count_4miles"] = float(count)
    competitors_list = comp.get("competitors") or []
    if competitors_list:
        c1 = competitors_list[0]
        dm = c1.get("distance_miles")
        gr = c1.get("rating")
        rc1 = c1.get("user_rating_count") or c1.get("rating_count")
        if dm is not None:
            feature_values["competitor_1_distance_miles"] = dm
            location_features["competitor_1_distance_miles"] = float(dm)
        if gr is not None:
            feature_values["competitor_1_google_rating"] = gr
            location_features["competitor_1_google_rating"] = float(gr)
        if rc1 is not None:
            feature_values["competitor_1_rating_count"] = rc1
            location_features["competitor_1_rating_count"] = float(rc1)

    # Tunnel count: use provided value if given, otherwise let the predictor impute from KNN.
    if tunnel_count is not None and 1 <= int(tunnel_count) <= 4:
        feature_values["tunnel_count"] = int(tunnel_count)
        location_features["tunnel_count"] = float(tunnel_count)

    # Car wash type: 1=Express Tunnel, 2=Mobile, 3=Hand Wash/Detail. Improves quantile prediction.
    if carwash_type_encoded is not None and 1 <= int(carwash_type_encoded) <= 3:
        feature_values["carwash_type_encoded"] = int(carwash_type_encoded)
        location_features["carwash_type_encoded"] = float(carwash_type_encoded)

    # Engineered features for v3
    comp_qual = location_features.get("competitor_1_google_rating"), location_features.get("competitor_1_rating_count")
    if comp_qual[0] is not None and comp_qual[1] is not None:
        location_features["competition_quality"] = float(comp_qual[0]) * math.log1p(float(comp_qual[1]))
    gs_rating = location_features.get("nearest_gas_station_rating")
    gs_count = location_features.get("nearest_gas_station_rating_count")
    if gs_rating is not None and gs_count is not None:
        location_features["gas_station_draw"] = float(gs_rating) * math.log1p(float(gs_count))
    wm = location_features.get("distance_nearest_walmart(5 mile)")
    tg = location_features.get("distance_nearest_target (5 mile)")
    if wm is not None and tg is not None:
        location_features["retail_proximity"] = 1.0 / (float(wm) + float(tg) + 0.1)
    pleasant = location_features.get("weather_days_pleasant_temp")
    freeze = location_features.get("weather_days_below_freezing")
    if pleasant is not None and freeze is not None:
        location_features["weather_drive_score"] = float(pleasant) - float(freeze)

    # Effective capacity: tunnel_count × is_express.
    # Only Express Tunnel (carwash_type_encoded=1) has a physical conveyor tunnel.
    # Mobile, Hand Wash, Flex etc → effective_capacity=0 regardless of tunnel_count.
    # If carwash_type unknown, defaults to tunnel_count (75% of sites are Express).
    tc = location_features.get("tunnel_count", 1.0)
    cw = location_features.get("carwash_type_encoded")
    is_express = float(cw == 1) if cw is not None else 1.0
    location_features["effective_capacity"] = float(tc) * is_express
    feature_values["effective_capacity"] = location_features["effective_capacity"]

    return feature_values, location_features


@celery_app.task(bind=True, name="analyse_site")
def run_site_analysis(
    self,
    address: str,
    tunnel_count: Optional[int] = None,
    carwash_type_encoded: Optional[int] = None,
    tier_strategy: str = "4-class-90pct-custom",
    run_quantile: bool = True,
    run_narratives: bool = True,
) -> Dict[str, Any]:
    """
    Celery task + implementation: geocode → fetch → quantile → narratives.
    Call via run_site_analysis.delay(...). Registered Celery name: analyse_site.
    """
    task_id = self.request.id
    logger.info(
        "analyse_site task started: task_id=%s address=%s tunnel_count=%s carwash_type=%s strategy=%s",
        task_id,
        address,
        tunnel_count,
        carwash_type_encoded,
        tier_strategy,
    )
    normalized_address = " ".join((address or "").strip().lower().split())
    lat, lon = calib.resolve_lat_lon(address)
    cached_response = get_cached_site_analysis_by_latlon(
        lat, lon, tolerance=SITE_RESPONSE_CACHE_TOLERANCE
    )
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
        logger.info(
            "run_site_analysis: cache miss, fetching address=%s tunnel_count=%s carwash_type=%s",
            address, tunnel_count, carwash_type_encoded,
        )
        logger.info("run_site_analysis: geocode done lat=%.4f lon=%.4f, fetching features", lat, lon)
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
    logger.info("run_site_analysis: fetch done, building feature_values and quantile input")
    feature_values, location_features = build_feature_values_and_v3_input(
        fetched,
        tunnel_count=tunnel_count,
        carwash_type_encoded=carwash_type_encoded,
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
        "feature_values": feature_values,
        "fetched": fetched,
    }

    quantile_result: Optional[Dict[str, Any]] = None
    if run_quantile and location_features:
        try:
            from app.site_analysis.modelling.ds.prediction import QuantilePredictorV4
            predictor = QuantilePredictorV4(tier_strategy=tier_strategy)
            quantile_result = predictor.analyze(location_features, llm_narrative=False)
            result["quantile_result"] = quantile_result
            logger.info("run_site_analysis: quantile prediction done (Q%s)", quantile_result.get("predicted_wash_quantile"))
        except Exception as e:
            logger.exception("Quantile prediction failed: %s", e)
            result["quantile_error"] = str(e)

    if run_narratives and quantile_result:
        try:
            from app.site_analysis.modelling.ai import (
                get_feature_narratives,
                get_overall_narrative,
                get_competition_narrative,
                get_retail_narrative,
                get_gas_narrative,
            )
            feature_narratives = get_feature_narratives(quantile_result, feature_values)
            result["narratives"] = {
                "feature": feature_narratives,
                "overall": get_overall_narrative(
                    quantile_result, feature_values, feature_narratives=feature_narratives
                ),
                "competition": get_competition_narrative(
                    quantile_result,
                    feature_narratives,
                    feature_values=feature_values,
                ),
                "retail": get_retail_narrative(
                    quantile_result,
                    feature_narratives,
                    feature_values=feature_values,
                ),
                "gas": get_gas_narrative(quantile_result, feature_narratives),
            }
            logger.info("run_site_analysis: narratives done")
        except Exception as e:
            logger.exception("Narratives failed: %s", e)
            result["narrative_error"] = str(e)

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
