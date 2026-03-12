"""
Single-fetch site analysis: geocode → fetch all features (weather, gas, retail, competitors)
→ build feature_values + v3 location_features → run quantile prediction → optional narratives.

Used by the analyse-site Celery task. All feature fetching uses app.features.active.
When FETCH_WEATHER_ONLY is True, only weather is fetched; gas/stores/retail/competitors
use random placeholder data so quantile prediction can still run.
Return value includes feature_values (for existing routes) and quantile_result (v3 output).
"""

from __future__ import annotations

import logging
import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, Optional

from app.utils import common as calib
from app.server.app import get_climate
from app.features.active.nearbyStores.nearby_stores import get_nearby_stores_data
from app.features.active.nearbyGasStations.get_nearby_gas_stations import get_nearby_gas_stations
from app.features.active.nearbyRetailers.get_nearby_retailers import get_nearby_retailers
from app.features.active.nearbyCompetitors.get_nearby_competitors import get_nearby_competitors
from app.features.active.weather.open_meteo import get_default_weather_range

logger = logging.getLogger(__name__)

# When True, only fetch weather; use random data for gas, stores, retailers, competitors (for quantile).
FETCH_WEATHER_ONLY = True


def _geocode(address: str) -> tuple[float, float]:
    """Return (lat, lon). Raises ValueError if geocoding fails."""
    if not address or not address.strip():
        raise ValueError("Address is required")
    geo = calib.get_lat_long(address)
    lat = geo.get("lat")
    lon = geo.get("lon")
    if lat is None or lon is None:
        raise ValueError("Could not geocode address")
    return float(lat), float(lon)


def _random_placeholder_fetched(lat: float, lon: float) -> Dict[str, Any]:
    """Return placeholder data for gas, stores, retailers, competitors (for quantile when FETCH_WEATHER_ONLY)."""
    # Seed by lat/lon so same address gives same placeholders
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
        "stores": {
            "distance_from_nearest_costco": round(rng.uniform(0.5, 4.0), 2),
            "distance_from_nearest_walmart": round(rng.uniform(1.0, 3.0), 2),
            "distance_from_nearest_target": round(rng.uniform(1.0, 3.0), 2),
        },
        "retailers_data": {"retailers": [], "count": 0},
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
    Fetch all feature data for (lat, lon) using features/active, in parallel.
    Returns a dict with keys: climate, gas_stations, stores, retailers_data, competitors_data.
    When FETCH_WEATHER_ONLY is True, only climate is fetched; others get random placeholder data.
    """
    start_date, end_date = get_default_weather_range()
    api_key = calib.GOOGLE_MAPS_API_KEY or ""

    def _fetch_climate() -> Dict[str, Any]:
        out = get_climate(lat, lon, start_date=start_date, end_date=end_date)
        return out if out and not out.get("error") else {}

    if FETCH_WEATHER_ONLY:
        climate = _fetch_climate()
        placeholders = _random_placeholder_fetched(lat, lon)
        return {
            "climate": climate,
            "gas_stations": placeholders["gas_stations"],
            "stores": placeholders["stores"],
            "retailers_data": placeholders["retailers_data"],
            "competitors_data": placeholders["competitors_data"],
        }

    def _fetch_gas() -> list:
        if not api_key:
            return []
        try:
            return get_nearby_gas_stations(
                api_key, lat, lon,
                radius_miles=2.0,
                max_results=6,
                fetch_place_details=True,
            ) or []
        except Exception as e:
            logger.warning("Gas stations fetch failed: %s", e)
            return []

    def _fetch_stores() -> Dict[str, Any]:
        try:
            return get_nearby_stores_data(lat, lon) or {}
        except Exception as e:
            logger.warning("Stores fetch failed: %s", e)
            return {}

    def _fetch_retailers() -> Dict[str, Any]:
        if not api_key:
            return {}
        try:
            return get_nearby_retailers(
                api_key, lat, lon,
                radius_miles=0.5,
                fetch_place_details=False,
            ) or {}
        except Exception as e:
            logger.warning("Retailers fetch failed: %s", e)
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
        "stores": {},
        "retailers_data": {},
        "competitors_data": {},
    }
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_key = {
            executor.submit(_fetch_climate): "climate",
            executor.submit(_fetch_gas): "gas_stations",
            executor.submit(_fetch_stores): "stores",
            executor.submit(_fetch_retailers): "retailers_data",
            executor.submit(_fetch_competitors): "competitors_data",
        }
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                result[key] = future.result()
            except Exception as e:
                logger.warning("Feature fetch %s failed: %s", key, e)

    competitors_list = (result["competitors_data"] or {}).get("competitors") or []
    if competitors_list:
        try:
            from app.features.active.nearbyCompetitors.classify_competitor_types import classify_competitors
            competitors_list = classify_competitors(competitors_list)
        except Exception as e:
            logger.warning("Competitor classification skipped: %s", e)
        result["competitors_data"] = {**result["competitors_data"], "competitors": competitors_list}

    return {
        "climate": result["climate"],
        "gas_stations": result["gas_stations"],
        "stores": result["stores"],
        "retailers_data": result["retailers_data"],
        "competitors_data": result["competitors_data"],
    }


def build_feature_values_and_v3_input(fetched: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, float]]:
    """
    From fetch_all_features output, build:
    1) feature_values: task-style keys (for existing routes / profiler).
    2) location_features: v3 predictor keys (for QuantilePredictorV3.analyze).
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

    stores = fetched.get("stores") or {}
    costco_dist = stores.get("distance_from_nearest_costco")
    walmart_dist = stores.get("distance_from_nearest_walmart")
    target_dist = stores.get("distance_from_nearest_target")
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

    retailers_data = fetched.get("retailers_data") or {}
    raw_list = retailers_data.get("retailers") or []
    grocery_1mi = [r for r in raw_list if r.get("category") == "Grocery" and (r.get("distance_miles") or 0) <= 1.0]
    food_05mi = [r for r in raw_list if r.get("category") == "Food Joint" and (r.get("distance_miles") or 0) <= 0.5]
    other_grocery = len(grocery_1mi)
    food_joints = len(food_05mi)
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

    return feature_values, location_features


def run_site_analysis(
    address: str,
    *,
    run_quantile: bool = True,
    run_narratives: bool = False,
    set_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Single entry point: geocode → fetch all features once → build feature_values + v3 input
    → run quantile prediction → optionally run narrative agents.
    If set_progress is provided (e.g. from Celery task), it is called after fetch (data
    available quickly), after quantile, and after narratives so clients can poll partial results.
    Returns dict with address, lat, lon, feature_values, quantile_result (if run_quantile),
    narratives (if run_narratives).
    """
    lat, lon = _geocode(address)
    fetched = fetch_all_features(lat, lon)
    feature_values, location_features = build_feature_values_and_v3_input(fetched)

    result: Dict[str, Any] = {
        "address": address,
        "lat": lat,
        "lon": lon,
        "feature_values": feature_values,
        "fetched": fetched,
    }
    if set_progress:
        try:
            set_progress(dict(result))
        except Exception as e:
            logger.warning("set_progress after fetch failed: %s", e)

    quantile_result: Optional[Dict[str, Any]] = None
    if run_quantile and location_features:
        try:
            from app.modelling.ds.quantile_predictor import QuantilePredictorV3
            predictor = QuantilePredictorV3()
            quantile_result = predictor.analyze(location_features, llm_narrative=False)
            result["quantile_result"] = quantile_result
            if set_progress:
                try:
                    set_progress(dict(result))
                except Exception as e:
                    logger.warning("set_progress after quantile failed: %s", e)
        except Exception as e:
            logger.exception("Quantile prediction failed: %s", e)
            result["quantile_error"] = str(e)

    if run_narratives and quantile_result:
        try:
            from app.modelling.ai import get_feature_narratives, get_overall_narrative
            feature_narratives = get_feature_narratives(quantile_result, feature_values)
            result["narratives"] = {
                "feature": feature_narratives,
                "overall": get_overall_narrative(
                    quantile_result, feature_values, feature_narratives=feature_narratives
                ),
            }
            if set_progress:
                try:
                    set_progress(dict(result))
                except Exception as e:
                    logger.warning("set_progress after narratives failed: %s", e)
        except Exception as e:
            logger.exception("Narratives failed: %s", e)
            result["narrative_error"] = str(e)

    return result


# Alias for backward compatibility with Celery / routes that expect analyze_site_from_dict
def analyze_site_from_dict(address: str) -> Dict[str, Any]:
    """Run full site analysis (single fetch + quantile). No narratives by default."""
    return run_site_analysis(address, run_quantile=True, run_narratives=False)
