import redis
import logging
from app.utils import common as calib
from datetime import datetime, timezone
from celery.result import AsyncResult
from fastapi import APIRouter, HTTPException
from app.server.models import *
from app.server.app import get_climate, get_competitors, get_traffic_lights, get_nearby_stores
from app.features.nearbyStores.nearby_stores import get_nearby_stores_data
from app.features.weather.weather_period import get_annual_weather_plot_data
from app.features.nearbyGasStations.get_nearby_gas_stations import get_nearby_gas_stations, get_nearest_gas_station_only
from app.features.nearbyRetailers.get_nearby_retailers import get_nearby_retailers
from app.features.nearbyCompetitors.get_nearby_competitors import get_nearby_competitors
from app.ai.analysis import analyze_site_from_dict
from app.celery.tasks import analyse_site
from app.celery.celery_app import celery_app
from app.agentic_reference.profiler_engine import QuantileProfiler, DIMENSION_GROUPS
from app.agentic_reference.dimension_profiler import DimensionProfiler
from app.agentic_reference.agentic_rationale import generate_dimension_rationale, generate_rationale
from app.scoring.approach2_scorer import (
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
from app.scoring.approach2_summary import (
    get_dimension_summary_approach2,
    DIMENSION_FEATURE_MAP,
)


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


def _get_task_result_or_raise(task_id: str):
    """Resolve Celery task by task_id. Returns result dict if SUCCESS; raises HTTPException otherwise."""
    task_result = AsyncResult(task_id, app=celery_app)
    if task_result.state != TaskStatus.SUCCESS.value:
        if task_result.state == TaskStatus.FAILURE.value:
            raise HTTPException(
                status_code=422,
                detail=f"Task {task_id} failed: {str(task_result.result) if task_result.result else 'Task failed'}",
            )
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not completed yet (status={task_result.state}). Poll GET /v1/task/{task_id} until status is SUCCESS.",
        )
    result = task_result.result
    if result is None:
        raise HTTPException(status_code=422, detail=f"Task {task_id} completed but result is empty.")
    return result


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


def _get_weather_response(req: WeatherRequest):
    """Shared logic for weather data: geocode, resolve dates, return climate data with Approach 2 categories and dimension score."""
    lat, lon = _geocode(req.address)
    start_date, end_date = _resolve_weather_dates(req.start_date, req.end_date)
    data = get_climate(lat, lon, start_date=start_date, end_date=end_date)

    feature_scores = {}
    dimension_score = None
    if data and "error" not in data:
        data = enrich_features_with_categories(data)
        raw_vals = {k: v.get("value") if isinstance(v, dict) else v for k, v in data.items()}
        scores = get_feature_final_scores(raw_vals, WEATHER_API_TO_PROFILER)
        dimension_score = compute_dimension_score(scores, "Weather")
        # Build feature_scores keyed by profiler key (matches summary feature_breakdown keys)
        for api_key, profiler_key in WEATHER_API_TO_PROFILER.items():
            val = raw_vals.get(api_key)
            if val is not None:
                cat = (data.get(api_key) or {}).get("category", "N/A")
                feature_scores[profiler_key] = {"value": val, "category": cat}

    return {
        "address": req.address,
        "lat": lat,
        "lon": lon,
        "start_date": start_date,
        "end_date": end_date,
        "data": data,
        "feature_scores": feature_scores,
        "dimension_score": {"Weather": dimension_score} if dimension_score is not None else {},
    }


@router.post("/weather")
def get_weather(req: WeatherRequest):
    """Legacy endpoint: same as /weather/data."""
    try:
        return _get_weather_response(req)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Weather fetch failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/weather/data")
def get_weather_data(req: WeatherRequest):
    """Weather data for a location and optional date range (current v1/weather behavior)."""
    try:
        return _get_weather_response(req)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Weather data fetch failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/weather/plot")
def get_weather_plot(req: WeatherPlotRequest):
    """
    Annual weather data for the Monthly Weather Distribution chart and full monthly stats.
    Returns sunny_days, pleasant_days, rainy_days, snow_days per month (Jan–Dec) plus all other metrics.
    """
    try:
        lat, lon = _geocode(req.address)
        result = get_annual_weather_plot_data(lat, lon, req.year)
        if result is None:
            raise HTTPException(status_code=502, detail="Could not fetch weather data for the given year.")
        return {
            "address": req.address,
            "lat": lat,
            "lon": lon,
            **result,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Weather plot data fetch failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/weather/summary/{task_id}")
def get_weather_summary_by_task(task_id: str):
    """
    Executive summary for Weather dimension using quantile-based profiling.
    Scores weather features against Low/Avg/High IQR ranges and generates LLM rationale.
    """
    return _dimension_summary(task_id, "Weather")


@router.post("/gas_station")
def get_gas_stations_endpoint(req: GasStationRequest):
    """
    Nearby gas stations within the given radius (default 2 miles).
    All station fields are from Google Places API (Nearby Search and optionally Place Details).
    No inferred or hardcoded data: name, rating, address, regular_opening_hours from search;
    optional fuel_options and types from Place Details when fetch_place_details is true.
    """
    try:
        lat, lon = _geocode(req.address)
        api_key = calib.GOOGLE_MAPS_API_KEY
        if not api_key:
            raise HTTPException(status_code=503, detail="Google Maps API key not configured")
        radius = req.radius_miles if req.radius_miles is not None else 2.0
        # Cap at 6 closest stations
        max_results = min(req.max_results if req.max_results is not None else 6, 6)
        fetch_details = req.fetch_place_details if req.fetch_place_details is not None else True
        stations = get_nearby_gas_stations(
            api_key, lat, lon,
            radius_miles=radius,
            max_results=max_results,
            fetch_place_details=fetch_details,
        )

        # Feature scores and dimension score from the nearest station only
        feature_scores = {}
        dimension_score = None
        if stations:
            s0 = stations[0]
            feature_scores = enrich_gas_features_with_categories(s0)
            flat = {k: v for k, v in {
                "distance_miles": s0.get("distance_miles"),
                "rating": s0.get("rating"),
                "rating_count": s0.get("rating_count"),
            }.items() if v is not None}
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
    """
    Executive summary for Gas dimension (nearest gas station distance/rating).
    """
    return _dimension_summary(task_id, "Gas")


@router.post("/datafetch/nearbygasstations")
def datafetch_nearby_gas_stations(req: DataFetchNearestGasRequest):
    """
    Returns only the single nearest gas station by driving distance (no radius or count limit).
    Uses the same logic and fields as /gas_station: Google Nearby Search (gas_station type),
    Distance Matrix for driving distance, and optionally Place Details. Only returns a place
    confirmed as a gas station. Intended for batch data collection over many addresses.
    """
    try:
        lat, lon = _geocode(req.address)
        api_key = calib.GOOGLE_MAPS_API_KEY
        if not api_key:
            raise HTTPException(status_code=503, detail="Google Maps API key not configured")
        fetch_details = req.fetch_place_details if req.fetch_place_details is not None else True
        nearest = get_nearest_gas_station_only(
            api_key, lat, lon,
            fetch_place_details=fetch_details,
        )
        feature_scores = enrich_gas_features_with_categories(nearest)
        dimension_score = None
        if nearest:
            flat = {k: nearest.get(k) for k in ("distance_miles", "rating", "rating_count") if nearest.get(k) is not None}
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


@router.post("/retailers")
def get_retailers_endpoint(req: RetailersRequest):
    """
    Returns only 5 categories: Costco, Walmart, Target (with details), other_grocery (count + avg),
    food_joints (count + avg). No list of individual places.
    """
    try:
        lat, lon = _geocode(req.address)
        api_key = calib.GOOGLE_MAPS_API_KEY
        if not api_key:
            raise HTTPException(status_code=503, detail="Google Maps API key not configured")
        radius = req.radius_miles if req.radius_miles is not None else 0.5
        # Cost-friendly: no Place Details; we only need count and rating/distance for averages
        data = get_nearby_retailers(
            api_key, lat, lon,
            radius_miles=radius,
            fetch_place_details=False,
        )
        stores = get_nearby_stores_data(lat, lon)

        # Use list only for counts and averages — do not expose in response
        raw_list = data.get("retailers") or []
        grocery_within_1mi = [r for r in raw_list if r.get("category") == "Grocery" and (r.get("distance_miles") or 0) <= 1.0]
        food_within_0_5mi = [r for r in raw_list if r.get("category") == "Food Joint" and (r.get("distance_miles") or 0) <= 0.5]
        other_grocery_count = len(grocery_within_1mi)
        food_joint_count = len(food_within_0_5mi)

        def _avg(items, key):
            vals = [x[key] for x in items if x.get(key) is not None]
            return round(sum(vals) / len(vals), 2) if vals else None

        flat_retailer = {k: v for k, v in {
            "distance_from_nearest_costco": stores.get("distance_from_nearest_costco"),
            "distance_from_nearest_walmart": stores.get("distance_from_nearest_walmart"),
            "distance_from_nearest_target": stores.get("distance_from_nearest_target"),
            "other_grocery_count_1mile": other_grocery_count,
            "count_food_joints_0_5miles": food_joint_count,
        }.items() if v is not None}

        feature_scores = enrich_retailers_features_with_categories(flat_retailer) if flat_retailer else {}
        dimension_score = None
        if flat_retailer:
            scores = get_feature_final_scores(flat_retailer, RETAILER_API_TO_PROFILER)
            dimension_score = compute_dimension_score(scores, "Retail Proximity")

        def _anchor_item(display_name: str, details):
            if details is None:
                return {"name": display_name, "found": False, "message": "No store within 5 mile radius"}
            # Keep category name (Costco/Walmart/Target); add store details without overwriting name
            rest = {k: v for k, v in details.items() if k != "name"}
            return {"name": display_name, "found": True, "store_name": details.get("name"), **rest}

        # retailers: exactly 5 items for UI — Costco, Walmart, Target, Other groceries, Food joints (per P-2.0 Sheet4)
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
    """
    Executive summary for Retail Proximity (Costco, Walmart, Target, other grocery, food joints).
    """
    return _dimension_summary(task_id, "Retail Proximity")


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


@router.post("/competitors/dynamics")
def get_competitors_dynamics_endpoint(req: CompetitorsDynamicsRequest):
    """
    Nearby car wash competitors within radius by route (driving) distance.
    Real data only: name, distance_miles, rating, user_rating_count, address; optional website, google_maps_uri, primary_type_display_name.
    No market share or threat level (not from API).
    """
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
    """
    Executive summary for Competition dimension using quantile-based profiling.
    """
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


@router.get("/overall-score/{task_id}")
def get_overall_score(task_id: str):
    """
    Feature-weight-based overall score (0-100) and dimension scores for a completed analyze-site task.
    Uses Approach 2 scoring and weights from feature_weights_config.
    """
    result = _get_task_result_or_raise(task_id)
    feature_values = result.get("feature_values") or {}
    if not feature_values:
        raise HTTPException(
            status_code=422,
            detail=f"Task {task_id} has no feature_values.",
        )
    profiler_scores = get_all_profiler_scores_from_task_feature_values(feature_values)
    if not profiler_scores:
        raise HTTPException(
            status_code=422,
            detail="Could not score any features for this task.",
        )
    overall = compute_overall_score(profiler_scores)
    dimension_scores = {}
    for dim in ("Weather", "Gas", "Retail Proximity", "Competition"):
        s = compute_dimension_score(profiler_scores, dim)
        if s is not None:
            dimension_scores[dim] = s
    return {
        "task_id": task_id,
        "overall_score": overall,
        "dimension_scores": dimension_scores,
    }


# ── Agentic Quantile Profiling Helpers ────────────────────────────

# Singleton profiler (loaded once on first request)
_profiler_singleton = None
_dim_profiler_singleton = None


def _get_profilers():
    """Lazy-load the profilers (dataset loaded once, then reused)."""
    global _profiler_singleton, _dim_profiler_singleton
    if _profiler_singleton is None:
        _profiler_singleton = QuantileProfiler()
        _dim_profiler_singleton = DimensionProfiler(_profiler_singleton)
    return _profiler_singleton, _dim_profiler_singleton


def _dimension_summary(task_id: str, dimension: str) -> dict:
    """
    Shared logic for per-dimension summary endpoints.
    Uses Approach 2 (percentile + category) for Weather, Retail Proximity, Competition;
    falls back to quantile profiling for other dimensions.
    """
    result = _get_task_result_or_raise(task_id)
    feature_values = result.get("feature_values") or {}

    # Approach 2: Weather, Retail Proximity (gas), Competition
    if dimension in DIMENSION_FEATURE_MAP:
        try:
            a2 = get_dimension_summary_approach2(dimension, feature_values)
        except Exception as e:
            logger.warning(f"Approach 2 summary failed for {dimension}: {e}")
            a2 = None
        if a2 and a2.get("features_scored", 0) > 0:
            scored = a2.get("feature_scores", [])
            fit_avg = (
                sum(s.get("final_score", 0) for s in scored) / len(scored)
                if scored else 0
            )
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
            fv_slice = {
                tk: feature_values.get(tk)
                for tk in mapping
                if feature_values.get(tk) is not None
            }
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

    # Fallback: Quantile profiling
    profiler, dim_profiler = _get_profilers()
    dim_result = dim_profiler.score_dimension(dimension, feature_values)
    overall = profiler.predict(feature_values)
    summary_text = generate_dimension_rationale(
        dimension_name=dimension,
        dimension_result=dim_result,
        dimension_strength_info=dim_profiler.dimension_strength.get(dimension, {}),
        overall_category=overall["predicted_category"],
    )
    dim_features = DIMENSION_GROUPS.get(dimension, [])
    fv_slice = {k: v for k, v in feature_values.items() if k in dim_features}
    feat_perf = {}
    if dim_result.get("feature_details"):
        for feat, details in dim_result["feature_details"].items():
            feat_perf[feat] = details.get("best_fit", "N/A")

    return DimensionSummaryResponse(
        task_id=task_id,
        dimension=dimension,
        predicted_tier=dim_result["predicted"],
        fit_score=dim_result["fit_score"],
        features_scored=dim_result["features_scored"],
        feature_breakdown=dim_result.get("feature_details", {}),
        discriminatory_power=dim_profiler.dimension_strength.get(dimension, {}),
        summary=summary_text,
        feature_values_slice=fv_slice,
        feature_performance=feat_perf,
    ).model_dump()


@router.get("/profiling/summary/{task_id}")
def get_full_profiling_summary(task_id: str):
    """
    Full quantile profiling summary across ALL dimensions.
    Includes overall tier, dimension breakdown, vote, and LLM rationale.
    """
    result = _get_task_result_or_raise(task_id)
    feature_values = result.get("feature_values") or {}

    profiler, dim_profiler = _get_profilers()
    overall = profiler.predict(feature_values, return_details=True)
    dim_results = dim_profiler.score_all_dimensions(feature_values)
    vote = dim_profiler.majority_vote(dim_results)
    strengths, weaknesses, neutrals = dim_profiler.get_strengths_weaknesses(dim_results)

    rationale = generate_rationale(
        location_features=feature_values,
        overall_prediction=overall,
        dimension_results=dim_results,
        dimension_strength=dim_profiler.dimension_strength,
        vote_result=vote,
        strengths=strengths,
        weaknesses=weaknesses,
        neutrals=neutrals,
    )

    return QuantileSummaryResponse(
        task_id=task_id,
        overall_tier=vote["category"],
        overall_fit_score=overall["fit_score"],
        expected_volume=overall["expected_volume"],
        dimensions={
            dim: {
                "predicted": res["predicted"],
                "fit_score": res["fit_score"],
                "features_scored": res["features_scored"],
            }
            for dim, res in dim_results.items()
        },
        vote=vote,
        strengths=strengths,
        weaknesses=weaknesses,
        rationale=rationale,
    ).model_dump()


@router.get("/health")
def health_check():
    return {"status": "healthy", "service": "site-analysis-pipeline"}