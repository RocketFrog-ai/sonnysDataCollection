import json
import logging
import warnings
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from celery.result import AsyncResult
from fastapi import APIRouter, HTTPException

from app.utils import common as calib
from app.server.config import (
    DIMENSIONS,
    WEATHER_METRIC_CONFIG,
    WEATHER_METRIC_DISPLAY,
    WEATHER_METRIC_TO_V3_FEATURE,
    get_weather_metric_value_from_climate,
    COMPETITION_METRIC_TO_V3_FEATURE,
    nearest_brand_strength_from_quantile,
    RETAIL_RADIUS_NEAR_MILES,
    RETAIL_RADIUS_FAR_MILES,
    RETAIL_SCORE_V3_KEYS,
    GAS_RADIUS_NEAR_MILES,
    GAS_RADIUS_FAR_MILES,
    GAS_SCORE_V3_KEYS,
    is_high_traffic_gas_brand,
    SITE_SCORE_WEIGHTS,
    SITE_SCORE_CATEGORY_WEIGHTS,
    SITE_SCORE_FEATURE_CATEGORY,
)
from app.server.site_verdict import build_overall_site_analysis_verdict
from app.server.models import (
    AnalyseRequest,
    ClusterProjectionRequest,
    ClusterRangeCheckRequest,
    TaskResponse,
    TaskStatus,
    TaskStatusResponse,
    DimensionSummaryResponse,
    QuantileSummaryResponse,
)
from app.features.active.trafficLights.nearby_traffic_lights import get_traffic_lights_summary
from app.features.active.nearbyStores.nearby_stores import get_nearby_stores_data
from app.server.db_cache import get_all_site_analysis_cache
from app.modelling.site_analysis import run_site_analysis
from app.celery.celery_app import celery_app
from app.modelling.ds.scorer import (
    get_all_profiler_scores_from_task_feature_values,
    compute_dimension_score,
    compute_overall_score,
)
from app.modelling.ds.dimension_summary import (
    get_dimension_summary_approach2,
    build_full_profiling_rationale,
    _overall_score_to_category,
    DIMENSION_FEATURE_MAP,
)
from app.modelling.ds.prediction import get_category_for_quantile
from app.features.active.nearbyCompetitors.classify_competitor_types import classify_competitors
from app.server.cluster_portable_model import load_portable, predict_wash_count, range_stats_dataframe

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Router
# -----------------------------------------------------------------------------

logger = logging.getLogger(__name__)
router = APIRouter()
PLACE_DETAILS_URL = "https://places.googleapis.com/v1/places/"
ROOT_DIR = Path(__file__).resolve().parents[2]
_CLUSTER_RUNTIME_CACHE: Dict[str, Dict[str, Any]] = {}
_CLUSTER_CONFIG = {
    "12km": {"cluster_col": "dbscan_cluster_12km", "radius_km": 12.0},
    "18km": {"cluster_col": "dbscan_cluster_18km", "radius_km": 18.0},
}
# Hide Ridge + projection outputs when nearest train centroid is farther than this (km).
MAX_NEAREST_CLUSTER_DISPLAY_KM = 20.0
_CLUSTER_SEGMENT_CONFIG = {
    "more_than_2yrs": {
        "data_path": ROOT_DIR / "daily_data" / "daily-data-modelling" / "more_than-2yrs.csv",
        "model_dir": ROOT_DIR / "daily_data" / "daily-data-modelling" / "clustering" / "models",
    },
    "less_than_2yrs": {
        "data_path": ROOT_DIR / "daily_data" / "daily-data-modelling" / "less_than-2yrs-clustering-ready.csv",
        "model_dir": ROOT_DIR / "daily_data" / "daily-data-modelling" / "clustering" / "models" / "less_than_2yrs",
    },
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

# Retail / map: cap how many anchors we show per type (narrative + map stay aligned).
_ANCHOR_DISPLAY_CAP = 5


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


def _haversine_km_scalar_to_many(lat: float, lon: float, lat_arr: np.ndarray, lon_arr: np.ndarray) -> np.ndarray:
    r = 6371.0
    lat1 = np.radians(lat)
    lon1 = np.radians(lon)
    lat2 = np.radians(lat_arr)
    lon2 = np.radians(lon_arr)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return r * (2 * np.arcsin(np.sqrt(a)))


def _load_cluster_runtime_assets(radius: str, segment: str) -> Dict[str, Any]:
    cache_key = f"{segment}:{radius}"
    if cache_key in _CLUSTER_RUNTIME_CACHE:
        return _CLUSTER_RUNTIME_CACHE[cache_key]

    if radius not in _CLUSTER_CONFIG:
        raise HTTPException(status_code=400, detail="radius must be one of: 12km, 18km")
    if segment not in _CLUSTER_SEGMENT_CONFIG:
        raise HTTPException(status_code=400, detail=f"Unsupported segment: {segment}")

    cfg = _CLUSTER_CONFIG[radius]
    seg_cfg = _CLUSTER_SEGMENT_CONFIG[segment]
    cluster_col = cfg["cluster_col"]
    radius_km = cfg["radius_km"]
    data_path = seg_cfg["data_path"]
    model_dir = seg_cfg["model_dir"]
    portable_path = model_dir / f"wash_count_model_{radius}.portable.json"
    if not portable_path.exists():
        raise HTTPException(
            status_code=500,
            detail=(
                f"Portable model missing: {portable_path}. "
                "Generate it with: python daily_data/daily-data-modelling/clustering/cluster_model_eval.py"
            ),
        )
    try:
        portable = load_portable(portable_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load portable model: {str(e)}")

    required = [
        "features",
        "cluster_col",
        "imputer_statistics",
        "scaler_mean",
        "scaler_scale",
        "ridge_coef",
        "ridge_intercept",
        "train_range_stats",
    ]
    missing = [k for k in required if k not in portable]
    if missing:
        raise HTTPException(status_code=500, detail=f"Portable model invalid; missing keys: {missing}")

    if portable["cluster_col"] != cluster_col:
        raise HTTPException(
            status_code=500,
            detail=f"Portable model cluster_col mismatch: expected {cluster_col}, got {portable['cluster_col']}",
        )

    try:
        range_stats = range_stats_dataframe(portable)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Invalid train_range_stats in portable model: {str(e)}")

    if not data_path.exists():
        raise HTTPException(status_code=500, detail=f"Cluster source data missing: {data_path}")

    try:
        df = pd.read_csv(data_path, low_memory=False)
        df["calendar_day"] = pd.to_datetime(df["calendar_day"])
        sub = df[
            (df[cluster_col] != -1)
            & df["latitude"].notna()
            & df["longitude"].notna()
            & df["wash_count_total"].notna()
        ].copy()
        sub = sub.sort_values("calendar_day").reset_index(drop=True)
        train = sub.iloc[: int(len(sub) * 0.80)].copy()
        centroids = (
            train.groupby(cluster_col)
            .agg(centroid_lat=("latitude", "mean"), centroid_lon=("longitude", "mean"))
            .reset_index()
            .rename(columns={cluster_col: "cluster_id"})
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed loading cluster centroids: {str(e)}")

    assets = {
        "segment": segment,
        "portable": portable,
        "range_stats": range_stats,
        "centroids": centroids,
        "history_df": sub[[cluster_col, "calendar_day", "site_client_id", "wash_count_total"]].copy(),
        "cluster_col": cluster_col,
        "radius_km": radius_km,
        "portable_path": str(portable_path),
    }
    _CLUSTER_RUNTIME_CACHE[cache_key] = assets
    return assets


def _assign_cluster_from_latlon(
    lat: float, lon: float, centroids: pd.DataFrame, radius_km: float
) -> Tuple[int, float, bool]:
    """Nearest train centroid by haversine km. Always assigns that cluster (V2-style).

    ``within_gate`` is True iff distance <= ``radius_km`` (legacy DBSCAN gate for disclosure).
    """
    if centroids.empty:
        return -1, float("nan"), False
    dists = _haversine_km_scalar_to_many(
        lat, lon, centroids["centroid_lat"].to_numpy(), centroids["centroid_lon"].to_numpy()
    )
    best_idx = int(np.argmin(dists))
    best_dist = float(dists[best_idx])
    best_cluster = int(centroids.iloc[best_idx]["cluster_id"])
    within_gate = bool(best_dist <= radius_km)
    return best_cluster, best_dist, within_gate


def _peer_coverage_message(within_gate: bool, dist_km: float, radius_km: float) -> Optional[str]:
    if within_gate or not np.isfinite(dist_km):
        return None
    return (
        f"Nearest cluster assigned; {dist_km:.1f} km to centroid exceeds {radius_km:.0f} km peer gate "
        "(weak local coverage — interpret with caution)."
    )


def _v1_cumulative_from_bars(segment: Dict[str, Any], horizon_months: int) -> Optional[float]:
    bars = (segment.get("projection") or {}).get("bar_graph_data") or []
    for b in bars:
        if int(b.get("horizon_months", -1)) == int(horizon_months) and b.get("wash_count") is not None:
            return float(b["wash_count"])
    return None


def _bridge_v1_more_forecast_to_opening_last_month(more: Dict[str, Any], less: Dict[str, Any]) -> None:
    """Scale >2y monthly_forecast_next_24 so month 25 matches <2y month 24 before mature remap."""
    if more.get("assigned_cluster_id") is None or less.get("assigned_cluster_id") is None:
        return
    mp = more.get("projection") or {}
    lp = less.get("projection") or {}
    fc = mp.get("monthly_forecast_next_24")
    fc_less = lp.get("monthly_forecast_next_24")
    if not isinstance(fc, list) or len(fc) < 24 or not isinstance(fc_less, list) or len(fc_less) < 24:
        return
    lt_last = float(fc_less[23])
    gt_first = float(fc[0])
    if not np.isfinite(gt_first) or gt_first <= 1e-12 or not np.isfinite(lt_last):
        return
    factor = float(lt_last / gt_first)
    mp["monthly_forecast_next_24"] = [float(max(float(v) * factor, 0.0)) for v in fc]
    mp["opening_to_mature_bridge"] = {
        "method": "scale_entire_mature_monthly_track",
        "scale_factor": factor,
        "aligned_operational_month_24_lt_to_month_25_gt": True,
    }


def _remap_v1_more_than_projection(more: Dict[str, Any], less: Dict[str, Any]) -> None:
    """>2y bars: four disjoint 6-month mature sums (months 25–30 … 43–48), not running cumulative."""
    if more.get("assigned_cluster_id") is None:
        return
    mp = more.get("projection") or {}
    fc = mp.get("monthly_forecast_next_24")
    if not isinstance(fc, list) or len(fc) < 24:
        return
    w = [float(x) for x in fc]
    bars = []
    periodv: Dict[str, float] = {}
    avgv: Dict[str, float] = {}
    for end_m, lo, hi in ((30, 0, 6), (36, 6, 12), (42, 12, 18), (48, 18, 24)):
        inc = float(np.sum(np.maximum(np.array(w[lo:hi], dtype=float), 0.0)))
        bars.append({"horizon_months": end_m, "wash_count": inc})
        periodv[str(end_m)] = inc
        avgv[str(end_m)] = inc / float(hi - lo) if hi > lo else 0.0
    mp["bar_graph_data"] = bars
    mp["horizons_months"] = [30, 36, 42, 48]
    mp["six_month_period_wash_count"] = periodv
    mp["avg_monthly_wash_in_period"] = avgv
    mp["monthly_forecast_next_24"] = [float(x) for x in w[5:]]
    mp["monthly_forecast_operational_indices"] = list(range(30, 49))
    mp["operational_phase"] = "mature_phase_months_30_48"
    mp["operational_calendar_months"] = (
        "Months 30–48 after site open (>2y: each bar is washes in that 6-month window only)"
    )


def _brand_new_site_continuation_v1(
    more_seg: Dict[str, Any], less_seg: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Independent 6-month period sums plus optional running total through each horizon."""
    opening: Dict[str, float] = {}
    for hm in (6, 12, 18, 24):
        v = _v1_cumulative_from_bars(less_seg, hm)
        if v is None:
            return None
        opening[str(hm)] = v
    mature: Dict[str, float] = {}
    for hm in (30, 36, 42, 48):
        v = _v1_cumulative_from_bars(more_seg, hm)
        if v is None:
            return None
        mature[str(hm)] = v
    run = 0.0
    running: Dict[str, float] = {}
    for hm in (6, 12, 18, 24, 30, 36, 42, 48):
        run += opening[str(hm)] if hm <= 24 else mature[str(hm)]
        running[str(hm)] = run
    return {
        "narrative": (
            "Each bar is washes in a single 6-month window. running_total_through_operational_month sums "
            "those windows in calendar order through month 48."
        ),
        "six_month_period_wash_count_by_label": {f"opening_{k}": v for k, v in opening.items()},
        "six_month_period_wash_count_mature": mature,
        "running_total_through_operational_month": running,
    }


def _monthly_cluster_series(df: pd.DataFrame, cluster_col: str, cluster_id: int) -> pd.DataFrame:
    """Monthly cluster profile from site-month totals."""
    sub = df[df[cluster_col].astype(int) == int(cluster_id)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["month", "cluster_monthly_mean_site", "cluster_monthly_median_site"])
    sub["month"] = pd.to_datetime(sub["calendar_day"]).dt.to_period("M").dt.to_timestamp()
    site_month = (
        sub.groupby(["month", "site_client_id"], as_index=False)["wash_count_total"]
        .sum()
        .rename(columns={"wash_count_total": "site_month_total"})
    )
    monthly_profile = (
        site_month.groupby("month")["site_month_total"]
        .agg(cluster_monthly_mean_site="mean", cluster_monthly_median_site="median")
        .reset_index()
        .sort_values("month")
    )
    return monthly_profile


def _naive_monthly(series: pd.Series, horizon: int) -> np.ndarray:
    vals = series.dropna().astype(float).values
    if len(vals) == 0:
        return np.zeros(horizon, dtype=float)
    if len(vals) < 3:
        return np.repeat(max(vals.mean(), 0.0), horizon)
    return np.repeat(max(vals[-3:].mean(), 0.0), horizon)


def _forecast_monthly(series: pd.Series, horizon: int, method: str) -> np.ndarray:
    y = series.dropna().astype(float)
    if len(y) < 6:
        return _naive_monthly(y, horizon)

    forecasts: Dict[str, np.ndarray] = {}
    try:
        if len(y) >= 24:
            hw = ExponentialSmoothing(
                y,
                trend="add",
                seasonal="add",
                seasonal_periods=12,
                initialization_method="estimated",
            ).fit(optimized=True, use_brute=False)
        else:
            hw = ExponentialSmoothing(
                y, trend="add", seasonal=None, initialization_method="estimated"
            ).fit(optimized=True, use_brute=False)
        forecasts["holt_winters"] = np.clip(hw.forecast(horizon).values.astype(float), 0.0, None)
    except Exception:
        forecasts["holt_winters"] = _naive_monthly(y, horizon)

    try:
        ar = ARIMA(
            y,
            order=(1, 1, 1),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(method_kwargs={"maxiter": 80})
        forecasts["arima"] = np.clip(ar.forecast(horizon).values.astype(float), 0.0, None)
    except Exception:
        forecasts["arima"] = _naive_monthly(y, horizon)

    m = (method or "blend").strip().lower()
    if m == "holt_winters":
        return forecasts["holt_winters"]
    if m == "arima":
        return forecasts["arima"]
    # default blend
    return (forecasts["holt_winters"] + forecasts["arima"]) / 2.0


def _projection_payload_for_segment(
    *,
    segment: str,
    radius: str,
    lat: float,
    lon: float,
    method: str,
) -> Dict[str, Any]:
    assets = _load_cluster_runtime_assets(radius, segment)
    assigned_cluster, assign_distance_km, within_gate = _assign_cluster_from_latlon(
        lat, lon, assets["centroids"], assets["radius_km"]
    )
    if assigned_cluster == -1:
        return {
            "assigned_cluster_id": None,
            "distance_to_cluster_km": assign_distance_km,
            "within_radius_gate_km": False,
            "predicted_wash_count_ridge": None,
            "historical_daily_wash_count": None,
            "projection": {
                "method": method,
                "horizons_months": [6, 12, 18, 24],
                "six_month_period_wash_count": {"6": None, "12": None, "18": None, "24": None},
                "avg_monthly_wash_in_period": {"6": None, "12": None, "18": None, "24": None},
                "bar_graph_data": [],
            },
            "message": "No cluster centroids available (unassigned).",
        }

    if np.isfinite(assign_distance_km) and float(assign_distance_km) > MAX_NEAREST_CLUSTER_DISPLAY_KM:
        return {
            "assigned_cluster_id": None,
            "distance_to_cluster_km": assign_distance_km,
            "within_radius_gate_km": within_gate,
            "predicted_wash_count_ridge": None,
            "historical_daily_wash_count": None,
            "projection": {
                "method": method,
                "horizons_months": [6, 12, 18, 24],
                "six_month_period_wash_count": {"6": None, "12": None, "18": None, "24": None},
                "avg_monthly_wash_in_period": {"6": None, "12": None, "18": None, "24": None},
                "bar_graph_data": [],
            },
            "message": (
                f"Nearest cluster centroid is {assign_distance_km:.1f} km away; "
                f"projections are suppressed beyond {MAX_NEAREST_CLUSTER_DISPLAY_KM:.0f} km."
            ),
        }

    rs = assets["range_stats"]
    cc = assets["cluster_col"]
    row = rs[rs[cc].astype(int) == int(assigned_cluster)]
    if row.empty:
        raise HTTPException(
            status_code=500,
            detail=f"No train range stats found for assigned cluster {assigned_cluster} ({segment})",
        )
    r = row.iloc[0]

    feature_map: Dict[str, Any] = {
        "latitude": float(lat),
        "longitude": float(lon),
        cc: int(assigned_cluster),
    }
    try:
        predicted = predict_wash_count(assets["portable"], feature_map)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed ({segment}): {str(e)}")

    monthly_profile = _monthly_cluster_series(assets["history_df"], cc, int(assigned_cluster))
    if monthly_profile.empty:
        raise HTTPException(
            status_code=500,
            detail=f"No monthly history found for cluster {assigned_cluster} ({segment})",
        )

    base_series = monthly_profile["cluster_monthly_median_site"]
    fc = _forecast_monthly(base_series, horizon=24, method=method)

    # Anchor forecast to site-specific model projection while preserving cluster trend shape.
    ridge_monthly_anchor = predicted * 30.4 if segment == "more_than_2yrs" else predicted
    last_cluster_median = float(base_series.iloc[-1]) if len(base_series) else 0.0
    if ridge_monthly_anchor and last_cluster_median > 0:
        scale = float(np.clip(ridge_monthly_anchor / last_cluster_median, 0.7, 1.3))
        fc = fc * scale

    horizons = [6, 12, 18, 24]
    period_sums: Dict[str, float] = {}
    avg_in_period: Dict[str, float] = {}
    for hi in horizons:
        lo = hi - 6
        period_sums[str(hi)] = float(np.sum(fc[lo:hi]))
        avg_in_period[str(hi)] = float(np.mean(fc[lo:hi])) if hi > lo else 0.0
    bars = [{"horizon_months": h, "wash_count": period_sums[str(h)]} for h in horizons]

    if segment == "less_than_2yrs":
        phase_label = "opening_phase_years_1_2"
        cal_label = "Months 1–24 after site open (<2y cohort model)"
    else:
        phase_label = "mature_phase_years_3_4"
        cal_label = "Months 25–48 after site open (>2y cohort model, mature-site dynamics)"

    out: Dict[str, Any] = {
        "assigned_cluster_id": int(assigned_cluster),
        "distance_to_cluster_km": assign_distance_km,
        "within_radius_gate_km": within_gate,
        "predicted_wash_count_ridge": predicted,
        "historical_daily_wash_count": {
            "min": float(r["train_min"]),
            "p10": float(r["train_p10"]),
            "median": float(r["train_median"]),
            "p90": float(r["train_p90"]),
            "max": float(r["train_max"]),
        },
        "projection": {
            "method": method,
            "horizons_months": horizons,
            "six_month_period_wash_count": period_sums,
            "avg_monthly_wash_in_period": avg_in_period,
            "bar_graph_data": bars,
            "monthly_forecast_next_24": [float(x) for x in fc.tolist()],
            "monthly_forecast_operational_indices": (
                list(range(1, 25)) if segment == "less_than_2yrs" else list(range(25, 49))
            ),
            "operational_phase": phase_label,
            "operational_calendar_months": cal_label,
        },
    }
    msg = _peer_coverage_message(within_gate, assign_distance_km, float(assets["radius_km"]))
    if msg:
        out["message"] = msg
    return out


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
    """
    Resolve marker coordinates from:
    1) embedded latitude/longitude (new tasks),
    2) Place Details lookup by place_id (older tasks),
    3) address geocode fallback.
    """
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
# Standalone Cluster API (independent of /analyze-site task flow)
# -----------------------------------------------------------------------------

@router.post("/cluster/standalone/range-check")
def cluster_range_check(req: ClusterRangeCheckRequest):
    """
    New-site / site-selection: address OR lat/lon + radius.
    Returns Ridge prediction, train-time historical min/p10/median/p90/max for the assigned cluster, and distance to cluster centroid (km).
    """
    radius = (req.radius or "12km").strip().lower()

    lat, lon = req.latitude, req.longitude
    if lat is None or lon is None:
        if not req.address:
            raise HTTPException(
                status_code=400,
                detail="Provide either latitude/longitude or address for cluster assignment.",
            )
        lat, lon = _lat_lon_from_address_or_400(req.address)

    def _single_segment_payload(segment: str) -> Dict[str, Any]:
        assets = _load_cluster_runtime_assets(radius, segment)
        assigned_cluster, assign_distance_km, within_gate = _assign_cluster_from_latlon(
            float(lat), float(lon), assets["centroids"], assets["radius_km"]
        )

        if assigned_cluster == -1:
            return {
                "assigned_cluster_id": None,
                "distance_to_cluster_km": assign_distance_km,
                "within_radius_gate_km": False,
                "predicted_wash_count_ridge": None,
                "historical_daily_wash_count": None,
                "message": "No cluster centroids available (unassigned).",
            }

        if np.isfinite(assign_distance_km) and float(assign_distance_km) > MAX_NEAREST_CLUSTER_DISPLAY_KM:
            return {
                "assigned_cluster_id": None,
                "distance_to_cluster_km": assign_distance_km,
                "within_radius_gate_km": within_gate,
                "predicted_wash_count_ridge": None,
                "historical_daily_wash_count": None,
                "message": (
                    f"Nearest cluster centroid is {assign_distance_km:.1f} km away; "
                    f"range check suppressed beyond {MAX_NEAREST_CLUSTER_DISPLAY_KM:.0f} km."
                ),
            }

        rs = assets["range_stats"]
        cc = assets["cluster_col"]
        row = rs[rs[cc].astype(int) == int(assigned_cluster)]
        if row.empty:
            raise HTTPException(
                status_code=500,
                detail=f"No train range stats found for assigned cluster {assigned_cluster} ({segment})",
            )
        r = row.iloc[0]

        feature_map: Dict[str, Any] = {
            "latitude": float(lat),
            "longitude": float(lon),
            cc: int(assigned_cluster),
        }
        try:
            predicted = predict_wash_count(assets["portable"], feature_map)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed ({segment}): {str(e)}")

        seg_out: Dict[str, Any] = {
            "assigned_cluster_id": int(assigned_cluster),
            "distance_to_cluster_km": assign_distance_km,
            "within_radius_gate_km": within_gate,
            "predicted_wash_count_ridge": predicted,
            "historical_daily_wash_count": {
                "min": float(r["train_min"]),
                "p10": float(r["train_p10"]),
                "median": float(r["train_median"]),
                "p90": float(r["train_p90"]),
                "max": float(r["train_max"]),
            },
        }
        msg = _peer_coverage_message(within_gate, assign_distance_km, float(assets["radius_km"]))
        if msg:
            seg_out["message"] = msg
        return seg_out

    return {
        "radius": radius,
        "more_than_2yrs": _single_segment_payload("more_than_2yrs"),
        "less_than_2yrs": _single_segment_payload("less_than_2yrs"),
    }


@router.post("/cluster/standalone/projection")
def cluster_projection(req: ClusterProjectionRequest):
    """
    New-site projection for 6/12/18/24 months.
    Uses cluster assignment + cluster monthly history + time-series forecast
    (holt-winters/arima/blend), and returns both >2y and <2y cohorts.
    """
    radius = (req.radius or "12km").strip().lower()
    method = (req.method or "blend").strip().lower()
    if method not in {"holt_winters", "arima", "blend"}:
        raise HTTPException(status_code=400, detail="method must be one of: holt_winters, arima, blend")

    lat, lon = req.latitude, req.longitude
    if lat is None or lon is None:
        if not req.address:
            raise HTTPException(
                status_code=400,
                detail="Provide either latitude/longitude or address for projection.",
            )
        lat, lon = _lat_lon_from_address_or_400(req.address)

    lat = float(lat)
    lon = float(lon)
    more = _projection_payload_for_segment(
        segment="more_than_2yrs",
        radius=radius,
        lat=lat,
        lon=lon,
        method=method,
    )
    less = _projection_payload_for_segment(
        segment="less_than_2yrs",
        radius=radius,
        lat=lat,
        lon=lon,
        method=method,
    )
    _bridge_v1_more_forecast_to_opening_last_month(more, less)
    _remap_v1_more_than_projection(more, less)
    out: Dict[str, Any] = {
        "radius": radius,
        "method": method,
        "more_than_2yrs": more,
        "less_than_2yrs": less,
    }
    stitched = _brand_new_site_continuation_v1(more, less)
    if stitched is not None:
        out["brand_new_site_continuation"] = stitched
    return out


@router.post("/analyze-site")
def analyze_site_endpoint(features: AnalyseRequest):
    """
    Enqueue site analysis: geocode → single fetch (features/active) → quantile (v3).
    Returns task_id; use GET /task/{task_id} for status and result (feature_values, quantile_result).
    tunnel_count (1–4): optional number of wash tunnels — strongly improves prediction accuracy.
    carwash_type_encoded (1–3): optional car wash type — improves quantile prediction.
    """
    if not features.address:
        raise HTTPException(status_code=400, detail="No site address provided")
    try:
        result = run_site_analysis.delay(
            features.address,
            tunnel_count=features.tunnel_count,
            carwash_type_encoded=features.carwash_type_encoded,
            tier_strategy=features.tier_strategy,
        )
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
# Metric keys: dirt-creation-days, dirt-deposit-severity, comfortable-washing-days, shutdown-risk-days.
# -----------------------------------------------------------------------------

@router.get("/weather/summary/{task_id}")
def get_weather_summary_by_task(task_id: str):
    """Dimension summary for Weather from task result (feature_values)."""
    dimension = "Weather"
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
        logger.warning("Approach 2 summary failed for %s: %s", dimension, e)
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


@router.get("/weather/data-by-task/{task_id}")
def get_weather_data_by_task(task_id: str):
    """
    Return all 4 weather metrics from the task result: value, unit, quantile_score
    (percentile e.g. 50.1), quantile (Q1–Q4), category (Poor/Fair/Good/Strong), min, max,
    and narrative summary. Category from v3: Q1→Poor, Q2→Fair, Q3→Good, Q4→Strong.
    Only available once task is fully complete (fetch + quantile + narratives).
    """
    result = _get_task_result_or_raise(task_id)
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
    metric_key_alias = {
        "dirt-trigger-days": "dirt-creation-days",
    }
    percentiles_for_score = []  # 0–100 per metric for weather_score (25% weight each)
    for metric_key in WEATHER_METRIC_CONFIG:
        value, unit = get_weather_metric_value_from_climate(climate, metric_key)
        if value is None:
            continue
        v3_key = WEATHER_METRIC_TO_V3_FEATURE.get(metric_key)
        fa = feature_analysis.get(v3_key, {}) if v3_key else {}
        narrative = narrative_by_v3_key.get(v3_key, {}) if v3_key else {}
        
        pct = narrative.get("percentile")
        if pct is None:
            pct = fa.get("adjusted_percentile")
            
        if pct is not None:
            percentiles_for_score.append(float(pct))
            
        boundaries = fa.get("quantile_boundaries") or []
        dist_min = fa.get("dist_min")
        dist_max = fa.get("dist_max")
        
        wash_q = narrative.get("wash_q")
        if wash_q is None:
            wash_q = fa.get("wash_correlated_q")
            
        feature_q = fa.get("feature_quantile_adj")
        quantile_str = f"Q{int(wash_q)}" if wash_q is not None else (f"Q{int(feature_q)}" if feature_q is not None else None)
        
        category = narrative.get("category")
        if category is None:
            category = get_category_for_quantile(wash_q) or get_category_for_quantile(feature_q)
            
        summary = narrative.get("summary")
        impact_classification = narrative.get("impact_classification")
        min_val = float(dist_min) if dist_min is not None else (float(boundaries[0]) if len(boundaries) > 0 else None)
        max_val = float(dist_max) if dist_max is not None else (float(boundaries[-1]) if len(boundaries) > 0 else None)
        display_name, subtitle = WEATHER_METRIC_DISPLAY.get(metric_key, (metric_key, ""))
        metrics.append({
            "metric_key": metric_key_alias.get(metric_key, metric_key),
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
            "impact_classification": impact_classification,
        })
    # weather_score: 25% of each metric's percentile (0–100) → mean of 4 percentiles, range 0–100
    weather_score = None
    if percentiles_for_score:
        weather_score = round(sum(percentiles_for_score) / len(percentiles_for_score), 1)
    narratives_overall = (result.get("narratives") or {}).get("overall") or {}
    has_overall = any(
        narratives_overall.get(k)
        for k in ("insight", "observation", "pro", "con", "conclusion")
    )
    complete = bool(quantile_result and has_overall)
    return {
        "task_id": task_id,
        "address": result.get("address"),
        "complete": complete,
        "success": complete,
        "weather_score": weather_score,
        "metrics": metrics,
        "overall": {
            "insight": narratives_overall.get("insight"),
            "observation": {
                "pro": narratives_overall.get("pro"),
                "con": narratives_overall.get("con"),
            },
            "conclusion": narratives_overall.get("conclusion"),
        },
    }


# -----------------------------------------------------------------------------
# Competition (nearby same-format car washes, 4-mile radius)
# -----------------------------------------------------------------------------

@router.get("/competition/data-by-task/{task_id}")
def get_competition_data_by_task(task_id: str):
    result = _get_task_result_or_raise(task_id)
    fetched = result.get("fetched") or {}
    competitors_data = fetched.get("competitors_data") or {}
    feature_values = result.get("feature_values") or {}
    quantile_result = result.get("quantile_result") or {}
    feature_analysis = quantile_result.get("feature_analysis") or {}
    narratives_comp = (result.get("narratives") or {}).get("competition") or {}

    competitors_list = competitors_data.get("competitors") or []
    count = competitors_data.get("count") or len(competitors_list) or feature_values.get("count", 0)
    nearest = competitors_list[0] if competitors_list else {}
    distance_to_nearest = nearest.get("distance_miles") if nearest else feature_values.get("competitor_1_distance_miles")
    nearest_rating = nearest.get("rating") if nearest else feature_values.get("competitor_1_google_rating")
    nearest_review_count = nearest.get("user_rating_count") or nearest.get("rating_count") if nearest else feature_values.get("competitor_1_rating_count")
    fa_quality = feature_analysis.get("competition_quality") or {}
    nearest_brand_strength = nearest_brand_strength_from_quantile(
        category=fa_quality.get("category"),
        wash_q=fa_quality.get("wash_correlated_q") or fa_quality.get("feature_quantile_adj"),
    )

    percentiles_for_score = []
    for v3_key in COMPETITION_METRIC_TO_V3_FEATURE.values():
        fa = feature_analysis.get(v3_key, {})
        pct = fa.get("adjusted_percentile")
        if pct is not None:
            percentiles_for_score.append(float(pct))
    competition_score = None
    if percentiles_for_score:
        competition_score = round(sum(percentiles_for_score) / len(percentiles_for_score), 1)

    # Run AI classification pipeline (scrape → Gemini AI) on each competitor.
    # classify_competitors handles: DB cache read, website scraping, AI call, DB cache write.
    # Each competitor dict already has "website" from Place Details (get_nearby_competitors).
    classified_list = classify_competitors(competitors_list) if competitors_list else []

    nearby_list = []
    for c in classified_list:
        classification = c.get("classification") or {}
        entry = {
            "name": c.get("name"),
            "rating": float(c["rating"]) if c.get("rating") is not None else None,
            "user_rating_count": c.get("user_rating_count") or c.get("rating_count"),
            "address": c.get("address"),
            "distance_miles": float(c["distance_miles"]) if c.get("distance_miles") is not None else None,
            # official_website: the URL that was actually used for classification
            # (Place Details websiteUri, or fallback found by find_official_website)
            "official_website": c.get("website"),
            # primary_carwash_type: AI-classified type (e.g. "Express Tunnel", "Full Service")
            "primary_carwash_type": classification.get("primary_type") if classification else None,
            # Full classification payload for richer downstream use,
        }
        nearby_list.append(entry)

    has_comp_narrative = any(
        narratives_comp.get(k) for k in ("insight", "observation", "pro", "con")
    )
    complete = bool(quantile_result and has_comp_narrative)

    return {
        "task_id": task_id,
        "address": result.get("address"),
        "complete": complete,
        "success": complete,
        "competition_score": competition_score,
        "nearby_car_washes": {
            "count": count,
            "list": nearby_list,
        },
        "nearest": {
            "distance_miles": float(distance_to_nearest) if distance_to_nearest is not None else None,
            "brand_strength": nearest_brand_strength,
            "rating": float(nearest_rating) if nearest_rating is not None else None,
            "user_rating_count": nearest_review_count,
        },
        "overall": {
            "insight": narratives_comp.get("insight"),
            "observation": {
                "pro": narratives_comp.get("pro"),
                "con": narratives_comp.get("con"),
            },
        },
    }


# -----------------------------------------------------------------------------
# Retail — data-by-task
# -----------------------------------------------------------------------------

@router.get("/retail/data-by-task/{task_id}")
def get_retail_data_by_task(task_id: str):
    result = _get_task_result_or_raise(task_id)

    fetched = result.get("fetched") or {}
    retail_anchors_data = fetched.get("retail_anchors") or {}
    quantile_result = result.get("quantile_result") or {}
    feature_analysis = quantile_result.get("feature_analysis") or {}
    narratives_retail = (result.get("narratives") or {}).get("retail") or {}

    anchors = retail_anchors_data.get("anchors") or []
    within_1 = [a for a in anchors if a.get("distance_miles") is not None and a["distance_miles"] <= RETAIL_RADIUS_NEAR_MILES]
    within_3 = [a for a in anchors if a.get("distance_miles") is not None and RETAIL_RADIUS_NEAR_MILES < a["distance_miles"] <= RETAIL_RADIUS_FAR_MILES]
    nearest = anchors[0] if anchors else {}

    within_1 = _cap_anchors_by_type(within_1, _ANCHOR_DISPLAY_CAP)
    within_3 = _cap_anchors_by_type(within_3, _ANCHOR_DISPLAY_CAP)

    # Named anchor lookup: nearest per class from fetched v3 values (pre-computed)
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

    # retail_score: mean of key v3 percentiles
    percentiles_for_score = []
    for v3_key in RETAIL_SCORE_V3_KEYS:
        fa = feature_analysis.get(v3_key, {})
        pct = fa.get("adjusted_percentile")
        if pct is not None:
            percentiles_for_score.append(float(pct))
    retail_score = round(sum(percentiles_for_score) / len(percentiles_for_score), 1) if percentiles_for_score else None

    has_narrative = any(
        narratives_retail.get(k)
        for k in ("insight", "observation", "pro", "con", "conclusion")
    )
    complete = bool(quantile_result and has_narrative)

    return {
        "task_id": task_id,
        "address": result.get("address"),
        "complete": complete,
        "success": complete,
        "retail_score": retail_score,
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
        "narratives": {
            "insight": narratives_retail.get("insight"),
            "observation": {
                "pro": narratives_retail.get("pro"),
                "con": narratives_retail.get("con"),
            },
            "conclusion": narratives_retail.get("conclusion"),
        },
    }


# -----------------------------------------------------------------------------
# Gas — data-by-task
# -----------------------------------------------------------------------------

@router.get("/gas/data-by-task/{task_id}")
def get_gas_data_by_task(task_id: str):
    result = _get_task_result_or_raise(task_id)

    fetched = result.get("fetched") or {}
    quantile_result = result.get("quantile_result") or {}
    feature_analysis = quantile_result.get("feature_analysis") or {}
    narratives_gas = (result.get("narratives") or {}).get("gas") or {}

    gas_list_raw = fetched.get("gas_stations") or []

    # Normalise each station (distance_miles, rating, user_rating_count/rating_count, name)
    stations: list = []
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
    #within_3 = [s for s in stations if s.get("distance_miles") is not None and s["distance_miles"] <= GAS_RADIUS_FAR_MILES]
    within_3 = [s for s in stations if s.get("distance_miles") is not None and GAS_RADIUS_NEAR_MILES < s["distance_miles"] <= GAS_RADIUS_FAR_MILES]
    nearest = stations[0] if stations else {}

    # Fallback nearest details from feature_values if no stations in fetched
    if not nearest:
        fv = result.get("feature_values") or {}
        nearest = {
            "name": None,
            "distance_miles": fv.get("distance_from_nearest_gas_station"),
            "rating": fv.get("nearest_gas_station_rating"),
            "user_rating_count": fv.get("nearest_gas_station_rating_count"),
            "high_traffic_brand": False,
        }

    # ---- gas_score: mean of key v3 percentiles ----
    percentiles_for_score = []
    for v3_key in GAS_SCORE_V3_KEYS:
        fa = feature_analysis.get(v3_key, {})
        pct = fa.get("adjusted_percentile")
        if pct is not None:
            percentiles_for_score.append(float(pct))
    gas_score = round(sum(percentiles_for_score) / len(percentiles_for_score), 1) if percentiles_for_score else None

    has_narrative = any(
        narratives_gas.get(k)
        for k in ("insight", "observation", "pro", "con", "conclusion")
    )
    complete = bool(quantile_result and has_narrative)

    return {
        "task_id": task_id,
        "address": result.get("address"),
        "complete": complete,
        "success": complete,
        "gas_score": gas_score,
        "nearest": {
            "name": nearest.get("name"),
            "distance_miles": nearest.get("distance_miles"),
            "high_traffic_brand": nearest.get("high_traffic_brand", False),
        },
        "gas_stations": {
            "within_1_mile": {
                "count": len(within_1),
                "list": within_1,
            },
            "within_3_miles": {
                "count": len(within_3),
                "list": within_3,
            },
        },
        "overall": {
            "insight": narratives_gas.get("insight"),
            "observation": {
                "pro": narratives_gas.get("pro"),
                "con": narratives_gas.get("con"),
            },
            "conclusion": narratives_gas.get("conclusion"),
        },
    }


@router.get("/map/data-by-task/{task_id}")
def get_map_data_by_task(task_id: str):
    """
    Return map-ready markers for frontend rendering (Leaflet/Mapbox/etc).
    Includes origin site + nearby competitors, gas stations, and retail anchors.
    Requires the analyse-site task to be fully complete (same as other data-by-task endpoints).
    """
    result = _get_task_result_or_raise(task_id)
    fetched = result.get("fetched") or {}

    lat = result.get("lat")
    lon = result.get("lon")
    if lat is None or lon is None:
        raise HTTPException(
            status_code=422,
            detail=f"Task {task_id} has no geocoded site coordinates yet.",
        )

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
        markers.append(
            {
                "id": raw.get("place_id") or fallback_id,
                "name": raw.get("name"),
                "category": category,
                "latitude": float(mlat),
                "longitude": float(mlon),
                "distance_miles": float(distance) if distance is not None else None,
                "rating": float(raw["rating"]) if raw.get("rating") is not None else None,
                "user_rating_count": int(rating_count) if rating_count is not None else None,
                "address": raw.get("address"),
            }
        )

    # Match gas endpoint scope: within_1_mile + within_3_miles only.
    gas_list_raw = fetched.get("gas_stations") or []
    gas_stations = []
    for s in gas_list_raw:
        d = s.get("distance_miles")
        gas_stations.append(
            {
                **s,
                "distance_miles": float(d) if d is not None else None,
            }
        )
    gas_stations.sort(key=lambda s: (s.get("distance_miles") is None, s.get("distance_miles") or float("inf")))
    gas_within_1 = [s for s in gas_stations if s.get("distance_miles") is not None and s["distance_miles"] <= GAS_RADIUS_NEAR_MILES]
    gas_within_3 = [
        s
        for s in gas_stations
        if s.get("distance_miles") is not None and GAS_RADIUS_NEAR_MILES < s["distance_miles"] <= GAS_RADIUS_FAR_MILES
    ]
    gas_for_map = gas_within_1 + gas_within_3
    for idx, station in enumerate(gas_for_map, start=1):
        _add_marker(station, "gas_station", f"gas_{idx}")

    competitors = ((fetched.get("competitors_data") or {}).get("competitors") or [])
    for idx, comp in enumerate(competitors, start=1):
        _add_marker(comp, "car_wash", f"competitor_{idx}")

    # Match retail endpoint scope:
    # - within_1_mile and within_3_miles buckets
    # - cap display lists to 5 per anchor type
    retail_anchors_all = (fetched.get("retail_anchors") or {}).get("anchors") or []
    retail_within_1 = [
        a
        for a in retail_anchors_all
        if a.get("distance_miles") is not None and a["distance_miles"] <= RETAIL_RADIUS_NEAR_MILES
    ]
    retail_within_3 = [
        a
        for a in retail_anchors_all
        if a.get("distance_miles") is not None and RETAIL_RADIUS_NEAR_MILES < a["distance_miles"] <= RETAIL_RADIUS_FAR_MILES
    ]

    retail_for_map = _cap_anchors_by_type(retail_within_1, _ANCHOR_DISPLAY_CAP) + _cap_anchors_by_type(
        retail_within_3, _ANCHOR_DISPLAY_CAP
    )
    for idx, anchor in enumerate(retail_for_map, start=1):
        category = _retail_anchor_category(anchor.get("type"), anchor.get("name"))
        _add_marker(anchor, category, f"retail_{idx}")

    return {
        "task_id": task_id,
        "address": result.get("address"),
        "lat": float(lat),
        "lon": float(lon),
        "complete": bool(result.get("quantile_result")),
        "counts": {
            "markers_total": len(markers),
            "gas_stations": len([m for m in markers if m["category"] == "gas_station"]),
            "competitors": len([m for m in markers if m["category"] == "car_wash"]),
            "retail_anchors": len(
                [m for m in markers if m["category"] not in {"origin", "gas_station", "car_wash"}]
            ),
        },
        "markers": markers,
    }


# -----------------------------------------------------------------------------
# Traffic lights
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


# -----------------------------------------------------------------------------
# Nearby stores
# -----------------------------------------------------------------------------

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
# Task status & profiling
# -----------------------------------------------------------------------------

@router.get("/task/{task_id}", response_model=TaskStatusResponse)
def get_task_status(task_id: str):
    """
    Task status and result from Celery. Full `result` is present only when status is success.
    Poll until success for the complete analyse-site payload (fetch + quantile + narratives).
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
    return response


@router.get("/result/{task_id}")
def get_result_by_task(task_id: str):
    """
    Get analyse-site result by task_id. Returns full result only once all stages
    (fetch + quantile + narratives) are complete. Poll until status = success.
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
    return {"task_id": task_id, "status": task_result.state.lower(), "result": None}


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
# Overall site score (business-logic weights × v3 percentiles) + quantile
# -----------------------------------------------------------------------------

@router.get("/overall/{task_id}")
def get_overall(task_id: str):
    """
    Returns:
    - site_score (0–100): weighted composite using business-logic feature weights.
      Formula: Σ (adjusted_percentile_i × weight_i) for each feature in SITE_SCORE_WEIGHTS.
    - category_scores: per-category weighted scores (Weather, Competition, Retail, Gas).
    - feature_scores: per-feature adjusted_percentile, weight, and weighted contribution.
    - predicted_quantile: Q1–Q4 tier from v3 model.
    - predicted_tier: plain tier label (poor, fair, good, strong).
    - expected_annual_volume: min / max / label (cars/year).
    - quantile_probabilities: model confidence per tier.
    """
    result = _get_task_result_or_raise(task_id)
    quantile_result = result.get("quantile_result") or {}
    feature_analysis = quantile_result.get("feature_analysis") or {}

    if not quantile_result:
        feature_values = result.get("feature_values") or {}
        site_score = None
        category_scores: dict = {}
        feature_scores: dict = {}
        if feature_values:
            profiler_scores = get_all_profiler_scores_from_task_feature_values(feature_values)
            if profiler_scores:
                overall = compute_overall_score(profiler_scores)
                site_score = round(overall, 1)
        return {
            "task_id": task_id,
            "status": "no_quantile_result",
            "message": f"Task {task_id} has no v3 quantile_result (task may predate v3 or quantile was not run).",
            "site_score": site_score,
            # category_scores / feature_scores intentionally omitted for now
            "predicted_quantile": None,
            "predicted_tier": None,
            "expected_annual_volume": None,
            "quantile_probabilities": {},
            "tunnel_count": (result.get("feature_values") or {}).get("tunnel_count"),
            "carwash_type_encoded": (result.get("feature_values") or {}).get("carwash_type_encoded"),
            "overall_site_analysis_verdict": build_overall_site_analysis_verdict(
                predicted_tier=None,
                expected_volume_label=None,
                site_score=site_score,
                category_scores=None,
                predicted_quantile=None,
                quantile_probabilities=None,
            ),
        }

    # Compute site score
    weighted_sum = 0.0
    total_weight_used = 0.0
    feature_scores: dict = {}
    category_accum: dict = {cat: {"weighted_sum": 0.0, "weight_used": 0.0} for cat in SITE_SCORE_CATEGORY_WEIGHTS}

    for v3_key, weight in SITE_SCORE_WEIGHTS.items():
        fa = feature_analysis.get(v3_key) or {}
        pct = fa.get("adjusted_percentile")
        if pct is None:
            continue
        contribution = float(pct) * weight
        weighted_sum += contribution
        total_weight_used += weight
        category = SITE_SCORE_FEATURE_CATEGORY.get(v3_key, "Other")
        feature_scores[v3_key] = {
            "label": fa.get("label", v3_key),
            "category": category,
            "adjusted_percentile": round(float(pct), 1),
            "weight": weight,
            "weighted_contribution": round(contribution, 3),
            "imputed": fa.get("imputed", False),
        }
        if category in category_accum:
            category_accum[category]["weighted_sum"] += contribution
            category_accum[category]["weight_used"] += weight

    # Normalise: if some features were missing, scale score to 0–100 range
    site_score = round(weighted_sum / total_weight_used, 1) if total_weight_used > 0 else None

    # Small business rule boost: Express + tunnel_count > 1 tends to perform better.
    # Apply a slight uplift to the overall site score (2–3 points) for scoring display.
    fv = result.get("feature_values") or {}
    try:
        tc_val = fv.get("tunnel_count")
        cw_val = fv.get("carwash_type_encoded")
        if site_score is not None and cw_val is not None and int(cw_val) == 1 and tc_val is not None and int(tc_val) > 1:
            site_score = min(100.0, round(site_score + 2.5, 1))
    except Exception:
        pass

    # Per-category scores (0–100)
    category_scores: dict = {}
    for cat, cat_weight in SITE_SCORE_CATEGORY_WEIGHTS.items():
        acc = category_accum.get(cat, {})
        w_used = acc.get("weight_used", 0)
        w_sum = acc.get("weighted_sum", 0.0)
        category_scores[cat] = {
            "score": round(w_sum / w_used, 1) if w_used > 0 else None,
            "category_weight": cat_weight,
            "features_scored": sum(1 for f in feature_scores.values() if f["category"] == cat),
        }

    # Quantile fields
    predicted_q = quantile_result.get("predicted_wash_quantile")
    predicted_quantile = f"Q{predicted_q}" if predicted_q else None
    predicted_tier_map = {1: "poor", 2: "fair", 3: "good", 4: "strong"}
    predicted_tier = predicted_tier_map.get(predicted_q)
    wash_range = quantile_result.get("predicted_wash_range") or {}
    proba = quantile_result.get("quantile_probabilities") or {}

    response = {
        "task_id": task_id,
        "address": result.get("address"),
        "site_score": site_score,
        "predicted_quantile": predicted_quantile,
        "predicted_tier": predicted_tier,
        "expected_annual_volume": {
            "min": wash_range.get("min"),
            "max": wash_range.get("max"),
            "label": wash_range.get("label"),
        },
        "weighted_volume_prediction": quantile_result.get("weighted_volume_prediction"),
        "prediction_confidence": quantile_result.get("prediction_confidence"),
        "operational_range": {
            "low":  ((quantile_result.get("volume_uncertainty") or {}).get("low")
                     if (quantile_result.get("volume_uncertainty") or {}).get("low") is not None
                     else ((quantile_result.get("weighted_volume_prediction") - 20000)
                           if quantile_result.get("weighted_volume_prediction") is not None else None)),
            "high": ((quantile_result.get("volume_uncertainty") or {}).get("high")
                     if (quantile_result.get("volume_uncertainty") or {}).get("high") is not None
                     else ((quantile_result.get("weighted_volume_prediction") + 20000)
                           if quantile_result.get("weighted_volume_prediction") is not None else None)),
        },
        "tier_metadata": quantile_result.get("tier_metadata"),
        "quantile_probabilities": {
            f"Q{k}": round(v * 100, 1) for k, v in proba.items()
        },
        "tunnel_count": (result.get("feature_values") or {}).get("tunnel_count"),
        "carwash_type_encoded": (result.get("feature_values") or {}).get("carwash_type_encoded"),
        "overall_site_analysis_verdict": build_overall_site_analysis_verdict(
            predicted_tier=predicted_tier,
            expected_volume_label=(wash_range.get("label") if wash_range else None),
            site_score=site_score,
            category_scores=category_scores,
            predicted_quantile=predicted_quantile,
            quantile_probabilities=proba,
            weighted_volume_prediction=quantile_result.get("weighted_volume_prediction"),
            operational_buffer=20000,
        ),
    }
    return response


# -----------------------------------------------------------------------------
# Health
# -----------------------------------------------------------------------------

@router.get("/health")
def health_check():
    return {"status": "healthy", "service": "site-analysis-pipeline"}


@router.get("/cache/site-analysis/all")
def get_site_analysis_cache_all():
    data = get_all_site_analysis_cache(
        page=1,
        page_size=50,
        include_response=True,
    )
    return data
