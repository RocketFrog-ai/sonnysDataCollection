from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.celery.celery_app import celery_app
from app.pnl_analysis.clustering_v2.runtime import (
    build_monthly_wash_projection_48mo,
    run_projection,
    run_quantile_projection,
)

# "Best accuracy" defaults (see daily_data/daily-data-modelling/clustering_v2/results/SYSTEM_ACCURACY_REPORT.md).
DEFAULT_TS_METHOD = "arima"
DEFAULT_LEVEL_MODEL = "rf"
DEFAULT_USE_OPENING_PREFIX = True
DEFAULT_BRIDGE_OPENING_TO_MATURE = True
DEFAULT_ALLOW_DISTANT_NEAREST_CLUSTER = False


def _json_sanitize(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    # numpy / pandas scalars often expose `.item()`
    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return _json_sanitize(item())
        except Exception:
            pass
    return str(obj)


def _effective_dollar_per_wash(wash_prices: List[float], wash_pcts: List[float]) -> float:
    return float(sum(float(p) * max(float(c), 0.0) / 100.0 for p, c in zip(wash_prices, wash_pcts)))


@celery_app.task(bind=True, name="pnl_clustering_v2_projection")
def run_clustering_v2_projection_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Celery task: clustering_v2 greenfield projection (no plots; returns data only).
    """
    address = payload.get("address")
    lat = payload.get("latitude", payload.get("lat"))
    lon = payload.get("longitude", payload.get("lon"))
    method = payload.get("method") or "arima"
    level_model = payload.get("level_model") or "ridge"

    resp_raw = run_projection(
        address=str(address).strip() if isinstance(address, str) and address.strip() else None,
        lat=float(lat) if lat is not None else None,
        lon=float(lon) if lon is not None else None,
        method=str(method),
        use_opening_prefix_for_mature_forecast=bool(payload.get("use_opening_prefix_for_mature_forecast", True)),
        bridge_opening_to_mature_when_prefix=bool(payload.get("bridge_opening_to_mature_when_prefix", True)),
        allow_nearest_cluster_beyond_distance_cap=bool(payload.get("allow_nearest_cluster_beyond_distance_cap", False)),
        level_model=str(level_model),
    )
    resp = _json_sanitize(resp_raw)

    calendar_year_washes = resp.get("calendar_year_washes") or {}
    timeline = build_monthly_wash_projection_48mo(resp)

    out: Dict[str, Any] = {
        "task_id": self.request.id,
        "input": resp.get("input") or {"address": address, "lat": lat, "lon": lon},
        "method": resp.get("method", method),
        "level_model": resp.get("level_model", level_model),
        "calendar_year_washes": calendar_year_washes,
        "monthly_projection_48mo": timeline,
        "raw": resp,
    }

    # Optional PnL inputs: compute revenue/opex/profit summaries if present.
    wash_prices = payload.get("wash_prices")
    wash_pcts = payload.get("wash_pcts")
    opex_years = payload.get("opex_years")
    if isinstance(wash_prices, list) and isinstance(wash_pcts, list) and len(wash_prices) and len(wash_prices) == len(wash_pcts):
        dpw = _effective_dollar_per_wash([float(x) for x in wash_prices], [float(x) for x in wash_pcts])
        out["dollars_per_wash"] = dpw
        years = ["year_1", "year_2", "year_3", "year_4"]
        washes = [float(calendar_year_washes.get(y, 0.0) or 0.0) for y in years]
        revenue = [float(dpw * w) for w in washes]
        out["revenue_by_year"] = dict(zip(years, revenue))
        if isinstance(opex_years, list) and len(opex_years) == 4:
            opex = [float(x) for x in opex_years]
            out["opex_by_year"] = dict(zip(years, opex))
            out["profit_by_year"] = dict(zip(years, [float(r - e) for r, e in zip(revenue, opex)]))

    return out


@celery_app.task(bind=True, name="pnl_central_input_form")
def run_pnl_central_input_form_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Central input-form task for PnL analysis.
    Uses fixed "best accuracy" model configs; request does not include model tuning knobs.
    Returns minimal result blocks used by dedicated endpoints.
    """
    address = payload.get("address")
    lat = payload.get("latitude", payload.get("lat"))
    lon = payload.get("longitude", payload.get("lon"))

    resp_point = run_projection(
        address=str(address).strip() if isinstance(address, str) and address.strip() else None,
        lat=float(lat) if lat is not None else None,
        lon=float(lon) if lon is not None else None,
        method=DEFAULT_TS_METHOD,
        use_opening_prefix_for_mature_forecast=DEFAULT_USE_OPENING_PREFIX,
        bridge_opening_to_mature_when_prefix=DEFAULT_BRIDGE_OPENING_TO_MATURE,
        allow_nearest_cluster_beyond_distance_cap=DEFAULT_ALLOW_DISTANT_NEAREST_CLUSTER,
        level_model=DEFAULT_LEVEL_MODEL,
    )
    resp_point = _json_sanitize(resp_point)
    inp = resp_point.get("input") or {}
    out_input = {
        "address": inp.get("address") or address,
        "lat": inp.get("lat") if inp.get("lat") is not None else lat,
        "lon": inp.get("lon") if inp.get("lon") is not None else lon,
    }

    point_years = resp_point.get("calendar_year_washes") or {}
    wash_volume_projection = {
        "year_1": float(point_years.get("year_1", 0.0) or 0.0),
        "year_2": float(point_years.get("year_2", 0.0) or 0.0),
        "year_3": float(point_years.get("year_3", 0.0) or 0.0),
        "year_4": float(point_years.get("year_4", 0.0) or 0.0),
    }

    wash_volume_range_minmax: Dict[str, Any] | None = None
    try:
        q = run_quantile_projection(
            lat=float(out_input["lat"]),
            lon=float(out_input["lon"]),
            method=DEFAULT_TS_METHOD,
            address=str(out_input["address"]) if out_input.get("address") else None,
            use_opening_prefix_for_mature_forecast=DEFAULT_USE_OPENING_PREFIX,
            bridge_opening_to_mature_when_prefix=DEFAULT_BRIDGE_OPENING_TO_MATURE,
            allow_nearest_cluster_beyond_distance_cap=DEFAULT_ALLOW_DISTANT_NEAREST_CLUSTER,
        )
        q = _json_sanitize(q)
        cyq = q.get("calendar_year_washes") or {}

        def _range_block(year_key: str) -> Dict[str, float]:
            row = cyq.get(year_key) or {}
            return {
                "min": float(row.get("q10", 0.0) or 0.0),
                "median": float(row.get("q50", 0.0) or 0.0),
                "max": float(row.get("q90", 0.0) or 0.0),
            }

        wash_volume_range_minmax = {
            "year_1": _range_block("year_1"),
            "year_2": _range_block("year_2"),
            "year_3": _range_block("year_3"),
            "year_4": _range_block("year_4"),
        }
    except Exception:
        wash_volume_range_minmax = None

    out: Dict[str, Any] = {
        "task_id": self.request.id,
        "input": out_input,
        "wash_volume_projection": wash_volume_projection,
        "wash_volume_range_minmax": wash_volume_range_minmax,
    }

    wash_prices = payload.get("wash_prices")
    wash_pcts = payload.get("wash_pcts")
    opex_years = payload.get("opex_years")
    if isinstance(wash_prices, list) and isinstance(wash_pcts, list) and len(wash_prices) and len(wash_prices) == len(wash_pcts):
        dpw = _effective_dollar_per_wash([float(x) for x in wash_prices], [float(x) for x in wash_pcts])
        out["dollars_per_wash"] = dpw
        years = ["year_1", "year_2", "year_3", "year_4"]
        washes = [float(wash_volume_projection.get(y, 0.0) or 0.0) for y in years]
        revenue = [float(dpw * w) for w in washes]
        out["revenue_by_year"] = dict(zip(years, revenue))
        if isinstance(opex_years, list) and len(opex_years) == 4:
            opex = [float(x) for x in opex_years]
            out["opex_by_year"] = dict(zip(years, opex))
            out["profit_by_year"] = dict(zip(years, [float(r - e) for r, e in zip(revenue, opex)]))

    return out
