from __future__ import annotations

from functools import lru_cache
import json
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.celery.celery_app import celery_app
from app.pnl_analysis.clustering_v2.runtime import (
    build_monthly_wash_projection_48mo,
    run_projection,
    run_quantile_projection,
)

# "Best accuracy" defaults (see daily_data/daily-data-modelling/clustering_v2/results/SYSTEM_ACCURACY_REPORT.md).
DEFAULT_TS_METHOD = "arima"
DEFAULT_LEVEL_MODEL = "rf"
DEFAULT_USE_OPENING_PREFIX = False
DEFAULT_BRIDGE_OPENING_TO_MATURE = False
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


@lru_cache(maxsize=1)
def _cohort_membership_retail_shares() -> Dict[str, Dict[str, float]]:
    """
    Cohort fallback mix used when a cluster has no usable peer history.
    """
    base = "daily_data/daily-data-modelling"
    cfg = {
        "less_than": f"{base}/less_than-2yrs-clustering-ready.csv",
        "more_than": f"{base}/master_more_than-2yrs.csv",
    }
    out: Dict[str, Dict[str, float]] = {}
    for cohort, path in cfg.items():
        df = pd.read_csv(path, usecols=["wash_count_retail", "wash_count_membership"])
        retail = float(df["wash_count_retail"].fillna(0).sum())
        membership = float(df["wash_count_membership"].fillna(0).sum())
        denom = retail + membership
        if denom <= 0:
            out[cohort] = {"retail_share": 0.0, "membership_share": 0.0}
        else:
            out[cohort] = {
                "retail_share": retail / denom,
                "membership_share": membership / denom,
            }
    return out


def _haversine_km_series(lat: float, lon: float, lat_s: pd.Series, lon_s: pd.Series) -> pd.Series:
    rad = 6371.0088
    la1 = pd.Series(np.radians(float(lat)), index=lat_s.index)
    lo1 = pd.Series(np.radians(float(lon)), index=lon_s.index)
    la2 = lat_s.astype(float).map(np.radians)
    lo2 = lon_s.astype(float).map(np.radians)
    dlat = la2 - la1
    dlon = lo2 - lo1
    a = (dlat / 2).map(np.sin) ** 2 + la1.map(np.cos) * la2.map(np.cos) * (dlon / 2).map(np.sin) ** 2
    return 2.0 * rad * a.clip(lower=0.0, upper=1.0).map(np.sqrt).map(np.arcsin)


@lru_cache(maxsize=1)
def _cluster_peer_membership_retail_shares() -> Dict[str, Dict[int, Dict[str, float]]]:
    """
    Cluster-level membership/retail shares from peer sites.
    <2y uses the explicit DBSCAN cluster in the panel.
    >2y assigns each historical row to the nearest clustering_v2 centroid, matching projection behavior.
    """
    base = "daily_data/daily-data-modelling"
    out: Dict[str, Dict[int, Dict[str, float]]] = {"less_than": {}, "more_than": {}}

    lt = pd.read_csv(
        f"{base}/less_than-2yrs-clustering-ready.csv",
        usecols=["dbscan_cluster_12km", "wash_count_retail", "wash_count_membership"],
    )
    lt = lt[lt["dbscan_cluster_12km"].notna()].copy()
    lt["dbscan_cluster_12km"] = pd.to_numeric(lt["dbscan_cluster_12km"], errors="coerce").astype("Int64")
    lt = lt[lt["dbscan_cluster_12km"].notna() & (lt["dbscan_cluster_12km"] >= 0)].copy()
    lt_grouped = lt.groupby("dbscan_cluster_12km", as_index=True)[["wash_count_retail", "wash_count_membership"]].sum()
    for cluster_id, row in lt_grouped.iterrows():
        retail = float(row["wash_count_retail"] or 0.0)
        membership = float(row["wash_count_membership"] or 0.0)
        denom = retail + membership
        if denom > 0:
            out["less_than"][int(cluster_id)] = {
                "retail_share": retail / denom,
                "membership_share": membership / denom,
            }

    gt = pd.read_csv(
        f"{base}/master_more_than-2yrs.csv",
        usecols=["latitude", "longitude", "wash_count_retail", "wash_count_membership"],
    ).dropna(subset=["latitude", "longitude"])
    centroid_path = f"{base}/clustering_v2/models/more_than/cluster_centroids_12km.json"
    centroids = json.load(open(centroid_path))["centroids"]
    cdf = pd.DataFrame(
        {
            "cluster_id": [int(c["cluster_id"]) for c in centroids],
            "lat": [float(c["lat"]) for c in centroids],
            "lon": [float(c["lon"]) for c in centroids],
        }
    )
    unique_sites = gt[["latitude", "longitude"]].drop_duplicates().reset_index(drop=True)
    assigned_cluster: List[int] = []
    for row in unique_sites.itertuples(index=False):
        d = _haversine_km_series(float(row.latitude), float(row.longitude), cdf["lat"], cdf["lon"])
        assigned_cluster.append(int(cdf.iloc[int(d.idxmin())]["cluster_id"]))
    unique_sites = unique_sites.assign(cluster_id=assigned_cluster)
    gt = gt.merge(unique_sites, on=["latitude", "longitude"], how="left")
    gt_grouped = gt.groupby("cluster_id", as_index=True)[["wash_count_retail", "wash_count_membership"]].sum()
    for cluster_id, row in gt_grouped.iterrows():
        retail = float(row["wash_count_retail"] or 0.0)
        membership = float(row["wash_count_membership"] or 0.0)
        denom = retail + membership
        if denom > 0:
            out["more_than"][int(cluster_id)] = {
                "retail_share": retail / denom,
                "membership_share": membership / denom,
            }

    return out


def _cluster_peer_share_for_projection(resp_point: Dict[str, Any], cohort: str) -> Dict[str, float]:
    fallback = _cohort_membership_retail_shares()[cohort]
    block = (resp_point.get("less_than_2yrs") if cohort == "less_than" else resp_point.get("more_than_2yrs")) or {}
    cluster = block.get("cluster") or {}
    try:
        cluster_id = int(cluster["cluster_id"])
    except (KeyError, TypeError, ValueError):
        return fallback
    return _cluster_peer_membership_retail_shares().get(cohort, {}).get(cluster_id, fallback)


def _membership_retail_count_projection(
    wash_volume_projection: Dict[str, float],
    resp_point: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    shares_by_year = {
        "year_1": _cluster_peer_share_for_projection(resp_point, "less_than"),
        "year_2": _cluster_peer_share_for_projection(resp_point, "less_than"),
        "year_3": _cluster_peer_share_for_projection(resp_point, "more_than"),
        "year_4": _cluster_peer_share_for_projection(resp_point, "more_than"),
    }
    out: Dict[str, Dict[str, float]] = {}
    for year_key, year_shares in shares_by_year.items():
        total = float(wash_volume_projection.get(year_key, 0.0) or 0.0)
        out[year_key] = {
            "membership": float(total * float(year_shares["membership_share"])),
            "retail": float(total * float(year_shares["retail_share"])),
        }
    return out


def _membership_retail_revenue_projection(
    membership_retail_count: Dict[str, Dict[str, float]],
    dollars_per_wash: float,
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for year_key, counts in membership_retail_count.items():
        out[year_key] = {
            "membership": float(float(counts.get("membership", 0.0) or 0.0) * dollars_per_wash),
            "retail": float(float(counts.get("retail", 0.0) or 0.0) * dollars_per_wash),
        }
    return out


def _cash_flow_projection(
    wash_volume_projection: Dict[str, float],
    wash_prices: List[float] | None,
    wash_pcts: List[float] | None,
    opex_years: List[float] | None,
) -> Dict[str, Dict[str, float]] | None:
    if not (
        isinstance(wash_prices, list)
        and isinstance(wash_pcts, list)
        and len(wash_prices)
        and len(wash_prices) == len(wash_pcts)
        and isinstance(opex_years, list)
        and len(opex_years) == 4
    ):
        return None
    dollars_per_wash = _effective_dollar_per_wash([float(x) for x in wash_prices], [float(x) for x in wash_pcts])
    out: Dict[str, Dict[str, float]] = {}
    for year_key, expense in zip(("year_1", "year_2", "year_3", "year_4"), [float(x) for x in opex_years]):
        total_revenue = float(float(wash_volume_projection.get(year_key, 0.0) or 0.0) * dollars_per_wash)
        total_expense = float(expense)
        out[year_key] = {
            "total_revenue": total_revenue,
            "total_expense": total_expense,
            "net_cash_flow": float(total_revenue - total_expense),
        }
    return out


@celery_app.task(bind=True, name="pnl_clustering_v2_projection")
def run_clustering_v2_projection_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Celery task: clustering_v2 greenfield projection (no plots; returns data only).
    """
    address = payload.get("address")
    lat = payload.get("latitude", payload.get("lat"))
    lon = payload.get("longitude", payload.get("lon"))
    method = payload.get("method") or "arima"
    level_model = payload.get("level_model") or DEFAULT_LEVEL_MODEL

    resp_raw = run_projection(
        address=str(address).strip() if isinstance(address, str) and address.strip() else None,
        lat=float(lat) if lat is not None else None,
        lon=float(lon) if lon is not None else None,
        method=str(method),
        use_opening_prefix_for_mature_forecast=bool(
            payload.get("use_opening_prefix_for_mature_forecast", DEFAULT_USE_OPENING_PREFIX)
        ),
        bridge_opening_to_mature_when_prefix=bool(
            payload.get("bridge_opening_to_mature_when_prefix", DEFAULT_BRIDGE_OPENING_TO_MATURE)
        ),
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
    membership_retail_count = _membership_retail_count_projection(wash_volume_projection, resp_point)
    out["membership_retail_count"] = membership_retail_count

    if isinstance(wash_prices, list) and isinstance(wash_pcts, list) and len(wash_prices) and len(wash_prices) == len(wash_pcts):
        dpw = _effective_dollar_per_wash([float(x) for x in wash_prices], [float(x) for x in wash_pcts])
        out["dollars_per_wash"] = dpw
        out["membership_retail_revenue"] = _membership_retail_revenue_projection(membership_retail_count, dpw)
        years = ["year_1", "year_2", "year_3", "year_4"]
        washes = [float(wash_volume_projection.get(y, 0.0) or 0.0) for y in years]
        revenue = [float(dpw * w) for w in washes]
        out["revenue_by_year"] = dict(zip(years, revenue))
        if isinstance(opex_years, list) and len(opex_years) == 4:
            opex = [float(x) for x in opex_years]
            out["opex_by_year"] = dict(zip(years, opex))
            out["profit_by_year"] = dict(zip(years, [float(r - e) for r, e in zip(revenue, opex)]))

    cash_flow_projections = _cash_flow_projection(wash_volume_projection, wash_prices, wash_pcts, opex_years)
    if cash_flow_projections is not None:
        out["cash_flow_projections"] = cash_flow_projections

    return out
