"""Greenfield Ridge projection (both cohorts). Outputs results/projection_demo/. See APPROACH.md.

Peer overlays read training panels via ``data_paths`` — the same
``daily_data/daily-data-modelling/less_than-2yrs-clustering-ready.csv`` and
``master_more_than-2yrs.csv`` used by ``build_v2.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import data_paths as _cohort_csv

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from statsmodels.tsa.api import ARIMA, ExponentialSmoothing


REPO_ROOT = Path(__file__).resolve().parents[3]
V2_DIR = Path(__file__).resolve().parent
MODELS_DIR = V2_DIR / "models"
DEMO_OUT = V2_DIR / "results" / "projection_demo"
# Peer summary on Ridge/RF charts (same radius as Streamlit peer panel).
PEER_REF_MAX_KM_FROM_PROJECTION = 20.0
# Reject nearest-centroid assignment beyond this unless --allow-distant-nearest-cluster.
MAX_NEAREST_CLUSTER_DISTANCE_KM = 20.0
MAX_NEAREST_CLUSTER_DISPLAY_KM = MAX_NEAREST_CLUSTER_DISTANCE_KM  # legacy name

sys.path.insert(0, str(REPO_ROOT))
warnings.filterwarnings("ignore")


def _load_json(p: Path) -> Any:
    return json.loads(p.read_text())


def _load_cohort(cohort: str) -> dict[str, Any]:
    d = MODELS_DIR / cohort
    out: dict[str, Any] = {
        "model": _load_json(d / "wash_count_model_12km.portable.json"),
        "centroids": _load_json(d / "cluster_centroids_12km.json"),
        "context": _load_json(d / "cluster_context_12km.json"),
        "series": _load_json(d / "cluster_monthly_series_12km.json"),
        "spec": _load_json(d / "feature_spec_12km.json"),
    }
    rf_path = d / "wash_count_model_12km.rf.joblib"
    if rf_path.is_file():
        try:
            out["rf_bundle"] = joblib.load(rf_path)
        except Exception:
            out["rf_bundle"] = None
    else:
        out["rf_bundle"] = None
    return out


def _rf_models_available() -> bool:
    return bool(
        (MODELS_DIR / "more_than" / "wash_count_model_12km.rf.joblib").is_file()
        and (MODELS_DIR / "less_than" / "wash_count_model_12km.rf.joblib").is_file()
    )


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))


def _nearest_cluster(centroids: list[dict[str, Any]], lat: float, lon: float) -> dict[str, Any]:
    best = min(centroids, key=lambda c: _haversine_km(lat, lon, c["lat"], c["lon"]))
    best = dict(best)
    best["distance_km"] = _haversine_km(lat, lon, best["lat"], best["lon"])
    return best


def _coords_from_geocode(res: dict[str, Any] | None) -> tuple[float | None, float | None]:
    if not res:
        return None, None
    la = res.get("latitude", res.get("lat"))
    lo = res.get("longitude", res.get("lon"))
    if la is None or lo is None:
        return None, None
    return float(la), float(lo)


def _resolve_latlon(address: str | None, lat: float | None, lon: float | None) -> tuple[float, float, str | None]:
    if lat is not None and lon is not None:
        return float(lat), float(lon), address
    if not address:
        raise SystemExit("Must provide --address or --lat/--lon")
    from app.utils.common import get_lat_long  # type: ignore

    stripped = str(address).strip()
    la, lo = _coords_from_geocode(get_lat_long(stripped))
    if la is None:
        if not stripped.upper().endswith("USA") and not stripped.upper().endswith("UNITED STATES"):
            la, lo = _coords_from_geocode(get_lat_long(stripped + ", USA"))
    if la is None or lo is None:
        raise SystemExit(
            f"Geocoding failed for address: {address}\n"
            "Try --lat/--lon, a fuller address (with ZIP), or set TOMTOM_GEOCODE_API_URL and "
            "TOMTOM_API_KEY in .env if geocoding is disabled."
        )
    return la, lo, address


def _apply_local_feature_medians(
    feature_vec: dict[str, float],
    feature_order: list[str],
    ctx_row: dict[str, Any] | None,
) -> dict[str, float]:
    local = (ctx_row or {}).get("local_feature_medians") or {}
    if not isinstance(local, dict) or not local:
        return feature_vec
    out = dict(feature_vec)
    for c in feature_order:
        v = out.get(c, np.nan)
        if np.isfinite(v):
            continue
        lv = local.get(c)
        if lv is None:
            continue
        try:
            fv = float(lv)
        except (TypeError, ValueError):
            continue
        if np.isfinite(fv):
            out[c] = fv
    return out


def _score_portable(
    model: dict[str, Any],
    feature_vec: dict[str, float],
    ctx_row: dict[str, Any] | None = None,
) -> float:
    order = model["feature_order"]
    vec = _apply_local_feature_medians(feature_vec, order, ctx_row)
    x = np.array([vec.get(c, np.nan) for c in order], dtype=float)
    stats = np.array(model["imputer"]["statistics"], dtype=float)
    x = np.where(np.isfinite(x), x, stats)
    mean = np.array(model["scaler"]["mean"], dtype=float)
    scale = np.array(model["scaler"]["scale"], dtype=float)
    scale = np.where(scale == 0, 1.0, scale)
    xs = (x - mean) / scale
    coef = np.array(model["ridge"]["coef"], dtype=float)
    intercept = float(model["ridge"]["intercept"])
    return float(xs @ coef + intercept)


def _score_rf_bundle(bundle: dict[str, Any], feature_vec: dict[str, float], ctx_row: dict[str, Any] | None) -> float:
    order = bundle["feature_order"]
    vec = _apply_local_feature_medians(feature_vec, order, ctx_row)
    x = np.array([[vec.get(c, np.nan) for c in order]], dtype=float)
    pipe = bundle["pipeline"]
    return float(pipe.predict(x)[0])


def _feature_vector(
    spec: dict[str, Any],
    context_row: dict[str, Any] | None,
    lat: float,
    lon: float,
    cluster_id: int,
) -> dict[str, float]:
    vec: dict[str, float] = {
        "latitude": lat,
        "longitude": lon,
        "dbscan_cluster_12km": float(cluster_id),
    }
    if context_row:
        for k, v in context_row.items():
            if k in ("cluster_id", "local_feature_medians"):
                continue
            if isinstance(v, dict):
                continue
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                continue
            vec[k] = float(v)
    return vec


def _series_to_df(series: list[dict[str, Any]]) -> pd.Series:
    if not series:
        return pd.Series(dtype=float)
    df = pd.DataFrame(series)
    df["month"] = pd.to_datetime(df["month"])
    s = df.set_index("month")["value"].astype(float).sort_index()
    s = s.asfreq("MS").interpolate(limit_direction="both")
    return s


def _forecast(series: pd.Series, horizon: int, method: str) -> pd.Series:
    if len(series) < 3:
        last = float(series.iloc[-1]) if len(series) else 0.0
        idx = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(1) if len(series) else pd.Timestamp("2025-01-01"),
                            periods=horizon, freq="MS")
        return pd.Series([last] * horizon, index=idx)

    idx_future = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    hw = None
    ar = None

    if method in ("holt_winters", "blend"):
        try:
            if len(series) >= 24:
                hw_model = ExponentialSmoothing(
                    series, trend="add", seasonal="add", seasonal_periods=12
                ).fit(optimized=True, use_brute=True)
            else:
                hw_model = ExponentialSmoothing(series, trend="add").fit()
            hw = pd.Series(hw_model.forecast(horizon).to_numpy(), index=idx_future)
        except Exception:
            hw = None

    if method in ("arima", "blend"):
        try:
            ar_model = ARIMA(series, order=(1, 1, 1)).fit()
            ar = pd.Series(ar_model.forecast(horizon).to_numpy(), index=idx_future)
        except Exception:
            ar = None

    if method == "holt_winters" and hw is not None:
        return hw
    if method == "arima" and ar is not None:
        return ar
    if method == "blend":
        if hw is not None and ar is not None:
            return (hw + ar) / 2.0
        if hw is not None:
            return hw
        if ar is not None:
            return ar
    return pd.Series([float(series.iloc[-1])] * horizon, index=idx_future)


def _series_for_mature_forecast_with_opening_context(
    cluster_series: pd.Series,
    opening_monthly_wash: list[float],
    *,
    tail_months: int = 72,
) -> pd.Series:
    o = np.maximum(np.array(opening_monthly_wash, dtype=float), 0.0)
    if len(o) != 24:
        return cluster_series
    if cluster_series is None or len(cluster_series) == 0:
        idx0 = pd.Timestamp("2000-01-01")
        return pd.Series(o, index=pd.date_range(idx0, periods=24, freq="MS")).asfreq("MS")
    tail_n = min(int(tail_months), len(cluster_series))
    tail = cluster_series.tail(tail_n)
    next_idx = tail.index[-1] + pd.offsets.MonthBegin(1)
    opening_s = pd.Series(o, index=pd.date_range(next_idx, periods=24, freq="MS"))
    combined = pd.concat([tail, opening_s]).sort_index()
    combined = combined.astype(float).asfreq("MS").interpolate(limit_direction="both")
    return combined


def _project_cohort(
    cohort: str,
    cohort_label: str,
    lat: float,
    lon: float,
    method: str,
    cohort_assets: dict[str, Any],
    days_per_month: int,
    is_daily_model: bool,
    opening_prefix_monthly: list[float] | None = None,
    *,
    allow_nearest_cluster_beyond_distance_cap: bool = False,
    level_model: str = "ridge",
) -> dict[str, Any]:
    centroids = cohort_assets["centroids"]["centroids"]
    if not centroids:
        return {"error": f"No centroids for {cohort_label}"}

    nearest = _nearest_cluster(centroids, lat, lon)
    cluster_id = int(nearest["cluster_id"])
    d_km = float(nearest["distance_km"])
    if (
        not allow_nearest_cluster_beyond_distance_cap
        and np.isfinite(d_km)
        and d_km > MAX_NEAREST_CLUSTER_DISTANCE_KM
    ):
        return {
            "error": (
                f"Nearest cluster is {d_km:.1f} km away; projections hidden when centroid distance "
                f"exceeds {MAX_NEAREST_CLUSTER_DISTANCE_KM:.0f} km (use --allow-distant-nearest-cluster to force)."
            ),
            "cohort": cohort_label,
            "cluster": {
                "cluster_id": cluster_id,
                "distance_km": d_km,
                "size": nearest.get("size"),
            },
        }

    ctx_row = None
    for r in cohort_assets["context"]["records"]:
        if int(r.get("cluster_id", -999)) == cluster_id:
            ctx_row = r
            break

    vec = _feature_vector(cohort_assets["spec"], ctx_row, lat, lon, cluster_id)
    if level_model == "ridge":
        anchor_raw = _score_portable(cohort_assets["model"], vec, ctx_row)
        pred_key = "ridge_prediction"
    elif level_model == "rf":
        bundle = cohort_assets.get("rf_bundle")
        if not bundle:
            return {
                "error": (
                    "RandomForest level model not found. Run `python build_v2.py` from clustering_v2 "
                    "to write models/*/wash_count_model_12km.rf.joblib"
                ),
                "cohort": cohort_label,
                "cluster": {
                    "cluster_id": cluster_id,
                    "distance_km": d_km,
                    "size": nearest.get("size"),
                },
            }
        anchor_raw = _score_rf_bundle(bundle, vec, ctx_row)
        pred_key = "rf_prediction"
    else:
        raise ValueError(f"Unknown level_model={level_model!r}")

    anchor_monthly = anchor_raw * days_per_month if is_daily_model else anchor_raw

    series = _series_to_df(cohort_assets["series"]["series"].get(str(cluster_id), []))
    cluster_monthly_level = float(series.tail(6).mean()) if len(series) else float("nan")

    if len(series) >= 6 and np.isfinite(anchor_monthly) and cluster_monthly_level > 0:
        scale = anchor_monthly / cluster_monthly_level
    else:
        scale = 1.0

    used_lt_prefix = (
        cohort == "more_than"
        and opening_prefix_monthly is not None
        and len(opening_prefix_monthly) == 24
    )
    forecast_input = (
        _series_for_mature_forecast_with_opening_context(series, opening_prefix_monthly)
        if used_lt_prefix
        else series
    )
    fc_raw = _forecast(forecast_input, 24, method)
    fc = fc_raw * scale

    horizons: dict[str, Any] = {}
    for hi in (6, 12, 18, 24):
        lo = hi - 6
        slice_ = fc.iloc[lo:hi]
        s = float(np.maximum(slice_, 0).sum()) if len(slice_) else 0.0
        horizons[f"{hi}m"] = {
            "six_month_period_sum": s,
            "monthly_avg_in_period": float(np.maximum(slice_, 0).mean()) if len(slice_) else 0.0,
            "period_operational_month_start": lo + 1,
            "period_operational_month_end": hi,
            "first_month": slice_.index[0].strftime("%Y-%m-%d") if len(slice_) else None,
            "last_month": slice_.index[-1].strftime("%Y-%m-%d") if len(slice_) else None,
        }

    pred_block = {
        "raw_level": anchor_raw,
        "monthly_level": anchor_monthly if np.isfinite(anchor_monthly) else None,
        "anchor_scale": scale,
        "cluster_context_used": ctx_row is not None,
        "level_model": level_model,
    }
    out: dict[str, Any] = {
        "cohort": cohort_label,
        "level_model": level_model,
        "cluster": {
            "cluster_id": cluster_id,
            "distance_km": nearest["distance_km"],
            "size": nearest["size"],
            "centroid_lat": nearest["lat"],
            "centroid_lon": nearest["lon"],
            "cluster_monthly_level_last_6mo": cluster_monthly_level if np.isfinite(cluster_monthly_level) else None,
            "nearest_cluster_max_distance_km": MAX_NEAREST_CLUSTER_DISTANCE_KM,
            "distance_cap_relaxed": bool(allow_nearest_cluster_beyond_distance_cap),
        },
        pred_key: pred_block,
        "mature_forecast_series_mode": (
            "cluster_tail_72m_plus_lt2y_monthly_wash_24m_prefix_then_forecast_25_48"
            if used_lt_prefix
            else "cluster_monthly_only"
        ),
        "used_lt2y_monthly_forecast_as_mature_forecast_context": bool(used_lt_prefix),
        "method": method,
        "horizons": horizons,
        "monthly_projection": [
            {"month": d.strftime("%Y-%m-%d"), "wash_count": float(max(v, 0))}
            for d, v in fc.items()
        ],
    }
    return out


def _bridge_mature_monthly_to_opening_last_month(
    resp: dict[str, Any],
    *,
    skip_bridge_when_prefix: bool = False,
) -> None:
    lt = resp.get("less_than_2yrs")
    gt = resp.get("more_than_2yrs")
    if (
        skip_bridge_when_prefix
        and gt
        and gt.get("used_lt2y_monthly_forecast_as_mature_forecast_context")
    ):
        return
    if not lt or not gt or "error" in lt or "error" in gt:
        return
    lp = lt.get("monthly_projection") or []
    gp = gt.get("monthly_projection") or []
    if len(lp) < 24 or len(gp) < 24:
        return
    lt_last = float(lp[-1].get("wash_count", 0) or 0)
    gt_first = float(gp[0].get("wash_count", 0) or 0)
    if not np.isfinite(gt_first) or gt_first <= 1e-12 or not np.isfinite(lt_last):
        return
    factor = float(lt_last / gt_first)
    for row in gp:
        row["wash_count"] = float(max(float(row.get("wash_count", 0) or 0) * factor, 0.0))
    for pred_key in ("ridge_prediction", "rf_prediction"):
        rp = gt.get(pred_key)
        if not isinstance(rp, dict):
            continue
        if rp.get("monthly_level") is not None and np.isfinite(float(rp["monthly_level"])):
            rp["monthly_level"] = float(rp["monthly_level"]) * factor
        if rp.get("raw_level") is not None and np.isfinite(float(rp["raw_level"])):
            rp["raw_level"] = float(rp["raw_level"]) * factor
        gt[pred_key] = rp
    gt["opening_to_mature_bridge"] = {
        "method": "scale_entire_mature_monthly_track",
        "scale_factor": float(factor),
        "aligned_month_24_lt_to_month_25_gt": True,
    }


def _enrich_brand_new_site_timeline(resp: dict[str, Any]) -> None:
    lt = resp.get("less_than_2yrs")
    gt = resp.get("more_than_2yrs")
    if lt and "error" not in lt and lt.get("monthly_projection"):
        for i, row in enumerate(lt["monthly_projection"], start=1):
            row["operational_month_index"] = i
        lt["operational_phase"] = "opening_phase_years_1_2"
        lt["operational_calendar_months"] = "Months 1–24 after site open"

    if not gt or "error" in gt or "monthly_projection" not in gt:
        return
    if not lt or "error" in lt or "horizons" not in lt:
        return
    try:
        lt_opening_total = sum(
            float(lt["horizons"][f"{m}m"]["six_month_period_sum"]) for m in (6, 12, 18, 24)
        )
    except (KeyError, TypeError, ValueError):
        return

    rows = gt["monthly_projection"]
    if len(rows) < 24:
        return
    gt["monthly_projection_mature_25_48"] = [dict(r) for r in rows]
    w = [float(r.get("wash_count", 0) or 0) for r in rows]

    new_hz: dict[str, Any] = {}
    for end_m, lo, hi in ((30, 0, 6), (36, 6, 12), (42, 12, 18), (48, 18, 24)):
        seg = np.maximum(np.array(w[lo:hi], dtype=float), 0.0)
        inc = float(seg.sum())
        new_hz[f"{end_m}m"] = {
            "six_month_period_sum": inc,
            "monthly_avg_in_period": float(seg.mean()) if len(seg) else 0.0,
            "operational_month_end": end_m,
            "period_operational_month_start": 25 + lo,
            "period_operational_month_end": 25 + hi - 1,
            "first_month": rows[lo].get("month"),
            "last_month": rows[hi - 1].get("month"),
        }
    gt["horizons"] = new_hz
    gt["horizon_definition"] = (
        "Each bar is washes in a single 6-month window only (no running cumulative). "
        ">2y panel: months 25–30, 31–36, 37–42, 43–48 (labeled by window end 30/36/42/48)."
    )
    gt["operational_phase"] = "mature_phase_months_30_48"
    gt["operational_calendar_months"] = "Months 30–48 after site open (published >2y view)"

    mature_pub: list[dict[str, Any]] = []
    for i, r in enumerate(rows[5:], start=30):
        nr = dict(r)
        nr["operational_month_index"] = i
        mature_pub.append(nr)
    gt["monthly_projection"] = mature_pub

    hz = gt["horizons"]
    run = 0.0
    running: dict[str, float] = {}
    for m in (6, 12, 18, 24):
        run += float(lt["horizons"][f"{m}m"]["six_month_period_sum"])
        running[str(m)] = run
    for end_m in (30, 36, 42, 48):
        run += float(hz[f"{end_m}m"]["six_month_period_sum"])
        running[str(end_m)] = run
    resp["brand_new_site_continuation"] = {
        "narrative": (
            "Bars are independent 6-month wash totals per period. Optional running_total_through_operational_month "
            "sums those periods in calendar order through month 48."
        ),
        "six_month_period_sum_by_label": {
            "opening_6m": float(lt["horizons"]["6m"]["six_month_period_sum"]),
            "opening_12m": float(lt["horizons"]["12m"]["six_month_period_sum"]),
            "opening_18m": float(lt["horizons"]["18m"]["six_month_period_sum"]),
            "opening_24m": float(lt["horizons"]["24m"]["six_month_period_sum"]),
            "mature_30m": float(hz["30m"]["six_month_period_sum"]),
            "mature_36m": float(hz["36m"]["six_month_period_sum"]),
            "mature_42m": float(hz["42m"]["six_month_period_sum"]),
            "mature_48m": float(hz["48m"]["six_month_period_sum"]),
        },
        "running_total_through_operational_month": running,
        "opening_total_first_24_months": lt_opening_total,
    }


def _append_calendar_year_washes_ridge(resp: dict[str, Any]) -> None:
    lt = resp.get("less_than_2yrs") or {}
    gt = resp.get("more_than_2yrs") or {}
    if "error" in lt or "error" in gt:
        return
    lm = lt.get("monthly_projection") or []
    mature = gt.get("monthly_projection_mature_25_48") or []
    if len(lm) < 24 or len(mature) < 24:
        return
    y1 = float(sum(float(lm[i]["wash_count"]) for i in range(12)))
    y2 = float(sum(float(lm[i]["wash_count"]) for i in range(12, 24)))
    # y3 / y4 = mature operational years (months 25–36 vs 37–48). With <2y monthly prefix, the >2y forecast is
    # ARIMA(1,1,1) on cluster tail + those 24 months; the extrapolation often levels off, so y3 ≈ y4 is common
    # (near-constant monthly washes), not a calendar indexing bug.
    y3 = float(sum(float(mature[i]["wash_count"]) for i in range(12)))
    y4 = float(sum(float(mature[i]["wash_count"]) for i in range(12, 24)))
    resp["calendar_year_washes"] = {
        "year_1": y1,
        "year_2": y2,
        "year_3": y3,
        "year_4": y4,
        "definition": (
            "Each value is the sum of forecast monthly washes in that operational year after open "
            "(year 1 = months 1–12, year 2 = 13–24, year 3 = 25–36, year 4 = 37–48)."
        ),
    }


def _nearest_cluster_caption(resp: dict[str, Any]) -> str:
    """One line: <2y and >2y nearest train centroid id, distance km, peer count."""
    parts: list[str] = []
    for label, key in (("<2y", "less_than_2yrs"), (">2y", "more_than_2yrs")):
        block = resp.get(key) or {}
        if "error" in block:
            parts.append(f"{label}: {block.get('error', 'error')[:40]}")
            continue
        cl = block.get("cluster") or {}
        cid = cl.get("cluster_id", "?")
        dk = cl.get("distance_km")
        sz = cl.get("size", "?")
        if dk is not None and np.isfinite(float(dk)):
            parts.append(f"{label} nearest cl {cid}, {float(dk):.2f} km (peers={sz})")
        else:
            parts.append(f"{label} nearest cl {cid} (peers={sz})")
    return "  |  ".join(parts)


def _wash_totals_2024_2025_for_site(
    panel: pd.DataFrame,
    site_client_id: float,
    *,
    day_col: str = "calendar_day",
    val_col: str = "wash_count_total",
) -> tuple[float, float]:
    s = panel.loc[panel["site_client_id"].astype(float) == float(site_client_id)].copy()
    if s.empty:
        return 0.0, 0.0
    s["_cy"] = pd.to_datetime(s[day_col], errors="coerce").dt.year
    s = s.loc[s["_cy"].isin([2024, 2025])]
    if s.empty:
        return 0.0, 0.0
    sums = s.groupby("_cy", sort=False)[val_col].sum()
    w24 = float(max(sums.loc[2024], 0.0)) if 2024 in sums.index and pd.notna(sums.loc[2024]) else 0.0
    w25 = float(max(sums.loc[2025], 0.0)) if 2025 in sums.index and pd.notna(sums.loc[2025]) else 0.0
    return w24, w25


def _peer_hist_avg_2024_2025_reference(resp: dict[str, Any]) -> dict[str, float] | None:
    """
    Distribution of per-site ``(wash_2024 + wash_2025) / 2`` for training peers near the projection:
    <2y: same ``dbscan_cluster_12km`` and ≤20 km from projected lat/lon; >2y: all panel sites in that radius
    (daily CSV has no cluster id). Summarized on the Ridge vs RF four-panel chart suptitle.
    """
    inp = resp.get("input") or {}
    try:
        p_lat = float(inp["lat"])
        p_lon = float(inp["lon"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (np.isfinite(p_lat) and np.isfinite(p_lon)):
        return None
    lt_b = resp.get("less_than_2yrs") or {}
    gt_b = resp.get("more_than_2yrs") or {}
    if "error" in lt_b or "error" in gt_b:
        return None
    try:
        cl_lt = lt_b["cluster"]
        cid_lt = int(cl_lt["cluster_id"])
        cl_gt = gt_b["cluster"]
        _ = int(cl_gt["cluster_id"])
        _ = float(cl_gt["centroid_lat"])
        _ = float(cl_gt["centroid_lon"])
    except (KeyError, TypeError, ValueError):
        return None
    if not _cohort_csv.LESS_THAN_CLUSTERING_READY_CSV.is_file() or not _cohort_csv.MASTER_MORE_THAN_2YRS_CSV.is_file():
        return None
    try:
        dflt = pd.read_csv(
            _cohort_csv.LESS_THAN_CLUSTERING_READY_CSV,
            usecols=[
                "site_client_id",
                "latitude",
                "longitude",
                "calendar_day",
                "wash_count_total",
                "dbscan_cluster_12km",
            ],
            low_memory=False,
        )
        dfgt = pd.read_csv(
            _cohort_csv.MASTER_MORE_THAN_2YRS_CSV,
            usecols=["site_client_id", "latitude", "longitude", "calendar_day", "wash_count_total"],
            low_memory=False,
        )
    except (OSError, ValueError, KeyError):
        return None

    avgs: list[float] = []
    seen: set[float] = set()

    cc = pd.to_numeric(dflt["dbscan_cluster_12km"], errors="coerce").fillna(-999).astype(int)
    sub = dflt.loc[cc == cid_lt].copy()
    if not sub.empty:
        g = (
            sub.groupby("site_client_id", sort=False)
            .agg(lat=("latitude", "mean"), lon=("longitude", "mean"))
            .replace([np.inf, -np.inf], np.nan)
            .dropna(subset=["lat", "lon"])
        )
        rows = [
            (float(sid), _haversine_km(p_lat, p_lon, float(row["lat"]), float(row["lon"])))
            for sid, row in g.iterrows()
        ]
        rows.sort(key=lambda t: t[1])
        for sid, d_proj in rows:
            if d_proj > PEER_REF_MAX_KM_FROM_PROJECTION:
                continue
            if sid in seen:
                continue
            seen.add(sid)
            w24, w25 = _wash_totals_2024_2025_for_site(dflt, sid)
            avgs.append((w24 + w25) / 2.0)

    g2 = (
        dfgt.groupby("site_client_id", sort=False)
        .agg(lat=("latitude", "mean"), lon=("longitude", "mean"))
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["lat", "lon"])
    )
    rows2 = [
        (float(sid), _haversine_km(p_lat, p_lon, float(row["lat"]), float(row["lon"])))
        for sid, row in g2.iterrows()
    ]
    rows2.sort(key=lambda t: t[1])
    for sid, d_proj in rows2:
        if d_proj > PEER_REF_MAX_KM_FROM_PROJECTION:
            continue
        if sid in seen:
            continue
        seen.add(sid)
        w24, w25 = _wash_totals_2024_2025_for_site(dfgt, sid)
        avgs.append((w24 + w25) / 2.0)

    if not avgs:
        return None
    arr = np.asarray(avgs, dtype=float)
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "n": int(arr.size),
    }


def _plot_projection(resp: dict[str, Any], out_path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    cy = resp.get("calendar_year_washes") or {}
    if all(k in cy for k in ("year_1", "year_2", "year_3", "year_4")):
        labels = ["Year 1", "Year 2", "Year 3", "Year 4"]
        vals = [float(cy["year_1"]), float(cy["year_2"]), float(cy["year_3"]), float(cy["year_4"])]
        bars = ax.bar(labels, vals, color="#3b82f6")
        ax.set_ylabel("Washes in calendar year (12-mo sum)")
        lt = resp.get("less_than_2yrs") or {}
        gt = resp.get("more_than_2yrs") or {}
        pfx = ""
        if gt.get("used_lt2y_monthly_forecast_as_mature_forecast_context"):
            pfx = " | >2y forecast used <2y monthly as TS context"
        ax.set_title(
            f"V2 Ridge | calendar years 1–4 (opening cl {lt.get('cluster', {}).get('cluster_id', '?')}, "
            f"mature cl {gt.get('cluster', {}).get('cluster_id', '?')}){pfx}",
            fontsize=10,
        )
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{int(v):,}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    else:
        ax.text(0.5, 0.5, "Insufficient data for calendar-year plot", ha="center", transform=ax.transAxes)
    inp = resp["input"]
    loc = inp.get("address") or f"{inp['lat']:.4f},{inp['lon']:.4f}"
    fig.suptitle(
        f"V2 projection for {loc}   (method={resp['method']})\nNearest train centroids: {_nearest_cluster_caption(resp)}",
        fontsize=10,
        y=1.03,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_ridge_rf_four_panels(
    resps: tuple[tuple[str, dict[str, Any]], ...],
    out_path: Path | None,
    loc_title: str,
    method: str,
    *,
    return_figure: bool = False,
) -> Figure | None:
    """2×2: Ridge no/prefix × RF no/prefix (calendar-year bars).

    Default (return_figure=False): write ``out_path`` and close the figure (CLI).
    With ``return_figure=True``: return the Matplotlib figure; ``out_path`` is ignored; caller must ``plt.close``.
    """
    labels = ["Year 1", "Year 2", "Year 3", "Year 4"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9), sharey=True)
    cluster_line = _nearest_cluster_caption(resps[0][1])
    peer_ref = _peer_hist_avg_2024_2025_reference(resps[0][1])
    colors = ("#3b82f6", "#3b82f6", "#16a34a", "#16a34a")
    for ax, (subtitle, resp), color in zip(np.ravel(axes), resps, colors):
        ax.set_axisbelow(True)
        cy = resp.get("calendar_year_washes") or {}
        if all(k in cy for k in ("year_1", "year_2", "year_3", "year_4")):
            vals = [float(cy["year_1"]), float(cy["year_2"]), float(cy["year_3"]), float(cy["year_4"])]
            ax.bar(labels, vals, color=color, alpha=0.88, zorder=4)
            ax.set_title(subtitle, fontsize=9)
            ax.grid(axis="y", alpha=0.3)
            for i, v in enumerate(vals):
                ax.text(i, v, f"{int(v):,}", ha="center", va="bottom", fontsize=7)
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", transform=ax.transAxes)
            ax.set_title(subtitle, fontsize=9)
        ax.tick_params(axis="x", rotation=12)
    axes[0, 0].set_ylabel("Washes (12-mo sum)")
    axes[1, 0].set_ylabel("Washes (12-mo sum)")
    peer_note = ""
    if peer_ref is not None:
        pr = peer_ref
        peer_note = (
            f"\nPeer sites ≤{PEER_REF_MAX_KM_FROM_PROJECTION:.0f} km (training CSVs): n={pr['n']}, "
            f"per-site avg(2024,2025) washes - min={pr['min']:,.0f}, med={pr['median']:,.0f}, "
            f"mean={pr['mean']:,.0f}, max={pr['max']:,.0f}"
        )
    fig.suptitle(
        f"V2 Ridge vs RF level (method={method})  |  {loc_title}\n"
        f"Top: Ridge — bottom: RF (same TS path; anchor scale from level model)\n{cluster_line}{peer_note}",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90 if peer_ref is not None else 0.92))
    if return_figure:
        return fig
    if out_path is None:
        raise ValueError("out_path is required when return_figure is False")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return None


def _plot_compare_panels(
    resps: tuple[tuple[str, dict[str, Any]], ...],
    out_path: Path | None,
    loc_title: str,
    method: str,
    *,
    return_figure: bool = False,
) -> Figure | None:
    labels = ["Year 1", "Year 2", "Year 3", "Year 4"]
    n = len(resps)
    fig_w = 5.5 * n + 1.0
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 5.5), sharey=True)
    if n == 1:
        axes = [axes]
    cluster_line = _nearest_cluster_caption(resps[0][1])
    for ax, (subtitle, resp) in zip(axes, resps):
        cy = resp.get("calendar_year_washes") or {}
        if all(k in cy for k in ("year_1", "year_2", "year_3", "year_4")):
            vals = [float(cy["year_1"]), float(cy["year_2"]), float(cy["year_3"]), float(cy["year_4"])]
            ax.bar(labels, vals, color="#3b82f6")
            ax.set_title(subtitle, fontsize=10)
            ax.grid(axis="y", alpha=0.3)
            for i, v in enumerate(vals):
                ax.text(i, v, f"{int(v):,}", ha="center", va="bottom", fontsize=8)
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", transform=ax.transAxes)
            ax.set_title(subtitle, fontsize=10)
        ax.tick_params(axis="x", rotation=15)
    axes[0].set_ylabel("Washes in calendar year (12-mo sum)")
    fig.suptitle(
        f"V2 Ridge compare (method={method})  |  {loc_title}\nNearest train centroids: {cluster_line}",
        fontsize=10,
        y=1.03,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    if return_figure:
        return fig
    if out_path is None:
        raise ValueError("out_path is required when return_figure is False")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return None


def run_projection(
    address: str | None,
    lat: float | None,
    lon: float | None,
    method: str,
    *,
    use_opening_prefix_for_mature_forecast: bool = True,
    bridge_opening_to_mature_when_prefix: bool = True,
    allow_nearest_cluster_beyond_distance_cap: bool = False,
    level_model: str = "ridge",
) -> dict[str, Any]:
    lat, lon, addr = _resolve_latlon(address, lat, lon)
    print(f"[project] address={addr!r}  lat={lat:.5f}  lon={lon:.5f}  method={method}  level={level_model}")

    more_assets = _load_cohort("more_than")
    less_assets = _load_cohort("less_than")

    less_block = _project_cohort(
        cohort="less_than",
        cohort_label="less_than_2yrs",
        lat=lat, lon=lon, method=method,
        cohort_assets=less_assets,
        days_per_month=30,
        is_daily_model=False,
        opening_prefix_monthly=None,
        allow_nearest_cluster_beyond_distance_cap=allow_nearest_cluster_beyond_distance_cap,
        level_model=level_model,
    )
    prefix: list[float] | None = None
    if use_opening_prefix_for_mature_forecast and "error" not in less_block:
        mp = less_block.get("monthly_projection") or []
        if len(mp) >= 24:
            prefix = [float(r["wash_count"]) for r in mp[:24]]

    more_block = _project_cohort(
        cohort="more_than",
        cohort_label="more_than_2yrs",
        lat=lat, lon=lon, method=method,
        cohort_assets=more_assets,
        days_per_month=30,
        is_daily_model=True,
        opening_prefix_monthly=prefix,
        allow_nearest_cluster_beyond_distance_cap=allow_nearest_cluster_beyond_distance_cap,
        level_model=level_model,
    )

    resp = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "radius_km": 12.0,
        "method": method,
        "level_model": level_model,
        "input": {"address": addr, "lat": lat, "lon": lon},
        "nearest_cluster_max_distance_km": MAX_NEAREST_CLUSTER_DISTANCE_KM,
        "allow_nearest_cluster_beyond_distance_cap": bool(allow_nearest_cluster_beyond_distance_cap),
        "less_than_2yrs": less_block,
        "more_than_2yrs": more_block,
    }
    _bridge_mature_monthly_to_opening_last_month(
        resp,
        skip_bridge_when_prefix=bool(
            use_opening_prefix_for_mature_forecast and not bridge_opening_to_mature_when_prefix
        ),
    )
    _enrich_brand_new_site_timeline(resp)
    _append_calendar_year_washes_ridge(resp)
    resp["use_opening_prefix_for_mature_forecast"] = bool(use_opening_prefix_for_mature_forecast)
    resp["bridge_opening_to_mature_when_prefix"] = bool(bridge_opening_to_mature_when_prefix)
    return resp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--address", type=str, default=None)
    ap.add_argument("--lat", type=float, default=None)
    ap.add_argument("--lon", type=float, default=None)
    ap.add_argument(
        "--method",
        type=str,
        default="arima",
        choices=["holt_winters", "arima", "blend"],
        help="TS extrapolation on cluster monthly track (default arima: see results/ts_arima_vs_holt_pick.json).",
    )
    ap.add_argument("--out-name", type=str, default=None,
                    help="tag appended to output filenames")
    ap.add_argument(
        "--no-opening-prefix",
        action="store_true",
        help="Do not append <2y monthly wash forecast as context before >2y TS extrapolation.",
    )
    ap.add_argument(
        "--legacy-prefix-no-bridge",
        action="store_true",
        help="With prefix: skip month-24→25 scale (old behavior). Default bridges when prefix is on.",
    )
    ap.add_argument(
        "--plot-two-way",
        action="store_true",
        help="Two panels: no <2y TS prefix vs prefix + month-24→25 bridge (default). Writes *_two_way.png/json.",
    )
    ap.add_argument(
        "--plot-ridge-rf-four-way",
        action="store_true",
        help=(
            "2×2 figure: Ridge no/prefix vs RF no/prefix (requires models/*/wash_count_model_12km.rf.joblib from build_v2). "
            "Writes projection_{method}_{tag}_ridge_rf_four.png/json."
        ),
    )
    ap.add_argument(
        "--plot-three-way",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--allow-distant-nearest-cluster",
        action="store_true",
        help=(
            "Always assign the geographically nearest train centroid, even if >20 km away. "
            "Default: refuse projection when nearest centroid exceeds 20 km."
        ),
    )
    args = ap.parse_args()

    DEMO_OUT.mkdir(parents=True, exist_ok=True)
    tag = args.out_name or datetime.utcnow().strftime("%Y%m%d-%H%M%S")

    plot_compare = args.plot_two_way or args.plot_three_way
    if args.plot_three_way and not args.plot_two_way:
        print(
            "[project] --plot-three-way is deprecated (same as --plot-two-way: 2 panels only).",
            file=sys.stderr,
        )
    if args.plot_ridge_rf_four_way:
        if not _rf_models_available():
            raise SystemExit(
                "RF models missing. Run from clustering_v2: python build_v2.py\n"
                "Expected: models/more_than/wash_count_model_12km.rf.joblib and models/less_than/…"
            )
        lat, lon, addr = _resolve_latlon(args.address, args.lat, args.lon)
        loc_title = addr or f"{lat:.4f},{lon:.4f}"
        print(f"[project 4-way Ridge vs RF] {loc_title!r}  method={args.method}")
        cap_kw = {"allow_nearest_cluster_beyond_distance_cap": args.allow_distant_nearest_cluster}
        rr0 = run_projection(
            None, lat, lon, args.method,
            use_opening_prefix_for_mature_forecast=False,
            bridge_opening_to_mature_when_prefix=True,
            level_model="ridge",
            **cap_kw,
        )
        rr1 = run_projection(
            None, lat, lon, args.method,
            use_opening_prefix_for_mature_forecast=True,
            bridge_opening_to_mature_when_prefix=True,
            level_model="ridge",
            **cap_kw,
        )
        rf0 = run_projection(
            None, lat, lon, args.method,
            use_opening_prefix_for_mature_forecast=False,
            bridge_opening_to_mature_when_prefix=True,
            level_model="rf",
            **cap_kw,
        )
        rf1 = run_projection(
            None, lat, lon, args.method,
            use_opening_prefix_for_mature_forecast=True,
            bridge_opening_to_mature_when_prefix=True,
            level_model="rf",
            **cap_kw,
        )
        combined = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "method": args.method,
            "input": {"address": addr, "lat": lat, "lon": lon},
            "compare": "ridge_vs_rf × (no_prefix | prefix+bridge)",
            "ridge": {"no_prefix": rr0, "with_opening_prefix_and_bridge": rr1},
            "random_forest": {"no_prefix": rf0, "with_opening_prefix_and_bridge": rf1},
        }
        json_path = DEMO_OUT / f"projection_{args.method}_{tag}_ridge_rf_four.json"
        png_path = DEMO_OUT / f"projection_{args.method}_{tag}_ridge_rf_four.png"
        json_path.write_text(json.dumps(combined, indent=2, default=str))
        _plot_ridge_rf_four_panels(
            (
                ("Ridge | No <2y TS prefix", rr0),
                ("Ridge | <2y prefix + 24→25 bridge", rr1),
                ("RF | No <2y TS prefix", rf0),
                ("RF | <2y prefix + 24→25 bridge", rf1),
            ),
            png_path,
            loc_title,
            args.method,
        )
        print(f"\nwrote {json_path.relative_to(REPO_ROOT)}")
        print(f"wrote {png_path.relative_to(REPO_ROOT)}")
        return

    if plot_compare:
        lat, lon, addr = _resolve_latlon(args.address, args.lat, args.lon)
        loc_title = addr or f"{lat:.4f},{lon:.4f}"
        print(f"[project 2-way] {loc_title!r}  lat={lat:.5f}  lon={lon:.5f}  method={args.method}")
        cap_kw = {"allow_nearest_cluster_beyond_distance_cap": args.allow_distant_nearest_cluster}
        r_no = run_projection(
            None,
            lat,
            lon,
            args.method,
            use_opening_prefix_for_mature_forecast=False,
            bridge_opening_to_mature_when_prefix=True,
            level_model="ridge",
            **cap_kw,
        )
        r_prefix = run_projection(
            None,
            lat,
            lon,
            args.method,
            use_opening_prefix_for_mature_forecast=True,
            bridge_opening_to_mature_when_prefix=True,
            level_model="ridge",
            **cap_kw,
        )
        combined = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "method": args.method,
            "input": {"address": addr, "lat": lat, "lon": lon},
            "compare": "no_prefix vs opening_prefix_with_month24_to_25_bridge (default fixed behavior)",
            "no_prefix": r_no,
            "with_opening_prefix_and_bridge": r_prefix,
        }
        json_path = DEMO_OUT / f"projection_{args.method}_{tag}_two_way.json"
        png_path = DEMO_OUT / f"projection_{args.method}_{tag}_two_way.png"
        json_path.write_text(json.dumps(combined, indent=2, default=str))
        _plot_compare_panels(
            (
                ("No <2y TS prefix", r_no),
                ("<2y prefix + 24→25 bridge (default)", r_prefix),
            ),
            png_path,
            loc_title,
            args.method,
        )
        print(f"\nwrote {json_path.relative_to(REPO_ROOT)}")
        print(f"wrote {png_path.relative_to(REPO_ROOT)}")
        return

    resp = run_projection(
        args.address,
        args.lat,
        args.lon,
        args.method,
        use_opening_prefix_for_mature_forecast=not args.no_opening_prefix,
        bridge_opening_to_mature_when_prefix=not args.legacy_prefix_no_bridge,
        allow_nearest_cluster_beyond_distance_cap=args.allow_distant_nearest_cluster,
    )

    json_path = DEMO_OUT / f"projection_{args.method}_{tag}.json"
    png_path = DEMO_OUT / f"projection_{args.method}_{tag}.png"
    json_path.write_text(json.dumps(resp, indent=2, default=str))
    _plot_projection(resp, png_path)
    print(f"\nwrote {json_path.relative_to(REPO_ROOT)}")
    print(f"wrote {png_path.relative_to(REPO_ROOT)}")

    for cohort in ("less_than_2yrs", "more_than_2yrs"):
        block = resp[cohort]
        if "error" in block:
            print(f"\n[{cohort}]  {block.get('error', 'error')}")
            continue
        h = block["horizons"]
        cl = block["cluster"]
        print(f"\n[{cohort}]  cluster={cl['cluster_id']}  size={cl['size']}  dist={cl['distance_km']:.2f}km")
        keys = ("30m", "36m", "42m", "48m") if cohort == "more_than_2yrs" and "30m" in h else ("6m", "12m", "18m", "24m")
        for k in keys:
            row = h[k]
            ma = row.get("monthly_avg_in_period", row.get("monthly_avg"))
            print(f"  {k:>3}: 6mo_sum={row['six_month_period_sum']:,.0f}  avg_in_period={ma:,.0f}")


if __name__ == "__main__":
    main()
