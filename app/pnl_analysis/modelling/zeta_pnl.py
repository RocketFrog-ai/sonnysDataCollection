"""
PnL wash-volume projection using zeta_modelling (model_1 + data_1).

Replaces clustering_v2 for central input-form and standalone projection tasks.
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.celery.celery_app import celery_app
from app.utils import common as calib

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ZETA_DATA = _REPO_ROOT / "zeta_modelling" / "data_1"
_ZETA_MODEL = _REPO_ROOT / "zeta_modelling" / "model_1"
_ARTIFACTS_PATH = _ZETA_MODEL / "phase3_artifacts.joblib"
_ADVANCED_REPORT = _ZETA_DATA / "phase3_advanced_report.json"
_YEAR_BANDS = (
    ("year_1", 1, 12),
    ("year_2", 13, 24),
    ("year_3", 25, 36),
    ("year_4", 37, 48),
    ("year_5", 49, 60),
)


def _regional_cluster_distance_cap_km(lat: float, lon: float) -> float:
    # Requested regional caps:
    # Northeast: 100km, Midwest: 150km, South: 160km, West: 180km.
    if lon >= -80 and lat >= 38:
        return 100.0
    if -104 <= lon < -80 and lat >= 36:
        return 150.0
    if -105 <= lon < -75 and lat < 36:
        return 160.0
    if lon < -104:
        return 180.0
    return 150.0


def _ensure_repo_on_path() -> None:
    r = str(_REPO_ROOT)
    if r not in sys.path:
        sys.path.insert(0, r)


@lru_cache(maxsize=1)
def _load_artifacts():
    _ensure_repo_on_path()
    from zeta_modelling.model_1.phase3_advanced_forecast import load_artifacts

    if not _ARTIFACTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {_ARTIFACTS_PATH}. Run: python zeta_modelling/model_1/build_phase3_artifacts.py"
        )
    return load_artifacts(_ARTIFACTS_PATH)


def _read_calibration_coverage() -> float:
    if _ADVANCED_REPORT.exists():
        try:
            payload = json.loads(_ADVANCED_REPORT.read_text())
            return float(payload.get("backtest", {}).get("p10_p90_coverage", 0.454))
        except Exception:
            pass
    return 0.454


def _json_sanitize(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return _json_sanitize(item())
        except Exception:
            pass
    return str(obj)


def _effective_dollar_per_wash(wash_prices: List[float], wash_pcts: List[float]) -> float:
    return float(sum(float(p) * max(float(c), 0.0) / 100.0 for p, c in zip(wash_prices, wash_pcts)))


def _normalize_central_task_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return payload
    out = dict(payload)
    ci = payload.get("customer_information")
    if isinstance(ci, dict):
        site = (ci.get("site_address") or "").strip()
        if site:
            out["address"] = site
        lat, lon = ci.get("latitude"), ci.get("longitude")
        if lat is not None:
            try:
                out["latitude"] = out["lat"] = float(lat)
            except (TypeError, ValueError):
                pass
        if lon is not None:
            try:
                out["longitude"] = out["lon"] = float(lon)
            except (TypeError, ValueError):
                pass
    mp = payload.get("menu_packages")
    if isinstance(mp, list) and mp:
        if not out.get("wash_prices") or not out.get("wash_pcts"):
            prices: List[float] = []
            pcts: List[float] = []
            for p in mp:
                if not isinstance(p, dict):
                    continue
                pr, pc = p.get("price"), p.get("customer_percentage")
                if pr is None or pc is None:
                    continue
                try:
                    prices.append(float(pr))
                    pcts.append(float(pc))
                except (TypeError, ValueError):
                    continue
            if prices and len(prices) == len(pcts):
                out["wash_prices"] = prices
                out["wash_pcts"] = pcts
    return out


def _zeta_params(payload: Dict[str, Any]) -> Dict[str, Any]:
    z = payload.get("zeta_forecast")
    if not isinstance(z, dict):
        fi = payload.get("financial_inputs")
        if isinstance(fi, dict) and isinstance(fi.get("zeta_forecast"), dict):
            z = fi.get("zeta_forecast")
    if z is None:
        z = {}
    if not isinstance(z, dict):
        z = {}
    out = {
        "margin_per_wash": float(z.get("margin_per_wash", 4.0)),
        "fixed_monthly_cost": float(z.get("fixed_monthly_cost", 50_000.0)),
        "ramp_up_cost": float(z.get("ramp_up_cost", 150_000.0)),
        "scenario": str(z.get("scenario", "Expected")),
        "forecast_months": int(z.get("forecast_months", 48)),
        "target_calibration_coverage": float(z.get("target_calibration_coverage", 0.80)),
        "forecast_start_date": str(z.get("forecast_start_date", "2026-01-01")),
        "enable_mature_yoy_control": bool(z.get("enable_mature_yoy_control", True)),
        "mature_yoy_start_year": int(min(10, max(2, int(z.get("mature_yoy_start_year", 3))))),
        "mature_min_yoy": float(z.get("mature_min_yoy", 0.005)),
        "mature_max_yoy": float(z.get("mature_max_yoy", 0.05)),
        "lifecycle_min_growth": float(z.get("lifecycle_min_growth", 0.005)),
        "lifecycle_max_growth": float(z.get("lifecycle_max_growth", 0.05)),
        "cluster_distance_policy": str(z.get("cluster_distance_policy", "regional")).strip().lower(),
        "max_cluster_distance_km": (
            float(z["max_cluster_distance_km"])
            if z.get("max_cluster_distance_km") is not None
            else None
        ),
    }
    if out["mature_min_yoy"] > out["mature_max_yoy"]:
        out["mature_min_yoy"], out["mature_max_yoy"] = out["mature_max_yoy"], out["mature_min_yoy"]
    return out


def _resolve_site_lat_lon(payload: Dict[str, Any]) -> tuple[float, float, Optional[str]]:
    """
    Resolve site coordinates from payload.
    Priority:
    1) explicit latitude/longitude
    2) geocode address
    """
    address = str(payload.get("address") or "").strip() or None
    lat = payload.get("latitude", payload.get("lat"))
    lon = payload.get("longitude", payload.get("lon"))

    if lat is not None and lon is not None:
        return float(lat), float(lon), address

    if address:
        try:
            geocoded_lat, geocoded_lon = calib.resolve_lat_lon(address)
            return float(geocoded_lat), float(geocoded_lon), address
        except ValueError as e:
            raise ValueError(f"Could not geocode address: {e}")

    raise ValueError("Provide address or both latitude and longitude for zeta projection")


def _scenario_adjust(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    out = df.copy()
    if scenario == "Conservative":
        out["volume"] = out["low"]
    elif scenario == "Aggressive":
        out["volume"] = out["high"]
    return out


def _run_zeta_forecast_df(
    lat: float,
    lon: float,
    *,
    months: int,
    margin_per_wash: float,
    fixed_monthly_cost: float,
    ramp_up_cost: float,
    scenario: str,
    target_coverage: float,
    start_date: str,
    enable_mature_yoy_control: bool = True,
    mature_yoy_start_year: int = 3,
    mature_min_yoy: float = 0.005,
    mature_max_yoy: float = 0.05,
    max_cluster_distance_km: float = 100.0,
    lifecycle_min_growth: float = 0.005,
    lifecycle_max_growth: float = 0.05,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    _ensure_repo_on_path()
    from zeta_modelling.model_1.phase3_advanced_forecast import (
        add_confidence_label,
        apply_global_uncertainty_calibration,
        final_report,
    )

    artifacts = _load_artifacts()
    forecast, summary = final_report(
        lat=float(lat),
        lon=float(lon),
        artifacts=artifacts,
        months=int(months),
        start_date=start_date,
        margin_per_wash=margin_per_wash,
        fixed_monthly_cost=fixed_monthly_cost,
        ramp_up_cost=ramp_up_cost,
        enable_mature_yoy_control=enable_mature_yoy_control,
        mature_yoy_start_year=mature_yoy_start_year,
        mature_min_yoy=mature_min_yoy,
        mature_max_yoy=mature_max_yoy,
        max_cluster_distance_km=max_cluster_distance_km,
        lifecycle_min_growth=lifecycle_min_growth,
        lifecycle_max_growth=lifecycle_max_growth,
    )
    cov = _read_calibration_coverage()
    forecast, _scale = apply_global_uncertainty_calibration(
        forecast=forecast,
        current_coverage=cov,
        target_coverage=target_coverage,
    )
    forecast, _conf = add_confidence_label(forecast)
    forecast = _scenario_adjust(forecast, scenario)
    forecast["cumulative_volume"] = forecast["volume"].cumsum()
    return forecast, summary


def _year_sum_volume(df: pd.DataFrame, start_m: int, end_m: int, col: str) -> float:
    mask = (df["age_in_months"] >= start_m) & (df["age_in_months"] <= end_m)
    return float(df.loc[mask, col].sum())


def _wash_volume_projection_from_forecast(forecast: pd.DataFrame) -> Dict[str, float]:
    return {y: _year_sum_volume(forecast, s, e, "volume") for y, s, e in _YEAR_BANDS}


def _wash_volume_range_minmax_from_forecast(forecast: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Year bands from zeta_modelling quantile tracks (blend_clusters).

    min / median / max: annual sums of monthly p10 / p50 / p90.

    q1 / q3: annual sums of monthly **p25 / p75** when present (trained LGBM quantiles at
    0.25 and 0.75 in rebuilt artifacts). If the artifact predates that (no p25/p75 columns),
    q1/q3 fall back to the same linear-in-probability interpolation between p10–p50 and
    p50–p90 per month (then summed), so older deployments keep working.
    """
    c_lo = "p10" if "p10" in forecast.columns else "low"
    c_mid = "p50" if "p50" in forecast.columns else "volume"
    c_hi = "p90" if "p90" in forecast.columns else "high"

    if "p25" in forecast.columns and "p75" in forecast.columns:
        tmp = forecast
        q1_key, q3_key = "p25", "p75"
    else:
        lo = pd.to_numeric(forecast[c_lo], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        mid = pd.to_numeric(forecast[c_mid], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        hi = pd.to_numeric(forecast[c_hi], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        mid = np.maximum(mid, lo)
        hi = np.maximum(hi, mid)
        tmp = forecast.copy()
        tmp["_q1_month"] = lo + ((0.25 - 0.1) / (0.5 - 0.1)) * (mid - lo)
        tmp["_q3_month"] = mid + ((0.75 - 0.5) / (0.9 - 0.5)) * (hi - mid)
        q1_key, q3_key = "_q1_month", "_q3_month"

    return {
        y: {
            "min": _year_sum_volume(forecast, s, e, c_lo),
            "q1": _year_sum_volume(tmp, s, e, q1_key),
            "median": _year_sum_volume(forecast, s, e, c_mid),
            "q3": _year_sum_volume(tmp, s, e, q3_key),
            "max": _year_sum_volume(forecast, s, e, c_hi),
        }
        for y, s, e in _YEAR_BANDS
    }


def _monthly_projection(forecast: pd.DataFrame, months: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for _, r in forecast.head(int(months)).iterrows():
        try:
            rows.append(
                {
                    "month": str(r["date"])[:10],
                    "wash_count": float(r["volume"]),
                    "wash_count_low": float(r["low"]),
                    "wash_count_high": float(r["high"]),
                    "operational_month_index": int(r["age_in_months"]),
                }
            )
        except Exception:
            continue
    return rows


@lru_cache(maxsize=1)
def _cohort_membership_retail_shares() -> Dict[str, Dict[str, float]]:
    cfg = {
        "less_than": _ZETA_DATA / "less_than-2yrs.csv",
        "more_than": _ZETA_DATA / "more_than-2yrs.csv",
    }
    out: Dict[str, Dict[str, float]] = {}
    for cohort, path in cfg.items():
        if not path.exists():
            out[cohort] = {"retail_share": 0.5, "membership_share": 0.5}
            continue
        df = pd.read_csv(path, usecols=["wash_count_retail", "wash_count_membership"])
        retail = float(df["wash_count_retail"].fillna(0).sum())
        membership = float(df["wash_count_membership"].fillna(0).sum())
        denom = retail + membership
        if denom <= 0:
            out[cohort] = {"retail_share": 0.5, "membership_share": 0.5}
        else:
            out[cohort] = {
                "retail_share": retail / denom,
                "membership_share": membership / denom,
            }
    return out


@lru_cache(maxsize=1)
def _cluster_peer_membership_retail_shares() -> Dict[str, Dict[int, Dict[str, float]]]:
    out: Dict[str, Dict[int, Dict[str, float]]] = {"less_than": {}, "more_than": {}}

    lt_path = _ZETA_DATA / "less_than-2yrs.csv"
    if lt_path.exists():
        lt = pd.read_csv(
            lt_path,
            usecols=["dbscan_cluster_12km", "wash_count_retail", "wash_count_membership"],
        )
        lt = lt[lt["dbscan_cluster_12km"].notna()].copy()
        lt["dbscan_cluster_12km"] = pd.to_numeric(lt["dbscan_cluster_12km"], errors="coerce").astype("Int64")
        lt = lt[lt["dbscan_cluster_12km"].notna() & (lt["dbscan_cluster_12km"] >= 0)].copy()
        g = lt.groupby("dbscan_cluster_12km", as_index=True)[["wash_count_retail", "wash_count_membership"]].sum()
        for cluster_id, row in g.iterrows():
            retail = float(row["wash_count_retail"] or 0.0)
            membership = float(row["wash_count_membership"] or 0.0)
            denom = retail + membership
            if denom > 0:
                out["less_than"][int(cluster_id)] = {
                    "retail_share": retail / denom,
                    "membership_share": membership / denom,
                }

    gt_path = _ZETA_DATA / "more_than-2yrs.csv"
    if gt_path.exists():
        gt = pd.read_csv(
            gt_path,
            usecols=["dbscan_cluster_12km", "wash_count_retail", "wash_count_membership"],
        )
        gt = gt[gt["dbscan_cluster_12km"].notna()].copy()
        gt["dbscan_cluster_12km"] = pd.to_numeric(gt["dbscan_cluster_12km"], errors="coerce").astype("Int64")
        gt = gt[gt["dbscan_cluster_12km"].notna() & (gt["dbscan_cluster_12km"] >= 0)].copy()
        g2 = gt.groupby("dbscan_cluster_12km", as_index=True)[["wash_count_retail", "wash_count_membership"]].sum()
        for cluster_id, row in g2.iterrows():
            retail = float(row["wash_count_retail"] or 0.0)
            membership = float(row["wash_count_membership"] or 0.0)
            denom = retail + membership
            if denom > 0:
                out["more_than"][int(cluster_id)] = {
                    "retail_share": retail / denom,
                    "membership_share": membership / denom,
                }

    return out


def _top3_weighted_peer_shares(summary_top3: List[Dict[str, Any]], cohort: str) -> Dict[str, float]:
    """Blend retail/membership mix across zeta top-3 nearest clusters by forecast weight."""
    peer = _cluster_peer_membership_retail_shares()
    cohort_map = peer.get(cohort, {})
    fallback = _cohort_membership_retail_shares()[cohort]
    if not summary_top3:
        return fallback
    retail_acc = 0.0
    mem_acc = 0.0
    w_sum = 0.0
    for row in summary_top3[:3]:
        if not isinstance(row, dict):
            continue
        try:
            cid = int(float(str(row.get("cluster_id"))))
        except (TypeError, ValueError):
            continue
        try:
            w = float(row.get("weight", 0.0) or 0.0)
        except (TypeError, ValueError):
            w = 0.0
        if w <= 0:
            continue
        sh = cohort_map.get(cid, fallback)
        retail_acc += w * float(sh["retail_share"])
        mem_acc += w * float(sh["membership_share"])
        w_sum += w
    if w_sum <= 0:
        return fallback
    return {"retail_share": retail_acc / w_sum, "membership_share": mem_acc / w_sum}


def _membership_retail_count_projection(
    wash_volume_projection: Dict[str, float],
    summary_top3: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    shares_by_year = {
        "year_1": _top3_weighted_peer_shares(summary_top3, "less_than"),
        "year_2": _top3_weighted_peer_shares(summary_top3, "less_than"),
        "year_3": _top3_weighted_peer_shares(summary_top3, "more_than"),
        "year_4": _top3_weighted_peer_shares(summary_top3, "more_than"),
        "year_5": _top3_weighted_peer_shares(summary_top3, "more_than"),
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


def _opex_percent_from_input_form(payload: Dict[str, Any]) -> Optional[float]:
    fi = payload.get("financial_inputs")
    if not isinstance(fi, dict):
        return None
    rows = fi.get("operational_expenses")
    if not isinstance(rows, list):
        return None
    total_pct = 0.0
    found = False
    for r in rows:
        if not isinstance(r, dict):
            continue
        p = r.get("percent_of_sales")
        if p is None:
            continue
        try:
            total_pct += float(p)
            found = True
        except (TypeError, ValueError):
            continue
    return float(total_pct) if found else None


def _project_cost_from_input_form(payload: Dict[str, Any]) -> Optional[float]:
    # Strict source only: financial_inputs.acquisition_budget[].category == "projectCost"
    fi = payload.get("financial_inputs")
    if not isinstance(fi, dict):
        return None
    ab = fi.get("acquisition_budget")
    if not isinstance(ab, list):
        return None
    for r in ab:
        if not isinstance(r, dict):
            continue
        cat = str(r.get("category", "")).strip().lower()
        if cat != "projectcost":
            continue
        v = r.get("total_investment")
        if v is None:
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    return None


def _annual_cash_flow_from_percent(
    wash_volume_projection: Dict[str, float],
    dollars_per_wash: float,
    opex_percent: float,
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    pct = max(float(opex_percent), 0.0) / 100.0
    for y, _s, _e in _YEAR_BANDS:
        washes = float(wash_volume_projection.get(y, 0.0) or 0.0)
        total_revenue = float(washes * dollars_per_wash)
        total_expense = float(total_revenue * pct)
        out[y] = {
            "total_revenue": total_revenue,
            "total_expense": total_expense,
            "net_cash_flow": float(total_revenue - total_expense),
        }
    return out


def _expense_breakdown_from_percent(
    payload: Dict[str, Any],
    revenue_by_year: Dict[str, float],
) -> Optional[Dict[str, Any]]:
    fi = payload.get("financial_inputs")
    if not isinstance(fi, dict):
        return None
    rows = fi.get("operational_expenses")
    if not isinstance(rows, list):
        return None
    out_rows: List[Dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        cat = str(r.get("category", "")).strip()
        p = r.get("percent_of_sales")
        if not cat or p is None:
            continue
        try:
            pct = float(p)
        except (TypeError, ValueError):
            continue
        amounts = {
            y: float((float(revenue_by_year.get(y, 0.0) or 0.0) * pct) / 100.0)
            for y, _s, _e in _YEAR_BANDS
        }
        out_rows.append(
            {
                "category": cat,
                "percent_of_sales": pct,
                "amount_by_year": amounts,
            }
        )
    if not out_rows:
        return None
    return {"items": out_rows}


def _break_even_volume_per_month_year2_vs_project_cost(
    *,
    project_cost: Optional[float],
    dpw: float,
    uses_fixed_annual_opex: bool,
    opex_year1: Optional[float],
    opex_year2: Optional[float],
    opex_percent_total: Optional[float],
) -> Optional[float]:
    """
    Minimum constant wash count per month (same every month in Year 1 and Year 2) so that,
    by the end of Year 2, cumulative operating cash covers ``project_cost``.

    - ``dpw`` = effective $/wash (Σ price_i × pct_i/100).
    - **Fixed annual opex** (``opex_years`` list): need 24·V·dpw ≥ project_cost + Opex_Y1 + Opex_Y2.
    - **Percent-of-revenue opex**: net cash per $ revenue is (1 − total_pct/100), so
      need 24·V·dpw·(1 − pct/100) ≥ project_cost.

    This is independent of the zeta wash forecast volumes (those drive *actual* revenue;
    this headline answers *how many washes/month* would recover project cost by end of Y2
    under the stated pricing and opex rules).
    """
    if project_cost is None or float(project_cost) <= 0 or dpw <= 1e-12:
        return None
    pc = float(project_cost)
    if uses_fixed_annual_opex:
        if opex_year1 is None or opex_year2 is None:
            return None
        num = pc + float(opex_year1) + float(opex_year2)
        return float(num / (24.0 * dpw))
    if opex_percent_total is not None:
        k = max(min(float(opex_percent_total) / 100.0, 0.9999), 0.0)
        net = 1.0 - k
        if net <= 1e-12:
            return None
        return float(pc / (24.0 * dpw * net))
    return None


def _attach_financial_outputs(
    out: Dict[str, Any],
    payload: Dict[str, Any],
    wash_volume_projection: Dict[str, float],
    forecast: pd.DataFrame,
) -> None:
    wash_prices = payload.get("wash_prices")
    wash_pcts = payload.get("wash_pcts")
    if not (isinstance(wash_prices, list) and isinstance(wash_pcts, list) and wash_prices and len(wash_prices) == len(wash_pcts)):
        return

    years = tuple(y for y, _s, _e in _YEAR_BANDS)
    dpw = _effective_dollar_per_wash([float(x) for x in wash_prices], [float(x) for x in wash_pcts])
    out["dollars_per_wash"] = dpw
    revenue_by_year = {y: float(float(wash_volume_projection.get(y, 0.0) or 0.0) * dpw) for y in years}
    out["revenue_by_year"] = revenue_by_year

    opex_years = payload.get("opex_years")
    cash_flow: Optional[Dict[str, Dict[str, float]]] = None
    uses_fixed_annual_opex = False
    if isinstance(opex_years, list) and len(opex_years) in (4, 5):
        uses_fixed_annual_opex = True
        opex_vals = [float(v) for v in opex_years]
        if len(opex_vals) == 4:
            opex_vals.append(opex_vals[-1])
        opex_by_year = {y: float(v) for y, v in zip(years, opex_vals)}
        out["opex_by_year"] = opex_by_year
        out["profit_by_year"] = {y: float(revenue_by_year[y] - opex_by_year[y]) for y in years}
        cash_flow = {
            y: {
                "total_revenue": revenue_by_year[y],
                "total_expense": opex_by_year[y],
                "net_cash_flow": float(revenue_by_year[y] - opex_by_year[y]),
            }
            for y in years
        }
    else:
        opex_pct = _opex_percent_from_input_form(payload)
        if opex_pct is not None:
            out["opex_percent_total"] = float(opex_pct)
            cash_flow = _annual_cash_flow_from_percent(wash_volume_projection, dpw, opex_pct)
            out["opex_by_year"] = {y: float(v["total_expense"]) for y, v in cash_flow.items()}
            out["profit_by_year"] = {y: float(v["net_cash_flow"]) for y, v in cash_flow.items()}
            expense_breakdown = _expense_breakdown_from_percent(payload, revenue_by_year)
            if expense_breakdown is not None:
                out["expense_breakdown"] = expense_breakdown
    if cash_flow is None and "profit_by_year" in out:
        profit_by_year = {y: float((out.get("profit_by_year") or {}).get(y, 0.0) or 0.0) for y in years}
        cash_flow = {
            y: {
                "total_revenue": revenue_by_year[y],
                "total_expense": float(revenue_by_year[y] - profit_by_year[y]),
                "net_cash_flow": profit_by_year[y],
            }
            for y in years
        }
    if cash_flow is not None:
        out["cash_flow_projections"] = cash_flow

    project_cost = _project_cost_from_input_form(payload)
    out["project_cost"] = project_cost
    profit_by_year = {y: float((out.get("profit_by_year") or {}).get(y, 0.0) or 0.0) for y in years}
    if project_cost is not None and project_cost > 0:
        out["cash_on_cash_return"] = {y: float((profit_by_year[y] / project_cost) * 100.0) for y in years}

    y2_rev = revenue_by_year["year_2"]
    y2_net = profit_by_year["year_2"]
    oby = out.get("opex_by_year")
    opex_y1_fixed = float(oby["year_1"]) if uses_fixed_annual_opex and isinstance(oby, dict) else None
    opex_y2_fixed = float(oby["year_2"]) if uses_fixed_annual_opex and isinstance(oby, dict) else None
    opex_pct_total = out.get("opex_percent_total") if not uses_fixed_annual_opex else None
    pct_for_be = float(opex_pct_total) if isinstance(opex_pct_total, (int, float)) else None

    break_even_vol_pm = _break_even_volume_per_month_year2_vs_project_cost(
        project_cost=project_cost,
        dpw=float(dpw),
        uses_fixed_annual_opex=uses_fixed_annual_opex,
        opex_year1=opex_y1_fixed,
        opex_year2=opex_y2_fixed,
        opex_percent_total=pct_for_be,
    )
    total_revenue_5y = float(forecast["volume"].sum() * dpw) if len(forecast) else None
    out["headlines_washcast"] = {
        "revenue_total_5y": total_revenue_5y,
        "net_cash_flow_year_2": y2_net,
        "project_cost": project_cost,
        "effective_dollars_per_wash": float(dpw),
        "break_even_volume_per_month": break_even_vol_pm,
        "break_even_volume_per_month_basis": (
            "Constant washes/month in Year 1–2 (24 months) such that cumulative net operating cash "
            "covers projectCost; revenue = washes × effective_dollars_per_wash; opex from opex_years "
            "(fixed $ per year) or operational_expenses percent-of-sales total."
        ),
        "avg_net_margin_year_2_pct": float((y2_net / y2_rev) * 100.0) if y2_rev > 0 else None,
    }


@celery_app.task(bind=True, name="pnl_zeta_projection")
def run_zeta_projection_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Greenfield wash projection via zeta_modelling (quantile + calibrated bands).
    Compatible shape with legacy /wash-count-plot consumers.
    """
    payload = _normalize_central_task_payload(payload)
    lat, lon, address = _resolve_site_lat_lon(payload)

    zp = _zeta_params(payload)
    if zp["cluster_distance_policy"] == "fixed":
        max_cluster_distance_km = float(
            zp["max_cluster_distance_km"] if zp["max_cluster_distance_km"] is not None else 100.0
        )
    else:
        max_cluster_distance_km = _regional_cluster_distance_cap_km(float(lat), float(lon))
    months_use = min(max(int(zp["forecast_months"]), 12), 60)
    forecast, summary = _run_zeta_forecast_df(
        lat,
        lon,
        months=months_use,
        margin_per_wash=zp["margin_per_wash"],
        fixed_monthly_cost=zp["fixed_monthly_cost"],
        ramp_up_cost=zp["ramp_up_cost"],
        scenario=zp["scenario"],
        target_coverage=zp["target_calibration_coverage"],
        start_date=zp["forecast_start_date"],
        enable_mature_yoy_control=zp["enable_mature_yoy_control"],
        mature_yoy_start_year=zp["mature_yoy_start_year"],
        mature_min_yoy=zp["mature_min_yoy"],
        mature_max_yoy=zp["mature_max_yoy"],
        max_cluster_distance_km=max_cluster_distance_km,
        lifecycle_min_growth=zp["lifecycle_min_growth"],
        lifecycle_max_growth=zp["lifecycle_max_growth"],
    )

    wash_vol = _wash_volume_projection_from_forecast(forecast)
    wash_volume_range_minmax = _wash_volume_range_minmax_from_forecast(forecast)
    timeline_60 = _monthly_projection(forecast, 60)
    timeline_48 = timeline_60[:48]
    top3 = summary.get("top3_clusters") or []

    out: Dict[str, Any] = {
        "task_id": self.request.id,
        "input": {"address": address, "lat": lat, "lon": lon},
        "method": "zeta_modelling",
        "level_model": "quantile_lgbm",
        "calendar_year_washes": wash_vol,
        "wash_volume_range_minmax": wash_volume_range_minmax,
        "monthly_projection_48mo": timeline_48,
        "monthly_projection_60mo": timeline_60,
        "zeta_forecast_summary": _json_sanitize(summary),
        "zeta_model_path": str(_ARTIFACTS_PATH),
        "zeta_data_path": str(_ZETA_DATA),
    }

    wash_prices = payload.get("wash_prices")
    wash_pcts = payload.get("wash_pcts")
    if isinstance(wash_prices, list) and isinstance(wash_pcts, list) and len(wash_prices) == len(wash_pcts) and wash_prices:
        # Blended $/wash = Σ (price_i × pct_i/100); annual revenue = that × P50 year wash volume.
        dpw = _effective_dollar_per_wash([float(x) for x in wash_prices], [float(x) for x in wash_pcts])
        mrc = _membership_retail_count_projection(wash_vol, top3)
        out["membership_retail_count"] = mrc
        out["membership_retail_revenue"] = _membership_retail_revenue_projection(mrc, dpw)
    _attach_financial_outputs(out, payload, wash_vol, forecast)

    return _json_sanitize(out)


@celery_app.task(bind=True, name="pnl_central_input_form")
def run_pnl_central_input_form_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Central input-form task: zeta_modelling volume + P10/P90-derived year ranges.
    """
    payload = _normalize_central_task_payload(payload)
    lat, lon, _address = _resolve_site_lat_lon(payload)

    zp = _zeta_params(payload)
    if zp["cluster_distance_policy"] == "fixed":
        max_cluster_distance_km = float(
            zp["max_cluster_distance_km"] if zp["max_cluster_distance_km"] is not None else 100.0
        )
    else:
        max_cluster_distance_km = _regional_cluster_distance_cap_km(float(lat), float(lon))
    months_use = min(max(int(zp["forecast_months"]), 12), 60)
    forecast, summary = _run_zeta_forecast_df(
        lat,
        lon,
        months=months_use,
        margin_per_wash=zp["margin_per_wash"],
        fixed_monthly_cost=zp["fixed_monthly_cost"],
        ramp_up_cost=zp["ramp_up_cost"],
        scenario=zp["scenario"],
        target_coverage=zp["target_calibration_coverage"],
        start_date=zp["forecast_start_date"],
        enable_mature_yoy_control=zp["enable_mature_yoy_control"],
        mature_yoy_start_year=zp["mature_yoy_start_year"],
        mature_min_yoy=zp["mature_min_yoy"],
        mature_max_yoy=zp["mature_max_yoy"],
        max_cluster_distance_km=max_cluster_distance_km,
        lifecycle_min_growth=zp["lifecycle_min_growth"],
        lifecycle_max_growth=zp["lifecycle_max_growth"],
    )

    wash_volume_projection = _wash_volume_projection_from_forecast(forecast)
    wash_volume_range_minmax = _wash_volume_range_minmax_from_forecast(forecast)

    inp = payload.get("customer_information") or {}
    out_input = {
        "address": inp.get("site_address") if isinstance(inp, dict) else None,
        "lat": lat,
        "lon": lon,
    }

    out: Dict[str, Any] = {
        "task_id": self.request.id,
        "input": out_input,
        "wash_volume_projection": wash_volume_projection,
        "wash_volume_range_minmax": wash_volume_range_minmax,
        "zeta_forecast_summary": _json_sanitize(summary),
        "zeta_model_path": str(_ARTIFACTS_PATH),
        "zeta_data_path": str(_ZETA_DATA),
    }

    wash_prices = payload.get("wash_prices")
    wash_pcts = payload.get("wash_pcts")
    top3 = summary.get("top3_clusters") or []
    membership_retail_count = _membership_retail_count_projection(wash_volume_projection, top3)
    out["membership_retail_count"] = membership_retail_count

    if isinstance(wash_prices, list) and isinstance(wash_pcts, list) and len(wash_prices) == len(wash_pcts) and wash_prices:
        dpw = _effective_dollar_per_wash([float(x) for x in wash_prices], [float(x) for x in wash_pcts])
        out["membership_retail_revenue"] = _membership_retail_revenue_projection(membership_retail_count, dpw)
    _attach_financial_outputs(out, payload, wash_volume_projection, forecast)

    return _json_sanitize(out)


# Backward-compatible Celery name for clients still enqueueing the old task.
run_clustering_v2_projection_task = run_zeta_projection_task