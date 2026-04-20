"""
V2 end-to-end projection for a NEW site.

Given an address (or lat/lon) we:
  1. Geocode it (via app.utils.common.get_lat_long when we need it).
  2. Assign it to the nearest DBSCAN-12km cluster for BOTH cohorts
     (>2y daily model, <2y monthly model) using train-only centroids.
  3. Build a serving-time feature vector using cluster medians (for the
     site features we don't have for a brand-new site) + cluster-context
     aggregates (the train-only "local market" signal).
  4. Score the Ridge portable model to get an expected wash-count level
     (per-day for >2y, per-month for <2y).
  5. Pull the train-only per-cluster monthly wash-count series and
     project it forward 24 months using Holt-Winters / ARIMA / blend.
  6. Anchor the forecast to the Ridge ballpark so the projection
     reflects THIS specific site and not just the cluster average.
  7. Write a JSON response and a two-panel bar plot (months on x,
     wash-count on y) into ``results/projection_demo``.

Usage:
  python project_site.py --address "1234 Main St, Austin TX"
  python project_site.py --lat 30.27 --lon -97.74 --method blend
  python project_site.py --address "..." --out-name custom_tag
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ARIMA, ExponentialSmoothing


REPO_ROOT = Path(__file__).resolve().parents[3]
V2_DIR = Path(__file__).resolve().parent
MODELS_DIR = V2_DIR / "models"
DEMO_OUT = V2_DIR / "results" / "projection_demo"
# Suppress Ridge + forecast bars when nearest train centroid exceeds this distance (km).
MAX_NEAREST_CLUSTER_DISPLAY_KM = 20.0

sys.path.insert(0, str(REPO_ROOT))
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------

def _load_json(p: Path) -> Any:
    return json.loads(p.read_text())


def _load_cohort(cohort: str) -> dict[str, Any]:
    d = MODELS_DIR / cohort
    return {
        "model": _load_json(d / "wash_count_model_12km.portable.json"),
        "centroids": _load_json(d / "cluster_centroids_12km.json"),
        "context": _load_json(d / "cluster_context_12km.json"),
        "series": _load_json(d / "cluster_monthly_series_12km.json"),
        "spec": _load_json(d / "feature_spec_12km.json"),
    }


# ---------------------------------------------------------------------------
# Geocoding + cluster assignment
# ---------------------------------------------------------------------------

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
    """TomTom helper returns lat/lon; other callers may use latitude/longitude."""
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
        # Disambiguate city/state-only strings (TomTom often needs country)
        if not stripped.upper().endswith("USA") and not stripped.upper().endswith("UNITED STATES"):
            la, lo = _coords_from_geocode(get_lat_long(stripped + ", USA"))
    if la is None or lo is None:
        raise SystemExit(
            f"Geocoding failed for address: {address}\n"
            "Try --lat/--lon, a fuller address (with ZIP), or set TOMTOM_GEOCODE_API_URL and "
            "TOMTOM_API_KEY in .env if geocoding is disabled."
        )
    return la, lo, address


# ---------------------------------------------------------------------------
# Ridge scoring (pure-numpy, no sklearn at serve time)
# ---------------------------------------------------------------------------

def _score_portable(model: dict[str, Any], feature_vec: dict[str, float]) -> float:
    order = model["feature_order"]
    x = np.array([feature_vec.get(c, np.nan) for c in order], dtype=float)
    # imputer (median statistics)
    stats = np.array(model["imputer"]["statistics"], dtype=float)
    x = np.where(np.isfinite(x), x, stats)
    # scaler
    mean = np.array(model["scaler"]["mean"], dtype=float)
    scale = np.array(model["scaler"]["scale"], dtype=float)
    scale = np.where(scale == 0, 1.0, scale)
    xs = (x - mean) / scale
    # ridge
    coef = np.array(model["ridge"]["coef"], dtype=float)
    intercept = float(model["ridge"]["intercept"])
    return float(xs @ coef + intercept)


def _feature_vector(
    spec: dict[str, Any],
    context_row: dict[str, Any] | None,
    lat: float,
    lon: float,
    cluster_id: int,
) -> dict[str, float]:
    """Build a Ridge input for a BRAND-NEW site.

    The time/lag features in training (``day_number``, ``month_number``,
    ``year_number``, ``prev_wash_count``...) are SITE-RELATIVE ages and
    lagged history, not calendar. For a brand-new site we have none of
    these, so we leave them as NaN and let the imputer fill them with
    training medians. This gives Ridge a "typical site at median age"
    reading - exactly the right serve for greenfield projection.

    We supply only the quantities that are actually known for the new
    location: lat/lon, the assigned cluster id, and the train-only
    cluster-context aggregates (which carry the local-market signal).
    """
    vec: dict[str, float] = {
        "latitude": lat,
        "longitude": lon,
        "dbscan_cluster_12km": float(cluster_id),
    }
    if context_row:
        for k, v in context_row.items():
            if k == "cluster_id":
                continue
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                continue
            vec[k] = float(v)
    return vec


# ---------------------------------------------------------------------------
# Time-series forecasting
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Per-cohort projection payload
# ---------------------------------------------------------------------------

def _project_cohort(
    cohort: str,
    cohort_label: str,
    lat: float,
    lon: float,
    method: str,
    cohort_assets: dict[str, Any],
    days_per_month: int,
    is_daily_model: bool,
) -> dict[str, Any]:
    centroids = cohort_assets["centroids"]["centroids"]
    if not centroids:
        return {"error": f"No centroids for {cohort_label}"}

    nearest = _nearest_cluster(centroids, lat, lon)
    cluster_id = int(nearest["cluster_id"])
    d_km = float(nearest["distance_km"])
    if np.isfinite(d_km) and d_km > MAX_NEAREST_CLUSTER_DISPLAY_KM:
        return {
            "error": (
                f"Nearest cluster is {d_km:.1f} km away; projections hidden when centroid distance "
                f"exceeds {MAX_NEAREST_CLUSTER_DISPLAY_KM:.0f} km."
            ),
            "cohort": cohort_label,
            "cluster": {
                "cluster_id": cluster_id,
                "distance_km": d_km,
                "size": nearest.get("size"),
            },
        }

    # cluster-context row (dict from records)
    ctx_row = None
    for r in cohort_assets["context"]["records"]:
        if int(r.get("cluster_id", -999)) == cluster_id:
            ctx_row = r
            break

    vec = _feature_vector(cohort_assets["spec"], ctx_row, lat, lon, cluster_id)
    ridge_raw = _score_portable(cohort_assets["model"], vec)
    ridge_monthly = ridge_raw * days_per_month if is_daily_model else ridge_raw

    # cluster monthly series (train only)
    series = _series_to_df(cohort_assets["series"]["series"].get(str(cluster_id), []))
    cluster_monthly_level = float(series.tail(6).mean()) if len(series) else float("nan")

    # anchor forecast so level tracks this specific site
    if len(series) >= 6 and np.isfinite(ridge_monthly) and cluster_monthly_level > 0:
        scale = ridge_monthly / cluster_monthly_level
    else:
        scale = 1.0

    fc_raw = _forecast(series, 24, method)
    fc = fc_raw * scale

    # Disjoint 6-month period sums (not running cumulative): 1–6, 7–12, 13–18, 19–24.
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

    return {
        "cohort": cohort_label,
        "cluster": {
            "cluster_id": cluster_id,
            "distance_km": nearest["distance_km"],
            "size": nearest["size"],
            "centroid_lat": nearest["lat"],
            "centroid_lon": nearest["lon"],
            "cluster_monthly_level_last_6mo": cluster_monthly_level if np.isfinite(cluster_monthly_level) else None,
        },
        "ridge_prediction": {
            "raw_level": ridge_raw,
            "monthly_level": ridge_monthly if np.isfinite(ridge_monthly) else None,
            "anchor_scale": scale,
            "cluster_context_used": ctx_row is not None,
        },
        "method": method,
        "horizons": horizons,
        "monthly_projection": [
            {"month": d.strftime("%Y-%m-%d"), "wash_count": float(max(v, 0))}
            for d, v in fc.items()
        ],
    }


def _bridge_mature_monthly_to_opening_last_month(resp: dict[str, Any]) -> None:
    """Scale >2y monthly forecast so operational month 25 matches <2y month 24 (level handoff)."""
    lt = resp.get("less_than_2yrs")
    gt = resp.get("more_than_2yrs")
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
    rp = gt.get("ridge_prediction") or {}
    if rp.get("monthly_level") is not None and np.isfinite(float(rp["monthly_level"])):
        rp["monthly_level"] = float(rp["monthly_level"]) * factor
    if rp.get("raw_level") is not None and np.isfinite(float(rp["raw_level"])):
        rp["raw_level"] = float(rp["raw_level"]) * factor
    gt["ridge_prediction"] = rp
    gt["opening_to_mature_bridge"] = {
        "method": "scale_entire_mature_monthly_track",
        "scale_factor": float(factor),
        "aligned_month_24_lt_to_month_25_gt": True,
    }


def _enrich_brand_new_site_timeline(resp: dict[str, Any]) -> None:
    """Opening: <2y months 1–24. Continuation: >2y public view is months 30–48 (bars + monthly rows)."""
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
    w = [float(r.get("wash_count", 0) or 0) for r in rows]

    # Mature track: four disjoint 6-month sums (operational 25–30, 31–36, 37–42, 43–48).
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

    # Publish only mature months 30–48 (skip forecast months 1–5 of mature series = operational 25–29).
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


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_projection(resp: dict[str, Any], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, cohort_key, title in [
        (axes[0], "less_than_2yrs", "<2y | years 1–2 (operational mo 1–24)"),
        (axes[1], "more_than_2yrs", ">2y | continuation mo 30–48 (site cumulative)"),
    ]:
        block = resp.get(cohort_key, {})
        if "error" in block or "horizons" not in block:
            ax.text(0.5, 0.5, block.get("error", "no data"), ha="center")
            ax.set_title(title)
            continue
        hz = block["horizons"]
        if cohort_key == "more_than_2yrs" and "30m" in hz:
            horizon_labels = ["30m", "36m", "42m", "48m"]
        else:
            horizon_labels = ["6m", "12m", "18m", "24m"]
        vals = [hz[h]["six_month_period_sum"] for h in horizon_labels]
        bars = ax.bar(horizon_labels, vals, color="#3b82f6")
        ax.set_title(f"{title}\ncluster {block['cluster']['cluster_id']} ({block['cluster']['size']} sites, "
                     f"{block['cluster']['distance_km']:.1f} km away)")
        ax.set_ylabel("Washes in 6-month period")
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{int(v):,}", ha="center", va="bottom", fontsize=9)
    inp = resp["input"]
    loc = inp.get("address") or f"{inp['lat']:.4f},{inp['lon']:.4f}"
    fig.suptitle(f"V2 projection for {loc}   (method={resp['method']})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run_projection(address: str | None, lat: float | None, lon: float | None, method: str) -> dict[str, Any]:
    lat, lon, addr = _resolve_latlon(address, lat, lon)
    print(f"[project] address={addr!r}  lat={lat:.5f}  lon={lon:.5f}  method={method}")

    more = _load_cohort("more_than")
    less = _load_cohort("less_than")

    resp = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "radius_km": 12.0,
        "method": method,
        "input": {"address": addr, "lat": lat, "lon": lon},
        "more_than_2yrs": _project_cohort(
            cohort="more_than",
            cohort_label="more_than_2yrs",
            lat=lat, lon=lon, method=method,
            cohort_assets=more,
            days_per_month=30,
            is_daily_model=True,
        ),
        "less_than_2yrs": _project_cohort(
            cohort="less_than",
            cohort_label="less_than_2yrs",
            lat=lat, lon=lon, method=method,
            cohort_assets=less,
            days_per_month=30,
            is_daily_model=False,
        ),
    }
    _bridge_mature_monthly_to_opening_last_month(resp)
    _enrich_brand_new_site_timeline(resp)
    return resp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--address", type=str, default=None)
    ap.add_argument("--lat", type=float, default=None)
    ap.add_argument("--lon", type=float, default=None)
    ap.add_argument("--method", type=str, default="blend",
                    choices=["holt_winters", "arima", "blend"])
    ap.add_argument("--out-name", type=str, default=None,
                    help="tag appended to output filenames")
    args = ap.parse_args()

    resp = run_projection(args.address, args.lat, args.lon, args.method)

    DEMO_OUT.mkdir(parents=True, exist_ok=True)
    tag = args.out_name or datetime.utcnow().strftime("%Y%m%d-%H%M%S")
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
