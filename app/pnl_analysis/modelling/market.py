"""
Local-market engine for the P&L explorer backend.

Wraps the cold-start forecaster (earnest-proforma-2.0/streamlits/coldstart_model.py) and
re-implements the pure data/trend/forecast helpers from that app's Streamlit UI so the FastAPI
routes can return plot-ready data for two tabs:

  1. explore_market(...)     — per-site monthly series within a neighbour radius (line charts).
  2. pinpoint_forecast(...)  — a dropped pin's 5-year trajectory + the local market's history-plus-forecast.

The cold-start artifacts (LightGBM plateau models + empirical ramp curves + learned cannibalization)
are loaded once and cached; the monthly panel (main-ds.csv) is loaded + clustered once and cached.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
COLDSTART_DIR = ROOT / "earnest-proforma-2.0" / "streamlits"
CSV = ROOT / "earnest-proforma-2.0" / "data" / "main-ds.csv"
EARTH_KM = 6371.0088

# import the cold-start model module (lives alongside the Streamlit app)
if str(COLDSTART_DIR) not in sys.path:
    sys.path.insert(0, str(COLDSTART_DIR))
import coldstart_model as cm  # noqa: E402

# tab-1 metrics: label -> dataframe column
METRICS: Dict[str, str] = {
    "Total washes": "tot_wash_count",
    "Membership washes": "mem_wash_count",
    "Retail washes": "ret_wash_count",
    "Total revenue ($)": "tot_revenue",
    "Membership share of washes": "mem_share_wash",
}
METRIC_LABEL_BY_COL = {v: k for k, v in METRICS.items()}

_PANEL_CACHE: Dict[str, Any] = {}
_MODEL_CACHE: Dict[str, Any] = {}


# ─────────────────────────── data ───────────────────────────
def _load_panel():
    """Monthly site-level panel + per-site table (with adaptive local-market clusters). Cached."""
    if "df" in _PANEL_CACHE:
        return _PANEL_CACHE["df"], _PANEL_CACHE["site"]

    raw = pd.read_csv(CSV, low_memory=False)
    raw["date"] = pd.to_datetime(dict(year=raw.year, month=raw.month, day=1))
    raw["op_start"] = pd.to_datetime(raw["operational_start"], format="%m-%Y", errors="coerce")
    raw["site_key"] = raw.client_id.astype(str) + "::" + raw.site_id.astype(str)

    df = raw.copy()
    asp_r = np.where(df.ret_wash_count > 0, df.ret_revenue / df.ret_wash_count, np.nan)
    asp_m = np.where(df.mem_wash_count > 0, df.mem_revenue / df.mem_wash_count, np.nan)
    df.loc[asp_r > 200, "ret_revenue"] = np.nan
    df.loc[asp_m > 200, "mem_revenue"] = np.nan
    df["tot_wash_count"] = df.mem_wash_count + df.ret_wash_count
    df["tot_revenue"] = df[["mem_revenue", "ret_revenue"]].sum(axis=1, min_count=1)
    df["mem_share_wash"] = np.where(df.tot_wash_count > 0, df.mem_wash_count / df.tot_wash_count, np.nan)

    site = (
        df.groupby("site_key")
        .agg(client_name=("client_name", "first"), lat=("lat", "first"), lon=("lon", "first"),
             state=("state", "first"), region=("region", "first"), op_start=("op_start", "first"),
             first_obs=("date", "min"), last_obs=("date", "max"), n_obs=("date", "size"))
        .reset_index()
    )
    site["left_censored"] = site.op_start <= pd.Timestamp("2020-01-01")
    site["has_coords"] = site[["lat", "lon"]].notna().all(axis=1)
    site["cluster"] = cm.assign_clusters(site, "adaptive")

    _PANEL_CACHE["df"] = df
    _PANEL_CACHE["site"] = site
    return df, site


def _load_model():
    if "art" not in _MODEL_CACHE:
        _MODEL_CACHE["art"] = cm.load()
    return _MODEL_CACHE["art"]


def haversine_km(lat1, lon1, lat2, lon2):
    r = np.radians
    lat1, lon1, lat2, lon2 = r(lat1), r(lon1), r(lat2), r(lon2)
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ─────────────────────────── trend / forecast helpers (from the Streamlit app) ───────────────────────────
TAU_Y = 2.0   # post-maturity growth saturates over ~2 yr (sites plateau ~24 mo empirically)


def _sat_years(years):
    """Saturating 'effective years' of drift: ≈ linear near 0, asymptotes to TAU_Y (booms decelerate)."""
    return TAU_Y * (1.0 - np.exp(-np.asarray(years, dtype=float) / TAU_Y))


def _robust_slope(arr):
    """Per-month log-growth slope (Theil-Sen on 6-mo-smoothed log level, last ~30 mo) + its SE."""
    arr = np.asarray(arr, dtype=float); arr = arr[np.isfinite(arr)]
    if len(arr) < 18:
        return None
    sm = pd.Series(arr).rolling(6, min_periods=3).mean().dropna().to_numpy()
    K = min(30, len(sm))
    if K < 8:
        return None
    yv = np.log(np.clip(sm[-K:], 1.0, None)); xv = np.arange(K, dtype=float)
    try:
        from scipy.stats import theilslopes
        sl, _, lo, hi = theilslopes(yv, xv)
    except Exception:
        sl = float(np.polyfit(xv, yv, 1)[0]); lo = hi = sl
    se = max((hi - lo) / (2 * 1.96), 1e-9) * np.sqrt(6.0)
    return float(sl), float(se)


def _shrink_annualize(sl, se):
    """Shrink a per-month slope toward 0 by its signal-to-noise t²/(1+t²), annualize → (g, g_lo, g_hi)."""
    SANE = lambda r: float(np.clip(r, -0.40, 0.40))
    t = abs(sl) / se
    sl_c = sl * (t * t / (1.0 + t * t))
    return SANE(np.exp(sl_c * 12) - 1), SANE(np.exp((sl - 1.96 * se) * 12) - 1), SANE(np.exp((sl + 1.96 * se) * 12) - 1)


def robust_growth(arr):
    """Annual growth (central + data-based CI band) for one series."""
    g = _robust_slope(arr)
    return _shrink_annualize(*g) if g else (0.0, 0.0, 0.0)


def market_trend(piv):
    """Composition-robust market trend from a date×site pivot: each site's own slope, pooled by median."""
    slopes, ses = [], []
    for col in getattr(piv, "columns", []):
        r = _robust_slope(piv[col].dropna().to_numpy())
        if r:
            slopes.append(r[0]); ses.append(r[1])
    if not slopes:
        return 0.0, 0.0, 0.0
    slopes = np.asarray(slopes); ses = np.asarray(ses); n = len(slopes)
    sl = float(np.median(slopes))
    between = float(np.std(slopes, ddof=1)) if n >= 2 else 0.0
    se = max(np.hypot(between, float(np.median(ses))) / np.sqrt(n), 1e-9)
    return _shrink_annualize(sl, se)


def forecast_series(s, H, g=None):
    """Smooth 5-yr expected-trend forecast: starts at the last actual, blends into a trend line at the recent
    deseasonalized level, growing at a robust annual rate that saturates over ~2 yr."""
    s = pd.Series(s).astype(float).dropna()
    n = len(s)
    if n == 0:
        return np.zeros(H)
    arr = s.to_numpy()
    last = float(arr[-1])
    level = float(arr[-min(12, n):].mean())
    if g is None:
        g = robust_growth(arr)[0]
    t = np.arange(1, H + 1)
    trend = level * (1 + g) ** _sat_years(t / 12.0)
    w = np.exp(-(t - 1) / 3.0)
    return np.clip(last * w + trend * (1 - w), 0, None)


# ─────────────────────────── tab 1: explore market ───────────────────────────
def explore_market(lat: float, lon: float, metric_col: str, radius_km: float,
                   smoothing: int, max_sites: int, x_axis: str) -> Dict[str, Any]:
    """Per-site monthly series for every site within `radius_km` of the pin (the local market).

    `x_axis` ∈ {"date", "months_since_open"}. Series are role-tagged (entrant vs incumbent);
    an entrant is a genuinely-observed opening that came after the market's earliest site.
    """
    df, site = _load_panel()
    metric_label = METRIC_LABEL_BY_COL.get(metric_col, metric_col)

    d = haversine_km(lat, lon, site.lat.values, site.lon.values)
    in_radius = (d <= radius_km) & site.has_coords.values
    nb = site.loc[in_radius].copy()
    nb["dist_km"] = d[in_radius]
    nb = nb.sort_values("op_start")
    n_in_market = len(nb)

    if n_in_market:
        earliest = nb.op_start.min()
        nb["is_entrant"] = (~nb.left_censored) & (nb.op_start > earliest)
    else:
        nb["is_entrant"] = False

    # cap clutter: always keep every entrant, fill the rest with the nearest incumbents
    entrants = nb[nb.is_entrant]
    n_inc = max(0, max_sites - len(entrants))
    inc = nb[~nb.is_entrant].nsmallest(n_inc, "dist_km")
    shown = pd.concat([entrants, inc]).drop_duplicates("site_key").sort_values("op_start")

    series: List[Dict[str, Any]] = []
    for _, s in shown.iterrows():
        ts = df.loc[df.site_key == s.site_key].set_index("date")[metric_col].sort_index()
        if ts.dropna().empty:
            continue
        ts = ts.reindex(pd.date_range(ts.index.min(), ts.index.max(), freq="MS"))
        if smoothing and smoothing > 1:
            ts = ts.rolling(smoothing, center=True, min_periods=1).mean()
        if x_axis == "months_since_open":
            x = ((ts.index.year - s.op_start.year) * 12 + (ts.index.month - s.op_start.month)).tolist()
        else:
            x = [d.strftime("%Y-%m-%d") for d in ts.index]
        y = [None if pd.isna(v) else float(v) for v in ts.values]
        series.append({
            "site_key": s.site_key,
            "name": s.client_name,
            "role": "entrant" if s.is_entrant else "incumbent",
            "dist_km": round(float(s.dist_km), 2),
            "op_start": s.op_start.strftime("%Y-%m") if pd.notna(s.op_start) else None,
            "x": x,
            "y": y,
        })

    entry_markers = [
        {"site_key": s.site_key, "name": s.client_name, "op_start": s.op_start.strftime("%Y-%m-%d")}
        for _, s in entrants.iterrows() if pd.notna(s.op_start)
    ]

    return {
        "lat": lat, "lon": lon,
        "metric": metric_col, "metric_label": metric_label,
        "x_axis": x_axis,
        "x_axis_label": "date" if x_axis == "date" else "months since each site opened",
        "y_axis_label": metric_label,
        "radius_km": radius_km, "smoothing": smoothing,
        "n_sites_in_market": int(n_in_market),
        "n_shown": len(series),
        "n_entrants": int(len(entrants)),
        "series": series,
        "entry_markers": entry_markers,
    }


# ─────────────────────────── tab 2: pin-point forecast ───────────────────────────
def pinpoint_forecast(lat: float, lon: float, brand: Optional[str], plateau_override: Optional[float],
                      mem_growth_pct: float, ret_growth_pct: float, horizon_months: int) -> Dict[str, Any]:
    """A dropped pin's 5-year trajectory + the local market's history-and-forecast total wash count.

    `mem_growth_pct` / `ret_growth_pct` are extra yr3-5 drift (%/yr) added on top of the market's own trend.
    """
    df, site = _load_panel()
    art = _load_model()
    gm = mem_growth_pct / 100.0
    gr = ret_growth_pct / 100.0

    # learn this local market's per-component trend (membership vs retail) from neighbours' own series
    nb = site[site.has_coords]
    d = haversine_km(lat, lon, nb.lat.values, nb.lon.values)
    keys = nb.site_key[(d <= 20) & (d > 1e-6)].tolist()
    if keys:
        sub = df[df.site_key.isin(keys)]
        pm = sub.pivot_table(index="date", columns="site_key", values="mem_wash_count")
        pr = sub.pivot_table(index="date", columns="site_key", values="ret_wash_count")
        mem_g, mem_lo, mem_hi = market_trend(pm)
        ret_g, ret_lo, ret_hi = market_trend(pr)
    else:
        mem_g = mem_lo = mem_hi = ret_g = ret_lo = ret_hi = 0.0

    traj, info = cm.predict_site(
        lat, lon, brand=brand, plateau_override=(plateau_override or None),
        annual_mem_growth=mem_g + gm, annual_ret_change=ret_g + gr,
        mem_growth_band=(mem_lo + gm, mem_hi + gm),
        ret_change_band=(ret_lo + gr, ret_hi + gr),
        horizon=horizon_months, art=art,
    )
    g = traj.set_index("month")
    months = [int(m) for m in g.index]

    trajectory = {
        "months": months,
        "total_med": [float(v) for v in g.total_med],
        "total_lo": [float(v) for v in g.total_lo],
        "total_hi": [float(v) for v in g.total_hi],
        "mem_med": [float(v) for v in g.mem_med],
        "ret_med": [float(v) for v in g.ret_med],
    }

    # ── total local-market wash count — history + 5-year forecast ──
    today = pd.Timestamp(df.date.max())
    H = horizon_months
    fdates = pd.date_range(today + pd.DateOffset(months=1), periods=H, freq="MS")
    _tj = traj.set_index("month")
    new_traj = _tj["total_med"].reindex(range(H)).to_numpy()
    new_lo = _tj["total_lo"].reindex(range(H)).to_numpy()
    new_hi = _tj["total_hi"].reindex(range(H)).to_numpy()

    cp = cm.cannib_params(art, lat, lon)
    market_forecast: Dict[str, Any] = {
        "open_date": (today + pd.DateOffset(months=1)).strftime("%Y-%m-%d"),
        "cannib": {"a": float(cp["a"]), "L": float(cp["L"]), "fallback": bool(cp.get("fallback", False))},
    }

    if keys:
        comp = df[df.site_key.isin(keys)].groupby("date")[["mem_wash_count", "ret_wash_count"]].sum()
        idx = pd.date_range(comp.index.min(), today, freq="MS")
        hist_mem = comp["mem_wash_count"].reindex(idx)
        hist_ret = comp["ret_wash_count"].reindex(idx)
        hist = hist_mem.add(hist_ret, fill_value=0)

        base_mem = forecast_series(hist_mem, H, g=mem_g)
        base_ret = forecast_series(hist_ret, H, g=ret_g)
        base_fc = base_mem + base_ret
        base_mem_lo = forecast_series(hist_mem, H, g=mem_lo); base_mem_hi = forecast_series(hist_mem, H, g=mem_hi)
        base_ret_lo = forecast_series(hist_ret, H, g=ret_lo); base_ret_hi = forecast_series(hist_ret, H, g=ret_hi)

        # cannibalization hits retail (learned a·exp(-d/L) for this region), phased over the 1st year
        nb_keyed = nb.set_index("site_key").loc[keys]
        dist_by_key = pd.Series(haversine_km(lat, lon, nb_keyed.lat.values, nb_keyed.lon.values), index=keys)
        rec_ret = (df[df.site_key.isin(keys)].sort_values("date").groupby("site_key").tail(12)
                   .groupby("site_key")["ret_wash_count"].mean())
        cannib_full = float((cm._cannib_ret(dist_by_key.reindex(rec_ret.index).values, cp) * rec_ret.values).sum())
        phase = np.minimum(1.0, np.arange(1, H + 1) / 12.0)

        with_fc = np.clip(base_mem + np.clip(base_ret - cannib_full * phase, 0, None) + new_traj, 0, None)
        with_lo = np.clip(base_mem_lo + np.clip(base_ret_lo - cannib_full * phase, 0, None) + new_lo, 0, None)
        with_hi = np.clip(base_mem_hi + np.clip(base_ret_hi - cannib_full * phase, 0, None) + new_hi, 0, None)

        market_forecast.update({
            "has_neighbours": True,
            "history": {
                "dates": [d.strftime("%Y-%m-%d") for d in hist.index],
                "values": [None if pd.isna(v) else float(v) for v in hist.values],
            },
            "forecast": {
                "dates": [d.strftime("%Y-%m-%d") for d in fdates],
                "with_new_site": [float(v) for v in with_fc],
                "without_new_site": [float(v) for v in base_fc],
                "band_lo": [float(v) for v in with_lo],
                "band_hi": [float(v) for v in with_hi],
                "new_entrant_journey": [float(v) for v in new_traj],
            },
            "net_change_year5": float(with_fc[-1] - base_fc[-1]),
        })
    else:
        market_forecast.update({
            "has_neighbours": False,
            "history": {"dates": [], "values": []},
            "forecast": {
                "dates": [d.strftime("%Y-%m-%d") for d in fdates],
                "with_new_site": [float(v) for v in new_traj],
                "without_new_site": [0.0] * H,
                "band_lo": [float(v) for v in new_lo],
                "band_hi": [float(v) for v in new_hi],
                "new_entrant_journey": [float(v) for v in new_traj],
            },
            "net_change_year5": float(new_traj[-1]),
        })

    return {
        "lat": lat, "lon": lon, "brand": brand,
        "summary": {
            "plateau_med": float(info["plateau_med"]),
            "plateau_lo": float(info["plateau_lo"]),
            "plateau_hi": float(info["plateau_hi"]),
            "mem_share": float(info["mem_share"]),
            "n_neighbours_20km": int(info["n_neighbours_20km"]),
            "brand_known": bool(info["brand_known"]),
            "ramp_source": info["ramp_source"],
            "mem_growth": float(mem_g + gm),
            "ret_growth": float(ret_g + gr),
        },
        "trajectory": trajectory,
        "market_forecast": market_forecast,
    }
