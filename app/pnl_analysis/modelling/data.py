"""
Shared, cached data loaders for the pnl_analysis backend (Explore-markets + Forecast).

Every modelling module (market / pnl / campaign) loads its data through here so the heavy
artifacts are read once and shared in-process:

  • load_panel()         — the monthly site panel (main-ds.csv) + a per-site table with adaptive
                            local-market clusters. The single source of truth for "the sites".
  • load_model()         — the cold-start LightGBM artifacts (plateau + ramp + cannibalization).
  • load_pnl_annual()    — per-(location, state, year) operating P&L from opex-data.csv.
  • load_pnl_monthly()   — per-(location, state, year, month) opex + monthly wash snapshot (with age).
  • load_campaign_panel()— opex-data.csv keyed by site_key (for campaign-spike detection / snapshots).

These mirror the loaders in earnest-proforma-2.0/streamlits/app.py exactly (PNL_EXCLUDE, the
ASP>200 nulling, the adaptive cluster assignment), so the API returns the same numbers as the app.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PROFORMA = ROOT / "earnest-proforma-2.0"
COLDSTART_DIR = PROFORMA / "streamlits"
MAIN_CSV = PROFORMA / "data" / "main-ds.csv"
OPEX_CSV = PROFORMA / "data" / "opex-data.csv"
EARTH_KM = 6371.0088

# sites kept OUT of the P&L analysis (matched on client_id) — mirrors app.py PNL_EXCLUDE
PNL_EXCLUDE = {"alpinecarwash_000087"}

# import the cold-start model module (lives alongside the Streamlit app)
if str(COLDSTART_DIR) not in sys.path:
    sys.path.insert(0, str(COLDSTART_DIR))
import coldstart_model as cm  # noqa: E402

# Explore-markets metric label <-> dataframe column
METRICS: Dict[str, str] = {
    "Total washes": "tot_wash_count",
    "Membership washes": "mem_wash_count",
    "Retail washes": "ret_wash_count",
    "Total revenue ($)": "tot_revenue",
    "Membership share of washes": "mem_share_wash",
}
METRIC_LABEL_BY_COL = {v: k for k, v in METRICS.items()}

_CACHE: Dict[str, Any] = {}


# ─────────────────────────── geometry ───────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    r = np.radians
    lat1, lon1, lat2, lon2 = r(lat1), r(lon1), r(lat2), r(lon2)
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ─────────────────────────── monthly panel (main-ds) ───────────────────────────
def load_panel() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Monthly site-level panel `df` + per-site table `site` (with adaptive local-market clusters).

    Mirrors `load_data()` in streamlits/app.py: builds tot/share columns, nulls implausible ASP
    (>$200/wash), aggregates one row per site_key, tags left-censored / has_coords, and assigns the
    adopted adaptive clusters. Cached for the process lifetime.
    """
    if "df" in _CACHE:
        return _CACHE["df"], _CACHE["site"]

    raw = pd.read_csv(MAIN_CSV, low_memory=False)
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
        .agg(client_id=("client_id", "first"), client_name=("client_name", "first"),
             lat=("lat", "first"), lon=("lon", "first"),
             state=("state", "first"), region=("region", "first"), op_start=("op_start", "first"),
             first_obs=("date", "min"), last_obs=("date", "max"), n_obs=("date", "size"))
        .reset_index()
    )
    site["left_censored"] = site.op_start <= pd.Timestamp("2020-01-01")
    site["has_coords"] = site[["lat", "lon"]].notna().all(axis=1)
    site["cluster"] = cm.assign_clusters(site, "adaptive")

    _CACHE["df"] = df
    _CACHE["site"] = site
    return df, site


def load_model() -> Dict[str, Any]:
    """Cold-start artifacts (LightGBM plateau/share models + ramp curves + learned cannibalization)."""
    if "art" not in _CACHE:
        _CACHE["art"] = cm.load()
    return _CACHE["art"]


def state_to_region(art: Dict[str, Any]) -> Dict[str, Any]:
    """state -> modal region map, derived from the model's site table (used to scope P&L stats)."""
    if "s2r" not in _CACHE:
        _CACHE["s2r"] = (art["sites_rl"].dropna(subset=["state"]).groupby("state").region
                         .agg(lambda x: x.mode().iloc[0] if len(x.mode()) else None).to_dict())
    return _CACHE["s2r"]


# ─────────────────────────── operating P&L (opex-data) ───────────────────────────
def load_pnl_annual() -> pd.DataFrame:
    """Per-(location, state, year) annual operating P&L. Sums the sub-monthly report rows into an
    annual opex / income per site; keeps near-full years (>=11 months) in 2022–2025. Mirrors app.load_pnl()."""
    if "pnl_annual" in _CACHE:
        return _CACHE["pnl_annual"]
    p = pd.read_csv(OPEX_CSV, low_memory=False)
    p = p[~p.client_id.astype(str).isin(PNL_EXCLUDE)]
    g = (p.groupby(["location_name", "state", "year"])
         .agg(opex=("total_expenses", "sum"), income=("total_income", "sum"), cogs=("cogs", "sum"),
              months=("month", "nunique"), asp_mem=("ASP_mem", "median"), asp_ret=("ASP_ret", "median"),
              lat=("lat", "first"), lon=("lon", "first"))
         .reset_index())
    out = g[(g.months >= 11) & (g.year.between(2022, 2025))].copy()
    _CACHE["pnl_annual"] = out
    return out


def load_pnl_monthly() -> pd.DataFrame:
    """Per-(location, state, year, month) monthly P&L: opex = sum of sub-monthly report rows; washes =
    the monthly snapshot. Adds `age` (months since the site's first P&L row) and total `wash`. Mirrors
    app.load_pnl_monthly()."""
    if "pnl_monthly" in _CACHE:
        return _CACHE["pnl_monthly"]
    p = pd.read_csv(OPEX_CSV, low_memory=False)
    p = p[~p.client_id.astype(str).isin(PNL_EXCLUDE)]
    m = (p.groupby(["location_name", "state", "year", "month"])
         .agg(opex=("total_expenses", "sum"), mem_wash=("mem_wash_count", "first"),
              ret_wash=("ret_wash_count", "first"), lat=("lat", "first"), lon=("lon", "first")).reset_index())
    m = m[m.year.between(2022, 2025)].copy()
    m["date"] = pd.to_datetime(dict(year=m.year, month=m.month, day=1))
    first = m.groupby("location_name").date.transform("min")
    m["age"] = (m.date.dt.year - first.dt.year) * 12 + (m.date.dt.month - first.dt.month)
    m["wash"] = m.mem_wash.fillna(0) + m.ret_wash.fillna(0)
    _CACHE["pnl_monthly"] = m
    return m


def load_campaign_panel() -> pd.DataFrame:
    """opex-data.csv keyed by site_key (client_id::site_id) — the raw rows for campaign-spike detection
    and the book_v4 OPEX/Revenue/Profit/Membership snapshot. Mirrors app._campaign_data()."""
    if "campaign_panel" in _CACHE:
        return _CACHE["campaign_panel"]
    d = pd.read_csv(OPEX_CSV, low_memory=False)
    d["site_key"] = d.client_id.astype(str) + "::" + d.site_id.astype(str)
    _CACHE["campaign_panel"] = d
    return d
