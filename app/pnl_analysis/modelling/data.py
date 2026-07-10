"""
Shared, cached data loaders for the pnl_analysis backend (Explore-markets + Forecast).

Every modelling module (market / pnl / campaign) loads its data through here so the heavy
artifacts are read once and shared in-process:

  • load_panel()         — the monthly site panel (proforma/data/panel/main-data-v2-stitched.csv)
                            + a per-site table with adaptive local-market clusters. The single
                            source of truth for "the sites". (This docstring used to say
                            main-ds.csv; that is the superseded legacy schema and is not read here.)
  • load_model()         — the cold-start LightGBM artifacts (plateau + ramp + cannibalization).
  • load_pnl_annual()    — per-(location, state, year) operating P&L from opex-data.csv.
  • load_pnl_monthly()   — per-(location, state, year, month) opex + monthly wash snapshot (with age).
  • load_campaign_panel()— opex-data.csv keyed by site_key (for campaign-spike detection / snapshots).

These mirror the loaders in proforma/ui/app.py (PNL_EXCLUDE, the ASP>200 nulling, the
adaptive cluster assignment), so the API returns the same numbers as the app. "Mirror" is literal:
this is a second implementation of the same math, and the two have drifted. See docs/DIVERGENCES.md
before changing either.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from proforma.models import coldstart as cm

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "proforma" / "data"
MAIN_CSV = DATA / "panel" / "main-data-v2-stitched.csv"
OPEX_CSV = DATA / "opex" / "opex-data.csv"
TYPES_CSV = DATA / "ref" / "site_carwash_types.csv"
EARTH_KM = 6371.0088

# sites kept OUT of the P&L analysis (matched on client_id) — mirrors the UI's PNL_EXCLUDE
PNL_EXCLUDE = {"alpinecarwash_000087"}

# "express only" mode — mirrors proforma/ui/panels/_shared.py exactly.
EXPRESS_TYPE = "Express Tunnel"   # the filter keeps just this primary_carwash_type
EXPRESS_MIN_MONTHS = 30           # express mode also requires >=30 monthly records -> richer history

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
def load_carwash_types() -> pd.Series:
    """site_key -> primary_carwash_type. Mirrors `_shared.load_carwash_types()` in the UI."""
    if "types" not in _CACHE:
        t = pd.read_csv(TYPES_CSV, low_memory=False)
        t["site_key"] = t.client_id.astype(str) + "::" + t.site_id.astype(str)
        t = t.dropna(subset=["primary_carwash_type"]).drop_duplicates("site_key", keep="first")
        _CACHE["types"] = t.set_index("site_key").primary_carwash_type
    return _CACHE["types"]


def load_panel(express_only: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Monthly site-level panel `df` + per-site table `site` (with adaptive local-market clusters).

    Mirrors `load_data()` in proforma/ui/panels/_shared.py: builds tot/share columns, nulls
    implausible ASP (>$200/wash), aggregates one row per site_key, tags left-censored / has_coords,
    and assigns the adopted adaptive clusters. Cached per `express_only` for the process lifetime.

    express_only=True keeps only `EXPRESS_TYPE` sites with >= `EXPRESS_MIN_MONTHS` monthly records,
    then re-clusters that subset — so neighbours, the cluster gate and the level anchor all see
    express sites only. Same two filters, in the same order, as the UI.

    When express_only=False this function is byte-for-byte what it always was: the `carwash_type` /
    `is_express` columns are attached ONLY in express mode, so the default panel's column set is
    unchanged and no downstream consumer sees a new field. (The UI attaches them unconditionally;
    that asymmetry is deliberate here, to keep the default API response identical.)
    """
    ckey = "df_express" if express_only else "df"
    skey = "site_express" if express_only else "site"
    if ckey in _CACHE:
        return _CACHE[ckey], _CACHE[skey]

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

    # express only: drop Flex / full-service / etc. up front, so the market, cluster and level-anchor
    # views all operate on express sites. Mirrors _shared.load_data.
    types = None
    if express_only:
        types = load_carwash_types()
        df["carwash_type"] = df.site_key.map(types)
        df = df[df.carwash_type == EXPRESS_TYPE].copy()

    site = (
        df.groupby("site_key")
        .agg(client_id=("client_id", "first"), client_name=("client_name", "first"),
             lat=("lat", "first"), lon=("lon", "first"),
             state=("state", "first"), region=("region", "first"), op_start=("op_start", "first"),
             first_obs=("date", "min"), last_obs=("date", "max"), n_obs=("date", "size"))
        .reset_index()
    )
    if express_only:
        site["carwash_type"] = site.site_key.map(types)
        site["is_express"] = site.carwash_type.eq(EXPRESS_TYPE)
    site["left_censored"] = site.op_start <= pd.Timestamp("2020-01-01")
    site["has_coords"] = site[["lat", "lon"]].notna().all(axis=1)

    # express only: keep just the richer-history sites (>=30 monthly records) BEFORE clustering, so
    # series, clusters and forecasts are well-grounded. Order matters: filter, then cluster.
    if express_only:
        rich_keys = set(site.site_key[site.n_obs >= EXPRESS_MIN_MONTHS])
        site = site[site.site_key.isin(rich_keys)].reset_index(drop=True)
        df = df[df.site_key.isin(rich_keys)].copy()

    site["cluster"] = cm.assign_clusters(site, "adaptive")

    _CACHE[ckey] = df
    _CACHE[skey] = site
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
         .agg(opex=("total_expenses", "sum"), income=("total_income", "sum"),
              mem_wash=("mem_wash_count", "first"),
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
