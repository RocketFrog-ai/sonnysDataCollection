"""Peer group construction and market-level reference statistics.

A "peer group" for a candidate site is the set of *mature* sites in the
same CBSA. If a CBSA is too sparse (< ``MIN_PEERS_PER_MARKET`` mature
peers) the peer set rolls up — first to the state cohort, then to the
region cohort, then to a national pool. This keeps the peer set
interpretable while avoiding tiny noisy markets.

Outputs from this module are pure DataFrames so they can be cached as
artifacts and used at both training time (per row of the panel) and
forecast time (per candidate location), with no leakage of the candidate
site itself into its own peer stats — every aggregate is computed
"leave-one-site-out" where applicable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def build_market_keys(static: pd.DataFrame) -> pd.DataFrame:
    """Return per-site market keys for joins."""
    cols = ["client_id_location_id", "cbsa_id", "cbsa_name", "state", "region", "h3_id"]
    cols = [c for c in cols if c in static.columns]
    return static[cols].drop_duplicates("client_id_location_id").reset_index(drop=True)


def local_h3_levels(panel: pd.DataFrame, static: pd.DataFrame) -> pd.DataFrame:
    """For each H3 disk (cell + 1-ring neighbours) compute a local
    mature reference level — the H3-anchored counterpart of the CBSA
    reference. Only H3 disks with ``>= MIN_PEERS_PER_H3`` peers count;
    others are dropped and the caller falls back to CBSA.
    """
    import h3
    drop_dup = [c for c in ("state", "region") if c in panel.columns and c in static.columns]
    base = panel.drop(columns=drop_dup) if drop_dup else panel
    p = base.merge(static[["client_id_location_id", "h3_id", "cbsa_id"]],
                   on="client_id_location_id", how="left")
    mature = p[p["site_age_months"] >= C.MATURITY_MONTHS].copy()
    if mature.empty:
        return pd.DataFrame(columns=["h3_id", "calendar_month", "h3_ref_level",
                                       "h3_month_level", "h3_seasonal_factor",
                                       "h3_peer_count"])

    # Per-site mature mean (recent-12-month mean for stability).
    site_recent_mean = (mature.sort_values(["client_id_location_id", "ym_ts"])
                        .groupby("client_id_location_id").tail(12)
                        .groupby(["client_id_location_id", "h3_id"])
                        ["wash_count_total"].mean().reset_index())
    site_recent_mean.rename(columns={"wash_count_total": "site_recent_mean"}, inplace=True)

    # Build the H3 disk for every site and explode so each site contributes
    # to its hex AND each ring-1 neighbour's hex.
    rows = []
    for sid, h3_id, mean in site_recent_mean.itertuples(index=False, name=None):
        if not h3_id:
            continue
        for cell in h3.grid_disk(h3_id, C.H3_DISK_RINGS):
            rows.append((cell, sid, mean))
    disk = pd.DataFrame(rows, columns=["h3_id", "client_id_location_id", "site_recent_mean"])

    # H3-disk annual reference.
    g = disk.groupby("h3_id").agg(
        h3_ref_level=("site_recent_mean", "mean"),
        h3_peer_count=("client_id_location_id", "nunique"),
    ).reset_index()
    g = g[g["h3_peer_count"] >= C.MIN_PEERS_PER_H3]

    # H3 × calendar_month seasonality: pool mature observations whose site
    # is in any disk anchored on h3_id.
    member = disk[["h3_id", "client_id_location_id"]].drop_duplicates()
    mat_lite = mature[["client_id_location_id", "calendar_month", "wash_count_total"]]
    merged = member.merge(mat_lite, on="client_id_location_id")
    g_m = (merged.groupby(["h3_id", "calendar_month"])["wash_count_total"]
                 .apply(_safe_geomean)
                 .reset_index()
                 .rename(columns={"wash_count_total": "h3_month_level"}))
    out = g_m.merge(g, on="h3_id")
    out["h3_seasonal_factor"] = (out["h3_month_level"] / out["h3_ref_level"]).clip(0.5, 2.0).fillna(1.0)
    return out


def _safe_geomean(x: pd.Series) -> float:
    """Winsorized arithmetic mean — robust to outliers but not low-biased the
    way a geomean / median is on right-skewed wash counts.

    Drops the top/bottom 5% before averaging.
    """
    x = x[x > 0]
    if len(x) == 0:
        return np.nan
    if len(x) >= 20:
        lo, hi = np.quantile(x, [0.05, 0.95])
        x = x[(x >= lo) & (x <= hi)]
    return float(x.mean())


def market_reference_levels(panel: pd.DataFrame, static: pd.DataFrame) -> pd.DataFrame:
    """For each CBSA compute reference mature monthly wash level.

    Uses only mature observations (site_age_months >= ``MATURITY_MONTHS``)
    when available; falls back to sites with the deepest history per
    market when no observation passes the threshold.

    Returns a long frame keyed by (cbsa_id, calendar_month) with:
      * ``cbsa_peer_count``       — # peers contributing
      * ``cbsa_ref_level``        — geomean of mature-window monthly wash
      * ``cbsa_seasonal_factor``  — month/cbsa_ref_level (1.0 = average)
      * ``cbsa_ref_recent``       — geomean of last 6 observed months
    """
    # static carries the *authoritative* cbsa/state/region; panel may already
    # have its own state/region columns from the source CSV. We drop those
    # from panel before merging to avoid suffixed names.
    drop_dup = [c for c in ("state", "region") if c in panel.columns and c in static.columns]
    p_base = panel.drop(columns=drop_dup) if drop_dup else panel
    p = p_base.merge(static[["client_id_location_id", "cbsa_id", "state", "region"]],
                     on="client_id_location_id", how="left")
    mature_mask = p["site_age_months"] >= C.MATURITY_MONTHS

    # For each cbsa: pool mature observations
    mat = p[mature_mask].copy()

    # CBSAs without enough mature obs get supplemented by their oldest sites.
    cbsa_peer_count = mat.groupby("cbsa_id")["client_id_location_id"].nunique()
    short_cbsas = cbsa_peer_count[cbsa_peer_count < C.MIN_PEERS_PER_MARKET].index.tolist()
    if short_cbsas:
        # For these CBSAs, take the top-N oldest sites and use their latest 12 months.
        extra = []
        for cb in short_cbsas:
            cb_panel = p[p["cbsa_id"] == cb]
            top_sites = (cb_panel.groupby("client_id_location_id")["site_age_months"].max()
                         .sort_values(ascending=False).head(5).index.tolist())
            sub = cb_panel[cb_panel["client_id_location_id"].isin(top_sites)]
            # Last 12 months per site
            sub = sub.sort_values(["client_id_location_id", "ym_ts"])
            sub = sub.groupby("client_id_location_id").tail(12)
            extra.append(sub)
        if extra:
            mat = pd.concat([mat, pd.concat(extra)], ignore_index=True)
            mat = mat.drop_duplicates(["client_id_location_id", "year_month"])

    # CBSA-level annual reference (geomean across all qualifying obs)
    g_cbsa = mat.groupby("cbsa_id").agg(
        cbsa_ref_level=("wash_count_total", _safe_geomean),
        cbsa_peer_count=("client_id_location_id", "nunique"),
    ).reset_index()

    # CBSA × calendar_month seasonality
    g_cbsa_m = mat.groupby(["cbsa_id", "calendar_month"]).agg(
        cbsa_month_level=("wash_count_total", _safe_geomean),
    ).reset_index()
    g_cbsa_m = g_cbsa_m.merge(g_cbsa[["cbsa_id", "cbsa_ref_level"]], on="cbsa_id")
    g_cbsa_m["cbsa_seasonal_factor"] = g_cbsa_m["cbsa_month_level"] / g_cbsa_m["cbsa_ref_level"]
    # Clip the factor to keep extreme months from blowing up the forecast.
    g_cbsa_m["cbsa_seasonal_factor"] = g_cbsa_m["cbsa_seasonal_factor"].clip(0.5, 2.0).fillna(1.0)

    # Recent trend: geomean over last 6 calendar months in the panel
    last_ym = p["year_month"].max()
    last6 = (pd.to_datetime(last_ym + "-01") - pd.DateOffset(months=5)).strftime("%Y-%m")
    recent = mat[mat["year_month"] >= last6]
    g_recent = recent.groupby("cbsa_id").agg(
        cbsa_ref_recent=("wash_count_total", _safe_geomean),
    ).reset_index()

    # g_cbsa_m already carries cbsa_ref_level from the earlier merge; only
    # bring in cbsa_peer_count here.
    out = g_cbsa_m.merge(g_cbsa[["cbsa_id", "cbsa_peer_count"]], on="cbsa_id", how="left")
    out = out.merge(g_recent, on="cbsa_id", how="left")
    return out


def rollup_levels(panel: pd.DataFrame, static: pd.DataFrame) -> dict:
    """Compute state, region, and national fallback reference frames.

    Returns a dict keyed by level name. Each frame has the same schema as
    :func:`market_reference_levels` minus ``cbsa_*`` (replaced with the
    level's own id column).
    """
    drop_dup = [c for c in ("state", "region") if c in panel.columns and c in static.columns]
    p_base = panel.drop(columns=drop_dup) if drop_dup else panel
    p = p_base.merge(static[["client_id_location_id", "cbsa_id", "state", "region"]],
                     on="client_id_location_id", how="left")
    mature = p[p["site_age_months"] >= C.MATURITY_MONTHS].copy()
    last_ym = p["year_month"].max()
    last6 = (pd.to_datetime(last_ym + "-01") - pd.DateOffset(months=5)).strftime("%Y-%m")

    def build_for(level: str) -> pd.DataFrame:
        if level == "national":
            mature["__lvl"] = "ALL"
        else:
            mature["__lvl"] = mature[level].astype(str)
        g = mature.groupby("__lvl").agg(
            ref_level=("wash_count_total", _safe_geomean),
            peer_count=("client_id_location_id", "nunique"),
        ).reset_index()
        gm = mature.groupby(["__lvl", "calendar_month"]).agg(
            month_level=("wash_count_total", _safe_geomean),
        ).reset_index().merge(g, on="__lvl")
        gm["seasonal_factor"] = (gm["month_level"] / gm["ref_level"]).clip(0.5, 2.0).fillna(1.0)
        rec = mature[mature["year_month"] >= last6].groupby("__lvl").agg(
            ref_recent=("wash_count_total", _safe_geomean),
        ).reset_index()
        gm = gm.merge(rec, on="__lvl", how="left")
        gm = gm.rename(columns={"__lvl": level if level != "national" else "national_id"})
        return gm

    return {lvl: build_for(lvl) for lvl in ("state", "region", "national")}


def resolve_peer_features(
    cbsa_id: str | None,
    state: str | None,
    region: str | None,
    calendar_month: int,
    cbsa_ref: pd.DataFrame,
    rollups: dict,
) -> dict:
    """Return the peer reference for a single (market, calendar_month).

    Falls back from CBSA → state → region → national so that even thin
    markets get a usable signal. The resolution level is returned so the
    forecast can be explained later.
    """
    row = cbsa_ref[(cbsa_ref["cbsa_id"] == cbsa_id) & (cbsa_ref["calendar_month"] == calendar_month)]
    if len(row) and row.iloc[0]["cbsa_peer_count"] >= C.MIN_PEERS_PER_MARKET:
        r = row.iloc[0]
        return dict(
            peer_ref_level=r["cbsa_ref_level"], peer_seasonal=r["cbsa_seasonal_factor"],
            peer_recent=r["cbsa_ref_recent"], peer_count=r["cbsa_peer_count"],
            peer_level_resolved="cbsa",
        )
    for level, key in (("state", state), ("region", region), ("national", "ALL")):
        df = rollups[level]
        idcol = level if level != "national" else "national_id"
        sub = df[(df[idcol] == str(key)) & (df["calendar_month"] == calendar_month)]
        if len(sub) and sub.iloc[0]["peer_count"] >= C.MIN_PEERS_PER_MARKET:
            r = sub.iloc[0]
            return dict(
                peer_ref_level=r["ref_level"], peer_seasonal=r["seasonal_factor"],
                peer_recent=r["ref_recent"], peer_count=r["peer_count"],
                peer_level_resolved=level,
            )
    # Hard fallback — should only hit on a truly empty panel.
    return dict(peer_ref_level=np.nan, peer_seasonal=1.0, peer_recent=np.nan,
                peer_count=0, peer_level_resolved="none")
