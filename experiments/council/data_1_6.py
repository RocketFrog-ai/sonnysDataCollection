"""
Single-source loader for the council — reads ONLY the council's own copy
`experiments/council/data/Council--historical-data.csv` (byte-identical to the shared stitched panel).

Replicates the panel transforms from `app/pnl_analysis/modelling/data.py::load_panel`
(date, op_start, site_key, ASP>$200 nulling, tot columns, left_censored, has_coords) but
points at the shared panel CSV and adds the per-site maturity fields the backtest needs
(`mat_n`, `mature_level`). Copies the 6-line `haversine_km` locally so this package never
imports app/pnl_analysis/modelling/data.py (which now imports proforma.models.coldstart directly).

Everything downstream (snapshot / scorer / council / harness) gets its data through here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

COUNCIL_DIR = Path(__file__).resolve().parent                 # <repo>/experiments/council
DATA_DIR = COUNCIL_DIR / "data"                               # the council's own copied datasets
# The council reads its OWN copy of the panel under experiments/council/data/. That file
# (`Council--historical-data.csv`) is byte-identical (same sha256, 13,269,792 bytes) to the shared
# `proforma/data/panel/main-data-v2-stitched.csv`, so the council reads exactly the same bytes while
# keeping every input inside its own folder (Dhruv's "council datasets only" constraint).
MAIN_CSV = DATA_DIR / "Council--historical-data.csv"
EARTH_KM = 6371.0088

# maturity window (months since a site's own opening) — matches the cold-start model's label window
MAT_LO, MAT_HI = 18, 30
LEFT_CENSOR_TS = pd.Timestamp("2020-01-01")   # op_start at/before this = reporting predates the data → not a true opening

_CACHE: Dict[str, Any] = {}


# ─────────────────────────── geometry (copied from data.py, not imported) ───────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    r = np.radians
    lat1, lon1, lat2, lon2 = r(lat1), r(lon1), r(lat2), r(lon2)
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ─────────────────────────── panel + per-site table ───────────────────────────
def load_panel_1_6() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Monthly panel `df` + per-site table `site`, from the shared stitched panel CSV. Cached for the process.

    `df` columns (superset of what compute_metrics needs): date, site_key, client_id, client_name,
    lat, lon, state, region, op_start, mem/ret wash+revenue, mem_purchase_count, tot_wash_count,
    tot_revenue, mem_share_wash, rel (months since the site's own op_start).

    `site` columns: site_key, client_id, client_name, lat, lon, state, region, op_start, first_obs,
    last_obs, n_obs, left_censored, has_coords, mat_n (# months in the 18–30 window), mature_level
    (mean tot_wash over that window; NaN if < 4 mature months).
    """
    if "df" in _CACHE:
        return _CACHE["df"], _CACHE["site"]

    # pandas' C CSV parser SEGFAULTS inside Streamlit's ScriptRunner thread with this numpy/pyarrow stack
    # (fine on the main thread, hence plain-python runs never crash). A pre-pickled cache read via
    # read_pickle sidesteps the C parser entirely. Built once from the CSV; regenerated if missing/stale.
    _pkl = MAIN_CSV.with_suffix(".pkl")
    if _pkl.exists():
        raw = pd.read_pickle(_pkl)
    else:
        raw = pd.read_csv(MAIN_CSV, low_memory=False)
        try:
            raw.to_pickle(_pkl)
        except Exception:
            pass
    raw["date"] = pd.to_datetime(dict(year=raw.year, month=raw.month, day=1))
    raw["op_start"] = pd.to_datetime(raw["operational_start"], format="%m-%Y", errors="coerce")
    raw["site_key"] = raw.client_id.astype(str) + "::" + raw.site_id.astype(str)

    df = raw.copy()
    # implausibly high per-wash ASP (>$200/wash) → null the revenue leg (mirrors load_panel)
    asp_r = np.where(df.ret_wash_count > 0, df.ret_revenue / df.ret_wash_count, np.nan)
    asp_m = np.where(df.mem_wash_count > 0, df.mem_revenue / df.mem_wash_count, np.nan)
    df.loc[asp_r > 200, "ret_revenue"] = np.nan
    df.loc[asp_m > 200, "mem_revenue"] = np.nan
    df["tot_wash_count"] = df.mem_wash_count + df.ret_wash_count
    df["tot_revenue"] = df[["mem_revenue", "ret_revenue"]].sum(axis=1, min_count=1)
    df["mem_share_wash"] = np.where(df.tot_wash_count > 0, df.mem_wash_count / df.tot_wash_count, np.nan)
    # months since the site's OWN opening (used for maturity windows everywhere)
    df["rel"] = (df.date.dt.year - df.op_start.dt.year) * 12 + (df.date.dt.month - df.op_start.dt.month)

    site = (
        df.groupby("site_key")
        .agg(client_id=("client_id", "first"), client_name=("client_name", "first"),
             lat=("lat", "first"), lon=("lon", "first"), state=("state", "first"),
             region=("region", "first"), op_start=("op_start", "first"),
             first_obs=("date", "min"), last_obs=("date", "max"), n_obs=("date", "size"))
        .reset_index()
    )
    site["left_censored"] = site.op_start <= LEFT_CENSOR_TS
    site["has_coords"] = site[["lat", "lon"]].notna().all(axis=1)

    # per-site maturity: mean total washes over the site's own months 18–30 (the level a mature site holds)
    mat = df[(df.rel >= MAT_LO) & (df.rel <= MAT_HI)]
    mat_g = mat.groupby("site_key")["tot_wash_count"]
    site["mat_n"] = site.site_key.map(mat_g.apply(lambda s: int(s.notna().sum()))).fillna(0).astype(int)
    mature_level = mat_g.mean()
    site["mature_level"] = site.site_key.map(mature_level)
    site.loc[site.mat_n < 4, "mature_level"] = np.nan   # too few mature months to trust the level

    # membership share in the mature window → an "express tunnel"-like proxy (the insight suite assumes
    # express economics; a pure-retail, low-volume wash is a different business the council can't size well).
    mem_mat = mat.groupby("site_key")["mem_wash_count"].sum()
    tot_mat = mat.groupby("site_key")["tot_wash_count"].sum()
    share = (mem_mat / tot_mat.replace(0, np.nan))
    site["mat_mem_share"] = site.site_key.map(share)
    site["express_like"] = (site.mat_mem_share >= 0.10) & (site.mature_level >= 3000)

    _CACHE["df"] = df
    _CACHE["site"] = site
    return df, site


def mature_floor(site: Optional[pd.DataFrame] = None) -> float:
    """Data-relative 'healthy mature site' floor = median mature_level across all sufficiently-matured sites.
    Used both to derive seat leans and to define a realized 'good build' in the scorer."""
    if "mature_floor" in _CACHE:
        return _CACHE["mature_floor"]
    if site is None:
        _, site = load_panel_1_6()
    floor = float(site.mature_level.dropna().median())
    _CACHE["mature_floor"] = floor
    return floor


# ─────────────────────────── neighbour geometry ───────────────────────────
def neighbours_within(site: pd.DataFrame, lat: float, lon: float, radius_km: float,
                      *, exclude_key: Optional[str] = None) -> pd.DataFrame:
    """Sites within `radius_km` of (lat, lon), with a `dist_km` column. Coord-valid only; optional
    self-exclusion by site_key. No time filter here — callers apply the as-of `op_start < T` cut."""
    s = site[site.has_coords].copy()
    s["dist_km"] = haversine_km(lat, lon, s.lat.values, s.lon.values)
    s = s[s.dist_km <= radius_km]
    if exclude_key is not None:
        s = s[s.site_key != exclude_key]
    return s.sort_values("dist_km").reset_index(drop=True)


# ─────────────────────────── backtest sample selection ───────────────────────────
def focal_candidates(*, radius_km: float = 20.0, min_mature_months: int = 4, min_neighbours: int = 2,
                     min_nbr_months: int = 6, min_year: int = 2021) -> pd.DataFrame:
    """The backtest sample: sites usable as a *focal* build test.

    A site qualifies when it is (a) not left-censored (a genuinely observed opening), (b) opened in
    `min_year` or later (so pre-T neighbours have ≥12 mo of context given the 2020-01 data start),
    (c) has ≥ `min_mature_months` months in its own 18–30 window (so a realized mature level exists),
    and (d) has ≥ `min_neighbours` OTHER sites that opened strictly before it within `radius_km` AND
    carry ≥ `min_nbr_months` of pre-T history — the SAME filter `build_snapshot` applies, so the count
    here never overstates the neighbours the snapshot will actually have (that mismatch left some sites
    with an empty pre-T panel).

    Returns the qualifying `site` rows plus `n_pre_neighbours` and `t_open` (= op_start). Deterministic.
    """
    df, site = load_panel_1_6()
    cand = site[(~site.left_censored) & (site.op_start.dt.year >= min_year)
                & (site.mat_n >= min_mature_months) & site.has_coords].copy()
    coord_site = site[site.has_coords]
    n_pre = []
    for _, r in cand.iterrows():
        d = haversine_km(r.lat, r.lon, coord_site.lat.values, coord_site.lon.values)
        pre = coord_site[(d <= radius_km) & (coord_site.site_key != r.site_key)
                         & (coord_site.op_start < r.op_start)]
        if len(pre):                                     # keep only neighbours with ≥ min_nbr_months pre-T rows
            obs = df[df.site_key.isin(pre.site_key) & (df.date < r.op_start)].groupby("site_key").size()
            n_pre.append(int((obs >= min_nbr_months).sum()))
        else:
            n_pre.append(0)
    cand["n_pre_neighbours"] = n_pre
    cand["t_open"] = cand.op_start   # the build date T (named t_open — `.T` is the DataFrame transpose)
    out = cand[cand.n_pre_neighbours >= min_neighbours].sort_values(["op_start", "site_key"]).reset_index(drop=True)
    return out
