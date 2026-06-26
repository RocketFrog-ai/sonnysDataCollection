"""
Node 1 of the Key-Insights pipeline — `compute_metrics`.

PURE python, no LLM, no network. Turns one local market's monthly per-site panel into a rich,
structured set of investor KPIs the LLM narrates: levels & peaks, recent momentum, the new
entrant's ramp and its effect on incumbents, revenue mix, ASP momentum, AND the per-site data
coverage (start/end of each site's reporting) so the narrative never mistakes a reporting gap for
real demand decline.

Discipline (matches the rest of the app):
  • Compute at MONTHLY resolution, window-agnostic.
  • Counts and $ are SUMMED across the market (min_count=1 so a truly empty month stays NaN);
    ASPs are REVENUE-WEIGHTED ratios (Σrevenue ÷ Σunits), never a mean of per-site ASPs.
  • Every ratio goes through `_pct` via `.shift`/at-or-before lookups — NEVER `Series.pct_change()`
    (pandas 3.0 propagates NaN across gaps).
  • Trend wording uses the composition-robust estimators in `trend.py`.
  • Membership-ASP uses mem_revenue ÷ mem_purchase_count — the SAME definition the Streamlit chart
    draws — so the text matches the line.
  • A coverage-robust "same-panel YoY" (only sites reporting in BOTH periods) is reported alongside
    the raw market-sum YoY, so the narrative can separate true growth from sites entering/leaving.

Every numeric leaf is a plain float / int / None (NaN -> None) so the whole dict is json.dumps-able.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.pnl_analysis.modelling.trend import market_trend, robust_growth

_MIN_SIDE_MONTHS = 3     # need >=3 non-NaN months each side of an entry to call a before/after change
_RAMP_MIN_MONTHS = 18    # below this the focal history is too short for a robust ramp slope
_SMOOTH = 3              # months — de-spike window so a single spiky month is never called a "peak"


# ─────────────────────────── small numeric helpers ───────────────────────────
def _f(x: Any) -> Optional[float]:
    """JSON-safe float: None/NaN/inf -> None."""
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _pct(cur: Any, prev: Any) -> Optional[float]:
    """Guarded fractional change (cur-prev)/prev. None on any non-finite / zero base."""
    cur, prev = _f(cur), _f(prev)
    if cur is None or prev is None or prev == 0:
        return None
    return (cur - prev) / prev


def _at_or_before(s: pd.Series, ts: Optional[pd.Timestamp]) -> float:
    """Most recent finite value at-or-before `ts` (point-in-time lookup; tolerant of a missing month)."""
    if ts is None:
        return np.nan
    ss = s.loc[:ts].dropna()
    return float(ss.iloc[-1]) if len(ss) else np.nan


def _last_valid_idx(s: pd.Series) -> Optional[pd.Timestamp]:
    s = s.dropna()
    return s.index[-1] if len(s) else None


def _ym(ts) -> Optional[str]:
    return ts.strftime("%Y-%m") if ts is not None and pd.notna(ts) else None


def _smooth(s: pd.Series) -> pd.Series:
    """3-month rolling mean — removes single-month spikes so 'peak'/'current' reflect sustained levels."""
    return s.rolling(_SMOOTH, center=True, min_periods=1).mean()


def _wavg(s: pd.Series, end: Optional[pd.Timestamp], w: int = _SMOOTH) -> float:
    """Trailing w-month average ending at `end` (de-spiked, point-in-time)."""
    if end is None:
        return np.nan
    seg = s.loc[end - pd.DateOffset(months=w - 1):end].dropna()
    return float(seg.mean()) if len(seg) else np.nan


def _trend_annual(P: pd.DataFrame, col: str) -> Optional[float]:
    """Composition-robust annual growth of `col` across the market (immune to sites entering/leaving)."""
    piv = P.pivot_table(index="date", columns="site_key", values=col, aggfunc="sum")
    return _f(market_trend(piv)[0])


def _extremes(s: pd.Series) -> Dict[str, Any]:
    """Level summary of a monthly market series on a 3-month-AVERAGED basis (so a single spiky month is
    never reported as the peak): span, current (recent 3-mo avg), sustained peak/trough, drawdown."""
    sd = s.dropna()
    if sd.empty:
        return {"start": None, "end": None, "current": None, "peak": None, "peak_date": None,
                "trough": None, "trough_date": None, "current_vs_peak": None}
    sm = _smooth(sd)
    cur = sm.iloc[-1]
    pk_i, tr_i = sm.idxmax(), sm.idxmin()
    return {
        "start": _ym(sd.index[0]), "end": _ym(sd.index[-1]),
        "current": _f(cur),                                   # recent 3-mo average
        "peak": _f(sm.loc[pk_i]), "peak_date": _ym(pk_i),     # sustained (3-mo-avg) peak
        "trough": _f(sm.loc[tr_i]), "trough_date": _ym(tr_i),
        "current_vs_peak": _pct(cur, sm.loc[pk_i]),
    }


def _yearly(s: pd.Series) -> List[Dict[str, Any]]:
    """Compact yearly-average trajectory of a monthly series — actual data points for the LLM."""
    sd = s.dropna()
    if sd.empty:
        return []
    g = sd.groupby(sd.index.year).mean()
    return [{"year": int(y), "avg": _f(v)} for y, v in g.items()]


def _quarterly(s: pd.Series) -> List[Dict[str, Any]]:
    """Quarterly-average trajectory of a monthly series — finer data points (the shape read off the plot)."""
    sd = s.dropna()
    if sd.empty:
        return []
    g = sd.resample("QS").mean().dropna()
    return [{"q": f"{idx.year}Q{(idx.month - 1) // 3 + 1}", "avg": _f(v)} for idx, v in g.items()]


def _weighted_asp(frame: pd.DataFrame, rev_col: str, unit_col: str) -> float:
    """Revenue-weighted ASP over a set of rows: Σrev ÷ Σunits."""
    units = frame[unit_col].sum(min_count=1)
    rev = frame[rev_col].sum(min_count=1)
    if not np.isfinite(units) or units == 0 or not np.isfinite(rev):
        return np.nan
    return float(rev / units)


def deseason_pct_change(df: pd.DataFrame, site_key: str, metric: str, entry_date: pd.Timestamp,
                        pre=(-6, -1), post=(1, 12)) -> float:
    """Deseasonalized % change (post vs pre) for one site around an entry date. Ported from
    streamlits/app.py so the cannibalization cross-check matches the app. Returns a PERCENT (×100)."""
    s = df.loc[df.site_key == site_key].set_index("date")[metric].sort_index()
    if s.empty:
        return np.nan
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="MS"))
    moy = s.index.month
    factor = pd.Series(s.values, index=moy).groupby(level=0).transform("mean") / np.nanmean(s.values)
    des = pd.Series(s.values / factor.values, index=s.index)
    k = (des.index.year - entry_date.year) * 12 + (des.index.month - entry_date.month)
    o = pd.Series(des.values, index=k)
    a = o[(o.index >= pre[0]) & (o.index <= pre[1])].mean()
    b = o[(o.index >= post[0]) & (o.index <= post[1])].mean()
    if not np.isfinite(a) or a == 0:
        return np.nan
    return (b - a) / a * 100


# ─────────────────────────── per-series KPI block ───────────────────────────
def _series_block(M: pd.DataFrame, P: pd.DataFrame, col: str, last: pd.Timestamp,
                  win_start: pd.Timestamp, both_keys_now_prev) -> Dict[str, Any]:
    """Levels (current/peak/trough/start/drawdown, all 3-mo-averaged) + de-spiked YoY (3-mo avg vs the
    3-mo avg a year ago) + coverage-robust same-store YoY + robust annual trend + change over the window."""
    s = M[col]
    yoy = _pct(_wavg(s, last), _wavg(s, last - pd.DateOffset(months=12)))
    change_window = _pct(_wavg(s, last), _wavg(s, win_start))
    block = _extremes(s)
    block.update({
        "yoy": yoy,
        "yoy_same_store": _same_store_yoy(P, col, last, both_keys_now_prev),
        "trend_annual": _trend_annual(P, col),
        "change_last_n": change_window,
    })
    return block


def _same_store_yoy(P: pd.DataFrame, col: str, last: pd.Timestamp, both_keys, w: int = _SMOOTH) -> Optional[float]:
    """Same-store (like-for-like) YoY: trailing 3-mo average vs the 3-mo average a year earlier, using ONLY
    sites that reported in BOTH periods — strips the coverage artifact where the market sum drops just
    because incumbents stopped reporting."""
    if not both_keys:
        return None
    sub = P[P.site_key.isin(both_keys)]

    def avg(end):
        seg = sub[(sub.date > end - pd.DateOffset(months=w)) & (sub.date <= end)]
        return seg.groupby("date")[col].sum(min_count=1).mean()

    return _pct(avg(last), avg(last - pd.DateOffset(months=12)))


# ─────────────────────────── main entry ───────────────────────────
def compute_metrics(panel: pd.DataFrame, sites_meta: pd.DataFrame, focal_key: str,
                    last_n_months: int = 12) -> Dict[str, Any]:
    """Rich structured investor metrics for one local market.

    panel       — long monthly rows for the market sites (the Streamlit `sub`).
    sites_meta  — one row per market site: site_key, name (demo-safe), op_start, dist_km,
                  is_entrant, left_censored.
    focal_key   — the NEW entrant site (or nearest incumbent if the market has no entrant).
    """
    P = panel.copy()
    P["date"] = pd.to_datetime(P["date"])
    P = P[P["date"].notna()]
    meta = sites_meta.copy()

    sum_cols = ["mem_wash_count", "ret_wash_count", "tot_wash_count",
                "mem_revenue", "ret_revenue", "tot_revenue", "mem_purchase_count"]
    sum_cols = [c for c in sum_cols if c in P.columns]
    full_idx = pd.date_range(P["date"].min(), P["date"].max(), freq="MS")
    M = P.groupby("date")[sum_cols].sum(min_count=1).reindex(full_idx)
    M.index.name = "date"
    last = _last_valid_idx(M["tot_wash_count"]) or full_idx[-1]
    prev_year = last - pd.DateOffset(months=12)
    win_start = last - pd.DateOffset(months=max(last_n_months - 1, 0))

    # sites reporting at BOTH `last` and `last - 12mo` → coverage-robust comparisons
    keys_now = set(P.loc[P.date == last, "site_key"])
    keys_prev = set(P.loc[P.date == prev_year, "site_key"])
    both_keys = keys_now & keys_prev

    # ── focal / entrant context ──
    frow = meta[meta.site_key == focal_key]
    n_entrants = int(meta.is_entrant.sum()) if "is_entrant" in meta else 0
    has_entrant = n_entrants > 0
    focal_is_entrant = bool(frow.is_entrant.iloc[0]) if len(frow) and "is_entrant" in frow else False
    focal_left_censored = bool(frow.left_censored.iloc[0]) if len(frow) and "left_censored" in frow else False
    entry_date = pd.to_datetime(frow["op_start"].iloc[0]) if len(frow) else pd.NaT
    focal_name = (str(frow["name"].iloc[0]) if len(frow) and "name" in frow and pd.notna(frow["name"].iloc[0])
                  else "the new site")
    inc_keys = meta.loc[~meta.is_entrant, "site_key"].tolist() if "is_entrant" in meta else []
    has_clean_entry = bool(has_entrant and focal_is_entrant and pd.notna(entry_date) and not focal_left_censored)

    # ════════════════════════ COVERAGE (per-site reporting spans) ════════════════════════
    sites_cov: List[Dict[str, Any]] = []
    gsite = {k: g for k, g in P.groupby("site_key")}
    for _, r in meta.iterrows():
        g = gsite.get(r.site_key)
        first_obs = g.date.min() if g is not None and len(g) else pd.NaT
        last_obs = g.date.max() if g is not None and len(g) else pd.NaT
        avg12 = (g.loc[g.date >= last - pd.DateOffset(months=11), "tot_wash_count"].mean()
                 if g is not None and len(g) else np.nan)
        sites_cov.append({
            "name": str(r.get("name", r.site_key)),
            "op_start": _ym(pd.to_datetime(r.get("op_start"))) if pd.notna(r.get("op_start")) else None,
            "first_obs": _ym(first_obs), "last_obs": _ym(last_obs),
            "months": int(g.date.nunique()) if g is not None else 0,
            "is_entrant": bool(r.get("is_entrant", False)),
            "dist_km": _f(r.get("dist_km")),
            "active_recent": bool(pd.notna(last_obs) and last_obs >= last - pd.DateOffset(months=2)),
            "avg_washes_last12": _f(avg12),
        })
    active_now, active_year_ago = len(keys_now), len(keys_prev)
    coverage_note = None
    if active_now < active_year_ago:
        coverage_note = (f"{active_year_ago - active_now} of {len(meta)} sites stopped reporting in the last "
                         f"year (active sites {active_year_ago}→{active_now}); the raw market-sum YoY is "
                         f"depressed by this coverage drop — prefer the same-panel YoY.")
    coverage = {
        "market_start": _ym(full_idx[0]), "market_end": _ym(last),
        "n_sites": int(len(meta)), "active_now": active_now, "active_year_ago": active_year_ago,
        "sites": sites_cov, "note": coverage_note,
    }

    flags = {
        "single_site": len(meta) == 1,
        "no_entrant": not has_entrant,
        "short_history": len(full_idx) < last_n_months,
        "focal_left_censored": focal_left_censored,
        "coverage_drop": active_now < active_year_ago,
    }

    # ════════════════════════ WASHES ════════════════════════
    washes: Dict[str, Any] = {
        "total": _series_block(M, P, "tot_wash_count", last, win_start, both_keys),
        "retail": _series_block(M, P, "ret_wash_count", last, win_start, both_keys),
        "membership": _series_block(M, P, "mem_wash_count", last, win_start, both_keys),
        "trajectory_yearly": {
            "total": _yearly(M["tot_wash_count"]), "retail": _yearly(M["ret_wash_count"]),
            "membership": _yearly(M["mem_wash_count"]),
        },
    }
    share = (M["mem_wash_count"] / M["tot_wash_count"].replace(0, np.nan))
    share_now, share_yoy = _wavg(share, last), _wavg(share, prev_year)
    sh_s = _smooth(share.dropna())
    share_pk = sh_s.idxmax() if len(sh_s) else None
    washes["membership_share"] = {
        "current": _f(share_now),
        "yoy_delta_pp": _f((share_now - share_yoy) * 100) if np.isfinite(share_now) and np.isfinite(share_yoy) else None,
        "peak": _f(sh_s.loc[share_pk]) if share_pk is not None else None,
        "peak_date": _ym(share_pk),
    }

    washes["entry_effect"] = None
    if has_clean_entry:
        tot = M["tot_wash_count"]
        pre = tot.loc[entry_date - pd.DateOffset(months=12):entry_date - pd.DateOffset(months=1)].dropna()
        post = tot.loc[entry_date + pd.DateOffset(months=1):entry_date + pd.DateOffset(months=12)].dropna()
        if len(pre) >= _MIN_SIDE_MONTHS and len(post) >= _MIN_SIDE_MONTHS:
            washes["entry_effect"] = {
                "entry_date": _ym(entry_date),
                "pre_per_month": _f(pre.mean()), "post_per_month": _f(post.mean()),
                "change": _pct(post.mean(), pre.mean()),
                "months_pre": int(len(pre)), "months_post": int(len(post)),
            }

    washes["focal_ramp"] = None
    if has_clean_entry:
        f = (P[P.site_key == focal_key].groupby("date")["tot_wash_count"].sum(min_count=1).reindex(full_idx))
        arr = f.loc[entry_date:].dropna()
        if len(arr):
            first3 = arr.iloc[:3].mean()
            arr_s = _smooth(arr)
            pk_i = arr_s.idxmax()
            washes["focal_ramp"] = {
                "name": focal_name, "entry_date": _ym(entry_date),
                "first3_per_month": _f(first3), "current_per_month": _f(arr.iloc[-1]),
                "peak_per_month": _f(arr_s.loc[pk_i]), "peak_date": _ym(pk_i),
                "ramp": _pct(arr.iloc[-1], first3),
                "current_vs_peak": _pct(arr.iloc[-1], arr_s.loc[pk_i]),
                "growth_annual": _f(robust_growth(arr.to_numpy())[0]),
                "months_open": int(len(arr)),
                "short_history": bool(len(arr) < _RAMP_MIN_MONTHS),
            }

    washes["cannibalization"] = None
    if has_clean_entry and len(inc_keys) >= 2:
        block = {"entry_date": _ym(entry_date), "n_incumbents": int(len(inc_keys))}
        for label, col in (("retail", "ret_wash_count"), ("total", "tot_wash_count")):
            INC = (P[P.site_key.isin(inc_keys)].groupby("date")[col].sum(min_count=1).reindex(full_idx))
            pre = INC.loc[entry_date - pd.DateOffset(months=12):entry_date - pd.DateOffset(months=1)].dropna()
            post = INC.loc[entry_date + pd.DateOffset(months=1):entry_date + pd.DateOffset(months=12)].dropna()
            block[f"{label}_change"] = (_pct(post.mean(), pre.mean())
                                        if len(pre) >= _MIN_SIDE_MONTHS and len(post) >= _MIN_SIDE_MONTHS else None)
        des = [deseason_pct_change(P, k, "ret_wash_count", entry_date) for k in inc_keys]
        des = [d for d in des if np.isfinite(d)]
        block["retail_change_deseason_median"] = _f(np.median(des) / 100) if des else None
        washes["cannibalization"] = block

    # ════════════════════════ REVENUE ════════════════════════
    rev_total = _series_block(M, P, "tot_revenue", last, win_start, both_keys)
    rev_share_series = (M["mem_revenue"] / M["tot_revenue"].replace(0, np.nan))
    rs_now, rs_yoy = _wavg(rev_share_series, last), _wavg(rev_share_series, prev_year)
    rs_win = rev_share_series.loc[last - pd.DateOffset(months=23):last].dropna()
    slope_pp = None
    if len(rs_win) >= 6:
        t = (rs_win.index.year - rs_win.index[0].year) * 12 + (rs_win.index.month - rs_win.index[0].month)
        slope_pp = _f(np.polyfit(t.to_numpy(dtype=float), rs_win.to_numpy(dtype=float), 1)[0] * 12 * 100)
    revenue: Dict[str, Any] = {
        "total": rev_total,
        "mem_share": {
            "current": _f(rs_now),
            "retail_share_current": _f(1 - rs_now) if np.isfinite(rs_now) else None,
            "yoy_delta_pp": _f((rs_now - rs_yoy) * 100) if np.isfinite(rs_now) and np.isfinite(rs_yoy) else None,
            "slope_pp_per_yr": slope_pp,
        },
        "trajectory_yearly": {"total": _yearly(M["tot_revenue"]), "membership": _yearly(M["mem_revenue"]),
                              "retail": _yearly(M["ret_revenue"])},
        "focal_contribution": None,
    }
    if has_clean_entry:
        focal_rev = (P[P.site_key == focal_key].groupby("date")["tot_revenue"].sum(min_count=1).reindex(full_idx))
        fr_now = _at_or_before(focal_rev, last)
        mkt_now = _at_or_before(M["tot_revenue"], last)
        f12 = focal_rev.loc[last - pd.DateOffset(months=11):last].mean()
        m12 = M["tot_revenue"].loc[last - pd.DateOffset(months=11):last].mean()
        revenue["focal_contribution"] = {
            "name": focal_name,
            "focal_revenue_per_month": _f(fr_now),
            "share_of_market": (_f(fr_now / mkt_now) if np.isfinite(fr_now) and np.isfinite(mkt_now) and mkt_now else None),
            "share_of_market_last12": (_f(f12 / m12) if np.isfinite(f12) and np.isfinite(m12) and m12 else None),
        }

    # ════════════════════════ ASPs (revenue-weighted) ════════════════════════
    asp_ret_w = M["ret_revenue"] / M["ret_wash_count"].replace(0, np.nan)
    asp_mem_w = M["mem_revenue"] / M["mem_purchase_count"].replace(0, np.nan)

    def _asp_block(series: pd.Series, rev_col: str, unit_col: str) -> Dict[str, Any]:
        mom = _pct(_at_or_before(series, last), _at_or_before(series, last - pd.DateOffset(months=1)))
        last3 = M.loc[last - pd.DateOffset(months=2):last]
        prev3 = M.loc[last - pd.DateOffset(months=5):last - pd.DateOffset(months=3)]
        mom_smoothed = _pct(_weighted_asp(last3, rev_col, unit_col), _weighted_asp(prev3, rev_col, unit_col))
        block = _extremes(series)                                 # current/peak on a 3-mo-averaged basis
        block.update({"mom": mom, "mom_smoothed": mom_smoothed,
                      "yoy": _pct(_wavg(series, last), _wavg(series, prev_year)),     # de-spiked YoY
                      "change_last_n": _pct(_wavg(series, last), _wavg(series, win_start))})
        return block

    asps: Dict[str, Any] = {
        "retail": _asp_block(asp_ret_w, "ret_revenue", "ret_wash_count"),
        "membership": _asp_block(asp_mem_w, "mem_revenue", "mem_purchase_count"),
        "trajectory_yearly": {"retail": _yearly(asp_ret_w), "membership": _yearly(asp_mem_w)},
        "definitions": "MoM = latest month vs prior month; YoY = latest vs same month a year earlier (point-in-time).",
        "focal_gap": None,
    }
    if has_clean_entry and inc_keys:
        win = P[(P["date"] >= win_start) & (P["date"] <= last)]
        wf, wi = win[win.site_key == focal_key], win[win.site_key.isin(inc_keys)]
        f_ret, i_ret = _weighted_asp(wf, "ret_revenue", "ret_wash_count"), _weighted_asp(wi, "ret_revenue", "ret_wash_count")
        f_mem, i_mem = _weighted_asp(wf, "mem_revenue", "mem_purchase_count"), _weighted_asp(wi, "mem_revenue", "mem_purchase_count")
        asps["focal_gap"] = {
            "name": focal_name,
            "retail_focal": _f(f_ret), "retail_incumbent": _f(i_ret),
            "retail_gap": _f(f_ret - i_ret) if np.isfinite(f_ret) and np.isfinite(i_ret) else None,
            "retail_gap_pct": _pct(f_ret, i_ret),
            "membership_focal": _f(f_mem), "membership_incumbent": _f(i_mem),
            "membership_gap": _f(f_mem - i_mem) if np.isfinite(f_mem) and np.isfinite(i_mem) else None,
        }
    # membership pricing premium over retail (pricing power of the sticky product)
    ret_cur, mem_cur = asps["retail"].get("current"), asps["membership"].get("current")
    asps["membership_premium"] = {
        "abs": _f(mem_cur - ret_cur) if mem_cur is not None and ret_cur is not None else None,
        "ratio": _f(mem_cur / ret_cur) if mem_cur is not None and ret_cur else None,
    }

    # ════════════════════════ DATA POINTS (quarterly market series — the shape read off the plots) ════════════════════════
    data_points_quarterly = {
        "total_washes": _quarterly(M["tot_wash_count"]),
        "membership_washes": _quarterly(M["mem_wash_count"]),
        "retail_washes": _quarterly(M["ret_wash_count"]),
        "total_revenue": _quarterly(M["tot_revenue"]),
        "membership_share_washes": _quarterly(share),
        "asp_retail": _quarterly(asp_ret_w),
        "asp_membership": _quarterly(asp_mem_w),
    }

    # ════════════════════════ REVENUE — monetization add-ons ════════════════════════
    rev_now, wash_now = _at_or_before(M["tot_revenue"], last), _at_or_before(M["tot_wash_count"], last)
    rev_prev, wash_prev = _at_or_before(M["tot_revenue"], prev_year), _at_or_before(M["tot_wash_count"], prev_year)
    pw_now = (rev_now / wash_now) if np.isfinite(rev_now) and np.isfinite(wash_now) and wash_now else np.nan
    pw_prev = (rev_prev / wash_prev) if np.isfinite(rev_prev) and np.isfinite(wash_prev) and wash_prev else np.nan
    revenue["per_wash"] = {"current": _f(pw_now), "yoy": _pct(pw_now, pw_prev)}      # blended $/wash monetization
    revenue["mem_vs_ret_yoy"] = {                                                    # which stream is driving growth
        "membership": _pct(_at_or_before(M["mem_revenue"], last), _at_or_before(M["mem_revenue"], prev_year)),
        "retail": _pct(_at_or_before(M["ret_revenue"], last), _at_or_before(M["ret_revenue"], prev_year)),
    }

    # ════════════════════════ SITE SELECTION (the investor is choosing where to BUILD) ════════════════════════
    dist_vals = pd.to_numeric(meta["dist_km"], errors="coerce").dropna() if "dist_km" in meta else pd.Series([], dtype=float)
    playbook = None
    if has_clean_entry:
        fa = (P[P.site_key == focal_key].groupby("date")["tot_wash_count"].sum(min_count=1)
              .reindex(full_idx)).loc[entry_date:].dropna()
        if len(fa):
            fa_s = _smooth(fa)
            pk, first3 = fa_s.idxmax(), fa.iloc[:3].mean()
            playbook = {
                "name": focal_name, "months_open": int(len(fa)),
                "first3_per_month": _f(first3), "peak_per_month": _f(fa_s.loc[pk]),
                "months_to_peak": int((pk.year - entry_date.year) * 12 + (pk.month - entry_date.month)),
                "ramp_multiple": _f(fa.iloc[-1] / first3) if first3 else None,
                "current_per_month": _f(fa.iloc[-1]),
                "share_of_market_washes": _f(fa.iloc[-1] / wash_now) if np.isfinite(wash_now) and wash_now else None,
            }
    site_selection = {
        "sites_in_market": int(len(meta)),
        "nearest_site_km": _f(dist_vals.min()) if len(dist_vals) else None,
        "median_site_km": _f(dist_vals.median()) if len(dist_vals) else None,
        "washes_per_active_site": _f(wash_now / active_now) if active_now and np.isfinite(wash_now) else None,
        "revenue_per_active_site": _f(rev_now / active_now) if active_now and np.isfinite(rev_now) else None,
        "last_entrant_playbook": playbook,   # the best analog for what a NEW build here could achieve
    }

    market_meta = {
        "n_sites": int(len(meta)), "n_entrants": n_entrants, "has_entrant": has_entrant,
        "focal_name": focal_name, "focal_is_entrant": focal_is_entrant,
        "focal_op_start": _ym(entry_date),
        "market_start": _ym(full_idx[0]), "history_months": int(len(full_idx)),
        "last_month": _ym(last), "last_n_months": int(last_n_months),
    }

    return {"meta": market_meta, "coverage": coverage, "site_selection": site_selection,
            "data_points_quarterly": data_points_quarterly,
            "washes": washes, "revenue": revenue, "asps": asps, "flags": flags}
