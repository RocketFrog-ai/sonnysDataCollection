"""
General — the estate-wide picture that sits underneath every other section.

Nothing here is specific to tunnels or to the proforma backtest. It answers two questions that any
of the later sections would otherwise have to assume:

  • **How does a new car wash ramp up?** (the maths lives in `tunnel_data.ramp_*`, which reads the
    same monthly panel; it is surfaced in the General section because it is a fact about car washes
    rather than a fact about tunnel length)
  • **Where is the business actually done?** — every site we hold, on a map, by wash volume.

Input: `conclusion/data/historical_data_5yrs_monthly.csv` — the monthly wash panel, ~2,100 sites.
Nothing here is Streamlit-aware, so the notebook imports it unchanged.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PANEL = REPO / "conclusion" / "data" / "historical_data_5yrs_monthly.csv"

# Continental-US bounding box — a handful of rows carry 0/0 or otherwise impossible coordinates and
# would drag the map's centre into the Atlantic.
LAT_RANGE, LON_RANGE = (20.0, 50.0), (-130.0, -65.0)


def site_map() -> pd.DataFrame:
    """One row per site: total washes, location, state, and the years it traded.

    `washes_per_year` is what the map is drawn on — a site open for six years would otherwise
    dominate one open for two purely by having existed longer.
    """
    d = pd.read_csv(PANEL, low_memory=False)
    d["site_key"] = d.client_id.astype(str) + "___" + d.site_id.astype(str)
    d["washes"] = d.mem_wash_count.fillna(0) + d.ret_wash_count.fillna(0)
    d["revenue"] = d.mem_revenue.fillna(0) + d.ret_revenue.fillna(0)
    d = d[d.washes > 0]

    g = (d.groupby("site_key")
           .agg(washes=("washes", "sum"), revenue=("revenue", "sum"),
                months=("washes", "size"), lat=("lat", "first"), lon=("lon", "first"),
                state=("state", "first"), operator=("client_name", "first"),
                city=("address1", "first"))
           .reset_index())
    g = g[g.lat.between(*LAT_RANGE) & g.lon.between(*LON_RANGE)]
    g["years"] = g.months / 12
    g["washes_per_year"] = g.washes / g.years
    g["washes_per_month"] = g.washes / g.months
    return g.sort_values("washes_per_year", ascending=False).reset_index(drop=True)


def state_totals(min_sites: int = 5) -> pd.DataFrame:
    """Per-state roll-up, ranked by the TYPICAL site rather than the state total.

    A state total mostly measures how many sites we happen to own there, not how strong the market
    is — we are not the only operator in any of these states. The median site is the comparable
    figure: it says what a wash in that state actually does.

    `min_sites` drops states too small to have a meaningful median (a 4-site state would otherwise
    top the ranking on noise). Dropped states are still returned, flagged `enough_sites=False`, so
    the caller can show them greyed rather than silently losing them.
    """
    s = site_map()
    out = (s.groupby("state")
             .agg(sites=("site_key", "size"),
                  median_site=("washes_per_year", "median"),
                  p25_site=("washes_per_year", lambda x: x.quantile(.25)),
                  p75_site=("washes_per_year", lambda x: x.quantile(.75)),
                  washes_per_year=("washes_per_year", "sum"))
             .reset_index())
    out["enough_sites"] = out.sites >= min_sites
    return out.sort_values("median_site", ascending=False).reset_index(drop=True)


# =================================================================================================
# Membership vs retail
# =================================================================================================

def _site_years(min_months: int = 6) -> pd.DataFrame:
    """One row per site × calendar year: membership and retail washes, and the split between them.

    `min_months` drops part-years. A site that opened in November contributes two months to that
    year, and those two months are not a fair read of its mix — a brand-new wash sells memberships
    hard in its first weeks, so a part-year would swing the state it sits in.
    """
    d = pd.read_csv(PANEL, low_memory=False)
    d["site_key"] = d.client_id.astype(str) + "___" + d.site_id.astype(str)
    d["mem"] = d.mem_wash_count.fillna(0)
    d["ret"] = d.ret_wash_count.fillna(0)

    g = (d.groupby(["site_key", "year"])
           .agg(mem=("mem", "sum"), ret=("ret", "sum"), months=("mem", "size"),
                state=("state", "first"), operator=("client_name", "first"))
           .reset_index())
    g = g[(g.months >= min_months) & ((g.mem + g.ret) > 0)]
    g["washes"] = g.mem + g.ret
    g["mem_share"] = g.mem / g.washes
    return g


def mix_by_year(min_months: int = 6) -> pd.DataFrame:
    """The membership/retail split by calendar year, on two bases that disagree for a reason.

    `share` pools every site trading that year. The panel triples in size over the window, so that
    line moves partly because *different sites* are in it — newer sites sell memberships harder.

    `share_same_sites` uses only the sites present in **every** year of the window, so the mix is
    the only thing that can change. Where the two lines diverge, the aggregate is measuring the
    intake, not the customer.
    """
    g = _site_years(min_months)
    years = sorted(g.year.unique())
    per_year = len(years)
    keep = g.groupby("site_key").year.nunique()
    same = g[g.site_key.isin(keep[keep == per_year].index)]

    def roll(frame: pd.DataFrame) -> pd.DataFrame:
        return (frame.groupby("year")
                     .agg(mem=("mem", "sum"), ret=("ret", "sum"), sites=("site_key", "nunique"))
                     .reset_index())

    out = roll(g).rename(columns={"sites": "sites_all"})
    out["washes"] = out.mem + out.ret
    out["share"] = out.mem / out.washes
    s = roll(same).rename(columns={"mem": "mem_s", "ret": "ret_s", "sites": "sites_same"})
    out = out.merge(s, on="year", how="left")
    out["share_same_sites"] = out.mem_s / (out.mem_s + out.ret_s)
    out.attrs["n_same_sites"] = int(same.site_key.nunique())
    out.attrs["years"] = years
    return out


def mix_by_state_year(min_sites: int = 4, min_months: int = 6) -> pd.DataFrame:
    """Membership share per state per year — the grid the heat map is drawn on.

    A state-year with fewer than `min_sites` sites is left blank rather than plotted: with two
    sites the "state" is one operator's habit, and operators differ far more than states do
    (§④ puts the operator at 39% of the variance and the trade area at none).
    """
    g = _site_years(min_months)
    out = (g.groupby(["state", "year"])
             .agg(mem=("mem", "sum"), ret=("ret", "sum"), sites=("site_key", "nunique"))
             .reset_index())
    out["share"] = out.mem / (out.mem + out.ret)
    out["washes"] = out.mem + out.ret
    return out[out.sites >= min_sites].reset_index(drop=True)


def mix_headline(min_months: int = 6) -> dict:
    """First year vs last year, on both bases, plus which states moved most."""
    y = mix_by_year(min_months)
    if y.empty:
        return {}
    first, last = y.iloc[0], y.iloc[-1]
    st_ = mix_by_state_year(min_months=min_months)
    moves = []
    for s, gg in st_.groupby("state"):
        gg = gg.sort_values("year")
        if len(gg) >= 2:
            moves.append(dict(state=s, first=float(gg.share.iloc[0]), last=float(gg.share.iloc[-1]),
                              change=float(gg.share.iloc[-1] - gg.share.iloc[0]),
                              sites=int(gg.sites.iloc[-1])))
    mv = pd.DataFrame(moves).sort_values("change", ascending=False)
    latest = st_[st_.year == st_.year.max()].sort_values("share", ascending=False)
    return dict(
        first_year=int(first.year), last_year=int(last.year),
        first_share=float(first.share), last_share=float(last.share),
        first_same=float(first.share_same_sites), last_same=float(last.share_same_sites),
        n_same_sites=int(y.attrs["n_same_sites"]),
        top_state=str(latest.state.iloc[0]) if len(latest) else "—",
        top_share=float(latest.share.iloc[0]) if len(latest) else np.nan,
        bottom_state=str(latest.state.iloc[-1]) if len(latest) else "—",
        bottom_share=float(latest.share.iloc[-1]) if len(latest) else np.nan,
        riser=str(mv.state.iloc[0]) if len(mv) else "—",
        riser_change=float(mv.change.iloc[0]) if len(mv) else np.nan,
        faller=str(mv.state.iloc[-1]) if len(mv) else "—",
        faller_change=float(mv.change.iloc[-1]) if len(mv) else np.nan,
        n_states=int(latest.state.nunique()),
    )


def map_headline(min_sites: int = 5) -> dict:
    """Headline numbers — all per-site, never state totals."""
    s, st = site_map(), state_totals(min_sites)
    ranked = st[st.enough_sites]
    return dict(
        n_sites=int(len(s)), n_states=int(st.state.nunique()),
        median_site=float(s.washes_per_year.median()),
        p25_site=float(s.washes_per_year.quantile(.25)),
        p75_site=float(s.washes_per_year.quantile(.75)),
        busiest_state=str(ranked.state.iloc[0]),
        busiest_state_site=float(ranked.median_site.iloc[0]),
        busiest_state_n=int(ranked.sites.iloc[0]),
        quietest_state=str(ranked.state.iloc[-1]),
        quietest_state_site=float(ranked.median_site.iloc[-1]),
        most_sites_state=str(st.sort_values("sites", ascending=False).state.iloc[0]),
        most_sites_n=int(st.sites.max()),
        most_sites_median=float(st.sort_values("sites", ascending=False).median_site.iloc[0]),
        spread=float(ranked.median_site.iloc[0] / ranked.median_site.iloc[-1]),
        n_ranked=int(len(ranked)),
    )
