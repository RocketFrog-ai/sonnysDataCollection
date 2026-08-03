"""
Section ③ — promotional campaigns: do they work, and who pays for the lift? The maths layer.

ONE copy of the math, imported by BOTH consumers so they can never disagree:
  • conclusion/demo/section_campaign.py         (Streamlit)
  • conclusion/notebook/book_v4_revolt.ipynb    (the working notebook this is ported from)

ONE input file: `proforma/data/opex/opex-data.csv` — the monthly P&L panel (client_id + site_id,
one row per site-month), carrying revenue, COGS, expenses, wash counts, ASP, member sign-ups and
lat/lon. Nothing is joined in from anywhere else, so this section shares no data with ① or ②.

The chain, in the order the notebook builds it:

  1. **Campaigns are inferred, not recorded.** There is no campaign table anywhere in this business,
     so a campaign is detected as an **OPEX spike**: a month where `cogs + expenses` exceeds that
     site's own trailing 6-month mean by 1.2x. Consecutive spike months (gap <= 1) merge into one
     campaign. That inference is the weakest link in the whole section and `robustness()` re-runs
     the estimate on a different trigger to show how much it matters.
  2. **Event studies** around the spike — the site itself, then every neighbour within 20 km, each
     normalised to its OWN pre-spike baseline so site size cannot drive the direction.
  3. **The counterfactual.** Everything in (2) measures a campaign against the site's own past,
     which is not a counterfactual: a site that opened last year grows anyway. `Counterfactual`
     rebuilds every estimate as a stacked difference-in-differences against matched sites that ran
     no campaign over the SAME calendar months, and runs the placebo tests that decide whether the
     estimate can be believed at all. It cannot: the raw numbers are mostly the opening ramp.

Nothing here is Streamlit-aware; nothing here prints. Functions return frames and dicts, and the
caller decides how to show them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATASET = REPO / "proforma" / "data" / "opex" / "opex-data.csv"

# --- event-study geometry -------------------------------------------------------------------------
# Everything is measured on the same window so the panels stack: 6 months of pre-period to form a
# baseline, 12 months after to watch it decay.
PRE_MONTHS, POST_MONTHS = 6, 12
SPIKE_THRESHOLD = 1.2          # x trailing 6-month OPEX
MIN_TRAILING = 4               # months of history needed before a spike can be detected
RADIUS_KM = 20.0               # what counts as "the local market"
MIN_EVENTS = 5                 # a month offset needs this many events before it is plotted
POST_WINDOW = (1, 3)           # the three months after a spike — the headline window

# US census region. Weather and the seasonal wash calendar are regional, so controls are matched
# on this before anything else.
STATE_TO_REGION = {
    "CT": "Northeast", "ME": "Northeast", "MA": "Northeast", "NH": "Northeast", "RI": "Northeast",
    "VT": "Northeast", "NJ": "Northeast", "NY": "Northeast", "PA": "Northeast",
    "IL": "Midwest", "IN": "Midwest", "MI": "Midwest", "OH": "Midwest", "WI": "Midwest",
    "IA": "Midwest", "KS": "Midwest", "MN": "Midwest", "MO": "Midwest", "NE": "Midwest",
    "ND": "Midwest", "SD": "Midwest",
    "DE": "South", "FL": "South", "GA": "South", "MD": "South", "NC": "South", "SC": "South",
    "VA": "South", "WV": "South", "DC": "South", "AL": "South", "KY": "South", "MS": "South",
    "TN": "South", "AR": "South", "LA": "South", "OK": "South", "TX": "South",
    "AZ": "West", "CO": "West", "ID": "West", "MT": "West", "NV": "West", "NM": "West",
    "UT": "West", "WY": "West", "AK": "West", "CA": "West", "HI": "West", "OR": "West",
    "WA": "West",
}

MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ==================================================================================================
# Load
# ==================================================================================================
def load(path: Path | str | None = None) -> pd.DataFrame:
    """The monthly P&L panel, with the three derived columns everything downstream assumes.

    `true_opex = cogs + expenses` is the repo's definition of what a site actually spends (see the
    opex-data memo); `total_income` is already a total, not a sum of parts.
    """
    d = pd.read_csv(path or DATASET)
    d["report_date"] = pd.to_datetime(d["report_date"])
    d["created_date"] = pd.to_datetime(d["created_date"])
    # the site key is client_id + site_id — site_id alone is a within-brand index and collides
    d["site_key"] = d["client_id"].astype(str) + "__" + d["site_id"].astype(str)
    d["true_opex"] = d["cogs"] + d["expenses"]
    d["total_washes"] = d["mem_wash_count"].fillna(0) + d["ret_wash_count"].fillna(0)
    d["region"] = d["state"].map(STATE_TO_REGION)
    return d


def site_frame(data: pd.DataFrame) -> pd.DataFrame:
    """One row per site — location and how long it has been reporting."""
    sp = data.groupby(
        ["site_key", "client_id", "client_name", "site_id", "location_name",
         "city", "state", "region"]
    ).agg(
        avg_monthly_revenue=("total_income", "mean"),
        first_report=("report_date", "min"),
        lat=("lat", "first"),
        lon=("lon", "first"),
    ).reset_index()
    max_date = data["report_date"].max()
    sp["months_of_data"] = (
        (max_date.year - sp["first_report"].dt.year) * 12
        + (max_date.month - sp["first_report"].dt.month) + 1
    )
    return sp


# ==================================================================================================
# 1 — the event: an OPEX spike
# ==================================================================================================
def detect_opex_spikes(data: pd.DataFrame, threshold: float = SPIKE_THRESHOLD,
                       min_trailing_months: int = MIN_TRAILING) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Months where a site's OPEX exceeds its own trailing 6-month mean by `threshold`x.

    The baseline is shifted one month so the spike month cannot inflate the bar it has to clear.
    """
    sub = data.sort_values(["site_key", "report_date"]).copy()
    sub["opex_baseline"] = (
        sub.groupby("site_key")["true_opex"]
        .transform(lambda s: s.shift(1).rolling(6, min_periods=min_trailing_months).mean())
    )
    sub["opex_vs_baseline"] = sub["true_opex"] / sub["opex_baseline"]
    sub["is_spike"] = sub["opex_vs_baseline"] > threshold
    return sub, sub[sub["is_spike"]].copy()


def build_spike_event_study(data: pd.DataFrame, spikes: pd.DataFrame,
                            pre_months: int = PRE_MONTHS,
                            post_months: int = POST_MONTHS) -> pd.DataFrame:
    """The focal site's own metrics around each spike, each normalised to its pre-spike mean.

    An event is dropped entirely if ANY tracked metric has a non-positive baseline — a site with
    no retail washes before the spike would otherwise divide by zero and dominate the median.
    """
    metrics = ["total_income", "true_opex", "ASP_mem", "ASP_ret",
               "total_washes", "mem_wash_count", "ret_wash_count"]
    by_site = {sk: g.sort_values("report_date") for sk, g in data.groupby("site_key")}

    records = []
    for _, spike in spikes.iterrows():
        ts = by_site.get(spike["site_key"])
        if ts is None:
            continue
        spike_date = spike["report_date"]
        ts = ts.copy()
        ts["months_from_spike"] = (
            (ts["report_date"].dt.year - spike_date.year) * 12
            + (ts["report_date"].dt.month - spike_date.month)
        )
        window = ts[(ts["months_from_spike"] >= -pre_months)
                    & (ts["months_from_spike"] <= post_months)]
        pre = window[window["months_from_spike"] < 0]
        baselines = {m: pre[m].mean() for m in metrics}
        if any(pd.isna(v) or v <= 0 for v in baselines.values()):
            continue
        for _, row in window.iterrows():
            rec = {"site_key": spike["site_key"], "spike_date": spike_date,
                   "months_from_spike": int(row["months_from_spike"])}
            for m in metrics:
                rec[f"{m}_norm"] = row[m] / baselines[m] if pd.notna(row[m]) else np.nan
            records.append(rec)
    return pd.DataFrame(records)


# ==================================================================================================
# 2 — the local market
# ==================================================================================================
def build_distance_matrix(site_pnl: pd.DataFrame, radius_km: float = RADIUS_KM) -> pd.DataFrame:
    """Directed site pairs within `radius_km`, straight-line (haversine) on lat/lon."""
    sites = (site_pnl.dropna(subset=["lat", "lon"])[["site_key", "lat", "lon"]]
             .drop_duplicates("site_key").reset_index(drop=True))
    lats, lons = np.radians(sites["lat"].values), np.radians(sites["lon"].values)
    dlat = lats[:, None] - lats[None, :]
    dlon = lons[:, None] - lons[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lats[:, None]) * np.cos(lats[None, :]) * np.sin(dlon / 2) ** 2
    dist = 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    np.fill_diagonal(dist, np.inf)
    fi, ni = np.where(dist <= radius_km)
    return pd.DataFrame({
        "focal_key": sites["site_key"].values[fi],
        "neighbor_key": sites["site_key"].values[ni],
        "distance_km": dist[fi, ni],
    })


def _neighbour_lookup(dist_df: pd.DataFrame) -> dict[str, list[tuple[str, float]]]:
    return {focal: list(zip(g["neighbor_key"], g["distance_km"]))
            for focal, g in dist_df.groupby("focal_key")}


def build_neighbor_event_study(data: pd.DataFrame, spikes: pd.DataFrame, dist_df: pd.DataFrame,
                               pre_months: int = PRE_MONTHS, post_months: int = POST_MONTHS,
                               min_pre_obs: int = 3) -> pd.DataFrame:
    """Every neighbour's metrics over the focal site's spike window.

    Each neighbour is normalised by ITS OWN pre-spike baseline, not the focal site's — otherwise a
    big neighbour next to a small promoter would look like it gained.
    """
    metrics = ["total_income", "total_washes", "mem_wash_count",
               "ret_wash_count", "ASP_mem", "ASP_ret"]
    revenue_metrics = {"total_income"}          # the only one that must have a positive baseline
    by_site = {sk: g.sort_values("report_date") for sk, g in data.groupby("site_key")}
    nbrs_of = _neighbour_lookup(dist_df)

    records, isolated = [], 0
    for _, spike in spikes.iterrows():
        focal_key, spike_date = spike["site_key"], spike["report_date"]
        nbrs = nbrs_of.get(focal_key, [])
        if not nbrs:
            isolated += 1
            continue
        for nbr_key, dist_km in nbrs:
            ts = by_site.get(nbr_key)
            if ts is None:
                continue
            ts = ts.copy()
            ts["months_from_spike"] = (
                (ts["report_date"].dt.year - spike_date.year) * 12
                + (ts["report_date"].dt.month - spike_date.month)
            )
            window = ts[(ts["months_from_spike"] >= -pre_months)
                        & (ts["months_from_spike"] <= post_months)]
            pre = window[window["months_from_spike"] < 0]
            if len(pre) < min_pre_obs:
                continue
            baselines, skip = {}, False
            for m in metrics:
                bv = pre[m].mean()
                if m in revenue_metrics and (pd.isna(bv) or bv <= 0):
                    skip = True
                    break
                baselines[m] = max(bv, 1) if (pd.isna(bv) or bv <= 0) else bv
            if skip:
                continue
            for _, row in window.iterrows():
                rec = {"focal_key": focal_key, "spike_date": spike_date,
                       "neighbor_key": nbr_key, "distance_km": dist_km,
                       "months_from_spike": int(row["months_from_spike"])}
                for m in metrics:
                    rv = row[m]
                    rec[f"{m}_norm"] = (rv / baselines[m]
                                        if pd.notna(rv) and baselines[m] > 0 else np.nan)
                records.append(rec)
    out = pd.DataFrame(records)
    out.attrs["isolated_events"] = isolated
    return out


def build_market_share_panel(data: pd.DataFrame, spikes: pd.DataFrame, dist_df: pd.DataFrame,
                             pre_months: int = PRE_MONTHS,
                             post_months: int = POST_MONTHS) -> pd.DataFrame:
    """The focal site's share of all washes done inside its own 20 km market, month by month.

    Share is the one metric that cannot rise for both sides at once, so it is what separates
    "the market grew" from "the market moved".
    """
    # index by absolute month (year*12 + month), not by timestamp — month offsets are calendar
    # arithmetic, and the panel's dates are month-END stamped so date maths would be fiddly
    d = data.assign(t=data["report_date"].dt.year * 12 + data["report_date"].dt.month)
    washes = d.pivot_table(index="t", columns="site_key", values="total_washes", aggfunc="sum")
    income = d.pivot_table(index="t", columns="site_key", values="total_income", aggfunc="sum")
    nbrs_of = _neighbour_lookup(dist_df)

    records = []
    for _, spike in spikes.iterrows():
        focal_key, spike_date = spike["site_key"], spike["report_date"]
        nbrs = nbrs_of.get(focal_key, [])
        if not nbrs or focal_key not in washes.columns:
            continue
        market = [sk for sk in [focal_key] + [s for s, _ in nbrs] if sk in washes.columns]
        anchor = spike_date.year * 12 + spike_date.month
        for mfs in range(-pre_months, post_months + 1):
            t = anchor + mfs
            if t not in washes.index:
                continue
            f_w, f_i = washes.at[t, focal_key], income.at[t, focal_key]
            if pd.isna(f_w) or pd.isna(f_i):
                continue
            mkt_w = float(washes.loc[t, market].fillna(0).sum())
            mkt_i = float(income.loc[t, market].fillna(0).sum())
            records.append({
                "focal_key": focal_key, "spike_date": spike_date, "months_from_spike": int(mfs),
                "focal_total_income": f_i, "focal_total_washes": f_w,
                "market_total_income": mkt_i, "market_total_washes": mkt_w,
                "focal_income_share": f_i / mkt_i if mkt_i > 0 else np.nan,
                "focal_wash_share": f_w / mkt_w if mkt_w > 0 else np.nan,
            })
    return pd.DataFrame(records)


def compute_spillover_stats(events_df: pd.DataFrame, neighbor_events_df: pd.DataFrame,
                            market_share_df: pd.DataFrame,
                            post_window: tuple[int, int] = POST_WINDOW) -> dict:
    """The headline spillover numbers, as medians of per-pair % changes.

    Median of per-event changes, never % change of the medians — so a 200-site operator and a
    single-site operator weigh the same.
    """
    lo, hi = post_window
    n_post = neighbor_events_df[neighbor_events_df["months_from_spike"].between(lo, hi)]
    f_post = events_df[events_df["months_from_spike"].between(lo, hi)]

    def nbr(col):
        return (n_post.groupby(["focal_key", "spike_date", "neighbor_key"])[col]
                .median().dropna())

    def focal(col):
        return f_post.groupby(["site_key", "spike_date"])[col].median().dropna()

    nbr_ret, nbr_mem = nbr("ret_wash_count_norm"), nbr("mem_wash_count_norm")
    nbr_tot, nbr_rev = nbr("total_washes_norm"), nbr("total_income_norm")
    f_ret, f_mem = focal("ret_wash_count_norm"), focal("mem_wash_count_norm")
    f_tot, f_rev = focal("total_washes_norm"), focal("total_income_norm")

    pre_ms = market_share_df[market_share_df["months_from_spike"].between(-PRE_MONTHS, -1)]
    post_ms = market_share_df[market_share_df["months_from_spike"].between(lo, hi)]
    share = pd.DataFrame({
        "pre": pre_ms.groupby(["focal_key", "spike_date"])["focal_wash_share"].mean(),
        "post": post_ms.groupby(["focal_key", "spike_date"])["focal_wash_share"].mean(),
    }).dropna()

    def pct(s):
        return float((s - 1).median() * 100)

    return {
        "n_neighbor_event_pairs": int(len(nbr_ret)),
        "n_neighbors": int(neighbor_events_df["neighbor_key"].nunique()) if len(neighbor_events_df) else 0,
        "n_focal_with_neighbors": int(neighbor_events_df["focal_key"].nunique()) if len(neighbor_events_df) else 0,
        "median_nbr_ret_wash_pct_change": pct(nbr_ret),
        "median_nbr_mem_wash_pct_change": pct(nbr_mem),
        "median_nbr_total_wash_pct_change": pct(nbr_tot),
        "median_nbr_revenue_pct_change": pct(nbr_rev),
        "pct_nbr_sites_revenue_decline": float((nbr_rev < 1).mean() * 100),
        "median_focal_ret_wash_pct_change": pct(f_ret),
        "median_focal_mem_wash_pct_change": pct(f_mem),
        "median_focal_total_wash_pct_change": pct(f_tot),
        "median_focal_revenue_pct_change": pct(f_rev),
        "median_focal_wash_share_pre": float(share["pre"].median() * 100),
        "median_focal_wash_share_post": float(share["post"].median() * 100),
        "share_gain_pp": float((share["post"] - share["pre"]).median() * 100),
        "n_market_share_events": int(len(share)),
    }


def event_curve(df: pd.DataFrame, col: str, group: str = "months_from_spike",
                min_events: int = MIN_EVENTS) -> pd.DataFrame:
    """Median + IQR of `col` at each month offset, dropping thin offsets."""
    agg = (df.groupby(group)[col]
           .agg(median="median", q25=lambda s: s.quantile(0.25),
                q75=lambda s: s.quantile(0.75), n="count")
           .reset_index())
    return agg[agg["n"] >= min_events].reset_index(drop=True)


def focal_state_mix(neighbor_events_df: pd.DataFrame, site_pnl: pd.DataFrame) -> pd.DataFrame:
    """Where the spillover evidence actually comes from — it is not evenly spread."""
    state = site_pnl.set_index("site_key")["state"].to_dict()
    d = neighbor_events_df.assign(focal_state=neighbor_events_df["focal_key"].map(state))
    out = (d.groupby("focal_state")
           .agg(focal_sites=("focal_key", "nunique"), rows=("focal_key", "size"))
           .sort_values("focal_sites", ascending=False).reset_index())
    out["share_of_sites"] = out["focal_sites"] / out["focal_sites"].sum() * 100
    return out


# ==================================================================================================
# 3 — campaigns: consecutive spike months, and what they cost
# ==================================================================================================
def cluster_campaigns(spikes: pd.DataFrame) -> pd.DataFrame:
    """Merge consecutive spike months (gap <= 1) into campaigns.

    Campaign spend is the sum of each month's OPEX ABOVE its own baseline — the incremental cost of
    the promotion, not the site's whole cost base.
    """
    records = []
    for site_key, grp in spikes.sort_values("report_date").groupby("site_key"):
        rows = grp.reset_index(drop=True)
        i = 0
        while i < len(rows):
            months = [rows.loc[i, "report_date"]]
            incr = [rows.loc[i, "true_opex"] - rows.loc[i, "opex_baseline"]]
            ratios = [rows.loc[i, "opex_vs_baseline"]]
            j = i + 1
            while j < len(rows):
                gap = ((rows.loc[j, "report_date"].year - rows.loc[j - 1, "report_date"].year) * 12
                       + (rows.loc[j, "report_date"].month - rows.loc[j - 1, "report_date"].month))
                if gap > 1:
                    break
                months.append(rows.loc[j, "report_date"])
                incr.append(rows.loc[j, "true_opex"] - rows.loc[j, "opex_baseline"])
                ratios.append(rows.loc[j, "opex_vs_baseline"])
                j += 1
            records.append({
                "site_key": site_key,
                "campaign_start": months[0], "campaign_end": months[-1],
                "duration_months": len(months),
                "total_incremental_opex": max(sum(incr), 0.0),
                "peak_opex_ratio": max(ratios), "n_spikes": len(months),
            })
            i = j
    c = pd.DataFrame(records)
    c["duration_bucket"] = c["duration_months"].apply(
        lambda d: "1 month" if d == 1 else ("2 months" if d == 2 else "3+ months"))
    return c


def campaign_spend_table(campaigns_df: pd.DataFrame) -> pd.DataFrame:
    """Incremental spend by campaign length, plus an All row."""
    order = ["1 month", "2 months", "3+ months"]
    rows = []
    total = len(campaigns_df)
    for label in order + ["All"]:
        sub = campaigns_df if label == "All" else campaigns_df[campaigns_df["duration_bucket"] == label]
        if not len(sub):
            continue
        s = sub["total_incremental_opex"]
        rows.append({"Campaign length": label, "Campaigns": len(sub),
                     "Share": len(sub) / total * 100,
                     "Mean spend": s.mean(), "p25": s.quantile(0.25),
                     "Median spend": s.median(), "p75": s.quantile(0.75)})
    return pd.DataFrame(rows)


def campaign_snapshot(data: pd.DataFrame, campaigns_df: pd.DataFrame,
                      lo: int = -3, hi: int = 6) -> pd.DataFrame:
    """Raw dollars around a campaign — one row per (campaign, month offset).

    Deliberately NOT normalised: the point of this panel is that a reader can see what a campaign
    costs and what comes back, in money.
    """
    by_site = {sk: g.sort_values("report_date") for sk, g in data.groupby("site_key")}
    records = []
    for _, camp in campaigns_df.iterrows():
        ts = by_site.get(camp["site_key"])
        if ts is None:
            continue
        anchor = camp["campaign_start"]
        ts = ts.copy()
        ts["mfs"] = ((ts["report_date"].dt.year - anchor.year) * 12
                     + (ts["report_date"].dt.month - anchor.month))
        for _, row in ts[(ts["mfs"] >= lo) & (ts["mfs"] <= hi)].iterrows():
            records.append({
                "bucket": camp["duration_bucket"], "duration": int(camp["duration_months"]),
                "mfs": int(row["mfs"]), "opex": row["true_opex"], "revenue": row["total_income"],
                "profit": row["total_income"] - row["true_opex"],
                "mem_purchases": row["mem_purchase_count"],
            })
    return pd.DataFrame(records)


# ==================================================================================================
# 4 — lifecycle: every site aligned on its own first month
# ==================================================================================================
def site_ramp(data: pd.DataFrame, dist_df: pd.DataFrame, max_age_months: int = 18,
              min_months_data: int = 6) -> pd.DataFrame:
    """Re-index every site to months since its own first report, so vintages are comparable."""
    washes = data.pivot_table(index="report_date", columns="site_key",
                              values="total_washes", aggfunc="sum")
    first = data.groupby("site_key")["report_date"].min().to_dict()
    nbrs_of = {focal: list(g["neighbor_key"]) for focal, g in dist_df.groupby("focal_key")}

    records = []
    for site_key, ts in data.groupby("site_key"):
        f = first[site_key]
        ts = ts.sort_values("report_date").copy()
        ts["age_months"] = ((ts["report_date"].dt.year - f.year) * 12
                            + (ts["report_date"].dt.month - f.month) + 1)
        ts = ts[(ts["age_months"] >= 1) & (ts["age_months"] <= max_age_months)]
        if len(ts) < min_months_data:
            continue
        nbrs = [n for n in nbrs_of.get(site_key, []) if n in washes.columns]
        for _, row in ts.iterrows():
            dt, f_w = row["report_date"], row["total_washes"]
            share = np.nan
            if nbrs and dt in washes.index:
                mkt = f_w + float(washes.loc[dt, nbrs].fillna(0).sum())
                share = f_w / mkt if mkt > 0 else np.nan
            records.append({
                "site_key": site_key, "age_months": int(row["age_months"]),
                "opex": row["true_opex"], "revenue": row["total_income"],
                "mem_wash_count": row["mem_wash_count"], "total_washes": f_w,
                "market_share": share, "has_neighbors": bool(nbrs),
            })
    return pd.DataFrame(records)


# ==================================================================================================
# 5 — the counterfactual
# ==================================================================================================
class Counterfactual:
    """Stacked difference-in-differences against sites that ran no campaign.

    For every campaign: take the treated site's log metric as a deviation from its OWN baseline
    months, take the same quantity averaged over matched control sites observed over the SAME
    calendar months, and subtract. The control curve is the estimate of "what would have happened
    anyway"; the gap is the campaign effect.

    Controls are matched on census region (same weather, same seasonal calendar) and on site age
    (+/- 6 months since opening), because a site still ramping up after opening is the other thing
    that makes revenue rise on its own. Baseline months are -6..-4, deliberately early, which
    leaves -3..-1 free to serve as the placebo window: a clean design must read zero there.

    Confidence intervals bootstrap over EVENTS, the unit of independence — resampling site-months
    would treat 18 rows from one campaign as 18 independent observations.
    """

    LO, HI = -6, 12
    BASE_K = [-6, -5, -4]
    MIN_CTRL = 5
    METRICS = ["total_income", "total_washes", "ret_wash_count",
               "mem_wash_count", "mem_purchase_count"]

    def __init__(self, data: pd.DataFrame, spikes: pd.DataFrame, seed: int = 0):
        d = data.copy()
        d["t"] = d["report_date"].dt.year * 12 + d["report_date"].dt.month
        self._d = d
        self.sites = sorted(d["site_key"].unique())
        self._si = {s: i for i, s in enumerate(self.sites)}
        self._tmin = int(d["t"].min())
        self._ncol = int(d["t"].max()) - self._tmin + 1

        self.mat: dict[str, np.ndarray] = {}
        for m in self.METRICS:
            M = np.full((len(self.sites), self._ncol), np.nan)
            for (sk, t), v in d.groupby(["site_key", "t"])[m].mean().items():
                if pd.notna(v) and v > 0:
                    M[self._si[sk], int(t) - self._tmin] = np.log(v)
            self.mat[m] = M

        self.region = d.drop_duplicates("site_key").set_index("site_key")["region"].to_dict()
        self.first = d.groupby("site_key")["t"].min().to_dict()
        self.spike_map = {sk: [x.year * 12 + x.month for x in dates] for sk, dates
                          in spikes.groupby("site_key")["report_date"].apply(list).items()}
        self._rng = np.random.default_rng(seed)

    # -- primitives --------------------------------------------------------------------------
    def series(self, site_key: str, anchor: int, metric: str, hi: int | None = None):
        """log(metric) at each month offset from `anchor`; NaN where unobserved."""
        hi = self.HI if hi is None else hi
        i = self._si.get(site_key)
        if i is None:
            return None
        a, b = anchor + self.LO - self._tmin, anchor + hi - self._tmin
        out = np.full(hi - self.LO + 1, np.nan)
        a_c, b_c = max(a, 0), min(b, self._ncol - 1)
        if a_c <= b_c:
            out[a_c - a: b_c - a + 1] = self.mat[metric][i, a_c:b_c + 1]
        return out

    def build(self, events: pd.DataFrame, metric: str = "total_income",
              spike_map: dict | None = None, shift: int = 0, hi: int | None = None,
              match_region: bool = True, match_age: bool = True) -> pd.DataFrame:
        """The DiD panel — one row per (event, month offset).

        `shift` fakes the campaign date (placebo-in-time). A faked anchor must itself be clean,
        otherwise the placebo would sit on top of a real campaign.
        """
        hi = self.HI if hi is None else hi
        spike_map = self.spike_map if spike_map is None else spike_map
        base_idx = [k - self.LO for k in self.BASE_K]

        def clean(sk, anchor):
            return not any(self.LO <= m - anchor <= hi for m in spike_map.get(sk, ()))

        rows = []
        for _, ev in events.iterrows():
            sk = ev["site_key"]
            if sk not in self._si:
                continue
            anchor = ev["campaign_start"].year * 12 + ev["campaign_start"].month + shift
            if shift and not clean(sk, anchor):
                continue

            pool = [s for s in self.sites if s != sk and clean(s, anchor)]
            if match_region:
                same = [s for s in pool if self.region.get(s) == self.region.get(sk)]
                if len(same) >= self.MIN_CTRL:
                    pool = same
            if match_age:
                age = anchor - self.first[sk]
                same = [s for s in pool if abs((anchor - self.first[s]) - age) <= 6]
                if len(same) < self.MIN_CTRL:
                    continue
                pool = same

            t = self.series(sk, anchor, metric, hi=hi)
            if t is None or np.isfinite(t[base_idx]).sum() < 2:
                continue
            t = t - np.nanmean(t[base_idx])

            ctrl = []
            for s in pool:
                c = self.series(s, anchor, metric, hi=hi)
                if c is None or np.isfinite(c[base_idx]).sum() < 2:
                    continue
                ctrl.append(c - np.nanmean(c[base_idx]))
            if len(ctrl) < self.MIN_CTRL:
                continue
            C = np.vstack(ctrl)
            ok = np.isfinite(C)
            n_ok = ok.sum(axis=0)
            c_mean = np.divide(np.where(ok, C, 0).sum(axis=0), n_ok,
                               out=np.full(C.shape[1], np.nan), where=n_ok > 0)

            for j, k in enumerate(range(self.LO, hi + 1)):
                if not np.isfinite(t[j]) or n_ok[j] < self.MIN_CTRL:
                    continue
                rows.append({"site_key": sk, "campaign_start": ev["campaign_start"],
                             "metric": metric, "k": k, "treated": float(t[j]),
                             "control": float(c_mean[j]), "did": float(t[j] - c_mean[j]),
                             "n_ctrl": int(n_ok[j])})
        return pd.DataFrame(rows)

    # -- statistics --------------------------------------------------------------------------
    def ci(self, values, n_boot: int = 4000):
        """Mean and 95% bootstrap CI, resampling events."""
        v = np.asarray(values, float)
        v = v[np.isfinite(v)]
        if len(v) < 3:
            return np.nan, np.nan, np.nan, len(v)
        bs = v[self._rng.integers(0, len(v), (n_boot, len(v)))].mean(axis=1)
        return v.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(v)

    def window(self, panel: pd.DataFrame, ks) -> dict:
        ev = panel[panel["k"].isin(list(ks))].groupby(["site_key", "campaign_start"])
        m, lo, hi, n = self.ci(ev["did"].mean().values)
        return {"naive": pct(ev["treated"].mean().mean()), "did": pct(m),
                "lo": pct(lo), "hi": pct(hi), "n": n, "sig": not (lo < 0 < hi)}

    def report(self, panel: pd.DataFrame, windows=None) -> pd.DataFrame:
        """One row per window: the naive number, the counterfactual number, and its interval."""
        windows = WINDOWS if windows is None else windows
        rows = []
        if not len(panel):
            return pd.DataFrame(rows)
        for name, ks in windows:
            r = self.window(panel, ks)
            if not r["n"]:
                continue
            rows.append({"Window": name, "Naive": r["naive"], "Counterfactual": r["did"],
                         "CI low": r["lo"], "CI high": r["hi"], "Events": r["n"],
                         "Significant": r["sig"]})
        return pd.DataFrame(rows)

    def path(self, panel: pd.DataFrame, min_events: int = 0) -> pd.DataFrame:
        """The event-study curve: treated, control and the gap, month by month, in %."""
        rows = []
        for k in sorted(panel["k"].unique()):
            s = panel[panel["k"] == k]
            m, lo, hi, n = self.ci(s.groupby(["site_key", "campaign_start"])["did"].mean().values)
            rows.append({"k": int(k), "treated": pct(s["treated"].mean()),
                         "control": pct(s["control"].mean()), "did": pct(m),
                         "lo": pct(lo), "hi": pct(hi), "n": n})
        out = pd.DataFrame(rows)
        if min_events and len(out):
            out.attrs["dropped"] = out[out["n"] < min_events]
            out = out[out["n"] >= min_events].reset_index(drop=True)
        return out

    def detrended(self, panel: pd.DataFrame) -> dict:
        """The effect net of the divergence already underway before the campaign started."""
        post = panel[panel["k"].between(1, 3)].groupby(["site_key", "campaign_start"])["did"].mean()
        pre = panel[panel["k"].between(-3, -1)].groupby(["site_key", "campaign_start"])["did"].mean()
        j = pd.concat([post.rename("post"), pre.rename("pre")], axis=1).dropna()
        m, lo, hi, n = self.ci((j["post"] - j["pre"]).values)
        return {"post": pct(post.mean()), "pre": pct(pre.mean()),
                "detrended": pct(m), "lo": pct(lo), "hi": pct(hi), "n": n}

    def age_of(self, campaigns_df: pd.DataFrame) -> pd.DataFrame:
        """Campaigns with the site's age (months of history) at campaign start."""
        c = campaigns_df.copy()
        c["age_months"] = [d.year * 12 + d.month - self.first.get(sk, 10 ** 6)
                           for sk, d in zip(c["site_key"], c["campaign_start"])]
        return c

    def censored_sites(self) -> int:
        """Sites already present in the panel's first month — for them 'age' is left-censored."""
        return sum(1 for sk in self.sites if self.first[sk] == self._tmin)


def pct(x):
    """log-difference -> percent."""
    return (np.exp(x) - 1) * 100 if np.isfinite(x) else np.nan


WINDOWS = [("pre-trend −3..−1 (placebo: should be ~0)", range(-3, 0)),
           ("effect +1..+3", range(1, 4)),
           ("effect +4..+6", range(4, 7))]

METRIC_WINDOWS = [("pre-trend −3..−1 (placebo)", range(-3, 0)),
                  ("effect +1..+3", range(1, 4)),
                  ("effect +4..+12", range(4, 13))]


# --- seasonality ----------------------------------------------------------------------------------
def seasonal_index(cf: Counterfactual) -> pd.DataFrame:
    """The month-of-year signature of revenue, on campaign-free site-months only.

    Log revenue minus the site's own average for that calendar year strips out site size and
    year-over-year growth, leaving the calendar. Campaign windows are excluded so the thing being
    measured is not contaminated by the thing being tested.
    """
    s = cf._d.copy()
    s["lrev"] = np.log(s["total_income"].where(s["total_income"] > 0))
    in_window = [any(cf.LO <= m - t <= cf.HI for m in cf.spike_map.get(sk, ()))
                 for sk, t in zip(s["site_key"], s["t"])]
    s = s[~np.array(in_window) & s["lrev"].notna()].copy()
    s["site_year"] = s["site_key"] + "_" + s["report_date"].dt.year.astype(str)
    s["resid"] = s["lrev"] - s.groupby("site_year")["lrev"].transform("mean")
    out = (s.groupby(s["report_date"].dt.month)["resid"].agg(["mean", "count"])
           .rename_axis("month").reset_index())
    out["pct_vs_site_year_avg"] = out["mean"].apply(pct)
    out["label"] = [MONTH_ABBR[m - 1] for m in out["month"]]
    return out


def deseasonalised_naive(cf: Counterfactual, campaigns_df: pd.DataFrame,
                         seasonal: pd.DataFrame) -> dict:
    """Recompute the SAME naive pre/post lift with the calendar removed.

    If seasonality were the whole story, this number would collapse. It does not.
    """
    month_ix = seasonal.set_index("month")["mean"].to_dict()
    d = cf._d
    lrev = np.log(d["total_income"].where(d["total_income"] > 0))
    cols = {"_raw": lrev, "_sa": lrev - d["report_date"].dt.month.map(month_ix)}
    for key, col in cols.items():
        M = np.full((len(cf.sites), cf._ncol), np.nan)
        for (sk, t), v in d.assign(_v=col).groupby(["site_key", "t"])["_v"].mean().items():
            if pd.notna(v):
                M[cf._si[sk], int(t) - cf._tmin] = v
        cf.mat[key] = M

    naive = {"_raw": [], "_sa": []}
    for _, ev in campaigns_df.iterrows():
        anchor = ev["campaign_start"].year * 12 + ev["campaign_start"].month
        for key in naive:
            s = cf.series(ev["site_key"], anchor, key)
            if s is None:
                continue
            b = s[[k - cf.LO for k in cf.BASE_K]]
            if np.isfinite(b).sum() < 2:
                continue
            post = s[1 - cf.LO: 4 - cf.LO]
            if np.isfinite(post).sum() == 0:
                continue
            naive[key].append(np.nanmean(post) - np.nanmean(b))
    raw, sa = pct(np.mean(naive["_raw"])), pct(np.mean(naive["_sa"]))
    amp = seasonal["pct_vs_site_year_avg"]
    return {"raw": raw, "deseasonalised": sa, "seasonality_pp": raw - sa,
            "n": len(naive["_raw"]), "swing_lo": amp.min(), "swing_hi": amp.max(),
            "peak_to_trough": amp.max() - amp.min()}


def campaign_start_months(campaigns_df: pd.DataFrame) -> pd.DataFrame:
    """When campaigns are launched — if they bunched before strong months, seasonality would bite."""
    v = campaigns_df["campaign_start"].dt.month.value_counts().reindex(range(1, 13), fill_value=0)
    return pd.DataFrame({"month": v.index, "label": [MONTH_ABBR[m - 1] for m in v.index],
                         "campaigns": v.values})


# --- robustness -----------------------------------------------------------------------------------
def expenses_only_events(data: pd.DataFrame, threshold: float = SPIKE_THRESHOLD):
    """Re-detect campaigns on FIXED expenses only, with COGS excluded.

    `true_opex = cogs + expenses`, and COGS is a variable cost that rises mechanically with wash
    volume — so a busy month can look like a campaign. This is the trigger's own control.
    """
    a = data.sort_values(["site_key", "report_date"]).copy()
    a["exp_baseline"] = (a.groupby("site_key")["expenses"]
                         .transform(lambda s: s.shift(1).rolling(6, min_periods=4).mean()))
    alt = a[a["expenses"] / a["exp_baseline"] > threshold]
    spike_map = {sk: [d.year * 12 + d.month for d in dates] for sk, dates
                 in alt.groupby("site_key")["report_date"].apply(list).items()}
    events = []
    for sk, g in alt.sort_values("report_date").groupby("site_key"):
        r = g.reset_index(drop=True)
        i = 0
        while i < len(r):
            j = i + 1
            while j < len(r) and ((r.loc[j, "report_date"].year - r.loc[j - 1, "report_date"].year) * 12
                                  + (r.loc[j, "report_date"].month - r.loc[j - 1, "report_date"].month)) <= 1:
                j += 1
            events.append({"site_key": sk, "campaign_start": r.loc[i, "report_date"]})
            i = j
    return pd.DataFrame(events), spike_map


def cogs_revenue_correlation(data: pd.DataFrame) -> float:
    """Within-site correlation between COGS and revenue — why the trigger check above matters."""
    long = data.groupby("site_key")["total_income"].transform("size") > 5
    cc = (data[long].groupby("site_key")[["cogs", "total_income"]].corr()
          .xs("cogs", level=1)["total_income"].dropna())
    return float(cc.median())


def age_sweep(cf: Counterfactual, campaigns_df: pd.DataFrame,
              thresholds=(0, 6, 12, 15, 18, 24)) -> pd.DataFrame:
    """Raise the minimum site age and watch contamination drain out — along with the sample.

    The right cutoff is the LOWEST one whose placebo passes. Below it the estimate is contaminated;
    above it there is nothing left to measure.
    """
    aged = cf.age_of(campaigns_df)
    rows = []
    for thresh in thresholds:
        sub = aged[aged["age_months"] >= thresh]
        panel = cf.build(sub)
        if panel.empty:
            continue
        pre, post = cf.window(panel, range(-3, 0)), cf.window(panel, range(1, 4))
        det = cf.detrended(panel)
        rows.append({"min_age": thresh, "campaigns": len(sub), "n_events": post["n"],
                     "pre_trend": pre["did"], "pre_lo": pre["lo"], "pre_hi": pre["hi"],
                     "placebo_passes": not pre["sig"],
                     "effect": post["did"], "eff_lo": post["lo"], "eff_hi": post["hi"],
                     "detrended": det["detrended"]})
    return pd.DataFrame(rows)


def near_vs_far(cf: Counterfactual, campaigns_df: pd.DataFrame,
                metrics=("ret_wash_count", "total_income"),
                near_km: float = 20.0, far_km: float = 100.0) -> pd.DataFrame:
    """The cannibalization claim, given a control group.

    Neighbours (<= 20 km) CAN be stolen from; sites 100+ km away cannot. Both are measured the same
    way over the same months, so whatever the far sites do is the market-wide trend. Only the
    difference is attributable to proximity.
    """
    geo = (cf._d.dropna(subset=["lat", "lon"]).drop_duplicates("site_key")
           [["site_key", "lat", "lon"]].reset_index(drop=True))
    la, lo_ = np.radians(geo["lat"].values), np.radians(geo["lon"].values)
    dlat, dlon = la[:, None] - la[None, :], lo_[:, None] - lo_[None, :]
    h = np.sin(dlat / 2) ** 2 + np.cos(la)[:, None] * np.cos(la)[None, :] * np.sin(dlon / 2) ** 2
    dist = 6371 * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
    keys = geo["site_key"].values
    gi = {k: i for i, k in enumerate(keys)}
    base_idx = [k - cf.LO for k in cf.BASE_K]

    rows = []
    for metric in metrics:
        near, far = [], []
        for _, ev in campaigns_df.iterrows():
            if ev["site_key"] not in gi:
                continue
            anchor = ev["campaign_start"].year * 12 + ev["campaign_start"].month
            d = dist[gi[ev["site_key"]]]
            for j, other in enumerate(keys):
                if other == ev["site_key"]:
                    continue
                if d[j] > near_km and d[j] < far_km:
                    continue
                if any(cf.LO <= m - anchor <= cf.HI for m in cf.spike_map.get(other, ())):
                    continue                      # the same exclusion applies to both groups
                s = cf.series(other, anchor, metric)
                if s is None or np.isfinite(s[base_idx]).sum() < 2:
                    continue
                s = s - np.nanmean(s[base_idx])
                post = s[1 - cf.LO: 4 - cf.LO]
                if not np.isfinite(post).any():
                    continue
                (near if d[j] <= near_km else far).append(np.nanmean(post))
        nm, nlo, nhi, nn = cf.ci(near)
        fm, flo, fhi, fn = cf.ci(far)
        rows.append({"metric": metric,
                     "near": pct(nm), "near_lo": pct(nlo), "near_hi": pct(nhi), "near_n": nn,
                     "far": pct(fm), "far_lo": pct(flo), "far_hi": pct(fhi), "far_n": fn,
                     "proximity": pct(nm - fm)})
    return pd.DataFrame(rows)


def roi_repricing(cf: Counterfactual, panel: pd.DataFrame,
                  campaigns_df: pd.DataFrame) -> pd.DataFrame:
    """Section-2 ROI arithmetic, but with incremental revenue measured against the counterfactual.

    Same divisor (campaign spend), different numerator: the gap vs matched controls instead of the
    gap vs the site's own past.
    """
    spend = campaigns_df.set_index(["site_key", "campaign_start"])["total_incremental_opex"]
    d = cf._d
    rows = []
    for (sk, cs), g in panel.groupby(["site_key", "campaign_start"]):
        anchor = cs.year * 12 + cs.month
        base = d[(d["site_key"] == sk) & (d["t"].between(anchor - 6, anchor - 4))]["total_income"]
        sp = spend.get((sk, cs), np.nan)
        g = g[g["k"] >= 0]
        if not len(base) or not np.isfinite(sp) or sp <= 0 or g.empty:
            continue
        bl = base.mean()
        if not np.isfinite(bl):
            continue
        rows.append({"site_key": sk, "campaign_start": cs, "spend": sp, "months": len(g),
                     "naive_incr": float((bl * (np.exp(g["treated"]) - 1)).sum()),
                     "did_incr": float((bl * (np.exp(g["did"]) - 1)).sum())})
    roi = pd.DataFrame(rows)
    if len(roi):
        roi["naive_roi"] = roi["naive_incr"] / roi["spend"]
        roi["did_roi"] = roi["did_incr"] / roi["spend"]
    return roi
