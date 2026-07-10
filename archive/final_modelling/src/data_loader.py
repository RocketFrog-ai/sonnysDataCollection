"""Load and clean monthly_kpi.csv, then build the structures the model needs.

Pipeline (mirrors notebooks/cluster_pattern.ipynb):
    raw CSV -> Monthly Recurring Plan subset -> site_uid + ym
            -> winsorize KPIs, drop flatline sites
            -> sites table (one row per site) + PANELS (site x month per KPI)

All heavy work is pure pandas/numpy so it is reusable by evaluate.py and app.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C


# --------------------------------------------------------------------------- #
# Load + clean
# --------------------------------------------------------------------------- #
def load_clean_df(path=C.DATA_PATH) -> pd.DataFrame:
    """Return the cleaned Monthly-Recurring-Plan long table.

    Columns guaranteed: client_id, site_id, site_uid, ym, latitude, longitude,
    state, region, and the 5 TARGET_KPIS.
    """
    raw = pd.read_csv(path)

    # Defensive: older files repeated the membership_wash_count header, so the
    # real asp_per_membership column loaded as `membership_wash_count.1`.
    raw = raw.rename(columns=C.LEGACY_RENAME)
    if "asp_per_membership" not in raw.columns and raw.shape[1] > 8:
        # last-resort positional fix if the name differs again
        cols = list(raw.columns)
        cols[8] = "asp_per_membership"
        raw.columns = cols

    raw["site_uid"] = raw["client_id"].astype(str) + "__" + raw["site_id"].astype(str)

    df = raw[raw["package_name"] == C.PACKAGE].copy()
    df["ym"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))

    for kpi in C.TARGET_KPIS:
        df[kpi] = pd.to_numeric(df[kpi], errors="coerce")

    dup = df.duplicated(subset=["site_uid", "ym"]).sum()
    assert dup == 0, f"Expected clean monthly granularity, found {dup} duplicate (site, month) rows"

    if C.WINSORIZE:
        df = _winsorize(df, C.TARGET_KPIS, C.WINSOR_PCT)

    if C.DROP_FLATLINE:
        df = _drop_flatline_sites(df, C.TARGET_KPIS)

    return df.reset_index(drop=True)


def _winsorize(df: pd.DataFrame, kpis, pct) -> pd.DataFrame:
    """Clip each KPI to its [lo, hi] percentile caps (computed on this subset)."""
    lo_q, hi_q = pct
    for kpi in kpis:
        lo, hi = df[kpi].quantile(lo_q), df[kpi].quantile(hi_q)
        df[kpi] = df[kpi].clip(lower=lo, upper=hi)
    return df


def _drop_flatline_sites(df: pd.DataFrame, kpis) -> pd.DataFrame:
    """Drop sites whose entire history is zero across ALL target KPIs.

    A site that is all-zero on a *single* KPI is kept (it may still be a useful
    neighbour for the other metrics); only sites that are dead on every metric
    are removed so they never drag a neighbour average toward zero.
    """
    per_site = df.groupby("site_uid")[kpis].sum()
    dead = per_site.index[(per_site == 0).all(axis=1)]
    return df[~df["site_uid"].isin(dead)]


# --------------------------------------------------------------------------- #
# Sites table + panels
# --------------------------------------------------------------------------- #
def build_sites_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per site: median coordinate + first-seen metadata + n_months."""
    sites = (
        df.groupby("site_uid")
        .agg(
            lat=("latitude", "median"),
            lon=("longitude", "median"),
            state=("state", "first"),
            region=("region", "first"),
            client=("client_id", "first"),
            n_months=("ym", "nunique"),
        )
        .reset_index()
    )
    return sites


def build_panels(df: pd.DataFrame, kpis=C.TARGET_KPIS):
    """Return (PANELS, month_axis).

    PANELS[kpi] is a (site_uid x month) DataFrame of KPI values over a shared,
    sorted monthly PeriodIndex. aggfunc='mean' is correct for the ASP ratios and
    a no-op for the others (one row per site-month).
    """
    month_axis = pd.PeriodIndex(
        pd.to_datetime(df["ym"]).dt.to_period("M").unique(), freq="M"
    ).sort_values()

    panels = {}
    for kpi in kpis:
        p = df.pivot_table(index="site_uid", columns="ym", values=kpi, aggfunc="mean")
        p.columns = pd.PeriodIndex(p.columns, freq="M")
        panels[kpi] = p.reindex(columns=month_axis)
    return panels, month_axis


def get_cohort(sites: pd.DataFrame, min_months=C.COHORT_MIN_MONTHS) -> set:
    """site_uids with >= min_months of history (drives validation)."""
    return set(sites.loc[sites["n_months"] >= min_months, "site_uid"])


# --------------------------------------------------------------------------- #
# Membership vs retail share panels
# --------------------------------------------------------------------------- #
def load_pct_panels(path=C.PCT_DATA_PATH, month_axis=None) -> dict:
    """Build site-month panels for the 4 membership/retail share KPIs.

    `monthly_withpackage.csv` is split one row per package_name, so the raw
    share columns are per-package (sales) or repeated (washes). We aggregate to
    a true site-month level and recompute each share from the underlying totals:

        membership_pct_sales = mem_sales / (mem_sales + retail_sales) * 100
        membership_pct_wash  = mem_wash  / (mem_wash  + retail_wash)  * 100

    with the retail_* share as the 100-complement. Returns PCT_KPIS -> panel
    (site_uid x month), reindexed onto `month_axis` when given so the columns
    line up with the rest of the model.
    """
    raw = pd.read_csv(path)
    raw["site_uid"] = raw["client_id"].astype(str) + "__" + raw["site_id"].astype(str)

    df = raw[raw["package_plan"] == C.PCT_PACKAGE_PLAN].copy()
    df["ym"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))

    # membership_sales_amount is per-package (summed); retail and wash counts are
    # site-level constants within a site-month (first is enough).
    for col in ("membership_sales_amount", "retail_sales_amount",
                "membership_wash_count", "retail_wash_count"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    g = (
        df.groupby(["site_uid", "ym"])
        .agg(
            mem_sales=("membership_sales_amount", "sum"),
            ret_sales=("retail_sales_amount", "first"),
            mem_wash=("membership_wash_count", "first"),
            ret_wash=("retail_wash_count", "first"),
        )
        .reset_index()
    )

    sales_tot = g["mem_sales"] + g["ret_sales"]
    wash_tot = g["mem_wash"] + g["ret_wash"]
    with np.errstate(invalid="ignore", divide="ignore"):
        g["membership_pct_sales"] = np.where(sales_tot > 0, g["mem_sales"] / sales_tot * 100, np.nan)
        g["membership_pct_wash"] = np.where(wash_tot > 0, g["mem_wash"] / wash_tot * 100, np.nan)
    g["retail_pct_sales"] = 100 - g["membership_pct_sales"]
    g["retail_pct_wash"] = 100 - g["membership_pct_wash"]

    panels = {}
    for kpi in C.PCT_KPIS:
        p = g.pivot_table(index="site_uid", columns="ym", values=kpi, aggfunc="mean")
        p.columns = pd.PeriodIndex(p.columns, freq="M")
        panels[kpi] = p.reindex(columns=month_axis) if month_axis is not None else p
    return panels


def load_pct_breakdowns(path=C.PCT_DATA_PATH) -> dict:
    """Per-site membership-vs-retail totals behind the share pies.

    Aggregated at the package_plan level (all Monthly-Recurring packages summed
    together), over all available months:
        wash  : DataFrame[site_uid] -> membership_wash, retail_wash counts
        sales : DataFrame[site_uid] -> membership_sales, retail_sales dollars

    Wash counts and retail sales are site-level (constant within a site-month),
    so they are taken once per month then summed. Membership sales are split per
    package, so they are summed across every package row.
    """
    raw = pd.read_csv(path)
    raw["site_uid"] = raw["client_id"].astype(str) + "__" + raw["site_id"].astype(str)

    df = raw[raw["package_plan"] == C.PCT_PACKAGE_PLAN].copy()
    df["ym"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
    for col in ("membership_sales_amount", "retail_sales_amount",
                "membership_wash_count", "retail_wash_count"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Membership sales: sum every package row per site.
    mem_sales = df.groupby("site_uid")["membership_sales_amount"].sum()

    # Site-level columns: take once per site-month, then sum across months.
    per_sm = (
        df.groupby(["site_uid", "ym"])
        .agg(
            mem_wash=("membership_wash_count", "first"),
            ret_wash=("retail_wash_count", "first"),
            ret_sales=("retail_sales_amount", "first"),
        )
        .reset_index()
    )
    wash = per_sm.groupby("site_uid")[["mem_wash", "ret_wash"]].sum()
    wash.columns = ["membership_wash", "retail_wash"]

    sales = per_sm.groupby("site_uid")[["ret_sales"]].sum()
    sales.columns = ["retail_sales"]
    sales["membership_sales"] = mem_sales
    sales = sales[["membership_sales", "retail_sales"]]

    return {"wash": wash, "sales": sales}


# --------------------------------------------------------------------------- #
# Convenience bundle
# --------------------------------------------------------------------------- #
@dataclass
class Dataset:
    df: pd.DataFrame
    sites: pd.DataFrame
    panels: dict
    month_axis: pd.PeriodIndex
    cohort: set
    pct_panels: dict = field(default_factory=dict)
    pct_breakdowns: dict = field(default_factory=dict)


def load_all(path=C.DATA_PATH) -> Dataset:
    """One-call bundle of every structure the model/app needs."""
    df = load_clean_df(path)
    sites = build_sites_table(df)
    panels, month_axis = build_panels(df)
    cohort = get_cohort(sites)
    try:
        pct_panels = load_pct_panels(month_axis=month_axis)
        pct_breakdowns = load_pct_breakdowns()
    except FileNotFoundError:
        pct_panels, pct_breakdowns = {}, {}
    return Dataset(
        df=df, sites=sites, panels=panels, month_axis=month_axis,
        cohort=cohort, pct_panels=pct_panels, pct_breakdowns=pct_breakdowns,
    )


if __name__ == "__main__":
    ds = load_all()
    print(f"rows               : {len(ds.df):,}")
    print(f"sites              : {ds.sites.site_uid.nunique():,}")
    print(f"months (month_axis): {len(ds.month_axis)}  {ds.month_axis.min()} .. {ds.month_axis.max()}")
    print(f"cohort (>=12 mo)   : {len(ds.cohort):,}")
    print(f"asp_per_membership : mean=${ds.df.asp_per_membership.mean():.2f} "
          f"(max=${ds.df.asp_per_membership.max():.2f} after winsorize)")
    print(f"panels             : {list(ds.panels)}")
