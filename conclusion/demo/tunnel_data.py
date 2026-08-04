"""
Section ① — tunnel length, wash trajectories and year-5 volume. The shared data + analysis layer.

ONE copy of the math, imported by BOTH consumers so they can never disagree:
  • conclusion/demo/section_tunnel.py        (Streamlit)
  • conclusion/notebook/conclusions.ipynb    (static plots / working)

ONE input file: `conclusion/data/tunnel_length_with_wash.csv` — 78 sites, each with a measured built
tunnel length, its measured peak hourly throughput, and its annual wash counts. Nothing is joined in.

The work that matters here is turning raw calendar-year wash counts into a comparable **year-5**
number for every site:

  • the file's wash columns are CALENDAR years, but sites open mid-year, so the first and last
    columns are part-years — each one is scaled up by the fraction of the year the site was
    actually open and inside the data window;
  • that gives an **operating-year** series (year 1 = first 12 months open), which is what makes
    two sites of different vintage comparable;
  • sites younger than five years are carried to year 5 on a **maturity ramp measured from the
    sites that do have five years**, and that ramp is validated by holdout (see `validation()`).

Nothing here is Streamlit-aware, so the notebook imports it unchanged.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATASET = REPO / "conclusion" / "data" / "tunnel_length_with_wash.csv"

# --- units --------------------------------------------------------------------------------------
# `tunnel_length_actual_ft` is the measurement in whole metres × 3.2. 3.2 is a rounded conversion;
# the real factor is 3.28084. Verified: dividing by 3.2 gives exact integers 13-60.
FILE_FT_PER_M = 3.2
FT_PER_M = 3.28084

# --- data window ----------------------------------------------------------------------------------
# The extract ends mid-2026. Derived from the data rather than assumed: across 34 mature sites that
# traded through both years, 2026 washes are a median 0.539 of 2025 washes (IQR 0.52-0.56), i.e.
# ~6.5 months of coverage.
CUTOFF = pd.Timestamp("2026-07-16")
FIRST_YEAR, LAST_YEAR = 2020, 2026
MIN_COVERAGE = 0.25   # a year seen for less than 3 months is too thin to annualise

# --- the maturity ramp ----------------------------------------------------------------------------
# Annualised wash rate in operating year N as a share of the site's year-5 rate, fitted on the sites
# that actually reached year 5 (see `fit_ramp()` / `validation()`). Sites are effectively mature from
# year 3.
RAMP = {1: 0.67, 2: 0.88, 3: 1.00, 4: 1.00, 5: 1.00}

# --- the sizing rule ------------------------------------------------------------------------------
# The proforma sizes a tunnel as `tunnel length in feet = year-5 peak hourly volume`: one car per
# hour per foot of tunnel.
CARS_PER_HOUR_PER_FT = 1.0
MIN_VIABLE_FT = 13 * FT_PER_M

# Every operating day gives the site ONE peak figure. A site with 1,600 trading days therefore has
# 1,600 daily peaks, and the four columns below are percentiles across that distribution: the median
# day, the p75 day, the p90 day, and the single highest day on record. So "highest daily peak" means
# one day out of ~1,600 — not a typical day, and not a clock-hour claim.
PEAK_BASIS = {
    "Median daily peak": "median_daily_peak_volume_cars",
    "p75 daily peak": "p75_daily_peak_volume_cars",
    "p90 daily peak": "p90_daily_peak_volume_cars",
    "Highest daily peak": "max_daily_peak_volume_cars",
}
DEFAULT_BASIS = "Highest daily peak"
PEAK_ORDER = list(PEAK_BASIS)

BAND_OVERBUILT = 0.50
BAND_TIGHT = 0.80

# How much history a site has. Sites with only one or two years of trading, and sites that stopped
# reporting before the end of the window, are DROPPED everywhere — a year-5 number carried up from a
# single part-year, or from a site that went dark, is extrapolation rather than evidence. Everything
# below (the panel, the ramp, its validation, the build) sees only the analysed set.
ALL_MATURITY = ["5+ years observed", "3–4 years observed", "1–2 years observed",
                "Stopped reporting"]
MATURITY_ORDER = ["5+ years observed", "3–4 years observed"]


# =================================================================================================
# loading
# =================================================================================================
def load_raw() -> pd.DataFrame:
    d = pd.read_csv(DATASET)
    d["opened"] = pd.to_datetime(d.created_date)
    d["tunnel_m"] = (d.tunnel_length_actual_ft / FILE_FT_PER_M).round().astype(int)
    d["tunnel_ft"] = d.tunnel_m * FT_PER_M
    d["site"] = d.name.fillna(d.site_key)
    d["where"] = d.city.fillna("—").astype(str) + ", " + d.state.fillna("—").astype(str)
    return d


def operating_panel(d: pd.DataFrame | None = None) -> pd.DataFrame:
    """Long panel for the analysed sites: one row per site × calendar year, in operating years.

    `rate` is the annualised run-rate — the calendar year's washes scaled up by the fraction of that
    year the site was open and inside the data window. Without this a site that opened in October
    looks like a catastrophe in its first column.

    Sites outside `MATURITY_ORDER` (1–2 years of history, or stopped reporting) are removed here, so
    every consumer — ramp, validation, cohort curve, build — works off the same set.
    """
    p = full_panel(d)
    keep = site_summary(p).query("maturity in @MATURITY_ORDER").index
    return p[p.site_key.isin(keep)].reset_index(drop=True)


def full_panel(d: pd.DataFrame | None = None) -> pd.DataFrame:
    """The same panel *before* the maturity filter — needed only to classify maturity itself."""
    d = load_raw() if d is None else d
    rows = []
    for r in d.itertuples():
        for y in range(FIRST_YEAR, LAST_YEAR + 1):
            washes = getattr(r, f"washes_year_{y}", np.nan)
            start, end = pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31")
            lo, hi = max(start, r.opened), min(end, CUTOFF)
            coverage = max((hi - lo).days + 1, 0) / 365
            if coverage <= 0:
                continue
            mid_age = ((lo + (hi - lo) / 2) - r.opened).days / 365.25
            rows.append(dict(site_key=r.site_key, site=r.site, year=y, washes=washes,
                             coverage=coverage, age=mid_age,
                             opyear=int(np.floor(mid_age)) + 1))
    p = pd.DataFrame(rows)
    p["usable"] = p.washes.notna() & (p.washes > 0) & (p.coverage >= MIN_COVERAGE)
    p["rate"] = np.where(p.usable, p.washes / p.coverage, np.nan)
    return p


def site_summary(p: pd.DataFrame) -> pd.DataFrame:
    """One row per site_key: how much history it has, its year-5 volume, and its maturity class.

    Split out of `build()` because the maturity class is what decides which sites are analysed at
    all, and that has to be known before the panel is filtered.
    """
    g = p[p.usable]
    per = []
    for key, s in g.groupby("site_key"):
        s = s.sort_values("opyear")
        last = s.iloc[-1]
        observed5 = s[s.opyear >= 5]
        still_trading = bool(s.year.max() >= LAST_YEAR)
        if len(observed5):
            year5, source = float(observed5.rate.median()), "Observed"
        elif not still_trading:
            year5, source = float(last.rate), "Stopped reporting"
        else:
            year5, source = float(last.rate / RAMP.get(min(int(last.opyear), 5), 1.0)), "Projected"
        per.append(dict(site_key=key, years_observed=int(s.opyear.max()),
                        n_years_data=int(len(s)), latest_opyear=int(last.opyear),
                        latest_rate=float(last.rate), first_year_rate=float(s.iloc[0].rate),
                        year5_washes=year5, year5_source=source, still_trading=still_trading))
    out = pd.DataFrame(per).set_index("site_key")
    out["maturity"] = np.select(
        [~out.still_trading.to_numpy(), out.years_observed >= 5, out.years_observed >= 3],
        ["Stopped reporting", "5+ years observed", "3–4 years observed"],
        default="1–2 years observed")
    return out


# =================================================================================================
# the maturity ramp, and its validation
# =================================================================================================
def fit_ramp(p: pd.DataFrame | None = None) -> pd.DataFrame:
    """Observed shape of the ramp: annualised rate by operating year, relative to year 5."""
    p = operating_panel() if p is None else p
    g = p[p.usable]
    anchor = g[g.opyear >= 5].groupby("site_key").rate.median()
    anchor = anchor[anchor > 0]
    n = g[g.site_key.isin(anchor.index)].copy()
    n["norm"] = n.rate / n.site_key.map(anchor)
    out = (n.groupby("opyear").norm.agg(median="median", sites="count").reset_index()
             .rename(columns={"opyear": "operating_year"}))
    out["ramp_used"] = out.operating_year.map(lambda k: RAMP.get(min(k, 5), 1.0))
    return out


def validation(p: pd.DataFrame | None = None) -> pd.DataFrame:
    """Holdout: on sites that DID reach year 5, how well does each earlier year predict it?

    `naive` is what you get by assuming the site never grows again; `ramp` applies the maturity
    curve. The gap between them is what the ramp is worth.
    """
    p = operating_panel() if p is None else p
    g = p[p.usable]
    truth = g[g.opyear == 5].set_index("site_key").rate
    rows = []
    for n in (1, 2, 3, 4):
        src = g[g.opyear == n].set_index("site_key").rate
        common = truth.index.intersection(src.index)
        if len(common) < 5:
            continue
        actual = truth[common]
        naive = src[common]
        ramped = naive / RAMP.get(n, 1.0)
        rows.append(dict(from_operating_year=n, sites=len(common),
                         mdape_naive=float(np.median((naive - actual).abs() / actual) * 100),
                         mdape_ramp=float(np.median((ramped - actual).abs() / actual) * 100),
                         bias=float(np.median(ramped / actual))))
    return pd.DataFrame(rows)


# =================================================================================================
# per-site build
# =================================================================================================
def build(basis: str = DEFAULT_BASIS,
          cars_per_hour_per_ft: float = CARS_PER_HOUR_PER_FT) -> pd.DataFrame:
    """One row per analysed site: history, year-5 volume (observed or carried forward), capacity."""
    d = load_raw()
    per = site_summary(full_panel(d))
    per = per[per.maturity.isin(MATURITY_ORDER)]
    out = d.merge(per.reset_index(), on="site_key", how="inner")
    out["maturity"] = pd.Categorical(out.maturity, MATURITY_ORDER, ordered=True)

    # capacity
    out["peak_cars_per_hour"] = out[PEAK_BASIS[basis]]
    out["basis"] = basis
    out["capacity_cars_per_hour"] = out.tunnel_ft * cars_per_hour_per_ft
    out["utilisation"] = out.peak_cars_per_hour / out.capacity_cars_per_hour
    out["required_ft"] = (out.peak_cars_per_hour / cars_per_hour_per_ft).clip(lower=MIN_VIABLE_FT)
    out["excess_ft"] = (out.tunnel_ft - out.required_ft).clip(lower=0)
    out["excess_share"] = out.excess_ft / out.tunnel_ft
    out["verdict"] = np.select(
        [out.utilisation < BAND_OVERBUILT, out.utilisation < BAND_TIGHT],
        ["Overbuilt", "Right-sized"], default="At capacity")

    out["washes_per_ft"] = out.year5_washes / out.tunnel_ft
    out["growth_y1_to_y5"] = out.year5_washes / out.first_year_rate
    out["tier"] = pd.cut(out.tunnel_ft, [0, 100, 120, 140, 400],
                         labels=["<100 ft", "100–120 ft", "120–140 ft", "140 ft+"])
    return out.sort_values(["maturity", "year5_washes"], ascending=[True, False]).reset_index(drop=True)


def site_trajectory(site_key: str) -> pd.DataFrame:
    """The year-by-year annualised trajectory for one site, with the projection to year 5."""
    p = operating_panel()
    s = p[(p.site_key == site_key) & p.usable].sort_values("opyear").copy()
    if s.empty:
        return s
    s["kind"] = "Observed"
    if s.opyear.max() < 5:
        last = s.iloc[-1]
        base = last.rate / RAMP.get(min(int(last.opyear), 5), 1.0)
        proj = [dict(site_key=site_key, site=last.site, year=np.nan, washes=np.nan,
                     coverage=np.nan, age=np.nan, opyear=k, usable=False,
                     rate=base * RAMP.get(min(k, 5), 1.0), kind="Projected")
                for k in range(int(last.opyear) + 1, 6)]
        s = pd.concat([s, pd.DataFrame(proj)], ignore_index=True)
    return s


def cohort_curve(through: int = 5) -> pd.DataFrame:
    """Median annualised wash rate by operating year, on a BALANCED panel.

    Only sites observed in *every* year 1..`through` are included. Without that restriction the
    curve moves as sites enter and leave the sample, which reads as a decline that is really just
    composition changing underneath it.
    """
    p = operating_panel()
    g = p[p.usable]
    wide = g.pivot_table(index="site_key", columns="opyear", values="rate")
    cols = [c for c in range(1, through + 1) if c in wide.columns]
    bal = wide[cols].dropna()
    # share_of_year5 is the median of each site's OWN ratio to its own year 5 — a within-site
    # statistic. (The ratio of the medians is a different number and does not answer
    # "what share of its year-5 volume does a site do in year N".)
    ratios = bal.div(bal[cols[-1]], axis=0)
    out = pd.DataFrame({
        "operating_year": cols,
        "median": [bal[c].median() for c in cols],
        "p25": [bal[c].quantile(.25) for c in cols],
        "p75": [bal[c].quantile(.75) for c in cols],
        "share_of_year5": [ratios[c].median() for c in cols],
        "sites": len(bal),
    })
    return out


def utilisation_by_basis(cars_per_hour_per_ft: float = CARS_PER_HOUR_PER_FT) -> pd.DataFrame:
    rows = []
    for label in PEAK_BASIS:
        u = build(label, cars_per_hour_per_ft).utilisation
        rows.append(dict(basis=label, p25=u.quantile(.25), median=u.median(),
                         p75=u.quantile(.75), max=u.max(),
                         pct_overbuilt=float((u < BAND_OVERBUILT).mean())))
    return pd.DataFrame(rows)


def length_vs_volume(d: pd.DataFrame) -> dict:
    from scipy import stats
    s = d[d.tunnel_ft.notna() & d.year5_washes.notna()]
    r, p = stats.pearsonr(s.tunnel_ft, s.year5_washes)
    slope, intercept = np.polyfit(s.tunnel_ft, s.year5_washes, 1)
    return dict(n=len(s), r=float(r), p=float(p), r2=float(r ** 2),
                slope=float(slope), intercept=float(intercept))


def headline(d: pd.DataFrame) -> dict:
    over = d[d.verdict == "Overbuilt"]
    obs = d[d.year5_source == "Observed"]
    return dict(
        n_sites=int(len(d)),
        n_observed5=int(len(obs)),
        median_tunnel_ft=float(d.tunnel_ft.median()),
        median_year5=float(d.year5_washes.median()),
        median_utilisation=float(d.utilisation.median()),
        max_utilisation=float(d.utilisation.max()),
        pct_overbuilt=float((d.verdict == "Overbuilt").mean()),
        n_overbuilt=int(len(over)),
        median_excess_ft=float(over.excess_ft.median()) if len(over) else 0.0,
        median_excess_share=float(over.excess_share.median()) if len(over) else 0.0,
        median_growth=float(d.growth_y1_to_y5.median()),
    )

# =================================================================================================
# Chart A — peak demand against tunnel length, grouped by how mature the site is
# =================================================================================================
# A year gets its own facet while it still holds this many sites; the thin tail folds into one
# "Year N+" facet rather than being drawn as a panel of four dots that reads as solid as a panel
# of forty.
MIN_COHORT_SITES = 10


def cohort_peaks() -> pd.DataFrame:
    """One row per site × **operating year**: the peak demand in that year, against tunnel length.

    Everything comes from `tunnel_length_with_wash.csv`. That file gives one peak figure per site
    per level (median / p75 / p90 / highest daily peak) measured across all of the site's trading
    days, plus washes for every calendar year. A peak *for a given year* is recovered by scaling the
    site-level peak by how busy that year was relative to the site's own best year:

        peak in year N  =  site peak  x  (annualised washes in year N / best year's washes)

    In the site's best year that returns the site-level peak unchanged; a quieter year scales down
    in proportion to its volume. This is arithmetic on that one file — no outside assumption about
    hours or seasonality.

    **One row per operating year, not per calendar year.** A site that opened in October has two
    calendar rows — the stub of its opening year and most of the next — that both sit inside
    operating year 1. Left alone that puts the same site on the chart twice in one facet, at two
    different heights, and makes its "path" look like it moved when it did not. The calendar
    fragments are folded together first, coverage-weighted: `rate = Σ washes ÷ Σ coverage`, which
    is the annualised run-rate over the whole operating year. 20 of 306 site-years need this.

    **This is the one chart that uses all sites in the file rather than the 3-year analysis set** —
    the question is whether under-use shrinks with age, which cannot be seen without young sites.
    """
    fp = full_panel()
    g = fp[fp.usable]

    per = (g.groupby(["site_key", "opyear"])
             .agg(washes=("washes", "sum"), coverage=("coverage", "sum"),
                  site=("site", "first"), calendar_years=("year", "nunique"),
                  first_year=("year", "min"), last_year=("year", "max"))
             .reset_index())
    per["rate"] = per.washes / per.coverage

    best = per.groupby("site_key").rate.max()
    per["vol_share"] = per.rate / per.site_key.map(best)

    raw = load_raw().set_index("site_key")
    per = per.join(raw[["where", "tunnel_ft", "tunnel_m", "client_id", "site_id"]
                       + list(PEAK_BASIS.values())], on="site_key")
    for col in PEAK_BASIS.values():
        per[col] = per[col] * per.vol_share

    per["cohort"] = _cohort_labels(per)
    per = per[per.cohort.notna()].copy()
    # Inside the folded tail facet a site could otherwise appear once per year it has beyond the
    # fold point. Keep its most mature year there — the facet's question is "where has this site
    # got to", not "every year it has been open".
    fold = per.cohort.cat.categories[-1]
    tail = per[per.cohort == fold]
    if len(tail):
        keep = tail.groupby("site_key").opyear.idxmax()
        per = pd.concat([per[per.cohort != fold], per.loc[keep]])
    return per.sort_values(["site_key", "opyear"]).reset_index(drop=True)


def _cohort_labels(per: pd.DataFrame) -> pd.Categorical:
    """Solo facets while a year is well populated, then one folded tail facet.

    The old fixed `["Year 1", "Year 2", "Year 3", "Year 4+"]` buried years 4 through 9 in a single
    panel, which is exactly the range where the "does under-use close with age?" question is
    answered. The cut is derived from the data instead: the tail starts at the first year too thin
    to stand on its own.
    """
    counts = per.groupby("opyear").site_key.nunique()
    solo = [int(y) for y in sorted(counts.index) if counts[y] >= MIN_COHORT_SITES]
    fold_at = (max(solo) if solo else 1)          # last well-populated year opens the tail facet
    labels = [f"Year {y}" for y in range(1, fold_at)] + [f"Year {fold_at}+"]
    assigned = np.where(per.opyear >= fold_at, f"Year {fold_at}+",
                        "Year " + per.opyear.astype(int).astype(str))
    return pd.Categorical(assigned, labels, ordered=True)


def cohorts() -> list[str]:
    """The facet labels actually present, in order — the section reads this rather than a constant."""
    c = cohort_peaks()
    return [str(x) for x in c.cohort.cat.categories if (c.cohort == x).any()]


def site_picker() -> dict[str, str]:
    """Label → `site_key`, guaranteed one-to-one.

    **The site key is `client_id + site_id`.** An earlier version built this map with the site's
    *name* as the dict key, so two sites sharing a name would silently collapse into one and the
    picker would quietly point at the wrong dots. Names happen to be unique in this file today
    (78 rows, 78 names, but only 64 client_ids — one client runs up to 6 sites), which is exactly
    the kind of accident that stops being true after the next data refresh. The label carries the
    real key so the reader can see which site they are looking at.
    """
    raw = load_raw()
    out: dict[str, str] = {}
    for r in raw.sort_values(["name", "site_id"]).itertuples():
        label = f"{r.site} — {r.where}  ·  {r.client_id} #{int(r.site_id)}"
        out[label] = r.site_key
    return out


def site_utilisation(site_key: str,
                     cars_per_hour_per_ft: float = CARS_PER_HOUR_PER_FT) -> pd.DataFrame:
    """One site's own path through the capacity chart — a row per operating year, per peak level.

    Same arithmetic as `cohort_utilisation`, not aggregated: this is what a single dot in the
    cohort facets does as it ages, which the facets themselves cannot show because a site appears
    once in each of them as a different dot.
    """
    c = cohort_peaks()
    g = c[c.site_key == site_key].sort_values("opyear")
    rows = []
    for label, col in PEAK_BASIS.items():
        for r in g.itertuples():
            cap = getattr(r, "tunnel_ft") * cars_per_hour_per_ft
            rows.append(dict(peak_level=label, opyear=int(r.opyear), cohort=str(r.cohort),
                             cars=float(getattr(r, col)), tunnel_ft=float(r.tunnel_ft),
                             share=float(getattr(r, col) / cap) if cap > 0 else np.nan,
                             washes_rate=float(r.rate)))
    out = pd.DataFrame(rows)
    if not out.empty:
        out.attrs["site"] = str(g.site.iloc[0]) if len(g) else site_key
        # `g["where"]`, never `g.where` — that attribute is DataFrame.where, the method.
        out.attrs["where"] = str(g["where"].iloc[0]) if "where" in g.columns and len(g) else ""
        out.attrs["tunnel_ft"] = float(g.tunnel_ft.iloc[0]) if len(g) else np.nan
    return out


def cohort_utilisation(cars_per_hour_per_ft: float = CARS_PER_HOUR_PER_FT) -> pd.DataFrame:
    """Share of the tunnel used at each peak level, by maturity cohort."""
    c = cohort_peaks()
    rows = []
    for label, col in PEAK_BASIS.items():
        share = c[col] / (c.tunnel_ft * cars_per_hour_per_ft)
        for coh, v in share.groupby(c.cohort, observed=True):
            rows.append(dict(peak_level=label, cohort=str(coh), sites=int(v.notna().sum()),
                             median_share=float(v.median())))
    return pd.DataFrame(rows)

def operating_days(d: pd.DataFrame | None = None) -> pd.Series:
    """Trading days behind each site's peak percentiles — open date to the end of the extract.

    This is what makes the percentiles readable: the "highest daily peak" is the best single day out
    of this many, and the p90 is beaten on roughly a tenth of them.
    """
    d = load_raw() if d is None else d
    return ((CUTOFF - d.opened).dt.days.clip(lower=0)).rename("operating_days")


def peak_context() -> dict:
    """Plain-language scale for the percentile columns, for captions."""
    days = operating_days()
    med = float(days.median())
    return dict(median_days=med, p90_days=med * 0.10, median_day_count=med * 0.50,
                min_days=float(days.min()), max_days=float(days.max()))


# =================================================================================================
# The maturity ramp, measured on the wide monthly panel
# =================================================================================================
# `tunnel_length_with_wash.csv` carries only 27 sites with a complete first five years, which is too
# thin to read a ramp from. `historical_data_5yrs_monthly.csv` carries 2,103 sites with a real
# `operational_start` and monthly washes, so operating years can be cut exactly rather than
# approximated from calendar columns. It is used ONLY for the shape of the ramp — it is never joined
# to the tunnel sites, which stay on their own file.
RAMP_PANEL = REPO / "conclusion" / "data" / "historical_data_5yrs_monthly.csv"
RAMP_MIN_YEARS = 4     # a site must have this many COMPLETE operating years to count


def ramp_panel_years() -> pd.DataFrame:
    """One row per site × complete operating year, from the wide monthly panel.

    A year counts only if all 12 of its months are present, so nothing is scaled or extrapolated.
    """
    d = pd.read_csv(RAMP_PANEL, low_memory=False)
    d["site_key"] = d.client_id.astype(str) + "___" + d.site_id.astype(str)
    d["washes"] = d.mem_wash_count.fillna(0) + d.ret_wash_count.fillna(0)
    d["opened"] = pd.to_datetime(d.operational_start, format="mixed", errors="coerce")
    d["date"] = pd.to_datetime(dict(year=d.year, month=d.month, day=1))
    g = d[(d.washes > 0) & d.opened.notna()].copy()
    g["op_month"] = ((g.date.dt.year - g.opened.dt.year) * 12
                     + (g.date.dt.month - g.opened.dt.month))
    g = g[g.op_month >= 0]
    g["opyear"] = g.op_month // 12 + 1
    out = (g.groupby(["site_key", "opyear"])
             .agg(months=("washes", "size"), washes=("washes", "sum"),
                  opened=("opened", "first"))
             .reset_index())
    return out[out.months >= 12]


def ramp_curve(min_years: int = RAMP_MIN_YEARS) -> pd.DataFrame:
    """Median washes by operating year, on sites with `min_years` complete years.

    `share_of_final` is the median of each site's OWN ratio to its `min_years` year — a within-site
    statistic, so it is not distorted by big and small sites entering at different points.
    """
    y = ramp_panel_years()
    wide = y.pivot(index="site_key", columns="opyear", values="washes")
    cols = [c for c in range(1, min_years + 1) if c in wide.columns]
    bal = wide[cols].dropna()
    ratios = bal.div(bal[cols[-1]], axis=0)
    rows = []
    for c in cols:
        rows.append(dict(operating_year=c, sites=int(len(bal)),
                         median=float(bal[c].median()),
                         p25=float(bal[c].quantile(.25)), p75=float(bal[c].quantile(.75)),
                         share_of_final=float(ratios[c].median())))
    out = pd.DataFrame(rows)
    out.attrs["anchor_year"] = cols[-1]
    out.attrs["sites"] = int(len(bal))
    return out


def ramp_validation(min_years: int = RAMP_MIN_YEARS) -> pd.DataFrame:
    """Holdout on the wide panel: predict the final year from each earlier one."""
    y = ramp_panel_years()
    wide = y.pivot(index="site_key", columns="opyear", values="washes")
    cols = [c for c in range(1, min_years + 1) if c in wide.columns]
    bal = wide[cols].dropna()
    target, ratios = bal[cols[-1]], bal.div(bal[cols[-1]], axis=0)
    rows = []
    for c in cols[:-1]:
        factor = ratios[c].median()
        naive, ramped = bal[c], bal[c] / factor
        rows.append(dict(from_operating_year=c, sites=int(len(bal)), factor=float(factor),
                         mdape_naive=float(np.median((naive - target).abs() / target) * 100),
                         mdape_ramp=float(np.median((ramped - target).abs() / target) * 100),
                         bias=float(np.median(ramped / target))))
    return pd.DataFrame(rows)
