"""
Demographics — does the market a site sits in explain how much it washes?

Every site-selection proforma in the industry is built on the same premise: score the trade area
(population, income, vehicles, traffic, competitors, nearby retail), and the score tells you what
the wash will do. This module tests that premise against a clean slice of the real estate.

**The cohort.** `historical_data_5yrs_monthly.csv` is joined to `historical_data_sitewise.csv` on
`client_id_1 + site_id` — the number-first client id is the semantically correct key — and then cut
down to sites with **all twelve months of 2025 present and trading**. That is 1,263 sites. The
twelve-month gate matters: it means every site's annual volume is a real sum, never annualised from
a partial year, so a site cannot look small merely because it opened in August. 2025 is the last
complete calendar year in the panel.

**The targets.** Total washes, membership washes and retail washes for calendar 2025. They are kept
separate because they behave differently — a membership wash is a subscription being used, a retail
wash is a stranger driving in, and the market has more to say about the second than the first.

**What is deliberately not done here.** No merging with the tunnel file or the n70 backtest file;
those sections stand alone. Nothing here feeds a modelled number back into the forecaster — this is
evidence about which inputs are worth carrying, not a new model.

Streamlit-free, so the notebook imports the same numbers the app shows.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
PANEL = REPO / "conclusion" / "data" / "historical_data_5yrs_monthly.csv"
SITEWISE = REPO / "conclusion" / "data" / "historical_data_sitewise.csv"

YEAR = 2025
MIN_STATE_SITES = 10          # below this a state median is noise, not a reading
MIN_OPERATOR_SITES = 4        # below this an operator cannot rank its own sites

# Continental-US box; a few rows carry impossible coordinates and would drag the map into the sea.
LAT_RANGE, LON_RANGE = (20.0, 50.0), (-130.0, -65.0)

# A trade area with a population of exactly zero, or zero vehicles passing all day, is a failed
# geocode in the sitewise file rather than a real desert. Those cells are read as missing so they
# cannot anchor the bottom of a correlation or a quintile. 14 sites have zero population and 19
# have zero traffic; both are reported by `data_gaps()` rather than silently dropped.
ZERO_IS_MISSING = ("2025 Estimate", "Labor Force", "Total Vehicles Available in the Market",
                   "traffic_total")

# label -> (column, family). The label is what the reviewer reads; the column is what the file
# calls it. Grouped into families so the evidence can be read a family at a time — if population
# has nothing to say, that is a statement about all six population columns at once.
FEATURES: dict[str, tuple[str, str]] = {
    "Population in the trade area":       ("2025 Estimate", "Population"),
    "Population growth, 2020→2025":       ("Growth 2025-2020", "Population"),
    "Population growth, 2025→2030":       ("Growth 2030-2025", "Population"),
    "Average age":                        ("2025 Average Age", "Population"),
    "Labour force":                       ("Labor Force", "Population"),
    "Renter-occupied homes":              ("Renter-Occupied", "Population"),

    "Median household income":            ("Median Household Income", "Income"),
    "Average household income":           ("Average Household Income", "Income"),
    "% of households over $50k":          ("2025 % HH with Income $50K+", "Income"),
    "Households, $100–125k":              ("$100,000 to $124,999", "Income"),
    "Households, $150–175k":              ("$150,000 to $174,999", "Income"),
    "Households, $200–250k":              ("$200,000 to $249,999", "Income"),
    "Owner-occupied home value":          ("Current Year Estimated Owner-Occupied Housing Units "
                                           "by Value", "Income"),

    "Vehicles in the market":             ("Total Vehicles Available in the Market", "Vehicles"),
    "Vehicles per household":             ("Average Number of Vehicles Available", "Vehicles"),
    "Households with 1 vehicle":          ("1 vehicle", "Vehicles"),
    "Households with 2 vehicles":         ("2 vehicles", "Vehicles"),
    "Households with 3+ vehicles":        ("3 vehicles", "Vehicles"),

    "Car washes nearby":                  ("Count of Car Wash Competitors", "Competition"),
    "Distance to nearest car wash":       ("Nearest Car Wash Competitors-Distance", "Competition"),
    "Distance to 3rd nearest car wash":   ("3rd Nearest Car Wash Competitors-Distance",
                                           "Competition"),

    "Mass merchants nearby":              ("Count of ChainXY VT - Mass Merchant", "Retail anchors"),
    "Distance to nearest mass merchant":  ("Nearest ChainXY VT - Mass Merchant-Distance",
                                           "Retail anchors"),
    "Grocery stores nearby":              ("Count of ChainXY VT - Grocery", "Retail anchors"),
    "Department stores nearby":           ("Count of ChainXY VT - Department Store",
                                           "Retail anchors"),

    "Traffic, all day":                   ("traffic_total", "Traffic"),
    "Traffic, morning":                   ("Nearest StreetLight US Hourly-ttl_breakfast", "Traffic"),
    "Traffic, midday":                    ("Nearest StreetLight US Hourly-ttl_lunch", "Traffic"),
    "Traffic, afternoon":                 ("Nearest StreetLight US Hourly-ttl_afternoon", "Traffic"),
    "Traffic, evening":                   ("Nearest StreetLight US Hourly-ttl_dinner", "Traffic"),
    "Traffic, overnight":                 ("Nearest StreetLight US Hourly-ttl_overnight", "Traffic"),
}

FAMILIES = ["Population", "Income", "Vehicles", "Competition", "Retail anchors", "Traffic"]

TARGETS: dict[str, str] = {
    "Total washes": "total_washes",
    "Membership washes": "mem_washes",
    "Retail washes": "ret_washes",
}

# The site's own trading facts — NOT market inputs. They are carried so the evidence can show what
# *does* move with volume once the market has been ruled out, and they are never mixed into a
# demographic model (membership count is an outcome of the wash, not a feature of the location).
OWN_FACTS: dict[str, str] = {
    "Membership customers": "mem_purchases",
    "Membership share of washes": "mem_share",
    "Washes per member per year": "washes_per_member",
    "Retail price per wash": "ret_asp",
}

TRAFFIC_COLS = ["Nearest StreetLight US Hourly-ttl_overnight",
                "Nearest StreetLight US Hourly-ttl_breakfast",
                "Nearest StreetLight US Hourly-ttl_lunch",
                "Nearest StreetLight US Hourly-ttl_afternoon",
                "Nearest StreetLight US Hourly-ttl_dinner",
                "Nearest StreetLight US Hourly-ttl_night"]


# --- the cohort ----------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def cohort() -> pd.DataFrame:
    """One row per site: calendar-2025 trading, joined to its trade-area demographics.

    A site qualifies only if all twelve months of 2025 are present *and* every one of them recorded
    at least one wash. That removes the two failure modes that would otherwise masquerade as a weak
    market: a site that opened mid-year, and a site that stopped reporting mid-year.
    """
    m = pd.read_csv(PANEL, low_memory=False)
    m = m[m.year == YEAR].copy()
    m["washes"] = m.mem_wash_count.fillna(0) + m.ret_wash_count.fillna(0)

    traded = m[m.washes > 0].groupby(["client_id", "site_id"]).month.nunique()
    full = traded[traded == 12].index

    agg = (m.set_index(["client_id", "site_id"]).loc[full].groupby(level=[0, 1])
             .agg(total_washes=("washes", "sum"),
                  mem_washes=("mem_wash_count", "sum"),
                  ret_washes=("ret_wash_count", "sum"),
                  mem_revenue=("mem_revenue", "sum"),
                  ret_revenue=("ret_revenue", "sum"),
                  mem_purchases=("mem_purchase_count", "sum"),
                  operator=("client_name", "first"), state=("state", "first"),
                  region=("region", "first"), city=("address1", "first"),
                  lat=("lat", "first"), lon=("lon", "first"))
             .reset_index())

    sw = pd.read_csv(SITEWISE, low_memory=False)
    # `client_id_1` is the number-first id and is the correct key; `client_id` in this file is
    # name-first and matches more rows only because the monthly panel carries both styles.
    sw = sw[sw.client_id_1.notna()].copy()
    sw["client_id"] = sw.client_id_1.astype(str)
    sw = sw.drop(columns=[c for c in ("client_name", "lat", "lon", "Name", "Latitude", "Longitude",
                                      "_Match", "__longitude", "__latitude", "__name",
                                      "Median Household Income.1", "Average Household Income.1")
                          if c in sw.columns])

    d = agg.merge(sw, on=["client_id", "site_id"], how="inner")
    d["traffic_total"] = d[TRAFFIC_COLS].sum(axis=1)
    for c in ZERO_IS_MISSING:
        d.loc[d[c] <= 0, c] = np.nan

    d["revenue"] = d.mem_revenue + d.ret_revenue
    d["mem_share"] = d.mem_washes / d.total_washes
    d["ret_asp"] = d.ret_revenue / d.ret_washes.replace(0, np.nan)
    d["washes_per_member"] = d.mem_washes / d.mem_purchases.replace(0, np.nan)
    d["on_map"] = d.lat.between(*LAT_RANGE) & d.lon.between(*LON_RANGE)
    d["site"] = d.operator.astype(str) + " · " + d.city.astype(str).str.slice(0, 34)
    return d.reset_index(drop=True)


def headline() -> dict:
    d = cohort()
    big = d.state.map(d.state.value_counts()) >= MIN_STATE_SITES
    return dict(
        sites=int(len(d)), states=int(d.state.nunique()),
        ranked_states=int((d[big].state.nunique())),
        operators=int(d.client_id.nunique()),
        features=len(FEATURES),
        median_washes=float(d.total_washes.median()),
        p25=float(d.total_washes.quantile(.25)), p75=float(d.total_washes.quantile(.75)),
        spread=float(d.total_washes.quantile(.9) / d.total_washes.quantile(.1)),
        median_mem_share=float(d.mem_share.median()),
    )


def data_gaps() -> pd.DataFrame:
    """Cells read as missing, so the reviewer can see what was set aside and why."""
    raw = pd.read_csv(SITEWISE, low_memory=False)
    raw = raw[raw.client_id_1.notna()].copy()
    raw["client_id"] = raw.client_id_1.astype(str)
    raw["traffic_total"] = raw[TRAFFIC_COLS].sum(axis=1)
    keys = set(zip(cohort().client_id, cohort().site_id))
    raw = raw[[k in keys for k in zip(raw.client_id, raw.site_id)]]
    rows = [dict(field={"2025 Estimate": "Population in the trade area",
                        "Labor Force": "Labour force",
                        "Total Vehicles Available in the Market": "Vehicles in the market",
                        "traffic_total": "Traffic, all day"}[c],
                 sites=int((raw[c] <= 0).sum()),
                 pct=float((raw[c] <= 0).mean())) for c in ZERO_IS_MISSING]
    return pd.DataFrame(rows).sort_values("sites", ascending=False).reset_index(drop=True)


# --- correlation evidence ------------------------------------------------------------------------

def _bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values. 31 features x 3 targets is enough tests that a raw p of .04
    means nothing on its own."""
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / (np.arange(n) + 1))[::-1])[::-1]
    return np.minimum(q, 1.0)


@lru_cache(maxsize=8)
def correlations(target: str = "Total washes") -> pd.DataFrame:
    """Every feature against one target, three ways.

    `rho` is the plain rank correlation across all 1,263 sites. `rho_within_state` re-ranks both
    sides inside each state first, so a correlation that is really "Texas is busy and Texas is
    sunny" collapses. `rho_within_operator` does the same inside each operator with 4+ sites, which
    is the question a site-selection team actually faces: *given it is us building it, does this
    number tell us which of our candidate sites will do better?*
    """
    d = cohort()
    tcol = TARGETS[target]
    big_state = d.state.map(d.state.value_counts()) >= MIN_STATE_SITES
    big_op = d.client_id.map(d.client_id.value_counts()) >= MIN_OPERATOR_SITES

    rows = []
    for label, (col, family) in FEATURES.items():
        x, y = d[col].astype(float), d[tcol].astype(float)
        ok = x.notna() & y.notna()
        rho, p = stats.spearmanr(x[ok], y[ok])

        def nested(mask, by):
            m = mask & ok
            xr = x[m].groupby(d.loc[m, by]).rank(pct=True)
            yr = y[m].groupby(d.loc[m, by]).rank(pct=True)
            return stats.spearmanr(xr, yr) if m.sum() > 40 else (np.nan, np.nan)

        rs, _ = nested(big_state, "state")
        ro, po = nested(big_op, "client_id")
        rows.append(dict(feature=label, family=family, n=int(ok.sum()), rho=rho, p=p,
                         rho_within_state=rs, rho_within_operator=ro, p_within_operator=po))

    out = pd.DataFrame(rows)
    out["q"] = _bh(out.p.values)
    out["abs_rho"] = out.rho.abs()
    # A rank correlation of 0.10 on 1,263 sites is "statistically significant" and commercially
    # nothing. The verdict column says so in words rather than leaving a q-value to be misread.
    out["verdict"] = np.where(out.q > .05, "No signal",
                       np.where(out.abs_rho < .10, "Detectable, too small to use",
                                "Real but weak"))
    return out.sort_values("abs_rho", ascending=False).reset_index(drop=True)


def quintiles(feature: str, target: str = "Total washes") -> pd.DataFrame:
    """Sites sorted into five equal groups on one feature; median volume in each.

    This is the exhibit that survives contact with a non-statistician: if the bottom fifth of sites
    on population wash roughly what the top fifth do, population is not a lever.
    """
    d = cohort()
    col, tcol = FEATURES[feature][0], TARGETS[target]
    ok = d[col].notna() & d[tcol].notna()
    sub = d[ok].copy()
    sub["bucket"] = pd.qcut(sub[col].rank(method="first"), 5,
                            labels=["Lowest fifth", "2nd", "Middle", "4th", "Highest fifth"])
    out = (sub.groupby("bucket", observed=True)
              .agg(sites=(tcol, "size"), median_washes=(tcol, "median"),
                   p25=(tcol, lambda v: v.quantile(.25)), p75=(tcol, lambda v: v.quantile(.75)),
                   feature_median=(col, "median"), feature_lo=(col, "min"), feature_hi=(col, "max"))
              .reset_index())
    out.attrs["spread"] = float(out.median_washes.iloc[-1] / out.median_washes.iloc[0])
    out.attrs["monotonic"] = bool(out.median_washes.is_monotonic_increasing
                                  or out.median_washes.is_monotonic_decreasing)
    return out


def own_fact_quintiles(factor: str = "Membership customers",
                       target: str = "Total washes") -> pd.DataFrame:
    """`quintiles()` for one of the site's own trading facts — the contrast exhibit.

    Same shape and same axis as the market version, so the two can be read side by side without
    anyone having to rescale in their head.
    """
    d = cohort()
    col, tcol = OWN_FACTS[factor], TARGETS[target]
    ok = d[col].notna() & d[tcol].notna()
    sub = d[ok].copy()
    sub["bucket"] = pd.qcut(sub[col].rank(method="first"), 5,
                            labels=["Lowest fifth", "2nd", "Middle", "4th", "Highest fifth"])
    out = (sub.groupby("bucket", observed=True)
              .agg(sites=(tcol, "size"), median_washes=(tcol, "median"))
              .reset_index())
    out.attrs["spread"] = float(out.median_washes.iloc[-1] / out.median_washes.iloc[0])
    out.attrs["monotonic"] = bool(out.median_washes.is_monotonic_increasing)
    return out


def own_facts_correlations() -> pd.DataFrame:
    """The site's own trading facts against total washes — the contrast to the market table."""
    d = cohort()
    rows = []
    for label, col in OWN_FACTS.items():
        x, y = d[col].astype(float), d.total_washes.astype(float)
        ok = x.notna() & y.notna()
        rho, p = stats.spearmanr(x[ok], y[ok])
        rows.append(dict(factor=label, n=int(ok.sum()), rho=rho, p=p))
    return pd.DataFrame(rows).sort_values("rho", key=abs, ascending=False).reset_index(drop=True)


# --- can a model built on these features actually forecast? ---------------------------------------

def _feature_matrix(d: pd.DataFrame) -> np.ndarray:
    return d[[c for c, _ in FEATURES.values()]].astype(float).values


@lru_cache(maxsize=1)
def oos_scores() -> pd.DataFrame:
    """Out-of-sample R² for a model given every demographic feature at once.

    Folds are split **by state**, so the model is always scored on markets it has never seen — the
    honest version of the question "would this have helped us underwrite the next site?". Two model
    families are run so the answer cannot be blamed on the choice of model: a gradient-boosted tree
    (finds interactions and non-linearity) and a ridge regression (cannot overfit).

    R² of 0 means "no better than quoting the estate-wide median". Below 0 means worse than that.
    """
    d = cohort()
    X = _feature_matrix(d)
    groups = d.state.values
    rows = []
    for tname, tcol in TARGETS.items():
        y = np.log1p(d[tcol].astype(float).values)
        for mname in ("Gradient-boosted trees", "Ridge regression"):
            pred = np.zeros(len(y))
            for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
                if mname.startswith("Gradient"):
                    mdl = HistGradientBoostingRegressor(max_depth=3, learning_rate=.05,
                                                        max_iter=300, random_state=0)
                    mdl.fit(X[tr], y[tr])
                    pred[te] = mdl.predict(X[te])
                else:
                    sc = StandardScaler().fit(np.nan_to_num(X[tr]))
                    mdl = RidgeCV(alphas=np.logspace(-2, 4, 25))
                    mdl.fit(sc.transform(np.nan_to_num(X[tr])), y[tr])
                    pred[te] = mdl.predict(sc.transform(np.nan_to_num(X[te])))
            r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
            # A handful of sites did zero membership washes all year; they cannot carry a
            # percentage error, so the median is taken over the rest.
            act = d[tcol].astype(float).values
            nz = act > 0
            err = np.abs(np.expm1(pred[nz]) - act[nz]) / act[nz]
            rows.append(dict(target=tname, model=mname, r2=float(r2),
                             mdape=float(np.median(err) * 100)))
    return pd.DataFrame(rows)


@lru_cache(maxsize=1)
def variance_decomposition() -> pd.DataFrame:
    """What *does* explain a site's volume — scored the same honest way.

    Each row is a candidate explanation, scored by leave-one-out R² on log volume: predict a site
    from the average of the *other* sites that share its label, never from itself. Demographics are
    scored by the cross-market model above, so all rows are on one scale.
    """
    d = cohort()
    y = np.log1p(d.total_washes.astype(float).values)

    def loo(labels: np.ndarray, mask: np.ndarray | None = None):
        m = np.ones(len(y), bool) if mask is None else mask
        yy, gg = y[m], pd.Series(labels[m])
        tot = pd.Series(yy).groupby(gg.values).transform("sum").values
        n = pd.Series(yy).groupby(gg.values).transform("size").values
        pred = np.where(n > 1, (tot - yy) / np.maximum(n - 1, 1), yy.mean())
        return (1 - np.sum((yy - pred) ** 2) / np.sum((yy - yy.mean()) ** 2),
                int(gg.nunique()), int(m.sum()))

    big_op = (d.client_id.map(d.client_id.value_counts()) >= 3).values
    rows = []
    r2, k, n = loo(d.client_id.values, big_op)
    rows.append(dict(explanation="Who operates it", detail=f"{k} operators with 3+ sites",
                     r2=r2, sites=n))
    r2, k, n = loo(d.state.values)
    rows.append(dict(explanation="Which state it is in", detail=f"{k} states", r2=r2, sites=n))
    r2, k, n = loo(d.region.values)
    rows.append(dict(explanation="Which region it is in", detail=f"{k} regions", r2=r2, sites=n))

    nb = neighbour_curve()
    best = nb.loc[nb.r2.idxmax()]
    rows.append(dict(explanation="How its nearest neighbours do",
                     detail=f"median of the {int(best.k)} closest other sites",
                     r2=float(best.r2), sites=int(best.sites)))

    dem = oos_scores().query("target == 'Total washes'").r2.max()
    rows.append(dict(explanation="All 31 demographic features",
                     detail="gradient boosting / ridge, scored on unseen states",
                     r2=float(dem), sites=int(len(d))))
    return pd.DataFrame(rows).sort_values("r2", ascending=False).reset_index(drop=True)


@lru_cache(maxsize=1)
def neighbour_curve() -> pd.DataFrame:
    """Does geography carry signal even where demographics do not?

    For each site, predict its volume from the median of its k nearest *other* sites. If the
    neighbours know something the census does not, that is an argument for anchoring a forecast on
    local trading rather than on a trade-area score.
    """
    d = cohort()
    m = d.on_map.values
    lat, lon = np.radians(d.lat.values[m]), np.radians(d.lon.values[m])
    y = np.log1p(d.total_washes.astype(float).values[m])
    dlat, dlon = lat[:, None] - lat[None, :], lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2) ** 2
    D = 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    np.fill_diagonal(D, np.inf)
    order = np.argsort(D, axis=1)
    sst = np.sum((y - y.mean()) ** 2)
    rows = []
    for k in (1, 3, 5, 10, 20):
        pred = np.median(y[order[:, :k]], axis=1)
        rows.append(dict(k=k, r2=float(1 - np.sum((y - pred) ** 2) / sst),
                         rho=float(stats.spearmanr(pred, y).statistic),
                         median_km=float(np.median(np.take_along_axis(D, order[:, :k], 1)[:, -1])),
                         sites=int(m.sum())))
    return pd.DataFrame(rows)


# --- geography -----------------------------------------------------------------------------------

def state_table(min_sites: int = MIN_STATE_SITES) -> pd.DataFrame:
    """Per-state medians. Ranked on the typical site, never on the state total — a state total
    mostly measures how many sites we happen to own there."""
    d = cohort()
    out = (d.groupby("state")
             .agg(sites=("total_washes", "size"), median_washes=("total_washes", "median"),
                  median_mem_share=("mem_share", "median"),
                  median_pop=("2025 Estimate", "median"),
                  median_income=("Median Household Income", "median"),
                  median_traffic=("traffic_total", "median"),
                  median_competitors=("Count of Car Wash Competitors", "median"))
             .reset_index())
    out["enough_sites"] = out.sites >= min_sites
    return out.sort_values("median_washes", ascending=False).reset_index(drop=True)


def state_feature_link(min_sites: int = MIN_STATE_SITES) -> pd.DataFrame:
    """The map's companion number: across states, does the state's median feature track its median
    volume? A feature that genuinely drives volume should make busy states look different from quiet
    ones on the map."""
    d = cohort()
    keep = d.state.map(d.state.value_counts()) >= min_sites
    sub = d[keep]
    med_y = sub.groupby("state").total_washes.median()
    rows = []
    for label, (col, family) in FEATURES.items():
        v = sub.groupby("state")[col].median().reindex(med_y.index)
        ok = v.notna()
        rho, p = stats.spearmanr(v[ok], med_y[ok])
        rows.append(dict(feature=label, family=family, states=int(ok.sum()), rho=rho, p=p))
    ms = sub.groupby("state").mem_share.median().reindex(med_y.index)
    rho, p = stats.spearmanr(ms, med_y)
    rows.append(dict(feature="Membership share of washes (not a market input)",
                     family="The site itself", states=int(len(med_y)), rho=rho, p=p))
    return (pd.DataFrame(rows).sort_values("rho", key=abs, ascending=False)
            .reset_index(drop=True))


def map_frame(colour_by: str = "Total washes") -> pd.DataFrame:
    """Sites plottable on a US map, with the chosen field as `value` and a percentile for colour.

    Percentile rather than raw value on purpose: population and washes differ by three orders of
    magnitude, and the point of the map is to compare *patterns*, which only works if both are on
    the same 0–100 scale.
    """
    d = cohort()[lambda x: x.on_map].copy()
    col = TARGETS.get(colour_by) or FEATURES.get(colour_by, (colour_by,))[0]
    d["value"] = d[col].astype(float)
    d["pct"] = d.value.rank(pct=True) * 100
    d["wash_pct"] = d.total_washes.rank(pct=True) * 100
    return d[["site", "operator", "state", "city", "lat", "lon", "value", "pct", "wash_pct",
              "total_washes", "mem_share"]].reset_index(drop=True)
