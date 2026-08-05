"""
Operator clusters — where one operator has packed several sites into a single place.

The question this answers is not "where are our sites" (§⓪ does that, nationally) but "when an
operator builds **five washes inside one town**, what does that look like from the inside?" —
how far apart they sit, in what order they were opened, and what each one has washed since.

The unit is an **operator cluster**: a set of sites that share a `client_id` and all sit within
`max_km` of each other. Clustering is complete-linkage on great-circle distance, so `max_km` is a
hard cap on the cluster's *diameter* — every site in a cluster is within `max_km` of every other
one, not merely of its nearest neighbour. Single linkage would chain: the Rio Grande Valley would
join up into one 120 km "place" through a string of 15 km hops, which is not a place you can put on
one screen. See `docs/DIVERGENCES.md` for the repo's other clustering (20 km complete-linkage on
KPIs, same reasoning).

**A coordinate defect this module has to work around.** 93 sites in the panel carry a *placeholder*
latitude/longitude: an identical coordinate pair shared by several sites of the same operator whose
street addresses are all different. BlueWave stamps 21 Houston-area sites on one point; Buckeye
stamps 10 sites spread over six Ohio towns (Norwalk, Ontario, Mansfield, Ashland, Lorain, Elyria)
on another. Their wash data is real; their location is not. Every one of them is dropped before
clustering, because a section whose entire subject is *how far apart sites are* cannot include
sites whose distance is a geocoding artefact. `coord_quality()` reports exactly what was dropped so
the app can say so out loud rather than quietly showing 21 sites as one dot.

Sites that share a coordinate **and** a street address are a different thing — a genuine
second tunnel at one address, or an operator handoff — and are kept, at a true distance of 0.

Input: `conclusion/data/historical_data_5yrs_monthly.csv`, the monthly wash panel. It is
byte-identical to `proforma/data/panel/main-data-v2-stitched.csv`; this module reads it through
`conclusion/data/` because every other section's data module does, not because they differ.

Streamlit-free, so the notebook can import it unchanged.
"""
from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

REPO = Path(__file__).resolve().parents[2]
PANEL = REPO / "conclusion" / "data" / "historical_data_5yrs_monthly.csv"

# Continental-US bounding box. A handful of rows carry 0/0, 90/180 or otherwise impossible
# coordinates; §⓪ uses the same gate.
LAT_RANGE, LON_RANGE = (20.0, 50.0), (-130.0, -65.0)

EARTH_KM = 6371.0088
KM_PER_MILE = 1.609344

# The panel runs to 2026-06. Calendar-year charts stop at the last year with all twelve months in
# it, so a half-collected 2026 cannot be read as a collapse. Derived, never hard-coded — see
# `last_complete_year()`.
MONTHS_IN_YEAR = 12


# =================================================================================================
# Loading
# =================================================================================================

# The three loaders below are memoised at module level, not only through Streamlit's `@st.cache_data`
# in the section. `opening_effect_all()` walks ~110 clusters and every one of them re-enters
# `build()` → `site_index()` → `_panel()`; uncached that is 110 reads of a 71k-row CSV, and the
# notebook (which has no Streamlit cache at all) would pay it too. Callers that mutate take a
# `.copy()` first — check that before adding one.
@lru_cache(maxsize=1)
def _panel() -> pd.DataFrame:
    d = pd.read_csv(PANEL, low_memory=False)
    d["washes"] = d.mem_wash_count.fillna(0) + d.ret_wash_count.fillna(0)
    d["revenue"] = d.mem_revenue.fillna(0) + d.ret_revenue.fillna(0)
    d["site_key"] = d.client_id.astype(str) + "___" + d.site_id.astype(str)
    # A month is "trading" if it washed a car. Pre-opening and post-closure rows sit in the panel
    # as zeros and would otherwise deflate every per-month rate in the section.
    d = d[d.washes > 0]
    d["ym"] = d.year * MONTHS_IN_YEAR + (d.month - 1)
    return d


def last_complete_year() -> int:
    """The last calendar year for which the panel holds all twelve months.

    One stray 2027 row exists carrying three washes; requiring twelve months discards it without
    naming it, which is the point — the rule is stated once and survives the next data refresh.
    """
    d = _panel()
    full = d.groupby("year").month.nunique()
    return int(full[full >= MONTHS_IN_YEAR].index.max())


def _brand(names: pd.Series, frac: float = 0.6) -> str:
    """The operator's brand, read off its own site names.

    `client_name` is not the operator — it is the *site* name, so an operator with 92 sites has 92
    distinct `client_name` values. Most brands leave a fingerprint in them ("BlueWave Katy",
    "Whistle Express Car Wash - Ramsey"); some name their sites purely after the street ("Sunset",
    "Rainbow") and leave none.

    A token joins the brand if it is the modal word in that position across at least `frac` of the
    operator's sites. A strict common prefix was tried first and is too brittle — one oddly-named
    site out of 92 collapses BlueWave's brand to the empty string.
    """
    toks = [str(n).split() for n in names]
    n = len(toks)
    if not n:
        return ""
    out: list[str] = []
    for i in range(8):
        col = Counter(t[i] for t in toks if len(t) > i)
        if not col:
            break
        word, hits = col.most_common(1)[0]
        if hits / n < frac:
            break
        out.append(word)
    return re.sub(r"[\s\-–—:·]+$", "", " ".join(out))


def _operator_label(client_id: str, brand: str) -> str:
    """Brand if the site names carry one, else the `client_id` stem, prettified.

    The fallback is deliberately literal rather than clever: `client_id` is the real key and the
    reader can always trace it back. Every table in the section prints the raw `client_id` too.
    """
    if brand:
        return brand
    stem = re.sub(r"_\d+$", "", str(client_id))
    stem = re.sub(r"cw$", "", stem)
    return stem.title() or str(client_id)


def _short_name(site_name: str, brand: str) -> str:
    """The site name with the operator's brand taken off the front — "Ramsey", not
    "Whistle Express Car Wash - Ramsey". Everything in the section is already inside one operator,
    so repeating the brand on every row costs width and says nothing."""
    s = str(site_name)
    if brand and s.lower().startswith(brand.lower()):
        s = re.sub(r"^[\s\-–—:·]+", "", s[len(brand):])
    return s or str(site_name)


# =================================================================================================
# Sites, and the coordinate defect
# =================================================================================================

@lru_cache(maxsize=1)
def site_index() -> pd.DataFrame:
    """One row per site: identity, location, opening, and lifetime trading.

    `placeholder_coord` flags the defect described in the module docstring. It is computed on the
    **whole panel**, not per operator, because the marker of a placeholder is that one coordinate
    carries more than one street address — whoever put it there.
    """
    d = _panel()
    g = (d.groupby(["client_id", "site_id"])
           .agg(site_name=("client_name", "first"), address=("address1", "first"),
                state=("state", "first"), postal_code=("postal_code", "first"),
                region=("region", "first"), lat=("lat", "first"), lon=("lon", "first"),
                operational_start=("operational_start", "first"),
                washes=("washes", "sum"), revenue=("revenue", "sum"),
                mem_washes=("mem_wash_count", "sum"), ret_washes=("ret_wash_count", "sum"),
                months=("washes", "size"), first_ym=("ym", "min"), last_ym=("ym", "max"))
           .reset_index())
    g["site_key"] = g.client_id.astype(str) + "___" + g.site_id.astype(str)

    brands = {cid: _brand(sub.site_name) for cid, sub in g.groupby("client_id")}
    g["operator"] = [_operator_label(c, brands[c]) for c in g.client_id]
    g["site"] = [_short_name(n, brands[c]) for n, c in zip(g.site_name, g.client_id)]

    g["in_us"] = g.lat.between(*LAT_RANGE) & g.lon.between(*LON_RANGE)

    key = g.lat.round(6).astype(str) + "," + g.lon.round(6).astype(str)
    n_here = key.map(key.value_counts())
    n_addr = key.map(g.groupby(key).address.nunique())
    g["placeholder_coord"] = (n_here > 1) & (n_addr > 1)
    g["co_located"] = (n_here > 1) & (n_addr == 1)

    # `washes / months * 12` rather than a calendar-year mean: it is comparable across a site that
    # opened in November and one that has traded six years.
    g["washes_per_year"] = g.washes / g.months * MONTHS_IN_YEAR
    g["washes_per_month"] = g.washes / g.months
    g["mem_share"] = g.mem_washes / g.washes.replace(0, np.nan)
    g["asp"] = g.revenue / g.washes.replace(0, np.nan)
    g["opened_ym"] = g.operational_start.map(_parse_start)
    # Fall back to the first month that actually washed a car — a few sites have no start stamp.
    g["opened_ym"] = g.opened_ym.fillna(g.first_ym)
    g["opened"] = g.opened_ym.map(_ym_label)
    return g


def _parse_start(v) -> float:
    """`operational_start` is "MM-YYYY". Returned on the same `ym` scale as the panel."""
    if not isinstance(v, str) or "-" not in v:
        return np.nan
    mm, yyyy = v.split("-", 1)
    try:
        return int(yyyy) * MONTHS_IN_YEAR + (int(mm) - 1)
    except ValueError:
        return np.nan


def _ym_label(ym: float) -> str:
    if pd.isna(ym):
        return "—"
    y, m = divmod(int(ym), MONTHS_IN_YEAR)
    return f"{['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][m]} {y}"


def coord_quality() -> dict:
    """What the coordinate gate dropped, so the app can state it rather than hide it."""
    s = site_index()
    bad = s[s.placeholder_coord]
    worst = (bad.groupby(["lat", "lon"])
                .agg(sites=("site_key", "size"), operator=("operator", "first"),
                     states=("state", "nunique"), addresses=("address", "nunique"))
                .reset_index().sort_values("sites", ascending=False))
    return dict(
        n_sites=int(len(s)),
        n_usable=int((s.in_us & ~s.placeholder_coord).sum()),
        n_placeholder=int(len(bad)),
        n_placeholder_points=int(worst.shape[0]),
        n_outside_box=int((~s.in_us).sum()),
        n_co_located=int(s.co_located.sum()),
        placeholder_washes=float(bad.washes.sum()),
        share_of_washes=float(bad.washes.sum() / s.washes.sum()),
        worst=worst.head(6),
    )


# =================================================================================================
# Clustering
# =================================================================================================

def haversine_matrix(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Full great-circle distance matrix, km. Straight-line — not drive time."""
    la, lo = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    dla = la[:, None] - la[None, :]
    dlo = lo[:, None] - lo[None, :]
    a = np.sin(dla / 2) ** 2 + np.cos(la)[:, None] * np.cos(la)[None, :] * np.sin(dlo / 2) ** 2
    d = 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    np.fill_diagonal(d, 0.0)
    return d


@lru_cache(maxsize=16)
def build(max_km: float = 25.0, min_sites: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster every operator's sites, and return `(clusters, sites)`.

    `sites` is `site_index()` restricted to sites that landed in a qualifying cluster, with
    `cluster_id`, `nearest_km` (distance to the closest sibling) and `open_rank` (1 = the first
    site the operator opened in that place) attached.

    Complete linkage, cut at `max_km`: the returned `diameter_km` is therefore guaranteed
    ≤ `max_km`, which is what makes "one place, one screen" true rather than hopeful.
    """
    s = site_index()
    s = s[s.in_us & ~s.placeholder_coord].copy()

    cl_rows, members = [], []
    for cid, g in s.groupby("client_id"):
        if len(g) < min_sites:
            continue
        g = g.reset_index(drop=True)
        d = haversine_matrix(g.lat.values, g.lon.values)
        if len(g) == 1:
            labels = np.array([1])
        else:
            labels = fcluster(linkage(squareform(d, checks=False), method="complete"),
                              t=float(max_km), criterion="distance")
        for k in np.unique(labels):
            idx = np.where(labels == k)[0]
            if len(idx) < min_sites:
                continue
            sub = d[np.ix_(idx, idx)]
            off = sub[np.triu_indices(len(idx), 1)]
            m = g.iloc[idx].copy()
            cluster_id = f"{cid}::{int(k)}"
            m["cluster_id"] = cluster_id
            # Nearest sibling: smallest off-diagonal entry of each row.
            near = sub.copy()
            np.fill_diagonal(near, np.inf)
            m["nearest_km"] = near.min(axis=1)
            m["open_rank"] = m.opened_ym.rank(method="first").astype(int)
            members.append(m)

            opened = m.opened_ym.dropna()
            cl_rows.append(dict(
                cluster_id=cluster_id, client_id=cid, operator=m.operator.iloc[0],
                state=m.state.mode().iloc[0] if m.state.notna().any() else "—",
                n_sites=int(len(idx)),
                diameter_km=float(sub.max()), median_km=float(np.median(off)),
                nearest_km=float(off[off > 0].min()) if (off > 0).any() else 0.0,
                lat=float(m.lat.mean()), lon=float(m.lon.mean()),
                lat_min=float(m.lat.min()), lat_max=float(m.lat.max()),
                lon_min=float(m.lon.min()), lon_max=float(m.lon.max()),
                washes=float(m.washes.sum()),
                washes_per_year=float(m.washes_per_year.sum()),
                median_site=float(m.washes_per_year.median()),
                first_open=_ym_label(opened.min()) if len(opened) else "—",
                last_open=_ym_label(opened.max()) if len(opened) else "—",
                build_out_months=int(opened.max() - opened.min()) if len(opened) else 0,
                anchor=m.sort_values("opened_ym").site.iloc[0],
                anchor_address=m.sort_values("opened_ym").address.iloc[0],
            ))

    if not cl_rows:
        empty = pd.DataFrame(columns=["cluster_id"])
        return empty, empty

    clusters = pd.DataFrame(cl_rows)
    # An operator can hold several clusters, sometimes several in one state. The label has to stay
    # unique or the picker silently merges two different places.
    place = clusters.operator + " · " + clusters.state
    dupes = place.duplicated(keep=False)
    if dupes.any():
        order = clusters[dupes].assign(place=place[dupes]).sort_values(
            ["place", "n_sites"], ascending=[True, False])
        place.loc[order.index] = order.place + " #" + (order.groupby("place").cumcount() + 1
                                                       ).astype(str)
    clusters["place"] = place
    clusters["label"] = place + " · " + clusters.n_sites.astype(str) + " sites"

    sites = pd.concat(members, ignore_index=True)
    sites = sites.merge(clusters[["cluster_id", "label"]], on="cluster_id", how="left")
    clusters = clusters.sort_values(["n_sites", "washes_per_year"],
                                    ascending=False).reset_index(drop=True)
    return clusters, sites


def headline(max_km: float = 25.0, min_sites: int = 3) -> dict:
    """Estate-wide summary of how much of the business sits inside an operator cluster."""
    clusters, sites = build(max_km, min_sites)
    s = site_index()
    usable = s[s.in_us & ~s.placeholder_coord]
    if clusters.empty:
        return dict(n_clusters=0, n_sites=0, max_km=max_km, min_sites=min_sites)
    return dict(
        n_clusters=int(len(clusters)), n_operators=int(clusters.client_id.nunique()),
        n_sites=int(len(sites)), n_usable=int(len(usable)),
        share_of_sites=float(len(sites) / len(usable)),
        share_of_washes=float(sites.washes.sum() / usable.washes.sum()),
        max_km=max_km, min_sites=min_sites,
        biggest=str(clusters.label.iloc[0]), biggest_n=int(clusters.n_sites.iloc[0]),
        biggest_diam=float(clusters.diameter_km.iloc[0]),
        median_cluster_sites=float(clusters.n_sites.median()),
        median_diameter=float(clusters.diameter_km.median()),
        median_nearest=float(clusters.nearest_km.median()),
        median_gap=float(sites.nearest_km.median()),
        tightest=str(clusters.sort_values("nearest_km").label.iloc[0]),
        tightest_gap=float(clusters.nearest_km.min()),
        median_build_out=float(clusters.build_out_months.median()),
    )


# =================================================================================================
# Inside one cluster
# =================================================================================================

def cluster_sites(cluster_id: str, max_km: float = 25.0, min_sites: int = 3) -> pd.DataFrame:
    """The sites of one cluster, in the order the operator opened them."""
    _, sites = build(max_km, min_sites)
    m = sites[sites.cluster_id == cluster_id].copy()
    return m.sort_values(["open_rank", "site"]).reset_index(drop=True)


def distance_table(cluster_id: str, max_km: float = 25.0,
                   min_sites: int = 3) -> pd.DataFrame:
    """Symmetric site-to-site distance matrix, km, labelled by short site name."""
    m = cluster_sites(cluster_id, max_km, min_sites)
    d = haversine_matrix(m.lat.values, m.lon.values)
    return pd.DataFrame(d, index=m.site.values, columns=m.site.values)


def site_years(cluster_id: str, max_km: float = 25.0, min_sites: int = 3,
               through: int | None = None) -> pd.DataFrame:
    """Washes per site per calendar year, for complete panel years only.

    A site's own opening year is genuinely partial and stays in — that part-year *is* what the site
    did. What is cut is the year the **panel** is only half-collected for (2026), because a
    half-year plotted against full years reads as a collapse that did not happen.
    """
    m = cluster_sites(cluster_id, max_km, min_sites)
    if m.empty:
        return m
    cut = last_complete_year() if through is None else through
    d = _panel()
    d = d[d.site_key.isin(set(m.site_key)) & (d.year <= cut)]
    y = (d.groupby(["site_key", "year"])
           .agg(washes=("washes", "sum"), mem=("mem_wash_count", "sum"),
                ret=("ret_wash_count", "sum"), revenue=("revenue", "sum"),
                months=("washes", "size"))
           .reset_index())
    y = y.merge(m[["site_key", "site", "address", "open_rank", "opened_ym", "cluster_id"]],
                on="site_key", how="left")
    y["opening_year"] = y.year == (y.opened_ym // MONTHS_IN_YEAR)
    y["operating_year"] = y.year - (y.opened_ym // MONTHS_IN_YEAR) + 1
    y["part_year"] = y.months < MONTHS_IN_YEAR
    y.attrs["through"] = cut
    return y.sort_values(["open_rank", "year"]).reset_index(drop=True)


def site_months(cluster_id: str, max_km: float = 25.0, min_sites: int = 3) -> pd.DataFrame:
    """Washes per site per month — the view where one site's opening and its neighbour's dip are
    visible in the same picture. Runs to the end of the panel; a monthly axis has no part-year
    problem to hide."""
    m = cluster_sites(cluster_id, max_km, min_sites)
    if m.empty:
        return m
    d = _panel()
    d = d[d.site_key.isin(set(m.site_key))].copy()
    d["date"] = pd.to_datetime(dict(year=d.year, month=d.month, day=1))
    out = (d.groupby(["site_key", "date"])
             .agg(washes=("washes", "sum"), revenue=("revenue", "sum"), ym=("ym", "first"))
             .reset_index())
    return out.merge(m[["site_key", "site", "open_rank", "opened_ym"]], on="site_key", how="left") \
              .sort_values(["open_rank", "date"]).reset_index(drop=True)


# =================================================================================================
# What happens to the sites already there when the operator adds another one
# =================================================================================================

def opening_effect(cluster_id: str, max_km: float = 25.0, min_sites: int = 3,
                   window: int = 6, settled_months: int = 12) -> pd.DataFrame:
    """For each new site in a cluster: what the *incumbents* did either side of the opening.

    One row per opening that has anything to compare. `incumbent_change` is the incumbents' total
    washes over the `window` months after the opening against the `window` months before it.

    Two guards, because the obvious version of this measurement is wrong:

      • an incumbent counts only once it has `settled_months` of trading behind it. §⓪ measures a
        new wash reaching ~98% of its eventual volume by year 2, so a younger "incumbent" is still
        climbing and would show growth that has nothing to do with the neighbour;
      • `control_change` is the same before/after ratio for the **same operator's sites outside
        this cluster**, over the very same calendar months, held to the identical settled test.
        It absorbs anything seasonal or company-wide. `excess` is the difference.

    Both sides are also **balanced**: a site counts only if it traded in the before window *and*
    the after window, so neither ratio can move because its membership changed. Without that the
    control is badly wrong in one direction — an operator opening a site in one town is usually
    opening sites elsewhere too, and an unbalanced, unsettled control pool reads +13% growth that
    is nothing but its own new sites ramping, which would then be charged to the neighbour as
    cannibalization.

    This is descriptive, not a causal estimate. Openings inside one cluster are months apart, so
    their windows overlap and the events are not independent. §⑤ Competition is where entry is
    identified properly.
    """
    m = cluster_sites(cluster_id, max_km, min_sites)
    if m.empty or len(m) < 2:
        return pd.DataFrame()
    client_id = m.client_id.iloc[0]

    d = _panel()
    in_cluster = set(m.site_key)
    fam = d[d.client_id == client_id]
    starts = site_index().set_index("site_key").opened_ym

    rows = []
    for _, new in m.sort_values("open_rank").iterrows():
        m0 = new.opened_ym
        if pd.isna(m0):
            continue
        settled = set(starts[starts <= m0 - settled_months].index)

        incumbents = (in_cluster & settled) - {new.site_key}
        before, after, n_inc = _balanced_window(d, incumbents, m0, window)
        if before <= 0 or after <= 0:
            continue

        controls = set(fam.site_key.unique()) & settled - in_cluster
        c_before, c_after, n_ctl = _balanced_window(d, controls, m0, window)
        control = (c_after / c_before - 1) if c_before > 0 and c_after > 0 else np.nan

        change = after / before - 1
        rows.append(dict(
            site=new.site, opened=new.opened, opened_ym=float(m0), open_rank=int(new.open_rank),
            n_incumbents=n_inc, nearest_km=float(new.nearest_km),
            incumbent_before=float(before), incumbent_after=float(after),
            incumbent_change=float(change),
            control_sites=n_ctl,
            control_change=float(control) if pd.notna(control) else np.nan,
            excess=float(change - control) if pd.notna(control) else np.nan,
        ))
    return pd.DataFrame(rows)


def _balanced_window(d: pd.DataFrame, keys: set, m0: float,
                     window: int) -> tuple[float, float, int]:
    """Total washes before and after `m0`, over only the sites that traded in *both* windows.

    Returns `(before, after, n_sites)`. Dropping the unbalanced sites is what stops a site that
    opened, or closed, mid-window from being read as the neighbour's doing.
    """
    if not keys:
        return 0.0, 0.0, 0
    sub = d[d.site_key.isin(keys)]
    pre = sub[sub.ym.between(m0 - window, m0 - 1)]
    post = sub[sub.ym.between(m0 + 1, m0 + window)]
    both = set(pre.site_key.unique()) & set(post.site_key.unique())
    if not both:
        return 0.0, 0.0, 0
    return (float(pre[pre.site_key.isin(both)].washes.sum()),
            float(post[post.site_key.isin(both)].washes.sum()), len(both))


def opening_effect_all(max_km: float = 25.0, min_sites: int = 3, window: int = 6,
                       settled_months: int = 12) -> pd.DataFrame:
    """`opening_effect` over every cluster — the estate-wide version of the same descriptive
    measurement, so a single cluster's number can be read against the population."""
    clusters, _ = build(max_km, min_sites)
    out = []
    for cid, label in zip(clusters.cluster_id, clusters.label):
        e = opening_effect(cid, max_km, min_sites, window, settled_months)
        if not e.empty:
            e["cluster_id"], e["label"] = cid, label
            out.append(e)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
