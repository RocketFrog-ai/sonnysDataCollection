"""
Competition — what happens to a site, and to its neighbours, when somebody opens nearby.

**One input file**, `conclusion/data/historical_data_5yrs_monthly.csv` — the monthly wash panel
(2,103 sites, 2020-01 → 2026-06, with each site's `operational_start`, lat/lon and state). Nothing
is joined in from anywhere else, exactly as sections ① ② ③ each read their own single file.

The unit of analysis is an **event**: a site opens, and every already-trading site within a radius
of it is an *incumbent* exposed to that opening.

    two-body    one entrant  ×  one incumbent          — every such pair, not just the nearest
    three-body  one entrant  ×  its two nearest incumbents
    n-body      one incumbent × however many entrants landed on it inside the window

Everything is measured on a symmetric window around the opening month: the incumbent's mean over
the `window` months **before** the opening vs its mean over the `window` months **from** the
opening. The entrant contributes only a post window — it has no past.

Three things this module does that the raw pre/post comparison does not, all of them load-bearing
because §③ of this pack was built on exactly the mistake they prevent:

  • **A counterfactual.** The same before/after change is computed for every *untouched* site —
    trading in the same months, in the same census region, in the same age bracket, with **no**
    opening within the radius anywhere in the window. The difference-in-differences is the
    incumbent's change minus that. A raw −8% means nothing if untouched sites also fell 8%.
  • **An age floor.** A young incumbent is still climbing its own opening ramp, which reads as
    growth that cancels the entrant's damage. `min_incumbent_age_months` excludes them.
  • **A minimum distance.** Sites at ~0 miles from each other are operator handoffs — the same
    physical wash changing hands and reappearing under a new `client_id`. They are not entrants.

## Left-censoring — the one thing to know about the dates

`operational_start` equals the site's first month in the panel for **every** site, and the panel
starts 2020-01. So the 348 sites stamped `2020-01` are "open on or before January 2020", not
"opened in January 2020". They are perfectly good *incumbents* (they are mature by construction)
but they can never be *entrants*. Entrants are therefore openings from **2020-07** onward — the
first month with a full six-month pre-window inside the panel.

Streamlit-free on purpose: the notebook imports the same functions, so the app and the notebook
cannot report different numbers.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PANEL = REPO / "conclusion" / "data" / "historical_data_5yrs_monthly.csv"

# Continental-US bounding box — a handful of rows carry impossible coordinates, and a site sitting
# in the Atlantic would collect spurious neighbours at plausible-looking distances.
LAT_RANGE, LON_RANGE = (20.0, 50.0), (-130.0, -65.0)

EARTH_MILES = 3958.7613

#: What we measure. Total washes is the headline; the retail/membership split is where the story
#: is, because an entrant takes *retail* traffic (a member is on a contract and does not switch on
#: a whim).
METRICS = {
    "total":      ("Total washes",      "washes"),
    "retail":     ("Retail washes",     "retail washes"),
    "membership": ("Membership washes", "membership washes"),
    "revenue":    ("Revenue",           "revenue"),
}

DIST_BINS = [0, 1, 2, 3, 5, 10]
DIST_LABELS = ["<1 mi", "1-2 mi", "2-3 mi", "3-5 mi", "5-10 mi"]

#: Age brackets an incumbent is matched to its controls on. A site's own opening ramp is the single
#: biggest confound in this data (see §③), so a 10-month-old incumbent is never compared against a
#: 5-year-old untouched site.
AGE_BINS = [0, 18, 36, 10_000]
AGE_LABELS = ["<18 mo", "18-36 mo", "36+ mo"]


# =================================================================================================
# Panel → matrices
# =================================================================================================

@lru_cache(maxsize=1)
def _matrices() -> tuple[pd.DataFrame, pd.DatetimeIndex, dict[str, np.ndarray]]:
    """Load the panel once and reshape it to `site × month` matrices, one per metric.

    Every window mean in this module is then a pair of array lookups against a prefix sum, which is
    what makes "recompute the whole thing when the reviewer moves the radius slider" viable at all —
    the naive per-site `DataFrame` filter is ~3,000× slower and the app has 2,782 pairs to do it for.
    """
    d = pd.read_csv(PANEL, low_memory=False)
    d["site_key"] = d.client_id.astype(str) + "___" + d.site_id.astype(str)
    d["ym"] = pd.to_datetime(dict(year=d.year, month=d.month, day=1))

    d["m_retail"] = d.ret_wash_count.fillna(0.0)
    d["m_membership"] = d.mem_wash_count.fillna(0.0)
    d["m_total"] = d.m_retail + d.m_membership
    d["m_revenue"] = d.ret_revenue.fillna(0.0) + d.mem_revenue.fillna(0.0)

    sites = (d.groupby("site_key")
               .agg(operator=("client_name", "first"), state=("state", "first"),
                    region=("region", "first"), zipcode=("postal_code", "first"),
                    address=("address1", "first"), lat=("lat", "first"), lon=("lon", "first"),
                    opened=("operational_start", "first"), first_month=("ym", "min"),
                    last_month=("ym", "max"), months=("ym", "size"))
               .reset_index())
    sites["opened"] = pd.to_datetime(sites.opened, format="%m-%Y", errors="coerce")
    sites = sites[sites.lat.between(*LAT_RANGE) & sites.lon.between(*LON_RANGE)]
    sites = sites.dropna(subset=["opened"]).reset_index(drop=True)

    months = pd.date_range(d.ym.min(), d.ym.max(), freq="MS")
    row = pd.Series(sites.index.values, index=sites.site_key)
    col = pd.Series(np.arange(len(months)), index=months)

    d = d[d.site_key.isin(row.index)]
    ri, ci = row.reindex(d.site_key).to_numpy(), col.reindex(d.ym).to_numpy()

    mats = {}
    for key in METRICS:
        m = np.full((len(sites), len(months)), np.nan)
        m[ri, ci] = d[f"m_{key}"].to_numpy(float)
        mats[key] = m

    sites["open_idx"] = col.reindex(sites.opened).to_numpy()
    # 2020-01 openings are the left-censored pile: "open by then", not "opened then".
    sites["censored"] = sites.opened <= months[0]
    return sites, months, mats


@lru_cache(maxsize=1)
def _prefix() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per metric, (sum, count) prefix arrays with a leading zero column, NaNs treated as absent."""
    _, _, mats = _matrices()
    out = {}
    for key, m in mats.items():
        present = ~np.isnan(m)
        s = np.concatenate([np.zeros((m.shape[0], 1)), np.nancumsum(m, axis=1)], axis=1)
        c = np.concatenate([np.zeros((m.shape[0], 1)), np.cumsum(present, axis=1)], axis=1)
        out[key] = (s, c)
    return out


def _window(key: str, a: int, b: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean and month-count for every site over the half-open month window `[a, b)`.

    Out-of-panel edges are clipped rather than rejected, and the count comes back with them — the
    caller decides whether 4 months of history is enough, this function does not.
    """
    s, c = _prefix()[key]
    a, b = max(a, 0), min(b, s.shape[1] - 1)
    if b <= a:
        n = np.zeros(s.shape[0])
        return np.full(s.shape[0], np.nan), n
    n = c[:, b] - c[:, a]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, (s[:, b] - s[:, a]) / np.maximum(n, 1), np.nan)
    return mean, n


@lru_cache(maxsize=1)
def distances() -> np.ndarray:
    """Great-circle miles between every pair of sites. 2,103² floats — 35 MB, computed once."""
    sites, _, _ = _matrices()
    la, lo = np.radians(sites.lat.to_numpy(float)), np.radians(sites.lon.to_numpy(float))
    dlat, dlon = la[:, None] - la[None, :], lo[:, None] - lo[None, :]
    h = np.sin(dlat / 2) ** 2 + np.cos(la)[:, None] * np.cos(la)[None, :] * np.sin(dlon / 2) ** 2
    return 2 * EARTH_MILES * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def sites() -> pd.DataFrame:
    """One row per site, with `opened`, coordinates and the panel window it covers."""
    s, _, _ = _matrices()
    return s.copy()


def _pct(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Percent change, undefined when the base is zero or missing (not clipped, not floored)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((before > 0) & np.isfinite(before) & np.isfinite(after),
                        (after - before) / np.where(before > 0, before, np.nan) * 100, np.nan)


# =================================================================================================
# The counterfactual
# =================================================================================================

@lru_cache(maxsize=8)
def control_table(radius_mi: float = 10.0, window: int = 6, min_months: int = 4,
                  min_control: int = 10) -> dict[str, pd.DataFrame]:
    """What an **untouched** site did over the same months, by region × age bracket.

    A site is a valid control at month `e` when no opening landed within `radius_mi` of it anywhere
    in `[e - window, e + window]` — including its own. So the control group is doing nothing but
    ageing, in the same calendar months, in the same part of the country.

    Returned as **three nested levels**, because the tightest match is not always populated: in a
    quiet month a single region × age cell can hold two sites, and a two-site median is not a
    counterfactual. `_attach_controls` takes the tightest level with at least `min_control` sites
    behind it and records which one it used in `ctrl_level`, so a reviewer can see when the match
    was loosened rather than have it happen silently.

        region_age   same census region, same age bracket, same month   — the intended match
        age          same age bracket, same month, nationwide
        month        same month, nationwide                             — always populated
    """
    s, months, _ = _matrices()
    n_sites, n_months = len(s), len(months)
    D = distances()
    open_idx = s.open_idx.to_numpy(float)

    # contaminated[j, e] — site j had an opening within the radius somewhere in e ± window
    entrant = np.where(~s.censored.to_numpy() & np.isfinite(open_idx))[0]
    contaminated = np.zeros((n_sites, n_months), dtype=bool)
    for i in entrant:
        near = D[i] <= radius_mi
        lo = max(int(open_idx[i]) - window, 0)
        hi = min(int(open_idx[i]) + window + 1, n_months)
        contaminated[np.ix_(near, np.arange(lo, hi))] = True

    rows = []
    region = s.region.fillna("—").to_numpy()
    for e in range(window, n_months - window + 1):
        clean = ~contaminated[:, e] & (open_idx <= e - window)
        if clean.sum() < 5:
            continue
        age = e - open_idx
        age_band = pd.cut(age, AGE_BINS, labels=AGE_LABELS, right=False)
        rec = {}
        for key in METRICS:
            pre, n_pre = _window(key, e - window, e)
            post, n_post = _window(key, e, e + window)
            ok = clean & (n_pre >= min_months) & (n_post >= min_months)
            rec[key] = np.where(ok, _pct(pre, post), np.nan)
        frame = pd.DataFrame(rec)
        frame["region"], frame["age_band"], frame["event_idx"] = region, age_band, e
        rows.append(frame.dropna(subset=list(METRICS), how="all"))

    if not rows:
        return {}
    allc = pd.concat(rows, ignore_index=True)

    def roll(keys: list[str]) -> pd.DataFrame:
        g = (allc.groupby(keys, observed=True)
                 .agg(n_control=("total", "size"),
                      **{f"ctrl_{k}": (k, "median") for k in METRICS})
                 .reset_index())
        return g[g.n_control >= min_control]

    return {"region_age": roll(["event_idx", "region", "age_band"]),
            "age": roll(["event_idx", "age_band"]),
            "month": roll(["event_idx"])}


def _attach_controls(out: pd.DataFrame, ctrl: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge the counterfactual on, tightest match first, and say which match each row got."""
    levels = [("region_age", ["event_idx", "region", "age_band"]),
              ("age", ["event_idx", "age_band"]), ("month", ["event_idx"])]
    out["ctrl_level"] = None
    for k in METRICS:
        out[f"ctrl_{k}"] = np.nan
    out["n_control"] = np.nan

    for name, keys in levels:
        table = ctrl.get(name)
        if table is None or table.empty:
            continue
        cols = [f"ctrl_{k}" for k in METRICS] + ["n_control"]
        m = out.merge(table.rename(columns={c: f"__{c}" for c in cols}), on=keys, how="left")
        fill = out.ctrl_level.isna() & m[f"__ctrl_total"].notna().to_numpy()
        for c in cols:
            out.loc[fill, c] = m.loc[fill, f"__{c}"].to_numpy()
        out.loc[fill, "ctrl_level"] = name
    return out


# =================================================================================================
# Two-body: one entrant, one incumbent
# =================================================================================================

@lru_cache(maxsize=8)
def pair_events(radius_mi: float = 10.0, window: int = 6, min_months: int = 4,
                min_incumbent_age: int = 12, min_dist_mi: float = 0.2,
                min_pre_volume: float = 500.0, min_entrant_volume: float = 250.0,
                dead_months: int = 2) -> pd.DataFrame:
    """Every (entrant, incumbent) pair inside the radius — one row per pair, not per entrant.

    An incumbent qualifies when it opened at least `min_incumbent_age` months before the entrant,
    has `min_months` of trading in the pre window and in the post window, and sits between
    `min_dist_mi` and `radius_mi` away. The entrant needs only a post window; the columns that
    describe it are `entrant_post_*`.

    Four exclusions, none of them cosmetic — each removes a way of manufacturing a huge fake
    effect, and together they take out ~7% of pairs and move the headline by 0.2 pp, which is the
    reassuring result:

      `min_dist_mi`         Below ~0.2 miles the "pair" is one physical car wash whose operator
                            changed, reappearing under a new `client_id` — the old key dies the
                            month the new one starts, reading as a 100% collapse caused by an
                            entrant 30 feet away.
      `min_pre_volume`      Total washes a month the incumbent must have been doing to count as
                            trading. 2% of the panel's site-months sit under 100 washes and a few
                            are negative; a percent change on a base of 11 washes is noise with a
                            big number attached.
      `min_entrant_volume`  Same test on the entrant. If the "opening" never washes a car, no
                            competitor actually arrived — the row is a placeholder, and pairing it
                            with an incumbent that happened to close is how you get a −79% effect
                            caused by nothing.
      `dead_months`         Months in the post window at under 5% of the incumbent's own past. Two
                            or more and the site stopped operating: the series goes to a literal
                            zero and stays there (verified by eye on the offenders). A wash two
                            miles away does not do that. Deliberately **not** a threshold on the
                            window average — that would also throw out an incumbent that genuinely
                            halved, which is the finding, not the artefact. §② excluded collapsed
                            sites on the same reasoning.

    Every row carries both the raw incumbent change and `did_*` — that change minus what untouched
    sites in the same region, age bracket and calendar months did. Counts of what each exclusion
    removed are on `.attrs`.
    """
    s, months, mats = _matrices()
    D = distances()
    open_idx = s.open_idx.to_numpy(float)
    n_months = len(months)
    total = mats["total"]
    # A 3-month window cannot contain 4 months of trading. Clamping here rather than rejecting the
    # combination keeps the window slider usable across its whole range.
    min_months = min(min_months, window)

    entrants = np.where(~s.censored.to_numpy() & (open_idx >= window))[0]
    by_month: dict[int, list[int]] = {}
    for i in entrants:
        by_month.setdefault(int(open_idx[i]), []).append(i)

    rows: list[pd.DataFrame] = []
    skipped_entrants: list[int] = []
    for e, ents in sorted(by_month.items()):
        if e + min_months > n_months:
            continue
        stats = {}
        for key in METRICS:
            pre, n_pre = _window(key, e - window, e)
            post, n_post = _window(key, e, e + window)
            stats[key] = (pre, post, n_pre, n_post)
        n_pre_all, n_post_all = stats["total"][2], stats["total"][3]
        pre_all = stats["total"][0]
        # Months in the post window at under 5% of the site's own past — a site that stopped, not
        # one that lost customers. Counted on the raw months, never on the window average.
        post_slice = total[:, e:min(e + window, n_months)]
        with np.errstate(invalid="ignore"):
            n_dead = np.nansum(post_slice < 0.05 * pre_all[:, None], axis=1)
        eligible = ((open_idx <= e - min_incumbent_age) & (n_pre_all >= min_months) &
                    (n_post_all >= min_months) & (pre_all >= min_pre_volume) &
                    (n_dead < dead_months))

        for i in ents:
            if not (stats["total"][3][i] >= min_months
                    and stats["total"][1][i] >= min_entrant_volume):
                skipped_entrants.append(i)      # nothing opened here that washes cars
                continue
            d = D[i]
            js = np.where(eligible & (d >= min_dist_mi) & (d <= radius_mi))[0]
            if len(js) == 0:
                continue
            rec = pd.DataFrame({
                "entrant": s.site_key.to_numpy()[i], "incumbent": s.site_key.to_numpy()[js],
                "distance_mi": d[js], "event_idx": e, "event_month": months[e],
                "entrant_opened": s.opened.to_numpy()[i], "incumbent_opened": s.opened.to_numpy()[js],
                "incumbent_age_mo": (e - open_idx[js]).astype(int),
                "entrant_operator": s.operator.to_numpy()[i],
                "incumbent_operator": s.operator.to_numpy()[js],
                "entrant_address": s.address.to_numpy()[i],
                "incumbent_address": s.address.to_numpy()[js],
                "state": s.state.to_numpy()[js], "region": s.region.to_numpy()[js],
                "entrant_lat": s.lat.to_numpy()[i], "entrant_lon": s.lon.to_numpy()[i],
                "incumbent_lat": s.lat.to_numpy()[js], "incumbent_lon": s.lon.to_numpy()[js],
                "n_pre": n_pre_all[js].astype(int), "n_post": n_post_all[js].astype(int),
                "same_operator": s.operator.to_numpy()[js] == s.operator.to_numpy()[i],
            })
            for key, (pre, post, _, n_post_k) in stats.items():
                rec[f"pre_{key}"] = pre[js]
                rec[f"post_{key}"] = post[js]
                rec[f"pct_{key}"] = _pct(pre[js], post[js])
                ent_post = post[i] if n_post_k[i] >= min_months else np.nan
                rec[f"entrant_post_{key}"] = ent_post
                combined = np.where(np.isfinite(ent_post), post[js] + ent_post, np.nan)
                rec[f"combined_{key}"] = combined
                rec[f"combined_pct_{key}"] = _pct(pre[js], combined)
            rows.append(rec)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)

    out["distance_band"] = pd.cut(out.distance_mi, DIST_BINS, labels=DIST_LABELS, right=False)
    out["age_band"] = pd.cut(out.incumbent_age_mo, AGE_BINS, labels=AGE_LABELS, right=False)

    out = _attach_controls(out, control_table(radius_mi, window, min_months))
    for key in METRICS:
        out[f"did_{key}"] = out[f"pct_{key}"] - out[f"ctrl_{key}"]

    out["regime"] = [_regime(a, b) for a, b in zip(out.pct_total, out.combined_pct_total)]
    out["label"] = (out.incumbent_operator.astype(str) + " (" + out.state.astype(str) + ") ← "
                    + out.entrant_operator.astype(str) + ", "
                    + out.distance_mi.round(1).astype(str) + " mi, "
                    + out.event_month.dt.strftime("%b %Y"))
    out = out.sort_values(["distance_mi", "event_month"]).reset_index(drop=True)
    out.attrs.update(n_skipped_entrants=len(set(skipped_entrants)),
                     min_pre_volume=min_pre_volume, min_entrant_volume=min_entrant_volume,
                     dead_months=dead_months, min_dist_mi=min_dist_mi, radius_mi=radius_mi,
                     window=window, min_incumbent_age=min_incumbent_age)
    return out


def _regime(incumbent_pct: float, combined_pct: float) -> str:
    """The four ways an opening can land, on the ±5% band the archive analysis used.

    Read the pair as one market: did the incumbent lose, and did the two of them together gain?
    """
    if not np.isfinite(incumbent_pct) or not np.isfinite(combined_pct):
        return "unknown"
    if incumbent_pct <= -5 and combined_pct <= 5:
        return "Pure cannibalisation"
    if incumbent_pct <= -5 and combined_pct > 5:
        return "Cannibalisation + growth"
    if incumbent_pct > -5 and combined_pct > 5:
        return "Market expansion"
    return "Flat / mixed"


def pair_headline(pairs: pd.DataFrame, metric: str = "total") -> dict:
    """The numbers the section leads with, all medians — the distribution has a long tail."""
    if pairs.empty:
        return {}
    p = pairs
    near = p[p.distance_mi < 3]
    far = p[p.distance_mi >= 5]
    return dict(
        n_pairs=int(len(p)), n_entrants=int(p.entrant.nunique()),
        n_incumbents=int(p.incumbent.nunique()),
        raw=float(p[f"pct_{metric}"].median()),
        did=float(p[f"did_{metric}"].median()),
        ctrl=float(p[f"ctrl_{metric}"].median()),
        share_down=float((p[f"pct_{metric}"] < 0).mean()),
        share_down_did=float((p[f"did_{metric}"] < 0).mean()),
        near_did=float(near[f"did_{metric}"].median()) if len(near) else np.nan,
        far_did=float(far[f"did_{metric}"].median()) if len(far) else np.nan,
        n_near=int(len(near)), n_far=int(len(far)),
        combined=float(p[f"combined_pct_{metric}"].median()),
        entrant_vs_incumbent=float((p[f"entrant_post_{metric}"] / p[f"pre_{metric}"]).median()),
        median_distance=float(p.distance_mi.median()),
    )


def by_distance(pairs: pd.DataFrame, metric: str = "total") -> pd.DataFrame:
    """Incumbent effect per distance band — raw, counterfactual, and the difference."""
    if pairs.empty:
        return pd.DataFrame()
    g = (pairs.groupby("distance_band", observed=True)
              .agg(pairs=("incumbent", "size"),
                   raw=(f"pct_{metric}", "median"),
                   control=(f"ctrl_{metric}", "median"),
                   did=(f"did_{metric}", "median"),
                   p25=(f"did_{metric}", lambda x: x.quantile(.25)),
                   p75=(f"did_{metric}", lambda x: x.quantile(.75)),
                   share_down=(f"did_{metric}", lambda x: (x < 0).mean()))
              .reset_index())
    return g[g.pairs > 0]


def regime_mix(pairs: pd.DataFrame) -> pd.DataFrame:
    """How the pairs split across the four outcomes, in a fixed order (not by count)."""
    order = ["Pure cannibalisation", "Cannibalisation + growth", "Market expansion",
             "Flat / mixed", "unknown"]
    counts = pairs.regime.value_counts()
    out = pd.DataFrame({"regime": order})
    out["pairs"] = out.regime.map(counts).fillna(0).astype(int)
    out["share"] = out.pairs / max(out.pairs.sum(), 1)
    return out[out.pairs > 0].reset_index(drop=True)


# =================================================================================================
# Event-time trajectories
# =================================================================================================

def event_profile(pairs: pd.DataFrame, metric: str = "total", span: int = 12,
                  window: int = 6, group: str | None = "distance_band") -> pd.DataFrame:
    """Median incumbent trajectory in event time, rebased so its own pre-window = 100.

    Rebasing is what makes a 4,000-wash site and a 30,000-wash site averageable at all. Month 0 is
    the entrant's first month. `group=None` gives one curve over everything.
    """
    _, _, mats = _matrices()
    s, months, _ = _matrices()
    idx = pd.Series(np.arange(len(s)), index=s.site_key)
    m = mats[metric]

    inc = idx.reindex(pairs.incumbent).to_numpy()
    e = pairs.event_idx.to_numpy()
    keys = pairs[group].astype(str).to_numpy() if group else np.full(len(pairs), "all")

    base = np.full(len(pairs), np.nan)
    for ev in np.unique(e):
        pre, n_pre = _window(metric, ev - window, ev)
        sel = e == ev
        base[sel] = np.where(n_pre[inc[sel]] > 0, pre[inc[sel]], np.nan)

    rows = []
    for t in range(-span, span + 1):
        col = e + t
        ok = (col >= 0) & (col < m.shape[1]) & np.isfinite(base) & (base > 0)
        vals = np.full(len(pairs), np.nan)
        vals[ok] = m[inc[ok], col[ok]] / base[ok] * 100
        rows.append(pd.DataFrame({"offset": t, "group": keys, "value": vals}))
    long = pd.concat(rows, ignore_index=True).dropna(subset=["value"])
    return (long.groupby(["group", "offset"])
                .agg(median=("value", "median"), p25=("value", lambda x: x.quantile(.25)),
                     p75=("value", lambda x: x.quantile(.75)), n=("value", "size"))
                .reset_index())


def entrant_profile(pairs: pd.DataFrame, metric: str = "total", span: int = 12,
                    window: int = 6) -> pd.DataFrame:
    """The entrant's own curve on the incumbent's scale — its washes as a % of what the incumbent
    was doing before it opened. Plotted on the same axis, the two curves answer "did the entrant
    take what the incumbent lost, or did it find new cars?" by eye."""
    s, months, mats = _matrices()
    idx = pd.Series(np.arange(len(s)), index=s.site_key)
    m = mats[metric]
    ent, inc = idx.reindex(pairs.entrant).to_numpy(), idx.reindex(pairs.incumbent).to_numpy()
    e = pairs.event_idx.to_numpy()

    base = np.full(len(pairs), np.nan)
    for ev in np.unique(e):
        pre, n_pre = _window(metric, ev - window, ev)
        sel = e == ev
        base[sel] = np.where(n_pre[inc[sel]] > 0, pre[inc[sel]], np.nan)

    rows = []
    for t in range(0, span + 1):
        col = e + t
        ok = (col >= 0) & (col < m.shape[1]) & np.isfinite(base) & (base > 0)
        vals = np.full(len(pairs), np.nan)
        vals[ok] = m[ent[ok], col[ok]] / base[ok] * 100
        rows.append(pd.DataFrame({"offset": t, "value": vals}))
    long = pd.concat(rows, ignore_index=True).dropna(subset=["value"])
    return (long.groupby("offset")
                .agg(median=("value", "median"), p25=("value", lambda x: x.quantile(.25)),
                     p75=("value", lambda x: x.quantile(.75)), n=("value", "size")).reset_index())


def site_series(site_key: str, metric: str = "total") -> pd.DataFrame:
    """One site's actual monthly series — no rebasing, no smoothing. What the case charts draw."""
    s, months, mats = _matrices()
    i = int(pd.Series(np.arange(len(s)), index=s.site_key)[site_key])
    return pd.DataFrame({"month": months, "value": mats[metric][i]}).dropna(subset=["value"])


# =================================================================================================
# Three-body: one entrant, its two nearest incumbents
# =================================================================================================

def triple_events(pairs: pd.DataFrame) -> pd.DataFrame:
    """Entrants that landed on **two** incumbents at once, as one row per entrant.

    The two are ranked by distance, so `near_*` and `far_*` are always the closer and the further.
    That ranking is the whole point: if proximity is what does the damage, the near incumbent must
    lose more than the far one *within the same event*, which controls for the market, the calendar
    and the entrant all at once — a comparison the two-body distance chart cannot make.
    """
    if pairs.empty:
        return pd.DataFrame()
    p = pairs.sort_values(["entrant", "distance_mi"])
    g = p.groupby("entrant")
    ranked = p[g.cumcount() < 2].copy()
    ranked["role"] = np.where(g.cumcount()[ranked.index] == 0, "near", "far")
    keep = ranked.entrant.value_counts()
    ranked = ranked[ranked.entrant.isin(keep[keep == 2].index)]
    if ranked.empty:
        return pd.DataFrame()

    wide = ranked.pivot(index="entrant", columns="role")
    cols = ["incumbent", "incumbent_operator", "incumbent_lat", "incumbent_lon", "distance_mi",
            "incumbent_age_mo", "incumbent_opened", "state"] + \
           [f"{p_}_{k}" for k in METRICS for p_ in ("pct", "did", "pre", "post")]
    out = pd.DataFrame(index=wide.index)
    for c in cols:
        for role in ("near", "far"):
            out[f"{role}_{c}"] = wide[(c, role)]
    flat = ranked[ranked.role == "near"].set_index("entrant")
    for c in ["event_month", "event_idx", "entrant_operator", "entrant_lat", "entrant_lon",
              "region", "state"] + [f"entrant_post_{k}" for k in METRICS]:
        out[c] = flat[c]
    out["between_mi"] = _between_distance(out)

    for k in METRICS:
        out[f"market_pre_{k}"] = out[f"near_pre_{k}"] + out[f"far_pre_{k}"]
        out[f"market_post_{k}"] = (out[f"near_post_{k}"] + out[f"far_post_{k}"]
                                   + out[f"entrant_post_{k}"])
        out[f"market_pct_{k}"] = _pct(out[f"market_pre_{k}"].to_numpy(),
                                      out[f"market_post_{k}"].to_numpy())
    out["label"] = (out.entrant_operator.astype(str) + " opens in " + out.state.astype(str)
                    + " · " + out.event_month.dt.strftime("%b %Y") + " · nearest "
                    + out.near_distance_mi.round(1).astype(str) + " mi / "
                    + out.far_distance_mi.round(1).astype(str) + " mi")
    return out.reset_index().sort_values("event_month").reset_index(drop=True)


def _between_distance(out: pd.DataFrame) -> np.ndarray:
    """How far apart the two incumbents are — a triple where they sit on top of each other is a
    different market shape from one where the entrant splits them."""
    la1, lo1 = np.radians(out.near_incumbent_lat.to_numpy(float)), np.radians(out.near_incumbent_lon.to_numpy(float))
    la2, lo2 = np.radians(out.far_incumbent_lat.to_numpy(float)), np.radians(out.far_incumbent_lon.to_numpy(float))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * EARTH_MILES * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def triple_headline(triples: pd.DataFrame, metric: str = "total",
                    close_mi: float = 2.0) -> dict:
    """Near vs far, paired — the difference is taken **within** each event, then the median.

    `close_*` repeats the test on the triples where the near incumbent is genuinely close. The
    full-sample version compares a median 3.6-mile incumbent against a median 5.8-mile one, and the
    two-body distance curve says the damage is essentially over by 3 miles — so a null on the full
    sample is what that curve predicts, not evidence against it. The close cut is the real test.
    """
    if triples.empty:
        return {}
    d = (triples[f"near_did_{metric}"] - triples[f"far_did_{metric}"]).dropna()
    close = triples[triples.near_distance_mi < close_mi]
    dc = (close[f"near_did_{metric}"] - close[f"far_did_{metric}"]).dropna()
    return dict(
        n=int(len(triples)),
        near=float(triples[f"near_did_{metric}"].median()),
        far=float(triples[f"far_did_{metric}"].median()),
        gap=float(d.median()) if len(d) else np.nan,
        share_near_worse=float((d < 0).mean()) if len(d) else np.nan,
        n_paired=int(len(d)),
        market=float(triples[f"market_pct_{metric}"].median()),
        near_mi=float(triples.near_distance_mi.median()),
        far_mi=float(triples.far_distance_mi.median()),
        close_mi=close_mi, n_close=int(len(dc)),
        close_near=float(close[f"near_did_{metric}"].median()) if len(close) else np.nan,
        close_far=float(close[f"far_did_{metric}"].median()) if len(close) else np.nan,
        close_gap=float(dc.median()) if len(dc) else np.nan,
        close_share_near_worse=float((dc < 0).mean()) if len(dc) else np.nan,
        close_far_mi=float(close.far_distance_mi.median()) if len(close) else np.nan,
    )


# =================================================================================================
# The entrant's own side of it
# =================================================================================================

def entrant_outcomes(pairs: pd.DataFrame, metric: str = "total", window: int = 6,
                     mature_from: int = 12, mature_to: int = 24) -> pd.DataFrame:
    """One row per **entrant**: how it did, against how crowded the market it walked into was.

    Two volumes, because they answer different questions. `opening` is the same post window the
    incumbent side is measured on — comparable, but it is the bottom of the entrant's ramp.
    `mature` is months 12–24, which for a car wash is roughly its settled level (§⓪: a site is at
    ~98% of its eventual volume by year two), and is missing for anything opened in the last two
    years.
    """
    if pairs.empty:
        return pd.DataFrame()
    s, months, _ = _matrices()
    idx = pd.Series(np.arange(len(s)), index=s.site_key)

    g = (pairs.groupby("entrant")
              .agg(neighbours=("incumbent", "nunique"), nearest_mi=("distance_mi", "min"),
                   event_idx=("event_idx", "first"), event_month=("event_month", "first"),
                   operator=("entrant_operator", "first"), state=("state", "first"),
                   region=("region", "first"), lat=("entrant_lat", "first"),
                   lon=("entrant_lon", "first"),
                   opening=(f"entrant_post_{metric}", "first"))
              .reset_index())
    g["neighbours_3mi"] = (pairs[pairs.distance_mi < 3].groupby("entrant").incumbent.nunique()
                           .reindex(g.entrant).fillna(0).to_numpy())

    i = idx.reindex(g.entrant).to_numpy()
    mature = np.full(len(g), np.nan)
    for ev in np.unique(g.event_idx):
        mean, n = _window(metric, ev + mature_from, ev + mature_to)
        sel = g.event_idx.to_numpy() == ev
        mature[sel] = np.where(n[i[sel]] >= 6, mean[i[sel]], np.nan)
    g["mature"] = mature
    g["nearest_band"] = pd.cut(g.nearest_mi, DIST_BINS, labels=DIST_LABELS, right=False)
    g["crowding"] = pd.cut(g.neighbours_3mi, [0, 1, 2, 3, 100],
                           labels=["none <3 mi", "1", "2", "3+"], right=False)
    return g


def entrant_by_crowding(entrants: pd.DataFrame, by: str = "nearest_band") -> pd.DataFrame:
    """Median entrant volume by how close / how many the competition was."""
    if entrants.empty:
        return pd.DataFrame()
    out = (entrants.groupby(by, observed=True)
                   .agg(entrants=("entrant", "size"), opening=("opening", "median"),
                        mature=("mature", "median"), n_mature=("mature", "count"))
                   .reset_index())
    return out[out.entrants > 0]


# =================================================================================================
# N-body: how many entrants can one market absorb
# =================================================================================================

def saturation(radius_mi: float = 10.0, window: int = 6, min_months: int = 4,
               min_incumbent_age: int = 12, metric: str = "total",
               max_n: int = 4) -> pd.DataFrame:
    """One incumbent-month per row, grouped by **how many** entrants landed inside the window.

    The two-body view treats every pair as its own event, so a site hit by three openings in one
    quarter appears three times, each read as a single opening. This is the same data re-cut so
    that a site hit three times is counted once, against three.
    """
    pairs = pair_events(radius_mi, window, min_months, min_incumbent_age)
    if pairs.empty:
        return pd.DataFrame()
    # One row per (incumbent, event month), carrying the count of entrants that share it. Pairs
    # from the same incumbent within `window` months of each other are the same exposure.
    p = pairs.sort_values(["incumbent", "event_idx"]).copy()
    p["burst"] = (p.groupby("incumbent").event_idx.diff().fillna(999) > window).groupby(
        p.incumbent).cumsum()
    g = (p.groupby(["incumbent", "burst"])
           .agg(entrants=("entrant", "nunique"),
                did=(f"did_{metric}", "median"), raw=(f"pct_{metric}", "median"),
                nearest_mi=("distance_mi", "min"), event_month=("event_month", "first"))
           .reset_index())
    g["entrants_capped"] = np.minimum(g.entrants, max_n)
    out = (g.groupby("entrants_capped")
             .agg(incumbents=("incumbent", "size"), did=("did", "median"), raw=("raw", "median"),
                  p25=("did", lambda x: x.quantile(.25)), p75=("did", lambda x: x.quantile(.75)),
                  nearest_mi=("nearest_mi", "median"))
             .reset_index())
    out["label"] = [f"{int(n)}" + ("+" if n == max_n else "") for n in out.entrants_capped]
    return out
