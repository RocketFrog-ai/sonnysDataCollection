"""
Section 2 — proforma backtest. The shared data + analysis layer.

ONE copy of the math, imported by BOTH consumers so they can never disagree:
  • conclusion/demo/app.py                          (Streamlit, for the CIO)
  • conclusion/notebook/conclusions.ipynb           (static plots / working)

The question: **when the Excel proforma projected a wash count for a site we then built, how close
did it land?** And if it missed, does our model do better on the same sites?

Data: `conclusion/data/n70_backtest_dataset.csv` — one row per site for the 70 mature, matched
sites, carrying every proforma input, all three forecasters, the actual washes, and both tunnel
lengths (the formula's and the one actually built) side by side.

On tunnel length, verified against this file:
  • `tunnel_length_ft` is exactly `year5_max_hourly` — the sizing formula with **no +20 added**:
    one foot of tunnel per car per hour of the proforma's own year-5 peak projection;
  • `tunnel_length_m` is that same formula figure in metres (it matches ft x 0.3048 to within 0.5%);
  • `tunnel_length_actual_m` is the tunnel that was **actually built**, in metres, for 61 of the 70.

So the file lets us ask two separate questions: did the proforma get the *volume* right, and did the
length formula it drives get the *tunnel* right.

Nothing here is Streamlit-aware, so the notebook imports it unchanged.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
DATASET = REPO / "conclusion" / "data" / "n70_backtest_dataset.csv"

# The three forecasters, in the order they are always presented.
FORECASTERS = [
    ("proforma", "Proforma (Excel)", "proforma_y{y}", "proforma_y5",
     "The projection the site was underwritten on."),
    ("coldstart", "Cold-start v15", "coldstart_v15_y{y}", "coldstart_v15_y5",
     "Location-only model — knows where the site is, nothing about the build."),
    ("model5", "Model 5 (ensemble)", "model5_y{y}", "model5_mature",
     "Super-ensemble, scored leave-one-site-out so the site never sees itself."),
]

# The 10 factors the proforma scores a site on, plus the two cumulative roll-ups.
FACTOR_COLS = ["area_profile", "nearest_competition", "weekly_hours", "type_of_site",
               "site_accessibility", "entrance_stack_up", "free_vacuum_slots", "pay_stations",
               "visibility", "traffic_speed"]

# A site whose mature washes fell below 40% of its own early peak has closed or stopped reporting;
# it is not a forecasting miss. The dataset README names both cases explicitly.
COLLAPSE_RATIO = 0.40

PROVENANCE = [
    ("Actual washes", "MEASURED",
     "Recomputed live from proforma/data/panel/main-data-v2-stitched.csv — mean monthly washes over "
     "the observed (non-imputed) months of each operating year, aligned to the site's TRUE open "
     "date. Reconciled cell-for-cell against the panel (0 mismatches)."),
    ("Proforma projection", "MEASURED (as proposed)",
     "`proforma_y1..5`, extracted from the 70 original proforma Excel files."),
    ("Proforma factor scores", "MEASURED (as proposed)",
     "The 10 `factor_*_score` columns and the demographics block, as the proforma itself scored "
     "them — the inputs that drove the projection."),
    ("Cold-start v15", "MODEL OUTPUT",
     "Location-only forecaster from ensemble/results/ensemble_features.csv."),
    ("Model 5", "MODEL OUTPUT",
     "Super-ensemble, leave-one-site-out (`model5_loso.csv`) — each site is predicted by a model "
     "that never saw it, so this is an honest out-of-sample comparison."),
    ("Collapse filter", "JUDGEMENT",
     "2 of 70 sites (Splash Car Wash, Rock N Roll Stuart) collapsed far below their own early peak "
     "— closure or stopped reporting, not a forecast miss. Excluded from maturity scoring; the "
     "toggle in the sidebar puts them back."),
]


# =================================================================================================
# loading
# =================================================================================================
def load(drop_collapsed: bool = True) -> pd.DataFrame:
    """One row per site, with the collapse flag applied."""
    d = pd.read_csv(DATASET, low_memory=False)
    peak = d[["actual_y1", "actual_y2", "actual_y3"]].max(axis=1)
    d["early_peak"] = peak
    d["collapsed"] = d.actual_mature_wash < COLLAPSE_RATIO * peak
    d["site_key"] = d.client_id.astype(str) + "___" + d.site_id.astype(str)
    return d[~d.collapsed].copy() if drop_collapsed else d


# =================================================================================================
# scoring
# =================================================================================================
def _mask(pred: pd.Series, act: pd.Series) -> pd.Series:
    return (act > 0) & act.notna() & pred.notna()


def mdape(pred: pd.Series, act: pd.Series) -> float:
    """Median absolute percentage error — the headline accuracy number.

    Median, not mean: one site whose proforma projected 4x reality would otherwise carry the score.
    """
    m = _mask(pred, act)
    return float(np.median(np.abs(pred[m] - act[m]) / act[m]) * 100) if m.any() else np.nan


def bias(pred: pd.Series, act: pd.Series) -> float:
    """Median predicted/actual. >1 = over-projects, <1 = under-projects, 1.0 = unbiased."""
    m = _mask(pred, act)
    return float(np.median(pred[m] / act[m])) if m.any() else np.nan


def scorecard(d: pd.DataFrame) -> pd.DataFrame:
    """The three forecasters against actual mature washes, on identical sites."""
    a = d.actual_mature_wash
    rows = []
    for key, label, _, mature_col, blurb in FORECASTERS:
        p = d[mature_col]
        m = _mask(p, a)
        ratio = (p[m] / a[m])
        rows.append(dict(key=key, forecaster=label, n=int(m.sum()),
                         mdape=mdape(p, a), bias=bias(p, a),
                         over_share=float((ratio > 1).mean()),
                         p10=float(ratio.quantile(.10)), p90=float(ratio.quantile(.90)),
                         iqr=float(ratio.quantile(.75) - ratio.quantile(.25)),
                         within_25=float((np.abs(ratio - 1) <= 0.25).mean()),
                         blurb=blurb))
    return pd.DataFrame(rows)


def by_year(d: pd.DataFrame, full_years_only: bool = True) -> pd.DataFrame:
    """Accuracy per operating year.

    `full_years_only` keeps sites with 12 observed months in that year — a partial year is
    rate-extrapolated from as little as one month and is noisy as a scoring target.
    """
    rows = []
    for y in range(1, 6):
        a = d[f"actual_y{y}"]
        m = a.notna() & (a > 0)
        if full_years_only:
            m &= d[f"actual_nobs_y{y}"] >= 12
        for key, label, tmpl, _, _ in FORECASTERS:
            p = d[tmpl.format(y=y)]
            rows.append(dict(year=y, key=key, forecaster=label, n=int((m & p.notna()).sum()),
                             mdape=mdape(p[m], a[m]), bias=bias(p[m], a[m])))
    return pd.DataFrame(rows)


def projection_pairs(d: pd.DataFrame, year: int | None = None, full_years_only: bool = True,
                     forecaster: str = "proforma") -> pd.DataFrame:
    """One row per site: what was projected **for a given operating year** vs what it actually did.

    `year=None` is the maturity comparison — the proforma's year-5 number against the site's settled
    volume, which is the number the site was underwritten on. `year=1..5` aligns the two on the same
    operating year instead, so year 1 is judged against the projection *for year 1* rather than
    against a five-year-out promise. The two answer different questions: maturity asks "was the
    investment case right", year-by-year asks "was the trajectory right".

    `full_years_only` keeps sites with all 12 months observed in that year. A part-year is
    rate-extrapolated from as little as one month, and against a projection that is noise.

    Summary statistics come back on `.attrs` so the caller never re-derives them from the frame.
    """
    tmpl, mature_col = {k: (t, m) for k, _, t, m, _ in FORECASTERS}[forecaster]
    if year is None:
        a, p = d.actual_mature_wash, d[mature_col]
        label, sub = "At maturity", "the site's settled volume vs the year-5 projection"
    else:
        a, p = d[f"actual_y{year}"], d[tmpl.format(y=year)]
        label = f"Operating year {year}"
        sub = f"what the site washed in year {year} vs what was projected for year {year}"

    m = _mask(p, a)
    if year is not None and full_years_only:
        m &= d[f"actual_nobs_y{year}"] >= 12

    out = d.loc[m, ["site_key", "client_name", "state", "open_year", "proforma_type"]].copy()
    out["actual"], out["projected"] = a[m], p[m]
    out["ratio"] = out.projected / out.actual
    out["over"] = out.ratio > 1

    r = out.ratio
    out.attrs.update(
        label=label, sub=sub, year=year, n=int(len(out)),
        n_dropped=int((_mask(p, a)).sum() - len(out)),
        mdape=mdape(p[m], a[m]), bias=float(r.median()) if len(r) else np.nan,
        over_share=float(out.over.mean()) if len(out) else np.nan,
        p90=float(r.quantile(.90)) if len(r) else np.nan,
        p10=float(r.quantile(.10)) if len(r) else np.nan,
        within_25=float((np.abs(r - 1) <= 0.25).mean()) if len(r) else np.nan)
    return out


def head_to_head(d: pd.DataFrame, a_col: str = "model5_mature",
                 b_col: str = "proforma_y5") -> dict:
    """Paired test on the same sites: how often is `a` closer than `b`, and is that luck?"""
    act = d.actual_mature_wash
    m = _mask(d[a_col], act) & _mask(d[b_col], act)
    ea = (d.loc[m, a_col] - act[m]).abs() / act[m]
    eb = (d.loc[m, b_col] - act[m]).abs() / act[m]
    wins = int((ea < eb).sum())
    n = int(m.sum())
    return dict(n=n, wins=wins, win_rate=wins / n if n else np.nan,
                binom_p=float(stats.binomtest(wins, n).pvalue) if n else np.nan,
                wilcoxon_p=float(stats.wilcoxon(ea, eb).pvalue) if n > 5 else np.nan)


def factor_table(d: pd.DataFrame) -> pd.DataFrame:
    """Does each proforma factor actually track the washes that showed up?

    Spearman (rank) rather than Pearson: most factors are 3-5 level ordinal choices, and we care
    whether a better score means more washes, not whether the relationship is linear.
    """
    a = d.actual_mature_wash
    rows = []
    for f in FACTOR_COLS:
        col = f"factor_{f}_score"
        if col not in d.columns or d[col].nunique() < 2:
            continue
        rho, p = stats.spearmanr(d[col], a)
        rows.append(dict(factor=f.replace("_", " "), kind="Site factor",
                         levels=int(d[col].nunique()), rho=float(rho), p=float(p)))
    for col, lab, kind in [("cumulative_site_score", "cumulative site score", "Roll-up"),
                           ("cumulative_demographic_score", "cumulative demographic", "Roll-up"),
                           ("traffic_count", "traffic count", "Market"),
                           ("demog_avg_household_size_value", "avg household size", "Market"),
                           ("demog_pct_hh_income_35k_value", "% hh income $35k+", "Market")]:
        if col not in d.columns:
            continue
        rho, p = stats.spearmanr(d[col], a)
        rows.append(dict(factor=lab, kind=kind, levels=int(d[col].nunique()),
                         rho=float(rho), p=float(p)))
    out = pd.DataFrame(rows).sort_values("rho", ascending=False).reset_index(drop=True)
    # Benjamini-Hochberg across the whole family — 15 correlations on 68 sites will throw up a
    # "significant" one by chance otherwise.
    m = len(out)
    ranks = out.p.rank(method="first")
    out["q"] = (out.p * m / ranks).clip(upper=1.0)
    out["signif"] = np.where(out.q < 0.05, "yes", np.where(out.q < 0.10, "marginal", "no"))
    return out


def headline(d: pd.DataFrame) -> dict:
    """The numbers the CIO reads first."""
    sc = scorecard(d).set_index("key")
    h2h = head_to_head(d)
    return dict(
        n_sites=int(len(d)),
        proforma_mdape=float(sc.loc["proforma", "mdape"]),
        proforma_bias=float(sc.loc["proforma", "bias"]),
        proforma_over_share=float(sc.loc["proforma", "over_share"]),
        proforma_p90=float(sc.loc["proforma", "p90"]),
        model5_mdape=float(sc.loc["model5", "mdape"]),
        model5_bias=float(sc.loc["model5", "bias"]),
        error_cut=float(sc.loc["proforma", "mdape"] - sc.loc["model5", "mdape"]),
        win_rate=h2h["win_rate"], win_p=h2h["binom_p"],
    )

def site_trajectory(d: pd.DataFrame, site_key: str) -> pd.DataFrame:
    """Year-by-year actual vs every forecaster, for one site.

    `observed_months` carries the coverage behind each actual so a part-year can be shown as such
    rather than read as a collapse.
    """
    r = d[d.site_key == site_key]
    if r.empty:
        return pd.DataFrame()
    r = r.iloc[0]
    rows = []
    for y in range(1, 6):
        rows.append(dict(
            year=y,
            actual=r.get(f"actual_y{y}", np.nan),
            observed_months=r.get(f"actual_nobs_y{y}", np.nan),
            proforma=r.get(f"proforma_y{y}", np.nan),
            coldstart=r.get(f"coldstart_v15_y{y}", np.nan),
            model5=r.get(f"model5_y{y}", np.nan)))
    return pd.DataFrame(rows)


def state_summary(d: pd.DataFrame) -> pd.DataFrame:
    """Proforma accuracy by state — only states with enough sites to mean anything."""
    a = d.actual_mature_wash
    g = d.assign(ratio=d.proforma_y5 / a)
    out = (g.groupby("state")
            .agg(sites=("site_key", "size"), median_ratio=("ratio", "median"),
                 median_actual=("actual_mature_wash", "median"))
            .reset_index())
    return out[out.sites >= 3].sort_values("median_ratio", ascending=False)

# =================================================================================================
# tunnel length — the formula against the tunnel that got built
# =================================================================================================
FT_PER_M = 0.3048


def tunnel_lengths(d: pd.DataFrame) -> pd.DataFrame:
    """Per-site formula length vs the length actually built, for the sites that have both.

    `formula_m` is recomputed here from `year5_max_hourly` x 0.3048 so the arithmetic is explicit;
    it agrees with the file's own `tunnel_length_m` to within 0.5%.
    """
    s = d[d.tunnel_length_actual_m.notna()].copy()
    s["formula_ft"] = s.year5_max_hourly
    s["formula_m"] = s.year5_max_hourly * FT_PER_M
    s["actual_m"] = s.tunnel_length_actual_m
    s["gap_m"] = s.actual_m - s.formula_m           # +ve = built longer than the formula asked for
    s["ratio"] = s.formula_m / s.actual_m
    return s


def tunnel_length_stats(d: pd.DataFrame) -> dict:
    from scipy import stats as _st
    s = tunnel_lengths(d)
    r, p = _st.pearsonr(s.formula_m, s.actual_m)
    rho, prho = _st.spearmanr(s.formula_m, s.actual_m)
    return dict(n=len(s),
                r=float(r), p=float(p), rho=float(rho), p_rho=float(prho),
                median_actual=float(s.actual_m.median()),
                median_formula=float(s.formula_m.median()),
                median_gap=float(s.gap_m.median()),
                mae=float(s.gap_m.abs().mean()),
                median_ratio=float(s.ratio.median()),
                built_longer_share=float((s.gap_m > 0).mean()))


def length_signal_check(d: pd.DataFrame) -> pd.DataFrame:
    """What does each length measure actually track — the tunnel, or the volume projection?

    The formula length is a restatement of the proforma's year-5 volume, so if it correlates with
    volume more strongly than with the built tunnel, it is a volume proxy wearing a length label.
    """
    from scipy import stats as _st
    s = tunnel_lengths(d)
    rows = []
    for xcol, xlab in [("formula_m", "Formula length"), ("actual_m", "Actual built length")]:
        for ycol, ylab in [("actual_mature_wash", "Actual mature washes"),
                           ("proforma_y5", "Proforma year-5 projection"),
                           ("actual_m", "Actual built length")]:
            if xcol == ycol:
                continue
            rho, p = _st.spearmanr(s[xcol], s[ycol])
            rows.append(dict(measure=xlab, tracks=ylab, rho=float(rho), p=float(p)))
    return pd.DataFrame(rows)


# =================================================================================================
# factor evidence — does a better score actually mean more washes?
# =================================================================================================
FACTOR_LABEL = {
    "pay_stations": "Pay stations", "free_vacuum_slots": "Free vacuum slots",
    "type_of_site": "Type of site", "entrance_stack_up": "Entrance stack-up",
    "visibility": "Visibility", "traffic_speed": "Traffic speed",
    "nearest_competition": "Nearest competition", "area_profile": "Area profile",
    "site_accessibility": "Site accessibility", "weekly_hours": "Weekly hours",
}
IMPACT_FACTORS = ["pay_stations", "free_vacuum_slots", "type_of_site", "entrance_stack_up",
                  "visibility", "traffic_speed", "nearest_competition", "area_profile",
                  "site_accessibility"]
SCORE_STEP = 0.05   # the proforma's scores move in 0.05 increments

# `traffic_count` — vehicles a day past the site — is the one proforma input that is a real measured
# quantity rather than a 0–0.15 scored level, so it sits outside FACTOR_COLS in the file. It belongs
# in the feature analysis all the same: it is on every proforma, present for all 70 sites, and it is
# the input people most expect to drive volume. It is modelled in logs (a 5k→10k road is the same
# kind of step as 25k→50k) and reported on its own scale.
TRAFFIC_KEY = "traffic_count"
FEATURE_LABEL = {**FACTOR_LABEL, TRAFFIC_KEY: "Traffic count"}

# The three build-sheet features Model 5 actually carries (`m5in_pay`, `m5in_vac`, `m5in_tos`).
# Everything else it uses is location, not build sheet.
MODEL5_FEATURES = ["pay_stations", "free_vacuum_slots", "type_of_site"]


def feature_frame(d: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """The full candidate feature set — nine scored factors plus measured traffic count.

    Returns `(X, y, keys)` with y = log(mature washes) and X **rank-transformed**, then standardised.
    Ranking is not cosmetic, for two reasons:

      • the proforma's score weights (0.05 / 0.075 / 0.125 / 0.15) are chosen weights, not measured
        quantities — only their ORDER carries information, so a rank is the honest reading;
      • one site scores −0.25 on free vacuum slots ("Coin or None"), a single-site level sitting five
        standard deviations below every other site. On the raw scale that one row has enough leverage
        to drag vacuum slots from the third-best feature to the worst. Ranking removes the leverage
        without removing the site.

    Traffic count is ranked on the same footing, so a 30,000-vehicle road and a 0.05 score notch land
    on one comparable axis.
    """
    s = d[d.actual_mature_wash > 0].copy()
    keys, cols = [], {}
    for f in IMPACT_FACTORS:
        c = f"factor_{f}_score"
        if c in s.columns and s[c].nunique() > 1:
            keys.append(f)
            cols[f] = s[c].astype(float)
    if TRAFFIC_KEY in s.columns and s[TRAFFIC_KEY].notna().all():
        keys.append(TRAFFIC_KEY)
        cols[TRAFFIC_KEY] = s[TRAFFIC_KEY].astype(float)
    raw = pd.DataFrame(cols)[keys]
    X = raw.rank(pct=True)
    X = (X - X.mean()) / X.std()
    X.attrs["raw"] = raw
    return X, np.log(s.actual_mature_wash.values), keys


def _loo_r2(X: pd.DataFrame, y: np.ndarray, keys: list[str]) -> float:
    """Leave-one-site-out R^2 — how well the features predict a site the fit never saw.

    This is the number that matters. In-sample R^2 only ever rises as you add features, so on 68
    sites it will happily reward nine correlated predictors for memorising noise; the leave-one-out
    figure goes NEGATIVE when that happens, which is exactly the signal we want to show.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import LeaveOneOut, cross_val_predict

    if not keys:
        return 0.0
    Z = X[keys].values
    p = cross_val_predict(LinearRegression(), Z, y, cv=LeaveOneOut())
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def factor_levels(d: pd.DataFrame, factor: str, min_n: int = 3) -> pd.DataFrame:
    """Median washes at each score level of one factor — the raw, unmodelled evidence.

    Grouped by the numeric SCORE rather than the raw choice text, because the text carries
    case variants and combined entries ("Corner Lot With Light / Inside Lot No Light") that would
    split identical levels apart.
    """
    sc, ch = f"factor_{factor}_score", f"factor_{factor}_choice"
    if sc not in d.columns:
        return pd.DataFrame()
    agg = dict(sites=("actual_mature_wash", "size"),
               median_washes=("actual_mature_wash", "median"),
               example=(ch, lambda s: s.mode().iloc[0] if len(s.mode()) else ""))
    # carry the two leading capacity factors alongside, so a level's raw gap can be checked
    # against what ELSE changes across those levels
    for extra, col in [("pay", "factor_pay_stations_score"),
                       ("vac", "factor_free_vacuum_slots_score")]:
        if col in d.columns:
            agg[f"median_{extra}"] = (col, "median")
    g = d.groupby(sc).agg(**agg).reset_index().rename(columns={sc: "score"})
    g["kept"] = g.sites >= min_n
    return g.sort_values("score")


def factor_significance(d: pd.DataFrame, factor: str) -> dict:
    """Is the difference between this factor's levels real, or small-sample noise?

    Two independent checks: a rank correlation against actual washes (does a better score go with
    more washes at all), and Kruskal-Wallis across the levels (are the groups distinguishable).
    """
    sc = f"factor_{factor}_score"
    if sc not in d.columns:
        return {}
    s = d.dropna(subset=[sc])
    rho, p_rho = stats.spearmanr(s[sc], s.actual_mature_wash)
    groups = [x.actual_mature_wash.values for _, x in d.groupby(sc) if len(x) >= 3]
    try:
        _, p_kw = stats.kruskal(*groups) if len(groups) >= 2 else (np.nan, np.nan)
    except ValueError:
        p_kw = np.nan
    p_kw = float(p_kw) if p_kw == p_kw else np.nan
    real = bool(p_rho < 0.05 and (np.isnan(p_kw) or p_kw < 0.10))
    return dict(rho=float(rho), p_rho=float(p_rho), p_levels=p_kw, real=real,
                smallest_group=int(min((len(x) for _, x in d.groupby(sc)), default=0)))


def factor_monotonic(d: pd.DataFrame, factor: str, min_n: int = 3) -> bool:
    g = factor_levels(d, factor, min_n)
    g = g[g.kept]
    return bool(len(g) >= 3 and g.median_washes.is_monotonic_increasing)


def factor_impact(d: pd.DataFrame) -> pd.DataFrame:
    """Per feature, how good it LOOKS versus how good it actually IS.

    One feature at a time, two numbers, both on the same scale — share of the site-to-site spread in
    mature washes that the feature accounts for:

      • `r2_fitted` — measured on the same 68 sites the line was drawn through. This is what a
                      spreadsheet correlation shows you, and it can only ever look good.
      • `loo_real`  — the same feature, but every site is predicted by a line fitted **without that
                      site**. A feature carrying no real information scores BELOW ZERO here: using
                      it leaves you worse off than just quoting the estate average.

    The gap between the two is exactly the amount of flattery in the first number. `raw_effect`
    (washes per one step better) is carried for the hover.
    """
    from sklearn.linear_model import LinearRegression

    X, y, keys = feature_frame(d)
    raw = X.attrs["raw"]

    rows = []
    for f in keys:
        fitted = LinearRegression().fit(X[[f]], y)
        # per-step effect back on the feature's OWN scale: a 0.05 score notch, or a 25% busier road
        if f == TRAFFIC_KEY:
            b = LinearRegression().fit(np.log(raw[[f]]), y).coef_[0] * np.log(1.25)
        else:
            b = LinearRegression().fit(raw[[f]], y).coef_[0] * SCORE_STEP
        rho, p = stats.spearmanr(raw[f], y)
        rows.append(dict(
            factor=FEATURE_LABEL.get(f, f), key=f,
            r2_fitted=float(fitted.score(X[[f]], y)),
            loo_real=_loo_r2(X, y, [f]),
            raw_effect=float(np.exp(b) - 1),
            step_label="a 25% busier road" if f == TRAFFIC_KEY else "one level better",
            rho=float(rho), p=float(p),
            in_model5=f in MODEL5_FEATURES))
    out = pd.DataFrame(rows).sort_values("loo_real", ascending=False).reset_index(drop=True)
    out.attrs["fitted_all"] = float(LinearRegression().fit(X, y).score(X, y))
    out.attrs["loo_all"] = _loo_r2(X, y, keys)
    out.attrs["n_survive"] = int((out.loo_real > 0).sum())
    out.attrs["n"] = int(len(y))
    return out


def selection_path(d: pd.DataFrame) -> pd.DataFrame:
    """Add features one at a time, best first, scoring each set on sites it never saw.

    Answers the only question that matters for a build sheet: **how many of these are worth
    keeping?** At each step the feature that most improves the leave-one-site-out score is added, so
    the curve is the best case for every set size — if it turns down, no other ordering saves it.

    Columns: `step`, `added`, `loo_real` (tested on sites the fit never saw), `r2_fitted` (what it
    looks like if you never check).
    """
    from sklearn.linear_model import LinearRegression

    X, y, keys = feature_frame(d)
    chosen, rest, rows = [], list(keys), []
    while rest:
        best = max(rest, key=lambda k: _loo_r2(X, y, chosen + [k]))
        chosen.append(best)
        rest.remove(best)
        rows.append(dict(step=len(chosen), added=FEATURE_LABEL.get(best, best), key=best,
                         loo_real=_loo_r2(X, y, chosen),
                         r2_fitted=float(LinearRegression().fit(X[chosen], y).score(X[chosen], y))))
    out = pd.DataFrame(rows)
    m5 = [k for k in MODEL5_FEATURES if k in keys]
    out.attrs.update(
        n=int(len(y)),
        best_step=int(out.loc[out.loo_real.idxmax(), "step"]),
        best_loo=float(out.loo_real.max()),
        worst_loo=float(out.loo_real.iloc[-1]),
        end_fitted=float(out.r2_fitted.iloc[-1]),
        model5_loo=_loo_r2(X, y, m5),
        model5_labels=[FEATURE_LABEL.get(k, k) for k in m5])
    out.attrs["model5_share"] = out.attrs["model5_loo"] / out.attrs["best_loo"]
    return out


def traffic_bands(d: pd.DataFrame, q: int = 4) -> pd.DataFrame:
    """Sites grouped by how busy the road is: washes won, and share of the traffic captured.

    Capture rate = daily washes / vehicles a day. It is the number that decides whether a busier
    road is worth paying for, and it is the one the raw wash figure hides.
    """
    s = d[(d.actual_mature_wash > 0) & d[TRAFFIC_KEY].notna()].copy()
    s["daily_washes"] = s.actual_mature_wash * 12 / 365
    s["capture"] = s.daily_washes / s[TRAFFIC_KEY]
    s["band"] = pd.qcut(s[TRAFFIC_KEY], q, labels=False)
    g = (s.groupby("band", observed=True)
           .agg(sites=(TRAFFIC_KEY, "size"), lo=(TRAFFIC_KEY, "min"), hi=(TRAFFIC_KEY, "max"),
                median_traffic=(TRAFFIC_KEY, "median"),
                median_washes=("actual_mature_wash", "median"),
                capture=("capture", "median"))
           .reset_index())
    g["label"] = [f"{r.lo:,.0f}–{r.hi:,.0f}" for r in g.itertuples()]
    rho, p = stats.spearmanr(s[TRAFFIC_KEY], s.actual_mature_wash)
    g.attrs.update(rho=float(rho), p=float(p), n=int(len(s)),
                   traffic_ratio=float(g.median_traffic.iloc[-1] / g.median_traffic.iloc[0]),
                   wash_ratio=float(g.median_washes.iloc[-1] / g.median_washes.iloc[0]),
                   capture_ratio=float(g.capture.iloc[0] / g.capture.iloc[-1]))
    return g


def factor_correlations(d: pd.DataFrame, keys=("pay_stations", "free_vacuum_slots",
                                               "type_of_site")) -> pd.DataFrame:
    """How entangled the leading factors are with each other."""
    cols = [f"factor_{k}_score" for k in keys if f"factor_{k}_score" in d.columns]
    c = d[cols].corr(method="spearman")
    c.index = [FACTOR_LABEL.get(k, k) for k in keys]
    c.columns = [FACTOR_LABEL.get(k, k) for k in keys]
    return c


# =================================================================================================
# one site, everything about it
# =================================================================================================
def site_volume_views(d: pd.DataFrame, site_key: str,
                      open_hours: float | None = None) -> pd.DataFrame:
    """Year 1-5 actual washes for one site, expressed per month, per day and per open hour.

    `open_hours` overrides the site's own `avg_daily_wash_hours`. The per-hour view is the one that
    the tunnel is actually sized against — the sizing rule is *cars per hour*, so how many hours a
    day the site is assumed to trade moves the recommended tunnel directly, and the proforma's own
    figure is an assumption made before the site opened rather than a measurement. Both are carried
    on `.attrs` so the caller can say which one is on screen.
    """
    t = site_trajectory(d, site_key)
    if t.empty:
        return t
    r = d[d.site_key == site_key].iloc[0]
    proforma_hours = r.get("avg_daily_wash_hours", np.nan)
    hours = open_hours if open_hours else proforma_hours
    for col in ("actual", "proforma", "model5"):
        t[f"{col}_daily"] = t[col] * 12 / 365
        t[f"{col}_hourly"] = t[f"{col}_daily"] / hours if hours and hours > 0 else np.nan
    t.attrs["open_hours_per_day"] = float(hours) if pd.notna(hours) else np.nan
    t.attrs["proforma_hours"] = float(proforma_hours) if pd.notna(proforma_hours) else np.nan
    t.attrs["hours_overridden"] = bool(open_hours and pd.notna(proforma_hours)
                                       and abs(open_hours - proforma_hours) > 1e-9)
    return t


def site_profile(d: pd.DataFrame, site_key: str) -> pd.DataFrame:
    """The choices the proforma scored this site on — what the site was actually built with."""
    r = d[d.site_key == site_key]
    if r.empty:
        return pd.DataFrame()
    r = r.iloc[0]
    rows = []
    for f in IMPACT_FACTORS + ["weekly_hours"]:
        ch, sc = f"factor_{f}_choice", f"factor_{f}_score"
        if ch not in d.columns:
            continue
        rows.append(dict(Factor=FACTOR_LABEL.get(f, f),
                         Chosen=str(r.get(ch, "—")).title(),
                         Score=r.get(sc, np.nan)))
    return pd.DataFrame(rows)


def site_tunnel(d: pd.DataFrame, site_key: str) -> dict:
    """Recommended (formula) vs actually built tunnel length for one site."""
    r = d[d.site_key == site_key]
    if r.empty:
        return {}
    r = r.iloc[0]
    rec_ft = r.get("year5_max_hourly", np.nan)
    actual_m = r.get("tunnel_length_actual_m", np.nan)
    return dict(recommended_ft=float(rec_ft) if pd.notna(rec_ft) else np.nan,
                recommended_m=float(rec_ft * FT_PER_M) if pd.notna(rec_ft) else np.nan,
                actual_m=float(actual_m) if pd.notna(actual_m) else np.nan,
                actual_ft=float(actual_m / FT_PER_M) if pd.notna(actual_m) else np.nan,
                gap_m=float(actual_m - rec_ft * FT_PER_M)
                if pd.notna(actual_m) and pd.notna(rec_ft) else np.nan,
                open_hours=float(r.get("avg_daily_wash_hours", np.nan)),
                weekly_hours=float(r.get("weekly_hours_operation", np.nan)))


def tunnel_cohorts(d: pd.DataFrame) -> pd.DataFrame:
    """Projected peak demand vs the tunnel actually built, grouped by how long the site has traded.

    The n70 file carries only ONE peak number per site — `year5_max_hourly`, the proforma's own
    year-5 projection — so unlike section ①'s measured percentiles this is a *promised* peak, not an
    observed one. Cohorts come from how many operating years the site has actual washes for.
    """
    s = tunnel_lengths(d).copy()
    years = sum((s[f"actual_y{y}"].notna() & (s[f"actual_y{y}"] > 0)).astype(int)
                for y in range(1, 6))
    s["years_trading"] = years
    s["cohort"] = pd.Categorical(
        pd.cut(years, [0, 1, 2, 3, 99], labels=["Year 1", "Year 2", "Year 3", "Year 4+"]),
        ["Year 1", "Year 2", "Year 3", "Year 4+"], ordered=True)
    s["actual_ft"] = s.actual_m / FT_PER_M
    s["share_used"] = s.year5_max_hourly / s.actual_ft
    return s
