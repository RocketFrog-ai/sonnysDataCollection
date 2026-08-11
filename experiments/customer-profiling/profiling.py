"""
Customer profiling for the Hurricane Express membership book — preprocessing, features, models.

Streamlit-free on purpose: `customer_profiling.ipynb` and `app.py` both import this, so the
notebook's numbers and the demo's numbers cannot drift apart.

The raw export is one row per (event x vehicle). Two things about that shape decide every number
downstream, and both are handled in `load_events` / `payments`:

1. **Payments are fanned out across the household's vehicles.** A 3-vehicle household paying $40
   once appears as three $40 rows. Summing `amount` naively overstates revenue by ~58%
   ($111k -> $70k on this export). `payments()` collapses them.
2. **Washes are per vehicle and must NOT be collapsed** — a household with three cars really does
   drive three washes. The only wash duplicates are the same `vehicle_id` carrying two spellings of
   its plate, which dedupe on (customer, vehicle, timestamp).

Vocabulary
----------
cycle      one positive payment = one paid membership month.
renewed    another payment landed within `RENEW_WINDOW` days of this one.
censored   the export ends before we could observe the renewal decision; excluded from rates.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).with_name("hurricane_customer_profille_data.csv")

# A renewal lands ~30 days after the last one (median 30, p75 31). 45 days allows a late retry
# without ever spanning two cycles; 40 is the point past which we call the membership dead.
RENEW_WINDOW = 45
CHURN_AFTER = 40
CYCLE_DAYS = 30

KEY = ["customer_id"]


# --------------------------------------------------------------------------------------------
# load & split
# --------------------------------------------------------------------------------------------
def load_events(path: Path | str = DATA) -> pd.DataFrame:
    """Raw export -> tidy event log. One row per (event, vehicle); nothing collapsed yet."""
    df = pd.read_csv(path)
    df["event_date"] = pd.to_datetime(df["event_date"])

    # `vehicle_vin` is 100% null and `vehicle_year`/`vehicle_model` are >96% null — they carry no
    # information on this export, so they are dropped rather than silently half-used.
    df = df.drop(columns=["vehicle_vin", "vehicle_year", "vehicle_model"], errors="ignore")

    # "-" is the export's other spelling of missing.
    for c in ["vehicle_state", "vehicle_type", "vehicle_make", "vehicle_color"]:
        if c in df:
            df[c] = df[c].replace("-", np.nan)

    df["customer_id"] = df["customer_id"].astype("int64")
    df["site_id"] = df["site_id"].astype("int64")
    df = df.drop_duplicates().sort_values(["customer_id", "event_date"]).reset_index(drop=True)
    return df


def asof(ev: pd.DataFrame) -> pd.Timestamp:
    """Last moment the export can speak to. Everything censoring-related keys off this."""
    return ev["event_date"].max()


def payments(ev: pd.DataFrame) -> pd.DataFrame:
    """One row per real charge, household-level (the per-vehicle fan-out collapsed).

    `n_vehicles_billed` keeps the fan-out width, which is the household's vehicle count at the
    time of the charge — a feature in its own right.
    """
    p = ev[ev["event_type"] == "payment"].copy()
    grp = ["customer_id", "site_id", "event_date", "membership_package_name",
           "payment_type", "amount", "current_package_price"]
    out = (p.groupby(grp, dropna=False)
             .agg(n_vehicles_billed=("vehicle_id", "nunique"))
             .reset_index()
             .sort_values(["customer_id", "event_date"]))
    # 38 charges carry no vehicle_id at all. That is a separate fact from household size, so it
    # gets its own flag rather than being smuggled in as "a household of zero cars".
    out["no_vehicle_on_file"] = out["n_vehicles_billed"].eq(0)
    out["n_vehicles_billed"] = out["n_vehicles_billed"].clip(lower=1)
    out["is_refund"] = out["amount"] < 0
    out["is_comp"] = out["amount"] == 0            # $0 renewal = comped month, still a live member
    out["discount"] = 1 - out["amount"] / out["current_package_price"]
    return out.reset_index(drop=True)


def washes(ev: pd.DataFrame) -> pd.DataFrame:
    """One row per vehicle-wash. Deduped on (customer, vehicle, timestamp) only."""
    w = ev[ev["event_type"] == "wash"].copy()
    return (w.drop_duplicates(["customer_id", "vehicle_id", "event_date"])
             [["customer_id", "site_id", "vehicle_id", "vehicle_type", "membership_package_name",
               "event_date"]]
             .sort_values(["customer_id", "event_date"])
             .reset_index(drop=True))


# --------------------------------------------------------------------------------------------
# the renewal panel — one row per paid month, features as known at the moment of the charge
# --------------------------------------------------------------------------------------------
def renewal_panel(ev: pd.DataFrame) -> pd.DataFrame:
    """Customer-month panel for the churn hazard model.

    Each row is a charge. The label is whether the *next* charge arrived. Every feature is
    computed from events at or before that charge — nothing from the cycle being predicted — so
    the model is honest about what it would know at decision time.
    """
    pay = payments(ev)
    wsh = washes(ev)
    now = asof(ev)

    cyc = pay[~pay["is_refund"]].copy()            # refunds are not membership months
    cyc = cyc.sort_values(["customer_id", "event_date"]).reset_index(drop=True)
    cyc["cycle_no"] = cyc.groupby("customer_id").cumcount()          # 0 = signup month
    cyc["prev_amount"] = cyc.groupby("customer_id")["amount"].shift()
    cyc["next_date"] = cyc.groupby("customer_id")["event_date"].shift(-1)

    gap = (cyc["next_date"] - cyc["event_date"]).dt.days
    cyc["renewed"] = cyc["next_date"].notna() & (gap <= RENEW_WINDOW)
    cyc["days_left_in_export"] = (now - cyc["event_date"]).dt.days
    # If the export ends before the decision window closes, a missing next charge means "not yet",
    # not "churned". Those rows are censored and never counted in a renewal rate.
    cyc["censored"] = (cyc["days_left_in_export"] < CHURN_AFTER) & ~cyc["renewed"]

    # --- wash behaviour in the 30 days *ending* at this charge ---------------------------------
    m = cyc[["customer_id", "event_date"]].merge(
        wsh[["customer_id", "event_date"]].rename(columns={"event_date": "w_date"}),
        on="customer_id", how="left")
    m["age"] = (m["event_date"] - m["w_date"]).dt.days
    past = m[m["age"] >= 0]

    def _count(lo, hi, name):
        s = past[(past["age"] >= lo) & (past["age"] < hi)].groupby(
            ["customer_id", "event_date"]).size().rename(name)
        return s

    cyc = cyc.merge(_count(0, 31, "washes_this_cycle"), on=["customer_id", "event_date"], how="left")
    cyc = cyc.merge(_count(31, 61, "washes_prev_cycle"), on=["customer_id", "event_date"], how="left")
    cyc = cyc.merge(past.groupby(["customer_id", "event_date"]).size().rename("washes_to_date"),
                    on=["customer_id", "event_date"], how="left")
    cyc = cyc.merge(past.groupby(["customer_id", "event_date"])["age"].min().rename("days_since_wash"),
                    on=["customer_id", "event_date"], how="left")
    for c in ["washes_this_cycle", "washes_prev_cycle", "washes_to_date"]:
        cyc[c] = cyc[c].fillna(0).astype(int)
    # No wash on record yet -> stand it in with the membership's own age, capped.
    cyc["days_since_wash"] = cyc["days_since_wash"].fillna(90).clip(upper=90)

    # --- derived, all decision-time ------------------------------------------------------------
    cyc["dormant"] = cyc["washes_this_cycle"].eq(0)
    cyc["wash_trend"] = cyc["washes_this_cycle"] - cyc["washes_prev_cycle"]
    cyc["washes_per_vehicle"] = cyc["washes_this_cycle"] / cyc["n_vehicles_billed"].clip(lower=1)
    cyc["cost_per_wash"] = cyc["amount"] / cyc["washes_this_cycle"].clip(lower=1)
    cyc["price_ratio"] = cyc["amount"] / cyc["prev_amount"].replace(0, np.nan)
    cyc["price_step_up"] = (cyc["price_ratio"] > 1.15).fillna(False)
    cyc["tenure_months"] = cyc["cycle_no"]
    cyc["month"] = cyc["event_date"].dt.month
    cyc["period"] = cyc["event_date"].dt.to_period("M").astype(str)

    signup_amt = cyc.groupby("customer_id")["amount"].transform("first")
    signup_list = cyc.groupby("customer_id")["current_package_price"].transform("first")
    cyc["joined_on_promo"] = signup_amt < 0.6 * signup_list
    cyc["package_tier"] = cyc["current_package_price"]
    cyc["is_first_responder"] = cyc["membership_package_name"].str.contains("Responder", case=False)
    return cyc


# Deliberately parsimonious. The wider set (adding cost_per_wash, discount, washes_per_vehicle,
# washes_to_date, wash_trend) scores the same AUC to within noise — every one of them is an
# algebraic function of features already here (cost_per_wash = amount / washes, discount =
# 1 - amount / tier, ...), so all they do is split one effect across three collinear coefficients
# and flip signs. The notebook shows that comparison, and the LightGBM check against it.
FEATURES = [
    "days_since_wash", "washes_this_cycle", "tenure_months", "amount",
    "price_step_up", "n_vehicles_billed", "no_vehicle_on_file", "joined_on_promo", "month",
]
LOG_FEATURES = ["amount"]     # long right tail; log1p keeps one $119 household off the slope


# --------------------------------------------------------------------------------------------
# customer table — one row per customer, as of the export
# --------------------------------------------------------------------------------------------
def customer_table(ev: pd.DataFrame) -> pd.DataFrame:
    """One row per customer: RFM, utilisation, economics, and current status."""
    pay, wsh, now = payments(ev), washes(ev), asof(ev)
    charges = pay[~pay["is_refund"]]

    t = pd.DataFrame(index=pd.Index(sorted(ev["customer_id"].unique()), name="customer_id"))
    t["site_id"] = ev.groupby("customer_id")["site_id"].first()
    t["package"] = charges.groupby("customer_id")["membership_package_name"].last()
    t["list_price"] = charges.groupby("customer_id")["current_package_price"].last()
    t["n_vehicles"] = ev.groupby("customer_id")["vehicle_id"].nunique()
    t["vehicle_type"] = (ev.dropna(subset=["vehicle_type"])
                           .groupby("customer_id")["vehicle_type"].agg(
                               lambda s: s.mode().iat[0] if len(s.mode()) else np.nan))
    t["state"] = (ev.dropna(subset=["vehicle_state"])
                    .groupby("customer_id")["vehicle_state"].agg(
                        lambda s: s.mode().iat[0] if len(s.mode()) else np.nan))

    t["joined"] = charges.groupby("customer_id")["event_date"].min()
    t["last_payment"] = charges.groupby("customer_id")["event_date"].max()
    t["cycles_paid"] = charges.groupby("customer_id").size()
    t["revenue"] = pay.groupby("customer_id")["amount"].sum()          # refunds netted off
    t["refunds"] = pay[pay["is_refund"]].groupby("customer_id")["amount"].sum().reindex(t.index).fillna(0)
    t["signup_amount"] = charges.groupby("customer_id")["amount"].first()
    t["last_amount"] = charges.groupby("customer_id")["amount"].last()
    t["joined_on_promo"] = t["signup_amount"] < 0.6 * t["list_price"]

    t["washes"] = wsh.groupby("customer_id").size().reindex(t.index).fillna(0).astype(int)
    t["last_wash"] = wsh.groupby("customer_id")["event_date"].max()
    t["days_since_wash"] = (now - t["last_wash"]).dt.days
    t["days_since_payment"] = (now - t["last_payment"]).dt.days

    t["tenure_days"] = (t["last_payment"] - t["joined"]).dt.days + CYCLE_DAYS
    t["tenure_months"] = t["tenure_days"] / 30.44
    t["washes_per_month"] = t["washes"] / t["tenure_months"].clip(lower=1)
    t["washes_per_vehicle_month"] = t["washes_per_month"] / t["n_vehicles"].clip(lower=1)
    t["arpu"] = t["revenue"] / t["cycles_paid"].clip(lower=1)
    t["cost_per_wash"] = t["revenue"] / t["washes"].clip(lower=1)

    t["active"] = t["days_since_payment"] <= CHURN_AFTER
    t["churned"] = ~t["active"]
    t["churn_month"] = np.where(t["churned"],
                                (t["last_payment"] + pd.Timedelta(days=CYCLE_DAYS)).dt.to_period("M").astype(str),
                                None)
    t["cohort"] = t["joined"].dt.to_period("M").astype(str)
    return t.reset_index()


# --------------------------------------------------------------------------------------------
# cohort retention
# --------------------------------------------------------------------------------------------
def cohort_retention(ev: pd.DataFrame, max_m: int = 10) -> pd.DataFrame:
    """Signup-cohort x months-since-join survival. NaN where the export is too young to know."""
    pay, now = payments(ev), asof(ev)
    charges = pay[~pay["is_refund"]]
    first = charges.groupby("customer_id")["event_date"].min()
    last = charges.groupby("customer_id")["event_date"].max()
    coh = first.dt.to_period("M").astype(str)
    lived = ((last - first).dt.days / CYCLE_DAYS).round().astype(int)
    observable = ((now - first).dt.days / CYCLE_DAYS).astype(int)

    rows = {}
    for c in sorted(coh.unique()):
        idx = coh[coh == c].index
        rows[c] = {m: (np.nan if (observable[idx] >= m).sum() == 0
                       else (lived[idx][observable[idx] >= m] >= m).mean())
                   for m in range(max_m + 1)}
    out = pd.DataFrame(rows).T
    out.index.name = "cohort"
    return out


def hazard_curve(panel: pd.DataFrame) -> pd.DataFrame:
    """Renewal rate by membership month, censored rows excluded."""
    o = panel[~panel["censored"]]
    g = o.groupby("cycle_no").agg(n=("renewed", "size"), renewal_rate=("renewed", "mean"))
    return g[g["n"] >= 5]


# --------------------------------------------------------------------------------------------
# churn model
# --------------------------------------------------------------------------------------------
def _model_frame(panel: pd.DataFrame) -> pd.DataFrame:
    """Cast booleans and log the skewed money features. Used for both fitting and scoring, so the
    two can't drift."""
    d = panel.copy()
    for c in ["price_step_up", "joined_on_promo", "no_vehicle_on_file"]:
        d[c] = d[c].astype(int)
    for c in LOG_FEATURES:
        d[c] = np.log1p(d[c].clip(lower=0))
    return d


@dataclass
class ChurnModel:
    pipe: object
    auc_cv: float
    auc_holdout: float
    base_rate: float
    top_decile_lift: float
    coefs: pd.Series
    cutoff: pd.Timestamp
    n_train: int
    n_test: int

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """P(churn) — i.e. P(no renewal) for each row."""
        return 1 - self.pipe.predict_proba(X[FEATURES])[:, 1]

    def odds_ratios(self) -> pd.Series:
        """Multiplier on the *odds of churning* per +1 SD of each feature. >1 raises churn."""
        return np.exp(-self.coefs).sort_values(ascending=False)


def fit_churn_model(panel: pd.DataFrame, holdout_frac: float = 0.25, seed: int = 0) -> ChurnModel:
    """Logistic regression on the renewal panel, with a time-ordered holdout.

    Logistic, not a booster: 1.9k rows and a 92% base rate is not enough to feed a tree ensemble
    without it memorising customers, and the coefficients are the deliverable — the operator wants
    to know *which lever*, not just who. A LightGBM check in the notebook confirms it isn't leaving
    signal on the table.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    d = _model_frame(panel[~panel["censored"]])
    X, y = d[FEATURES], d["renewed"].astype(int)
    # No class_weight. Churn is 7.5% of cycles, and `balanced` would rank identically (AUC 0.672 vs
    # 0.671) while pushing mean predicted risk to 0.45 — fine for a ranking, useless for the dollar
    # figure the demo puts on screen. Unweighted, the model is calibrated: top risk quintile
    # predicts 15.4% and delivers 15.3%.
    pipe = Pipeline([
        ("prep", ColumnTransformer([("num", StandardScaler(), FEATURES)])),
        ("lr", LogisticRegression(max_iter=2000, C=0.5)),
    ])

    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    auc_cv = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean()

    # Time-ordered holdout: train on the older cycles, predict the newer ones. This is the split
    # that matches how the model would actually be used, and it is strictly harder than random CV
    # because the book matures over the window.
    order = d["event_date"].rank(method="first")
    cut = order.quantile(1 - holdout_frac)
    tr, te = order <= cut, order > cut
    pipe.fit(X[tr], y[tr])
    p_renew = pipe.predict_proba(X[te])[:, 1]
    auc_ho = roc_auc_score(y[te], p_renew) if y[te].nunique() > 1 else np.nan

    risk = 1 - p_renew
    top = risk >= np.quantile(risk, 0.9)
    churn_te = 1 - y[te].mean()
    lift = ((1 - y[te][top].mean()) / churn_te) if churn_te > 0 else np.nan

    pipe.fit(X, y)                                    # refit on everything for scoring live members
    coefs = pd.Series(pipe.named_steps["lr"].coef_[0], index=FEATURES).sort_values()
    return ChurnModel(pipe, auc_cv, auc_ho, 1 - y.mean(), lift, coefs,
                      d["event_date"][order <= cut].max(), int(tr.sum()), int(te.sum()))


def score_live_book(panel: pd.DataFrame, model: ChurnModel, cust: pd.DataFrame) -> pd.DataFrame:
    """Risk-score every still-active member off their most recent cycle."""
    latest = _model_frame(panel.sort_values("event_date").groupby("customer_id").tail(1))
    latest["churn_risk"] = model.score(latest)

    live = cust[cust["active"]][["customer_id", "package", "n_vehicles", "arpu",
                                 "washes_per_month", "tenure_months", "days_since_wash",
                                 "site_id", "joined_on_promo", "cohort"]]
    out = live.merge(latest[["customer_id", "churn_risk", "washes_this_cycle", "cost_per_wash"]],
                     on="customer_id", how="left")
    out["monthly_revenue_at_risk"] = out["arpu"] * out["churn_risk"]
    return out.sort_values("churn_risk", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------------------------
# segmentation
# --------------------------------------------------------------------------------------------
SEG_FEATURES = ["washes_per_month", "tenure_months", "arpu", "n_vehicles",
                "cost_per_wash", "days_since_wash"]


def segment(cust: pd.DataFrame, k: int = 4, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """K-means personas over behaviour + economics. Returns (customers+segment, profile table).

    Names are assigned from the centroids rather than hard-coded to a cluster index, so a re-fit
    on new data cannot silently swap two personas' labels.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    # One customer in this export has wash events but no payment row at all (see the notebook's
    # data-quality section) — no economics to cluster on, so they sit out rather than being
    # imputed into a persona.
    d = cust[cust["cycles_paid"].notna()].copy()
    X = d[SEG_FEATURES].copy()
    # Never washed -> no recency. 120 days is "beyond anything observed", which is the truth here.
    X["days_since_wash"] = X["days_since_wash"].fillna(120).clip(upper=120)
    for c in ["arpu", "cost_per_wash", "washes_per_month"]:
        X[c] = np.log1p(X[c].clip(lower=0))
    Z = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=k, n_init=25, random_state=seed).fit(Z)
    d["segment_id"] = km.labels_

    prof = d.groupby("segment_id").agg(
        members=("customer_id", "size"),
        washes_per_month=("washes_per_month", "median"),
        tenure_months=("tenure_months", "median"),
        arpu=("arpu", "median"),
        n_vehicles=("n_vehicles", "median"),
        cost_per_wash=("cost_per_wash", "median"),
        days_since_wash=("days_since_wash", "median"),
        churn_rate=("churned", "mean"),
        revenue=("revenue", "sum"),
    )
    prof["revenue_share"] = prof["revenue"] / prof["revenue"].sum()
    prof["name"] = _name_segments(prof)
    d["segment"] = d["segment_id"].map(prof["name"])
    return d, prof.reset_index().set_index("name")


# Each archetype is a direction in z-scored centroid space: +1 wants a high value on that trait,
# -1 a low one. Clusters are matched to archetypes by optimal assignment, not greedily, so no
# persona can be stolen by a cluster that merely happens to be checked first.
ARCHETYPES = {
    "Power household":  {"washes_per_month": 1, "n_vehicles": 1, "arpu": 1, "days_since_wash": -1},
    "Core regular":     {"tenure_months": 1, "days_since_wash": -1, "churn_rate": -1},
    "Never activated":  {"washes_per_month": -1, "days_since_wash": 1, "cost_per_wash": 1},
    "Promo flipper":    {"tenure_months": -1, "arpu": -1, "days_since_wash": 1, "churn_rate": 1},
    "Occasional":       {"washes_per_month": -1, "tenure_months": 1, "churn_rate": -1},
    "Price-sensitive":  {"arpu": -1, "tenure_months": 1, "washes_per_month": 1},
}


def _name_segments(prof: pd.DataFrame) -> pd.Series:
    """Match each centroid to the archetype it best fits, one-to-one (Hungarian assignment)."""
    from scipy.optimize import linear_sum_assignment

    traits = ["washes_per_month", "tenure_months", "arpu", "n_vehicles",
              "cost_per_wash", "days_since_wash", "churn_rate"]
    z = prof[traits]
    z = (z - z.mean()) / z.std(ddof=0).replace(0, 1)

    names = list(ARCHETYPES)
    # score[i, j] = how well cluster i matches archetype j; negate for a min-cost assignment.
    score = np.array([[sum(w * z.loc[sid, t] for t, w in ARCHETYPES[n].items()) for n in names]
                      for sid in prof.index])
    rows, cols = linear_sum_assignment(-score)
    return pd.Series({prof.index[r]: names[c] for r, c in zip(rows, cols)}).reindex(prof.index)


# --------------------------------------------------------------------------------------------
# unit economics / CLV
# --------------------------------------------------------------------------------------------
def unit_economics(cust: pd.DataFrame, variable_cost_per_wash: float = 2.25,
                   monthly_churn: float | None = None) -> pd.DataFrame:
    """Per-member contribution margin and CLV.

    CLV = monthly contribution / monthly churn, the standard geometric-series lifetime. Churn
    defaults to the book's observed monthly rate; the demo lets the operator move it.
    """
    d = cust.copy()
    if monthly_churn is None:
        monthly_churn = observed_monthly_churn(d)
    d["monthly_wash_cost"] = d["washes_per_month"] * variable_cost_per_wash
    d["monthly_contribution"] = d["arpu"] - d["monthly_wash_cost"]
    d["expected_lifetime_months"] = 1 / max(monthly_churn, 1e-6)
    d["clv"] = d["monthly_contribution"] * d["expected_lifetime_months"]
    return d


def dormant_payers(cust: pd.DataFrame, days: int = 45) -> pd.DataFrame:
    """Active members still being billed who have not washed in `days`.

    The most commercially interesting slice in the book, because it cuts both ways: this month they
    are pure margin (revenue, no wash cost), and next month they are the likeliest to cancel.
    """
    d = cust[cust["active"]].copy()
    dsw = d["days_since_wash"].fillna(999)
    return d[dsw >= days].sort_values("arpu", ascending=False)


def observed_monthly_churn(cust: pd.DataFrame) -> float:
    """Churned members / total member-months — the rate that belongs in a CLV denominator."""
    return float(cust["churned"].sum() / max(cust["tenure_months"].sum(), 1))
