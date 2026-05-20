"""Feature engineering.

Two modes:

* **Known-site features** — for sites with observed history we compute
  per-site statistics (mean/peak/std/recent) over a designated *training
  window* and attach them as features. The booster can then learn each
  site's level directly.
* **Cold-start features** — for a brand-new site with no history these
  per-site stats are imputed from the CBSA peer reference. The exact
  same feature schema is used at training and inference.

This matches the strategy used in the previous Model 1 / Model 2
pipelines (cluster-level peer features + a per-site identity signal)
while keeping the cold-start path interpretable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .peer_groups import resolve_peer_features
from .lifecycle import get_ramp


REGION_ORDER = ["Northeast", "Midwest", "South", "West", "Other"]


def _region_code(r):
    try:
        return REGION_ORDER.index(str(r))
    except ValueError:
        return len(REGION_ORDER) - 1  # "Other"


def site_training_stats(panel: pd.DataFrame, train_end_ym: str) -> pd.DataFrame:
    """Per-site stats computed strictly from rows where ``year_month <= train_end_ym``.

    These are the "I have seen this site already" features. For brand-new
    sites they're not available and get imputed from peer references at
    forecast time.
    """
    tr = panel[panel["year_month"] <= train_end_ym]
    g = tr.groupby("client_id_location_id").agg(
        site_train_mean=("wash_count_total", "mean"),
        site_train_median=("wash_count_total", "median"),
        site_train_max=("wash_count_total", "max"),
        site_train_std=("wash_count_total", "std"),
        site_train_months=("wash_count_total", "size"),
    ).reset_index()
    # Recent 6-month mean over training window.
    tr_sorted = tr.sort_values(["client_id_location_id", "ym_ts"])
    last6 = tr_sorted.groupby("client_id_location_id").tail(6)
    g_recent = last6.groupby("client_id_location_id").agg(
        site_train_recent6=("wash_count_total", "mean"),
    ).reset_index()
    return g.merge(g_recent, on="client_id_location_id", how="left")


def build_training_frame(
    panel: pd.DataFrame,
    static: pd.DataFrame,
    cbsa_ref: pd.DataFrame,
    rollups: dict,
    ramp: pd.DataFrame,
    site_stats: pd.DataFrame | None = None,
    h3_ref: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach peer + ramp + H3 local + (optional) site-level features."""
    drop_dup = [c for c in ("state", "region") if c in panel.columns and c in static.columns]
    base = panel.drop(columns=drop_dup) if drop_dup else panel
    static_cols = ["client_id_location_id", "cbsa_id", "state", "region"]
    if "h3_id" in static.columns:
        static_cols.append("h3_id")
    df = base.merge(static[static_cols], on="client_id_location_id", how="left")
    df["calendar_month"] = df["ym_ts"].dt.month

    # ---------- peer features (vectorised CBSA merge then fallback) ----------
    peer = df.merge(
        cbsa_ref[["cbsa_id", "calendar_month", "cbsa_ref_level", "cbsa_seasonal_factor",
                  "cbsa_ref_recent", "cbsa_peer_count"]],
        on=["cbsa_id", "calendar_month"], how="left",
    )
    peer = peer.rename(columns={
        "cbsa_ref_level": "peer_ref_level",
        "cbsa_seasonal_factor": "peer_seasonal",
        "cbsa_ref_recent": "peer_recent",
        "cbsa_peer_count": "peer_count",
    })
    peer["peer_level_resolved"] = np.where(
        peer["peer_count"].fillna(0) >= C.MIN_PEERS_PER_MARKET, "cbsa", None
    )

    # Fallback rows
    need_fb = peer["peer_level_resolved"].isna()
    if need_fb.any():
        fb = peer.loc[need_fb].apply(
            lambda r: pd.Series(resolve_peer_features(
                r["cbsa_id"], r["state"], r["region"], int(r["calendar_month"]),
                cbsa_ref, rollups,
            )),
            axis=1,
        )
        for col in fb.columns:
            peer.loc[need_fb, col] = fb[col].values

    # ---------- H3 local sub-market features ----------
    if h3_ref is not None and "h3_id" in peer.columns and len(h3_ref):
        peer = peer.merge(
            h3_ref[["h3_id", "calendar_month", "h3_ref_level", "h3_seasonal_factor", "h3_peer_count"]],
            on=["h3_id", "calendar_month"], how="left",
        )
    if "h3_ref_level" not in peer.columns:
        peer["h3_ref_level"] = np.nan
        peer["h3_seasonal_factor"] = np.nan
        peer["h3_peer_count"] = 0
    # Where H3 local is missing, fall back to CBSA values (so the schema
    # never has NaN and the booster sees graceful degradation).
    peer["h3_ref_level"] = peer["h3_ref_level"].fillna(peer["peer_ref_level"])
    peer["h3_seasonal_factor"] = peer["h3_seasonal_factor"].fillna(peer["peer_seasonal"])
    peer["h3_peer_count"] = peer["h3_peer_count"].fillna(0).astype(int)
    # Local vs macro ratio — captures "this neighbourhood is hotter/colder
    # than the broader metro by X%".
    peer["h3_vs_cbsa_ratio"] = (peer["h3_ref_level"] / peer["peer_ref_level"]).clip(0.3, 3.0).fillna(1.0)
    peer["h3_used"] = (peer["h3_peer_count"] >= C.MIN_PEERS_PER_H3).astype(int)

    # ---------- ramp features ----------
    # Vectorised join on age — region-specific where available, national else.
    nat = ramp[ramp["level"] == "national"][["age_months", "ramp_factor"]]
    nat = nat.rename(columns={"ramp_factor": "ramp_national"})
    peer = peer.merge(nat, left_on="site_age_months", right_on="age_months", how="left").drop(columns=["age_months"])

    reg = ramp[ramp["level"] == "region"][["level_value", "age_months", "ramp_factor"]]
    reg = reg.rename(columns={"level_value": "region", "ramp_factor": "ramp_region"})
    peer = peer.merge(reg, left_on=["region", "site_age_months"],
                       right_on=["region", "age_months"], how="left").drop(columns=["age_months"])
    peer["ramp_factor"] = peer["ramp_region"].fillna(peer["ramp_national"]).fillna(1.0)

    # ---------- structural features ----------
    peer["calendar_month_sin"] = np.sin(2 * np.pi * peer["calendar_month"] / 12)
    peer["calendar_month_cos"] = np.cos(2 * np.pi * peer["calendar_month"] / 12)
    peer["region_code"] = peer["region"].apply(_region_code)
    peer["maturity_indicator"] = (peer["site_age_months"] >= C.MATURITY_MONTHS).astype(int)
    # CBSA as a categorical id — LightGBM can learn per-market level.
    peer["cbsa_code"] = peer["cbsa_id"].astype("category").cat.codes.astype("int64")

    # ---------- site-level features (mean/peak/std from training window) ----------
    if site_stats is not None:
        peer = peer.merge(site_stats, on="client_id_location_id", how="left")
        # For sites missing training stats (cold-start at inference) impute
        # from the peer reference so the booster sees the same schema.
        peer["site_train_mean"] = peer["site_train_mean"].fillna(peer["peer_ref_level"])
        peer["site_train_median"] = peer["site_train_median"].fillna(peer["peer_ref_level"])
        peer["site_train_max"] = peer["site_train_max"].fillna(peer["peer_ref_level"] * 1.5)
        peer["site_train_std"] = peer["site_train_std"].fillna(peer["peer_ref_level"] * 0.2)
        peer["site_train_recent6"] = peer["site_train_recent6"].fillna(peer["peer_recent"].fillna(peer["peer_ref_level"]))
        peer["site_train_months"] = peer["site_train_months"].fillna(0)
        # Useful interactions.
        peer["site_vs_peer_ratio"] = (peer["site_train_mean"] / peer["peer_ref_level"]).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.2, 5.0)
    else:
        for c in ["site_train_mean", "site_train_median", "site_train_max",
                   "site_train_std", "site_train_recent6", "site_train_months",
                   "site_vs_peer_ratio"]:
            if c not in peer.columns:
                peer[c] = peer["peer_ref_level"] if "ratio" not in c and "months" not in c and "std" not in c else 0.0

    # Anchor prediction.
    # Priority order for the "level" component:
    #   1. site-specific training mean (if known)
    #   2. H3 local sub-market reference (if enough local peers)
    #   3. CBSA peer reference (with state/region/national fallback)
    if "site_train_mean" in peer.columns:
        known_site = peer["site_train_months"] > 0
    else:
        known_site = pd.Series(False, index=peer.index)
    h3_ok = peer["h3_used"] == 1 if "h3_used" in peer.columns else pd.Series(False, index=peer.index)
    level_h3_or_cbsa = peer["h3_ref_level"].where(h3_ok, peer["peer_ref_level"])
    seasonal_h3_or_cbsa = peer["h3_seasonal_factor"].where(h3_ok, peer["peer_seasonal"])
    site_level = (peer["site_train_mean"].where(known_site, level_h3_or_cbsa)
                  if "site_train_mean" in peer.columns else level_h3_or_cbsa)
    peer["anchor_wash"] = site_level * seasonal_h3_or_cbsa * peer["ramp_factor"]

    # Extra interaction-flavoured features.
    peer["peer_trend_ratio"] = (peer["peer_recent"] / peer["peer_ref_level"]).clip(0.5, 2.0).fillna(1.0)
    peer["log_anchor"] = np.log1p(peer["anchor_wash"].clip(lower=1))

    # Targets.
    peer["y"] = np.log1p(peer["wash_count_total"].clip(lower=1))
    peer["y_anchor"] = peer["log_anchor"]
    peer["y_residual"] = peer["y"] - peer["y_anchor"]

    return peer


FEATURE_COLS = [
    "site_age_months",
    "calendar_month_sin", "calendar_month_cos",
    "latitude", "longitude",
    "region_code", "cbsa_code",
    "peer_ref_level", "peer_recent", "peer_seasonal", "peer_count",
    "peer_trend_ratio",
    # H3 local sub-market layer
    "h3_ref_level", "h3_seasonal_factor", "h3_peer_count",
    "h3_vs_cbsa_ratio", "h3_used",
    "ramp_factor",
    "log_anchor",
    "maturity_indicator",
    # Site-level features (imputed for cold-start)
    "site_train_mean", "site_train_median", "site_train_max",
    "site_train_std", "site_train_recent6", "site_train_months",
    "site_vs_peer_ratio",
]
