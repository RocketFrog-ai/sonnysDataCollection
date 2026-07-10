"""End-to-end pipeline: clean, build markets, train, evaluate, forecast samples.

Run with: ``python -m zeta_modelling.model_3.src.pipeline``
Outputs land under ``zeta_modelling/model_3/outputs/``.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as C
from .data_prep import load_and_clean, site_static_table
from .geo_markets import load_cbsa, assign_markets, add_h3
from .peer_groups import build_market_keys, market_reference_levels, rollup_levels, local_h3_levels
from .lifecycle import estimate_ramp_curves
from .features import build_training_frame, FEATURE_COLS, site_training_stats
from .train import train_cohort_models, predict_blend
from .forecast_engine import forecast_site, year_totals, explain_forecast
from .evaluate import (
    regression_metrics, site_year_hit_miss, summarize_hit_miss,
    segment_metrics, market_level_metrics, annual_hit_miss,
)


def _dump_json(obj, path):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)): return o.item()
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, pd.Timestamp): return o.isoformat()
        return str(o)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=conv)


def _fit_lean_cohorts(train_rows: pd.DataFrame) -> dict:
    """Fast two-cohort booster fit (fixed rounds, no inner CV).

    Used by the cold-start k-fold so the per-fold retrain is cheap. Same
    objective / params as :func:`train.train_cohort_models` — the booster
    learns the log-space residual on top of the peer anchor.
    """
    import lightgbm as lgb

    params = dict(objective="regression", metric="rmse", learning_rate=0.03,
                  num_leaves=31, min_data_in_leaf=80, feature_fraction=0.80,
                  bagging_fraction=0.80, bagging_freq=5, lambda_l2=2.0,
                  verbosity=-1, seed=C.SEED)
    out = {}
    for seg_name, mask in (
        ("young", train_rows["site_age_months"] < C.MATURITY_MONTHS),
        ("mature", train_rows["site_age_months"] >= C.MATURITY_MONTHS),
    ):
        seg = train_rows[mask].dropna(subset=FEATURE_COLS + ["y", "y_residual"])
        if len(seg) < 200:
            continue
        dtrain = lgb.Dataset(seg[FEATURE_COLS], label=seg["y_residual"])
        booster = lgb.train(params, dtrain, num_boost_round=800)
        pred = booster.predict(seg[FEATURE_COLS])
        cal = float(seg["wash_count_total"].sum()
                    / max(np.expm1(seg["y_anchor"].values + pred).sum(), 1.0))
        out[seg_name] = {"booster": booster, "calibration_multiplier": cal,
                         "residual_sd_log": float(np.std(seg["y_residual"].values - pred))}
    return out


def coldstart_kfold_eval(panel: pd.DataFrame, static: pd.DataFrame,
                         n_folds: int = C.COLD_START_KFOLDS) -> pd.DataFrame:
    """K-fold cold-start cross-validation — the honest, full-coverage metric.

    Every site is held out exactly once. For each fold the held-out sites are
    treated as brand-new:

    * their CBSA / H3 / rollup references are rebuilt from the OTHER folds
      only (leave-fold-out — a held-out site never sees itself as a peer),
      and from the 2024 temporal-training window only (``COLD_REF_*`` knobs) —
      i.e. predicting a new site purely from the *temporal trend of its
      neighbours*;
    * a cohort booster is retrained on the other folds with the same
      reference construction, so the booster never trains on the held-out
      sites' rows;
    * the held-out sites are scored with zero own-history (site-level
      features imputed from peers) and ``cbsa_code = 0`` to match how the
      production engine scores a brand-new market.

    Returns one pooled monthly frame (every site scored once) with
    ``pred_wash`` / ``pred_anchor`` / ``tenure_year`` ready for grading.
    """
    all_sites = np.array(sorted(static["client_id_location_id"].unique()))
    rng = np.random.default_rng(C.SEED)
    folds = np.array_split(rng.permutation(all_sites), n_folds)
    pred_frames = []
    for fi, fold_sites in enumerate(folds):
        fold_sites = set(fold_sites.tolist())
        non_cold = panel[~panel["client_id_location_id"].isin(fold_sites)]
        cold = panel[panel["client_id_location_id"].isin(fold_sites)]
        neigh = (non_cold[non_cold["year_month"] <= C.TRAIN_END_YM]
                 if C.COLD_REF_TRAIN_WINDOW_ONLY else non_cold)
        cbsa_ref_f = market_reference_levels(neigh, static)
        rollups_f = rollup_levels(neigh, static)
        h3_ref_f = local_h3_levels(neigh, static)
        ramp_f = estimate_ramp_curves(non_cold, static)
        site_stats_f = site_training_stats(non_cold, C.TRAIN_END_YM)
        # Booster: trained on the OTHER folds only, 2024 window, same refs.
        tf = build_training_frame(non_cold, static, cbsa_ref_f, rollups_f,
                                  ramp_f, site_stats_f, h3_ref_f)
        tf = tf.dropna(subset=["peer_ref_level"])
        models_f = _fit_lean_cohorts(tf[tf["year_month"] <= C.TRAIN_END_YM])
        if not models_f:
            continue
        # Cold scoring: held-out sites, zero own-history, unknown market code.
        cf = build_training_frame(cold, static, cbsa_ref_f, rollups_f,
                                  ramp_f, site_stats_f, h3_ref_f)
        cf = cf.dropna(subset=["peer_ref_level"]).copy()
        cf["cbsa_code"] = 0
        cf = cf.dropna(subset=FEATURE_COLS)
        if "_imputed" in cf.columns:
            cf = cf[~cf["_imputed"].fillna(False).astype(bool)].copy()
        cf["pred_wash"] = predict_blend(models_f, cf)
        cf["pred_anchor"] = cf["anchor_wash"].clip(lower=0)
        cf["fold"] = fi
        pred_frames.append(cf)
        print(f"  fold {fi + 1}/{n_folds}: {len(fold_sites)} held-out sites, "
              f"{len(cf)} scored rows")
    allcold = pd.concat(pred_frames, ignore_index=True)
    allcold["tenure_year"] = (allcold["site_age_months"] // 12 + 1).astype(int)
    allcold["calendar_year"] = allcold["ym_ts"].dt.year
    return allcold


def run():
    print("\n=== 1. Loading and cleaning data ===")
    panel = load_and_clean()
    static = site_static_table(panel)

    print("\n=== 2. CBSA / MSA market assignment + H3 local hexes ===")
    cbsa_layer = load_cbsa()
    static = assign_markets(static, cbsa_layer)
    static = add_h3(static, res=C.H3_RES)
    print(static[["cbsa_name", "client_id_location_id"]]
          .groupby("cbsa_name").nunique()
          .rename(columns={"client_id_location_id": "n_sites"})
          .sort_values("n_sites", ascending=False).head(15))
    static.to_csv(C.SUMMARY_DIR / "site_market_assignments.csv", index=False)

    cbsa_counts = static["cbsa_id"].value_counts()
    h3_counts = static["h3_id"].value_counts()
    print(f"unique CBSAs: {cbsa_counts.size}, median sites/CBSA: {cbsa_counts.median():.1f}")
    print(f"unique H3 (r={C.H3_RES}) cells: {h3_counts.size}, "
          f"cells with >=2 sites: {(h3_counts>=2).sum()}")

    print("\n=== 3. Peer market reference levels (CBSA + H3 local) ===")
    cbsa_ref = market_reference_levels(panel, static)
    rollups = rollup_levels(panel, static)
    h3_ref = local_h3_levels(panel, static)
    cbsa_ref.to_csv(C.SUMMARY_DIR / "cbsa_reference.csv", index=False)
    h3_ref.to_csv(C.SUMMARY_DIR / "h3_local_reference.csv", index=False)
    for lvl, df in rollups.items():
        df.to_csv(C.SUMMARY_DIR / f"rollup_{lvl}.csv", index=False)
    print(f"H3 reference rows: {len(h3_ref)} (covering "
          f"{h3_ref['h3_id'].nunique() if len(h3_ref) else 0} unique hex disks)")

    print("\n=== 4. Lifecycle ramp curves ===")
    ramp = estimate_ramp_curves(panel, static)
    ramp.to_csv(C.SUMMARY_DIR / "ramp_curves.csv", index=False)

    print("\n=== 5. Per-site training-window stats ===")
    all_sites = sorted(static["client_id_location_id"].unique())
    site_stats = site_training_stats(panel, C.TRAIN_END_YM)
    print(f"site stats computed for {len(site_stats)} sites (train end {C.TRAIN_END_YM})")
    site_stats.to_csv(C.SUMMARY_DIR / "site_training_stats.csv", index=False)

    print("\n=== 6. Feature engineering — temporal split (known sites) ===")
    train_frame = build_training_frame(panel, static, cbsa_ref, rollups, ramp, site_stats, h3_ref)
    train_frame = train_frame.dropna(subset=["peer_ref_level"]).reset_index(drop=True)
    print(f"feature rows: {len(train_frame)}")
    # Temporal split: 2024 rows train, 2025+ rows test. Same sites in both
    # windows — the booster has each site's training-window history.
    in_train_period = train_frame["year_month"] <= C.TRAIN_END_YM
    train_rows = train_frame[in_train_period].copy()
    temporal_test_rows = train_frame[~in_train_period].copy()
    if "_imputed" in temporal_test_rows.columns:
        m = temporal_test_rows["_imputed"].fillna(False).astype(bool)
        temporal_test_rows = temporal_test_rows[~m].copy()
    print(f"train rows (<= {C.TRAIN_END_YM}): {len(train_rows)}")
    print(f"temporal-test rows (> {C.TRAIN_END_YM}): {len(temporal_test_rows)}")

    print("\n=== 7. Train cohort models (production artifact + temporal eval) ===")
    models = train_cohort_models(train_rows, n_folds=5)
    for seg_name, seg_mask in (
        ("young", train_rows["site_age_months"] < C.MATURITY_MONTHS),
        ("mature", train_rows["site_age_months"] >= C.MATURITY_MONTHS),
    ):
        if seg_name not in models: continue
        seg = train_rows[seg_mask].dropna(subset=FEATURE_COLS + ["y"])
        if len(seg) == 0: continue
        b = models[seg_name]["booster"]
        pred = b.predict(seg[FEATURE_COLS])
        models[seg_name]["residual_sd_log"] = float(np.std(seg["y"].values - pred))

    print("\n=== 8. Cold-start k-fold cross-validation ===")
    cold_all = coldstart_kfold_eval(panel, static)
    print(f"cold-start CV: {cold_all['client_id_location_id'].nunique()} sites scored "
          f"as never-seen, {len(cold_all)} monthly rows")

    print("\n=== 9. Evaluate ===")
    def _eval(df, label, annual_mode, predict=True):
        """Evaluate one test split.

        ``annual_mode`` decides the headline annual bucket:
          * ``"tenure"``   — site-age aligned (Year-1 = age 0-11). Correct for
            cold-start: held-out sites opened across all 12 months of 2024.
          * ``"calendar"`` — calendar-year split. Used for the temporal eval,
            which is a calendar split by construction.
        Both bucketings are always computed and stored; only the headline
        print uses ``annual_mode``. When ``predict`` is False the frame is
        expected to already carry ``pred_wash`` / ``pred_anchor`` (the
        cold-start k-fold scores per fold, so it is pre-predicted).
        """
        if len(df) == 0:
            return None
        if predict:
            df = df.dropna(subset=FEATURE_COLS).copy()
            df["pred_wash"] = predict_blend(models, df)
            df["pred_anchor"] = df["anchor_wash"].clip(lower=0)
        else:
            df = df.copy()
        model_m = segment_metrics(df, "pred_wash", "wash_count_total")
        anchor_m = segment_metrics(df, "pred_anchor", "wash_count_total")
        df["calendar_year"] = df["ym_ts"].dt.year
        df["tenure_year"] = (df["site_age_months"] // 12 + 1).astype(int)
        cal_sum, cal_full = annual_hit_miss(df, "calendar_year")
        ten_sum, ten_full = annual_hit_miss(df, "tenure_year")
        if annual_mode == "tenure":
            primary, primary_full, ylab = ten_sum, ten_full, "tenure-year"
        else:
            primary, primary_full, ylab = cal_sum, cal_full, "calendar-year"

        print(f"\n--- {label} ---")
        print(f"  monthly  MAE={model_m['overall']['MAE']:,.0f}"
              f"  RMSE={model_m['overall']['RMSE']:,.0f}"
              f"  WMAPE={model_m['overall']['WMAPE']:.3f}"
              f"  bias={model_m['overall']['bias']:,.0f}"
              f"   (peer-anchor MAE={anchor_m['overall']['MAE']:,.0f})")
        if primary.get("n_site_years"):
            bp = int(primary["band_pct"] * 100)
            bk = int(primary["band_abs"] / 1000)
            print(f"  annual ({ylab} buckets — {primary['n_site_years']} full site-years):")
            print(f"    HIT% within ±{bp}%       = {primary['hit_rate_pct']*100:.1f}%")
            print(f"    HIT% within ±{int(primary['band_abs']):,}   = {primary['hit_rate_abs']*100:.1f}%")
            print(f"    annual WMAPE           = {primary['annual_wmape']:.3f}")
            print(f"    median abs %% error    = {primary['median_pct_error']*100:.1f}%")
            for y, info in sorted(primary["by_year"].items()):
                print(f"      Year {y}: n={info['n_site_years']:3d}  "
                      f"HIT±{bp}%={info['hit_rate_pct']*100:5.1f}%  "
                      f"HIT±{bk}k={info['hit_rate_abs']*100:5.1f}%  "
                      f"med%err={info['median_pct_error']*100:5.1f}%  "
                      f"MAE={info['annual_mae']:,.0f}")
        return {"label": label, "model": model_m, "anchor": anchor_m,
                "annual_mode": annual_mode,
                "annual_tenure": ten_sum, "annual_calendar": cal_sum,
                "annual_df": primary_full, "df": df}

    temporal_eval = _eval(temporal_test_rows,
                          "TEMPORAL split (known sites, 2025+ months)", "calendar")
    cold_eval = _eval(cold_all,
                      "COLD-START k-fold CV (never-seen sites, tenure-aligned)",
                      "tenure", predict=False)
    test_df = temporal_eval["df"] if temporal_eval else cold_eval["df"]

    if temporal_eval:
        mkt_metrics = market_level_metrics(temporal_eval["df"], "pred_wash", "wash_count_total")
        mkt_metrics.to_csv(C.MET_DIR / "market_level_metrics.csv", index=False)
        temporal_eval["annual_df"].to_csv(C.MET_DIR / "temporal_site_year_hits.csv", index=False)
    if cold_eval:
        cold_eval["annual_df"].to_csv(C.MET_DIR / "coldstart_site_year_hits.csv", index=False)

    metrics_out = {
        "temporal_split": {
            "model": temporal_eval["model"],
            "anchor": temporal_eval["anchor"],
            "annual_calendar": temporal_eval["annual_calendar"],
            "annual_tenure": temporal_eval["annual_tenure"],
            "n_test_rows": int(len(temporal_test_rows)),
        } if temporal_eval else None,
        "cold_start_kfold": {
            "model": cold_eval["model"],
            "anchor": cold_eval["anchor"],
            "annual_tenure": cold_eval["annual_tenure"],
            "annual_calendar_unaligned": cold_eval["annual_calendar"],
            "n_test_rows": int(len(cold_all)),
            "n_sites_scored_cold": int(cold_all["client_id_location_id"].nunique()),
            "protocol": {
                "method": f"{C.COLD_START_KFOLDS}-fold cross-validation "
                          "(every site held out once as never-seen)",
                "grading": "tenure-year (Year-1 = site age 0-11 months), "
                           "NOT calendar year — sites open across all of 2024",
                "neighbour_refs": "leave-fold-out" + (
                    ", 2024 temporal-train window only" if C.COLD_REF_TRAIN_WINDOW_ONLY
                    else ", full neighbour history"),
            },
        } if cold_eval else None,
        "train_end_ym": C.TRAIN_END_YM,
        "hit_band": C.HIT_BAND,
        "hit_pct_band": C.ANNUAL_PCT_BAND,
        "exclude_chem": C.EXCLUDE_CHEM,
    }
    _dump_json(metrics_out, C.MET_DIR / "metrics.json")
    # Set model_metrics for the rest of pipeline (plots) — prefer temporal.
    model_metrics = (temporal_eval or cold_eval)["model"]

    print("\n=== 10. Feature importance ===")
    fi = {seg: models[seg]["feature_importance"] for seg in models}
    _dump_json(fi, C.MET_DIR / "feature_importance.json")

    print("\n=== 11. Sample forecasts ===")
    samples = [
        ("Dallas, TX", 32.7767, -96.7970, "TX"),
        ("Atlanta, GA", 33.7490, -84.3880, "GA"),
        ("Phoenix, AZ", 33.4484, -112.0740, "AZ"),
        ("Cleveland, OH", 41.4993, -81.6944, "OH"),
        ("Tampa, FL", 27.9506, -82.4572, "FL"),
        ("Boise, ID", 43.6150, -116.2023, "ID"),
    ]
    from .geo_markets import assign_single
    sample_rows = []
    explanations = []
    for name, lat, lon, st in samples:
        m = assign_single(lat, lon, st, cbsa_layer)
        fc = forecast_site(
            lat, lon, st, pd.Timestamp("2026-06-01"),
            m["cbsa_id"], m["cbsa_name"],
            models, cbsa_ref, rollups, ramp,
            h3_ref=h3_ref, h3_id=m.get("h3_id"),
        )
        fc["sample_label"] = name
        fc["sample_lat"] = lat; fc["sample_lon"] = lon
        sample_rows.append(fc)
        yt = year_totals(fc)
        print(f"  {name:18s} CBSA={(m['cbsa_name'] or 'unknown')[:35]:35s}  "
              f"Yr1={yt['predicted_year_total'].iloc[0]:>8,.0f}  "
              f"Yr5={yt['predicted_year_total'].iloc[4]:>8,.0f}")
        exp = explain_forecast(fc)
        exp["sample_label"] = name
        exp["site_lat"] = lat; exp["site_lon"] = lon; exp["state"] = st
        explanations.append(exp)
    pd.concat(sample_rows, ignore_index=True).to_csv(C.FORECAST_DIR / "sample_forecasts.csv", index=False)
    _dump_json(explanations, C.FORECAST_DIR / "sample_explanations.json")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for fc in sample_rows:
        ax.plot(fc["month_offset"], fc["predicted_washes"], label=fc["sample_label"].iloc[0])
        ax.fill_between(fc["month_offset"], fc["predicted_washes_low"],
                        fc["predicted_washes_high"], alpha=0.07)
    ax.set_title("Sample 60-month cold-start forecasts (model_3)")
    ax.set_xlabel("Month since opening"); ax.set_ylabel("Predicted monthly washes")
    ax.legend(loc="best"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(C.PLOT_DIR / "sample_forecasts.png", dpi=120)
    plt.close(fig)

    # Ramp curves plot
    fig, ax = plt.subplots(figsize=(9, 5))
    nat = ramp[ramp["level"] == "national"]
    ax.plot(nat["age_months"], nat["ramp_factor"], lw=2.5, label="national", color="black")
    for region, g in ramp[ramp["level"] == "region"].groupby("level_value"):
        ax.plot(g["age_months"], g["ramp_factor"], lw=1.2, alpha=0.8, label=region)
    ax.set_title("Lifecycle ramp curves (fraction of mature wash level)")
    ax.set_xlabel("Site age (months)"); ax.set_ylabel("Ramp factor")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(C.PLOT_DIR / "ramp_curves.png", dpi=120)
    plt.close(fig)

    # Market scatter — CBSA peer count
    cbsa_counts_df = static.groupby(["cbsa_id", "cbsa_name"]).size().reset_index(name="n_sites")
    cbsa_counts_df = cbsa_counts_df.sort_values("n_sites", ascending=False)
    cbsa_counts_df.to_csv(C.SUMMARY_DIR / "cbsa_site_counts.csv", index=False)
    fig, ax = plt.subplots(figsize=(11, 5))
    top = cbsa_counts_df.head(25)
    ax.barh(range(len(top)), top["n_sites"])
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([n[:40] for n in top["cbsa_name"].fillna("?")])
    ax.invert_yaxis()
    ax.set_xlabel("# sites"); ax.set_title("Top 25 CBSAs by site count")
    fig.tight_layout(); fig.savefig(C.PLOT_DIR / "top_cbsa_site_counts.png", dpi=120)
    plt.close(fig)

    # Feature importance plot
    if models:
        fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 5))
        if len(models) == 1: axes = [axes]
        for ax_, (seg, m) in zip(axes, models.items()):
            fi_items = sorted(m["feature_importance"].items(), key=lambda kv: kv[1], reverse=True)
            names, vals = zip(*fi_items)
            ax_.barh(range(len(names)), vals)
            ax_.set_yticks(range(len(names))); ax_.set_yticklabels(names)
            ax_.invert_yaxis(); ax_.set_title(f"{seg} model — gain importance")
        fig.tight_layout(); fig.savefig(C.PLOT_DIR / "feature_importance.png", dpi=120)
        plt.close(fig)

    # Holdout pred vs actual scatter
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(test_df["wash_count_total"], test_df["pred_wash"], s=6, alpha=0.35)
    mx = max(test_df["wash_count_total"].max(), test_df["pred_wash"].max())
    ax.plot([0, mx], [0, mx], color="red", lw=1, ls="--")
    ax.set_xlabel("Actual monthly washes"); ax.set_ylabel("Predicted")
    ax.set_title(f"Holdout — pred vs actual (MAE={model_metrics['overall']['MAE']:,.0f})")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(C.PLOT_DIR / "holdout_pred_vs_actual.png", dpi=120)
    plt.close(fig)

    print("\n=== 12. Save artifacts ===")
    artifact = {
        "models": {seg: {
            "booster_txt": models[seg]["booster"].model_to_string(),
            "feature_importance": models[seg]["feature_importance"],
            "residual_sd_log": models[seg].get("residual_sd_log"),
            "final_rounds": models[seg]["final_rounds"],
            "n_rows": models[seg]["n_rows"],
        } for seg in models},
        "feature_cols": FEATURE_COLS,
        "cbsa_ref": cbsa_ref.to_dict(orient="list"),
        "rollups": {k: v.to_dict(orient="list") for k, v in rollups.items()},
        "ramp": ramp.to_dict(orient="list"),
        "h3_ref": h3_ref.to_dict(orient="list") if h3_ref is not None and len(h3_ref) else None,
        "config": {
            "OBSERVED_CUTOFF_YM": C.OBSERVED_CUTOFF_YM,
            "MIN_PEERS_PER_MARKET": C.MIN_PEERS_PER_MARKET,
            "MATURITY_MONTHS": C.MATURITY_MONTHS,
            "FORECAST_HORIZON_MONTHS": C.FORECAST_HORIZON_MONTHS,
            "HIT_BAND": C.HIT_BAND,
        },
    }
    with open(C.ART_DIR / "model3_artifacts.pkl", "wb") as f:
        pickle.dump(artifact, f)
    print(f"saved: {C.ART_DIR / 'model3_artifacts.pkl'}")

    print("\n=== DONE ===")
    return metrics_out


if __name__ == "__main__":
    run()
