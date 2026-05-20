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
    segment_metrics, market_level_metrics,
)


def _dump_json(obj, path):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)): return o.item()
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, pd.Timestamp): return o.isoformat()
        return str(o)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=conv)


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

    print("\n=== 5. Split design (cold holdout first) ===")
    # Decide cold-start holdout BEFORE computing site stats so we can null
    # them for held-out sites and prevent leakage.
    all_sites = sorted(static["client_id_location_id"].unique())
    rng = np.random.default_rng(C.SEED)
    cold_holdout = set(rng.choice(all_sites,
                                   size=max(1, int(C.COLD_START_HOLDOUT_FRAC * len(all_sites))),
                                   replace=False))
    print(f"cold holdout sites: {len(cold_holdout)} of {len(all_sites)}")

    print("\n=== 6. Per-site training-window stats (excluding cold holdout) ===")
    # Site stats use rows year_month <= TRAIN_END_YM AND only NON-cold sites,
    # so held-out sites genuinely have no own-history signal at score time.
    non_cold_panel = panel[~panel["client_id_location_id"].isin(cold_holdout)]
    site_stats = site_training_stats(non_cold_panel, C.TRAIN_END_YM)
    print(f"site stats computed for {len(site_stats)} non-cold sites (train end {C.TRAIN_END_YM})")
    site_stats.to_csv(C.SUMMARY_DIR / "site_training_stats.csv", index=False)

    print("\n=== 7. Feature engineering ===")
    train_frame = build_training_frame(panel, static, cbsa_ref, rollups, ramp, site_stats, h3_ref)
    train_frame = train_frame.dropna(subset=["peer_ref_level"]).reset_index(drop=True)
    print(f"feature rows: {len(train_frame)}")
    # Temporal split: train rows = (not in cold holdout) AND (year_month <= TRAIN_END_YM)
    # Temporal test rows = (not in cold holdout) AND (year_month > TRAIN_END_YM)
    in_cold = train_frame["client_id_location_id"].isin(cold_holdout)
    in_train_period = train_frame["year_month"] <= C.TRAIN_END_YM
    train_rows = train_frame[~in_cold & in_train_period].copy()
    temporal_test_rows = train_frame[~in_cold & ~in_train_period].copy()
    # Cold-start sites: site_train_* are already imputed from peer reference
    # (the fillna in build_training_frame), so this evaluation is genuinely
    # zero-history.
    cold_test_rows = train_frame[in_cold].copy()
    # For cold-start eval we only score on actually-observed rows (drop our
    # own imputed months so we're not testing the imputer).
    if "_imputed" in cold_test_rows.columns:
        mask = cold_test_rows["_imputed"].fillna(False).astype(bool)
        cold_test_rows = cold_test_rows[~mask].copy()
    if "_imputed" in temporal_test_rows.columns:
        mask = temporal_test_rows["_imputed"].fillna(False).astype(bool)
        temporal_test_rows = temporal_test_rows[~mask].copy()
    print(f"cold holdout sites: {len(cold_holdout)}")
    print(f"train rows (pre-{C.TRAIN_END_YM}, known sites): {len(train_rows)}")
    print(f"temporal-test rows (post-{C.TRAIN_END_YM}, known sites, observed only): {len(temporal_test_rows)}")
    print(f"cold-test rows (held-out sites, observed only): {len(cold_test_rows)}")

    print("\n=== 8. Train ===")
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

    print("\n=== 9. Evaluate ===")
    def _eval(df, label):
        if len(df) == 0:
            return None
        df = df.dropna(subset=FEATURE_COLS).copy()
        df["pred_wash"] = predict_blend(models, df)
        df["pred_anchor"] = df["anchor_wash"].clip(lower=0)
        model_m = segment_metrics(df, "pred_wash", "wash_count_total")
        anchor_m = segment_metrics(df, "pred_anchor", "wash_count_total")
        # Annual HIT/MISS by calendar year on the test period.
        df["calendar_year"] = df["ym_ts"].dt.year
        annual = df.groupby(["client_id_location_id", "calendar_year"]).agg(
            actual=("wash_count_total", "sum"),
            predicted=("pred_wash", "sum"),
            n_months=("wash_count_total", "size"),
        ).reset_index()
        # Match the upstream report: require near-full year of observations.
        annual_full = annual[annual["n_months"] >= 10].copy()
        annual_full["abs_error"] = (annual_full["actual"] - annual_full["predicted"]).abs()
        annual_full["hit"] = (annual_full["abs_error"] <= C.HIT_BAND).astype(int)
        hm = {
            "overall_hit_rate": float(annual_full["hit"].mean()) if len(annual_full) else float("nan"),
            "n_site_years_fullyear": int(len(annual_full)),
            "n_site_years_any":      int(len(annual)),
            "annual_mae":            float(annual_full["abs_error"].mean()) if len(annual_full) else float("nan"),
            "by_year":               {int(y): {
                "hit_rate": float(annual_full[annual_full["calendar_year"]==y]["hit"].mean()) if (annual_full["calendar_year"]==y).any() else None,
                "n": int((annual_full["calendar_year"]==y).sum()),
            } for y in sorted(annual_full["calendar_year"].unique())},
        }
        print(f"\n--- {label} ---")
        print("  monthly MAE:    ", f"{model_m['overall']['MAE']:,.0f}",
              " | RMSE:", f"{model_m['overall']['RMSE']:,.0f}",
              " | WMAPE:", f"{model_m['overall']['WMAPE']:.3f}")
        print("  anchor MAE:     ", f"{anchor_m['overall']['MAE']:,.0f}")
        print(f"  annual HIT%(±{C.HIT_BAND:,}): "
              f"{hm['overall_hit_rate']*100:.1f}%  "
              f"({hm['n_site_years_fullyear']} full-year site-years; {hm['n_site_years_any']} any-months)")
        for y, info in hm["by_year"].items():
            print(f"    {y}: hit={info['hit_rate']*100:.1f}%  n={info['n']}")
        return {"label": label, "model": model_m, "anchor": anchor_m,
                "hit_miss_annual": hm, "annual_df": annual_full, "df": df}

    temporal_eval = _eval(temporal_test_rows, "TEMPORAL split (known sites, 2025+ months)")
    cold_eval = _eval(cold_test_rows, "COLD-START split (never-seen sites)")
    test_df = temporal_eval["df"] if temporal_eval else cold_eval["df"]

    if temporal_eval:
        mkt_metrics = market_level_metrics(temporal_eval["df"], "pred_wash", "wash_count_total")
        mkt_metrics.to_csv(C.MET_DIR / "market_level_metrics.csv", index=False)
        temporal_eval["annual_df"].to_csv(C.MET_DIR / "temporal_site_year_hits.csv", index=False)
    if cold_eval:
        cold_eval["annual_df"].to_csv(C.MET_DIR / "coldstart_site_year_hits.csv", index=False)

    metrics_out = {
        "temporal_split": {
            "model": temporal_eval["model"] if temporal_eval else None,
            "anchor": temporal_eval["anchor"] if temporal_eval else None,
            "hit_miss_annual": temporal_eval["hit_miss_annual"] if temporal_eval else None,
            "n_test_rows": int(len(temporal_test_rows)),
        } if temporal_eval else None,
        "cold_start_split": {
            "model": cold_eval["model"] if cold_eval else None,
            "anchor": cold_eval["anchor"] if cold_eval else None,
            "hit_miss_annual": cold_eval["hit_miss_annual"] if cold_eval else None,
            "n_test_rows": int(len(cold_test_rows)),
            "n_cold_sites": len(cold_holdout),
        } if cold_eval else None,
        "train_end_ym": C.TRAIN_END_YM,
        "hit_band": C.HIT_BAND,
    }
    _dump_json({k: v for k, v in metrics_out.items() if not isinstance(v, dict) or "annual_df" not in v},
               C.MET_DIR / "metrics.json")
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
