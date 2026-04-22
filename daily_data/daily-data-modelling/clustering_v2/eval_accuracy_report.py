"""Monthly holdout eval: Ridge + RandomForest (same splits as V2). Writes results/monthly_level_holdout_eval.json."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

V2 = Path(__file__).resolve().parent
REPO_ROOT = V2.parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(V2))

import build_v2 as bv  # noqa: E402

TARGET = bv.TARGET


def _pipe_ridge() -> Pipeline:
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("m", Ridge(alpha=1.0, random_state=0)),
        ]
    )


def _pipe_rf() -> Pipeline:
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            (
                "m",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=16,
                    min_samples_leaf=4,
                    random_state=0,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def _rel_bands(y: np.ndarray, p: np.ndarray, thresholds=(0.10, 0.15, 0.20, 0.30)) -> dict[str, Any]:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p) & (y > 0)
    y, p = y[m], p[m]
    if len(y) == 0:
        return {"n": 0}
    ape = np.abs(p - y) / y
    out: dict[str, Any] = {"n": int(len(y))}
    for t in thresholds:
        out[f"pct_error_within_{int(t * 100)}pct"] = round(100 * float(np.mean(ape <= t)), 2)
    out["median_abs_pct_error"] = round(float(np.median(ape)) * 100, 2)
    out["mean_abs_pct_error"] = round(float(np.mean(ape)) * 100, 2)
    return out


def _train_monthly_wash_min_max_by_cluster(train_df: pd.DataFrame) -> pd.DataFrame:
    """Per cluster: min / max of train **monthly** wash totals (sum daily rows to site×month). Same grain as >2y eval."""
    t = train_df[["site_client_id", "dbscan_cluster_12km", "calendar_day", TARGET]].copy()
    t["calendar_day"] = pd.to_datetime(t["calendar_day"], errors="coerce")
    t = t.dropna(subset=["calendar_day"])
    t = t.loc[t["dbscan_cluster_12km"].astype(int) != -1]
    t["year_month"] = t["calendar_day"].dt.to_period("M")
    per_sm = t.groupby(["site_client_id", "dbscan_cluster_12km", "year_month"], as_index=False)[TARGET].sum()
    return per_sm.groupby("dbscan_cluster_12km", as_index=False).agg(
        ctx_train_monthly_wash_sum_min=(TARGET, "min"),
        ctx_train_monthly_wash_sum_max=(TARGET, "max"),
    )


def _subset(y, p, min_y, thresholds=(0.10, 0.15, 0.20, 0.30)):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p) & (y >= min_y)
    y, p = y[m], p[m]
    if len(y) == 0:
        return {"n": 0, "note": f"actual_month_total >= {min_y}"}
    ape = np.abs(p - y) / y
    out: dict[str, Any] = {"n": int(len(y)), "note": f"actual_month_total >= {min_y}"}
    for t in thresholds:
        out[f"pct_error_within_{int(t * 100)}pct"] = round(100 * float(np.mean(ape <= t)), 2)
    out["median_abs_pct_error"] = round(float(np.median(ape)) * 100, 2)
    return out


def _prepare_more_than() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    master = pd.read_csv(REPO_ROOT / "daily_data/daily-data-modelling/master_more_than-2yrs.csv", low_memory=False)
    more = pd.read_csv(REPO_ROOT / "daily_data/daily-data-modelling/more_than-2yrs.csv", low_memory=False)
    master["calendar_day"] = pd.to_datetime(master["calendar_day"], errors="coerce")
    more["calendar_day"] = pd.to_datetime(more["calendar_day"], errors="coerce")
    extra = [
        c
        for c in (
            ["dbscan_cluster_12km"]
            + bv.TIME_LAG_FEATURES_DAILY
            + bv.ANNUAL_WEATHER_FEATURES
            + ["region_enc", "state_enc", "costco_enc", "carwash_type_encoded"]
        )
        if c in more.columns and c not in master.columns
    ]
    df = master.merge(more[["site_client_id", "calendar_day"] + extra], on=["site_client_id", "calendar_day"], how="left")
    train = df[df["calendar_day"] < pd.Timestamp("2025-07-01")].copy().dropna(subset=[TARGET])
    test = df[df["calendar_day"] >= pd.Timestamp("2025-07-01")].copy().dropna(subset=[TARGET])
    st = (
        train.dropna(subset=["latitude", "longitude"])
        .drop_duplicates("site_client_id")[["site_client_id", "latitude", "longitude"]]
    )
    st = st.merge(bv._fit_dbscan(st, "cluster_id_train"), on="site_client_id", how="left")
    train, test = bv.align_train_test_clusters_to_train_refit(train, test, st)
    ctx = bv._build_cluster_context(train, "dbscan_cluster_12km", bv.CONTEXT_BASE_FEATURES)
    train_ctx = train.merge(ctx, left_on="dbscan_cluster_12km", right_on="cluster_id", how="left").drop(columns=["cluster_id"])
    test_ctx = test.merge(ctx, left_on="dbscan_cluster_12km", right_on="cluster_id", how="left").drop(columns=["cluster_id"])
    ctx_cols = [c for c in train_ctx.columns if c.startswith("ctx_")]
    f_v2 = bv.SITE_FEATURES_STATIC + bv.ANNUAL_WEATHER_FEATURES + bv.TIME_LAG_FEATURES_DAILY + bv.DAILY_WEATHER_FEATURES + ctx_cols
    f_v2 = [c for c in f_v2 if c in train_ctx.columns and train_ctx[c].notna().any()]
    return train_ctx, test_ctx, f_v2


def _eval_more_than_monthly_aggregate(
    train_ctx: pd.DataFrame,
    test_ctx: pd.DataFrame,
    f_v2: list[str],
    model: Pipeline,
    model_label: str,
) -> dict[str, Any]:
    model.fit(train_ctx[f_v2].to_numpy(float), train_ctx[TARGET].to_numpy(float))
    pred = model.predict(test_ctx[f_v2].to_numpy(float))
    tc = test_ctx.copy()
    tc["_pred"] = pred
    tc["year_month"] = tc["calendar_day"].dt.to_period("M")
    g = tc.groupby(["site_client_id", "year_month"], as_index=False).agg(
        y_sum=(TARGET, "sum"),
        p_sum=("_pred", "sum"),
        cluster=("dbscan_cluster_12km", "first"),
    )
    ym_y = g["y_sum"].to_numpy(float)
    ym_p = g["p_sum"].to_numpy(float)

    band = _train_monthly_wash_min_max_by_cluster(train_ctx)
    g = g.merge(band, left_on="cluster", right_on="dbscan_cluster_12km", how="left")
    lo = g["ctx_train_monthly_wash_sum_min"].to_numpy(float)
    hi = g["ctx_train_monthly_wash_sum_max"].to_numpy(float)
    m_band = np.isfinite(lo) & np.isfinite(hi)
    pr2 = (
        round(100 * float(np.mean((ym_p[m_band] >= lo[m_band]) & (ym_p[m_band] <= hi[m_band]))), 2)
        if int(m_band.sum()) > 0
        else None
    )
    ir2 = (
        round(100 * float(np.mean((ym_y[m_band] >= lo[m_band]) & (ym_y[m_band] <= hi[m_band]))), 2)
        if int(m_band.sum()) > 0
        else None
    )

    return {
        "cohort": "more_than_2yrs",
        "level_model": model_label,
        "grain": "monthly_aggregate_from_daily_predictions",
        "definition": "Each row = one site × one calendar month in test window; actual = sum(daily washes); pred = sum(daily level-model preds).",
        "split": "train calendar_day < 2025-07-01; test calendar_day >= 2025-07-01",
        "peer_band_definition": (
            "Train-only, per cluster: min and max of peer **monthly** wash totals "
            "(each total = sum of TARGET over calendar month for one site). "
            "Compared to eval rows at the same monthly grain."
        ),
        "n_site_months": int(len(g)),
        "mae_total_washes_per_month": round(float(mean_absolute_error(ym_y, ym_p)), 2),
        "rmse_total_washes_per_month": round(float(np.sqrt(np.mean((ym_y - ym_p) ** 2))), 2),
        "r2": round(float(r2_score(ym_y, ym_p)), 4),
        "wape": round(float(bv._wape(ym_y, ym_p)), 4),
        "pred_inside_cluster_train_min_max_pct": pr2,
        "actual_inside_cluster_train_min_max_pct": ir2,
        "pct_error_vs_actual_month_total": _rel_bands(ym_y, ym_p),
        "pct_error_subset_heavy_months_actual_ge_3000": _subset(ym_y, ym_p, 3000),
    }


def _prepare_less_than() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    df = pd.read_csv(REPO_ROOT / "daily_data/daily-data-modelling/less_than-2yrs-clustering-ready.csv", low_memory=False)
    df["calendar_day"] = pd.to_datetime(df["calendar_day"], errors="coerce")
    train = df[df["period_index"] <= 12].copy().dropna(subset=[TARGET])
    test = df[df["period_index"] >= 25].copy().dropna(subset=[TARGET])
    st = (
        train.dropna(subset=["latitude", "longitude"])
        .drop_duplicates("site_client_id")[["site_client_id", "latitude", "longitude"]]
    )
    st = st.merge(bv._fit_dbscan(st, "cluster_id_train"), on="site_client_id", how="left")
    train, test = bv.align_train_test_clusters_to_train_refit(train, test, st)
    ctx = bv._build_cluster_context(
        train, "dbscan_cluster_12km", bv.CONTEXT_BASE_FEATURES_MONTHLY, include_target_aggs=True
    )
    train_ctx = train.merge(ctx, left_on="dbscan_cluster_12km", right_on="cluster_id", how="left").drop(columns=["cluster_id"])
    test_ctx = test.merge(ctx, left_on="dbscan_cluster_12km", right_on="cluster_id", how="left").drop(columns=["cluster_id"])
    ctx_cols = [c for c in train_ctx.columns if c.startswith("ctx_")]
    f2 = bv.SITE_FEATURES_MONTHLY + bv.TIME_LAG_FEATURES_MONTHLY + ["dbscan_cluster_12km"] + ctx_cols
    f2 = [c for c in f2 if c in train_ctx.columns and train_ctx[c].notna().any()]
    return train_ctx, test_ctx, f2


def _eval_less_than_monthly_rows(
    train_ctx: pd.DataFrame,
    test_ctx: pd.DataFrame,
    f2: list[str],
    model: Pipeline,
    model_label: str,
) -> dict[str, Any]:
    model.fit(train_ctx[f2].to_numpy(float), train_ctx[TARGET].to_numpy(float))
    p2 = model.predict(test_ctx[f2].to_numpy(float))
    y2 = test_ctx[TARGET].to_numpy(float)
    lo2 = test_ctx["ctx_wash_count_total_min"].to_numpy(float)
    hi2 = test_ctx["ctx_wash_count_total_max"].to_numpy(float)
    pr2 = round(100 * float(np.mean((p2 >= lo2) & (p2 <= hi2))), 2)
    ir2 = round(100 * float(np.mean((y2 >= lo2) & (y2 <= hi2))), 2)
    return {
        "cohort": "less_than_2yrs",
        "level_model": model_label,
        "grain": "monthly_panel_one_row_per_site_month",
        "definition": "Each test row is already monthly wash_count_total; model predicts that value.",
        "split": "train period_index<=12; test period_index>=25 (see APPROACH.md)",
        "n_test_rows": int(len(test_ctx)),
        "mae_washes_per_month": round(float(mean_absolute_error(y2, p2)), 2),
        "rmse_washes_per_month": round(float(np.sqrt(np.mean((y2 - p2) ** 2))), 2),
        "r2": round(float(r2_score(y2, p2)), 4),
        "wape": round(float(bv._wape(y2, p2)), 4),
        "pred_inside_cluster_train_min_max_pct": pr2,
        "actual_inside_cluster_train_min_max_pct": ir2,
        "pct_error_vs_actual_month_total": _rel_bands(y2, p2),
        "pct_error_subset_actual_ge_2000": _subset(y2, p2, 2000),
    }


def main() -> None:
    tr_m, te_m, f_m = _prepare_more_than()
    tr_l, te_l, f_l = _prepare_less_than()

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evaluation": "V2 holdout — Ridge + RandomForest level models (monthly figures)",
        "more_than_2yrs": {
            "ridge": _eval_more_than_monthly_aggregate(tr_m, te_m, f_m, _pipe_ridge(), "ridge"),
            "random_forest": _eval_more_than_monthly_aggregate(tr_m, te_m, f_m, _pipe_rf(), "random_forest"),
        },
        "less_than_2yrs": {
            "ridge": _eval_less_than_monthly_rows(tr_l, te_l, f_l, _pipe_ridge(), "ridge"),
            "random_forest": _eval_less_than_monthly_rows(tr_l, te_l, f_l, _pipe_rf(), "random_forest"),
        },
    }
    outp = V2 / "results" / "monthly_level_holdout_eval.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    # Back-compat copy for readers of old filename
    legacy = V2 / "results" / "monthly_ridge_holdout_eval.json"
    legacy.write_text(json.dumps(out, indent=2))
    old = V2 / "results" / "accuracy_pct_bands.json"
    if old.exists():
        old.unlink()
        print("removed legacy results/accuracy_pct_bands.json", file=sys.stderr)
    print(json.dumps(out, indent=2))
    print(f"\nwrote {outp.relative_to(REPO_ROOT)}")
    print(f"wrote {legacy.relative_to(REPO_ROOT)} (same content; legacy name)")


if __name__ == "__main__":
    main()
