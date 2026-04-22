"""Ridge vs tree models — same splits/features as V2 eval; extra slices for a decision.

Computes:
  >2y: daily row-level (same grain as backtest_v2.py), monthly aggregate (accuracy report grain),
       heavy-day subset, Q3 vs Q4 2025 test windows.
  <2y: monthly panel test rows, subset actual >= 2000.

Writes results/ridge_vs_trees_decision_report.json (and prints summary).

Run:
  python daily_data/daily-data-modelling/clustering_v2/eval_ridge_vs_trees_holdout.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

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

try:
    from xgboost import XGBRegressor

    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False


def _base_pipe(estimator) -> Pipeline:
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("m", estimator)])


def _metrics_block(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    if len(y) == 0:
        return {"n": 0, "mae": None, "rmse": None, "r2": None, "wape": None}
    return {
        "n": int(len(y)),
        "mae": round(float(mean_absolute_error(y, p)), 4),
        "rmse": round(float(np.sqrt(np.mean((y - p) ** 2))), 4),
        "r2": round(float(r2_score(y, p)), 4),
        "wape": round(float(bv._wape(y, p)), 4),
    }


def _prepare_more_than() -> dict[str, Any]:
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
    return {"train_ctx": train_ctx, "test_ctx": test_ctx, "f_v2": f_v2}


def _prepare_less_than() -> dict[str, Any]:
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
    return {"train_ctx": train_ctx, "test_ctx": test_ctx, "f2": f2}


def _analyze_more_than(name: str, model: Pipeline, bundle: dict[str, Any]) -> dict[str, Any]:
    train_ctx, test_ctx, f_v2 = bundle["train_ctx"], bundle["test_ctx"], bundle["f_v2"]
    model.fit(train_ctx[f_v2].to_numpy(float), train_ctx[TARGET].to_numpy(float))
    p = model.predict(test_ctx[f_v2].to_numpy(float))
    y = test_ctx[TARGET].to_numpy(float)
    cd = test_ctx["calendar_day"]

    out: dict[str, Any] = {"estimator": name, "cohort": "more_than_2yrs"}
    out["daily_test_rows"] = _metrics_block(y, p)

    thr = float(np.nanpercentile(y, 80))
    mask80 = y >= thr
    out["daily_subset_actual_ge_p80_of_test_y"] = {
        "threshold": round(thr, 2),
        "definition": "Top 20% busiest test days by actual wash_count_total.",
        **_metrics_block(y[mask80], p[mask80]),
    }
    mask200 = y >= 200
    out["daily_subset_actual_ge_200"] = _metrics_block(y[mask200], p[mask200])

    tc = test_ctx.copy()
    tc["_pred"] = p
    tc["year_month"] = tc["calendar_day"].dt.to_period("M")
    g = tc.groupby(["site_client_id", "year_month"], as_index=False).agg(y_sum=(TARGET, "sum"), p_sum=("_pred", "sum"))
    out["monthly_aggregate_site_month"] = _metrics_block(g["y_sum"].to_numpy(float), g["p_sum"].to_numpy(float))
    out["n_site_months"] = int(len(g))

    q3 = (cd >= pd.Timestamp("2025-07-01")) & (cd < pd.Timestamp("2025-10-01"))
    q4 = (cd >= pd.Timestamp("2025-10-01")) & (cd <= pd.Timestamp("2025-12-31"))
    out["daily_by_test_window"] = {
        "2025_q3_jul_sep": _metrics_block(y[q3], p[q3]),
        "2025_q4_oct_dec": _metrics_block(y[q4], p[q4]),
    }
    return out


def _analyze_less_than(name: str, model: Pipeline, bundle: dict[str, Any]) -> dict[str, Any]:
    train_ctx, test_ctx, f2 = bundle["train_ctx"], bundle["test_ctx"], bundle["f2"]
    model.fit(train_ctx[f2].to_numpy(float), train_ctx[TARGET].to_numpy(float))
    p2 = model.predict(test_ctx[f2].to_numpy(float))
    y2 = test_ctx[TARGET].to_numpy(float)
    out: dict[str, Any] = {
        "estimator": name,
        "cohort": "less_than_2yrs",
        "monthly_all_test_rows": _metrics_block(y2, p2),
        "n_test_rows": int(len(test_ctx)),
    }
    m2k = y2 >= 2000
    out["monthly_subset_actual_ge_2000"] = _metrics_block(y2[m2k], p2[m2k])
    return out


def _estimator_specs() -> list[tuple[str, Any]]:
    specs: list[tuple[str, Any]] = [
        ("ridge", Ridge(alpha=1.0, random_state=0)),
        (
            "random_forest",
            RandomForestRegressor(
                n_estimators=300,
                max_depth=16,
                min_samples_leaf=4,
                random_state=0,
                n_jobs=-1,
            ),
        ),
    ]
    if _HAS_XGB:
        specs.append(
            (
                "xgboost",
                XGBRegressor(
                    n_estimators=400,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_lambda=1.0,
                    random_state=0,
                    n_jobs=-1,
                ),
            )
        )
    return specs


def _decision_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Heuristic summary for humans; not a statistical test."""

    def pick(cohort: str, key_path: list[str]) -> dict[str, dict[str, Any]]:
        out = {}
        for r in rows:
            if r.get("cohort") != cohort:
                continue
            est = r["estimator"]
            cur: Any = r
            for k in key_path:
                cur = cur[k]
            out[est] = cur
        return out

    more_daily = pick("more_than_2yrs", ["daily_test_rows"])
    more_mo = pick("more_than_2yrs", ["monthly_aggregate_site_month"])
    less_all = pick("less_than_2yrs", ["monthly_all_test_rows"])

    def best_wape(d: dict[str, dict[str, Any]]) -> str | None:
        ok = {k: v["wape"] for k, v in d.items() if v.get("wape") is not None}
        if not ok:
            return None
        return min(ok, key=lambda k: ok[k])

    ridge_d_wape = more_daily.get("ridge", {}).get("wape")
    rf_d_wape = more_daily.get("random_forest", {}).get("wape")

    bullets = [
        f">2y daily row WAPE — best: {best_wape(more_daily)} (same grain as backtest_v2).",
        f">2y monthly aggregate WAPE — best: {best_wape(more_mo)} (accuracy-report grain).",
        f"<2y monthly WAPE — best: {best_wape(less_all)}.",
    ]
    if ridge_d_wape is not None and rf_d_wape is not None and rf_d_wape < ridge_d_wape - 0.02:
        bullets.append(
            "Trees beat Ridge on **daily** >2y by a clear margin — uplift is not only from monthly summation."
        )
    elif ridge_d_wape is not None and rf_d_wape is not None:
        bullets.append(
            ">2y daily: trees vs Ridge are close — large monthly-only gains deserve scrutiny."
        )

    return {"bullets": bullets, "best_by_slice": {"more_daily_wape": best_wape(more_daily), "more_monthly_wape": best_wape(more_mo), "less_monthly_wape": best_wape(less_all)}}


def main() -> None:
    print("[prep] >2y...", flush=True)
    more_b = _prepare_more_than()
    print("[prep] <2y...", flush=True)
    less_b = _prepare_less_than()

    rows: list[dict[str, Any]] = []
    for name, est in _estimator_specs():
        print(f"[fit] {name} >2y...", flush=True)
        rows.append(_analyze_more_than(name, _base_pipe(est), more_b))
        print(f"[fit] {name} <2y...", flush=True)
        rows.append(_analyze_less_than(name, _base_pipe(est), less_b))

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reference": {
            "ridge_backtest_v2_daily_wape_more_than": 0.385,
            "note": "backtest_summary.json V2_final wape for >2y daily rows (~0.385). Compare to daily_test_rows.wape below.",
        },
        "xgboost_included": _HAS_XGB,
        "splits": {
            "more_than": "train calendar_day < 2025-07-01; test >= 2025-07-01 (Jul–Dec 2025 in data)",
            "less_than": "train period_index<=12; test period_index>=25",
        },
        "results": rows,
        "decision_summary": _decision_summary(rows),
    }
    outp = V2 / "results" / "ridge_vs_trees_decision_report.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    print(json.dumps(out["decision_summary"], indent=2))
    print(json.dumps(rows, indent=2))
    print(f"\nwrote {outp.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
