"""Ridge backtest: >2y V1 / V1+daily / V2; <2y V1 vs V2 (wash-only ctx). Writes results/backtest_summary.json."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import build_v2 as bv

REPO_ROOT = bv.REPO_ROOT
TARGET = bv.TARGET
V2_DIR = bv.V2_DIR


def _fit_eval(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> dict[str, float]:
    features = [c for c in features if c in train.columns and train[c].notna().any()]
    X_tr, y_tr = train[features].to_numpy(dtype=float), train[TARGET].to_numpy(dtype=float)
    X_te, y_te = test[features].to_numpy(dtype=float), test[TARGET].to_numpy(dtype=float)
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("m", Ridge(alpha=1.0, random_state=0)),
    ])
    pipe.fit(X_tr, y_tr)
    pr = pipe.predict(X_te)
    return {
        "n_features": len(features),
        "mae": float(mean_absolute_error(y_te, pr)),
        "rmse": float(np.sqrt(np.mean((y_te - pr) ** 2))),
        "r2": float(r2_score(y_te, pr)),
        "wape": bv._wape(y_te, pr),
    }


def backtest_more_than() -> dict[str, dict[str, float]]:
    print("[>2y backtest] loading and merging...")
    master = pd.read_csv(REPO_ROOT / "daily_data/daily-data-modelling/master_more_than-2yrs.csv", low_memory=False)
    more = pd.read_csv(REPO_ROOT / "daily_data/daily-data-modelling/more_than-2yrs.csv", low_memory=False)
    master["calendar_day"] = pd.to_datetime(master["calendar_day"], errors="coerce")
    more["calendar_day"] = pd.to_datetime(more["calendar_day"], errors="coerce")
    extra = [
        c for c in (
            ["dbscan_cluster_12km"] + bv.TIME_LAG_FEATURES_DAILY
            + bv.ANNUAL_WEATHER_FEATURES
            + ["region_enc", "state_enc", "costco_enc", "carwash_type_encoded"]
        ) if c in more.columns and c not in master.columns
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

    f_v1base = bv.SITE_FEATURES_STATIC + bv.ANNUAL_WEATHER_FEATURES + bv.TIME_LAG_FEATURES_DAILY + ["dbscan_cluster_12km"]
    f_v1plus = f_v1base + bv.DAILY_WEATHER_FEATURES
    f_v2 = f_v1plus + ctx_cols

    print("[>2y] V1-baseline...")
    m_v1 = _fit_eval(train, test, f_v1base)
    print("[>2y] V1-plus (+ daily weather)...")
    m_v1p = _fit_eval(train, test, f_v1plus)
    print("[>2y] V2 (+ cluster context)...")
    m_v2 = _fit_eval(train_ctx, test_ctx, f_v2)
    return {"V1_baseline": m_v1, "V1_plus_daily_weather": m_v1p, "V2_final": m_v2}


def backtest_less_than() -> dict[str, dict[str, float]]:
    print("[<2y backtest] loading...")
    df = pd.read_csv(REPO_ROOT / "daily_data/daily-data-modelling/less_than-2yrs-clustering-ready.csv", low_memory=False)
    df["calendar_day"] = pd.to_datetime(df["calendar_day"], errors="coerce")
    # period_index encoding: year1 -> 1-12, year2 -> 25-36 (month_number 13-24); see APPROACH.md.
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

    f_v1base = bv.SITE_FEATURES_MONTHLY + bv.TIME_LAG_FEATURES_MONTHLY + ["dbscan_cluster_12km"]
    f_v2 = f_v1base + ctx_cols

    print("[<2y] V1-baseline...")
    m_v1 = _fit_eval(train, test, f_v1base)
    print("[<2y] V2 (+ cluster context)...")
    m_v2 = _fit_eval(train_ctx, test_ctx, f_v2)
    return {"V1_baseline": m_v1, "V2_final": m_v2}


def main() -> None:
    out = {
        "split_more_than_2yrs": "train<2025-07-01, test>=2025-07-01",
        "split_less_than_2yrs": (
            "train period_index<=12 (year_number=1, month_number 1-12); "
            "test period_index>=25 (year_number=2, month_number 13-24); see APPROACH.md"
        ),
        "more_than_2yrs": backtest_more_than(),
        "less_than_2yrs": backtest_less_than(),
    }
    (V2_DIR / "results" / "backtest_summary.json").write_text(json.dumps(out, indent=2))
    print("\n=== Backtest summary ===")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
