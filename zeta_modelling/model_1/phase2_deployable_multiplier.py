from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values(["site_id", "date"]).reset_index(drop=True)
    out["month"] = out["date"].dt.month
    out["year"] = out["date"].dt.year
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    out["age_sq"] = out["age_in_months"] ** 2
    out["log_age"] = np.log1p(out["age_in_months"])
    out["is_early"] = (out["age_in_months"] <= 6).astype(int)
    out["is_growth"] = ((out["age_in_months"] > 6) & (out["age_in_months"] <= 18)).astype(int)
    out["is_mature"] = (out["age_in_months"] > 18).astype(int)
    out["lat_lon_interaction"] = out["latitude"] * out["longitude"]
    return out


def add_cluster_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cluster_month_avg = (
        out.groupby(["cluster_id", "month"], as_index=False)["monthly_volume"]
        .mean()
        .rename(columns={"monthly_volume": "cluster_month_avg"})
    )
    cluster_age_avg = (
        out.groupby(["cluster_id", "age_in_months"], as_index=False)["monthly_volume"]
        .mean()
        .rename(columns={"monthly_volume": "cluster_age_avg"})
    )
    out = out.merge(cluster_month_avg, on=["cluster_id", "month"], how="left")
    out = out.merge(cluster_age_avg, on=["cluster_id", "age_in_months"], how="left")
    out["cluster_month_avg"] = out["cluster_month_avg"].replace(0, np.nan)
    out["age_saturation"] = np.tanh(out["age_in_months"] / 12.0)
    out["growth_velocity"] = out["cluster_age_avg"] / (out["age_in_months"] + 1)
    out["distance_from_peak"] = out["cluster_age_avg"] - out["cluster_month_avg"]
    return out


def derive_site_type_labels(train_df: pd.DataFrame) -> pd.DataFrame:
    site_stats = (
        train_df.groupby("site_id", as_index=False)["monthly_volume"]
        .mean()
        .rename(columns={"monthly_volume": "site_avg_volume_train"})
    )
    q = site_stats["site_avg_volume_train"].quantile([0.33, 0.66]).to_list()
    q1, q2 = q[0], q[1]

    def bucket(v: float) -> str:
        if v <= q1:
            return "low"
        if v <= q2:
            return "mid"
        return "high"

    site_stats["site_type"] = site_stats["site_avg_volume_train"].apply(bucket)
    return site_stats[["site_id", "site_type"]]


def fit_site_type_inference_model(train_df: pd.DataFrame) -> tuple[LGBMClassifier, dict[str, str]]:
    label_df = derive_site_type_labels(train_df)
    labeled_rows = train_df.merge(label_df, on="site_id", how="left")

    X = labeled_rows[["cluster_id", "latitude", "longitude"]].copy()
    y = labeled_rows["site_type"].astype(str)

    X["cluster_id"] = X["cluster_id"].astype("category")
    fill_values = X.median(numeric_only=True)
    X = X.fillna(fill_values)

    clf = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
    )
    clf.fit(X, y, categorical_feature=["cluster_id"])

    # Fallback for unseen geography/cluster combos.
    cluster_mode = (
        labeled_rows.groupby("cluster_id")["site_type"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "mid")
        .to_dict()
    )
    return clf, {str(k): str(v) for k, v in cluster_mode.items()}


def infer_site_type(
    df: pd.DataFrame,
    clf: LGBMClassifier,
    cluster_mode: dict[str, str],
) -> pd.Series:
    X = df[["cluster_id", "latitude", "longitude"]].copy()
    X["cluster_id"] = X["cluster_id"].astype("category")
    fill_values = X.median(numeric_only=True)
    X = X.fillna(fill_values)

    pred = pd.Series(clf.predict(X), index=df.index).astype(str)
    missing = pred.isna()
    if missing.any():
        fallback = df.loc[missing, "cluster_id"].astype(str).map(cluster_mode).fillna("mid")
        pred.loc[missing] = fallback
    return pred


def train_multiplier_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
) -> dict[str, float]:
    eps = 1e-6
    train = train_df.copy()
    test = test_df.copy()
    train["target_multiplier"] = train["monthly_volume"] / (train["cluster_month_avg"] + eps)
    test["target_multiplier"] = test["monthly_volume"] / (test["cluster_month_avg"] + eps)

    X_train = train[features].copy()
    X_test = test[features].copy()
    y_train = train["target_multiplier"].astype(float)

    cat_cols = ["cluster_id", "maturity_bucket", "site_type"]
    cat_cols = [c for c in cat_cols if c in X_train.columns]
    for c in cat_cols:
        X_train[c] = X_train[c].astype("category")
        X_test[c] = X_test[c].astype("category")

    fill_values = X_train.median(numeric_only=True)
    X_train = X_train.fillna(fill_values)
    X_test = X_test.fillna(fill_values)

    model = LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=30,
        random_state=42,
    )
    model.fit(X_train, y_train, categorical_feature=cat_cols)
    pred_multiplier = model.predict(X_test)
    pred_volume = pred_multiplier * (test["cluster_month_avg"].to_numpy(dtype=float) + eps)

    y_true = test["monthly_volume"].to_numpy(dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, pred_volume)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred_volume))),
    }


def simulate_cold_start(
    full_df: pd.DataFrame,
    features: list[str],
    holdout_frac: float = 0.2,
) -> dict[str, float]:
    # Realistic simulation: hold out entire sites, evaluate on their first 6 months.
    sites = pd.Series(full_df["site_id"].astype(str).unique()).sort_values().reset_index(drop=True)
    holdout_n = max(1, int(len(sites) * holdout_frac))
    holdout_sites = set(sites.tail(holdout_n).tolist())

    cold_test = full_df[
        full_df["site_id"].astype(str).isin(holdout_sites) & (full_df["age_in_months"] <= 6)
    ].copy()
    cold_train = full_df[~full_df["site_id"].astype(str).isin(holdout_sites)].copy()

    if len(cold_test) == 0 or len(cold_train) == 0:
        return {"mae_early_months": float("nan"), "rmse_early_months": float("nan"), "rows": 0}

    metrics = train_multiplier_model(cold_train, cold_test, features)
    metrics["rows"] = int(len(cold_test))
    return {
        "mae_early_months": metrics["mae"],
        "rmse_early_months": metrics["rmse"],
        "rows": metrics["rows"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deployable no-leakage multiplier model with inferred site_type.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("zeta_modelling/data/phase1_final_monthly_2024_2025.csv"),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("zeta_modelling/data/phase2_deployable_multiplier_metrics.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input, low_memory=False)
    feat = add_cluster_features(add_base_features(df))

    train_df = feat[feat["date"] < pd.Timestamp("2025-01-01")].copy()
    test_df = feat[feat["date"] >= pd.Timestamp("2025-01-01")].copy()

    clf, cluster_mode = fit_site_type_inference_model(train_df)
    train_df["site_type"] = infer_site_type(train_df, clf, cluster_mode)
    test_df["site_type"] = infer_site_type(test_df, clf, cluster_mode)

    final_features = [
        "age_in_months",
        "real_age_months",
        "cluster_id",
        "cluster_month_avg",
        "cluster_age_avg",
        "month",
        "month_sin",
        "month_cos",
        "latitude",
        "longitude",
        "maturity_bucket",
        "site_type",
        "age_sq",
        "log_age",
        "is_early",
        "is_growth",
        "is_mature",
        "age_saturation",
        "growth_velocity",
        "distance_from_peak",
        "lat_lon_interaction",
    ]
    final_features = [c for c in final_features if c in train_df.columns]

    main_metrics = train_multiplier_model(train_df, test_df, final_features)
    cold_df = pd.concat([train_df, test_df], ignore_index=True)
    cold_metrics = simulate_cold_start(cold_df, final_features)

    payload = {
        "input": str(args.input),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "model_type": "deployable_no_lag_multiplier",
        "site_type_inference": "LightGBM classifier using cluster_id + latitude + longitude",
        "metrics_full_test": main_metrics,
        "metrics_cold_start_early_months": cold_metrics,
        "features": final_features,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
