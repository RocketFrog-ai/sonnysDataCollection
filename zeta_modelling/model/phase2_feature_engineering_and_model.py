from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["month"] = out["date"].dt.month
    out["year"] = out["date"].dt.year
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    return out


def add_lifecycle_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["age_sq"] = out["age_in_months"] ** 2
    out["log_age"] = np.log1p(out["age_in_months"])
    out["is_early"] = (out["age_in_months"] <= 6).astype(int)
    out["is_growth"] = ((out["age_in_months"] > 6) & (out["age_in_months"] <= 18)).astype(int)
    out["is_mature"] = (out["age_in_months"] > 18).astype(int)
    return out


def add_cluster_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    cluster_month_avg = (
        out.groupby(["cluster_id", "month"], as_index=False)["monthly_volume"]
        .mean()
        .rename(columns={"monthly_volume": "cluster_month_avg"})
    )
    out = out.merge(cluster_month_avg, on=["cluster_id", "month"], how="left")

    cluster_age_avg = (
        out.groupby(["cluster_id", "age_in_months"], as_index=False)["monthly_volume"]
        .mean()
        .rename(columns={"monthly_volume": "cluster_age_avg"})
    )
    out = out.merge(cluster_age_avg, on=["cluster_id", "age_in_months"], how="left")
    return out


def add_global_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    month_avg = out.groupby("month")["monthly_volume"].mean()
    global_avg = out["monthly_volume"].mean()
    seasonality_factor = (month_avg / global_avg).to_dict()
    out["seasonality_factor"] = out["month"].map(seasonality_factor)
    return out


def add_location_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["lat_lon_interaction"] = out["latitude"] * out["longitude"]
    return out


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["site_id", "date"]).copy()
    grouped = out.groupby("site_id")["monthly_volume"]
    out["lag_1"] = grouped.shift(1)
    out["lag_3"] = grouped.shift(3)
    out["rolling_mean_3"] = grouped.transform(
        lambda s: s.shift(1).rolling(window=3, min_periods=1).mean()
    )
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = add_time_features(out)
    out = add_lifecycle_features(out)
    out = add_cluster_features(out)
    out = add_global_seasonality(out)
    out = add_location_features(out)
    out = add_lag_features(out)
    return out


def train_and_evaluate(
    df: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    model_name: str,
) -> dict[str, float | int | str]:
    train_df = df[df["date"] < pd.Timestamp("2025-01-01")].copy()
    test_df = df[df["date"] >= pd.Timestamp("2025-01-01")].copy()

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError("Empty train or test split. Check date filters.")

    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    y_train = train_df["monthly_volume"].astype(float)
    y_test = test_df["monthly_volume"].astype(float)

    for col in categorical_cols:
        X_train[col] = X_train[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    # Median imputation keeps all rows for both lag and non-lag models.
    fill_values = X_train.median(numeric_only=True)
    X_train = X_train.fillna(fill_values)
    X_test = X_test.fillna(fill_values)

    model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
    )
    model.fit(X_train, y_train, categorical_feature=categorical_cols)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return {
        "model_name": model_name,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "mae": float(mae),
        "rmse": float(rmse),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 feature engineering + LightGBM baseline.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("zeta_modelling/data/phase1_final_monthly_2024_2025.csv"),
    )
    parser.add_argument(
        "--output-features",
        type=Path,
        default=Path("zeta_modelling/data/phase2_features_2024_2025.csv"),
    )
    parser.add_argument(
        "--output-metrics",
        type=Path,
        default=Path("zeta_modelling/data/phase2_metrics_2024_2025.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values(["site_id", "date"]).reset_index(drop=True)

    feat_df = build_features(df)

    base_features = [
        "age_in_months",
        "real_age_months",
        "age_sq",
        "log_age",
        "is_early",
        "is_growth",
        "is_mature",
        "cluster_id",
        "cluster_month_avg",
        "cluster_age_avg",
        "month",
        "month_sin",
        "month_cos",
        "seasonality_factor",
        "latitude",
        "longitude",
        "lat_lon_interaction",
        "maturity_bucket",
    ]
    lag_features = ["lag_1", "lag_3", "rolling_mean_3"]

    categorical_cols = ["cluster_id", "maturity_bucket"]
    metrics_no_lag = train_and_evaluate(
        feat_df,
        feature_cols=base_features,
        categorical_cols=categorical_cols,
        model_name="lightgbm_no_lag",
    )
    metrics_with_lag = train_and_evaluate(
        feat_df,
        feature_cols=base_features + lag_features,
        categorical_cols=categorical_cols,
        model_name="lightgbm_with_lag",
    )

    args.output_features.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_csv(args.output_features, index=False)

    metrics = {
        "input_path": str(args.input),
        "output_features_path": str(args.output_features),
        "time_split": {"train_before": "2025-01-01", "test_from": "2025-01-01"},
        "metrics": [metrics_no_lag, metrics_with_lag],
    }
    args.output_metrics.write_text(json.dumps(metrics, indent=2))

    print("Saved features to:", args.output_features)
    print("Saved metrics to:", args.output_metrics)
    for row in metrics["metrics"]:
        print(
            f"{row['model_name']}: train={row['train_rows']}, test={row['test_rows']}, "
            f"MAE={row['mae']:.3f}, RMSE={row['rmse']:.3f}"
        )


if __name__ == "__main__":
    main()
