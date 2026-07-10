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


def add_cluster_core_features(df: pd.DataFrame) -> pd.DataFrame:
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


def add_cluster_pseudo_lags_and_trend(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cluster_calendar = (
        out.groupby(["cluster_id", "date"], as_index=False)["monthly_volume"]
        .mean()
        .sort_values(["cluster_id", "date"])
        .rename(columns={"monthly_volume": "cluster_date_avg"})
    )
    grouped = cluster_calendar.groupby("cluster_id")["cluster_date_avg"]
    cluster_calendar["cluster_lag_1"] = grouped.shift(1)
    cluster_calendar["cluster_lag_3"] = grouped.shift(1).rolling(window=3, min_periods=1).mean()
    cluster_calendar["cluster_growth_rate"] = grouped.pct_change().replace([np.inf, -np.inf], np.nan)
    cluster_calendar["cluster_rolling_mean"] = grouped.shift(1).rolling(window=3, min_periods=1).mean()

    # Cluster variance view by seasonality month (captures uncertainty by cluster + month).
    cluster_month_std = (
        out.groupby(["cluster_id", "month"], as_index=False)["monthly_volume"]
        .std()
        .rename(columns={"monthly_volume": "cluster_std"})
    )

    out = out.merge(
        cluster_calendar[
            [
                "cluster_id",
                "date",
                "cluster_lag_1",
                "cluster_lag_3",
                "cluster_growth_rate",
                "cluster_rolling_mean",
            ]
        ],
        on=["cluster_id", "date"],
        how="left",
    )
    out = out.merge(cluster_month_std, on=["cluster_id", "month"], how="left")
    return out


def add_global_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    month_avg = out.groupby("month")["monthly_volume"].mean()
    global_avg = out["monthly_volume"].mean()
    out["seasonality_factor"] = out["month"].map((month_avg / global_avg).to_dict())
    return out


def add_location_and_interactions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["lat_lon_interaction"] = out["latitude"] * out["longitude"]
    out["cluster_id_num"] = pd.to_numeric(out["cluster_id"], errors="coerce").fillna(-1)
    out["age_cluster_interaction"] = out["age_in_months"] * out["cluster_id_num"]
    out["age_seasonality_interaction"] = out["age_in_months"] * out["seasonality_factor"]
    return out


def add_true_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["site_id", "date"]).copy()
    grouped = out.groupby("site_id")["monthly_volume"]
    out["lag_1"] = grouped.shift(1)
    out["lag_3"] = grouped.shift(3)
    out["rolling_mean_3"] = grouped.transform(lambda s: s.shift(1).rolling(window=3, min_periods=1).mean())
    return out


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = add_time_features(out)
    out = add_lifecycle_features(out)
    out = add_cluster_core_features(out)
    out = add_cluster_pseudo_lags_and_trend(out)
    out = add_global_seasonality(out)
    out = add_location_and_interactions(out)
    out = add_true_lag_features(out)
    return out


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df[df["date"] < pd.Timestamp("2025-01-01")].copy()
    test_df = df[df["date"] >= pd.Timestamp("2025-01-01")].copy()
    return train_df, test_df


def tune_lightgbm(train_df: pd.DataFrame, features: list[str], categorical_cols: list[str]) -> dict[str, int | float]:
    # Time-respecting inner validation inside train period.
    train_sub = train_df[train_df["date"] < pd.Timestamp("2024-10-01")].copy()
    val_sub = train_df[train_df["date"] >= pd.Timestamp("2024-10-01")].copy()
    if len(train_sub) == 0 or len(val_sub) == 0:
        raise ValueError("Tuning split failed; not enough train-period coverage.")

    X_tr = train_sub[features].copy()
    y_tr = train_sub["monthly_volume"].astype(float)
    X_val = val_sub[features].copy()
    y_val = val_sub["monthly_volume"].astype(float)

    for c in categorical_cols:
        X_tr[c] = X_tr[c].astype("category")
        X_val[c] = X_val[c].astype("category")

    fill_values = X_tr.median(numeric_only=True)
    X_tr = X_tr.fillna(fill_values)
    X_val = X_val.fillna(fill_values)

    grid = [
        {"num_leaves": 31, "max_depth": -1, "min_child_samples": 20, "learning_rate": 0.10},
        {"num_leaves": 63, "max_depth": -1, "min_child_samples": 30, "learning_rate": 0.05},
        {"num_leaves": 127, "max_depth": -1, "min_child_samples": 40, "learning_rate": 0.03},
        {"num_leaves": 63, "max_depth": 8, "min_child_samples": 50, "learning_rate": 0.05},
        {"num_leaves": 95, "max_depth": 10, "min_child_samples": 80, "learning_rate": 0.03},
        {"num_leaves": 127, "max_depth": 12, "min_child_samples": 100, "learning_rate": 0.02},
    ]

    best_rmse = float("inf")
    best_params: dict[str, int | float] = {}
    for params in grid:
        model = LGBMRegressor(
            n_estimators=1400,
            random_state=42,
            **params,
        )
        model.fit(X_tr, y_tr, categorical_feature=categorical_cols)
        pred_val = model.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, pred_val)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params
    best_params["best_val_rmse"] = best_rmse
    return best_params


def train_eval(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    categorical_cols: list[str],
    params: dict[str, int | float],
) -> tuple[LGBMRegressor, dict[str, float | int], pd.DataFrame]:
    X_train = train_df[features].copy()
    y_train = train_df["monthly_volume"].astype(float)
    X_test = test_df[features].copy()
    y_test = test_df["monthly_volume"].astype(float)

    for c in categorical_cols:
        X_train[c] = X_train[c].astype("category")
        X_test[c] = X_test[c].astype("category")

    fill_values = X_train.median(numeric_only=True)
    X_train = X_train.fillna(fill_values)
    X_test = X_test.fillna(fill_values)

    model = LGBMRegressor(
        n_estimators=1400,
        random_state=42,
        num_leaves=int(params.get("num_leaves", 63)),
        max_depth=int(params.get("max_depth", -1)),
        min_child_samples=int(params.get("min_child_samples", 30)),
        learning_rate=float(params.get("learning_rate", 0.05)),
    )
    model.fit(X_train, y_train, categorical_feature=categorical_cols)
    preds = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    result = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "mae": mae,
        "rmse": rmse,
    }
    pred_df = test_df[["site_id", "date", "cluster_id", "age_in_months", "monthly_volume"]].copy()
    pred_df["pred"] = preds
    pred_df["abs_error"] = (pred_df["monthly_volume"] - pred_df["pred"]).abs()
    return model, result, pred_df


def error_analysis(pred_df: pd.DataFrame, out_dir: Path) -> dict[str, str]:
    by_site = (
        pred_df.groupby("site_id", as_index=False)["abs_error"]
        .mean()
        .rename(columns={"abs_error": "mae_site"})
        .sort_values("mae_site", ascending=False)
    )
    by_cluster = (
        pred_df.groupby("cluster_id", as_index=False)["abs_error"]
        .mean()
        .rename(columns={"abs_error": "mae_cluster"})
        .sort_values("mae_cluster", ascending=False)
    )
    pred_df["age_phase"] = np.select(
        [pred_df["age_in_months"] <= 6, pred_df["age_in_months"] <= 18],
        ["early", "growth"],
        default="mature",
    )
    by_age_phase = (
        pred_df.groupby("age_phase", as_index=False)["abs_error"]
        .mean()
        .rename(columns={"abs_error": "mae_age_phase"})
        .sort_values("mae_age_phase", ascending=False)
    )

    p1 = out_dir / "phase2_error_by_site.csv"
    p2 = out_dir / "phase2_error_by_cluster.csv"
    p3 = out_dir / "phase2_error_by_age_phase.csv"
    by_site.to_csv(p1, index=False)
    by_cluster.to_csv(p2, index=False)
    by_age_phase.to_csv(p3, index=False)
    return {"error_by_site": str(p1), "error_by_cluster": str(p2), "error_by_age_phase": str(p3)}


def feature_importance_df(model: LGBMRegressor, features: list[str], out_dir: Path) -> str:
    fi = pd.DataFrame(
        {"feature": features, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    path = out_dir / "phase2_feature_importance.csv"
    fi.to_csv(path, index=False)
    return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 no-lag upgrade with pseudo-lags and tuning.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("zeta_modelling/data/phase1_final_monthly_2024_2025.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("zeta_modelling/data"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values(["site_id", "date"]).reset_index(drop=True)

    feat_df = build_feature_table(df)
    feat_path = args.out_dir / "phase2_features_no_lag_upgrade_2024_2025.csv"
    feat_df.to_csv(feat_path, index=False)

    train_df, test_df = split_train_test(feat_df)
    categorical_cols = ["cluster_id", "maturity_bucket"]

    baseline_features = [
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
    upgraded_no_lag_features = baseline_features + [
        "cluster_lag_1",
        "cluster_lag_3",
        "cluster_growth_rate",
        "cluster_std",
        "cluster_rolling_mean",
        "age_cluster_interaction",
        "age_seasonality_interaction",
    ]
    warm_features = upgraded_no_lag_features + ["lag_1", "lag_3", "rolling_mean_3"]

    baseline_params = {"num_leaves": 63, "max_depth": 6, "min_child_samples": 30, "learning_rate": 0.05}
    baseline_model, baseline_metrics, _ = train_eval(
        train_df, test_df, baseline_features, categorical_cols, baseline_params
    )

    best_params = tune_lightgbm(train_df, upgraded_no_lag_features, categorical_cols)
    tuned_no_lag_model, tuned_no_lag_metrics, tuned_pred = train_eval(
        train_df, test_df, upgraded_no_lag_features, categorical_cols, best_params
    )
    _, warm_metrics, _ = train_eval(
        train_df, test_df, warm_features, categorical_cols, best_params
    )

    error_paths = error_analysis(tuned_pred, args.out_dir)
    fi_path = feature_importance_df(tuned_no_lag_model, upgraded_no_lag_features, args.out_dir)

    metrics_payload = {
        "input_path": str(args.input),
        "features_path": str(feat_path),
        "time_split": {"train_before": "2025-01-01", "test_from": "2025-01-01"},
        "best_tuned_params_no_lag": best_params,
        "baseline_no_lag": baseline_metrics,
        "upgraded_no_lag": tuned_no_lag_metrics,
        "warm_model_with_true_lags": warm_metrics,
        "gap_reduction_mae": baseline_metrics["mae"] - tuned_no_lag_metrics["mae"],
        "error_analysis_files": error_paths,
        "feature_importance_file": fi_path,
    }
    metrics_path = args.out_dir / "phase2_no_lag_upgrade_metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2))

    print("Saved feature table:", feat_path)
    print("Saved metrics:", metrics_path)
    print(
        "Baseline no-lag -> "
        f"MAE={baseline_metrics['mae']:.3f}, RMSE={baseline_metrics['rmse']:.3f}"
    )
    print(
        "Upgraded no-lag -> "
        f"MAE={tuned_no_lag_metrics['mae']:.3f}, RMSE={tuned_no_lag_metrics['rmse']:.3f}"
    )
    print(
        "Warm model (true lags) -> "
        f"MAE={warm_metrics['mae']:.3f}, RMSE={warm_metrics['rmse']:.3f}"
    )
    print(f"MAE gap reduction (no-lag baseline -> upgraded): {metrics_payload['gap_reduction_mae']:.3f}")


if __name__ == "__main__":
    main()
