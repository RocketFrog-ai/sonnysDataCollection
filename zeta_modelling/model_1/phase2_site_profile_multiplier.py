from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error


def base_features(df: pd.DataFrame) -> pd.DataFrame:
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
    return out


def add_lifecycle_curve_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["age_saturation"] = np.tanh(out["age_in_months"] / 12.0)
    out["growth_velocity"] = out["cluster_age_avg"] / (out["age_in_months"] + 1)
    out["distance_from_peak"] = out["cluster_age_avg"] - out["cluster_month_avg"]
    return out


def _slope_first_6(group: pd.DataFrame) -> float:
    g = group.sort_values("date").head(6)
    if len(g) < 2:
        return 0.0
    x = np.arange(len(g), dtype=float)
    y = g["monthly_volume"].to_numpy(dtype=float)
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)


def add_site_behavior_profiles(df: pd.DataFrame, train_cutoff: str = "2025-01-01") -> pd.DataFrame:
    out = df.copy()
    train_only = out[out["date"] < pd.Timestamp(train_cutoff)].copy()

    site_stats = (
        train_only.groupby("site_id", as_index=False)
        .agg(
            site_avg_volume=("monthly_volume", "mean"),
            site_peak=("monthly_volume", "max"),
            site_volatility=("monthly_volume", "std"),
        )
    )
    slopes = (
        train_only.groupby("site_id", as_index=False)
        .apply(_slope_first_6, include_groups=False)
        .rename(columns={None: "site_growth", 0: "site_growth"})
    )
    site_stats = site_stats.merge(slopes, on="site_id", how="left")
    site_stats["site_volatility"] = site_stats["site_volatility"].fillna(0.0)
    site_stats["site_growth"] = site_stats["site_growth"].fillna(0.0)

    # Cluster site behavior into types using only training-period behavior.
    kmeans_features = ["site_avg_volume", "site_peak", "site_growth", "site_volatility"]
    X = site_stats[kmeans_features].copy()
    X = X.fillna(X.median(numeric_only=True))
    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    site_stats["site_type"] = kmeans.fit_predict(X).astype(str)

    out = out.merge(site_stats, on="site_id", how="left")

    # Cold-start fallback for unseen sites in test.
    med = site_stats[kmeans_features].median(numeric_only=True)
    for col in kmeans_features:
        out[col] = out[col].fillna(float(med[col]))
    out["site_type"] = out["site_type"].fillna("unknown")
    return out


def add_lags_for_warm_model(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["site_id", "date"]).copy()
    grp = out.groupby("site_id")["monthly_volume"]
    out["lag_1"] = grp.shift(1)
    out["lag_3"] = grp.shift(3)
    out["rolling_mean_3"] = grp.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    return out


def train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return df[df["date"] < "2025-01-01"].copy(), df[df["date"] >= "2025-01-01"].copy()


def train_predict_multiplier(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    cat_cols: list[str],
) -> tuple[dict[str, float], np.ndarray]:
    train = train_df.copy()
    test = test_df.copy()

    eps = 1e-6
    train["target_multiplier"] = train["monthly_volume"] / (train["cluster_month_avg"] + eps)
    test["target_multiplier"] = test["monthly_volume"] / (test["cluster_month_avg"] + eps)

    X_train = train[features].copy()
    X_test = test[features].copy()
    y_train = train["target_multiplier"].astype(float)
    y_test = test["target_multiplier"].astype(float)

    used_cat_cols = [c for c in cat_cols if c in X_train.columns]
    for c in used_cat_cols:
        X_train[c] = X_train[c].astype("category")
        X_test[c] = X_test[c].astype("category")

    fill_values = X_train.median(numeric_only=True)
    X_train = X_train.fillna(fill_values)
    X_test = X_test.fillna(fill_values)

    model = LGBMRegressor(
        n_estimators=1400,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=30,
        random_state=42,
    )
    model.fit(X_train, y_train, categorical_feature=used_cat_cols)
    pred_mult = model.predict(X_test)

    pred_abs = pred_mult * (test["cluster_month_avg"].to_numpy(dtype=float) + eps)
    y_abs = test["monthly_volume"].to_numpy(dtype=float)
    mae = float(mean_absolute_error(y_abs, pred_abs))
    rmse = float(np.sqrt(mean_squared_error(y_abs, pred_abs)))

    # Report multiplier fit quality too.
    mae_mult = float(mean_absolute_error(y_test, pred_mult))
    rmse_mult = float(np.sqrt(mean_squared_error(y_test, pred_mult)))

    return {"mae": mae, "rmse": rmse, "mae_multiplier": mae_mult, "rmse_multiplier": rmse_mult}, pred_abs


def train_predict_absolute(
    train_df: pd.DataFrame, test_df: pd.DataFrame, features: list[str], cat_cols: list[str]
) -> dict[str, float]:
    X_train = train_df[features].copy()
    X_test = test_df[features].copy()
    y_train = train_df["monthly_volume"].astype(float)
    y_test = test_df["monthly_volume"].astype(float)

    used_cat_cols = [c for c in cat_cols if c in X_train.columns]
    for c in used_cat_cols:
        X_train[c] = X_train[c].astype("category")
        X_test[c] = X_test[c].astype("category")

    fill_values = X_train.median(numeric_only=True)
    X_train = X_train.fillna(fill_values)
    X_test = X_test.fillna(fill_values)

    model = LGBMRegressor(
        n_estimators=1400,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=30,
        random_state=42,
    )
    model.fit(X_train, y_train, categorical_feature=used_cat_cols)
    pred = model.predict(X_test)
    return {
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase2: site profiles + multiplier target.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("zeta_modelling/data/phase1_final_monthly_2024_2025.csv"),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("zeta_modelling/data/phase2_site_profile_multiplier_metrics.json"),
    )
    return parser.parse_args()


def _unique_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input, low_memory=False)

    feat = base_features(df)
    feat = add_cluster_features(feat)
    feat = add_site_behavior_profiles(feat)
    feat = add_lifecycle_curve_features(feat)
    feat = add_lags_for_warm_model(feat)

    train_df, test_df = train_test(feat)

    baseline_features = [
        "age_in_months",
        "real_age_months",
        "age_sq",
        "log_age",
        "month",
        "month_sin",
        "month_cos",
        "cluster_id",
        "cluster_month_avg",
        "cluster_age_avg",
        "seasonality_factor" if "seasonality_factor" in feat.columns else "month_cos",
        "latitude",
        "longitude",
        "maturity_bucket",
    ]
    # ensure all exist
    baseline_features = [c for c in baseline_features if c in feat.columns]
    baseline_features = _unique_preserve_order(baseline_features)

    no_lag_new_features = baseline_features + [
        "site_avg_volume",
        "site_peak",
        "site_growth",
        "site_volatility",
        "site_type",
        "age_saturation",
        "growth_velocity",
        "distance_from_peak",
        "lat_lon_interaction",
    ]
    no_lag_new_features = [c for c in no_lag_new_features if c in feat.columns]
    no_lag_new_features = _unique_preserve_order(no_lag_new_features)

    warm_features = no_lag_new_features + ["lag_1", "lag_3", "rolling_mean_3"]
    warm_features = [c for c in warm_features if c in feat.columns]
    warm_features = _unique_preserve_order(warm_features)

    cat_cols = [c for c in ["cluster_id", "maturity_bucket", "site_type"] if c in feat.columns]

    baseline_abs = train_predict_absolute(train_df, test_df, baseline_features, cat_cols)
    no_lag_mult, pred_abs = train_predict_multiplier(train_df, test_df, no_lag_new_features, cat_cols)
    warm_abs = train_predict_absolute(train_df, test_df, warm_features, cat_cols)

    # Error diagnostics on upgraded no-lag predictions.
    diag = test_df[["site_id", "cluster_id", "age_in_months", "monthly_volume"]].copy()
    diag["pred"] = pred_abs
    diag["abs_error"] = (diag["monthly_volume"] - diag["pred"]).abs()
    by_age_phase = (
        diag.assign(
            age_phase=np.select(
                [diag["age_in_months"] <= 6, diag["age_in_months"] <= 18],
                ["early", "growth"],
                default="mature",
            )
        )
        .groupby("age_phase", as_index=False)["abs_error"]
        .mean()
        .rename(columns={"abs_error": "mae"})
    )
    by_age_phase_path = args.out_json.parent / "phase2_site_profile_multiplier_error_by_age_phase.csv"
    by_age_phase.to_csv(by_age_phase_path, index=False)

    payload = {
        "input": str(args.input),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "baseline_no_lag_absolute_target": baseline_abs,
        "upgraded_no_lag_multiplier_target": no_lag_mult,
        "warm_lag_absolute_target": warm_abs,
        "mae_gap_vs_warm_before": float(baseline_abs["mae"] - warm_abs["mae"]),
        "mae_gap_vs_warm_after": float(no_lag_mult["mae"] - warm_abs["mae"]),
        "mae_gap_reduction": float((baseline_abs["mae"] - warm_abs["mae"]) - (no_lag_mult["mae"] - warm_abs["mae"])),
        "error_by_age_phase_file": str(by_age_phase_path),
        "features_used_no_lag_upgraded": no_lag_new_features,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
