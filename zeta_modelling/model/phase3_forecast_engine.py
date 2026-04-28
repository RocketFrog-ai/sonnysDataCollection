from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.cluster import KMeans


@dataclass
class TrainedArtifacts:
    site_type_model: LGBMClassifier
    early_model: LGBMRegressor
    main_model: LGBMRegressor
    cluster_mode_site_type: dict[str, str]
    cluster_centroids: pd.DataFrame
    cluster_month_avg: pd.DataFrame
    cluster_age_avg: pd.DataFrame
    cluster_month_std: pd.DataFrame
    feature_cols: list[str]
    cat_cols: list[str]


def prepare_base(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values(["site_id", "date"]).reset_index(drop=True)
    out["month"] = out["date"].dt.month
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    out["age_sq"] = out["age_in_months"] ** 2
    out["log_age"] = np.log1p(out["age_in_months"])
    out["is_early"] = (out["age_in_months"] <= 6).astype(int)
    out["is_growth"] = ((out["age_in_months"] > 6) & (out["age_in_months"] <= 18)).astype(int)
    out["is_mature"] = (out["age_in_months"] > 18).astype(int)
    out["lat_lon_interaction"] = out["latitude"] * out["longitude"]
    out["age_saturation"] = np.tanh(out["age_in_months"] / 12.0)
    return out


def add_cluster_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    work["cluster_id"] = work["cluster_id"].astype(str)
    cluster_month_avg = (
        work.groupby(["cluster_id", "month"], as_index=False)["monthly_volume"]
        .mean()
        .rename(columns={"monthly_volume": "cluster_month_avg"})
    )
    cluster_month_std = (
        work.groupby(["cluster_id", "month"], as_index=False)["monthly_volume"]
        .std()
        .fillna(0.0)
        .rename(columns={"monthly_volume": "cluster_std"})
    )
    cluster_age_avg = (
        work.groupby(["cluster_id", "age_in_months"], as_index=False)["monthly_volume"]
        .mean()
        .rename(columns={"monthly_volume": "cluster_age_avg"})
    )
    return cluster_month_avg, cluster_month_std, cluster_age_avg


def add_cluster_features(
    df: pd.DataFrame,
    cluster_month_avg: pd.DataFrame,
    cluster_month_std: pd.DataFrame,
    cluster_age_avg: pd.DataFrame,
) -> pd.DataFrame:
    out = df.copy()
    out["cluster_id"] = out["cluster_id"].astype(str)
    out = out.merge(cluster_month_avg, on=["cluster_id", "month"], how="left")
    out = out.merge(cluster_month_std, on=["cluster_id", "month"], how="left")
    out = out.merge(cluster_age_avg, on=["cluster_id", "age_in_months"], how="left")
    out["cluster_month_avg"] = out["cluster_month_avg"].replace(0, np.nan)
    out["cluster_growth_curve"] = out["cluster_age_avg"] - out.groupby("cluster_id")["cluster_age_avg"].shift(1)
    out["cluster_growth_curve"] = out["cluster_growth_curve"].fillna(0.0)
    out["growth_velocity"] = out["cluster_age_avg"] / (out["age_in_months"] + 1)
    out["distance_from_peak"] = out["cluster_age_avg"] - out["cluster_month_avg"]
    return out


def fit_site_type_model(train_df: pd.DataFrame, n_types: int = 6) -> tuple[LGBMClassifier, dict[str, str]]:
    # Better site_type labeling: 6-cluster KMeans on train-only site behavior.
    train = train_df.copy()
    train["cluster_id"] = train["cluster_id"].astype(str)
    stats = (
        train.groupby("site_id", as_index=False)
        .agg(
            site_avg_volume=("monthly_volume", "mean"),
            site_peak=("monthly_volume", "max"),
            site_volatility=("monthly_volume", "std"),
        )
        .fillna(0.0)
    )
    Xk = stats[["site_avg_volume", "site_peak", "site_volatility"]].copy()
    km = KMeans(n_clusters=n_types, random_state=42, n_init=10)
    stats["site_type"] = km.fit_predict(Xk).astype(str)

    labeled = train.merge(stats[["site_id", "site_type"]], on="site_id", how="left")
    cluster_density = labeled.groupby("cluster_id")["site_id"].nunique().rename("cluster_density")
    cluster_avg_volume = labeled.groupby("cluster_id")["monthly_volume"].mean().rename("cluster_avg_volume")
    labeled = labeled.join(cluster_density, on="cluster_id")
    labeled = labeled.join(cluster_avg_volume, on="cluster_id")

    X = labeled[["cluster_id", "latitude", "longitude", "cluster_density", "cluster_avg_volume"]].copy()
    y = labeled["site_type"].astype(str)
    X["cluster_id"] = X["cluster_id"].astype("category")
    fill_values = X.median(numeric_only=True)
    X = X.fillna(fill_values)

    clf = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
    )
    clf.fit(X, y, categorical_feature=["cluster_id"])

    cluster_mode = (
        labeled.groupby("cluster_id")["site_type"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "0")
        .to_dict()
    )
    return clf, {str(k): str(v) for k, v in cluster_mode.items()}


def infer_site_type(df: pd.DataFrame, clf: LGBMClassifier, cluster_mode: dict[str, str]) -> pd.Series:
    temp = df.copy()
    temp["cluster_id"] = temp["cluster_id"].astype(str)
    temp["cluster_density"] = temp.groupby("cluster_id")["cluster_id"].transform("count")
    temp["cluster_avg_volume"] = temp.groupby("cluster_id")["cluster_month_avg"].transform("mean")
    X = temp[["cluster_id", "latitude", "longitude", "cluster_density", "cluster_avg_volume"]].copy()
    X["cluster_id"] = X["cluster_id"].astype("category")
    fill_values = X.median(numeric_only=True)
    X = X.fillna(fill_values)
    pred = pd.Series(clf.predict(X), index=df.index).astype(str)
    pred = pred.fillna(df["cluster_id"].astype(str).map(cluster_mode).fillna("0"))
    return pred


def train_regressor(train_df: pd.DataFrame, feature_cols: list[str], cat_cols: list[str]) -> LGBMRegressor:
    train = train_df.copy()
    eps = 1e-6
    train["target_multiplier"] = train["monthly_volume"] / (train["cluster_month_avg"] + eps)

    X = train[feature_cols].copy()
    y = train["target_multiplier"].astype(float)
    use_cat = [c for c in cat_cols if c in X.columns]
    for c in use_cat:
        X[c] = X[c].astype("category")
    fill_values = X.median(numeric_only=True)
    X = X.fillna(fill_values)

    model = LGBMRegressor(
        n_estimators=1400,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=30,
        random_state=42,
    )
    model.fit(X, y, categorical_feature=use_cat)
    return model


def nearest_cluster(lat: float, lon: float, centroids: pd.DataFrame) -> str:
    c = centroids.copy()
    d2 = (c["latitude"] - lat) ** 2 + (c["longitude"] - lon) ** 2
    idx = d2.idxmin()
    return str(c.loc[idx, "cluster_id"])


def make_forecast_rows(lat: float, lon: float, start_date: str, months: int) -> pd.DataFrame:
    start = pd.Timestamp(start_date)
    dates = pd.date_range(start=start, periods=months, freq="MS")
    out = pd.DataFrame(
        {
            "date": dates,
            "month": dates.month,
            "age_in_months": np.arange(1, months + 1, dtype=int),
            "latitude": lat,
            "longitude": lon,
            "real_age_months": np.arange(1, months + 1, dtype=int),
        }
    )
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    out["age_sq"] = out["age_in_months"] ** 2
    out["log_age"] = np.log1p(out["age_in_months"])
    out["is_early"] = (out["age_in_months"] <= 6).astype(int)
    out["is_growth"] = ((out["age_in_months"] > 6) & (out["age_in_months"] <= 18)).astype(int)
    out["is_mature"] = (out["age_in_months"] > 18).astype(int)
    out["lat_lon_interaction"] = out["latitude"] * out["longitude"]
    out["age_saturation"] = np.tanh(out["age_in_months"] / 12.0)
    out["maturity_bucket"] = np.select(
        [out["age_in_months"] < 36, out["age_in_months"] < 84], ["young", "mid"], default="mature"
    )
    return out


def predict_new_site(
    lat: float,
    lon: float,
    artifacts: TrainedArtifacts,
    months: int = 60,
    start_date: str = "2026-01-01",
    break_even_threshold_volume: float = 250000.0,
) -> pd.DataFrame:
    forecast = make_forecast_rows(lat, lon, start_date, months)
    assigned_cluster = nearest_cluster(lat, lon, artifacts.cluster_centroids)
    forecast["cluster_id"] = assigned_cluster

    forecast = forecast.merge(
        artifacts.cluster_month_avg, on=["cluster_id", "month"], how="left"
    ).merge(
        artifacts.cluster_month_std, on=["cluster_id", "month"], how="left"
    ).merge(
        artifacts.cluster_age_avg, on=["cluster_id", "age_in_months"], how="left"
    )
    forecast["cluster_std"] = forecast["cluster_std"].fillna(
        artifacts.cluster_month_std["cluster_std"].median()
    )
    forecast["cluster_growth_curve"] = (
        forecast["cluster_age_avg"] - forecast["cluster_age_avg"].shift(1)
    ).fillna(0.0)
    forecast["growth_velocity"] = forecast["cluster_age_avg"] / (forecast["age_in_months"] + 1)
    forecast["distance_from_peak"] = forecast["cluster_age_avg"] - forecast["cluster_month_avg"]

    forecast["site_type"] = infer_site_type(
        forecast, artifacts.site_type_model, artifacts.cluster_mode_site_type
    )
    for c in artifacts.cat_cols:
        if c in forecast.columns:
            forecast[c] = forecast[c].astype("category")

    # Early vs main model split.
    eps = 1e-6
    X = forecast[artifacts.feature_cols].copy()
    fill_values = X.median(numeric_only=True)
    X = X.fillna(fill_values)

    early_mask = forecast["age_in_months"] <= 6
    pred_multiplier = np.zeros(len(forecast), dtype=float)
    if early_mask.any():
        pred_multiplier[early_mask.to_numpy()] = artifacts.early_model.predict(X.loc[early_mask])
    if (~early_mask).any():
        pred_multiplier[(~early_mask).to_numpy()] = artifacts.main_model.predict(X.loc[~early_mask])

    forecast["pred_multiplier"] = pred_multiplier
    forecast["pred_volume"] = forecast["pred_multiplier"] * (forecast["cluster_month_avg"] + eps)

    # Uncertainty band (fast win): +-1 * cluster_std
    forecast["low"] = (forecast["pred_volume"] - forecast["cluster_std"]).clip(lower=0)
    forecast["high"] = (forecast["pred_volume"] + forecast["cluster_std"]).clip(lower=0)

    forecast["cumulative_volume"] = forecast["pred_volume"].cumsum()
    break_even = forecast[forecast["cumulative_volume"] >= break_even_threshold_volume]
    break_even_month = int(break_even["age_in_months"].iloc[0]) if len(break_even) else None
    forecast["break_even_month_estimate"] = break_even_month
    return forecast


def train_artifacts(df: pd.DataFrame) -> TrainedArtifacts:
    base = prepare_base(df)
    base["cluster_id"] = base["cluster_id"].astype(str)
    cluster_month_avg, cluster_month_std, cluster_age_avg = add_cluster_tables(base)
    feat = add_cluster_features(base, cluster_month_avg, cluster_month_std, cluster_age_avg)

    train = feat[feat["date"] < pd.Timestamp("2025-01-01")].copy()
    clf, cluster_mode = fit_site_type_model(train, n_types=6)
    feat["site_type"] = infer_site_type(feat, clf, cluster_mode)

    feature_cols = [
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
        "cluster_growth_curve",
    ]
    feature_cols = [c for c in feature_cols if c in feat.columns]
    cat_cols = [c for c in ["cluster_id", "maturity_bucket", "site_type"] if c in feature_cols]

    early_train = feat[(feat["date"] < "2025-01-01") & (feat["age_in_months"] <= 6)].copy()
    main_train = feat[(feat["date"] < "2025-01-01") & (feat["age_in_months"] > 6)].copy()

    early_model = train_regressor(early_train, feature_cols, cat_cols)
    main_model = train_regressor(main_train, feature_cols, cat_cols)

    centroids = (
        base.groupby("cluster_id", as_index=False)[["latitude", "longitude"]]
        .mean()
        .dropna(subset=["latitude", "longitude"])
    )

    return TrainedArtifacts(
        site_type_model=clf,
        early_model=early_model,
        main_model=main_model,
        cluster_mode_site_type=cluster_mode,
        cluster_centroids=centroids,
        cluster_month_avg=cluster_month_avg,
        cluster_age_avg=cluster_age_avg,
        cluster_month_std=cluster_month_std,
        feature_cols=feature_cols,
        cat_cols=cat_cols,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 forecast engine with uncertainty bands.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("zeta_modelling/data/phase1_final_monthly_2024_2025.csv"),
    )
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--months", type=int, default=60)
    parser.add_argument("--start-date", type=str, default="2026-01-01")
    parser.add_argument("--break-even-threshold-volume", type=float, default=250000.0)
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("zeta_modelling/data/phase3_new_site_forecast_5y.csv"),
    )
    parser.add_argument(
        "--out-summary-json",
        type=Path,
        default=Path("zeta_modelling/data/phase3_forecast_summary.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input, low_memory=False)
    artifacts = train_artifacts(df)
    forecast = predict_new_site(
        lat=args.lat,
        lon=args.lon,
        artifacts=artifacts,
        months=args.months,
        start_date=args.start_date,
        break_even_threshold_volume=args.break_even_threshold_volume,
    )
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    forecast_out = forecast[
        [
            "date",
            "age_in_months",
            "pred_volume",
            "low",
            "high",
            "cumulative_volume",
            "break_even_month_estimate",
            "cluster_id",
            "site_type",
        ]
    ].copy()
    forecast_out.rename(columns={"pred_volume": "volume"}, inplace=True)
    forecast_out.to_csv(args.out_csv, index=False)

    summary = {
        "input": str(args.input),
        "forecast_rows": int(len(forecast_out)),
        "lat": args.lat,
        "lon": args.lon,
        "assigned_cluster": str(forecast_out["cluster_id"].iloc[0]),
        "inferred_site_type": str(forecast_out["site_type"].iloc[0]),
        "total_5y_volume": float(forecast_out["volume"].sum()),
        "break_even_month_estimate": (
            int(forecast_out["break_even_month_estimate"].iloc[0])
            if pd.notna(forecast_out["break_even_month_estimate"].iloc[0])
            else None
        ),
        "output_csv": str(args.out_csv),
    }
    args.out_summary_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
