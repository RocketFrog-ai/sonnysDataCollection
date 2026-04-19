"""
Cluster-Based Forecast Evaluation
---------------------------------
Runs a train/test model for each DBSCAN radius cluster column and reports:
  1) Model prediction accuracy (MAE, RMSE, MAPE, R2)
  2) Whether test actuals fall inside expected train-time cluster ranges
  3) How far off predictions are by cluster
  4) Saved trained model artifacts

Outputs:
  - results/model_metrics_summary.json
  - results/model_eval_{radius}.json
  - results/test_predictions_{radius}.json
  - results/cluster_range_eval_{radius}.json
  - models/wash_count_model_{radius}.portable.json (version-agnostic Ridge weights)
  - plots/model_actual_vs_pred_{radius}.png
  - plots/model_residual_by_cluster_{radius}.png
"""

import json
import os
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Paths
BASE = Path(__file__).resolve().parent
DATA_PATH = Path(os.getenv("CLUSTER_DATA_PATH", str(BASE.parent / "master_daily_with_site_metadata.csv")))
RESULTS_DIR = Path(os.getenv("CLUSTER_RESULTS_DIR", str(BASE / "results")))
PLOTS_DIR = Path(os.getenv("CLUSTER_PLOTS_DIR", str(BASE / "plots")))
MODELS_DIR = Path(os.getenv("CLUSTER_MODELS_DIR", str(BASE / "models")))
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

TARGET_COL = "wash_count_total"
CLUSTER_COLS = {
    "dbscan_cluster_12km": "12km",
    "dbscan_cluster_18km": "18km",
}

# Features designed to avoid direct target leakage.
# (Do NOT use wash_count_retail/membership/voucher because they sum into target.)
BASE_FEATURES = [
    # Time and lag signals
    "day_number",
    "month_number",
    "year_number",
    "day_of_week_feature",
    "last_week_same_day",
    "running_avg_7_days",
    "prev_wash_count",
    # Weather
    "weather_total_precipitation_mm",
    "weather_rainy_days",
    "weather_total_snowfall_cm",
    "weather_days_below_freezing",
    "weather_total_sunshine_hours",
    "weather_days_pleasant_temp",
    "weather_avg_daily_max_windspeed_ms",
    # Site/context
    "nearest_gas_station_distance_miles",
    "nearest_gas_station_rating",
    "nearest_gas_station_rating_count",
    "competitors_count_4miles",
    "competitor_1_google_rating",
    "competitor_1_distance_miles",
    "competitor_1_rating_count",
    "distance_nearest_costco(5 mile)",
    "distance_nearest_walmart(5 mile)",
    "distance_nearest_target (5 mile)",
    "other_grocery_count_1mile",
    "count_food_joints_0_5miles (0.5 mile)",
    "current_count",
    "previous_count",
    "age_on_30_sep_25",
    "region_enc",
    "state_enc",
    "costco_enc",
    "tunnel_count",
    "carwash_type_encoded",
    "latitude",
    "longitude",
]


def safe_float(v):
    return None if pd.isna(v) else float(v)


def pipeline_to_portable(
    pipeline: Pipeline,
    feature_cols: List[str],
    cluster_col: str,
    train_range_df: pd.DataFrame,
) -> dict:
    """Serialize fitted Ridge pipeline to JSON-safe dict (no sklearn pickle)."""
    imputer = pipeline.named_steps["imputer"]
    scaler = pipeline.named_steps["scaler"]
    ridge = pipeline.named_steps["ridge"]
    return {
        "format_version": 1,
        "model_type": "ridge_pipeline_numeric",
        "features": feature_cols,
        "cluster_col": cluster_col,
        "target_col": TARGET_COL,
        "imputer_statistics": imputer.statistics_.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "ridge_coef": ridge.coef_.tolist(),
        "ridge_intercept": float(ridge.intercept_),
        "train_range_stats": train_range_df.to_dict(orient="records"),
    }


def make_cluster_expected_ranges(train_df, cluster_col):
    """
    Expected ranges from train only, per cluster:
    - min/max (wide range)
    - p10/p90 (operating range)
    - p25/p75 (tight range / IQR)
    """
    stats = (
        train_df.groupby(cluster_col)[TARGET_COL]
        .agg(
            train_count="count",
            train_min="min",
            train_max="max",
            train_p10=lambda x: x.quantile(0.10),
            train_p25=lambda x: x.quantile(0.25),
            train_median="median",
            train_p75=lambda x: x.quantile(0.75),
            train_p90=lambda x: x.quantile(0.90),
            train_mean="mean",
            train_std="std",
        )
        .reset_index()
    )
    return stats


def build_test_range_flags(test_df, range_df, cluster_col):
    merged = test_df.merge(range_df, on=cluster_col, how="left")
    y = merged[TARGET_COL]
    merged["in_minmax"] = ((y >= merged["train_min"]) & (y <= merged["train_max"])).astype(int)
    merged["in_p10_p90"] = ((y >= merged["train_p10"]) & (y <= merged["train_p90"])).astype(int)
    merged["in_iqr"] = ((y >= merged["train_p25"]) & (y <= merged["train_p75"])).astype(int)
    return merged


def calc_global_metrics(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mape = mean_absolute_percentage_error(y_true, np.clip(y_pred, 0, None))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse),
        "mape": float(mape),
        "r2": float(r2_score(y_true, y_pred)),
    }


def eval_by_cluster(test_eval, cluster_col):
    by_cluster = (
        test_eval.groupby(cluster_col)
        .agg(
            test_rows=(TARGET_COL, "count"),
            actual_mean=(TARGET_COL, "mean"),
            actual_median=(TARGET_COL, "median"),
            pred_mean=("predicted_wash_count_total", "mean"),
            mae=("abs_error", "mean"),
            rmse=("squared_error", lambda x: np.sqrt(np.mean(x))),
            in_minmax_rate=("in_minmax", "mean"),
            in_p10_p90_rate=("in_p10_p90", "mean"),
            in_iqr_rate=("in_iqr", "mean"),
            residual_mean=("residual", "mean"),
        )
        .reset_index()
        .sort_values("test_rows", ascending=False)
    )
    return by_cluster


def plot_actual_vs_pred(test_eval, radius):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        test_eval[TARGET_COL],
        test_eval["predicted_wash_count_total"],
        alpha=0.15,
        s=8,
        color="#2563eb",
        edgecolor="none",
    )
    lo = min(test_eval[TARGET_COL].min(), test_eval["predicted_wash_count_total"].min())
    hi = max(test_eval[TARGET_COL].max(), test_eval["predicted_wash_count_total"].max())
    ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1.2, label="Ideal")
    ax.set_title(f"Actual vs Predicted Wash Count ({radius})")
    ax.set_xlabel("Actual wash_count_total")
    ax.set_ylabel("Predicted wash_count_total")
    ax.legend()
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / f"model_actual_vs_pred_{radius}.png", dpi=160)
    plt.close()


def plot_residual_by_cluster(test_eval, cluster_col, radius):
    top_clusters = (
        test_eval.groupby(cluster_col)
        .size()
        .sort_values(ascending=False)
        .head(12)
        .index
    )
    sub = test_eval[test_eval[cluster_col].isin(top_clusters)].copy()
    sub[cluster_col] = sub[cluster_col].astype(str)
    order = [str(c) for c in top_clusters]

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.boxplot(
        data=sub,
        x=cluster_col,
        y="residual",
        order=order,
        fliersize=1.8,
        linewidth=0.8,
        ax=ax,
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title(f"Residual Distribution by Cluster (Top test-volume clusters, {radius})")
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Residual (actual - predicted)")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / f"model_residual_by_cluster_{radius}.png", dpi=160)
    plt.close()


def main():
    print("Loading data …")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["calendar_day"] = pd.to_datetime(df["calendar_day"])

    # Keep only rows with target
    df = df[df[TARGET_COL].notna()].copy()

    summary_rows = []

    for cluster_col, radius in CLUSTER_COLS.items():
        print(f"\n{'=' * 70}")
        print(f"Modeling for {radius} ({cluster_col})")
        print(f"{'=' * 70}")

        # Use only rows with real cluster assignment
        sub = df[df[cluster_col] != -1].copy()

        feature_cols = [c for c in BASE_FEATURES if c in sub.columns and sub[c].notna().any()] + [cluster_col]
        model_df = sub[["calendar_day", TARGET_COL] + feature_cols].copy()

        # Impute numeric missing values with train medians later
        model_df = model_df.sort_values("calendar_day").reset_index(drop=True)

        # Time split: first 80% train, last 20% test
        split_idx = int(len(model_df) * 0.80)
        train_df = model_df.iloc[:split_idx].copy()
        test_df = model_df.iloc[split_idx:].copy()
        split_date = test_df["calendar_day"].min().date()
        print(f"Rows: {len(model_df):,} | Train: {len(train_df):,} | Test: {len(test_df):,} | Test starts: {split_date}")

        # Expected range per cluster from train only
        cluster_ranges = make_cluster_expected_ranges(train_df, cluster_col)

        # Prepare X/y
        X_train = train_df[feature_cols].copy()
        y_train = train_df[TARGET_COL].copy()
        X_test = test_df[feature_cols].copy()
        y_test = test_df[TARGET_COL].copy()

        # Fast, stable baseline model with all selected features.
        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=5.0, random_state=42)),
            ]
        )
        model.fit(X_train, y_train)
        y_pred = np.clip(model.predict(X_test), 0, None)

        # Metrics
        metrics = calc_global_metrics(y_test, y_pred)

        test_eval = test_df.copy()
        test_eval["predicted_wash_count_total"] = y_pred
        test_eval["residual"] = test_eval[TARGET_COL] - test_eval["predicted_wash_count_total"]
        test_eval["abs_error"] = np.abs(test_eval["residual"])
        test_eval["squared_error"] = test_eval["residual"] ** 2

        # Join expected train-range stats and evaluate in-range accuracy
        test_eval = build_test_range_flags(test_eval, cluster_ranges, cluster_col)
        cluster_level_eval = eval_by_cluster(test_eval, cluster_col)

        global_range_metrics = {
            "actual_in_train_minmax_rate": float(test_eval["in_minmax"].mean()),
            "actual_in_train_p10_p90_rate": float(test_eval["in_p10_p90"].mean()),
            "actual_in_train_iqr_rate": float(test_eval["in_iqr"].mean()),
        }

        # Save portable model (no sklearn pickle — loads in any sklearn version)
        portable_path = MODELS_DIR / f"wash_count_model_{radius}.portable.json"
        portable_payload = pipeline_to_portable(model, feature_cols, cluster_col, cluster_ranges)
        portable_payload["split"] = {
            "method": "time_based_80_20",
            "test_start_date": str(split_date),
        }
        with open(portable_path, "w", encoding="utf-8") as f:
            json.dump(portable_payload, f, indent=2)

        # Save full test predictions (with key columns)
        keep_cols = [
            "calendar_day",
            cluster_col,
            TARGET_COL,
            "predicted_wash_count_total",
            "residual",
            "abs_error",
            "in_minmax",
            "in_p10_p90",
            "in_iqr",
        ]
        test_eval[keep_cols].to_json(
            RESULTS_DIR / f"test_predictions_{radius}.json", orient="records", date_format="iso", indent=2
        )
        cluster_level_eval.to_json(
            RESULTS_DIR / f"cluster_range_eval_{radius}.json", orient="records", date_format="iso", indent=2
        )

        eval_payload = {
            "radius": radius,
            "cluster_col": cluster_col,
            "n_rows_total": int(len(model_df)),
            "n_rows_train": int(len(train_df)),
            "n_rows_test": int(len(test_df)),
            "split": {
                "type": "time_based",
                "train_fraction": 0.8,
                "test_fraction": 0.2,
                "test_start_date": str(split_date),
            },
            "model_metrics": metrics,
            "range_check_metrics": global_range_metrics,
            "top_clusters_by_test_volume": (
                cluster_level_eval.head(15)[
                    [
                        cluster_col,
                        "test_rows",
                        "actual_mean",
                        "pred_mean",
                        "mae",
                        "in_minmax_rate",
                        "in_p10_p90_rate",
                        "in_iqr_rate",
                    ]
                ]
                .to_dict(orient="records")
            ),
            "saved_model_path": str(portable_path),
        }
        with open(RESULTS_DIR / f"model_eval_{radius}.json", "w") as f:
            json.dump(eval_payload, f, indent=2)

        # Plots
        plot_actual_vs_pred(test_eval, radius)
        plot_residual_by_cluster(test_eval, cluster_col, radius)

        summary_rows.append(
            {
                "radius": radius,
                "cluster_col": cluster_col,
                "n_rows_train": len(train_df),
                "n_rows_test": len(test_df),
                **metrics,
                **global_range_metrics,
                "saved_model_path": str(portable_path),
            }
        )

        print(
            "Metrics:"
            f" MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f},"
            f" MAPE={metrics['mape']:.4f}, R2={metrics['r2']:.4f}"
        )
        print(
            "Range match:"
            f" min-max={global_range_metrics['actual_in_train_minmax_rate']:.3f},"
            f" p10-p90={global_range_metrics['actual_in_train_p10_p90_rate']:.3f},"
            f" iqr={global_range_metrics['actual_in_train_iqr_rate']:.3f}"
        )
        print(f"Saved portable model: {portable_path.name}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_json(RESULTS_DIR / "model_metrics_summary.json", orient="records", date_format="iso", indent=2)
    print("\nSaved summary: results/model_metrics_summary.json")
    print("Done.")


if __name__ == "__main__":
    main()
