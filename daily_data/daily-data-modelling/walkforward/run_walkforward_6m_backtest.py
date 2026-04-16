from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_FORECAST_BACKTEST_DIR = Path(__file__).resolve().parent.parent / "forecast_backtest"
sys.path.insert(0, str(_FORECAST_BACKTEST_DIR))

_MODELLING_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent

from run_forecast_backtest import (
    DATE_COL,
    SITE_COL,
    TARGET_COL,
    add_calendar_features,
    build_group_forecasts,
    build_site_share_tables,
    merge_level_predictions,
    safe_fill_features,
)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(np.square(err))))
    denom = np.maximum(np.abs(y_true), 1.0)
    mape = float(np.mean(np.abs(err) / denom) * 100)
    wape = float(np.sum(np.abs(err)) / max(np.sum(np.abs(y_true)), 1.0) * 100)
    return {"MAE": mae, "RMSE": rmse, "MAPE_pct": mape, "WAPE_pct": wape}


def evaluate_scope(eval_df: pd.DataFrame, model_cols: list[str], scope_name: str, fold_name: str) -> pd.DataFrame:
    y_true = eval_df[TARGET_COL].to_numpy(dtype=float)
    rows = []
    for m in model_cols:
        y_pred = np.clip(eval_df[m].to_numpy(dtype=float), 0.0, None)
        rows.append({"fold": fold_name, "scope": scope_name, "model": m, **metrics(y_true, y_pred)})
    return pd.DataFrame(rows)


def fold_dates() -> list[dict[str, pd.Timestamp]]:
    # Expanding windows on 24-month panel:
    # 6m->6m, 12m->6m, 18m->6m
    return [
        {
            "fold": "train_6m_test_next_6m",
            "train_start": pd.Timestamp("2024-01-01"),
            "train_end": pd.Timestamp("2024-06-30"),
            "test_start": pd.Timestamp("2024-07-01"),
            "test_end": pd.Timestamp("2024-12-31"),
        },
        {
            "fold": "train_12m_test_next_6m",
            "train_start": pd.Timestamp("2024-01-01"),
            "train_end": pd.Timestamp("2024-12-31"),
            "test_start": pd.Timestamp("2025-01-01"),
            "test_end": pd.Timestamp("2025-06-30"),
        },
        {
            "fold": "train_18m_test_next_6m",
            "train_start": pd.Timestamp("2024-01-01"),
            "train_end": pd.Timestamp("2025-06-30"),
            "test_start": pd.Timestamp("2025-07-01"),
            "test_end": pd.Timestamp("2025-12-31"),
        },
    ]


def run_fold(df: pd.DataFrame, cfg: dict[str, pd.Timestamp]) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold = cfg["fold"]
    train_end = cfg["train_end"]
    test_start = cfg["test_start"]
    test_end = cfg["test_end"]
    future_dates = pd.date_range(test_start, test_end, freq="D")

    train_df = df[(df[DATE_COL] >= cfg["train_start"]) & (df[DATE_COL] <= train_end)].copy()
    test_actual = df[(df[DATE_COL] >= test_start) & (df[DATE_COL] <= test_end)].copy()

    grouped_site = df.groupby([SITE_COL, DATE_COL], as_index=False)[TARGET_COL].sum()
    grouped_city = df.groupby(["city", DATE_COL], as_index=False)[TARGET_COL].sum()
    grouped_zip = df.groupby(["zip", DATE_COL], as_index=False)[TARGET_COL].sum()
    grouped_region = df.groupby(["region", DATE_COL], as_index=False)[TARGET_COL].sum()

    site_preds = build_group_forecasts(grouped_site, SITE_COL, train_end, future_dates)
    city_preds = build_group_forecasts(grouped_city, "city", train_end, future_dates)
    zip_preds = build_group_forecasts(grouped_zip, "zip", train_end, future_dates)
    region_preds = build_group_forecasts(grouped_region, "region", train_end, future_dates)

    test_actual["day_of_week"] = test_actual[DATE_COL].dt.day_name()
    train_df["day_of_week"] = train_df[DATE_COL].dt.day_name()
    share_tables = {
        "city": build_site_share_tables(train_df, "city"),
        "zip": build_site_share_tables(train_df, "zip"),
        "region": build_site_share_tables(train_df, "region"),
    }

    base_rows = test_actual[[SITE_COL, DATE_COL, TARGET_COL, "city", "zip", "region", "day_of_week"]].copy()
    test_full = merge_level_predictions(base_rows, site_preds, city_preds, zip_preds, region_preds, share_tables)

    feature_cols = [
        f"{SITE_COL}_arima",
        f"{SITE_COL}_holt_winters",
        "city_arima_site_alloc",
        "city_holt_winters_site_alloc",
        "zip_arima_site_alloc",
        "zip_holt_winters_site_alloc",
        "region_arima_site_alloc",
        "region_holt_winters_site_alloc",
    ]
    test_full = safe_fill_features(test_full, feature_cols)
    test_full = add_calendar_features(test_full)
    test_full["simple_avg_ensemble"] = np.clip(test_full[feature_cols].mean(axis=1), 0.0, None)

    model_cols = feature_cols + ["simple_avg_ensemble"]

    site_day = evaluate_scope(test_full, model_cols, "site_day", fold)

    site_month_df = (
        test_full.assign(month=test_full[DATE_COL].dt.to_period("M").astype(str))
        .groupby([SITE_COL, "month"], as_index=False)[[TARGET_COL] + model_cols]
        .sum(numeric_only=True)
    )
    site_month = evaluate_scope(site_month_df, model_cols, "site_month", fold)

    site_6m_df = test_full.groupby([SITE_COL], as_index=False)[[TARGET_COL] + model_cols].sum(numeric_only=True)
    site_6m = evaluate_scope(site_6m_df, model_cols, "site_6month", fold)

    fold_metrics = pd.concat([site_day, site_month, site_6m], ignore_index=True)

    # Monthly overall comparison for the simple average model.
    monthly_overall = (
        test_full.assign(month=test_full[DATE_COL].dt.to_period("M").astype(str))
        .groupby("month", as_index=False)[[TARGET_COL, "simple_avg_ensemble"]]
        .sum(numeric_only=True)
    )
    monthly_overall["fold"] = fold
    monthly_overall["diff"] = monthly_overall["simple_avg_ensemble"] - monthly_overall[TARGET_COL]
    monthly_overall["pct_diff"] = monthly_overall["diff"] / monthly_overall[TARGET_COL].clip(lower=1.0) * 100

    return fold_metrics, monthly_overall


def main() -> None:
    parser = argparse.ArgumentParser(description="6m walk-forward backtest (6->6, 12->6, 18->6)")
    parser.add_argument(
        "--input",
        type=str,
        default=str(_MODELLING_ROOT / "master_daily_with_site_metadata.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(_SCRIPT_DIR),
        help="Directory for output files (defaults to this script's folder)",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL, TARGET_COL, SITE_COL, "city", "zip", "region"]).copy()
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0.0)

    df["zip"] = df["zip"].astype(str)
    df[SITE_COL] = df[SITE_COL].astype(str)
    df["city"] = df["city"].astype(str)
    df["region"] = df["region"].astype(str)
    df = df.sort_values([SITE_COL, DATE_COL]).reset_index(drop=True)

    metrics_frames = []
    monthly_frames = []
    for cfg in fold_dates():
        m, mo = run_fold(df, cfg)
        metrics_frames.append(m)
        monthly_frames.append(mo)

    all_metrics = pd.concat(metrics_frames, ignore_index=True)
    all_monthly = pd.concat(monthly_frames, ignore_index=True)

    # Best model by fold/scope.
    best_by_fold_scope = (
        all_metrics.sort_values(["fold", "scope", "WAPE_pct", "RMSE"])
        .groupby(["fold", "scope"], as_index=False)
        .first()
    )

    simple_only = all_metrics[all_metrics["model"] == "simple_avg_ensemble"].copy()
    simple_pivot = simple_only.pivot(index="fold", columns="scope", values="WAPE_pct").reset_index()
    simple_pivot = simple_pivot.rename_axis(None, axis=1)
    simple_pivot = simple_pivot[
        ["fold", "site_day", "site_month", "site_6month"]
    ].rename(
        columns={
            "site_day": "WAPE_site_day_pct",
            "site_month": "WAPE_site_month_pct",
            "site_6month": "WAPE_site_6month_pct",
        }
    )

    all_metrics.to_csv(out_dir / "walkforward_fold_metrics.csv", index=False)
    best_by_fold_scope.to_csv(out_dir / "walkforward_best_model_by_fold_scope.csv", index=False)
    simple_pivot.to_csv(out_dir / "walkforward_simple_avg_wape_trend.csv", index=False)
    all_monthly.to_csv(out_dir / "walkforward_monthly_overall_simple_avg.csv", index=False)

    summary = {
        "input_file": str(input_path),
        "n_rows": int(len(df)),
        "n_sites": int(df[SITE_COL].nunique()),
        "folds": [x["fold"] for x in fold_dates()],
        "simple_avg_wape_trend": simple_pivot.to_dict(orient="records"),
        "best_model_by_fold_scope": best_by_fold_scope.to_dict(orient="records"),
    }
    (out_dir / "walkforward_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
