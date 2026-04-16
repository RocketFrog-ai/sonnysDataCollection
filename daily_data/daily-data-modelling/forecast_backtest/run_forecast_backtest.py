from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

_MODELLING_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent

TARGET_COL = "wash_count_total"
DATE_COL = "calendar_day"
SITE_COL = "site_client_id"


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(np.square(err))))
    denom = np.maximum(np.abs(y_true), 1.0)
    mape = float(np.mean(np.abs(err) / denom) * 100)
    wape = float(np.sum(np.abs(err)) / max(np.sum(np.abs(y_true)), 1.0) * 100)
    return {"MAE": mae, "RMSE": rmse, "MAPE_pct": mape, "WAPE_pct": wape}


def seasonal_naive_forecast(train: pd.Series, horizon: int, season_len: int = 7) -> np.ndarray:
    vals = train.values.astype(float)
    if len(vals) == 0:
        return np.zeros(horizon, dtype=float)
    if len(vals) < season_len:
        return np.repeat(max(vals.mean(), 0.0), horizon)
    repeats = int(np.ceil(horizon / season_len))
    out = np.tile(vals[-season_len:], repeats)[:horizon]
    return np.clip(out, 0.0, None)


def holt_winters_forecast(train: pd.Series, horizon: int) -> np.ndarray:
    if train.nunique() <= 1 or len(train) < 21:
        return seasonal_naive_forecast(train, horizon)
    try:
        model = ExponentialSmoothing(
            train.astype(float),
            trend="add",
            seasonal="add",
            seasonal_periods=7,
            initialization_method="estimated",
        )
        fit = model.fit(optimized=True, use_brute=False)
        return np.clip(fit.forecast(horizon).values.astype(float), 0.0, None)
    except Exception:
        return seasonal_naive_forecast(train, horizon)


def arima_forecast(train: pd.Series, horizon: int) -> np.ndarray:
    if train.nunique() <= 1 or len(train) < 30:
        return seasonal_naive_forecast(train, horizon)
    try:
        model = ARIMA(
            train.astype(float),
            order=(1, 0, 1),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = model.fit(method_kwargs={"maxiter": 80})
        return np.clip(fit.forecast(horizon).values.astype(float), 0.0, None)
    except Exception:
        return seasonal_naive_forecast(train, horizon)


def build_group_forecasts(
    daily_grouped: pd.DataFrame,
    group_col: str,
    train_end: pd.Timestamp,
    future_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    out = []
    horizon = len(future_dates)
    all_train_dates = pd.date_range(daily_grouped[DATE_COL].min(), train_end, freq="D")

    for group, gdf in daily_grouped.groupby(group_col, sort=False):
        ts = (
            gdf.set_index(DATE_COL)[TARGET_COL]
            .reindex(all_train_dates, fill_value=0.0)
            .astype(float)
        )
        pred_arima = arima_forecast(ts, horizon)
        pred_hw = holt_winters_forecast(ts, horizon)

        out.append(
            pd.DataFrame(
                {
                    group_col: group,
                    DATE_COL: future_dates,
                    f"{group_col}_arima": pred_arima,
                    f"{group_col}_holt_winters": pred_hw,
                }
            )
        )

    return pd.concat(out, ignore_index=True)


def safe_fill_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    row_means = df[feature_cols].mean(axis=1, skipna=True)
    for c in feature_cols:
        df[c] = df[c].fillna(row_means)
        df[c] = df[c].fillna(df[c].median(skipna=True))
        df[c] = df[c].fillna(0.0)
        df[c] = np.clip(df[c], 0.0, None)
    return df


def build_site_share_tables(train_df: pd.DataFrame, level_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Day-of-week share lets higher-level forecasts inherit weekly demand shape.
    site_level_dow = (
        train_df.groupby([SITE_COL, level_col, "day_of_week"], as_index=False)[TARGET_COL]
        .sum()
        .rename(columns={TARGET_COL: "site_level_total"})
    )
    level_dow = (
        train_df.groupby([level_col, "day_of_week"], as_index=False)[TARGET_COL]
        .sum()
        .rename(columns={TARGET_COL: "level_total_dow"})
    )
    dow_share = site_level_dow.merge(level_dow, on=[level_col, "day_of_week"], how="left")
    dow_share["share_dow"] = np.where(
        dow_share["level_total_dow"] > 0,
        dow_share["site_level_total"] / dow_share["level_total_dow"],
        np.nan,
    )
    dow_share = dow_share[[SITE_COL, level_col, "day_of_week", "share_dow"]]

    site_level = (
        train_df.groupby([SITE_COL, level_col], as_index=False)[TARGET_COL]
        .sum()
        .rename(columns={TARGET_COL: "site_level_total"})
    )
    level_total = (
        train_df.groupby(level_col, as_index=False)[TARGET_COL]
        .sum()
        .rename(columns={TARGET_COL: "level_total"})
    )
    level_share = site_level.merge(level_total, on=[level_col], how="left")
    level_share["share_level"] = np.where(
        level_share["level_total"] > 0,
        level_share["site_level_total"] / level_share["level_total"],
        np.nan,
    )
    level_share = level_share[[SITE_COL, level_col, "share_level"]]
    return dow_share, level_share


def merge_level_predictions(
    base_rows: pd.DataFrame,
    preds_site: pd.DataFrame,
    preds_city: pd.DataFrame,
    preds_zip: pd.DataFrame,
    preds_region: pd.DataFrame,
    share_tables: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
) -> pd.DataFrame:
    merged = base_rows.merge(
        preds_site,
        on=[SITE_COL, DATE_COL],
        how="left",
    )
    for level_col, preds in [("city", preds_city), ("zip", preds_zip), ("region", preds_region)]:
        share_dow, share_level = share_tables[level_col]
        merged = merged.merge(preds, on=[level_col, DATE_COL], how="left")
        merged = merged.merge(share_dow, on=[SITE_COL, level_col, "day_of_week"], how="left")
        merged = merged.merge(share_level, on=[SITE_COL, level_col], how="left")
        merged["site_share"] = merged["share_dow"].fillna(merged["share_level"]).fillna(0.0)
        merged[f"{level_col}_arima_site_alloc"] = merged[f"{level_col}_arima"] * merged["site_share"]
        merged[f"{level_col}_holt_winters_site_alloc"] = (
            merged[f"{level_col}_holt_winters"] * merged["site_share"]
        )
        merged = merged.drop(columns=["share_dow", "share_level", "site_share"])
    return merged


def evaluate_models(
    eval_df: pd.DataFrame,
    y_true_col: str,
    pred_cols: list[str],
    output_scope: str,
) -> pd.DataFrame:
    rows = []
    y_true = eval_df[y_true_col].to_numpy(dtype=float)
    for col in pred_cols:
        y_pred = np.clip(eval_df[col].to_numpy(dtype=float), 0.0, None)
        m = metrics(y_true, y_pred)
        rows.append({"scope": output_scope, "model": col, **m})
    return pd.DataFrame(rows)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dow"] = out[DATE_COL].dt.dayofweek
    out["month"] = out[DATE_COL].dt.month
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    out["is_month_start"] = out[DATE_COL].dt.is_month_start.astype(int)
    out["is_month_end"] = out[DATE_COL].dt.is_month_end.astype(int)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest ARIMA/Holt-Winters + stacked ensemble")
    parser.add_argument(
        "--input",
        type=str,
        default=str(_MODELLING_ROOT / "master_daily_with_site_metadata.csv"),
        help="Path to the modelling CSV",
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
    df["state"] = df["state"].astype(str)

    df = df.sort_values([SITE_COL, DATE_COL])
    site_meta = df[[SITE_COL, "city", "zip", "region"]].drop_duplicates(SITE_COL)
    site_meta_full = (
        df.sort_values([SITE_COL, DATE_COL])
        .groupby(SITE_COL, as_index=False)
        .last()
    )

    split_valid_start = pd.Timestamp("2025-01-01")
    split_test_start = pd.Timestamp("2025-07-01")
    split_end = pd.Timestamp("2025-12-31")

    train_for_valid_end = split_valid_start - pd.Timedelta(days=1)  # 2024-12-31
    train_for_test_end = split_test_start - pd.Timedelta(days=1)  # 2025-06-30

    valid_dates = pd.date_range(split_valid_start, split_test_start - pd.Timedelta(days=1), freq="D")
    test_dates = pd.date_range(split_test_start, split_end, freq="D")

    # Aggregate at each hierarchy level.
    grouped_site = df.groupby([SITE_COL, DATE_COL], as_index=False)[TARGET_COL].sum()
    grouped_city = df.groupby(["city", DATE_COL], as_index=False)[TARGET_COL].sum()
    grouped_zip = df.groupby(["zip", DATE_COL], as_index=False)[TARGET_COL].sum()
    grouped_region = df.groupby(["region", DATE_COL], as_index=False)[TARGET_COL].sum()

    # Fit on 2024 to forecast 2025-H1 (stacking training target window)
    site_valid_preds = build_group_forecasts(grouped_site, SITE_COL, train_for_valid_end, valid_dates)
    city_valid_preds = build_group_forecasts(grouped_city, "city", train_for_valid_end, valid_dates)
    zip_valid_preds = build_group_forecasts(grouped_zip, "zip", train_for_valid_end, valid_dates)
    region_valid_preds = build_group_forecasts(grouped_region, "region", train_for_valid_end, valid_dates)

    # Fit on first 18 months to forecast 2025-H2 (final evaluation window)
    site_test_preds = build_group_forecasts(grouped_site, SITE_COL, train_for_test_end, test_dates)
    city_test_preds = build_group_forecasts(grouped_city, "city", train_for_test_end, test_dates)
    zip_test_preds = build_group_forecasts(grouped_zip, "zip", train_for_test_end, test_dates)
    region_test_preds = build_group_forecasts(grouped_region, "region", train_for_test_end, test_dates)

    valid_actual = df[(df[DATE_COL] >= split_valid_start) & (df[DATE_COL] < split_test_start)][
        [SITE_COL, DATE_COL, TARGET_COL]
    ].copy()
    test_actual = df[(df[DATE_COL] >= split_test_start) & (df[DATE_COL] <= split_end)][
        [SITE_COL, DATE_COL, TARGET_COL]
    ].copy()

    valid_actual = valid_actual.merge(site_meta, on=SITE_COL, how="left")
    test_actual = test_actual.merge(site_meta, on=SITE_COL, how="left")
    valid_actual["day_of_week"] = valid_actual[DATE_COL].dt.day_name()
    test_actual["day_of_week"] = test_actual[DATE_COL].dt.day_name()

    train_valid = df[df[DATE_COL] <= train_for_valid_end].copy()
    train_test = df[df[DATE_COL] <= train_for_test_end].copy()
    train_valid["day_of_week"] = train_valid[DATE_COL].dt.day_name()
    train_test["day_of_week"] = train_test[DATE_COL].dt.day_name()

    share_tables_valid = {
        "city": build_site_share_tables(train_valid, "city"),
        "zip": build_site_share_tables(train_valid, "zip"),
        "region": build_site_share_tables(train_valid, "region"),
    }
    share_tables_test = {
        "city": build_site_share_tables(train_test, "city"),
        "zip": build_site_share_tables(train_test, "zip"),
        "region": build_site_share_tables(train_test, "region"),
    }

    valid_full = merge_level_predictions(
        valid_actual,
        site_valid_preds,
        city_valid_preds,
        zip_valid_preds,
        region_valid_preds,
        share_tables_valid,
    )
    test_full = merge_level_predictions(
        test_actual,
        site_test_preds,
        city_test_preds,
        zip_test_preds,
        region_test_preds,
        share_tables_test,
    )

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

    valid_full = safe_fill_features(valid_full, feature_cols)
    test_full = safe_fill_features(test_full, feature_cols)
    valid_full = add_calendar_features(valid_full)
    test_full = add_calendar_features(test_full)

    # Train stacker on 2025-H1 and evaluate on 2025-H2.
    stacker = LinearRegression(positive=True)
    stacker.fit(valid_full[feature_cols], valid_full[TARGET_COL])
    valid_full["stacked_ensemble"] = np.clip(stacker.predict(valid_full[feature_cols]), 0.0, None)
    test_full["stacked_ensemble"] = np.clip(stacker.predict(test_full[feature_cols]), 0.0, None)
    valid_full["simple_avg_ensemble"] = np.clip(valid_full[feature_cols].mean(axis=1), 0.0, None)
    test_full["simple_avg_ensemble"] = np.clip(test_full[feature_cols].mean(axis=1), 0.0, None)

    # Stronger ensemble with exogenous metadata + calendar + base forecasts.
    static_feature_cols = [
        "weather_total_precipitation_mm",
        "weather_rainy_days",
        "weather_total_snowfall_cm",
        "weather_days_below_freezing",
        "weather_total_sunshine_hours",
        "weather_days_pleasant_temp",
        "weather_avg_daily_max_windspeed_ms",
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
        "age_on_30_sep_25",
        "region_enc",
        "state_enc",
        "costco_enc",
        "tunnel_count",
        "carwash_type_encoded",
        "latitude",
        "longitude",
    ]

    available_static = [c for c in static_feature_cols if c in site_meta_full.columns]
    model_site_meta = site_meta_full[[SITE_COL] + available_static].copy()
    for c in available_static:
        model_site_meta[c] = pd.to_numeric(model_site_meta[c], errors="coerce")

    valid_full = valid_full.merge(model_site_meta, on=SITE_COL, how="left")
    test_full = test_full.merge(model_site_meta, on=SITE_COL, how="left")

    gbm_features = feature_cols + ["dow", "month", "is_weekend", "is_month_start", "is_month_end"] + available_static
    for c in gbm_features:
        valid_full[c] = pd.to_numeric(valid_full[c], errors="coerce")
        test_full[c] = pd.to_numeric(test_full[c], errors="coerce")
        med = valid_full[c].median(skipna=True)
        valid_full[c] = valid_full[c].fillna(med if pd.notna(med) else 0.0)
        test_full[c] = test_full[c].fillna(med if pd.notna(med) else 0.0)

    gbm = GradientBoostingRegressor(
        loss="squared_error",
        max_depth=6,
        learning_rate=0.05,
        n_estimators=300,
        min_samples_leaf=50,
        subsample=0.8,
        random_state=42,
    )
    gbm.fit(valid_full[gbm_features], valid_full[TARGET_COL])
    valid_full["gbm_stacked_exog"] = np.clip(gbm.predict(valid_full[gbm_features]), 0.0, None)
    test_full["gbm_stacked_exog"] = np.clip(gbm.predict(test_full[gbm_features]), 0.0, None)

    # Post-hoc site-level calibration: learn actual-vs-predicted mapping on validation.
    calibrator = LinearRegression()
    calibrator.fit(valid_full[["simple_avg_ensemble"]], valid_full[TARGET_COL])
    valid_full["simple_avg_calibrated"] = np.clip(
        calibrator.predict(valid_full[["simple_avg_ensemble"]]),
        0.0,
        None,
    )
    test_full["simple_avg_calibrated"] = np.clip(
        calibrator.predict(test_full[["simple_avg_ensemble"]]),
        0.0,
        None,
    )

    # Site-level (site-day) accuracy.
    model_cols = feature_cols + [
        "simple_avg_ensemble",
        "simple_avg_calibrated",
        "stacked_ensemble",
        "gbm_stacked_exog",
    ]
    site_day_metrics = evaluate_models(test_full, TARGET_COL, model_cols, output_scope="site_day")

    # National daily totals accuracy.
    daily_totals = (
        test_full[[DATE_COL, TARGET_COL] + model_cols]
        .groupby(DATE_COL, as_index=False)
        .sum(numeric_only=True)
    )
    national_metrics = evaluate_models(daily_totals, TARGET_COL, model_cols, output_scope="national_daily_total")

    metrics_df = pd.concat([site_day_metrics, national_metrics], ignore_index=True)
    metrics_df = metrics_df.sort_values(["scope", "WAPE_pct", "RMSE"])

    weight_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "weight": stacker.coef_,
        }
    ).sort_values("weight", ascending=False)
    gbm_importance = pd.DataFrame(
        {
            "feature": gbm_features,
            "importance": gbm.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    monthly_compare = (
        test_full.assign(month=test_full[DATE_COL].dt.to_period("M").astype(str))
        .groupby("month", as_index=False)[[TARGET_COL] + model_cols]
        .sum(numeric_only=True)
    )
    for c in model_cols:
        monthly_compare[f"{c}_abs_error"] = (monthly_compare[c] - monthly_compare[TARGET_COL]).abs()
        monthly_compare[f"{c}_pct_error"] = (
            (monthly_compare[c] - monthly_compare[TARGET_COL]) / monthly_compare[TARGET_COL].clip(lower=1.0) * 100
        )

    test_full.to_csv(out_dir / "test_predictions_site_day.csv", index=False)
    daily_totals.to_csv(out_dir / "test_predictions_national_daily.csv", index=False)
    monthly_compare.to_csv(out_dir / "monthly_actual_vs_predicted.csv", index=False)
    metrics_df.to_csv(out_dir / "metrics_summary.csv", index=False)
    weight_df.to_csv(out_dir / "stacking_weights.csv", index=False)
    gbm_importance.to_csv(out_dir / "gbm_feature_importance.csv", index=False)

    summary = {
        "input_file": str(input_path),
        "date_min": str(df[DATE_COL].min().date()),
        "date_max": str(df[DATE_COL].max().date()),
        "n_rows": int(len(df)),
        "n_sites": int(df[SITE_COL].nunique()),
        "n_city": int(df["city"].nunique()),
        "n_zip": int(df["zip"].nunique()),
        "n_region": int(df["region"].nunique()),
        "validation_window": [str(valid_dates.min().date()), str(valid_dates.max().date())],
        "test_window": [str(test_dates.min().date()), str(test_dates.max().date())],
        "best_site_day_model": metrics_df[metrics_df["scope"] == "site_day"]
        .sort_values("WAPE_pct")
        .iloc[0][["model", "WAPE_pct", "RMSE", "MAE", "MAPE_pct"]]
        .to_dict(),
        "best_national_daily_model": metrics_df[metrics_df["scope"] == "national_daily_total"]
        .sort_values("WAPE_pct")
        .iloc[0][["model", "WAPE_pct", "RMSE", "MAE", "MAPE_pct"]]
        .to_dict(),
    }

    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
