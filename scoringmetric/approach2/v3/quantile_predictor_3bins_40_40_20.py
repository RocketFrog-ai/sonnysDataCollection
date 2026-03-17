from __future__ import annotations

"""
Three-bin (Low / Medium / High) quantile predictor — business-driven 40/40/20 bins.

This script:
- Loads the same merged dataset as QuantilePredictorV4
- Builds a 3-class target using:
    Low    = bottom ~40% of sites by current_count
    Medium = middle ~40%
    High   = top ~20%
- Trains an ExtraTreesClassifier (same hyperparams as v4)
- Reports 3-class cross-validated accuracy

CLI-only; intended for experimentation.
"""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score

from scoringmetric.approach2.v3.quantile_predictor_v4 import (
    _load_and_merge,
    FEATURE_LABELS,
    DISPLAY_ONLY_FEATURES,
)


def build_features_and_target_3bins_40_40_20() -> tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build X, y for a 3-class (Low / Medium / High) 40/40/20 problem.

    - current_count is split at the 40th and 80th percentiles:
        Low    = bottom 40%
        Medium = 40–80%
        High   = top 20%
    - Features: same 28 ML features as v4.
    """
    base = Path(__file__).resolve().parents[1]
    excel_path = base / "Proforma-v2-data-final (1).xlsx"
    csv_path = base / "extrapolation" / "temp_extrapolated.csv"

    df = _load_and_merge(excel_path, csv_path)

    counts = df["current_count"].dropna().values
    p40, p80 = np.percentile(counts, [40, 80])
    bounds = np.array([counts.min(), p40, p80, counts.max()])

    df["wash_3bin_40_40_20"] = pd.cut(
        df["current_count"],
        bins=bounds,
        labels=[1, 2, 3],  # 1=Low, 2=Medium, 3=High
        include_lowest=True,
    ).astype("Int64")

    ml_feature_order: List[str] = [
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
        "costco_enc",
        "distance_nearest_walmart(5 mile)",
        "distance_nearest_target (5 mile)",
        "other_grocery_count_1mile",
        "count_food_joints_0_5miles (0.5 mile)",
        "age_on_30_sep_25",
        "region_enc",
        "state_enc",
        "competition_quality",
        "gas_station_draw",
        "retail_proximity",
        "weather_drive_score",
        "tunnel_count",
        "effective_capacity",
    ]

    available = set(df.columns) - {
        "Address",
        "current_count",
        "location_id",
        "client_id",
        "street",
        "city",
        "zip",
        "region",
        "state",
        "wash_q",
        "_match_type",
        "primary_carwash_type",
        *DISPLAY_ONLY_FEATURES,
    }
    feature_cols = [f for f in ml_feature_order if f in available and f in FEATURE_LABELS]

    mask = df["wash_3bin_40_40_20"].notna()
    y = df.loc[mask, "wash_3bin_40_40_20"].astype(int).values

    X_raw = df.loc[mask, feature_cols].copy()
    for c in X_raw.columns:
        X_raw[c] = pd.to_numeric(X_raw[c], errors="coerce")

    X = KNNImputer(n_neighbors=5).fit_transform(X_raw)
    return X, y, feature_cols


def main() -> None:
    X, y, feature_cols = build_features_and_target_3bins_40_40_20()
    print(f"3-bin 40/40/20 target: y shape={y.shape}, classes={sorted(set(y))}")
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = ExtraTreesClassifier(
        n_estimators=600,
        max_depth=8,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    cv = cross_val_score(clf, X, y, cv=skf, scoring="accuracy")
    print(f"3-bin 40/40/20 ExtraTrees — CV per fold: {cv}")
    print(f"Mean CV accuracy: {cv.mean():.3%}")


if __name__ == "__main__":
    main()

