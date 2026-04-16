from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from tqdm.auto import tqdm


DEFAULT_INPUT = Path(__file__).resolve().parent / "master_daily_with_site_metadata.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "master_test_and_score_results.csv"
TARGET_COL = "wash_count_total"


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]) * 100.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 5-fold Test & Score model comparison on master daily dataset."
    )
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT), help="Input CSV path.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output metrics CSV path.")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of CV folds.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for CV/model reproducibility.")
    parser.add_argument(
        "--rf-estimators",
        type=int,
        default=300,
        help="Number of trees for Random Forest model.",
    )
    parser.add_argument(
        "--include-categorical",
        action="store_true",
        help="Include city/zip categorical one-hot features (slower, but closest to Orange setup).",
    )
    return parser.parse_args()


def build_feature_lists(include_categorical: bool) -> tuple[list[str], list[str], list[str]]:
    numeric_cols = [
        "day_of_week_feature",
        "prev_wash_count",
        "last_week_same_day",
        "running_avg_7_days",
        "dbscan_cluster_6km",
        "dbscan_cluster_12km",
        "dbscan_cluster_18km",
        "weather_total_precipitation_mm",
        "weather_total_snowfall_cm",
        "weather_avg_daily_max_windspeed_ms",
        "latitude",
        "longitude",
    ]
    categorical_cols = ["city", "zip"] if include_categorical else []
    feature_cols = numeric_cols + categorical_cols
    return feature_cols, numeric_cols, categorical_cols


def validate_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading data: {input_path}")
    df = pd.read_csv(input_path)

    feature_cols, numeric_cols, categorical_cols = build_feature_lists(args.include_categorical)
    validate_columns(df, feature_cols + [TARGET_COL])

    work = df[feature_cols + [TARGET_COL]].dropna(subset=[TARGET_COL]).copy()
    y = work[TARGET_COL].to_numpy(dtype=float)
    X = work[feature_cols]

    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    transformers: list[tuple[str, Pipeline, list[str]]] = [("num", numeric_transformer, numeric_cols)]
    if categorical_cols:
        categorical_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("cat", categorical_transformer, categorical_cols))

    preprocessor = ColumnTransformer(transformers)

    models = {
        "AdaBoost": AdaBoostRegressor(random_state=args.seed),
        "RF1": RandomForestRegressor(
            n_estimators=args.rf_estimators,
            random_state=args.seed,
            n_jobs=-1,
        ),
        "TREE-1": DecisionTreeRegressor(random_state=args.seed),
        "Gradient Boosting": GradientBoostingRegressor(random_state=args.seed),
        "KNN1": KNeighborsRegressor(n_neighbors=5),
        "Linear Regression": LinearRegression(),
    }

    cv = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    rows: list[dict[str, float | str]] = []

    print(
        f"Rows: {len(work):,} | Features: {len(feature_cols)} "
        f"| Categorical: {bool(categorical_cols)} | Models: {len(models)} | Folds: {args.n_splits}"
    )

    for model_name, model in tqdm(models.items(), total=len(models), desc="Models"):
        fold_metrics = []
        fold_iter = tqdm(
            cv.split(X),
            total=cv.get_n_splits(),
            desc=f"Folds ({model_name})",
            leave=False,
        )

        for fold_idx, (train_idx, test_idx) in enumerate(fold_iter, start=1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            pipe = Pipeline(
                [
                    ("prep", clone(preprocessor)),
                    ("model", clone(model)),
                ]
            )

            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            mse_v = mean_squared_error(y_test, y_pred)
            rmse_v = float(np.sqrt(mse_v))
            mae_v = mean_absolute_error(y_test, y_pred)
            mape_v = mape(y_test, y_pred)
            smape_v = smape(y_test, y_pred)
            r2_v = r2_score(y_test, y_pred)

            fold_metrics.append((mse_v, rmse_v, mae_v, mape_v, smape_v, r2_v))
            fold_iter.set_postfix({"fold": fold_idx, "mape": f"{mape_v:.2f}", "r2": f"{r2_v:.3f}"})

        arr = np.array(fold_metrics, dtype=float)
        rows.append(
            {
                "Model": model_name,
                "MSE": float(np.nanmean(arr[:, 0])),
                "RMSE": float(np.nanmean(arr[:, 1])),
                "MAE": float(np.nanmean(arr[:, 2])),
                "MAPE": float(np.nanmean(arr[:, 3])),
                "SMAPE": float(np.nanmean(arr[:, 4])),
                "R2": float(np.nanmean(arr[:, 5])),
            }
        )

    result = pd.DataFrame(rows).sort_values("MAPE", ascending=True).reset_index(drop=True)
    result.to_csv(output_path, index=False)

    print(f"\nSaved results to: {output_path}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
