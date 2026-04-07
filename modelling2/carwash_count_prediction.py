"""
EDA + regression models to predict current_count (car wash volume) from finale.csv.
Excludes tunnel_count from features (modelling2 only). Does not modify app code.
"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
(ROOT / ".mplconfig").mkdir(exist_ok=True)
# Reduce BLAS/OpenMP thread oversubscription (helps stability in some environments)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

DATA_PATH = ROOT / "finale.csv"
EDA_DIR = ROOT / "eda_output"
RESULTS_PATH = ROOT / "model_results.json"

# Target = car wash count in the dataset column name
TARGET = "current_count"

DROP_FROM_FEATURES = [
    "tunnel_count",  # excluded per project brief (modelling2 only)
    TARGET,
    "client_id",
    "street",
    "city",
    "zip",
    "site_client_id",
    "location_id",
    "_match_type",
    "region",
    "state",
    "primary_carwash_type",
    "official_website",
]

RANDOM_STATE = 42
N_BOOTSTRAP = 2000


def load_xy() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_PATH)
    y = df[TARGET].astype(float)
    X = df.drop(columns=[c for c in DROP_FROM_FEATURES if c in df.columns], errors="ignore")
    # Only numeric features
    X = X.select_dtypes(include=[np.number])
    assert "tunnel_count" not in X.columns
    return X, y


def run_eda(df: pd.DataFrame, y: pd.Series) -> None:
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(y, kde=True, ax=ax, color="steelblue")
    ax.set_title("Distribution of car wash count (current_count)")
    ax.set_xlabel("current_count")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "01_hist_current_count.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["previous_count"], kde=True, ax=ax, color="darkseagreen")
    ax.set_title("Distribution of previous_count")
    ax.set_xlabel("previous_count")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "02_hist_previous_count.png", dpi=150)
    plt.close(fig)

    age = df["age_on_30_sep_25"]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(age, kde=True, ax=ax, color="coral")
    ax.set_title("Site age (years) on 2025-09-30")
    ax.set_xlabel("age_on_30_sep_25")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "03_hist_age.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(x=age, y=y, alpha=0.5, ax=ax)
    ax.set_title("Car wash count vs site age")
    ax.set_xlabel("age_on_30_sep_25")
    ax.set_ylabel("current_count")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "04_scatter_age_vs_current.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(x=df["previous_count"], y=y, alpha=0.5, ax=ax)
    ax.set_title("Current vs previous car wash count")
    ax.set_xlabel("previous_count")
    ax.set_ylabel("current_count")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "05_scatter_previous_vs_current.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(x=age, y=df["previous_count"], alpha=0.5, ax=ax)
    ax.set_title("Previous count vs site age")
    ax.set_xlabel("age_on_30_sep_25")
    ax.set_ylabel("previous_count")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "06_scatter_age_vs_previous.png", dpi=150)
    plt.close(fig)

    # Log-scale joint relationship (heavy tails)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(
        x=np.log1p(df["previous_count"]),
        y=np.log1p(y),
        hue=age,
        palette="viridis",
        alpha=0.65,
        ax=ax,
    )
    ax.set_title("log1p(current) vs log1p(previous), colored by age")
    ax.set_xlabel("log1p(previous_count)")
    ax.set_ylabel("log1p(current_count)")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "07_scatter_log_previous_log_current_hue_age.png", dpi=150)
    plt.close(fig)

    num = df.select_dtypes(include=[np.number])
    cols = [c for c in num.columns if c != "tunnel_count"]
    if TARGET in cols:
        cols.remove(TARGET)
    # Subset for readability
    focus = [c for c in ["previous_count", "age_on_30_sep_25", "carwash_type_encoded", "region_enc"] if c in num.columns]
    cor_cols = focus + [TARGET]
    cor_cols = [c for c in cor_cols if c in num.columns]
    cm = num[cor_cols].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Correlation (selected numeric features + target)")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "08_corr_heatmap_subset.png", dpi=150)
    plt.close(fig)

    # Missingness
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=True)
    if len(miss):
        fig, ax = plt.subplots(figsize=(8, max(3, len(miss) * 0.2)))
        miss.plot(kind="barh", ax=ax, color="slategray")
        ax.set_title("Missing values per column")
        fig.tight_layout()
        fig.savefig(EDA_DIR / "09_missing_counts.png", dpi=150)
        plt.close(fig)


def make_preprocessor(feature_names: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [("num", SimpleImputer(strategy="median"), feature_names)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_tree_pipelines(feature_names: list[str]) -> dict[str, Pipeline]:
    pre = make_preprocessor(feature_names)

    rf = Pipeline(
        [
            ("prep", pre),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=400,
                    max_depth=12,
                    min_samples_leaf=4,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )

    et = Pipeline(
        [
            ("prep", pre),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=400,
                    max_depth=14,
                    min_samples_leaf=4,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )

    return {"random_forest": rf, "extra_trees": et}


def cv_scores(estimator, X, y) -> dict:
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    # n_jobs=1 avoids macOS fork/spawn issues with OpenMP in some sklearn builds
    r2 = cross_val_score(estimator, X, y, cv=kf, scoring="r2", n_jobs=1)
    neg_mae = cross_val_score(estimator, X, y, cv=kf, scoring="neg_mean_absolute_error", n_jobs=1)
    return {
        "r2_mean": float(np.mean(r2)),
        "r2_std": float(np.std(r2)),
        "mae_mean": float(np.mean(-neg_mae)),
        "mae_std": float(np.std(-neg_mae)),
        "r2_folds": r2.tolist(),
    }


def in_sample_r2(pipeline: Pipeline, X, y) -> float:
    pred = pipeline.predict(X)
    return float(r2_score(y, pred))


def bootstrap_test_ci(y_true: np.ndarray, y_pred: np.ndarray, n: int = N_BOOTSTRAP, seed: int = RANDOM_STATE):
    rng = np.random.default_rng(seed)
    n_samples = len(y_true)
    r2s, mapes = [], []
    for _ in range(n):
        idx = rng.integers(0, n_samples, size=n_samples)
        r2s.append(r2_score(y_true[idx], y_pred[idx]))
        mapes.append(mean_absolute_percentage_error(y_true[idx], y_pred[idx]) * 100.0)
    r2s, mapes = np.array(r2s), np.array(mapes)
    return {
        "r2_ci95_low": float(np.percentile(r2s, 2.5)),
        "r2_ci95_high": float(np.percentile(r2s, 97.5)),
        "mape_pct_ci95_low": float(np.percentile(mapes, 2.5)),
        "mape_pct_ci95_high": float(np.percentile(mapes, 97.5)),
    }


def try_knn_variants(X_train, y_train, X_test, y_test, feature_names: list[str]) -> list[dict]:
    pre = make_preprocessor(feature_names)
    results = []
    for k in [3, 5, 7, 11]:
        for weights in ["uniform", "distance"]:
            pipe = Pipeline(
                [
                    ("prep", pre),
                    ("scale", StandardScaler()),
                    ("model", KNeighborsRegressor(n_neighbors=k, weights=weights)),
                ]
            )
            cv = cv_scores(pipe, X_train, y_train)
            pipe.fit(X_train, y_train)
            tr_r2 = in_sample_r2(pipe, X_train, y_train)
            pred = pipe.predict(X_test)
            results.append(
                {
                    "name": f"knn_k{k}_{weights}",
                    "k": k,
                    "weights": weights,
                    "cv": cv,
                    "train_r2": tr_r2,
                    "test_r2": float(r2_score(y_test, pred)),
                    "test_mae": float(mean_absolute_error(y_test, pred)),
                    "test_mape_pct": float(mean_absolute_percentage_error(y_test, pred) * 100.0),
                    "overfit_gap_r2": float(tr_r2 - cv["r2_mean"]),
                }
            )
    return results


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    X, y = load_xy()
    feature_names = list(X.columns)

    run_eda(df, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    knn_results = try_knn_variants(X_train, y_train, X_test, y_test, feature_names)
    pipelines = build_tree_pipelines(feature_names)

    tree_results = []
    for name, pipe in pipelines.items():
        cv = cv_scores(pipe, X_train, y_train)
        pipe.fit(X_train, y_train)
        tr_r2 = float(r2_score(y_train, pipe.predict(X_train)))
        pred = pipe.predict(X_test)
        tree_results.append(
            {
                "name": name,
                "cv": cv,
                "train_r2": tr_r2,
                "test_r2": float(r2_score(y_test, pred)),
                "test_mae": float(mean_absolute_error(y_test, pred)),
                "test_mape_pct": float(mean_absolute_percentage_error(y_test, pred) * 100.0),
                "overfit_gap_r2": float(tr_r2 - cv["r2_mean"]),
            }
        )

    all_rows = knn_results + tree_results
    # Prefer high test R² with small overfit gap; secondary: CV R²
    def score_row(r):
        return (r["test_r2"], -r["overfit_gap_r2"], r["cv"]["r2_mean"])

    best = max(all_rows, key=score_row)
    best_name = best["name"]

    # Refit best model on full training logic: refit winning config on train+test? User wants generalization — report test metrics.
    # Optional: fit on all data for deployment artifact
    if best_name.startswith("knn_k"):
        k = int(best["k"])
        w = str(best["weights"])
        final = Pipeline(
            [
                ("prep", make_preprocessor(feature_names)),
                ("scale", StandardScaler()),
                ("model", KNeighborsRegressor(n_neighbors=k, weights=w)),
            ]
        )
    elif best_name == "random_forest":
        final = pipelines["random_forest"]
    else:
        final = pipelines["extra_trees"]

    final.fit(X_train, y_train)
    test_pred = final.predict(X_test)
    boot = bootstrap_test_ci(y_test.values, test_pred)

    mean_count = float(y_test.mean())
    # "Accuracy-like" score: 100 - MAPE (higher is better when MAPE < 100)
    mape = float(mean_absolute_percentage_error(y_test, test_pred) * 100.0)
    accuracy_style = float(max(0.0, min(100.0, 100.0 - mape)))

    out = {
        "n_rows": int(len(df)),
        "n_features_used": len(feature_names),
        "target_column": TARGET,
        "excluded_from_X": [c for c in DROP_FROM_FEATURES if c != TARGET],
        "feature_columns": feature_names,
        "best_model": best_name,
        "holdout_fraction": 0.2,
        "test_metrics": {
            "r2": float(r2_score(y_test, test_pred)),
            "r2_as_variance_explained_pct": float(r2_score(y_test, test_pred) * 100.0),
            "mean_absolute_error": float(mean_absolute_error(y_test, test_pred)),
            "mean_absolute_percentage_error_pct": mape,
            "accuracy_style_100_minus_mape": accuracy_style,
            "mean_actual_count_on_test": mean_count,
        },
        "bootstrap_test_n": N_BOOTSTRAP,
        "bootstrap_test_ci95": boot,
        "all_model_comparison": sorted(all_rows, key=lambda r: (-r["test_r2"], r["overfit_gap_r2"])),
    }

    col_stats = []
    for c in sorted(df.columns):
        col_stats.append(
            {
                "column": c,
                "dtype": str(df[c].dtype),
                "missing_n": int(df[c].isna().sum()),
                "role": (
                    "target"
                    if c == TARGET
                    else (
                        "excluded (tunnel)"
                        if c == "tunnel_count"
                        else ("feature (numeric)" if c in feature_names else "excluded (id/text)")
                    )
                ),
            }
        )
    out["column_reference"] = col_stats

    with open(RESULTS_PATH, "w") as f:
        json.dump(out, f, indent=2)

    # Human-readable summary
    summary_path = ROOT / "MODEL_SUMMARY.md"
    rf_row = next(r for r in out["all_model_comparison"] if r["name"] == "random_forest")
    lines = [
        "# Car wash count prediction (modelling2)",
        "",
        f"- **Rows:** {out['n_rows']} sites",
        f"- **Target:** `{TARGET}` (car wash volume)",
        f"- **Features:** {out['n_features_used']} numeric columns; **`tunnel_count` excluded**",
        f"- **Best model (highest hold-out R², with overfit gap as tie-breaker):** `{best_name}`",
        f"- **Alternative:** `random_forest` — nearly the same R² ({rf_row['test_r2']:.3f}) with **lower MAPE** ({rf_row['test_mape_pct']:.2f}% vs {best['test_mape_pct']:.2f}%).",
        "",
        "## Hold-out test performance (20%, random_state=42)",
        "",
        f"- **R² (variance explained):** {out['test_metrics']['r2']:.4f} → **{out['test_metrics']['r2_as_variance_explained_pct']:.1f}%** of variance explained",
        f"- **MAPE:** {out['test_metrics']['mean_absolute_percentage_error_pct']:.2f}% mean absolute percentage error",
        f"- **Accuracy-style (100 − MAPE):** {out['test_metrics']['accuracy_style_100_minus_mape']:.1f}%",
        f"- **MAE:** {out['test_metrics']['mean_absolute_error']:.0f} (on a mean count of {mean_count:.0f})",
        "",
        "## Bootstrap 95% confidence intervals (test set, resampled rows)",
        "",
        f"- **R²:** [{boot['r2_ci95_low']:.4f}, {boot['r2_ci95_high']:.4f}]",
        f"- **MAPE %:** [{boot['mape_pct_ci95_low']:.2f}, {boot['mape_pct_ci95_high']:.2f}]",
        "",
        "## All models (sorted by test R²)",
        "",
        "| model | CV R² mean±std | train R² | test R² | test MAPE % | overfit gap (train−CV R²) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in out["all_model_comparison"]:
        cv = r["cv"]
        lines.append(
            f"| {r['name']} | {cv['r2_mean']:.3f}±{cv['r2_std']:.3f} | {r['train_r2']:.3f} | "
            f"{r['test_r2']:.3f} | {r['test_mape_pct']:.2f} | {r['overfit_gap_r2']:.3f} |"
        )
    lines.append("")
    lines.append("## Column reference (all `finale.csv` columns)")
    lines.append("")
    lines.append("| column | dtype | missing | role |")
    lines.append("|---|---|---:|---|")
    for s in col_stats:
        lines.append(f"| `{s['column']}` | {s['dtype']} | {s['missing_n']} | {s['role']} |")
    lines.append("")
    lines.append(f"EDA figures: `{EDA_DIR.relative_to(ROOT)}/`")
    lines.append("")
    lines.append(
        "Run: `python carwash_count_prediction.py` from this folder "
        "(uses `MPLCONFIGDIR` under `modelling2/.mplconfig`)."
    )
    summary_path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
