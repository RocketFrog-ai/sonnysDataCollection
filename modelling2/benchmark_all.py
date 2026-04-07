"""
Unified benchmarks for modelling2/finale.csv — plain-language "accuracy" + parity with app/modelling.

- Production (`app.modelling.ds.prediction`) reports **classification accuracy**:
  predict wash tier Q1–Q4 from site features; "exact" = predicted tier equals true tier (% of sites).
- Earlier modelling2 regression reported **R²** and **MAPE** — that is not the same as tier accuracy.

This script reports both, plus quantile classifiers (KNN / RF / ExtraTrees), regression, and
previous→current style targets. **tunnel_count** and **effective_capacity** are excluded from the
primary modelling2 policy; a separate labeled row uses them only as a reference to match production docs.

Run:  python benchmark_all.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import accuracy_score, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

DATA_PATH = ROOT / "finale.csv"
OUT_JSON = ROOT / "benchmark_results.json"
OUT_MD = ROOT / "ACCURACY_REPORT.md"

RANDOM_STATE = 42
N_SPLITS = 5
# Same tier preset as default production analyse_site / v4 docs
TIER_PERCENTILE_SPLITS = [19, 31, 31, 19]  # "4-class-90pct-custom"


def add_engineered_features(df: pd.DataFrame, *, include_effective_capacity: bool) -> pd.DataFrame:
    """Mirror app.modelling.ds.prediction._add_engineered_features (subset)."""
    df = df.copy()

    cr = df.get("competitor_1_rating_count", pd.Series(np.nan, index=df.index))
    cg = df.get("competitor_1_google_rating", pd.Series(np.nan, index=df.index))
    cr_f = pd.to_numeric(cr, errors="coerce").fillna(100)
    cg_f = pd.to_numeric(cg, errors="coerce").fillna(4.0)
    df["competition_quality"] = cg_f * np.log1p(cr_f)

    gr = df.get("nearest_gas_station_rating", pd.Series(np.nan, index=df.index))
    grc = df.get("nearest_gas_station_rating_count", pd.Series(np.nan, index=df.index))
    gr_f = pd.to_numeric(gr, errors="coerce").fillna(3.5)
    grc_f = pd.to_numeric(grc, errors="coerce").fillna(30)
    df["gas_station_draw"] = gr_f * np.log1p(grc_f)

    wd = df.get("distance_nearest_walmart(5 mile)", pd.Series(np.nan, index=df.index))
    td = df.get("distance_nearest_target (5 mile)", pd.Series(np.nan, index=df.index))
    wd_f = pd.to_numeric(wd, errors="coerce").fillna(5.0)
    td_f = pd.to_numeric(td, errors="coerce").fillna(5.0)
    df["retail_proximity"] = 1.0 / (wd_f + td_f + 0.1)

    pd_col = df.get("weather_days_pleasant_temp", pd.Series(np.nan, index=df.index))
    bf = df.get("weather_days_below_freezing", pd.Series(np.nan, index=df.index))
    pd_f = pd.to_numeric(pd_col, errors="coerce")
    bf_f = pd.to_numeric(bf, errors="coerce")
    df["weather_drive_score"] = pd_f - bf_f

    cw = df.get("carwash_type_encoded", pd.Series(1, index=df.index))
    is_express = (pd.to_numeric(cw, errors="coerce").fillna(1) == 1).astype(float)
    df["is_express"] = is_express

    if include_effective_capacity and "tunnel_count" in df.columns:
        tc = pd.to_numeric(df["tunnel_count"], errors="coerce").fillna(1.0)
        df["effective_capacity"] = tc * is_express
    elif include_effective_capacity:
        df["effective_capacity"] = 0.0

    return df


def ensure_costco_enc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "costco_enc" not in df.columns and "distance_nearest_costco(5 mile)" in df.columns:
        df["costco_enc"] = pd.to_numeric(
            df["distance_nearest_costco(5 mile)"], errors="coerce"
        ).fillna(99)
    return df


def assign_wash_quantile_labels(counts: np.ndarray, splits: list[int]) -> np.ndarray:
    cum = np.cumsum([0] + list(splits))
    bounds = np.percentile(counts, cum)
    bounds[0] = float(np.min(counts))
    bounds[-1] = float(np.max(counts))
    q = pd.cut(
        pd.Series(counts),
        bins=bounds,
        labels=list(range(1, len(splits) + 1)),
        include_lowest=True,
    )
    return q.astype(int).values


# Features aligned with prediction.ML_FEATURE_ORDER (no tunnel / no EC variant)
FEATURES_NO_TUNNEL = [
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
    "is_express",
]

FEATURES_WITH_TUNNEL = FEATURES_NO_TUNNEL[:-1] + ["tunnel_count", "effective_capacity"]


def materialize_X(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    X = df[cols].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def cv_classify_metrics(pipe: Pipeline, X: pd.DataFrame, y: np.ndarray) -> dict:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    pred = cross_val_predict(pipe, X, y, cv=skf, n_jobs=1)
    exact = float(accuracy_score(y, pred))
    within_1 = float(np.mean(np.abs(pred - y) <= 1))
    return {
        "exact_accuracy_pct": round(exact * 100, 2),
        "within_1_tier_pct": round(within_1 * 100, 2),
        "n_samples": int(len(y)),
    }


def cv_regression_metrics(pipe: Pipeline, X: pd.DataFrame, y: np.ndarray) -> dict:
    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    pred = cross_val_predict(pipe, X, y, cv=kfold, n_jobs=1)
    pred = np.asarray(pred, dtype=float)
    y = np.asarray(y, dtype=float)
    # MAPE is unstable when OOF predictions go negative or near zero (trees/KNN).
    pred_pos = np.maximum(pred, 1.0)
    ape = np.abs(pred_pos - y) / np.maximum(y, 1.0)
    mape = float(np.mean(ape) * 100.0)
    mdape = float(np.median(ape) * 100.0)
    # Mean APE after dropping worst 5% of sites (OOF trees sometimes blow up on a few rows)
    trim = int(max(1, np.ceil(0.05 * len(ape))))
    mape_trim_pct = float(np.mean(np.sort(ape)[: len(ape) - trim]) * 100.0)
    r2 = float(r2_score(y, pred))
    rel_err = np.abs(pred_pos - y) / np.maximum(y, 1.0)
    within_10 = float(np.mean(rel_err <= 0.10) * 100.0)
    within_20 = float(np.mean(rel_err <= 0.20) * 100.0)
    return {
        "r2_oof_pct": round(r2 * 100, 2),
        "mape_pct": round(mape, 2),
        "mape_trimmed_worst5pct_pct": round(mape_trim_pct, 2),
        "median_ape_pct": round(mdape, 2),
        "within_10pct_of_actual_pct": round(within_10, 2),
        "within_20pct_of_actual_pct": round(within_20, 2),
        "n_samples": int(len(y)),
    }


def build_classifiers() -> dict[str, Pipeline]:
    imputer = KNNImputer(n_neighbors=5)
    knn = Pipeline(
        [
            ("imp", imputer),
            ("scale", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7, weights="distance")),
        ]
    )
    rf = Pipeline(
        [
            ("imp", imputer),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=10,
                    min_samples_leaf=4,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )
    et = Pipeline(
        [
            ("imp", imputer),
            (
                "clf",
                ExtraTreesClassifier(
                    n_estimators=600,
                    max_depth=8,
                    min_samples_leaf=5,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )
    return {"knn_classifier": knn, "random_forest_classifier": rf, "extra_trees_classifier": et}


def build_regressors(imputer_kind: str = "median") -> dict[str, Pipeline]:
    if imputer_kind == "knn":
        imp = KNNImputer(n_neighbors=5)
    else:
        imp = SimpleImputer(strategy="median")

    knn = Pipeline(
        [
            ("imp", imp),
            ("scale", StandardScaler()),
            ("reg", KNeighborsRegressor(n_neighbors=7, weights="distance")),
        ]
    )
    rf = Pipeline(
        [
            ("imp", imp),
            (
                "reg",
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
            ("imp", imp),
            (
                "reg",
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
    return {"knn_regressor": knn, "random_forest_regressor": rf, "extra_trees_regressor": et}


def feature_importance_ranking(
    pipe: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    model_step: str,
) -> list[dict[str, float | str]]:
    """Tree models only: Gini-based importances (relative split contribution, not causality)."""
    p = clone(pipe)
    p.fit(X, y)
    model = p.named_steps[model_step]
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return []
    names = list(X.columns)
    total = float(np.sum(imp)) or 1.0
    ranked = sorted(zip(names, imp), key=lambda x: -x[1])
    return [
        {
            "feature": n,
            "importance": float(v),
            "importance_pct_of_model_total": round(float(v / total * 100), 3),
        }
        for n, v in ranked
    ]


def importance_md_table(rows: list[dict[str, float | str]], top_n: int = 12) -> str:
    lines = [
        "| Rank | Feature | Importance % (of model total) |",
        "|---:|---|---:|",
    ]
    for i, r in enumerate(rows[:top_n], start=1):
        lines.append(
            f"| {i} | `{r['feature']}` | {r['importance_pct_of_model_total']} |"
        )
    return "\n".join(lines)


def main() -> None:
    raw = pd.read_csv(DATA_PATH)
    raw = ensure_costco_enc(raw)
    counts = pd.to_numeric(raw["current_count"], errors="coerce").values
    prev = pd.to_numeric(raw["previous_count"], errors="coerce").values

    y_q = assign_wash_quantile_labels(counts, TIER_PERCENTILE_SPLITS)

    results: dict = {
        "n_rows": len(raw),
        "tier_percentile_splits": TIER_PERCENTILE_SPLITS,
        "note_app_modelling": (
            "app/modelling quantile predictor uses ExtraTreesClassifier on wash tiers Q1–Q4 "
            "and reports exact tier match % (5-fold CV). It includes tunnel_count and "
            "effective_capacity; modelling2 policy excludes those from the primary feature set."
        ),
    }

    # --- Quantile classification (no tunnel) ---
    df_nt = add_engineered_features(raw, include_effective_capacity=False)
    X_nt = materialize_X(df_nt, FEATURES_NO_TUNNEL)
    results["quantile_classification_no_tunnel"] = {}
    for name, pipe in build_classifiers().items():
        results["quantile_classification_no_tunnel"][name] = cv_classify_metrics(
            pipe, X_nt, y_q
        )

    # KNN = "top 5 most similar sites" in (imputed + scaled) feature space; neighbors vote their wash tier.
    def _pipe_knn5(weights: str) -> Pipeline:
        return Pipeline(
            [
                ("imp", KNNImputer(n_neighbors=5)),
                ("scale", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=5, weights=weights)),
            ]
        )

    results["quantile_classification_no_tunnel_knn_top5_neighbors"] = {
        "knn_k5_uniform": cv_classify_metrics(_pipe_knn5("uniform"), X_nt, y_q),
        "knn_k5_distance": cv_classify_metrics(_pipe_knn5("distance"), X_nt, y_q),
    }

    # Optional: with tunnel + EC (reference — same spirit as production prediction module)
    df_wt = add_engineered_features(raw, include_effective_capacity=True)
    X_wt = materialize_X(df_wt, FEATURES_WITH_TUNNEL)
    results["quantile_classification_REFERENCE_with_tunnel_and_effective_capacity"] = {}
    for name, pipe in build_classifiers().items():
        results["quantile_classification_REFERENCE_with_tunnel_and_effective_capacity"][name] = (
            cv_classify_metrics(pipe, X_wt, y_q)
        )

    # With previous year in X (operational scenario: you know last year's volume)
    X_prev = X_nt.copy()
    X_prev["log1p_previous_count"] = np.log1p(np.maximum(prev, 0))
    results["quantile_classification_no_tunnel_plus_log1p_previous"] = {}
    for name, pipe in build_classifiers().items():
        results["quantile_classification_no_tunnel_plus_log1p_previous"][name] = cv_classify_metrics(
            pipe, X_prev, y_q
        )

    # --- Regression: predict current_count ---
    y_curr = counts.astype(float)
    results["regression_predict_current_no_tunnel"] = {}
    for name, pipe in build_regressors("median").items():
        results["regression_predict_current_no_tunnel"][name] = cv_regression_metrics(pipe, X_nt, y_curr)

    X_nt_prev = X_nt.copy()
    X_nt_prev["log1p_previous_count"] = np.log1p(np.maximum(prev, 0))
    results["regression_predict_current_no_tunnel_plus_log1p_previous"] = {}
    for name, pipe in build_regressors("median").items():
        results["regression_predict_current_no_tunnel_plus_log1p_previous"][name] = cv_regression_metrics(
            pipe, X_nt_prev, y_curr
        )

    # --- YoY ratio and delta (loss-style targets) ---
    ratio = y_curr / np.maximum(prev, 1.0)
    delta = y_curr - prev
    results["regression_predict_yoy_ratio_current_over_previous"] = {}
    for name, pipe in build_regressors("median").items():
        results["regression_predict_yoy_ratio_current_over_previous"][name] = cv_regression_metrics(
            pipe, X_nt, ratio
        )
    results["regression_predict_yoy_delta_current_minus_previous"] = {}
    for name, pipe in build_regressors("median").items():
        results["regression_predict_yoy_delta_current_minus_previous"][name] = cv_regression_metrics(
            pipe, X_nt, delta
        )

    # Naive baseline: predict current = previous (operational lower bound)
    naive_mape = float(mean_absolute_percentage_error(y_curr, prev) * 100.0)
    naive_r2 = float(r2_score(y_curr, prev))
    naive_rel = np.abs(prev - y_curr) / np.maximum(y_curr, 1.0)
    results["baseline_predict_current_equals_previous"] = {
        "mape_pct": round(naive_mape, 2),
        "r2_pct": round(naive_r2 * 100, 2),
        "within_20pct_of_actual_pct": round(float(np.mean(naive_rel <= 0.20) * 100), 2),
    }

    # --- Feature importances (full-data fit; tree Gini importances) ---
    clfs = build_classifiers()
    regs = build_regressors("median")
    results["feature_importances"] = {
        "note": (
            "RandomForest / ExtraTrees only (KNN has no importances). "
            "Trained on all rows; importances are Gini-based split contributions — "
            "useful for ranking drivers, not causal effects."
        ),
        "quantile_classification_with_log1p_previous": {
            "random_forest_classifier": feature_importance_ranking(
                clfs["random_forest_classifier"], X_prev, y_q, "clf"
            ),
            "extra_trees_classifier": feature_importance_ranking(
                clfs["extra_trees_classifier"], X_prev, y_q, "clf"
            ),
        },
        "quantile_classification_no_previous": {
            "random_forest_classifier": feature_importance_ranking(
                clfs["random_forest_classifier"], X_nt, y_q, "clf"
            ),
            "extra_trees_classifier": feature_importance_ranking(
                clfs["extra_trees_classifier"], X_nt, y_q, "clf"
            ),
        },
        "regression_predict_current_with_log1p_previous": {
            "random_forest_regressor": feature_importance_ranking(
                regs["random_forest_regressor"], X_nt_prev, y_curr, "reg"
            ),
            "extra_trees_regressor": feature_importance_ranking(
                regs["extra_trees_regressor"], X_nt_prev, y_curr, "reg"
            ),
        },
        "regression_predict_current_no_previous": {
            "random_forest_regressor": feature_importance_ranking(
                regs["random_forest_regressor"], X_nt, y_curr, "reg"
            ),
            "extra_trees_regressor": feature_importance_ranking(
                regs["extra_trees_regressor"], X_nt, y_curr, "reg"
            ),
        },
    }

    OUT_JSON.write_text(json.dumps(results, indent=2))

    # --- Markdown: plain English ---
    nt = results["quantile_classification_no_tunnel"]
    ref = results["quantile_classification_REFERENCE_with_tunnel_and_effective_capacity"]
    wprev = results["quantile_classification_no_tunnel_plus_log1p_previous"]
    reg = results["regression_predict_current_no_tunnel"]
    regp = results["regression_predict_current_no_tunnel_plus_log1p_previous"]
    yoy = results["regression_predict_yoy_ratio_current_over_previous"]
    fi = results["feature_importances"]

    def tier_table(d: dict) -> str:
        lines = ["| Model | Exact tier % | Within ±1 tier % |", "|---|---:|---:|"]
        for k, v in d.items():
            lines.append(f"| {k} | {v['exact_accuracy_pct']} | {v['within_1_tier_pct']} |")
        return "\n".join(lines)

    def reg_table(d: dict) -> str:
        lines = [
            "| Model | R² OOF (×100) | Median APE % | MAPE trim worst 5% % | Within 20% of actual % |",
            "|---|---:|---:|---:|---:|",
        ]
        for k, v in d.items():
            lines.append(
                f"| {k} | {v['r2_oof_pct']} | {v['median_ape_pct']} | {v['mape_trimmed_worst5pct_pct']} | {v['within_20pct_of_actual_pct']} |"
            )
        return "\n".join(lines)

    yoy_lines = "\n".join(
        f"| {k} | {v['r2_oof_pct']} | {v['median_ape_pct']} | {v['within_20pct_of_actual_pct']} |"
        for k, v in yoy.items()
    )
    md = f"""# Accuracy report (modelling2) — plain language

**Dataset:** `{DATA_PATH.name}` — **{results['n_rows']}** sites.

## Why this felt different from `app/modelling`

| What you saw | What it measures |
|---|---|
| **`app/modelling`** (`ds.prediction`) | **Classification:** each site gets a **tier** Q1–Q4 from `current_count`. **Accuracy** = % of sites where the **predicted tier equals the true tier** (5-fold CV). Docs cite ~**63.5%** exact and ~**98%** within one tier when **tunnel_count + effective_capacity** are in the model. |
| **Earlier modelling2 regression** | **Regression:** predict the **number** of washes. **R²** = variance explained (not “% correct”). **MAPE** = average **percent** error. There is no single “accuracy %” unless we define one (e.g. % predictions within 20% of true). |

So: **production “accuracy” is tier hit rate; regression “accuracy” must be defined** (we use median APE + “within 20% of actual” below).

---

## 1) Quantile / tier classification (like production)

Tiers use the same style as production default: percentile splits **{TIER_PERCENTILE_SPLITS}** on `current_count`.

### 1a) **Modelling2 policy — NO `tunnel_count`, NO `effective_capacity`**

{tier_table(nt)}

**1a-note — KNN “top 5 similar sites” only** (same features as 1a; no `previous_count`): find the **5 nearest** sites in imputed + scaled feature space; predicted tier = **vote** among those 5 sites’ true tiers (`uniform` = simple majority; `distance` = closer neighbors count more).

{tier_table(results["quantile_classification_no_tunnel_knn_top5_neighbors"])}

### 1b) **Reference only — WITH `tunnel_count` + `effective_capacity`** (matches production feature idea)

{tier_table(ref)}

### 1c) **No tunnel, but you know last year** — feature `log1p(previous_count)` added

{tier_table(wprev)}

---

## 2) Regression — predict **current** wash count (number, not tier)

### 2a) Site features only (no tunnel / EC)

{reg_table(reg)}

### 2b) Same + `log1p(previous_count)` (previous year known)

{reg_table(regp)}

**Naive baseline:** always guess **current = previous** → MAPE **{results['baseline_predict_current_equals_previous']['mape_pct']}%**, R² **{results['baseline_predict_current_equals_previous']['r2_pct']}**, within 20% of actual **{results['baseline_predict_current_equals_previous']['within_20pct_of_actual_pct']}%**.

---

## 3) YoY-style targets (predict **ratio** `current ÷ previous` from site features, no tunnel)

| Model | R² OOF (×100) | Median APE % | Within 20% of actual % |
|---|---:|---:|---:|
{yoy_lines}

*(Delta `current − previous` is in `benchmark_results.json`; percentage error on raw deltas is misleading without scaling.)*

On raw counts, **mean** MAPE is often inflated by a few catastrophic OOF rows; prefer **median APE** or **trimmed MAPE** (table in §2).

---

## 4) Which features mattered most? (tree importances)

{fi['note']}

### 4a) **Quantile (Q1–Q4)** — with `log1p_previous_count`

**Random forest classifier — top features**

{importance_md_table(fi['quantile_classification_with_log1p_previous']['random_forest_classifier'])}

**Extra trees classifier — top features**

{importance_md_table(fi['quantile_classification_with_log1p_previous']['extra_trees_classifier'])}

### 4b) **Quantile** — site features only (no previous year)

**Random forest**

{importance_md_table(fi['quantile_classification_no_previous']['random_forest_classifier'])}

### 4c) **Regression** (predict `current_count`) — with `log1p_previous_count`

**Random forest regressor**

{importance_md_table(fi['regression_predict_current_with_log1p_previous']['random_forest_regressor'])}

**Extra trees regressor**

{importance_md_table(fi['regression_predict_current_with_log1p_previous']['extra_trees_regressor'])}

### 4d) **Regression** — site features only (no previous year)

**Random forest regressor**

{importance_md_table(fi['regression_predict_current_no_previous']['random_forest_regressor'])}

Full ranked lists: key `feature_importances` in `{OUT_JSON.name}`.

---

## Bottom line (copy-paste)

- **Tier “accuracy” without tunnel:** best exact tier match ≈ **{max(v['exact_accuracy_pct'] for v in nt.values())}%** (ExtraTrees in this run).
- **Tier “accuracy” with tunnel+EC (reference):** best exact ≈ **{max(v['exact_accuracy_pct'] for v in ref.values())}%** (aligns with production ~63–69% depending on exact pipeline).
- **Count prediction (median APE, site features only):** best ≈ **{min(v['median_ape_pct'] for v in reg.values())}%**; **within 20% of true count** on ≈ **{max(v['within_20pct_of_actual_pct'] for v in reg.values())}%** of sites.
- **Count prediction with last year in the model:** best median APE ≈ **{min(v['median_ape_pct'] for v in regp.values())}%**; **within 20%** on ≈ **{max(v['within_20pct_of_actual_pct'] for v in regp.values())}%** of sites.

Full numbers: `{OUT_JSON.name}`.
"""
    OUT_MD.write_text(md)


if __name__ == "__main__":
    main()
