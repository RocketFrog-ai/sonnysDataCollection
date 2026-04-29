from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from zeta_modelling.model_1.benchmarking import default_volume_sample_path, load_model1_benchmark_metrics
from zeta_modelling.model_1.phase3_advanced_forecast import (
    add_confidence_label,
    apply_global_uncertainty_calibration,
    final_report,
    load_artifacts,
)
from zeta_modelling.model_2.cohort_forecast_pipeline import (
    LessModelPair,
    _fit_lgbm_multiplier,
    add_less_features,
    add_more_features,
    add_site_type_feature,
    assign_dbscan_clusters,
    fit_cluster_ts_models,
    fit_site_type_classifier,
    forecast_new_site_60m,
    prepare_less,
    prepare_more,
)


ARTIFACTS_PATH = _REPO_ROOT / "zeta_modelling" / "model_1" / "phase3_artifacts.joblib"
DEFAULT_REPORT_PATH = _REPO_ROOT / "zeta_modelling" / "data_1" / "phase3_advanced_report.json"
DATA_1 = _REPO_ROOT / "zeta_modelling" / "data_1"
DATA_2 = _REPO_ROOT / "zeta_modelling" / "data_2"
MODEL2_OUT = DATA_2 / "model_2_outputs"
MODEL2_METRICS_PATH = MODEL2_OUT / "model2_metrics.json"
MODEL2_SAMPLE_PATH = MODEL2_OUT / "model2_sample_forecast_60m.csv"
MODEL2_ARTIFACTS_PATH = _REPO_ROOT / "zeta_modelling" / "model_2" / "model2_artifacts.joblib"

VOLUME_MODEL_COLUMNS: dict[str, str] = {
    "LightGBM no lag (phase 2 feature set)": "pred_lightgbm_no_lag",
    "LightGBM with true lags (leakage benchmark)": "pred_lightgbm_with_lag",
    "No-lag baseline (upgrade script)": "pred_p2_baseline_no_lag",
    "No-lag upgraded (cluster pseudo-lags)": "pred_p2_upgraded_no_lag",
    "Warm model with true lags": "pred_p2_warm_lags",
    "Deployable multiplier + inferred site_type": "pred_deployable_multiplier",
    "Site profile + multiplier target": "pred_site_profile_multiplier",
    "Phase 3 quantile forecaster (P50)": "pred_phase3_p50",
}


@st.cache_resource
def get_artifacts(path: str):
    return load_artifacts(Path(path))


@st.cache_data
def _load_volume_benchmark_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


@st.cache_data
def _load_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text())


@st.cache_data
def _load_model2_sample_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def _safe_float(x: object) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _model2_feature_lists(df_less: pd.DataFrame, df_more: pd.DataFrame) -> tuple[list[str], list[str]]:
    less_features = [
        "age_in_months",
        "age_sq",
        "log_age",
        "age_saturation",
        "growth_velocity",
        "is_early",
        "is_growth",
        "is_mature",
        "cluster_id",
        "cluster_month_avg",
        "cluster_month_std",
        "cluster_month_mean",
        "cluster_month_median",
        "cluster_month_p25",
        "cluster_month_p75",
        "cluster_age_avg",
        "cluster_growth_rate",
        "latitude",
        "longitude",
        "lat_lon_interaction",
        "month_sin",
        "month_cos",
        "site_type",
    ]
    more_features = [
        "age_in_months",
        "age_sq",
        "log_age",
        "age_saturation",
        "growth_velocity",
        "is_early",
        "is_growth",
        "is_mature",
        "cluster_id",
        "cluster_daily_avg",
        "cluster_std",
        "cluster_month_avg",
        "cluster_month_median",
        "cluster_month_p25",
        "cluster_month_p75",
        "cluster_age_avg",
        "cluster_growth_rate",
        "latitude",
        "longitude",
        "lat_lon_interaction",
        "month_sin",
        "month_cos",
        "site_type",
    ]
    return [c for c in less_features if c in df_less.columns], [c for c in more_features if c in df_more.columns]


def _train_model2_artifacts() -> dict[str, Any]:
    less = prepare_less(DATA_2 / "less_than-2yrs.csv")
    more = prepare_more(DATA_2 / "more_than-2yrs.csv")
    less, less_cent = assign_dbscan_clusters(less, eps_km=12.0)
    more, more_cent = assign_dbscan_clusters(more, eps_km=12.0)
    less_f = add_site_type_feature(add_less_features(less))
    more_f = add_site_type_feature(add_more_features(more))

    split = pd.Timestamp("2025-01-01")
    less_train = less_f[less_f["date"] < split].copy()
    more_train = more_f[more_f["date"] < split].copy()
    less_feats, more_feats = _model2_feature_lists(less_train, more_train)

    stm = fit_site_type_classifier(less_train)
    less_train_early = less_train[less_train["age_in_months"] <= 6].copy()
    less_train_main = less_train[less_train["age_in_months"] > 6].copy()
    if len(less_train_early) < 80:
        less_train_early = less_train.copy()
    if len(less_train_main) < 80:
        less_train_main = less_train.copy()
    less_pair = LessModelPair(
        early=_fit_lgbm_multiplier(less_train_early, less_feats, cat_cols=["cluster_id", "site_type"]),
        main=_fit_lgbm_multiplier(less_train_main, less_feats, cat_cols=["cluster_id", "site_type"]),
    )
    more_bundle = _fit_lgbm_multiplier(more_train, more_feats, cat_cols=["cluster_id", "site_type"])
    cluster_ts = fit_cluster_ts_models(more_train, horizon=36)

    metrics = _load_json(str(MODEL2_METRICS_PATH))
    interval_scale = _safe_float(metrics.get("uncertainty_calibration", {}).get("interval_width_scale"))
    if not np.isfinite(interval_scale) or interval_scale <= 0:
        interval_scale = 1.0

    return {
        "less_pair": less_pair,
        "more_bundle": more_bundle,
        "less_feat": less_f,
        "more_feat": more_f,
        "less_centroids": less_cent,
        "more_centroids": more_cent,
        "cluster_ts": cluster_ts,
        "site_type_model": stm,
        "interval_width_scale": interval_scale,
        "trained_at": pd.Timestamp.utcnow().isoformat(),
    }


@st.cache_resource
def get_model2_artifacts(path: str) -> dict[str, Any]:
    p = Path(path)
    if p.exists():
        return joblib.load(p)
    payload = _train_model2_artifacts()
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, p)
    return payload


def forecast_model2_for_input(
    lat: float,
    lon: float,
    months: int,
    scenario: str,
    margin_per_wash: float,
    fixed_monthly_cost: float,
    ramp_up_cost: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    art = get_model2_artifacts(str(MODEL2_ARTIFACTS_PATH))
    fc = forecast_new_site_60m(
        lat=lat,
        lon=lon,
        less_pair=art["less_pair"],
        more_bundle=art["more_bundle"],
        less_feat=art["less_feat"],
        more_feat=art["more_feat"],
        less_centroids=art["less_centroids"],
        more_centroids=art["more_centroids"],
        cluster_ts=art["cluster_ts"],
        stm=art["site_type_model"],
        interval_width_scale=art["interval_width_scale"],
        start_date="2026-01-01",
    ).head(months)
    fc["volume"] = fc["p50"]
    fc["low"] = fc["p10"]
    fc["high"] = fc["p90"]
    fc = scenario_adjust(fc, scenario)
    fc["monthly_profit"] = fc["volume"] * margin_per_wash - fixed_monthly_cost
    if len(fc):
        fc.loc[fc.index[0], "monthly_profit"] -= ramp_up_cost
    fc["cumulative_profit"] = fc["monthly_profit"].cumsum()
    fc["cumulative_volume"] = fc["volume"].cumsum()
    width = (fc["high"] - fc["low"]).clip(lower=0.0)
    base = fc["volume"].clip(lower=1.0)
    fc["interval_width_ratio"] = width / base
    fc["confidence_label"] = np.where(fc["interval_width_ratio"] > 1.2, "Low confidence", "Moderate confidence")
    be_idx = fc.index[fc["cumulative_profit"] >= 0]
    break_even = int(fc.loc[be_idx[0], "age_in_months"]) if len(be_idx) else None
    summary = {
        "expected_volume": float(fc["volume"].sum()),
        "range_low": float(fc["low"].sum()),
        "range_high": float(fc["high"].sum()),
        "total_profit": float(fc["monthly_profit"].sum()),
        "break_even_month": break_even,
    }
    return fc, summary


def _encode_model_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = df[feature_cols].copy()
    for c in X.columns:
        if str(X[c].dtype) == "category" or X[c].dtype == object:
            X[c] = X[c].astype("category").cat.codes
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    return X


@st.cache_data(show_spinner=False)
def _run_model1_ml_experiments(data1_dir: str) -> pd.DataFrame:
    d1 = Path(data1_dir)
    src = d1 / "phase1_final_monthly_2024_2025.csv"
    if not src.exists():
        return pd.DataFrame()
    df = pd.read_csv(src, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = df["date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["age_sq"] = pd.to_numeric(df["age_in_months"], errors="coerce") ** 2
    split = pd.Timestamp("2025-01-01")
    tr = df[df["date"] < split].copy()
    te = df[df["date"] >= split].copy()
    feats = [c for c in ["age_in_months", "real_age_months", "age_sq", "latitude", "longitude", "cluster_id", "month_sin", "month_cos"] if c in df.columns]
    if tr.empty or te.empty or not feats:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        "LightGBM": __import__("lightgbm").LGBMRegressor(n_estimators=900, learning_rate=0.05, num_leaves=63, random_state=42, verbose=-1),
    }
    X_tr = _encode_model_features(tr, feats)
    X_te = _encode_model_features(te, feats)
    y_tr = tr["monthly_volume"].to_numpy(dtype=float)
    y_te = te["monthly_volume"].to_numpy(dtype=float)
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        rows.append(
            {
                "dataset": "model_1/data_1",
                "model": name,
                "mae": float(mean_absolute_error(y_te, pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_te, pred))),
                "rows_test": int(len(te)),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def _run_model2_ml_experiments(data2_dir: str) -> pd.DataFrame:
    d2 = Path(data2_dir)
    less = prepare_less(d2 / "less_than-2yrs.csv")
    more = prepare_more(d2 / "more_than-2yrs.csv")
    less, _ = assign_dbscan_clusters(less, eps_km=12.0)
    more, _ = assign_dbscan_clusters(more, eps_km=12.0)
    less_f = add_site_type_feature(add_less_features(less))
    more_f = add_site_type_feature(add_more_features(more))
    split = pd.Timestamp("2025-01-01")
    less_train = less_f[less_f["date"] < split].copy()
    less_test = less_f[less_f["date"] >= split].copy()
    more_train = more_f[more_f["date"] < split].copy()
    more_test = more_f[more_f["date"] >= split].copy()
    less_feats, more_feats = _model2_feature_lists(less_train, more_train)
    rows: list[dict[str, object]] = []

    def _fit_eval(name: str, model: object, tr: pd.DataFrame, te: pd.DataFrame, feats: list[str], cohort: str) -> None:
        if tr.empty or te.empty:
            return
        X_tr = _encode_model_features(tr, feats)
        X_te = _encode_model_features(te, feats)
        y_tr = tr["monthly_volume"].to_numpy(dtype=float)
        y_te = te["monthly_volume"].to_numpy(dtype=float)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        rows.append({"dataset": "model_2/data_2", "cohort": cohort, "model": name, "mae": float(mean_absolute_error(y_te, pred)), "rmse": float(np.sqrt(mean_squared_error(y_te, pred))), "rows_test": int(len(te))})

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1),
        "LightGBM": __import__("lightgbm").LGBMRegressor(n_estimators=800, learning_rate=0.05, num_leaves=63, random_state=42, verbose=-1),
    }
    for n, m in models.items():
        _fit_eval(n, m, less_train, less_test, less_feats, "<2y")
        _fit_eval(n, m, more_train, more_test, more_feats, ">2y")
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def _run_ts_experiments(data2_dir: str) -> pd.DataFrame:
    d2 = Path(data2_dir)
    more = prepare_more(d2 / "more_than-2yrs.csv")
    more, _ = assign_dbscan_clusters(more, eps_km=12.0)
    more_f = add_more_features(more)
    split = pd.Timestamp("2025-01-01")
    grouped = more_f.groupby(["cluster_id", "date"], as_index=False)["monthly_volume"].median()
    train_g = grouped[grouped["date"] < split]
    test_g = grouped[grouped["date"] >= split]
    eligible = train_g.groupby("cluster_id").size().rename("n_tr").to_frame().join(test_g.groupby("cluster_id").size().rename("n_te"), how="inner")
    use_clusters = eligible[(eligible["n_tr"] >= 12) & (eligible["n_te"] >= 3)].sort_values("n_te", ascending=False).head(12).index.tolist()
    if not use_clusters:
        return pd.DataFrame()

    def seasonal_naive(tr: np.ndarray, h: int, season: int = 12) -> np.ndarray:
        if len(tr) >= season:
            return np.tile(tr[-season:], int(np.ceil(h / season)))[:h]
        return np.array([tr[-1] if len(tr) else 0.0] * h, dtype=float)

    def arima(tr: np.ndarray, h: int) -> np.ndarray:
        from statsmodels.tsa.arima.model import ARIMA

        return np.asarray(ARIMA(tr, order=(1, 1, 1)).fit().forecast(steps=h), dtype=float)

    def sarima(tr: np.ndarray, h: int) -> np.ndarray:
        fit = SARIMAX(tr, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        return np.asarray(fit.forecast(steps=h), dtype=float)

    def holt_winters(tr: np.ndarray, h: int) -> np.ndarray:
        fit = ExponentialSmoothing(tr, trend="add", seasonal="add", seasonal_periods=12).fit(optimized=True, use_brute=True)
        return np.asarray(fit.forecast(h), dtype=float)

    prophet_available = True
    try:
        from prophet import Prophet
    except Exception:
        prophet_available = False
        Prophet = None  # type: ignore[assignment]

    def prophet_fc(tr_dates: pd.Series, tr_vals: np.ndarray, h: int) -> np.ndarray:
        if not prophet_available or Prophet is None:
            raise RuntimeError("Prophet unavailable")
        pdf = pd.DataFrame({"ds": pd.to_datetime(tr_dates.values), "y": tr_vals})
        model = Prophet(weekly_seasonality=False, daily_seasonality=False, yearly_seasonality=True)
        model.fit(pdf)
        fut = model.make_future_dataframe(periods=h, freq="MS")
        pred = model.predict(fut)["yhat"].tail(h).to_numpy(dtype=float)
        return pred

    model_errs: dict[str, list[float]] = {"ARIMA": [], "SARIMA": [], "HoltWinters": [], "Prophet": []}
    for cid in use_clusters:
        tr_b = train_g[train_g["cluster_id"] == cid].sort_values("date")
        te_b = test_g[test_g["cluster_id"] == cid].sort_values("date")
        tr_y = tr_b["monthly_volume"].to_numpy(dtype=float)
        te_y = te_b["monthly_volume"].to_numpy(dtype=float)
        if len(te_y) == 0:
            continue
        candidates: dict[str, np.ndarray] = {}
        for name, fn in {
            "ARIMA": lambda: arima(tr_y, len(te_y)),
            "SARIMA": lambda: sarima(tr_y, len(te_y)),
            "HoltWinters": lambda: holt_winters(tr_y, len(te_y)),
            "Prophet": lambda: prophet_fc(tr_b["date"], tr_y, len(te_y)),
        }.items():
            try:
                candidates[name] = fn()
            except Exception:
                candidates[name] = seasonal_naive(tr_y, len(te_y))
        for k, p in candidates.items():
            model_errs[k].extend(np.abs(te_y - p).tolist())

    rows: list[dict[str, object]] = []
    for name, errs in model_errs.items():
        if not errs and name == "Prophet" and not prophet_available:
            rows.append({"dataset": "model_2/data_2", "model": "Prophet", "mae": np.nan, "rmse": np.nan, "rows_test": 0, "notes": "package not installed"})
            continue
        if errs:
            rows.append({"dataset": "model_2/data_2", "model": name, "mae": float(np.mean(errs)), "rmse": float(np.sqrt(np.mean(np.square(errs)))), "rows_test": int(len(errs)), "notes": ""})
    return pd.DataFrame(rows)


def get_current_coverage() -> float:
    if DEFAULT_REPORT_PATH.exists():
        payload = json.loads(DEFAULT_REPORT_PATH.read_text())
        return float(payload.get("backtest", {}).get("p10_p90_coverage", 0.454))
    return 0.454


def scenario_adjust(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    out = df.copy()
    if scenario == "Conservative":
        out["volume"] = out["low"]
    elif scenario == "Aggressive":
        out["volume"] = out["high"]
    out["cumulative_volume"] = out["volume"].cumsum()
    return out


def recommendation_text(confidence: str, break_even: str, risk: str, margin: float) -> str:
    rec = []
    if "Low" in confidence:
        rec.append("High uncertainty; validate with local demand checks before commit.")
    if "Not reached" in break_even:
        rec.append(f"Proceed only if margin_per_wash can improve above {margin:.1f}.")
    if "High" in risk:
        rec.append("Use phased rollout due to high variability cluster behavior.")
    if not rec:
        rec.append("Scenario is investable with current assumptions.")
    return "\n".join([f"- {r}" for r in rec])


def _annual_volume_by_year(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date", "volume"]).copy()
    out["year"] = out["date"].dt.year.astype(int)
    annual = (
        out.groupby("year", as_index=False)["volume"]
        .sum()
        .rename(columns={"volume": "annual_volume"})
        .sort_values("year")
    )
    return annual


def render_model1_benchmarking() -> None:
    st.subheader("Model 1 / Data 1 — holdout benchmarks")
    st.caption(
        "Tabular metrics are read from JSON reports under data_1. "
        "Default split is train before Jan 2025 and test from Jan 2025 monthly rows."
    )

    metrics_df = load_model1_benchmark_metrics(DATA_1, DEFAULT_REPORT_PATH)
    if metrics_df.empty:
        st.warning("No benchmark JSON files found in data_1.")
        return

    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    plot_df = metrics_df.dropna(subset=["mae"]).copy()
    if not plot_df.empty:
        plot_df = plot_df.sort_values("mae", ascending=True)
        labels = (plot_df["stage"].astype(str) + ": " + plot_df["model"].astype(str)).str.slice(0, 72)
        fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(plot_df))))
        ax.barh(labels, plot_df["mae"].to_numpy(), color="steelblue", alpha=0.85)
        ax.set_xlabel("MAE (monthly wash count)")
        ax.set_title("Mean absolute error by benchmark (lower is better)")
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

        fig2, ax2 = plt.subplots(figsize=(10, max(4, 0.35 * len(plot_df))))
        ax2.barh(labels, plot_df["rmse"].to_numpy(), color="darkseagreen", alpha=0.85)
        ax2.set_xlabel("RMSE (monthly wash count)")
        ax2.set_title("Root mean squared error by benchmark")
        plt.tight_layout()
        st.pyplot(fig2, clear_figure=True)

    st.subheader("Volume diagnostics (per model)")
    vol_path = default_volume_sample_path(DATA_1)
    if not vol_path.exists():
        st.info(
            "Aligned actual vs predicted volumes are not on disk yet. Generate them with:\n\n"
            "`python zeta_modelling/model_1/export_benchmark_volume_sample.py`"
        )
        return

    vol_df = _load_volume_benchmark_csv(str(vol_path))
    choice = st.selectbox("Model for volume plots", list(VOLUME_MODEL_COLUMNS.keys()))
    pred_col = VOLUME_MODEL_COLUMNS[choice]
    if pred_col not in vol_df.columns:
        st.error(f"Column {pred_col} missing from {vol_path.name}. Re-run the export script.")
        return

    sub = vol_df[["y_actual", pred_col]].dropna()
    if sub.empty:
        st.warning("No rows with valid predictions for this model.")
        return

    max_pts = 5000
    if len(sub) > max_pts:
        sub = sub.sample(max_pts, random_state=42)

    y_a = sub["y_actual"].to_numpy(dtype=float)
    y_p = sub[pred_col].to_numpy(dtype=float)
    mae = float(np.mean(np.abs(y_a - y_p)))
    rmse = float(np.sqrt(np.mean((y_a - y_p) ** 2)))

    c1, c2 = st.columns(2)
    c1.metric("MAE (this sample)", f"{mae:,.1f}")
    c2.metric("RMSE (this sample)", f"{rmse:,.1f}")

    fig_s, ax_s = plt.subplots(figsize=(6, 6))
    ax_s.scatter(y_a, y_p, alpha=0.25, s=8, c="tab:blue")
    lim = max(float(y_a.max()), float(y_p.max()), 1.0)
    ax_s.plot([0, lim], [0, lim], color="gray", linestyle="--", linewidth=1, label="y = x")
    ax_s.set_xlabel("Actual monthly volume")
    ax_s.set_ylabel("Predicted monthly volume")
    ax_s.set_title(f"{choice}\n(actual vs predicted, test months)")
    ax_s.legend(loc="upper left")
    plt.tight_layout()
    st.pyplot(fig_s, clear_figure=True)

    fig_h, ax_h = plt.subplots(figsize=(10, 4))
    bins = 50
    ax_h.hist(y_a, bins=bins, alpha=0.55, label="Actual", color="tab:blue", density=True)
    ax_h.hist(y_p, bins=bins, alpha=0.55, label="Predicted", color="tab:orange", density=True)
    ax_h.set_xlabel("Monthly volume")
    ax_h.set_ylabel("Density")
    ax_h.set_title("Volume distributions (normalized histograms)")
    ax_h.legend()
    plt.tight_layout()
    st.pyplot(fig_h, clear_figure=True)

    with st.expander("Export script details"):
        st.markdown(
            f"- Volume CSV: `{vol_path.relative_to(_REPO_ROOT)}`\n"
            f"- Rows: {len(vol_df):,} (all 2025+ test months, aligned across models)\n"
            "- Regenerate after changing training code or phase-1 inputs."
        )


def render_model2_diagnostics() -> None:
    st.subheader("Model 2 outputs")
    metrics = _load_json(str(MODEL2_METRICS_PATH))
    if not metrics:
        st.warning(
            "No Model 2 metrics found. Run:\n\n"
            "`python zeta_modelling/model_2/cohort_forecast_pipeline.py`"
        )
        return

    less = metrics.get("less_than_2y", {})
    more = metrics.get("more_than_2y", {})
    unc = metrics.get("uncertainty_calibration", {})
    cold = metrics.get("cold_start_holdout_first6", {})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("<2y MAE", f"{_safe_float(less.get('mae')):,.1f}")
    c2.metric(">2y MAE", f"{_safe_float(more.get('mae')):,.1f}")
    c3.metric("Cold-start MAE", f"{_safe_float(cold.get('mae')):,.1f}")
    c4.metric("Coverage (calibrated)", f"{_safe_float(unc.get('calibrated_coverage_p10_p90')):.3f}")

    rows = [
        {"cohort": "<2y", "mae": _safe_float(less.get("mae")), "wmape": _safe_float(less.get("wmape"))},
        {"cohort": ">2y", "mae": _safe_float(more.get("mae")), "wmape": _safe_float(more.get("wmape"))},
    ]
    cohort_df = pd.DataFrame(rows)
    st.dataframe(cohort_df, use_container_width=True, hide_index=True)

    fig_b, ax_b = plt.subplots(figsize=(8, 3.8))
    ax_b.bar(cohort_df["cohort"], cohort_df["mae"], color=["#4C78A8", "#59A14F"], alpha=0.9)
    ax_b.set_ylabel("MAE")
    ax_b.set_title("Model 2 cohort MAE (test)")
    st.pyplot(fig_b, clear_figure=True)

    fig_cov, ax_cov = plt.subplots(figsize=(8, 3.8))
    cov_df = pd.DataFrame(
        {
            "type": ["raw", "calibrated", "target"],
            "coverage": [
                _safe_float(unc.get("raw_coverage_p10_p90")),
                _safe_float(unc.get("calibrated_coverage_p10_p90")),
                _safe_float(unc.get("target_coverage")),
            ],
        }
    )
    ax_cov.bar(cov_df["type"], cov_df["coverage"], color=["#F28E2B", "#76B7B2", "#9C755F"], alpha=0.9)
    ax_cov.set_ylim(0, 1.0)
    ax_cov.set_ylabel("Coverage")
    ax_cov.set_title("P10–P90 coverage calibration (Model 2)")
    st.pyplot(fig_cov, clear_figure=True)

    sample = _load_model2_sample_csv(str(MODEL2_SAMPLE_PATH))
    if sample.empty:
        st.info("Sample forecast CSV missing. Generate pipeline outputs first.")
        return

    st.subheader("Sample 60-month forecast")
    fig_f, ax_f = plt.subplots(figsize=(10, 4))
    ax_f.plot(sample["date"], sample["p50"], label="P50", linewidth=2)
    ax_f.fill_between(sample["date"], sample["p10"], sample["p90"], alpha=0.2, label="P10-P90")
    ax_f.axvline(sample.iloc[min(23, len(sample) - 1)]["date"], color="gray", linestyle="--", linewidth=1)
    ax_f.set_ylabel("Monthly Volume")
    ax_f.set_title("Model 2 sample forecast (stitch boundary near month 24)")
    ax_f.legend()
    st.pyplot(fig_f, clear_figure=True)

    fig_age, ax_age = plt.subplots(figsize=(10, 4))
    ax_age.plot(sample["age_in_months"], sample["p50"], linewidth=2, color="tab:blue", label="P50")
    ax_age.fill_between(sample["age_in_months"], sample["p10"], sample["p90"], alpha=0.18, color="tab:blue")
    ax_age.axvline(24, color="gray", linestyle="--", linewidth=1, label="stitch split")
    ax_age.set_xlabel("Age in months")
    ax_age.set_ylabel("Monthly Volume")
    ax_age.set_title("Age curve with uncertainty")
    ax_age.legend()
    st.pyplot(fig_age, clear_figure=True)

    with st.expander("Model 2 artifact paths"):
        st.markdown(
            f"- Metrics JSON: `{MODEL2_METRICS_PATH.relative_to(_REPO_ROOT)}`\n"
            f"- Sample forecast CSV: `{MODEL2_SAMPLE_PATH.relative_to(_REPO_ROOT)}`\n"
            f"- Interval width scale: `{_safe_float(unc.get('interval_width_scale')):.4f}`"
        )


def render_model_benchmarking() -> None:
    st.header("Model Benchmarking")
    st.caption("Model-family experiments across ML and TS candidates.")
    t1, t2, t3 = st.tabs(["Model 1 ML", "Model 2 ML", "TS (Prophet/SARIMA/Holt-Winters/ARIMA)"])
    with t1:
        m1_df = _run_model1_ml_experiments(str(DATA_1))
        if m1_df.empty:
            st.warning("No Model 1 ML benchmark results produced.")
        else:
            st.dataframe(m1_df.sort_values("mae"), use_container_width=True, hide_index=True)
            fig_ml, ax_ml = plt.subplots(figsize=(9, 3.8))
            plot_ml = m1_df.sort_values("mae")
            ax_ml.bar(plot_ml["model"], plot_ml["mae"], color=["#4E79A7", "#F28E2B", "#59A14F"][: len(plot_ml)])
            ax_ml.set_xlabel("MAE")
            ax_ml.set_title("Model 1 ML benchmark (lower is better)")
            st.pyplot(fig_ml, clear_figure=True)
    with t2:
        m2_ml_df = _run_model2_ml_experiments(str(DATA_2))
        if m2_ml_df.empty:
            st.warning("No Model 2 ML benchmark results produced.")
        else:
            st.dataframe(m2_ml_df.sort_values(["cohort", "mae"]), use_container_width=True, hide_index=True)
            fig_ml2, ax_ml2 = plt.subplots(figsize=(10, max(4, 0.35 * len(m2_ml_df))))
            plot_ml2 = m2_ml_df.sort_values("mae")
            labels = plot_ml2["cohort"].astype(str) + " | " + plot_ml2["model"].astype(str)
            ax_ml2.barh(labels, plot_ml2["mae"], color="steelblue", alpha=0.85)
            ax_ml2.set_xlabel("MAE")
            ax_ml2.set_title("Model 2 ML benchmark by cohort")
            st.pyplot(fig_ml2, clear_figure=True)
    with t3:
        ts_df = _run_ts_experiments(str(DATA_2))
        if ts_df.empty:
            st.warning("No TS benchmark results produced.")
        else:
            st.dataframe(ts_df.sort_values("mae"), use_container_width=True, hide_index=True)
            fig_ts, ax_ts = plt.subplots(figsize=(10, 3.8))
            plot_ts = ts_df.dropna(subset=["mae"]).sort_values("mae")
            ax_ts.bar(plot_ts["model"], plot_ts["mae"], color=["#4E79A7", "#F28E2B", "#59A14F", "#E15759"][: len(plot_ts)])
            ax_ts.set_ylabel("MAE")
            ax_ts.set_title("TS benchmark on mature-cluster holdout")
            st.pyplot(fig_ts, clear_figure=True)


st.set_page_config(page_title="Car Wash Forecast Engine", layout="wide")
st.title("Car Wash Site Decision Engine")
st.caption("Forecasting tool for 3y/5y volume, uncertainty, and business viability.")

with st.sidebar:
    page = st.radio("Page", ["Site forecast", "Model benchmarking"], index=0)

if page == "Model benchmarking":
    render_model_benchmarking()
    st.stop()

if not ARTIFACTS_PATH.exists():
    st.error(
        "Artifacts not found. Run this once before using app:\n"
        "`python zeta_modelling/model_1/build_phase3_artifacts.py`"
    )
    st.stop()

with st.sidebar:
    st.header("Inputs")
    st.subheader("Location")
    lat = st.number_input("Latitude", value=32.646615, format="%.6f")
    lon = st.number_input("Longitude", value=-96.533319, format="%.6f")

    st.subheader("Business")
    margin_per_wash = st.number_input("margin_per_wash", value=4.0, min_value=0.0, step=0.1)
    fixed_monthly_cost = st.number_input("fixed_monthly_cost", value=50000.0, min_value=0.0, step=1000.0)
    ramp_up_cost = st.number_input("ramp_up_cost", value=150000.0, min_value=0.0, step=5000.0)

    with st.expander("Advanced", expanded=False):
        scenario = st.selectbox("Scenario", ["Expected", "Conservative", "Aggressive"], index=0)
        horizon_choice = st.selectbox("Time horizon", ["3y", "5y"], index=1)
        target_coverage = st.number_input("Target coverage", value=0.80, min_value=0.1, max_value=0.99, step=0.01)
        rebuild_m2 = st.checkbox("Rebuild and save Model 2 artifacts", value=False)
        st.caption("Model 1 mature-year YoY band: only if late years are strictly down YoY; else unchanged.")
        enable_mature_yoy = st.checkbox("Enable mature YoY band", value=True)
        mature_yoy_start_year = st.number_input("Start at forecast year", min_value=2, max_value=10, value=4, step=1)
        mature_min_yoy = st.number_input("Min YoY vs prior year (fraction)", value=0.005, min_value=0.0, max_value=0.5, step=0.005, format="%.4f")
        mature_max_yoy = st.number_input("Max YoY vs prior year (fraction)", value=0.05, min_value=0.0, max_value=1.0, step=0.01, format="%.4f")

    run = st.button("Run Forecast", type="primary", use_container_width=True)

if run:
    artifacts = get_artifacts(str(ARTIFACTS_PATH))
    months = 36 if horizon_choice == "3y" else 60
    _mmn, _mmx = float(mature_min_yoy), float(mature_max_yoy)
    if _mmn > _mmx:
        _mmn, _mmx = _mmx, _mmn

    forecast, summary = final_report(
        lat=lat,
        lon=lon,
        artifacts=artifacts,
        months=months,
        start_date="2026-01-01",
        margin_per_wash=margin_per_wash,
        fixed_monthly_cost=fixed_monthly_cost,
        ramp_up_cost=ramp_up_cost,
        enable_mature_yoy_control=enable_mature_yoy,
        mature_yoy_start_year=int(mature_yoy_start_year),
        mature_min_yoy=_mmn,
        mature_max_yoy=_mmx,
    )
    current_coverage = get_current_coverage()
    forecast, scale = apply_global_uncertainty_calibration(
        forecast=forecast,
        current_coverage=current_coverage,
        target_coverage=target_coverage,
    )
    forecast, overall_confidence = add_confidence_label(forecast)
    forecast = scenario_adjust(forecast, scenario)

    risk = "High variability" if float(forecast["interval_width_ratio"].mean()) > 1.0 else "Moderate variability"
    break_even_text = (
        f"Month {summary['break_even_month']}"
        if summary["break_even_month"] is not None
        else f"Not reached (at margin={margin_per_wash:g})"
    )

    expected_volume = float(forecast["volume"].sum())
    range_low = float(forecast["low"].sum())
    range_high = float(forecast["high"].sum())
    total_profit = float(forecast["monthly_profit"].sum())

    st.subheader("Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Expected Volume", f"{expected_volume:,.0f}")
    c2.metric("Range", f"{range_low:,.0f} - {range_high:,.0f}")
    c3.metric("Break-even", break_even_text)
    c4, c5, c6 = st.columns(3)
    c4.metric("Total Profit", f"{total_profit:,.0f}")
    c5.metric("Confidence", overall_confidence)
    c6.metric("Risk", risk)

    with st.expander("Model 1 mature-year YoY control (pre-calibration)"):
        st.json(summary.get("mature_yoy_control") or {})

    st.subheader("Forecast (P50 with P10-P90 band)")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(forecast["date"], forecast["volume"], label="P50", linewidth=2)
    ax.fill_between(forecast["date"], forecast["low"], forecast["high"], alpha=0.2, label="P10-P90")
    ax.set_ylabel("Monthly Volume")
    ax.legend()
    st.pyplot(fig, clear_figure=True)

    st.subheader("Cluster Trend Context")
    primary_cluster = str(pd.DataFrame(summary["top3_clusters"]).iloc[0]["cluster_id"])
    cluster_curve = artifacts.cluster_age_avg[
        artifacts.cluster_age_avg["cluster_id"].astype(str) == primary_cluster
    ].copy()
    compare = forecast[["age_in_months", "volume"]].copy()
    fig_cluster, ax_cluster = plt.subplots(figsize=(10, 4))
    if not cluster_curve.empty:
        cluster_curve = cluster_curve.sort_values("age_in_months")
        ax_cluster.plot(
            cluster_curve["age_in_months"],
            cluster_curve["cluster_age_avg"],
            label=f"Cluster {primary_cluster} avg trend",
            linewidth=2,
            alpha=0.8,
        )
    ax_cluster.plot(
        compare["age_in_months"],
        compare["volume"],
        label="New site expected (P50/scenario-adjusted)",
        linewidth=2,
        linestyle="--",
    )
    ax_cluster.set_xlabel("Age in months")
    ax_cluster.set_ylabel("Monthly Volume")
    ax_cluster.legend()
    st.pyplot(fig_cluster, clear_figure=True)

    st.subheader("Cumulative Profit")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(forecast["date"], forecast["cumulative_profit"], linewidth=2)
    if summary["break_even_month"] is not None:
        be_row = forecast[forecast["age_in_months"] == summary["break_even_month"]].head(1)
        if len(be_row):
            ax2.scatter(be_row["date"], be_row["cumulative_profit"], color="green", s=80, label="Break-even")
            ax2.legend()
    ax2.set_ylabel("Cumulative Profit")
    st.pyplot(fig2, clear_figure=True)

    with st.expander("Monthly Table", expanded=False):
        show_cols = ["date", "age_in_months", "volume", "low", "high", "monthly_profit", "cumulative_profit", "confidence_label"]
        st.dataframe(forecast[show_cols], use_container_width=True)

    st.subheader("Cluster Info")
    st.dataframe(pd.DataFrame(summary["top3_clusters"]), use_container_width=True)

    st.subheader("Risk + Recommendation")
    st.info(recommendation_text(overall_confidence, break_even_text, risk, margin_per_wash))

    st.subheader("Downloads")
    csv_bytes = forecast.to_csv(index=False).encode("utf-8")
    summary_payload = {
        "expected_volume": expected_volume,
        "range_low": range_low,
        "range_high": range_high,
        "break_even": break_even_text,
        "total_profit": total_profit,
        "confidence": overall_confidence,
        "risk": risk,
        "scenario": scenario,
        "horizon_months": months,
        "calibration_scale": scale,
        "current_coverage": current_coverage,
    }
    st.download_button("Download forecast CSV", data=csv_bytes, file_name="forecast_output.csv", mime="text/csv")
    st.download_button(
        "Download summary JSON",
        data=json.dumps(summary_payload, indent=2).encode("utf-8"),
        file_name="forecast_summary.json",
        mime="application/json",
    )

    st.subheader("Map")
    st.map(pd.DataFrame([{"lat": lat, "lon": lon}]))

    st.divider()
    if rebuild_m2 and MODEL2_ARTIFACTS_PATH.exists():
        MODEL2_ARTIFACTS_PATH.unlink()
        get_model2_artifacts.clear()
    m2_forecast, m2_summary = forecast_model2_for_input(
        lat=lat,
        lon=lon,
        months=months,
        scenario=scenario,
        margin_per_wash=margin_per_wash,
        fixed_monthly_cost=fixed_monthly_cost,
        ramp_up_cost=ramp_up_cost,
    )

    m2_break_even_text = (
        f"Month {m2_summary['break_even_month']}"
        if m2_summary["break_even_month"] is not None
        else f"Not reached (at margin={margin_per_wash:g})"
    )
    m2_conf = "Low confidence" if float(m2_forecast["interval_width_ratio"].mean()) > 1.0 else "Moderate confidence"
    m2_risk = "High variability" if float(m2_forecast["interval_width_ratio"].mean()) > 1.0 else "Moderate variability"

    st.header("Model comparison on same input")
    cmp1, cmp2, cmp3 = st.columns(3)
    cmp1.metric("Model 1 expected volume", f"{expected_volume:,.0f}")
    cmp2.metric("Model 2 expected volume", f"{m2_summary['expected_volume']:,.0f}")
    cmp3.metric("Volume delta (M2-M1)", f"{(m2_summary['expected_volume'] - expected_volume):,.0f}")
    fig_cmp, ax_cmp = plt.subplots(figsize=(10, 4))
    ax_cmp.plot(forecast["date"], forecast["volume"], label="Model 1 P50", linewidth=2)
    ax_cmp.plot(m2_forecast["date"], m2_forecast["volume"], label="Model 2 P50", linewidth=2)
    ax_cmp.fill_between(m2_forecast["date"], m2_forecast["low"], m2_forecast["high"], alpha=0.15, label="Model 2 P10-P90")
    ax_cmp.fill_between(forecast["date"], forecast["low"], forecast["high"], alpha=0.12, label="Model 1 P10-P90")
    ax_cmp.set_ylabel("Monthly Volume")
    ax_cmp.set_title("Forecast comparison for entered site")
    ax_cmp.legend(loc="upper left")
    st.pyplot(fig_cmp, clear_figure=True)
    st.subheader("Annual volume by year (Model 1 vs Model 2)")
    m1_annual = _annual_volume_by_year(forecast).rename(columns={"annual_volume": "model1_annual_volume"})
    m2_annual = _annual_volume_by_year(m2_forecast).rename(columns={"annual_volume": "model2_annual_volume"})
    annual_cmp = (
        m1_annual.merge(m2_annual, on="year", how="outer")
        .sort_values("year")
        .fillna(0.0)
    )
    st.dataframe(annual_cmp, use_container_width=True, hide_index=True)
    years = annual_cmp["year"].astype(int).to_numpy()
    x = np.arange(len(years), dtype=float)
    width = 0.36
    fig_annual, ax_annual = plt.subplots(figsize=(10, 4))
    ax_annual.bar(x - width / 2, annual_cmp["model1_annual_volume"].to_numpy(dtype=float), width=width, label="Model 1")
    ax_annual.bar(x + width / 2, annual_cmp["model2_annual_volume"].to_numpy(dtype=float), width=width, label="Model 2")
    ax_annual.set_xticks(x)
    ax_annual.set_xticklabels([str(y) for y in years])
    ax_annual.set_xlabel("Year")
    ax_annual.set_ylabel("Annual Volume")
    ax_annual.set_title("Annual volume comparison by year")
    ax_annual.legend()
    st.pyplot(fig_annual, clear_figure=True)

    st.header("Model 2 site forecast (same input)")
    d1, d2, d3 = st.columns(3)
    d1.metric("Expected Volume", f"{m2_summary['expected_volume']:,.0f}")
    d2.metric("Range", f"{m2_summary['range_low']:,.0f} - {m2_summary['range_high']:,.0f}")
    d3.metric("Break-even", m2_break_even_text)
    d4, d5, d6 = st.columns(3)
    d4.metric("Total Profit", f"{m2_summary['total_profit']:,.0f}")
    d5.metric("Confidence", m2_conf)
    d6.metric("Risk", m2_risk)

    st.subheader("Model 2 Forecast (P50 with P10-P90 band)")
    fig_m2, ax_m2 = plt.subplots(figsize=(10, 4))
    ax_m2.plot(m2_forecast["date"], m2_forecast["volume"], label="P50", linewidth=2)
    ax_m2.fill_between(m2_forecast["date"], m2_forecast["low"], m2_forecast["high"], alpha=0.2, label="P10-P90")
    ax_m2.set_ylabel("Monthly Volume")
    ax_m2.legend()
    st.pyplot(fig_m2, clear_figure=True)

    st.subheader("Model 2 Age Trend Context")
    fig_m2_age, ax_m2_age = plt.subplots(figsize=(10, 4))
    ax_m2_age.plot(m2_forecast["age_in_months"], m2_forecast["volume"], linewidth=2, label="New site expected (P50)")
    ax_m2_age.axvline(24, color="gray", linestyle="--", linewidth=1, label="stitch split")
    ax_m2_age.set_xlabel("Age in months")
    ax_m2_age.set_ylabel("Monthly Volume")
    ax_m2_age.legend()
    st.pyplot(fig_m2_age, clear_figure=True)

    st.subheader("Model 2 Cumulative Profit")
    fig_m2p, ax_m2p = plt.subplots(figsize=(10, 4))
    ax_m2p.plot(m2_forecast["date"], m2_forecast["cumulative_profit"], linewidth=2)
    if m2_summary["break_even_month"] is not None:
        be_row = m2_forecast[m2_forecast["age_in_months"] == m2_summary["break_even_month"]].head(1)
        if len(be_row):
            ax_m2p.scatter(be_row["date"], be_row["cumulative_profit"], color="green", s=80, label="Break-even")
            ax_m2p.legend()
    ax_m2p.set_ylabel("Cumulative Profit")
    st.pyplot(fig_m2p, clear_figure=True)

    with st.expander("Model 2 Monthly Table", expanded=False):
        cols = ["date", "age_in_months", "volume", "low", "high", "monthly_profit", "cumulative_profit", "confidence_label"]
        st.dataframe(m2_forecast[cols], use_container_width=True)

    st.subheader("Model 2 Recommendation")
    st.info(recommendation_text(m2_conf, m2_break_even_text, m2_risk, margin_per_wash))
