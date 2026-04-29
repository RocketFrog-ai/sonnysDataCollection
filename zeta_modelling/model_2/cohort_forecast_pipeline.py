from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA


EARTH_RADIUS_KM = 6371.0088


@dataclass
class LGBMBundle:
    model: LGBMRegressor
    feature_columns: list[str]
    categorical_columns: list[str]
    target_mode: str  # "multiplier"


@dataclass
class LessModelPair:
    early: LGBMBundle
    main: LGBMBundle


@dataclass
class SiteTypeModel:
    clf: LGBMClassifier
    classes: list[str]


def _normalize_site_id(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    num = pd.to_numeric(s, errors="coerce")
    int_like = num.notna() & (num % 1 == 0)
    s.loc[int_like] = num.loc[int_like].astype("Int64").astype(str)
    float_like = num.notna() & ~int_like
    s.loc[float_like] = num.loc[float_like].map(lambda x: f"{x:.6f}".rstrip("0").rstrip("."))
    return s


def _make_site_id(df: pd.DataFrame) -> pd.Series:
    if "site_client_id" in df.columns:
        return _normalize_site_id(df["site_client_id"])
    return df["latitude"].round(6).astype(str) + "_" + df["longitude"].round(6).astype(str)


def _date_from_month_number(df: pd.DataFrame) -> pd.Series:
    m = pd.to_numeric(df["month_number"], errors="coerce")
    y = pd.Series(pd.NA, index=df.index, dtype="Int64")
    mo = pd.Series(pd.NA, index=df.index, dtype="Int64")
    y.loc[m.between(1, 12)] = 2024
    mo.loc[m.between(1, 12)] = m.loc[m.between(1, 12)].astype("Int64")
    y.loc[m.between(13, 24)] = 2025
    mo.loc[m.between(13, 24)] = (m.loc[m.between(13, 24)] - 12).astype("Int64")
    return pd.to_datetime({"year": y, "month": mo, "day": 1}, errors="coerce")


def prepare_less(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False).copy()
    df["site_id"] = _make_site_id(df)
    df["date"] = _date_from_month_number(df)
    df = df.dropna(subset=["site_id", "date", "latitude", "longitude"]).copy()
    agg = (
        df.groupby(["site_id", "date"], as_index=False)
        .agg(
            monthly_volume=("wash_count_total", "sum"),
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
            period_index=("period_index", "first"),
        )
        .sort_values(["site_id", "date"])
    )
    agg["age_in_months"] = pd.to_numeric(agg["period_index"], errors="coerce")
    missing_age = agg["age_in_months"].isna()
    agg.loc[missing_age, "age_in_months"] = agg[missing_age].groupby("site_id").cumcount() + 1
    agg["age_in_months"] = agg["age_in_months"].astype(int)
    agg["cohort"] = "less_than_2y"
    return agg


def prepare_more(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False).copy()
    df["site_id"] = _make_site_id(df)
    df["date"] = pd.to_datetime(df["calendar_day"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    df = df.dropna(subset=["site_id", "date", "latitude", "longitude"]).copy()
    df = df[(df["date"] >= pd.Timestamp("2024-01-01")) & (df["date"] <= pd.Timestamp("2025-12-31"))]
    agg = (
        df.groupby(["site_id", "date"], as_index=False)
        .agg(
            monthly_volume=("wash_count_total", "sum"),
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
        )
        .sort_values(["site_id", "date"])
    )
    agg["age_in_months"] = agg.groupby("site_id").cumcount() + 1 + 24
    agg["age_bucket"] = "mature"
    agg["cohort"] = "more_than_2y"
    return agg


def assign_dbscan_clusters(df: pd.DataFrame, eps_km: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sites = df[["site_id", "latitude", "longitude"]].drop_duplicates().copy()
    coords_rad = np.radians(sites[["latitude", "longitude"]].to_numpy())
    eps_rad = eps_km / EARTH_RADIUS_KM
    labels = DBSCAN(eps=eps_rad, min_samples=5, metric="haversine").fit_predict(coords_rad)
    sites["cluster_id"] = labels.astype(int)
    out = df.merge(sites[["site_id", "cluster_id"]], on="site_id", how="left")
    centroids = sites.groupby("cluster_id", as_index=False)[["latitude", "longitude"]].mean()
    return out, centroids


def add_less_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["month"] = out["date"].dt.month
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    out["age_sq"] = out["age_in_months"] ** 2
    out["log_age"] = np.log1p(out["age_in_months"])
    out["is_early"] = (out["age_in_months"] <= 6).astype(int)
    out["is_growth"] = ((out["age_in_months"] > 6) & (out["age_in_months"] <= 18)).astype(int)
    out["is_mature"] = (out["age_in_months"] > 18).astype(int)
    stats = (
        out.groupby(["cluster_id", "month"], as_index=False)["monthly_volume"]
        .agg(
            cluster_month_avg="mean",
            cluster_month_std="std",
            cluster_month_mean="mean",
            cluster_month_median="median",
            cluster_month_p25=lambda s: float(np.nanpercentile(s, 25)),
            cluster_month_p75=lambda s: float(np.nanpercentile(s, 75)),
        )
    )
    out = out.merge(stats, on=["cluster_id", "month"], how="left")
    age_curve = (
        out.groupby(["cluster_id", "age_in_months"], as_index=False)["monthly_volume"]
        .mean()
        .rename(columns={"monthly_volume": "cluster_age_avg"})
    )
    out = out.merge(age_curve, on=["cluster_id", "age_in_months"], how="left")
    out["cluster_growth_rate"] = (
        out.sort_values(["cluster_id", "date"])
        .groupby("cluster_id")["cluster_month_avg"]
        .pct_change()
        .replace([np.inf, -np.inf], np.nan)
    )
    out["lat_lon_interaction"] = out["latitude"] * out["longitude"]
    out["cluster_month_avg"] = out["cluster_month_avg"].replace(0, np.nan)
    out["target_multiplier"] = out["monthly_volume"] / (out["cluster_month_avg"] + 1e-6)
    out["cluster_month_std"] = out["cluster_month_std"].fillna(out["cluster_month_std"].median())
    out["cluster_growth_rate"] = out["cluster_growth_rate"].fillna(0.0)
    out["age_saturation"] = np.tanh(out["age_in_months"] / 12.0)
    out["growth_velocity"] = out["cluster_age_avg"] / (out["age_in_months"] + 1)
    return out


def add_more_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["month"] = out["date"].dt.month
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    out["age_sq"] = out["age_in_months"] ** 2
    out["log_age"] = np.log1p(out["age_in_months"])
    out["is_early"] = (out["age_in_months"] <= 6).astype(int)
    out["is_growth"] = ((out["age_in_months"] > 6) & (out["age_in_months"] <= 18)).astype(int)
    out["is_mature"] = (out["age_in_months"] > 18).astype(int)
    stats = (
        out.groupby(["cluster_id", "month"], as_index=False)["monthly_volume"]
        .agg(
            cluster_daily_avg="mean",
            cluster_std="std",
            cluster_month_avg="mean",
            cluster_month_median="median",
            cluster_month_p25=lambda s: float(np.nanpercentile(s, 25)),
            cluster_month_p75=lambda s: float(np.nanpercentile(s, 75)),
        )
    )
    out = out.merge(stats, on=["cluster_id", "month"], how="left")
    age_curve = (
        out.groupby(["cluster_id", "age_in_months"], as_index=False)["monthly_volume"]
        .mean()
        .rename(columns={"monthly_volume": "cluster_age_avg"})
    )
    out = out.merge(age_curve, on=["cluster_id", "age_in_months"], how="left")
    out["cluster_growth_rate"] = (
        out.sort_values(["cluster_id", "date"])
        .groupby("cluster_id")["cluster_month_avg"]
        .pct_change()
        .replace([np.inf, -np.inf], np.nan)
    )
    out["lat_lon_interaction"] = out["latitude"] * out["longitude"]
    out["cluster_month_avg"] = out["cluster_month_avg"].replace(0, np.nan)
    out["target_multiplier"] = out["monthly_volume"] / (out["cluster_month_avg"] + 1e-6)
    out["cluster_std"] = out["cluster_std"].fillna(out["cluster_std"].median())
    out["cluster_growth_rate"] = out["cluster_growth_rate"].fillna(0.0)
    out["age_saturation"] = np.tanh(out["age_in_months"] / 12.0)
    out["growth_velocity"] = out["cluster_age_avg"] / (out["age_in_months"] + 1)
    return out


def _site_behavior_stats(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("site_id", as_index=False)
        .agg(
            site_avg_volume=("monthly_volume", "mean"),
            site_peak=("monthly_volume", "max"),
            site_std=("monthly_volume", "std"),
        )
        .fillna(0.0)
    )


def add_site_type_feature(df: pd.DataFrame, train_cutoff: str = "2025-01-01") -> pd.DataFrame:
    out = df.copy()
    tr = out[out["date"] < pd.Timestamp(train_cutoff)].copy()
    stats = _site_behavior_stats(tr)
    n_types = 8
    if len(stats) >= n_types:
        km = KMeans(n_clusters=n_types, random_state=42, n_init=10)
        stats["site_type"] = km.fit_predict(stats[["site_avg_volume", "site_peak", "site_std"]]).astype(str)
    else:
        stats["site_type"] = "0"
    out = out.merge(stats[["site_id", "site_type"]], on="site_id", how="left")
    cluster_mode = (
        out[out["date"] < pd.Timestamp(train_cutoff)]
        .groupby("cluster_id")["site_type"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "0")
        .to_dict()
    )
    out["site_type"] = out["site_type"].fillna(out["cluster_id"].map(cluster_mode).fillna("0")).astype(str)
    return out


def fit_site_type_classifier(less_train: pd.DataFrame) -> SiteTypeModel:
    st = _site_behavior_stats(less_train)
    km = KMeans(n_clusters=8, random_state=42, n_init=10) if len(st) >= 8 else KMeans(n_clusters=max(2, len(st) // 3), random_state=42, n_init=10)
    st["site_type_true"] = km.fit_predict(st[["site_avg_volume", "site_peak", "site_std"]]).astype(str)
    labeled = less_train.merge(
        st[["site_id", "site_avg_volume", "site_peak", "site_std", "site_type_true"]],
        on="site_id",
        how="left",
    )
    cluster_density = labeled.groupby("cluster_id")["site_id"].transform("nunique")
    cluster_avg_vol = labeled.groupby("cluster_id")["monthly_volume"].transform("mean")
    X = labeled[["cluster_id", "latitude", "longitude", "site_avg_volume", "site_peak", "site_std"]].copy()
    X["cluster_density"] = cluster_density
    X["cluster_avg_volume"] = cluster_avg_vol
    X["cluster_id"] = X["cluster_id"].astype("category")
    y = labeled["site_type_true"].astype(str)
    X = X.fillna(X.median(numeric_only=True))
    clf = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.08,
        num_leaves=31,
        random_state=42,
        verbose=-1,
    )
    clf.fit(X, y, categorical_feature=["cluster_id"])
    classes = [str(c) for c in clf.classes_]
    return SiteTypeModel(clf=clf, classes=classes)


def infer_site_type_proba(df: pd.DataFrame, stm: SiteTypeModel) -> tuple[np.ndarray, np.ndarray]:
    tmp = df.copy()
    if "site_id" not in tmp.columns:
        tmp["site_id"] = "_synthetic_"
    tmp["cluster_density"] = tmp.groupby("cluster_id")["site_id"].transform("nunique")
    tmp["cluster_avg_volume"] = tmp.groupby("cluster_id")["monthly_volume"].transform("mean")
    st = _site_behavior_stats(tmp)
    tmp = tmp.merge(st[["site_id", "site_avg_volume", "site_peak", "site_std"]], on="site_id", how="left")
    med = float(tmp["site_avg_volume"].median())
    if not np.isfinite(med):
        med = 0.0
    X = tmp[["cluster_id", "latitude", "longitude"]].copy()
    X["site_avg_volume"] = tmp["site_avg_volume"].fillna(med)
    X["site_peak"] = tmp["site_peak"].fillna(0.0)
    X["site_std"] = tmp["site_std"].fillna(0.0)
    X["cluster_density"] = tmp["cluster_density"]
    X["cluster_avg_volume"] = tmp["cluster_avg_volume"]
    X["cluster_id"] = X["cluster_id"].astype("category")
    X = X.fillna(X.median(numeric_only=True))
    proba = stm.clf.predict_proba(X)
    pred_idx = np.argmax(proba, axis=1)
    pred_type = np.array([stm.classes[i] for i in pred_idx])
    return proba, pred_type


def _predict_lgbm_volume_probabilistic_site_type(
    bundle: LGBMBundle,
    df: pd.DataFrame,
    stm: SiteTypeModel,
    proba: np.ndarray,
    top_k: int = 3,
) -> np.ndarray:
    """Blend predictions across top-k site types by classifier probability."""
    out = np.zeros(len(df), dtype=float)
    top_idx = np.argsort(-proba, axis=1)[:, :top_k]
    for i in range(len(df)):
        row = df.iloc[[i]].copy()
        wsum = 0.0
        acc = 0.0
        for k in range(top_k):
            j = int(top_idx[i, k])
            w = float(proba[i, j])
            if w <= 0:
                continue
            st = stm.classes[j]
            row["site_type"] = st
            row["site_type"] = row["site_type"].astype("category")
            acc += w * float(_predict_lgbm_volume(bundle, row)[0])
            wsum += w
        out[i] = acc / wsum if wsum > 0 else float(_predict_lgbm_volume(bundle, df.iloc[[i]])[0])
    return out


def _predict_less_volume_mixed(
    pair: LessModelPair,
    df: pd.DataFrame,
    stm: Optional[SiteTypeModel],
    proba: Optional[np.ndarray],
) -> np.ndarray:
    out = np.zeros(len(df), dtype=float)
    early_m = df["age_in_months"] <= 6
    if early_m.any():
        if stm is not None and proba is not None:
            out[early_m.to_numpy()] = _predict_lgbm_volume_probabilistic_site_type(
                pair.early, df.loc[early_m], stm, proba[early_m.to_numpy()]
            )
        else:
            out[early_m.to_numpy()] = _predict_lgbm_volume(pair.early, df.loc[early_m])
    if (~early_m).any():
        if stm is not None and proba is not None:
            out[(~early_m).to_numpy()] = _predict_lgbm_volume_probabilistic_site_type(
                pair.main, df.loc[~early_m], stm, proba[(~early_m).to_numpy()]
            )
        else:
            out[(~early_m).to_numpy()] = _predict_lgbm_volume(pair.main, df.loc[~early_m])
    return out


def _fit_lgbm_multiplier(df: pd.DataFrame, feature_cols: list[str], cat_cols: list[str]) -> LGBMBundle:
    work = df.copy()
    X = work[feature_cols].copy()
    y = work["target_multiplier"].astype(float)
    for c in cat_cols:
        X[c] = X[c].astype("category")
    fill_vals = X.median(numeric_only=True)
    X = X.fillna(fill_vals)
    model = LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=30,
        random_state=42,
        verbose=-1,
    )
    model.fit(X, y, categorical_feature=cat_cols)
    return LGBMBundle(model=model, feature_columns=feature_cols, categorical_columns=cat_cols, target_mode="multiplier")


def _predict_lgbm_volume(bundle: LGBMBundle, df: pd.DataFrame) -> np.ndarray:
    X = df[bundle.feature_columns].copy()
    for c in bundle.categorical_columns:
        X[c] = X[c].astype("category")
    fill_vals = X.median(numeric_only=True)
    X = X.fillna(fill_vals)
    pred_mult = bundle.model.predict(X)
    return pred_mult * (df["cluster_month_avg"].to_numpy(dtype=float) + 1e-6)


def _raw_p10_p90_from_std(p50: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return p50 - 1.28 * std, p50 + 1.28 * std


def _calibrate_interval_width(
    p50: np.ndarray,
    p10: np.ndarray,
    p90: np.ndarray,
    y_true: np.ndarray,
    target_coverage: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Scale (p90-p10) so empirical coverage moves toward target_coverage."""
    width = np.maximum(p90 - p10, 1e-6)
    inside = ((y_true >= p10) & (y_true <= p90)).mean()
    current = float(inside) if np.isfinite(inside) else 0.5
    scale = float(target_coverage / max(current, 1e-6))
    half = 0.5 * width * scale
    return p50 - half, p50 + half, current, scale


def _wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true).sum()
    if denom <= 1e-9:
        return float("nan")
    return float(np.abs(y_true - y_pred).sum() / denom)


def fit_cluster_ts_models(more_train: pd.DataFrame, horizon: int = 36) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    grouped = more_train.groupby(["cluster_id", "date"], as_index=False)["monthly_volume"].median()
    for cid, block in grouped.groupby("cluster_id"):
        s = block.sort_values("date")["monthly_volume"].astype(float).to_numpy()
        if len(s) < 6:
            out[int(cid)] = np.array([float(np.median(s)) if len(s) else 0.0] * horizon)
            continue
        try:
            model = ARIMA(s, order=(1, 1, 1))
            fit = model.fit()
            fc = fit.forecast(steps=horizon)
            out[int(cid)] = np.asarray(fc, dtype=float)
        except Exception:
            out[int(cid)] = np.array([float(np.median(s))] * horizon)
    return out


def _nearest_cluster(lat: float, lon: float, centroids: pd.DataFrame) -> int:
    d2 = (centroids["latitude"] - lat) ** 2 + (centroids["longitude"] - lon) ** 2
    return int(centroids.iloc[int(d2.idxmin())]["cluster_id"])


def forecast_new_site_60m(
    lat: float,
    lon: float,
    less_pair: LessModelPair,
    more_bundle: LGBMBundle,
    less_feat: pd.DataFrame,
    more_feat: pd.DataFrame,
    less_centroids: pd.DataFrame,
    more_centroids: pd.DataFrame,
    cluster_ts: Dict[int, np.ndarray],
    stm: Optional[SiteTypeModel] = None,
    interval_width_scale: float = 1.0,
    start_date: str = "2026-01-01",
) -> pd.DataFrame:
    less_cluster = _nearest_cluster(lat, lon, less_centroids)
    more_cluster = _nearest_cluster(lat, lon, more_centroids)
    dates = pd.date_range(start=pd.Timestamp(start_date), periods=60, freq="MS")
    out = pd.DataFrame({"date": dates, "age_in_months": np.arange(1, 61), "latitude": lat, "longitude": lon})
    out["month"] = out["date"].dt.month
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    out["less_cluster_id"] = less_cluster
    out["more_cluster_id"] = more_cluster

    less_stats = less_feat.groupby(["cluster_id", "month"], as_index=False)[
        [
            "cluster_month_avg",
            "cluster_month_std",
            "cluster_month_mean",
            "cluster_month_median",
            "cluster_month_p25",
            "cluster_month_p75",
        ]
    ].mean().rename(
        columns={
            "cluster_id": "less_cluster_id",
            "cluster_month_avg": "less_cluster_month_avg",
            "cluster_month_std": "less_cluster_month_std",
            "cluster_month_mean": "less_cluster_month_mean",
            "cluster_month_median": "less_cluster_month_median",
            "cluster_month_p25": "less_cluster_month_p25",
            "cluster_month_p75": "less_cluster_month_p75",
        }
    )
    more_stats = (
        more_feat.groupby(["cluster_id", "month"], as_index=False)[
            [
                "cluster_daily_avg",
                "cluster_std",
                "cluster_month_avg",
                "cluster_month_median",
                "cluster_month_p25",
                "cluster_month_p75",
            ]
        ]
        .mean()
        .rename(
            columns={
                "cluster_id": "more_cluster_id",
                "cluster_daily_avg": "more_cluster_daily_avg",
                "cluster_std": "more_cluster_std",
                "cluster_month_avg": "more_cluster_month_avg",
                "cluster_month_median": "more_cluster_month_median",
                "cluster_month_p25": "more_cluster_month_p25",
                "cluster_month_p75": "more_cluster_month_p75",
            }
        )
    )
    out = out.merge(less_stats, on=["less_cluster_id", "month"], how="left")
    out = out.merge(more_stats, on=["more_cluster_id", "month"], how="left")
    out["cluster_id"] = out["less_cluster_id"]
    out["site_id"] = "_forecast_new_"
    out["monthly_volume"] = out["less_cluster_month_avg"].fillna(less_feat["cluster_month_avg"].median())
    out["cluster_growth_rate"] = out["less_cluster_month_avg"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    age_curve_less = (
        less_feat.groupby(["cluster_id", "age_in_months"], as_index=False)["cluster_age_avg"]
        .mean()
        .rename(columns={"cluster_id": "less_cluster_id"})
    )
    out = out.merge(age_curve_less, on=["less_cluster_id", "age_in_months"], how="left")
    out["cluster_age_avg"] = out["cluster_age_avg"].fillna(out["less_cluster_month_avg"])
    out["age_sq"] = out["age_in_months"] ** 2
    out["log_age"] = np.log1p(out["age_in_months"])
    out["is_early"] = (out["age_in_months"] <= 6).astype(int)
    out["is_growth"] = ((out["age_in_months"] > 6) & (out["age_in_months"] <= 18)).astype(int)
    out["is_mature"] = (out["age_in_months"] > 18).astype(int)
    out["lat_lon_interaction"] = out["latitude"] * out["longitude"]
    out["age_saturation"] = np.tanh(out["age_in_months"] / 12.0)
    out["growth_velocity"] = out["cluster_age_avg"] / (out["age_in_months"] + 1)

    less_rows = out[out["age_in_months"] <= 24].copy()
    less_rows["cluster_id"] = less_rows["less_cluster_id"]
    less_rows["cluster_month_avg"] = less_rows["less_cluster_month_avg"].fillna(less_feat["cluster_month_avg"].median())
    less_rows["cluster_month_std"] = less_rows["less_cluster_month_std"]
    less_rows["cluster_month_mean"] = less_rows["less_cluster_month_mean"]
    less_rows["cluster_month_median"] = less_rows["less_cluster_month_median"]
    less_rows["cluster_month_p25"] = less_rows["less_cluster_month_p25"]
    less_rows["cluster_month_p75"] = less_rows["less_cluster_month_p75"]
    mode_st = (
        less_feat[less_feat["cluster_id"] == less_cluster]["site_type"].mode().iloc[0]
        if len(less_feat[less_feat["cluster_id"] == less_cluster])
        else "0"
    )
    less_rows["site_type"] = mode_st
    less_rows["site_type"] = less_rows["site_type"].astype(str)
    proba_less: Optional[np.ndarray] = None
    if stm is not None:
        proba_less, pred_st = infer_site_type_proba(less_rows.assign(monthly_volume=less_rows["cluster_month_avg"]), stm)
        less_rows["site_type"] = pred_st.astype(str)
    less_pred = _predict_less_volume_mixed(less_pair, less_rows, stm, proba_less)
    out.loc[out["age_in_months"] <= 24, "p50"] = less_pred

    mature_rows = out[out["age_in_months"] > 24].copy()
    mature_rows = mature_rows.drop(columns=["cluster_age_avg", "growth_velocity"], errors="ignore")
    mature_rows["cluster_id"] = mature_rows["more_cluster_id"]
    mature_rows["cluster_month_avg"] = mature_rows["more_cluster_month_avg"].fillna(more_feat["cluster_month_avg"].median())
    mature_rows["cluster_daily_avg"] = mature_rows["more_cluster_daily_avg"]
    mature_rows["cluster_std"] = mature_rows["more_cluster_std"]
    mature_rows["cluster_month_median"] = mature_rows["more_cluster_month_median"]
    mature_rows["cluster_month_p25"] = mature_rows["more_cluster_month_p25"]
    mature_rows["cluster_month_p75"] = mature_rows["more_cluster_month_p75"]
    age_curve_more = (
        more_feat.groupby(["cluster_id", "age_in_months"], as_index=False)["cluster_age_avg"]
        .mean()
        .rename(columns={"cluster_id": "more_cluster_id"})
    )
    mature_rows = mature_rows.merge(age_curve_more, on=["more_cluster_id", "age_in_months"], how="left")
    mature_rows["cluster_age_avg"] = mature_rows["cluster_age_avg"].fillna(mature_rows["more_cluster_month_avg"])
    mature_rows["age_saturation"] = np.tanh(mature_rows["age_in_months"] / 12.0)
    mature_rows["growth_velocity"] = mature_rows["cluster_age_avg"] / (mature_rows["age_in_months"] + 1)
    mature_rows["site_type"] = (
        more_feat[more_feat["cluster_id"] == more_cluster]["site_type"].mode().iloc[0]
        if len(more_feat[more_feat["cluster_id"] == more_cluster])
        else "0"
    )
    baseline = _predict_lgbm_volume(
        more_bundle,
        mature_rows,
    )
    ts_fc = cluster_ts.get(int(more_cluster), np.array([np.nan] * 36))
    ts_fc = ts_fc[: len(mature_rows)]
    anchor = float(np.nanmean(out.loc[out["age_in_months"].between(22, 24), "p50"]))
    cluster_recent = float(np.nanmean(more_feat[(more_feat["cluster_id"] == more_cluster) & (more_feat["date"] < pd.Timestamp("2025-01-01"))]["monthly_volume"]))
    if not np.isfinite(cluster_recent) or abs(cluster_recent) < 1e-6:
        cluster_recent = 1.0
    ts_scale = anchor / cluster_recent if np.isfinite(anchor) else 1.0
    ma = mature_rows["age_in_months"].to_numpy(dtype=float)
    t = np.clip((ma - 25.0) / (48.0 - 25.0), 0.0, 1.0)
    w_baseline = 0.70 - (0.70 - 0.32) * t
    stitched = w_baseline * baseline + (1.0 - w_baseline) * (ts_fc * ts_scale)
    out.loc[out["age_in_months"] > 24, "p50"] = stitched

    out["std_use"] = np.where(out["age_in_months"] <= 24, out["less_cluster_month_std"], out["more_cluster_std"])
    out["std_use"] = out["std_use"].fillna(np.nanmedian(out["std_use"]))
    half = 1.28 * out["std_use"].to_numpy(dtype=float) * float(interval_width_scale)
    out["p10"] = (out["p50"] - half).clip(lower=0)
    out["p90"] = (out["p50"] + half).clip(lower=0)
    return out


def _model2_artifact_path(model_save_dir: Path) -> Path:
    return model_save_dir / "model2_artifacts.joblib"


def _load_artifacts(model_save_dir: Path) -> Optional[dict]:
    path = _model2_artifact_path(model_save_dir)
    if not path.exists():
        return None
    return joblib.load(path)


def _save_artifacts(model_save_dir: Path, payload: dict) -> Path:
    model_save_dir.mkdir(parents=True, exist_ok=True)
    path = _model2_artifact_path(model_save_dir)
    joblib.dump(payload, path)
    return path


def run_pipeline(
    data_dir: Path,
    out_dir: Path,
    model_save_dir: Path,
    retrain: bool = False,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    less = prepare_less(data_dir / "less_than-2yrs.csv")
    more = prepare_more(data_dir / "more_than-2yrs.csv")

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
    saved = None if retrain else _load_artifacts(model_save_dir)
    if saved is None:
        less, less_cent = assign_dbscan_clusters(less, eps_km=12.0)
        more, more_cent = assign_dbscan_clusters(more, eps_km=12.0)
        less_f = add_site_type_feature(add_less_features(less))
        more_f = add_site_type_feature(add_more_features(more))

        less_train = less_f[less_f["date"] < pd.Timestamp("2025-01-01")].copy()
        less_test = less_f[less_f["date"] >= pd.Timestamp("2025-01-01")].copy()
        more_train = more_f[more_f["date"] < pd.Timestamp("2025-01-01")].copy()
        more_test = more_f[more_f["date"] >= pd.Timestamp("2025-01-01")].copy()

        less_features = [c for c in less_features if c in less_train.columns]
        more_features = [c for c in more_features if c in more_train.columns]

        stm = fit_site_type_classifier(less_train)
        less_train_early = less_train[less_train["age_in_months"] <= 6].copy()
        less_train_main = less_train[less_train["age_in_months"] > 6].copy()
        if len(less_train_early) < 80:
            less_train_early = less_train.copy()
        if len(less_train_main) < 80:
            less_train_main = less_train.copy()
        less_early = _fit_lgbm_multiplier(less_train_early, less_features, cat_cols=["cluster_id", "site_type"])
        less_main = _fit_lgbm_multiplier(less_train_main, less_features, cat_cols=["cluster_id", "site_type"])
        less_pair = LessModelPair(early=less_early, main=less_main)
        more_bundle = _fit_lgbm_multiplier(more_train, more_features, cat_cols=["cluster_id", "site_type"])

        less_std_arr = less_test["cluster_month_std"].fillna(less_test["cluster_month_std"].median()).to_numpy(dtype=float)
        more_std_arr = more_test["cluster_std"].fillna(more_test["cluster_std"].median()).to_numpy(dtype=float)
        _, pred_site_less = infer_site_type_proba(less_test, stm)
        less_test_typed = less_test.copy()
        less_test_typed["site_type"] = pred_site_less.astype(str)
        less_pred = _predict_less_volume_mixed(less_pair, less_test_typed, None, None)
        more_pred = _predict_lgbm_volume(more_bundle, more_test)
        less_y = less_test["monthly_volume"].to_numpy(dtype=float)
        more_y = more_test["monthly_volume"].to_numpy(dtype=float)
        less_p10_raw, less_p90_raw = _raw_p10_p90_from_std(less_pred, less_std_arr)
        more_p10_raw, more_p90_raw = _raw_p10_p90_from_std(more_pred, more_std_arr)
        all_y = np.concatenate([less_y, more_y])
        all_p50 = np.concatenate([less_pred, more_pred])
        all_p10 = np.concatenate([less_p10_raw, more_p10_raw])
        all_p90 = np.concatenate([less_p90_raw, more_p90_raw])
        coverage_raw = float(
            (((all_y >= all_p10) & (all_y <= all_p90)).sum()) / max(len(all_y), 1)
        )
        target_cov = 0.8
        all_p10_cal = np.asarray(all_p10, dtype=float)
        all_p90_cal = np.asarray(all_p90, dtype=float)
        interval_width_scale = 1.0
        for _ in range(8):
            cur = float(((all_y >= all_p10_cal) & (all_y <= all_p90_cal)).sum() / max(len(all_y), 1))
            if cur <= target_cov + 0.04:
                break
            all_p10_cal, all_p90_cal, _, sc = _calibrate_interval_width(
                all_p50, all_p10_cal, all_p90_cal, all_y, target_coverage=target_cov
            )
            interval_width_scale *= sc

        cluster_ts = fit_cluster_ts_models(more_train, horizon=36)
        _save_artifacts(
            model_save_dir,
            {
                "less_pair": less_pair,
                "more_bundle": more_bundle,
                "stm": stm,
                "cluster_ts": cluster_ts,
                "less_cent": less_cent,
                "more_cent": more_cent,
                "interval_width_scale": interval_width_scale,
                "less_features": less_features,
                "more_features": more_features,
            },
        )
    else:
        less, less_cent = assign_dbscan_clusters(less, eps_km=12.0)
        more, more_cent = assign_dbscan_clusters(more, eps_km=12.0)
        less_f = add_site_type_feature(add_less_features(less))
        more_f = add_site_type_feature(add_more_features(more))
        less_train = less_f[less_f["date"] < pd.Timestamp("2025-01-01")].copy()
        less_test = less_f[less_f["date"] >= pd.Timestamp("2025-01-01")].copy()
        more_train = more_f[more_f["date"] < pd.Timestamp("2025-01-01")].copy()
        more_test = more_f[more_f["date"] >= pd.Timestamp("2025-01-01")].copy()
        less_pair = saved["less_pair"]
        more_bundle = saved["more_bundle"]
        stm = saved["stm"]
        cluster_ts = saved["cluster_ts"]
        less_cent = saved["less_cent"]
        more_cent = saved["more_cent"]
        interval_width_scale = float(saved.get("interval_width_scale", 1.0))

    _, pred_site_less = infer_site_type_proba(less_test, stm)
    less_test_typed = less_test.copy()
    less_test_typed["site_type"] = pred_site_less.astype(str)
    less_pred = _predict_less_volume_mixed(less_pair, less_test_typed, None, None)
    more_pred = _predict_lgbm_volume(more_bundle, more_test)
    less_y = less_test["monthly_volume"].to_numpy(dtype=float)
    more_y = more_test["monthly_volume"].to_numpy(dtype=float)

    less_mae = float(mean_absolute_error(less_y, less_pred))
    more_mae = float(mean_absolute_error(more_y, more_pred))
    less_wmape = _wmape(less_y, less_pred)
    more_wmape = _wmape(more_y, more_pred)

    less_std_arr = less_test["cluster_month_std"].fillna(less_test["cluster_month_std"].median()).to_numpy(dtype=float)
    more_std_arr = more_test["cluster_std"].fillna(more_test["cluster_std"].median()).to_numpy(dtype=float)
    less_p10_raw, less_p90_raw = _raw_p10_p90_from_std(less_pred, less_std_arr)
    more_p10_raw, more_p90_raw = _raw_p10_p90_from_std(more_pred, more_std_arr)
    all_y = np.concatenate([less_y, more_y])
    all_p50 = np.concatenate([less_pred, more_pred])
    all_p10 = np.concatenate([less_p10_raw, more_p10_raw])
    all_p90 = np.concatenate([less_p90_raw, more_p90_raw])
    coverage_raw = float(
        (((all_y >= all_p10) & (all_y <= all_p90)).sum()) / max(len(all_y), 1)
    )
    target_cov = 0.8
    all_p10_cal = np.asarray(all_p10, dtype=float)
    all_p90_cal = np.asarray(all_p90, dtype=float)
    current_cov = coverage_raw
    if saved is not None:
        all_p10_cal = all_p50 - 0.5 * (all_p90 - all_p10) * interval_width_scale
        all_p90_cal = all_p50 + 0.5 * (all_p90 - all_p10) * interval_width_scale
    coverage_calibrated = float(((all_y >= all_p10_cal) & (all_y <= all_p90_cal)).sum() / max(len(all_y), 1))

    # Cold-start: hold out 20% sites in <2y, evaluate first 6 months.
    sites = pd.Series(less_f["site_id"].unique()).sort_values().reset_index(drop=True)
    hold_n = max(1, int(0.2 * len(sites)))
    hold_sites = set(sites.tail(hold_n).tolist())
    cold_test = less_f[less_f["site_id"].isin(hold_sites) & (less_f["age_in_months"] <= 6)].copy()
    cold_train = less_f[~less_f["site_id"].isin(hold_sites) & (less_f["date"] < pd.Timestamp("2025-01-01"))].copy()
    cold_early_df = cold_train[cold_train["age_in_months"] <= 6].copy()
    cold_main_df = cold_train[cold_train["age_in_months"] > 6].copy()
    if len(cold_early_df) < 50:
        cold_early_df = cold_train.copy()
    if len(cold_main_df) < 50:
        cold_main_df = cold_train.copy()
    cold_pair = LessModelPair(
        early=_fit_lgbm_multiplier(cold_early_df, less_features, cat_cols=["cluster_id", "site_type"]),
        main=_fit_lgbm_multiplier(cold_main_df, less_features, cat_cols=["cluster_id", "site_type"]),
    )
    _, pred_site_cold = infer_site_type_proba(cold_test, stm)
    cold_test_typed = cold_test.copy()
    cold_test_typed["site_type"] = pred_site_cold.astype(str)
    cold_pred = _predict_less_volume_mixed(cold_pair, cold_test_typed, None, None)
    cold_mae = float(mean_absolute_error(cold_test["monthly_volume"].to_numpy(dtype=float), cold_pred)) if len(cold_test) else float("nan")

    sample_fc = forecast_new_site_60m(
        lat=float(less_f["latitude"].iloc[0]),
        lon=float(less_f["longitude"].iloc[0]),
        less_pair=less_pair,
        more_bundle=more_bundle,
        less_feat=less_f,
        more_feat=more_f,
        less_centroids=less_cent,
        more_centroids=more_cent,
        cluster_ts=cluster_ts,
        stm=stm,
        interval_width_scale=interval_width_scale,
        start_date="2026-01-01",
    )
    sample_fc.to_csv(out_dir / "model2_sample_forecast_60m.csv", index=False)
    less_cent.to_csv(out_dir / "model2_less_cluster_centroids.csv", index=False)
    more_cent.to_csv(out_dir / "model2_more_cluster_centroids.csv", index=False)

    metrics = {
        "data_dir": str(data_dir),
        "split": {"train_before": "2025-01-01", "test_from": "2025-01-01"},
        "less_than_2y": {"mae": less_mae, "wmape": less_wmape, "rows_test": int(len(less_test))},
        "more_than_2y": {"mae": more_mae, "wmape": more_wmape, "rows_test": int(len(more_test))},
        "uncertainty_calibration": {
            "target_coverage": target_cov,
            "raw_coverage_p10_p90": coverage_raw,
            "calibrated_coverage_p10_p90": coverage_calibrated,
            "interval_width_scale": interval_width_scale,
            "empirical_coverage_before_scale": coverage_raw,
        },
        "cold_start_holdout_first6": {"mae": cold_mae, "rows_test": int(len(cold_test))},
        "artifacts": {
            "saved_models": str(_model2_artifact_path(model_save_dir)),
            "sample_forecast": str(out_dir / "model2_sample_forecast_60m.csv"),
            "less_cluster_centroids": str(out_dir / "model2_less_cluster_centroids.csv"),
            "more_cluster_centroids": str(out_dir / "model2_more_cluster_centroids.csv"),
        },
    }
    (out_dir / "model2_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Model 2 cohort-separated forecast pipeline (data_2).")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("zeta_modelling/data_2"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("zeta_modelling/data_2/model_2_outputs"),
    )
    parser.add_argument(
        "--model-save-dir",
        type=Path,
        default=Path("zeta_modelling/data_2/model_saves_2"),
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force retraining even if saved models exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_pipeline(args.data_dir, args.out_dir, args.model_save_dir, retrain=args.retrain)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
