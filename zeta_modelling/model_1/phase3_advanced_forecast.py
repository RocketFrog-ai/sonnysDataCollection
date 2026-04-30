from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error


QUANTILES = {"p10": 0.1, "p50": 0.5, "p90": 0.9}


@dataclass
class Artifacts:
    site_type_clf: LGBMClassifier
    site_type_classes: list[str]
    early_models: dict[str, LGBMRegressor]
    main_models: dict[str, LGBMRegressor]
    cluster_month_avg: pd.DataFrame
    cluster_age_avg: pd.DataFrame
    cluster_month_std: pd.DataFrame
    cluster_age_std: pd.DataFrame
    cluster_centroids: pd.DataFrame
    feature_cols: list[str]
    cat_cols: list[str]
    cat_maps: dict[str, dict[str, int]]


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values(["site_id", "date"]).reset_index(drop=True)
    out["cluster_id"] = out["cluster_id"].astype(str)
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


def build_cluster_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    cma = (
        df.groupby(["cluster_id", "month"], as_index=False)["monthly_volume"]
        .mean()
        .rename(columns={"monthly_volume": "cluster_month_avg"})
    )
    cas = (
        df.groupby(["cluster_id", "age_in_months"], as_index=False)["monthly_volume"]
        .mean()
        .rename(columns={"monthly_volume": "cluster_age_avg"})
    )
    cms = (
        df.groupby(["cluster_id", "month"], as_index=False)["monthly_volume"]
        .std()
        .fillna(0.0)
        .rename(columns={"monthly_volume": "cluster_month_std"})
    )
    age_std = (
        df.groupby(["cluster_id", "age_in_months"], as_index=False)["monthly_volume"]
        .std()
        .fillna(0.0)
        .rename(columns={"monthly_volume": "cluster_age_std"})
    )
    return {"cma": cma, "cas": cas, "cms": cms, "age_std": age_std}


def attach_cluster_features(df: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = df.copy()
    out = out.merge(tables["cma"], on=["cluster_id", "month"], how="left")
    out = out.merge(tables["cas"], on=["cluster_id", "age_in_months"], how="left")
    out = out.merge(tables["cms"], on=["cluster_id", "month"], how="left")
    out = out.merge(tables["age_std"], on=["cluster_id", "age_in_months"], how="left")
    out["cluster_month_avg"] = out["cluster_month_avg"].replace(0, np.nan)
    out["cluster_growth_curve"] = out["cluster_age_avg"] - out.groupby("cluster_id")["cluster_age_avg"].shift(1)
    out["cluster_growth_curve"] = out["cluster_growth_curve"].fillna(0.0)
    out["growth_velocity"] = out["cluster_age_avg"] / (out["age_in_months"] + 1)
    out["distance_from_peak"] = out["cluster_age_avg"] - out["cluster_month_avg"]
    return out


def build_site_type_labels(train_df: pd.DataFrame, n_types: int = 6) -> pd.DataFrame:
    stats = (
        train_df.groupby("site_id", as_index=False)
        .agg(
            site_avg_volume=("monthly_volume", "mean"),
            site_peak=("monthly_volume", "max"),
            site_std=("monthly_volume", "std"),
        )
        .fillna(0.0)
    )
    X = stats[["site_avg_volume", "site_peak", "site_std"]]
    km = KMeans(n_clusters=n_types, random_state=42, n_init=10)
    stats["site_type"] = km.fit_predict(X).astype(str)
    return stats[["site_id", "site_type"]]


def fit_site_type_classifier(train_df: pd.DataFrame) -> tuple[LGBMClassifier, list[str]]:
    labels = build_site_type_labels(train_df, n_types=6)
    tr = train_df.merge(labels, on="site_id", how="left")
    cluster_density = tr.groupby("cluster_id")["site_id"].nunique().rename("cluster_density")
    cluster_avg = tr.groupby("cluster_id")["monthly_volume"].mean().rename("cluster_avg_volume")
    tr = tr.join(cluster_density, on="cluster_id").join(cluster_avg, on="cluster_id")

    X = tr[["cluster_id", "latitude", "longitude", "cluster_density", "cluster_avg_volume"]].copy()
    y = tr["site_type"].astype(str)
    X["cluster_id"] = X["cluster_id"].astype("category")
    X = X.fillna(X.median(numeric_only=True))

    clf = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
    )
    clf.fit(X, y, categorical_feature=["cluster_id"])
    return clf, [str(c) for c in clf.classes_]


def infer_site_type_proba(df: pd.DataFrame, clf: LGBMClassifier, classes: list[str]) -> pd.DataFrame:
    tmp = df.copy()
    tmp["cluster_density"] = tmp.groupby("cluster_id")["cluster_id"].transform("count")
    tmp["cluster_avg_volume"] = tmp.groupby("cluster_id")["cluster_month_avg"].transform("mean")
    X = tmp[["cluster_id", "latitude", "longitude", "cluster_density", "cluster_avg_volume"]].copy()
    X["cluster_id"] = X["cluster_id"].astype("category")
    X = X.fillna(X.median(numeric_only=True))

    probs = clf.predict_proba(X)
    out = pd.DataFrame(probs, columns=[f"site_type_prob_{c}" for c in classes], index=df.index)
    top = np.argmax(probs, axis=1)
    out["site_type"] = [classes[i] for i in top]
    return out


def train_quantile_models(
    train_df: pd.DataFrame, feature_cols: list[str]
) -> tuple[dict[str, LGBMRegressor], dict[str, LGBMRegressor]]:
    eps = 1e-6
    tr = train_df.copy()
    tr["target_multiplier"] = tr["monthly_volume"] / (tr["cluster_month_avg"] + eps)
    early = tr[tr["age_in_months"] <= 6].copy()
    main = tr[tr["age_in_months"] > 6].copy()

    def fit_block(block: pd.DataFrame) -> dict[str, LGBMRegressor]:
        X = block[feature_cols].copy()
        y = block["target_multiplier"].astype(float)
        fill_vals = X.median(numeric_only=True)
        X = X.fillna(fill_vals)
        models: dict[str, LGBMRegressor] = {}
        for name, alpha in QUANTILES.items():
            m = LGBMRegressor(
                objective="quantile",
                alpha=alpha,
                n_estimators=1200,
                learning_rate=0.05,
                num_leaves=63,
                max_depth=-1,
                min_child_samples=30,
                random_state=42,
            )
            m.fit(X, y)
            models[name] = m
        return models

    return fit_block(early), fit_block(main)


def build_cat_maps(df: pd.DataFrame, cols: list[str]) -> dict[str, dict[str, int]]:
    maps: dict[str, dict[str, int]] = {}
    for c in cols:
        vals = sorted(df[c].astype(str).fillna("unknown").unique().tolist())
        maps[c] = {v: i for i, v in enumerate(vals)}
    return maps


def apply_cat_maps(df: pd.DataFrame, maps: dict[str, dict[str, int]]) -> pd.DataFrame:
    out = df.copy()
    for c, mapping in maps.items():
        out[f"{c}_code"] = out[c].astype(str).map(mapping).fillna(-1).astype(int)
    return out


def top_k_cluster_weights(lat: float, lon: float, centroids: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    c = centroids.copy()
    lat1 = np.radians(float(lat))
    lon1 = np.radians(float(lon))
    lat2 = np.radians(c["latitude"].astype(float).to_numpy())
    lon2 = np.radians(c["longitude"].astype(float).to_numpy())
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    cc = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    c["distance_km"] = 6371.0 * cc
    top = c.sort_values("distance_km").head(k).copy()
    top["w_raw"] = 1.0 / (top["distance_km"] + 1e-6)
    top["weight"] = top["w_raw"] / top["w_raw"].sum()
    return top[["cluster_id", "distance_km", "weight"]]


def make_future_rows(lat: float, lon: float, months: int, start_date: str) -> pd.DataFrame:
    dates = pd.date_range(start=pd.Timestamp(start_date), periods=months, freq="MS")
    out = pd.DataFrame(
        {
            "date": dates,
            "month": dates.month,
            "age_in_months": np.arange(1, months + 1),
            "real_age_months": np.arange(1, months + 1),
            "latitude": lat,
            "longitude": lon,
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


def predict_for_cluster(
    rows: pd.DataFrame, cluster_id: str, artifacts: Artifacts
) -> pd.DataFrame:
    out = rows.copy()
    out["cluster_id"] = str(cluster_id)
    out = attach_cluster_features(
        out,
        {"cma": artifacts.cluster_month_avg, "cas": artifacts.cluster_age_avg, "cms": artifacts.cluster_month_std, "age_std": artifacts.cluster_age_std},
    )
    proba = infer_site_type_proba(out, artifacts.site_type_clf, artifacts.site_type_classes)
    out = pd.concat([out, proba], axis=1)
    eps = 1e-6

    pred = pd.DataFrame(index=out.index)
    for q in QUANTILES.keys():
        pred[q] = 0.0

    # Probabilistic site type: weighted over top-3 types.
    prob_cols = [c for c in out.columns if c.startswith("site_type_prob_")]
    top3_idx = np.argsort(-out[prob_cols].to_numpy(), axis=1)[:, :3]

    out = apply_cat_maps(out, artifacts.cat_maps)
    X_base = out[artifacts.feature_cols].copy()
    X_base = X_base.fillna(X_base.median(numeric_only=True))
    early_mask = out["age_in_months"] <= 6

    for i in range(len(out)):
        row = X_base.iloc[[i]].copy()
        probs_row = out[prob_cols].iloc[i].to_numpy()
        idxs = top3_idx[i]
        weights = probs_row[idxs]
        wsum = weights.sum()
        if wsum <= 0:
            weights = np.array([1 / len(idxs)] * len(idxs))
        else:
            weights = weights / wsum
        for local_k, idx in enumerate(idxs):
            stype = artifacts.site_type_classes[idx]
            row_mod = row.copy()
            if "site_type_code" in row_mod.columns:
                row_mod["site_type_code"] = artifacts.cat_maps["site_type"].get(stype, -1)
            model_set = artifacts.early_models if bool(early_mask.iloc[i]) else artifacts.main_models
            for q in QUANTILES.keys():
                pred.at[out.index[i], q] += weights[local_k] * model_set[q].predict(row_mod)[0]

    out["pred_p10"] = pred["p10"] * (out["cluster_month_avg"] + eps)
    out["pred_p50"] = pred["p50"] * (out["cluster_month_avg"] + eps)
    out["pred_p90"] = pred["p90"] * (out["cluster_month_avg"] + eps)

    # Early-stage correction from empirical cluster-age ratio.
    corr = 1.0 + np.clip((out["cluster_age_std"] / (out["cluster_age_avg"].abs() + 1e-6)) * 0.10, -0.2, 0.2)
    early = out["age_in_months"] <= 3
    out.loc[early, "pred_p10"] *= 1 / corr.loc[early]
    out.loc[early, "pred_p50"] *= 1.0
    out.loc[early, "pred_p90"] *= corr.loc[early]
    return out


def blend_clusters(rows: pd.DataFrame, cluster_weights: pd.DataFrame, artifacts: Artifacts) -> pd.DataFrame:
    all_preds = []
    for _, c in cluster_weights.iterrows():
        p = predict_for_cluster(rows, str(c["cluster_id"]), artifacts)
        p["cluster_weight"] = c["weight"]
        all_preds.append(p)
    stack = pd.concat(all_preds, ignore_index=True)

    g = stack.groupby("date", as_index=False).apply(
        lambda d: pd.Series(
            {
                "age_in_months": int(d["age_in_months"].iloc[0]),
                "p50": float(np.sum(d["pred_p50"] * d["cluster_weight"])),
                "p10": float(np.sum(d["pred_p10"] * d["cluster_weight"])),
                "p90": float(np.sum(d["pred_p90"] * d["cluster_weight"])),
            }
        ),
        include_groups=False,
    )
    g["volume"] = g["p50"]
    g["low"] = g["p10"]
    g["high"] = g["p90"]
    g["cumulative_volume"] = g["volume"].cumsum()
    return g.sort_values("date").reset_index(drop=True)


def apply_mature_yoy_control(
    forecast: pd.DataFrame,
    *,
    start_year: int = 4,
    min_yoy: float = 0.005,
    max_yoy: float = 0.05,
) -> tuple[pd.DataFrame, dict]:
    """
    When (and only when) late-horizon annual P50 totals are strictly decreasing
    year-on-year from ``start_year`` onward, re-scale those years so each annual
    total lies in [prev * (1+min_yoy), prev * (1+max_yoy)].

    If the raw forecast is flat, up, or mixed in that tail, the dataframe is
    returned unchanged (summary explains skip).
    """
    out = forecast.copy()
    if out.empty or "age_in_months" not in out.columns or "p50" not in out.columns:
        return out, {"enabled": False, "reason": "missing_required_columns"}

    years = ((pd.to_numeric(out["age_in_months"], errors="coerce") - 1) // 12 + 1).astype(int)
    out["_forecast_year"] = years
    year_order = sorted(out["_forecast_year"].dropna().unique().tolist())
    if not year_order or max(year_order) < int(start_year):
        out = out.drop(columns=["_forecast_year"], errors="ignore")
        return out, {"enabled": False, "reason": "insufficient_horizon"}

    # Annual P50 totals by forecast year (1-based).
    totals_by_year: dict[int, float] = {}
    for y in year_order:
        y_mask = out["_forecast_year"] == int(y)
        totals_by_year[int(y)] = float(pd.to_numeric(out.loc[y_mask, "p50"], errors="coerce").fillna(0.0).sum())

    late_years = [y for y in year_order if int(y) >= int(start_year)]
    if len(late_years) < 2:
        out = out.drop(columns=["_forecast_year"], errors="ignore")
        return out, {
            "enabled": False,
            "reason": "skipped_need_at_least_two_late_years",
            "start_year": int(start_year),
            "late_year_totals": {str(y): totals_by_year.get(y, 0.0) for y in late_years},
        }

    late_vals = [totals_by_year[y] for y in late_years]
    strict_monotonic_down = all(late_vals[i] < late_vals[i - 1] for i in range(1, len(late_vals)))
    if not strict_monotonic_down:
        out = out.drop(columns=["_forecast_year"], errors="ignore")
        return out, {
            "enabled": False,
            "reason": "skipped_not_strict_monotonic_down_late_years",
            "start_year": int(start_year),
            "late_year_totals": {str(y): float(totals_by_year[y]) for y in late_years},
        }

    scaled_years: list[int] = []
    year_scale_factors: dict[str, float] = {}
    prev_target_total: float | None = None
    bands: dict[str, dict[str, float]] = {}

    for y in year_order:
        y_mask = out["_forecast_year"] == int(y)
        cur_total = float(pd.to_numeric(out.loc[y_mask, "p50"], errors="coerce").fillna(0.0).sum())
        if prev_target_total is None:
            prev_target_total = cur_total
            continue
        if y < int(start_year):
            prev_target_total = cur_total
            continue
        if cur_total <= 0:
            prev_target_total = max(prev_target_total, 1.0)
            continue

        lower = float(prev_target_total * (1.0 + float(min_yoy)))
        upper = float(prev_target_total * (1.0 + float(max_yoy)))
        target = float(np.clip(cur_total, lower, upper))
        factor = float(target / cur_total) if cur_total > 0 else 1.0
        bands[str(int(y))] = {"lower": lower, "upper": upper, "raw": cur_total, "target": target}

        if abs(factor - 1.0) > 1e-9:
            for c in ("pred_p10", "pred_p50", "pred_p90", "p10", "p50", "p90", "volume", "low", "high"):
                if c in out.columns:
                    out.loc[y_mask, c] = pd.to_numeric(out.loc[y_mask, c], errors="coerce") * factor
            scaled_years.append(int(y))
            year_scale_factors[str(int(y))] = factor

        prev_target_total = target

    if "volume" in out.columns:
        out["cumulative_volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0.0).cumsum()
    out = out.drop(columns=["_forecast_year"], errors="ignore")
    return out, {
        "enabled": True,
        "applied": True,
        "trigger": "strict_monotonic_down_late_years",
        "start_year": int(start_year),
        "min_yoy": float(min_yoy),
        "max_yoy": float(max_yoy),
        "late_year_totals_raw": {str(y): float(totals_by_year[y]) for y in late_years},
        "scaled_years": scaled_years,
        "year_scale_factors": year_scale_factors,
        "year_bands": bands,
    }


def apply_lifecycle_smoothing_control(
    forecast: pd.DataFrame,
    *,
    min_annual_growth: float = 0.005,
    max_annual_growth: float = 0.05,
    year2_max_growth: float = 0.35,
) -> tuple[pd.DataFrame, dict]:
    """
    Enforce a business-realistic upward lifecycle for annual wash volumes.

    Rules (applied year-by-year, each anchored to previous target):
    - Year 2: always >= Year 1 * 1.02 (ramp-up), capped at year2_max_growth.
    - Year 3+: if model already grew vs prior → keep it (capped at max_annual_growth).
              if model declined → scale UP to at least prev * (1 + min_annual_growth).
    - We NEVER let any year go below the prior year; the floor is always +min_annual_growth.
    - Scale is applied proportionally within the year so monthly shape is preserved.
    """
    out = forecast.copy()
    if out.empty or "age_in_months" not in out.columns or "p50" not in out.columns:
        return out, {"enabled": False, "reason": "missing_required_columns"}

    years = ((pd.to_numeric(out["age_in_months"], errors="coerce") - 1) // 12 + 1).astype(int)
    out["_forecast_year"] = years
    year_order = sorted(out["_forecast_year"].dropna().unique().tolist())
    if len(year_order) < 2:
        out = out.drop(columns=["_forecast_year"], errors="ignore")
        return out, {"enabled": False, "reason": "insufficient_horizon"}

    totals_raw: dict[int, float] = {}
    for y in year_order:
        y_mask = out["_forecast_year"] == int(y)
        totals_raw[int(y)] = float(pd.to_numeric(out.loc[y_mask, "p50"], errors="coerce").fillna(0.0).sum())

    prev_target: float | None = None
    scaled_years: list[int] = []
    year_scale_factors: dict[str, float] = {}
    bands: dict[str, dict[str, float]] = {}

    for y in year_order:
        y_mask = out["_forecast_year"] == int(y)
        cur_total = float(pd.to_numeric(out.loc[y_mask, "p50"], errors="coerce").fillna(0.0).sum())

        if prev_target is None:
            prev_target = max(cur_total, 1.0)
            continue
        if cur_total <= 0:
            prev_target = max(prev_target, 1.0)
            continue

        if int(y) == 2:
            # Ramp-up: minimum +2%, max +35%.
            lower = float(prev_target * 1.02)
            upper = float(prev_target * (1.0 + float(year2_max_growth)))
        else:
            # Mature lifecycle: always grow, min_annual_growth to max_annual_growth.
            lower = float(prev_target * (1.0 + float(min_annual_growth)))
            upper = float(prev_target * (1.0 + float(max_annual_growth)))

        # If raw model already landed above lower (grew naturally), respect it
        # up to the upper cap.  If raw fell below lower, lift it.
        target = float(np.clip(cur_total, lower, upper))
        factor = float(target / cur_total)
        bands[str(int(y))] = {
            "lower": lower,
            "upper": upper,
            "raw": cur_total,
            "target": target,
            "yoy_pct": float((target / prev_target - 1.0) * 100),
        }

        if abs(factor - 1.0) > 1e-9:
            for c in ("pred_p10", "pred_p50", "pred_p90", "p10", "p50", "p90", "volume", "low", "high"):
                if c in out.columns:
                    out.loc[y_mask, c] = pd.to_numeric(out.loc[y_mask, c], errors="coerce") * factor
            scaled_years.append(int(y))
            year_scale_factors[str(int(y))] = round(factor, 6)

        prev_target = target

    if "volume" in out.columns:
        out["cumulative_volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0.0).cumsum()

    totals_adj: dict[int, float] = {}
    for y in year_order:
        y_mask = out["_forecast_year"] == int(y)
        totals_adj[int(y)] = float(pd.to_numeric(out.loc[y_mask, "p50"], errors="coerce").fillna(0.0).sum())
    out = out.drop(columns=["_forecast_year"], errors="ignore")

    return out, {
        "enabled": True,
        "applied": len(scaled_years) > 0,
        "min_annual_growth_pct": float(min_annual_growth * 100),
        "max_annual_growth_pct": float(max_annual_growth * 100),
        "scaled_years": scaled_years,
        "year_scale_factors": year_scale_factors,
        "year_bands": bands,
        "annual_totals_raw": {str(k): float(v) for k, v in totals_raw.items()},
        "annual_totals_adjusted": {str(k): float(v) for k, v in totals_adj.items()},
    }


def break_even_from_costs(
    forecast: pd.DataFrame,
    margin_per_wash: float,
    fixed_monthly_cost: float,
    ramp_up_cost: float,
) -> tuple[pd.DataFrame, int | None]:
    out = forecast.copy()
    out["monthly_profit"] = out["volume"] * margin_per_wash - fixed_monthly_cost
    out.loc[out.index[0], "monthly_profit"] -= ramp_up_cost
    out["cumulative_profit"] = out["monthly_profit"].cumsum()
    hit = out[out["cumulative_profit"] > 0]
    month = int(hit["age_in_months"].iloc[0]) if len(hit) else None
    out["break_even_month"] = month
    return out, month


def backtest_predictions_merge(
    df: pd.DataFrame,
    artifacts: Artifacts,
    max_rows: int | None = 3000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Test-period rows with Phase 3 quantile predictions (row-wise cluster assignment)."""
    base = add_base_features(df)
    test = base[base["date"] >= pd.Timestamp("2025-01-01")].copy()
    if max_rows is not None:
        test = test.sample(min(len(test), max_rows), random_state=random_state).copy()
    pred_rows = []
    for cid, part in test.groupby("cluster_id"):
        pred_rows.append(predict_for_cluster(part, str(cid), artifacts))
    preds = pd.concat(pred_rows, ignore_index=True)
    merged = preds[
        ["site_id", "date", "cluster_id", "age_in_months", "monthly_volume", "pred_p50", "pred_p10", "pred_p90"]
    ].copy()
    merged["abs_error"] = (merged["monthly_volume"] - merged["pred_p50"]).abs()
    return merged


def evaluate_backtest(df: pd.DataFrame, artifacts: Artifacts, out_dir: Path) -> dict[str, float | str]:
    merged = backtest_predictions_merge(df, artifacts, max_rows=3000, random_state=42)
    mae = float(mean_absolute_error(merged["monthly_volume"], merged["pred_p50"]))
    rmse = float(np.sqrt(mean_squared_error(merged["monthly_volume"], merged["pred_p50"])))
    coverage = float(((merged["monthly_volume"] >= merged["pred_p10"]) & (merged["monthly_volume"] <= merged["pred_p90"])).mean())

    # Plots
    out_dir.mkdir(parents=True, exist_ok=True)
    rand_sites = merged["site_id"].drop_duplicates().sample(min(4, merged["site_id"].nunique()), random_state=42)
    plt.figure(figsize=(10, 6))
    for sid in rand_sites:
        d = merged[merged["site_id"] == sid].sort_values("date").head(20)
        plt.plot(d["date"], d["monthly_volume"], alpha=0.5)
        plt.plot(d["date"], d["pred_p50"], linestyle="--", alpha=0.6)
    plt.title("Random Sites: Actual vs Predicted")
    plt.tight_layout()
    p1 = out_dir / "backtest_random_sites.png"
    plt.savefig(p1)
    plt.close()

    hi_sites = merged.groupby("site_id", as_index=False)["abs_error"].mean().sort_values("abs_error", ascending=False).head(4)["site_id"]
    plt.figure(figsize=(10, 6))
    for sid in hi_sites:
        d = merged[merged["site_id"] == sid].sort_values("date").head(20)
        plt.plot(d["date"], d["monthly_volume"], alpha=0.5)
        plt.plot(d["date"], d["pred_p50"], linestyle="--", alpha=0.6)
    plt.title("High-Error Sites: Actual vs Predicted")
    plt.tight_layout()
    p2 = out_dir / "backtest_high_error_sites.png"
    plt.savefig(p2)
    plt.close()

    cluster_curve = merged.groupby(["cluster_id", "age_in_months"], as_index=False)[["monthly_volume", "pred_p50"]].mean()
    plt.figure(figsize=(10, 6))
    for cid in cluster_curve["cluster_id"].drop_duplicates().head(5):
        d = cluster_curve[cluster_curve["cluster_id"] == cid].sort_values("age_in_months")
        plt.plot(d["age_in_months"], d["monthly_volume"], alpha=0.5)
        plt.plot(d["age_in_months"], d["pred_p50"], linestyle="--", alpha=0.7)
    plt.title("Cluster Average Curves")
    plt.tight_layout()
    p3 = out_dir / "backtest_cluster_curves.png"
    plt.savefig(p3)
    plt.close()

    heat = merged.assign(age_phase=np.select([merged["age_in_months"] <= 6, merged["age_in_months"] <= 18], ["early", "growth"], default="mature"))
    heat_tbl = heat.pivot_table(index="cluster_id", columns="age_phase", values="abs_error", aggfunc="mean").fillna(0)
    plt.figure(figsize=(8, 6))
    plt.imshow(heat_tbl.values, aspect="auto")
    plt.xticks(range(len(heat_tbl.columns)), heat_tbl.columns)
    plt.yticks(range(len(heat_tbl.index)), heat_tbl.index)
    plt.colorbar(label="MAE")
    plt.title("Error Heatmap: Cluster x Age Phase")
    plt.tight_layout()
    p4 = out_dir / "backtest_error_heatmap.png"
    plt.savefig(p4)
    plt.close()

    return {
        "mae": mae,
        "rmse": rmse,
        "p10_p90_coverage": coverage,
        "plot_random_sites": str(p1),
        "plot_high_error_sites": str(p2),
        "plot_cluster_curves": str(p3),
        "plot_error_heatmap": str(p4),
    }


def train_artifacts(df: pd.DataFrame) -> Artifacts:
    base = add_base_features(df)
    tables = build_cluster_tables(base)
    feat = attach_cluster_features(base, tables)
    train = feat[feat["date"] < pd.Timestamp("2025-01-01")].copy()

    site_type_clf, classes = fit_site_type_classifier(train)
    st = infer_site_type_proba(feat, site_type_clf, classes)
    feat["site_type"] = st["site_type"]
    cat_raw = ["cluster_id", "maturity_bucket", "site_type"]
    cat_maps = build_cat_maps(feat, cat_raw)
    feat = apply_cat_maps(feat, cat_maps)

    feature_cols = [
        "age_in_months", "real_age_months", "cluster_id_code", "cluster_month_avg", "cluster_age_avg",
        "month", "month_sin", "month_cos", "latitude", "longitude", "maturity_bucket_code", "site_type_code",
        "age_sq", "log_age", "is_early", "is_growth", "is_mature", "age_saturation", "growth_velocity",
        "distance_from_peak", "lat_lon_interaction", "cluster_growth_curve",
    ]
    feature_cols = [c for c in feature_cols if c in feat.columns]
    early_models, main_models = train_quantile_models(
        feat[feat["date"] < pd.Timestamp("2025-01-01")].copy(), feature_cols
    )
    centroids = base.groupby("cluster_id", as_index=False)[["latitude", "longitude"]].mean().dropna()
    return Artifacts(
        site_type_clf=site_type_clf,
        site_type_classes=classes,
        early_models=early_models,
        main_models=main_models,
        cluster_month_avg=tables["cma"],
        cluster_age_avg=tables["cas"],
        cluster_month_std=tables["cms"],
        cluster_age_std=tables["age_std"],
        cluster_centroids=centroids,
        feature_cols=feature_cols,
        cat_cols=[],
        cat_maps=cat_maps,
    )


def save_artifacts(artifacts: Artifacts, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, path)


def load_artifacts(path: Path) -> Artifacts:
    return joblib.load(path)


def final_report(
    lat: float,
    lon: float,
    artifacts: Artifacts,
    months: int,
    start_date: str,
    margin_per_wash: float,
    fixed_monthly_cost: float,
    ramp_up_cost: float,
    enable_mature_yoy_control: bool = True,
    mature_yoy_start_year: int = 4,
    mature_min_yoy: float = 0.005,
    mature_max_yoy: float = 0.05,
    max_cluster_distance_km: float | None = 100.0,
    enable_lifecycle_smoothing: bool = True,
    lifecycle_min_growth: float = 0.005,
    lifecycle_max_growth: float = 0.05,
) -> tuple[pd.DataFrame, dict]:
    weights = top_k_cluster_weights(lat, lon, artifacts.cluster_centroids, k=3)
    if max_cluster_distance_km is not None:
        max_d = float(max_cluster_distance_km)
        within = weights[weights["distance_km"] <= max_d].copy()
        if len(within) < 3:
            raise ValueError(
                "No forecast: need 3 nearest clusters within "
                f"{max_d:.1f} km (found {len(within)} within threshold)."
            )
        weights = within.sort_values("distance_km").head(3).copy()
    rows = make_future_rows(lat, lon, months, start_date)
    fc = blend_clusters(rows, weights, artifacts)
    mature_yoy_summary = {"enabled": False}
    if enable_mature_yoy_control:
        fc, mature_yoy_summary = apply_mature_yoy_control(
            fc,
            start_year=mature_yoy_start_year,
            min_yoy=mature_min_yoy,
            max_yoy=mature_max_yoy,
        )
    lifecycle_summary = {"enabled": False}
    if enable_lifecycle_smoothing:
        fc, lifecycle_summary = apply_lifecycle_smoothing_control(
            fc,
            min_annual_growth=lifecycle_min_growth,
            max_annual_growth=lifecycle_max_growth,
        )
    fc, be_month = break_even_from_costs(fc, margin_per_wash, fixed_monthly_cost, ramp_up_cost)
    summary = {
        "lat": lat,
        "lon": lon,
        "top3_clusters": weights.to_dict(orient="records"),
        "total_volume": float(fc["volume"].sum()),
        "total_profit": float(fc["monthly_profit"].sum()),
        "break_even_month": be_month,
        "mature_yoy_control": mature_yoy_summary,
        "lifecycle_smoothing_control": lifecycle_summary,
    }
    return fc, summary


def apply_global_uncertainty_calibration(
    forecast: pd.DataFrame,
    current_coverage: float,
    target_coverage: float = 0.80,
) -> tuple[pd.DataFrame, float]:
    out = forecast.copy()
    safe_cov = max(float(current_coverage), 1e-6)
    scale = float(target_coverage / safe_cov)
    # Apply calibration to output using p10/p50/p90 anchors.
    lower_spread = out["p50"] - out["p10"]
    upper_spread = out["p90"] - out["p50"]
    out["low"] = (out["p50"] - scale * lower_spread).clip(lower=0)
    out["high"] = (out["p50"] + scale * upper_spread).clip(lower=0)
    out["volume"] = out["p50"]
    return out, scale


def add_confidence_label(forecast: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    out = forecast.copy()
    rel_width = (out["high"] - out["low"]) / (out["volume"].abs() + 1e-6)
    out["interval_width_ratio"] = rel_width

    q1 = float(rel_width.quantile(0.33))
    q2 = float(rel_width.quantile(0.66))

    def _label(x: float) -> str:
        if x <= q1:
            return "High confidence"
        if x <= q2:
            return "Medium confidence"
        return "Low confidence"

    out["confidence_label"] = rel_width.apply(_label)
    overall = _label(float(rel_width.mean()))
    return out, overall


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Advanced forecast engine with uncertainty and business outputs.")
    p.add_argument("--input", type=Path, default=Path("zeta_modelling/data/phase1_final_monthly_2024_2025.csv"))
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--months", type=int, default=60)
    p.add_argument("--start-date", type=str, default="2026-01-01")
    p.add_argument("--margin-per-wash", type=float, default=4.0)
    p.add_argument("--fixed-monthly-cost", type=float, default=50000.0)
    p.add_argument("--ramp-up-cost", type=float, default=150000.0)
    p.add_argument("--out-csv", type=Path, default=Path("zeta_modelling/data/phase3_advanced_forecast.csv"))
    p.add_argument("--out-report", type=Path, default=Path("zeta_modelling/data/phase3_advanced_report.json"))
    p.add_argument("--out-backtest-dir", type=Path, default=Path("zeta_modelling/data/phase3_backtest_plots"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input, low_memory=False)
    artifacts = train_artifacts(df)
    forecast, summary = final_report(
        lat=args.lat,
        lon=args.lon,
        artifacts=artifacts,
        months=args.months,
        start_date=args.start_date,
        margin_per_wash=args.margin_per_wash,
        fixed_monthly_cost=args.fixed_monthly_cost,
        ramp_up_cost=args.ramp_up_cost,
    )
    backtest = evaluate_backtest(add_base_features(df), artifacts, args.out_backtest_dir)
    forecast, uncertainty_scale = apply_global_uncertainty_calibration(
        forecast=forecast,
        current_coverage=float(backtest["p10_p90_coverage"]),
        target_coverage=0.80,
    )
    forecast, overall_confidence = add_confidence_label(forecast)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(args.out_csv, index=False)
    range_low = float(forecast["low"].sum())
    range_high = float(forecast["high"].sum())
    risk_label = "High variability cluster" if float(forecast["interval_width_ratio"].mean()) > 1.0 else "Moderate variability cluster"
    report = {
        "forecast_summary": summary,
        "uncertainty_calibration": {
            "method": "global_single_scale",
            "target_coverage": 0.80,
            "current_coverage_before_calibration": float(backtest["p10_p90_coverage"]),
            "scale": uncertainty_scale,
        },
        "final_report": {
            "expected_volume_5y": float(forecast["volume"].sum()),
            "range_5y_low": range_low,
            "range_5y_high": range_high,
            "break_even": (
                f"Month {summary['break_even_month']}"
                if summary["break_even_month"] is not None
                else f"Not reached (margin={args.margin_per_wash})"
            ),
            "confidence": overall_confidence,
            "risk": risk_label,
        },
        "backtest": backtest,
        "output_csv": str(args.out_csv),
    }
    args.out_report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
