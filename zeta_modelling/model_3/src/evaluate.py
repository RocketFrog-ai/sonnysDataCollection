"""Evaluation: metrics, HIT/MISS, segment slices."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denom = np.sum(np.abs(y_true))
    wmape = float(np.sum(np.abs(y_true - y_pred)) / denom) if denom > 0 else float("nan")
    bias = float(np.mean(y_pred - y_true))
    return {"MAE": mae, "RMSE": rmse, "WMAPE": wmape, "bias": bias, "n": int(len(y_true))}


def site_year_hit_miss(forecast_long: pd.DataFrame, actual_long: pd.DataFrame,
                       band: float = C.HIT_BAND) -> pd.DataFrame:
    """Aggregate predictions and actuals to (site, year_idx) annual totals
    and report HIT (|pred-actual| <= band) per row.

    Parameters
    ----------
    forecast_long : columns [client_id_location_id, month_offset, predicted_washes]
    actual_long   : columns [client_id_location_id, month_offset, wash_count_total]
    """
    f = forecast_long.copy()
    a = actual_long.copy()
    f["year_idx"] = f["month_offset"] // 12 + 1
    a["year_idx"] = a["month_offset"] // 12 + 1
    fp = f.groupby(["client_id_location_id", "year_idx"])["predicted_washes"].sum().reset_index()
    ap = a.groupby(["client_id_location_id", "year_idx"])["wash_count_total"].sum().reset_index()
    j = fp.merge(ap, on=["client_id_location_id", "year_idx"], how="inner")
    j["abs_err"] = (j["predicted_washes"] - j["wash_count_total"]).abs()
    j["hit"] = (j["abs_err"] <= band).astype(int)
    return j


def summarize_hit_miss(hit_df: pd.DataFrame) -> dict:
    if len(hit_df) == 0:
        return {"overall_hit_rate": float("nan"), "n": 0}
    overall = float(hit_df["hit"].mean())
    by_year = hit_df.groupby("year_idx")["hit"].mean().to_dict()
    return {
        "overall_hit_rate": overall,
        "by_year_hit_rate": {int(k): float(v) for k, v in by_year.items()},
        "n_site_years": int(len(hit_df)),
    }


def segment_metrics(eval_df: pd.DataFrame, pred_col: str, actual_col: str) -> dict:
    out = {"overall": regression_metrics(eval_df[actual_col], eval_df[pred_col])}
    for col in ("region", "peer_level_resolved", "maturity_indicator"):
        if col not in eval_df.columns:
            continue
        out[f"by_{col}"] = {
            str(k): regression_metrics(g[actual_col], g[pred_col])
            for k, g in eval_df.groupby(col)
        }
    return out


def market_level_metrics(eval_df: pd.DataFrame, pred_col: str, actual_col: str) -> pd.DataFrame:
    g = eval_df.groupby("cbsa_id").apply(
        lambda d: pd.Series(regression_metrics(d[actual_col], d[pred_col]))
    ).reset_index()
    return g.sort_values("WMAPE")
