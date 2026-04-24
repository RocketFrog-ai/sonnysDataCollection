"""End-to-end greenfield projection evaluation across current sites.

What this measures:
  - ``<2y``: projected Year 1 / Year 2 vs actual opening-year totals from
    ``less_than-2yrs-clustering-ready.csv``.
  - ``>2y`` literal calendar analogy: projected Year 1 / Year 2 vs actual
    2024 / 2025 totals from ``master_more_than-2yrs.csv``.
  - ``>2y`` mature analogy: projected Year 3 / Year 4 vs actual
    2024 / 2025 totals from ``master_more_than-2yrs.csv``.

This is intentionally different from the level-model holdout reports:
it scores the full address/lat-lon -> nearest cluster -> level anchor ->
cluster TS forecast pipeline.

Run:
  python daily_data/daily-data-modelling/clustering_v2/eval_end_to_end_projection_accuracy.py
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

V2 = Path(__file__).resolve().parent
REPO_ROOT = V2.parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(V2))

import project_site as ps  # noqa: E402
import project_site_quantile as psq  # noqa: E402


RESULTS_DIR = V2 / "results"
OUT_JSON = RESULTS_DIR / "end_to_end_projection_accuracy.json"
_ORIG_LOAD_COHORT = ps._load_cohort
_ORIG_JOBLIB_LOAD = joblib.load

warnings.filterwarnings(
    "ignore",
    message="`sklearn.utils.parallel.delayed` should be used",
    category=UserWarning,
)


VARIANTS: list[dict[str, Any]] = [
    {
        "name": "ridge_arima_prefix_bridge",
        "family": "ridge_rf",
        "level_model": "ridge",
        "method": "arima",
        "use_prefix": True,
        "bridge": True,
    },
    {
        "name": "ridge_blend_prefix_bridge",
        "family": "ridge_rf",
        "level_model": "ridge",
        "method": "blend",
        "use_prefix": True,
        "bridge": True,
    },
    {
        "name": "rf_arima_prefix_bridge",
        "family": "ridge_rf",
        "level_model": "rf",
        "method": "arima",
        "use_prefix": True,
        "bridge": True,
    },
    {
        "name": "rf_blend_prefix_bridge",
        "family": "ridge_rf",
        "level_model": "rf",
        "method": "blend",
        "use_prefix": True,
        "bridge": True,
    },
    {
        "name": "rf_arima_no_prefix",
        "family": "ridge_rf",
        "level_model": "rf",
        "method": "arima",
        "use_prefix": False,
        "bridge": False,
    },
]


def _metric_block(actual: list[float], pred: list[float]) -> dict[str, Any]:
    y = np.asarray(actual, float)
    p = np.asarray(pred, float)
    m = np.isfinite(y) & np.isfinite(p) & (y >= 0)
    y = y[m]
    p = p[m]
    if len(y) == 0:
        return {"n": 0}
    err = p - y
    abs_err = np.abs(err)
    out: dict[str, Any] = {
        "n": int(len(y)),
        "actual_sum": round(float(np.sum(y)), 2),
        "pred_sum": round(float(np.sum(p)), 2),
        "bias_pct_of_actual_sum": round(float(np.sum(err) / max(np.sum(y), 1e-9)) * 100.0, 2),
        "mae": round(float(np.mean(abs_err)), 2),
        "rmse": round(float(np.sqrt(np.mean(err ** 2))), 2),
        "wape": round(float(np.sum(abs_err) / max(np.sum(np.abs(y)), 1e-9)), 4),
    }
    pos = y > 0
    if np.any(pos):
        ape = abs_err[pos] / y[pos]
        out["median_abs_pct_error"] = round(float(np.median(ape)) * 100.0, 2)
        out["mean_abs_pct_error"] = round(float(np.mean(ape)) * 100.0, 2)
        for thr in (0.10, 0.15, 0.20, 0.30):
            out[f"pct_within_{int(thr * 100)}pct"] = round(float(np.mean(ape <= thr)) * 100.0, 2)
    return out


def _site_metric_rows(rows: list[dict[str, Any]], actual_key: str, pred_key: str) -> tuple[list[float], list[float]]:
    actual: list[float] = []
    pred: list[float] = []
    for row in rows:
        a = row.get(actual_key)
        p = row.get(pred_key)
        if a is None or p is None:
            continue
        try:
            af = float(a)
            pf = float(p)
        except (TypeError, ValueError):
            continue
        if np.isfinite(af) and np.isfinite(pf):
            actual.append(af)
            pred.append(pf)
    return actual, pred


def _year_summary(rows: list[dict[str, Any]], actual_key: str, pred_key: str) -> dict[str, Any]:
    actual, pred = _site_metric_rows(rows, actual_key, pred_key)
    return _metric_block(actual, pred)


def _combined_year_summary(
    rows: list[dict[str, Any]],
    year_pairs: list[tuple[str, str]],
) -> dict[str, Any]:
    actual: list[float] = []
    pred: list[float] = []
    for a_key, p_key in year_pairs:
        a, p = _site_metric_rows(rows, a_key, p_key)
        actual.extend(a)
        pred.extend(p)
    return _metric_block(actual, pred)


def _distance_summary(distances: list[float]) -> dict[str, Any]:
    d = np.asarray([x for x in distances if np.isfinite(x)], float)
    if len(d) == 0:
        return {"n": 0}
    return {
        "n": int(len(d)),
        "median_km": round(float(np.median(d)), 3),
        "p90_km": round(float(np.quantile(d, 0.9)), 3),
        "max_km": round(float(np.max(d)), 3),
        "pct_within_20km": round(float(np.mean(d <= ps.MAX_NEAREST_CLUSTER_DISTANCE_KM)) * 100.0, 2),
    }


def _rows_to_summary(
    lt_rows: list[dict[str, Any]],
    gt_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "less_than_2yrs": {
            "n_sites": len(lt_rows),
            "cluster_distance_summary_km": _distance_summary([float(r["cluster_distance_km"]) for r in lt_rows]),
            "year_1": _year_summary(lt_rows, "actual_year_1", "pred_year_1"),
            "year_2": _year_summary(lt_rows, "actual_year_2", "pred_year_2"),
            "combined_years_1_2": _combined_year_summary(
                lt_rows,
                [("actual_year_1", "pred_year_1"), ("actual_year_2", "pred_year_2")],
            ),
        },
        "more_than_2yrs_literal_calendar_analogy": {
            "definition": "Compare projected operational Year 1/2 to actual calendar-year 2024/2025 totals of mature sites. This is not a true greenfield validation.",
            "n_sites": len(gt_rows),
            "cluster_distance_summary_km": _distance_summary([float(r["cluster_distance_km"]) for r in gt_rows]),
            "year_1_vs_2024": _year_summary(gt_rows, "actual_2024", "pred_year_1"),
            "year_2_vs_2025": _year_summary(gt_rows, "actual_2025", "pred_year_2"),
            "combined": _combined_year_summary(
                gt_rows,
                [("actual_2024", "pred_year_1"), ("actual_2025", "pred_year_2")],
            ),
        },
        "more_than_2yrs_mature_analogy": {
            "definition": "Compare projected operational Year 3/4 to actual calendar-year 2024/2025 totals of mature sites. This is the more stage-aligned >2y analogue.",
            "n_sites": len(gt_rows),
            "cluster_distance_summary_km": _distance_summary([float(r["cluster_distance_km"]) for r in gt_rows]),
            "year_3_vs_2024": _year_summary(gt_rows, "actual_2024", "pred_year_3"),
            "year_4_vs_2025": _year_summary(gt_rows, "actual_2025", "pred_year_4"),
            "combined": _combined_year_summary(
                gt_rows,
                [("actual_2024", "pred_year_3"), ("actual_2025", "pred_year_4")],
            ),
        },
    }


@lru_cache(maxsize=None)
def _ridge_rf_assets(cohort: str) -> dict[str, Any]:
    return _ORIG_LOAD_COHORT(cohort)


@lru_cache(maxsize=None)
def _quantile_support(cohort: str) -> dict[str, Any]:
    return _ORIG_LOAD_COHORT(cohort)


@lru_cache(maxsize=None)
def _cached_joblib(path_str: str) -> Any:
    return _ORIG_JOBLIB_LOAD(path_str)


@lru_cache(maxsize=None)
def _cached_feature_order(path_str: str) -> list[str]:
    return json.loads(Path(path_str).read_text())


def _project_ridge_rf_variant(lat: float, lon: float, variant: dict[str, Any]) -> dict[str, Any]:
    less_assets = _ridge_rf_assets("less_than")
    more_assets = _ridge_rf_assets("more_than")

    less_block = ps._project_cohort(
        cohort="less_than",
        cohort_label="less_than_2yrs",
        lat=lat,
        lon=lon,
        method=variant["method"],
        cohort_assets=less_assets,
        days_per_month=30,
        is_daily_model=False,
        opening_prefix_monthly=None,
        allow_nearest_cluster_beyond_distance_cap=True,
        level_model=variant["level_model"],
    )
    prefix: list[float] | None = None
    if variant["use_prefix"] and "error" not in less_block:
        mp = less_block.get("monthly_projection") or []
        if len(mp) >= 24:
            prefix = [float(r["wash_count"]) for r in mp[:24]]

    more_block = ps._project_cohort(
        cohort="more_than",
        cohort_label="more_than_2yrs",
        lat=lat,
        lon=lon,
        method=variant["method"],
        cohort_assets=more_assets,
        days_per_month=30,
        is_daily_model=True,
        opening_prefix_monthly=prefix,
        allow_nearest_cluster_beyond_distance_cap=True,
        level_model=variant["level_model"],
    )

    resp = {
        "method": variant["method"],
        "level_model": variant["level_model"],
        "less_than_2yrs": less_block,
        "more_than_2yrs": more_block,
        "input": {"lat": lat, "lon": lon},
    }
    ps._bridge_mature_monthly_to_opening_last_month(
        resp,
        skip_bridge_when_prefix=bool(variant["use_prefix"] and not variant["bridge"]),
    )
    ps._enrich_brand_new_site_timeline(resp)
    ps._append_calendar_year_washes_ridge(resp)
    resp["use_opening_prefix_for_mature_forecast"] = bool(variant["use_prefix"])
    resp["bridge_opening_to_mature_when_prefix"] = bool(variant["bridge"])
    return resp


def _project_quantile_variant(lat: float, lon: float, variant: dict[str, Any]) -> dict[str, Any]:
    orig_load_cohort = ps._load_cohort
    orig_joblib_load = psq.joblib.load
    orig_load_feature_order = psq._load_feature_order
    try:
        ps._load_cohort = _quantile_support
        psq.joblib.load = lambda p: _cached_joblib(str(Path(p).resolve()))
        psq._load_feature_order = lambda cohort_dir: _cached_feature_order(
            str((Path(cohort_dir) / "feature_order.json").resolve())
        )
        resp = psq.build_quantile_projection_response(
            lat,
            lon,
            variant["method"],
            None,
            use_opening_prefix_for_mature_forecast=variant["use_prefix"],
            bridge_opening_to_mature_when_prefix=variant["bridge"],
            allow_nearest_cluster_beyond_distance_cap=True,
        )
    finally:
        ps._load_cohort = orig_load_cohort
        psq.joblib.load = orig_joblib_load
        psq._load_feature_order = orig_load_feature_order
    resp["level_model"] = "quantile"
    return resp


def _project_variant(lat: float, lon: float, variant: dict[str, Any]) -> dict[str, Any]:
    if variant["family"] == "quantile":
        return _project_quantile_variant(lat, lon, variant)
    return _project_ridge_rf_variant(lat, lon, variant)


def _build_less_than_sites() -> pd.DataFrame:
    df = pd.read_csv(REPO_ROOT / "daily_data/daily-data-modelling/less_than-2yrs-clustering-ready.csv", low_memory=False)
    actual = (
        df.groupby(["site_client_id", "year_number"], as_index=False)["wash_count_total"]
        .sum()
        .pivot(index="site_client_id", columns="year_number", values="wash_count_total")
        .reset_index()
        .rename(columns={1: "actual_year_1", 2: "actual_year_2"})
    )
    meta = (
        df.sort_values(["site_client_id", "period_index"])
        .drop_duplicates("site_client_id")[["site_client_id", "latitude", "longitude", "Address"]]
    )
    out = meta.merge(actual, on="site_client_id", how="inner")
    out = out.dropna(subset=["latitude", "longitude", "actual_year_1", "actual_year_2"])
    return out


def _build_more_than_sites() -> pd.DataFrame:
    df = pd.read_csv(REPO_ROOT / "daily_data/daily-data-modelling/master_more_than-2yrs.csv", low_memory=False)
    df["calendar_day"] = pd.to_datetime(df["calendar_day"], errors="coerce")
    df = df.dropna(subset=["calendar_day"])
    df["calendar_year"] = df["calendar_day"].dt.year
    actual = (
        df.groupby(["site_client_id", "calendar_year"], as_index=False)["wash_count_total"]
        .sum()
        .pivot(index="site_client_id", columns="calendar_year", values="wash_count_total")
        .reset_index()
        .rename(columns={2024: "actual_2024", 2025: "actual_2025"})
    )
    meta = (
        df.sort_values(["site_client_id", "calendar_day"])
        .drop_duplicates("site_client_id")[["site_client_id", "latitude", "longitude", "Address"]]
    )
    out = meta.merge(actual, on="site_client_id", how="inner")
    out = out.dropna(subset=["latitude", "longitude", "actual_2024", "actual_2025"])
    return out


def _extract_lt_row(site_row: pd.Series, resp: dict[str, Any]) -> dict[str, Any] | None:
    lt = resp.get("less_than_2yrs") or {}
    cy = resp.get("calendar_year_washes") or {}
    if "error" in lt:
        return None
    if resp.get("level_model") == "quantile":
        y1 = ((cy.get("year_1") or {}).get("q50"))
        y2 = ((cy.get("year_2") or {}).get("q50"))
    else:
        y1 = cy.get("year_1")
        y2 = cy.get("year_2")
    if y1 is None or y2 is None:
        return None
    return {
        "site_client_id": int(site_row["site_client_id"]),
        "cluster_distance_km": float((lt.get("cluster") or {}).get("distance_km", np.nan)),
        "actual_year_1": float(site_row["actual_year_1"]),
        "actual_year_2": float(site_row["actual_year_2"]),
        "pred_year_1": float(y1),
        "pred_year_2": float(y2),
    }


def _extract_gt_row(site_row: pd.Series, resp: dict[str, Any]) -> dict[str, Any] | None:
    gt = resp.get("more_than_2yrs") or {}
    cy = resp.get("calendar_year_washes") or {}
    if "error" in gt:
        return None
    if resp.get("level_model") == "quantile":
        y1 = ((cy.get("year_1") or {}).get("q50"))
        y2 = ((cy.get("year_2") or {}).get("q50"))
        y3 = ((cy.get("year_3") or {}).get("q50"))
        y4 = ((cy.get("year_4") or {}).get("q50"))
    else:
        y1 = cy.get("year_1")
        y2 = cy.get("year_2")
        y3 = cy.get("year_3")
        y4 = cy.get("year_4")
    if any(v is None for v in (y1, y2, y3, y4)):
        return None
    return {
        "site_client_id": int(site_row["site_client_id"]),
        "cluster_distance_km": float((gt.get("cluster") or {}).get("distance_km", np.nan)),
        "actual_2024": float(site_row["actual_2024"]),
        "actual_2025": float(site_row["actual_2025"]),
        "pred_year_1": float(y1),
        "pred_year_2": float(y2),
        "pred_year_3": float(y3),
        "pred_year_4": float(y4),
    }


def _evaluate_variant(
    variant: dict[str, Any],
    lt_sites: pd.DataFrame,
    gt_sites: pd.DataFrame,
) -> dict[str, Any]:
    lt_rows: list[dict[str, Any]] = []
    gt_rows: list[dict[str, Any]] = []

    for _, row in lt_sites.iterrows():
        resp = _project_variant(float(row["latitude"]), float(row["longitude"]), variant)
        out = _extract_lt_row(row, resp)
        if out is not None:
            lt_rows.append(out)

    for _, row in gt_sites.iterrows():
        resp = _project_variant(float(row["latitude"]), float(row["longitude"]), variant)
        out = _extract_gt_row(row, resp)
        if out is not None:
            gt_rows.append(out)

    return {
        "variant": variant,
        "site_counts": {
            "less_than_requested": int(len(lt_sites)),
            "less_than_scored": len(lt_rows),
            "more_than_requested": int(len(gt_sites)),
            "more_than_scored": len(gt_rows),
        },
        "summary": _rows_to_summary(lt_rows, gt_rows),
    }


def _scoreboard(results: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for r in results:
        name = r["variant"]["name"]
        s = r["summary"]
        rows.extend(
            [
                {
                    "variant": name,
                    "evaluation_slice": "<2y_combined_years_1_2",
                    "wape": s["less_than_2yrs"]["combined_years_1_2"]["wape"],
                    "mae": s["less_than_2yrs"]["combined_years_1_2"]["mae"],
                },
                {
                    "variant": name,
                    "evaluation_slice": ">2y_literal_combined",
                    "wape": s["more_than_2yrs_literal_calendar_analogy"]["combined"]["wape"],
                    "mae": s["more_than_2yrs_literal_calendar_analogy"]["combined"]["mae"],
                },
                {
                    "variant": name,
                    "evaluation_slice": ">2y_mature_combined",
                    "wape": s["more_than_2yrs_mature_analogy"]["combined"]["wape"],
                    "mae": s["more_than_2yrs_mature_analogy"]["combined"]["mae"],
                },
            ]
        )
    df = pd.DataFrame(rows)
    best: dict[str, Any] = {}
    for key, grp in df.groupby("evaluation_slice", sort=False):
        grp = grp.sort_values(["wape", "mae", "variant"], kind="stable").reset_index(drop=True)
        best[key] = grp.to_dict(orient="records")
    return {"rows": rows, "ranked_by_slice": best}


def main() -> None:
    lt_sites = _build_less_than_sites()
    gt_sites = _build_more_than_sites()

    results: list[dict[str, Any]] = []
    for variant in VARIANTS:
        print(f"[eval] {variant['name']}")
        results.append(_evaluate_variant(variant, lt_sites, gt_sites))

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "End-to-end projection accuracy for current greenfield pipeline variants.",
        "notes": [
            "This scores the full projection pipeline, not just the level-model holdout.",
            "<2y year-1 numbers are still optimistic relative to a true leave-site-out greenfield test because the deployed artifacts were trained using year-1 rows from the same cohort.",
            ">2y comparisons are analogies, not direct greenfield ground truth, because the mature file only contains realized 2024/2025 volumes for already-operating sites.",
            "All variants were evaluated with allow_nearest_cluster_beyond_distance_cap=True so every site could be scored; distance summaries show how often a site is within the production 20 km cap.",
            "Quantile q50 was not included in this end-to-end sweep because the saved joblib artifacts are not compatible with the current sklearn runtime (`SimpleImputer` deserialization error).",
        ],
        "dataset_counts": {
            "less_than_sites": int(len(lt_sites)),
            "more_than_sites": int(len(gt_sites)),
        },
        "variants": results,
        "scoreboard": _scoreboard(results),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out["scoreboard"], indent=2))
    print(f"\nwrote {OUT_JSON.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
