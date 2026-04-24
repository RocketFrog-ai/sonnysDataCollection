"""Evaluate projections against same-cluster peer ranges on test sites.

For each evaluated site:
  1. Run the projection pipeline to get the assigned cluster and yearly forecasts.
  2. Take the *other* sites in that same cohort+cluster.
  3. Build peer actual totals for the relevant year comparison.
  4. Check whether the prediction sits inside the peer min/max range and how it compares
     with the peer median baseline.

Default evaluation uses the current production-like config:
  RF + ARIMA + no opening prefix for >2y.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

V2 = Path(__file__).resolve().parent
REPO_ROOT = V2.parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(V2))

import eval_end_to_end_projection_accuracy as e2e  # noqa: E402


OUT_JSON = V2 / "results" / "cluster_peer_range_accuracy.json"


def _peer_stats(values: list[float]) -> dict[str, float] | None:
    arr = np.asarray([float(v) for v in values if np.isfinite(float(v))], float)
    if len(arr) == 0:
        return None
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "n": float(len(arr)),
    }


def _nearest_boundary_gap(value: float, lo: float, hi: float) -> float:
    if lo <= value <= hi:
        return 0.0
    if value < lo:
        return float(lo - value)
    return float(value - hi)


def _range_eval(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [
        r
        for r in rows
        if all(k in r for k in ("actual", "pred", "peer_min", "peer_max", "peer_median"))
        and np.isfinite(float(r["actual"]))
        and np.isfinite(float(r["pred"]))
        and np.isfinite(float(r["peer_min"]))
        and np.isfinite(float(r["peer_max"]))
        and np.isfinite(float(r["peer_median"]))
    ]
    if not usable:
        return {"n": 0}

    actual = np.asarray([float(r["actual"]) for r in usable], float)
    pred = np.asarray([float(r["pred"]) for r in usable], float)
    med = np.asarray([float(r["peer_median"]) for r in usable], float)
    lo = np.asarray([float(r["peer_min"]) for r in usable], float)
    hi = np.asarray([float(r["peer_max"]) for r in usable], float)

    pred_gap = np.asarray([_nearest_boundary_gap(p, l, h) for p, l, h in zip(pred, lo, hi)], float)
    act_gap = np.asarray([_nearest_boundary_gap(a, l, h) for a, l, h in zip(actual, lo, hi)], float)
    pred_inside = (pred >= lo) & (pred <= hi)
    act_inside = (actual >= lo) & (actual <= hi)

    abs_err_model = np.abs(pred - actual)
    abs_err_med = np.abs(med - actual)
    ape_model = abs_err_model / np.maximum(actual, 1e-9)
    ape_med = abs_err_med / np.maximum(actual, 1e-9)

    return {
        "n": int(len(usable)),
        "pred_inside_peer_min_max_pct": round(float(np.mean(pred_inside)) * 100.0, 2),
        "actual_inside_peer_min_max_pct": round(float(np.mean(act_inside)) * 100.0, 2),
        "pred_below_peer_range_pct": round(float(np.mean(pred < lo)) * 100.0, 2),
        "pred_above_peer_range_pct": round(float(np.mean(pred > hi)) * 100.0, 2),
        "median_gap_to_peer_range_when_outside": round(float(np.median(pred_gap[pred_gap > 0])) if np.any(pred_gap > 0) else 0.0, 2),
        "mean_gap_to_peer_range_when_outside": round(float(np.mean(pred_gap[pred_gap > 0])) if np.any(pred_gap > 0) else 0.0, 2),
        "model_mae_vs_actual": round(float(np.mean(abs_err_model)), 2),
        "cluster_median_baseline_mae_vs_actual": round(float(np.mean(abs_err_med)), 2),
        "model_wape_vs_actual": round(float(np.sum(abs_err_model) / max(np.sum(actual), 1e-9)), 4),
        "cluster_median_baseline_wape_vs_actual": round(float(np.sum(abs_err_med) / max(np.sum(actual), 1e-9)), 4),
        "pct_sites_model_better_than_cluster_median": round(float(np.mean(abs_err_model < abs_err_med)) * 100.0, 2),
        "median_abs_pct_error_model": round(float(np.median(ape_model)) * 100.0, 2),
        "median_abs_pct_error_cluster_median": round(float(np.median(ape_med)) * 100.0, 2),
        "median_gap_pred_to_peer_median": round(float(np.median(np.abs(pred - med))), 2),
        "median_gap_actual_to_peer_median": round(float(np.median(np.abs(actual - med))), 2),
        "median_gap_actual_to_peer_range": round(float(np.median(act_gap)), 2),
    }


def _top_counts(counter: Counter[int], top_n: int = 15) -> list[dict[str, int]]:
    return [{"cluster_id": int(cid), "count": int(cnt)} for cid, cnt in counter.most_common(top_n)]


def _build_less_than_peer_lookup() -> dict[int, dict[int, dict[str, float]]]:
    df = pd.read_csv(REPO_ROOT / "daily_data/daily-data-modelling/less_than-2yrs-clustering-ready.csv", low_memory=False)
    agg = (
        df.groupby(["dbscan_cluster_12km", "site_client_id", "year_number"], as_index=False)["wash_count_total"]
        .sum()
    )
    out: dict[int, dict[int, dict[str, float]]] = {}
    for (cluster_id, site_id), grp in agg.groupby(["dbscan_cluster_12km", "site_client_id"], sort=False):
        try:
            cid = int(cluster_id)
            sid = int(site_id)
        except (TypeError, ValueError):
            continue
        years = {int(r["year_number"]): float(r["wash_count_total"]) for _, r in grp.iterrows()}
        out.setdefault(cid, {})[sid] = years
    return out


def _build_more_than_peer_lookup() -> dict[int, dict[int, dict[int, float]]]:
    master = pd.read_csv(REPO_ROOT / "daily_data/daily-data-modelling/master_more_than-2yrs.csv", low_memory=False)
    more = pd.read_csv(REPO_ROOT / "daily_data/daily-data-modelling/more_than-2yrs.csv", low_memory=False)
    extra = [c for c in ("dbscan_cluster_12km",) if c in more.columns and c not in master.columns]
    if extra:
        df = master.merge(
            more[["site_client_id", "calendar_day"] + extra],
            on=["site_client_id", "calendar_day"],
            how="left",
        )
    else:
        df = master
    df["calendar_day"] = pd.to_datetime(df["calendar_day"], errors="coerce")
    df = df.dropna(subset=["calendar_day"])
    df["calendar_year"] = df["calendar_day"].dt.year
    agg = (
        df.groupby(["dbscan_cluster_12km", "site_client_id", "calendar_year"], as_index=False)["wash_count_total"]
        .sum()
    )
    out: dict[int, dict[int, dict[int, float]]] = {}
    for (cluster_id, site_id), grp in agg.groupby(["dbscan_cluster_12km", "site_client_id"], sort=False):
        try:
            cid = int(cluster_id)
            sid = int(site_id)
        except (TypeError, ValueError):
            continue
        years = {int(r["calendar_year"]): float(r["wash_count_total"]) for _, r in grp.iterrows()}
        out.setdefault(cid, {})[sid] = years
    return out


def _cluster_example(
    cluster_id: int,
    peer_lookup: dict[int, dict[int, dict[int, float]]],
    year_a: int,
    year_b: int,
) -> dict[str, Any] | None:
    peers = peer_lookup.get(int(cluster_id))
    if not peers:
        return None
    vals_a = [float(v.get(year_a)) for v in peers.values() if year_a in v]
    vals_b = [float(v.get(year_b)) for v in peers.values() if year_b in v]
    sa = _peer_stats(vals_a)
    sb = _peer_stats(vals_b)
    if sa is None and sb is None:
        return None
    return {
        "cluster_id": int(cluster_id),
        f"year_{year_a}_peer_stats": sa,
        f"year_{year_b}_peer_stats": sb,
    }


def main() -> None:
    variant = {
        "name": "rf_arima_no_prefix",
        "family": "ridge_rf",
        "level_model": "rf",
        "method": "arima",
        "use_prefix": False,
        "bridge": False,
    }

    lt_sites = e2e._build_less_than_sites()
    gt_sites = e2e._build_more_than_sites()
    lt_peer_lookup = _build_less_than_peer_lookup()
    gt_peer_lookup = _build_more_than_peer_lookup()

    lt_cluster_counts: Counter[int] = Counter()
    gt_cluster_counts: Counter[int] = Counter()

    lt_y1_rows: list[dict[str, Any]] = []
    lt_y2_rows: list[dict[str, Any]] = []
    gt_lit_y1_rows: list[dict[str, Any]] = []
    gt_lit_y2_rows: list[dict[str, Any]] = []
    gt_mat_y3_rows: list[dict[str, Any]] = []
    gt_mat_y4_rows: list[dict[str, Any]] = []

    for _, row in lt_sites.iterrows():
        resp = e2e._project_variant(float(row["latitude"]), float(row["longitude"]), variant)
        out = e2e._extract_lt_row(row, resp)
        if out is None:
            continue
        cid = int((resp.get("less_than_2yrs") or {}).get("cluster", {}).get("cluster_id", -999))
        sid = int(out["site_client_id"])
        lt_cluster_counts[cid] += 1
        peers = {
            psid: vals
            for psid, vals in lt_peer_lookup.get(cid, {}).items()
            if int(psid) != sid
        }
        vals_y1 = [float(v[1]) for v in peers.values() if 1 in v]
        vals_y2 = [float(v[2]) for v in peers.values() if 2 in v]
        s1 = _peer_stats(vals_y1)
        s2 = _peer_stats(vals_y2)
        if s1 is not None:
            lt_y1_rows.append(
                {
                    "site_client_id": sid,
                    "cluster_id": cid,
                    "actual": float(out["actual_year_1"]),
                    "pred": float(out["pred_year_1"]),
                    "peer_min": s1["min"],
                    "peer_max": s1["max"],
                    "peer_median": s1["median"],
                }
            )
        if s2 is not None:
            lt_y2_rows.append(
                {
                    "site_client_id": sid,
                    "cluster_id": cid,
                    "actual": float(out["actual_year_2"]),
                    "pred": float(out["pred_year_2"]),
                    "peer_min": s2["min"],
                    "peer_max": s2["max"],
                    "peer_median": s2["median"],
                }
            )

    for _, row in gt_sites.iterrows():
        resp = e2e._project_variant(float(row["latitude"]), float(row["longitude"]), variant)
        out = e2e._extract_gt_row(row, resp)
        if out is None:
            continue
        cid = int((resp.get("more_than_2yrs") or {}).get("cluster", {}).get("cluster_id", -999))
        sid = int(out["site_client_id"])
        gt_cluster_counts[cid] += 1
        peers = {
            psid: vals
            for psid, vals in gt_peer_lookup.get(cid, {}).items()
            if int(psid) != sid
        }
        vals_2024 = [float(v[2024]) for v in peers.values() if 2024 in v]
        vals_2025 = [float(v[2025]) for v in peers.values() if 2025 in v]
        s24 = _peer_stats(vals_2024)
        s25 = _peer_stats(vals_2025)
        if s24 is not None:
            gt_lit_y1_rows.append(
                {
                    "site_client_id": sid,
                    "cluster_id": cid,
                    "actual": float(out["actual_2024"]),
                    "pred": float(out["pred_year_1"]),
                    "peer_min": s24["min"],
                    "peer_max": s24["max"],
                    "peer_median": s24["median"],
                }
            )
            gt_mat_y3_rows.append(
                {
                    "site_client_id": sid,
                    "cluster_id": cid,
                    "actual": float(out["actual_2024"]),
                    "pred": float(out["pred_year_3"]),
                    "peer_min": s24["min"],
                    "peer_max": s24["max"],
                    "peer_median": s24["median"],
                }
            )
        if s25 is not None:
            gt_lit_y2_rows.append(
                {
                    "site_client_id": sid,
                    "cluster_id": cid,
                    "actual": float(out["actual_2025"]),
                    "pred": float(out["pred_year_2"]),
                    "peer_min": s25["min"],
                    "peer_max": s25["max"],
                    "peer_median": s25["median"],
                }
            )
            gt_mat_y4_rows.append(
                {
                    "site_client_id": sid,
                    "cluster_id": cid,
                    "actual": float(out["actual_2025"]),
                    "pred": float(out["pred_year_4"]),
                    "peer_min": s25["min"],
                    "peer_max": s25["max"],
                    "peer_median": s25["median"],
                }
            )

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": variant,
        "site_counts": {
            "less_than_test_sites": int(len(lt_sites)),
            "more_than_test_sites": int(len(gt_sites)),
        },
        "less_than_2yrs": {
            "note": "For this file, Year 1 / Year 2 are operational-year totals from year_number=1/2.",
            "assigned_cluster_counts_top15": _top_counts(lt_cluster_counts),
            "cluster_53_peer_example": _cluster_example(53, lt_peer_lookup, 1, 2),
            "year_1_peer_range_check": _range_eval(lt_y1_rows),
            "year_2_peer_range_check": _range_eval(lt_y2_rows),
            "combined_years_1_2_peer_range_check": _range_eval(lt_y1_rows + lt_y2_rows),
        },
        "more_than_2yrs_literal_analogy": {
            "note": "Compare predicted Year 1 to peer actual 2024 range and predicted Year 2 to peer actual 2025 range.",
            "assigned_cluster_counts_top15": _top_counts(gt_cluster_counts),
            "cluster_53_peer_example": _cluster_example(53, gt_peer_lookup, 2024, 2025),
            "year_1_vs_2024_peer_range_check": _range_eval(gt_lit_y1_rows),
            "year_2_vs_2025_peer_range_check": _range_eval(gt_lit_y2_rows),
            "combined_peer_range_check": _range_eval(gt_lit_y1_rows + gt_lit_y2_rows),
        },
        "more_than_2yrs_mature_analogy": {
            "note": "Compare predicted Year 3 to peer actual 2024 range and predicted Year 4 to peer actual 2025 range.",
            "year_3_vs_2024_peer_range_check": _range_eval(gt_mat_y3_rows),
            "year_4_vs_2025_peer_range_check": _range_eval(gt_mat_y4_rows),
            "combined_peer_range_check": _range_eval(gt_mat_y3_rows + gt_mat_y4_rows),
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {OUT_JSON.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
