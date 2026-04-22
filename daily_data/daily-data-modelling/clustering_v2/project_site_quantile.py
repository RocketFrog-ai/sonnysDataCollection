"""Quantile projection (q10/q50/q90). Reuses project_site helpers. See APPROACH.md."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

import project_site as base

REPO_ROOT = base.REPO_ROOT
V2_DIR = base.V2_DIR
QMODELS_DIR = V2_DIR / "models_quantile"
DEMO_OUT = V2_DIR / "results" / "projection_demo"


def _load_feature_order(cohort_dir: Path) -> list[str]:
    return json.loads((cohort_dir / "feature_order.json").read_text())


def _predict_quantiles(models: dict[str, Any], feature_order: list[str], vec: dict[str, float]) -> dict[str, float]:
    row = {c: vec.get(c, np.nan) for c in feature_order}
    X = pd.DataFrame([row], columns=feature_order)
    out = {}
    for key, mdl in models.items():
        out[key] = float(mdl.predict(X)[0])
    return out


def _project_cohort_quantile(
    cohort_key: str,
    lat: float,
    lon: float,
    method: str,
    days_per_month: int,
    is_daily_model: bool,
    opening_lt_q50_monthly: list[float] | None = None,
    *,
    allow_nearest_cluster_beyond_distance_cap: bool = False,
) -> dict[str, Any]:
    support = base._load_cohort(cohort_key)

    q_dir = QMODELS_DIR / cohort_key
    feature_order = _load_feature_order(q_dir)
    models = {
        "q10": joblib.load(q_dir / "q10.joblib"),
        "q50": joblib.load(q_dir / "q50.joblib"),
        "q90": joblib.load(q_dir / "q90.joblib"),
    }

    centroids = support["centroids"]["centroids"]
    nearest = base._nearest_cluster(centroids, lat, lon)
    cluster_id = int(nearest["cluster_id"])
    d_km = float(nearest["distance_km"])
    if (
        not allow_nearest_cluster_beyond_distance_cap
        and np.isfinite(d_km)
        and d_km > base.MAX_NEAREST_CLUSTER_DISTANCE_KM
    ):
        return {
            "error": (
                f"Nearest cluster is {d_km:.1f} km away; projections hidden when centroid distance "
                f"exceeds {base.MAX_NEAREST_CLUSTER_DISTANCE_KM:.0f} km (use --allow-distant-nearest-cluster to force)."
            ),
            "cluster": {
                "cluster_id": cluster_id,
                "distance_km": d_km,
                "size": nearest.get("size"),
            },
        }

    ctx_row = None
    for r in support["context"]["records"]:
        if int(r.get("cluster_id", -999)) == cluster_id:
            ctx_row = r
            break

    vec = base._feature_vector(support["spec"], ctx_row, lat, lon, cluster_id)
    vec = base._apply_local_feature_medians(vec, feature_order, ctx_row)
    q_raw = _predict_quantiles(models, feature_order, vec)

    if is_daily_model:
        q_monthly = {k: v * days_per_month for k, v in q_raw.items()}
    else:
        q_monthly = dict(q_raw)

    q_monthly["q10"] = min(q_monthly["q10"], q_monthly["q50"])
    q_monthly["q90"] = max(q_monthly["q90"], q_monthly["q50"])

    series = base._series_to_df(support["series"]["series"].get(str(cluster_id), []))
    cluster_level = float(series.tail(6).mean()) if len(series) else float("nan")

    used_lt_prefix = (
        cohort_key == "more_than"
        and opening_lt_q50_monthly is not None
        and len(opening_lt_q50_monthly) == 24
    )
    forecast_series = (
        base._series_for_mature_forecast_with_opening_context(series, opening_lt_q50_monthly)
        if used_lt_prefix
        else series
    )
    fc_base = base._forecast(forecast_series, 24, method)

    def _scale(target: float) -> float:
        if len(series) >= 6 and np.isfinite(target) and cluster_level > 0:
            return float(target / cluster_level)
        return 1.0

    s10 = _scale(q_monthly["q10"])
    s50 = _scale(q_monthly["q50"])
    s90 = _scale(q_monthly["q90"])

    fc10 = np.maximum(fc_base * s10, 0)
    fc50 = np.maximum(fc_base * s50, 0)
    fc90 = np.maximum(fc_base * s90, 0)

    fc10 = np.minimum(fc10, fc50)
    fc90 = np.maximum(fc90, fc50)

    horizons: dict[str, Any] = {}
    for m in (6, 12, 18, 24):
        lo = m - 6
        s10 = fc10.iloc[lo:m]
        s50 = fc50.iloc[lo:m]
        s90 = fc90.iloc[lo:m]
        horizons[f"{m}m"] = {
            "six_month_period_q10": float(np.maximum(s10, 0).sum()),
            "six_month_period_q50": float(np.maximum(s50, 0).sum()),
            "six_month_period_q90": float(np.maximum(s90, 0).sum()),
            "monthly_avg_q50_in_period": float(np.maximum(s50, 0).mean()) if len(s50) else 0.0,
            "period_operational_month_start": lo + 1,
            "period_operational_month_end": m,
        }

    return {
        "cluster": {
            "cluster_id": cluster_id,
            "distance_km": nearest["distance_km"],
            "size": nearest["size"],
            "cluster_monthly_level_last_6mo": cluster_level if np.isfinite(cluster_level) else None,
            "nearest_cluster_max_distance_km": base.MAX_NEAREST_CLUSTER_DISTANCE_KM,
            "distance_cap_relaxed": bool(allow_nearest_cluster_beyond_distance_cap),
        },
        "mature_forecast_series_mode": (
            "cluster_tail_72m_plus_lt2y_q50_monthly_24m_prefix_then_forecast_25_48"
            if used_lt_prefix
            else "cluster_monthly_only"
        ),
        "used_lt2y_monthly_forecast_as_mature_forecast_context": bool(used_lt_prefix),
        "quantile_prediction": {
            "raw": q_raw,
            "monthly": q_monthly,
            "anchor_scale": {"q10": s10, "q50": s50, "q90": s90},
        },
        "method": method,
        "horizons": horizons,
        "monthly_projection": [
            {
                "month": d.strftime("%Y-%m-%d"),
                "q10": float(fc10.loc[d]),
                "q50": float(fc50.loc[d]),
                "q90": float(fc90.loc[d]),
            }
            for d in fc50.index
        ],
    }


def _bridge_quantile_mature_to_opening_last_month(
    resp: dict[str, Any],
    *,
    skip_bridge_when_prefix: bool = False,
) -> None:
    lt = resp.get("less_than_2yrs")
    gt = resp.get("more_than_2yrs")
    if (
        skip_bridge_when_prefix
        and gt
        and gt.get("used_lt2y_monthly_forecast_as_mature_forecast_context")
    ):
        return
    if not lt or not gt or "error" in lt or "error" in gt:
        return
    lp = lt.get("monthly_projection") or []
    gp = gt.get("monthly_projection") or []
    if len(lp) < 24 or len(gp) < 24:
        return
    lt_last = float(lp[-1].get("q50", 0) or 0)
    gt_first = float(gp[0].get("q50", 0) or 0)
    if not np.isfinite(gt_first) or gt_first <= 1e-12 or not np.isfinite(lt_last):
        return
    factor = float(lt_last / gt_first)
    for row in gp:
        for k in ("q10", "q50", "q90"):
            row[k] = float(max(float(row.get(k, 0) or 0) * factor, 0.0))
        row["q10"] = min(row["q10"], row["q50"])
        row["q90"] = max(row["q90"], row["q50"])
    qp = gt.get("quantile_prediction") or {}
    mon = qp.get("monthly")
    if isinstance(mon, dict):
        for k in ("q10", "q50", "q90"):
            if k in mon and mon[k] is not None and np.isfinite(float(mon[k])):
                mon[k] = float(max(float(mon[k]) * factor, 0.0))
        if all(k in mon for k in ("q10", "q50", "q90")):
            mon["q10"] = min(float(mon["q10"]), float(mon["q50"]))
            mon["q90"] = max(float(mon["q90"]), float(mon["q50"]))
        qp["monthly"] = mon
        gt["quantile_prediction"] = qp
    gt["opening_to_mature_bridge"] = {
        "method": "scale_entire_mature_monthly_track",
        "scale_factor": factor,
        "aligned_month_24_lt_q50_to_month_25_gt_q50": True,
    }


def _slice_sum(values: list[float], lo: int, hi: int) -> float:
    return float(np.sum(np.maximum(np.array(values[lo:hi], dtype=float), 0.0)))


def _enrich_brand_new_site_timeline_quantile(resp: dict[str, Any]) -> None:
    lt = resp.get("less_than_2yrs")
    gt = resp.get("more_than_2yrs")
    if lt and "error" not in lt and lt.get("monthly_projection"):
        for i, row in enumerate(lt["monthly_projection"], start=1):
            row["operational_month_index"] = i
        lt["operational_phase"] = "opening_phase_years_1_2"
        lt["operational_calendar_months"] = "Months 1–24 after site open"

    if not gt or "error" in gt or "monthly_projection" not in gt:
        return
    if not lt or "error" in lt or "horizons" not in lt:
        return

    rows = gt["monthly_projection"]
    if len(rows) < 24:
        return
    gt["monthly_projection_mature_25_48"] = [dict(r) for r in rows]
    q10s = [float(r["q10"]) for r in rows]
    q50s = [float(r["q50"]) for r in rows]
    q90s = [float(r["q90"]) for r in rows]
    new_hz: dict[str, Any] = {}
    for end_m, lo, hi in ((30, 0, 6), (36, 6, 12), (42, 12, 18), (48, 18, 24)):
        p10 = _slice_sum(q10s, lo, hi)
        p50 = _slice_sum(q50s, lo, hi)
        p90 = _slice_sum(q90s, lo, hi)
        new_hz[f"{end_m}m"] = {
            "six_month_period_q10": p10,
            "six_month_period_q50": p50,
            "six_month_period_q90": p90,
            "monthly_avg_q50_in_period": p50 / float(hi - lo) if hi > lo else 0.0,
            "period_operational_month_start": 25 + lo,
            "period_operational_month_end": 25 + hi - 1,
        }
    gt["horizons"] = new_hz
    gt["horizon_definition"] = (
        "Each bar is washes in one 6-month window only. >2y: months 25–30, 31–36, 37–42, 43–48 (labels 30/36/42/48)."
    )
    gt["operational_phase"] = "mature_phase_months_30_48"
    gt["operational_calendar_months"] = "Months 30–48 after site open (>2y quantile view)"

    pub: list[dict[str, Any]] = []
    for i, r in enumerate(rows[5:], start=30):
        nr = dict(r)
        nr["operational_month_index"] = i
        pub.append(nr)
    gt["monthly_projection"] = pub

    hz = gt["horizons"]
    run10 = run50 = run90 = 0.0
    r10: dict[str, float] = {}
    r50: dict[str, float] = {}
    r90: dict[str, float] = {}
    for m in (6, 12, 18, 24):
        lm = lt["horizons"][f"{m}m"]
        run10 += float(lm["six_month_period_q10"])
        run50 += float(lm["six_month_period_q50"])
        run90 += float(lm["six_month_period_q90"])
        r10[str(m)] = run10
        r50[str(m)] = run50
        r90[str(m)] = run90
    for end_m in (30, 36, 42, 48):
        hm = hz[f"{end_m}m"]
        run10 += float(hm["six_month_period_q10"])
        run50 += float(hm["six_month_period_q50"])
        run90 += float(hm["six_month_period_q90"])
        r10[str(end_m)] = run10
        r50[str(end_m)] = run50
        r90[str(end_m)] = run90
    resp["brand_new_site_continuation"] = {
        "narrative": (
            "Bars are independent 6-month period sums. running_total_* fields sum periods in order through 48."
        ),
        "six_month_period_q50_by_label": {
            "opening_6m": float(lt["horizons"]["6m"]["six_month_period_q50"]),
            "opening_12m": float(lt["horizons"]["12m"]["six_month_period_q50"]),
            "opening_18m": float(lt["horizons"]["18m"]["six_month_period_q50"]),
            "opening_24m": float(lt["horizons"]["24m"]["six_month_period_q50"]),
            "mature_30m": float(hz["30m"]["six_month_period_q50"]),
            "mature_36m": float(hz["36m"]["six_month_period_q50"]),
            "mature_42m": float(hz["42m"]["six_month_period_q50"]),
            "mature_48m": float(hz["48m"]["six_month_period_q50"]),
        },
        "running_total_q50_through_operational_month": r50,
        "running_total_q10_through_operational_month": r10,
        "running_total_q90_through_operational_month": r90,
    }


def _append_calendar_year_washes_quantile(resp: dict[str, Any]) -> None:
    lt = resp.get("less_than_2yrs") or {}
    gt = resp.get("more_than_2yrs") or {}
    if "error" in lt or "error" in gt:
        return
    lm = lt.get("monthly_projection") or []
    mature = gt.get("monthly_projection_mature_25_48") or []
    if len(lm) < 24 or len(mature) < 24:
        return

    def _ysum(rows: list[dict[str, Any]], lo: int, hi: int, qk: str) -> float:
        return float(sum(max(float(r.get(qk, 0) or 0), 0.0) for r in rows[lo:hi]))

    resp["calendar_year_washes"] = {
        "year_1": {
            "q10": _ysum(lm, 0, 12, "q10"),
            "q50": _ysum(lm, 0, 12, "q50"),
            "q90": _ysum(lm, 0, 12, "q90"),
        },
        "year_2": {
            "q10": _ysum(lm, 12, 24, "q10"),
            "q50": _ysum(lm, 12, 24, "q50"),
            "q90": _ysum(lm, 12, 24, "q90"),
        },
        "year_3": {
            "q10": _ysum(mature, 0, 12, "q10"),
            "q50": _ysum(mature, 0, 12, "q50"),
            "q90": _ysum(mature, 0, 12, "q90"),
        },
        "year_4": {
            "q10": _ysum(mature, 12, 24, "q10"),
            "q50": _ysum(mature, 12, 24, "q50"),
            "q90": _ysum(mature, 12, 24, "q90"),
        },
        "definition": (
            "Per-year sums of monthly quantile tracks (years 1–2 from <2y; years 3–4 from >2y, months 25–48)."
        ),
    }


def build_quantile_projection_response(
    lat: float,
    lon: float,
    method: str,
    addr: str | None,
    *,
    use_opening_prefix_for_mature_forecast: bool = True,
    bridge_opening_to_mature_when_prefix: bool = True,
    allow_nearest_cluster_beyond_distance_cap: bool = False,
) -> dict[str, Any]:
    lt_block = _project_cohort_quantile(
        "less_than",
        lat,
        lon,
        method,
        30,
        False,
        None,
        allow_nearest_cluster_beyond_distance_cap=allow_nearest_cluster_beyond_distance_cap,
    )
    pfx: list[float] | None = None
    if use_opening_prefix_for_mature_forecast and "error" not in lt_block:
        mp = lt_block.get("monthly_projection") or []
        if len(mp) >= 24:
            pfx = [float(r["q50"]) for r in mp[:24]]
    gt_block = _project_cohort_quantile(
        "more_than",
        lat,
        lon,
        method,
        30,
        True,
        pfx,
        allow_nearest_cluster_beyond_distance_cap=allow_nearest_cluster_beyond_distance_cap,
    )
    resp = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "radius_km": 12.0,
        "method": method,
        "input": {"address": addr, "lat": lat, "lon": lon},
        "nearest_cluster_max_distance_km": base.MAX_NEAREST_CLUSTER_DISTANCE_KM,
        "allow_nearest_cluster_beyond_distance_cap": bool(allow_nearest_cluster_beyond_distance_cap),
        "less_than_2yrs": lt_block,
        "more_than_2yrs": gt_block,
    }
    _bridge_quantile_mature_to_opening_last_month(
        resp,
        skip_bridge_when_prefix=bool(
            use_opening_prefix_for_mature_forecast and not bridge_opening_to_mature_when_prefix
        ),
    )
    _enrich_brand_new_site_timeline_quantile(resp)
    _append_calendar_year_washes_quantile(resp)
    resp["use_opening_prefix_for_mature_forecast"] = bool(use_opening_prefix_for_mature_forecast)
    resp["bridge_opening_to_mature_when_prefix"] = bool(bridge_opening_to_mature_when_prefix)
    return resp


def _plot_quantile_compare_panels(
    resps: tuple[tuple[str, dict[str, Any]], ...],
    out_path: Path | None,
    loc_title: str,
    method: str,
    *,
    return_figure: bool = False,
) -> Figure | None:
    labels = ["Year 1", "Year 2", "Year 3", "Year 4"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    cluster_line = base._nearest_cluster_caption(resps[0][1])
    for ax, (subtitle, resp) in zip(axes, resps):
        cy = resp.get("calendar_year_washes") or {}
        ok = all(f"year_{i}" in cy for i in (1, 2, 3, 4))
        if ok and all(isinstance(cy.get(f"year_{i}"), dict) for i in (1, 2, 3, 4)):
            mids = [float(cy[f"year_{i}"]["q50"]) for i in (1, 2, 3, 4)]
            lows = [float(cy[f"year_{i}"]["q10"]) for i in (1, 2, 3, 4)]
            highs = [float(cy[f"year_{i}"]["q90"]) for i in (1, 2, 3, 4)]
            yerr = np.array([np.array(mids) - np.array(lows), np.array(highs) - np.array(mids)])
            ax.bar(labels, mids, color="#16a34a", alpha=0.85)
            ax.errorbar(labels, mids, yerr=yerr, fmt="none", ecolor="black", elinewidth=1.5, capsize=4)
            ax.set_title(subtitle, fontsize=10)
            ax.grid(axis="y", alpha=0.3)
            for i, v in enumerate(mids):
                ax.text(i, v, f"{int(v):,}", ha="center", va="bottom", fontsize=8)
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", transform=ax.transAxes)
            ax.set_title(subtitle, fontsize=10)
        ax.tick_params(axis="x", rotation=15)
    axes[0].set_ylabel("Washes in calendar year (12-mo sum)")
    fig.suptitle(
        f"V2 Quantile compare (method={method})  |  {loc_title}\nNearest train centroids: {cluster_line}",
        fontsize=10,
        y=1.03,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    if return_figure:
        return fig
    if out_path is None:
        raise ValueError("out_path is required when return_figure is False")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return None


def _plot(resp: dict[str, Any], out_path: Path | None, *, return_figure: bool = False) -> Figure | None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    cy = resp.get("calendar_year_washes") or {}
    if all(f"year_{i}" in cy for i in (1, 2, 3, 4)):
        labels = ["Year 1", "Year 2", "Year 3", "Year 4"]
        mids = [float(cy[f"year_{i}"]["q50"]) for i in (1, 2, 3, 4)]
        lows = [float(cy[f"year_{i}"]["q10"]) for i in (1, 2, 3, 4)]
        highs = [float(cy[f"year_{i}"]["q90"]) for i in (1, 2, 3, 4)]
        yerr = np.array([np.array(mids) - np.array(lows), np.array(highs) - np.array(mids)])
        ax.bar(labels, mids, color="#16a34a", alpha=0.85)
        ax.errorbar(labels, mids, yerr=yerr, fmt="none", ecolor="black", elinewidth=1.5, capsize=4)
        lt = resp.get("less_than_2yrs") or {}
        gt = resp.get("more_than_2yrs") or {}
        pfx = ""
        if gt.get("used_lt2y_monthly_forecast_as_mature_forecast_context"):
            pfx = " | >2y used <2y q50 as TS context"
        ax.set_title(f"V2 quantile | calendar years 1–4{pfx}", fontsize=10)
        ax.set_ylabel("Washes in calendar year (12-mo sum)")
        ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(mids):
            ax.text(i, v, f"{int(v):,}", ha="center", va="bottom", fontsize=9)
    else:
        ax.text(0.5, 0.5, "Insufficient data for calendar-year plot", ha="center", transform=ax.transAxes)

    inp = resp["input"]
    if inp.get("address"):
        loc = inp["address"]
    else:
        loc = f"{inp['lat']:.4f}, {inp['lon']:.4f}"
    fig.suptitle(
        f"Quantile projection for {loc} (method={resp['method']})\nNearest train centroids: {base._nearest_cluster_caption(resp)}",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    if return_figure:
        return fig
    if out_path is None:
        raise ValueError("out_path is required when return_figure is False")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--address", type=str, default=None)
    ap.add_argument("--lat", type=float, default=None)
    ap.add_argument("--lon", type=float, default=None)
    ap.add_argument(
        "--method",
        type=str,
        default="arima",
        choices=["holt_winters", "arima", "blend"],
        help="TS on cluster monthly track (default arima; see results/ts_arima_vs_holt_pick.json).",
    )
    ap.add_argument("--out-name", type=str, default=None)
    ap.add_argument(
        "--no-opening-prefix",
        action="store_true",
        help="Do not append <2y q50 monthly series as context before >2y TS extrapolation.",
    )
    ap.add_argument(
        "--legacy-prefix-no-bridge",
        action="store_true",
        help="With prefix: skip q50 month-24→25 scale on >2y track (old behavior).",
    )
    ap.add_argument(
        "--allow-distant-nearest-cluster",
        action="store_true",
        help=(
            "Always assign nearest train centroid even if >20 km. "
            "Default: refuse when distance exceeds 20 km."
        ),
    )
    ap.add_argument(
        "--plot-two-way",
        action="store_true",
        help="Two panels: no <2y q50 prefix vs prefix + 24→25 bridge. Writes projection_quantile_*_two_way.png/json.",
    )
    ap.add_argument("--plot-three-way", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    lat, lon, addr = base._resolve_latlon(args.address, args.lat, args.lon)
    DEMO_OUT.mkdir(parents=True, exist_ok=True)
    tag = args.out_name or datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    cap_kw = {"allow_nearest_cluster_beyond_distance_cap": args.allow_distant_nearest_cluster}

    plot_compare = args.plot_two_way or args.plot_three_way
    if args.plot_three_way and not args.plot_two_way:
        print(
            "[quantile] --plot-three-way is deprecated (same as --plot-two-way).",
            file=sys.stderr,
        )
    if plot_compare:
        loc_title = addr or f"{lat:.4f},{lon:.4f}"
        r_no = build_quantile_projection_response(
            lat,
            lon,
            args.method,
            addr,
            use_opening_prefix_for_mature_forecast=False,
            bridge_opening_to_mature_when_prefix=True,
            **cap_kw,
        )
        r_prefix = build_quantile_projection_response(
            lat,
            lon,
            args.method,
            addr,
            use_opening_prefix_for_mature_forecast=True,
            bridge_opening_to_mature_when_prefix=True,
            **cap_kw,
        )
        combined = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "method": args.method,
            "input": {"address": addr, "lat": lat, "lon": lon},
            "compare": "no_prefix vs opening_prefix_with_month24_to_25_bridge",
            "no_prefix": r_no,
            "with_opening_prefix_and_bridge": r_prefix,
        }
        json_path = DEMO_OUT / f"projection_quantile_{args.method}_{tag}_two_way.json"
        png_path = DEMO_OUT / f"projection_quantile_{args.method}_{tag}_two_way.png"
        json_path.write_text(json.dumps(combined, indent=2, default=str))
        _plot_quantile_compare_panels(
            (
                ("No <2y q50 TS prefix", r_no),
                ("<2y q50 prefix + 24→25 bridge (default)", r_prefix),
            ),
            png_path,
            loc_title,
            args.method,
        )
        print(f"wrote {json_path.relative_to(REPO_ROOT)}")
        print(f"wrote {png_path.relative_to(REPO_ROOT)}")
        return

    resp = build_quantile_projection_response(
        lat,
        lon,
        args.method,
        addr,
        use_opening_prefix_for_mature_forecast=not args.no_opening_prefix,
        bridge_opening_to_mature_when_prefix=not args.legacy_prefix_no_bridge,
        **cap_kw,
    )

    json_path = DEMO_OUT / f"projection_quantile_{args.method}_{tag}.json"
    png_path = DEMO_OUT / f"projection_quantile_{args.method}_{tag}.png"
    json_path.write_text(json.dumps(resp, indent=2, default=str))
    _plot(resp, png_path)

    print(f"wrote {json_path.relative_to(REPO_ROOT)}")
    print(f"wrote {png_path.relative_to(REPO_ROOT)}")
    for cohort in ("less_than_2yrs", "more_than_2yrs"):
        b = resp[cohort]
        if "error" in b:
            print(f"[{cohort}] {b.get('error', 'error')}")
            continue
        h = b["horizons"]
        c = b["cluster"]
        print(f"[{cohort}] cluster={c['cluster_id']} dist={c['distance_km']:.2f}km size={c['size']}")
        keys = ("30m", "36m", "42m", "48m") if cohort == "more_than_2yrs" and "30m" in h else ("6m", "12m", "18m", "24m")
        for k in keys:
            row = h[k]
            print(
                f"  {k}: q10={row['six_month_period_q10']:,.0f} "
                f"q50={row['six_month_period_q50']:,.0f} q90={row['six_month_period_q90']:,.0f}"
            )


if __name__ == "__main__":
    main()
