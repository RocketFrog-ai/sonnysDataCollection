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

# reuse robust helpers from ridge projection flow
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
) -> dict[str, Any]:
    # Supporting artifacts from portable V2 folder (centroids/context/series)
    support = base._load_cohort(cohort_key)

    # Quantile model artifacts
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
    if np.isfinite(d_km) and d_km > base.MAX_NEAREST_CLUSTER_DISPLAY_KM:
        return {
            "error": (
                f"Nearest cluster is {d_km:.1f} km away; projections hidden when centroid distance "
                f"exceeds {base.MAX_NEAREST_CLUSTER_DISPLAY_KM:.0f} km."
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
    q_raw = _predict_quantiles(models, feature_order, vec)

    if is_daily_model:
        q_monthly = {k: v * days_per_month for k, v in q_raw.items()}
    else:
        q_monthly = dict(q_raw)

    # keep quantiles ordered
    q_monthly["q10"] = min(q_monthly["q10"], q_monthly["q50"])
    q_monthly["q90"] = max(q_monthly["q90"], q_monthly["q50"])

    series = base._series_to_df(support["series"]["series"].get(str(cluster_id), []))
    cluster_level = float(series.tail(6).mean()) if len(series) else float("nan")

    fc_base = base._forecast(series, 24, method)

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

    # monotonic bounds
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
        },
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


def _bridge_quantile_mature_to_opening_last_month(resp: dict[str, Any]) -> None:
    """Scale >2y q10/q50/q90 monthlies so operational month 25 matches <2y month 24 (q50 handoff)."""
    lt = resp.get("less_than_2yrs")
    gt = resp.get("more_than_2yrs")
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


def _plot(resp: dict[str, Any], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, key, title in [
        (axes[0], "less_than_2yrs", "<2y quantile | years 1–2 (mo 1–24)"),
        (axes[1], "more_than_2yrs", ">2y quantile | continuation mo 30–48"),
    ]:
        b = resp[key]
        if "error" in b or "horizons" not in b:
            ax.text(0.5, 0.5, b.get("error", "no data"), ha="center", fontsize=9, transform=ax.transAxes)
            ax.set_title(title)
            continue
        hz = b["horizons"]
        if key == "more_than_2yrs" and "30m" in hz:
            labels = ["30m", "36m", "42m", "48m"]
        else:
            labels = ["6m", "12m", "18m", "24m"]
        mids = [hz[k]["six_month_period_q50"] for k in labels]
        lows = [hz[k]["six_month_period_q10"] for k in labels]
        highs = [hz[k]["six_month_period_q90"] for k in labels]
        yerr = np.array([np.array(mids) - np.array(lows), np.array(highs) - np.array(mids)])

        bars = ax.bar(labels, mids, color="#16a34a", alpha=0.85)
        ax.errorbar(labels, mids, yerr=yerr, fmt='none', ecolor='black', elinewidth=1.5, capsize=4)
        ax.set_title(f"{title}\ncluster {b['cluster']['cluster_id']} ({b['cluster']['size']} sites, {b['cluster']['distance_km']:.1f}km)")
        ax.set_ylabel("Washes in 6-month period")
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, mids):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{int(v):,}", ha='center', va='bottom', fontsize=9)

    inp = resp["input"]
    if inp.get("address"):
        loc = inp["address"]
    else:
        loc = f"{inp['lat']:.4f}, {inp['lon']:.4f}"
    fig.suptitle(f"Quantile projection for {loc} (method={resp['method']})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--address", type=str, default=None)
    ap.add_argument("--lat", type=float, default=None)
    ap.add_argument("--lon", type=float, default=None)
    ap.add_argument("--method", type=str, default="blend", choices=["holt_winters", "arima", "blend"])
    ap.add_argument("--out-name", type=str, default=None)
    args = ap.parse_args()

    lat, lon, addr = base._resolve_latlon(args.address, args.lat, args.lon)

    resp = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "radius_km": 12.0,
        "method": args.method,
        "input": {"address": addr, "lat": lat, "lon": lon},
        "more_than_2yrs": _project_cohort_quantile("more_than", lat, lon, args.method, 30, True),
        "less_than_2yrs": _project_cohort_quantile("less_than", lat, lon, args.method, 30, False),
    }
    _bridge_quantile_mature_to_opening_last_month(resp)
    _enrich_brand_new_site_timeline_quantile(resp)

    DEMO_OUT.mkdir(parents=True, exist_ok=True)
    tag = args.out_name or datetime.utcnow().strftime("%Y%m%d-%H%M%S")
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
