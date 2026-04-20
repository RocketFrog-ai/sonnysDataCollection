#!/usr/bin/env python3
"""
Run V1 projection API + V2 Ridge + V2 quantile for one site; save one combined PNG + JSON.

Geocodes once (V2 helpers), then POSTs lat/lon to V1 so all three align on the same coordinates.

Example:
  python run_projection_bundle.py --address "5360 Laurel Springs Pkwy, Suwanee, GA 30024"
  python run_projection_bundle.py --address "..." --base-url http://localhost:8001 --radius 12km
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests

MODELLING_ROOT = Path(__file__).resolve().parent
REPO_ROOT = MODELLING_ROOT.parents[2]
V1_DIR = MODELLING_ROOT / "clustering_v1"
V2_DIR = MODELLING_ROOT / "clustering_v2"
OUT_DEFAULT = MODELLING_ROOT / "results" / "projection_bundle"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(V2_DIR))

import project_site as v2_site  # noqa: E402
import project_site_quantile as v2_q  # noqa: E402


def _load_v1_demo_helpers():
    p = V1_DIR / "run_projection_demo.py"
    spec = importlib.util.spec_from_file_location("run_projection_demo_v1", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._extract_bars, mod._segment_status


_extract_bars_v1, _segment_status_v1 = _load_v1_demo_helpers()


def _draw_v1_row(
    axes: tuple[Any, Any],
    payload: dict[str, Any],
) -> None:
    err = payload.get("_bundle_error")
    if err:
        for ax in axes:
            ax.text(0.5, 0.5, str(err), ha="center", va="center", fontsize=8, wrap=True, transform=ax.transAxes)
            ax.set_title("V1 API (failed)", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        return

    segments = [
        (axes[0], "less_than_2yrs", "#16a34a"),
        (axes[1], "more_than_2yrs", "#2563eb"),
    ]
    for ax, name, color in segments:
        bars = _extract_bars_v1(payload, name)
        if not bars:
            ax.set_title(name, fontsize=9)
            ax.text(
                0.5,
                0.5,
                _segment_status_v1(payload, name) or "no bars",
                ha="center",
                va="center",
                fontsize=8,
                transform=ax.transAxes,
                wrap=True,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        x = [str(b["horizon_months"]) for b in bars]
        y = [b["wash_count"] for b in bars]
        rects = ax.bar(x, y, color=color, alpha=0.9)
        dist = (payload.get(name) or {}).get("distance_to_cluster_km")
        gate = (payload.get(name) or {}).get("within_radius_gate_km")
        sub = f"dist={dist:.1f}km" if isinstance(dist, (int, float)) and np.isfinite(dist) else ""
        if gate is False and sub:
            sub += " (outside peer gate)"
        ax.set_title(f"V1 | {name}\n{sub}", fontsize=9)
        ax.set_xlabel("Horizon (mo)", fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        for r, v in zip(rects, y):
            ax.text(
                r.get_x() + r.get_width() / 2.0,
                r.get_height(),
                f"{v:,.0f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )


def _draw_v2_ridge_row(axes: tuple[Any, Any], resp: dict[str, Any]) -> None:
    for ax, cohort_key, title in [
        (axes[0], "less_than_2yrs", "<2y Ridge | yrs 1–2"),
        (axes[1], "more_than_2yrs", ">2y Ridge | mo 30–48"),
    ]:
        block = resp.get(cohort_key, {})
        if "error" in block or "horizons" not in block:
            ax.text(0.5, 0.5, block.get("error", "no data"), ha="center", fontsize=8, transform=ax.transAxes)
            ax.set_title(f"V2 | {title}", fontsize=9)
            continue
        hz = block["horizons"]
        labels = ["30m", "36m", "42m", "48m"] if cohort_key == "more_than_2yrs" and "30m" in hz else ["6m", "12m", "18m", "24m"]
        vals = [hz[h]["six_month_period_sum"] for h in labels]
        ax.bar(labels, vals, color="#3b82f6", alpha=0.9)
        c = block["cluster"]
        ax.set_title(
            f"V2 | {title}\ncl {c['cluster_id']} n={c['size']} {c['distance_km']:.1f}km",
            fontsize=9,
        )
        ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{int(v):,}", ha="center", va="bottom", fontsize=7)


def _draw_v2_quantile_row(axes: tuple[Any, Any], resp: dict[str, Any]) -> None:
    for ax, key, title in [
        (axes[0], "less_than_2yrs", "<2y Q | yrs 1–2"),
        (axes[1], "more_than_2yrs", ">2y Q | mo 30–48"),
    ]:
        b = resp.get(key) or {}
        if "error" in b or "horizons" not in b:
            ax.text(0.5, 0.5, b.get("error", "no data"), ha="center", fontsize=8, transform=ax.transAxes, wrap=True)
            ax.set_title(f"V2 quantile | {title}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        hz = b["horizons"]
        labels = ["30m", "36m", "42m", "48m"] if key == "more_than_2yrs" and "30m" in hz else ["6m", "12m", "18m", "24m"]
        mids = [hz[k]["six_month_period_q50"] for k in labels]
        lows = [hz[k]["six_month_period_q10"] for k in labels]
        highs = [hz[k]["six_month_period_q90"] for k in labels]
        yerr = np.array([np.array(mids) - np.array(lows), np.array(highs) - np.array(mids)])
        ax.bar(labels, mids, color="#16a34a", alpha=0.85)
        ax.errorbar(labels, mids, yerr=yerr, fmt="none", ecolor="black", elinewidth=1.0, capsize=3)
        c = b["cluster"]
        ax.set_title(
            f"V2 quantile | {title}\ncl {c['cluster_id']} n={c['size']} {c['distance_km']:.1f}km",
            fontsize=9,
        )
        ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(mids):
            ax.text(i, v, f"{int(v):,}", ha="center", va="bottom", fontsize=7)


def _fetch_v1(
    base_url: str, radius: str, method: str, lat: float, lon: float
) -> dict[str, Any]:
    endpoint = base_url.rstrip("/") + "/v1/cluster/standalone/projection"
    body = {"latitude": lat, "longitude": lon, "radius": radius, "method": method}
    r = requests.post(endpoint, json=body, timeout=120)
    if r.status_code >= 400:
        raise RuntimeError(f"V1 API {r.status_code}: {r.text}")
    return r.json()


def _build_quantile_resp(lat: float, lon: float, addr: str | None, method: str) -> dict[str, Any]:
    resp = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "radius_km": 12.0,
        "method": method,
        "input": {"address": addr, "lat": lat, "lon": lon},
        "more_than_2yrs": v2_q._project_cohort_quantile("more_than", lat, lon, method, 30, True),
        "less_than_2yrs": v2_q._project_cohort_quantile("less_than", lat, lon, method, 30, False),
    }
    v2_q._bridge_quantile_mature_to_opening_last_month(resp)
    v2_q._enrich_brand_new_site_timeline_quantile(resp)
    return resp


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V1 API + V2 Ridge + V2 quantile → one PNG + JSON bundle.")
    p.add_argument("--address", type=str, default=None)
    p.add_argument("--lat", type=float, default=None)
    p.add_argument("--lon", type=float, default=None)
    p.add_argument("--radius", type=str, default="12km", choices=["12km", "18km"])
    p.add_argument("--method", type=str, default="blend", choices=["blend", "arima", "holt_winters"])
    p.add_argument("--base-url", type=str, default="http://localhost:8001")
    p.add_argument("--output-dir", type=str, default=str(OUT_DEFAULT))
    p.add_argument("--out-name", type=str, default=None, help="filename tag (default: timestamp)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    lat, lon, addr = v2_site._resolve_latlon(args.address, args.lat, args.lon)

    try:
        v1_payload = _fetch_v1(args.base_url, args.radius, args.method, lat, lon)
    except Exception as e:
        v1_payload = {
            "radius": args.radius,
            "method": args.method,
            "more_than_2yrs": {},
            "less_than_2yrs": {},
            "_bundle_error": f"V1 API unreachable or error: {e}",
        }
    else:
        v1_payload["radius"] = args.radius
        v1_payload["method"] = args.method

    v2_resp = v2_site.run_projection(addr, lat, lon, args.method)
    q_resp = _build_quantile_resp(lat, lon, addr, args.method)

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.out_name or datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    bundle_path = out_dir / f"projection_bundle_{args.radius}_{args.method}_{tag}.json"
    png_path = out_dir / f"projection_bundle_{args.radius}_{args.method}_{tag}.png"

    bundle_doc = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "input": {"address": addr, "lat": lat, "lon": lon},
        "radius": args.radius,
        "method": args.method,
        "v1_projection_api": v1_payload,
        "v2_ridge_local": v2_resp,
        "v2_quantile_local": q_resp,
    }
    bundle_path.write_text(json.dumps(bundle_doc, indent=2, default=str), encoding="utf-8")

    # sharey='row': left/right in each row use the same y limits so <2y vs >2y bars are comparable.
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True, sharey="row")
    _draw_v1_row((axes[0, 0], axes[0, 1]), v1_payload)
    _draw_v2_ridge_row((axes[1, 0], axes[1, 1]), v2_resp)
    _draw_v2_quantile_row((axes[2, 0], axes[2, 1]), q_resp)
    for ax in axes.ravel():
        if ax.get_ylabel() == "" and ax.has_data():
            ax.set_ylabel("washes / 6-mo period", fontsize=8)
    loc = addr or f"{lat:.4f}, {lon:.4f}"
    fig.suptitle(
        f"Projection bundle | {loc} | V1 {args.radius} + V2 12km | method={args.method}",
        fontsize=12,
    )
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    def _rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(REPO_ROOT.resolve()))
        except ValueError:
            return str(p.resolve())

    print(f"lat={lat:.5f} lon={lon:.5f}")
    print(f"wrote {_rel(bundle_path)}")
    print(f"wrote {_rel(png_path)}")


if __name__ == "__main__":
    main()
