#!/usr/bin/env python3
"""
V2 quantile projection only: geocode → nearest cluster → quantile models + TS forecast;
save yearly PNG + JSON.

Example:
  python run_projection_bundle.py --address "5360 Laurel Springs Pkwy, Suwanee, GA 30024"
  python run_projection_bundle.py --address "..." --compare-opening-prefix
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODELLING_ROOT = Path(__file__).resolve().parent
REPO_ROOT = MODELLING_ROOT.parents[2]
V2_DIR = MODELLING_ROOT / "clustering_v2"
OUT_DEFAULT = MODELLING_ROOT / "results" / "projection_bundle"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(V2_DIR))

import project_site_quantile as v2_q  # noqa: E402


def _v2_quantile_calendar_year_series(resp: dict[str, Any]) -> tuple[list[float], list[float], list[float]] | None:
    cy = resp.get("calendar_year_washes") or {}
    if not all(f"year_{i}" in cy for i in (1, 2, 3, 4)):
        return None
    mids = [float(cy[f"year_{i}"]["q50"]) for i in (1, 2, 3, 4)]
    lows = [float(cy[f"year_{i}"]["q10"]) for i in (1, 2, 3, 4)]
    highs = [float(cy[f"year_{i}"]["q90"]) for i in (1, 2, 3, 4)]
    return mids, lows, highs


def _draw_yearly_quantile_four_bars(ax: Any, resp: dict[str, Any], title: str) -> None:
    t = _v2_quantile_calendar_year_series(resp)
    labels = ["Year 1", "Year 2", "Year 3", "Year 4"]
    if t is None:
        ax.text(0.5, 0.5, "No yearly data", ha="center", fontsize=8, transform=ax.transAxes)
        ax.set_title(title, fontsize=9)
        return
    mids, lows, highs = t
    yerr = np.array([np.array(mids) - np.array(lows), np.array(highs) - np.array(mids)])
    ax.bar(labels, mids, color="#16a34a", alpha=0.88)
    ax.errorbar(labels, mids, yerr=yerr, fmt="none", ecolor="black", elinewidth=1.0, capsize=3)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("Washes in year (12-mo sum)", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(mids):
        ax.text(i, v, f"{int(v):,}", ha="center", va="bottom", fontsize=7)


def _build_quantile_resp(
    lat: float,
    lon: float,
    addr: str | None,
    method: str,
    *,
    use_opening_prefix_for_mature_forecast: bool = True,
) -> dict[str, Any]:
    return v2_q.build_quantile_projection_response(
        lat, lon, method, addr, use_opening_prefix_for_mature_forecast=use_opening_prefix_for_mature_forecast
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V2 quantile projection → yearly PNG + JSON.")
    p.add_argument("--address", type=str, default=None)
    p.add_argument("--lat", type=float, default=None)
    p.add_argument("--lon", type=float, default=None)
    p.add_argument("--radius", type=str, default="12km", choices=["12km", "18km"])
    p.add_argument("--method", type=str, default="blend", choices=["blend", "arima", "holt_winters"])
    p.add_argument("--output-dir", type=str, default=str(OUT_DEFAULT))
    p.add_argument("--out-name", type=str, default=None, help="filename tag (default: timestamp)")
    p.add_argument(
        "--no-opening-prefix",
        action="store_true",
        help="Do not append <2y q50 monthly series as context before >2y TS extrapolation.",
    )
    p.add_argument(
        "--compare-opening-prefix",
        action="store_true",
        help="Write comparison PNG+JSON: left=no prefix, right=with prefix (quantile only).",
    )
    return p.parse_args()


def _plot_quantile_yearly_single(q_resp: dict[str, Any], *, suptitle: str, out_path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(9, 4.5), constrained_layout=True)
    use_pfx = q_resp.get("use_opening_prefix_for_mature_forecast", True)
    pfx = "with <2y q50 prefix for >2y TS" if use_pfx else "no <2y prefix for >2y TS"
    _draw_yearly_quantile_four_bars(ax, q_resp, f"V2 quantile | calendar years 1–4 ({pfx})")
    fig.suptitle(suptitle, fontsize=11)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_quantile_yearly_compare(q_off: dict[str, Any], q_on: dict[str, Any], *, suptitle: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True, sharey=True)
    _draw_yearly_quantile_four_bars(axes[0], q_off, "V2 quantile | no <2y prefix")
    _draw_yearly_quantile_four_bars(axes[1], q_on, "V2 quantile | with <2y q50 prefix")
    fig.suptitle(suptitle, fontsize=11)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    lat, lon, addr = v2_q.base._resolve_latlon(args.address, args.lat, args.lon)
    use_pfx = not args.no_opening_prefix

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.out_name or datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    loc = addr or f"{lat:.4f}, {lon:.4f}"
    suptitle = f"V2 quantile | {loc} | 12km | method={args.method}"

    if args.compare_opening_prefix:
        q_off = _build_quantile_resp(lat, lon, addr, args.method, use_opening_prefix_for_mature_forecast=False)
        q_on = _build_quantile_resp(lat, lon, addr, args.method, use_opening_prefix_for_mature_forecast=True)
        bundle_path = out_dir / f"projection_bundle_{args.radius}_{args.method}_{tag}_compare_prefix.json"
        png_path = out_dir / f"projection_bundle_{args.radius}_{args.method}_{tag}_compare_prefix.png"
        bundle_path.write_text(
            json.dumps(
                {
                    "generated_at": datetime.utcnow().isoformat() + "Z",
                    "input": {"address": addr, "lat": lat, "lon": lon},
                    "radius": args.radius,
                    "method": args.method,
                    "model": "v2_quantile_only",
                    "opening_prefix_variant": "side_by_side_off_vs_on",
                    "v2_quantile_local_no_prefix": q_off,
                    "v2_quantile_local_with_prefix": q_on,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        _plot_quantile_yearly_compare(
            q_off, q_on, suptitle=suptitle + " | compare opening→mature prefix", out_path=png_path
        )
    else:
        q_resp = _build_quantile_resp(lat, lon, addr, args.method, use_opening_prefix_for_mature_forecast=use_pfx)
        bundle_path = out_dir / f"projection_bundle_{args.radius}_{args.method}_{tag}.json"
        png_path = out_dir / f"projection_bundle_{args.radius}_{args.method}_{tag}.png"
        bundle_doc = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "input": {"address": addr, "lat": lat, "lon": lon},
            "radius": args.radius,
            "method": args.method,
            "model": "v2_quantile_only",
            "use_opening_prefix_for_mature_forecast": use_pfx,
            "v2_quantile_local": q_resp,
        }
        bundle_path.write_text(json.dumps(bundle_doc, indent=2, default=str), encoding="utf-8")
        _plot_quantile_yearly_single(q_resp, suptitle=suptitle, out_path=png_path)

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
