from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call projection endpoint for a sample site and plot 6/12/18/24 month bars."
    )
    parser.add_argument(
        "--address",
        type=str,
        default="5360 Laurel Springs Pkwy, Suwanee, GA 30024",
        help="Sample input site address.",
    )
    parser.add_argument(
        "--radius",
        type=str,
        default="12km",
        choices=["12km", "18km"],
        help="Cluster radius.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="blend",
        choices=["blend", "arima", "holt_winters"],
        help="Projection method.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8001",
        help="FastAPI base URL.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "results" / "projection_demo"),
        help="Directory to save JSON and plot.",
    )
    return parser.parse_args()


def _extract_bars(payload: Dict[str, Any], key: str) -> List[Dict[str, float]]:
    segment = payload.get(key) or {}
    proj = segment.get("projection") or {}
    bars = proj.get("bar_graph_data") or []
    out = []
    for b in bars:
        h = b.get("horizon_months")
        v = b.get("wash_count")
        if h is None or v is None:
            continue
        out.append({"horizon_months": int(h), "wash_count": float(v)})
    return out


def _plot_bars(payload: Dict[str, Any], out_path: Path) -> None:
    bars_more = _extract_bars(payload, "more_than_2yrs")
    bars_less = _extract_bars(payload, "less_than_2yrs")

    if not bars_more and not bars_less:
        raise ValueError("No bar_graph_data present in projection response.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    segments = [
        ("more_than_2yrs", bars_more, "#2563eb"),
        ("less_than_2yrs", bars_less, "#16a34a"),
    ]

    for ax, (name, bars, color) in zip(axes, segments):
        if not bars:
            ax.set_title(f"{name} (no projection data)")
            ax.axis("off")
            continue
        x = [str(b["horizon_months"]) for b in bars]
        y = [b["wash_count"] for b in bars]
        rects = ax.bar(x, y, color=color, alpha=0.9)
        ax.set_title(name)
        ax.set_xlabel("Horizon (months)")
        ax.set_ylabel("Projected cumulative wash count")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        for r, v in zip(rects, y):
            ax.text(
                r.get_x() + r.get_width() / 2.0,
                r.get_height(),
                f"{v:,.0f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    title = (
        f"Projection Bars | radius={payload.get('radius')} | method={payload.get('method')}"
    )
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    endpoint = args.base_url.rstrip("/") + "/v1/cluster/standalone/projection"
    body = {"address": args.address, "radius": args.radius, "method": args.method}

    resp = requests.post(endpoint, json=body, timeout=60)
    resp.raise_for_status()
    payload = resp.json()

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = out_dir / f"projection_response_{args.radius}_{args.method}_{ts}.json"
    png_path = out_dir / f"projection_bars_{args.radius}_{args.method}_{ts}.png"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    _plot_bars(payload, png_path)

    print(f"Endpoint: {endpoint}")
    print(f"Address:  {args.address}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved plot: {png_path}")


if __name__ == "__main__":
    main()
