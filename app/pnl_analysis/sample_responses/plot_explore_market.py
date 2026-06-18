"""
Plot a TAB 1 (explore-market) response.

Run it:
    python plot_explore_market.py                                  # uses explore_market.json
    python plot_explore_market.py explore_market_months_since_open.json
    python plot_explore_market.py some_response.json --save out.png

Reads the parallel arrays from the JSON and draws one line per site, with
entrants emphasized and a vertical marker at each entrant's open date.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

HERE = Path(__file__).resolve().parent


def plot_explore_market(resp: dict, ax=None):
    """Draw an explore-market response onto `ax` (creates one if None). Returns the ax."""
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 6))

    is_date = resp.get("x_axis") == "date"

    def to_x(xs):
        return [datetime.strptime(x, "%Y-%m-%d") for x in xs] if is_date else xs

    for s in resp.get("series", []):
        entrant = s.get("role") == "entrant"
        x = to_x(s["x"])
        ax.plot(
            x, s["y"],
            label=f'{s["name"]} ({s["role"]}, {s["dist_km"]}km)',
            linewidth=2.6 if entrant else 1.4,
            alpha=1.0 if entrant else 0.7,
            zorder=3 if entrant else 2,
        )

    # vertical markers at entrant open dates (only meaningful on a date axis)
    if is_date:
        for m in resp.get("entry_markers", []):
            xd = datetime.strptime(m["op_start"], "%Y-%m-%d")
            ax.axvline(xd, color="crimson", linestyle="--", alpha=0.5, linewidth=1)
            ax.text(xd, ax.get_ylim()[1], f' opened: {m["name"]}',
                    rotation=90, va="top", ha="left", fontsize=7, color="crimson")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    title = (f'Local market — {resp.get("metric_label", "")}  '
             f'(r={resp.get("radius_km")}km, {resp.get("n_shown")}/{resp.get("n_sites_in_market")} sites)')
    ax.set_title(title)
    ax.set_xlabel(resp.get("x_axis_label", "x"))
    ax.set_ylabel(resp.get("y_axis_label", "value"))
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.25)
    return ax


def _positional(argv):
    """argv minus flags and the value that follows --save."""
    out, skip = [], False
    for a in argv[1:]:
        if skip:
            skip = False
            continue
        if a == "--save":
            skip = True
            continue
        if a.startswith("--"):
            continue
        out.append(a)
    return out


def _load(argv):
    args = _positional(argv)
    path = Path(args[0]) if args else HERE / "explore_market.json"
    if not path.is_absolute():
        path = HERE / path
    with open(path) as f:
        return json.load(f), path


if __name__ == "__main__":
    resp, path = _load(sys.argv)
    print(f"Plotting {path.name}: {resp.get('n_shown')} site(s), metric={resp.get('metric')}")
    plot_explore_market(resp)
    plt.tight_layout()
    if "--save" in sys.argv:
        out = sys.argv[sys.argv.index("--save") + 1]
        plt.savefig(out, dpi=130, bbox_inches="tight")
        print("saved", out)
    else:
        plt.show()
