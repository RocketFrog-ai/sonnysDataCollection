"""
Plot a TAB 2 (pinpoint-forecast) response.

Run it:
    python plot_pinpoint_forecast.py                                  # uses pinpoint_forecast.json
    python plot_pinpoint_forecast.py pinpoint_forecast_no_neighbours.json
    python plot_pinpoint_forecast.py some_response.json --save out.png

Draws two charts side by side:
  A) the new site's own 5-yr trajectory (total med + P10-P90 band, mem/ret split)
  B) the whole local market: history + forecast, with vs without the new site
Plus the KPI summary printed to the console.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE = Path(__file__).resolve().parent


def _dates(strs):
    return [datetime.strptime(s, "%Y-%m-%d") for s in strs]


def plot_trajectory(resp: dict, ax):
    """Chart A — the new site alone."""
    t = resp["trajectory"]
    m = t["months"]
    ax.fill_between(m, t["total_lo"], t["total_hi"], alpha=0.18, color="tab:blue",
                    label="P10-P90 band")
    ax.plot(m, t["total_med"], color="tab:blue", linewidth=2.5, label="total (median)")
    ax.plot(m, t["mem_med"], color="tab:green", linewidth=1.4, linestyle="--", label="membership")
    ax.plot(m, t["ret_med"], color="tab:orange", linewidth=1.4, linestyle="--", label="retail")
    s = resp["summary"]
    ax.axhline(s["plateau_med"], color="grey", linestyle=":", alpha=0.7,
               label=f'plateau ~{s["plateau_med"]:.0f}/mo')
    ax.set_title("A) New site — 5-yr trajectory")
    ax.set_xlabel("months since open")
    ax.set_ylabel("washes / month")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)


def plot_market(resp: dict, ax):
    """Chart B — the whole local market: history + forecast."""
    mf = resp["market_forecast"]
    hist, fc = mf["history"], mf["forecast"]

    if hist["dates"]:
        ax.plot(_dates(hist["dates"]), hist["values"], color="black",
                linewidth=1.8, label="history (actual)")

    fd = _dates(fc["dates"])
    ax.fill_between(fd, fc["band_lo"], fc["band_hi"], alpha=0.15, color="tab:blue",
                    label="forecast band")
    ax.plot(fd, fc["with_new_site"], color="tab:blue", linewidth=2.4, label="with new site")
    ax.plot(fd, fc["without_new_site"], color="tab:red", linewidth=1.8, linestyle="--",
            label="without new site (baseline)")
    ax.plot(fd, fc["new_entrant_journey"], color="tab:green", linewidth=1.2, linestyle=":",
            label="new entrant contribution")

    od = mf.get("open_date")
    if od:
        ax.axvline(datetime.strptime(od, "%Y-%m-%d"), color="crimson",
                   linestyle="--", alpha=0.5, linewidth=1)

    nb = "with neighbours" if mf.get("has_neighbours") else "NO neighbours in radius"
    ax.set_title(f'B) Local market — history + forecast ({nb})\n'
                 f'net change yr5: {mf.get("net_change_year5"):+.0f}/mo')
    ax.set_xlabel("date")
    ax.set_ylabel("market washes / month")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)


def plot_pinpoint_forecast(resp: dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    plot_trajectory(resp, ax1)
    plot_market(resp, ax2)
    s = resp["summary"]
    fig.suptitle(f'Pinpoint forecast — brand={resp.get("brand")}  '
                 f'(plateau {s["plateau_lo"]:.0f}-{s["plateau_hi"]:.0f}/mo, '
                 f'mem_share {s["mem_share"]:.0%}, {s["n_neighbours_20km"]} neighbours)',
                 fontsize=12)
    return fig


def _summary(resp):
    s = resp["summary"]
    print(f"  plateau_med ......... {s['plateau_med']:.0f} washes/mo "
          f"(lo {s['plateau_lo']:.0f} / hi {s['plateau_hi']:.0f})")
    print(f"  mem_share ........... {s['mem_share']:.0%}")
    print(f"  neighbours (20km) ... {s['n_neighbours_20km']}")
    print(f"  brand_known ......... {s['brand_known']}   ramp_source: {s['ramp_source']}")
    print(f"  growth mem/ret ...... {s['mem_growth']:+.1%} / {s['ret_growth']:+.1%}")
    print(f"  net change yr5 ...... {resp['market_forecast']['net_change_year5']:+.0f}/mo")


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
    path = Path(args[0]) if args else HERE / "pinpoint_forecast.json"
    if not path.is_absolute():
        path = HERE / path
    with open(path) as f:
        return json.load(f), path


if __name__ == "__main__":
    resp, path = _load(sys.argv)
    print(f"Plotting {path.name}:")
    _summary(resp)
    plot_pinpoint_forecast(resp)
    plt.tight_layout()
    if "--save" in sys.argv:
        out = sys.argv[sys.argv.index("--save") + 1]
        plt.savefig(out, dpi=130, bbox_inches="tight")
        print("saved", out)
    else:
        plt.show()
