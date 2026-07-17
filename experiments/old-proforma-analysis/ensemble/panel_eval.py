"""Coldstart LOO predictions for the 862-site panel-wide eligible set.

Same leave-one-out artifact surgery as the proforma benchmark: the target site and any
sites_rl record <=0.2km are removed before predicting, production config (model_kind='et',
local market-trend drift). Outputs per-year trajectory means + plateau + anchor metadata.
"""
import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/dhruvsood/sonnysDataCollection")
from proforma.models import coldstart as cm            # noqa: E402
import proforma.pnl.data as D                          # noqa: E402
from app.pnl_analysis.modelling.market import market_trend  # noqa: E402

SCRATCH = "/Users/dhruvsood/sonnysDataCollection/experiments/old-proforma-analysis/ensemble/results"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=10 ** 9)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sites = pd.read_csv(f"{SCRATCH}/panel_eligible.csv").iloc[a.start:a.end]
    df, sitetab = D.load_panel(False)
    art = D.load_model()
    rl = art["sites_rl"]
    nb = sitetab[sitetab.has_coords]
    rows = []
    for _, r in sites.iterrows():
        try:
            lat, lon = float(r.lat), float(r.lon)
            d_rl = cm._haversine(lat, lon, rl.lat.values, rl.lon.values)
            excl = set(rl.site_key[d_rl <= 0.2].tolist()) | {r.key}
            d_nb = cm._haversine(lat, lon, nb.lat.values, nb.lon.values)
            keys = [k for k in nb.site_key[(d_nb <= 20.0) & (d_nb > 1e-6)].tolist() if k not in excl]
            if keys:
                sub = df[df.site_key.isin(keys)]
                pm = sub.pivot_table(index="date", columns="site_key", values="mem_wash_count")
                pr = sub.pivot_table(index="date", columns="site_key", values="ret_wash_count")
                mem_g, mem_lo, mem_hi = market_trend(pm)
                ret_g, ret_lo, ret_hi = market_trend(pr)
            else:
                mem_g = mem_lo = mem_hi = ret_g = ret_lo = ret_hi = 0.0
            art2 = {**art, "sites_rl": rl[~rl.site_key.isin(excl)].reset_index(drop=True)}
            traj, info = cm.predict_site(
                lat, lon, brand=None,
                annual_mem_growth=mem_g, annual_ret_change=ret_g,
                mem_growth_band=(mem_lo, mem_hi), ret_change_band=(ret_lo, ret_hi),
                horizon=60, art=art2, model_kind="et")
            t = traj.set_index("month")["total_med"]
            out = {"key": r.key, "n_excluded": len(excl)}
            for y in range(1, 6):
                out[f"pred_y{y}"] = float(t.loc[12 * (y - 1):12 * y - 1].mean())
            out.update(plateau=info["plateau_med"], model_plateau=info["model_plateau"],
                       anchor_level=info["anchor_level"], n_local_mature=info["n_local_mature"],
                       proxy_used=bool(info["proxy_used"]), region=info["region"])
            rows.append(out)
        except Exception as e:
            rows.append({"key": r.key, "error": str(e)[:200]})
    pd.DataFrame(rows).to_csv(a.out, index=False)
    errs = sum(1 for x in rows if "error" in x and x.get("error"))
    print(f"wrote {len(rows)} rows ({errs} errors) -> {a.out}")

if __name__ == "__main__":
    main()
