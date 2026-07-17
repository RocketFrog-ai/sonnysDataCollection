"""Run the coldstart pin-forecast at each matched old-proforma site (LOO + leaky variants).

Replicates app/pnl_analysis/modelling/market.compute_trajectory (production config:
model_kind='et', local market trend drift) but with leave-one-out artifact surgery:
the target site, its handoff twin, and any panel record <=0.2km of the pin are removed
from art['sites_rl'] so neighbour features + the local-mature anchor cannot see the
site's own actuals. Trend neighbours likewise exclude those keys.
"""
import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/dhruvsood/sonnysDataCollection")
from proforma.models import coldstart as cm            # noqa: E402
import proforma.pnl.data as D                          # noqa: E402
from app.pnl_analysis.modelling.market import market_trend  # noqa: E402

ROOT = "/Users/dhruvsood/sonnysDataCollection/experiments/old-proforma-analysis"

def ym(x):
    try:
        mo, yr = str(x).split("-"); return int(yr) * 12 + int(mo)
    except Exception:
        return np.nan

def eval_sites():
    s = pd.read_csv(f"{ROOT}/old-proforma-combined.csv")
    site = s[s.match_status.str.startswith("matched") & (s.actuals_suspect != True)].copy()
    site = site.dropna(subset=["match_lat", "match_lon"]).sort_values("source_file").reset_index(drop=True)
    return site

def predict_one(row, df, sitetab, art):
    lat, lon = float(row.match_lat), float(row.match_lon)
    # keys to exclude: primary + secondary handoff segments + anything <=0.2km
    excl_keys = {f"{row.match_client_id}::{int(row.match_site_id)}"}
    if isinstance(row.match_secondary_segments, str) and row.match_secondary_segments:
        for part in row.match_secondary_segments.split(";"):
            c, i = part.rsplit(":", 1)
            excl_keys.add(f"{c}::{int(i)}")
    rl = art["sites_rl"]
    d_rl = cm._haversine(lat, lon, rl.lat.values, rl.lon.values)
    near = rl.site_key[(d_rl <= 0.2)].tolist()
    excl_keys |= set(near)

    # market trend from neighbours (production: <=20km, >1e-6km), minus excluded keys
    nb = sitetab[sitetab.has_coords]
    d_nb = cm._haversine(lat, lon, nb.lat.values, nb.lon.values)
    keys = [k for k in nb.site_key[(d_nb <= 20.0) & (d_nb > 1e-6)].tolist() if k not in excl_keys]
    if keys:
        sub = df[df.site_key.isin(keys)]
        pm = sub.pivot_table(index="date", columns="site_key", values="mem_wash_count")
        pr = sub.pivot_table(index="date", columns="site_key", values="ret_wash_count")
        mem_g, mem_lo, mem_hi = market_trend(pm)
        ret_g, ret_lo, ret_hi = market_trend(pr)
    else:
        mem_g = mem_lo = mem_hi = ret_g = ret_lo = ret_hi = 0.0

    out = {"source_file": row.source_file, "n_excluded": int(len(excl_keys & set(rl.site_key)))}
    for tag, artx in [("loo", {**art, "sites_rl": rl[~rl.site_key.isin(excl_keys)].reset_index(drop=True)}),
                      ("leaky", art)]:
        traj, info = cm.predict_site(
            lat, lon, brand=None,
            annual_mem_growth=mem_g, annual_ret_change=ret_g,
            mem_growth_band=(mem_lo, mem_hi), ret_change_band=(ret_lo, ret_hi),
            horizon=60, art=artx, model_kind="et")
        t = traj.set_index("month")["total_med"]
        for y in range(1, 6):
            out[f"pred_{tag}_y{y}"] = float(t.loc[12 * (y - 1):12 * y - 1].mean())
        out[f"plateau_{tag}"] = info["plateau_med"]
        out[f"proxy_{tag}"] = bool(info["proxy_used"])
        out[f"nmat_{tag}"] = int(info["n_local_mature"])
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=10 ** 9)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sites = eval_sites().iloc[a.start:a.end]
    df, sitetab = D.load_panel(False)
    art = D.load_model()
    rows = []
    for _, r in sites.iterrows():
        try:
            rows.append(predict_one(r, df, sitetab, art))
        except Exception as e:
            rows.append({"source_file": r.source_file, "error": str(e)[:200]})
        print(f"done {r.source_file[:60]}", flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"wrote {len(rows)} rows -> {a.out}")

if __name__ == "__main__":
    main()
