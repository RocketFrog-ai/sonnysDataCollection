"""Post-open Bayesian re-forecaster curve -> ensemble/results/postopen_curve.json.

After a site opens, blend the ex-ante prior (the model plateau) with the site's own first k
observed months: de-ramp those months to a mature-equivalent obs_level, then
  posterior_log_level = (k*log(obs) + tau*log(prior)) / (k + tau),
with tau chosen out-of-fold (10-fold) to minimise MdAPE. Evaluated on the 862 panel sites
against the true mature level, only where age >= k+18 so the observation window never overlaps
the mature (last-12-month) window. Reads only committed data. Run in conda `sonnys`.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PANEL = REPO / "proforma" / "data" / "panel" / "main-data-v2-stitched.csv"
PS = HERE / "results" / "panel_scored.csv"
OUT = HERE / "results" / "postopen_curve.json"
KS = [1, 2, 3, 6, 9, 12]
TAUS = [0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24]

def ym(x):
    try:
        mo, yr = str(x).split("-"); return int(yr) * 12 + int(mo)
    except Exception:
        return np.nan

def mdape(a, p): return float(np.median(np.abs(a - p) / a) * 100)
def within(a, p, b=.2): return float((np.abs(a / p - 1) <= b).mean() * 100)

ps = pd.read_csv(PS)[["key", "plateau", "mature", "open_idx"]].dropna()
ps = ps[(ps.plateau > 0) & (ps.mature > 0)].set_index("key")

p = pd.read_csv(PANEL, low_memory=False)
p["wash"] = p.mem_wash_count.fillna(0) + p.ret_wash_count.fillna(0)
p = p[p.imputed == 0].copy()
p["key"] = p.client_id.astype(str) + "::" + p.site_id.astype(str)
p = p[p.key.isin(ps.index)]
p["m"] = (p.year * 12 + p.month) - p.operational_start.map(ym)     # months since open (0 = first)
p = p.merge(ps.mature.rename("mat"), left_on="key", right_index=True)

# global monthly ramp: median across sites of wash[m] / that site's mature, m = 0..23
ramp = (p[p.m.between(0, 23)].assign(frac=lambda d: d.wash / d.mat)
        .groupby("m").frac.median().reindex(range(24)).interpolate().clip(lower=0.05))
age = p.groupby("key").m.max() + 1

rng_rows = []
for k in KS:
    rk = float(ramp.loc[0:k - 1].mean())                          # mean ramp fraction over the observed window
    obs = p[p.m.between(0, k - 1)].groupby("key").wash.agg(["mean", "size"])
    obs = obs[obs["size"] >= k]
    d = ps.loc[ps.index.intersection(obs.index)].copy()
    d["obs_level"] = obs["mean"].reindex(d.index) / rk
    d["age"] = age.reindex(d.index)
    d = d[(d.age >= k + 18) & (d.obs_level > 0)]
    lp, lo, lt = np.log(d.plateau.values), np.log(d.obs_level.values), np.log(d.mature.values)
    # out-of-fold posterior: pick tau on train (min MdAPE), predict test
    post = np.empty(len(d))
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    for tr, te in kf.split(lp):
        best_t, best_l = TAUS[0], np.inf
        for t in TAUS:
            pr = np.exp((k * lo[tr] + t * lp[tr]) / (k + t))
            l = mdape(np.exp(lt[tr]), pr)
            if l < best_l: best_l, best_t = l, t
        post[te] = np.exp((k * lo[te] + best_t * lp[te]) / (k + best_t))
    # final full-data tau (for reporting)
    ft = min(TAUS, key=lambda t: mdape(d.mature.values, np.exp((k * lo + t * lp) / (k + t))))
    rng_rows.append(dict(k=k, n=len(d), tau=ft,
        prior_mdape=mdape(d.mature.values, d.plateau.values), prior_w20=within(d.mature.values, d.plateau.values),
        obs_mdape=mdape(d.mature.values, d.obs_level.values), obs_w20=within(d.mature.values, d.obs_level.values),
        post_mdape=mdape(d.mature.values, post), post_w20=within(d.mature.values, post)))

json.dump({"ramp_0_23": [round(float(x), 3) for x in ramp.values], "curve": rng_rows}, open(OUT, "w"), indent=1)
print("wrote", OUT)
print(pd.DataFrame(rng_rows).round(1).to_string(index=False))
