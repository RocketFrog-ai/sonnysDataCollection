"""Panel-scale (862 sites) validation: raw coldstart LOO vs calibrated ensemble vs naive baselines."""
import glob
import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

pd.set_option("display.width", 240)
SCRATCH = "/Users/dhruvsood/sonnysDataCollection/experiments/old-proforma-analysis/ensemble/results"
PANEL = "/Users/dhruvsood/sonnysDataCollection/proforma/data/panel/main-data-v2-stitched.csv"

def ym(x):
    try:
        mo, yr = str(x).split("-"); return int(yr) * 12 + int(mo)
    except Exception:
        return np.nan

def mdape(a, p): return float(np.median(np.abs(np.asarray(a) - np.asarray(p)) / np.asarray(a)) * 100)
def within(a, p, b=0.2): return float((np.abs(np.asarray(a) / np.asarray(p) - 1) <= b).mean() * 100)
def missf(a, p): return float(np.exp(np.median(np.abs(np.log(np.asarray(a) / np.asarray(p))))))

pred = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{SCRATCH}/panel_chunk_*.csv"))], ignore_index=True)
pred = pred[~pred.get("error", pd.Series(index=pred.index)).notna()] if "error" in pred else pred
print("panel predictions:", len(pred))

p = pd.read_csv(PANEL, low_memory=False)
p["wash"] = p.mem_wash_count.fillna(0) + p.ret_wash_count.fillna(0)
p = p[p.imputed == 0]
p["key"] = p.client_id.astype(str) + "::" + p.site_id.astype(str)
p["open_idx"] = p.operational_start.map(ym)
p["cal_idx"] = p.year * 12 + p.month
p["op_year"] = (p.cal_idx - p.open_idx) // 12 + 1

mat = (p.sort_values("cal_idx").groupby("key").tail(12).groupby("key").wash.median().rename("mature"))
opy = (p[p.op_year.between(1, 5)].groupby(["key", "op_year"]).wash.agg(["mean", "size"]).unstack())
opy.columns = [f"{a}_y{int(b)}" for a, b in opy.columns]
meta = p.groupby("key").agg(open_idx=("open_idx", "first")).assign(
    open_year=lambda d: (d.open_idx - 1) // 12)

E = pred.merge(mat, on="key").merge(opy, on="key", how="left").merge(meta, on="key", how="left")
E = E[(E.mature > 0) & (E.plateau > 0)].reset_index(drop=True)
print("scored sites:", len(E))

# ── A. raw coldstart LOO at panel scale: mature + op-year accuracy ──
print("\n=== A. RAW coldstart LOO at 862-site scale ===")
r = E.mature / E.plateau
print(f"mature: MdAPE={mdape(E.mature, E.plateau):.1f}%  within10/20/30/50="
      f"{within(E.mature, E.plateau, .1):.0f}/{within(E.mature, E.plateau, .2):.0f}/"
      f"{within(E.mature, E.plateau, .3):.0f}/{within(E.mature, E.plateau, .5):.0f}%  "
      f"rho={stats.spearmanr(E.plateau, E.mature)[0]:.2f}  ratio_med={r.median():.2f}  miss={missf(E.mature, E.plateau):.2f}x")
acc = []
for y in range(1, 6):
    d = E[E.get(f"size_y{y}", pd.Series(dtype=float)).ge(10) & E[f"mean_y{y}"].gt(0) & E[f"pred_y{y}"].gt(0)]
    if len(d) < 30: continue
    acc.append({"yr": f"Y{y}", "n": len(d), "MdAPE": mdape(d[f"mean_y{y}"], d[f"pred_y{y}"]),
                "within20": within(d[f"mean_y{y}"], d[f"pred_y{y}"]),
                "ratio_med": float((d[f"mean_y{y}"] / d[f"pred_y{y}"]).median()),
                "rho": float(stats.spearmanr(d[f"pred_y{y}"], d[f"mean_y{y}"])[0])})
print(pd.DataFrame(acc).round(2).to_string(index=False))

# ── B. naive baselines ──
print("\n=== B. naive baselines (mature level) ===")
have_anchor = E[np.isfinite(E.anchor_level) & (E.anchor_level > 0)]
print(f"local mature median (anchor) only: n={len(have_anchor)}  MdAPE={mdape(have_anchor.mature, have_anchor.anchor_level):.1f}%  "
      f"rho={stats.spearmanr(have_anchor.anchor_level, have_anchor.mature)[0]:.2f}")
print(f"global median level ({E.mature.median():.0f}/mo): MdAPE={mdape(E.mature, np.full(len(E), E.mature.median())):.1f}%")

# ── C. calibrated ensemble at panel scale (grouped 10-fold, nothing sees its own site) ──
print("\n=== C. CALIBRATED ensemble (ridge on log level, 10-fold by site) ===")
X = pd.DataFrame({
    "lp": np.log(E.plateau),
    "lmp": np.log(E.model_plateau.clip(lower=1)),
    "lanchor": np.where(np.isfinite(E.anchor_level) & (E.anchor_level > 0),
                        np.log(E.anchor_level.clip(lower=1)) - np.log(E.plateau), 0.0),
    "has_anchor": (np.isfinite(E.anchor_level) & (E.anchor_level > 0)).astype(float),
    "nmat": np.log1p(E.n_local_mature.fillna(0)),
    "oy": E.open_year - 2022.0,
})
for reg in ["South", "West", "Midwest", "Northeast"]:
    X[f"r_{reg}"] = (E.region == reg).astype(float)
yv = np.log(E.mature.values)
gkf = GroupKFold(n_splits=10)
hat = np.empty(len(E)); hat[:] = np.nan
cy_store = []
for tr, te in gkf.split(X, yv, groups=E.key):
    Xtr, Xte = X.values[tr], X.values[te]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    mdl = Ridge(alpha=1.0).fit((Xtr - mu) / sd, yv[tr])
    raw = mdl.predict((Xte - mu) / sd)
    # median debias in-fold (train residuals, cross-fitted approximation via train fit)
    c = np.median(yv[tr] - mdl.predict((Xtr - mu) / sd))
    hat[te] = raw + c
E["cal"] = np.exp(hat)
print(f"mature: MdAPE={mdape(E.mature, E.cal):.1f}%  within10/20/30/50="
      f"{within(E.mature, E.cal, .1):.0f}/{within(E.mature, E.cal, .2):.0f}/"
      f"{within(E.mature, E.cal, .3):.0f}/{within(E.mature, E.cal, .5):.0f}%  "
      f"rho={stats.spearmanr(E.cal, E.mature)[0]:.2f}  ratio_med={(E.mature/E.cal).median():.2f}  miss={missf(E.mature, E.cal):.2f}x")

# per-year with the calibrated level + model ramp + cross-fitted per-year debias
print("\nper-year, calibrated level x model ramp (+ in-fold year debias):")
E["ramp1"] = E.pred_y1 / E.plateau
for y in range(2, 6):
    E[f"ramp{y}"] = E[f"pred_y{y}"] / E.plateau
acc2 = []
for y in range(1, 6):
    ok = E[f"size_y{y}"].ge(10) & E[f"mean_y{y}"].gt(0) & E[f"pred_y{y}"].gt(0)
    d = E[ok].copy()
    if len(d) < 30: continue
    hat_y = np.empty(len(d)); hat_y[:] = np.nan
    dv = np.log(d[f"mean_y{y}"].values)
    base = np.log(d.cal.values * d[f"ramp{y}"].values)
    gk = GroupKFold(n_splits=10)
    for tr, te in gk.split(base.reshape(-1, 1), dv, groups=d.key):
        c = np.median(dv[tr] - base[tr])
        hat_y[te] = base[te] + c
    pr_ = np.exp(hat_y)
    acc2.append({"yr": f"Y{y}", "n": len(d), "MdAPE": mdape(d[f"mean_y{y}"], pr_),
                 "within20": within(d[f"mean_y{y}"], pr_),
                 "ratio_med": float(np.median(d[f"mean_y{y}"] / pr_)),
                 "rho": float(stats.spearmanr(pr_, d[f"mean_y{y}"])[0])})
    E.loc[ok, f"cal_y{y}"] = pr_
print(pd.DataFrame(acc2).round(2).to_string(index=False))

E.to_csv(f"{SCRATCH}/panel_scored.csv", index=False)
out = {"raw_mature": {"MdAPE": mdape(E.mature, E.plateau), "rho": float(stats.spearmanr(E.plateau, E.mature)[0]),
                      "within20": within(E.mature, E.plateau, .2), "ratio": float((E.mature/E.plateau).median())},
       "cal_mature": {"MdAPE": mdape(E.mature, E.cal), "rho": float(stats.spearmanr(E.cal, E.mature)[0]),
                      "within20": within(E.mature, E.cal, .2), "ratio": float((E.mature/E.cal).median())},
       "raw_years": acc, "cal_years": acc2, "n": len(E)}
json.dump(out, open(f"{SCRATCH}/panel_results.json", "w"), indent=1)
print("\nwrote panel_scored.csv / panel_results.json")
