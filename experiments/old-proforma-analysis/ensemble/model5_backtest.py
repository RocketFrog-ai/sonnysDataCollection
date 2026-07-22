"""Model 5 (SUPER ensemble, input-calibrated level_A) leave-one-site-out backtest.

Model 5's level_A recalibrates the cold-start plateau using the site's capacity inputs —
pay stations, vacuum slots, site type (from the OLD PROFORMA data) + traffic. This scores it
honestly: for each mature site its level is predicted from a ridge trained on the OTHER mature
sites only (nested leave-one-out debias); non-mature sites use the all-mature-trained model
(never in their own training). Per-year forecast = level x model ramp x per-year median debias.
Output: ensemble/results/model5_loso.csv (per-site) + a printed proforma/v1.5/Model-5 table.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
E = pd.read_csv(HERE / "results" / "ensemble_features.csv")
E["lp"] = np.log(E.plateau_loo)
E["ltr"] = np.log(E.traffic_count.clip(lower=1))
FEATS = ["lp", "pay", "vac", "tos", "ltr"]            # plateau + the 3 proven factors + traffic
MAT = E[E.mature_ok & E.mature_wash.gt(0)].reset_index(drop=True)

def fit(train):
    X, y = train[FEATS].values, np.log(train.mature_wash.values)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    mdl = Ridge(alpha=1.0).fit((X - mu) / sd, y)
    nh = np.empty(len(train))                         # nested-LOO debias (never in-sample)
    for j in range(len(train)):
        sub = train.drop(index=train.index[j])
        Xs, ys = sub[FEATS].values, np.log(sub.mature_wash.values)
        m2, s2 = Xs.mean(0), Xs.std(0) + 1e-9
        r2 = Ridge(alpha=1.0).fit((Xs - m2) / s2, ys)
        nh[j] = r2.predict(((train[FEATS].values[j] - m2) / s2).reshape(1, -1))[0]
    return mdl, mu, sd, float(np.median(np.log(train.mature_wash.values) - nh))

def level_of(mdl, mu, sd, c, rows):
    return np.exp(mdl.predict((rows[FEATS].values - mu) / sd) + c)

# leave-one-out mature level for the mature sites
m5_mat = {}
for i in range(len(MAT)):
    mdl, mu, sd, c = fit(MAT.drop(index=i))
    m5_mat[MAT.loc[i, "source_file"]] = float(level_of(mdl, mu, sd, c, MAT.iloc[[i]])[0])
# all-mature model for the rest
mdl_a, mu_a, sd_a, c_a = fit(MAT)
level = {}
for _, r in E.iterrows():
    if r.source_file in m5_mat:
        level[r.source_file] = m5_mat[r.source_file]
    elif np.isfinite(r[FEATS].astype(float).values).all():
        level[r.source_file] = float(level_of(mdl_a, mu_a, sd_a, c_a, E[E.source_file == r.source_file])[0])
    else:
        level[r.source_file] = np.nan

# per-year median debias, fit on the mature sites' LOO levels x model ramp
c_y = {}
for y in range(1, 6):
    d = MAT[MAT.open_observed & MAT[f"nobs_y{y}"].ge(10) & MAT[f"act_y{y}"].gt(0) & MAT[f"cs_y{y}"].gt(0)]
    lv = np.array([m5_mat[sf] for sf in d.source_file])
    ramp = d[f"cs_y{y}"].values / d.plateau_loo.values
    c_y[y] = float(np.median(np.log(d[f"act_y{y}"].values) - np.log(lv * ramp)))

rows = []
for _, r in E.iterrows():
    o = {"source_file": r.source_file, "m5_mature": level.get(r.source_file, np.nan)}
    for y in range(1, 6):
        lv = level.get(r.source_file, np.nan)
        if pd.notna(lv) and r.plateau_loo > 0 and pd.notna(r[f"cs_y{y}"]):
            o[f"m5_y{y}"] = lv * (r[f"cs_y{y}"] / r.plateau_loo) * np.exp(c_y[y])
    rows.append(o)
M5 = pd.DataFrame(rows)
M5.to_csv(HERE / "results" / "model5_loso.csv", index=False)

# ---- comparison table ----
def mdape(a, p): a, p = np.asarray(a, float), np.asarray(p, float); return np.median(np.abs(a - p) / a) * 100
def within(a, p): a, p = np.asarray(a, float), np.asarray(p, float); return (np.abs(a / p - 1) <= .2).mean() * 100
def rho(a, p): return stats.spearmanr(p, a)[0]

D = E.merge(M5, on="source_file")
dm = D[D.mature_ok & D.mature_wash.gt(0)].dropna(subset=["pf_y5", "plateau_loo", "m5_mature"])
print("MATURE (n=%d)  MdAPE / within20 / rho / bias" % len(dm))
for nm, c in [("Proforma v1.0", "pf_y5"), ("Model v1.5", "plateau_loo"), ("Model 5 SUPER", "m5_mature")]:
    print(f"  {nm:16s}: {mdape(dm.mature_wash, dm[c]):5.1f}%  {within(dm.mature_wash, dm[c]):5.1f}%  "
          f"{rho(dm.mature_wash, dm[c]):.2f}  {np.median(dm.mature_wash/dm[c]):.2f}")
print("\nPER OPERATING YEAR  MdAPE:  proforma / v1.5 / Model 5")
for y in range(1, 6):
    d = D[D.open_observed & D[f"nobs_y{y}"].ge(10) & D[f"act_y{y}"].gt(0) & D[f"pf_y{y}"].gt(0)
          & D[f"cs_y{y}"].gt(0) & D[f"m5_y{y}"].gt(0)]
    if len(d) < 8: continue
    print(f"  Y{y} (n={len(d):2d}): {mdape(d[f'act_y{y}'], d[f'pf_y{y}']):.0f}%  "
          f"{mdape(d[f'act_y{y}'], d[f'cs_y{y}']):.0f}%  {mdape(d[f'act_y{y}'], d[f'm5_y{y}']):.0f}%")
print("\nwrote", HERE / "results" / "model5_loso.csv")
