"""Honest 5-fold CV, deployed reality (brand=None). Model 1 = base LightGBM; Model 2 = + local-matured
anchor (k, blended out-of-fold using TRAIN sites only — no leakage). Also tries anchor + merged features."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
import coldstart_model as cm

panel, site = cm.load_panel_site()
S, _ = cm.build_features(panel, site)
S = S[S.mat_n >= 4].reset_index(drop=True)
y = np.log1p(S.mat_total.values)
lat, lon, mat = S.lat.values, S.lon.values, S.mat_total.values
NB = [c for c in cm.FEAT if not c.startswith("brand")]      # deployed path drops brand_loo/brand_n
CAT = cm.CAT
for c in CAT:
    S[c] = S[c].astype("category")
X = S[NB + CAT]


def run(k, anchor=True):
    kf = KFold(5, shuffle=True, random_state=0)
    p1 = np.zeros(len(S)); p2 = np.zeros(len(S))
    for tr, te in kf.split(S):
        mdl = lgb.LGBMRegressor(objective="quantile", alpha=0.5, n_estimators=500, learning_rate=0.03,
                                num_leaves=31, min_child_samples=20, subsample=0.8, colsample_bytree=0.8, verbose=-1)
        mdl.fit(X.iloc[tr], y[tr], categorical_feature=CAT)
        pm = np.expm1(mdl.predict(X.iloc[te]))
        for pos, i in enumerate(te):
            p1[i] = pm[pos]
            if anchor:
                d = cm._haversine(lat[i], lon[i], lat[tr], lon[tr])
                loc = d <= 20
                n = int(loc.sum())
                if n > 0:
                    a = float(np.mean(mat[tr][loc])); w = n / (n + k)
                    p2[i] = w * a + (1 - w) * pm[pos]
                else:
                    p2[i] = pm[pos]
            else:
                p2[i] = pm[pos]
    return p1, p2


def rep(pred, label):
    ape = np.abs(pred - mat) / mat
    wape = np.sum(np.abs(pred - mat)) / np.sum(mat) * 100
    print(f"{label:34s}: medAPE {np.median(ape)*100:5.1f}%   WAPE {wape:5.1f}%")
    return wape


print(f"sites {len(S)} (deployed path: brand=None)\n")
p1, _ = run(4, anchor=False)
rep(p1, "Model 1 — base (no brand)")
for k in [2, 3, 4, 6]:
    _, p2 = run(k)
    rep(p2, f"Model 2 — anchor k={k}")
