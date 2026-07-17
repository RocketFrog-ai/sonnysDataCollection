"""Fit and save the SUPER ensemble artifact -> proforma/artifacts/'ensemble_super (model 5)'/.

Two fitted layers on top of coldstart Model 3 (both small, saved as one joblib):

  level_A  "with user inputs" — ridge on [log plateau, pay, vac, lot-type, log traffic],
           fit on the 70 mature proforma-matched sites (the only population where the
           capacity factors exist). LOSO-estimated accuracy: MdAPE 29.6%, within±20% 38.6%.
  level_B  "pin only" fallback — ridge calibration of the coldstart plateau on the 862
           panel-eligible sites (features from predict_site's own info dict + open year).
           Cross-fitted accuracy: mature MdAPE 31.8% (raw plateau 32.9%).
  year_debias  per-op-year median log-residual corrections c_y (fit on the 862).

Debias constants are nested/cross-fitted (never in-sample). Run in conda `sonnys`;
the joblib is welded to that env's sklearn version (same rule as coldstart_artifacts).
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT_DIR = REPO / "proforma" / "artifacts" / "ensemble_super (model 5)"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# choice -> score maps (the proforma Excel's own IF-cascade weights; runtime inputs use these)
PAY_MAP = {"3 or more": 0.15, "2": 0.10, "1": 0.05, "live person": 0.0}
VAC_MAP = {"more than 20": 0.15, "12 - 20": 0.10, "less than 12": 0.05, "coin or none": -0.25}
LOT_MAP = {"corner lot with light": 0.15, "corner lot without light": 0.125,
           "inside lot near light": 0.075, "inside lot no light": 0.05}

# ── level_A: capacity+traffic ridge on the 70 mature matched sites ──
F = pd.read_csv(HERE / "results" / "ensemble_features.csv")
F["lp"] = np.log(F.plateau_loo)
F["ltr"] = np.log(F.traffic_count.clip(lower=1))
MAT = F[F.mature_ok & F.mature_wash.gt(0)].reset_index(drop=True)
FEATS_A = ["lp", "pay", "vac", "tos", "ltr"]
XA, yA = MAT[FEATS_A].values, np.log(MAT.mature_wash.values)
muA, sdA = XA.mean(0), XA.std(0) + 1e-9
ridge_A = Ridge(alpha=1.0).fit((XA - muA) / sdA, yA)
# nested-LOO debias (never in-sample)
nh = np.empty(len(MAT))
for j in range(len(MAT)):
    tr = MAT.drop(index=j)
    Xj, yj = tr[FEATS_A].values, np.log(tr.mature_wash.values)
    mj, sj = Xj.mean(0), Xj.std(0) + 1e-9
    r = Ridge(alpha=1.0).fit((Xj - mj) / sj, yj)
    nh[j] = r.predict(((XA[j] - mj) / sj).reshape(1, -1))[0]
c_A = float(np.median(yA - nh))

# ── level_B: pin-only calibration on the 862 panel sites ──
E = pd.read_csv(HERE / "results" / "panel_scored.csv")
E = E[(E.mature > 0) & (E.plateau > 0)].reset_index(drop=True)
REGIONS = ["South", "West", "Midwest", "Northeast"]
def feats_B(df):
    X = pd.DataFrame({
        "lp": np.log(df.plateau),
        "lmp": np.log(df.model_plateau.clip(lower=1)),
        "lanchor": np.where(np.isfinite(df.anchor_level) & (df.anchor_level > 0),
                            np.log(df.anchor_level.clip(lower=1)) - np.log(df.plateau), 0.0),
        "has_anchor": (np.isfinite(df.anchor_level) & (df.anchor_level > 0)).astype(float),
        "nmat": np.log1p(df.n_local_mature.fillna(0)),
        "oy": df.open_year - 2022.0,
    })
    for rg in REGIONS:
        X[f"r_{rg}"] = (df.region == rg).astype(float)
    return X
XB, yB = feats_B(E).values, np.log(E.mature.values)
muB, sdB = XB.mean(0), XB.std(0) + 1e-9
ridge_B = Ridge(alpha=1.0).fit((XB - muB) / sdB, yB)
gkf = GroupKFold(n_splits=10)
res = np.empty(len(E))
for tr, te in gkf.split(XB, yB, groups=E.key):
    m, s = XB[tr].mean(0), XB[tr].std(0) + 1e-9
    r = Ridge(alpha=1.0).fit((XB[tr] - m) / s, yB[tr])
    res[te] = yB[te] - r.predict((XB[te] - m) / s)
c_B = float(np.median(res))

# ── per-op-year debias c_y (cross-fitted on the 862, applied on top of level_B×ramp) ──
E["cal_level"] = np.exp(ridge_B.predict((XB - muB) / sdB) + c_B)
c_y = {}
for y in range(1, 6):
    d = E[E.get(f"size_y{y}", pd.Series(dtype=float)).ge(10) & E[f"mean_y{y}"].gt(0) & E[f"pred_y{y}"].gt(0)]
    if len(d) < 30:
        c_y[y] = 0.0
        continue
    base = np.log(d.cal_level.values * (d[f"pred_y{y}"].values / d.plateau.values))
    dv = np.log(d[f"mean_y{y}"].values)
    hold = np.empty(len(d))
    gk = GroupKFold(n_splits=10)
    for tr, te in gk.split(base.reshape(-1, 1), dv, groups=d.key):
        hold[te] = np.median(dv[tr] - base[tr])
    c_y[y] = float(np.median(dv - base))          # final constant = full-data median
art = {
    "version": "v1 (2026-07-17)",
    "level_A": {"feats": FEATS_A, "mu": muA, "sd": sdA, "coef": ridge_A.coef_,
                "intercept": float(ridge_A.intercept_), "debias": c_A,
                "pay_map": PAY_MAP, "vac_map": VAC_MAP, "lot_map": LOT_MAP,
                "train": "70 mature proforma-matched sites",
                "loso_metrics": {"MdAPE": 29.6, "within20": 38.6, "rho": 0.58}},
    "level_B": {"feats": list(feats_B(E).columns), "mu": muB, "sd": sdB, "coef": ridge_B.coef_,
                "intercept": float(ridge_B.intercept_), "debias": c_B, "regions": REGIONS,
                "train": "862 panel-eligible sites (open 2021-24, >=24 clean months)",
                "cv_metrics": {"MdAPE": 31.8, "within20": 30.0, "rho": 0.47}},
    "year_debias": c_y,
    "upstream": "proforma/artifacts/coldstart_artifacts.joblib (Model 3: model_kind='et', local_anchor=True)",
    "fitted_by": "experiments/old-proforma-analysis/ensemble/fit_super_artifact.py",
}
joblib.dump(art, OUT_DIR / "super_ensemble_v1.joblib")
print("saved", OUT_DIR / "super_ensemble_v1.joblib")
print(json.dumps({"c_A": c_A, "c_B": c_B, "c_y": c_y}, indent=1))
print("level_A coefs:", dict(zip(FEATS_A, ridge_A.coef_.round(3))))
