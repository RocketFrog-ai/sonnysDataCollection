"""Honest 5-fold CV of the mature-LEVEL model: does adding merged_all_sites demand features lower WAPE?

Target y = log1p(mat_total) over matured sites (mat_n>=4). LightGBM retrained per fold (no in-sample
optimism). Compares: base coldstart features vs + each merged feature group vs + all, with importances.
"""
import warnings; warnings.filterwarnings("ignore")
import re
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
import coldstart_model as cm

def safe(c):
    return re.sub(r"[^0-9A-Za-z]+", "_", c).strip("_")

panel, site = cm.load_panel_site()
S, _ = cm.build_features(panel, site)
S = S[S.mat_n >= 4].copy()
S["site_key"] = S.site_key.astype(str)
y = np.log1p(S.mat_total.values)

# merged demand features
m = pd.read_csv("../data/merged_all_sites.csv")
m["site_key"] = m["client_id"].astype(str) + "::" + m["site_id"].astype(str)
skip = {"Name", "Latitude", "Longitude", "client_name", "client_id", "site_id", "lat", "lon",
        "_Match", "__longitude", "__latitude", "__name", "site_key", "client_id_1"}
MNUM = [c for c in m.columns if c not in skip and pd.api.types.is_numeric_dtype(m[c])]
m2 = m[["site_key"] + MNUM].drop_duplicates("site_key").rename(columns={c: safe(c) for c in MNUM})
S = S.merge(m2, on="site_key", how="left")

GROUPS = {
    "pop/demo":   ["2025 Estimate", "Growth 2025-2020", "Growth 2030-2025", "2025 Average Age",
                   "Labor Force", "Renter-Occupied", "2025 % HH with Income $50K+"],
    "income":     ["Average Household Income", "Median Household Income", "Median Household Income.1",
                   "Average Household Income.1", "Current Year Estimated Owner-Occupied Housing Units by Value",
                   "$100,000 to $124,999", "$125,000 to $149,999", "$150,000 to $174,999",
                   "$175,000 to $199,999", "$200,000 to $249,999"],
    "vehicles":   ["1 vehicle", "2 vehicles", "3 vehicles", "4 vehicles", "5 or more vehicles",
                   "Total Vehicles Available in the Market", "Average Number of Vehicles Available"],
    "competition":["Count of Car Wash Competitors", "Nearest Car Wash Competitors-Distance",
                   "2nd Nearest Car Wash Competitors-Distance", "3rd Nearest Car Wash Competitors-Distance"],
    "retail-anchors":["Count of ChainXY VT - Mass Merchant", "Nearest ChainXY VT - Mass Merchant-Distance",
                   "2nd Nearest ChainXY VT - Mass Merchant-Distance", "3rd Nearest ChainXY VT - Mass Merchant-Distance",
                   "Count of ChainXY VT - Grocery", "Count of ChainXY VT - Department Store"],
    "traffic":    ["Nearest StreetLight US Hourly-ttl_overnight", "Nearest StreetLight US Hourly-ttl_breakfast",
                   "Nearest StreetLight US Hourly-ttl_lunch", "Nearest StreetLight US Hourly-ttl_afternoon",
                   "Nearest StreetLight US Hourly-ttl_dinner", "Nearest StreetLight US Hourly-ttl_night"],
}
GROUPS = {g: [safe(c) for c in cols] for g, cols in GROUPS.items()}
ALL_MERGED = [c for g in GROUPS.values() for c in g]

BASE_NUM = [c for c in cm.FEAT]; CAT = cm.CAT
for c in CAT:
    S[c] = S[c].astype("category")


def cv_wape(feat_num, label):
    X = S[feat_num + CAT].copy()
    yt = y
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    pred = np.zeros(len(S))
    for tr, te in kf.split(X):
        mdl = lgb.LGBMRegressor(objective="quantile", alpha=0.5, n_estimators=500, learning_rate=0.03,
                                num_leaves=31, min_child_samples=20, subsample=0.8, colsample_bytree=0.8, verbose=-1)
        mdl.fit(X.iloc[tr], yt[tr], categorical_feature=CAT)
        pred[te] = mdl.predict(X.iloc[te])
    yhat = np.expm1(pred); ytrue = S.mat_total.values
    ape = np.abs(yhat - ytrue) / ytrue
    wape = np.sum(np.abs(yhat - ytrue)) / np.sum(ytrue) * 100
    print(f"{label:28s}: medAPE {np.median(ape)*100:5.1f}%   WAPE {wape:5.1f}%   (nfeat {len(feat_num)})")
    return wape


print(f"matured sites: {len(S)} | mat_total range {S.mat_total.min():.0f}–{S.mat_total.max():.0f} "
      f"median {S.mat_total.median():.0f}  CoV {S.mat_total.std()/S.mat_total.mean():.2f}\n")
b = cv_wape(BASE_NUM, "BASE (current model)")
for g, cols in GROUPS.items():
    cv_wape(BASE_NUM + cols, f"base + {g}")
allw = cv_wape(BASE_NUM + ALL_MERGED, "base + ALL merged")
cv_wape(["lat", "lon"] + ALL_MERGED, "merged-only (+latlon)")
print(f"\nBASE WAPE {b:.1f}%  ->  base+ALL {allw:.1f}%  (delta {b-allw:+.1f}pp)")

# ── DEPLOYED REALITY: brand is None in drop_pin_ui, so brand_loo/brand_n are NaN. Does demand data help HERE? ──
print("\n=== deployed path: brand_loo/brand_n DROPPED (what the drop-pin forecast actually sees) ===")
NB = [c for c in BASE_NUM if not c.startswith("brand")]
nb = cv_wape(NB, "no-brand BASE (deployed)")
for g, cols in GROUPS.items():
    cv_wape(NB + cols, f"no-brand + {g}")
nbest = cv_wape(NB + ALL_MERGED, "no-brand + ALL merged")
print(f"\nno-brand BASE {nb:.1f}%  vs  +ALL merged {nbest:.1f}%  (delta {nb-nbest:+.1f}pp);  with-brand BASE {b:.1f}%")

# importance: which features matter (train once on all)
X = S[BASE_NUM + ALL_MERGED + CAT].copy()
mdl = lgb.LGBMRegressor(objective="quantile", alpha=0.5, n_estimators=500, learning_rate=0.03,
                        num_leaves=31, min_child_samples=20, subsample=0.8, colsample_bytree=0.8, verbose=-1)
mdl.fit(X, y, categorical_feature=CAT)
imp = pd.Series(mdl.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 20 features by gain:")
print(imp.head(20).to_string())
