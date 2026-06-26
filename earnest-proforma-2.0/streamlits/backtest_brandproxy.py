"""Trick test (no retrain): in the deployed path brand=None -> brand_loo=NaN (the model's #1 feature is blank).
Instead feed brand_loo = local-matured mean (mat_total of OTHER sites <=20km). Does the SAVED model use it
better than a post-hoc anchor blend? In-sample for all variants (saved model trained on these), so the DELTA
across variants is the signal. LOO features (self excluded from neighbour stats)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import coldstart_model as cm

art = cm.load()
rl = art["sites_rl"].reset_index(drop=True)
lat, lon, mat = rl.lat.values, rl.lon.values, rl.mat_total.values
rec = rl.rec_total.fillna(0).values; recm = rl.rec_mem.fillna(0).values
FEAT, CAT = art["FEAT"], art["CAT"]; q50 = art["models"]["q50"]
idxs = np.where(np.isfinite(mat) & (mat > 0))[0]


def feats(i, brand_proxy):
    d = cm._haversine(lat[i], lon[i], lat, lon)
    other = np.ones(len(rl), bool); other[i] = False
    row = {"lat": lat[i], "lon": lon[i]}
    for rkm in [5, 10, 20]:
        mask = (d <= rkm) & other
        row[f"n_nbr_{rkm}"] = int(mask.sum())
        vt = rec[mask]; row[f"mean_nbr_{rkm}"] = float(np.mean(vt)) if mask.any() else np.nan
        if rkm == 20:
            row["logsum_20"] = float(np.log1p(np.sum(vt)))
            sh = recm[mask] / np.where(rec[mask] > 0, rec[mask], np.nan)
            row["shr_nbr_20"] = float(np.nanmean(sh)) if np.isfinite(sh).any() else np.nan
    dd = d.copy(); dd[i] = np.inf; nn = int(np.argmin(dd))
    row["dist_nearest_km"] = float(dd[nn]); row["nearest_level"] = float(rec[nn])
    row["cluster_size"] = float(((d <= 20) & other).sum())
    row["region"] = rl.region.iloc[nn]; row["state"] = rl.state.iloc[nn]
    loc = (d <= 20) & other & np.isfinite(mat) & (mat > 0); n = int(loc.sum())
    if brand_proxy and n > 0:
        row["brand_loo"] = float(np.mean(mat[loc])); row["brand_n"] = float(n)
    else:
        row["brand_loo"] = np.nan; row["brand_n"] = np.nan
    X = pd.DataFrame([row])[FEAT + CAT]
    for c in CAT:
        X[c] = pd.Categorical(X[c], categories=art["cat_values"][c])
    return X, (float(np.mean(mat[loc])) if n > 0 else np.nan), n


def wape(pred, y):
    ape = np.abs(pred - y) / y
    return np.median(ape) * 100, np.sum(np.abs(pred - y)) / np.sum(y) * 100


y = mat[idxs]
# variant A: brand_loo = NaN (current deployed). variant B: brand_loo = local matured mean.
pA, pB, anc, ns = [], [], [], []
for i in idxs:
    XA, a, n = feats(i, False); pA.append(float(np.expm1(q50.predict(XA)[0])))
    XB, _, _ = feats(i, True);  pB.append(float(np.expm1(q50.predict(XB)[0])))
    anc.append(a); ns.append(n)
pA, pB, anc, ns = map(np.array, (pA, pB, anc, ns))
print(f"sites {len(idxs)}\n")
print("variant                                    medAPE  WAPE")
md = wape(pA, y); print(f"A: brand_loo=NaN (current deployed)      : {md[0]:5.1f}% {md[1]:5.1f}%")
mb = wape(pB, y); print(f"B: brand_loo=local-matured mean          : {mb[0]:5.1f}% {mb[1]:5.1f}%")
# post-hoc anchor blend on variant A for reference (k=4)
m = ns > 0; w = np.where(m, ns / (ns + 4.0), 0.0)
pAnchor = np.where(m, w * np.nan_to_num(anc) + (1 - w) * pA, pA)
mc = wape(pAnchor, y); print(f"C: A + post-hoc anchor blend k=4         : {mc[0]:5.1f}% {mc[1]:5.1f}%")
