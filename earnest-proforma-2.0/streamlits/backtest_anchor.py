"""Leave-one-out backtest: does the local-mature anchor lower the new-site LEVEL error vs the base model?

For each matured site (mat_total known), we predict its mature level using ONLY the other sites:
 - model  = the LightGBM q50 on leave-one-out features (self excluded from neighbour stats),
 - anchor = mean mat_total of OTHER matured sites within 20 km,
 - blend  = w*anchor + (1-w)*model,  w = n/(n+k).
Compared against the site's true mat_total. NOTE: the LightGBM was trained on these sites, so the
model column is in-sample (optimistic). If the blend still wins, the anchor genuinely adds signal.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import coldstart_model as cm

art = cm.load()
rl = art["sites_rl"].reset_index(drop=True)
lat, lon = rl.lat.values, rl.lon.values
mat = rl.mat_total.values
rec = rl.rec_total.fillna(0).values
recm = rl.rec_mem.fillna(0).values
FEAT, CAT = art["FEAT"], art["CAT"]
q50 = art["models"]["q50"]
idxs = np.where(np.isfinite(mat) & (mat > 0))[0]


def loo_features(i):
    d = cm._haversine(lat[i], lon[i], lat, lon)
    other = np.ones(len(rl), bool); other[i] = False
    row = {"lat": lat[i], "lon": lon[i]}
    for rkm in [5, 10, 20]:
        mask = (d <= rkm) & other
        row[f"n_nbr_{rkm}"] = int(mask.sum())
        vt = rec[mask]
        row[f"mean_nbr_{rkm}"] = float(np.mean(vt)) if mask.any() else np.nan
        if rkm == 20:
            row["logsum_20"] = float(np.log1p(np.sum(vt)))
            sh = recm[mask] / np.where(rec[mask] > 0, rec[mask], np.nan)
            row["shr_nbr_20"] = float(np.nanmean(sh)) if np.isfinite(sh).any() else np.nan
    dd = d.copy(); dd[i] = np.inf; nn = int(np.argmin(dd))
    row["dist_nearest_km"] = float(dd[nn]); row["nearest_level"] = float(rec[nn])
    row["cluster_size"] = float(((d <= 20) & other).sum())
    row["region"] = rl.region.iloc[nn]; row["state"] = rl.state.iloc[nn]
    row["brand_loo"] = np.nan; row["brand_n"] = np.nan
    X = pd.DataFrame([row])[FEAT + CAT]
    for c in CAT:
        X[c] = pd.Categorical(X[c], categories=art["cat_values"][c])
    return X


recs = []
for i in idxs:
    m = float(np.expm1(q50.predict(loo_features(i))[0]))
    d = cm._haversine(lat[i], lon[i], lat, lon)
    other = np.ones(len(rl), bool); other[i] = False
    loc = (d <= 20) & other & np.isfinite(mat) & (mat > 0)
    n = int(loc.sum())
    a = float(np.mean(mat[loc])) if n > 0 else np.nan
    recs.append((i, float(mat[i]), m, a, n))
R = pd.DataFrame(recs, columns=["i", "y", "m", "a", "n"])


def metrics(pred, y):
    ape = np.abs(pred - y) / y
    return np.median(ape) * 100, np.mean(ape) * 100, np.sum(np.abs(pred - y)) / np.sum(y) * 100


print(f"matured sites: {len(R)} · with >=1 matured neighbour <=20km: {(R.n > 0).sum()}")
sub = R[R.n > 0].copy()
y, m, a, n = sub.y.values, sub.m.values, sub.a.values, sub.n.values
print(f"\n--- evaluated where the anchor is active (n={len(sub)} sites) ---")
print("strategy           medAPE  meanAPE   WAPE")
md = metrics(m, y); print(f"Model only        : {md[0]:6.1f}% {md[1]:7.1f}% {md[2]:6.1f}%")
ad = metrics(a, y); print(f"Anchor only       : {ad[0]:6.1f}% {ad[1]:7.1f}% {ad[2]:6.1f}%")
best = None
for k in [1, 2, 3, 4, 6, 8, 12, 20]:
    w = n / (n + k); blend = w * a + (1 - w) * m
    bd = metrics(blend, y)
    star = ""
    if best is None or bd[0] < best[1]:
        best = (k, bd[0]); star = "  <-- best medAPE"
    print(f"Blend k={k:<2}        : {bd[0]:6.1f}% {bd[1]:7.1f}% {bd[2]:6.1f}%{star}")
print(f"\nBest k by median APE: {best[0]} (medAPE {best[1]:.1f}%)")
