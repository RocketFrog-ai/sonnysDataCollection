"""Where does the local-matured proxy BREAK, and what's more robust? For each matured site (LOO),
fill brand_loo four ways and compare WAPE, stratified by local sample size n and local dispersion (CoV):
  p1  = NaN (Model 1)
  mean   = mean of matured neighbours <=20km (current Model 2)
  median = median (robust to one flagship skewing the mean)
  shrink = n/(n+4)*local_mean + rest*region_mean (leans on region when local sample is thin)
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import coldstart_model as cm

art = cm.load(); rl = art["sites_rl"].reset_index(drop=True)
lat, lon, mat = rl.lat.values, rl.lon.values, rl.mat_total.values
rec = rl.rec_total.fillna(0).values; recm = rl.rec_mem.fillna(0).values
FEAT, CAT = art["FEAT"], art["CAT"]; q50 = art["models"]["q50"]
md = np.isfinite(mat) & (mat > 0)
idxs = np.where(md)[0]
reg_mean = rl[md].groupby("region").mat_total.mean().to_dict()
glob_mean = float(rl[md].mat_total.mean())


def base_row(i):
    d = cm._haversine(lat[i], lon[i], lat, lon)
    other = np.ones(len(rl), bool); other[i] = False
    row = {"lat": lat[i], "lon": lon[i]}
    for rkm in [5, 10, 20]:
        m = (d <= rkm) & other; row[f"n_nbr_{rkm}"] = int(m.sum()); vt = rec[m]
        row[f"mean_nbr_{rkm}"] = float(np.mean(vt)) if m.any() else np.nan
        if rkm == 20:
            row["logsum_20"] = float(np.log1p(np.sum(vt)))
            sh = recm[m] / np.where(rec[m] > 0, rec[m], np.nan)
            row["shr_nbr_20"] = float(np.nanmean(sh)) if np.isfinite(sh).any() else np.nan
    dd = d.copy(); dd[i] = np.inf; nn = int(np.argmin(dd))
    row["dist_nearest_km"] = float(dd[nn]); row["nearest_level"] = float(rec[nn])
    row["cluster_size"] = float(((d <= 20) & other).sum())
    reg = rl.region.iloc[nn]; row["region"] = reg; row["state"] = rl.state.iloc[nn]
    loc = (d <= 20) & other & md
    return row, mat[loc], reg


def pred(row, bl, bn):
    r = dict(row); r["brand_loo"] = bl; r["brand_n"] = bn
    X = pd.DataFrame([r])[FEAT + CAT]
    for c in CAT:
        X[c] = pd.Categorical(X[c], categories=art["cat_values"][c])
    return float(np.expm1(q50.predict(X)[0]))


rows = []
for i in idxs:
    row, v, reg = base_row(i); n = len(v)
    p1 = pred(row, np.nan, np.nan)
    if n > 0:
        mn, mdn = float(np.mean(v)), float(np.median(v))
        cov = float(np.std(v) / mn) if mn > 0 else np.nan
        shr = (n / (n + 4)) * mn + (4 / (n + 4)) * reg_mean.get(reg, glob_mean)
        rows.append((mat[i], n, cov, p1, pred(row, mn, n), pred(row, mdn, n), pred(row, shr, n)))
    else:
        rows.append((mat[i], 0, np.nan, p1, p1, p1, p1))
R = pd.DataFrame(rows, columns=["y", "n", "cov", "p1", "mean", "median", "shrink"])


def wape(df, col):
    return np.sum(np.abs(df[col] - df.y)) / np.sum(df.y) * 100


def line(df, tag):
    print(f"{tag:22s} (n={len(df):4d}): Model1 {wape(df,'p1'):5.1f}  mean {wape(df,'mean'):5.1f}  "
          f"median {wape(df,'median'):5.1f}  shrink {wape(df,'shrink'):5.1f}")


print(f"all matured sites {len(R)}  |  with local matured nbr: {(R.n>0).sum()}\n")
line(R, "ALL")
A = R[R.n > 0]
print("\n-- by local sample size n --")
for lo, hi, t in [(1, 2, "n=1"), (2, 4, "n=2-3"), (4, 8, "n=4-7"), (8, 1e9, "n>=8")]:
    s = A[(A.n >= lo) & (A.n < hi)]
    if len(s): line(s, t)
print("\n-- by local dispersion CoV (n>=3) --")
B = A[A.n >= 3]
for lo, hi, t in [(0, .4, "homogeneous <.4"), (.4, .7, "mixed .4-.7"), (.7, 1e9, "heterogeneous >.7")]:
    s = B[(B["cov"] >= lo) & (B["cov"] < hi)]
    if len(s): line(s, t)

# SHIPPED rule: median proxy, but fall back to Model 1 when local CoV > 0.7 (or n==0)
R["guard"] = np.where((R.n > 0) & (R["cov"] <= 0.7), R["median"], R["p1"])
het = R[(R.n > 0) & (R["cov"] > 0.7)]
print(f"\nSHIPPED (guarded median): overall WAPE {wape(R,'guard'):.1f}%  "
      f"(plain mean {wape(R,'mean'):.1f}, plain median {wape(R,'median'):.1f})")
print(f"  heterogeneous>0.7 stratum: guarded {wape(het,'guard'):.1f}%  vs  median {wape(het,'median'):.1f}  "
      f"vs Model1 {wape(het,'p1'):.1f}")
