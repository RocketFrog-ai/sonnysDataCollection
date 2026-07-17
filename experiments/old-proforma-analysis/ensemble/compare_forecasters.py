"""Head-to-head: old proforma projections vs coldstart pin-forecast (LOO + leaky) vs actuals."""
import glob
import json

import numpy as np
import pandas as pd
from scipy import stats

pd.set_option("display.width", 260)
SCRATCH = "/Users/dhruvsood/sonnysDataCollection/experiments/old-proforma-analysis/ensemble/results"
ROOT = "/Users/dhruvsood/sonnysDataCollection/experiments/old-proforma-analysis"

pred = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{SCRATCH}/chunk_[0-9]*.csv"))], ignore_index=True)
pred = pred.drop_duplicates("source_file")
print("prediction rows:", len(pred), "| errors:", pred.get("error", pd.Series(dtype=object)).notna().sum()
      if "error" in pred else 0)

s = pd.read_csv(f"{ROOT}/old-proforma-combined.csv")
m = pd.read_csv(f"{ROOT}/old-proforma-combined-monthly.csv")

def ym(x):
    try:
        mo, yr = str(x).split("-"); return int(yr) * 12 + int(mo)
    except Exception:
        return np.nan

site = s[s.match_status.str.startswith("matched") & (s.actuals_suspect != True)].copy()
site["open_idx"] = site.match_operational_start.map(ym)
mm = m[m.year.notna() & (m.imputed == 0)].copy()
mm = mm[mm.source_file.isin(site.source_file)]
mm["cal_idx"] = mm.year.astype(int) * 12 + mm.month.astype(int)
mm = mm.merge(site[["source_file", "open_idx"]], on="source_file")
mm["mo_open"] = mm.cal_idx - mm.open_idx
mm["op_year"] = mm.mo_open // 12 + 1
fc = mm.groupby("source_file").cal_idx.min(); lc = mm.groupby("source_file").cal_idx.max()
site = site.merge(fc.rename("first_cal"), left_on="source_file", right_index=True, how="left")
site = site.merge(lc.rename("last_cal"), left_on="source_file", right_index=True, how="left")
site["open_observed"] = (site.open_idx > 2020 * 12 + 1) & (site.open_idx >= site.first_cal - 1)
site["age_mo"] = site.last_cal - site.open_idx + 1
opy = (mm[mm.op_year.between(1, 5)].groupby(["source_file", "op_year"]).wash_count
       .agg(["mean", "size"]).unstack())
opy.columns = [f"{a}_y{int(b)}" for a, b in opy.columns]
site = site.merge(opy, left_on="source_file", right_index=True, how="left")
last12 = (mm.sort_values(["source_file", "cal_idx"]).groupby("source_file")
          .apply(lambda g: pd.Series({"mature_wash": g.tail(12).wash_count.median(),
                                      "mature_n": len(g.tail(12))}), include_groups=False))
site = site.merge(last12, left_on="source_file", right_index=True, how="left")
site["mature_ok"] = (site.age_mo >= 24) & (site.mature_n >= 6) & (site.mature_wash > 0)
site = site.merge(pred, on="source_file", how="left")

BANDS = [0.10, 0.20, 0.30, 0.50]

def block(act, prd):
    act, prd = np.asarray(act, float), np.asarray(prd, float)
    ratio = act / prd
    lr = np.log(ratio)
    out = {"n": len(act), "median_ratio": float(np.median(ratio)),
           "MdAPE_vs_act": float(np.median(np.abs(act - prd) / act) * 100),
           "wMAPE": float(np.abs(act - prd).sum() / act.sum() * 100),
           "miss_factor": float(np.exp(np.median(np.abs(lr)))),
           "spearman": float(stats.spearmanr(prd, act)[0]) if len(act) >= 5 else np.nan}
    for b in BANDS:
        out[f"within_{int(b*100)}"] = float((np.abs(act / prd - 1) <= b).mean() * 100)
    return out

results = {"years": {}, "sites": {}}
for y in range(1, 6):
    cols = [f"mean_y{y}", f"vol_monthly_y{y}", f"pred_loo_y{y}", f"pred_leaky_y{y}"]
    d = site[site.open_observed & site[f"size_y{y}"].ge(10)].dropna(subset=cols)
    d = d[(d[f"mean_y{y}"] > 0) & (d[f"vol_monthly_y{y}"] > 0) & (d[f"pred_loo_y{y}"] > 0)]
    if len(d) < 5:
        continue
    act = d[f"mean_y{y}"]
    yr = {"proforma": block(act, d[f"vol_monthly_y{y}"]),
          "coldstart_loo": block(act, d[f"pred_loo_y{y}"]),
          "coldstart_leaky": block(act, d[f"pred_leaky_y{y}"])}
    # head-to-head: coldstart_loo vs proforma on identical sites
    e_m = np.abs(np.log(act.values / d[f"pred_loo_y{y}"].values))
    e_p = np.abs(np.log(act.values / d[f"vol_monthly_y{y}"].values))
    wins = int((e_m < e_p).sum()); nn = len(d)
    yr["model_win_rate"] = wins / nn * 100
    yr["sign_test_p"] = float(stats.binomtest(wins, nn, 0.5).pvalue)
    results["years"][f"Y{y}"] = yr
    results["sites"][f"Y{y}"] = [
        {"name": str(r.match_client_name), "addr": str(r.address)[:70],
         "act": float(r[f"mean_y{y}"]), "proforma": float(r[f"vol_monthly_y{y}"]),
         "model": float(r[f"pred_loo_y{y}"])} for _, r in d.iterrows()]

# pooled site-years
frames = []
for y in range(1, 6):
    cols = [f"mean_y{y}", f"vol_monthly_y{y}", f"pred_loo_y{y}", f"pred_leaky_y{y}"]
    d = site[site.open_observed & site.get(f"size_y{y}", pd.Series(dtype=float)).ge(10)].dropna(subset=cols)
    d = d[(d[f"mean_y{y}"] > 0) & (d[f"vol_monthly_y{y}"] > 0) & (d[f"pred_loo_y{y}"] > 0)]
    frames.append(pd.DataFrame({"act": d[f"mean_y{y}"], "pf": d[f"vol_monthly_y{y}"],
                                "loo": d[f"pred_loo_y{y}"], "leaky": d[f"pred_leaky_y{y}"]}))
pool = pd.concat(frames, ignore_index=True)
pooled = {"proforma": block(pool.act, pool.pf), "coldstart_loo": block(pool.act, pool.loo),
          "coldstart_leaky": block(pool.act, pool.leaky)}
e_m = np.abs(np.log(pool.act / pool.loo)); e_p = np.abs(np.log(pool.act / pool.pf))
pooled["model_win_rate"] = float((e_m < e_p).mean() * 100)
pooled["sign_test_p"] = float(stats.binomtest(int((e_m < e_p).sum()), len(pool), 0.5).pvalue)
results["pooled"] = pooled

# mature: age-gated actual vs proforma Y5 target vs model plateau (LOO)
dm = site[site.mature_ok].dropna(subset=["vol_monthly_y5", "plateau_loo", "mature_wash"])
dm = dm[(dm.plateau_loo > 0) & (dm.vol_monthly_y5 > 0)]
results["mature"] = {"proforma_y5": block(dm.mature_wash, dm.vol_monthly_y5),
                     "coldstart_plateau_loo": block(dm.mature_wash, dm.plateau_loo),
                     "coldstart_plateau_leaky": block(dm.mature_wash, dm.plateau_leaky)}
e_m = np.abs(np.log(dm.mature_wash / dm.plateau_loo)); e_p = np.abs(np.log(dm.mature_wash / dm.vol_monthly_y5))
results["mature"]["model_win_rate"] = float((e_m < e_p).mean() * 100)
results["mature"]["sign_test_p"] = float(stats.binomtest(int((e_m < e_p).sum()), len(dm), 0.5).pvalue)
results["mature"]["sites"] = [
    {"name": str(r.match_client_name), "addr": str(r.address)[:70], "act": float(r.mature_wash),
     "proforma": float(r.vol_monthly_y5), "model": float(r.plateau_loo)} for _, r in dm.iterrows()]

# Y1 ratio strip data (for the median-vs-band explainer)
d1 = site[site.open_observed & site.size_y1.ge(10)].dropna(subset=["mean_y1", "vol_monthly_y1"])
d1 = d1[(d1.mean_y1 > 0) & (d1.vol_monthly_y1 > 0)]
results["y1_ratios"] = sorted((d1.mean_y1 / d1.vol_monthly_y1).round(3).tolist())

with open(f"{SCRATCH}/compare_results.json", "w") as f:
    json.dump(results, f, indent=1)

print("\n=== per-year head-to-head (identical sites) ===")
for y, yr in results["years"].items():
    p, c = yr["proforma"], yr["coldstart_loo"]
    print(f"{y}: n={p['n']:3d} | ratio {p['median_ratio']:.2f} vs {c['median_ratio']:.2f}"
          f" | MdAPE {p['MdAPE_vs_act']:.0f}% vs {c['MdAPE_vs_act']:.0f}%"
          f" | within20 {p['within_20']:.0f}% vs {c['within_20']:.0f}%"
          f" | miss {p['miss_factor']:.2f}x vs {c['miss_factor']:.2f}x"
          f" | rho {p['spearman']:.2f} vs {c['spearman']:.2f}"
          f" | model wins {yr['model_win_rate']:.0f}% (p={yr['sign_test_p']:.3f})")
pp, cc, ll = pooled["proforma"], pooled["coldstart_loo"], pooled["coldstart_leaky"]
print(f"\nPOOLED (n={pp['n']}): ratio {pp['median_ratio']:.2f} vs {cc['median_ratio']:.2f} (leaky {ll['median_ratio']:.2f})"
      f" | MdAPE {pp['MdAPE_vs_act']:.0f}% vs {cc['MdAPE_vs_act']:.0f}% (leaky {ll['MdAPE_vs_act']:.0f}%)"
      f" | within20 {pp['within_20']:.0f}% vs {cc['within_20']:.0f}% (leaky {ll['within_20']:.0f}%)"
      f" | miss {pp['miss_factor']:.2f} vs {cc['miss_factor']:.2f} (leaky {ll['miss_factor']:.2f})"
      f" | rho {pp['spearman']:.2f} vs {cc['spearman']:.2f} (leaky {ll['spearman']:.2f})"
      f" | model wins {pooled['model_win_rate']:.0f}% (p={pooled['sign_test_p']:.4f})")
mm_ = results["mature"]
print(f"\nMATURE (n={mm_['proforma_y5']['n']}): proforma ratio {mm_['proforma_y5']['median_ratio']:.2f}"
      f" vs model {mm_['coldstart_plateau_loo']['median_ratio']:.2f}"
      f" | MdAPE {mm_['proforma_y5']['MdAPE_vs_act']:.0f}% vs {mm_['coldstart_plateau_loo']['MdAPE_vs_act']:.0f}%"
      f" | rho {mm_['proforma_y5']['spearman']:.2f} vs {mm_['coldstart_plateau_loo']['spearman']:.2f}"
      f" | model wins {mm_['model_win_rate']:.0f}% (p={mm_['sign_test_p']:.4f})")
