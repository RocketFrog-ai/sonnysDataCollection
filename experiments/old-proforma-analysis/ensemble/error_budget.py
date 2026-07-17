"""Error budget: where does the ~40% forecast error come from, and what would fix it?"""
import glob

import numpy as np
import pandas as pd
from scipy import stats

pd.set_option("display.width", 240)
SCRATCH = "/Users/dhruvsood/sonnysDataCollection/experiments/old-proforma-analysis/ensemble/results"
ROOT = "/Users/dhruvsood/sonnysDataCollection/experiments/old-proforma-analysis"
PANEL = "/Users/dhruvsood/sonnysDataCollection/proforma/data/panel/main-data-v2-stitched.csv"

def ym(x):
    try:
        mo, yr = str(x).split("-"); return int(yr) * 12 + int(mo)
    except Exception:
        return np.nan

def mdape(act, prd):
    act, prd = np.asarray(act, float), np.asarray(prd, float)
    return np.median(np.abs(act - prd) / act) * 100

def within(act, prd, b=0.2):
    r = np.asarray(act, float) / np.asarray(prd, float)
    return (np.abs(r - 1) <= b).mean() * 100

# ---------- eval table (same construction as before) ----------
pred = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{SCRATCH}/chunk_[0-9]*.csv"))], ignore_index=True)
s = pd.read_csv(f"{ROOT}/old-proforma-combined.csv")
m = pd.read_csv(f"{ROOT}/old-proforma-combined-monthly.csv")
site = s[s.match_status.str.startswith("matched") & (s.actuals_suspect != True)].copy()
site["open_idx"] = site.match_operational_start.map(ym)
mm = m[m.year.notna() & (m.imputed == 0)].copy()
mm = mm[mm.source_file.isin(site.source_file)]
mm["cal_idx"] = mm.year.astype(int) * 12 + mm.month.astype(int)
mm = mm.merge(site[["source_file", "open_idx"]], on="source_file")
mm["op_year"] = (mm.cal_idx - mm.open_idx) // 12 + 1
fc = mm.groupby("source_file").cal_idx.min(); lc = mm.groupby("source_file").cal_idx.max()
site = site.merge(fc.rename("first_cal"), left_on="source_file", right_index=True, how="left")
site = site.merge(lc.rename("last_cal"), left_on="source_file", right_index=True, how="left")
site["open_observed"] = (site.open_idx > 2020 * 12 + 1) & (site.open_idx >= site.first_cal - 1)
site["age_mo"] = site.last_cal - site.open_idx + 1
opy = (mm[mm.op_year.between(1, 6)].groupby(["source_file", "op_year"]).wash_count
       .agg(["mean", "size"]).unstack())
opy.columns = [f"{a}_y{int(b)}" for a, b in opy.columns]
site = site.merge(opy, left_on="source_file", right_index=True, how="left")
last12 = (mm.sort_values(["source_file", "cal_idx"]).groupby("source_file")
          .apply(lambda g: pd.Series({"mature_wash": g.tail(12).wash_count.median(),
                                      "mature_n": len(g.tail(12))}), include_groups=False))
site = site.merge(last12, left_on="source_file", right_index=True, how="left")
site["mature_ok"] = (site.age_mo >= 24) & (site.mature_n >= 6) & (site.mature_wash > 0)
site = site.merge(pred, on="source_file", how="left")

print("=" * 96)
print("A. PERSISTENCE: how well does the site's OWN previous year predict next year? (the 112 sites)")
print("=" * 96)
for y in (1, 2, 3, 4):
    d = site[site.open_observed & site[f"size_y{y}"].ge(10) & site.get(f"size_y{y+1}", pd.Series(dtype=float)).ge(10)]
    d = d[(d[f"mean_y{y}"] > 0) & (d[f"mean_y{y+1}"] > 0)]
    if len(d) < 8: continue
    print(f"  Y{y}->Y{y+1}: n={len(d):3d}  MdAPE={mdape(d[f'mean_y{y+1}'], d[f'mean_y{y}']):5.1f}%  "
          f"within±20%={within(d[f'mean_y{y+1}'], d[f'mean_y{y}']):4.0f}%  "
          f"spearman={stats.spearmanr(d[f'mean_y{y}'], d[f'mean_y{y+1}'])[0]:.2f}")

print()
print("=" * 96)
print("B. PANEL-WIDE year-over-year noise (ALL panel sites, mature years >=3, >=10 obs months each)")
print("=" * 96)
p = pd.read_csv(PANEL, low_memory=False)
p["wash"] = p.mem_wash_count.fillna(0) + p.ret_wash_count.fillna(0)
p = p[p.imputed == 0]
p["open_idx"] = p.operational_start.map(ym)
p["cal_idx"] = p.year * 12 + p.month
p["op_year"] = (p.cal_idx - p.open_idx) // 12 + 1
p["key"] = p.client_id.astype(str) + "::" + p.site_id.astype(str)
g = (p[p.op_year.between(1, 8)].groupby(["key", "op_year"]).wash
     .agg(["mean", "size"]).reset_index())
g = g[(g["size"] >= 10) & (g["mean"] > 0)]
g = g.pivot(index="key", columns="op_year", values="mean")
pairs_mat, pairs_ramp = [], []
for y in range(1, 8):
    if y in g.columns and y + 1 in g.columns:
        dd = g[[y, y + 1]].dropna()
        tgt = pairs_mat if y >= 3 else pairs_ramp
        tgt.append(pd.DataFrame({"a": dd[y], "b": dd[y + 1]}))
mat = pd.concat(pairs_mat); ramp = pd.concat(pairs_ramp)
print(f"  mature pairs (Y3+ -> next): n={len(mat)}  MdAPE={mdape(mat.b, mat.a):.1f}%  within±20%={within(mat.b, mat.a):.0f}%")
print(f"  ramp pairs (Y1/Y2 -> next): n={len(ramp)}  MdAPE={mdape(ramp.b, ramp.a):.1f}%  within±20%={within(ramp.b, ramp.a):.0f}%")

print()
print("=" * 96)
print("C. ORACLE LEVEL x MODEL RAMP: if we knew the TRUE mature level, how good are Y1..Y3?")
print("=" * 96)
d = site[site.mature_ok & site.open_observed & (site.plateau_loo > 0)].copy()
for y in (1, 2, 3):
    dd = d[d[f"size_y{y}"].ge(10) & d[f"mean_y{y}"].gt(0)].dropna(subset=[f"pred_loo_y{y}"])
    oracle = dd.mature_wash * (dd[f"pred_loo_y{y}"] / dd.plateau_loo)
    print(f"  Y{y}: n={len(dd):3d}  model-LOO MdAPE={mdape(dd[f'mean_y{y}'], dd[f'pred_loo_y{y}']):5.1f}%  "
          f"ORACLE-level MdAPE={mdape(dd[f'mean_y{y}'], oracle):5.1f}%  "
          f"oracle within±20%={within(dd[f'mean_y{y}'], oracle):4.0f}%")

print()
print("=" * 96)
print("D. CHEAP FIXES on the model: global level calibration (x0.78, in-sample) ")
print("=" * 96)
rows = []
for y in range(1, 6):
    dd = site[site.open_observed & site[f"size_y{y}"].ge(10) & site[f"mean_y{y}"].gt(0)
              ].dropna(subset=[f"pred_loo_y{y}", f"vol_monthly_y{y}"])
    cal = dd[f"pred_loo_y{y}"] * 0.78
    rows.append({"yr": f"Y{y}", "n": len(dd),
                 "proforma": mdape(dd[f"mean_y{y}"], dd[f"vol_monthly_y{y}"]),
                 "model_raw": mdape(dd[f"mean_y{y}"], dd[f"pred_loo_y{y}"]),
                 "model_cal": mdape(dd[f"mean_y{y}"], cal),
                 "cal_within20": within(dd[f"mean_y{y}"], cal),
                 "pf_within20": within(dd[f"mean_y{y}"], dd[f"vol_monthly_y{y}"])})
print(pd.DataFrame(rows).round(1).to_string(index=False))

print()
print("=" * 96)
print("E. UNUSED SIGNAL: do the proforma's site factors explain the model's residual error?")
print("=" * 96)
d = site[site.mature_ok & (site.plateau_loo > 0)].copy()
d["resid"] = np.log(d.mature_wash / d.plateau_loo)
for f in ["factor_pay_stations_score", "factor_free_vacuum_slots_score",
          "factor_type_of_site_score", "cumulative_site_score", "traffic_count"]:
    ok = d[f].notna()
    r, pv = stats.spearmanr(d.loc[ok, f], d.loc[ok, "resid"])
    print(f"  {f:34s} vs LOO residual: r={r:+.2f} (p={pv:.3f}, n={ok.sum()})")
# combined: capacity-adjusted plateau (median-ratio per pay-station tier, LOO-ish rough cut)
d["paytier"] = d.factor_pay_stations_choice.astype(str).str.upper().str.strip().replace({"2.0": "2"})
tier_adj = d.groupby("paytier").resid.transform(lambda v: (np.exp(v.median()) if len(v) >= 5 else 1.0))
adj = d.plateau_loo * tier_adj
print(f"\n  mature MdAPE: model plateau LOO {mdape(d.mature_wash, d.plateau_loo):.1f}%  "
      f"-> + pay-station tier adjustment {mdape(d.mature_wash, adj):.1f}%  (in-sample, illustrative)")
print(f"  open-cohort effect on residual: r={stats.spearmanr(d.match_operational_start.str[-4:].astype(int), d.resid)[0]:+.2f}")
