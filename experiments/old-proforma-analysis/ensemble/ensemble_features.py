"""Build the per-site feature table for the ensemble (all leakage-safe by construction).

Features per matched proforma site:
  - coldstart Model-3 LOO plateau + per-year ramp multipliers (from the benchmark chunks)
  - proforma projections (vol_monthly_y1..y5) + capacity factor scores + traffic
  - era-aware local anchor: median monthly washes of <=20km neighbours that were >=24mo old
    DURING the site's first 12 calendar months (excl. self/twin/<=0.2km) -- the 'what did this
    market actually support in that era' time-series signal
  - open cohort year
Targets: mature_wash (age-gated) and actual op-year means y1..y5.
"""
import glob

import numpy as np
import pandas as pd

SCRATCH = "/Users/dhruvsood/sonnysDataCollection/experiments/old-proforma-analysis/ensemble/results"
ROOT = "/Users/dhruvsood/sonnysDataCollection/experiments/old-proforma-analysis"
PANEL = "/Users/dhruvsood/sonnysDataCollection/proforma/data/panel/main-data-v2-stitched.csv"
EARTH_KM = 6371.0088

def hav(lat1, lon1, lat2, lon2):
    r = np.radians
    lat1, lon1, lat2, lon2 = r(lat1), r(lon1), r(lat2), r(lon2)
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def ym(x):
    try:
        mo, yr = str(x).split("-"); return int(yr) * 12 + int(mo)
    except Exception:
        return np.nan

# ── eval sites + actuals (identical construction to the benchmark) ──
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

# ── normalized capacity scores (already numeric) + choices ──
def norm_choice(v):
    if pd.isna(v): return np.nan
    return str(v).upper().strip().replace("MUTIPLE", "MULTIPLE").replace("2.0", "2")
site["pay_choice"] = site.factor_pay_stations_choice.map(norm_choice)

# ── era-aware local anchor from the full panel ──
p = pd.read_csv(PANEL, low_memory=False)
p["wash"] = p.mem_wash_count.fillna(0) + p.ret_wash_count.fillna(0)
p = p[p.imputed == 0]
p["key"] = p.client_id.astype(str) + "::" + p.site_id.astype(str)
p["open_idx"] = p.operational_start.map(ym)
p["cal_idx"] = p.year * 12 + p.month
psite = p.groupby("key").agg(lat=("lat", "first"), lon=("lon", "first"),
                             p_open=("open_idx", "first")).reset_index()
psite = psite.dropna(subset=["lat", "lon"])
pw = p[["key", "cal_idx", "wash"]]

rows = []
for _, r in site.iterrows():
    lat, lon, o = r.match_lat, r.match_lon, r.open_idx
    excl = {f"{r.match_client_id}::{int(r.match_site_id)}"}
    if isinstance(r.match_secondary_segments, str) and r.match_secondary_segments:
        for part in r.match_secondary_segments.split(";"):
            c, i = part.rsplit(":", 1)
            excl.add(f"{c}::{int(i)}")
    d = hav(lat, lon, psite.lat.values, psite.lon.values)
    near = psite[(d <= 20.0)].copy()
    near["d"] = d[d <= 20.0]
    near = near[(near.d > 0.2) | (~near.key.isin(excl)) & (near.d > 0.2)]  # drop <=0.2km AND excl keys
    near = near[~near.key.isin(excl)]
    # neighbours >=24mo old at the target's open
    near = near[(o - near.p_open) >= 24]
    if len(near):
        w = pw[pw.key.isin(set(near.key)) & pw.cal_idx.between(o, o + 11)]
        per = w.groupby("key").agg(mu=("wash", "mean"), n=("wash", "size"))
        per = per[per.n >= 6]
        era_anchor = float(per.mu.median()) if len(per) else np.nan
        n_era = int(len(per))
    else:
        era_anchor, n_era = np.nan, 0
    rows.append({"source_file": r.source_file, "era_anchor": era_anchor, "n_era": n_era})
site = site.merge(pd.DataFrame(rows), on="source_file")

# ── assemble the modelling table ──
out_cols = {
    "source_file": site.source_file,
    "open_year": site.match_operational_start.str[-4:].astype(float),
    "open_observed": site.open_observed,
    "mature_ok": site.mature_ok,
    "mature_wash": site.mature_wash,
    "plateau_loo": site.plateau_loo,
    "era_anchor": site.era_anchor,
    "n_era": site.n_era,
    "nmat_loo": site.nmat_loo,
    "traffic_count": site.traffic_count,
    "pay": site.factor_pay_stations_score,
    "vac": site.factor_free_vacuum_slots_score,
    "tos": site.factor_type_of_site_score,
    "css": site.cumulative_site_score,
    "pay_choice": site.pay_choice,
}
for y in range(1, 6):
    out_cols[f"act_y{y}"] = site[f"mean_y{y}"]
    out_cols[f"nobs_y{y}"] = site[f"size_y{y}"]
    out_cols[f"pf_y{y}"] = site[f"vol_monthly_y{y}"]
    out_cols[f"cs_y{y}"] = site[f"pred_loo_y{y}"]
F = pd.DataFrame(out_cols)
F["ramp_y1"] = F.cs_y1 / F.plateau_loo
for y in range(2, 6):
    F[f"ramp_y{y}"] = F[f"cs_y{y}"] / F.plateau_loo
F.to_csv(f"{SCRATCH}/ensemble_features.csv", index=False)
print(f"wrote {len(F)} sites; mature_ok={F.mature_ok.sum()}, era_anchor coverage={F.era_anchor.notna().sum()}")
print(F[["era_anchor", "n_era", "plateau_loo", "mature_wash"]].describe().round(1).to_string())
r = F[F.mature_ok & F.era_anchor.notna()]
import scipy.stats as st
print("\nera_anchor vs mature actual: spearman",
      round(st.spearmanr(r.era_anchor, r.mature_wash)[0], 2),
      "| plateau_loo vs mature:", round(st.spearmanr(r.plateau_loo, r.mature_wash)[0], 2))
print("residual log(act/plateau) vs log(era/plateau):",
      round(st.spearmanr(np.log(r.mature_wash / r.plateau_loo), np.log(r.era_anchor / r.plateau_loo))[0], 2))
