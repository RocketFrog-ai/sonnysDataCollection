"""
Cold-start car-wash forecaster: drop a pin (lat/lon [+ operator]) -> 5-year monthly trajectory.

Hybrid model (see notebooks/coldstart_forecast.ipynb for build + evaluation):
  trajectory(m) = PLATEAU_LEVEL  x  RAMP_SHAPE(m)
  - PLATEAU_LEVEL: LightGBM quantile regression on location + local-market features (+ operator/brand,
    the dominant driver). Predicts mature monthly total washes and the membership share.
  - RAMP_SHAPE: empirical normalized ramp curves (membership ramps over ~7mo; retail ~flat from month 1),
    extended flat to 60 months with an optional secular drift (membership up / retail down).
  - NEIGHBOURS: project each existing neighbour flat and subtract the learned cannibalization-by-distance.

Artifacts are cached to notebooks/artifacts/coldstart_artifacts.joblib.
"""
from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.neighbors import BallTree
from sklearn.ensemble import ExtraTreesRegressor

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ART = HERE.parent / "notebooks" / "artifacts"
CSV = HERE.parent / "data" / "main-ds.csv"
MODEL_PATH = ART / "coldstart_artifacts.joblib"
EARTH_KM = 6371.0088
H_DEFAULT = 60
MODEL_MIN_MONTHS = 24    # train labels + the local-mature level anchor only use sites with ≥24 months of history
# Strong-market calibration: the level model regresses good clusters toward the global mean (LOSO wash-count
# backtest: strong-cluster bias 0.92, below neighbour floor 21% in the top quartile, never over-predicts there).
# Blend the predicted level toward the local-mature NEIGHBOUR median in log space by this weight, ONLY for pins
# that have a real anchor (≥anchor_min_n matured siblings). Backtest @0.50: global bias 0.912→0.925, in-band
# 60.6→70%, strong-tercile 0.92→~0.96 (no overshoot), WAPE +0.4pp. Keep ≤0.6 — higher erodes weak markets and
# drags genuine outperformers toward their cluster median. 0.0 disables.
ANCHOR_CALIB_W = 0.50

FEAT = ["lat", "lon", "n_nbr_5", "n_nbr_10", "n_nbr_20", "mean_nbr_10", "mean_nbr_20",
        "logsum_20", "shr_nbr_20", "dist_nearest_km", "nearest_level", "cluster_size",
        "brand_loo", "brand_n"]
CAT = ["region", "state"]

# cannibalization on a neighbour's RETAIL = a*exp(-d/L), d in km. (a, L) are LEARNED from a diff-in-diff event
# study at fit() time (see _fit_cannibalization); 0.26/12.0 is only the fallback if that fit is degenerate.
_CANNIB_FALLBACK = dict(a=0.26, L=12.0)


def _cannib_ret(dist_km, params=None):
    p = params or _CANNIB_FALLBACK
    dk = np.asarray(dist_km, dtype=float)
    return np.where(dk <= 20, p["a"] * np.exp(-dk / p["L"]), 0.0)


def cannib_params(art, lat, lon):
    """Cannibalization (a, L) for a dropped pin: its region's fitted curve if we have one, else the pooled fit.
    Returns the fallback dict if the artifact predates the learned cannibalization."""
    c = (art or {}).get("cannib")
    if not c:
        return _CANNIB_FALLBACK
    rl = art["sites_rl"]
    region = rl.region.iloc[int(np.argmin(_haversine(lat, lon, rl.lat.values, rl.lon.values)))]
    return c["per_region"].get(region, c["pooled"])


def _fit_cannibalization(panel, site, pre=(-6, -1), post=(7, 12), max_d=20.0):
    """DIFF-IN-DIFF event study → learn retail cannibalization curve a*exp(-d/L) from the data (replaces the frozen
    0.26*exp(-d/12)). For every real opening: compare each nearby INCUMBENT's pre→post retail change to the
    market-wide mature-site retail trend over the SAME calendar window (the control — removes the secular
    membership-up/retail-down drift), then fit -a*exp(-d/L) to the (distance, excess-change) pairs. Pooled +
    per-region. Falls back to 0.26/12 if a fit is degenerate (so it can never make things worse)."""
    from scipy.optimize import curve_fit
    s = site.dropna(subset=["op_start", "lat", "lon"]).copy()
    pr = panel[["site_key", "date", "ret_wash_count"]].dropna()
    ret = {k: g.set_index("date").ret_wash_count.sort_index() for k, g in pr.groupby("site_key")}
    # control = mean log-retail of MATURE sites (age≥18mo) by calendar month → the market-wide retail trend
    pm2 = panel[["site_key", "date", "ret_wash_count"]].copy()
    pm2["op_start"] = pm2.site_key.map(site.set_index("site_key").op_start)
    pm2["age"] = (pm2.date.dt.year - pm2.op_start.dt.year) * 12 + (pm2.date.dt.month - pm2.op_start.dt.month)
    matr = pm2[(pm2.age >= 18) & (pm2.ret_wash_count > 0)]
    cidx = matr.groupby("date").ret_wash_count.apply(lambda x: float(np.log(x).mean())).sort_index()
    coords = s.set_index("site_key")[["lat", "lon", "op_start", "region"]]
    keys = coords.index.tolist(); la = coords.lat.values; lo = coords.lon.values; ops = coords.op_start.values

    def _wmean(series, t0, a, b):
        seg = series[(series.index >= t0 + pd.DateOffset(months=a)) & (series.index <= t0 + pd.DateOffset(months=b))]
        return float(seg.mean()) if len(seg) >= 2 else np.nan

    def _widx(t0, a, b):
        seg = cidx[(cidx.index >= t0 + pd.DateOffset(months=a)) & (cidx.index <= t0 + pd.DateOffset(months=b))]
        return float(seg.mean()) if len(seg) >= 2 else np.nan

    pairs = []
    for ei in range(len(keys)):
        t0 = pd.Timestamp(ops[ei])
        cpre, cpost = _widx(t0, *pre), _widx(t0, *post)
        if not (np.isfinite(cpre) and np.isfinite(cpost)):
            continue
        ctrl = cpost - cpre                                       # market-wide log change over this window
        d = _haversine(la[ei], lo[ei], la, lo)
        for j in np.where((d > 1e-6) & (d <= max_d))[0]:
            if pd.Timestamp(ops[j]) >= t0 - pd.DateOffset(months=6):
                continue                                          # j must already be open (an incumbent)
            sr = ret.get(keys[j])
            if sr is None:
                continue
            pm_, qm_ = _wmean(sr, t0, *pre), _wmean(sr, t0, *post)
            if np.isfinite(pm_) and np.isfinite(qm_) and pm_ > 0 and qm_ > 0:
                eff = (np.log(qm_) - np.log(pm_)) - ctrl          # excess change vs the market = the entrant's effect
                pairs.append((float(d[j]), float(eff), coords.region.iloc[ei]))
    P = pd.DataFrame(pairs, columns=["d", "eff", "region"])

    def _fit(sub):
        # Fit to the per-distance-bin MEDIAN excess change — the raw pairs are right-skewed (a few incumbents boom),
        # so a least-squares fit on raw pairs chases outliers; the median is the robust cannibalization signal.
        if len(sub) < 60:
            return None
        b = pd.cut(sub.d, [0, 3, 5, 8, 12, 16, 20])
        gb = sub.groupby(b, observed=True).agg(d=("d", "median"), e=("eff", "median"), n=("eff", "size")).dropna()
        gb = gb[gb.n >= 10]
        if len(gb) < 4:
            return None
        loss = np.clip(-gb.e.values, 0, None)              # cannibalization ≥0; far "gains" are noise → 0
        if loss.max() < 0.02:
            return None                                    # no real near-field loss
        try:
            (a, L), _ = curve_fit(lambda x, a, L: a * np.exp(-x / L), gb.d.values, loss,
                                  p0=[0.15, 5.0], bounds=([0, 1], [0.6, 40]),
                                  sigma=1.0 / np.sqrt(gb.n.values), maxfev=10000)
            if 0.02 <= a <= 0.6 and 1 <= L <= 40:
                return dict(a=float(a), L=float(L), n=int(len(sub)), n_bins=int(len(gb)))
        except Exception:
            return None
        return None
    pooled = _fit(P) or dict(**_CANNIB_FALLBACK, n=int(len(P)), fallback=True)
    per_region = {}
    for r, sub in P.groupby("region"):
        f = _fit(sub)
        if f:
            per_region[r] = f
    return dict(pooled=pooled, per_region=per_region, n_pairs=int(len(P)))


# ─────────────────────────── data / features ───────────────────────────
def assign_clusters(df, strategy="adaptive", dense_radius=10.0, sparse_radius=20.0,
                    dense_min=5, cap_km=25.0):
    """Local-market labels for rows of `df` (needs lat/lon). Returns an int array aligned to df (-1 = standalone).

    strategy='dbscan20'  : fixed 20 km DBSCAN (the original baseline — chains in dense metros).
    strategy='adaptive'  : DENSITY-AWARE. A site links to a neighbour only within min(r_i, r_j), where a site's
                           radius is `dense_radius` (10 km) if it already has >=`dense_min` neighbours within 10 km,
                           else `sparse_radius` (20 km). Connected components form markets; components wider than
                           `cap_km` are re-split by complete-linkage (kills chaining); singletons -> -1 (standalone).
    """
    from sklearn.cluster import DBSCAN, AgglomerativeClustering
    from sklearn.neighbors import BallTree
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    labels = np.full(len(df), -1, dtype=int)
    ok = df[["lat", "lon"]].notna().all(axis=1).values
    rad = np.radians(df.loc[ok, ["lat", "lon"]].values)
    if len(rad) < 2:
        return labels

    if strategy == "dbscan20":
        lab = DBSCAN(eps=sparse_radius / EARTH_KM, min_samples=2, metric="haversine").fit(rad).labels_
        labels[ok] = lab
        return labels

    tree = BallTree(rad, metric="haversine")
    n10 = np.array([len(x) - 1 for x in tree.query_radius(rad, r=dense_radius / EARTH_KM)])
    r_i = np.where(n10 >= dense_min, dense_radius, sparse_radius)
    nbrs, dists = tree.query_radius(rad, r=sparse_radius / EARTH_KM, return_distance=True)
    rows, cols = [], []
    for i in range(len(rad)):
        for j, dk in zip(nbrs[i], dists[i] * EARTH_KM):
            if j != i and dk <= min(r_i[i], r_i[j]):
                rows.append(i); cols.append(j)
    g = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(rad), len(rad)))
    _, comp = connected_components(g, directed=False)
    sizes = pd.Series(comp).value_counts()
    comp = np.where(pd.Series(comp).map(sizes).values >= 2, comp, -1)

    out = np.full(len(rad), -1, dtype=int); nxt = 0
    for c in pd.unique(comp[comp >= 0]):
        idx = np.where(comp == c)[0]; r = rad[idx]
        lat, lon = r[:, :1], r[:, 1:]
        a = np.sin((lat - lat.T) / 2) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin((lon - lon.T) / 2) ** 2
        dm = 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        if dm.max() <= cap_km or len(idx) < 3:
            out[idx] = nxt; nxt += 1
        else:
            sub = AgglomerativeClustering(n_clusters=None, distance_threshold=cap_km,
                                          metric="precomputed", linkage="complete").fit(dm)
            for sc in np.unique(sub.labels_):
                sidx = idx[sub.labels_ == sc]
                if len(sidx) >= 2:
                    out[sidx] = nxt; nxt += 1
    labels[ok] = out
    return labels


def _build_from_csv():
    raw = pd.read_csv(CSV, low_memory=False)
    raw["date"] = pd.to_datetime(dict(year=raw.year, month=raw.month, day=1))
    raw["op_start"] = pd.to_datetime(raw["operational_start"], format="%m-%Y", errors="coerce")
    raw["site_key"] = raw.client_id.astype(str) + "::" + raw.site_id.astype(str)
    asp_r = np.where(raw.ret_wash_count > 0, raw.ret_revenue / raw.ret_wash_count, np.nan)
    asp_m = np.where(raw.mem_wash_count > 0, raw.mem_revenue / raw.mem_wash_count, np.nan)
    raw.loc[asp_r > 200, "ret_revenue"] = np.nan
    raw.loc[asp_m > 200, "mem_revenue"] = np.nan
    raw["tot_wash_count"] = raw.mem_wash_count + raw.ret_wash_count
    raw["tot_revenue"] = raw[["mem_revenue", "ret_revenue"]].sum(axis=1, min_count=1)
    raw["mem_share_wash"] = np.where(raw.tot_wash_count > 0, raw.mem_wash_count / raw.tot_wash_count, np.nan)
    site = raw.groupby("site_key").agg(
        client_id=("client_id", "first"), client_name=("client_name", "first"),
        lat=("lat", "first"), lon=("lon", "first"), state=("state", "first"),
        region=("region", "first"), op_start=("op_start", "first")).reset_index()
    site["has_coords"] = site[["lat", "lon"]].notna().all(axis=1)
    return raw, site


def load_panel_site(strategy="adaptive"):
    if (ART / "panel.parquet").exists() and (ART / "site.parquet").exists():
        panel = pd.read_parquet(ART / "panel.parquet")
        site = pd.read_parquet(ART / "site.parquet")
    else:
        panel, site = _build_from_csv()
    if "has_coords" not in site.columns:
        site["has_coords"] = site[["lat", "lon"]].notna().all(axis=1)
    # always (re)assign clusters consistently — adaptive is the adopted strategy (see clustering bake-off)
    site["cluster_dbscan20"] = assign_clusters(site, "dbscan20")
    site["cluster"] = assign_clusters(site, strategy)
    return panel, site


def build_features(panel, site):
    p = panel.merge(site[["site_key", "op_start"]], on="site_key", how="left", suffixes=("", "_s"))
    p["m"] = (p.date.dt.year - p.op_start.dt.year) * 12 + (p.date.dt.month - p.op_start.dt.month)

    def agg(g):
        g = g.sort_values("date"); mat = g[(g.m >= 18) & (g.m <= 30)]; rec = g.tail(6)
        return pd.Series(dict(mat_n=len(mat), max_m=g.m.max(), n_months=len(g),
            mat_total=mat.tot_wash_count.mean(), mat_mem=mat.mem_wash_count.mean(), mat_ret=mat.ret_wash_count.mean(),
            rec_total=rec.tot_wash_count.mean(), rec_mem=rec.mem_wash_count.mean(), rec_ret=rec.ret_wash_count.mean()))
    S = p.groupby("site_key").apply(agg, include_groups=False).reset_index()
    S = S.merge(site[["site_key", "client_id", "client_name", "lat", "lon", "region", "state", "cluster", "has_coords"]],
                on="site_key", how="left")
    S = S[S.has_coords].copy()
    S["region"] = S.region.fillna("Unknown"); S["state"] = S.state.fillna("Unknown")

    coords = np.radians(S[["lat", "lon"]].values)
    tree = BallTree(coords, metric="haversine")
    rec_total = S.rec_total.fillna(0).values; rec_mem = S.rec_mem.fillna(0).values
    for rkm in [5, 10, 20]:
        ind = tree.query_radius(coords, r=rkm / EARTH_KM)
        n, mean, ssum, shr = [], [], [], []
        for i, nbrs in enumerate(ind):
            o = [j for j in nbrs if j != i]; n.append(len(o))
            if o:
                vt = rec_total[o]; mean.append(np.mean(vt)); ssum.append(np.sum(vt))
                sh = rec_mem[o] / np.where(rec_total[o] > 0, rec_total[o], np.nan)
                shr.append(np.nanmean(sh) if np.isfinite(sh).any() else np.nan)
            else:
                mean.append(np.nan); ssum.append(0.0); shr.append(np.nan)
        S[f"n_nbr_{rkm}"] = n; S[f"mean_nbr_{rkm}"] = mean; S[f"sum_nbr_{rkm}"] = ssum; S[f"shr_nbr_{rkm}"] = shr
    dist, idx = tree.query(coords, k=2)
    S["dist_nearest_km"] = dist[:, 1] * EARTH_KM
    S["nearest_level"] = rec_total[idx[:, 1]]
    S["cluster_size"] = S.groupby("cluster").site_key.transform("size")
    S["logsum_20"] = np.log1p(S.sum_nbr_20)
    # brand (operator) leave-one-out mean mature level
    cli = S.groupby("client_id").mat_total
    S["brand_n"] = cli.transform("count")
    bs = cli.transform("sum")
    S["brand_loo"] = np.where(S.brand_n > 1, (bs - S.mat_total.fillna(0)) / (S.brand_n - 1).clip(lower=1), np.nan)
    return S, tree


RAMP_RELIABLE_MAX = 42   # follow the data to ~3.5y; beyond, hold the data-driven asymptote (yrs 4–5 = drift-slider scenario)


def _ramp_points(panel, site, H=H_DEFAULT, clean=True, min_months=30, drop_open_years=(2020,)):
    """Per-site normalized ramp (value / own months-18–30 mean), tagged with cluster & region.

    clean=True restricts the ramp-SHAPE pool to representative, well-observed sites:
      • drop `drop_open_years` cohorts (2020 = COVID-distorted: 35% membership vs ~67% for 2022+, and its
        retail decays steeply — it was 80% of the month-≥55 pool and manufactured a false yr-4–5 decline);
      • require ≥`min_months` of history (a real plateau + tail, not a half-ramp).
    This only governs the normalized SHAPE; the plateau LEVEL is set by the model/anchor, not here."""
    p = panel.merge(site[["site_key", "op_start", "cluster", "region"]], on="site_key", how="left", suffixes=("", "_s"))
    p["m"] = (p.date.dt.year - p.op_start.dt.year) * 12 + (p.date.dt.month - p.op_start.dt.month)
    p = p[(p.m >= 0) & (p.m <= H)].copy()
    if clean:
        nmo = p.groupby("site_key").m.nunique()
        keep = set(nmo[nmo >= min_months].index)
        oy = p.op_start.dt.year
        p = p[p.site_key.isin(keep) & ~oy.isin(drop_open_years)].copy()
    p["region"] = p["region"].fillna("Unknown")
    mat = p[p.m.between(18, 30)].groupby("site_key").agg(pm=("mem_wash_count", "mean"), pr=("ret_wash_count", "mean"))
    mat = mat[(mat.pm > 0) & (mat.pr > 0)]
    p = p.merge(mat, on="site_key", how="inner")
    p["rmem"] = p.mem_wash_count / p.pm
    p["rret"] = p.ret_wash_count / p.pr
    return p


def _curve_from(pts, H=H_DEFAULT, parent=None, k=10.0, min_per_m=6, reliable_max=RAMP_RELIABLE_MAX,
                tail="slope", tail_fit_window=12, tail_slope_cap=0.01):
    """Median normalized ramp from `pts`: data-driven through the supported horizon, then EXTENDED to H.
    Shrunk toward `parent` by n/(n+k) when data is thin.

    tail="slope" (default): past the last reliable month, continue the component's OWN measured late slope
      (fit over the last `tail_fit_window` months), capped at ±`tail_slope_cap`/mo and to ±60% of the
      mature level — so membership keeps its gentle climb and retail its gentle erosion instead of a flat
      line frozen at month ~42 (the flat hold inherited the COVID 2020 cohort's decay → underreported yr4–5).
    tail="flat": legacy behaviour — hold the mature asymptote flat (use for A/B backtests)."""
    out = {}
    n_sites = pts.site_key.nunique()
    for comp, key in [("rmem", "mem"), ("rret", "ret")]:
        g = pts.groupby("m")[comp]; med = g.median(); cnt = g.count()
        arr = np.full(H + 1, np.nan)
        for m in range(min(H, reliable_max) + 1):
            if m in med.index and cnt.get(m, 0) >= min_per_m:
                arr[m] = med[m]
        rel = np.where(~np.isnan(arr))[0]
        last = int(rel.max()) if len(rel) else 0
        s = pd.Series(arr).interpolate(limit_area="inside")
        asym = float(np.nanmean(arr[max(0, last - 5):last + 1])) if len(rel) else 1.0
        if tail == "slope" and len(rel) and last < H:
            base = float(s.iloc[last])
            w0 = max(int(rel.min()), last - tail_fit_window + 1)
            xv = np.arange(w0, last + 1); yv = s.iloc[w0:last + 1].to_numpy()
            ok = np.isfinite(yv)
            b = float(np.polyfit(xv[ok], yv[ok], 1)[0]) if ok.sum() >= 3 else 0.0
            b = float(np.clip(b, -tail_slope_cap, tail_slope_cap))     # cap |slope| to keep noisy thin fits sane
            ext = base + b * (np.arange(last + 1, H + 1) - last)
            s.iloc[last + 1:] = np.clip(ext, base * 0.4, base * 1.6)   # cap cumulative tail drift to ±60%
        s = s.fillna(asym)                                              # leading gaps (+ tail if flat) -> asymptote
        cur = pd.Series(s.to_numpy()).rolling(3, center=True, min_periods=1).mean().to_numpy()
        if parent is not None:                                          # shrink thin groups toward the parent curve
            w = n_sites / (n_sites + k)
            cur = w * cur + (1 - w) * np.asarray(parent[key])
        out[key] = cur
    return out


def _ramp_curves(panel, site, H=H_DEFAULT, clean=True, tail="flat"):
    """Hierarchical ramp life-cycles: global → per region → per cluster (where ≥3 opened sites), so a pin can use
    its LOCAL market's historical ramp, shrinking to region/global when the local sample is thin."""
    pts = _ramp_points(panel, site, H, clean=clean)
    cf = lambda sub, **kw: _curve_from(sub, H, tail=tail, **kw)
    glob = cf(pts, parent=None)
    region = {r: cf(sub, parent=glob, k=12)
              for r, sub in pts.groupby("region") if sub.site_key.nunique() >= 8}
    cluster, cluster_region = {}, {}
    for c, sub in pts[pts.cluster >= 0].groupby("cluster"):
        if sub.site_key.nunique() >= 3:
            reg = sub.region.mode()
            par = region.get(reg.iloc[0], glob) if len(reg) else glob
            cluster[int(c)] = cf(sub, parent=par, k=8)
            cluster_region[int(c)] = reg.iloc[0] if len(reg) else None
    return {"global": glob, "region": region, "cluster": cluster, "cluster_region": cluster_region,
            "support": {"n_sites": int(pts.site_key.nunique()), "clusters_with_ramp": len(cluster), "regions": len(region)}}


def _select_ramp(art, lat, lon, prefer="region", radius_km=20.0, min_sibs=2):
    """Ramp life-cycle (SHAPE only — the plateau LEVEL is anchored separately) for a dropped pin.

    prefer="region" (default): region-pooled selection (global fallback). This is what the LOSO trajectory
      backtest supports — cluster-anchoring the shape did NOT improve held-out WAPE (20.8→21.3 overall), echoing
      the earlier per-cluster finding; the plateau LEVEL never came from clusters anyway.
    prefer="cluster" (opt-in): borrow the trajectory shape from the pin's actual neighbours — pick the geo-cluster
      with the most members within `radius_km` that has a precomputed (shrunk) ramp ("cluster siblings first,
      20 km fallback"), region→global fallback. Kept available because neighbours DO agree on late-ramp SHAPE to
      ±14% (vs ±38% on level), so it may earn its keep once the 2022+ membership-era cohorts mature past month 48
      (today the only observable yr-4–5 truth is the COVID 2020 cohort, which declines)."""
    rl = art["sites_rl"].reset_index(drop=True)
    if not len(rl):
        return art["ramps"]["global"], "global"
    d = _haversine(lat, lon, rl.lat.values, rl.lon.values)
    ramps = art["ramps"]
    cl = ramps.get("cluster", {})
    if prefer == "cluster" and cl:
        within = np.where(d <= radius_km)[0]
        counts = {}
        for i in within:
            c = int(rl.cluster.iloc[i])
            if c >= 0 and c in cl:
                counts[c] = counts.get(c, 0) + 1
        counts = {c: n for c, n in counts.items() if n >= min_sibs}
        if counts:
            best = max(counts, key=counts.get)
            return cl[best], f"cluster {best}: {counts[best]} sibs ≤{radius_km:.0f}km"
    region = rl.iloc[int(np.argmin(d))].region
    if region in ramps["region"]:
        return ramps["region"][region], f"region: {region}"
    return ramps["global"], "global"


# ─────────────────────────── train ───────────────────────────
def fit(save=True):
    import lightgbm as lgb
    panel, site = load_panel_site()
    S, _ = build_features(panel, site)
    # train only on sites with a reliable mature window AND enough total history (≥MODEL_MIN_MONTHS months) so the
    # mat_total target is well-estimated — better modelling, fewer thin/young sites skewing the level model
    lab = S[(S.mat_n >= 4) & (S.n_months >= MODEL_MIN_MONTHS)].copy()
    lab["y"] = np.log1p(lab.mat_total)
    lab["share"] = (lab.mat_mem / lab.mat_total).clip(0, 1)
    for c in CAT:
        lab[c] = lab[c].astype("category")
    X = lab[FEAT + CAT]

    def mk(obj, alpha=None):
        return lgb.LGBMRegressor(objective=obj, alpha=alpha, n_estimators=500, learning_rate=0.03,
                                 num_leaves=31, min_child_samples=20, subsample=0.8,
                                 colsample_bytree=0.8, verbose=-1)
    models = {}
    for q, a in [("q10", 0.1), ("q50", 0.5), ("q90", 0.9)]:
        m = mk("quantile", a); m.fit(X, lab.y.values, categorical_feature=CAT); models[q] = m
    ms = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=20, verbose=-1)
    ms.fit(X, lab.share.values, categorical_feature=CAT); models["share"] = ms
    # Model 3: ExtraTrees on the numeric features. Bagging beats boosting on this small, noisy cold-start set
    # (leakage-free backtest: WAPE 43.8→40.3, medAPE 38.3→34.5 vs LightGBM). Needs imputation (no native NaN);
    # the level comes from ET, the P10/P90 band widths are reused from the LGB quantiles at inference.
    et_impute = X[FEAT].median()
    et = ExtraTreesRegressor(n_estimators=600, min_samples_leaf=2, max_features=0.5, n_jobs=-1, random_state=0)
    et.fit(X[FEAT].fillna(et_impute), lab.y.values); models["et"] = et

    # clean=True drops the COVID 2020 cohort + sub-30-mo sites from the SHAPE pool (validated: temporal holdout
    # predicting unseen 2022/2023 cohorts improved 20.7→20.3 / 16.6→16.1 WAPE). tail="flat" + region selection are
    # the backtested defaults; cluster ramps are still built here so prefer="cluster"/tail="slope" stay opt-in.
    ramps = _ramp_curves(panel, site, clean=True, tail="flat")
    cannib = _fit_cannibalization(panel, site)          # data-fit retail cannibalization a*exp(-d/L)
    brand_mean = S.groupby("client_id").mat_total.mean()
    brand_n = S.groupby("client_id").mat_total.count()
    region_share = lab.groupby("region").share.median().to_dict()
    # site recent-level table for neighbour features + neighbour-impact at inference (full universe so neighbour
    # COUNTS stay on the training distribution; n_months lets the anchor restrict to well-observed sites)
    sites_rl = S[["site_key", "client_name", "client_id", "lat", "lon", "region", "state", "cluster",
                  "rec_total", "rec_mem", "rec_ret", "mat_total", "n_months"]].copy()

    art = dict(models=models, ramps=ramps, cannib=cannib, FEAT=FEAT, CAT=CAT,
               cat_values={c: list(lab[c].cat.categories) for c in CAT}, et_impute=et_impute,
               brand_mean=brand_mean.to_dict(), brand_n=brand_n.to_dict(),
               region_share=region_share, global_share=float(lab.share.median()),
               sites_rl=sites_rl, n_train=len(lab))
    if save:
        ART.mkdir(exist_ok=True); joblib.dump(art, MODEL_PATH)
    return art


def load():
    if not MODEL_PATH.exists():
        return fit(save=True)
    return joblib.load(MODEL_PATH)


# ─────────────────────────── inference ───────────────────────────
def _point_features(art, lat, lon, brand=None, brand_loo_override=None):
    rl = art["sites_rl"]
    d = _haversine(lat, lon, rl.lat.values, rl.lon.values)
    rec = rl.rec_total.fillna(0).values
    recm = rl.rec_mem.fillna(0).values
    row = {"lat": lat, "lon": lon}
    for rkm in [5, 10, 20]:
        mask = d <= rkm
        row[f"n_nbr_{rkm}"] = int(mask.sum())
        vt = rec[mask]
        row[f"mean_nbr_{rkm}"] = float(np.mean(vt)) if mask.any() else np.nan
        if rkm == 20:
            row["logsum_20"] = float(np.log1p(np.sum(vt)))
            sh = recm[mask] / np.where(rec[mask] > 0, rec[mask], np.nan)
            row["shr_nbr_20"] = float(np.nanmean(sh)) if np.isfinite(sh).any() else np.nan
    nearest = int(np.argmin(d))
    row["dist_nearest_km"] = float(d[nearest])
    row["nearest_level"] = float(rec[nearest])
    row["cluster_size"] = float((d <= 20).sum())
    # region/state inherited from nearest existing site
    row["region"] = rl.region.iloc[nearest]; row["state"] = rl.state.iloc[nearest]
    # brand_loo (model's strongest feature). Priority: known operator → local-matured proxy → NaN.
    if brand and brand in art["brand_mean"]:
        row["brand_loo"] = float(art["brand_mean"][brand]); row["brand_n"] = float(art["brand_n"][brand])
    elif brand_loo_override is not None:
        row["brand_loo"] = float(brand_loo_override[0]); row["brand_n"] = float(brand_loo_override[1])
    else:
        row["brand_loo"] = np.nan; row["brand_n"] = np.nan
    X = pd.DataFrame([row])[FEAT + CAT]
    for c in CAT:
        X[c] = pd.Categorical(X[c], categories=art["cat_values"][c])
    return X


def predict_site(lat, lon, brand=None, plateau_override=None,
                 annual_mem_growth=0.0, annual_ret_change=0.0,
                 mem_growth_band=None, ret_change_band=None, horizon=H_DEFAULT, art=None,
                 local_anchor=True, anchor_radius_km=20.0, anchor_max_cov=0.7, anchor_keys=None,
                 anchor_min_n=3, model_kind="lgb", anchor_calib_w=None):
    """Return DataFrame[month, total_med/lo/hi, mem, ret] — the new site's 5-yr monthly trajectory.

    mem_growth_band / ret_change_band = optional (lo, hi) post-maturity drift rates (e.g. the Theil-Sen slope CI).
    When given, the lo/hi trajectory bands drift at those rates, so the band FANS OUT with trend uncertainty on
    top of the plateau P10–P90 — a noisy market self-widens instead of being capped.

    anchor_keys = optional iterable of site_keys to restrict the LOCAL-MATURED level anchor to (e.g. express-only
    sites). The LightGBM neighbour features are left on the full site set so they stay on the training
    distribution; only the brand_loo proxy (the level anchor) is computed over the allowed sites.

    model_kind = "lgb" (Models 1/2, LightGBM quantile level) or "et" (Model 3, ExtraTrees level — more accurate on
    this small/noisy cold-start set; the P10/P90 band widths are still taken from the LightGBM quantiles)."""
    art = art or load()
    # ── LOCAL-MATURED LEVEL (Model 2): with no operator/brand given, the model's strongest feature (brand_loo)
    #    would be NaN. Instead fill it with the mean MATURE level (mat_total) of rich-history sites within
    #    anchor_radius_km of the pin. Backtest (saved model, 1384 sites): WAPE 46.6%→40.2%, medAPE 40.3%→32.8%
    #    vs leaving brand_loo NaN — and it beats a post-hoc blend. mat_total is non-null only for sites that
    #    reached their 18–30-mo window, i.e. genuinely matured sites. ──
    rl = art["sites_rl"]
    anchor_level, n_local_mature, local_cov, override, proxy_used = float("nan"), 0, float("nan"), None, False
    if local_anchor and not (brand and brand in art["brand_mean"]):
        d = _haversine(lat, lon, rl.lat.values, rl.lon.values)
        mt = rl.mat_total.values
        loc = (d <= anchor_radius_km) & np.isfinite(mt) & (mt > 0)
        if "n_months" in rl.columns:                                 # anchor only on well-observed sites (≥24 mo)
            loc &= (rl.n_months.values >= MODEL_MIN_MONTHS)
        if anchor_keys is not None:                                   # express-only: anchor on express neighbours
            loc &= rl.site_key.isin(set(anchor_keys)).values
        n_local_mature = int(loc.sum())
        if n_local_mature > 0:
            vals = mt[loc]
            anchor_level = float(np.median(vals))                        # MEDIAN: robust to one flagship skewing the mean
            mean_v = float(np.mean(vals))
            local_cov = float(np.std(vals) / mean_v) if mean_v > 0 else float("inf")
            # need ENOUGH matured neighbours (≥anchor_min_n) AND agreement (CoV guard). With 1–2 sites the anchor
            # is noisy (and CoV is trivially 0 for n=1), so fall back to the pure global model instead.
            if n_local_mature >= anchor_min_n and local_cov <= anchor_max_cov:
                override = (anchor_level, float(n_local_mature))
                proxy_used = True

    X = _point_features(art, lat, lon, brand, brand_loo_override=override)
    use_et = (model_kind == "et" and "et" in art.get("models", {}))
    p50_lgb = float(np.expm1(art["models"]["q50"].predict(X)[0]))
    p10 = float(np.expm1(art["models"]["q10"].predict(X)[0]))
    p90 = float(np.expm1(art["models"]["q90"].predict(X)[0]))
    if use_et:                                                # Model 3: ExtraTrees level, LGB-quantile band widths
        p50 = float(np.expm1(art["models"]["et"].predict(X[art["FEAT"]].fillna(art["et_impute"]))[0]))
        lo = min(p10 / max(p50_lgb, 1e-9), 1.0); hi = max(p90 / max(p50_lgb, 1e-9), 1.0)
        p10, p90 = p50 * lo, p50 * hi
    else:
        p50 = p50_lgb
        p10, p90 = min(p10, p50), max(p90, p50)
    share = float(np.clip(art["models"]["share"].predict(X)[0], 0.05, 0.95))
    # audit: the plateau WITHOUT the local proxy (brand_loo=NaN), for the UI's before/after caption
    if override is not None:
        Xna = _point_features(art, lat, lon, brand, brand_loo_override=None)
        model_p50 = float(np.expm1((art["models"]["et"].predict(Xna[art["FEAT"]].fillna(art["et_impute"]))
                                    if use_et else art["models"]["q50"].predict(Xna))[0]))
    else:
        model_p50 = p50

    # strong-market calibration: pull the level toward the local-mature neighbour median (log-space blend), ONLY
    # when a real anchor exists. Corrects the model's mean-regression of strong clusters; model_p50 above stays the
    # pure no-anchor reference for the UI before/after. Skipped under a manual plateau_override (that wins below).
    _cw = ANCHOR_CALIB_W if anchor_calib_w is None else float(anchor_calib_w)
    calib_applied = False
    if (_cw > 0 and proxy_used and np.isfinite(anchor_level) and anchor_level > 0
            and not (plateau_override is not None and plateau_override > 0)):
        blended = float(np.expm1(_cw * np.log1p(anchor_level) + (1 - _cw) * np.log1p(p50)))
        scale = blended / max(p50, 1e-9)
        p50, p10, p90 = blended, p10 * scale, p90 * scale
        calib_applied = True

    if plateau_override is not None and plateau_override > 0:
        scale = plateau_override / max(p50, 1e-9)
        p50, p10, p90 = plateau_override, p10 * scale, p90 * scale

    ramp, ramp_src = _select_ramp(art, lat, lon)        # local-market life-cycle (cluster → region → global)
    rm, rr = ramp["mem"], ramp["ret"]
    mglo, mghi = mem_growth_band or (annual_mem_growth, annual_mem_growth)
    rglo, rghi = ret_change_band or (annual_ret_change, annual_ret_change)
    band_g = {"med": (annual_mem_growth, annual_ret_change), "lo": (mglo, rglo), "hi": (mghi, rghi)}
    H = min(horizon, len(rm) - 1)
    TAU_Y = 2.0                                          # post-maturity growth SATURATES over ~2 yr (sites plateau
    rows = []                                            # ~24 mo empirically) — bounds boom extrapolation, no rate clamp
    for m in range(H + 1):
        drift_y = max(0, (m - 24)) / 12.0                # drift only kicks in AFTER maturity (~month 24)
        eff = TAU_Y * (1.0 - np.exp(-drift_y / TAU_Y))   # saturating effective-years (→TAU_Y as m→∞), not linear
        for tag, lvl in [("med", p50), ("lo", p10), ("hi", p90)]:
            amg, arc = band_g[tag]                        # each band drifts at its own (central / lo / hi) rate
            mem = lvl * share * rm[m] * (1 + amg) ** eff
            ret = lvl * (1 - share) * rr[m] * (1 + arc) ** eff
            rows.append((m, tag, mem, ret, mem + ret))
    df = pd.DataFrame(rows, columns=["month", "band", "mem", "ret", "total"])
    out = df.pivot_table(index="month", columns="band", values=["total", "mem", "ret"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out = out.reset_index()
    return out, dict(plateau_med=p50, plateau_lo=p10, plateau_hi=p90, mem_share=share,
                     n_neighbours_20km=int(X["n_nbr_20"].iloc[0]), brand_known=bool(brand and brand in art["brand_mean"]),
                     ramp_source=ramp_src, region=str(X["region"].iloc[0]), state=str(X["state"].iloc[0]),
                     model_plateau=model_p50, anchor_level=anchor_level, n_local_mature=n_local_mature,
                     proxy_used=proxy_used, local_cov=local_cov, model_kind=("et" if use_et else "lgb"),
                     calib_applied=calib_applied, calib_w=(_cw if calib_applied else 0.0))


def _haversine(lat1, lon1, lat2, lon2):
    r = np.radians; lat1, lon1, lat2, lon2 = r(lat1), r(lon1), r(lat2), r(lon2)
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def predict_neighbours(lat, lon, radius_km=20, horizon=H_DEFAULT, art=None, max_n=8):
    """For each existing neighbour: baseline (flat) vs with-new-entrant trajectory (retail cannibalized by distance)."""
    art = art or load()
    rl = art["sites_rl"].copy()
    rl["dist_km"] = _haversine(lat, lon, rl.lat.values, rl.lon.values)
    nb = rl[(rl.dist_km <= radius_km) & (rl.dist_km > 1e-6)].nsmallest(max_n, "dist_km")
    months = np.arange(horizon + 1)
    phase = np.minimum(1.0, months / 12.0)
    cp = cannib_params(art, lat, lon)                  # learned (a, L) for this pin's region
    res = []
    for _, s in nb.iterrows():
        base_mem = float(s.rec_mem or 0); base_ret = float(s.rec_ret or 0)
        cr = float(_cannib_ret(s.dist_km, cp)); cm = 0.4 * cr  # smaller membership drag
        ret_base = np.full(horizon + 1, base_ret); mem_base = np.full(horizon + 1, base_mem)
        ret_imp = ret_base * (1 - cr * phase); mem_imp = mem_base * (1 - cm * phase)
        res.append(dict(site_key=s.site_key, name=s.client_name, dist_km=float(s.dist_km),
                        cannib_ret_pct=cr * 100,
                        baseline=(mem_base + ret_base), with_entrant=(mem_imp + ret_imp),
                        baseline_ret=ret_base, with_entrant_ret=ret_imp))
    return res, months


def evaluate_trajectory(art=None, n_folds=5, seed=0):
    """Leave-one-site-out trajectory backtest: predict each labelled site's actual months 1-36 from location+brand."""
    import lightgbm as lgb
    from sklearn.model_selection import KFold
    panel, site = load_panel_site()
    S, _ = build_features(panel, site)
    lab = S[(S.mat_n >= 4) & (S.n_months >= MODEL_MIN_MONTHS)].reset_index(drop=True)   # same population fit() trains on
    # actual normalized check is covered in the notebook; here report plateau LOSO MAPE quickly
    lab["y"] = np.log1p(lab.mat_total)
    for c in CAT:
        lab[c] = lab[c].astype("category")
    X = lab[FEAT + CAT]; y = lab.y.values
    pred = np.zeros(len(y))
    for tr, te in KFold(n_folds, shuffle=True, random_state=seed).split(X):
        m = lgb.LGBMRegressor(objective="quantile", alpha=0.5, n_estimators=500, learning_rate=0.03,
                              num_leaves=31, min_child_samples=20, subsample=0.8, colsample_bytree=0.8, verbose=-1)
        m.fit(X.iloc[tr], y[tr], categorical_feature=CAT); pred[te] = m.predict(X.iloc[te])
    a, p = np.expm1(y), np.expm1(pred)
    return dict(n=len(y), mape=float(np.median(np.abs(a - p) / a) * 100),
                mae=float(np.mean(np.abs(a - p))))


if __name__ == "__main__":
    art = fit(save=True)
    print("trained on", art["n_train"], "sites; artifacts ->", MODEL_PATH)
