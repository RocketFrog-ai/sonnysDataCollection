"""Leave-one-site-out evaluation of ensemble variants vs the two baselines.

Protocol (identical for every variant; NOTHING is ever fit on the site being scored):
  outer loop over target sites; train = mature sites minus the target.
  Inside each fold:
    - fit the LEVEL model on train (log mature_wash target)
    - NESTED leave-one-out over the train set gives honest train-side level predictions,
      from which we fit: c_mat (median log-residual debias), per-year ramp debiases c_y,
      and convex per-year blend weights w_y toward the proforma
  target per-year forecast = level(target)*exp(c_mat) * ramp_yN * exp(c_y),
  then exp(w_y*log(ens) + (1-w_y)*log(proforma_yN)).
Non-mature sites are scored by the all-mature fold (they are never in any train set).
Scored on the benchmark populations: 227 site-years + 70 mature.

Usage: python ensemble_eval.py [--variant blend|blend_year|ridge|ridge_year|gbm_year|all]
                               [--shuffle SEED]   (negative control: permute targets)
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

SCRATCH = "/Users/dhruvsood/sonnysDataCollection/experiments/old-proforma-analysis/ensemble/results"
F = pd.read_csv(f"{SCRATCH}/ensemble_features.csv")

FEATS = ["lp", "lpf", "pay", "vac", "tos", "css", "ltr", "oy", "lera", "has_era"]
F["lp"] = np.log(F.plateau_loo)
F["lpf"] = np.log(F.pf_y5)
F["ltr"] = np.log(F.traffic_count.clip(lower=1))
F["oy"] = F.open_year - 2022.0
F["lera"] = np.where(F.era_anchor.notna() & (F.era_anchor > 0),
                     np.log(F.era_anchor.clip(lower=1) / F.plateau_loo), 0.0)
F["has_era"] = F.era_anchor.notna().astype(float)

MAT = F[F.mature_ok & F.mature_wash.gt(0)].reset_index(drop=True)
W_GRID = np.round(np.arange(0, 1.0001, 0.05), 2)

# ─────────────────────── level models ───────────────────────
def fit_level(train, kind):
    ytr = np.log(train.mature_wash.values)
    if kind == "blend":
        best_w, best_l = 0.5, np.inf
        for w in W_GRID:
            loss = np.median(np.abs(ytr - (w * train.lp.values + (1 - w) * train.lpf.values)))
            if loss < best_l:
                best_l, best_w = loss, w
        return lambda rows: np.exp(best_w * rows.lp.values + (1 - best_w) * rows.lpf.values)
    X = train[FEATS].values
    mu, sd = X.mean(0), X.std(0) + 1e-9
    if kind == "ridge":
        best_a, best_l = 1.0, np.inf
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        for a in [0.1, 0.3, 1.0, 3.0, 10.0]:
            errs = []
            for tr, te in kf.split(X):
                mdl = Ridge(alpha=a).fit((X[tr] - mu) / sd, ytr[tr])
                errs += list(np.abs(ytr[te] - mdl.predict((X[te] - mu) / sd)))
            if np.median(errs) < best_l:
                best_l, best_a = np.median(errs), a
        mdl = Ridge(alpha=best_a).fit((X - mu) / sd, ytr)
        return lambda rows: np.exp(mdl.predict((rows[FEATS].values.astype(float) - mu) / sd))
    if kind == "gbm":
        mdl = GradientBoostingRegressor(max_depth=2, n_estimators=150, learning_rate=0.05,
                                        subsample=0.8, random_state=0).fit(X, ytr)
        return lambda rows: np.exp(mdl.predict(rows[FEATS].values.astype(float)))
    raise ValueError(kind)

def nested_hats(train, kind):
    """Honest train-side level predictions: LOO within the train set."""
    hats = np.empty(len(train))
    for j in range(len(train)):
        sub = train.drop(index=train.index[j])
        f = fit_level(sub, kind)
        hats[j] = f(train.iloc[[j]])[0]
    return pd.Series(hats, index=train.source_file.values)

def fold_quantities(train, kind, year_blend):
    """Everything fit inside the fold: level fn, c_mat, per-year (c_y, w_y)."""
    f = fit_level(train, kind)
    nh = nested_hats(train, kind)
    c_mat = float(np.median(np.log(train.mature_wash.values) - np.log(nh.loc[train.source_file].values)))
    per_year = {}
    for y in range(1, 6):
        d = train[train.open_observed & train[f"nobs_y{y}"].ge(10) & train[f"act_y{y}"].gt(0)
                  & train[f"pf_y{y}"].gt(0) & train[f"ramp_y{y}"].notna()]
        if len(d) < 8:
            per_year[y] = (0.0, 1.0)
            continue
        ens = nh.loc[d.source_file].values * np.exp(c_mat) * d[f"ramp_y{y}"].values
        la = np.log(d[f"act_y{y}"].values)
        c_y = float(np.median(la - np.log(ens)))
        if not year_blend:
            per_year[y] = (c_y, 1.0)
            continue
        le = np.log(ens) + c_y
        lp_ = np.log(d[f"pf_y{y}"].values)
        best_w, best_l = 1.0, np.inf
        for w in W_GRID:
            loss = np.median(np.abs(la - (w * le + (1 - w) * lp_)))
            if loss < best_l:
                best_l, best_w = loss, w
        per_year[y] = (c_y, best_w)
    return f, c_mat, per_year

def run_variant(kind, year_blend, shuffle_seed=None):
    mat = MAT.copy()
    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        perm = rng.permutation(len(mat))
        for col in ["mature_wash"] + [f"act_y{y}" for y in range(1, 6)] + [f"nobs_y{y}" for y in range(1, 6)]:
            mat[col] = mat[col].values[perm]
    level_hat, preds = {}, {}
    # shared fold for all non-mature sites (train = ALL mature)
    f_all, c_all, py_all = fold_quantities(mat.reset_index(drop=True), kind, year_blend)
    mature_sfs = set(mat.source_file)
    for _, r in F.iterrows():
        if r.source_file in mature_sfs:
            train = mat[mat.source_file != r.source_file].reset_index(drop=True)
            f, c_mat, py = fold_quantities(train, kind, year_blend)
        else:
            f, c_mat, py = f_all, c_all, py_all
        lv = float(f(F[F.source_file == r.source_file])[0]) * np.exp(c_mat)
        level_hat[r.source_file] = lv
        row = {}
        for y in range(1, 6):
            if pd.notna(r[f"ramp_y{y}"]) and r[f"pf_y{y}"] > 0:
                c_y, w = py[y]
                ens = max(lv * r[f"ramp_y{y}"] * np.exp(c_y), 1e-9)
                row[y] = float(np.exp(w * np.log(ens) + (1 - w) * np.log(r[f"pf_y{y}"])))
        preds[r.source_file] = row
    return level_hat, preds

# ─────────────────────── scoring ───────────────────────
def mdape(a, p): return float(np.median(np.abs(np.asarray(a) - np.asarray(p)) / np.asarray(a)) * 100)
def within(a, p, b=0.2): return float((np.abs(np.asarray(a) / np.asarray(p) - 1) <= b).mean() * 100)
def missf(a, p): return float(np.exp(np.median(np.abs(np.log(np.asarray(a) / np.asarray(p))))))

def score(level_hat, preds, label):
    rows = []
    for y in range(1, 6):
        d = F[F.open_observed & F[f"nobs_y{y}"].ge(10) & F[f"act_y{y}"].gt(0)
              & F[f"pf_y{y}"].gt(0) & F[f"cs_y{y}"].gt(0)]
        for _, r in d.iterrows():
            if y in preds.get(r.source_file, {}):
                rows.append({"y": y, "act": r[f"act_y{y}"], "ens": preds[r.source_file][y],
                             "pf": r[f"pf_y{y}"], "cs": r[f"cs_y{y}"]})
    P = pd.DataFrame(rows)
    e_e = np.abs(np.log(P.act / P.ens)); e_p = np.abs(np.log(P.act / P.pf)); e_c = np.abs(np.log(P.act / P.cs))
    res = {"label": label, "n_site_years": len(P),
           "MdAPE": mdape(P.act, P.ens), "within20": within(P.act, P.ens),
           "miss": missf(P.act, P.ens), "ratio_med": float(np.median(P.act / P.ens)),
           "win_vs_pf": float((e_e < e_p).mean() * 100),
           "p_vs_pf": float(stats.binomtest(int((e_e < e_p).sum()), len(P), 0.5).pvalue),
           "win_vs_cs": float((e_e < e_c).mean() * 100),
           "p_vs_cs": float(stats.binomtest(int((e_e < e_c).sum()), len(P), 0.5).pvalue)}
    per_year = {}
    for y in range(1, 6):
        d = P[P.y == y]
        if len(d) >= 5:
            per_year[f"Y{y}"] = {"n": len(d), "MdAPE": mdape(d.act, d.ens), "within20": within(d.act, d.ens),
                                 "rho": float(stats.spearmanr(d.ens, d.act)[0])}
    res["per_year"] = per_year
    dm = F[F.mature_ok & F.mature_wash.gt(0) & F.pf_y5.gt(0) & F.plateau_loo.gt(0)].copy()
    dm["ens"] = dm.source_file.map(level_hat)
    e_e = np.abs(np.log(dm.mature_wash / dm.ens)); e_p = np.abs(np.log(dm.mature_wash / dm.pf_y5))
    e_c = np.abs(np.log(dm.mature_wash / dm.plateau_loo))
    res["mature"] = {"n": len(dm), "MdAPE": mdape(dm.mature_wash, dm.ens),
                     "within20": within(dm.mature_wash, dm.ens),
                     "rho": float(stats.spearmanr(dm.ens, dm.mature_wash)[0]),
                     "ratio_med": float(np.median(dm.mature_wash / dm.ens)),
                     "win_vs_pf": float((e_e < e_p).mean() * 100),
                     "win_vs_cs": float((e_e < e_c).mean() * 100)}
    return res

def baselines():
    out = []
    for tag in ["proforma", "coldstart_loo"]:
        col = "pf" if tag == "proforma" else "cs"
        level_hat = dict(zip(F.source_file, F["pf_y5" if tag == "proforma" else "plateau_loo"]))
        preds = {r.source_file: {y: r[f"{col}_y{y}"] for y in range(1, 6) if pd.notna(r[f"cs_y{y}"])}
                 for _, r in F.iterrows()}
        out.append(score(level_hat, preds, tag))
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="all")
    ap.add_argument("--shuffle", type=int, default=None)
    ap.add_argument("--out", default=f"{SCRATCH}/ensemble_results.json")
    a = ap.parse_args()
    VAR = {"blend": ("blend", False), "blend_year": ("blend", True),
           "ridge": ("ridge", False), "ridge_year": ("ridge", True),
           "gbm_year": ("gbm", True)}
    results = baselines() if a.variant == "all" else []
    todo = list(VAR.items()) if a.variant == "all" else [(a.variant, VAR[a.variant])]
    for name, (kind, yb) in todo:
        lh, pr = run_variant(kind, yb, shuffle_seed=a.shuffle)
        results.append(score(lh, pr, name + ("" if a.shuffle is None else f"_SHUF{a.shuffle}")))
    cols = ["label", "n_site_years", "MdAPE", "within20", "miss", "ratio_med",
            "win_vs_pf", "p_vs_pf", "win_vs_cs", "p_vs_cs"]
    T = pd.DataFrame([{k: r[k] for k in cols} for r in results])
    print(T.round(2).to_string(index=False))
    print("\nmature:")
    M = pd.DataFrame([{**{"label": r["label"]}, **r["mature"]} for r in results])
    print(M.round(2).to_string(index=False))
    print("\nper-year:")
    for r in results:
        py = " | ".join(f"{k} n={v['n']} MdAPE={v['MdAPE']:.0f}% w20={v['within20']:.0f}% rho={v['rho']:.2f}"
                        for k, v in r.get("per_year", {}).items())
        print(f"  {r['label']:16s} {py}")
    with open(a.out, "w") as fh:
        json.dump(results, fh, indent=1)
    print("wrote", a.out)
