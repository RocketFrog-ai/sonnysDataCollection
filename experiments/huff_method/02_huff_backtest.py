"""
Step 2: Huff-lite model backtest over the sites resolved by 01_resolve_site_keys.py.

Model (single demand-centroid simplification, per ss1.docx's own stated starting point):
  A_j = min-max(cumulative_site_score_j) -> [1,10]           (attractiveness, sites with a proforma)
  A_unknown = median(A_j over the known sites)                (attractiveness, unknown neighbours --
                                                                 real competitors with no proforma on
                                                                 file; see context.md for why this is
                                                                 a placeholder, not measured data)
  own_term_j  = A_j^alpha / d0^beta                            (d0 = 0.1 mi nominal self-distance)
  comp_term_j = sum_k( A_k^alpha / dist_jk^beta )              (real neighbours within 20km, haversine,
                                                                 from the real panel, not the proformas)
  Share_j = own_term_j / (own_term_j + comp_term_j)
  Volume_j_predicted = k * traffic_j * Share_j
(alpha, beta, k) calibrated by nonlinear least squares in log-space against each site's real
actual_mature_wash_mo (from MANIFEST.csv, passed through by step 1). alpha is bounded to [0.5, 2.5]
-- the literature-plausible range ss1.docx itself cites for convenience retail -- because an
unconstrained fit does not converge (see context.md "alpha does not converge" finding).

Baseline for comparison: the old proforma's OWN mature-month prediction (vol_monthly_y5), pulled
from the same extraction, not recomputed.

Run from the repo root: python experiments/huff-model-backtest/02_huff_backtest.py
Output: experiments/huff-model-backtest/backtest_results.csv (per-site predictions + both metrics)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "old-proforma-analysis" / "code"))

from proforma.pnl.data import load_panel, haversine_km  # noqa: E402
import extract_proformas as EP  # noqa: E402  (reused as-is, not reimplemented)

HERE = Path(__file__).resolve().parent
N70_DIR = REPO_ROOT / "experiments" / "old-proforma-analysis" / "n70-final-considered"
RESOLVED_JSON = HERE / "resolved_sites.json"
OUT_CSV = HERE / "backtest_results.csv"

S_MIN, S_MAX = -0.425, 1.5   # theoretical bounds of the 10-factor weighted sum, see
                              # app/pnl_analysis/modelling/bosch_forecast.py SITE_FACTOR_WEIGHTS
RADIUS_KM = 20.0
D0_MI = 0.1
KM_TO_MI = 0.621371
ALPHA_BOUNDS = (0.5, 2.5)   # see context.md: an unconstrained fit runs away, does not converge
BETA_BOUNDS = (1.0, 3.0)


def to_attractiveness(S):
    return 1 + 9 * (S - S_MIN) / (S_MAX - S_MIN)


def load_site_inputs():
    with open(RESOLVED_JSON) as f:
        resolved = json.load(f)

    records = []
    for r in resolved:
        if r["status"] != "unique":
            continue  # ambiguous matches (e.g. Seaford Car Wash, NY) need manual resolution first
        path = N70_DIR / r["source_file"]
        rec = EP.process_file(str(path))
        S, traffic = rec.get("cumulative_site_score"), rec.get("traffic_count")
        if S is None or traffic is None:
            print("SKIP (missing site score or traffic):", r["source_file"])
            continue
        records.append({
            "site_key": r["site_key"], "lat": r["lat"], "lon": r["lon"],
            "grounding": r["grounding"], "S": S, "D": rec.get("cumulative_demographic_score") or 0.0,
            "traffic": traffic, "old_proforma_pred_monthly": rec.get("vol_monthly_y5"),
            "actual_mature_wash_mo": r["actual_mature_wash_mo"],
        })
    return pd.DataFrame(records)


def find_neighbours(df, site_table):
    known_keys = set(df.site_key)
    a_unknown = df["A"].median()
    out = []
    for _, row in df.iterrows():
        d_km = haversine_km(row.lat, row.lon, site_table.lat.values, site_table.lon.values)
        mask = (d_km <= RADIUS_KM) & (d_km > 0.01) & (site_table.site_key.values != row.site_key)
        nb_keys, nb_dist = site_table.site_key.values[mask], d_km[mask]
        nb_A = np.array([df.loc[df.site_key == k, "A"].iloc[0] if k in known_keys else a_unknown
                          for k in nb_keys])
        out.append({"dist_km": nb_dist, "A": nb_A, "n": len(nb_keys)})
    return out, a_unknown


def predict(params, df, neighbours):
    alpha, beta, log_k = params
    k = np.exp(log_k)
    preds = []
    for i, row in df.iterrows():
        nb = neighbours[i]
        own = row.A ** alpha / (D0_MI ** beta)
        if nb["n"] > 0:
            dist_mi = nb["dist_km"] * KM_TO_MI
            comp = np.sum((nb["A"] ** alpha) / (np.maximum(dist_mi, 0.05) ** beta))
        else:
            comp = 0.0
        preds.append(k * row.traffic * (own / (own + comp)))
    return np.array(preds)


def residuals(params, df, neighbours, actual):
    pred = np.maximum(predict(params, df, neighbours), 1e-6)
    return np.log(pred) - np.log(actual)


def metrics(pred, actual, label):
    pred, actual = np.asarray(pred, dtype=float), np.asarray(actual, dtype=float)
    valid = np.isfinite(pred) & np.isfinite(actual) & (actual > 0)
    pred, actual = pred[valid], actual[valid]
    ape = np.abs(pred - actual) / actual * 100
    r2 = 1 - np.sum((pred - actual) ** 2) / np.sum((actual - actual.mean()) ** 2)
    print(f"{label:22s} n={valid.sum():3d}  MAPE={ape.mean():7.1f}%  MdAPE={np.median(ape):6.1f}%  "
          f"R^2={r2:+.3f}  median(pred/actual)={np.median(pred/actual):.2f}")


def main():
    df = load_site_inputs()
    print(f"Usable sites (site score + traffic present): {len(df)}")
    df["A"] = to_attractiveness(df["S"])
    print(f"Known-site A range: [{df.A.min():.2f}, {df.A.max():.2f}], median {df.A.median():.2f}")

    _, site_table = load_panel()
    site_table = site_table[site_table.has_coords].reset_index(drop=True)
    neighbours, a_unknown = find_neighbours(df, site_table)
    df["n_neighbours_20km"] = [n["n"] for n in neighbours]
    print(f"Unknown-neighbour A fallback (median of known sites): {a_unknown:.2f}")

    actual = df["actual_mature_wash_mo"].values.astype(float)
    x0 = [1.0, 2.0, np.log(0.05)]
    result = least_squares(residuals, x0, args=(df, neighbours, actual),
                            bounds=([ALPHA_BOUNDS[0], BETA_BOUNDS[0], -10],
                                    [ALPHA_BOUNDS[1], BETA_BOUNDS[1], 10]))
    alpha_fit, beta_fit, k_fit = result.x[0], result.x[1], np.exp(result.x[2])
    print(f"\nCalibrated: alpha={alpha_fit:.3f}  beta={beta_fit:.3f}  k={k_fit:.5f}")
    if abs(alpha_fit - ALPHA_BOUNDS[1]) < 1e-3:
        print("NOTE: alpha pinned at its upper bound -- see context.md, this does not converge freely.")

    df["huff_pred_monthly"] = predict(result.x, df, neighbours)

    print("\n=== Backtest accuracy (mature monthly wash volume) ===")
    metrics(df["huff_pred_monthly"], df["actual_mature_wash_mo"], "Huff-lite")
    metrics(df["old_proforma_pred_monthly"], df["actual_mature_wash_mo"], "Old proforma (y5)")

    print("\n=== By grounding tier ===")
    for tier in ["grounded", "thin", "isolated"]:
        sub = df[df.grounding == tier]
        if sub.empty:
            continue
        print(f"-- {tier} (n={len(sub)}) --")
        metrics(sub["huff_pred_monthly"], sub["actual_mature_wash_mo"], "  Huff-lite")
        metrics(sub["old_proforma_pred_monthly"], sub["actual_mature_wash_mo"], "  Old proforma")

    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote per-site results to {OUT_CSV}")


if __name__ == "__main__":
    main()
