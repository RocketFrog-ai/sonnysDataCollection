"""Leave-One-Out spatial cross-validation of the neighbour model.

For every cohort site (>=12 months) we hide it, predict its 5 KPI trajectories
from its <=20 km neighbours, and compare to its actual history. This answers the
user's question -- "how well synced do the neighbours and the new site move?" --
with MAE (real units), sMAPE, and Pearson r per KPI, benchmarked against a
naive global-mean baseline so the geographic uplift is explicit.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, pearsonr

warnings.filterwarnings("ignore", category=ConstantInputWarning)

from . import config as C
from .data_loader import load_all
from .predict import predict_site


def naive_baseline(panels: dict, month_axis, kpis=C.TARGET_KPIS) -> dict:
    """Per-KPI global mean trajectory (every site's average per month)."""
    return {kpi: panels[kpi].mean(axis=0).reindex(month_axis) for kpi in kpis}


def _smape(a: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(2 * np.abs(p - a) / (np.abs(p) + np.abs(a) + 1e-9)))


def loocv(ds, agg: str = "idw", kpis=C.TARGET_KPIS) -> pd.DataFrame:
    """Per (site, KPI) scores over the cohort. One row per scored cell."""
    base = naive_baseline(ds.panels, ds.month_axis, kpis)
    sites_idx = ds.sites.set_index("site_uid")
    rows = []
    n_no_nbr = 0

    for uid in sorted(ds.cohort):
        srow = sites_idx.loc[uid]
        res = predict_site(
            srow.lat, srow.lon,
            panels=ds.panels, sites=ds.sites, month_axis=ds.month_axis,
            kpis=kpis, agg=agg, exclude_uid=uid,
        )
        if res.meta["n_neighbours"] == 0:
            n_no_nbr += 1
            continue

        for kpi in kpis:
            actual = ds.panels[kpi].loc[uid]
            pred = res.predictions[kpi]
            m = actual.notna() & pred.notna()
            if m.sum() < C.MIN_OVERLAP_MONTHS:
                continue
            a = actual[m].to_numpy(float)
            p = pred[m].to_numpy(float)
            b = base[kpi][m].to_numpy(float)

            mae = float(np.mean(np.abs(p - a)))
            mae_base = float(np.mean(np.abs(b - a)))
            r = (float(pearsonr(a, p)[0]) if a.std() > 0 and p.std() > 0 else np.nan)
            rows.append(
                {
                    "site_uid": uid, "kpi": kpi, "mode": res.meta["mode"],
                    "n_overlap": int(m.sum()), "mae": mae, "smape": _smape(a, p),
                    "pearson_r": r, "mae_baseline": mae_base,
                }
            )

    df = pd.DataFrame(rows)
    df.attrs["n_no_neighbours"] = n_no_nbr
    return df


def summarise(loocv_df: pd.DataFrame) -> pd.DataFrame:
    """Median per-KPI scores + baseline + geographic uplift."""
    out = []
    for kpi, g in loocv_df.groupby("kpi"):
        med_mae = g["mae"].median()
        med_base = g["mae_baseline"].median()
        out.append(
            {
                "kpi": kpi,
                "label": C.KPI_LABELS.get(kpi, kpi),
                "n_sites": int(g["site_uid"].nunique()),
                "median_mae": round(med_mae, 2),
                "median_smape": round(g["smape"].median(), 3),
                "median_pearson_r": round(g["pearson_r"].median(), 3),
                "median_mae_baseline": round(med_base, 2),
                "uplift_pct": round(100 * (med_base - med_mae) / med_base, 1) if med_base else np.nan,
                "pct_radius": round(100 * (g["mode"] == "radius").mean(), 1),
            }
        )
    # order by the validated strength of the geographic signal (Pearson r)
    return pd.DataFrame(out).sort_values("median_pearson_r", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    ds = load_all()
    print(f"Running LOOCV over {len(ds.cohort)} cohort sites (IDW)...")
    raw = loocv(ds, agg="idw")
    summary = summarise(raw)

    C.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(C.EVAL_PATH, index=False)

    pd.set_option("display.width", 200)
    print(f"\nsites with no neighbours at all: {raw.attrs['n_no_neighbours']}")
    print(f"\nLOOCV summary (sorted by sync r):\n{summary.to_string(index=False)}")
    print(f"\nWrote -> {C.EVAL_PATH}")
