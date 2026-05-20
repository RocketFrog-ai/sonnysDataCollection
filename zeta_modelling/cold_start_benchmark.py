"""Cold-start benchmark: compare model_1, model_2, and model_3 on the same
held-out site set using a consistent methodology.

Methodology (same for all three):
- Use the corrected LT data: year_number=2 -> 2025 (not 2026)
- Hold out 15% of LT sites by site_id (same random seed)
- Train on remaining 85% of LT sites (2024 data only for training)
- For each held-out site: look up its DBSCAN cluster (model_1/2) or CBSA+H3
  peer anchor (model_3) and predict washes for 2025
- Compute monthly MAE, WMAPE, and annual HIT% at ±20,000 washes
"""
import sys
import os
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_LT = ROOT / "model_3/data/less_than-2yrs_monthly.csv"
DATA_MT = ROOT / "model_3/data/more_than-2yrs_monthly.csv"
SEED = 7
HOLDOUT_FRAC = 0.15
HIT_BAND = 20_000
EARTH_RADIUS_KM = 6371.0088


# ─── Data loading (shared) ───────────────────────────────────────────────────

def load_lt_fixed() -> pd.DataFrame:
    """Load LT panel with year_number=2 correctly remapped to 2025."""
    df = pd.read_csv(DATA_LT, low_memory=False)
    # Fix mislabelled year: year_number=2 rows have year_month=2026-MM but are actually 2025
    mask2 = df["year_number"] == 2
    cal_month = (df.loc[mask2, "month_number"] - 12).clip(1, 12)
    df.loc[mask2, "year_month"] = "2025-" + cal_month.astype(str).str.zfill(2)
    # Keep only 2024 and 2025
    df = df[df["year_month"].str[:4].isin(["2024", "2025"])].copy()
    df["site_id"] = df["client_id_location_id"].astype(str)
    df["date"] = pd.to_datetime(df["year_month"] + "-01")
    df["wash_count_total"] = pd.to_numeric(df["wash_count_total"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    # op_start: use operational_start_date where available, else first observed month
    df["operational_start_date"] = pd.to_datetime(df["operational_start_date"], errors="coerce")
    first_obs = df.groupby("site_id")["date"].min().rename("first_date")
    df = df.join(first_obs, on="site_id")
    df["operational_start_date"] = df["operational_start_date"].fillna(df["first_date"])
    df = df.dropna(subset=["site_id", "date", "wash_count_total", "latitude", "longitude"])
    df = df[df["wash_count_total"] > 0]
    # Dedupe (site, month)
    df = df.sort_values(["site_id", "date", "wash_count_total"])
    df = df.drop_duplicates(["site_id", "date"], keep="last")
    df["age_in_months"] = (
        (df["date"].dt.year - df["operational_start_date"].dt.year) * 12
        + (df["date"].dt.month - df["operational_start_date"].dt.month)
    ).clip(lower=0)
    df["calendar_month"] = df["date"].dt.month
    return df


def make_holdout_split(df: pd.DataFrame):
    """15% site holdout with fixed seed."""
    sites = sorted(df["site_id"].unique())
    rng = np.random.default_rng(SEED)
    rng.shuffle(sites)
    n_hold = max(1, int(HOLDOUT_FRAC * len(sites)))
    hold_sites = set(sites[:n_hold])
    train_df = df[~df["site_id"].isin(hold_sites)].copy()
    test_df = df[df["site_id"].isin(hold_sites)].copy()
    return train_df, test_df, hold_sites


# ─── Metrics ─────────────────────────────────────────────────────────────────

def wmape(actual, pred):
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    return float(np.sum(np.abs(actual - pred)) / np.maximum(np.sum(np.abs(actual)), 1e-9))


def annual_hit_rate(test_df: pd.DataFrame, pred_col: str = "pred") -> dict:
    """Compute annual HIT% at ±HIT_BAND for full calendar years only."""
    df = test_df.copy()
    df["year"] = df["date"].dt.year
    agg = df.groupby(["site_id", "year"]).agg(
        actual_total=("wash_count_total", "sum"),
        pred_total=(pred_col, "sum"),
        n_months=("wash_count_total", "count"),
    ).reset_index()
    # Only count site-years with all 12 months
    full = agg[agg["n_months"] == 12].copy()
    if len(full) == 0:
        return {"hit_pct": float("nan"), "n_site_years": 0}
    full["hit"] = (np.abs(full["actual_total"] - full["pred_total"]) <= HIT_BAND)
    return {
        "hit_pct": float(full["hit"].mean() * 100),
        "n_site_years": int(len(full)),
    }


# ─── MODEL 1: DBSCAN cluster median (same as model_1/2 peer lookup) ──────────

def run_model1_coldstart(df: pd.DataFrame, eps_km: float = 12.0) -> dict:
    """DBSCAN cluster peer median — model_1 approach."""
    train_df, test_df, hold_sites = make_holdout_split(df)

    # Cluster all sites (train + test) — DBSCAN uses spatial proximity
    sites_all = df[["site_id", "latitude", "longitude"]].drop_duplicates().copy()
    coords_rad = np.radians(sites_all[["latitude", "longitude"]].to_numpy())
    eps_rad = eps_km / EARTH_RADIUS_KM
    labels = DBSCAN(eps=eps_rad, min_samples=3, metric="haversine").fit_predict(coords_rad)
    sites_all["cluster_id"] = labels

    # Cluster monthly reference: median monthly volume per (cluster, calendar_month)
    # computed ONLY from training sites
    train_with_cluster = train_df.merge(sites_all[["site_id", "cluster_id"]], on="site_id", how="left")
    train_2024 = train_with_cluster[train_with_cluster["date"].dt.year == 2024]
    cluster_ref = (
        train_2024.groupby(["cluster_id", "calendar_month"])["wash_count_total"]
        .median()
        .reset_index()
        .rename(columns={"wash_count_total": "cluster_median"})
    )
    national_ref = train_2024.groupby("calendar_month")["wash_count_total"].median().reset_index()
    national_ref.columns = ["calendar_month", "national_median"]

    # Predict for test sites using cluster reference (2025 months)
    test_2025 = test_df[test_df["date"].dt.year == 2025].copy()
    test_2025 = test_2025.merge(sites_all[["site_id", "cluster_id"]], on="site_id", how="left")
    test_2025 = test_2025.merge(cluster_ref, on=["cluster_id", "calendar_month"], how="left")
    test_2025 = test_2025.merge(national_ref, on="calendar_month", how="left")
    test_2025["pred"] = test_2025["cluster_median"].fillna(test_2025["national_median"])

    # Also evaluate on 2024 (cold sites have both years)
    test_all = test_df.copy()
    test_all = test_all.merge(sites_all[["site_id", "cluster_id"]], on="site_id", how="left")
    test_all = test_all.merge(cluster_ref, on=["cluster_id", "calendar_month"], how="left")
    test_all = test_all.merge(national_ref, on="calendar_month", how="left")
    test_all["pred"] = test_all["cluster_median"].fillna(test_all["national_median"])
    test_all = test_all.dropna(subset=["pred"])

    mae = float(mean_absolute_error(test_all["wash_count_total"], test_all["pred"]))
    wm = wmape(test_all["wash_count_total"], test_all["pred"])
    hit = annual_hit_rate(test_all)
    return {"model": f"model_1 (DBSCAN {eps_km}km peer median)", "monthly_mae": round(mae),
            "wmape": round(wm, 3), **hit, "hold_sites": len(hold_sites),
            "test_rows": len(test_all)}


# ─── MODEL 2: DBSCAN + LightGBM multiplier ───────────────────────────────────

def run_model2_coldstart(df: pd.DataFrame, eps_km: float = 12.0) -> dict:
    """DBSCAN cluster + LightGBM multiplier — model_2 approach."""
    try:
        import lightgbm as lgb
    except ImportError:
        return {"model": "model_2 (DBSCAN+LGB)", "error": "lightgbm not installed"}

    train_df, test_df, hold_sites = make_holdout_split(df)

    # Cluster all sites
    sites_all = df[["site_id", "latitude", "longitude"]].drop_duplicates().copy()
    coords_rad = np.radians(sites_all[["latitude", "longitude"]].to_numpy())
    eps_rad = eps_km / EARTH_RADIUS_KM
    labels = DBSCAN(eps=eps_rad, min_samples=3, metric="haversine").fit_predict(coords_rad)
    sites_all["cluster_id"] = labels

    def add_features(frame: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
        f = frame.copy()
        f["month_sin"] = np.sin(2 * np.pi * f["calendar_month"] / 12)
        f["month_cos"] = np.cos(2 * np.pi * f["calendar_month"] / 12)
        f["age_sq"] = f["age_in_months"] ** 2
        f["log_age"] = np.log1p(f["age_in_months"])
        f = f.merge(ref, on=["cluster_id", "calendar_month"], how="left")
        f["cluster_median"] = f["cluster_median"].fillna(f["cluster_median"].median())
        return f

    train_with_c = train_df.merge(sites_all[["site_id", "cluster_id"]], on="site_id", how="left")
    train_2024 = train_with_c[train_with_c["date"].dt.year == 2024]
    cluster_ref = (
        train_2024.groupby(["cluster_id", "calendar_month"])["wash_count_total"]
        .median().reset_index().rename(columns={"wash_count_total": "cluster_median"})
    )
    national_ref_val = train_2024["wash_count_total"].median()
    cluster_ref_merged = cluster_ref.copy()
    cluster_ref_merged["cluster_median"] = cluster_ref_merged["cluster_median"].fillna(national_ref_val)

    train_feat = add_features(train_with_c, cluster_ref_merged)
    feat_cols = ["age_in_months", "age_sq", "log_age", "month_sin", "month_cos",
                 "latitude", "longitude", "cluster_median"]
    feat_cols = [c for c in feat_cols if c in train_feat.columns]
    train_feat = train_feat.dropna(subset=feat_cols + ["wash_count_total"])

    # Train LGB on multiplier = actual / cluster_median
    train_feat["multiplier"] = (train_feat["wash_count_total"] /
                                train_feat["cluster_median"].clip(lower=1)).clip(0.1, 10)
    valid_rows = train_feat["multiplier"].between(0.1, 10)
    X_tr = train_feat.loc[valid_rows, feat_cols]
    y_tr = np.log(train_feat.loc[valid_rows, "multiplier"])

    model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=31,
                               min_child_samples=20, random_state=SEED, verbose=-1)
    model.fit(X_tr, y_tr)

    # Predict for held-out sites
    test_all = test_df.copy()
    test_all = test_all.merge(sites_all[["site_id", "cluster_id"]], on="site_id", how="left")
    test_all = add_features(test_all, cluster_ref_merged)
    test_all["cluster_median"] = test_all["cluster_median"].fillna(national_ref_val)
    test_all = test_all.dropna(subset=feat_cols)
    X_te = test_all[feat_cols]
    test_all["pred_multiplier"] = np.exp(model.predict(X_te))
    test_all["pred"] = (test_all["cluster_median"] * test_all["pred_multiplier"]).clip(lower=1)

    mae = float(mean_absolute_error(test_all["wash_count_total"], test_all["pred"]))
    wm = wmape(test_all["wash_count_total"], test_all["pred"])
    hit = annual_hit_rate(test_all)
    return {"model": f"model_2 (DBSCAN {eps_km}km + LGB multiplier)", "monthly_mae": round(mae),
            "wmape": round(wm, 3), **hit, "hold_sites": len(hold_sites),
            "test_rows": len(test_all)}


# ─── MODEL 3: CBSA + H3 peer anchor ─────────────────────────────────────────

def run_model3_coldstart() -> dict:
    """Read model_3's saved cold-start metrics from its own pipeline output."""
    metrics_path = ROOT / "model_3/outputs/metrics/metrics.json"
    if not metrics_path.exists():
        return {"model": "model_3 (CBSA+H3+LGB)", "error": "metrics.json not found — run pipeline first",
                "monthly_mae": 0, "wmape": 0.0, "hit_pct": float("nan"), "n_site_years": 0}
    with open(metrics_path) as f:
        m = json.load(f)
    cs = m.get("cold_start_split", {})
    model_overall = cs.get("model", {}).get("overall", {})
    hit = cs.get("hit_miss_annual", {})
    return {
        "model": "model_3 (CBSA+H3+LGB)",
        "monthly_mae": int(round(model_overall.get("MAE", float("nan")))),
        "wmape": round(model_overall.get("WMAPE", float("nan")), 3),
        "hit_pct": round(hit.get("overall_hit_rate", float("nan")) * 100, 1),
        "n_site_years": hit.get("n_site_years_fullyear", "?"),
        "hold_sites": cs.get("n_cold_sites", "~141"),
        "test_rows": cs.get("n_test_rows", "?"),
    }


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    print("Loading LT data (year-corrected)...")
    df = load_lt_fixed()
    print(f"  {len(df)} rows, {df['site_id'].nunique()} sites, years: {sorted(df['date'].dt.year.unique())}")
    print(f"  Train/test split: {int((1-HOLDOUT_FRAC)*100)}% / {int(HOLDOUT_FRAC*100)}% sites, seed={SEED}\n")

    results = []

    print("Running model_1 (DBSCAN 12km peer median)...")
    r1 = run_model1_coldstart(df, eps_km=12.0)
    results.append(r1)
    print(f"  MAE={r1['monthly_mae']:,}  WMAPE={r1['wmape']:.3f}  HIT={r1.get('hit_pct', '?')}%  n={r1.get('n_site_years','?')} site-years\n")

    print("Running model_2 (DBSCAN 12km + LightGBM multiplier)...")
    r2 = run_model2_coldstart(df, eps_km=12.0)
    results.append(r2)
    print(f"  MAE={r2['monthly_mae']:,}  WMAPE={r2['wmape']:.3f}  HIT={r2.get('hit_pct', '?')}%  n={r2.get('n_site_years','?')} site-years\n")

    print("Reading model_3 cold-start from saved metrics...")
    r3 = run_model3_coldstart()
    results.append(r3)
    print(f"  MAE={r3['monthly_mae']:,}  WMAPE={r3['wmape']:.3f}  HIT={r3.get('hit_pct', '?')}%  n={r3.get('n_site_years','?')} site-years\n")

    print("=" * 70)
    print(f"{'MODEL':<45} {'MAE':>7} {'WMAPE':>7} {'HIT%':>8} {'N':>6}")
    print("-" * 70)
    for r in results:
        hit = r.get("hit_pct", float("nan"))
        hit_str = f"{hit:.1f}%" if isinstance(hit, float) and not np.isnan(hit) else str(hit)
        print(f"{r['model']:<45} {r['monthly_mae']:>7,} {r['wmape']:>7.3f} {hit_str:>8} {str(r.get('n_site_years','?')):>6}")
    print("=" * 70)
    print(f"\nHIT% = annual washes within ±{HIT_BAND:,} of actual (full 12-month site-years only)")
    print("All models: 15% site holdout (sites never seen during training), seed=7")
    print("LT data: year_number=2 correctly mapped to 2025 (not 2026)")


if __name__ == "__main__":
    main()
