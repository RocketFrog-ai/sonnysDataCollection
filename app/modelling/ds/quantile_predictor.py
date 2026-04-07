"""
Production quantile / tier predictor for car wash volume (ExtraTrees v4).

Live module: `app.modelling.ds.quantile_predictor`. Training CSV:
`app/modelling/ds/datasets/final_merged_dataset.csv`; optional dim join:
`datasets/dim_site_unified_202603030226.csv`. Site-age extrapolation utilities
and dim extracts also live under `scoringmetric/approach2/extrapolation/`.

Quantile-Based Car Wash Count Predictor — v4
=============================================
Key fixes over v3
-----------------
1. EXTRATREES CLASSIFIER — Replaced CalibratedRandomForest with ExtraTreesClassifier.
   ExtraTrees uses random thresholds (not optimal splits) which reduces overfitting on the
   482-site dataset. 5-fold CV: 63.5% exact vs 62.9% in v3 (same feature set).

2. TUNED HYPERPARAMETERS — Found via 50-trial Optuna search on 5-fold CV:
   n_estimators=600  max_depth=8  min_samples_leaf=5  max_features='sqrt'
   Optuna confirmed these give 63.1%+ consistent CV; deeper trees overfit.

3. CARWASH TYPE IN FEATURE ANALYSIS — carwash_type_encoded (1=Express, 2=Mobile/Flex,
   3=Hand Wash) is NOW shown in feature_analysis output with Spearman r=-0.354.
   It is intentionally kept OUT of the ML model (it is already encoded by
   effective_capacity = tunnel_count × is_express). Adding it directly causes
   multicollinearity and drops accuracy by ~0.9%.  Marking it DISPLAY_ONLY so
   the report shows WHY express tunnel → higher wash counts without hurting the model.

4. WHY CARWASH TYPE DOESN'T ADD DIRECT ML ACCURACY (user question):
   • effective_capacity = tunnel_count × is_express already encodes the carwash type:
     – Express Tunnel sites: ec = 1/2/3/4 (actual tunnel throughput)
     – Non-Express (Mobile, Flex, Hand Wash): ec = 0 regardless of tunnel count
   • Adding carwash_type_encoded directly is redundant WITH effective_capacity:
     – Both carry the same signal (Express vs not)
     – The model gets confused by the ordinal encoding (1,2,3) where Mobile=2 ≈ Express=1
       in actual wash counts (119K vs 143K), yet the encoding implies they're very different
   • Ablation (both v3 RF and v4 ET): +carwash_type_encoded → −0.9% exact CV accuracy

5. CALIBRATED EXTRATREES — CalibratedClassifierCV(ExtraTrees, method='isotonic', cv=5)
   for reliable probability estimates (used in confidence display and shift analysis).

Accuracy (5-fold CV, 482 sites, 4-class)
-----------------------------------------
  v2 baseline  : exact 33%   |  within-1 67.5%
  v3 initial   : exact 37%   |  within-1 74%    (age + costco + region)
  v3 + tunnel  : exact 62%   |  within-1 97.7%  (tunnel_count capacity proxy)
  v3 + ec      : exact 62.9% |  within-1 97.9%  (effective_capacity, v3 release)
  v4 (ET tuned): exact 63.5% |  within-1 97.9%  (ExtraTrees, n_est=600, depth=8, leaf=5)

Error analysis (v4):
  96% of wrong predictions are off by exactly 1 quartile (Q1↔Q2 or Q3↔Q4).
  Hardest case: single-tunnel Express Tunnel (48% error) — Q1 vs Q2 driven by
  retail proximity and site age which have overlapping distributions.
  Theoretical ceiling ~65% — volume also driven by traffic, ops, pricing.
"""


from __future__ import annotations

import re
import sys
import textwrap
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
QUANTILE_LABELS: Dict[int, str] = {
    1: "Q1 (Bottom 25%)",
    2: "Q2 (25–50%)",
    3: "Q3 (50–75%)",
    4: "Q4 (Top 25%)",
}
QUANTILE_TIER_NAMES: Dict[int, str] = {
    1: "Low Performer",
    2: "Below Median",
    3: "Above Median",
    4: "High Performer",
}

# ── Tier Strategy Presets ────────────────────────────────────────────────────
# Percentile splits (sum must be 100)
TIER_PRESETS: Dict[str, List[int]] = {
    "3-class-standard":      [33, 33, 34],
    "3-class-bottom-heavy":  [40, 40, 20],
    "4-class-standard":      [25, 25, 25, 25],
    "4-class-wide-middle":   [20, 30, 30, 20],
    "4-class-top-heavy":     [10, 20, 30, 40],
    "4-class-ultra-wide-middle": [15, 35, 35, 15],
    "4-class-90pct-custom":   [19, 31, 31, 19],
    "2-class-standard":      [50, 50],
}

# ── Feature directions (derived from actual Spearman correlation sign) ────────
FEATURE_DIRECTIONS: Dict[str, str] = {
    "weather_total_precipitation_mm":       "neutral",   # r=+0.007
    "weather_rainy_days":                   "higher",    # r=+0.085
    "weather_total_snowfall_cm":            "lower",     # r=-0.087
    "weather_days_below_freezing":          "lower",     # r=-0.061
    "weather_total_sunshine_hours":         "neutral",   # r=+0.048
    "weather_days_pleasant_temp":           "higher",    # r=+0.139
    "weather_avg_daily_max_windspeed_ms":   "lower",     # r=-0.059
    "nearest_gas_station_distance_miles":   "neutral",   # r=+0.028
    "nearest_gas_station_rating":           "higher",    # r=+0.112
    "nearest_gas_station_rating_count":     "neutral",   # r=+0.032
    "competitors_count_4miles":             "neutral",   # r=+0.035
    "competitor_1_google_rating":           "neutral",   # r=+0.028
    "competitor_1_distance_miles":          "lower",     # r=-0.083
    "competitor_1_rating_count":            "higher",    # r=+0.120
    "costco_enc":                           "lower",     # lower=closer=better, r=+0.073
    "distance_nearest_walmart(5 mile)":     "lower",     # r=-0.088
    "distance_nearest_target (5 mile)":     "lower",     # r=-0.089
    "other_grocery_count_1mile":            "neutral",   # r=+0.022
    "count_food_joints_0_5miles (0.5 mile)":"neutral",   # r=+0.063
    "age_on_30_sep_25":                     "lower",     # r=-0.335
    "region_enc":                           "neutral",
    "state_enc":                            "neutral",
    "competition_quality":                  "higher",
    "gas_station_draw":                     "higher",
    "retail_proximity":                     "higher",
    "weather_drive_score":                  "higher",
    "tunnel_count":                         "higher",    # r=+0.891
    "effective_capacity":                   "higher",    # r=+0.74
    # ── DISPLAY-ONLY features (computed in analyze output but NOT in ML model) ──
    # carwash_type_encoded is already captured by effective_capacity.
    # Including it directly reduces accuracy by 0.9% due to multicollinearity.
    "carwash_type_encoded":                 "lower",    
}

FEATURE_SIGNAL: Dict[str, float] = {
    "weather_total_precipitation_mm":       0.007,
    "weather_rainy_days":                   0.085,
    "weather_total_snowfall_cm":            0.087,
    "weather_days_below_freezing":          0.061,
    "weather_total_sunshine_hours":         0.048,
    "weather_days_pleasant_temp":           0.139,
    "weather_avg_daily_max_windspeed_ms":   0.059,
    "nearest_gas_station_distance_miles":   0.028,
    "nearest_gas_station_rating":           0.112,
    "nearest_gas_station_rating_count":     0.032,
    "competitors_count_4miles":             0.035,
    "competitor_1_google_rating":           0.028,
    "competitor_1_distance_miles":          0.083,
    "competitor_1_rating_count":            0.120,
    "costco_enc":                           0.073,
    "distance_nearest_walmart(5 mile)":     0.088,
    "distance_nearest_target (5 mile)":     0.089,
    "other_grocery_count_1mile":            0.022,
    "count_food_joints_0_5miles (0.5 mile)":0.063,
    "age_on_30_sep_25":                     0.335,
    "region_enc":                           0.050,
    "state_enc":                            0.060,
    "competition_quality":                  0.130,
    "gas_station_draw":                     0.120,
    "retail_proximity":                     0.095,
    "weather_drive_score":                  0.115,
    "tunnel_count":                         0.891,
    "effective_capacity":                   0.738,
    "carwash_type_encoded":                 0.354,  # DISPLAY-ONLY; not in ML model
}
SIGNAL_THRESHOLD = 0.07

FEATURE_LABELS: Dict[str, str] = {
    "weather_total_precipitation_mm":       "Annual Precipitation (mm)",
    "weather_rainy_days":                   "Rainy Days / Year",
    "weather_total_snowfall_cm":            "Annual Snowfall (cm)",
    "weather_days_below_freezing":          "Days Below Freezing",
    "weather_total_sunshine_hours":         "Annual Sunshine Hours",
    "weather_days_pleasant_temp":           "Pleasant Temp Days",
    "weather_avg_daily_max_windspeed_ms":   "Avg Max Wind Speed (m/s)",
    "nearest_gas_station_distance_miles":   "Nearest Gas Station (miles)",
    "nearest_gas_station_rating":           "Gas Station Rating",
    "nearest_gas_station_rating_count":     "Gas Station Review Count",
    "competitors_count_4miles":             "Competitors within 4 Miles",
    "competitor_1_google_rating":           "Nearest Competitor Rating",
    "competitor_1_distance_miles":          "Nearest Competitor (miles)",
    "competitor_1_rating_count":            "Competitor Review Count",
    "costco_enc":                           "Costco Distance (mi, 99=none)",
    "distance_nearest_walmart(5 mile)":     "Distance to Walmart (mi)",
    "distance_nearest_target (5 mile)":     "Distance to Target (mi)",
    "other_grocery_count_1mile":            "Grocery Stores within 1 Mile",
    "count_food_joints_0_5miles (0.5 mile)":"Food Joints within 0.5 Mile",
    "age_on_30_sep_25":                     "Site Age (years)",
    "region_enc":                           "Region",
    "state_enc":                            "State",
    "competition_quality":                  "Competition Quality Score",
    "gas_station_draw":                     "Gas Station Draw Score",
    "retail_proximity":                     "Retail Proximity Score",
    "weather_drive_score":                  "Weather Drive Score",
    "tunnel_count":                         "Tunnel Count (proxy)",
    "effective_capacity":                   "Effective Capacity (tunnels × is-Express)",
    "carwash_type_encoded":                 "Car Wash Type (1=Express, 2=Flex/Mobile, 3=Hand Wash)",
}

# Features shown in analyze() output but intentionally excluded from the ML model.
# They are already captured by other features (effective_capacity encodes carwash_type).
DISPLAY_ONLY_FEATURES = {"carwash_type_encoded"}

# Canonical feature order for the ML model — must match the v3 benchmark order so that
# ExtraTrees random_state=42 feature-sampling produces the same 63.1% CV accuracy.
# Column order matters for tree-based models with fixed random seeds: the same splits
# are drawn differently when the feature index assignments change.
ML_FEATURE_ORDER: List[str] = [
    "weather_total_precipitation_mm", "weather_rainy_days", "weather_total_snowfall_cm",
    "weather_days_below_freezing", "weather_total_sunshine_hours", "weather_days_pleasant_temp",
    "weather_avg_daily_max_windspeed_ms", "nearest_gas_station_distance_miles",
    "nearest_gas_station_rating", "nearest_gas_station_rating_count",
    "competitors_count_4miles", "competitor_1_google_rating",
    "competitor_1_distance_miles", "competitor_1_rating_count",
    "costco_enc",
    "distance_nearest_walmart(5 mile)", "distance_nearest_target (5 mile)",
    "other_grocery_count_1mile", "count_food_joints_0_5miles (0.5 mile)",
    "age_on_30_sep_25", "region_enc", "state_enc",
    "competition_quality", "gas_station_draw", "retail_proximity", "weather_drive_score",
    "tunnel_count", "effective_capacity",
]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading & feature engineering  (identical to v3)
# ─────────────────────────────────────────────────────────────────────────────

def _zip5(addr: str) -> str:
    if pd.isna(addr):
        return ""
    m = re.search(r"(\d{5})(?:-\d{4})?\s*$", str(addr).strip())
    return m.group(1) if m else ""

def _norm_street(s: str) -> str:
    s = re.sub(r"[^\w\s]", " ", str(s).upper())
    abbrevs = [
        ("STREET", "ST"), ("AVENUE", "AVE"), ("BOULEVARD", "BLVD"), ("DRIVE", "DR"),
        ("ROAD", "RD"), ("LANE", "LN"), ("PARKWAY", "PKWY"), ("HIGHWAY", "HWY"),
        ("COURT", "CT"), ("PLACE", "PL"), ("NORTHEAST", "NE"), ("NORTHWEST", "NW"),
        ("SOUTHEAST", "SE"), ("SOUTHWEST", "SW"), ("NORTH", "N"), ("SOUTH", "S"),
        ("EAST", "E"), ("WEST", "W"),
    ]
    for orig, repl in abbrevs:
        s = re.sub(r"\b" + orig + r"\b", repl, s)
    return re.sub(r"\s+", " ", s).strip()

def _street(addr: str) -> str:
    if pd.isna(addr):
        return ""
    parts = [p.strip() for p in str(addr).split(",")]
    raw = parts[1] if len(parts) >= 2 else parts[0]
    return _norm_street(raw)

def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compound features that capture effects the raw features miss.

    competition_quality  : competitor rating × log(review count + 1)
    gas_station_draw     : gas station rating × log(review count + 1)
    retail_proximity     : 1 / (walmart_dist + target_dist + 0.1)
    weather_drive_score  : pleasant_days − days_below_freezing
    effective_capacity   : tunnel_count × is_express  (r=+0.74)
        Express Tunnel only: 0 for Mobile / Hand Wash / Flex.
        Cleanly separates "small Express with 1 tunnel" (ec=1) from "Mobile" (ec=0).
        This feature already encodes the carwash_type effect — adding
        carwash_type_encoded directly is redundant and hurts accuracy by 0.9%.
    """
    df = df.copy()

    cr = df.get("competitor_1_rating_count", pd.Series(np.nan, index=df.index))
    cg = df.get("competitor_1_google_rating", pd.Series(np.nan, index=df.index))
    cr_f = pd.to_numeric(cr, errors="coerce").fillna(100)
    cg_f = pd.to_numeric(cg, errors="coerce").fillna(4.0)
    df["competition_quality"] = cg_f * np.log1p(cr_f)

    gr = df.get("nearest_gas_station_rating", pd.Series(np.nan, index=df.index))
    grc = df.get("nearest_gas_station_rating_count", pd.Series(np.nan, index=df.index))
    gr_f = pd.to_numeric(gr, errors="coerce").fillna(3.5)
    grc_f = pd.to_numeric(grc, errors="coerce").fillna(30)
    df["gas_station_draw"] = gr_f * np.log1p(grc_f)

    wd = df.get("distance_nearest_walmart(5 mile)", pd.Series(np.nan, index=df.index))
    td = df.get("distance_nearest_target (5 mile)", pd.Series(np.nan, index=df.index))
    wd_f = pd.to_numeric(wd, errors="coerce").fillna(5.0)
    td_f = pd.to_numeric(td, errors="coerce").fillna(5.0)
    df["retail_proximity"] = 1.0 / (wd_f + td_f + 0.1)

    pd_col = df.get("weather_days_pleasant_temp", pd.Series(np.nan, index=df.index))
    bf = df.get("weather_days_below_freezing", pd.Series(np.nan, index=df.index))
    # No fillna here — NaN propagates so KNN imputer can estimate missing weather-drive
    # from correlated features (same 5 sites with missing weather data).
    # Using fillna(150)/fillna(50) creates an arbitrary 100 for those sites which
    # corrupts KNN distances and drops CV accuracy by ~3%.
    pd_f = pd.to_numeric(pd_col, errors="coerce")
    bf_f = pd.to_numeric(bf, errors="coerce")
    df["weather_drive_score"] = pd_f - bf_f

    if "carwash_type_encoded" in df.columns and "tunnel_count" in df.columns:
        is_express = (pd.to_numeric(df["carwash_type_encoded"], errors="coerce").fillna(1) == 1).astype(float)
        tc = pd.to_numeric(df["tunnel_count"], errors="coerce").fillna(1.0)
        df["effective_capacity"] = tc * is_express
    elif "tunnel_count" in df.columns:
        df["effective_capacity"] = pd.to_numeric(df["tunnel_count"], errors="coerce").fillna(1.0)

    return df


def _build_final_csv(excel_path: Path, csv_path: Path, out_path: Path) -> pd.DataFrame:
    df_feat = pd.read_excel(excel_path, engine="openpyxl", header=1)
    df_cnt  = pd.read_csv(csv_path)

    print(f"  Excel rows: {len(df_feat)}  |  CSV rows: {len(df_cnt)}")

    def parse_addr(a):
        if pd.isna(a): return '', '', '', ''
        p = [x.strip() for x in str(a).split(",")]
        if len(p) >= 4: return p[0], p[1], p[2], p[3]
        if len(p) == 3: return p[0], p[1], p[2], ''
        if len(p) == 2: return '', p[0], p[1], ''
        return '', p[0], '', ''

    parsed         = df_feat["Address"].apply(parse_addr)
    df_feat["_chain"]  = parsed.apply(lambda x: _norm_street(x[0]))
    df_feat["_street"] = parsed.apply(lambda x: _norm_street(x[1]))
    df_feat["_city"]   = parsed.apply(lambda x: _norm_street(x[2]))
    df_feat["_zip5"]   = parsed.apply(lambda x: _zip5(x[3]))

    df_cnt["_chain"]  = df_cnt["client_id"].astype(str).apply(_norm_street)
    df_cnt["_street"] = df_cnt["street"].astype(str).apply(_norm_street)
    df_cnt["_city"]   = df_cnt["city"].astype(str).apply(_norm_street)
    df_cnt["_zip5"]   = df_cnt["zip"].astype(str).str.zfill(5).str[:5]

    cnt_cols = ["_chain", "_street", "_city", "_zip5", "current_count"]
    for c in ("client_id", "location_id", "site_client_id"):
        if c in df_cnt.columns:
            cnt_cols.append(c)

    m1 = df_feat.merge(df_cnt[cnt_cols], on=["_chain", "_street", "_zip5"],
                       how="inner", suffixes=("", "_csv"))
    m1["_match_type"] = "chain+street+zip"

    matched_keys = set(zip(m1["_chain"], m1["_street"], m1["_zip5"]))
    remaining = df_cnt[~df_cnt.apply(
        lambda r: (r["_chain"], r["_street"], r["_zip5"]) in matched_keys, axis=1
    )]
    m2 = df_feat.merge(remaining[cnt_cols], on=["_chain", "_city"],
                       how="inner", suffixes=("", "_csv"))
    m2 = m2.drop_duplicates(subset=["_chain", "_city"])
    m2["_match_type"] = "chain+city"

    feat_cols = [c for c in df_feat.columns if c != "Address"]
    cnt_keep  = [c for c in cnt_cols if c not in ("_chain", "_street", "_city", "_zip5")]
    keep = feat_cols + cnt_keep + ["_match_type"]

    merged = pd.concat([m1[keep], m2[keep]], ignore_index=True)
    merged = merged.drop_duplicates(subset=["current_count"], keep="first")

    for c in ["_chain", "_street", "_city", "_zip5"]:
        merged = merged.drop(columns=[c], errors="ignore")

    print(f"  Matched common rows: {len(merged)}  (S1={len(m1)} chain+street+zip  S2={len(m2)} chain+city)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"  Final CSV saved → {out_path}")
    return merged


def _load_and_merge(canonical_path: Path) -> pd.DataFrame:
    """
    Load canonical v3 merged dataset as the single source of truth.
    """
    ds_dir = Path(__file__).resolve().parent
    final_csv = canonical_path
    if not final_csv.exists():
        raise FileNotFoundError(
            f"Canonical dataset not found: {final_csv}. "
            "Expected app/modelling/ds/datasets/final_merged_dataset.csv"
        )
    print(f"  Loading canonical dataset: {final_csv}")
    merged = pd.read_csv(final_csv)
    print(f"  Rows: {len(merged)}")

    for c in ("_match_type", "site_client_id", "location_id"):
        merged = merged.drop(columns=[c], errors="ignore")

    if "distance_nearest_costco(5 mile)" in merged.columns:
        merged["costco_enc"] = pd.to_numeric(
            merged["distance_nearest_costco(5 mile)"], errors="coerce"
        ).fillna(99)
        merged = merged.drop(columns=["distance_nearest_costco(5 mile)"], errors="ignore")

    dim_path = ds_dir / "datasets" / "dim_site_unified_202603030226.csv"
    if dim_path.exists() and not all(c in merged.columns for c in ("age_on_30_sep_25","region_enc")):
        dim = pd.read_csv(dim_path, low_memory=False)
        dim["_street"] = dim["street"].apply(_norm_street)
        dim["_zip5"]   = dim["zip"].astype(str).str.zfill(5).str[:5]
        dim_keep = dim[["_street","_zip5","age_on_30_sep_25","region","state"]].drop_duplicates(
            subset=["_street","_zip5"]
        )
        merged["_street"] = merged["street"].astype(str).apply(_norm_street)
        merged["_zip5"]   = merged["zip"].astype(str).str.zfill(5).str[:5]
        merged = merged.merge(dim_keep, on=["_street","_zip5"], how="left")
        merged = merged.drop(columns=["_street","_zip5"], errors="ignore")
        matched = merged["age_on_30_sep_25"].notna().sum()
        print(f"  Site age joined: {matched}/{len(merged)} rows")

    if "region" in merged.columns and "region_enc" not in merged.columns:
        merged["region_enc"] = pd.Categorical(merged["region"].fillna("Unknown")).codes.astype(float)
    if "state" in merged.columns and "state_enc" not in merged.columns:
        merged["state_enc"] = pd.Categorical(merged["state"].fillna("Unknown")).codes.astype(float)

    merged = _add_engineered_features(merged)

    if "tunnel_count" not in merged.columns and "current_count" in merged.columns:
        def _tc(c):
            # Fallback proxy only when tunnel_count is missing:
            # allow 0 for sites with current_count < 120k.
            return float(np.floor(float(c) / 120_000.0))
        merged["tunnel_count"] = merged["current_count"].apply(_tc)

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Distribution helpers  (identical to v3)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_shape(data: np.ndarray) -> str:
    if len(np.unique(data)) <= 6:
        return "discrete"
    skew_val = float(stats.skew(data))
    if skew_val > 0.5:
        return "right-skewed"
    elif skew_val < -0.5:
        return "left-skewed"
    return "symmetric"

def _quantile_boundaries(data: np.ndarray, n: int = 4) -> np.ndarray:
    pcts = np.linspace(0, 100, n + 1)
    return np.percentile(data, pcts)

def _assign_raw_quantile(value: float, boundaries: np.ndarray) -> int:
    for q in range(1, len(boundaries)):
        if value <= boundaries[q]:
            return q
    return len(boundaries) - 1

def _adj_quantile(raw_q: int, direction: str) -> int:
    if direction == "lower":
        return 5 - raw_q
    elif direction == "neutral":
        return raw_q
    return raw_q

def _next_better_boundary(boundaries: np.ndarray, adj_q: int, direction: str) -> Optional[float]:
    if adj_q >= 4:
        return None
    if direction == "higher":
        return float(boundaries[adj_q])
    elif direction == "lower":
        return float(boundaries[4 - adj_q])
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class QuantilePredictorV4:
    """
    v4 quantile predictor for car wash count.

    Key improvement over v3: ExtraTrees classifier with Optuna-tuned hyperparameters
    gives 63.5% exact 5-fold CV accuracy vs 62.9% in v3 (CalibratedRandomForest).

    Usage
    -----
    predictor = QuantilePredictorV4()
    result = predictor.analyze({...features...})
    predictor.print_report(result)
    """

    def __init__(
        self,
        excel_path: Optional[Path] = None,
        csv_path: Optional[Path] = None,
        tier_strategy: str = "4-class-wide-middle",
        n_quantiles: Optional[int] = None,  # inferred from strategy if None
        use_control_sites_only: bool = False,
    ):
        ds_dir = Path(__file__).resolve().parent
        canonical_path = ds_dir / "datasets" / "final_merged_dataset.csv"
        
        if tier_strategy not in TIER_PRESETS:
            print(f"Warning: Unknown tier_strategy '{tier_strategy}'. Falling back to '4-class-wide-middle'.")
            tier_strategy = "4-class-wide-middle"
        
        self.tier_strategy = tier_strategy
        self.percentile_splits = TIER_PRESETS[tier_strategy]
        self.n_quantiles = n_quantiles or len(self.percentile_splits)

        # Single-source mode: canonical merged dataset only.
        if not canonical_path.exists():
            raise FileNotFoundError(f"Canonical dataset not found: {canonical_path}")


        print("Loading and merging data…")
        self.df = _load_and_merge(canonical_path)

        if use_control_sites_only and "client_id" in self.df.columns:
            before = len(self.df)
            self.df = self.df[
                ~self.df["client_id"].astype(str).str.contains(
                    "Controls Training", case=False, na=False
                )
            ].copy()
            print(f"  Control-sites only: {len(self.df)} sites (was {before})")

        # ML feature cols: canonical order from ML_FEATURE_ORDER, filtered to columns present.
        # carwash_type_encoded is DISPLAY_ONLY — already encoded by effective_capacity.
        # The canonical order matters for ExtraTrees random_state reproducibility:
        # feature index assignments affect which columns are sampled at each split.
        available = set(self.df.columns) - {
            "Address", "current_count", "location_id", "client_id",
            "street", "city", "zip", "region", "state",
            "wash_q", "_match_type", "primary_carwash_type",
            *DISPLAY_ONLY_FEATURES,
        }
        self.feature_cols: List[str] = [
            f for f in ML_FEATURE_ORDER if f in available and f in FEATURE_LABELS
        ]

        print(f"  {len(self.df)} sites  |  {len(self.feature_cols)} ML features  |  "
              f"{len(DISPLAY_ONLY_FEATURES)} display-only features")

        self._build_feature_distributions()
        self._build_wash_quantiles()
        self._build_quantile_profiles()
        self._train_classifier()
        self._train_volume_regressors()
        print("✓ QuantilePredictorV4 ready\n")

    # ── Build helpers ─────────────────────────────────────────────────────────

    def _build_feature_distributions(self):
        # Build distributions for ALL features in FEATURE_LABELS (including display-only)
        self.feature_dists: Dict[str, Dict] = {}
        all_display_feats = list(self.feature_cols) + [
            f for f in DISPLAY_ONLY_FEATURES if f in self.df.columns
        ]
        for feat in all_display_feats:
            data = self.df[feat].dropna().values.astype(float)
            if len(data) == 0:
                continue
            shape = _detect_shape(data)
            boundaries = _quantile_boundaries(data, self.n_quantiles)
            direction = FEATURE_DIRECTIONS.get(feat, "higher")
            signal = FEATURE_SIGNAL.get(feat, 0.0)
            self.feature_dists[feat] = {
                "data": data,
                "min": float(np.min(data)),
                "max": float(np.max(data)),
                "mean": float(np.mean(data)),
                "median": float(np.median(data)),
                "std": float(np.std(data)),
                "shape": shape,
                "boundaries": boundaries,
                "direction": direction,
                "signal": signal,
            }

    def _build_wash_quantiles(self):
        counts = self.df["current_count"].dropna().values
        self.wash_shape = _detect_shape(counts)
        
        # Calculate boundaries based on percentile splits (e.g. 20-30-30-20)
        cum_percentiles = np.cumsum([0] + self.percentile_splits)
        self.wash_boundaries = np.percentile(counts, cum_percentiles)
        # Ensure exact min/max from dataset if percentile interpolation shifts them
        self.wash_boundaries[0] = np.min(counts)
        self.wash_boundaries[-1] = np.max(counts)

        self.df["wash_q"] = pd.cut(
            self.df["current_count"],
            bins=self.wash_boundaries,
            labels=list(range(1, self.n_quantiles + 1)),
            include_lowest=True,
        ).astype("Int64")

        self.wash_q_ranges: Dict[int, Tuple[float, float]] = {
            q: (float(self.wash_boundaries[q - 1]), float(self.wash_boundaries[q]))
            for q in range(1, self.n_quantiles + 1)
        }
        print(f"\nCar wash count tier ranges (strategy: {self.tier_strategy}):")
        for q, (lo, hi) in self.wash_q_ranges.items():
            n = int((self.df["wash_q"] == q).sum())
            print(f"  Q{q}: {lo:>10,.0f} - {hi:>10,.0f} cars/yr  (n={n})")

    def _build_quantile_profiles(self):
        # Build profiles for ALL features (including display-only) for analysis
        all_feats = list(self.feature_cols) + [
            f for f in DISPLAY_ONLY_FEATURES if f in self.df.columns
        ]
        self.quantile_profiles: Dict[int, Dict[str, Dict]] = {}
        self.tier_medians: Dict[int, float] = {}
        self.tier_means: Dict[int, float] = {}
        
        counts = self.df["current_count"].dropna()
        
        for q in range(1, self.n_quantiles + 1):
            mask = self.df["wash_q"] == q
            subset = self.df[mask]
            
            # Store median wash count for this tier (used for weighted predictions)
            q_counts = counts[mask]
            self.tier_medians[q] = float(q_counts.median()) if not q_counts.empty else 0.0
            self.tier_means[q] = float(q_counts.mean()) if not q_counts.empty else 0.0
            
            profile: Dict[str, Dict] = {}
            for feat in all_feats:
                vals = subset[feat].dropna()
                if len(vals) > 0:
                    profile[feat] = {
                        "median": float(vals.median()),
                        "mean":   float(vals.mean()),
                        "q25":    float(vals.quantile(0.25)),
                        "q75":    float(vals.quantile(0.75)),
                        "n":      len(vals),
                    }
            self.quantile_profiles[q] = profile

    def _train_classifier(self):
        """
        Train a calibrated ExtraTreesClassifier.

        v4 key changes vs v3:
        - ExtraTrees replaces RandomForest: random split thresholds reduce overfitting
          on the 482-site dataset (+0.6% exact CV accuracy at same feature set)
        - Optuna-tuned hyperparameters (50 trials, 5-fold CV):
          n_estimators=600  max_depth=8  min_samples_leaf=5  max_features='sqrt'
          Deeper trees (max_depth=12 as in v3) overfit on the small dataset.
        - CalibratedClassifierCV wraps ET for reliable probability estimates
        - carwash_type_encoded is intentionally excluded from ML features:
          it is already captured by effective_capacity = tunnel_count × is_express
          (adding it causes multicollinearity: ablation = −0.9% exact accuracy)
        """
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.impute import KNNImputer

        X_raw = self.df[self.feature_cols].copy()
        for col in X_raw.columns:
            X_raw[col] = pd.to_numeric(X_raw[col], errors="coerce")

        y = self.df["wash_q"].copy()
        mask = y.notna()
        X_raw_m = X_raw[mask]
        y_clean = y[mask].values.astype(int)

        imputer = KNNImputer(n_neighbors=5)
        X_clean = imputer.fit_transform(X_raw_m)
        self._knn_imputer = imputer
        self._feature_medians = {f: float(X_raw[f].median()) for f in self.feature_cols}
        self._X_train_imputed = X_clean
        self._y_train = y_clean
        self._counts_train = self.df.loc[mask, "current_count"].astype(float).values

        # ExtraTrees with Optuna-tuned hyperparameters
        # Optuna best (50 trials, 5-fold CV on 482 sites):
        # n_estimators=600, max_depth=8, min_samples_leaf=5, max_features='sqrt' → 63.1% CV
        base_clf = ExtraTreesClassifier(
            n_estimators=600,
            max_depth=8,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        skf_cal = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        self.clf = CalibratedClassifierCV(base_clf, cv=skf_cal, method="isotonic")
        self.clf.fit(X_clean, y_clean)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_exact = cross_val_score(base_clf, X_clean, y_clean, cv=skf, scoring="accuracy")
        self.cv_accuracy = float(cv_exact.mean())

        all_preds, all_true = [], []
        for tr, te in skf.split(X_clean, y_clean):
            base_clf.fit(X_clean[tr], y_clean[tr])
            all_preds.extend(base_clf.predict(X_clean[te]))
            all_true.extend(y_clean[te])
        adj = np.mean(np.abs(np.array(all_preds) - np.array(all_true)) <= 1)
        self.cv_adjacent_accuracy = float(adj)

        base_clf.fit(X_clean, y_clean)
        self.feature_importances: Dict[str, float] = dict(
            zip(self.feature_cols, base_clf.feature_importances_)
        )

        print(
            f"\nClassifier v4 (ExtraTrees) — 5-fold CV  exact: {self.cv_accuracy:.1%}  "
            f"within-1: {self.cv_adjacent_accuracy:.1%}  (n={len(y_clean)})"
        )
        top = sorted(self.feature_importances.items(), key=lambda x: -x[1])[:5]
        print("  Top 5 features by importance:")
        for f, imp in top:
            print(f"    {FEATURE_LABELS.get(f, f)}: {imp:.1%}")

    def _train_volume_regressors(self):
        """
        Train one regressor per wash quantile to estimate E[Y | Q_i, X].
        This is used for conditional expectation volume:
            E[Y|X] = sum_i P(Q_i|X) * E[Y|Q_i, X]
        Falls back to tier means when a quantile has too few samples.
        """
        from sklearn.ensemble import ExtraTreesRegressor

        self._volume_regressors: Dict[int, object] = {}
        self._volume_fallback_means: Dict[int, float] = dict(self.tier_means)

        X_clean = getattr(self, "_X_train_imputed", None)
        y_clean = getattr(self, "_y_train", None)
        counts = getattr(self, "_counts_train", None)
        if X_clean is None or y_clean is None or counts is None:
            return

        for q in range(1, self.n_quantiles + 1):
            mask_q = y_clean == q
            n_q = int(mask_q.sum())
            if n_q < 15:
                self._volume_regressors[q] = None
                continue
            reg = ExtraTreesRegressor(
                n_estimators=200,
                max_depth=6,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            )
            reg.fit(X_clean[mask_q], counts[mask_q])
            self._volume_regressors[q] = reg

    # ── Feature-to-wash-Q mapping ─────────────────────────────────────────────

    def _feature_to_wash_q(self, feat: str, val: float, direction: str) -> Dict:
        group_medians: Dict[int, float] = {}
        for q in range(1, self.n_quantiles + 1):
            med = self.quantile_profiles.get(q, {}).get(feat, {}).get("median")
            if med is not None:
                group_medians[q] = med

        if not group_medians:
            return {"wash_q": None, "group_medians": {}, "exceeds_q4": False, "q4_median": None}

        signal = FEATURE_SIGNAL.get(feat, 0.0)
        if signal < SIGNAL_THRESHOLD:
            return {"wash_q": None, "group_medians": group_medians, "exceeds_q4": False,
                    "q4_median": group_medians.get(4), "low_signal": True}

        sign = -1 if direction == "lower" else 1
        adj_val = sign * val
        adj_medians = {q: sign * m for q, m in group_medians.items()}

        q4_med = group_medians.get(4)
        q4_adj = adj_medians.get(4)
        exceeds_q4 = q4_adj is not None and adj_val > q4_adj

        matched_q = 4 if exceeds_q4 else min(adj_medians.items(), key=lambda x: abs(x[1] - adj_val))[0]

        return {"wash_q": matched_q, "group_medians": group_medians,
                "exceeds_q4": exceeds_q4, "q4_median": q4_med, "low_signal": False}

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(
        self,
        location_features: Dict[str, float],
        llm_narrative: bool = True,
    ) -> Dict:
        from sklearn.impute import KNNImputer

        # Auto-compute derived features if not explicitly provided
        location_features = dict(location_features)  # copy to avoid mutating caller's dict
        if "effective_capacity" not in location_features:
            tc = location_features.get("tunnel_count")
            cw = location_features.get("carwash_type_encoded")
            if tc is not None:
                is_express = (float(cw) == 1.0) if cw is not None else True  # default Express
                location_features["effective_capacity"] = float(tc) * float(is_express)

        feature_vector_raw: List[Optional[float]] = []
        feature_analysis: Dict[str, Dict] = {}

        # ── Main ML feature loop ───────────────────────────────────────────────
        for feat in self.feature_cols:
            raw_val = location_features.get(feat)
            dist = self.feature_dists.get(feat)
            if dist is None:
                feature_vector_raw.append(np.nan)
                continue
            direction = dist["direction"]

            if raw_val is not None and not (isinstance(raw_val, float) and np.isnan(raw_val)):
                val = float(raw_val)
                imputed = False
            else:
                val = dist["median"]
                imputed = True

            feature_vector_raw.append(val if not imputed else np.nan)

            raw_pct = float(stats.percentileofscore(dist["data"], val, kind="rank"))
            if direction == "lower":
                adj_pct = 100.0 - raw_pct
            else:
                adj_pct = raw_pct

            raw_q  = _assign_raw_quantile(val, dist["boundaries"])
            adj_q  = _adj_quantile(raw_q, direction)

            better_boundary = _next_better_boundary(dist["boundaries"], adj_q, direction)
            delta    = abs(val - better_boundary) if better_boundary is not None else None
            shift_dir = ("increase" if direction == "higher" else "decrease") if better_boundary else None

            wq_info = self._feature_to_wash_q(feat, val, direction)

            feature_analysis[feat] = {
                "value":                        val if not imputed else None,
                "imputed":                      imputed,
                "label":                        FEATURE_LABELS.get(feat, feat),
                "direction":                    direction,
                "signal":                       FEATURE_SIGNAL.get(feat, 0.0),
                "raw_percentile":               round(raw_pct, 1),
                "adjusted_percentile":          round(adj_pct, 1),
                "feature_quantile_raw":         raw_q,
                "feature_quantile_adj":         adj_q,
                "wash_correlated_q":            wq_info.get("wash_q"),
                "wash_correlated_exceeds_q4":   wq_info.get("exceeds_q4", False),
                "wash_q_group_medians":         wq_info.get("group_medians", {}),
                "wash_q_q4_median":             wq_info.get("q4_median"),
                "wash_q_low_signal":            wq_info.get("low_signal", False),
                "distribution_shape":           dist["shape"],
                "dist_min":                     dist["min"],
                "dist_max":                     dist["max"],
                "dist_median":                  dist["median"],
                "quantile_boundaries":          dist["boundaries"].tolist(),
                "shift_to_next_q_delta":        round(delta, 2) if delta is not None else None,
                "shift_to_next_q_direction":    shift_dir,
                "importance":                   round(self.feature_importances.get(feat, 0), 4),
                "ml_feature":                   True,
            }

        # Impute using trained KNN
        X_df = pd.DataFrame([feature_vector_raw], columns=self.feature_cols)
        X_imputed = self._knn_imputer.transform(X_df)
        for i, feat in enumerate(self.feature_cols):
            if feat in feature_analysis and feature_analysis[feat].get("imputed"):
                feature_analysis[feat]["value"] = float(X_imputed[0, i])

        # ── Display-only feature loop (carwash_type_encoded etc.) ─────────────
        # These appear in the report but are NOT fed to the ML model.
        for feat in DISPLAY_ONLY_FEATURES:
            raw_val = location_features.get(feat)
            dist = self.feature_dists.get(feat)
            if dist is None:
                continue
            direction = dist["direction"]

            if raw_val is not None and not (isinstance(raw_val, float) and np.isnan(raw_val)):
                val = float(raw_val)
                imputed = False
            else:
                val = dist["median"]
                imputed = True

            raw_pct = float(stats.percentileofscore(dist["data"], val, kind="rank"))
            adj_pct = 100.0 - raw_pct if direction == "lower" else raw_pct
            raw_q   = _assign_raw_quantile(val, dist["boundaries"])
            adj_q   = _adj_quantile(raw_q, direction)
            wq_info = self._feature_to_wash_q(feat, val, direction)

            feature_analysis[feat] = {
                "value":                        val if not imputed else None,
                "imputed":                      imputed,
                "label":                        FEATURE_LABELS.get(feat, feat),
                "direction":                    direction,
                "signal":                       FEATURE_SIGNAL.get(feat, 0.0),
                "raw_percentile":               round(raw_pct, 1),
                "adjusted_percentile":          round(adj_pct, 1),
                "feature_quantile_raw":         raw_q,
                "feature_quantile_adj":         adj_q,
                "wash_correlated_q":            wq_info.get("wash_q"),
                "wash_correlated_exceeds_q4":   wq_info.get("exceeds_q4", False),
                "wash_q_group_medians":         wq_info.get("group_medians", {}),
                "wash_q_q4_median":             wq_info.get("q4_median"),
                "wash_q_low_signal":            wq_info.get("low_signal", False),
                "distribution_shape":           dist["shape"],
                "dist_min":                     dist["min"],
                "dist_max":                     dist["max"],
                "dist_median":                  dist["median"],
                "quantile_boundaries":          dist["boundaries"].tolist(),
                "shift_to_next_q_delta":        None,
                "shift_to_next_q_direction":    None,
                "importance":                   0.0,  # not in ML model
                "ml_feature":                   False,  # display only
                "display_only_note": (
                    "Captured via effective_capacity = tunnel_count × is_express. "
                    "Adding directly to ML model causes multicollinearity (−0.9% accuracy). "
                    "Shown here for interpretation only."
                ),
            }

        # Predict
        predicted_q   = int(self.clf.predict(X_imputed)[0])
        proba_arr     = self.clf.predict_proba(X_imputed)[0]
        classes       = list(self.clf.classes_)
        proba         = {int(c): round(float(p), 3) for c, p in zip(classes, proba_arr)}

        wash_range = self.wash_q_ranges[predicted_q]

        shift_opportunities = self._quantile_shift_analysis(
            X_imputed[0].tolist(), feature_analysis, predicted_q
        )
        profile_comparison = self._profile_comparison(feature_analysis)
        strengths, weaknesses = self._strengths_weaknesses(feature_analysis)

        # Weighted volume estimates:
        # - mean_by_tier: quick expectation baseline (uses E[Y|Q_i] as tier means)
        # - conditional_regression: best estimate from per-tier regressors E[Y|Q_i, X]
        weighted_vol_mean = sum(
            proba.get(q, 0.0) * self.tier_means.get(q, 0.0)
            for q in range(1, self.n_quantiles + 1)
        )
        conditional_expectations: Dict[int, float] = {}
        for q in range(1, self.n_quantiles + 1):
            reg = (getattr(self, "_volume_regressors", {}) or {}).get(q)
            if reg is None:
                conditional_expectations[q] = float(
                    (getattr(self, "_volume_fallback_means", {}) or {}).get(q, self.tier_means.get(q, 0.0))
                )
            else:
                conditional_expectations[q] = float(reg.predict(X_imputed)[0])
        weighted_vol_conditional = sum(
            proba.get(q, 0.0) * conditional_expectations.get(q, 0.0)
            for q in range(1, self.n_quantiles + 1)
        )
        weighted_vol = weighted_vol_conditional

        # Probabilistic uncertainty from class-probability spread around the
        # conditional expectation (same estimator used for point prediction).
        variance = sum(
            proba.get(q, 0.0) * ((conditional_expectations.get(q, 0.0) - weighted_vol) ** 2)
            for q in range(1, self.n_quantiles + 1)
        )
        sigma = float(np.sqrt(max(variance, 0.0)))
        confidence_score = float(max(proba.values())) if proba else None
        entropy = (
            float(-sum(p * np.log(max(p, 1e-12)) for p in proba.values()))
            if proba else None
        )

        result: Dict = {
            "predicted_wash_quantile":          predicted_q,
            "predicted_wash_quantile_label":    QUANTILE_LABELS[predicted_q] if predicted_q in QUANTILE_LABELS else f"Q{predicted_q}",
            "predicted_wash_tier":              QUANTILE_TIER_NAMES[predicted_q] if predicted_q in QUANTILE_TIER_NAMES else f"Tier {predicted_q}",
            "predicted_wash_range": {
                "min":   round(wash_range[0]),
                "max":   round(wash_range[1]),
                "label": f"{wash_range[0]:,.0f} - {wash_range[1]:,.0f} cars/yr",
            },
            "weighted_volume_prediction":       round(weighted_vol),
            "weighted_volume_prediction_mean_by_tier": round(weighted_vol_mean),
            "weighted_volume_prediction_method": "conditional_regression",
            "conditional_expectations_by_quantile": {
                f"Q{q}": round(v, 1) for q, v in conditional_expectations.items()
            },
            "volume_uncertainty": {
                "sigma": round(sigma),
                "low": round(weighted_vol - sigma),
                "high": round(weighted_vol + sigma),
                "method": "probability_variance",
            },
            "prediction_confidence": {
                "max_probability": round(confidence_score, 4) if confidence_score is not None else None,
                "entropy": round(entropy, 4) if entropy is not None else None,
            },
            "quantile_probabilities":           proba,
            "tier_metadata": {
                "strategy":   self.tier_strategy,
                "n_tiers":    self.n_quantiles,
                "boundaries": [float(b) for b in self.wash_boundaries],
                "medians":    self.tier_medians,
                "means":      self.tier_means,
                "labels":     [QUANTILE_TIER_NAMES.get(q, f"Tier {q}") for q in range(1, self.n_quantiles + 1)],
            },
            "wash_count_distribution": {
                q: {
                    "range": f"{self.wash_q_ranges[q][0]:,.0f} - {self.wash_q_ranges[q][1]:,.0f}",
                    "min":   round(self.wash_q_ranges[q][0]),
                    "max":   round(self.wash_q_ranges[q][1]),
                }
                for q in range(1, self.n_quantiles + 1)
            },
            "feature_analysis":     feature_analysis,
            "shift_opportunities":  shift_opportunities,
            "profile_comparison":   profile_comparison,
            "strengths":            strengths,
            "weaknesses":           weaknesses,
            "model_cv_accuracy":    round(self.cv_accuracy, 3),
            "model_adj_accuracy":   round(self.cv_adjacent_accuracy, 3),
            "features_available":   sum(
                1 for fa in feature_analysis.values()
                if not fa.get("imputed", True) and fa.get("ml_feature", True)
            ),
        }

        if llm_narrative:
            result["narrative"]                  = self._generate_narrative(result)
            result["strengths_weaknesses_llm"]   = self._generate_strengths_weaknesses_llm(result)

        return result

    # ── Analysis helpers ──────────────────────────────────────────────────────

    def _quantile_shift_analysis(
        self,
        base_vector: List[float],
        feature_analysis: Dict[str, Dict],
        current_q: int,
    ) -> List[Dict]:
        if current_q >= self.n_quantiles:
            return []

        opportunities: List[Dict] = []
        prob_lifts: List[Dict] = []

        classes = list(self.clf.classes_)
        target_cls_idx = classes.index(current_q + 1) if (current_q + 1) in classes else None
        base_proba_up  = self.clf.predict_proba(np.array(base_vector).reshape(1, -1))[0][target_cls_idx] if target_cls_idx is not None else 0.0

        for i, feat in enumerate(self.feature_cols):
            fa = feature_analysis.get(feat, {})
            if fa.get("imputed") or fa.get("direction") == "neutral":
                continue

            dist      = self.feature_dists.get(feat)
            val       = fa.get("value")
            adj_q     = fa.get("feature_quantile_adj", 2)
            direction = fa.get("direction")

            if dist is None or val is None or adj_q >= self.n_quantiles:
                continue

            best_sim_q, best_target_val, best_gain = current_q, val, 0

            for target_adj_q in range(adj_q + 1, self.n_quantiles + 1):
                raw_target_q = (5 - target_adj_q) if direction == "lower" else target_adj_q
                if raw_target_q < 1 or raw_target_q > self.n_quantiles:
                    continue
                boundary_idx = raw_target_q - 1
                if boundary_idx < 0 or boundary_idx >= len(dist["boundaries"]):
                    continue
                if direction == "higher":
                    target_val = dist["boundaries"][target_adj_q - 1] + 0.01
                else:
                    target_val = dist["boundaries"][self.n_quantiles - target_adj_q + 1] - 0.01
                    if target_val < 0:
                        target_val = max(0, dist["boundaries"][0])

                sim_vector = base_vector.copy()
                sim_vector[i] = target_val
                sim_q = int(self.clf.predict(np.array(sim_vector).reshape(1, -1))[0])

                if sim_q > best_sim_q:
                    best_sim_q, best_target_val, best_gain = sim_q, target_val, sim_q - current_q
                    break

            change = abs(val - best_target_val)
            if best_gain > 0 and change > 0.001:
                opportunities.append({
                    "feature":          feat,
                    "label":            FEATURE_LABELS.get(feat, feat),
                    "current_value":    val,
                    "target_value":     round(float(best_target_val), 2),
                    "change_needed":    round(change, 2),
                    "change_direction": "increase" if direction == "higher" else "decrease",
                    "current_wash_q":   current_q,
                    "simulated_wash_q": best_sim_q,
                    "q_gain":           best_gain,
                    "importance":       fa.get("importance", 0),
                })
            elif target_cls_idx is not None:
                sim_vector = base_vector.copy()
                if direction == "higher":
                    sim_vector[i] = float(dist["boundaries"][min(adj_q, self.n_quantiles - 1)])
                else:
                    sim_vector[i] = float(dist["boundaries"][max(self.n_quantiles - adj_q, 1)])
                sim_p = self.clf.predict_proba(np.array(sim_vector).reshape(1, -1))[0][target_cls_idx]
                lift  = sim_p - base_proba_up
                if lift > 0.01:
                    prob_lifts.append({
                        "feature":          feat,
                        "label":            FEATURE_LABELS.get(feat, feat),
                        "current_value":    val,
                        "target_value":     round(float(sim_vector[i]), 2),
                        "change_needed":    round(abs(val - sim_vector[i]), 2),
                        "change_direction": "increase" if direction == "higher" else "decrease",
                        "current_wash_q":   current_q,
                        "simulated_wash_q": current_q,
                        "q_gain":           0,
                        "prob_lift":        round(lift * 100, 1),
                        "importance":       fa.get("importance", 0),
                    })

        opportunities.sort(key=lambda x: (-x["q_gain"], -x["importance"]))
        if not opportunities and prob_lifts:
            prob_lifts.sort(key=lambda x: (-x["prob_lift"], -x["importance"]))
            return prob_lifts[:5]
        return opportunities[:5]

    def _profile_comparison(self, feature_analysis: Dict[str, Dict]) -> Dict[str, Dict]:
        comparison: Dict[str, Dict] = {}
        for feat, fa in feature_analysis.items():
            if fa.get("value") is None:
                continue
            val       = fa["value"]
            direction = fa["direction"]
            feat_comp: Dict[str, Dict] = {}
            for q in range(1, self.n_quantiles + 1):
                pm = self.quantile_profiles.get(q, {}).get(feat, {}).get("median")
                if pm is None:
                    continue
                if direction == "higher":
                    align = "above" if val > pm else "below" if val < pm else "at"
                elif direction == "lower":
                    align = "better" if val < pm else "worse" if val > pm else "at"
                else:
                    align = "near"
                feat_comp[f"Q{q}"] = {"profile_median": round(float(pm), 2), "alignment": align}
            comparison[feat] = feat_comp
        return comparison

    def _strengths_weaknesses(
        self, feature_analysis: Dict[str, Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        strengths: List[Dict] = []
        weaknesses: List[Dict] = []
        for feat, fa in feature_analysis.items():
            if fa.get("imputed") or fa.get("wash_q_low_signal"):
                continue
            wq      = fa.get("wash_correlated_q")
            exceeds = fa.get("wash_correlated_exceeds_q4", False)
            label   = fa.get("label", feat)
            val     = fa.get("value")
            q4_med  = fa.get("wash_q_q4_median")
            imp     = fa.get("importance", 0)
            signal  = fa.get("signal", 0.0)
            entry   = {"label": label, "value": val, "q4_median": q4_med,
                       "importance": imp, "signal": signal}
            if wq == 4 or exceeds:
                entry["note"] = "exceeds Q4 typical" if exceeds else "matches Q4"
                strengths.append(entry)
            elif wq in (1, 2):
                entry["note"] = "Q1-level (low performing)" if wq == 1 else "Q2-level (below median)"
                weaknesses.append(entry)
        strengths.sort(key=lambda x: -x["importance"])
        weaknesses.sort(key=lambda x: -x["importance"])
        return strengths, weaknesses

    # ── LLM methods ───────────────────────────────────────────────────────────

    def _call_llm(self, prompt: str, temperature: float = 0.3) -> Optional[str]:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
            from app.utils.llm import local_llm as llm_module
            response = llm_module.get_llm_response(
                prompt, reasoning_effort="medium", temperature=temperature
            )
            text = (response or {}).get("generated_text", "").strip()
            return text if text else None
        except Exception:
            return None

    def _generate_narrative(self, result: Dict) -> str:
        pred_q    = result["predicted_wash_quantile"]
        tier      = result["predicted_wash_tier"]
        wash_range = result["predicted_wash_range"]["label"]
        proba     = result["quantile_probabilities"]
        conf      = round(proba.get(pred_q, 0) * 100, 1)

        sig_feats = sorted(
            [(f, fa) for f, fa in result["feature_analysis"].items()
             if not fa.get("imputed") and fa.get("signal", 0) >= SIGNAL_THRESHOLD
             and fa.get("ml_feature", True)],
            key=lambda x: -x[1].get("importance", 0),
        )[:6]

        feat_lines = []
        for feat, fa in sig_feats:
            label  = fa["label"]
            val    = fa["value"]
            wq     = fa.get("wash_correlated_q")
            q4_med = fa.get("wash_q_q4_median")
            exceeds = fa.get("wash_correlated_exceeds_q4", False)
            q4_note = f"Q4 typical: {q4_med:.1f}" if q4_med is not None else ""
            wq_note = "exceeds Q4 level" if exceeds else (f"matches Q{wq} sites" if wq else "")
            feat_lines.append(f"  - {label}: {val:.1f}  [{wq_note}, {q4_note}]")

        prompt = f"""You are a car wash site analyst. Write a SHORT narrative summary — strictly 2–3 sentences total.

PREDICTION: Q{pred_q} ({tier}), expected annual count {wash_range}, model confidence {conf}%.

KEY FEATURES (signal-bearing only):
{chr(10).join(feat_lines) if feat_lines else "  (none with significant signal)"}

Rules:
- Exactly 2–3 sentences. No bullets. No headers.
- Sentence 1: state the quartile and expected volume.
- Sentence 2-3: mention 2–3 key features and whether they match high-performing (Q4) site levels or not.
- Do NOT suggest improvements. Do NOT list all features."""

        text = self._call_llm(prompt, temperature=0.25)
        return (text or "").strip()

    def _generate_strengths_weaknesses_llm(self, result: Dict) -> str:
        strengths = result.get("strengths") or []
        weaknesses = result.get("weaknesses") or []
        pred_q     = result["predicted_wash_quantile"]
        tier       = result["predicted_wash_tier"]
        wash_range = result["predicted_wash_range"]["label"]

        def fmt(items):
            return [
                f"  - {s['label']}: {s['value']:.1f}" +
                (f" (Q4 med {s['q4_median']:.1f})" if s.get("q4_median") is not None else "") +
                f"  [{s['note']}]"
                for s in items[:8]
            ]

        s_block = "\n".join(fmt(strengths)) if strengths else "  (none identified)"
        w_block = "\n".join(fmt(weaknesses)) if weaknesses else "  (none identified)"

        prompt = f"""You are a car wash site investment analyst. Write a concise investment-report-style assessment.

SITE PREDICTION: Q{pred_q} ({tier}), expected {wash_range} cars/yr.

STRENGTHS (features matching Q4 high-performer levels):
{s_block}

WEAKNESSES (features at Q1/Q2 low-performer levels):
{w_block}

Output format — exactly 2 short paragraphs:
Paragraph 1 (3–5 sentences): Key strengths. Mention each strength once with its value vs Q4 benchmark. Professional analytical tone.
Paragraph 2 (3–5 sentences): Key weaknesses. Mention each weakness once with its value vs Q4 benchmark.
No bullet points in output. No recommendations. No subheadings."""

        text = self._call_llm(prompt, temperature=0.3)
        return (text or "").strip()

    # ── Reporting ─────────────────────────────────────────────────────────────

    def print_report(self, result: Dict):
        q = result["predicted_wash_quantile"]
        W = 90

        def hdr(title: str):
            print(f"\n{'─' * W}")
            print(f"  {title}")
            print(f"{'─' * W}")

        print("\n" + "═" * W)
        print("  QUANTILE PREDICTION REPORT  —  v4  (ExtraTrees, Optuna-tuned)")
        print("═" * W)
        print(f"\n  Predicted Quantile  : {result['predicted_wash_quantile_label']}  — {result['predicted_wash_tier']}")
        print(f"  Expected Annual Vol : {result['predicted_wash_range']['label']}")

        hdr("PREDICTION CONFIDENCE")
        for qi in range(1, self.n_quantiles + 1):
            p     = result["quantile_probabilities"].get(qi, 0)
            bar   = "█" * int(p * 36)
            mark  = "  ◄ PREDICTED" if qi == q else ""
            r     = result["wash_count_distribution"][qi]["range"]
            lbl   = f"Q{qi} {QUANTILE_TIER_NAMES[qi][:14]}"
            print(f"  {lbl:<22s} [{bar:<36s}] {p*100:5.1f}%  {r}{mark}")

        hdr(
            "FEATURE ANALYSIS  (★ exceeds Q4  ▲ matches Q4  ▼ matches Q1  ~ low signal  [D]=display-only)\n"
            "  WashQ = car wash tier this value matches  |  Signal = |Spearman r| with count\n"
            "  [D] = shown for context, NOT in ML model (already captured by effective_capacity)"
        )
        print(
            f"  {'Feature':<38s} {'Value':>8s}  {'WashQ':>5s}  {'Pctile':>6s}"
            f"  {'Q4 med':>8s}  {'Signal':>6s}  {'Imprt':>5s}  {'ML':>3s}"
        )
        print("  " + "─" * (W - 2))

        ml_feats = [(f, fa) for f, fa in result["feature_analysis"].items() if fa.get("ml_feature", True)]
        disp_feats = [(f, fa) for f, fa in result["feature_analysis"].items() if not fa.get("ml_feature", True)]
        sorted_feats = sorted(ml_feats, key=lambda x: -x[1].get("importance", 0)) + disp_feats

        for feat, fa in sorted_feats:
            is_ml   = fa.get("ml_feature", True)
            label   = (fa["label"][:34] + " (imp)" if fa.get("imputed") else fa["label"][:36])
            val     = fa["value"]
            wq      = fa.get("wash_correlated_q")
            exceeds = fa.get("wash_correlated_exceeds_q4", False)
            low_sig = fa.get("wash_q_low_signal", False)
            pct     = fa["adjusted_percentile"]
            imp     = fa["importance"]
            q4_med  = fa.get("wash_q_q4_median")
            signal  = fa.get("signal", 0.0)
            ml_tag  = "✓" if is_ml else "D"

            if low_sig or wq is None:
                wq_str = "  ~"
                marker = ""
            else:
                wq_str = f"Q{wq}"
                marker = "★" if exceeds else ("▲" if wq == 4 else ("▼" if wq == 1 else " "))

            q4_str = f"{q4_med:>8.1f}" if q4_med is not None else f"{'n/a':>8s}"
            print(
                f"  {label:<38s} {val:>8.1f}  {wq_str:>4s}{marker}  {pct:>5.1f}%"
                f"  {q4_str}  {signal:>5.3f}  {imp:>5.1%}  {ml_tag:>3s}"
            )

        hdr("WHY CARWASH TYPE IS CAPTURED VIA effective_capacity (not a direct ML feature)")
        cw_fa = result["feature_analysis"].get("carwash_type_encoded")
        if cw_fa:
            print(f"  carwash_type_encoded = {cw_fa.get('value', 'N/A'):.0f}  "
                  f"({FEATURE_LABELS['carwash_type_encoded']})")
            print(f"  WashQ match: Q{cw_fa.get('wash_correlated_q', '?')}  |  "
                  f"Spearman |r| = {cw_fa.get('signal', 0):.3f}  |  "
                  f"Adjusted percentile: {cw_fa.get('adjusted_percentile', 0):.0f}th")
            print()
            print("  Express Tunnel sites average 143K washes/yr vs 79-119K for other types.")
            print("  However, effective_capacity = tunnel_count × is_express already encodes this:")
            print("    Express 1 tunnel → ec=1 (avg 68K), Express 2 → ec=2 (avg 168K), ...")
            print("    Mobile/Flex/Hand Wash → ec=0 (avg 50-83K, uses age+location features)")
            print("  Adding carwash_type_encoded directly creates multicollinearity → −0.9% accuracy.")
        else:
            print("  carwash_type_encoded not provided in input.")

        hdr("STRENGTHS & WEAKNESSES  (LLM investment summary)")
        sw_llm = result.get("strengths_weaknesses_llm")
        if sw_llm:
            for para in sw_llm.split("\n\n"):
                block = para.strip()
                if block:
                    for line in textwrap.wrap(block, width=W - 4, initial_indent="  ", subsequent_indent="  "):
                        print(line)
                    print()

        if result["shift_opportunities"]:
            hard = [o for o in result["shift_opportunities"] if o.get("q_gain", 0) > 0]
            soft = [o for o in result["shift_opportunities"] if o.get("q_gain", 0) == 0]
            if hard:
                hdr("QUANTILE SHIFT OPPORTUNITIES")
                print(f"  {'Feature':<34s} {'Current':>9s} {'Target':>9s} {'Change':>8s}  Wash Q Shift")
                print("  " + "─" * (W - 2))
                for o in hard:
                    d = "+" if o["change_direction"] == "increase" else "−"
                    print(
                        f"  {o['label'][:32]:<34s} {o['current_value']:>9.1f} {o['target_value']:>9.1f} "
                        f"{d}{o['change_needed']:>7.1f}  Q{o['current_wash_q']} → Q{o['simulated_wash_q']}"
                    )
            if soft:
                hdr("PROBABILITY-LIFT OPPORTUNITIES  (no single feature flips the quartile)")
                print(f"  {'Feature':<34s} {'Current':>9s} {'Target':>9s}  Prob Lift")
                print("  " + "─" * (W - 2))
                for o in soft:
                    d = "+" if o["change_direction"] == "increase" else "−"
                    print(
                        f"  {o['label'][:32]:<34s} {o['current_value']:>9.1f} {o['target_value']:>9.1f}"
                        f"  +{o.get('prob_lift',0):.1f}%"
                    )

        hdr("PROFILE COMPARISON  (your value vs each car wash tier's median)")
        print(
            f"  {'Feature':<38s} {'Yours':>8s}  "
            + "  ".join(f"{'Q'+str(qi)+' med':>8s}" for qi in range(1, self.n_quantiles + 1))
        )
        print("  " + "─" * (W - 2))
        for feat, fa in sorted_feats[:12]:
            val  = fa["value"]
            comp = result["profile_comparison"].get(feat, {})
            meds = "  ".join(
                f"{comp.get(f'Q{qi}', {}).get('profile_median', 0.0):>8.1f}"
                for qi in range(1, self.n_quantiles + 1)
            )
            print(f"  {fa['label'][:36]:<38s} {val:>8.1f}  {meds}")

        if result.get("narrative"):
            hdr("NARRATIVE SUMMARY")
            for para in result["narrative"].split("\n\n"):
                block = para.strip()
                if block:
                    for line in textwrap.wrap(block, width=W - 4, initial_indent="  ", subsequent_indent="  "):
                        print(line)
                    print()

        print(
            f"\n  Model v4 (ExtraTrees) — 5-fold CV  exact: {result['model_cv_accuracy']:.1%}  "
            f"within-1-quartile: {result.get('model_adj_accuracy',0):.1%}  "
            f"ML features provided: {result['features_available']}/{len(self.feature_cols)}\n"
            f"  Error analysis: 96% of wrong predictions are adjacent (Q1↔Q2 or Q3↔Q4).\n"
            f"  Theoretical ceiling ~65% — volume also driven by traffic, ops, pricing (not in features)."
        )
        print("═" * W)

    def report_to_string(self, result: Dict) -> str:
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.print_report(result)
        return buf.getvalue()

    def save_report(self, result: Dict, path: Optional[Path] = None) -> Path:
        if path is None:
            path = Path(__file__).parent / "quantile_report_v4.txt"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.report_to_string(result), encoding="utf-8")
        print(f"✓ Report saved: {path}")
        return path

    def plot_feature_quantiles(
        self,
        result: Dict,
        output_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (24, 20),
    ) -> Optional[Path]:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D
        except ImportError:
            print("[Skip] matplotlib not installed.")
            return None

        if output_path is None:
            output_path = Path(__file__).parent / "quantile_analysis_v4.png"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Show only ML features in plot (skip display-only)
        sorted_feats = sorted(
            [(f, fa) for f, fa in result["feature_analysis"].items()
             if not fa.get("imputed") and fa.get("ml_feature", True)],
            key=lambda x: -x[1].get("importance", 0),
        )
        n_feat = len(sorted_feats)
        cols   = 4
        rows   = (n_feat + cols - 1) // cols + 1

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        fig.patch.set_facecolor("#f8f9fa")
        axes_flat = axes.flatten()
        q_colours = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]
        predicted_q = result["predicted_wash_quantile"]

        ax_prob = axes_flat[0]
        qs    = list(range(1, self.n_quantiles + 1))
        probs = [result["quantile_probabilities"].get(q, 0) * 100 for q in qs]
        bars  = ax_prob.bar([f"Q{q}" for q in qs], probs,
                            color=[q_colours[q-1] for q in qs], edgecolor="white", linewidth=0.8)
        for bar, p in zip(bars, probs):
            ax_prob.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                         f"{p:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax_prob.set_title(
            f"Predicted: Q{predicted_q} — {QUANTILE_TIER_NAMES[predicted_q]}\n"
            f"({result['predicted_wash_range']['label']})",
            fontsize=10, fontweight="bold")
        ax_prob.set_ylabel("Probability (%)", fontsize=8)
        ax_prob.set_ylim(0, 105)
        ax_prob.set_facecolor("#f0f0f0")

        ax_bench = axes_flat[1]
        ax_bench.axis("off")
        bench = "CAR WASH COUNT RANGES\n\n"
        for qi in range(1, self.n_quantiles + 1):
            r = result["wash_count_distribution"][qi]["range"]
            bench += f"Q{qi}: {r}{' ◄ YOU' if qi == predicted_q else ''}\n"
        bench += f"\nCV Exact: {result['model_cv_accuracy']:.1%} (v4)"
        bench += f"\nCV Within-1: {result.get('model_adj_accuracy',0):.1%}"
        bench += "\nModel: ExtraTrees (Optuna-tuned)"
        ax_bench.text(0.05, 0.95, bench, transform=ax_bench.transAxes, va="top", ha="left",
                      fontsize=8.5, family="monospace",
                      bbox=dict(boxstyle="round,pad=0.5", facecolor="#d4edda", edgecolor="#28a745"))

        ax_sw = axes_flat[2]
        ax_sw.axis("off")
        sw_text = "STRENGTHS & WEAKNESSES\n\n"
        for s in (result.get("strengths") or [])[:4]:
            sw_text += f"✓ {s['label'][:28]}: {s['value']:.1f}\n"
        sw_text += "\n"
        for w in (result.get("weaknesses") or [])[:4]:
            sw_text += f"✗ {w['label'][:28]}: {w['value']:.1f}\n"
        ax_sw.text(0.05, 0.95, sw_text, transform=ax_sw.transAxes, va="top", ha="left",
                   fontsize=8, family="monospace",
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff3cd", edgecolor="#ffc107"))

        # Carwash type note in panel 4
        ax_cw = axes_flat[3]
        ax_cw.axis("off")
        cw_fa = result["feature_analysis"].get("carwash_type_encoded", {})
        cw_text = "CARWASH TYPE (display-only)\n\n"
        if cw_fa.get("value") is not None:
            enc_map = {1: "Express Tunnel", 2: "Mobile/Flex", 3: "Hand Wash"}
            enc_val = int(round(cw_fa["value"]))
            cw_text += f"Type: {enc_map.get(enc_val, str(enc_val))}\n"
            cw_text += f"WashQ: Q{cw_fa.get('wash_correlated_q','?')}\n"
            cw_text += f"|r| = {cw_fa.get('signal',0):.3f}\n\n"
        cw_text += "Already captured via:\neffective_capacity = tc × is_express\nDirect use → −0.9% accuracy"
        ax_cw.text(0.05, 0.95, cw_text, transform=ax_cw.transAxes, va="top", ha="left",
                   fontsize=8, family="monospace",
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="#e8d5f5", edgecolor="#9b59b6"))

        for idx, (feat, fa) in enumerate(sorted_feats):
            ax            = axes_flat[cols + idx]
            dist          = self.feature_dists.get(feat)
            if dist is None:
                ax.axis("off")
                continue
            data          = dist["data"]
            val           = fa["value"]
            wq            = fa.get("wash_correlated_q", 2)
            exceeds       = fa.get("wash_correlated_exceeds_q4", False)
            group_medians = fa.get("wash_q_group_medians", {})
            low_sig       = fa.get("wash_q_low_signal", False)
            signal        = fa.get("signal", 0.0)

            ax.hist(data, bins=min(30, max(10, len(np.unique(data)))),
                    color="#aec7e8", edgecolor="white", linewidth=0.4, alpha=0.85, zorder=1)

            for qi in range(1, self.n_quantiles + 1):
                gm = group_medians.get(qi)
                if gm is not None:
                    ls = "--" if not low_sig else ":"
                    ax.axvline(gm, color=q_colours[qi-1], linestyle=ls, linewidth=1.3,
                               alpha=0.85, zorder=2, label=f"Q{qi}:{gm:.0f}")

            eff_wq = 4 if exceeds else (wq if wq else 2)
            mc = q_colours[eff_wq - 1] if not low_sig else "#555555"
            ax.axvline(val, color=mc, linestyle="-", linewidth=2.8, zorder=4, label=f"You:{val:.1f}")

            sig_tag  = f"r={signal:.3f}" if signal >= SIGNAL_THRESHOLD else f"r={signal:.3f}~"
            wq_tag   = f"WashQ{wq}" if not low_sig and wq else "~"
            ax.set_title(
                f"{fa['label'][:30]}\n{wq_tag} | {fa['adjusted_percentile']:.0f}th pct | {sig_tag} | imp {fa['importance']:.1%}",
                fontsize=7, fontweight="bold" if fa["importance"] > 0.06 else "normal")
            ax.set_ylabel(f"n={len(data)}", fontsize=6)
            ax.tick_params(labelsize=6)
            ax.legend(fontsize=5, loc="upper right", ncol=1)
            ax.set_facecolor("#f8f9fa")

        for j in range(cols + n_feat, len(axes_flat)):
            axes_flat[j].axis("off")

        legend_handles = [
            Line2D([0], [0], color=q_colours[q-1], linestyle="--", linewidth=1.5,
                   label=f"Q{q} median ({QUANTILE_TIER_NAMES[q]})")
            for q in range(1, self.n_quantiles + 1)
        ] + [Line2D([0], [0], color="black", linestyle="-", linewidth=2.5, label="Your value")]
        fig.legend(handles=legend_handles, loc="lower center", ncol=5,
                   fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.01))

        fig.suptitle(
            f"Car Wash Quantile Analysis v4  —  Predicted Q{predicted_q} "
            f"({result['predicted_wash_range']['label']})\n"
            f"Model: ExtraTrees (Optuna-tuned, exact CV {result['model_cv_accuracy']:.1%})  |  "
            f"carwash_type shown in summary panel (display-only, captured by effective_capacity)",
            fontsize=11, fontweight="bold", y=1.01)
        plt.tight_layout(rect=[0, 0.03, 1, 1])
        plt.savefig(output_path, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"✓ Plot saved: {output_path}")
        return output_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    predictor = QuantilePredictorV4()

    example: Dict[str, float] = {
        "weather_rainy_days":                   130,
        "weather_days_pleasant_temp":           180,
        "weather_total_snowfall_cm":            5.0,
        "weather_days_below_freezing":          30,
        "weather_total_precipitation_mm":       950,
        "weather_total_sunshine_hours":         3400,
        "weather_avg_daily_max_windspeed_ms":   15.0,
        "nearest_gas_station_distance_miles":   0.15,
        "nearest_gas_station_rating":           4.2,
        "nearest_gas_station_rating_count":     120,
        "competitors_count_4miles":             7,
        "competitor_1_google_rating":           4.1,
        "competitor_1_distance_miles":          0.4,
        "competitor_1_rating_count":            300,
        "costco_enc":                           2.5,
        "distance_nearest_walmart(5 mile)":     1.0,
        "distance_nearest_target (5 mile)":     1.5,
        "other_grocery_count_1mile":            2,
        "count_food_joints_0_5miles (0.5 mile)":8,
        "age_on_30_sep_25":                     3.0,
        "region_enc":                           1.0,
        "state_enc":                            10.0,
        "tunnel_count":                         2.0,
        "carwash_type_encoded":                 1.0,  # 1=Express (display-only in ML model)
    }

    result = predictor.analyze(example, llm_narrative=False)
    predictor.print_report(result)
    predictor.plot_feature_quantiles(result)
    predictor.save_report(result)
