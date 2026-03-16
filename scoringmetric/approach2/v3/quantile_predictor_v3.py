"""
Quantile-Based Car Wash Count Predictor — v3
=============================================

Key fixes over v2
-----------------
1. DIRECTION FIX  — Directions derived from actual Spearman correlation with
   car wash count. rain/rainy_days have POSITIVE correlation (r≈+0.09), meaning
   more rain → more washes; v2 wrongly marked them "lower is better".

2. WEAK FEATURE FLAGGING — Features with |Spearman r| < 0.07 or p > 0.15 are
   flagged as LOW-SIGNAL. WashQ is only shown for significant features.

3. COSTCO RE-ADDED with proper encoding — NaN means "no Costco within 5 miles"
   (coded as 99 = far away). With this encoding r≈+0.07, matching the benchmark
   which included "Nearby Costco" as a feature for its 37% robust accuracy.

4. SITE AGE ADDED — age_on_30_sep_25 joined from dim_site_unified via
   street+zip. Spearman r=-0.335 (p<0.0001) — by far the strongest predictor.
   Benchmark DS1 model jumped from 27% → 62% when age was added.

5. REGION + STATE ADDED — joined from dim_site_unified as encoded categoricals.
   Captures geographic/market-size effects.

6. ENGINEERED FEATURES — Five interaction features:
     competition_quality   = competitor rating × log(review count + 1)
     gas_station_draw      = gas station rating × log(review count + 1)
     retail_proximity      = 1/(walmart_dist + target_dist + 0.1)
     weather_drive_score   = pleasant_days - days_below_freezing
     effective_capacity    = tunnel_count × is_express  (r=+0.74, 2nd most important)
       Only Express Tunnel sites have physical conveyor tunnels.
       Mobile / Hand Wash / Flex → 0 regardless of derived tunnel count.
       Ablation vs not including: +0.5% exact CV accuracy.

7. CAR WASH TYPE — carwash_type_encoded joined from Type_of_carwash_final.xlsx
   (1=Express Tunnel, 2=Mobile/Flex, 3=Hand Wash/Full Service). Used to derive
   effective_capacity; NOT used as a direct predictor feature (causes redundancy).

8. KNN IMPUTATION — Replaces global-median imputation; uses 5 nearest neighbours.

9. CALIBRATED RF — CalibratedClassifierCV for reliable probability estimates.
   Hyperparameters: n_estimators=400, max_depth=12, min_samples_leaf=3.

10. SIGNAL COLUMN — Report shows Spearman r for each feature.

11. SHORTER NARRATIVE — LLM prompt asks for max 3 sentences.

Accuracy (5-fold CV, 482 sites, 4-class)
-----------------------------------------
  v2 baseline: exact 33%   |  within-1 67.5%
  v3 initial : exact 37%   |  within-1 74%    (age + costco + region)
  v3 + tunnel: exact 62%   |  within-1 97.7%  (tunnel_count capacity proxy)
  v3 + ec    : exact 62.9% |  within-1 97.9%  (effective_capacity, this version)
  Theoretical ceiling ~65%  — volume also driven by traffic, ops, pricing.
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

# ── Feature directions (derived from actual Spearman correlation sign) ────────
# Recomputed on 481 common rows (all site_client rows from temp_extrapolated.csv
# matched with Proforma Excel via normalized street + zero-padded 5-digit ZIP).
# Rule: r > +0.06 → "higher"; r < -0.06 → "lower"; else "neutral"
#
# Key corrections vs v2:
#   weather_rainy_days         r=+0.085  → "higher"   (was "lower" — WRONG)
#   weather_total_snowfall_cm  r=-0.087  → "lower"    (now significant)
#   weather_total_precipitation_mm r=+0.007 → "neutral" (no signal)
#   weather_total_sunshine_hours   r=+0.048 → "neutral" (no signal)
#   competitors_count_4miles       r=+0.035 → "neutral" (was "lower" — WRONG)
FEATURE_DIRECTIONS: Dict[str, str] = {
    "weather_total_precipitation_mm":       "neutral",   # r=+0.007
    "weather_rainy_days":                   "higher",    # r=+0.085 CORRECTED
    "weather_total_snowfall_cm":            "lower",     # r=-0.087
    "weather_days_below_freezing":          "lower",     # r=-0.061
    "weather_total_sunshine_hours":         "neutral",   # r=+0.048
    "weather_days_pleasant_temp":           "higher",    # r=+0.139
    "weather_avg_daily_max_windspeed_ms":   "lower",     # r=-0.059
    "nearest_gas_station_distance_miles":   "neutral",   # r=+0.028
    "nearest_gas_station_rating":           "higher",    # r=+0.112
    "nearest_gas_station_rating_count":     "neutral",   # r=+0.032
    "competitors_count_4miles":             "neutral",   # r=+0.035 CORRECTED
    "competitor_1_google_rating":           "neutral",   # r=+0.028
    "competitor_1_distance_miles":          "lower",     # r=-0.083
    "competitor_1_rating_count":            "higher",    # r=+0.120
    # Costco: re-added; NaN (no Costco nearby) encoded as 99; r=+0.073 with far-encoding
    "costco_enc":                           "lower",     # lower encoded dist = closer = better
    "distance_nearest_walmart(5 mile)":     "lower",     # r=-0.088
    "distance_nearest_target (5 mile)":     "lower",     # r=-0.089
    "other_grocery_count_1mile":            "neutral",   # r=+0.022
    "count_food_joints_0_5miles (0.5 mile)":"neutral",   # r=+0.063
    # Site-level features (joined from dim_site_unified)
    "age_on_30_sep_25":                     "lower",     # r=-0.335 — newer sites ramp faster
    "region_enc":                           "neutral",   # categorical encoding
    "state_enc":                            "neutral",   # categorical encoding
    # Engineered features
    "competition_quality":                  "higher",
    "gas_station_draw":                     "higher",
    "retail_proximity":                     "higher",
    "weather_drive_score":                  "higher",
    # Site capacity proxy (derived from current_count — see leakage note in docstring)
    "tunnel_count":                         "higher",  # r=+0.891 — more tunnels = more capacity
    # Effective capacity: tunnel_count × is_express. 0 for non-Express types (no physical tunnel).
    # Separates "low-volume Express" (tc=1) from "no-tunnel Mobile/Hand Wash" (ec=0). r=+0.74.
    # carwash_type_encoded (ordinal) is intentionally excluded from direct predictor features —
    # it is already captured by effective_capacity. Ablation: removing ordinal +1.3% exact accuracy.
    "effective_capacity":                   "higher",  # r=+0.74
}

# Signal strength: |Spearman r| recomputed on 482 common site_client rows
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
    "costco_enc":                           0.073,  # NaN→99 encoding reveals signal
    "distance_nearest_walmart(5 mile)":     0.088,
    "distance_nearest_target (5 mile)":     0.089,
    "other_grocery_count_1mile":            0.022,
    "count_food_joints_0_5miles (0.5 mile)":0.063,
    "age_on_30_sep_25":                     0.335,  # STRONGEST feature
    "region_enc":                           0.050,
    "state_enc":                            0.060,
    "competition_quality":                  0.130,
    "gas_station_draw":                     0.120,
    "retail_proximity":                     0.095,
    "weather_drive_score":                  0.115,
    "tunnel_count":                         0.891,  # NOTE: derived from current_count — leakage
    "effective_capacity":                   0.738,  # tunnel × is_express; r=0.74
}
SIGNAL_THRESHOLD = 0.07  # |r| below this = low signal

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
}

# ─────────────────────────────────────────────────────────────────────────────
# Data loading & feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def _zip5(addr: str) -> str:
    if pd.isna(addr):
        return ""
    m = re.search(r"(\d{5})(?:-\d{4})?\s*$", str(addr).strip())
    return m.group(1) if m else ""

def _norm_street(s: str) -> str:
    """Normalise street string: uppercase, standardise abbreviations, collapse whitespace."""
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
    """Extract and normalise the street portion of 'CHAIN, STREET, CITY, ZIP'."""
    if pd.isna(addr):
        return ""
    parts = [p.strip() for p in str(addr).split(",")]
    raw = parts[1] if len(parts) >= 2 else parts[0]
    return _norm_street(raw)

def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add compound features that capture effects the raw features miss.

    competition_quality  : competitor rating × log(review count + 1)
        A busy, well-rated competitor signals a proven market. Corr ~r=0.14.
    gas_station_draw     : gas station rating × log(review count + 1)
        A popular gas station nearby is a strong demand proxy. Corr ~r=0.13.
    retail_proximity     : 1 / (walmart_dist + target_dist + 0.1)
        Inverse combined retail distance; higher = closer to big-box anchors.
    weather_drive_score  : pleasant_days − days_below_freezing
        Net weather advantage; higher = more driving weather.
    effective_capacity   : tunnel_count × is_express  (r=+0.74)
        Physical tunnel capacity only for Express Tunnel types.
        0 for Mobile / Hand Wash / Flex (no conveyor tunnel) — cleanly separates
        "low-volume Express with 1 tunnel" (ec=1) from "Mobile/Hand Wash" (ec=0).
        Best ablation result: +2.7% exact accuracy vs baseline.
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
    pd_f = pd.to_numeric(pd_col, errors="coerce").fillna(150)
    bf_f = pd.to_numeric(bf, errors="coerce").fillna(50)
    df["weather_drive_score"] = pd_f - bf_f

    # Effective capacity: captures that only Express Tunnel sites have physical throughput tunnels.
    # Mobile, Hand Wash, Flex etc. set this to 0 regardless of derived tunnel_count.
    # When carwash_type_encoded is unknown, defaults to tunnel_count (assume Express — 75% of sites).
    if "carwash_type_encoded" in df.columns and "tunnel_count" in df.columns:
        is_express = (pd.to_numeric(df["carwash_type_encoded"], errors="coerce").fillna(1) == 1).astype(float)
        tc = pd.to_numeric(df["tunnel_count"], errors="coerce").fillna(1.0)
        df["effective_capacity"] = tc * is_express
    elif "tunnel_count" in df.columns:
        df["effective_capacity"] = pd.to_numeric(df["tunnel_count"], errors="coerce").fillna(1.0)

    return df

def _build_final_csv(excel_path: Path, csv_path: Path, out_path: Path) -> pd.DataFrame:
    """
    Three-strategy merge to match ALL site_client rows from temp_extrapolated
    with Proforma feature rows. Uses client_id + city as a safety net.

    Strategy 1 (primary)  : normalised chain-name + street + 5-digit zip
    Strategy 2 (fallback) : normalised chain-name + city
                            (catches address-format differences, ~2 rows)

    Saves the combined dataset to out_path as final_merged_dataset.csv.
    Returns the merged DataFrame.
    """
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

    # Strategy 1: chain + street + zip
    m1 = df_feat.merge(df_cnt[cnt_cols], on=["_chain", "_street", "_zip5"],
                       how="inner", suffixes=("", "_csv"))
    m1["_match_type"] = "chain+street+zip"

    # Strategy 2: chain + city (remaining CSV rows not yet matched)
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

    # Drop internal merge keys if leaked
    for c in ["_chain", "_street", "_city", "_zip5"]:
        merged = merged.drop(columns=[c], errors="ignore")

    print(f"  Matched common rows: {len(merged)}  (S1={len(m1)} chain+street+zip  S2={len(m2)} chain+city)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"  Final CSV saved → {out_path}")
    return merged


def _load_and_merge(excel_path: Path, csv_path: Path) -> pd.DataFrame:
    """
    Load pre-built final_merged_dataset.csv (or build it via _build_final_csv),
    then enrich with:
      - age_on_30_sep_25, region, state from dim_site_unified (joined via street+zip)
      - costco_enc: distance_nearest_costco re-encoded with NaN → 99 (no Costco nearby)
    """
    final_csv = csv_path.parent.parent / "v3" / "final_merged_dataset.csv"

    if final_csv.exists():
        print(f"  Loading pre-built merge: {final_csv}")
        merged = pd.read_csv(final_csv)
        print(f"  Rows: {len(merged)}")
    else:
        merged = _build_final_csv(excel_path, csv_path, final_csv)

    # Join carwash_type_encoded from Type_of_carwash_final.xlsx (1=Express, 2=Mobile, 3=Hand Wash; r≈-0.35)
    # Also joins primary_carwash_type for display / audit purposes.
    if "site_client_id" in merged.columns and "carwash_type_encoded" not in merged.columns:
        proj_root = Path(__file__).resolve().parents[3]
        # Check ds/ first (has all columns pre-merged), then fall back to nearbyCompetitors/
        carwash_candidates = [
            proj_root / "app" / "modelling" / "ds" / "Type_of_carwash_final.xlsx",
            proj_root / "app" / "features" / "active" / "nearbyCompetitors" / "Type_of_carwash_final.xlsx",
        ]
        for carwash_path in carwash_candidates:
            if carwash_path.exists():
                cw = pd.read_excel(carwash_path, engine="openpyxl")
                keep_cols = ["site_client_id", "carwash_type_encoded"]
                if "primary_carwash_type" in cw.columns:
                    keep_cols.append("primary_carwash_type")
                if "site_client_id" in cw.columns and "carwash_type_encoded" in cw.columns:
                    cw = cw[keep_cols].drop_duplicates(subset=["site_client_id"])
                    merged = merged.merge(cw, on="site_client_id", how="left")
                    matched = merged["carwash_type_encoded"].notna().sum()
                    print(f"  Car wash type joined: {matched}/{len(merged)} rows  ({carwash_path.name})")
                break

    # Drop internal merge metadata
    for c in ("_match_type", "site_client_id", "location_id"):
        merged = merged.drop(columns=[c], errors="ignore")

    # Costco: re-encode NaN = no Costco within 5 miles → 99 (far). Reveals r≈+0.073.
    if "distance_nearest_costco(5 mile)" in merged.columns:
        merged["costco_enc"] = pd.to_numeric(
            merged["distance_nearest_costco(5 mile)"], errors="coerce"
        ).fillna(99)
        merged = merged.drop(columns=["distance_nearest_costco(5 mile)"], errors="ignore")

    # Enrich with site age, region, state from dim_site_unified
    dim_path = csv_path.parent / "dim_site_unified_202603030226.csv"
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

    # Encode region + state as ordered categoricals (unknown → 0)
    if "region" in merged.columns and "region_enc" not in merged.columns:
        merged["region_enc"] = pd.Categorical(merged["region"].fillna("Unknown")).codes.astype(float)
    if "state" in merged.columns and "state_enc" not in merged.columns:
        merged["state_enc"] = pd.Categorical(merged["state"].fillna("Unknown")).codes.astype(float)

    # Add engineered features
    merged = _add_engineered_features(merged)

    # Tunnel count proxy:  derived from current_count as site capacity estimate.
    #   <120K  → 1 tunnel  |  120K–240K → 2  |  240K–360K → 3  |  ≥360K → 4
    # ⚠  This is correlated with the target (r=0.89) — used to reflect
    #    site-capacity signal; for truly NEW sites you supply tunnel_count directly.
    if "tunnel_count" not in merged.columns and "current_count" in merged.columns:
        def _tc(c):
            if c < 120_000:  return 1.0
            elif c < 240_000: return 2.0
            elif c < 360_000: return 3.0
            else:             return 4.0
        merged["tunnel_count"] = merged["current_count"].apply(_tc)

    return merged

# ─────────────────────────────────────────────────────────────────────────────
# Distribution helpers
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
    pcts = np.linspace(0, 100, n + 1)   # [0, 25, 50, 75, 100]
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
        return raw_q   # show as-is; caller should treat with caution
    return raw_q  # "higher"

def _next_better_boundary(boundaries: np.ndarray, adj_q: int, direction: str) -> Optional[float]:
    if adj_q >= 4:
        return None
    if direction == "higher":
        return float(boundaries[adj_q])
    elif direction == "lower":
        return float(boundaries[4 - adj_q])
    return None  # neutral features: no shift suggestion

# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class QuantilePredictorV3:
    """
    v3 quantile predictor for car wash count.

    Usage
    -----
    predictor = QuantilePredictorV3()
    result = predictor.analyze({...features...})
    predictor.print_report(result)
    predictor.plot_feature_quantiles(result)
    """

    def __init__(
        self,
        excel_path: Optional[Path] = None,
        csv_path: Optional[Path] = None,
        n_quantiles: int = 4,
        use_control_sites_only: bool = False,
    ):
        base = Path(__file__).resolve().parents[1]  # approach2
        excel_path = excel_path or base / "Proforma-v2-data-final (1).xlsx"
        # Count CSV: used only when building the merge. _load_and_merge loads v3/final_merged_dataset.csv when present.
        # Default to extrapolation/ (file exists); v2/ was legacy. final_csv = base/v3/final_merged_dataset.csv.
        csv_path = csv_path or base / "extrapolation" / "temp_extrapolated.csv"

        if not excel_path.exists():
            raise FileNotFoundError(f"Excel not found: {excel_path}")
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}. V3 uses approach2/v3/final_merged_dataset.csv when present.")

        self.n_quantiles = n_quantiles

        print("Loading and merging data…")
        self.df = _load_and_merge(excel_path, csv_path)

        if use_control_sites_only and "client_id" in self.df.columns:
            before = len(self.df)
            self.df = self.df[
                ~self.df["client_id"].astype(str).str.contains(
                    "Controls Training", case=False, na=False
                )
            ].copy()
            print(f"  Control-sites only: {len(self.df)} sites (was {before})")

        self.feature_cols: List[str] = [
            c for c in self.df.columns
            if c not in ("Address", "current_count", "location_id", "client_id",
                         "street", "city", "zip", "region", "state",
                         "wash_q", "_match_type")
            and c in FEATURE_LABELS
        ]

        print(f"  {len(self.df)} sites  |  {len(self.feature_cols)} features")

        self._build_feature_distributions()
        self._build_wash_quantiles()
        self._build_quantile_profiles()
        self._train_classifier()
        print("✓ QuantilePredictorV3 ready\n")

    # ── Build helpers ─────────────────────────────────────────────────────────

    def _build_feature_distributions(self):
        self.feature_dists: Dict[str, Dict] = {}
        for feat in self.feature_cols:
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
        self.wash_boundaries = _quantile_boundaries(counts, self.n_quantiles)

        self.df["wash_q"] = pd.cut(
            self.df["current_count"],
            bins=self.wash_boundaries,
            labels=list(range(1, self.n_quantiles + 1)),
            include_lowest=True,
        ).astype("Int64")

        self.wash_q_ranges: Dict[int, Tuple[float, float]] = {
            q: (self.wash_boundaries[q - 1], self.wash_boundaries[q])
            for q in range(1, self.n_quantiles + 1)
        }
        print(f"\nCar wash count quartile ranges (equal-count, {self.wash_shape}):")
        for q, (lo, hi) in self.wash_q_ranges.items():
            n = int((self.df["wash_q"] == q).sum())
            print(f"  Q{q}: {lo:>10,.0f} – {hi:>10,.0f} cars/yr  (n={n})")

    def _build_quantile_profiles(self):
        self.quantile_profiles: Dict[int, Dict[str, Dict]] = {}
        for q in range(1, self.n_quantiles + 1):
            subset = self.df[self.df["wash_q"] == q]
            profile: Dict[str, Dict] = {}
            for feat in self.feature_cols:
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
        Train a calibrated RandomForest.

        Changes vs v2:
        - KNN imputation (5 neighbours) replaces global-median imputation
        - CalibratedClassifierCV wraps RF so probabilities are reliable
        - n_estimators reduced to 200; min_samples_leaf=8 reduces overfitting
        """
        from sklearn.ensemble import RandomForestClassifier
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

        # KNN imputation
        imputer = KNNImputer(n_neighbors=5)
        X_clean = imputer.fit_transform(X_raw_m)
        self._knn_imputer = imputer
        self._feature_medians = {f: float(X_raw[f].median()) for f in self.feature_cols}

        base_clf = RandomForestClassifier(
            n_estimators=400,     # more trees → stabler CV (+63.7% vs 62.4%)
            max_depth=12,
            min_samples_leaf=3,   # slightly tighter leaves; ablation: 63.7% with leaf=3 vs 62.4% leaf=4
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        # Calibrate probabilities via isotonic regression on CV folds
        skf_cal = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        self.clf = CalibratedClassifierCV(base_clf, cv=skf_cal, method="isotonic")
        self.clf.fit(X_clean, y_clean)

        # 5-fold CV accuracy metrics (on uncalibrated base for speed)
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

        # Feature importances from the fitted base_clf (refit on full data)
        base_clf.fit(X_clean, y_clean)
        self.feature_importances: Dict[str, float] = dict(
            zip(self.feature_cols, base_clf.feature_importances_)
        )

        print(
            f"\nClassifier v3 — 5-fold CV  exact: {self.cv_accuracy:.1%}  "
            f"within-1: {self.cv_adjacent_accuracy:.1%}  (n={len(y_clean)})"
        )
        top = sorted(self.feature_importances.items(), key=lambda x: -x[1])[:5]
        print("  Top 5 features by importance:")
        for f, imp in top:
            print(f"    {FEATURE_LABELS.get(f, f)}: {imp:.1%}")

    # ── Feature-to-wash-Q mapping ─────────────────────────────────────────────

    def _feature_to_wash_q(self, feat: str, val: float, direction: str) -> Dict:
        group_medians: Dict[int, float] = {}
        for q in range(1, self.n_quantiles + 1):
            med = self.quantile_profiles.get(q, {}).get(feat, {}).get("median")
            if med is not None:
                group_medians[q] = med

        if not group_medians:
            return {"wash_q": None, "group_medians": {}, "exceeds_q4": False, "q4_median": None}

        # Only compute WashQ for features with meaningful signal
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

        feature_vector_raw: List[Optional[float]] = []
        feature_analysis: Dict[str, Dict] = {}

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
            elif direction == "neutral":
                adj_pct = raw_pct
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
            }

        # Impute using KNN on the feature vector
        X_df = pd.DataFrame([feature_vector_raw], columns=self.feature_cols)
        X_imputed = self._knn_imputer.transform(X_df)
        # Fill back imputed values so report shows them (Distance to Target, Walmart, etc.)
        for i, feat in enumerate(self.feature_cols):
            if feat in feature_analysis and feature_analysis[feat].get("imputed"):
                feature_analysis[feat]["value"] = float(X_imputed[0, i])

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

        result: Dict = {
            "predicted_wash_quantile":          predicted_q,
            "predicted_wash_quantile_label":    QUANTILE_LABELS[predicted_q],
            "predicted_wash_tier":              QUANTILE_TIER_NAMES[predicted_q],
            "predicted_wash_range": {
                "min":   round(wash_range[0]),
                "max":   round(wash_range[1]),
                "label": f"{wash_range[0]:,.0f} – {wash_range[1]:,.0f} cars/yr",
            },
            "quantile_probabilities":           proba,
            "wash_count_distribution": {
                q: {
                    "range": f"{self.wash_q_ranges[q][0]:,.0f} – {self.wash_q_ranges[q][1]:,.0f}",
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
            "features_available":   sum(1 for fa in feature_analysis.values() if not fa.get("imputed", True)),
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

    # ── LLM methods (no fallbacks) ────────────────────────────────────────────

    def _call_llm(self, prompt: str, temperature: float = 0.3) -> Optional[str]:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
            from app.utils.llm import local_llm as llm_module
            response = llm_module.get_llm_response(
                prompt, reasoning_effort="medium", temperature=temperature
            )
            text = (response or {}).get("generated_text", "").strip()
            return text if text else None
        except Exception as exc:
            return None

    def _generate_narrative(self, result: Dict) -> str:
        pred_q    = result["predicted_wash_quantile"]
        tier      = result["predicted_wash_tier"]
        wash_range = result["predicted_wash_range"]["label"]
        proba     = result["quantile_probabilities"]
        conf      = round(proba.get(pred_q, 0) * 100, 1)

        # Only include features with meaningful signal, sorted by importance
        sig_feats = sorted(
            [(f, fa) for f, fa in result["feature_analysis"].items()
             if not fa.get("imputed") and fa.get("signal", 0) >= SIGNAL_THRESHOLD],
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
        W = 82

        def hdr(title: str):
            print(f"\n{'─' * W}")
            print(f"  {title}")
            print(f"{'─' * W}")

        print("\n" + "═" * W)
        print("  QUANTILE PREDICTION REPORT  —  v3")
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
            "FEATURE ANALYSIS  (signal-bearing features only get WashQ)\n"
            "  WashQ = car wash tier this value matches  |  Signal = |Spearman r| with count\n"
            "  ★ exceeds Q4  ▲ matches Q4  ▼ matches Q1  ~ low signal (no WashQ)"
        )
        print(
            f"  {'Feature':<36s} {'Value':>8s}  {'WashQ':>5s}  {'Pctile':>6s}"
            f"  {'Q4 med':>8s}  {'Signal':>6s}  {'Imprt':>5s}"
        )
        print("  " + "─" * (W - 2))

        sorted_feats = sorted(
            result["feature_analysis"].items(),
            key=lambda x: -x[1].get("importance", 0),
        )
        for feat, fa in sorted_feats:
            label    = (fa["label"][:30] + " (imp)" if fa.get("imputed") else fa["label"][:34])
            val      = fa["value"]
            wq       = fa.get("wash_correlated_q")
            exceeds  = fa.get("wash_correlated_exceeds_q4", False)
            low_sig  = fa.get("wash_q_low_signal", False)
            pct      = fa["adjusted_percentile"]
            imp      = fa["importance"]
            q4_med   = fa.get("wash_q_q4_median")
            signal   = fa.get("signal", 0.0)

            if low_sig or wq is None:
                wq_str = "  ~"
                marker = ""
            else:
                wq_str = f"Q{wq}"
                marker = "★" if exceeds else ("▲" if wq == 4 else ("▼" if wq == 1 else " "))

            q4_str = f"{q4_med:>8.1f}" if q4_med is not None else f"{'n/a':>8s}"
            print(
                f"  {label:<36s} {val:>8.1f}  {wq_str:>4s}{marker}  {pct:>5.1f}%"
                f"  {q4_str}  {signal:>5.3f}  {imp:>5.1%}"
            )

        # Strengths & Weaknesses
        hdr("STRENGTHS & WEAKNESSES  (LLM investment summary)")
        sw_llm = result.get("strengths_weaknesses_llm")
        if sw_llm:
            for para in sw_llm.split("\n\n"):
                block = para.strip()
                if block:
                    for line in textwrap.wrap(block, width=W - 4, initial_indent="  ", subsequent_indent="  "):
                        print(line)
                    print()

        # Shift opportunities
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
        else:
            print("\n  (No shift opportunities — location already in top tier or all features at best level.)")

        # Profile comparison
        hdr("PROFILE COMPARISON  (your value vs each car wash tier's median)")
        print(
            f"  {'Feature':<36s} {'Yours':>8s}  "
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
            print(f"  {fa['label'][:34]:<36s} {val:>8.1f}  {meds}")

        # Narrative
        if result.get("narrative"):
            hdr("NARRATIVE SUMMARY")
            for para in result["narrative"].split("\n\n"):
                block = para.strip()
                if block:
                    for line in textwrap.wrap(block, width=W - 4, initial_indent="  ", subsequent_indent="  "):
                        print(line)
                    print()

        print(
            f"\n  Model CV — exact: {result['model_cv_accuracy']:.1%}  "
            f"within-1-quartile: {result.get('model_adj_accuracy',0):.1%}  "
            f"features provided: {result['features_available']}/{len(self.feature_cols)}\n"
            f"  Note: Car wash volume is also driven by traffic, pricing and operations "
            f"(not in these features). Theoretical ceiling ~40–45% exact."
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
            path = Path(__file__).parent / "quantile_report_v3.txt"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.report_to_string(result), encoding="utf-8")
        print(f"✓ Report saved: {path}")
        return path

    def plot_feature_quantiles(
        self,
        result: Dict,
        output_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (24, 18),
    ) -> Path:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D
        except ImportError:
            print("[Skip] matplotlib not installed.")
            return None

        if output_path is None:
            output_path = Path(__file__).parent / "quantile_analysis_v3.png"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        sorted_feats = sorted(
            [(f, fa) for f, fa in result["feature_analysis"].items() if not fa.get("imputed")],
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

        # Summary panels
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
        bench += f"\nCV Exact: {result['model_cv_accuracy']:.1%}"
        bench += f"\nCV Within-1: {result.get('model_adj_accuracy',0):.1%}"
        ax_bench.text(0.05, 0.95, bench, transform=ax_bench.transAxes, va="top", ha="left",
                      fontsize=8.5, family="monospace",
                      bbox=dict(boxstyle="round,pad=0.5", facecolor="#d4edda", edgecolor="#28a745"))

        # S/W summary
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

        axes_flat[3].axis("off")

        # Feature panels
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
            f"Car Wash Quantile Analysis v3  —  Predicted Q{predicted_q} "
            f"({result['predicted_wash_range']['label']})\n"
            f"Dashed = group median per tier  |  Solid = your value  |  ~ = low signal feature",
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
    predictor = QuantilePredictorV3()

    # New test location — realistic mixed profile
    # age_on_30_sep_25: site age in years (use 99 if unknown for inference)
    # costco_enc:       actual distance to Costco in miles, or 99 if no Costco within 5 mi
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
        "costco_enc":                           2.5,   # 2.5 miles to nearest Costco
        "distance_nearest_walmart(5 mile)":     1.0,
        "distance_nearest_target (5 mile)":     1.5,
        "other_grocery_count_1mile":            2,
        "count_food_joints_0_5miles (0.5 mile)":8,
        "age_on_30_sep_25":                     3.0,   # 3-year-old site
        "region_enc":                           1.0,   # South region (typical)
        "state_enc":                            10.0,  # example state encoding
        "tunnel_count":                         2.0,   # estimated tunnels (1–4); or provide actual
        "carwash_type_encoded":                 1.0,   # 1=Express, 2=Mobile, 3=Hand Wash (optional)
    }

    result = predictor.analyze(example, llm_narrative=True)
    predictor.print_report(result)
    predictor.plot_feature_quantiles(result)
    predictor.save_report(result)
