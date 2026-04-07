"""Canonical merged CSV load and engineered columns for the tier model."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from app.modelling.ds.prediction.paths import datasets_dir


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
    """Add competition_quality, gas_station_draw, retail_proximity, weather_drive_score, effective_capacity."""
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
    df_cnt = pd.read_csv(csv_path)

    print(f"  Excel rows: {len(df_feat)}  |  CSV rows: {len(df_cnt)}")

    def parse_addr(a):
        if pd.isna(a):
            return "", "", "", ""
        p = [x.strip() for x in str(a).split(",")]
        if len(p) >= 4:
            return p[0], p[1], p[2], p[3]
        if len(p) == 3:
            return p[0], p[1], p[2], ""
        if len(p) == 2:
            return "", p[0], p[1], ""
        return "", p[0], "", ""

    parsed = df_feat["Address"].apply(parse_addr)
    df_feat["_chain"] = parsed.apply(lambda x: _norm_street(x[0]))
    df_feat["_street"] = parsed.apply(lambda x: _norm_street(x[1]))
    df_feat["_city"] = parsed.apply(lambda x: _norm_street(x[2]))
    df_feat["_zip5"] = parsed.apply(lambda x: _zip5(x[3]))

    df_cnt["_chain"] = df_cnt["client_id"].astype(str).apply(_norm_street)
    df_cnt["_street"] = df_cnt["street"].astype(str).apply(_norm_street)
    df_cnt["_city"] = df_cnt["city"].astype(str).apply(_norm_street)
    df_cnt["_zip5"] = df_cnt["zip"].astype(str).str.zfill(5).str[:5]

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
    cnt_keep = [c for c in cnt_cols if c not in ("_chain", "_street", "_city", "_zip5")]
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
    """Read canonical CSV, costco_enc, optional dim age/region join, engineered features."""
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

    dim_path = datasets_dir() / "dim_site_unified_202603030226.csv"
    if dim_path.exists() and not all(c in merged.columns for c in ("age_on_30_sep_25", "region_enc")):
        dim = pd.read_csv(dim_path, low_memory=False)
        dim["_street"] = dim["street"].apply(_norm_street)
        dim["_zip5"] = dim["zip"].astype(str).str.zfill(5).str[:5]
        dim_keep = dim[["_street", "_zip5", "age_on_30_sep_25", "region", "state"]].drop_duplicates(
            subset=["_street", "_zip5"]
        )
        merged["_street"] = merged["street"].astype(str).apply(_norm_street)
        merged["_zip5"] = merged["zip"].astype(str).str.zfill(5).str[:5]
        merged = merged.merge(dim_keep, on=["_street", "_zip5"], how="left")
        merged = merged.drop(columns=["_street", "_zip5"], errors="ignore")
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
