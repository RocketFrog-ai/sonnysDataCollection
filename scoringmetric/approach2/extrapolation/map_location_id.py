#!/usr/bin/env python3
"""
Map site_client_id + client_id from target quarters CSV to dim_site_unified to get location_id.
Match on site_id (= site_client_id) and client_id (case-insensitive, normalized).
Writes a new CSV with location_id added and prints match accuracy.
"""
import pandas as pd
import re

TARGET_CSV = "_WITH_target_quarters_AS_SELECT_DISTINCT_year_num_quarter_num_FR_202602130556.csv"
DIM_CSV = "dim_site_unified_202603030226.csv"
OUTPUT_CSV = "target_quarters_with_location_id.csv"


def normalize_client(s: str) -> str:
    """Normalize for matching: strip, lower, collapse whitespace."""
    if pd.isna(s) or s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def main():
    target = pd.read_csv(TARGET_CSV)
    dim = pd.read_csv(DIM_CSV)

    # Dim: use client_id if non-empty, else group_name (for chem rows)
    dim["_client"] = dim["client_id"].fillna("")
    empty = (dim["_client"].astype(str).str.strip() == "") | dim["_client"].isna()
    dim.loc[empty, "_client"] = dim.loc[empty, "group_name"].fillna("").astype(str)
    dim["_client_norm"] = dim["_client"].apply(normalize_client)

    # Target
    target["_client_norm"] = target["client_id"].apply(normalize_client)
    target["site_client_id"] = target["site_client_id"].astype(int)

    # Dim site_id as int for matching
    dim["site_id"] = pd.to_numeric(dim["site_id"], errors="coerce")
    dim = dim.dropna(subset=["site_id"])
    dim["site_id"] = dim["site_id"].astype(int)

    # Build lookup: (site_id, client_norm) -> list of (location_id, source) ; prefer site_client
    lookup = {}
    for _, row in dim.iterrows():
        sid = int(row["site_id"])
        cnorm = row["_client_norm"]
        if not cnorm:
            continue
        key = (sid, cnorm)
        loc = row["location_id"]
        src = row.get("source", "")
        if key not in lookup:
            lookup[key] = []
        lookup[key].append((loc, src))

    # Prefer site_client source when multiple (numeric location_id usually)
    for key in lookup:
        entries = lookup[key]
        site_client_entries = [e for e in entries if e[1] == "site_client"]
        if site_client_entries:
            lookup[key] = site_client_entries
        else:
            lookup[key] = entries

    # Match each target row
    location_ids = []
    match_source = []
    for _, row in target.iterrows():
        sid = row["site_client_id"]
        cnorm = row["_client_norm"]
        key = (sid, cnorm)
        loc = None
        src_used = ""
        if key in lookup:
            # Take first (or first site_client already filtered)
            loc = lookup[key][0][0]
            src_used = lookup[key][0][1]
        location_ids.append(loc)
        match_source.append(src_used if loc is not None else "")

    target["location_id"] = location_ids
    target["location_match_source"] = match_source

    # Drop temp columns and put location_id as second column
    out = target.drop(columns=["_client_norm"])
    cols = [c for c in out.columns if c != "location_id"]
    idx = cols.index("site_client_id") + 1
    out = out[cols[:idx] + ["location_id"] + cols[idx:]]

    out.to_csv(OUTPUT_CSV, index=False, float_format="%.2f")

    # Accuracy
    matched = out["location_id"].notna()
    n_total = len(out)
    n_matched = matched.sum()
    n_unmatched = n_total - n_matched
    pct = (n_matched / n_total * 100) if n_total else 0

    print("=== Location ID mapping accuracy ===")
    print(f"Total rows:              {n_total}")
    print(f"Matched (has location_id): {n_matched}")
    print(f"Unmatched:              {n_unmatched}")
    print(f"Match rate:             {pct:.2f}%")
    if n_matched > 0:
        by_source = out.loc[matched, "location_match_source"].value_counts()
        print("Matches by source:")
        for s, c in by_source.items():
            print(f"  {s}: {c}")

    # Sample unmatched for debugging
    if n_unmatched > 0:
        um = out[~matched][["site_client_id", "client_id"]].drop_duplicates().head(10)
        print("\nSample unmatched (site_client_id, client_id):")
        print(um.to_string(index=False))

    print(f"\nWrote: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
