#!/usr/bin/env python3
"""
Extrapolate missing gap data into temp.csv using daily average.
- Load gaps (5).json: key is location_id (inner dict keys), each has gaps[], prev_gaps[], curr_gaps[], client_id.
- Match temp.csv rows by location_id + normalized client_id.
- For 2024 gaps: add (previous_count/360)*gap_days to previous_count.
- For 2025 gaps: add (current_count/360)*gap_days to current_count.
- Recompute yoy_percent_change. Write updated CSV.
"""
import json
import re
from datetime import datetime
from collections import defaultdict

import pandas as pd

GAPS_JSON = "gaps (5).json"
TEMP_CSV = "temp.csv"
OUTPUT_CSV = "temp_extrapolated.csv"
DAYS_PER_YEAR = 360


def normalize_client(s: str) -> str:
    if pd.isna(s) or s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def parse_ts(ts: str) -> datetime:
    try:
        return datetime.fromisoformat(ts.replace("+00:00", "").replace("Z", ""))
    except Exception:
        return None


def main():
    with open(GAPS_JSON) as f:
        gaps_data = json.load(f)

    # Aggregate by (location_id, normalized client_id): (gap_days_2024, gap_days_2025)
    # Top-level keys are report names; inner keys are location_id.
    gap_by_loc_client = defaultdict(lambda: [0.0, 0.0])  # [gap_2024, gap_2025]

    for top_key, inner in gaps_data.items():
        if not isinstance(inner, dict):
            continue
        for loc_id, rec in inner.items():
            if not isinstance(rec, dict) or "gaps" not in rec or "prev_gaps" not in rec:
                continue
            gaps_list = rec.get("gaps") or []
            prev_gaps = rec.get("prev_gaps") or []
            client_id = rec.get("client_id") or ""
            if len(gaps_list) != len(prev_gaps):
                continue
            client_norm = normalize_client(client_id)
            key = (str(loc_id).strip(), client_norm)
            g_2024_add, g_2025_add = 0.0, 0.0
            for i, days in enumerate(gaps_list):
                if i >= len(prev_gaps):
                    break
                prev_ts = parse_ts(prev_gaps[i])
                if prev_ts is None:
                    continue
                year = prev_ts.year
                try:
                    d = float(days)
                except (TypeError, ValueError):
                    d = 0
                if year == 2024:
                    g_2024_add += d
                elif year == 2025:
                    g_2025_add += d
            # Accumulate across top-level keys (same location_id + client_id in multiple reports)
            g_2024, g_2025 = gap_by_loc_client[key]
            gap_by_loc_client[key] = [g_2024 + g_2024_add, g_2025 + g_2025_add]

    df = pd.read_csv(TEMP_CSV)
    df["_client_norm"] = df["client_id"].apply(normalize_client)
    df["location_id"] = df["location_id"].astype(str).str.strip()

    current_count = pd.to_numeric(df["current_count"], errors="coerce").fillna(0)
    previous_count = pd.to_numeric(df["previous_count"], errors="coerce").fillna(0)

    new_current = current_count.values.copy()
    new_previous = previous_count.values.copy()
    matched_count = 0
    applied = []

    for idx, row in df.iterrows():
        loc_id = row["location_id"]
        client_norm = row["_client_norm"]
        key = (loc_id, client_norm)
        if key not in gap_by_loc_client:
            continue
        g_2024, g_2025 = gap_by_loc_client[key]
        if g_2024 == 0 and g_2025 == 0:
            continue
        matched_count += 1
        prev = float(previous_count.iloc[idx])
        curr = float(current_count.iloc[idx])
        # Extrapolate: count + (count/360)*gap_days = count * (1 + gap_days/360)
        if g_2024 > 0 and prev > 0:
            new_previous[idx] = prev * (1 + g_2024 / DAYS_PER_YEAR)
        if g_2025 > 0 and curr > 0:
            new_current[idx] = curr * (1 + g_2025 / DAYS_PER_YEAR)
        applied.append(
            {
                "location_id": loc_id,
                "client_id": row["client_id"],
                "gap_days_2024": g_2024,
                "gap_days_2025": g_2025,
                "orig_prev": prev,
                "orig_curr": curr,
                "new_prev": new_previous[idx],
                "new_curr": new_current[idx],
            }
        )

    df["current_count"] = new_current
    df["previous_count"] = new_previous

    # Recompute yoy_percent_change
    def yoy(row):
        p = row["previous_count"]
        c = row["current_count"]
        if p == 0:
            return None
        return round(((c - p) / p) * 100, 2)

    df["yoy_percent_change"] = df.apply(yoy, axis=1)
    df = df.drop(columns=["_client_norm"])

    df.to_csv(OUTPUT_CSV, index=False, float_format="%.2f")

    print("=== Gap extrapolation ===")
    print(f"Total rows in temp.csv:     {len(df)}")
    print(f"Rows matched with gaps:     {matched_count}")
    print(f"Unique (location_id, client) in gaps: {len(gap_by_loc_client)}")
    if applied:
        print(f"\nSample applied (first 5):")
        for a in applied[:5]:
            print(f"  loc={a['location_id']} client={a['client_id'][:25]} gap_2024={a['gap_days_2024']:.0f} gap_2025={a['gap_days_2025']:.0f} -> prev {a['orig_prev']:.0f}->{a['new_prev']:.0f} curr {a['orig_curr']:.0f}->{a['new_curr']:.0f}")
    print(f"\nWrote: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
