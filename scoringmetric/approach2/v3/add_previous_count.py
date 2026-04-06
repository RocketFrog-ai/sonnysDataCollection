#!/usr/bin/env python3
"""
Join previous_count from extrapolation/temp_extrapolated.csv into final_merged_dataset.csv.

Match key: site_client_id (1:1; current_count values align on both files).
Inserts previous_count immediately after current_count.
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FINAL_CSV = ROOT / "final_merged_dataset.csv"
TEMP_CSV = ROOT.parent / "extrapolation" / "temp_extrapolated.csv"


def main():
    with open(TEMP_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        prev_by_site = {row["site_client_id"]: row["previous_count"] for row in reader}

    with open(FINAL_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        if "previous_count" in fieldnames:
            raise SystemExit("previous_count already present; aborting.")
        idx = fieldnames.index("current_count") + 1
        fieldnames.insert(idx, "previous_count")
        rows = list(reader)

    missing = [r["site_client_id"] for r in rows if r["site_client_id"] not in prev_by_site]
    if missing:
        raise SystemExit(f"Missing previous_count for site_client_id(s): {missing[:5]} ...")

    out_rows = []
    for row in rows:
        row = dict(row)
        row["previous_count"] = prev_by_site[row["site_client_id"]]
        out_rows.append(row)

    with open(FINAL_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    print(f"Wrote {len(out_rows)} rows with previous_count -> {FINAL_CSV}")


if __name__ == "__main__":
    main()
