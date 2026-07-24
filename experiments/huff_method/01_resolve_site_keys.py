"""
Step 1 of the Huff-model backtest pipeline: resolve MANIFEST.csv's (match_client_name,
match_state) pairs -- from experiments/old-proforma-analysis/n70-final-considered/MANIFEST.csv --
to an exact (client_id, site_id) in the real operating-site panel (proforma.pnl.data.load_panel()).

MANIFEST.csv already tells you WHICH real site each of the 70 curated proformas matched to (by
name/state) and its actual mature wash volume + a competitor-density "grounding" tier -- but
name+state isn't a precise key, and has no lat/lon. This script resolves that precisely, which is
what step 2 (the Huff calibration) needs to do a real neighbour search.

Run from the repo root: python experiments/huff-model-backtest/01_resolve_site_keys.py
Output: experiments/huff-model-backtest/resolved_sites.json (69 unique + 1 flagged ambiguous)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from proforma.pnl import data as D  # noqa: E402

MANIFEST = REPO_ROOT / "experiments" / "old-proforma-analysis" / "n70-final-considered" / "MANIFEST.csv"
OUT_JSON = Path(__file__).resolve().parent / "resolved_sites.json"


def norm(s) -> str:
    if pd.isna(s):
        return ""
    s = re.sub(r"[^A-Z0-9 ]+", " ", str(s).upper())
    return re.sub(r"\s+", " ", s).strip()


def main():
    manifest = pd.read_csv(MANIFEST)
    _, site = D.load_panel()

    site = site.copy()
    site["_name_norm"] = site["client_name"].map(norm)
    manifest["_name_norm"] = manifest["match_client_name"].map(norm)

    rows = []
    for _, r in manifest.iterrows():
        cands = site[(site["_name_norm"] == r["_name_norm"]) & (site["state"] == r["match_state"])]
        if len(cands) == 1:
            c = cands.iloc[0]
            status = "unique"
            site_key, lat, lon, op_start, n_obs = c.site_key, c.lat, c.lon, c.op_start, c.n_obs
        elif len(cands) > 1:
            status = f"ambiguous({len(cands)})"
            site_key = lat = lon = op_start = n_obs = None
        else:
            status = "no_match"
            site_key = lat = lon = op_start = n_obs = None
        rows.append({
            "source_file": r["source_file"], "match_client_name": r["match_client_name"],
            "match_state": r["match_state"], "status": status, "site_key": site_key,
            "lat": lat, "lon": lon, "op_start": op_start, "n_obs": n_obs,
            "actual_mature_wash_mo": r["actual_mature_wash_mo"], "grounding": r["grounding"],
        })

    out = pd.DataFrame(rows)
    print("=== Status counts ===")
    print(out["status"].value_counts())
    print("\n=== Non-unique rows (need manual resolution) ===")
    print(out[out["status"] != "unique"][["source_file", "match_client_name", "match_state", "status"]]
          .to_string(index=False))

    out.to_json(OUT_JSON, orient="records", indent=2, date_format="iso")
    print(f"\nWrote {len(out)} rows to {OUT_JSON}")


if __name__ == "__main__":
    main()
