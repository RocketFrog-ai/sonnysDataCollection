"""Compare greenfield projection (Year 1–4) to observed volumes for sample >2y test sites.

Observed (from ``master_more_than-2yrs.csv``):
  - **Operational year** = sum of ``wash_count_total`` where ``year_number`` is 1, 2, … (panel encoding).
  - **Calendar 2024 / 2025** = sum of daily washes in those Gregorian years (same file).

Greenfield: ``project_site.run_projection`` (Ridge level, default prefix + bridge, ARIMA), same as CLI.

**Important:** The projection is **greenfield** — it does **not** replay or fit this site’s past volumes. It is
“typical new site at this location / cluster.” So gaps vs **this** site’s realized 2024–2025 or op-year totals
are expected and are **not** the same metric as Ridge holdout MAE on daily rows.

Writes ``results/projection_vs_observed_operational_sample.json``.

Run from ``clustering_v2``::

    python eval_greenfield_vs_observed_volumes.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

V2 = Path(__file__).resolve().parent
REPO_ROOT = V2.parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(V2))

import project_site as ps  # noqa: E402


def _pct_err(actual: float, pred: float) -> float | None:
    if not (actual > 0 and pred == pred):
        return None
    return round(100.0 * (pred - actual) / actual, 2)


def main() -> None:
    master = pd.read_csv(
        REPO_ROOT / "daily_data/daily-data-modelling/master_more_than-2yrs.csv",
        usecols=[
            "site_client_id",
            "calendar_day",
            "wash_count_total",
            "year_number",
            "latitude",
            "longitude",
            "Address",
        ],
        low_memory=False,
    )
    master["calendar_day"] = pd.to_datetime(master["calendar_day"], errors="coerce")
    master = master.dropna(subset=["calendar_day", "wash_count_total"])

    test_site_ids = master.loc[master["calendar_day"] >= pd.Timestamp("2025-07-01"), "site_client_id"].unique()
    meta = (
        master.groupby("site_client_id", as_index=False)
        .agg(
            lat=("latitude", "first"),
            lon=("longitude", "first"),
            addr=("Address", "first"),
            cal_min=("calendar_day", "min"),
            cal_max=("calendar_day", "max"),
        )
    )
    meta = meta[meta["site_client_id"].isin(test_site_ids)]
    meta = meta.dropna(subset=["lat", "lon"])

    # Operational-year actuals (all dates, per site)
    op = (
        master.groupby(["site_client_id", "year_number"], as_index=False)["wash_count_total"]
        .sum()
        .rename(columns={"wash_count_total": "actual_washes"})
    )
    op = op[op["year_number"].isin([1, 2, 3, 4])]

    # Calendar-year actuals
    master["_y"] = master["calendar_day"].dt.year
    cy = master.groupby(["site_client_id", "_y"], as_index=False)["wash_count_total"].sum()
    cy = cy.rename(columns={"wash_count_total": "actual_washes", "_y": "calendar_year"})
    cy24 = cy[cy["calendar_year"] == 2024].set_index("site_client_id")["actual_washes"]
    cy25 = cy[cy["calendar_year"] == 2025].set_index("site_client_id")["actual_washes"]

    rows: list[dict[str, Any]] = []
    # Deterministic sample: first N meta rows that project successfully
    sample_n = 8
    taken = 0
    for _, r in meta.sort_values("site_client_id").iterrows():
        if taken >= sample_n:
            break
        sid = int(r["site_client_id"])
        lat, lon = float(r["lat"]), float(r["lon"])
        try:
            resp = ps.run_projection(
                None,
                lat,
                lon,
                "arima",
                use_opening_prefix_for_mature_forecast=True,
                bridge_opening_to_mature_when_prefix=True,
                allow_nearest_cluster_beyond_distance_cap=True,
                level_model="ridge",
            )
        except SystemExit:
            continue
        if resp.get("more_than_2yrs", {}).get("error") or resp.get("less_than_2yrs", {}).get("error"):
            continue
        cyw = resp.get("calendar_year_washes") or {}
        if not all(k in cyw for k in ("year_1", "year_2", "year_3", "year_4")):
            continue

        pred = {f"pred_year_{i}": float(cyw[f"year_{i}"]) for i in (1, 2, 3, 4)}
        op_site = op[op["site_client_id"] == sid].set_index("year_number")["actual_washes"].to_dict()
        actual_by_op_year = {int(k): float(v) for k, v in op_site.items()}

        row: dict[str, Any] = {
            "site_client_id": sid,
            "address_snip": (str(r["addr"])[:80] + "…") if r["addr"] is not None and len(str(r["addr"])) > 80 else r["addr"],
            "lat": lat,
            "lon": lon,
            "data_calendar_min": r["cal_min"].strftime("%Y-%m-%d"),
            "data_calendar_max": r["cal_max"].strftime("%Y-%m-%d"),
            "actual_calendar_2024_total_washes": float(cy24.get(sid, float("nan"))) if sid in cy24.index else None,
            "actual_calendar_2025_total_washes": float(cy25.get(sid, float("nan"))) if sid in cy25.index else None,
            "actual_operational_year_totals": {f"year_{k}": v for k, v in sorted(actual_by_op_year.items())},
            "greenfield_calendar_year_washes_projection": pred,
            "pct_error_pred_vs_actual_operational_year": {
                f"year_{k}": _pct_err(actual_by_op_year[k], pred[f"pred_year_{k}"])
                for k in (1, 2, 3, 4)
                if k in actual_by_op_year
            },
            "pct_error_pred_vs_calendar_year": {
                "2024_vs_pred_year_1": _pct_err(float(cy24[sid]), pred["pred_year_1"]) if sid in cy24.index else None,
                "2025_vs_pred_year_2": _pct_err(float(cy25[sid]), pred["pred_year_2"]) if sid in cy25.index else None,
            },
        }
        rows.append(row)
        taken += 1

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "caveat": (
            "Greenfield Year 1–4 are model defaults for a hypothetical opening at this location; "
            "they are not calibrated to each site's realized ramp. "
            "Operational year_number in CSV is the panel's year index (not always Jan–Dec calendar). "
            "2024 vs pred_year_1 / 2025 vs pred_year_2 are rough calendar-vs-operational analogies only."
        ),
        "n_sites_attempted_sample": sample_n,
        "n_sites_reported": len(rows),
        "sites": rows,
    }
    outp = V2 / "results" / "projection_vs_observed_operational_sample.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {outp.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
