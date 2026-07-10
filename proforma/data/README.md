# `proforma/data/` — the shared dataset store

One copy of every dataset, read by every model version. **Immutable**: nothing here is edited by
hand, and nothing here is version-specific. If you need a variant, add a new *filename*, not a new
folder — versioning is by dataset name, not by directory.

Full provenance, schemas, and the known data-quality issues are in **`../../docs/DATA.md`**. This
file is the map.

| path | what | who reads it |
|---|---|---|
| `panel/main-data-v2-stitched.csv` | **the canonical monthly site panel.** 2020-01→2027-01, 71,701 rows, 2,103 `client_id+site_id` keys, handoffs stitched | `../models/coldstart.py`, `../ui/app.py`, `app/pnl_analysis/modelling/data.py`, `../../experiments/council/data_1_6.py` |
| `panel/main-data-v2.csv` | raw export — input to the build script | `../scripts/process_main_data_v2.py` |
| `panel/main-data-v2-processed.csv` | intermediate, pre-stitch | same |
| `panel/main-data-6yr.csv` | older 6-year cut | the `same_location_timeline` + `tunnel_length_backtest` notebooks |
| `panel/main-ds.csv` | **legacy schema, superseded** by the stitched panel | `libs/carwash_type/classify_site_types.py` (as a site list) |
| `opex/opex-data.csv` | operating P&L. True opex = `cogs + expenses`; operator key `client_id` | `../ui/app.py`, `app/pnl_analysis/modelling/{data,campaign}.py` |
| `ref/site_carwash_types.csv` | **resolved** wash-type per site | `../ui/app.py` |
| `ref/merged_all_sites.csv` | site coordinate/name lookup | `app/site_analysis/server/site_features.py`, `../ui/site_visual_page.py`, `../backtests/backtest_features.py` |
| `ref/old-excel-proforma-data-enriched.csv` | 187 real builds (tunnel length → CAPEX) | `../models/tunnel_capex.py` |
| `ref/unknownsites_resolved.csv` | provenance for the resolved type table | *no current code reader* |
| `ref/site_carwash_types.csv.bak-pre-resolved` | pre-resolution snapshot | *no current code reader* |
| `ref/same_location_sites.csv` | same-address site pairs | the `same_location_timeline` notebook |
| `ref/merged_sites_with_2025_wash_counts_nonzero_with_region_state.csv` | 2025 cut with region/state | *no current code reader* |
| `derived/cannibalization_entrants.csv` | output of the entry-cannibalization study | *no current code reader* |

## Rebuilding the panel

```bash
python proforma/scripts/process_main_data_v2.py    # run from the repo root
```

Reads `panel/main-data-v2.csv`, writes `-processed` and `-stitched` beside it. It used to write two
copies, one per project tree, to keep mirrors in sync; the mirrors were proven byte-identical and
collapsed, so there is now exactly one output location.

## Two traps

1. **The site key is `client_id + site_id`.** `site_id` alone is a within-brand index and collides
   across operators. Every loader builds `site_key = client_id + "::" + site_id`.
2. **`libs/carwash_type/data/site_carwash_types.csv` is a *different file* with the same name** —
   it is the classifier's raw output, not the resolved table here. They differ (1,302,501 vs
   1,306,923 bytes). Do not merge them. See `../../docs/DIVERGENCES.md` §3.
