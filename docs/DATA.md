# Data

All datasets live **once**, under `proforma/data/`, versioned by **filename**, not by folder.
Model versions (`proforma/v1_5`, `proforma/v1_6`) declare in their own README which datasets they
consume. Nothing under `proforma/data/` is version-specific; nothing should be edited by hand.

```
proforma/data/
├── panel/    the monthly site panel and its build chain
├── opex/     operating P&L
├── ref/      reference / lookup tables
└── derived/  outputs of analyses
```

## The canonical panel

`proforma/data/panel/main-data-v2-stitched.csv` — **the source of truth for "the sites".**

- 2020-01 → 2027-01, 71,701 rows, 2,103 `client_id + site_id` keys.
- Operator handoffs stitched into single site identities; `imputed=0`.
- The real site key is **`client_id + site_id`**. `site_id` alone is a within-brand index and
  collides across operators.

Read by `proforma/v1_5/models/coldstart.py`, `proforma/v1_5/ui/app.py`,
`app/pnl_analysis/modelling/data.py`, and `proforma/v1_6/data_1_6.py`.

### Build chain

```
main-data-v2.csv            raw export
  └─ process_main_data_v2.py ─→ main-data-v2-processed.csv
                             └─→ main-data-v2-stitched.csv    ← use this one
```

```bash
python proforma/v1_5/scripts/process_main_data_v2.py    # from the repo root
```

Also in `panel/`: `main-data-6yr.csv` (older 6-year cut, no current code reader) and `main-ds.csv`
(**legacy schema, superseded** — read only by `libs/carwash_type/classify_site_types.py` as a site
list, and by the `moirai_ts` notebook).

## The rest

| file | consumer |
|---|---|
| `opex/opex-data.csv` | Streamlit P&L, `app/pnl_analysis/modelling/{data,campaign}.py`. True opex = `cogs + expenses`; operator key is `client_id`. |
| `ref/site_carwash_types.csv` | `proforma/v1_5/ui/app.py` — the **resolved** wash-type table |
| `ref/merged_all_sites.csv` | `app/site_analysis/server/site_features.py`, the Sitewise UI page, `backtest_features.py` |
| `ref/old-excel-proforma-data-enriched.csv` | `proforma/v1_5/models/tunnel_capex.py` (187 real builds → tunnel-length→CAPEX) |
| `ref/unknownsites_resolved.csv`, `ref/site_carwash_types.csv.bak-pre-resolved` | provenance for the resolved type table |
| `ref/same_location_sites.csv`, `ref/merged_sites_with_2025_...csv` | no current code reader |
| `derived/cannibalization_entrants.csv` | no current code reader; output of the entry-cannibalization study |

## Provenance of the collapse (2026-07)

The panel used to exist in **two** trees, `earnest-proforma-2.0/data/` and
`earnest-proforma-final-1.6/data/`. Three filenames appeared in both. Each pair was verified
**byte-identical by sha256** before the duplicate was removed:

| file | bytes |
|---|---:|
| `main-data-v2-stitched.csv` | 13,269,792 |
| `main-data-v2-processed.csv` | 13,123,055 |
| `main-data-v2.csv` | 13,578,108 |

~40 MB of duplication removed, and `process_main_data_v2.py` no longer has to write two copies to
keep them in sync. The council (`proforma/v1_6`) previously read the 1.6 copy specifically; it now
reads the shared one, which is the same bytes.

**`site_carwash_types.csv` was NOT collapsed.** `proforma/data/ref/` and `libs/carwash_type/data/`
hold two *different* files with that name (1,306,923 vs 1,302,501 bytes) — the resolved table and
the classifier's raw output. See `docs/DIVERGENCES.md` §3.

## Known data-quality issues

- **33 of 2,103 site keys have implausible coordinates** — several at exactly `lat=0.0, lon=0.0`,
  one at `lat=90.0, lon=180.0`, one in Australia. They fall outside any plausible US bounding box.
  `coldstart.load_panel_site` and the council both filter on `has_coords` (non-null), which does
  **not** exclude `0.0, 0.0`. Pre-existing; not addressed by the restructure.
- Some operators' revenue collapses to ~0 while wash counts hold, corrupting cluster ASP. The
  models defend against this with an ASP-corruption filter (`_drop_corrupt_asp_rows`, ASP > $200 →
  null the revenue leg). See `proforma/v1_5/MODELLING.md`.

## git-LFS

**Not in use.** `git lfs ls-files` reports zero LFS-tracked files at HEAD, and did so before the
restructure too. `.gitattributes` used to carry 11 `filter=lfs` patterns, all dead — 9 matched
untracked files, 2 matched nothing. It now carries an explanation instead.

The largest tracked file, `proforma/v1_5/artifacts/coldstart_artifacts.joblib` (~46 MB), is an
ordinary inline git blob. Do not convert it to LFS after the fact: a pattern added after a file is
committed silently does nothing to the existing history.
