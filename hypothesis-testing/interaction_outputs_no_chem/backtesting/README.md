# Backtesting — localisation clusters (1–4)

Matched **14/18** sites. Clusters follow `backtesting_localisation.tsv` / `del.tsv` (not auto geography).

Retail blank / "None" in Excel → **No major retail** (Colorado sites have no listed anchor).

## Clusters
- **Localisation 1 — DFW (Frisco / Lewisville / Prosper, TX)** — 5 sites, span 13.0 mi
- **Localisation 2 — Tampa Bay FL + Beaver Dam WI** — 5 sites, span 5.0 mi
- **Localisation 3 — Colorado metros** — 4 sites, span 19.1 mi
- **Localisation 4 — Tennessee** — 4 sites, span 16.2 mi

## plots/
| File | Content |
|------|---------|
| `cluster_localisation_1.png` … `_4.png` | 8-panel factor dashboard — **2024 car wash count** (sheet) |
| `cluster_localisation_*_washes_2024_2025.png` | Optional: 2024 vs 2025 side-by-side (not used on factor charts) |
| `factor_correlation_heatmap.png` | All-site exploratory (2024 washes) |
| `wash_vs_traffic.png` | Traffic vs **2024** washes, colored by localisation |

### Wash counts
All factor charts use **`2024 car wash count`** from `backtesting.xlsx`. Year comparison uses **`2024`** and **`2025 car wash count`** columns.

### Address match
`data/backtesting_address_match.csv` — exact normalized `Address` match to `less_than-2yrs.csv` / `more_than-2yrs_monthly.csv` + `operational_start_date` when present in those files.

Re-run: `python backtesting_analysis.py`
