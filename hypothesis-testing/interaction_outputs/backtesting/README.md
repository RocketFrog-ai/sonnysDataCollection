# Backtesting — localisation clusters (1–4)

Matched **18/18** sites. Clusters follow `backtesting_localisation.tsv` / `del.tsv` (not auto geography).

Retail blank / "None" in Excel → **No major retail** (Colorado sites have no listed anchor).

## Clusters
- **Localisation 1 — DFW (Frisco / Lewisville / Prosper, TX)** — 5 sites, span 13.0 mi
- **Localisation 2 — Tampa Bay FL + Beaver Dam WI** — 5 sites, span 11.6 mi
- **Localisation 3 — Colorado metros** — 4 sites, span 55.1 mi
- **Localisation 4 — Tennessee** — 4 sites, span 27.1 mi

## plots/
| File | Content |
|------|---------|
| `cluster_localisation_1.png` … `_4.png` | 8-panel factor dashboard — **2024 panel washes only** |
| `cluster_localisation_*_washes_2024_2025.png` | Optional: 2024 vs 2025 side-by-side (not used on factor charts) |
| `factor_correlation_heatmap.png` | All-site exploratory (2024 washes) |
| `wash_vs_traffic.png` | Traffic vs **2024** washes, colored by localisation |

### Wash counts
All factor charts use **`2024 car wash count`** from `backtesting.xlsx` (not panel CSV sums). Year comparison charts use **`2024 car wash count`** and **`2025 car wash count`** columns from the same sheet.

Re-run: `python backtesting_analysis.py`
