# Site interaction outputs

## plots/two_body/
- `examples_all_sites.png` — every usable pair (calendar grid)
- `avg_existing_single_new_multi_trend.png` — pooled median when existing=single, new=multi
- `trends_by_site_type_combo.png` — 2×2 grid for all four single/multi combinations

## plots/three_body/
- `examples_all_sites.png` — every usable triple
- `avg_all_triples_trend.png` — by role + single/multi (unique color per series)
- `avg_all_triples_trend_overall.png` — overall pool dual panel: month 0 = B (middle) | month 0 = C (newest)
- `avg_new_multi_intro_trend.png` — subset when newest site (C) is multi

## plots/four_body/
- `examples_all_sites.png` — every usable quad
- `avg_all_quads_trend.png` — by role + single/multi (unique color per series)
- `avg_all_quads_trend_overall.png` — overall pool dual panel: month 0 = C (middle) | month 0 = D (newest)

## plots/aggregate/
- `any_new_operator_effect.png` — all pairs: incumbent + combined market
- `market_saturation_threshold.png` — loss-making zone (incumbent ≤−5%, combined ≤+5%)

## data/
- `two_body_pair_deltas.csv`, `three_body_triple_deltas.csv`, `four_body_quad_deltas.csv`

## report/
- `site_interaction_report.md`

## backtesting/
Factor hypothesis from `backtesting.xlsx` matched to panel sites (n=18). See `backtesting/README.md`.

Re-run `site_interaction_analysis.ipynb`, `python run_site_interaction_plots.py`, or `python backtesting_analysis.py`.
