# Site interaction outputs

## plots/two_body/
- `examples_all_sites.png` — every usable pair (calendar grid)
- `avg_existing_single_new_multi_trend.png` — pooled median when existing=single, new=multi

## plots/three_body/
- `examples_all_sites.png` — every usable triple
- `avg_new_multi_intro_trend.png` — pooled trend when newest site (C) is multi

## plots/four_body/
- `examples_all_sites.png` — every usable quad

## plots/aggregate/
- `any_new_operator_effect.png` — all pairs: incumbent + combined market
- `market_saturation_threshold.png` — loss-making zone (incumbent ≤−5%, combined ≤+5%)

## data/
- `two_body_pair_deltas.csv`, `three_body_triple_deltas.csv`, `four_body_quad_deltas.csv`

## report/
- `site_interaction_report.md`

Re-run `site_interaction_analysis.ipynb` or `python run_site_interaction_plots.py`.
