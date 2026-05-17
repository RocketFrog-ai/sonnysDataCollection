# Site Interaction Report

## Method
- Data used (unified monthly panel):
  - `less_than-2yrs.csv`: 491 sites, 8,958 site-month rows (2024-01-01 to 2026-12-01).
  - `more_than-2yrs_monthly.csv`: 482 sites, 11,548 site-month rows (2024-01-01 to 2025-12-01).
- Combined panel: 973 unique sites, 20,506 site-month rows, calendar span 2024-01-01 to 2026-12-01.
- Typical roles: newer launches from lt2; older neighbors from gt2 monthly (true calendar months in `year_month`).
- Month indexing fix: the launch month is now the month that contains `operational_start_date`, so launch month = 1, the month before launch = 0, and earlier months are negative.
- Rows impacted by that correction in `less_than-2yrs.csv`: 0 rows across 0 sites.
- Prelaunch rows still present in the raw less-than-2-years panel: 0. These rows stay in the data for transparency but are excluded from new-site launch visuals before month 1.
- Local-market rule: prefer same-ZIP older sites when available, but still require the match to be within 10 miles.
- Pre/post window: 6 months before vs 6 months after the launch month.

## Two-body findings
- Usable nearby pairs: 84.
- Existing-site source: 77 from gt2 monthly, 7 from lt2.
- New-site source: 84 from lt2, 0 from gt2 monthly.
- Median existing-site total change: -2.7%.
- Median combined-market total change: +49.8%.
- Same-ZIP share among usable pairs: 2%.
- Most common regime: market expansion.
- In the event-time chart, the median combined market reaches 194.1 on the pre-launch index scale while the existing site bottoms at 91.9.

## Three-body findings
- Usable triples: 41.
- Median A-site total change after C launches: -7.4%.
- Median B-site total change after C launches: -0.9%.
- Median full A+B+C market total change: +31.3%.
- In the event-time chart, the full market peaks at 155.7 on the A+B pre-launch index scale.

## Plain-English takeaway
- The fatal month-indexing bug is fixed: all launch comparisons now align to the launch month, not the panel's January 2024 calendar.
- Nearby launches usually pressure the closest older site, but the local market often still expands once the new site is added.
- The cleaner event-time plots are the easiest way to read the result: blue and green older-site lines soften after launch, while the orange combined-market line usually stays above the pre-launch baseline.
