# Site Interaction Report

## Method
- Data used: `less_than-2yrs.csv` and `more_than-2yrs_monthly.csv`.
- Monthly panel coverage: 2024-01-01 to 2026-12-01 across 973 sites and 20,506 site-month rows.
- Month indexing fix: the launch month is now the month that contains `operational_start_date`, so launch month = 1, the month before launch = 0, and earlier months are negative.
- Rows impacted by that correction in `less_than-2yrs.csv`: 3,852 rows across 221 sites.
- Prelaunch rows still present in the raw less-than-2-years panel: 1,223. These rows stay in the data for transparency but are excluded from new-site launch visuals before month 1.
- Local-market rule: prefer same-ZIP older sites when available, but still require the match to be within 10 miles.
- Pre/post window: 6 months before vs 6 months after the launch month.

## Two-body findings
- Usable nearby pairs: 81.
- Median existing-site total change: -4.4%.
- Median combined-market total change: +89.0%.
- Same-ZIP share among usable pairs: 1%.
- Most common regime: market expansion.
- In the event-time chart, the median combined market reaches 178.2 on the pre-launch index scale while the existing site bottoms at 92.5.

## Three-body findings
- Usable triples: 33.
- Median A-site total change after C launches: -10.1%.
- Median B-site total change after C launches: -7.5%.
- Median full A+B+C market total change: +38.5%.
- In the event-time chart, the full market peaks at 142.6 on the A+B pre-launch index scale.

## Plain-English takeaway
- The fatal month-indexing bug is fixed: all launch comparisons now align to the launch month, not the panel's January 2024 calendar.
- Nearby launches usually pressure the closest older site, but the local market often still expands once the new site is added.
- The cleaner event-time plots are the easiest way to read the result: blue and green older-site lines soften after launch, while the orange combined-market line usually stays above the pre-launch baseline.
