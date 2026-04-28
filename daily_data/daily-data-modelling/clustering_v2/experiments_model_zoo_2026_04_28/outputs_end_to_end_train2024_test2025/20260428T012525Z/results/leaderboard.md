# End-to-end time split leaderboard

Definition:
- `<2y`: train 2024 monthly rows, test 2025 monthly totals (calendar_day for year2 is shifted in the CSV; corrected by -1 year).
- `>2y`: train 2024 daily rows, test 2025 monthly totals.

Score = lt_wape(test year monthly) + gt_wape(test year monthly). Lower is better.

| rank | variant | score | lt_wape | gt_wape | lt_r2 | gt_r2 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `xgb_arima` | 0.8348 | 0.479 | 0.3558 | -0.1478 | 0.4972 |
| 2 | `xgb_meta` | 0.8353 | 0.4791 | 0.3562 | -0.1521 | 0.4913 |
| 3 | `xgb_prophet` | 2.7435 | 1.0738 | 1.6697 | -8.6921 | -11.046 |
