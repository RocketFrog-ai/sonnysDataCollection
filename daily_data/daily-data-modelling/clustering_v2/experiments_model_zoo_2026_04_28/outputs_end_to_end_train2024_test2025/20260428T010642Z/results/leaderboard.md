# End-to-end time split leaderboard

Definition:
- `<2y`: train 2024 monthly rows, test 2025 monthly totals (calendar_day for year2 is shifted in the CSV; corrected by -1 year).
- `>2y`: train 2024 daily rows, test 2025 monthly totals.

Score = lt_wape(test year monthly) + gt_wape(test year monthly). Lower is better.

| rank | variant | score | lt_wape | gt_wape | lt_r2 | gt_r2 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `xgb_arima` | 0.8343 | 0.4785 | 0.3558 | -0.146 | 0.4971 |
| 2 | `xgb_meta` | 0.8398 | 0.4773 | 0.3625 | -0.1421 | 0.4776 |
| 3 | `xgb_prophet` | 0.8406 | 0.4871 | 0.3535 | -0.1875 | 0.4986 |
| 4 | `xgb_blend` | 0.8444 | 0.469 | 0.3754 | -0.1286 | 0.3891 |
| 5 | `xgb_holt` | 0.8704 | 0.4722 | 0.3982 | -0.145 | 0.3006 |
| 6 | `rf_arima` | 0.905 | 0.5036 | 0.4014 | -0.2774 | 0.3521 |
| 7 | `rf_blend` | 0.9083 | 0.4982 | 0.4101 | -0.2868 | 0.2695 |
| 8 | `rf_prophet` | 0.9113 | 0.5121 | 0.3992 | -0.3157 | 0.3542 |
| 9 | `rf_meta` | 0.9122 | 0.5089 | 0.4033 | -0.3262 | 0.3411 |
| 10 | `rf_holt` | 0.9333 | 0.5047 | 0.4286 | -0.3333 | 0.1952 |
| 11 | `ridge_arima` | 0.9993 | 0.5163 | 0.483 | -0.269 | 0.1832 |
| 12 | `ridge_blend` | 1.0005 | 0.5081 | 0.4924 | -0.2502 | 0.1119 |
| 13 | `ridge_meta` | 1.0017 | 0.5174 | 0.4843 | -0.2747 | 0.1726 |
| 14 | `ridge_prophet` | 1.0054 | 0.5226 | 0.4828 | -0.3092 | 0.1887 |
| 15 | `ridge_holt` | 1.0196 | 0.5099 | 0.5097 | -0.2621 | 0.0437 |
