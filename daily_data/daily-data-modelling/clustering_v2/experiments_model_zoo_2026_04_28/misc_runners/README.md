# misc_runners

Run from this directory (or pass paths accordingly). Each script imports `clustering_v2/build_v2.py` via `parents[2]` and writes under the parent experiment folder:

| Script | Output folder prefix |
|--------|----------------------|
| `run_site_holdout_zoo.py` | `../outputs_site_holdout_intersection/` |
| `run_time_split_gt_2024_2025.py` | `../outputs_mature_level_train2024_test2025/` |
| `run_time_split_end_to_end_2024_2025.py` | `../outputs_end_to_end_train2024_test2025/` |
| `run_lt_time_split_2024_6m6m.py` | `../outputs_lt2y_within_2024_train6m_test6m/` |
| `run_temporal_avg_ts_2024_2025_and_4y_outlook.py` | `../outputs_temporal_ts_mean_curves/` |

`run_time_split_end_to_end_2024_2025.py` accepts `--only xgb_arima,xgb_prophet` (etc.) for faster comparisons; conda env needs **xgboost** and **prophet**. `meta` works but is slow.

Copy key metrics back into `../REPORT.md` after significant reruns.
