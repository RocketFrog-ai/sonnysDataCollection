# app/modelling/ds — final v3 approach: percentile profiler, feature scoring, quantile prediction.
#
# Public surface:
#   profiler            — CTOExactProfiler (percentile-based feature scorer)
#   feature_weights_config — directions, dimension weights, overall weights
#   scorer              — API-facing enrichment & scoring helpers
#   dimension_summary   — per-dimension rationale generation
#   quantile_predictor  — QuantilePredictorV3 (calibrated RF, v3)
#   quantile_display    — QUANTILE_TO_CATEGORY (Q1→Poor, Q2→Fair, Q3→Good, Q4→Strong), get_category_for_quantile
