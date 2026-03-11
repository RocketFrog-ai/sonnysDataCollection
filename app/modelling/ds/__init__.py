# app/modelling/ds — final v3 approach: percentile profiler, feature scoring, quantile prediction.
#
# Public surface:
#   profiler            — CTOExactProfiler (percentile-based feature scorer)
#   feature_weights_config — directions, dimension weights, overall weights
#   scorer              — API-facing enrichment & scoring helpers
#   dimension_summary   — per-dimension rationale generation
#   quantile_predictor  — QuantilePredictorV3 (calibrated RF, v3)
