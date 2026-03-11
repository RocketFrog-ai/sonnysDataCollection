# Backward-compatibility shim.
# All scoring logic lives in app/modelling/ds/scorer.py.
from app.modelling.ds.scorer import *  # noqa: F401, F403
from app.modelling.ds.scorer import (  # noqa: F401 — explicit re-export for static analysis
    WEATHER_API_TO_PROFILER,
    GAS_API_TO_PROFILER,
    COMPETITORS_API_TO_PROFILER,
    RETAILER_API_TO_PROFILER,
    TASK_GAS_TO_PROFILER,
    score_feature_with_config,
    enrich_features_with_categories,
    enrich_gas_features_with_categories,
    enrich_competitors_features_with_categories,
    enrich_retailers_features_with_categories,
    get_feature_final_scores,
    compute_dimension_score,
    compute_overall_score,
    get_all_profiler_scores_from_task_feature_values,
)
