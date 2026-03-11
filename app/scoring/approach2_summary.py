# Backward-compatibility shim.
# All dimension summary logic lives in app/modelling/ds/dimension_summary.py.
from app.modelling.ds.dimension_summary import *  # noqa: F401, F403
from app.modelling.ds.dimension_summary import (  # noqa: F401
    DIMENSION_FEATURE_MAP,
    FEATURE_LABELS,
    PORTFOLIO_N_SITES,
    INCLUDE_LLM_SUMMARY_TAIL,
    get_dimension_summary_approach2,
    build_full_profiling_rationale,
    _overall_score_to_category,
    get_feature_performance_map,
)
