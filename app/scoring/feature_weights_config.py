# Backward-compatibility shim.
# All weight / direction config lives in app/modelling/ds/feature_weights_config.py.
from app.modelling.ds.feature_weights_config import *  # noqa: F401, F403
from app.modelling.ds.feature_weights_config import (  # noqa: F401
    FEATURE_DIRECTION,
    DIMENSION_FEATURE_WEIGHTS,
    FEATURE_WEIGHTS,
    DIMENSION_FEATURES,
    DIMENSION_WEIGHTS_FOR_OVERALL,
)
