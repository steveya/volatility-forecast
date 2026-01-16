from .stes_model import STESModel
from .tree_stes_model import XGBoostSTESModel

# Backwards-compatible alias for the older implementation.
from .xgboost_stes_model import XGBoostSTESModel as LegacyXGBoostSTESModel

try:
    from .neural_network_model import (
        RNNVolatilityModel,
        GRUVolatilityModel,
        RNNSTESModel,
    )
except ModuleNotFoundError as e:
    # Allow using the package without optional deep learning deps.
    if getattr(e, "name", None) == "torch":
        RNNVolatilityModel = None
        GRUVolatilityModel = None
        RNNSTESModel = None
    else:
        raise
