from .stes_model import STESModel
from .tree_stes_model import XGBoostSTESModel

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
