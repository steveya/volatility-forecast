from .stes_model import STESModel
from .pgarch_linear_model import PGARCHLinearModel
from .xgb_pgarch_full_model import XGBPGARCHModel
from .xgb_pgarch_model import XGBGPGARCHModel
from .tree_stes_model import XGBoostSTESModel
from .volgru_model import VolGRUModel

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
