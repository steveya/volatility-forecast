# Volatility Forecast

Deep learning and statistical models for time-series volatility forecasting, featuring VolGRU and STES implementations.

## Overview

This library provides a rigorous and extensible framework for estimating conditional variance, allowing you to train pure statistical models like Smooth Transition Exponential Smoothing (STES), gradient-boosted tree extensions (XGBSTES), and neural network models like Volatility Gated Recurrent Units (VolGRU).

The underlying philosophy treats standard econometric models (like STES and GARCH) as constrained recurrent neural networks, mapping the architectural space such that deep learning variants can act as strict relaxations of the baselines.

## Quickstart

```python
from volatility_forecast.model.volgru_model import VolGRUModel
from volatility_forecast.model.volgru_config import VolGRUConfig

config = VolGRUConfig(
    backend='torch',
    gate_mode='gru_linear',
    candidate_mode='mlp_pos',
    val_fraction=0.15,
    dtype='float32',  # use 'float64' for 1e-12 STES numerical parity
)
model = VolGRUModel(config=config)
model.fit(X_train, y_train, returns=returns_train)
```

> **Precision note (`VolGRUConfig.dtype`):** The Torch backend defaults to `dtype='float32'` for training throughput and backward-compatible outputs (`np.float32`, matching `STESModel` to `rtol=1e-6, atol=1e-6` in the STES-reduced `gate_mode='stes_linear', candidate_mode='stes_r2', state_dim=1` configuration at moderate gate logits). Set `dtype='float64'` when exact double-precision parity with `STESModel` (`rtol=0.0, atol=1e-12`, provided variance states remain above `VolGRUConfig.eps`) is required.

## Alphaforge Runtime

The repo's canonical data path now assumes adapter-backed Alphaforge contexts.
For local development, install the sibling Alphaforge checkout into `py312`
and validate that the runtime exposes `DataContext.from_adapters(...)` and
`DataContext.load(...)`:

```bash
conda run -n py312 python -m pip install -e ../alphaforge -e .
conda run -n py312 python -c "from alphaforge.data.context import DataContext; assert hasattr(DataContext, 'from_adapters'); assert hasattr(DataContext, 'load')"
conda run -n py312 python -m pytest -q
```

## Documentation

* [Training Protocol & Early Stopping](design_decisions/training_protocol.md)
