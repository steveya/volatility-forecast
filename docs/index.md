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
    val_fraction=0.15
)
model = VolGRUModel(config=config)
model.fit(X_train, y_train, returns=returns_train)
```

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
