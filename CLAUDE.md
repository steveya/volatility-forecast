# CLAUDE.md

## Project Overview

**volatility-forecast** is a Python library for time-series volatility forecasting combining statistical and deep learning models. It provides a scikit-learn compatible API and integrates with **alphaforge** for scalable dataset construction.

## Repository Structure

```
volatility_forecast/           # Main package
  model/                       # Forecasting models (ES, STES, XGBoost-STES, VolGRU)
  features/                    # Feature engineering (returns, signatures, range estimators)
  targets/                     # Target variable definitions (squared returns)
  sources/                     # Data source adapters (Tiingo EOD, simulated GARCH)
  data/                        # Data handling (PriceVolume dataclass)
  evaluation/                  # Metrics (RMSE, MAE, QLIKE), Diebold-Mariano test
  benchmark/                   # Benchmarking utilities
  reporting/                   # Reporting and visualization
  api/                         # Flask REST API
  forecast/                    # Forecast generation
  pipeline.py                  # Dataset building pipeline (VolDatasetSpec)
  storage.py                   # DuckDB/filesystem storage
  main.py                      # Entry point
tests/                         # Pytest test suite (24+ files)
examples/                      # Runnable example scripts
scripts/                       # Utility scripts
config/                        # YAML configs (model, data, app)
docs/                          # MkDocs documentation
notebooks/                     # Jupyter notebooks
```

## Development Environment

- **Python**: 3.11 or 3.12 (conda env `py312` on macOS)
- **Install**: `pip install -e .[dev]`
- **Environment variables**: stored in `.env` at repo root
  - `TIINGO_API` — Tiingo API key (required for live data tests)
  - `TIINGO_CACHE_DIR` — cache directory
  - `TIINGO_CACHE_MODE` — cache mode

## Commands

```bash
# Run all tests
pytest tests/ -v

# Run tests (conda)
conda run -n py312 pytest -q

# Run a specific test file
pytest tests/test_signature_features.py -v

# Run examples (some require TIINGO_API)
python examples/demo_forecast.py
```

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`):
- Triggers on push/PR to `main`
- Tests on Python 3.11 and 3.12 (Ubuntu)
- Installs with `pip install -e .[dev]` then runs `pytest tests/ -v`

## Architecture & Key Patterns

### Model hierarchy
All models inherit from `BaseVolatilityModel` (in `model/base_model.py`) which extends scikit-learn's `BaseEstimator`/`RegressorMixin`. Models implement `fit()` and `predict()`. Fitted attributes use trailing underscore convention (e.g., `params_`, `is_fitted_`).

### Key models
| Model | File | Description |
|-------|------|-------------|
| ES | `model/es_model.py` | Exponential Smoothing (constant alpha) |
| STES | `model/stes_model.py` | Smooth Transition ES with logistic smoothing |
| XGBoost-STES | `model/tree_stes_model.py` | XGBoost hybrid with STES |
| VolGRU | `model/volgru_model.py` | GRU neural network (PyTorch/JAX backends) |

### Feature system
Features use alphaforge `FeatureTemplate` patterns. Key feature types:
- **Return features** (`features/return_features.py`): lagged returns, abs/squared returns
- **Signature features** (`features/signature_features.py`): rough path theory via sktime
- **Range features** (`features/range_features.py`): high-low range estimators

### Data pipeline
- `pipeline.py` provides `VolDatasetSpec` and `build_vol_dataset()` for dataset construction
- Point-in-time (PIT) alignment via `asof_utc` columns is required throughout
- `storage.py` provides DuckDB-backed persistence for models and forecasts

### Evaluation
- Fixed-split evaluation: Train on `[is_index : os_index)`, Test on `[os_index : ]`
- STES prediction is aligned to next-step variance; compare to shifted target when needed
- Metrics: RMSE, MAE, QLIKE, hit_rate, correlation (in `evaluation/metrics.py`)
- Diebold-Mariano test for statistical comparison (`evaluation/dm_test.py`)

## Conventions

- **Naming**: PascalCase for classes, snake_case for functions/methods, leading underscore for private
- **Minimal edits**: prefer small, targeted changes over refactors
- **Compatibility**: maintain alphaforge conventions
- **Tests**: keep tests aligned with agreed conventions; test files mirror source structure
- **Caching**: Tiingo caching is pluggable via alphaforge cache backends; use CLI flags when available

## Configuration

- `config/model_config.yaml` — model definitions, hyperparameters, feature lists
- `config/data_config.yaml` — data sources, tickers (SPY), date ranges
- `config/app_config.yaml` — application settings

## Dependencies

Core: numpy, pandas, scipy, scikit-learn, joblib, requests, sqlalchemy, pandas-market-calendars, python-dotenv

Optional: xgboost, sktime (>=0.24.0), torch, jax, flask, apscheduler, matplotlib, seaborn
