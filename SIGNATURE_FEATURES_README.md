# Signature Features for Volatility Forecasting

This document describes the signature-based features implementation for volatility forecasting using sktime's `SignatureTransformer`.

## Overview

Signature features capture the geometric and topological properties of time series paths, making them particularly useful for modeling complex dependencies in financial time series. This implementation has been integrated into the volatility-forecast package.

## What Changed

### 1. Updated `volatility_forecast/features/signature_features.py`

**Key Changes:**
- **Migrated from `iisignature` to `sktime.transformations.series.signature_based.SignatureTransformer`**
- **Added comprehensive augmentation support** with parameter `augmentation_list`:
  - `"all"`: Applies both basepoint and addtime augmentations (most feature-rich)
  - `"none"`: No augmentation (baseline)
  - `"time"` or `"addtime"`: Time augmentation only
  - `"basepoint"`: Basepoint augmentation only
- **Bumped version** from "1.0" to "2.0" to reflect the breaking change
- **Enhanced catalog metadata** to track augmentation type used

**Benefits of sktime:**
- Consistent API with other sktime transformers
- Better integration with scikit-learn pipelines
- More augmentation options out-of-the-box
- Active maintenance and community support

### 2. Created Comprehensive Test Suite

**File:** `tests/test_signature_features.py`

**Test Coverage:**
- Template initialization and parameter space validation
- Basic signature feature transform
- All augmentation methods (`all`, `none`, `time`, `basepoint`, `addtime`)
- Different signature levels (depths 1-5)
- Different lag windows (3, 5, 10, 20 days)
- Feature consistency and reproducibility
- Catalog structure and metadata
- Feature value validity (finite values, no NaNs)
- Index alignment between features and catalog
- Integration tests with high signature levels and long lag windows
- Augmentation comparison across different configurations

**Run tests with:**
```bash
cd /Users/steveyang/Projects/Github/volatility-forecast
pytest tests/test_signature_features.py -v
```

### 3. Created Example Script

**File:** `examples/signature_in_volatility_forecast.py`

This script demonstrates practical usage of signature features for volatility forecasting. It's based on `volatility_forecast_2.py` but enhanced with signature features.

**Features:**
- Combines traditional lag features (returns, absolute returns, squared returns) with signature features
- Supports all augmentation methods exposed by `SignatureTransformer`
- Evaluates multiple models: ES, STES, and XGBSTES with signatures
- Includes comparison mode to evaluate all augmentation methods side-by-side
- Generates visualizations and result tables

**Usage Examples:**

```bash
# Basic usage with all augmentations (default)
python examples/signature_in_volatility_forecast.py --sig-lags 10 --sig-level 3

# Compare all augmentation methods
python examples/signature_in_volatility_forecast.py --compare-augmentations --sig-lags 10 --sig-level 3

# Use specific augmentation
python examples/signature_in_volatility_forecast.py --augmentation basepoint --sig-lags 15 --sig-level 2

# Custom configuration
python examples/signature_in_volatility_forecast.py \
    --n-lags 5 \
    --sig-lags 20 \
    --sig-level 4 \
    --augmentation all \
    --seed 42 \
    --output-dir outputs/my_signature_experiment
```

**Command-line Arguments:**
- `--n-lags`: Number of traditional lag features (default: 5)
- `--sig-lags`: Window size for signature computation (default: 10)
- `--sig-level`: Signature truncation level/depth (default: 3)
- `--seed`: Random seed for reproducibility (default: 42)
- `--augmentation`: Augmentation method: `none`, `time`, `basepoint`, `addtime`, `all` (default: all)
- `--compare-augmentations`: Flag to compare all augmentation methods
- `--output-dir`: Output directory for results (default: outputs/signature_volatility)

### 4. Updated Dependencies

**File:** `requirements.txt`

Added `sktime>=0.24.0` for signature-based features.

## Signature Method Parameters

### Template Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lags` | int | 5 | Number of time steps (window size) for signature computation |
| `sig_level` | int | 2 | Signature truncation depth (higher = more features) |
| `source` | categorical | "tiingo" | Data source |
| `table` | categorical | "market.ohlcv" | Data table |
| `price_col` | categorical | "close" | Price column to use |
| `augmentation_list` | categorical | "all" | Type of path augmentation |

### Augmentation Methods

1. **`"none"`**: No augmentation - computes signatures directly from the path
2. **`"time"` or `"addtime"`**: Adds a time dimension to the path (useful for capturing temporal dynamics)
3. **`"basepoint"`**: Adds a basepoint to make the path start from origin (translation invariance)
4. **`"all"`**: Applies both basepoint and addtime augmentations (maximum feature extraction)

### Signature Level (Depth)

The signature level controls the truncation depth:
- **Level 1**: First-order terms (linear features)
- **Level 2**: Up to second-order terms (captures pairwise interactions)
- **Level 3**: Up to third-order terms (captures three-way interactions)
- **Level 4+**: Higher-order terms (richer representation, but exponentially more features)

**Trade-off:** Higher levels create more features but also increase computational cost and risk of overfitting.

## Example Usage in Code

```python
from volatility_forecast.features.signature_features import SignatureFeaturesTemplate
from alphaforge.features.dataset_spec import FeatureRequest

# Create feature request with signature features
signature_features = FeatureRequest(
    template=SignatureFeaturesTemplate(),
    params={
        "lags": 10,              # 10-day window
        "sig_level": 3,          # Depth 3 signatures
        "source": "tiingo",
        "table": "market.ohlcv",
        "price_col": "close",
        "augmentation_list": "all",  # Use all augmentations
    },
)
```

## Understanding Signature Features

Signatures are a mathematical tool from rough path theory that extract features from paths (time series). They have several useful properties:

1. **Universal Approximation**: Can approximate any continuous functional on paths
2. **Geometric Invariance**: Invariant to time reparameterization
3. **Feature-rich**: Captures interactions at multiple orders
4. **Interpretable Terms**: Each signature term has a geometric interpretation

For volatility forecasting, signatures applied to squared returns can capture:
- The overall magnitude of volatility
- The path-dependent nature of volatility clustering
- Higher-order interactions between past volatility shocks
- Complex temporal patterns

## Performance Considerations

- **Feature Count**: Grows exponentially with signature level
  - Level 1: O(d) features
  - Level 2: O(d²) features  
  - Level 3: O(d³) features
  - where d = number of channels (augmented path dimensions)

- **Computational Cost**: 
  - Signature computation: O(n × lags × sig_level)
  - For typical values (n=1000, lags=10, sig_level=3), computation is fast (<1 second)

- **Recommended Starting Points**:
  - `lags=5-10`, `sig_level=2-3` for initial experiments
  - `lags=10-20`, `sig_level=3-4` for more sophisticated models
  - Use augmentation comparison mode to find best augmentation

## Output Files

When running the example script, the following outputs are generated:

### Single Augmentation Mode:
- `model_results.csv`: Model performance metrics (RMSE, MAE, MedAE)

### Comparison Mode (`--compare-augmentations`):
- `augmentation_comparison.csv`: Detailed comparison across augmentations
- `rmse_by_augmentation.png`: Bar plot of RMSE by augmentation
- `mae_by_augmentation.png`: Bar plot of MAE by augmentation  
- `medae_by_augmentation.png`: Bar plot of MedAE by augmentation

## Installation

```bash
# Install/upgrade sktime
pip install -r requirements.txt

# Or install sktime directly
pip install sktime>=0.24.0
```

## Next Steps

1. **Run the tests** to verify the implementation:
   ```bash
   pytest tests/test_signature_features.py -v
   ```

2. **Try the example script** to see signature features in action:
   ```bash
   python examples/signature_in_volatility_forecast.py --compare-augmentations
   ```

3. **Experiment with parameters**:
   - Try different `sig_lags` values (window sizes)
   - Test different `sig_level` depths
   - Compare augmentation methods for your specific dataset

4. **Integration**: Add signature features to your existing volatility forecasting pipeline by including `SignatureFeaturesTemplate` in your `FeatureRequest` list

## References

- sktime documentation: https://www.sktime.net/
- Signature methods in machine learning: Kidger et al. (2020) "Deep Signature Transforms"
- Rough path theory: Lyons (1998) "Differential equations driven by rough signals"

## Questions or Issues?

If you encounter any issues or have questions about the signature features implementation, please check:
1. That sktime is properly installed (`pip list | grep sktime`)
2. The test suite passes (`pytest tests/test_signature_features.py`)
3. The example script runs without errors

Common issues:
- **ImportError for SignatureTransformer**: Install/upgrade sktime (`pip install -U sktime`)
- **Memory issues with high signature levels**: Reduce `sig_level` or `sig_lags`
- **Slow computation**: Reduce dataset size, signature level, or window size
