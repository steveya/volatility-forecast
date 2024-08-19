# volatility-forecast
This repository contains the [notebook](https://github.com/steveta/volatility-forecast/blob/main/volatility_forecast/notebooks/stes_volatility_forecast.ipynb) that produced the blog posts [Volatility Forecasts](https://steveya.github.io/tags/volatility-forecast/). 

The repo also contains a package that aims to wrap all the volatility forecasting models into a single package that is compatible with the scikit-learn API. This repo also contains a work-in-progress Flask app that will produce daily volatility forecasts from each model (and new models.) The app package is currently under development and is not ready for use. 

## Executive Summary

### Volatility Forecasts: Part 1 - Baseline Model

In the first blog post, we explore traditional and advanced volatility forecasting methods. We focus on Exponential Smoothing (ES) and its advanced variant, Smooth Transition Exponential Smoothing (STES). STES adapts better to market shocks by using transition variables to modulate the smoothing parameter.

#### Key Findings

- **STES** generally outperforms **ES** on real market data (SPY returns).
- The performance is measured by Root Mean Squared Error (RMSE).

#### Results

| Model | Test RMSE | Train RMSE |
| --- | --- | --- |
| STES-E&AE&SE | 4.48e-04 | 4.98e-04 |
| STES-AE&SE   | 4.49e-04 | 4.96e-04 |
| STES-E&SE    | 4.50e-04 | 4.95e-04 |
| STES-E&AE    | 4.52e-04 | 4.93e-04 |
| ES           | 4.64e-04 | 4.99e-04 |

### Volatility Forecasts: Part 2 - XGBoost-STES

The second blog post investigates using XGBoost to enhance the STES model by better capturing non-linear dependencies in volatility time series. We replace the linear transition function in STES with a tree ensemble model.

#### Key Findings

- **XGBoost-STES** shows potential improvements in test RMSE.
- Performance varies significantly based on hyperparameter tuning.

#### Results

| Model | Test RMSE | Train RMSE |
| --- | --- | --- |
| XGB-STES (Untuned)  | 4.41e-04 | 5.01e-04 |
| XGB-STES (Sklearn Tuned)  | 4.43e-04 | 5.20e-04 |
| STES-AE&SE | 4.49e-04 | 4.96e-04 |
| STES-E&SE  | 4.50e-04 | 4.95e-04 |
| STES-E&AE  | 4.52e-04 | 4.93e-04 |
| ES         | 4.64e-04 | 4.99e-04 |
| XGB-STES (Optuna Tuned)  | 6.92e-04 | 6.93e-04 |

#### Conclusion

While the XGBoost-STES model can outperform simpler models, its performance depends heavily on tuning strategies. Further extensions and improvements are being explored to enhance volatility forecasting and develop robust trading strategies.
