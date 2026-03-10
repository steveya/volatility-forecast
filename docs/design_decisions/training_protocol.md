# Training Protocol and Evaluation Parity

This document outlines the standard training protocols for `volatility-forecast`, specifically focusing on how structural neural networks like `VolGRU` are evaluated against statistical benchmarks like STES.

## The Overfitting Challenge

Unlike pure statistical models or standard GARCH, Deep-learning representations like the VolGRU model are highly over-parameterized. This exposes them significantly to the risk of "memorizing noise" from the training data. If we let the models train entirely to convergence using Mean Squared Error loss mapping onto stochastic individual squared returns $r_t^2$, the In-Sample bounds drop aggressively while the Out-Of-Sample prediction capabilities explode.

## Early Stopping Mechanics

To safeguard against overfitting, our models leverage validation-based **Early Stopping**. We hold a portion of the data out (configured via `val_fraction=0.15`), and monitor the exact validation loss dynamically at each epoch. PyTorch's internal optimization (like Adam) inherently bounces through financial noise, meaning we instruct the training mechanism to forcefully stop updating parameters when validation loss stops improving over a fixed patience window.

### The Asymmetry Issue

If we merely use `val_fraction=0.15` during our final fit for the testing framework:
1.  **STES Pipeline**: Optimizes and converges on 100% of the training segment using precision functions (e.g. Scipy's L-BFGS-B).
2.  **Naive Neural Pipeline**: Leaves the final 15% out specifically to determine exactly when validation bottoms out, blinding the model effectively to recent market history immediately preceding the "Out Of Sample" period.

This inherently handicaps neural architectures in straight benchmark runs.

## The XGBoost-Style Refit Protocol

To ensure true apples-to-apples validation metrics and empirical bounds against statistical baselines:
1.  **Cross-Validation**: We pass the `val_fraction=0.15` configuration strictly inside the iterative `run_cv` loop structure. By monitoring validation runs across folds, we harvest precisely which epochs halted validation early. We capture both the *best hyper-parameter regularizers* and the **average optimal epoch stop**.
2.  **Final Base Refit**: The model dynamically overrides configuration parameter caps: it executes on `val_fraction=0.0`, processing 100% of the data exactly like STES, using a hard `max_epochs` value mapped from the average optimum validation calculation in CV.

This framework successfully suppresses neural overfitting vulnerabilities whilst preserving fair training window evaluations parity against GARCH / STES base regressions in any Time Series evaluation framework.
