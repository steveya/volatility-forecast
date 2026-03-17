# AGENTS

## Purpose

You are working in the `volatility-forecast` repository. This repo owns canonical volatility-model code, reusable experiment drivers, evaluation logic, and tests. For the cross-repo workflow, model implementations belong here, not in the blog repo.

This repo supports one long-running volatility forecasting workflow rather than a single PGARCH subproject. The current series frontier spans STES, XGBSTES, GARCH benchmarking under QLIKE, PGARCH variants, expanded feature work, and later VolGRU-style relaxations.

## Scope

This file applies to the repo. Keep compatibility with alphaforge conventions and avoid unnecessary refactors.

## Environment

- OS: macOS
- Python: use conda env `py312`

## Local .env

- Keep environment variables in a `.env` file at the repo root.
- Common keys: `TIINGO_API`, `TIINGO_CACHE_DIR`, `TIINGO_CACHE_MODE`.

## Layout

- `volatility_forecast/model/` contains the core model implementations.
- `volatility_forecast/features/`, `targets/`, `evaluation/`, and `reporting/` hold reusable pipeline pieces.
- `examples/` contains runnable experiment drivers and benchmark scripts.
- `tests/` contains regression and unit coverage.
- `docs/design_decisions/training_protocol.md` documents the preferred overfitting-control protocol for neural or high-capacity models.
- `outputs/` is for local experiment artifacts and summaries, not for canonical blog ownership.

## Commands

- Run tests: `conda run -n py312 pytest -q`
- If the exact experiment command is unclear, leave `TODO(user): add canonical experiment command` instead of inventing one.

## What to do

- Preserve public interfaces where possible. If you change constructor arguments, fit or predict behavior, output schemas, or shared utility semantics, update tests and affected examples in the same change.
- Keep code modular. Promote reusable logic into package modules rather than burying it in notebooks or one-off scripts.
- Prefer small, composable changes over broad refactors.
- Treat model work as part of the broader volatility-series workflow, not as an isolated family-specific branch.
- Record series rung, baseline family, candidate family, decision, and next branch in `../codex-workflows/volatility-forecasts-log.md` for substantive research iterations.
- Separate experiment axes when possible:
  - feature changes should hold regularization and structure fixed
  - regularization changes should hold features and structure fixed
  - structural model changes should begin from a frozen feature tier and loss definition
- Keep tests aligned with agreed conventions and update them when behavior intentionally changes.

## Experiment Protocol

- Start from an explicit baseline before introducing a candidate.
- Name the baseline precisely: for example `GARCH(1,1)`, `STES`, `XGBSTES`, `PGARCH-L`, `XGB-g-PGARCH`, `XGBPGARCHModel`, or a `VolGRU` variant.
- Hold the sample window, target, loss, and split constant while comparing a candidate to its baseline.
- Compare serious candidates against `GARCH(1,1)` and the current family baseline when the series stage requires both.
- Record whether the candidate changed:
  - features
  - regularization
  - structure
- Track three states for each branch:
  - baseline
  - candidate
  - best-so-far

## Overfitting Control

- Do not tune on the test segment.
- Use time-series validation or the documented early-stopping/refit protocol for high-capacity models.
- Report both point metrics and calibration-style diagnostics when they materially affect the conclusion.
- Treat gains that appear only after repeated test-set peeking as invalid.
- Prefer pruning weak directions quickly rather than expanding a high-capacity search with no validation support.

## Conventions

- Fixed-split evaluation aligns with `examples/volatility_forecast_2.py`:
  - Train: `[is_index : os_index)`
  - Test: `[os_index : ]`
- STES prediction is aligned to next-step variance; compare to shifted target when needed.

## Data and PIT

- `asof_utc` should be present by default in data flowing through the pipeline.

## Caching

- Tiingo caching is pluggable via alphaforge cache backends.
- Use CLI flags for cache control when available.

## What not to do

- Do not copy blog prose or blog-specific notebook glue into the core package unless it becomes reusable code.
- Do not ship one-off hacks that only work for a single notebook run.
- Do not silently change train/test alignment, target alignment, or metric definitions.
- Do not treat `outputs/` artifacts as a stable API contract.
- Do not make avoidable interface changes without updating the dependent examples and tests.

## Validation

- Run targeted tests for the modules you changed.
- Run broader regression coverage when shared model interfaces, pipeline helpers, or evaluation utilities changed.
- If you changed an example-backed workflow, run the smallest reproducible command that exercises it.
- Verify that metrics, path outputs, and reported baselines line up with the actual implementation.

## Deliverables

- Code changes in reusable modules or explicit experiment drivers.
- Updated tests for intentional behavior changes or new model behavior.
- A concise record of series rung, baseline, candidate, and best-so-far when the work is part of a search cycle.
- Clear validation notes: run, skipped, or blocked.
