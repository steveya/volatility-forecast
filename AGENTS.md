# AGENTS

## Scope
You are working in the volatility-forecast repository. Keep compatibility with alphaforge conventions and avoid unnecessary refactors.

## Environment
- OS: macOS
- Python: use conda env `py312`

## Local .env
- Keep environment variables in a .env file at the repo root.
- Common keys: `TIINGO_API`, `TIINGO_CACHE_DIR`, `TIINGO_CACHE_MODE`.

## Commands
- Run tests: `conda run -n py312 pytest -q`

## Conventions
- Fixed-split evaluation aligns with `volatility_forecast_2.py`:
  - Train: `[is_index : os_index)`
  - Test: `[os_index : ]`
- STES prediction is aligned to next-step variance; compare to shifted target when needed.

## Data & PIT
- `asof_utc` should be present by default in data flowing through the pipeline.

## Caching
- Tiingo caching is pluggable via alphaforge cache backends.
- Use CLI flags for cache control when available.

## Notes
- Prefer minimal edits.
- Keep tests aligned with agreed conventions.