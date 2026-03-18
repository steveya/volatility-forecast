"""Deprecated entry point. Use the CLI instead:

    python -m volatility_forecast run
    python -m volatility_forecast backfill --start 2024-03-01 --end 2026-03-16
    python -m volatility_forecast leaderboard

See ``volatility_forecast.cli`` for the full command reference.
"""

import warnings
from volatility_forecast.cli import main

if __name__ == "__main__":
    warnings.warn(
        "main.py is deprecated. Use 'python -m volatility_forecast' instead.",
        DeprecationWarning,
        stacklevel=1,
    )
    main()
