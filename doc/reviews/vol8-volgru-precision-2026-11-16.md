# VOL-8: VolGRU Torch/STES Gate Precision Parity (`VolGRUConfig.dtype`)

Review date: 2026-11-16, America/New_York.
Linear issue: `VOL-8`

## Root Cause

Commit `4dcacf7` switched `VolGRUModel` Torch initialization and tensor casting
from `torch.float64` to `torch.float32`, while `STESModel` remained double
precision (`np.float64`) and `tests/test_volgru_reduces_to_stes.py` still
asserted `rtol=0.0, atol=1e-10`.

On small-scale synthetic returns (`N(0, 0.01^2)`), variance values
($\sigma_t^2 \sim 10^{-4}$) kept `float32` rounding errors below `1e-10`, whereas
the logistic update gate ($z_t \sim 0.5$) carried $\approx 3 \times 10^{-8}$
`float32` rounding error relative to `float64` `STESModel`, failing
`test_volgru_torch_reduces_to_stes_exactly`.

## Design & Numerical Contract

- **`VolGRUConfig.dtype: Literal["float32", "float64"] = "float32"`**
  (`volatility_forecast/model/volgru_config.py`):
  - Default `"float32"` preserves backward-compatible training throughput and
    `np.float32` prediction outputs.
  - Because `VolGRUConfig` uses `@dataclass(slots=True)`, `VolGRUConfig.__setstate__`
    backfills any missing dataclass defaults (including `dtype="float32"` and
    `gate_feature_group_lambda=0.0`) so unpickling pre-`VOL-8` `.joblib`
    artifacts supports `config.dtype`, `repr`, `get_params()`,
    `dataclasses.asdict()`, and equality comparisons cleanly.
  - `VolGRUModel.__setstate__` coordinates with `_LEGACY_DTYPE_BACKFILLED_IDS`
    (cleaned up via `__post_init__` / `__del__`) so legacy pre-`4dcacf7`
    artifacts pickled with `torch.float64` weights and no `dtype` slot retain
    `config.dtype = "float64"` instead of being silently downcast on load, while
    post-`4dcacf7` / pre-`VOL-8` `float32` artifacts load as `"float32"`.
  - Explicit `"float64"` configures double-precision Torch parameters and
    sequence recursion across `VolGRUModel` (`_torch_dtype`,
    `_sync_torch_module_dtype`, `_ensure_torch_module`, `set_gate_beta`,
    `get_gate_beta`, `_fit_torch`, `_predict_torch`) and direct
    `VolGRUCellTorch` / `VolGRUModelTorch` construction (`dtype=None` resolves
    from `config.dtype`), plus `VolGRUCellTorch.forward_sequence` input casting
    against `self.gate_beta.dtype` / `self.gate_beta.device`.
- **`float32` Saturated Gate-Entropy Safety**
  (`volatility_forecast/model/volgru_utils.py`):
  - `gate_entropy_term_torch` clamps `safe_eps = min(max(float(eps), float(torch.finfo(z.dtype).eps)), 0.25)`
    when `z` is a floating-point tensor, preventing `1.0 - 1e-12` from
    rounding to `1.0` in `float32` and producing `0.0 * (-inf) = NaN` in
    loss/gradients when `gate_entropy_lambda > 0` and gates saturate.
- **Precision Parity & Boundaries** (`tests/test_volgru_reduces_to_stes.py`):
  - `dtype="float64"` matches `STESModel` to `rtol=0.0, atol=1e-12` across
    return scales `[0.01, 1.0]` when variance stays above `VolGRUConfig.eps`
    (verified with discriminating `1e-9` `gate_beta` perturbation).
  - `dtype="float32"` matches `STESModel` to `rtol=1e-6, atol=1e-6` in the
    STES-reduced configuration at moderate logits and matches a pure-NumPy
    `float32` STES reference to `rtol=0.0, atol=5e-7` (verified with
    discriminating `1e-4` perturbation).
  - Zero-return initialization (`returns == 0`, `init_var == 0.0`) explicitly
    verifies the documented semantic difference between `VolGRUModel` clamping
    ($\sigma^2 \ge \text{config.eps}$) and unclamped `STESModel` ($0.0$).
  - Covers `dtype` validation (`"float16"` -> `ValueError`), end-to-end `float64`
    `fit`/`predict`/`predict_with_gates`/`run_cv`/`save`/`load` round-trips,
    both pre-`VOL-8` (`float32`) and pre-`4dcacf7` (`float64`) pickled `.joblib`
    artifacts missing `dtype`/`gate_feature_group_lambda` slots, `config.dtype`
    mutation sync across `get_gate_beta()` / `predict()`, `float32` saturated
    `gate_entropy_term_torch`, direct `VolGRUCellTorch`/`VolGRUModelTorch`
    `float64` construction, general `gru_linear` + `mlp_pos` + `qlike` +
    `perform_cv=True` `float64` fit, and Torch-absent `ImportError`.

## CI Verification Workflow

Added `volgru-torch` job to `.github/workflows/ci.yml` (Ubuntu, Python `3.12`)
installing CPU `torch` (`https://download.pytorch.org/whl/cpu`) and `.[dev]`,
running:

```bash
pytest tests/test_stes_*.py tests/test_volgru_*.py -v
```

Keeping `tests/test_stes_*.py` before `tests/test_volgru_*.py` matches default
pytest discovery order and avoids invoking SciPy L-BFGS-B after Torch OpenMP
initialization (which segfaulted inside `scipy/optimize/_lbfgsb_py.py:461` on
macOS ARM64 `py312` with `torch 2.11.0.dev` + `scipy 1.16.3`).

## Local Verification Runs

- `py312` (`/Users/steveyang/miniforge3/envs/py312/bin/python`, Python 3.12.2,
  STES + VolGRU slice, post-final edit): `37 passed in 21.75s`.
- `base` (`/Users/steveyang/miniforge3/bin/python`, Python 3.10.6, STES +
  VolGRU + Synapse definition slice, post-final edit): `89 passed in 24.02s`.
- `py312` full suite prior to test import-order formatting edit:
  `311 passed, 7 skipped in 50.99s` (1 `TIINGO_API not set` skip in
  `tests/test_data.py`, 6 `synapse` skips in `tests/test_definition_slice.py`).
- `ruff check --no-cache`: `volgru_config.py`, `volgru_torch.py`,
  `volgru_utils.py`, and `tests/test_volgru_reduces_to_stes.py` clean
  (`All checks passed!`), and `ruff check --no-cache --select F,I` clean on
  `volgru_model.py` (pre-existing `E402` lines 16-17 untouched).

## Independent Tier 1 Review

- Reviewer conversation: `882acea2-784b-4dd8-bab6-4c286e37251d`
- Findings & dispositions:
  - **P1 #1 (Legacy `.joblib` / `slots=True` compatibility + pre-`4dcacf7` `float64` downcast):** Fixed via `VolGRUConfig.__setstate__` slot/default backfill, `_LEGACY_DTYPE_BACKFILLED_IDS` tracking with `__post_init__`/`__del__` cleanup, `VolGRUModel.__setstate__` pre-`4dcacf7` `float64` detection/preservation, and pickled legacy `.joblib` tests in `tests/test_volgru_reduces_to_stes.py`.
  - **P1 #2 (`gate_entropy_term_torch` `NaN` on saturated `float32` gates):** Fixed via `torch.finfo(z.dtype).eps` lower clamp bound in `gate_entropy_term_torch` and verified by finite forward/backward + `VolGRUModel.fit` regression test.
  - **P2 #1 (`VolGRUCellTorch`/`VolGRUModelTorch` ignoring `config.dtype` when `dtype` omitted):** Fixed via `_resolve_torch_dtype` default `dtype=None` and verified in test.
  - **P2 #2 (`_sync_torch_module_dtype` parameter coverage & `get_gate_beta()` sync):** Fixed to scan all `.parameters()` and floating `.buffers()` and sync inside `get_gate_beta()`; mutation test added.
  - **P2 #3 (Additional test coverage):** Added tests for pickled legacy `.joblib` roundtrips, `config.dtype` mutation sync, saturated `float32` entropy, direct Torch module `float64` init, `perform_cv=True` non-STES `float64` fit, and Torch-absent `ImportError`.
  - **P2 #4 (Doc clarifications & env labels):** Updated `docs/index.md` scope qualification and corrected env labels (`base` Python 3.10.6 vs `py312`) in this ledger.
- Delta outcome: `PASS ([DELTA_OK])`
