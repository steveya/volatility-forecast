# PGARCH Model Architecture

This note records the refactor boundary for post-Part-6 PGARCH-family work.

## Why This Refactor Exists

The post-6 search now has two competing needs:

- keep the accepted PGARCH recursion and channel semantics stable
- make it easy to try structural branches such as regime-routing mixtures or attention-style channel weighting

Before this refactor, the boosted PGARCH implementation kept all of these concerns inside one trainer:

- score links and inverse links
- PGARCH recursion
- loss geometry for rowwise gradients
- channel-specific XGBoost controls

That made the code harder to extend cleanly for future structural branches.

## New Split

### `volatility_forecast/model/pgarch_core.py`

Owns reusable PGARCH mechanics:

- bounds and score links
- raw score containers
- recursion state
- loss / score evaluation
- raw-score blending primitives for future expert mixtures or attention weighting

The design goal is that future models can emit raw channel scores and reuse the same PGARCH recursion without reimplementing the variance mechanics.

### `volatility_forecast/model/pgarch_channel_heads.py`

Owns channel-head planning for the boosted PGARCH family:

- base XGBoost head configuration
- per-channel override validation
- per-channel boosting-round planning

This keeps channel-capacity logic out of the trainer itself.

### `volatility_forecast/model/xgb_pgarch_full_model.py`

Now acts more like a training wrapper:

- fit the initializer
- fit or update channel heads
- delegate recursion and score geometry to `PGARCHCore`
- delegate channel-specific head planning to `XGBChannelHeadPlan`

## What This Enables Next

This refactor is intended to support the next structural branch after the current post-6 boosted survivor:

- regime-routing PGARCH mixtures
- attention-style weighting over channel memories or expert branches
- other channel-preserving PGARCH generalizations

The key implementation rule is unchanged:

- preserve explicit `mu`, `phi`, and `g`
- preserve the PGARCH outer recursion
- make new structure live in score generation or expert weighting, not in ad hoc notebook-side rewrites
