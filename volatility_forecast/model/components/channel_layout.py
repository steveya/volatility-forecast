"""Channel-routing helpers for PGARCH-family models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

VALID_PGARCH_CHANNELS = ("mu", "phi", "g")


@dataclass(frozen=True, slots=True)
class ChannelLayout:
    """Per-channel feature routing expressed in augmented-column space."""

    mu_aug_cols: np.ndarray
    phi_aug_cols: np.ndarray
    g_aug_cols: np.ndarray

    def __post_init__(self) -> None:
        for name in VALID_PGARCH_CHANNELS:
            value = np.asarray(getattr(self, f"{name}_aug_cols"), dtype=int)
            if value.ndim != 1:
                raise ValueError(
                    f"{name}_aug_cols must be 1-D, got shape {value.shape}."
                )
            object.__setattr__(self, f"{name}_aug_cols", value)

    @property
    def n_mu(self) -> int:
        return int(self.mu_aug_cols.size)

    @property
    def n_phi(self) -> int:
        return int(self.phi_aug_cols.size)

    @property
    def n_g(self) -> int:
        return int(self.g_aug_cols.size)

    def augmented_columns(self, channel: str) -> np.ndarray:
        if channel not in VALID_PGARCH_CHANNELS:
            raise ValueError(f"Unsupported PGARCH channel {channel!r}.")
        return np.asarray(getattr(self, f"{channel}_aug_cols"), dtype=int)

    def feature_columns(self, channel: str) -> np.ndarray:
        aug = self.augmented_columns(channel)
        return np.asarray(aug[1:] - 1, dtype=int)


def build_channel_layout(
    n_features: int,
    channel_features: dict[str, list[int]] | None,
) -> ChannelLayout:
    """Build a channel layout with intercept-aware augmented indices."""

    if int(n_features) < 0:
        raise ValueError("n_features must be non-negative.")

    if channel_features is None:
        all_aug = np.arange(1 + int(n_features), dtype=int)
        return ChannelLayout(
            mu_aug_cols=all_aug,
            phi_aug_cols=all_aug,
            g_aug_cols=all_aug,
        )

    channel_indices: dict[str, np.ndarray] = {}
    for channel in VALID_PGARCH_CHANNELS:
        if channel not in channel_features:
            raise ValueError(f"channel_features must contain key '{channel}'.")
        idx = np.asarray(channel_features[channel])
        if idx.ndim != 1:
            raise ValueError(
                f"channel_features['{channel}'] must be a 1-D sequence of column indices."
            )
        if idx.size and not np.issubdtype(idx.dtype, np.integer):
            raise ValueError(
                f"channel_features['{channel}'] must contain integer column indices."
            )
        if len(idx) > 0 and (idx.min() < 0 or idx.max() >= int(n_features)):
            raise ValueError(
                f"channel_features['{channel}'] indices must be in [0, {n_features})."
            )
        channel_indices[channel] = idx.astype(int, copy=False)

    return ChannelLayout(
        mu_aug_cols=np.concatenate([[0], channel_indices["mu"] + 1]).astype(
            int, copy=False
        ),
        phi_aug_cols=np.concatenate([[0], channel_indices["phi"] + 1]).astype(
            int, copy=False
        ),
        g_aug_cols=np.concatenate([[0], channel_indices["g"] + 1]).astype(
            int, copy=False
        ),
    )
