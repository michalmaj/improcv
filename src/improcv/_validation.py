"""Shared argument validation helpers for improcv's public functions."""

from __future__ import annotations

import numpy as np

__all__: list[str] = []


def require_image_ndim(image: np.ndarray, ndims: tuple[int, ...] = (2, 3)) -> None:
    """Raise ValueError unless `image.ndim` is one of `ndims`."""
    if image.ndim not in ndims:
        allowed = " or ".join(str(n) for n in ndims)
        raise ValueError(f"image must have {allowed} dimensions, got {image.ndim}")


def require_positive(value: float, name: str) -> None:
    """Raise ValueError unless `value` is positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def require_non_negative(value: float, name: str) -> None:
    """Raise ValueError unless `value` is non-negative."""
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def require_channels(image: np.ndarray, channels: int) -> None:
    """Raise ValueError unless `image` has exactly `channels` channels."""
    if image.ndim != 3 or image.shape[2] != channels:
        raise ValueError(
            f"image must have {channels} channels with shape (H, W, {channels}), "
            f"got shape {image.shape}"
        )
