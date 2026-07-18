"""Shared argument validation helpers for improcv's public functions."""

from __future__ import annotations

import math

import numpy as np

__all__: list[str] = []


def require_image_ndim(image: np.ndarray, ndims: tuple[int, ...] = (2, 3)) -> None:
    """Raise ValueError unless `image.ndim` is one of `ndims`."""
    if image.ndim not in ndims:
        allowed = " or ".join(str(n) for n in ndims)
        raise ValueError(f"image must have {allowed} dimensions, got {image.ndim}")


def _is_nan_or_inf(value: float) -> bool:
    return isinstance(value, float) and (math.isnan(value) or math.isinf(value))


def require_positive(value: float, name: str) -> None:
    """Raise ValueError unless `value` is a finite, positive number."""
    if _is_nan_or_inf(value):
        raise ValueError(f"{name} must be finite, got {value}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def require_non_negative(value: float, name: str) -> None:
    """Raise ValueError unless `value` is a finite, non-negative number."""
    if _is_nan_or_inf(value):
        raise ValueError(f"{name} must be finite, got {value}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def require_channels(image: np.ndarray, channels: int) -> None:
    """Raise ValueError unless `image` has exactly `channels` channels."""
    if image.ndim != 3 or image.shape[2] != channels:
        raise ValueError(
            f"image must have {channels} channels with shape (H, W, {channels}), "
            f"got shape {image.shape}"
        )
