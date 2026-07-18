"""Shared argument validation helpers for improcv's public functions."""

from __future__ import annotations

import math
from collections.abc import Collection

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


def require_positive_int(value: object, name: str) -> None:
    """Raise TypeError unless `value` is an int, then ValueError unless it's positive.

    Rejects `bool` (a `bool` is technically an `int` subclass in Python,
    but accepting `True`/`False` here would silently misinterpret a
    boolean argument as ``1``/``0``) and any non-int type, including
    floats — so this also rejects NaN and infinity, which are float-only
    concepts.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def require_one_of(value: object, allowed: Collection[object], name: str) -> None:
    """Raise ValueError unless `value` is one of `allowed`.

    Intended for runtime-checking string-literal (`Literal[...]`) parameters:
    type checkers only catch invalid values at static-analysis time, so
    every such parameter needs this check to reject bad values passed in
    at runtime (e.g. from user input or untyped call sites).
    """
    if value not in allowed:
        raise ValueError(f"{name} must be one of {tuple(allowed)}, got {value!r}")


def require_channels(image: np.ndarray, channels: int) -> None:
    """Raise ValueError unless `image` has exactly `channels` channels."""
    if image.ndim != 3 or image.shape[2] != channels:
        raise ValueError(
            f"image must have {channels} channels with shape (H, W, {channels}), "
            f"got shape {image.shape}"
        )
