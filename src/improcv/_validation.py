"""Shared argument validation helpers for improcv's public functions."""

from __future__ import annotations

import math
from collections.abc import Collection

import numpy as np

__all__: list[str] = []


def require_image_ndim(image: np.ndarray, ndims: tuple[int, ...] = (2, 3)) -> None:
    """Raise ValueError unless `image.ndim` is one of `ndims` and `image` is non-empty.

    Every public function calls this (directly or via a narrower `ndims`),
    so the empty-image check here is the single, global place that rejects
    a zero-height or zero-width image for the whole library.
    """
    if image.ndim not in ndims:
        allowed = " or ".join(str(n) for n in ndims)
        raise ValueError(f"image must have {allowed} dimensions, got {image.ndim}")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(f"image must not be empty, got shape {image.shape}")


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


def require_dtype(image: np.ndarray, dtypes: tuple[type, ...], name: str = "image") -> None:
    """Raise TypeError unless `image.dtype` is one of `dtypes`.

    For functions backed by an OpenCV call that only supports specific
    dtypes (e.g. ``cv2.equalizeHist`` requires 8-bit input) and would
    otherwise raise a raw, unfriendly ``cv2.error``.
    """
    if not any(image.dtype == dtype for dtype in dtypes):
        allowed = ", ".join(np.dtype(dtype).name for dtype in dtypes)
        raise TypeError(f"{name} must have dtype in ({allowed}), got {image.dtype}")


def require_channels(image: np.ndarray, channels: int) -> None:
    """Raise ValueError unless `image` has exactly `channels` channels."""
    if image.ndim != 3 or image.shape[2] != channels:
        raise ValueError(
            f"image must have {channels} channels with shape (H, W, {channels}), "
            f"got shape {image.shape}"
        )


def require_odd(value: int, name: str) -> None:
    """Raise ValueError unless `value` is odd."""
    if value % 2 == 0:
        raise ValueError(f"{name} must be odd, got {value}")


def require_range(value: float, low: float, high: float, name: str) -> None:
    """Raise ValueError unless `low <= value <= high`."""
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}, got {value}")


def require_same_shape_and_dtype(
    image_a: np.ndarray, image_b: np.ndarray, name_a: str = "image_a", name_b: str = "image_b"
) -> None:
    """Raise ValueError/TypeError unless `image_a` and `image_b` share shape and dtype.

    Mismatched dtype passed uncaught into OpenCV's element-wise ops (e.g.
    ``cv2.bitwise_and``, ``cv2.addWeighted``) surfaces as a raw, unfriendly
    ``cv2.error`` rather than a clear library error.
    """
    if image_a.shape != image_b.shape:
        raise ValueError(
            f"{name_a} and {name_b} must have the same shape, got "
            f"{image_a.shape} and {image_b.shape}"
        )
    if image_a.dtype != image_b.dtype:
        raise TypeError(
            f"{name_a} and {name_b} must have the same dtype, got "
            f"{image_a.dtype} and {image_b.dtype}"
        )
