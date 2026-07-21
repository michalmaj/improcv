"""Shared argument validation helpers for improcv's public functions."""

from __future__ import annotations

import math
import numbers
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


def require_real_number(value: object, name: str) -> None:
    """Raise TypeError unless `value` is a real number.

    Accepts plain Python `int`/`float` as well as NumPy scalar types
    (`np.float32`, `np.float64`, `np.int32`, ...) — anything registered as
    `numbers.Real` — but rejects `bool` (a `bool` is technically an `int`
    subclass, but accepting `True`/`False` here would silently
    misinterpret a boolean argument as ``1``/``0``) and non-numeric types
    such as `str`.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")


def _is_nan_or_inf(value: numbers.Real) -> bool:
    # float(value) normalizes any numbers.Real (including NumPy scalar
    # types like np.float32, which math.isnan/math.isinf don't accept
    # directly on some platforms) before checking finiteness.
    return not math.isfinite(float(value))


def require_positive(value: object, name: str) -> None:
    """Raise TypeError unless `value` is a real number, then ValueError unless
    it's finite and positive."""
    require_real_number(value, name)
    assert isinstance(value, numbers.Real)  # narrows for the type checker
    if _is_nan_or_inf(value):
        raise ValueError(f"{name} must be finite, got {value}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def require_non_negative(value: object, name: str) -> None:
    """Raise TypeError unless `value` is a real number, then ValueError unless
    it's finite and non-negative."""
    require_real_number(value, name)
    assert isinstance(value, numbers.Real)  # narrows for the type checker
    if _is_nan_or_inf(value):
        raise ValueError(f"{name} must be finite, got {value}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def require_finite(value: object, name: str) -> None:
    """Raise TypeError unless `value` is a real number, then ValueError unless
    it's finite (not NaN or infinite).

    Unlike `require_positive`/`require_non_negative`, this carries no sign
    constraint — for parameters where negative values are meaningful (e.g.
    a brightness delta) but NaN/infinity are not.
    """
    require_real_number(value, name)
    assert isinstance(value, numbers.Real)  # narrows for the type checker
    if _is_nan_or_inf(value):
        raise ValueError(f"{name} must be finite, got {value}")


def require_int(value: object, name: str) -> None:
    """Raise TypeError unless `value` is an int.

    Rejects `bool` (a `bool` is technically an `int` subclass in Python,
    but accepting `True`/`False` here would silently misinterpret a
    boolean argument as ``1``/``0``) and any non-int type, including
    floats — so this also rejects NaN and infinity, which are float-only
    concepts.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")


def require_integral(value: object, name: str) -> None:
    """Raise TypeError unless `value` is an integral number.

    Accepts plain Python `int` as well as NumPy integer scalar types
    (`np.int32`, `np.int64`, ...) — anything registered as `numbers.Integral`
    — but rejects `bool` (a `bool` is technically an `int` subclass, and
    therefore also registers as `numbers.Integral`, but accepting
    `True`/`False` here would silently misinterpret a boolean argument as
    `1`/`0`) and non-integral types such as `float`.

    Unlike `require_int`, which only accepts plain `int`, this accepts any
    `numbers.Integral` — for parameters where a coordinate might legitimately
    come straight out of a NumPy array (e.g. a centroid or contour point).
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")


def require_bool(value: object, name: str) -> None:
    """Raise TypeError unless `value` is an actual `bool`.

    OpenCV's Python bindings loosely coerce several types (including `int`
    and even `None`) into a boolean parameter — this rejects that before it
    reaches OpenCV, so a caller's mistake surfaces as a clear error instead
    of silently-wrong behavior.
    """
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")


def require_positive_int(value: object, name: str) -> None:
    """Raise TypeError unless `value` is an int, then ValueError unless it's positive."""
    require_int(value, name)
    assert isinstance(value, int)  # narrows for the type checker; require_int already enforced this
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def require_non_negative_int(value: object, name: str) -> None:
    """Raise TypeError unless `value` is an int, then ValueError unless it's non-negative."""
    require_int(value, name)
    assert isinstance(value, int)  # narrows for the type checker; require_int already enforced this
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def require_size_2d(value: object, name: str) -> None:
    """Raise ValueError/TypeError unless `value` is a 2-tuple of positive ints.

    For ``(width, height)``-style parameters (e.g. `output_size`), where
    a wrong-length tuple would otherwise reach an `IndexError` or a raw
    `cv2.error` deep inside OpenCV instead of a clear library error.
    """
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{name} must be a 2-tuple, got {value!r}")
    width, height = value
    require_positive_int(width, f"{name}[0]")
    require_positive_int(height, f"{name}[1]")


def require_point_2d(value: object, name: str) -> None:
    """Raise ValueError/TypeError unless `value` is a 2-tuple of finite real numbers.

    For ``(x, y)``-style parameters (e.g. `rotate`'s `center`), where a
    wrong-length tuple would otherwise reach an `IndexError` (too short)
    or a raw `cv2.error`/`TypeError` deep inside OpenCV (too long).
    """
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{name} must be a 2-tuple, got {value!r}")
    x, y = value
    require_finite(x, f"{name}[0]")
    require_finite(y, f"{name}[1]")


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


def require_transform_matrix(
    matrix: np.ndarray, shape: tuple[int, int], name: str = "matrix"
) -> None:
    """Raise ValueError/TypeError unless `matrix` is a finite float array of `shape`.

    Checks shape, then dtype (``float32``/``float64`` — an ``int32``
    matrix reaches a raw ``cv2.error``), then finiteness (a ``NaN`` in the
    matrix does not error at all; it silently produces a black image).
    Deliberately does not cast a wrong-dtype matrix for the caller — a
    silent cast could paper over the caller's own mistake (e.g. building
    the matrix from integer inputs by accident).
    """
    if matrix.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {matrix.shape}")
    require_dtype(matrix, (np.float32, np.float64), name)
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")


def require_channels(image: np.ndarray, channels: int) -> None:
    """Raise ValueError unless `image` has exactly `channels` channels and is non-empty.

    The emptiness check matters on its own: a 3-channel *empty* image
    (e.g. shape ``(0, 10, 3)``) previously passed this check and reached
    a raw ``cv2.error`` at the actual OpenCV call site.
    """
    if image.ndim != 3 or image.shape[2] != channels:
        raise ValueError(
            f"image must have {channels} channels with shape (H, W, {channels}), "
            f"got shape {image.shape}"
        )
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(f"image must not be empty, got shape {image.shape}")


def require_odd(value: int, name: str) -> None:
    """Raise ValueError unless `value` is odd."""
    if value % 2 == 0:
        raise ValueError(f"{name} must be odd, got {value}")


def require_spatial_mask(mask: np.ndarray, image: np.ndarray, name: str = "mask") -> None:
    """Raise ValueError/TypeError unless `mask` is a valid spatial mask for `image`.

    `mask` must be uint8, 2D, and match `image`'s spatial size (H, W) --
    regardless of `image`'s channel count or dtype, since a mask selects
    pixel positions, not per-channel values. Unlike
    `require_same_shape_and_dtype`, which wrongly demands full shape+dtype
    equality between two same-kind images.
    """
    require_dtype(mask, (np.uint8,), name)
    require_image_ndim(mask, ndims=(2,))
    spatial_shape = image.shape[:2]
    if mask.shape != spatial_shape:
        raise ValueError(
            f"{name} must have shape {spatial_shape} (matching image's spatial size), "
            f"got {mask.shape}"
        )


def require_positive_integral(value: object, name: str) -> None:
    """Raise TypeError unless `value` is an integral number, then ValueError unless it's positive.

    Unlike `require_positive_int`, which only accepts a plain Python `int`,
    this accepts any `numbers.Integral` (including NumPy integer scalars) --
    for parameters like a histogram bin count that may legitimately arrive
    as `np.int32`.
    """
    require_integral(value, name)
    assert isinstance(value, numbers.Integral)  # narrows for the type checker
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def require_range(value: object, low: float, high: float, name: str) -> None:
    """Raise TypeError unless `value` is a real number, then ValueError unless
    `low <= value <= high`."""
    require_real_number(value, name)
    assert isinstance(value, numbers.Real)  # narrows for the type checker
    if not low <= float(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}, got {value}")


def require_fits_dtype(value: object, dtype: np.dtype | type, name: str) -> None:
    """Raise ValueError unless `value` fits within `dtype`'s representable range.

    For parameters like `threshold`'s `max_value` that OpenCV silently
    saturates rather than rejects when out of range for the image's
    integer dtype (e.g. ``300`` silently becomes ``255`` for a ``uint8``
    image) — verified directly against ``cv2.threshold``. Floating-point
    dtypes have no meaningful bound here and are skipped.
    """
    require_real_number(value, name)
    assert isinstance(value, numbers.Real)  # narrows for the type checker
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        if not info.min <= float(value) <= info.max:
            raise ValueError(
                f"{name} must fit within the range of {np.dtype(dtype).name} "
                f"([{info.min}, {info.max}]), got {value}"
            )


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
