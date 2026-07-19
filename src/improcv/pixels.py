"""Pixel-level operations."""

from __future__ import annotations

from typing import cast

import cv2
import numpy as np

from improcv._validation import (
    require_dtype,
    require_finite,
    require_image_ndim,
    require_non_negative,
    require_range,
    require_same_shape_and_dtype,
)
from improcv.types import Image, ImageU8, Mask

__all__ = [
    "in_range",
    "invert",
    "adjust_brightness",
    "adjust_contrast",
    "alpha_blend",
    "bitwise_and",
    "bitwise_or",
    "apply_lut",
]

# Verified directly against cv2.inRange on OpenCV 4.13 and 5.0. bool and
# float16 are excluded even though each only crashes on *one* of the two
# lines (bool segfaults the interpreter on 5.0 but raises a normal
# cv2.error on 4.13; float16 segfaults on 4.13 but works on 5.0) — since
# improcv must support both lines, either dtype is unsafe everywhere.
# int32/int64 are excluded too: cv2.inRange silently produces wrong
# results for large-magnitude values (verified with +/-5_000_000_000 on
# an int64 image), not merely an unsupported-dtype error, so there is no
# safe subrange to carve out.
_IN_RANGE_DTYPES = (np.uint8, np.uint16, np.int16, np.float32, np.float64)


def in_range(image: Image, lower: tuple[float, ...], upper: tuple[float, ...]) -> Mask:
    """Return a mask of pixels within `[lower, upper]` (inclusive, per channel).

    Parameters
    ----------
    image : np.ndarray
        Input image with shape ``(H, W)`` or ``(H, W, C)``.
    lower, upper : tuple of float
        Inclusive per-channel bounds; each must have exactly one element
        per channel of `image` (1 for a grayscale image, `C` for an
        ``(H, W, C)`` image). A shorter "scalar" bound is deliberately not
        supported: ``cv2.inRange`` does not broadcast it the way one might
        expect (verified directly — it does not simply apply the same
        bound to every channel), so allowing it here would be a silent
        correctness trap rather than a convenience. Each element must be a
        finite real number — a fractional bound is meaningful for a
        ``float32`` image.

    Returns
    -------
    np.ndarray
        A new ``uint8`` array shaped like `image`'s spatial dimensions,
        with values ``0`` or ``255`` — improcv's mask convention (matches
        OpenCV's own native mask representation; see `harris_corner`,
        `threshold`, `auto_canny`).

    Raises
    ------
    ValueError
        If `image` does not have 2 or 3 dimensions, `lower`/`upper` does
        not have exactly one element per channel of `image`, or any
        element of `lower`/`upper` is not finite.
    TypeError
        If any element of `lower`/`upper` is not a real number (rejects
        ``bool`` and ``str`` — both otherwise reach ``cv2.inRange``
        directly: a ``bool`` element is silently reinterpreted as ``1``/
        ``0`` and a ``str`` element reaches a raw ``cv2.error``); or if
        `image` does not have dtype ``uint8``, ``uint16``, ``int16``,
        ``float32``, or ``float64`` (verified against ``cv2.inRange`` on
        both OpenCV 4 and 5 — ``bool`` and ``float16`` each crash the
        interpreter outright, with no exception to catch, on one of the
        two supported OpenCV lines; ``int32``/``int64`` silently produce
        wrong results for large-magnitude values rather than erroring).
    """
    require_image_ndim(image)
    require_dtype(image, _IN_RANGE_DTYPES)
    channels = 1 if image.ndim == 2 else image.shape[2]
    if len(lower) != channels or len(upper) != channels:
        raise ValueError(
            f"lower and upper must each have {channels} element(s) matching "
            f"image's channel count, got {len(lower)} and {len(upper)}"
        )
    for i, bound in enumerate(lower):
        require_finite(bound, f"lower[{i}]")
    for i, bound in enumerate(upper):
        require_finite(bound, f"upper[{i}]")
    # Bounds are always built as float64, regardless of image dtype or the
    # Python/NumPy types in lower/upper: cv2.inRange gives silently wrong
    # results (an all-zero mask) when a float32 image is paired with
    # float32-dtype bounds specifically — verified directly. float64
    # bounds work correctly against every dtype in _IN_RANGE_DTYPES, so
    # this sidesteps that landmine entirely rather than special-casing it.
    lower_array = np.array(lower, dtype=np.float64)
    upper_array = np.array(upper, dtype=np.float64)
    # cv2.inRange always produces uint8 {0, 255}; cv2's stubs don't say so.
    return cast(Mask, cv2.inRange(image, lower_array, upper_array))


def invert(image: ImageU8) -> ImageU8:
    """Invert pixel values (``255 - value``).

    Restricted to ``uint8``: inverting a float image's bit pattern (what
    ``cv2.bitwise_not`` does under the hood) does not correspond to a
    meaningful "inverted image" — it can even produce ``NaN``.

    Raises
    ------
    ValueError
        If `image` does not have 2 or 3 dimensions.
    TypeError
        If `image` does not have dtype ``uint8``.
    """
    require_image_ndim(image)
    require_dtype(image, (np.uint8,))
    # cv2.bitwise_not preserves uint8 here; cv2's stubs don't say so.
    return cast(ImageU8, cv2.bitwise_not(image))


def adjust_brightness(image: ImageU8, delta: float) -> ImageU8:
    """Add `delta` to every pixel value, clamped to the valid 8-bit range.

    Uses saturating (clamping) arithmetic in both directions: a negative
    `delta` that would push a pixel below 0 clamps to 0, it does not wrap
    or reflect back to a positive value (unlike a naive
    ``cv2.convertScaleAbs`` call, whose ``beta`` argument takes the
    absolute value of the result rather than clamping it). A fractional
    `delta` is rounded to the nearest integer, not truncated — truncating
    a positive float towards zero always rounds *down*, which loses up
    to a full unit of intended effect for a positive `delta` while a
    negative `delta` keeps almost all of its effect, an asymmetry with no
    principled justification.

    Restricted to ``uint8`` input: a float image would be truncated to
    integers before `delta` is even applied, silently destroying
    sub-integer data instead of producing a meaningful result.

    Raises
    ------
    ValueError
        If `image` does not have 2 or 3 dimensions, or `delta` is not finite.
    TypeError
        If `image` does not have dtype ``uint8``.
    """
    require_image_ndim(image)
    require_dtype(image, (np.uint8,))
    require_finite(delta, "delta")
    return np.clip(np.round(image.astype(np.float64) + delta), 0, 255).astype(np.uint8)


def adjust_contrast(image: ImageU8, factor: float) -> ImageU8:
    """Scale pixel values by `factor` around the mid-gray point (128).

    Scaling around the midpoint (rather than around 0) keeps average
    brightness roughly stable: values above 128 move further up, values
    below 128 move further down, matching how "contrast" is defined in
    standard image editors. Scaling around 0 would conflate contrast with
    brightness (every pixel would move in the same direction).

    The scaled result is rounded to the nearest integer, not truncated —
    see `adjust_brightness` for why truncation is asymmetric here.

    Restricted to ``uint8`` input, for the same reason as `adjust_brightness`.

    Raises
    ------
    ValueError
        If `image` does not have 2 or 3 dimensions, or `factor` is negative.
    TypeError
        If `image` does not have dtype ``uint8``.
    """
    require_image_ndim(image)
    require_dtype(image, (np.uint8,))
    require_non_negative(factor, "factor")
    scaled = (image.astype(np.float64) - 128.0) * factor + 128.0
    return np.clip(np.round(scaled), 0, 255).astype(np.uint8)


_ALPHA_BLEND_DTYPES = (np.uint8, np.uint16, np.int32, np.float32, np.float64)


def alpha_blend(image_a: Image, image_b: Image, alpha: float) -> Image:
    """Blend two same-shaped images: ``alpha * image_a + (1 - alpha) * image_b``.

    Restricted to an explicit set of dtypes verified to round-trip
    correctly through ``cv2.addWeighted`` (``uint8``, ``uint16``,
    ``int32``, ``float32``, ``float64``). Notably excludes ``int64``,
    which ``cv2.addWeighted`` silently downcasts to ``int32`` — verified
    directly — rather than rejecting or preserving it, and ``bool``,
    which raises a raw ``cv2.error``.

    Raises
    ------
    ValueError
        If `image_a` does not have 2 or 3 dimensions, the two images don't
        share a shape, or `alpha` is outside ``[0, 1]``.
    TypeError
        If `image_a` does not have dtype in the supported set, or
        `image_a` and `image_b` don't share a dtype.
    """
    require_image_ndim(image_a)
    require_dtype(image_a, _ALPHA_BLEND_DTYPES)
    require_same_shape_and_dtype(image_a, image_b)
    require_range(alpha, 0.0, 1.0, "alpha")
    return cv2.addWeighted(image_a, alpha, image_b, 1.0 - alpha, 0)


def bitwise_and(image_a: ImageU8, image_b: ImageU8) -> ImageU8:
    """Element-wise bitwise AND of two same-shaped images.

    Restricted to ``uint8``: a bitwise op on a float's bit pattern is
    technically possible but not a meaningful image operation.

    Raises
    ------
    ValueError
        If `image_a` does not have 2 or 3 dimensions, or shapes differ.
    TypeError
        If `image_a` does not have dtype ``uint8``, or `image_a` and
        `image_b` don't share a dtype.
    """
    require_image_ndim(image_a)
    require_dtype(image_a, (np.uint8,))
    require_same_shape_and_dtype(image_a, image_b)
    # cv2.bitwise_and preserves uint8 here; cv2's stubs don't say so.
    return cast(ImageU8, cv2.bitwise_and(image_a, image_b))


def bitwise_or(image_a: ImageU8, image_b: ImageU8) -> ImageU8:
    """Element-wise bitwise OR of two same-shaped images.

    Restricted to ``uint8``: a bitwise op on a float's bit pattern is
    technically possible but not a meaningful image operation.

    Raises
    ------
    ValueError
        If `image_a` does not have 2 or 3 dimensions, or shapes differ.
    TypeError
        If `image_a` does not have dtype ``uint8``, or `image_a` and
        `image_b` don't share a dtype.
    """
    require_image_ndim(image_a)
    require_dtype(image_a, (np.uint8,))
    require_same_shape_and_dtype(image_a, image_b)
    # cv2.bitwise_or preserves uint8 here; cv2's stubs don't say so.
    return cast(ImageU8, cv2.bitwise_or(image_a, image_b))


def apply_lut(image: ImageU8, table: np.ndarray) -> ImageU8:
    """Map each 8-bit pixel value through a 256-entry lookup table.

    Raises
    ------
    ValueError
        If `image` does not have 2 or 3 dimensions, or `table` is not
        shaped ``(256,)``.
    TypeError
        If `image` or `table` does not have dtype ``uint8`` (required by
        the underlying ``cv2.LUT`` call). `table` is not silently cast:
        a table built with an out-of-range value (e.g. ``-1``) would
        otherwise wrap around to ``255`` instead of raising.
    """
    require_image_ndim(image)
    require_dtype(image, (np.uint8,))
    if table.shape != (256,):
        raise ValueError(f"table must have shape (256,), got {table.shape}")
    require_dtype(table, (np.uint8,), "table")
    # cv2.LUT always produces uint8 here; cv2's stubs don't say so.
    return cast(ImageU8, cv2.LUT(image, table))
