"""Pixel-level operations."""

from __future__ import annotations

import cv2
import numpy as np

from improcv._validation import require_dtype, require_image_ndim, require_non_negative

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


def in_range(image: np.ndarray, lower: tuple[int, ...], upper: tuple[int, ...]) -> np.ndarray:
    """Return a mask of pixels within `[lower, upper]` (inclusive, per channel).

    Parameters
    ----------
    image : np.ndarray
        Input image with shape ``(H, W)`` or ``(H, W, C)``.
    lower, upper : tuple of int
        Inclusive per-channel bounds.

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
        If `image` does not have 2 or 3 dimensions.
    """
    require_image_ndim(image)
    return cv2.inRange(image, np.array(lower), np.array(upper))


def invert(image: np.ndarray) -> np.ndarray:
    """Invert pixel values (``255 - value`` for 8-bit images).

    Raises
    ------
    ValueError
        If `image` does not have 2 or 3 dimensions.
    """
    require_image_ndim(image)
    return cv2.bitwise_not(image)


def adjust_brightness(image: np.ndarray, delta: float) -> np.ndarray:
    """Add `delta` to every pixel value, clamped to the valid 8-bit range.

    Uses saturating (clamping) arithmetic in both directions: a negative
    `delta` that would push a pixel below 0 clamps to 0, it does not wrap
    or reflect back to a positive value (unlike a naive
    ``cv2.convertScaleAbs`` call, whose ``beta`` argument takes the
    absolute value of the result rather than clamping it).

    Raises
    ------
    ValueError
        If `image` does not have 2 or 3 dimensions.
    """
    require_image_ndim(image)
    return np.clip(image.astype(np.int32) + delta, 0, 255).astype(np.uint8)


def adjust_contrast(image: np.ndarray, factor: float) -> np.ndarray:
    """Scale pixel values by `factor` around the mid-gray point (128).

    Scaling around the midpoint (rather than around 0) keeps average
    brightness roughly stable: values above 128 move further up, values
    below 128 move further down, matching how "contrast" is defined in
    standard image editors. Scaling around 0 would conflate contrast with
    brightness (every pixel would move in the same direction).

    Raises
    ------
    ValueError
        If `image` does not have 2 or 3 dimensions, or `factor` is negative.
    """
    require_image_ndim(image)
    require_non_negative(factor, "factor")
    return np.clip((image.astype(np.float64) - 128.0) * factor + 128.0, 0, 255).astype(np.uint8)


def alpha_blend(image_a: np.ndarray, image_b: np.ndarray, alpha: float) -> np.ndarray:
    """Blend two same-shaped images: ``alpha * image_a + (1 - alpha) * image_b``.

    Raises
    ------
    ValueError
        If `image_a` does not have 2 or 3 dimensions, the two images don't
        share a shape, or `alpha` is outside ``[0, 1]``.
    """
    require_image_ndim(image_a)
    if image_a.shape != image_b.shape:
        raise ValueError(
            f"images must have the same shape, got {image_a.shape} and {image_b.shape}"
        )
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
    return cv2.addWeighted(image_a, alpha, image_b, 1.0 - alpha, 0)


def bitwise_and(image_a: np.ndarray, image_b: np.ndarray) -> np.ndarray:
    """Element-wise bitwise AND of two same-shaped images.

    Raises
    ------
    ValueError
        If `image_a` does not have 2 or 3 dimensions, or shapes differ.
    """
    require_image_ndim(image_a)
    if image_a.shape != image_b.shape:
        raise ValueError(
            f"images must have the same shape, got {image_a.shape} and {image_b.shape}"
        )
    return cv2.bitwise_and(image_a, image_b)


def bitwise_or(image_a: np.ndarray, image_b: np.ndarray) -> np.ndarray:
    """Element-wise bitwise OR of two same-shaped images.

    Raises
    ------
    ValueError
        If `image_a` does not have 2 or 3 dimensions, or shapes differ.
    """
    require_image_ndim(image_a)
    if image_a.shape != image_b.shape:
        raise ValueError(
            f"images must have the same shape, got {image_a.shape} and {image_b.shape}"
        )
    return cv2.bitwise_or(image_a, image_b)


def apply_lut(image: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Map each 8-bit pixel value through a 256-entry lookup table.

    Raises
    ------
    ValueError
        If `image` does not have 2 or 3 dimensions, or `table` is not
        shaped ``(256,)``.
    TypeError
        If `image` does not have dtype ``uint8`` (required by the
        underlying ``cv2.LUT`` call).
    """
    require_image_ndim(image)
    require_dtype(image, (np.uint8,))
    if table.shape != (256,):
        raise ValueError(f"table must have shape (256,), got {table.shape}")
    return cv2.LUT(image, table.astype(np.uint8))
