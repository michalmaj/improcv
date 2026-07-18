"""Pixel-level operations."""

from __future__ import annotations

import cv2
import numpy as np

from improcv._validation import require_image_ndim

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
    """Return a boolean mask of pixels within `[lower, upper]` (inclusive, per channel).

    Parameters
    ----------
    image : np.ndarray
        Input image with shape ``(H, W)`` or ``(H, W, C)``.
    lower, upper : tuple of int
        Inclusive per-channel bounds.

    Returns
    -------
    np.ndarray
        A new boolean array shaped like `image`'s spatial dimensions.

    Raises
    ------
    ValueError
        If `image` does not have 2 or 3 dimensions.
    """
    require_image_ndim(image)
    mask = cv2.inRange(image, np.array(lower), np.array(upper))
    return mask.astype(np.bool_)


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

    Raises
    ------
    ValueError
        If `image` does not have 2 or 3 dimensions.
    """
    require_image_ndim(image)
    return cv2.convertScaleAbs(image, alpha=1.0, beta=delta)


def adjust_contrast(image: np.ndarray, factor: float) -> np.ndarray:
    """Scale pixel values by `factor`, clamped to the valid 8-bit range.

    Raises
    ------
    ValueError
        If `image` does not have 2 or 3 dimensions.
    """
    require_image_ndim(image)
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)


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
    """
    require_image_ndim(image)
    if table.shape != (256,):
        raise ValueError(f"table must have shape (256,), got {table.shape}")
    return cv2.LUT(image, table.astype(np.uint8))
