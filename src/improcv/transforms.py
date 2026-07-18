"""Geometric image transformations."""

from __future__ import annotations

import cv2
import numpy as np

__all__ = ["resize"]


def resize(
    image: np.ndarray,
    width: int | None = None,
    height: int | None = None,
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    """Resize an image, optionally preserving its aspect ratio.

    Parameters
    ----------
    image : np.ndarray
        Input image with shape ``(H, W)`` or ``(H, W, C)``.
    width : int, optional
        Target width in pixels. If given without `height`, the height is
        computed to preserve the input's aspect ratio.
    height : int, optional
        Target height in pixels. If given without `width`, the width is
        computed to preserve the input's aspect ratio.
    interpolation : int, default ``cv2.INTER_AREA``
        Interpolation flag passed through to ``cv2.resize``.

    Returns
    -------
    np.ndarray
        A new array holding the resized image. `image` is never modified
        and the result never shares memory with it.

    Raises
    ------
    ValueError
        If both `width` and `height` are ``None``, if either is not a
        positive integer, or if `image` does not have 2 or 3 dimensions.
    """
    if image.ndim not in (2, 3):
        raise ValueError(f"image must have 2 or 3 dimensions, got {image.ndim}")
    if width is None and height is None:
        raise ValueError("at least one of width or height must be given")
    if width is not None and width <= 0:
        raise ValueError(f"width must be positive, got {width}")
    if height is not None and height <= 0:
        raise ValueError(f"height must be positive, got {height}")

    source_height, source_width = image.shape[:2]

    if width is not None and height is not None:
        target_size = (width, height)
    elif width is not None:
        target_size = (width, round(width * source_height / source_width))
    else:
        assert height is not None
        target_size = (round(height * source_width / source_height), height)

    return cv2.resize(image, target_size, interpolation=interpolation)
