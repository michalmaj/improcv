"""Thresholding and morphological operations."""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

from improcv._validation import require_image_ndim, require_one_of, require_positive

__all__ = [
    "threshold",
    "dilate",
    "erode",
    "morph_open",
    "morph_close",
    "morph_gradient",
    "tophat",
    "blackhat",
]

ThresholdMethod = Literal["binary", "otsu", "adaptive_mean", "adaptive_gaussian"]
_THRESHOLD_METHODS: tuple[ThresholdMethod, ...] = (
    "binary",
    "otsu",
    "adaptive_mean",
    "adaptive_gaussian",
)


def threshold(
    image: np.ndarray,
    value: float = 127,
    max_value: float = 255,
    method: ThresholdMethod = "binary",
    *,
    block_size: int = 11,
    constant: float = 2,
) -> np.ndarray:
    """Binarize a single-channel image.

    Parameters
    ----------
    image : np.ndarray
        Input image with shape ``(H, W)``.
    value : float, default 127
        Threshold value; ignored when `method` is ``"otsu"`` or adaptive.
    max_value : float, default 255
        Value assigned to pixels that pass the threshold.
    method : {"binary", "otsu", "adaptive_mean", "adaptive_gaussian"}, default "binary"
        Thresholding strategy. The two ``adaptive_*`` methods threshold
        each pixel using a local neighborhood of size `block_size`.
    block_size : int, default 11
        Neighborhood size for adaptive methods; must be an odd integer > 1.
    constant : float, default 2
        Constant subtracted from the local mean/weighted mean for adaptive
        methods.

    Returns
    -------
    np.ndarray
        A new single-channel array with values in ``{0, max_value}``.

    Raises
    ------
    ValueError
        If `image` does not have exactly 2 dimensions, `method` is not one
        of the accepted values, or `block_size` is not an odd integer
        greater than 1 (adaptive methods only).
    """
    require_image_ndim(image, ndims=(2,))
    require_one_of(method, _THRESHOLD_METHODS, "method")
    if method == "binary":
        _, result = cv2.threshold(image, value, max_value, cv2.THRESH_BINARY)
        return result
    if method == "otsu":
        _, result = cv2.threshold(image, 0, max_value, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return result

    if block_size % 2 == 0 or block_size <= 1:
        raise ValueError(f"block_size must be an odd integer > 1, got {block_size}")
    adaptive_method = (
        cv2.ADAPTIVE_THRESH_MEAN_C if method == "adaptive_mean" else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    )
    return cv2.adaptiveThreshold(
        image, max_value, adaptive_method, cv2.THRESH_BINARY, block_size, constant
    )


def _kernel(size: int) -> np.ndarray:
    require_positive(size, "kernel_size")
    return np.ones((size, size), dtype=np.uint8)


def dilate(image: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """Grow bright regions using a square structuring element."""
    require_image_ndim(image)
    return cv2.dilate(image, _kernel(kernel_size), iterations=iterations)


def erode(image: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """Shrink bright regions using a square structuring element."""
    require_image_ndim(image)
    return cv2.erode(image, _kernel(kernel_size), iterations=iterations)


def morph_open(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Erosion followed by dilation — removes small bright noise."""
    require_image_ndim(image)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, _kernel(kernel_size))


def morph_close(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Dilation followed by erosion — closes small dark holes."""
    require_image_ndim(image)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, _kernel(kernel_size))


def morph_gradient(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Difference between dilation and erosion — outlines regions."""
    require_image_ndim(image)
    return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, _kernel(kernel_size))


def tophat(image: np.ndarray, kernel_size: int = 9) -> np.ndarray:
    """Difference between the image and its opening — highlights small bright details."""
    require_image_ndim(image)
    return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, _kernel(kernel_size))


def blackhat(image: np.ndarray, kernel_size: int = 9) -> np.ndarray:
    """Difference between the image's closing and itself — highlights small dark details."""
    require_image_ndim(image)
    return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, _kernel(kernel_size))
