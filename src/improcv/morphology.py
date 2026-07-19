"""Thresholding and morphological operations."""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

from improcv._validation import (
    require_dtype,
    require_finite,
    require_fits_dtype,
    require_image_ndim,
    require_non_negative_int,
    require_odd,
    require_one_of,
    require_positive_int,
)
from improcv.types import Image

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

# Verified directly against cv2.threshold (THRESH_BINARY) and the
# structural/morphological ops (cv2.dilate/erode/morphologyEx) on OpenCV
# 4.13 and 5.0 (identical results on both): int32, int64, and bool reach a
# raw cv2.error.
_THRESHOLD_BINARY_DTYPES = (np.uint8, np.uint16, np.int16, np.float32, np.float64)
_MORPHOLOGY_DTYPES = (np.uint8, np.uint16, np.int16, np.float32, np.float64)


def threshold(
    image: Image,
    value: float = 127,
    max_value: float = 255,
    method: ThresholdMethod = "binary",
    *,
    block_size: int = 11,
    constant: float = 2,
) -> Image:
    """Binarize a single-channel image.

    Parameters
    ----------
    image : np.ndarray
        Input image with shape ``(H, W)``. Dtype-preserving for
        ``"binary"`` (e.g. a ``float32`` input stays ``float32``);
        ``"otsu"`` and the ``adaptive_*`` methods require and return
        ``uint8``.
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
        A new single-channel array with values in ``{0, max_value}``. Note
        this is *not* always improcv's ``uint8`` ``{0, 255}`` mask
        convention (see `in_range`, `auto_canny`, `harris_corner`):
        ``"binary"`` mode preserves `image`'s dtype and honors a custom
        `max_value`, so the result is only a conventional mask when
        `image` is ``uint8`` and `max_value` is left at its default 255.

    Raises
    ------
    ValueError
        If `image` does not have exactly 2 dimensions, `method` is not one
        of the accepted values, `value`/`max_value`/`constant` is not
        finite, `max_value` does not fit within `image`'s integer dtype
        range (e.g. ``300`` for a ``uint8`` image — OpenCV silently
        saturates this to ``255`` rather than rejecting it, verified
        directly), or `block_size` is not an odd integer greater than 1
        (adaptive methods only).
    TypeError
        If `method` is ``"otsu"`` or one of the ``adaptive_*`` methods and
        `image` does not have dtype ``uint8`` (OpenCV requires 8-bit input
        for those); or if `method` is ``"binary"`` and `image` does not
        have dtype ``uint8``, ``uint16``, ``int16``, ``float32``, or
        ``float64`` (verified against ``cv2.threshold`` on both OpenCV 4
        and 5; ``int32``/``int64``/``bool`` are not supported and
        otherwise reach a raw ``cv2.error``).
    """
    require_image_ndim(image, ndims=(2,))
    require_one_of(method, _THRESHOLD_METHODS, "method")
    require_finite(value, "value")
    require_finite(max_value, "max_value")
    require_finite(constant, "constant")
    if method == "binary":
        require_dtype(image, _THRESHOLD_BINARY_DTYPES)
        require_fits_dtype(max_value, image.dtype, "max_value")
        _, result = cv2.threshold(image, value, max_value, cv2.THRESH_BINARY)
        return result
    if method == "otsu":
        require_dtype(image, (np.uint8,))
        require_fits_dtype(max_value, image.dtype, "max_value")
        _, result = cv2.threshold(image, 0, max_value, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return result

    require_dtype(image, (np.uint8,))
    require_fits_dtype(max_value, image.dtype, "max_value")
    require_positive_int(block_size, "block_size")
    require_odd(block_size, "block_size")
    if block_size == 1:
        raise ValueError(f"block_size must be greater than 1, got {block_size}")
    adaptive_method = (
        cv2.ADAPTIVE_THRESH_MEAN_C if method == "adaptive_mean" else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    )
    return cv2.adaptiveThreshold(
        image, max_value, adaptive_method, cv2.THRESH_BINARY, block_size, constant
    )


def _kernel(size: int) -> np.ndarray:
    require_positive_int(size, "kernel_size")
    return np.ones((size, size), dtype=np.uint8)


def dilate(image: Image, kernel_size: int = 3, iterations: int = 1) -> Image:
    """Grow bright regions using a square structuring element.

    `iterations` may be ``0`` (a meaningful no-op, returns `image`
    unchanged) but not negative.

    Raises
    ------
    TypeError
        If `image` does not have dtype ``uint8``, ``uint16``, ``int16``,
        ``float32``, or ``float64`` (verified against ``cv2.dilate`` on
        both OpenCV 4 and 5; ``int32``/``int64``/``bool`` are not
        supported and otherwise reach a raw ``cv2.error``).
    """
    require_image_ndim(image)
    require_dtype(image, _MORPHOLOGY_DTYPES)
    require_non_negative_int(iterations, "iterations")
    return cv2.dilate(image, _kernel(kernel_size), iterations=iterations)


def erode(image: Image, kernel_size: int = 3, iterations: int = 1) -> Image:
    """Shrink bright regions using a square structuring element.

    `iterations` may be ``0`` (a meaningful no-op, returns `image`
    unchanged) but not negative.

    Raises
    ------
    TypeError
        If `image` does not have dtype ``uint8``, ``uint16``, ``int16``,
        ``float32``, or ``float64`` (verified against ``cv2.erode`` on
        both OpenCV 4 and 5; ``int32``/``int64``/``bool`` are not
        supported and otherwise reach a raw ``cv2.error``).
    """
    require_image_ndim(image)
    require_dtype(image, _MORPHOLOGY_DTYPES)
    require_non_negative_int(iterations, "iterations")
    return cv2.erode(image, _kernel(kernel_size), iterations=iterations)


def morph_open(image: Image, kernel_size: int = 3) -> Image:
    """Erosion followed by dilation — removes small bright noise.

    Raises
    ------
    TypeError
        If `image` does not have dtype ``uint8``, ``uint16``, ``int16``,
        ``float32``, or ``float64`` (verified against
        ``cv2.morphologyEx`` on both OpenCV 4 and 5;
        ``int32``/``int64``/``bool`` are not supported and otherwise
        reach a raw ``cv2.error``).
    """
    require_image_ndim(image)
    require_dtype(image, _MORPHOLOGY_DTYPES)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, _kernel(kernel_size))


def morph_close(image: Image, kernel_size: int = 3) -> Image:
    """Dilation followed by erosion — closes small dark holes.

    Raises
    ------
    TypeError
        If `image` does not have dtype ``uint8``, ``uint16``, ``int16``,
        ``float32``, or ``float64`` (verified against
        ``cv2.morphologyEx`` on both OpenCV 4 and 5;
        ``int32``/``int64``/``bool`` are not supported and otherwise
        reach a raw ``cv2.error``).
    """
    require_image_ndim(image)
    require_dtype(image, _MORPHOLOGY_DTYPES)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, _kernel(kernel_size))


def morph_gradient(image: Image, kernel_size: int = 3) -> Image:
    """Difference between dilation and erosion — outlines regions.

    Raises
    ------
    TypeError
        If `image` does not have dtype ``uint8``, ``uint16``, ``int16``,
        ``float32``, or ``float64`` (verified against
        ``cv2.morphologyEx`` on both OpenCV 4 and 5;
        ``int32``/``int64``/``bool`` are not supported and otherwise
        reach a raw ``cv2.error``).
    """
    require_image_ndim(image)
    require_dtype(image, _MORPHOLOGY_DTYPES)
    return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, _kernel(kernel_size))


def tophat(image: Image, kernel_size: int = 9) -> Image:
    """Difference between the image and its opening — highlights small bright details.

    Raises
    ------
    TypeError
        If `image` does not have dtype ``uint8``, ``uint16``, ``int16``,
        ``float32``, or ``float64`` (verified against
        ``cv2.morphologyEx`` on both OpenCV 4 and 5;
        ``int32``/``int64``/``bool`` are not supported and otherwise
        reach a raw ``cv2.error``).
    """
    require_image_ndim(image)
    require_dtype(image, _MORPHOLOGY_DTYPES)
    return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, _kernel(kernel_size))


def blackhat(image: Image, kernel_size: int = 9) -> Image:
    """Difference between the image's closing and itself — highlights small dark details.

    Raises
    ------
    TypeError
        If `image` does not have dtype ``uint8``, ``uint16``, ``int16``,
        ``float32``, or ``float64`` (verified against
        ``cv2.morphologyEx`` on both OpenCV 4 and 5;
        ``int32``/``int64``/``bool`` are not supported and otherwise
        reach a raw ``cv2.error``).
    """
    require_image_ndim(image)
    require_dtype(image, _MORPHOLOGY_DTYPES)
    return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, _kernel(kernel_size))
