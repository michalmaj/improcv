"""Image smoothing filters."""

from __future__ import annotations

from typing import cast

import cv2
import numpy as np

from improcv._validation import (
    require_dtype,
    require_image_ndim,
    require_non_negative,
    require_odd,
    require_positive,
    require_positive_int,
    require_size_2d,
)
from improcv.types import Image, ImageU8

__all__ = [
    "gaussian_blur",
    "median_blur",
    "bilateral_filter",
    "clahe",
    "gamma_correction",
    "histogram_equalization",
]

# Verified directly against cv2.GaussianBlur on OpenCV 4.13 and 5.0 (identical
# results on both): int32, int64, and bool reach a raw cv2.error.
_GAUSSIAN_BLUR_DTYPES = (np.uint8, np.uint16, np.int16, np.float32, np.float64)

# Verified directly against cv2.medianBlur on OpenCV 4.13 and 5.0 (identical
# results on both): int32, int64, float64, and bool reach a raw cv2.error.
_MEDIAN_BLUR_DTYPES = (np.uint8, np.uint16, np.int16, np.float32)

# Verified directly against cv2.bilateralFilter on OpenCV 4.13 and 5.0
# (identical results on both): only 8-bit and 32-bit float are supported;
# everything else (including uint16) reaches a raw cv2.error.
_BILATERAL_FILTER_DTYPES = (np.uint8, np.float32)


def gaussian_blur(image: Image, kernel_size: int, sigma: float = 0.0) -> Image:
    """Smooth an image with a Gaussian kernel.

    Parameters
    ----------
    image : np.ndarray
        Input image with shape ``(H, W)`` or ``(H, W, C)``.
    kernel_size : int
        Kernel width and height; must be a positive odd integer.
    sigma : float, default 0.0
        Gaussian standard deviation; ``0.0`` derives it from `kernel_size`.
        Must be finite and non-negative.

    Returns
    -------
    np.ndarray
        A new, blurred array with the same shape and dtype as `image`.

    Raises
    ------
    ValueError
        If `image` does not have 2 or 3 dimensions, `kernel_size` is not a
        positive odd integer, or `sigma` is not finite or is negative.
    TypeError
        If `image` does not have dtype ``uint8``, ``uint16``, ``int16``,
        ``float32``, or ``float64`` (verified against ``cv2.GaussianBlur``
        on both OpenCV 4 and 5; ``int32``/``int64``/``bool`` are not
        supported and otherwise reach a raw ``cv2.error``).
    """
    require_image_ndim(image)
    require_dtype(image, _GAUSSIAN_BLUR_DTYPES)
    require_positive_int(kernel_size, "kernel_size")
    require_odd(kernel_size, "kernel_size")
    require_non_negative(sigma, "sigma")
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def median_blur(image: Image, kernel_size: int) -> Image:
    """Smooth an image with a median filter — effective against salt-and-pepper noise.

    Parameters
    ----------
    image : np.ndarray
        Input image with shape ``(H, W)`` or ``(H, W, C)``.
    kernel_size : int
        Kernel width and height; must be a positive odd integer.

    Returns
    -------
    np.ndarray
        A new, filtered array with the same shape and dtype as `image`.

    Raises
    ------
    ValueError
        If `image` does not have 2 or 3 dimensions, or `kernel_size` is not
        a positive odd integer.
    TypeError
        If `image` does not have dtype ``uint8``, ``uint16``, ``int16``,
        or ``float32`` (verified against ``cv2.medianBlur`` on both OpenCV
        4 and 5; ``int32``/``int64``/``float64``/``bool`` are not
        supported and otherwise reach a raw ``cv2.error``).
    """
    require_image_ndim(image)
    require_dtype(image, _MEDIAN_BLUR_DTYPES)
    require_positive_int(kernel_size, "kernel_size")
    require_odd(kernel_size, "kernel_size")
    return cv2.medianBlur(image, kernel_size)


def bilateral_filter(image: Image, diameter: int, sigma_color: float, sigma_space: float) -> Image:
    """Smooth an image while preserving edges.

    Parameters
    ----------
    image : np.ndarray
        Input image with shape ``(H, W)`` or ``(H, W, C)``.
    diameter : int
        Diameter of the pixel neighborhood; must be positive.
    sigma_color : float
        Filter sigma in color space — larger values mix more distant
        colors. Must be finite and non-negative.
    sigma_space : float
        Filter sigma in coordinate space — larger values mix more distant
        pixels. Must be finite and non-negative.

    Returns
    -------
    np.ndarray
        A new, filtered array with the same shape and dtype as `image`.

    Raises
    ------
    ValueError
        If `image` does not have 2 or 3 dimensions, `diameter` is not
        positive, or `sigma_color`/`sigma_space` is not finite or is
        negative.
    TypeError
        If `diameter` is not an ``int``, or `image` does not have dtype
        ``uint8`` or ``float32`` (verified against ``cv2.bilateralFilter``
        on both OpenCV 4 and 5; every other dtype, including ``uint16``,
        is not supported and otherwise reaches a raw ``cv2.error``).
    """
    require_image_ndim(image)
    require_dtype(image, _BILATERAL_FILTER_DTYPES)
    require_positive_int(diameter, "diameter")
    require_non_negative(sigma_color, "sigma_color")
    require_non_negative(sigma_space, "sigma_space")
    return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)


def clahe(image: Image, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)) -> Image:
    """Apply Contrast Limited Adaptive Histogram Equalization.

    Parameters
    ----------
    image : np.ndarray
        Input image with shape ``(H, W)``.
    clip_limit : float, default 2.0
        Contrast limiting threshold.
    tile_grid_size : tuple of int, default (8, 8)
        Number of tiles in the row and column directions.

    Returns
    -------
    np.ndarray
        A new, contrast-enhanced array with the same shape and dtype as `image`.

    Raises
    ------
    ValueError
        If `image` does not have exactly 2 dimensions, `clip_limit` is not
        positive, or `tile_grid_size` is not a 2-tuple of positive ints.
        OpenCV's own CLAHE implementation does not validate these (a zero
        tile dimension causes a low-level crash on some builds), so this
        must be checked before calling into it.
    TypeError
        If `image` does not have dtype ``uint8`` or ``uint16`` (both are
        natively supported by OpenCV's CLAHE).
    """
    require_image_ndim(image, ndims=(2,))
    require_dtype(image, (np.uint8, np.uint16))
    require_positive(clip_limit, "clip_limit")
    require_size_2d(tile_grid_size, "tile_grid_size")
    clahe_op = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe_op.apply(image)


def gamma_correction(image: ImageU8, gamma: float) -> ImageU8:
    """Apply gamma correction to an 8-bit image via a precomputed lookup table.

    Parameters
    ----------
    image : np.ndarray
        Input image with shape ``(H, W)`` or ``(H, W, C)``, dtype ``uint8``.
    gamma : float
        Gamma value; must be positive. Values below 1 darken the image,
        values above 1 brighten it.

    Returns
    -------
    np.ndarray
        A new array with the same shape and dtype as `image`.

    Raises
    ------
    ValueError
        If `image` does not have 2 or 3 dimensions, or `gamma` is not positive.
    TypeError
        If `image` does not have dtype ``uint8`` (required by the
        underlying ``cv2.LUT`` call).
    """
    require_image_ndim(image)
    require_dtype(image, (np.uint8,))
    require_positive(gamma, "gamma")
    normalized = np.arange(256, dtype=np.float64) / 255.0
    table = np.clip((normalized ** (1.0 / gamma)) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    # cv2.LUT always produces uint8 here (both image and table are uint8);
    # cv2's stubs type the return as the loose `MatLike`.
    return cast(ImageU8, cv2.LUT(image, table))


def histogram_equalization(image: ImageU8) -> ImageU8:
    """Equalize the histogram of a single-channel image to spread out its intensity range.

    Parameters
    ----------
    image : np.ndarray
        Input image with shape ``(H, W)``.

    Returns
    -------
    np.ndarray
        A new, equalized array with the same shape and dtype as `image`.

    Raises
    ------
    ValueError
        If `image` does not have exactly 2 dimensions.
    TypeError
        If `image` does not have dtype ``uint8``.
    """
    require_image_ndim(image, ndims=(2,))
    require_dtype(image, (np.uint8,))
    # cv2.equalizeHist always produces uint8; cv2's stubs don't say so.
    return cast(ImageU8, cv2.equalizeHist(image))
