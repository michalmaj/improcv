"""Image smoothing filters."""

from __future__ import annotations

import cv2
import numpy as np

from improcv._validation import (
    require_dtype,
    require_image_ndim,
    require_positive,
    require_positive_int,
)

__all__ = [
    "gaussian_blur",
    "median_blur",
    "bilateral_filter",
    "clahe",
    "gamma_correction",
    "histogram_equalization",
]


def gaussian_blur(image: np.ndarray, kernel_size: int, sigma: float = 0.0) -> np.ndarray:
    """Smooth an image with a Gaussian kernel.

    Parameters
    ----------
    image : np.ndarray
        Input image with shape ``(H, W)`` or ``(H, W, C)``.
    kernel_size : int
        Kernel width and height; must be a positive odd integer.
    sigma : float, default 0.0
        Gaussian standard deviation; ``0.0`` derives it from `kernel_size`.

    Returns
    -------
    np.ndarray
        A new, blurred array with the same shape and dtype as `image`.

    Raises
    ------
    ValueError
        If `image` does not have 2 or 3 dimensions, or `kernel_size` is not
        a positive odd integer.
    """
    require_image_ndim(image)
    require_positive(kernel_size, "kernel_size")
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}")
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def median_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
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
    """
    require_image_ndim(image)
    require_positive(kernel_size, "kernel_size")
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}")
    return cv2.medianBlur(image, kernel_size)


def bilateral_filter(
    image: np.ndarray, diameter: int, sigma_color: float, sigma_space: float
) -> np.ndarray:
    """Smooth an image while preserving edges.

    Parameters
    ----------
    image : np.ndarray
        Input image with shape ``(H, W)`` or ``(H, W, C)``.
    diameter : int
        Diameter of the pixel neighborhood; must be positive.
    sigma_color : float
        Filter sigma in color space — larger values mix more distant colors.
    sigma_space : float
        Filter sigma in coordinate space — larger values mix more distant pixels.

    Returns
    -------
    np.ndarray
        A new, filtered array with the same shape and dtype as `image`.

    Raises
    ------
    ValueError
        If `image` does not have 2 or 3 dimensions, or `diameter` is not positive.
    """
    require_image_ndim(image)
    require_positive(diameter, "diameter")
    return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)


def clahe(
    image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)
) -> np.ndarray:
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
        positive, or either element of `tile_grid_size` is not a positive
        int. OpenCV's own CLAHE implementation does not validate these
        (a zero tile dimension causes a low-level crash on some builds),
        so this must be checked before calling into it.
    TypeError
        If `image` does not have dtype ``uint8`` or ``uint16`` (both are
        natively supported by OpenCV's CLAHE).
    """
    require_image_ndim(image, ndims=(2,))
    require_dtype(image, (np.uint8, np.uint16))
    require_positive(clip_limit, "clip_limit")
    tiles_x, tiles_y = tile_grid_size
    require_positive_int(tiles_x, "tile_grid_size[0]")
    require_positive_int(tiles_y, "tile_grid_size[1]")
    clahe_op = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe_op.apply(image)


def gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray:
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
    return cv2.LUT(image, table)


def histogram_equalization(image: np.ndarray) -> np.ndarray:
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
    return cv2.equalizeHist(image)
