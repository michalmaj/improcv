"""Image smoothing filters."""

from __future__ import annotations

import cv2
import numpy as np

from improcv._validation import require_image_ndim, require_positive

__all__ = ["gaussian_blur", "median_blur", "bilateral_filter"]


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
