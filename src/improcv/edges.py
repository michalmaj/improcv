"""Edge and corner detection."""

from __future__ import annotations

import cv2
import numpy as np

from improcv._validation import require_image_ndim

__all__ = ["auto_canny", "sobel_edge", "laplacian_edge", "harris_corner"]


def auto_canny(image: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """Detect edges with Canny, picking thresholds automatically from the image's median intensity.

    Parameters
    ----------
    image : np.ndarray
        Input image with shape ``(H, W)``.
    sigma : float, default 0.33
        Controls how far the lower/upper thresholds spread around the
        median intensity.

    Returns
    -------
    np.ndarray
        A new single-channel array with values in ``{0, 255}``.

    Raises
    ------
    ValueError
        If `image` does not have exactly 2 dimensions.
    """
    require_image_ndim(image, ndims=(2,))
    median = float(np.median(image))
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    return cv2.Canny(image, lower, upper)


def sobel_edge(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Detect edges with the Sobel operator.

    Computes the x and y gradients separately and combines them into a
    single gradient-magnitude edge map, so edges of any orientation are
    detected (a raw mixed ``dx=1, dy=1`` derivative would miss edges that
    are constant along one axis).

    Parameters
    ----------
    image : np.ndarray
        Input image with shape ``(H, W)``.
    kernel_size : int, default 3
        Sobel kernel size.

    Returns
    -------
    np.ndarray
        A new ``uint8`` array with the same shape as `image`, holding the
        gradient magnitude scaled to the 8-bit range.

    Raises
    ------
    ValueError
        If `image` does not have exactly 2 dimensions.
    """
    require_image_ndim(image, ndims=(2,))
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size)
    magnitude = cv2.magnitude(grad_x, grad_y)
    return cv2.convertScaleAbs(magnitude)


def laplacian_edge(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Detect edges with the Laplacian operator.

    Parameters
    ----------
    image : np.ndarray
        Input image with shape ``(H, W)``.
    kernel_size : int, default 3
        Laplacian kernel size.

    Returns
    -------
    np.ndarray
        A new ``uint8`` array with the same shape as `image`, holding the
        absolute Laplacian response scaled to the 8-bit range.

    Raises
    ------
    ValueError
        If `image` does not have exactly 2 dimensions.
    """
    require_image_ndim(image, ndims=(2,))
    gradient = cv2.Laplacian(image, cv2.CV_64F, ksize=kernel_size)
    return cv2.convertScaleAbs(gradient)


def harris_corner(
    image: np.ndarray,
    block_size: int = 2,
    kernel_size: int = 3,
    k: float = 0.04,
    threshold: float = 0.01,
) -> np.ndarray:
    """Detect corners with the Harris operator.

    Parameters
    ----------
    image : np.ndarray
        Input image with shape ``(H, W)``.
    block_size : int, default 2
        Neighborhood size considered for corner detection.
    kernel_size : int, default 3
        Sobel derivative kernel size used internally.
    k : float, default 0.04
        Harris detector free parameter.
    threshold : float, default 0.01
        Fraction of the maximum response above which a pixel is marked as
        a corner.

    Returns
    -------
    np.ndarray
        A new boolean array, shaped like `image`, ``True`` where the
        Harris response exceeds ``threshold * response.max()``.

    Raises
    ------
    ValueError
        If `image` does not have exactly 2 dimensions.
    """
    require_image_ndim(image, ndims=(2,))
    response = cv2.cornerHarris(image.astype(np.float32), block_size, kernel_size, k)
    return response > threshold * response.max()
