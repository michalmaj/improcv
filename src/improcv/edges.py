"""Edge and corner detection."""

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
    require_range,
)
from improcv.types import Image, ImageU8, Mask

__all__ = ["auto_canny", "sobel_edge", "laplacian_edge", "harris_corner"]

# Verified directly against cv2.Sobel on OpenCV 4.13 and 5.0 (identical
# results on both): int32, int64, and bool reach a raw cv2.error.
_SOBEL_DTYPES = (np.uint8, np.uint16, np.int16, np.float32, np.float64)

# Verified directly against cv2.Laplacian on OpenCV 4.13 and 5.0 (identical
# results on both). Notably excludes float32: this function always requests
# a CV_64F destination, and OpenCV rejects the float32-source/float64-dest
# combination specifically ("Unsupported combination of source format and
# destination format") even though float32 input works for every other
# function here — a float32 source only works when the destination is also
# float32 (CV_32F), which `laplacian_edge` does not use.
_LAPLACIAN_DTYPES = (np.uint8, np.uint16, np.int16, np.float64)


def auto_canny(image: ImageU8, sigma: float = 0.33) -> Mask:
    """Detect edges with Canny, picking thresholds automatically from the image's median intensity.

    Parameters
    ----------
    image : np.ndarray
        Input image with shape ``(H, W)``.
    sigma : float, default 0.33
        Controls how far the lower/upper thresholds spread around the
        median intensity; must be within ``[0, 1]``.

    Returns
    -------
    np.ndarray
        A new single-channel array with values in ``{0, 255}``.

    Raises
    ------
    ValueError
        If `image` does not have exactly 2 dimensions, or `sigma` is not
        finite or is outside ``[0, 1]``.
    TypeError
        If `image` does not have dtype ``uint8``.
    """
    require_image_ndim(image, ndims=(2,))
    require_dtype(image, (np.uint8,))
    # require_range alone already rejects NaN/infinity here: both fail
    # every comparison against a bounded [0, 1] range.
    require_range(sigma, 0.0, 1.0, "sigma")
    median = float(np.median(image))
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    # cv2's stubs type this as the loose `MatLike`; cv2.Canny always
    # produces a uint8 {0, 255} array in practice.
    return cast(Mask, cv2.Canny(image, lower, upper))


def sobel_edge(image: Image, kernel_size: int = 3) -> ImageU8:
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
        Sobel kernel size; must be a positive odd integer in ``[1, 31]``
        (OpenCV's own supported range).

    Returns
    -------
    np.ndarray
        A new ``uint8`` array with the same shape as `image`, holding the
        gradient magnitude scaled to the 8-bit range.

    Raises
    ------
    ValueError
        If `image` does not have exactly 2 dimensions, or `kernel_size` is
        not a positive odd integer.
    TypeError
        If `image` does not have dtype ``uint8``, ``uint16``, ``int16``,
        ``float32``, or ``float64`` (verified against ``cv2.Sobel`` on
        both OpenCV 4 and 5; ``int32``/``int64``/``bool`` are not
        supported and otherwise reach a raw ``cv2.error``).
    """
    require_image_ndim(image, ndims=(2,))
    require_dtype(image, _SOBEL_DTYPES)
    require_positive_int(kernel_size, "kernel_size")
    require_odd(kernel_size, "kernel_size")
    require_range(kernel_size, 1, 31, "kernel_size")
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size)
    magnitude = cv2.magnitude(grad_x, grad_y)
    # cv2.convertScaleAbs always produces uint8; cv2's stubs don't say so.
    return cast(ImageU8, cv2.convertScaleAbs(magnitude))


def laplacian_edge(image: Image, kernel_size: int = 3) -> ImageU8:
    """Detect edges with the Laplacian operator.

    Parameters
    ----------
    image : np.ndarray
        Input image with shape ``(H, W)``.
    kernel_size : int, default 3
        Laplacian kernel size; must be a positive odd integer in
        ``[1, 31]`` (OpenCV's own supported range).

    Returns
    -------
    np.ndarray
        A new ``uint8`` array with the same shape as `image`, holding the
        absolute Laplacian response scaled to the 8-bit range.

    Raises
    ------
    ValueError
        If `image` does not have exactly 2 dimensions, or `kernel_size` is
        not a positive odd integer.
    TypeError
        If `image` does not have dtype ``uint8``, ``uint16``, ``int16``,
        or ``float64`` (verified against ``cv2.Laplacian`` on both OpenCV
        4 and 5; notably excludes ``float32``, which — unlike every other
        function in this module — is rejected here because this function
        always requests a ``float64`` destination and OpenCV does not
        support that specific source/destination combination;
        ``int32``/``int64``/``bool`` are not supported either).
    """
    require_image_ndim(image, ndims=(2,))
    require_dtype(image, _LAPLACIAN_DTYPES)
    require_positive_int(kernel_size, "kernel_size")
    require_odd(kernel_size, "kernel_size")
    require_range(kernel_size, 1, 31, "kernel_size")
    gradient = cv2.Laplacian(image, cv2.CV_64F, ksize=kernel_size)
    # cv2.convertScaleAbs always produces uint8; cv2's stubs don't say so.
    return cast(ImageU8, cv2.convertScaleAbs(gradient))


def harris_corner(
    image: Image,
    block_size: int = 2,
    kernel_size: int = 3,
    k: float = 0.04,
    threshold: float = 0.01,
) -> Mask:
    """Detect corners with the Harris operator.

    Parameters
    ----------
    image : np.ndarray
        Input image with shape ``(H, W)``.
    block_size : int, default 2
        Neighborhood size considered for corner detection; must be a
        positive integer.
    kernel_size : int, default 3
        Sobel derivative kernel size used internally; must be a positive
        odd integer in ``[1, 31]`` (OpenCV's own supported range).
    k : float, default 0.04
        Harris detector free parameter; must be positive (a typical useful
        range is small, around 0.04-0.06, but no upper bound is enforced).
    threshold : float, default 0.01
        Fraction of the maximum response above which a pixel is marked as
        a corner; must be non-negative.

    Returns
    -------
    np.ndarray
        A new ``uint8`` array, shaped like `image`, with value ``255``
        where the Harris response exceeds ``threshold * response.max()``
        and ``0`` elsewhere — improcv's mask convention (matches OpenCV's
        own native mask representation; see `in_range`, `threshold`,
        `auto_canny`).

    Raises
    ------
    ValueError
        If `image` does not have exactly 2 dimensions, `block_size` is not
        a positive integer, `kernel_size` is not a positive odd integer,
        `k` is not positive, or `threshold` is negative.
    """
    require_image_ndim(image, ndims=(2,))
    require_positive_int(block_size, "block_size")
    require_positive_int(kernel_size, "kernel_size")
    require_odd(kernel_size, "kernel_size")
    require_range(kernel_size, 1, 31, "kernel_size")
    require_positive(k, "k")
    require_non_negative(threshold, "threshold")
    response = cv2.cornerHarris(image.astype(np.float32), block_size, kernel_size, k)
    return (response > threshold * response.max()).astype(np.uint8) * 255
