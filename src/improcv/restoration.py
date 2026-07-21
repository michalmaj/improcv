"""Image restoration: mask-guided inpainting."""

from __future__ import annotations

from typing import Literal, cast

import cv2
import numpy as np

from improcv._validation import (
    require_channels,
    require_dtype,
    require_image_ndim,
    require_one_of,
    require_positive,
    require_spatial_mask,
)
from improcv.types import Image, Mask

__all__ = [
    "inpaint",
    "InpaintMethod",
]

InpaintMethod = Literal["ns", "telea"]
_INPAINT_METHODS: dict[InpaintMethod, int] = {"ns": cv2.INPAINT_NS, "telea": cv2.INPAINT_TELEA}

_INPAINT_1CH_DTYPES = (np.uint8, np.uint16, np.float32)


def inpaint(
    image: Image,
    mask: Mask,
    radius: float,
    method: InpaintMethod = "telea",
) -> Image:
    """Reconstruct a masked region of an image from its surrounding pixels.

    Parameters
    ----------
    image : np.ndarray
        Input image. Either 2D (``(H, W)``), dtype ``uint8``, ``uint16``,
        or ``float32``, or 3D with exactly 3 channels, dtype ``uint8``
        only -- verified directly, an asymmetric contract: 3-channel input
        does not accept ``uint16``/``float32``, even though 1-channel
        input does. A ``(H, W, 1)`` array is not accepted as an implicit
        grayscale image.
    mask : np.ndarray
        Shape ``(H, W)``, dtype ``uint8``, matching `image`'s spatial
        size. Any nonzero value marks a pixel to be reconstructed. Must
        leave at least one non-masked (known) pixel -- verified directly
        that a fully-nonzero mask does not error, but deterministically
        returns the input unchanged on both OpenCV versions and both
        methods, since there is no known source pixel to reconstruct from;
        that silent no-op is rejected explicitly instead.
    radius : float
        Radius of the circular neighborhood considered by the algorithm.
        Must be finite and strictly positive -- verified directly that
        `cv2.inpaint` silently treats ``0``, a negative value, and ``1``
        as identical, so a non-positive value is rejected rather than
        silently coerced.
    method : {"ns", "telea"}, default "telea"
        Inpainting algorithm.

    Returns
    -------
    np.ndarray
        Same shape and dtype as `image`. A new, independent array; `image`
        is never modified (verified directly that `cv2.inpaint` does not
        mutate its input). For an all-zero `mask`, this is a copy of
        `image` with identical values.

    Raises
    ------
    ValueError
        If `image` is not 2D/3D or is empty, has a height or width below
        2 pixels, a 3D `image` does not have exactly 3 channels, `image`
        has dtype ``float32`` and contains a non-finite (``NaN``/``inf``)
        value, `mask` does not match `image`'s spatial size or leaves no
        non-masked pixel, `radius` is not finite or not positive, or
        `method` is not one of the accepted values.
    TypeError
        If `image` does not have one of the accepted dtypes for its
        channel count, `mask` does not have dtype ``uint8``, or `radius`
        is not a real number (rejecting `bool`).
    RuntimeError
        If `image` has dtype ``float32``, contains only finite values, but
        `cv2.inpaint` still produces a non-finite result -- verified
        directly that this happens for extreme-but-finite ``float32``
        magnitudes (e.g. `numpy.finfo(numpy.float32).max`), identically on
        both OpenCV versions; this is treated as an internal OpenCV
        failure rather than returned as a silently corrupted image.

    Notes
    -----
    A 1-pixel-tall or 1-pixel-wide `image` is rejected for every dtype, not
    only ``float32``: verified directly, repeatedly, across separate
    processes, that a ``(1, N)`` ``float32`` image produces nondeterministic
    output for the exact same input -- including ``NaN`` -- on both OpenCV
    4.13 and 5.0, and that a ``(1, N)`` `uint8`/`uint16` image produces a
    non-reproducible (if not `NaN`-capable) result across runs too. This is
    silent, nondeterministic result corruption, not an ordinary numerical
    difference, and is rejected uniformly rather than only for the dtype
    where it happens to be visible as `NaN`.
    """
    require_one_of(method, tuple(_INPAINT_METHODS), "method")
    require_image_ndim(image, ndims=(2, 3))
    if image.ndim == 2:
        require_dtype(image, _INPAINT_1CH_DTYPES)
    else:
        require_channels(image, 3)
        require_dtype(image, (np.uint8,))
    height, width = image.shape[:2]
    if height < 2 or width < 2:
        raise ValueError("image height and width must each be at least 2 pixels for inpainting")
    if image.dtype == np.float32 and not np.all(np.isfinite(image)):
        raise ValueError("a float32 image must contain only finite values")
    require_spatial_mask(mask, image)
    require_positive(radius, "radius")

    if not np.any(mask):
        return image.copy()
    if np.all(mask):
        raise ValueError("mask must leave at least one non-masked (known) pixel")

    result = cv2.inpaint(image, mask, float(radius), _INPAINT_METHODS[method])
    if image.dtype == np.float32 and not np.all(np.isfinite(result)):
        raise RuntimeError("OpenCV inpaint produced non-finite output for a finite float32 image")
    return cast(Image, result)
