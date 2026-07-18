"""Color space conversions."""

from __future__ import annotations

import cv2
import numpy as np

from improcv._validation import require_channels, require_dtype, require_image_ndim
from improcv.types import Image

__all__ = [
    "bgr_to_rgb",
    "rgb_to_bgr",
    "ensure_gray",
    "to_hsv",
    "to_lab",
    "to_ycrcb",
]

# cv2.cvtColor's supported dtypes are conversion-specific, verified directly
# against cv2 rather than assumed: a simple channel reorder or BGR<->GRAY
# supports uint8/uint16/float32, but HSV/LAB do not support uint16.
_CHANNEL_REORDER_DTYPES = (np.uint8, np.uint16, np.float32)
_HSV_LAB_DTYPES = (np.uint8, np.float32)


def bgr_to_rgb(image: Image) -> Image:
    """Convert a 3-channel BGR image to RGB channel order.

    Raises
    ------
    ValueError
        If `image` is not 3-channel or is empty.
    TypeError
        If `image` does not have dtype ``uint8``, ``uint16``, or ``float32``.
    """
    require_channels(image, 3)
    require_dtype(image, _CHANNEL_REORDER_DTYPES)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image: Image) -> Image:
    """Convert a 3-channel RGB image to BGR channel order.

    Raises
    ------
    ValueError
        If `image` is not 3-channel or is empty.
    TypeError
        If `image` does not have dtype ``uint8``, ``uint16``, or ``float32``.
    """
    require_channels(image, 3)
    require_dtype(image, _CHANNEL_REORDER_DTYPES)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def ensure_gray(image: Image) -> Image:
    """Return a single-channel grayscale version of `image`.

    Accepts either a 3-channel BGR image or an already-grayscale image;
    always returns a new array, never a view into `image`.

    Raises
    ------
    ValueError
        If `image` does not have 2 or 3 dimensions, is empty, or (for a
        3-channel input) is not 3-channel.
    TypeError
        If a 3-channel `image` does not have dtype ``uint8``, ``uint16``,
        or ``float32``.
    """
    require_image_ndim(image)
    if image.ndim == 2:
        return image.copy()
    require_channels(image, 3)
    require_dtype(image, _CHANNEL_REORDER_DTYPES)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def to_hsv(image: Image) -> Image:
    """Convert a 3-channel BGR image to HSV.

    Raises
    ------
    ValueError
        If `image` is not 3-channel or is empty.
    TypeError
        If `image` does not have dtype ``uint8`` or ``float32`` (unlike
        the other conversions here, HSV does not support ``uint16``).
    """
    require_channels(image, 3)
    require_dtype(image, _HSV_LAB_DTYPES)
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def to_lab(image: Image) -> Image:
    """Convert a 3-channel BGR image to CIE LAB.

    Raises
    ------
    ValueError
        If `image` is not 3-channel or is empty.
    TypeError
        If `image` does not have dtype ``uint8`` or ``float32`` (unlike
        the other conversions here, LAB does not support ``uint16``).
    """
    require_channels(image, 3)
    require_dtype(image, _HSV_LAB_DTYPES)
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)


def to_ycrcb(image: Image) -> Image:
    """Convert a 3-channel BGR image to YCrCb.

    Raises
    ------
    ValueError
        If `image` is not 3-channel or is empty.
    TypeError
        If `image` does not have dtype ``uint8``, ``uint16``, or ``float32``.
    """
    require_channels(image, 3)
    require_dtype(image, _CHANNEL_REORDER_DTYPES)
    return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
