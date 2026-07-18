"""Color space conversions."""

from __future__ import annotations

import cv2

from improcv._validation import require_channels, require_image_ndim
from improcv.types import Image

__all__ = [
    "bgr_to_rgb",
    "rgb_to_bgr",
    "ensure_gray",
    "to_hsv",
    "to_lab",
    "to_ycrcb",
]


def bgr_to_rgb(image: Image) -> Image:
    """Convert a 3-channel BGR image to RGB channel order."""
    require_channels(image, 3)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image: Image) -> Image:
    """Convert a 3-channel RGB image to BGR channel order."""
    require_channels(image, 3)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def ensure_gray(image: Image) -> Image:
    """Return a single-channel grayscale version of `image`.

    Accepts either a 3-channel BGR image or an already-grayscale image;
    always returns a new array, never a view into `image`.
    """
    require_image_ndim(image)
    if image.ndim == 2:
        return image.copy()
    require_channels(image, 3)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def to_hsv(image: Image) -> Image:
    """Convert a 3-channel BGR image to HSV."""
    require_channels(image, 3)
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def to_lab(image: Image) -> Image:
    """Convert a 3-channel BGR image to CIE LAB."""
    require_channels(image, 3)
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)


def to_ycrcb(image: Image) -> Image:
    """Convert a 3-channel BGR image to YCrCb."""
    require_channels(image, 3)
    return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
