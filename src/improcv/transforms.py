"""Geometric image transformations."""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

from improcv._validation import require_image_ndim, require_positive

__all__ = ["resize", "translate", "rotate", "rotate_bound", "flip"]


def resize(
    image: np.ndarray,
    width: int | None = None,
    height: int | None = None,
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    """Resize an image, optionally preserving its aspect ratio.

    Parameters
    ----------
    image : np.ndarray
        Input image with shape ``(H, W)`` or ``(H, W, C)``.
    width : int, optional
        Target width in pixels. If given without `height`, the height is
        computed to preserve the input's aspect ratio.
    height : int, optional
        Target height in pixels. If given without `width`, the width is
        computed to preserve the input's aspect ratio.
    interpolation : int, default ``cv2.INTER_AREA``
        Interpolation flag passed through to ``cv2.resize``.

    Returns
    -------
    np.ndarray
        A new array holding the resized image. `image` is never modified
        and the result never shares memory with it.

    Raises
    ------
    ValueError
        If both `width` and `height` are ``None``, if either is not a
        positive integer, or if `image` does not have 2 or 3 dimensions.
    """
    require_image_ndim(image)
    if width is None and height is None:
        raise ValueError("at least one of width or height must be given")
    if width is not None:
        require_positive(width, "width")
    if height is not None:
        require_positive(height, "height")

    source_height, source_width = image.shape[:2]

    if width is not None and height is not None:
        target_size = (width, height)
    elif width is not None:
        target_size = (width, round(width * source_height / source_width))
    else:
        assert height is not None
        target_size = (round(height * source_width / source_height), height)

    return cv2.resize(image, target_size, interpolation=interpolation)


def translate(
    image: np.ndarray,
    x: int,
    y: int,
    *,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: float | tuple[float, ...] = 0,
) -> np.ndarray:
    """Shift an image by `(x, y)` pixels.

    Parameters
    ----------
    image : np.ndarray
        Input image with shape ``(H, W)`` or ``(H, W, C)``.
    x : int
        Horizontal shift in pixels; positive moves content right.
    y : int
        Vertical shift in pixels; positive moves content down.
    interpolation : int, default ``cv2.INTER_LINEAR``
        Interpolation flag passed through to ``cv2.warpAffine``.
    border_mode : int, default ``cv2.BORDER_CONSTANT``
        Border mode for pixels exposed by the shift.
    border_value : float or tuple of float, default 0
        Fill value used when `border_mode` is ``cv2.BORDER_CONSTANT``.

    Returns
    -------
    np.ndarray
        A new array with the same shape and dtype as `image`. `image` is
        never modified.

    Raises
    ------
    ValueError
        If `image` does not have 2 or 3 dimensions.
    """
    require_image_ndim(image)
    height, width = image.shape[:2]
    matrix = np.array([[1.0, 0.0, float(x)], [0.0, 1.0, float(y)]], dtype=np.float32)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=interpolation,
        borderMode=border_mode,
        borderValue=border_value,
    )


def rotate(
    image: np.ndarray,
    angle: float,
    center: tuple[float, float] | None = None,
    scale: float = 1.0,
    *,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: float | tuple[float, ...] = 0,
) -> np.ndarray:
    """Rotate an image by `angle` degrees counter-clockwise around `center`.

    The output has the same size as `image`; content rotated outside the
    original canvas is cropped. Use `rotate_bound` to keep the full rotated
    content by expanding the canvas instead.

    Parameters
    ----------
    image : np.ndarray
        Input image with shape ``(H, W)`` or ``(H, W, C)``.
    angle : float
        Rotation angle in degrees, counter-clockwise.
    center : tuple of float, optional
        Rotation center in ``(x, y)`` pixel coordinates. Defaults to the
        image center.
    scale : float, default 1.0
        Isotropic scale factor applied together with the rotation.
    interpolation : int, default ``cv2.INTER_LINEAR``
        Interpolation flag passed through to ``cv2.warpAffine``.
    border_mode : int, default ``cv2.BORDER_CONSTANT``
        Border mode for pixels exposed by the rotation.
    border_value : float or tuple of float, default 0
        Fill value used when `border_mode` is ``cv2.BORDER_CONSTANT``.

    Returns
    -------
    np.ndarray
        A new array with the same shape and dtype as `image`.

    Raises
    ------
    ValueError
        If `image` does not have 2 or 3 dimensions.
    """
    require_image_ndim(image)
    height, width = image.shape[:2]
    if center is None:
        center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=interpolation,
        borderMode=border_mode,
        borderValue=border_value,
    )


def rotate_bound(
    image: np.ndarray,
    angle: float,
    *,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: float | tuple[float, ...] = 0,
) -> np.ndarray:
    """Rotate an image by `angle` degrees, expanding the canvas so no content is cropped.

    Parameters
    ----------
    image : np.ndarray
        Input image with shape ``(H, W)`` or ``(H, W, C)``.
    angle : float
        Rotation angle in degrees, counter-clockwise.
    interpolation : int, default ``cv2.INTER_LINEAR``
        Interpolation flag passed through to ``cv2.warpAffine``.
    border_mode : int, default ``cv2.BORDER_CONSTANT``
        Border mode for the newly exposed canvas corners.
    border_value : float or tuple of float, default 0
        Fill value used when `border_mode` is ``cv2.BORDER_CONSTANT``.

    Returns
    -------
    np.ndarray
        A new array holding the fully rotated image; its shape generally
        differs from `image`'s.

    Raises
    ------
    ValueError
        If `image` does not have 2 or 3 dimensions.
    """
    require_image_ndim(image)
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_width = int(height * sin + width * cos)
    new_height = int(height * cos + width * sin)

    matrix[0, 2] += (new_width / 2) - center[0]
    matrix[1, 2] += (new_height / 2) - center[1]

    return cv2.warpAffine(
        image,
        matrix,
        (new_width, new_height),
        flags=interpolation,
        borderMode=border_mode,
        borderValue=border_value,
    )


FlipDirection = Literal["horizontal", "vertical", "both"]

_FLIP_CODES: dict[FlipDirection, int] = {
    "horizontal": 1,
    "vertical": 0,
    "both": -1,
}


def flip(image: np.ndarray, direction: FlipDirection) -> np.ndarray:
    """Flip an image horizontally, vertically, or both.

    Parameters
    ----------
    image : np.ndarray
        Input image with shape ``(H, W)`` or ``(H, W, C)``.
    direction : {"horizontal", "vertical", "both"}
        Flip axis.

    Returns
    -------
    np.ndarray
        A new array with the same shape and dtype as `image`.

    Raises
    ------
    ValueError
        If `image` does not have 2 or 3 dimensions, or `direction` is not
        one of the accepted values.
    """
    require_image_ndim(image)
    if direction not in _FLIP_CODES:
        raise ValueError(f"direction must be one of {tuple(_FLIP_CODES)}, got {direction!r}")
    return cv2.flip(image, _FLIP_CODES[direction])
