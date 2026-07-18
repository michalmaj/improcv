"""Geometric image transformations."""

from __future__ import annotations

import math
from typing import Literal

import cv2
import numpy as np

from improcv._validation import (
    require_image_ndim,
    require_non_negative,
    require_one_of,
    require_positive,
    require_positive_int,
)
from improcv.types import Image, TransformMatrix

__all__ = [
    "resize",
    "translate",
    "rotate",
    "rotate_bound",
    "flip",
    "crop",
    "center_crop",
    "pad",
    "warp_affine",
    "warp_perspective",
]


def resize(
    image: Image,
    width: int | None = None,
    height: int | None = None,
    interpolation: int = cv2.INTER_AREA,
) -> Image:
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
        positive integer, or if `image` is empty or does not have 2 or 3
        dimensions.
    TypeError
        If `width` or `height` is given but is not an ``int``.
    """
    require_image_ndim(image)
    if width is None and height is None:
        raise ValueError("at least one of width or height must be given")
    if width is not None:
        require_positive_int(width, "width")
    if height is not None:
        require_positive_int(height, "height")

    source_height, source_width = image.shape[:2]

    if width is not None and height is not None:
        target_size = (width, height)
    elif width is not None:
        computed_height = max(1, round(width * source_height / source_width))
        target_size = (width, computed_height)
    else:
        assert height is not None
        computed_width = max(1, round(height * source_width / source_height))
        target_size = (computed_width, height)

    return cv2.resize(image, target_size, interpolation=interpolation)


def translate(
    image: Image,
    x: int,
    y: int,
    *,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: float | tuple[float, ...] = 0,
) -> Image:
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
    image: Image,
    angle: float,
    center: tuple[float, float] | None = None,
    scale: float = 1.0,
    *,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: float | tuple[float, ...] = 0,
) -> Image:
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
        # Pixel centers sit at integer coordinates 0..N-1, so the center of
        # the grid is (N-1)/2, not N/2 — the latter is off by half a pixel
        # and loses a full row/column on 90/180-degree rotations.
        center = ((width - 1) / 2, (height - 1) / 2)
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
    image: Image,
    angle: float,
    *,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: float | tuple[float, ...] = 0,
) -> Image:
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
    # Pixel centers sit at integer coordinates 0..N-1, so the center of the
    # grid is (N-1)/2, not N/2 — the latter is off by half a pixel and loses
    # a full row/column on 90/180-degree rotations.
    center = ((width - 1) / 2, (height - 1) / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    # ceil, not int(): truncating down can undercount the required canvas
    # by up to 1px (e.g. a 2x2 image at 45deg needs 3px, int() gives 2 —
    # no expansion at all — and rotate_bound's contract is "never crop").
    # round() first absorbs floating-point noise from cos/sin at exact
    # multiples of 90deg (e.g. 20.000000000000004), which would otherwise
    # make ceil() over-expand the canvas by a spurious extra pixel.
    new_width = math.ceil(round(height * sin + width * cos, 6))
    new_height = math.ceil(round(height * cos + width * sin, 6))

    # Same off-by-half-pixel correction for the new canvas's center.
    matrix[0, 2] += (new_width - 1) / 2 - center[0]
    matrix[1, 2] += (new_height - 1) / 2 - center[1]

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


def flip(image: Image, direction: FlipDirection) -> Image:
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
    require_one_of(direction, _FLIP_CODES, "direction")
    return cv2.flip(image, _FLIP_CODES[direction])


def crop(image: Image, x: int, y: int, width: int, height: int) -> Image:
    """Crop a rectangular region from an image.

    Parameters
    ----------
    image : np.ndarray
        Input image with shape ``(H, W)`` or ``(H, W, C)``.
    x, y : int
        Top-left corner of the region, in pixels.
    width, height : int
        Size of the region, in pixels.

    Returns
    -------
    np.ndarray
        A new array holding the cropped region — always a copy, never a
        view into `image`.

    Raises
    ------
    ValueError
        If `image` does not have 2 or 3 dimensions, `x`/`y` are negative,
        `width`/`height` are not positive, or the region exceeds the
        image bounds.
    """
    require_image_ndim(image)
    require_non_negative(x, "x")
    require_non_negative(y, "y")
    require_positive(width, "width")
    require_positive(height, "height")

    source_height, source_width = image.shape[:2]
    if x + width > source_width or y + height > source_height:
        raise ValueError(
            f"crop region ({x}, {y}, {width}, {height}) exceeds image bounds "
            f"({source_width}, {source_height})"
        )

    return image[y : y + height, x : x + width].copy()


def center_crop(image: Image, width: int, height: int) -> Image:
    """Crop a `width` x `height` region centered on `image`.

    Parameters
    ----------
    image : np.ndarray
        Input image with shape ``(H, W)`` or ``(H, W, C)``.
    width, height : int
        Size of the centered region, in pixels.

    Returns
    -------
    np.ndarray
        A new array holding the cropped region — always a copy.

    Raises
    ------
    ValueError
        Same conditions as `crop`; in particular, a `width`/`height` larger
        than `image` raises via a negative computed origin.
    """
    require_image_ndim(image)
    source_height, source_width = image.shape[:2]
    x = (source_width - width) // 2
    y = (source_height - height) // 2
    return crop(image, x, y, width, height)


PadMode = Literal["constant", "reflect", "replicate", "wrap"]

_BORDER_MODES: dict[PadMode, int] = {
    "constant": cv2.BORDER_CONSTANT,
    "reflect": cv2.BORDER_REFLECT_101,
    "replicate": cv2.BORDER_REPLICATE,
    "wrap": cv2.BORDER_WRAP,
}


def pad(
    image: Image,
    top: int,
    bottom: int,
    left: int,
    right: int,
    *,
    mode: PadMode = "constant",
    value: float | tuple[float, ...] = 0,
) -> Image:
    """Add a border around an image.

    Parameters
    ----------
    image : np.ndarray
        Input image with shape ``(H, W)`` or ``(H, W, C)``.
    top, bottom, left, right : int
        Border width in pixels on each side; must be non-negative.
    mode : {"constant", "reflect", "replicate", "wrap"}, default "constant"
        Border fill strategy.
    value : float or tuple of float, default 0
        Fill value used when `mode` is ``"constant"``.

    Returns
    -------
    np.ndarray
        A new, larger array with `image`'s content placed at
        ``[top:top+H, left:left+W]``.

    Raises
    ------
    ValueError
        If `image` does not have 2 or 3 dimensions, any border amount is
        negative, or `mode` is not one of the accepted values.
    """
    require_image_ndim(image)
    require_non_negative(top, "top")
    require_non_negative(bottom, "bottom")
    require_non_negative(left, "left")
    require_non_negative(right, "right")
    require_one_of(mode, _BORDER_MODES, "mode")

    return cv2.copyMakeBorder(
        image, top, bottom, left, right, borderType=_BORDER_MODES[mode], value=value
    )


def warp_affine(
    image: Image,
    matrix: TransformMatrix,
    output_size: tuple[int, int] | None = None,
    *,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: float | tuple[float, ...] = 0,
) -> Image:
    """Apply an affine transformation to an image.

    Parameters
    ----------
    image : np.ndarray
        Input image with shape ``(H, W)`` or ``(H, W, C)``.
    matrix : np.ndarray
        A ``(2, 3)`` affine transformation matrix.
    output_size : tuple of int, optional
        ``(width, height)`` of the output. Defaults to `image`'s size.
    interpolation : int, default ``cv2.INTER_LINEAR``
        Interpolation flag passed through to ``cv2.warpAffine``.
    border_mode : int, default ``cv2.BORDER_CONSTANT``
        Border mode for pixels exposed by the transform.
    border_value : float or tuple of float, default 0
        Fill value used when `border_mode` is ``cv2.BORDER_CONSTANT``.

    Returns
    -------
    np.ndarray
        A new array holding the warped image.

    Raises
    ------
    ValueError
        If `image` does not have 2 or 3 dimensions, `matrix` is not shaped
        ``(2, 3)``, or `output_size` is given but not a pair of positive
        ints. OpenCV silently ignores an invalid `dsize` and returns an
        array sized like the *input* instead of raising, so this must be
        checked here rather than left to ``cv2.warpAffine``.
    """
    require_image_ndim(image)
    if matrix.shape != (2, 3):
        raise ValueError(f"matrix must have shape (2, 3), got {matrix.shape}")
    height, width = image.shape[:2]
    if output_size is not None:
        require_positive_int(output_size[0], "output_size[0]")
        require_positive_int(output_size[1], "output_size[1]")
    size = output_size if output_size is not None else (width, height)
    return cv2.warpAffine(
        image, matrix, size, flags=interpolation, borderMode=border_mode, borderValue=border_value
    )


def warp_perspective(
    image: Image,
    matrix: TransformMatrix,
    output_size: tuple[int, int] | None = None,
    *,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: float | tuple[float, ...] = 0,
) -> Image:
    """Apply a perspective transformation to an image.

    Parameters
    ----------
    image : np.ndarray
        Input image with shape ``(H, W)`` or ``(H, W, C)``.
    matrix : np.ndarray
        A ``(3, 3)`` perspective transformation matrix.
    output_size : tuple of int, optional
        ``(width, height)`` of the output. Defaults to `image`'s size.
    interpolation : int, default ``cv2.INTER_LINEAR``
        Interpolation flag passed through to ``cv2.warpPerspective``.
    border_mode : int, default ``cv2.BORDER_CONSTANT``
        Border mode for pixels exposed by the transform.
    border_value : float or tuple of float, default 0
        Fill value used when `border_mode` is ``cv2.BORDER_CONSTANT``.

    Returns
    -------
    np.ndarray
        A new array holding the warped image.

    Raises
    ------
    ValueError
        If `image` does not have 2 or 3 dimensions, `matrix` is not shaped
        ``(3, 3)``, or `output_size` is given but not a pair of positive
        ints. OpenCV silently ignores an invalid `dsize` and returns an
        array sized like the *input* instead of raising, so this must be
        checked here rather than left to ``cv2.warpPerspective``.
    """
    require_image_ndim(image)
    if matrix.shape != (3, 3):
        raise ValueError(f"matrix must have shape (3, 3), got {matrix.shape}")
    height, width = image.shape[:2]
    if output_size is not None:
        require_positive_int(output_size[0], "output_size[0]")
        require_positive_int(output_size[1], "output_size[1]")
    size = output_size if output_size is not None else (width, height)
    return cv2.warpPerspective(
        image, matrix, size, flags=interpolation, borderMode=border_mode, borderValue=border_value
    )
