"""Annotation drawing: contours, bounding boxes, and image montages."""

from __future__ import annotations

import numbers
from collections.abc import Sequence

import cv2
import numpy as np

from improcv._validation import (
    require_channels,
    require_dtype,
    require_fits_dtype,
    require_integral,
    require_range,
)
from improcv.contours import Contour

__all__ = [
    "draw_contours",
]


def _normalize_bgr_color(color: object) -> tuple[int, int, int]:
    """Validate and normalize a BGR color argument.

    Requires exactly a `tuple` of length 3 (not just any sequence -- this
    catches a caller accidentally passing a list or array instead), each
    element an integral number (`bool` rejected) in ``[0, 255]``. Returns
    a plain-`int` 3-tuple so a NumPy scalar never reaches the `cv2.*` call.
    """
    if not isinstance(color, tuple):
        raise TypeError(f"color must be a tuple, got {type(color).__name__}")
    if len(color) != 3:
        raise ValueError(f"color must have exactly 3 elements, got {len(color)}")
    normalized: list[int] = []
    for i, channel in enumerate(color):
        require_integral(channel, f"color[{i}]")
        require_range(channel, 0, 255, f"color[{i}]")
        normalized.append(int(channel))
    return (normalized[0], normalized[1], normalized[2])


def _normalize_thickness(thickness: object) -> int:
    """Validate and normalize a `thickness` argument.

    Requires an integral number (`bool` rejected), fitting signed
    ``int32`` (OpenCV raises an unhelpful raw ``cv2.error`` for values far
    outside a sane range rather than a clear message), and nonzero -- `0`
    is rejected explicitly since OpenCV silently treats it as a thin
    outline rather than "draw nothing" (verified directly). Positive
    values draw an outline; negative values fill the shape's interior.
    """
    require_integral(thickness, "thickness")
    assert isinstance(thickness, numbers.Integral)  # narrows for the type checker
    require_fits_dtype(thickness, np.int32, "thickness")
    if thickness == 0:
        raise ValueError("thickness must not be 0, got 0")
    return int(thickness)


def _require_valid_contours(contours: object) -> list[np.ndarray]:
    """Raise TypeError/ValueError unless `contours` is a valid sequence of Contour arrays.

    Rejects a single `np.ndarray` passed directly (must be wrapped in a
    list/sequence of contours) -- otherwise it would be silently iterated
    point-by-point. Each element must be `int32`, shape ``(N, 1, 2)``,
    ``N > 0``. An empty sequence is valid (no-op).
    """
    if isinstance(contours, np.ndarray) or not isinstance(contours, Sequence):
        raise TypeError(
            f"contours must be a sequence of Contour arrays, not a single {type(contours).__name__}"
        )
    result = list(contours)
    for i, contour in enumerate(result):
        if not isinstance(contour, np.ndarray):
            raise TypeError(f"contours[{i}] must be an np.ndarray, got {type(contour).__name__}")
        if contour.dtype != np.int32:
            raise ValueError(f"contours[{i}] must have dtype int32, got {contour.dtype}")
        if contour.ndim != 3 or contour.shape[1:] != (1, 2) or contour.shape[0] == 0:
            raise ValueError(
                f"contours[{i}] must have shape (N, 1, 2) with N > 0, got shape {contour.shape}"
            )
    return result


def draw_contours(
    image: np.ndarray,
    contours: Sequence[Contour],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw contour outlines (or fills) onto a copy of `image`.

    Parameters
    ----------
    image : np.ndarray
        A `uint8` BGR image, shape ``(H, W, 3)``. Grayscale and BGRA are
        rejected explicitly: OpenCV silently uses only the first element
        of `color` as a grayscale value on a single-channel image rather
        than raising.
    contours : sequence of Contour
        As returned by `find_contours`. Must be an actual sequence, not a
        single `np.ndarray`. Each element must be an `int32` array of
        shape ``(N, 1, 2)`` with ``N > 0``. `[]` is accepted and is a
        no-op.
    color : tuple of int, default (0, 255, 0)
        BGR color; see `_normalize_bgr_color`.
    thickness : int, default 2
        Outline thickness in pixels; a negative value fills each contour's
        interior instead. When filling **multiple** contours without
        hierarchy, OpenCV applies the even-odd rule across the complete
        collection -- contours originating from unrelated groups should be
        drawn in separate calls to avoid an unintentional "hole." This
        function does not accept or thread hierarchy information.

    Returns
    -------
    np.ndarray
        A new array with the contours drawn on top of a copy of `image`.
        `image` itself is never modified.

    Raises
    ------
    ValueError
        If `image` does not have exactly 3 channels or is empty; if a
        `color` channel or `thickness` is out of range, or `thickness` is
        `0`; if any contour has the wrong shape or is empty.
    TypeError
        If `image` does not have dtype ``uint8``; if `contours` is a
        single `np.ndarray` instead of a sequence, or any element isn't an
        `int32` array; if `color`/`thickness` isn't an integral type, or
        `color` isn't a 3-tuple.
    """
    require_channels(image, 3)
    require_dtype(image, (np.uint8,))
    color_bgr = _normalize_bgr_color(color)
    thickness_int = _normalize_thickness(thickness)
    valid_contours = _require_valid_contours(contours)

    result = image.copy()
    cv2.drawContours(result, valid_contours, -1, color_bgr, thickness_int)
    return result
