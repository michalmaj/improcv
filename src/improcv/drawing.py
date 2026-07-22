"""Annotation drawing: contours, bounding boxes, and image montages."""

from __future__ import annotations

import math
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
from improcv.types import BoundingBox

__all__ = [
    "draw_bounding_boxes",
    "draw_contours",
    "montage",
]

_MAX_MONTAGE_BYTES = 512 * 1024 * 1024  # 512 MiB
_MAX_DRAWING_THICKNESS = 32767  # OpenCV's own internal MAX_THICKNESS limit


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

    Requires an integral number (`bool` rejected), and nonzero -- `0` is
    rejected explicitly since OpenCV silently treats it as a thin outline
    rather than "draw nothing" (verified directly). Positive values draw
    an outline and must not exceed `_MAX_DRAWING_THICKNESS` (``32767``),
    OpenCV's own internal ``MAX_THICKNESS`` limit (verified:
    `thickness=32768` reaches a raw ``cv2.error: thickness <=
    MAX_THICKNESS``; `32767` is accepted). Negative values fill the
    shape's interior instead and have no such cap, but must still fit
    signed `int32`.
    """
    require_integral(thickness, "thickness")
    assert isinstance(thickness, numbers.Integral)  # narrows for the type checker
    thickness_int = int(thickness)
    if thickness_int == 0:
        raise ValueError("thickness must not be 0, got 0")
    if thickness_int > 0:
        if thickness_int > _MAX_DRAWING_THICKNESS:
            raise ValueError(
                f"positive thickness must not exceed {_MAX_DRAWING_THICKNESS}, got {thickness_int}"
            )
    else:
        require_fits_dtype(thickness_int, np.int32, "thickness")
    return thickness_int


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
            raise TypeError(f"contours[{i}] must have dtype int32, got {contour.dtype}")
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


def _require_valid_boxes(boxes: object) -> list[BoundingBox]:
    """Raise TypeError/ValueError unless `boxes` is a valid sequence of BoundingBox values.

    Rejects a single `BoundingBox` passed directly (a `BoundingBox` is
    itself a tuple, so an unguarded loop would otherwise silently iterate
    its 4 individual int fields as if they were separate boxes). Each
    element's fields must be integral (no `bool`), `width`/`height`
    positive, and `x`, `y`, `width`, `height`, `x + width`, `y + height`
    must all fit signed `int32`. An empty sequence is valid (no-op).

    Fields are converted to plain Python `int` *before* computing `x +
    width`/`y + height` -- verified that adding two `np.int32` scalars
    close to `int32`'s max silently wraps around (with only a
    `RuntimeWarning`, easy to miss) rather than raising, which would let
    an out-of-range box slip past the very check meant to catch it. Python
    `int` addition never overflows, so the sum is always computed
    correctly before being checked.
    """
    if isinstance(boxes, BoundingBox) or not isinstance(boxes, Sequence):
        raise TypeError(
            f"boxes must be a sequence of BoundingBox values, not a single {type(boxes).__name__}"
        )
    result: list[BoundingBox] = []
    for i, box in enumerate(boxes):
        if not isinstance(box, BoundingBox):
            raise TypeError(f"boxes[{i}] must be a BoundingBox, got {type(box).__name__}")
        x, y, width, height = box
        for field_name, field_value in (
            ("x", x),
            ("y", y),
            ("width", width),
            ("height", height),
        ):
            require_integral(field_value, f"boxes[{i}].{field_name}")
            require_fits_dtype(field_value, np.int32, f"boxes[{i}].{field_name}")
        if width <= 0:
            raise ValueError(f"boxes[{i}].width must be positive, got {width}")
        if height <= 0:
            raise ValueError(f"boxes[{i}].height must be positive, got {height}")
        x_int, y_int, width_int, height_int = int(x), int(y), int(width), int(height)
        require_fits_dtype(x_int + width_int, np.int32, f"boxes[{i}].x + boxes[{i}].width")
        require_fits_dtype(y_int + height_int, np.int32, f"boxes[{i}].y + boxes[{i}].height")
        result.append(BoundingBox(x_int, y_int, width_int, height_int))
    return result


def draw_bounding_boxes(
    image: np.ndarray,
    boxes: Sequence[BoundingBox],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw axis-aligned bounding boxes onto a copy of `image`.

    Parameters
    ----------
    image : np.ndarray
        A `uint8` BGR image, shape ``(H, W, 3)`` -- same contract as
        `draw_contours`, for the same reason.
    boxes : sequence of BoundingBox
        As returned by `bounding_boxes`. Must be an actual sequence, not a
        single `BoundingBox`. Each element's `x`/`y`/`width`/`height` must
        be integral (no `bool`), `width > 0`, `height > 0`, and each of
        `x`, `y`, `width`, `height`, `x + width`, `y + height` must fit
        signed `int32`. A box may partially or entirely lie outside
        `image`'s bounds -- OpenCV legitimately clips it. `[]` is accepted
        and is a no-op.
    color : tuple of int, default (0, 255, 0)
        BGR color; see `_normalize_bgr_color`.
    thickness : int, default 2
        Border thickness in pixels; a negative value fills the box
        instead. See `_normalize_thickness`.

    Returns
    -------
    np.ndarray
        A new array with the boxes drawn on top of a copy of `image`.
        `image` itself is never modified.

    Raises
    ------
    ValueError
        If `image` does not have exactly 3 channels or is empty; if a
        `color` channel or `thickness` is out of range, or `thickness` is
        `0`; if any box's `width`/`height` isn't positive, or a
        field/computed edge doesn't fit signed `int32`.
    TypeError
        If `image` does not have dtype ``uint8``; if `boxes` is a single
        `BoundingBox` instead of a sequence, or any element isn't a
        `BoundingBox`; if `color`/`thickness` isn't an integral type, or
        `color` isn't a 3-tuple; if a box field is not integral or is
        `bool`.

    Notes
    -----
    Uses `cv2.rectangle`'s ``Rect`` overload (``(x, y, width, height)``)
    rather than its two-point overload (``(x, y)``, ``(x + width, y +
    height)``) -- verified the two-point overload treats both corners as
    inclusive, drawing a filled region one pixel wider and taller than
    intended (e.g. `BoundingBox(2, 3, 4, 5)` filled would extend to `x=6`,
    `y=8` instead of the correct `x=5`, `y=7`).
    """
    require_channels(image, 3)
    require_dtype(image, (np.uint8,))
    color_bgr = _normalize_bgr_color(color)
    thickness_int = _normalize_thickness(thickness)
    valid_boxes = _require_valid_boxes(boxes)

    result = image.copy()
    for box in valid_boxes:
        x, y, width, height = box
        cv2.rectangle(result, (int(x), int(y), int(width), int(height)), color_bgr, thickness_int)
    return result


def _require_valid_montage_images(images: object) -> list[np.ndarray]:
    """Raise TypeError/ValueError unless `images` is a valid, non-empty sequence of images.

    Rejects a single `np.ndarray` passed directly. Each element must be a
    `uint8` `np.ndarray` with positive height/width, and the whole list
    must share the same `ndim` (2 or 3) and, if 3D, the same channel count
    (3 or 4 only -- no other count accepted).
    """
    if isinstance(images, np.ndarray) or not isinstance(images, Sequence):
        raise TypeError(
            f"images must be a sequence of arrays, not a single {type(images).__name__}"
        )
    result = list(images)
    if len(result) == 0:
        raise ValueError("images must not be empty")

    reference_ndim: int | None = None
    reference_channels: int | None = None
    for i, image in enumerate(result):
        if not isinstance(image, np.ndarray):
            raise TypeError(f"images[{i}] must be an np.ndarray, got {type(image).__name__}")
        if image.dtype != np.uint8:
            raise TypeError(f"images[{i}] must have dtype uint8, got {image.dtype}")
        if image.ndim not in (2, 3) or image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError(f"images[{i}] must not be empty, got shape {image.shape}")
        if image.ndim == 3 and image.shape[2] not in (3, 4):
            raise ValueError(
                f"images[{i}] must have 3 or 4 channels if 3-dimensional, got shape {image.shape}"
            )
        channels = 1 if image.ndim == 2 else image.shape[2]
        if reference_ndim is None:
            reference_ndim = image.ndim
            reference_channels = channels
        elif image.ndim != reference_ndim or channels != reference_channels:
            raise ValueError(
                f"images[{i}] has {channels} channel(s) (ndim {image.ndim}), expected "
                f"{reference_channels} channel(s) (ndim {reference_ndim}) to match images[0]"
            )
    return result


def _require_valid_montage_dim(value: object, name: str) -> int:
    """Raise TypeError/ValueError unless `value` is a positive int32-bounded integral.

    Returns it normalized to plain `int`.
    """
    require_integral(value, name)
    assert isinstance(value, numbers.Integral)  # narrows for the type checker
    require_fits_dtype(value, np.int32, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return int(value)


def _require_valid_fill_value(fill_value: object) -> int:
    """Raise TypeError/ValueError unless `fill_value` is an integral in [0, 255].

    Returns it normalized to plain `int`.
    """
    require_integral(fill_value, "fill_value")
    assert isinstance(fill_value, numbers.Integral)  # narrows for the type checker
    require_range(fill_value, 0, 255, "fill_value")
    return int(fill_value)


def _pick_interpolation(image: np.ndarray, tile_width: int, tile_height: int) -> int:
    """Pick INTER_AREA for shrinking, INTER_LINEAR for enlarging or mixed scaling.

    OpenCV's own guidance: INTER_AREA gives moire-free results when
    shrinking but is not recommended for enlarging; INTER_LINEAR is the
    general-purpose choice for enlarging or mixed-direction scaling.
    """
    shrinking = tile_width <= image.shape[1] and tile_height <= image.shape[0]
    return cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR


def montage(
    images: Sequence[np.ndarray],
    tile_width: int,
    tile_height: int,
    columns: int | None = None,
    fill_value: int = 0,
) -> np.ndarray:
    """Tile multiple images into a single grid image.

    Parameters
    ----------
    images : sequence of np.ndarray
        Images to tile, in row-major grid order. Must be an actual
        sequence (not a single `np.ndarray`), non-empty, each element a
        `uint8` array with positive height/width, and either all 2D
        (grayscale, ``(H, W)``), all 3-channel BGR (``(H, W, 3)``), or all
        4-channel BGRA (``(H, W, 4)``) -- the whole list must share the
        same `ndim` and channel count. No other channel count is
        accepted.
    tile_width, tile_height : int
        Each image is resized to exactly this size before tiling -- a hard
        resize, not aspect-ratio-preserving (unlike `resize`). Must be a
        positive integral number (no `bool`/`float`) fitting signed
        `int32`. Shrinking uses `cv2.INTER_AREA`; enlarging or mixed
        scaling uses `cv2.INTER_LINEAR`, chosen per image.
    columns : int, optional
        Number of grid columns; same integral/`int32`/positive contract as
        `tile_width`/`tile_height` when given. If omitted, computed as
        ``math.isqrt(len(images) - 1) + 1`` (an integer-only equivalent of
        ``ceil(sqrt(len(images)))``) for a roughly square grid. Rows are
        always ``ceil(len(images) / columns)``.
    fill_value : int, default 0
        Value used for any leftover grid cells when `images` doesn't
        exactly fill the grid. Must be an integral number (no `bool`) in
        `[0, 255]`.

    Returns
    -------
    np.ndarray
        A new array, shape ``(rows * tile_height, columns * tile_width)``
        or ``(..., C)``, dtype ``uint8``. Never shares memory with any
        input image.

    Raises
    ------
    ValueError
        If `images` is empty, any image has non-positive height/width or a
        disallowed channel count, the images don't share a consistent
        `ndim`/channel count, any of `tile_width`/`tile_height`/`columns`
        is non-positive or `fill_value` is out of range, or the computed
        output size would exceed an internal safety limit.
    TypeError
        If `images` is a single `np.ndarray` instead of a sequence, any
        element isn't an `np.ndarray` or isn't `uint8`, or any of
        `tile_width`/`tile_height`/`columns`/`fill_value` isn't an
        integral type (including `bool`/`float`).
    """
    valid_images = _require_valid_montage_images(images)
    tile_width_int = _require_valid_montage_dim(tile_width, "tile_width")
    tile_height_int = _require_valid_montage_dim(tile_height, "tile_height")
    columns_int = (
        _require_valid_montage_dim(columns, "columns")
        if columns is not None
        else math.isqrt(len(valid_images) - 1) + 1
    )
    fill_value_int = _require_valid_fill_value(fill_value)

    rows = (len(valid_images) + columns_int - 1) // columns_int
    output_height = rows * tile_height_int
    output_width = columns_int * tile_width_int

    first = valid_images[0]
    channels = 1 if first.ndim == 2 else first.shape[2]
    output_bytes = output_height * output_width * channels
    if output_bytes > _MAX_MONTAGE_BYTES:
        raise ValueError(
            f"montage output would require {output_bytes} bytes, exceeding the "
            f"{_MAX_MONTAGE_BYTES}-byte safety limit -- reduce tile_width/tile_height/columns "
            "or the number of images"
        )

    shape = (
        (output_height, output_width)
        if first.ndim == 2
        else (output_height, output_width, channels)
    )
    canvas = np.full(shape, fill_value_int, dtype=np.uint8)

    for i, image in enumerate(valid_images):
        row, col = divmod(i, columns_int)
        interpolation = _pick_interpolation(image, tile_width_int, tile_height_int)
        tile = cv2.resize(image, (tile_width_int, tile_height_int), interpolation=interpolation)
        y0 = row * tile_height_int
        x0 = col * tile_width_int
        canvas[y0 : y0 + tile_height_int, x0 : x0 + tile_width_int] = tile

    return canvas
