"""Region analysis: connected components, distance transform, flood fill."""

from __future__ import annotations

import numbers
from collections.abc import Sequence
from typing import Literal, NamedTuple, cast

import cv2
import numpy as np
import numpy.typing as npt

from improcv._validation import (
    require_bool,
    require_dtype,
    require_finite,
    require_image_ndim,
    require_integral,
    require_non_negative,
    require_one_of,
)
from improcv.types import BoundingBox, Image, ImageFloat32, Mask

__all__ = [
    "connected_components",
    "connected_components_with_stats",
    "distance_transform",
    "flood_fill",
    "Connectivity",
    "Labels",
    "ComponentStats",
    "Centroids",
    "DistanceType",
    "DistanceMaskSize",
    "FloodFillResult",
]

Connectivity = Literal[4, 8]
_CONNECTIVITIES: tuple[Connectivity, ...] = (4, 8)


def _require_integral_choice(value: object, allowed: tuple[int, ...], name: str) -> int:
    """Validate `value` as an integral choice from `allowed`, returning it as plain `int`.

    A bare `require_one_of` membership check does not catch a `float` equal
    to an accepted value (`4.0 == 4` in Python) — verified directly that
    this let a `float` reach a raw `cv2.error` (or, for `flood_fill`'s
    `connectivity`, an unrelated `TypeError` from the `|` operator) instead
    of a clear validation error. `require_integral` rejects `bool` and any
    non-integral type first; the `int(...)` conversion then guarantees a
    plain `int` reaches the underlying `cv2.*` call regardless of whether
    `value` was a builtin `int` or a NumPy integer scalar.
    """
    require_integral(value, name)
    value_int = int(value)  # type: ignore[arg-type]
    require_one_of(value_int, allowed, name)
    return value_int


Labels = npt.NDArray[np.int32]
"""A label map: shape ``(H, W)``, dtype ``int32``. Label ``0`` is always the
background; labels ``1..N`` are the connected foreground components."""


def connected_components(mask: Mask, connectivity: Connectivity = 8) -> tuple[int, Labels]:
    """Label connected components of foreground pixels in a binary mask.

    Parameters
    ----------
    mask : np.ndarray
        Input mask with shape ``(H, W)``, dtype ``uint8``. Any nonzero value
        is treated as foreground, not only ``255`` (verified directly).
    connectivity : {4, 8}, default 8
        Pixel connectivity used to group foreground pixels into components.
        Verified directly: two foreground pixels touching only at a corner
        are counted as 2 separate components under 4-connectivity, but as
        1 merged component under 8-connectivity.

    Returns
    -------
    num_labels : int
        The number of distinct labels, **including the background** — e.g.
        ``num_labels == 3`` means "background plus 2 real components".
    labels : np.ndarray
        Shape ``(H, W)``, dtype ``int32``. Label ``0`` is always the
        background. A new, independent array; `mask` is never modified.

    Raises
    ------
    ValueError
        If `mask` does not have exactly 2 dimensions or is empty, or
        `connectivity` is not one of the accepted values.
    TypeError
        If `mask` does not have dtype ``uint8``.
    """
    require_image_ndim(mask, ndims=(2,))
    require_dtype(mask, (np.uint8,))
    connectivity_int = _require_integral_choice(connectivity, _CONNECTIVITIES, "connectivity")
    num_labels, labels = cv2.connectedComponents(mask, connectivity=connectivity_int)
    return num_labels, cast(Labels, labels)


ComponentStats = npt.NDArray[np.int32]
"""Per-component stats: shape ``(N, 5)``, dtype ``int32``. Columns are
``[x, y, width, height, area]`` (the bounding box and pixel area of each
labeled component). Row ``0`` is the background — see `connected_components_with_stats`."""

Centroids = npt.NDArray[np.float64]
"""Per-component centroids: shape ``(N, 2)``, dtype ``float64``. Columns are
``[cx, cy]``. Row ``0`` is the background — see `connected_components_with_stats`."""


def connected_components_with_stats(
    mask: Mask, connectivity: Connectivity = 8
) -> tuple[int, Labels, ComponentStats, Centroids]:
    """Label connected components and compute their bounding-box stats and centroids.

    A separate function from `connected_components` (not a flag), matching
    OpenCV's own split: computing stats/centroids is measurably more
    expensive, so a caller who only needs labels isn't forced to pay for it.

    Parameters
    ----------
    mask : np.ndarray
        Input mask with shape ``(H, W)``, dtype ``uint8``. Any nonzero value
        is treated as foreground, not only ``255`` (verified directly).
    connectivity : {4, 8}, default 8
        Pixel connectivity — see `connected_components`.

    Returns
    -------
    num_labels : int
        The number of distinct labels, including the background.
    labels : np.ndarray
        Shape ``(H, W)``, dtype ``int32``. Label ``0`` is always the
        background.
    stats : np.ndarray
        Shape ``(num_labels, 5)``, dtype ``int32``, columns
        ``[x, y, width, height, area]``. ``stats[0]`` contains OpenCV's
        statistics for the background label — this is **not** guaranteed to
        be the whole image; it reflects wherever the background pixels
        actually are (verified directly). When there are no background
        pixels at all (an all-foreground mask), the background area is zero
        and its bounding box is OpenCV's own degenerate sentinel value
        (verified directly, identical on OpenCV 4.13 and 5.0).
    centroids : np.ndarray
        Shape ``(num_labels, 2)``, dtype ``float64``, columns ``[cx, cy]``.
        ``centroids[0]`` is the background's centroid, and may contain
        ``NaN`` when the background area is zero (verified directly) —
        callers should inspect ``stats[0, 4]`` (the background area) before
        relying on ``centroids[0]``.

    Notes
    -----
    ``stats.shape[0] == centroids.shape[0] == num_labels`` always holds.
    All three outputs are fresh, independent arrays; `mask` is never
    modified.

    Raises
    ------
    ValueError
        If `mask` does not have exactly 2 dimensions or is empty, or
        `connectivity` is not one of the accepted values.
    TypeError
        If `mask` does not have dtype ``uint8``.
    """
    require_image_ndim(mask, ndims=(2,))
    require_dtype(mask, (np.uint8,))
    connectivity_int = _require_integral_choice(connectivity, _CONNECTIVITIES, "connectivity")
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=connectivity_int
    )
    return num_labels, cast(Labels, labels), cast(ComponentStats, stats), cast(Centroids, centroids)


DistanceType = Literal["l1", "l2", "c"]
_DISTANCE_TYPE_FLAGS: dict[DistanceType, int] = {
    "l1": cv2.DIST_L1,
    "l2": cv2.DIST_L2,
    "c": cv2.DIST_C,
}

DistanceMaskSize = Literal[0, 3, 5]
# Verified directly against cv2.distanceTransform on OpenCV 4.13 and 5.0
# (identical on both): "l1"/"c" silently ignore mask_size 0 or 5, producing
# output identical to mask_size=3 -- not an error, so this validation is
# the only thing that prevents a caller from believing they requested a
# precision that was never actually applied.
_VALID_MASK_SIZES: dict[DistanceType, tuple[DistanceMaskSize, ...]] = {
    "l2": (0, 3, 5),
    "l1": (3,),
    "c": (3,),
}
_DEFAULT_MASK_SIZE: dict[DistanceType, DistanceMaskSize] = {"l2": 5, "l1": 3, "c": 3}


def distance_transform(
    mask: Mask,
    distance_type: DistanceType = "l2",
    mask_size: DistanceMaskSize | None = None,
) -> ImageFloat32:
    """Compute, for every nonzero pixel, its distance to the nearest zero pixel.

    Parameters
    ----------
    mask : np.ndarray
        Input mask with shape ``(H, W)``, dtype ``uint8``.
    distance_type : {"l1", "l2", "c"}, default "l2"
        Distance metric: ``"l1"`` (city-block), ``"l2"`` (Euclidean), or
        ``"c"`` (chessboard).
    mask_size : {0, 3, 5} or None, default None
        Distance-transform mask size. ``None`` resolves to a metric-specific
        default: ``5`` for ``"l2"``, ``3`` for ``"l1"``/``"c"``. Only
        ``"l2"`` accepts all three explicit values (``0`` selects OpenCV's
        precise algorithm); ``"l1"``/``"c"`` accept only ``3`` — verified
        directly that OpenCV silently produces the ``mask_size=3`` result
        for any other value with `"l1"`/`"c"` rather than erroring, so this
        function rejects that combination instead of silently ignoring it.

    Returns
    -------
    np.ndarray
        An `ImageFloat32` shaped ``(H, W)``, matching `mask`'s shape. A new
        array; `mask` is never modified.

    Raises
    ------
    ValueError
        If `mask` does not have exactly 2 dimensions or is empty, `mask`
        does not contain at least one zero-valued (background) pixel,
        `distance_type` is not one of the accepted values, or `mask_size`
        (explicit or defaulted) is not valid for the chosen `distance_type`.
    TypeError
        If `mask` does not have dtype ``uint8``, or an explicit `mask_size`
        is not `numbers.Integral` (rejecting `bool`).
    """
    require_image_ndim(mask, ndims=(2,))
    require_dtype(mask, (np.uint8,))
    if not np.any(mask == 0):
        # An all-foreground mask has no zero pixel to measure distance to;
        # cv2.distanceTransform returns large sentinel-like values in that
        # case (verified directly) that look like plausible float32
        # distances but are meaningless.
        raise ValueError("mask must contain at least one zero-valued background pixel")
    require_one_of(distance_type, tuple(_DISTANCE_TYPE_FLAGS), "distance_type")
    if mask_size is None:
        resolved_mask_size: int = _DEFAULT_MASK_SIZE[distance_type]
    else:
        require_integral(mask_size, "mask_size")
        resolved_mask_size = int(mask_size)
    valid_sizes = _VALID_MASK_SIZES[distance_type]
    if resolved_mask_size not in valid_sizes:
        raise ValueError(
            f"mask_size {resolved_mask_size} is not valid for distance_type "
            f"{distance_type!r}; accepted values are {valid_sizes}"
        )
    result = cv2.distanceTransform(mask, _DISTANCE_TYPE_FLAGS[distance_type], resolved_mask_size)
    return cast(ImageFloat32, result)


class FloodFillResult(NamedTuple):
    """Result of `flood_fill`.

    `filled_count` is `cv2.floodFill`'s own return value (number of pixels
    repainted). `image`/`mask` are always fresh, independent arrays.
    """

    filled_count: int
    image: Image
    mask: Mask
    bounding_box: BoundingBox


_FLOOD_FILL_DTYPES = (np.uint8, np.float32)


def _resolve_channel_values(
    value: float | Sequence[float], channels: int, name: str, *, non_negative: bool = False
) -> tuple[float, ...]:
    """Resolve a scalar-or-per-channel parameter to an exact `channels`-length tuple of floats."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number or a sequence of real numbers, got bool")
    if isinstance(value, numbers.Real):
        if non_negative:
            require_non_negative(value, name)
        else:
            require_finite(value, name)
        return (float(value),) * channels
    if isinstance(value, (str, bytes, bytearray, memoryview)):
        # These are technically iterable (bytes/bytearray/memoryview yield
        # plain ints; str yields single-character strings), so without this
        # explicit rejection a bytes/bytearray value of the right length
        # would silently pass through the length/finiteness checks below as
        # if it were a genuine sequence of numbers -- verified directly that
        # b"x" (length 1) filled a grayscale image with 120 (ord("x")), no
        # error at all.
        raise TypeError(
            f"{name} must be a real number or a sequence of real numbers, "
            f"got {type(value).__name__}"
        )
    try:
        # `value` is a Sequence at this point (the numbers.Real case already
        # returned above), but pyright can't narrow numbers.Real exclusion
        # from the `float | Sequence[float]` union -- verified this is
        # exactly the case reached at runtime.
        values = list(value)  # type: ignore[arg-type]
    except TypeError:
        raise TypeError(
            f"{name} must be a real number or a sequence of real numbers, "
            f"got {type(value).__name__}"
        ) from None
    if len(values) != channels:
        raise ValueError(
            f"{name} must have {channels} element(s) matching image's channel count, "
            f"got {len(values)}"
        )
    for i, element in enumerate(values):
        if non_negative:
            require_non_negative(element, f"{name}[{i}]")
        else:
            require_finite(element, f"{name}[{i}]")
    return tuple(float(element) for element in values)


def _require_seed_point(
    value: object, width: int, height: int, name: str = "seed_point"
) -> tuple[int, int]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{name} must be a 2-tuple, got {value!r}")
    x, y = value
    require_integral(x, f"{name}[0]")
    require_integral(y, f"{name}[1]")
    x_int, y_int = int(x), int(y)
    if not (0 <= x_int < width and 0 <= y_int < height):
        raise ValueError(
            f"{name} must be within image bounds, got ({x_int}, {y_int}) "
            f"for a {width}x{height} image"
        )
    return x_int, y_int


def flood_fill(
    image: Image,
    seed_point: tuple[int, int],
    new_value: float | Sequence[float],
    lo_diff: float | Sequence[float] = 0,
    up_diff: float | Sequence[float] = 0,
    connectivity: Connectivity = 4,
    fixed_range: bool = False,
) -> FloodFillResult:
    """Fill a connected region of similar-colored pixels, starting from a seed point.

    Parameters
    ----------
    image : np.ndarray
        Input image with shape ``(H, W)`` or ``(H, W, C)``, dtype ``uint8``
        or ``float32``, 1 or 3 channels. Never modified: `cv2.floodFill`
        itself mutates its `image` argument in place by default (verified
        directly), but this function copies internally and fills the copy.
    seed_point : tuple of int
        ``(x, y)`` pixel coordinates (column, then row — matching OpenCV's
        point convention, not ``(row, column)``) where the fill starts.
        Accepts any `numbers.Integral`, including NumPy integer scalars
        (e.g. `np.int32`), not just plain `int` — coordinates routinely
        come straight out of NumPy arrays or other OpenCV results. Must be
        within `image`'s bounds.
    new_value : float or sequence of float
        Fill color. A scalar broadcasts to every channel; a sequence must
        have exactly as many elements as `image` has channels. For a
        ``uint8`` `image`, every element must be in ``[0, 255]`` — verified
        directly that `cv2.floodFill` itself silently saturates an
        out-of-range value instead of raising.
    lo_diff, up_diff : float or sequence of float, default 0
        Maximum lower/upper brightness/color difference between a
        candidate pixel and its comparison pixel (see `fixed_range`) still
        considered part of the region. Same scalar/sequence convention as
        `new_value`; must be non-negative.
    connectivity : {4, 8}, default 4
        Pixel connectivity. Defaults to 4, matching `cv2.floodFill`'s own
        conventional default (unlike `connected_components`, which
        defaults to 8 — each function keeps its own underlying OpenCV
        call's conventional default).
    fixed_range : bool, default False
        If ``False`` ("floating range"), each candidate pixel is compared
        to its already-filled neighbor. If ``True`` ("fixed range"), every
        candidate is compared to the seed pixel's original value instead —
        verified directly that these produce genuinely different fill
        regions on the same input.

    Returns
    -------
    FloodFillResult
        ``filled_count`` (number of pixels repainted), ``image`` (a new,
        filled copy), ``mask`` (a new ``uint8`` ``{0, 255}`` array shaped
        like `image`'s spatial dimensions, marking the filled region),
        ``bounding_box`` (the filled region's bounding box).

    Raises
    ------
    ValueError
        If `image` is not 2D/3D or is empty, does not have 1 or 3 channels,
        `seed_point` is not a 2-tuple (wrong type or length) or is not
        within `image`'s bounds, `new_value`/`lo_diff`/`up_diff` has the
        wrong element count for `image`'s channels or contains a
        non-finite value, `lo_diff`/`up_diff` contains a negative value, or
        (for a ``uint8`` `image`) `new_value` contains a value outside
        ``[0, 255]``.
    TypeError
        If `image` does not have dtype ``uint8`` or ``float32``,
        `seed_point`'s elements are not `numbers.Integral` (rejecting
        `bool`), or `new_value`/`lo_diff`/`up_diff` is not a real number or
        a sequence of real numbers.
    """
    require_image_ndim(image, ndims=(2, 3))
    require_dtype(image, _FLOOD_FILL_DTYPES)
    channels = 1 if image.ndim == 2 else image.shape[2]
    if channels not in (1, 3):
        raise ValueError(f"image must have 1 or 3 channels, got {channels}")
    connectivity_int = _require_integral_choice(connectivity, _CONNECTIVITIES, "connectivity")
    require_bool(fixed_range, "fixed_range")
    height, width = image.shape[:2]
    x, y = _require_seed_point(seed_point, width, height)

    new_value_resolved = _resolve_channel_values(new_value, channels, "new_value")
    if image.dtype == np.uint8:
        for i, element in enumerate(new_value_resolved):
            # cv2.floodFill silently rounds a fractional new_value instead
            # of rejecting it (0.5 -> 0, 254.5 -> 254) -- verified directly.
            if not float(element).is_integer():
                raise ValueError(
                    f"new_value[{i}] must be an integer value for a uint8 image, got {element}"
                )
            if not (0 <= element <= 255):
                raise ValueError(
                    f"new_value[{i}] must be in [0, 255] for a uint8 image, got {element}"
                )
    elif image.dtype == np.float32:
        for i, element in enumerate(new_value_resolved):
            # A finite Python float can still overflow float32 (e.g. 3.5e38
            # exceeds float32's max of ~3.4028235e38) -- cv2.floodFill
            # silently produces inf in the result instead of raising,
            # verified directly. np.errstate suppresses the (expected,
            # harmless) overflow-in-cast RuntimeWarning this check itself
            # triggers for an out-of-range element.
            with np.errstate(over="ignore"):
                representable = np.isfinite(np.float32(element))
            if not representable:
                raise ValueError(
                    f"new_value[{i}] must be representable as float32 "
                    f"without overflow, got {element}"
                )
    lo_diff_resolved = _resolve_channel_values(lo_diff, channels, "lo_diff", non_negative=True)
    up_diff_resolved = _resolve_channel_values(up_diff, channels, "up_diff", non_negative=True)

    image_copy = image.copy()
    internal_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
    # Packs the mask-fill value 255 into the flags word so the internal
    # mask comes back already {0, 255}-valued -- verified directly, no
    # separate normalization step needed.
    flags = connectivity_int | (255 << 8)
    if fixed_range:
        flags |= cv2.FLOODFILL_FIXED_RANGE

    new_value_cv2 = new_value_resolved[0] if channels == 1 else new_value_resolved
    lo_diff_cv2 = lo_diff_resolved[0] if channels == 1 else lo_diff_resolved
    up_diff_cv2 = up_diff_resolved[0] if channels == 1 else up_diff_resolved

    filled_count, _, mask_out, rect = cv2.floodFill(
        image_copy, internal_mask, (x, y), new_value_cv2, lo_diff_cv2, up_diff_cv2, flags
    )
    # mask_out[1:-1, 1:-1] is a view into mask_out (same reasoning as
    # find_contours' Hierarchy) -- .copy() guarantees a fresh array.
    mask_result = cast(Mask, mask_out[1:-1, 1:-1].copy())
    return FloodFillResult(filled_count, image_copy, mask_result, BoundingBox(*rect))
