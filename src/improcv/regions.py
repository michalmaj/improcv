"""Region analysis: connected components, distance transform, flood fill."""

from __future__ import annotations

from typing import Literal, cast

import cv2
import numpy as np
import numpy.typing as npt

from improcv._validation import require_dtype, require_image_ndim, require_one_of
from improcv.types import ImageFloat32, Mask

__all__ = ["connected_components", "connected_components_with_stats", "distance_transform"]

Connectivity = Literal[4, 8]
_CONNECTIVITIES: tuple[Connectivity, ...] = (4, 8)

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
    require_one_of(connectivity, _CONNECTIVITIES, "connectivity")
    num_labels, labels = cv2.connectedComponents(mask, connectivity=connectivity)
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
    require_one_of(connectivity, _CONNECTIVITIES, "connectivity")
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=connectivity
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
        If `mask` does not have exactly 2 dimensions or is empty,
        `distance_type` is not one of the accepted values, or `mask_size`
        (explicit or defaulted) is not valid for the chosen `distance_type`.
    TypeError
        If `mask` does not have dtype ``uint8``.
    """
    require_image_ndim(mask, ndims=(2,))
    require_dtype(mask, (np.uint8,))
    require_one_of(distance_type, tuple(_DISTANCE_TYPE_FLAGS), "distance_type")
    resolved_mask_size = _DEFAULT_MASK_SIZE[distance_type] if mask_size is None else mask_size
    valid_sizes = _VALID_MASK_SIZES[distance_type]
    if resolved_mask_size not in valid_sizes:
        raise ValueError(
            f"mask_size {resolved_mask_size} is not valid for distance_type "
            f"{distance_type!r}; accepted values are {valid_sizes}"
        )
    result = cv2.distanceTransform(mask, _DISTANCE_TYPE_FLAGS[distance_type], resolved_mask_size)
    return cast(ImageFloat32, result)
