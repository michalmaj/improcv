"""Region analysis: connected components, distance transform, flood fill."""

from __future__ import annotations

from typing import Literal, cast

import cv2
import numpy as np
import numpy.typing as npt

from improcv._validation import require_dtype, require_image_ndim, require_one_of
from improcv.types import Mask

__all__ = ["connected_components", "connected_components_with_stats"]

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
