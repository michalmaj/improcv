"""Region analysis: connected components, distance transform, flood fill."""

from __future__ import annotations

from typing import Literal, cast

import cv2
import numpy as np
import numpy.typing as npt

from improcv._validation import require_dtype, require_image_ndim, require_one_of
from improcv.types import Mask

__all__ = ["connected_components"]

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
