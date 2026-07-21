"""Seeded image segmentation: watershed and rectangle-initialized GrabCut."""

from __future__ import annotations

from typing import cast

import cv2
import numpy as np
import numpy.typing as npt

from improcv._validation import require_channels, require_dtype
from improcv.types import ImageU8

__all__ = [
    "watershed",
]


def watershed(
    image: ImageU8,
    markers: npt.NDArray[np.int32],
) -> npt.NDArray[np.int32]:
    """Segment an image via marker-based watershed.

    Parameters
    ----------
    image : np.ndarray
        Input image with shape ``(H, W, 3)``, dtype ``uint8``.
    markers : np.ndarray
        Shape ``(H, W)`` matching `image`'s spatial size, dtype ``int32``.
        ``0`` marks the unknown region to be filled in by the algorithm;
        positive values mark seed regions (not required to be contiguous
        or to start at ``1`` -- verified directly that non-contiguous
        labels like ``2`` and ``10`` are accepted and preserved). Negative
        values are rejected on input -- ``-1`` is reserved for the
        algorithm's own boundary output.

    Returns
    -------
    np.ndarray
        Shape ``(H, W)``, dtype ``int32``. Positive values mark pixels
        assigned to a region (matching one of the input seed labels);
        ``-1`` marks watershed boundaries between regions. A fresh,
        independent array; `image`/`markers` are never modified. ``0`` is
        not a guaranteed output class -- it is simply whatever seed-growth
        left unassigned, and may not appear at all for some inputs.

    Raises
    ------
    ValueError
        If `image` does not have exactly 3 channels or is empty, `markers`
        does not have exactly 2 dimensions or is empty, `markers`'s shape
        does not match `image`'s spatial size, `markers` contains a
        negative value, or `markers` contains no positive seed.
    TypeError
        If `image` does not have dtype ``uint8``, or `markers` does not
        have dtype ``int32``.
    """
    require_channels(image, 3)
    require_dtype(image, (np.uint8,))
    if markers.ndim != 2 or markers.size == 0:
        raise ValueError(
            f"markers must have exactly 2 dimensions and be non-empty, got shape {markers.shape}"
        )
    require_dtype(markers, (np.int32,), "markers")
    if markers.shape != image.shape[:2]:
        raise ValueError(
            f"markers must have shape {image.shape[:2]} matching image's spatial size, "
            f"got {markers.shape}"
        )
    if np.any(markers < 0):
        raise ValueError("markers must not contain negative values")
    if not np.any(markers > 0):
        raise ValueError("markers must contain at least one positive seed")

    markers_copy = markers.copy()
    result = cv2.watershed(image, markers_copy)
    # cv2's stubs type watershed's result as the loose MatLike; it always
    # returns the same int32 array it was given (mutated in place).
    return cast(npt.NDArray[np.int32], result)
