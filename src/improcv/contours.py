"""Contour detection and shape descriptors."""

from __future__ import annotations

from typing import Literal, NamedTuple, cast

import cv2
import numpy as np
import numpy.typing as npt

from improcv._validation import require_dtype, require_image_ndim, require_one_of
from improcv.types import Mask

__all__ = ["find_contours"]

Contour = npt.NDArray[np.int32]
"""A single contour: shape ``(N, 1, 2)``, dtype ``int32`` — the exact shape and
dtype ``cv2.findContours``/``convexHull``/``approxPolyDP`` produce and consume,
so a `Contour` can be passed directly to any ``cv2.*`` function expecting one."""

Hierarchy = npt.NDArray[np.int32]
"""Contour hierarchy: shape ``(N, 4)``, dtype ``int32``. Each row is
``[next, previous, first_child, parent]`` (``-1`` for none) — squeezed from
OpenCV's raw ``(1, N, 4)``; that leading dimension is never meaningful and is
never fed back into any ``cv2.*`` call."""


class BoundingBox(NamedTuple):
    """An axis-aligned bounding box, matching ``cv2.boundingRect``'s ``(x, y, w, h)``.

    Unpacks like a plain tuple (``x, y, w, h = box``), so it costs nothing in
    interop versus a bare tuple, while still giving named field access.
    """

    x: int
    y: int
    width: int
    height: int


class RotatedRect(NamedTuple):
    """A rotated rectangle, matching ``cv2.minAreaRect``'s ``((cx, cy), (w, h), angle)``.

    Verified directly to work as a drop-in argument to ``cv2.boxPoints`` in
    place of the raw tuple ``cv2.minAreaRect`` itself returns.
    """

    center: tuple[float, float]
    size: tuple[float, float]
    angle: float


RetrievalMode = Literal["external", "list", "ccomp", "tree"]
_RETRIEVAL_MODES: tuple[RetrievalMode, ...] = ("external", "list", "ccomp", "tree")
_RETRIEVAL_MODE_FLAGS: dict[RetrievalMode, int] = {
    "external": cv2.RETR_EXTERNAL,
    "list": cv2.RETR_LIST,
    "ccomp": cv2.RETR_CCOMP,
    "tree": cv2.RETR_TREE,
}

ApproxMethod = Literal["none", "simple", "tc89_l1", "tc89_kcos"]
_APPROX_METHODS: tuple[ApproxMethod, ...] = ("none", "simple", "tc89_l1", "tc89_kcos")
_APPROX_METHOD_FLAGS: dict[ApproxMethod, int] = {
    "none": cv2.CHAIN_APPROX_NONE,
    "simple": cv2.CHAIN_APPROX_SIMPLE,
    "tc89_l1": cv2.CHAIN_APPROX_TC89_L1,
    "tc89_kcos": cv2.CHAIN_APPROX_TC89_KCOS,
}


def find_contours(
    mask: Mask,
    retrieval_mode: RetrievalMode = "external",
    approximation: ApproxMethod = "simple",
) -> tuple[list[Contour], Hierarchy]:
    """Find contours in a binary mask.

    Parameters
    ----------
    mask : np.ndarray
        Input mask with shape ``(H, W)``, dtype ``uint8``. Any nonzero value
        is treated as foreground, not only ``255`` (verified directly).
    retrieval_mode : {"external", "list", "ccomp", "tree"}, default "external"
        Contour retrieval strategy: ``"external"`` finds only outermost
        contours; ``"list"`` finds all contours with a flat (non-nested)
        hierarchy; ``"ccomp"`` organizes contours into a two-level hierarchy
        (outer boundaries and holes); ``"tree"`` reconstructs the full nested
        hierarchy.
    approximation : {"none", "simple", "tc89_l1", "tc89_kcos"}, default "simple"
        Contour approximation method passed to ``cv2.findContours``.

    Returns
    -------
    contours : list of np.ndarray
        Each contour has shape ``(N, 1, 2)``, dtype ``int32`` — see `Contour`.
    hierarchy : np.ndarray
        Shape ``(N, 4)``, dtype ``int32``, one row per contour: columns
        ``[next, previous, first_child, parent]`` (``-1`` for none). Always a
        fresh, independent array — a shape ``(0, 4)`` array for zero
        contours, never `None`.

    Raises
    ------
    ValueError
        If `mask` does not have exactly 2 dimensions or is empty, or
        `retrieval_mode`/`approximation` is not one of the accepted values.
    TypeError
        If `mask` does not have dtype ``uint8``.

    Notes
    -----
    `mask` is never modified: verified directly that `cv2.findContours` does
    not mutate its input on OpenCV 4.x/5.x (unlike OpenCV 3.x, which mutated
    the input mask in place).
    """
    require_image_ndim(mask, ndims=(2,))
    require_dtype(mask, (np.uint8,))
    require_one_of(retrieval_mode, _RETRIEVAL_MODES, "retrieval_mode")
    require_one_of(approximation, _APPROX_METHODS, "approximation")
    contours, raw_hierarchy = cv2.findContours(
        mask, _RETRIEVAL_MODE_FLAGS[retrieval_mode], _APPROX_METHOD_FLAGS[approximation]
    )
    hierarchy: Hierarchy
    if raw_hierarchy is None:
        hierarchy = np.empty((0, 4), dtype=np.int32)
    else:
        # raw_hierarchy[0] is a view into raw_hierarchy (verified directly:
        # np.shares_memory is True) — .copy() guarantees a fresh, independent
        # array, per this module's copy/view contract.
        hierarchy = raw_hierarchy[0].copy()
    # cv2's stubs type contours as the loose `list[MatLike]`; cv2.findContours
    # always produces int32 (N, 1, 2) arrays in practice.
    return cast(list[Contour], list(contours)), hierarchy
