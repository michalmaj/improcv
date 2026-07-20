"""Contour detection and shape descriptors."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, NamedTuple, cast

import cv2
import numpy as np
import numpy.typing as npt

from improcv._validation import require_dtype, require_image_ndim, require_one_of
from improcv.types import Mask

__all__ = ["find_contours", "bounding_boxes", "sort_contours", "convex_hull"]

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


def _require_contour(contour: object, min_points: int, name: str = "contour") -> None:
    """Raise TypeError/ValueError unless `contour` is a valid `Contour` with `min_points` points.

    Checks, in order: `contour` is an `np.ndarray`, has dtype `int32`, has
    shape `(N, 1, 2)`, and `N >= min_points`. Does not check C-contiguity:
    `cv2.boundingRect`/`convexHull`/`approxPolyDP`/`minAreaRect` all accept a
    non-contiguous `(N, 1, 2)` int32 array directly (verified), so no
    contiguity check or implicit copy is made here.
    """
    if not isinstance(contour, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(contour).__name__}")
    if contour.dtype != np.int32:
        raise TypeError(f"{name} must have dtype int32, got {contour.dtype}")
    if contour.ndim != 3 or contour.shape[1:] != (1, 2):
        raise ValueError(f"{name} must have shape (N, 1, 2), got {contour.shape}")
    if contour.shape[0] < min_points:
        raise ValueError(f"{name} must have at least {min_points} point(s), got {contour.shape[0]}")


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


def bounding_boxes(contours: Sequence[Contour]) -> list[BoundingBox]:
    """Compute the axis-aligned bounding box of each contour.

    Parameters
    ----------
    contours : sequence of np.ndarray
        Each element a `Contour`. May be empty — an empty `contours` is not
        an error and produces an empty result, matching how "no shapes
        found" is a normal outcome in contour detection. Accepts the raw
        tuple `cv2.findContours` itself returns directly; no `list(...)`
        conversion is required from the caller.

    Returns
    -------
    list of BoundingBox
        One box per input contour, in input order.

    Raises
    ------
    ValueError
        If any element of `contours` does not have shape ``(N, 1, 2)``.
    TypeError
        If any element of `contours` is not an ``int32`` `np.ndarray`.
    """
    boxes = []
    for i, contour in enumerate(contours):
        _require_contour(contour, min_points=0, name=f"contours[{i}]")
        boxes.append(BoundingBox(*cv2.boundingRect(contour)))
    return boxes


SortOrder = Literal["left-to-right", "right-to-left", "top-to-bottom", "bottom-to-top"]
_SORT_ORDERS: tuple[SortOrder, ...] = (
    "left-to-right",
    "right-to-left",
    "top-to-bottom",
    "bottom-to-top",
)


def sort_contours(
    contours: Sequence[Contour],
    order: SortOrder = "left-to-right",
) -> tuple[list[Contour], list[BoundingBox]]:
    """Sort contours by the position of their bounding box.

    Parameters
    ----------
    contours : sequence of np.ndarray
        Each element a `Contour`. May be empty — ``sort_contours([])``
        returns ``([], [])`` rather than raising, matching `bounding_boxes`.
    order : {"left-to-right", "right-to-left", "top-to-bottom", "bottom-to-top"}
        Sort key and direction: horizontal orders sort by each box's `x`,
        vertical orders by `y`. Default ``"left-to-right"``.

    Returns
    -------
    sorted_contours : list of np.ndarray
    sorted_boxes : list of BoundingBox
        Parallel lists: ``sorted_boxes[i]`` is `sorted_contours[i]`'s
        bounding box, avoiding a second bounding-box pass for the caller.

    Notes
    -----
    Uses Python's stable sort: contours whose sort key (`x` or `y`) is
    exactly equal keep their original relative order — no secondary
    tie-break key is applied.

    Raises
    ------
    ValueError
        If `order` is not one of the accepted values, or any element of
        `contours` does not have shape ``(N, 1, 2)``.
    TypeError
        If any element of `contours` is not an ``int32`` `np.ndarray`.
    """
    require_one_of(order, _SORT_ORDERS, "order")
    boxes = bounding_boxes(contours)
    paired = list(zip(contours, boxes, strict=True))
    if order in ("left-to-right", "right-to-left"):
        paired.sort(key=lambda pair: pair[1].x, reverse=order == "right-to-left")
    else:
        paired.sort(key=lambda pair: pair[1].y, reverse=order == "bottom-to-top")
    sorted_contours = [pair[0] for pair in paired]
    sorted_boxes = [pair[1] for pair in paired]
    return sorted_contours, sorted_boxes


def convex_hull(contour: Contour) -> Contour:
    """Compute the convex hull of a contour.

    Parameters
    ----------
    contour : np.ndarray
        A `Contour`. A 1- or 2-point contour is well-defined and accepted
        (returned as-is, or as a 2-point hull, respectively) — only a
        genuinely empty (0-point) contour is rejected, since `cv2.convexHull`
        returns `None` for that case (verified directly), which would
        otherwise silently violate this function's ``-> Contour`` return
        contract.

    Returns
    -------
    np.ndarray
        The hull's vertices, shape ``(M, 1, 2)``, dtype ``int32``, ``M <= N``.
        A new array; `contour` is never modified.

    Raises
    ------
    ValueError
        If `contour` has 0 points or does not have shape ``(N, 1, 2)``.
    TypeError
        If `contour` is not an ``int32`` `np.ndarray`.
    """
    _require_contour(contour, min_points=1)
    return cast(Contour, cv2.convexHull(contour))
