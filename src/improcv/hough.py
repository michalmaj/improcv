"""Hough transform-based shape detection: lines, line segments, circles."""

from __future__ import annotations

from typing import Literal, NamedTuple

import cv2
import numpy as np

from improcv._compat.opencv import _normalize_hough_lines_p_output
from improcv._validation import (
    require_dtype,
    require_finite,
    require_image_ndim,
    require_positive_integral,
)
from improcv.types import Mask

__all__ = [
    "hough_line_segments",
    "hough_lines",
    "Circle",
    "HoughCircleMethod",
    "Line",
    "LineSegment",
]

_MAX_INT32 = int(np.iinfo(np.int32).max)

HoughCircleMethod = Literal["gradient", "gradient_alt"]


def _require_strictly_positive(value: float, name: str) -> None:
    """Raise TypeError/ValueError unless `value` is a finite, strictly positive real number."""
    require_finite(value, name)
    if not value > 0.0:
        raise ValueError(f"{name} must be positive, got {value}")


def _require_non_negative_real(value: float, name: str) -> None:
    """Raise TypeError/ValueError unless `value` is a finite, non-negative real number."""
    require_finite(value, name)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value}")


class Line(NamedTuple):
    """A line found by `hough_lines`, in polar form.

    ``rho`` is the distance from the coordinate origin (top-left corner of
    the image), in pixels; ``theta`` is the line's rotation angle in
    radians (``0`` ~ vertical line, ``pi/2`` ~ horizontal line).
    """

    rho: float
    theta: float


class LineSegment(NamedTuple):
    """A line segment found by `hough_line_segments`, as two endpoints."""

    x1: int
    y1: int
    x2: int
    y2: int


class Circle(NamedTuple):
    """A circle found by `hough_circles`, as a center and radius."""

    x: float
    y: float
    radius: float


def hough_lines(
    image: Mask,
    threshold: int,
    rho: float = 1.0,
    theta: float = np.pi / 180,
) -> list[Line]:
    """Detect lines with the standard Hough transform.

    Parameters
    ----------
    image : np.ndarray
        A 2D uint8 edge image. Zero is background and every nonzero pixel
        votes as an edge pixel. Typically produced by Canny or another
        binary edge detector.
    threshold : int
        Minimum accumulator votes for a line to be reported. A positive
        integer, at most ``2**31 - 1`` (the range of a signed C ``int``).
        Has no default -- how many votes are "enough" depends entirely on
        image size and content, so no single value is universally
        reasonable; verified directly that a non-positive threshold
        doesn't raise but silently returns a huge, low-quality flood of
        lines instead.
    rho : float, default 1.0
        Distance resolution of the accumulator, in pixels. Must be
        positive. `improcv`'s own default, not a native OpenCV one --
        `rho` is a required parameter in OpenCV's own signature; `1.0`
        pixel is a universal, content-independent accumulator-resolution
        choice.
    theta : float, default pi/180
        Angle resolution of the accumulator, in radians. Must be
        positive. `improcv`'s own default (1 degree), for the same reason
        as `rho`.

    Returns
    -------
    list of Line
        Empty if no lines are found. Verified directly, identically on
        both OpenCV versions: violating `rho > 0`/`theta > 0` is not
        handled safely by OpenCV itself (some invalid values attempt an
        ~18-exabyte allocation on one version, a raw assertion crash on
        the other) -- this function never lets a non-positive `rho`/`theta`
        reach OpenCV.

    Raises
    ------
    ValueError
        If `image` does not have exactly 2 dimensions or is empty, `rho`
        or `theta` is not finite or not strictly positive, or `threshold`
        is not positive or exceeds ``2**31 - 1``.
    TypeError
        If `image` does not have dtype ``uint8``, or `threshold` is not
        `numbers.Integral` (rejecting `bool`/`float`).
    RuntimeError
        If OpenCV's raw result is not a ``float32`` array with exactly 2
        fields per line -- an internally inconsistent result rather than
        a valid line list.
    """
    require_image_ndim(image, ndims=(2,))
    require_dtype(image, (np.uint8,))
    _require_strictly_positive(rho, "rho")
    _require_strictly_positive(theta, "theta")
    require_positive_integral(threshold, "threshold")
    threshold_int = int(threshold)
    if threshold_int > _MAX_INT32:
        raise ValueError(
            f"threshold must fit within the range of int32 ([1, {_MAX_INT32}]), got {threshold}"
        )

    # OpenCV's own docs state the input image may be modified by this
    # function -- verified directly that neither installed build actually
    # does, but that's an accident of implementation, not a promise, so a
    # copy is made to guarantee improcv's own no-mutation contract.
    raw = cv2.HoughLines(image.copy(), rho, theta, threshold_int)
    if raw is None:
        return []

    if raw.dtype != np.float32 or raw.ndim != 3 or raw.shape[1:] != (1, 2):
        raise RuntimeError(
            f"cv2.HoughLines returned an array of shape {raw.shape} dtype {raw.dtype} -- "
            "unexpected OpenCV output"
        )

    return [Line(rho=float(r), theta=float(t)) for r, t in raw[:, 0, :]]


def hough_line_segments(
    image: Mask,
    threshold: int,
    rho: float = 1.0,
    theta: float = np.pi / 180,
    min_line_length: float = 0.0,
    max_line_gap: float = 0.0,
) -> list[LineSegment]:
    """Detect line segments with the probabilistic Hough transform.

    Parameters
    ----------
    image : np.ndarray
        A 2D uint8 edge image. Zero is background and every nonzero pixel
        votes as an edge pixel. Typically produced by Canny or another
        binary edge detector.
    threshold : int
        Minimum accumulator votes for a line to be reported. A positive
        integer, at most ``2**31 - 1``. Has no default, for the same
        reason as `hough_lines`'s `threshold`.
    rho : float, default 1.0
        Distance resolution of the accumulator, in pixels. Must be
        positive. `improcv`'s own default, not a native OpenCV one -- see
        `hough_lines`.
    theta : float, default pi/180
        Angle resolution of the accumulator, in radians. Must be
        positive. `improcv`'s own default -- see `hough_lines`.
    min_line_length : float, default 0.0
        Minimum line length; shorter segments are rejected. Must be
        non-negative. OpenCV's own default.
    max_line_gap : float, default 0.0
        Maximum allowed gap between points on the same line to link them
        into one segment. Must be non-negative. OpenCV's own default.

    Returns
    -------
    list of LineSegment
        Empty if no segments are found.

    Raises
    ------
    ValueError
        If `image` does not have exactly 2 dimensions or is empty, `rho`
        or `theta` is not finite or not strictly positive, `threshold` is
        not positive or exceeds ``2**31 - 1``, or `min_line_length`/
        `max_line_gap` is not finite or is negative.
    TypeError
        If `image` does not have dtype ``uint8``, or `threshold` is not
        `numbers.Integral` (rejecting `bool`/`float`).
    RuntimeError
        If OpenCV's raw result is not an ``int32`` array with exactly 4
        fields per segment -- an internally inconsistent result rather
        than a valid segment list.
    """
    require_image_ndim(image, ndims=(2,))
    require_dtype(image, (np.uint8,))
    _require_strictly_positive(rho, "rho")
    _require_strictly_positive(theta, "theta")
    require_positive_integral(threshold, "threshold")
    threshold_int = int(threshold)
    if threshold_int > _MAX_INT32:
        raise ValueError(
            f"threshold must fit within the range of int32 ([1, {_MAX_INT32}]), got {threshold}"
        )
    _require_non_negative_real(min_line_length, "min_line_length")
    _require_non_negative_real(max_line_gap, "max_line_gap")

    # Same no-mutation rationale as hough_lines: OpenCV's own docs permit
    # HoughLinesP to modify its input too.
    raw = cv2.HoughLinesP(
        image.copy(),
        rho,
        theta,
        threshold_int,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if raw is None:
        return []

    normalized = _normalize_hough_lines_p_output(raw)
    return [
        LineSegment(x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2)) for x1, y1, x2, y2 in normalized
    ]
