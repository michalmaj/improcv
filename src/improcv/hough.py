"""Hough transform-based shape detection: lines, line segments, circles."""

from __future__ import annotations

import math
from typing import Literal, NamedTuple

import cv2
import numpy as np

from improcv._compat.opencv import _normalize_hough_lines_p_output
from improcv._validation import (
    require_dtype,
    require_finite,
    require_image_ndim,
    require_integral,
    require_one_of,
    require_positive_integral,
)
from improcv.types import Image, Mask

__all__ = [
    "hough_circles",
    "hough_line_segments",
    "hough_lines",
    "Circle",
    "HoughCircleMethod",
    "Line",
    "LineSegment",
]

_MAX_INT32 = int(np.iinfo(np.int32).max)
_MIN_INT32 = int(np.iinfo(np.int32).min)

HoughCircleMethod = Literal["gradient", "gradient_alt"]

_HOUGH_CIRCLE_METHODS: dict[HoughCircleMethod, int] = {
    "gradient": cv2.HOUGH_GRADIENT,
    "gradient_alt": cv2.HOUGH_GRADIENT_ALT,
}
_DEFAULT_PARAM2: dict[HoughCircleMethod, float] = {
    "gradient": 100.0,
    "gradient_alt": 0.9,
}


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


def hough_circles(
    image: Image,
    min_dist: float,
    method: HoughCircleMethod = "gradient",
    dp: float = 1.0,
    param1: float = 100.0,
    param2: float | None = None,
    min_radius: int = 0,
    max_radius: int = 0,
) -> list[Circle]:
    """Detect circles with the Hough transform.

    Parameters
    ----------
    image : np.ndarray
        A 2D uint8 grayscale image (not a binary edge mask -- unlike
        `hough_lines`/`hough_line_segments`, OpenCV's own docs call for a
        plain grayscale input here, typically blurred first).
    min_dist : float
        Minimum distance between the centers of detected circles, in
        pixels. Must be positive. Has no default -- genuinely
        image/circle-scale-dependent, no single value is universally
        reasonable.
    method : {"gradient", "gradient_alt"}, default "gradient"
        Detection method. `"gradient"` is OpenCV's classic
        ``HOUGH_GRADIENT``; `"gradient_alt"` is the newer
        ``HOUGH_GRADIENT_ALT``, generally better at fitting circle shape
        but with different `param2` semantics (see below) and no
        "centers only" support (see `max_radius`).
    dp : float, default 1.0
        Inverse ratio of the accumulator resolution to the image
        resolution. Must be positive. `improcv`'s own default (matching
        OpenCV's own C++ default) -- `1.5` is only a documented
        *recommendation* for `"gradient_alt"`, not required.
    param1 : float, default 100.0
        For both methods, the higher of the two Canny thresholds used
        internally. Must be positive. `improcv`'s own default (matching
        OpenCV's own C++ default) for both methods -- `300` is only a
        documented *recommendation* for `"gradient_alt"`, not required.
    param2 : float or None, default None
        Method-specific accumulator threshold. For `"gradient"`, must be
        positive (smaller values report more, potentially false, circles).
        For `"gradient_alt"`, must be strictly between ``0.0`` and ``1.0``
        (a circle "perfectness" measure, closer to ``1.0`` is stricter).
        When `None`, resolves to `100.0` for `"gradient"` or `0.9` for
        `"gradient_alt"` -- verified directly that OpenCV's own omitted-
        parameter default (`100.0` regardless of method) violates
        `"gradient_alt"`'s own required range and raises a raw
        `cv2.error`, so this function never lets that combination reach
        OpenCV.
    min_radius : int, default 0
        Minimum circle radius, in pixels. Must be a non-negative integer
        within the signed `int32` range. `0` means no lower bound
        (OpenCV's own default).
    max_radius : int, default 0
        Maximum circle radius, in pixels, within the signed `int32`
        range. Semantics:

        - ``0`` (default, OpenCV's own default): automatic upper bound
          (OpenCV uses the maximum image dimension).
        - Greater than `min_radius`: an explicit range, honored as given.
        - Negative, with `method="gradient"` only: "centers only" mode --
          every returned `radius` is `0.0`.
        - Negative, with `method="gradient_alt"`: rejected with
          `ValueError` -- OpenCV documents "centers only" for
          `HOUGH_GRADIENT` only; verified directly that `"gradient_alt"`
          does not raise for a negative `max_radius` but also does not
          enter centers-only mode, silently computing a normal nonzero
          radius instead, which would otherwise silently violate this
          function's own "centers only" promise.
        - Positive but not greater than `min_radius`: rejected with
          `ValueError` for both methods -- verified directly that neither
          method honors this range as given: `"gradient"` silently widens
          it, `"gradient_alt"` silently reorders it, rather than
          respecting or rejecting the ambiguous request.

    Returns
    -------
    list of Circle
        Empty if no circles are found.

    Raises
    ------
    ValueError
        If `image` does not have exactly 2 dimensions or is empty,
        `method` is not recognized, `dp`/`min_dist`/`param1` is not finite
        or not strictly positive, `param2` (once resolved) is out of its
        method-specific range, `min_radius`/`max_radius` is outside the
        signed `int32` range, `min_radius` is negative, or `max_radius`'s
        value is inconsistent with `method`/`min_radius` per the semantics
        above.
    TypeError
        If `image` does not have dtype ``uint8``, `dp`/`min_dist`/`param1`/
        `param2` is not a real number, or `min_radius`/`max_radius` is not
        `numbers.Integral` (rejecting `bool`/`float`).
    RuntimeError
        If OpenCV's raw result is not a ``float32`` array with 3 or 4
        fields per circle, or any returned `x`/`y`/`radius` is not finite
        -- an internally inconsistent result rather than a valid circle
        list.
    """
    require_image_ndim(image, ndims=(2,))
    require_dtype(image, (np.uint8,))
    require_one_of(method, ("gradient", "gradient_alt"), "method")
    _require_strictly_positive(dp, "dp")
    _require_strictly_positive(min_dist, "min_dist")
    _require_strictly_positive(param1, "param1")

    if param2 is None:
        param2_float = _DEFAULT_PARAM2[method]
    else:
        require_finite(param2, "param2")
        if method == "gradient_alt":
            if not (0.0 < param2 < 1.0):
                raise ValueError(
                    f"param2 must be strictly between 0.0 and 1.0 for method='gradient_alt', "
                    f"got {param2}"
                )
        else:
            if not param2 > 0.0:
                raise ValueError(f"param2 must be positive for method='gradient', got {param2}")
        param2_float = float(param2)

    require_integral(min_radius, "min_radius")
    min_radius_int = int(min_radius)
    if min_radius_int < 0 or min_radius_int > _MAX_INT32:
        raise ValueError(
            f"min_radius must fit within the range of a non-negative int32 "
            f"([0, {_MAX_INT32}]), got {min_radius}"
        )

    require_integral(max_radius, "max_radius")
    max_radius_int = int(max_radius)
    if max_radius_int < _MIN_INT32 or max_radius_int > _MAX_INT32:
        raise ValueError(
            f"max_radius must fit within the range of int32 "
            f"([{_MIN_INT32}, {_MAX_INT32}]), got {max_radius}"
        )
    if max_radius_int < 0 and method == "gradient_alt":
        raise ValueError(
            "max_radius must be non-negative for method='gradient_alt' -- 'centers only' mode "
            "(a negative max_radius) is only documented and supported for method='gradient'"
        )
    if max_radius_int > 0 and max_radius_int <= min_radius_int:
        raise ValueError(
            f"max_radius ({max_radius_int}) must be greater than min_radius ({min_radius_int}) "
            "-- verified directly that OpenCV silently widens or reorders this range instead of "
            "honoring or rejecting it, so this combination is rejected explicitly"
        )

    raw = cv2.HoughCircles(
        image,
        _HOUGH_CIRCLE_METHODS[method],
        dp,
        min_dist,
        param1=param1,
        param2=param2_float,
        minRadius=min_radius_int,
        maxRadius=max_radius_int,
    )
    if raw is None:
        return []

    if raw.dtype != np.float32 or raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[2] not in (3, 4):
        raise RuntimeError(
            f"cv2.HoughCircles returned an array of shape {raw.shape} dtype {raw.dtype} -- "
            "unexpected OpenCV output"
        )

    circles: list[Circle] = []
    for row in raw[0]:
        x, y, radius = float(row[0]), float(row[1]), float(row[2])
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(radius)):
            raise RuntimeError(
                f"cv2.HoughCircles returned a non-finite circle: x={x}, y={y}, radius={radius}"
            )
        circles.append(Circle(x=x, y=y, radius=radius))
    return circles
