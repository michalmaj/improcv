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

# Verified directly that without a cap, an individually "valid" (finite,
# positive, within the geometric rho/theta range) combination can still
# imply an accumulator large enough to be a practical denial-of-service
# footgun. 50 million cells is comfortably above any legitimate use of
# these functions while staying far short of causing memory pressure.
_MAX_ACCUMULATOR_CELLS = 50_000_000

HoughCircleMethod = Literal["gradient", "gradient_alt"]

_HOUGH_CIRCLE_METHODS: dict[HoughCircleMethod, int] = {
    "gradient": cv2.HOUGH_GRADIENT,
    "gradient_alt": cv2.HOUGH_GRADIENT_ALT,
}
_DEFAULT_PARAM2: dict[HoughCircleMethod, float] = {
    "gradient": 100.0,
    "gradient_alt": 0.9,
}


def _require_strictly_positive(value: float, name: str) -> float:
    """Raise TypeError/ValueError unless `value` is a finite, strictly positive real number.

    Returns `value` normalized to a plain `float`. `require_finite`
    accepts any `numbers.Real` (e.g. `fractions.Fraction`), but OpenCV's
    own binding can't always accept those directly -- verified directly
    that `rho=Fraction(1, 1)` passes this validation yet raises a raw,
    confusing `cv2.error` ("Argument 'rho' can not be treated as a
    double") if passed through unnormalized. Every real-valued OpenCV
    parameter in this module is normalized to `float` before the call.
    """
    require_finite(value, name)
    value_float = float(value)
    if not value_float > 0.0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value_float


def _require_non_negative_real(value: float, name: str) -> float:
    """Raise TypeError/ValueError unless `value` is a finite, non-negative real number.

    Returns `value` normalized to a plain `float` -- see
    `_require_strictly_positive`.
    """
    require_finite(value, name)
    value_float = float(value)
    if value_float < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value_float


def _require_safe_hough_line_params(
    image: np.ndarray, rho: float, theta: float
) -> tuple[float, float]:
    """Validate and normalize rho/theta against the image's Hough accumulator size.

    Returns `(rho, theta)` normalized to plain `float`.

    Verified directly that extreme, individually finite and positive
    rho/theta values can crash the process outright with a segmentation
    fault (not a raised exception) on both OpenCV versions:
    ``hough_line_segments(image, threshold=10, rho=1e6)`` and the same
    call with both `rho` and `theta` extreme both segfault reliably.
    OpenCV computes the accumulator's dimensions from `rho`/`theta` and
    the image size, then allocates it, without validating that the
    computed dimensions are sane.

    Bounds `theta` to `(0, pi]` (a Hough line angle is periodic over a
    half-turn; theta beyond that is geometrically meaningless) and `rho`
    to `(0, 2 * (height + width) + 1]` (the same expression OpenCV's own
    accumulator sizing uses for the image's diagonal-ish span) and
    additionally caps the total predicted accumulator cell count
    (`_MAX_ACCUMULATOR_CELLS`), so a combination that is individually
    "valid" but would still produce a degenerate or absurdly large
    accumulator is rejected before OpenCV ever sees it. The predicted
    cell count closely mirrors, but isn't required to bit-match,
    OpenCV's own internal `cvRound`-based accumulator sizing -- this is a
    defensive upper bound, not a re-implementation of OpenCV's math.
    """
    require_finite(rho, "rho")
    rho_float = float(rho)
    require_finite(theta, "theta")
    theta_float = float(theta)

    if not (0.0 < theta_float <= math.pi):
        raise ValueError(f"theta must be in (0, pi], got {theta}")

    height, width = image.shape
    max_rho = 2.0 * (height + width) + 1.0
    if not (0.0 < rho_float <= max_rho):
        raise ValueError(f"rho must be in (0, {max_rho}] for a {height}x{width} image, got {rho}")

    # Computed before rounding: a very small (but still individually
    # "valid") rho/theta can make this division overflow to infinity,
    # and round(inf) raises a raw OverflowError rather than a value this
    # function can compare -- verified directly with rho=theta=1e-320.
    num_angle_estimate = math.pi / theta_float
    num_rho_estimate = max_rho / rho_float
    if not (math.isfinite(num_angle_estimate) and math.isfinite(num_rho_estimate)):
        raise ValueError(
            f"rho={rho}, theta={theta} produce non-finite Hough accumulator dimensions"
        )
    if num_angle_estimate > _MAX_INT32 or num_rho_estimate > _MAX_INT32:
        raise ValueError(
            f"rho={rho}, theta={theta} produce an accumulator dimension exceeding int32 "
            f"(numangle~{num_angle_estimate}, numrho~{num_rho_estimate})"
        )

    # math.ceil rather than round: this is a defensive upper bound on the
    # cell count, not a re-implementation of OpenCV's own cvRound-based
    # sizing, so rounding up (never underestimating) is the safe choice.
    num_angle = math.ceil(num_angle_estimate)
    num_rho = math.ceil(num_rho_estimate)
    if num_angle <= 0 or num_rho <= 0:
        raise ValueError(
            f"rho={rho}, theta={theta} produce a degenerate accumulator "
            f"(numangle={num_angle}, numrho={num_rho}) for a {height}x{width} image"
        )
    num_cells = num_angle * num_rho
    if num_cells > _MAX_ACCUMULATOR_CELLS:
        raise ValueError(
            f"rho={rho}, theta={theta} would require an accumulator of {num_cells} cells "
            f"for a {height}x{width} image, exceeding the safety limit of "
            f"{_MAX_ACCUMULATOR_CELLS}"
        )

    return rho_float, theta_float


def _require_dp_at_least_one(value: float, name: str = "dp") -> float:
    """Raise TypeError/ValueError unless `value` is a finite real number >= 1.0.

    Returns `value` normalized to a plain `float`. Verified directly,
    identically for both `HOUGH_GRADIENT` and `HOUGH_GRADIENT_ALT`, that
    OpenCV silently clamps `dp` to `max(dp, 1.0)` internally -- `dp`
    values from `0.1` through `1.0` all produce identical results, only
    `dp > 1.0` actually changes anything. This function rejects
    `dp < 1.0` outright instead of silently accepting a value OpenCV
    itself ignores.
    """
    require_finite(value, name)
    value_float = float(value)
    if not value_float >= 1.0:
        raise ValueError(f"{name} must be at least 1.0, got {value}")
    return value_float


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
        Empty if no lines are found. `rho`/`theta` are validated not only
        for positivity but also against the image's predicted Hough
        accumulator size before ever calling OpenCV -- verified directly
        that an individually finite, positive but extreme value (e.g.
        ``rho=1e6``) can segfault the process outright rather than raise
        a catchable error.

    Raises
    ------
    ValueError
        If `image` does not have exactly 2 dimensions or is empty,
        `theta` is not in ``(0, pi]``, `rho` is not in
        ``(0, 2 * (height + width) + 1]`` for `image`'s size, `rho`/
        `theta` imply a degenerate or excessively large Hough
        accumulator, or `threshold` is not positive or exceeds
        ``2**31 - 1``.
    TypeError
        If `image` does not have dtype ``uint8``, `rho`/`theta` is not a
        real number, or `threshold` is not `numbers.Integral` (rejecting
        `bool`/`float`).
    RuntimeError
        If OpenCV's raw result is not a ``float32`` `np.ndarray` with
        exactly 2 finite fields per line -- an internally inconsistent
        result rather than a valid line list.
    """
    require_image_ndim(image, ndims=(2,))
    require_dtype(image, (np.uint8,))
    rho_float, theta_float = _require_safe_hough_line_params(image, rho, theta)
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
    raw = cv2.HoughLines(image.copy(), rho_float, theta_float, threshold_int)
    if raw is None:
        return []

    if (
        not isinstance(raw, np.ndarray)
        or raw.dtype != np.float32
        or raw.ndim != 3
        or raw.shape[1:] != (1, 2)
    ):
        raise RuntimeError(
            f"cv2.HoughLines returned an array of shape "
            f"{getattr(raw, 'shape', None)} dtype {getattr(raw, 'dtype', None)} -- "
            "unexpected OpenCV output"
        )

    lines: list[Line] = []
    for r, t in raw[:, 0, :]:
        r_float, t_float = float(r), float(t)
        if not (math.isfinite(r_float) and math.isfinite(t_float)):
            raise RuntimeError(
                f"cv2.HoughLines returned a non-finite line: rho={r_float}, theta={t_float}"
            )
        lines.append(Line(rho=r_float, theta=t_float))
    return lines


def hough_line_segments(
    image: Mask,
    threshold: int,
    rho: float = 1.0,
    theta: float = np.pi / 180,
    min_line_length: int = 0,
    max_line_gap: int = 0,
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
    min_line_length : int, default 0
        Minimum line length, in pixels; shorter segments are rejected.
        Must be a non-negative integer within the signed `int32` range.
        `improcv` accepts only an integer here even though OpenCV's own
        binding accepts a `float`: OpenCV rounds this value internally
        (``cvRound``) before use, so a fractional value promises a
        precision the algorithm doesn't actually honor.
    max_line_gap : int, default 0
        Maximum allowed gap, in pixels, between points on the same line
        to link them into one segment. Must be a non-negative integer
        within the signed `int32` range, for the same reason as
        `min_line_length`.

    Returns
    -------
    list of LineSegment
        Empty if no segments are found.

    Raises
    ------
    ValueError
        If `image` does not have exactly 2 dimensions or is empty,
        `theta` is not in ``(0, pi]``, `rho` is not in
        ``(0, 2 * (height + width) + 1]`` for `image`'s size, `rho`/
        `theta` imply a degenerate or excessively large Hough
        accumulator, `threshold` is not positive or exceeds
        ``2**31 - 1``, or `min_line_length`/`max_line_gap` is negative or
        exceeds ``2**31 - 1``.
    TypeError
        If `image` does not have dtype ``uint8``, `rho`/`theta` is not a
        real number, `threshold` is not `numbers.Integral` (rejecting
        `bool`/`float`), or `min_line_length`/`max_line_gap` is not
        `numbers.Integral` (rejecting `bool`/`float`).
    RuntimeError
        If OpenCV's raw result is not an ``int32`` `np.ndarray` with
        exactly 4 fields per segment -- an internally inconsistent
        result rather than a valid segment list.
    """
    require_image_ndim(image, ndims=(2,))
    require_dtype(image, (np.uint8,))
    rho_float, theta_float = _require_safe_hough_line_params(image, rho, theta)
    require_positive_integral(threshold, "threshold")
    threshold_int = int(threshold)
    if threshold_int > _MAX_INT32:
        raise ValueError(
            f"threshold must fit within the range of int32 ([1, {_MAX_INT32}]), got {threshold}"
        )

    require_integral(min_line_length, "min_line_length")
    min_line_length_int = int(min_line_length)
    if min_line_length_int < 0 or min_line_length_int > _MAX_INT32:
        raise ValueError(
            f"min_line_length must fit within the range of a non-negative int32 "
            f"([0, {_MAX_INT32}]), got {min_line_length}"
        )

    require_integral(max_line_gap, "max_line_gap")
    max_line_gap_int = int(max_line_gap)
    if max_line_gap_int < 0 or max_line_gap_int > _MAX_INT32:
        raise ValueError(
            f"max_line_gap must fit within the range of a non-negative int32 "
            f"([0, {_MAX_INT32}]), got {max_line_gap}"
        )

    # Same no-mutation rationale as hough_lines: OpenCV's own docs permit
    # HoughLinesP to modify its input too.
    raw = cv2.HoughLinesP(
        image.copy(),
        rho_float,
        theta_float,
        threshold_int,
        minLineLength=min_line_length_int,
        maxLineGap=max_line_gap_int,
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
        resolution. Must be at least ``1.0`` -- verified directly,
        identically for both methods, that OpenCV silently clamps `dp`
        to ``max(dp, 1.0)`` internally, so a value below `1.0` would be
        silently ignored rather than honored; this function rejects it
        instead. `improcv`'s own default (matching OpenCV's own C++
        default) -- `1.5` is only a documented *recommendation* for
        `"gradient_alt"`, not required.
    param1 : float, default 100.0
        For both methods, the higher of the two Canny thresholds used
        internally. Must be positive. For `method="gradient"` only, must
        also be at most `int32`'s max -- verified directly that OpenCV
        rounds `param1` to a C ``int`` internally for this method and
        silently wraps around beyond `int32` instead of raising, making
        the threshold meaningless rather than stricter. `improcv`'s own
        default (matching OpenCV's own C++ default) for both methods --
        `300` is only a documented *recommendation* for `"gradient_alt"`,
        not required.
    param2 : float or None, default None
        Method-specific accumulator threshold. For `"gradient"`, must be
        positive and at most `int32`'s max (same silent-wraparound risk
        as `param1` -- verified directly: `param2=2**31-1` correctly
        finds no circles, but `param2=2**31` or `param2=1e100` silently
        wrap to something small and find many). For `"gradient_alt"`,
        must be strictly between ``0.0`` and ``1.0`` (a circle
        "perfectness" measure, closer to ``1.0`` is stricter) -- verified
        this method does not exhibit the same wraparound. When `None`,
        resolves to `100.0` for `"gradient"` or `0.9` for `"gradient_alt"`
        -- verified directly that OpenCV's own omitted-parameter default
        (`100.0` regardless of method) violates `"gradient_alt"`'s own
        required range and raises a raw `cv2.error`, so this function
        never lets that combination reach OpenCV.
    min_radius : int, default 0
        Minimum circle radius, in pixels. Must be a non-negative integer
        within the signed `int32` range, and -- when `max_radius` is `0`
        (automatic upper bound) -- within a limit derived from `image`'s
        own dimensions and `method` (see `max_radius`). `0` means no
        lower bound (OpenCV's own default).
    max_radius : int, default 0
        Maximum circle radius, in pixels, within the signed `int32`
        range. Semantics:

        - ``0`` (default, OpenCV's own default): automatic upper bound
          (OpenCV uses the maximum image dimension). When this is used,
          `min_radius` must itself be less than ``max(height, width)``
          for `method="gradient"`, or at most
          ``min(height, width) // 2`` for `method="gradient_alt"`.
        - Greater than `min_radius`: an explicit range, honored as given
          -- but also capped at ``max(height, width)`` for
          `method="gradient"`, or ``min(height, width) // 2`` for
          `method="gradient_alt"`. Being within `int32` alone isn't
          enough: verified directly that `cv2.HoughCircles` allocates
          memory proportional to `max_radius` itself, not image size --
          `max_radius=50_000_000` on a 64x64 image measurably consumed
          gigabytes of memory and can be killed by the OS on a
          memory-constrained system, despite being an entirely ordinary
          `int32` value.
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
        `method` is not recognized, `dp` is not finite or below `1.0`,
        `min_dist`/`param1` is not finite or not strictly positive,
        `param2` (once resolved) is out of its method-specific range,
        `min_radius`/`max_radius` is outside the signed `int32` range,
        `min_radius` is negative, or `max_radius`'s value is inconsistent
        with `method`/`min_radius` per the semantics above.
    TypeError
        If `image` does not have dtype ``uint8``, `dp`/`min_dist`/`param1`/
        `param2` is not a real number, or `min_radius`/`max_radius` is not
        `numbers.Integral` (rejecting `bool`/`float`).
    RuntimeError
        If OpenCV's raw result is not a ``float32`` `np.ndarray` with 3 or
        4 fields per circle, any returned `x`/`y`/`radius` is not finite,
        or any returned `radius` is negative -- an internally inconsistent
        result rather than a valid circle list.
    """
    require_image_ndim(image, ndims=(2,))
    require_dtype(image, (np.uint8,))
    require_one_of(method, ("gradient", "gradient_alt"), "method")
    dp_float = _require_dp_at_least_one(dp, "dp")
    min_dist_float = _require_strictly_positive(min_dist, "min_dist")
    param1_float = _require_strictly_positive(param1, "param1")
    # Verified directly, identically on both OpenCV versions: for
    # method="gradient" only, cv2.HoughCircles rounds param1/param2 to a
    # C int internally, and a value beyond int32 silently wraps around
    # instead of raising -- param2=2**31-1 correctly finds 0 circles,
    # but param2=2**31 (and 1e100) silently wraps to something small and
    # finds many circles instead, as if the threshold barely mattered.
    # "gradient_alt" does not exhibit this (confirmed no such jump for
    # huge param1 there), so this bound is method-specific.
    if method == "gradient" and param1_float > _MAX_INT32:
        raise ValueError(
            f"param1 must be at most {_MAX_INT32} for method='gradient' -- verified directly "
            f"that OpenCV rounds param1 to a C int internally and silently wraps around "
            f"beyond int32, got {param1}"
        )

    if param2 is None:
        param2_float = _DEFAULT_PARAM2[method]
    else:
        require_finite(param2, "param2")
        param2_float = float(param2)
        if method == "gradient_alt":
            if not (0.0 < param2_float < 1.0):
                raise ValueError(
                    f"param2 must be strictly between 0.0 and 1.0 for method='gradient_alt', "
                    f"got {param2}"
                )
        else:
            if not param2_float > 0.0:
                raise ValueError(f"param2 must be positive for method='gradient', got {param2}")
            if param2_float > _MAX_INT32:
                raise ValueError(
                    f"param2 must be at most {_MAX_INT32} for method='gradient' -- verified "
                    f"directly that OpenCV rounds param2 to a C int internally and silently "
                    f"wraps around beyond int32, got {param2}"
                )

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

    # Being within the int32 range isn't enough: verified directly that
    # cv2.HoughCircles allocates memory proportional to max_radius
    # itself (not just image size) -- max_radius=50_000_000 on a tiny
    # 64x64 image measurably consumed gigabytes of memory and can be
    # killed by the OS (SIGKILL) on a memory-constrained system, even
    # though it's a perfectly ordinary int32 value. Bound max_radius (and,
    # for the auto-range max_radius=0 case, min_radius) to the image's own
    # dimensions instead.
    height, width = image.shape
    if max_radius_int > 0:
        radius_limit = max(height, width) if method == "gradient" else min(height, width) // 2
        if max_radius_int > radius_limit:
            raise ValueError(
                f"max_radius ({max_radius_int}) exceeds the safe limit ({radius_limit}) for a "
                f"{height}x{width} image with method={method!r} -- verified directly that "
                "larger values can exhaust memory or be killed by the OS"
            )
    elif max_radius_int == 0:
        if method == "gradient":
            radius_limit = max(height, width)
            if not min_radius_int < radius_limit:
                raise ValueError(
                    f"min_radius ({min_radius_int}) must be less than {radius_limit} for a "
                    f"{height}x{width} image with method='gradient' when max_radius=0 (automatic "
                    "upper bound)"
                )
        else:
            radius_limit = min(height, width) // 2
            if not min_radius_int <= radius_limit:
                raise ValueError(
                    f"min_radius ({min_radius_int}) must be at most {radius_limit} for a "
                    f"{height}x{width} image with method='gradient_alt' when max_radius=0 "
                    "(automatic upper bound)"
                )

    raw = cv2.HoughCircles(
        image,
        _HOUGH_CIRCLE_METHODS[method],
        dp_float,
        min_dist_float,
        param1=param1_float,
        param2=param2_float,
        minRadius=min_radius_int,
        maxRadius=max_radius_int,
    )
    if raw is None:
        return []

    if (
        not isinstance(raw, np.ndarray)
        or raw.dtype != np.float32
        or raw.ndim != 3
        or raw.shape[0] != 1
        or raw.shape[2] not in (3, 4)
    ):
        raise RuntimeError(
            f"cv2.HoughCircles returned an array of shape "
            f"{getattr(raw, 'shape', None)} dtype {getattr(raw, 'dtype', None)} -- "
            "unexpected OpenCV output"
        )

    circles: list[Circle] = []
    for row in raw[0]:
        x, y, radius = float(row[0]), float(row[1]), float(row[2])
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(radius)):
            raise RuntimeError(
                f"cv2.HoughCircles returned a non-finite circle: x={x}, y={y}, radius={radius}"
            )
        if radius < 0.0:
            raise RuntimeError(f"cv2.HoughCircles returned a negative radius: {radius}")
        circles.append(Circle(x=x, y=y, radius=radius))
    return circles
