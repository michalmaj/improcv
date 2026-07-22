"""Hough transform-based shape detection: lines, line segments, circles."""

from __future__ import annotations

from typing import Literal, NamedTuple

__all__ = [
    "Circle",
    "HoughCircleMethod",
    "Line",
    "LineSegment",
]

HoughCircleMethod = Literal["gradient", "gradient_alt"]


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
