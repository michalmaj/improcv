"""Standalone point/region detectors: FAST, blob (SimpleBlobDetector), MSER."""

from __future__ import annotations

import math
import numbers
from typing import Literal, NamedTuple

import cv2
import numpy as np
import numpy.typing as npt

from improcv._validation import (
    require_bool,
    require_dtype,
    require_fits_dtype,
    require_image_ndim,
    require_integral,
    require_one_of,
    require_spatial_mask,
)
from improcv.types import BoundingBox, Mask

__all__ = [
    "FastType",
    "MSERRegion",
    "detect_blob_keypoints",
    "detect_fast_keypoints",
    "detect_mser_regions",
]


def _require_valid_detector_image(image: np.ndarray) -> None:
    """Raise ValueError/TypeError unless `image` is a uint8 grayscale/BGR/BGRA array.

    Shared by every detector in this module -- all three verified
    directly to work correctly with 1, 3, or 4 channels; a 2-channel
    image reaches an unclear raw cv2.error otherwise.
    """
    require_image_ndim(image, ndims=(2, 3))
    channels = 1 if image.ndim == 2 else image.shape[2]
    require_one_of(channels, (1, 3, 4), "image channel count")
    require_dtype(image, (np.uint8,))


def _normalize_integral_param(value: object, name: str) -> int:
    """Raise TypeError/ValueError unless `value` is an int32-bounded integral; return plain int.

    Requires `numbers.Integral` (`bool` rejected), fitting signed
    `int32` (a huge value reaches a raw "integer won't fit into a C int"
    cv2 error otherwise). Shared by `threshold` (FAST) and
    `delta`/`min_area`/`max_area` (MSER).
    """
    require_integral(value, name)
    assert isinstance(value, numbers.Integral)  # narrows for the type checker
    require_fits_dtype(value, np.int32, name)
    return int(value)


def _require_valid_keypoints(raw: object) -> list[cv2.KeyPoint]:
    """Raise RuntimeError unless `raw` is a valid sequence of cv2.KeyPoint; return a fresh list.

    Shared by `detect_fast_keypoints`/`detect_blob_keypoints`. Checks
    `pt`/`size`/`response` are finite and `size` is non-negative --
    defensive, not independently reproduced against real OpenCV output.
    """
    if not isinstance(raw, (list, tuple)):
        raise RuntimeError(
            f"detector.detect returned a {type(raw).__name__}, expected a list/tuple of "
            "cv2.KeyPoint -- unexpected OpenCV output"
        )
    result: list[cv2.KeyPoint] = []
    for i, kp in enumerate(raw):
        if not isinstance(kp, cv2.KeyPoint):
            raise RuntimeError(
                f"detector.detect()[{i}] is a {type(kp).__name__}, expected cv2.KeyPoint -- "
                "unexpected OpenCV output"
            )
        if not (
            math.isfinite(kp.pt[0])
            and math.isfinite(kp.pt[1])
            and math.isfinite(kp.size)
            and math.isfinite(kp.response)
        ):
            raise RuntimeError(
                f"detector.detect()[{i}] has non-finite pt/size/response -- "
                "unexpected OpenCV output"
            )
        if kp.size < 0:
            raise RuntimeError(
                f"detector.detect()[{i}] has negative size {kp.size} -- unexpected OpenCV output"
            )
        result.append(kp)
    return result


FastType = Literal["type_5_8", "type_7_12", "type_9_16"]

_FAST_TYPES: dict[FastType, int] = {
    "type_5_8": cv2.FastFeatureDetector_TYPE_5_8,
    "type_7_12": cv2.FastFeatureDetector_TYPE_7_12,
    "type_9_16": cv2.FastFeatureDetector_TYPE_9_16,
}


def detect_fast_keypoints(
    image: np.ndarray,
    threshold: int = 10,
    nonmax_suppression: bool = True,
    fast_type: FastType = "type_9_16",
    mask: Mask | None = None,
) -> list[cv2.KeyPoint]:
    """Detect FAST corner keypoints.

    Parameters
    ----------
    image : np.ndarray
        A `uint8` image, grayscale (``(H, W)``), BGR (``(H, W, 3)``), or
        BGRA (``(H, W, 4)``).
    threshold : int, default 10
        Intensity difference threshold. Must be an integral number (no
        `bool`) fitting signed `int32`, further restricted to
        ``[0, 255]`` -- OpenCV silently accepts values outside this range
        with undefined-looking behavior rather than raising (verified
        directly).
    nonmax_suppression : bool, default True
        Whether to apply non-maximum suppression to detected corners.
    fast_type : {"type_5_8", "type_7_12", "type_9_16"}, default "type_9_16"
        FAST corner-detection neighborhood variant -- OpenCV silently
        accepts any int here with no validation at all (verified
        directly), so this is checked against the three real values.
    mask : np.ndarray, optional
        Optional `uint8` mask restricting detection to nonzero regions,
        matching `image`'s spatial size.

    Returns
    -------
    list of cv2.KeyPoint
        Detected keypoints, in OpenCV's own `cv2.KeyPoint` form (directly
        usable with `cv2.drawKeypoints`) -- no descriptors are computed
        (this is detection only, not `detect_and_compute`).

    Raises
    ------
    ValueError
        If `image` does not have exactly 2 dimensions or a channel count
        in ``{1, 3, 4}``, or is empty; if `threshold` is outside
        ``[0, 255]`` or doesn't fit signed `int32`; if `fast_type` isn't
        one of the three accepted values; if `mask`'s spatial shape
        doesn't match `image`'s.
    TypeError
        If `image` does not have dtype ``uint8``; if `threshold` isn't an
        integral type or is a `bool`; if `nonmax_suppression` isn't a
        `bool`; if `mask` isn't `uint8`.
    RuntimeError
        If OpenCV's raw keypoint output is internally inconsistent (wrong
        type, non-finite fields, negative size).
    """
    _require_valid_detector_image(image)
    threshold_int = _normalize_integral_param(threshold, "threshold")
    if not (0 <= threshold_int <= 255):
        raise ValueError(f"threshold must be in [0, 255], got {threshold_int}")
    require_bool(nonmax_suppression, "nonmax_suppression")
    require_one_of(fast_type, tuple(_FAST_TYPES.keys()), "fast_type")
    if mask is not None:
        require_spatial_mask(mask, image)

    detector = cv2.FastFeatureDetector.create(
        threshold_int, nonmax_suppression, _FAST_TYPES[fast_type]
    )
    return _require_valid_keypoints(detector.detect(image, mask))


def detect_blob_keypoints(
    image: np.ndarray,
    params: cv2.SimpleBlobDetector.Params | None = None,
    mask: Mask | None = None,
) -> list[cv2.KeyPoint]:
    """Detect blob keypoints.

    Parameters
    ----------
    image : np.ndarray
        A `uint8` image, grayscale (``(H, W)``), BGR (``(H, W, 3)``), or
        BGRA (``(H, W, 4)``).
    params : cv2.SimpleBlobDetector.Params, optional
        Detector configuration (threshold range, area/circularity/
        convexity/inertia/color filters, etc.), passed straight through to
        `cv2.SimpleBlobDetector.create` -- not re-exposed as separate
        function parameters given the 14-field surface. `None` uses
        OpenCV's own defaults.
    mask : np.ndarray, optional
        Optional `uint8` mask restricting detection to nonzero regions,
        matching `image`'s spatial size.

    Returns
    -------
    list of cv2.KeyPoint
        Detected keypoints; no descriptors.

    Raises
    ------
    ValueError
        If `image` does not have exactly 2 dimensions or a channel count
        in ``{1, 3, 4}``, or is empty; if `mask`'s spatial shape doesn't
        match `image`'s; if `params` is structurally valid but describes
        an internally invalid configuration (e.g. `thresholdStep <= 0`).
    TypeError
        If `image` does not have dtype ``uint8``; if `params` is given and
        isn't a `cv2.SimpleBlobDetector.Params`; if `mask` isn't `uint8`.
    RuntimeError
        If OpenCV's raw keypoint output is internally inconsistent.
    """
    _require_valid_detector_image(image)
    if params is not None and not isinstance(params, cv2.SimpleBlobDetector.Params):
        raise TypeError(
            f"params must be a cv2.SimpleBlobDetector.Params or None, got {type(params).__name__}"
        )
    if mask is not None:
        require_spatial_mask(mask, image)

    try:
        detector = (
            cv2.SimpleBlobDetector.create(params)
            if params is not None
            else cv2.SimpleBlobDetector.create()
        )
    except cv2.error as exc:
        raise ValueError("params contains an invalid SimpleBlobDetector configuration") from exc

    return _require_valid_keypoints(detector.detect(image, mask))


def _require_valid_mser_image(image: np.ndarray) -> None:
    """Raise ValueError/TypeError per `_require_valid_detector_image`, plus MSER's own 3x3 floor.

    `detectRegions` raises a raw cv2.error for anything smaller than 3x3
    in either dimension (verified directly) -- rejected here first for a
    clear, attributable message.
    """
    _require_valid_detector_image(image)
    if image.shape[0] < 3 or image.shape[1] < 3:
        raise ValueError(f"image must be at least 3x3 for MSER detection, got {image.shape[:2]}")


class MSERRegion(NamedTuple):
    """One region found by `detect_mser_regions`.

    `points` is every pixel belonging to the region as an unordered
    ``(N, 2)`` int32 point set -- verified directly (by rendering a
    region's points into its own bounding box) that these are scattered
    membership points covering the region's interior, *not* an ordered
    boundary walk. **Do not pass `points` to `draw_contours` or any other
    `Contour`-consuming function** -- it would connect points in
    arbitrary order, drawing a nonsensical zigzag polygon instead of the
    region's outline. To get an ordered boundary, build a mask from
    `points` and call `find_contours` on it, or compute a convex hull
    (`convex_hull`) if that's an acceptable approximation.
    """

    points: npt.NDArray[np.int32]
    bounding_box: BoundingBox


def _normalize_mser_region_points(region: object, index: int) -> np.ndarray:
    """Raise RuntimeError unless `region` is a valid MSER point set; return an int32 array.

    Handles the known pybind11 quirk where a region can come back with
    `dtype=object` (e.g. when several regions share the same point
    count) -- accepted only after confirming every element is genuinely
    integral and int32-safe; never an unconditional `astype(np.int32)`,
    which would silently truncate a corrupted float. Always returns an
    independent copy -- a region extracted from a collapsed outer
    container (see `_normalize_mser_regions_container`) is a view into a
    shared buffer, and every public `MSERRegion.points` must be
    independent.
    """
    region_arr = np.asarray(region)
    if region_arr.ndim != 2 or region_arr.shape[1] != 2 or region_arr.shape[0] == 0:
        raise RuntimeError(
            f"MSER region[{index}] has shape {getattr(region_arr, 'shape', None)}, expected "
            "(N, 2) with N > 0 -- unexpected OpenCV output"
        )
    if region_arr.dtype == np.int32:
        return region_arr.copy()
    if region_arr.dtype == object:
        for value in region_arr.reshape(-1):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise RuntimeError(
                    f"MSER region[{index}] has dtype object with a non-integral element "
                    f"{value!r} -- unexpected OpenCV output"
                )
            if not (-(2**31) <= int(value) <= 2**31 - 1):
                raise RuntimeError(
                    f"MSER region[{index}] has an element {value!r} outside int32 range -- "
                    "unexpected OpenCV output"
                )
        return region_arr.astype(np.int32)
    raise RuntimeError(
        f"MSER region[{index}] has dtype {region_arr.dtype}, expected int32 -- "
        "unexpected OpenCV output"
    )


def _normalize_mser_bbox(row: np.ndarray, index: int) -> BoundingBox:
    """Raise RuntimeError unless `row` is a valid (x, y, width, height) bbox.

    Returns a normalized `BoundingBox`.
    """
    x, y, width, height = (int(v) for v in row)
    if width <= 0 or height <= 0:
        raise RuntimeError(
            f"MSER bounding box[{index}] has non-positive width/height ({width}, {height}) -- "
            "unexpected OpenCV output"
        )
    return BoundingBox(x, y, width, height)


def _normalize_mser_regions_container(regions: object) -> list[object]:
    """Raise RuntimeError unless `regions` is a recognized MSER regions container; return a list.

    The known pybind11 quirk isn't limited to a single region's own
    `dtype=object` -- when every region happens to have the same point
    count, the *whole* `regions` output can collapse into one
    `(N, M, 2)` ndarray (either `int32` or `object` dtype) instead of
    staying a tuple of per-region arrays; when point counts differ, it
    can instead fall back to a 1D `dtype=object` array of per-region
    arrays. Each recognized variant is unpacked into a plain list here,
    with each element still validated by `_normalize_mser_region_points`
    afterward.
    """
    if isinstance(regions, (list, tuple)):
        return list(regions)

    if isinstance(regions, np.ndarray):
        if (
            regions.ndim == 3
            and regions.shape[0] > 0
            and regions.shape[2] == 2
            and regions.dtype in (np.int32, object)
        ):
            return [regions[i] for i in range(regions.shape[0])]

        if regions.ndim == 1 and regions.dtype == object:
            return list(regions)

    raise RuntimeError(
        f"MSER detectRegions returned an unexpected regions container: "
        f"{type(regions).__name__} {getattr(regions, 'shape', '')} "
        f"{getattr(regions, 'dtype', '')} -- unexpected OpenCV output"
    )


def _require_valid_mser_result(regions: object, bboxes: object) -> list[MSERRegion]:
    """Raise RuntimeError unless `(regions, bboxes)` is internally consistent; return MSERRegions.

    Handles the documented empty variant (`(), ()`), normalizes the raw
    `regions` container (see `_normalize_mser_regions_container`),
    validates each region's point array, and cross-checks each bbox
    against its own region's point min/max.
    """
    if isinstance(regions, tuple) and len(regions) == 0:
        if not (isinstance(bboxes, tuple) and len(bboxes) == 0):
            raise RuntimeError(
                "MSER detection reported zero regions but returned non-empty bboxes -- "
                "unexpected OpenCV output"
            )
        return []

    regions_list = _normalize_mser_regions_container(regions)

    bboxes_arr = np.asarray(bboxes)
    if (
        bboxes_arr.ndim != 2
        or bboxes_arr.shape[1] != 4
        or bboxes_arr.shape[0] != len(regions_list)
        or bboxes_arr.dtype != np.int32
    ):
        raise RuntimeError(
            f"MSER detectRegions returned {len(regions_list)} regions but bboxes shape "
            f"{bboxes_arr.shape} dtype {bboxes_arr.dtype} -- unexpected OpenCV output"
        )

    results: list[MSERRegion] = []
    for i, region in enumerate(regions_list):
        points = _normalize_mser_region_points(region, i)
        box = _normalize_mser_bbox(bboxes_arr[i], i)
        x_min, y_min = int(points[:, 0].min()), int(points[:, 1].min())
        x_max, y_max = int(points[:, 0].max()), int(points[:, 1].max())
        if (
            box.x != x_min
            or box.y != y_min
            or box.x + box.width - 1 != x_max
            or box.y + box.height - 1 != y_max
        ):
            raise RuntimeError(
                f"MSER bounding box[{i}] {box} does not match region[{i}]'s own point range "
                f"x[{x_min},{x_max}] y[{y_min},{y_max}] -- unexpected OpenCV output"
            )
        results.append(MSERRegion(points=points, bounding_box=box))
    return results


def detect_mser_regions(
    image: np.ndarray,
    delta: int = 5,
    min_area: int = 60,
    max_area: int = 14400,
) -> list[MSERRegion]:
    """Detect Maximally Stable Extremal Regions (MSER).

    Parameters
    ----------
    image : np.ndarray
        A `uint8` image, grayscale (``(H, W)``), BGR (``(H, W, 3)``), or
        BGRA (``(H, W, 4)``), at least 3x3 -- `detectRegions` raises a raw
        `cv2.error` below that size.
    delta, min_area, max_area : int, default 5, 60, 14400
        Must be integral (no `bool`) and each positive. `min_area`/
        `max_area` must additionally fit signed `int32`, and `min_area`
        must be less than `max_area` -- OpenCV silently returns zero
        regions for a non-positive value or an inverted range rather than
        raising (the same footgun class as `hough_circles`'s radius
        bounds). `delta` is further restricted to ``[1, 255]`` -- since
        this function only accepts `uint8` images, OpenCV's internal
        `val +/- delta` comparison on pixel levels has no meaningful
        effect (or can misbehave) outside that range.

    Returns
    -------
    list of MSERRegion
        One entry per detected region, `[]` if none are found -- MSER's
        own `bboxes` output is a plain empty tuple (not an array) in that
        case; normalized away here.

    Raises
    ------
    ValueError
        If `image` does not have exactly 2 dimensions, a channel count in
        ``{1, 3, 4}``, or at least 3x3 spatial size, or is empty; if
        `delta`/`min_area`/`max_area` isn't positive, if `min_area`/
        `max_area` doesn't fit signed `int32`, if `delta` isn't in
        ``[1, 255]``, or if `min_area >= max_area`.
    TypeError
        If `image` does not have dtype ``uint8``; if `delta`/`min_area`/
        `max_area` isn't an integral type or is a `bool`.
    RuntimeError
        If OpenCV's raw region/bbox output is internally inconsistent
        (mismatched region/bbox counts, wrong shapes, non-integral or
        out-of-int32-range point values, a non-positive bbox width/height,
        or a bbox that doesn't match its region's own point range).
    """
    _require_valid_mser_image(image)
    delta_int = _normalize_integral_param(delta, "delta")
    min_area_int = _normalize_integral_param(min_area, "min_area")
    max_area_int = _normalize_integral_param(max_area, "max_area")
    if not (1 <= delta_int <= 255):
        raise ValueError(f"delta must be in [1, 255] for uint8 MSER detection, got {delta_int}")
    if min_area_int <= 0:
        raise ValueError(f"min_area must be positive, got {min_area_int}")
    if max_area_int <= 0:
        raise ValueError(f"max_area must be positive, got {max_area_int}")
    if min_area_int >= max_area_int:
        raise ValueError(
            f"min_area must be less than max_area, got min_area={min_area_int}, "
            f"max_area={max_area_int}"
        )

    detector = cv2.MSER.create(delta=delta_int, min_area=min_area_int, max_area=max_area_int)
    regions, bboxes = detector.detectRegions(image)
    return _require_valid_mser_result(regions, bboxes)
