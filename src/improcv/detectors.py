"""Standalone point/region detectors: FAST, blob (SimpleBlobDetector), MSER."""

from __future__ import annotations

import math
import numbers
from typing import Literal

import cv2
import numpy as np

from improcv._validation import (
    require_bool,
    require_dtype,
    require_fits_dtype,
    require_image_ndim,
    require_integral,
    require_one_of,
    require_spatial_mask,
)
from improcv.types import Mask

__all__ = [
    "FastType",
    "detect_blob_keypoints",
    "detect_fast_keypoints",
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
