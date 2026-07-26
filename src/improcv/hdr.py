"""HDR-related operations: exposure fusion, radiance merging, and tone mapping."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Literal, cast

import cv2
import numpy as np

from improcv._compat.opencv import merge_hdr_supports_dtype
from improcv._validation import (
    require_bool,
    require_dtype,
    require_finite,
    require_fits_dtype,
    require_non_negative,
    require_positive,
    require_positive_integral,
    require_range,
)
from improcv.types import Image, ImageFloat32, ImageU8

__all__ = [
    "calibrate_camera_response_debevec",
    "calibrate_camera_response_robertson",
    "fuse_exposures",
    "merge_hdr_debevec",
    "merge_hdr_robertson",
    "tone_map",
    "tone_map_drago",
    "tone_map_mantiuk",
    "tone_map_reinhard",
]

_MIN_EXPOSURES = 2
_LDR_LUT_LENGTH = 256
_HDR_LUT_LENGTH = 65536
_MERGE_DTYPES = (np.uint8, np.uint16, np.float32)
_MANTIUK_MIN_DIM = 2


def _require_valid_float32_exposure_values(image: np.ndarray, name: str) -> None:
    """Raise ValueError unless a `float32` exposure image is finite and within ``[0, 1]``.

    OpenCV's HDR merge silently clips `float32` pixel values to ``[0, 1]``
    (treating `NaN`/`inf` as ordinary floats to clip) rather than rejecting
    them -- verified directly that an out-of-range value is clipped with no
    error or warning, silently discarding information.
    """
    if not np.all(np.isfinite(image)):
        raise ValueError(f"{name} must contain only finite values for float32 input")
    if image.min() < 0.0 or image.max() > 1.0:
        raise ValueError(
            f"{name} must have values in [0, 1] for float32 input, got range "
            f"[{image.min()}, {image.max()}]"
        )


def _require_valid_exposure_image(
    image: np.ndarray, name: str, allowed_dtypes: tuple[type, ...]
) -> None:
    """Raise ValueError/TypeError unless `image` is a valid grayscale or BGR
    exposure frame with one of `allowed_dtypes`.

    Accepts 2D grayscale ``(H, W)`` or 3D BGR ``(H, W, 3)`` -- ``(H, W, 1)``,
    2-channel, BGRA, and any other channel count are all rejected, with no
    automatic conversion. Unlike `improcv._validation.require_image_ndim`,
    every message here includes `name` (e.g. ``images[3]``), since this
    validates one element of a stack, where a generic "image" message would
    not say which element is invalid.
    """
    if image.ndim not in (2, 3):
        raise ValueError(f"{name} must be 2D grayscale or 3D BGR, got {image.ndim} dimensions")
    if image.size == 0:
        raise ValueError(f"{name} must not be empty, got shape {image.shape}")
    if image.ndim == 3 and image.shape[2] != 3:
        if image.shape[2] == 1:
            raise ValueError(
                f"{name} must be 2D grayscale (H, W) or 3-channel BGR (H, W, 3), got a "
                f"single-channel image with an explicit trailing axis -- drop it first "
                f"with {name}[..., 0]"
            )
        if image.shape[2] == 4:
            raise ValueError(
                f"{name} must be 2D grayscale (H, W) or 3-channel BGR (H, W, 3), got a "
                "4-channel (BGRA) image -- explicitly drop or composite the alpha "
                "channel before calling"
            )
        raise ValueError(
            f"{name} must be 2D grayscale (H, W) or 3-channel BGR (H, W, 3), got "
            f"{image.shape[2]} channels"
        )
    require_dtype(image, allowed_dtypes, name)
    if image.dtype == np.float32:
        _require_valid_float32_exposure_values(image, name)


def _require_valid_exposure_stack(
    images: object, *, allowed_dtypes: tuple[type, ...] = (np.uint8,)
) -> list[np.ndarray]:
    """Raise ValueError/TypeError unless `images` is a valid exposure stack,
    else return it as a plain `list`.

    `images` must be a real `collections.abc.Sequence` -- a single
    `np.ndarray` (including a 4D stack), a `str`/`bytes`, or a
    generator/iterator (none of which implement the `Sequence` protocol) are
    all rejected, even though OpenCV's own Python binding happens to accept
    a 4D array in place of a list. Every element must be a non-empty, 2D
    grayscale or 3D BGR `np.ndarray` with exactly the same shape and dtype
    as `images[0]` -- mixing grayscale and BGR frames in one stack is
    therefore rejected as a shape mismatch, and mixing dtypes as a dtype
    mismatch. `images[0]`'s dtype must be one of `allowed_dtypes`; a
    `float32` stack is additionally required to be finite and within
    ``[0, 1]`` (see `_require_valid_float32_exposure_values`). Every error
    message names the offending index (e.g. ``images[3]``).
    """
    if isinstance(images, (str, bytes)) or not isinstance(images, Sequence):
        raise TypeError(
            "images must be a Sequence of arrays (e.g. a list or tuple), not a single "
            f"array or {type(images).__name__}"
        )
    normalized = list(images)
    if len(normalized) < _MIN_EXPOSURES:
        raise ValueError(
            f"images must contain at least {_MIN_EXPOSURES} images, got {len(normalized)}"
        )

    first = normalized[0]
    if not isinstance(first, np.ndarray):
        raise TypeError(f"images[0] must be a NumPy array, got {type(first).__name__}")
    _require_valid_exposure_image(first, "images[0]", allowed_dtypes)

    for index, image in enumerate(normalized[1:], start=1):
        name = f"images[{index}]"
        if not isinstance(image, np.ndarray):
            raise TypeError(f"{name} must be a NumPy array, got {type(image).__name__}")
        if image.shape != first.shape:
            raise ValueError(
                f"{name} has shape {image.shape}, expected {first.shape} (matching images[0])"
            )
        if image.dtype != first.dtype:
            raise TypeError(
                f"{name} has dtype {image.dtype}, expected {first.dtype} (matching images[0])"
            )
        if image.dtype == np.float32:
            _require_valid_float32_exposure_values(image, name)

    return normalized


def _validated_weight(value: object, name: str) -> float:
    """Raise TypeError/ValueError unless `value` is a valid Mertens weight
    parameter, else return its `float32` value as a plain `float` -- the
    exact value OpenCV will receive.

    Non-negative, with `0` a legal value for all three weights (it is even
    `exposure_weight`'s own default). No OpenCV-documented upper bound, so
    (like `denoising.py`'s `h`) an extreme value can overflow to `inf` once
    converted to `float32` -- both the conversion and the finiteness check
    are wrapped so this never raises an uncontrolled `RuntimeWarning` or
    lets a non-finite value reach OpenCV silently.
    """
    require_non_negative(value, name)
    original = float(value)  # type: ignore[arg-type]

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        converted = np.float32(original)

    if not np.isfinite(converted):
        raise ValueError(
            f"{name} is too large to represent as OpenCV's float32 parameter, got {value}"
        )
    if original > 0.0 and converted == 0.0:
        raise ValueError(
            f"{name} must be positive, got {value}, which is too small to remain "
            "positive once converted to OpenCV's float32 parameter"
        )
    return float(converted)


def _validated_positive_float32(value: object, name: str) -> float:
    """Raise TypeError/ValueError unless `value` is a valid, strictly positive
    parameter safely representable as OpenCV's `float32`, else return its
    `float32` value as a plain `float` -- the exact value OpenCV will
    receive.

    Shared by exposure times and `smoothness` (`CalibrateDebevec`'s
    `lambda`) -- both require a strictly positive, finite value, validated
    on the `float32`-converted value (like `denoising.py`'s `h`) since an
    extreme-but-finite Python value can overflow to `inf`, or a tiny
    positive value can underflow to exactly `0.0`, once cast to `float32`.
    """
    require_positive(value, name)
    original = float(value)  # type: ignore[arg-type]

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        converted = np.float32(original)

    if not np.isfinite(converted):
        raise ValueError(
            f"{name} is too large to represent as OpenCV's float32 parameter, got {value}"
        )
    if converted == 0.0:
        raise ValueError(
            f"{name} must be positive, got {value}, which is too small to remain "
            "positive once converted to OpenCV's float32 parameter"
        )
    return float(converted)


def _validated_exposure_time(value: object, index: int) -> float:
    """Raise TypeError/ValueError unless `value` is a valid single exposure
    time, else return its `float32` value as a plain `float`.

    Verified directly that OpenCV performs no validation on exposure times
    at all: zero, negative, `NaN`, and `inf` are all silently accepted,
    producing numerically meaningless (though not always non-finite)
    results -- see `_validated_positive_float32` for the shared
    strictly-positive-`float32`-safe contract.
    """
    return _validated_positive_float32(value, f"exposure_times[{index}]")


def _validated_threshold(value: object, name: str) -> float:
    """Raise TypeError/ValueError unless `value` is a valid, non-negative
    `threshold` parameter (`CalibrateRobertson`) safely representable as
    OpenCV's `float32`, else return its `float32` value as a plain `float`.

    Unlike `_validated_positive_float32`, a positive value that underflows
    to `0.0f` is accepted rather than rejected: `threshold=0` is itself a
    legal, documented value here (it just means the iteration effectively
    never stops early, not that it runs zero iterations), so a tiny
    positive value reaching that same, already-legal `0.0` is not a hidden
    contract violation.
    """
    require_non_negative(value, name)
    original = float(value)  # type: ignore[arg-type]

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        converted = np.float32(original)

    if not np.isfinite(converted):
        raise ValueError(
            f"{name} is too large to represent as OpenCV's float32 parameter, got {value}"
        )
    return float(converted)


def _require_valid_exposure_times(exposure_times: object, count: int) -> np.ndarray:
    """Raise ValueError/TypeError unless `exposure_times` is a valid,
    index-paired sequence of exposure times, else return a fresh,
    contiguous `float32` NumPy array -- never the caller's own array or
    buffer.

    OpenCV's own Python binding rejects a plain Python list for this
    parameter outright (``Expected Ptr<cv::UMat>``), and verified directly
    that passing a `float64` NumPy array with numerically identical values
    to a working `float32` array can silently produce a fully non-finite
    (``inf``) merge result, with no error -- so this always builds a brand
    new `float32` array explicitly (`copy=True`, `order="C"`), regardless of
    what was passed in; `np.asarray` alone would not guarantee a fresh
    buffer.
    """
    if isinstance(exposure_times, (str, bytes)) or not isinstance(exposure_times, Sequence):
        raise TypeError(
            "exposure_times must be a Sequence of real numbers (e.g. a list or tuple), "
            f"not a single array or {type(exposure_times).__name__}"
        )
    times_list = list(exposure_times)
    if len(times_list) != count:
        raise ValueError(
            f"exposure_times must contain exactly {count} values (one per image), "
            f"got {len(times_list)}"
        )
    validated = [_validated_exposure_time(value, index) for index, value in enumerate(times_list)]
    return np.array(validated, dtype=np.float32, copy=True, order="C")


def _lut_length_for_dtype(dtype: np.dtype) -> int:
    """Return the response-curve LUT length OpenCV expects for `dtype`.

    ``256`` for `uint8`, ``65536`` for `uint16`/`float32` -- verified
    directly from OpenCV's source and confirmed empirically that a
    mismatched length raises a `cv2.error`, not a universal `(256, 1, 3)`
    for every dtype.
    """
    return _LDR_LUT_LENGTH if dtype == np.uint8 else _HDR_LUT_LENGTH


def _expected_response_curve_shape(image: np.ndarray) -> tuple[int, ...]:
    """Compute OpenCV's expected response-curve shape for `image`'s dtype
    and channel count.

    ``256``-entry LUT for `uint8`, ``65536`` for `uint16`/`float32`;
    ``(length, 1)`` for a grayscale image, ``(length, 1, channels)`` for a
    multi-channel one. Shared by `_require_valid_response_curve`
    (validating a curve supplied to `merge_hdr_debevec`/`merge_hdr_robertson`,
    both BGR-only today, so only ever exercised here with `channels == 3`)
    and by `calibrate_camera_response_debevec`/`_robertson`'s own output
    postcondition -- `calibrate_camera_response_debevec` can legally
    produce a grayscale curve (`CalibrateDebevec` genuinely supports
    grayscale, unlike `MergeDebevec`'s buggy default-response path), even
    though no current merge function can consume one.
    """
    lut_length = _lut_length_for_dtype(image.dtype)
    channels = 1 if image.ndim == 2 else image.shape[2]
    return (lut_length, 1) if channels == 1 else (lut_length, 1, channels)


def _require_valid_response_curve(
    response_curve: object, image: np.ndarray, *, method: Literal["debevec", "robertson"]
) -> np.ndarray:
    """Raise ValueError/TypeError unless `response_curve` is a valid response
    curve for `image`'s dtype and channel count, else return it unchanged.

    Shape depends on `image`'s dtype and channel count -- see
    `_expected_response_curve_shape`. Neither continuity nor writeability
    is required -- verified directly that OpenCV safely accepts a
    non-contiguous or read-only curve without mutating it, so no defensive
    copy is made here.

    `method` selects the value contract, which differs between the two
    merge algorithms: `merge_hdr_debevec` takes the curve's logarithm
    internally, so every entry must be strictly positive; `merge_hdr_robertson`
    uses the curve directly and tolerates zero entries, but not a curve that
    is zero everywhere or contains a negative entry. Verified directly that
    OpenCV itself enforces neither rule, instead silently producing
    corrupted-looking-but-finite or outright `NaN` output -- this is
    treated as a detectable input error rather than deferred to the output
    postcondition.
    """
    if not isinstance(response_curve, np.ndarray):
        raise TypeError(
            f"response_curve must be a NumPy array, got {type(response_curve).__name__}"
        )
    require_dtype(response_curve, (np.float32,), "response_curve")

    expected_shape = _expected_response_curve_shape(image)
    if response_curve.shape != expected_shape:
        channels = 1 if image.ndim == 2 else image.shape[2]
        kind = "grayscale" if channels == 1 else "BGR"
        raise ValueError(
            f"response_curve must have shape {expected_shape} for a {image.dtype} {kind} "
            f"stack, got {response_curve.shape}"
        )
    if not np.all(np.isfinite(response_curve)):
        raise ValueError("response_curve must contain only finite values")

    if method == "debevec":
        if not np.all(response_curve > 0.0):
            raise ValueError(
                "response_curve must be strictly positive everywhere for "
                "merge_hdr_debevec (its logarithm is used internally)"
            )
    else:
        if np.any(response_curve < 0.0):
            raise ValueError("response_curve must not contain negative values")
        if not np.any(response_curve > 0.0):
            raise ValueError("response_curve must not be all-zero")

    return response_curve


def _validated_positive_int_param(value: object, name: str) -> int:
    """Raise TypeError/ValueError unless `value` is a positive integral
    number fitting a signed 32-bit C++ int, else return it as a plain `int`.

    Shared by `max_iterations` (`CalibrateRobertson`) and, as the first
    step of its own further validation, `samples` (`CalibrateDebevec`).
    Both are OpenCV `int` parameters where `0` is rejected as a misleading
    no-op -- `max_iterations=0` runs zero calibration iterations, and
    `samples=0` cannot form any sampling grid -- rather than because
    OpenCV itself necessarily rejects `0` cleanly.
    """
    require_positive_integral(value, name)
    numeric = int(value)  # type: ignore[arg-type]
    require_fits_dtype(numeric, np.int32, name)
    return numeric


def _validated_debevec_samples(value: object, image: np.ndarray, random_sampling: bool) -> int:
    """Raise TypeError/ValueError unless `value` is a valid `samples`
    parameter for `calibrate_camera_response_debevec`, else return it as a
    plain `int`.

    For `random_sampling=True`, only the shared positive-integral-in-range
    contract applies (see `_validated_positive_int_param`) -- OpenCV samples
    pixel locations with replacement, so there is no grid to validate and
    no upper bound tied to the image's pixel count.

    For `random_sampling=False` (the default, grid-based mode), `samples`
    is only a *target* count: OpenCV computes
    ``x_points = int(sqrt(samples * width / height))`` and
    ``y_points = samples // x_points``, then samples that rectangular grid
    -- not exactly `samples` points (verified directly: e.g. `samples=4`
    and `samples=5` produce the identical 2x2 grid on a square image, due
    to integer truncation in this same formula). This replicates that
    exact formula to validate ``1 <= x_points <= width`` and
    ``1 <= y_points <= height`` *before* calling OpenCV, since a `samples`
    value violating either bound otherwise raises a raw, unindexed
    `CV_Assert` (verified directly for both `samples` too small, e.g. `0`,
    and too large relative to the image size).
    """
    numeric = _validated_positive_int_param(value, "samples")
    if random_sampling:
        return numeric

    height, width = image.shape[:2]
    ratio = float(numeric) * width / height
    if not math.isfinite(ratio):
        raise ValueError(
            f"samples={value} is too large to compute a sampling grid safely for a "
            f"{width}x{height} image"
        )
    x_points = int(math.sqrt(ratio))
    if not (1 <= x_points <= width):
        raise ValueError(
            f"samples={value} does not fit a valid sampling grid for a {width}x{height} "
            f"image with random_sampling=False -- samples is a target grid size, rounded "
            f"to x_points x y_points via OpenCV's own formula, and the resulting x_points "
            f"({x_points}) must be between 1 and {width} (the image width); try a "
            "different samples value, or use random_sampling=True"
        )
    y_points = numeric // x_points
    if not (1 <= y_points <= height):
        raise ValueError(
            f"samples={value} does not fit a valid sampling grid for a {width}x{height} "
            f"image with random_sampling=False -- the resulting y_points ({y_points}) must "
            f"be between 1 and {height} (the image height); try a different samples value, "
            "or use random_sampling=True"
        )
    return numeric


def _require_merge_dtype_supported(
    image: np.ndarray, merger_factory: Callable[[], object], operation: str
) -> None:
    """Raise ValueError unless the installed OpenCV build's HDR merge
    supports `image`'s dtype.

    Verified directly that `uint16`/`float32` support was added to OpenCV's
    HDR merge (`cv2.MergeDebevec`/`cv2.MergeRobertson`) at some point after
    this project's documented minimum supported OpenCV (`opencv-python>=4.9`
    -- `4.9.0` itself raises a raw, unindexed `CV_Assert(images[0].depth()
    == CV_8U)` for anything but `uint8`). Rather than letting that raw
    assertion surface, this checks the real capability first (see
    `improcv._compat.opencv.merge_hdr_supports_dtype`) and raises a clear,
    actionable message naming the actual dtype and limitation.
    """
    if not merge_hdr_supports_dtype(merger_factory, image.dtype.type):
        raise ValueError(
            f"images[0] has dtype {image.dtype}, but this OpenCV build's {operation} does "
            "not support it (only uint8) -- uint16/float32 support was added to OpenCV's "
            "HDR merge after this project's minimum supported OpenCV (opencv-python>=4.9); "
            "upgrade OpenCV or use a uint8 image stack instead"
        )


def _validated_float32_result(
    result: object, expected_shape: tuple[int, ...], operation: str
) -> ImageFloat32:
    """Raise RuntimeError unless `result` is a finite `float32` array of
    `expected_shape`, else return it cast to `ImageFloat32`.

    Shared postcondition for every OpenCV HDR call in this module -- verified
    directly that `MergeMertens` can silently return an all-`NaN` array for
    an extreme (but individually representable) weight, and that
    `MergeDebevec`/`MergeRobertson` can silently produce huge, corrupted-
    looking values or outright `NaN` for a malformed response curve.
    Returning any of those directly would silently hand the caller a
    non-finite or wrongly-shaped "answer" instead of signaling failure.
    """
    if not isinstance(result, np.ndarray):
        raise RuntimeError(f"{operation} did not return a NumPy array")
    if result.dtype != np.float32:
        raise RuntimeError(f"{operation} returned dtype {result.dtype} instead of float32")
    if result.shape != expected_shape:
        raise RuntimeError(f"{operation} returned shape {result.shape}, expected {expected_shape}")
    if not np.all(np.isfinite(result)):
        raise RuntimeError(f"{operation} did not produce a finite result for the given inputs")
    return cast(ImageFloat32, result)


def _validated_calibration_result(
    result: object, expected_shape: tuple[int, ...], *, method: Literal["debevec", "robertson"]
) -> ImageFloat32:
    """Raise RuntimeError unless `result` is a finite `float32` response
    curve of `expected_shape` that also satisfies the value contract the
    corresponding merge function's `response_curve` requires, else return
    it cast to `ImageFloat32`.

    First reuses `_validated_float32_result` for the shared ndarray/dtype/
    shape/finiteness checks. Then applies the same value rule
    `_require_valid_response_curve` enforces on a user-supplied curve --
    but raises `RuntimeError`, not `ValueError`, since a curve failing this
    check here is OpenCV's own calibration output, not a caller mistake.

    Verified directly, with a deterministic counterexample, that OpenCV's
    calibration can legally return a *finite* curve `merge_hdr_debevec`
    still cannot use safely: `CalibrateDebevec` estimates in log-space and
    then exponentiates, so a very negative (but finite) intermediate value
    can underflow `float32` to exactly `0.0` -- silently passing the
    existing finiteness check while still being unusable, since
    `merge_hdr_debevec` takes the curve's logarithm. `CalibrateRobertson`'s
    own merge counterpart tolerates zero entries (only a negative entry or
    an all-zero curve is rejected), so its value rule is correspondingly
    looser.
    """
    operation = "CalibrateDebevec" if method == "debevec" else "CalibrateRobertson"
    validated = _validated_float32_result(result, expected_shape, operation)

    if method == "debevec":
        if not np.all(validated > 0.0):
            raise RuntimeError(
                f"{operation} returned a response curve containing zero or negative "
                "values, which cannot be used safely by merge_hdr_debevec because it "
                "takes the curve's logarithm"
            )
    else:
        if np.any(validated < 0.0):
            raise RuntimeError(
                f"{operation} returned a response curve containing negative values, "
                "which merge_hdr_robertson's own response_curve contract rejects"
            )
        if not np.any(validated > 0.0):
            raise RuntimeError(
                f"{operation} returned an all-zero response curve, which "
                "merge_hdr_robertson's own response_curve contract rejects"
            )

    return validated


def fuse_exposures(
    images: Sequence[ImageU8],
    *,
    contrast_weight: float = 1.0,
    saturation_weight: float = 1.0,
    exposure_weight: float = 0.0,
) -> ImageFloat32:
    """Fuse a stack of differently-exposed images into one well-exposed image.

    Implemented via OpenCV's Mertens exposure fusion (`cv2.createMergeMertens`).
    Unlike `merge_hdr_debevec`/`merge_hdr_robertson`, this does **not**
    reconstruct a physical HDR radiance map and does **not** require exposure
    times -- it blends the input images directly, in the domain of a
    Laplacian pyramid, weighted by per-pixel measures of local contrast,
    color saturation, and "well-exposedness" (closeness to mid-gray). The
    result does not need tone mapping, though it does need explicit
    preparation before saving as `uint8` -- see `Returns`.

    Parameters
    ----------
    images : Sequence[np.ndarray]
        A real `Sequence` (e.g. a list or tuple) of at least 2 images -- a
        single `np.ndarray` (including a 4D stack), even though OpenCV's own
        Python binding happens to accept one in place of a list, is rejected
        explicitly, as are a `str`/`bytes` and a generator/iterator. Every
        image must be `uint8`, non-empty, and have exactly the same shape as
        `images[0]`: either 2D grayscale ``(H, W)`` or 3D BGR ``(H, W, 3)``
        -- ``(H, W, 1)``, 2-channel, and BGRA are all rejected, with no
        automatic conversion, and mixing grayscale and BGR frames in one
        stack is rejected as a shape mismatch. Neither the images nor the
        `images` container itself are modified.
    contrast_weight : float, optional
        Weight for the local contrast (Laplacian-magnitude) measure. Must be
        non-negative and finite. Default `1.0`, OpenCV's own default.
    saturation_weight : float, optional
        Weight for the per-pixel color saturation (deviation-from-mean)
        measure. Must be non-negative and finite. Default `1.0`, OpenCV's
        own default.
    exposure_weight : float, optional
        Weight for the "well-exposedness" measure. Must be non-negative and
        finite. Default `0.0`, OpenCV's own default -- verified directly.

    Returns
    -------
    np.ndarray
        Same shape as each input image, dtype `float32`. A new, independent
        array; never shares memory with any input image or with the
        `images` container. Nominally display-oriented (most values close
        to ``[0, 1]``), but **not** guaranteed to lie exactly within
        ``[0, 1]`` -- verified directly that the underlying Laplacian-
        pyramid reconstruction can produce a small undershoot below `0` or
        overshoot above `1`. Not clipped or quantized by this function;
        convert explicitly before saving as `uint8` (see the README).

    Raises
    ------
    ValueError
        If `images` has fewer than 2 elements, an element is empty, has a
        shape different from `images[0]`, or an unsupported channel count
        (``(H, W, 1)``, 2-channel, or BGRA), or a weight is negative,
        non-finite, or positive but too small to remain positive once
        converted to OpenCV's `float32` parameter.
    TypeError
        If `images` is not a `Sequence` (rejecting a single array, `str`,
        `bytes`, or a generator/iterator), an element is not a NumPy array,
        an element does not have dtype `uint8`, or a weight is not a real
        number (rejecting `bool`).
    RuntimeError
        If OpenCV's `MergeMertens` does not return a finite `float32` array
        of the expected shape for the given images and weights -- verified
        directly that an extreme but individually representable weight
        (e.g. `contrast_weight=1e10`) can trigger a non-finite result on
        both OpenCV 4.13 and 5.0, with no warning or error from OpenCV
        itself.

    Notes
    -----
    `MergeMertens` uses internal parallel summation; verified directly that
    two independent calls with identical arguments are not always
    bit-for-bit identical (differences on the order of a `float32` rounding
    unit), on both OpenCV 4.13 and 5.0. Do not rely on exact reproducibility
    across separate calls.
    """
    normalized_images = _require_valid_exposure_stack(images)
    contrast_weight = _validated_weight(contrast_weight, "contrast_weight")
    saturation_weight = _validated_weight(saturation_weight, "saturation_weight")
    exposure_weight = _validated_weight(exposure_weight, "exposure_weight")

    merger = cv2.createMergeMertens(contrast_weight, saturation_weight, exposure_weight)
    result = merger.process(normalized_images)
    return _validated_float32_result(result, normalized_images[0].shape, "MergeMertens")


def merge_hdr_debevec(
    images: Sequence[Image],
    exposure_times: Sequence[float],
    *,
    response_curve: ImageFloat32 | None = None,
) -> ImageFloat32:
    """Reconstruct an HDR radiance map from a bracketed exposure stack (Debevec).

    Implemented via OpenCV's `cv2.createMergeDebevec`. Unlike
    `fuse_exposures`, this reconstructs a physical HDR radiance map from the
    images **and** their exposure times, combined via a weighted log-average
    of the (calibrated or default-linear) camera response. The result is a
    raw radiance map, not a display-ready image -- it is not clipped,
    normalized, or tone-mapped by this function.

    Parameters
    ----------
    images : Sequence[np.ndarray]
        A real `Sequence` (e.g. a list or tuple) of at least 2 images -- a
        single `np.ndarray` (including a 4D stack), a `str`/`bytes`, and a
        generator/iterator are all rejected explicitly. **Grayscale input is
        not supported -- only 3-channel BGR ``(H, W, 3)``.** Confirmed
        directly in OpenCV's own C++ source: when `response_curve` is not
        given, `MergeDebevec` builds its default linear response with
        `linearResponse(channels, lutLength)` (correctly sized for the
        image's real channel count), but the very next line unconditionally
        writes through a hardcoded 3-channel (`Vec3f`) accessor regardless
        of that count -- for a genuinely 1-channel (grayscale) response
        array, this reads/writes 3 floats through a buffer that only has 1
        float per element, corrupting memory. This is undefined behavior:
        harmless on some platforms/allocators, a real, reproducible process
        crash on others (observed directly in CI). Every image must be
        non-empty and have exactly the same shape and dtype as `images[0]`
        -- ``(H, W, 1)``, 2-channel, BGRA, and any other channel count are
        all rejected too, with no automatic conversion. dtype must be
        `uint8`, `uint16`, or `float32` -- for `float32`, every value must
        be finite and within ``[0, 1]`` (verified directly that OpenCV
        silently clips values outside this range instead of rejecting
        them). `uint16`/`float32` additionally require an OpenCV build
        that supports them: verified directly that OpenCV `4.9.0` (this
        project's documented minimum) only supports `uint8` for HDR merge,
        raising a raw `CV_Assert` for anything else, while `4.13.0`/`5.0.0`
        support all three -- the exact version this was added is not
        pinned down, so the real capability is probed directly instead
        (see `improcv._compat.opencv.merge_hdr_supports_dtype`), raising a
        clear `ValueError` rather than letting that raw assertion surface
        on an older build. Not modified, nor is the `images` container
        itself.
    exposure_times : Sequence[float]
        Exactly as many values as `images`, paired by index (``images[i]``
        was taken with `exposure_times[i]`) -- no automatic reordering of
        either. Every value must be a finite, strictly positive real number
        (Python or NumPy scalar, `bool` rejected) -- verified directly that
        OpenCV performs no validation on exposure times at all, silently
        accepting zero, negative, non-finite, and duplicate values and
        producing numerically meaningless results. All times must use one
        consistent time unit (conventionally seconds); verified directly
        that uniformly rescaling every time by a constant factor rescales
        the entire output radiance map by the reciprocal of that factor.
        There is no requirement that the values be sorted. Always converted
        to a fresh, contiguous `float32` array before reaching OpenCV --
        verified directly that passing a `float64` array with numerically
        identical values can silently produce a fully non-finite (``inf``)
        result, with no error.
    response_curve : np.ndarray or None, optional
        The camera's inverse response function, as returned by
        `calibrate_camera_response_debevec`/`_robertson`. If `None` (the
        default), **no calibration is performed**: OpenCV uses its own fixed linear
        response instead -- this is a concrete, deterministic fallback, not
        an implicit/unpredictable one. If given, must be `float32`, finite,
        and strictly positive everywhere (its logarithm is used internally;
        verified directly that a zero or negative entry can silently
        corrupt the result rather than raising an error), with shape
        depending on `images[0]`'s dtype: ``(256, 1, 3)`` for `uint8`,
        ``(65536, 1, 3)`` for `uint16`/`float32` -- there is no universal
        shape. Not modified.

    Returns
    -------
    np.ndarray
        Same shape as each input image, dtype `float32`. A new, independent
        array; never shares memory with any input. A raw radiance map --
        typical values range far beyond ``[0, 1]`` (into the hundreds or
        thousands for a realistic bracket) and are not clipped or
        normalized; this function does not verify values are non-negative
        (not enough evidence to reject a small negative value outright).
        Deterministic: unlike `fuse_exposures`, repeated calls with
        identical arguments were verified to be bit-for-bit identical, on
        both OpenCV 4.13 and 5.0.

    Raises
    ------
    ValueError
        If `images` has fewer than 2 elements, an element is empty, has a
        shape or dtype different from `images[0]`, a grayscale or otherwise
        unsupported channel count, (for `float32`) a non-finite or
        out-of-``[0, 1]`` value, or a dtype the installed OpenCV build's HDR
        merge does not support; if `exposure_times` does not have exactly
        `len(images)` values, or a value is non-positive, non-finite, or
        too small to remain positive once converted to `float32`; if
        `response_curve` has the wrong shape for `images[0]`'s
        dtype/channels, contains a non-finite value, or contains a zero or
        negative value.
    TypeError
        If `images`/`exposure_times` is not a `Sequence` (rejecting a
        single array, `str`, `bytes`, or a generator/iterator), an image
        element is not a NumPy array or has an unsupported dtype, an
        exposure time is not a real number (rejecting `bool`), or
        `response_curve` is not a NumPy array or does not have dtype
        `float32`.
    RuntimeError
        If OpenCV does not return a finite `float32` array of the expected
        shape for the given inputs.

    Notes
    -----
    A response curve produced by `calibrate_camera_response_debevec`/
    `_robertson` can be passed via `response_curve`. Structurally, OpenCV
    accepts a curve regardless of which calibration algorithm produced it,
    as long as the shape/dtype match -- but pairing a Robertson-calibrated
    curve here is not recommended: the two calibration algorithms normalize
    their output differently, so cross-pairing remains numerically finite
    but is not verified to be physically meaningful.
    """
    normalized_images = _require_valid_exposure_stack(images, allowed_dtypes=_MERGE_DTYPES)
    if normalized_images[0].ndim == 2:
        raise ValueError(
            "merge_hdr_debevec does not support grayscale input -- verified directly, and "
            "confirmed in OpenCV's own C++ source, that its default (no explicit "
            "response_curve) linear response construction unconditionally writes through a "
            "3-channel accessor regardless of the image's actual channel count, corrupting "
            "memory for a genuinely 1-channel array (undefined behavior: silently harmless "
            "on some platforms, a real process crash observed on others) -- use a 3-channel "
            "BGR stack instead"
        )
    _require_merge_dtype_supported(normalized_images[0], cv2.createMergeDebevec, "MergeDebevec")
    times_array = _require_valid_exposure_times(exposure_times, len(normalized_images))
    if response_curve is not None:
        response_curve = _require_valid_response_curve(
            response_curve, normalized_images[0], method="debevec"
        )

    merger = cv2.createMergeDebevec()
    if response_curve is None:
        result = merger.process(normalized_images, times_array)
    else:
        result = merger.process(normalized_images, times_array, response_curve)
    return _validated_float32_result(result, normalized_images[0].shape, "MergeDebevec")


def merge_hdr_robertson(
    images: Sequence[Image],
    exposure_times: Sequence[float],
    *,
    response_curve: ImageFloat32 | None = None,
) -> ImageFloat32:
    """Reconstruct an HDR radiance map from a bracketed exposure stack (Robertson).

    Implemented via OpenCV's `cv2.createMergeRobertson`. Same purpose and
    image/time contract as `merge_hdr_debevec` -- see its docstring for the
    full parameter contract, which is identical here except for the
    grayscale restriction and `response_curve`'s value rule (both below).
    Robertson's weighted-linear reconstruction is a different algorithm from
    Debevec's weighted-log-average; the two are not guaranteed to produce
    comparable absolute radiance scales for the same input.

    Parameters
    ----------
    images : Sequence[np.ndarray]
        See `merge_hdr_debevec` -- grayscale is unsupported here too
        (neither `merge_hdr_debevec` nor `merge_hdr_robertson` currently
        support grayscale input, for different reasons). Verified directly
        that `cv2.MergeRobertson` raises a raw, unhelpful `cv2.error` for
        any stack that is not 3-channel BGR -- regardless of dtype, and
        regardless of whether `response_curve` is given -- so a grayscale
        stack is rejected here with a clear message before ever reaching
        OpenCV.
    exposure_times : Sequence[float]
        See `merge_hdr_debevec`.
    response_curve : np.ndarray or None, optional
        See `merge_hdr_debevec` for shape/dtype/finiteness requirements.
        Unlike `merge_hdr_debevec`, the curve is used directly (not
        logarithmically), so zero entries are tolerated -- only a negative
        entry, or a curve that is zero everywhere, is rejected. If `None`,
        OpenCV uses its own fixed linear response (no calibration is
        performed).

    Returns
    -------
    np.ndarray
        See `merge_hdr_debevec`. Deterministic: repeated calls with
        identical arguments were verified to be bit-for-bit identical, on
        both OpenCV 4.13 and 5.0.

    Raises
    ------
    ValueError
        As `merge_hdr_debevec`, plus a grayscale `images` stack, and except
        `response_curve`'s value rule: a negative entry, or a curve that is
        zero everywhere, is rejected instead of a non-strictly-positive
        entry.
    TypeError
        See `merge_hdr_debevec`.
    RuntimeError
        See `merge_hdr_debevec`.

    Notes
    -----
    See `merge_hdr_debevec`'s `Notes` on response-curve cross-compatibility.
    """
    normalized_images = _require_valid_exposure_stack(images, allowed_dtypes=_MERGE_DTYPES)
    if normalized_images[0].ndim == 2:
        raise ValueError(
            "merge_hdr_robertson does not support grayscale input -- verified directly "
            "that OpenCV's MergeRobertson raises a raw, unhelpful error for any channel "
            "count other than 3, regardless of dtype -- use a 3-channel BGR stack instead "
            "(merge_hdr_debevec does not support grayscale input either)"
        )
    _require_merge_dtype_supported(normalized_images[0], cv2.createMergeRobertson, "MergeRobertson")
    times_array = _require_valid_exposure_times(exposure_times, len(normalized_images))
    if response_curve is not None:
        response_curve = _require_valid_response_curve(
            response_curve, normalized_images[0], method="robertson"
        )

    merger = cv2.createMergeRobertson()
    if response_curve is None:
        result = merger.process(normalized_images, times_array)
    else:
        result = merger.process(normalized_images, times_array, response_curve)
    return _validated_float32_result(result, normalized_images[0].shape, "MergeRobertson")


def calibrate_camera_response_debevec(
    images: Sequence[ImageU8],
    exposure_times: Sequence[float],
    *,
    samples: int = 70,
    smoothness: float = 10.0,
    random_sampling: bool = False,
) -> ImageFloat32:
    """Estimate a camera's inverse response curve from a bracketed exposure stack (Debevec).

    Implemented via OpenCV's `cv2.createCalibrateDebevec`. The result is
    intended for `merge_hdr_debevec`'s `response_curve` parameter, to
    replace its default fixed linear response with one estimated from the
    actual camera/scene. Calibration is never performed implicitly by any
    merge function -- this must be called explicitly.

    Parameters
    ----------
    images : Sequence[np.ndarray]
        A real `Sequence` (e.g. a list or tuple) of at least 2 images -- a
        single `np.ndarray` (including a 4D stack), a `str`/`bytes`, and a
        generator/iterator are all rejected explicitly. Every image must be
        non-empty and have exactly the same shape and dtype as `images[0]`:
        either 2D grayscale ``(H, W)`` or 3D BGR ``(H, W, 3)`` -- ``(H, W, 1)``,
        2-channel, and BGRA are all rejected, with no automatic conversion.
        dtype must be exactly `uint8` -- verified directly, in OpenCV's own
        C++ source, that `CalibrateDebevec` asserts this unconditionally
        (unlike `merge_hdr_debevec`, which also accepts `uint16`/`float32`).
        Not modified, nor is the `images` container itself.
    exposure_times : Sequence[float]
        Exactly as many values as `images`, paired by index -- see
        `merge_hdr_debevec`'s `exposure_times` for the full contract
        (strictly positive, finite, one consistent time unit, no sorting
        required, always rebuilt into a fresh `float32` array).
    samples : int, optional
        Target number of pixel locations to sample. Must be a positive
        integer (Python or NumPy scalar, `bool` rejected) fitting a signed
        32-bit C++ int. When `random_sampling` is `False` (the default),
        this is only a *target*: OpenCV rounds it to a rectangular grid of
        ``x_points * y_points`` locations via its own formula (see
        `_validated_debevec_samples`), so e.g. `samples=4` and `samples=5`
        can produce the identical grid on the same image -- a `samples`
        value that does not fit any valid grid for `images[0]`'s size is
        rejected before calling OpenCV, with a message explaining the
        rounding. When `random_sampling` is `True`, this bound does not
        apply -- OpenCV samples with replacement, so there is no grid and
        no upper limit tied to the image's pixel count. Default `70`,
        OpenCV's own default.
    smoothness : float, optional
        Smoothness weight for the estimated curve (OpenCV's own parameter
        name is `lambda`) -- larger values produce a smoother, less
        scene-specific curve. Must be strictly positive, finite, and safely
        representable as `float32`. Verified directly that `smoothness=0`
        can produce an `inf`-valued curve, and that `NaN`/`inf` can reach
        OpenCV's internal SVD solver and trigger low-level LAPACK warnings
        rather than a clean error -- both are rejected here before ever
        calling OpenCV, not deferred to the output postcondition. No
        OpenCV-documented upper bound. Default `10.0`, OpenCV's own default.
    random_sampling : bool, optional
        `False` (the default) samples a deterministic rectangular grid;
        `True` samples pixel locations randomly, with replacement, using
        OpenCV's internal `rand()` -- this class exposes no seed parameter,
        so a `True` result is not guaranteed reproducible across calls or
        processes. Must be a genuine `bool` -- `0`/`1`, NumPy integers, and
        other truthy/falsy objects are all rejected, not silently
        interpreted as a boolean.

    Returns
    -------
    np.ndarray
        The estimated inverse camera response, dtype `float32`. Shape
        ``(256, 1)`` for a grayscale `images` stack, ``(256, 1, 3)`` for
        BGR (calibration is always `uint8`-only, so always a 256-entry
        LUT). A BGR curve has a real, public consumer:
        `merge_hdr_debevec`'s `response_curve` parameter. A grayscale
        curve does not -- both `merge_hdr_debevec` and
        `merge_hdr_robertson` are currently BGR-only, so there is no
        existing end-to-end grayscale calibration-to-merge pipeline in
        improcv today; grayscale calibration is still offered on its own
        merits (e.g. inspecting the estimated curve directly), not as
        half of a pipeline. A new, independent array; never shares memory
        with any input. Guaranteed finite and strictly positive -- both are
        enforced as a postcondition, since `merge_hdr_debevec` takes the
        curve's logarithm and cannot use a curve containing a zero or
        negative value. Not guaranteed monotonic -- not enough evidence to
        treat that as a universal contract of every valid input's result,
        though a legitimate curve is expected to be close to it in
        practice.

    Raises
    ------
    ValueError
        If `images` has fewer than 2 elements, an element is empty, has a
        shape or dtype different from `images[0]`, or an unsupported
        channel count; if `exposure_times` does not have exactly
        `len(images)` values, or a value is non-positive, non-finite, or
        too small to remain positive once converted to `float32`; if
        `samples` does not fit a valid sampling grid for `images[0]`'s size
        (`random_sampling=False` only); if `smoothness` is non-positive,
        non-finite, or too small to remain positive once converted to
        `float32`.
    TypeError
        If `images`/`exposure_times` is not a `Sequence` (rejecting a
        single array, `str`, `bytes`, or a generator/iterator), an image
        element is not a NumPy array or does not have dtype `uint8`, an
        exposure time is not a real number (rejecting `bool`), `samples` is
        not an integer (rejecting `bool`), `smoothness` is not a real
        number (rejecting `bool`), or `random_sampling` is not a `bool`.
    RuntimeError
        If OpenCV does not return a finite, strictly positive `float32`
        array of the expected shape for the given inputs -- verified
        directly that a non-finite result is a real, non-hypothetical
        outcome for a degenerate stack (e.g. a single-intensity-level
        image), though `CalibrateDebevec`'s smoothness regularization makes
        this markedly less likely than for
        `calibrate_camera_response_robertson`; whether a specific
        degenerate stack triggers it is not necessarily the same across
        every supported OpenCV version (verified directly for one such
        case: finite on OpenCV 4.13.0/5.0.0, non-finite on 4.9.0, this
        project's documented minimum). Also verified directly, with a
        deterministic counterexample, that OpenCV can return a *finite*
        curve containing exact-zero entries -- `CalibrateDebevec` estimates
        in log-space and then exponentiates, so a very negative but finite
        intermediate value can underflow `float32` to exactly `0.0`; such a
        curve is rejected here rather than passed through to
        `merge_hdr_debevec`, where it would fail less clearly.
    """
    normalized_images = _require_valid_exposure_stack(images)
    times_array = _require_valid_exposure_times(exposure_times, len(normalized_images))
    require_bool(random_sampling, "random_sampling")
    samples = _validated_debevec_samples(samples, normalized_images[0], random_sampling)
    smoothness = _validated_positive_float32(smoothness, "smoothness")

    calibrator = cv2.createCalibrateDebevec(samples, smoothness, random_sampling)
    result = calibrator.process(normalized_images, times_array)
    expected_shape = _expected_response_curve_shape(normalized_images[0])
    return _validated_calibration_result(result, expected_shape, method="debevec")


def calibrate_camera_response_robertson(
    images: Sequence[ImageU8],
    exposure_times: Sequence[float],
    *,
    max_iterations: int = 30,
    threshold: float = 0.01,
) -> ImageFloat32:
    """Estimate a camera's inverse response curve from a bracketed exposure stack (Robertson).

    Implemented via OpenCV's `cv2.createCalibrateRobertson`. Same purpose
    as `calibrate_camera_response_debevec` -- see its docstring for the
    full `images`/`exposure_times` contract, except for one restriction
    (below). This iterative estimation can be markedly more expensive than
    `calibrate_camera_response_debevec`'s closed-form solve, with cost
    scaling with `max_iterations`.

    Parameters
    ----------
    images : Sequence[np.ndarray]
        See `calibrate_camera_response_debevec`, with one additional
        restriction: **grayscale input is not supported**. Verified
        directly that `cv2.CalibrateRobertson` raises a raw, unhelpful
        `cv2.error` for any stack that is not 3-channel BGR -- so a
        grayscale stack is rejected here with a clear message before ever
        reaching OpenCV. Use `calibrate_camera_response_debevec` for a
        grayscale stack instead.
    exposure_times : Sequence[float]
        See `calibrate_camera_response_debevec`.
    max_iterations : int, optional
        Maximum number of Gauss-Seidel solver iterations. Must be a
        positive integer (Python or NumPy scalar, `bool` rejected) fitting
        a signed 32-bit C++ int -- `0` is rejected as a misleading no-op
        (verified directly that it silently returns the untouched initial
        linear response, not an error), not because OpenCV itself would
        reject it. No upper bound. Default `30`, OpenCV's own default.
    threshold : float, optional
        Target difference between successive iterations, below which the
        solver stops early. Must be non-negative, finite, and safely
        representable as `float32`. Unlike `smoothness`, a positive value
        underflowing to `0.0f` is accepted rather than rejected -- `0` is
        itself a legal value here (it just means early stopping is
        effectively disabled, not that zero iterations run). No upper
        bound. Default `0.01`, OpenCV's own default.

    Returns
    -------
    np.ndarray
        The estimated inverse camera response, dtype `float32`, shape
        ``(256, 1, 3)`` (calibration is always `uint8`-only, so always a
        256-entry LUT; grayscale is unsupported, so always 3 channels). A
        new, independent array; never shares memory with any input.
        Guaranteed finite, non-negative, and containing at least one
        strictly positive value -- all enforced as a postcondition, since
        `merge_hdr_robertson`'s own `response_curve` contract rejects a
        negative entry or an all-zero curve. Not guaranteed strictly
        positive throughout (unlike `calibrate_camera_response_debevec`'s
        curve) or monotonic.

    Raises
    ------
    ValueError
        If `images` has fewer than 2 elements, an element is empty, has a
        shape or dtype different from `images[0]`, an unsupported channel
        count, or is grayscale; if `exposure_times` does not have exactly
        `len(images)` values, or a value is non-positive, non-finite, or
        too small to remain positive once converted to `float32`; if
        `max_iterations` is non-positive; if `threshold` is negative,
        non-finite, or too large to represent as `float32`.
    TypeError
        If `images`/`exposure_times` is not a `Sequence` (rejecting a
        single array, `str`, `bytes`, or a generator/iterator), an image
        element is not a NumPy array or does not have dtype `uint8`, an
        exposure time is not a real number (rejecting `bool`),
        `max_iterations` is not an integer (rejecting `bool`), or
        `threshold` is not a real number (rejecting `bool`).
    RuntimeError
        If OpenCV does not return a finite `float32` array of the expected
        shape for the given inputs -- verified directly that this is a
        real, non-hypothetical outcome for otherwise-valid input: OpenCV's
        `CalibrateRobertson` normalizes by, for each of the 256 intensity
        levels, the count of pixels observed at that level across the
        whole stack; a level that never appears anywhere in the stack
        divides by zero and yields `NaN` at that entry. An image with few
        distinct intensity values (e.g. a solid color) can therefore never
        produce a finite curve here, regardless of image size -- confirmed
        identically on OpenCV 4.9.0, 4.13.0, and 5.0.0. This is not
        heuristically rejected before calling OpenCV; it surfaces as this
        `RuntimeError`. Also raised if OpenCV returns a finite curve
        containing a negative entry, or an all-zero curve -- both would
        otherwise be rejected downstream by `merge_hdr_robertson`'s own
        `response_curve` contract.

    Notes
    -----
    Needs a reasonably diverse intensity histogram across the stack to
    produce a finite result at all -- verified directly that an all-black
    or all-white image stack, or one with very few distinct intensity
    values, deterministically raises `RuntimeError` here regardless of
    image size, on every supported OpenCV version. `calibrate_camera_response_debevec`'s
    smoothness regularization makes it markedly more robust to the same
    stacks, but this is not an unconditional guarantee across every
    supported OpenCV version -- verified directly that the exact same
    all-black stack calibrates finitely with `calibrate_camera_response_debevec`
    on OpenCV 4.13.0/5.0.0, but not on OpenCV 4.9.0 (this project's
    documented minimum).
    """
    normalized_images = _require_valid_exposure_stack(images)
    if normalized_images[0].ndim == 2:
        raise ValueError(
            "calibrate_camera_response_robertson does not support grayscale input -- "
            "verified directly that OpenCV's CalibrateRobertson raises a raw, unhelpful "
            "error for any channel count other than 3, regardless of dtype -- use "
            "calibrate_camera_response_debevec for a grayscale stack instead"
        )
    times_array = _require_valid_exposure_times(exposure_times, len(normalized_images))
    max_iterations = _validated_positive_int_param(max_iterations, "max_iterations")
    threshold = _validated_threshold(threshold, "threshold")

    calibrator = cv2.createCalibrateRobertson(max_iterations, threshold)
    result = calibrator.process(normalized_images, times_array)
    expected_shape = _expected_response_curve_shape(normalized_images[0])
    return _validated_calibration_result(result, expected_shape, method="robertson")


def _require_valid_tonemap_hdr(hdr: object) -> ImageFloat32:
    """Raise TypeError/ValueError unless `hdr` is a valid HDR image for tone
    mapping, else return it unchanged (cast to `ImageFloat32`).

    Must be a non-empty, `float32`, 3-channel BGR `(H, W, 3)` array with
    only finite values -- verified directly, in OpenCV's own C++ source,
    that the base `Tonemap` class asserts exactly `CV_32FC3` (2D,
    `float32`, 3 channels), and that `TonemapDrago`/`TonemapReinhard`/
    `TonemapMantiuk` each enforce the identical contract *indirectly*, by
    calling the base `Tonemap` internally as their own first processing
    step -- so grayscale, ``(H, W, 1)``, 2-channel, `uint8`/`uint16`/
    `float64`, and BGRA are all rejected identically for all four
    functions in this module. Negative values are explicitly allowed:
    neither `merge_hdr_debevec` nor `merge_hdr_robertson` guarantees
    non-negative radiance, and verified directly that OpenCV's base linear
    tonemap normalizes purely by min/max regardless of sign, handling
    negative input numerically safely. Not modified, and never copied
    here -- verified directly that every tone-mapping operator accepts a
    non-contiguous, read-only, or Fortran-order array safely.
    """
    if not isinstance(hdr, np.ndarray):
        raise TypeError(f"hdr must be a NumPy array, got {type(hdr).__name__}")
    if hdr.ndim != 3 or hdr.shape[2] != 3:
        if hdr.ndim == 2:
            raise ValueError(
                f"hdr must be 3-channel BGR (H, W, 3), got a 2D grayscale array with shape "
                f"{hdr.shape} -- tone mapping does not support grayscale input"
            )
        if hdr.ndim == 3 and hdr.shape[2] == 1:
            raise ValueError(
                f"hdr must be 3-channel BGR (H, W, 3), got a single-channel image with an "
                f"explicit trailing axis, shape {hdr.shape} -- drop it first with "
                "hdr[..., 0], though tone mapping does not support a genuinely grayscale "
                "image either"
            )
        if hdr.ndim == 3 and hdr.shape[2] == 4:
            raise ValueError(
                f"hdr must be 3-channel BGR (H, W, 3), got a 4-channel (BGRA) image with "
                f"shape {hdr.shape} -- explicitly drop or composite the alpha channel first"
            )
        raise ValueError(f"hdr must have shape (H, W, 3), got {hdr.shape}")
    if hdr.size == 0:
        raise ValueError(f"hdr must not be empty, got shape {hdr.shape}")
    require_dtype(hdr, (np.float32,), "hdr")
    if not np.all(np.isfinite(hdr)):
        raise ValueError("hdr must contain only finite values")
    return cast(ImageFloat32, hdr)


def _validated_ranged_float32(value: object, low: float, high: float, name: str) -> float:
    """Raise TypeError/ValueError unless `value` is a real number within
    ``[low, high]``, else return its `float32` value as a plain `float`.

    Shared by `tone_map_reinhard`'s `intensity`, `light_adaptation`, and
    `color_adaptation`, and by `tone_map_drago`'s `bias` -- all four are
    bounded ranges where both endpoints (including `0`) are legal OpenCV
    values, so unlike `_validated_positive_float32`, a value underflowing
    to `0.0f` is never a contract violation, and overflow to `inf` cannot
    happen once the value is already confirmed within a finite range.
    """
    require_range(value, low, high, name)
    original = float(value)  # type: ignore[arg-type]
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        converted = np.float32(original)
    return float(converted)


def _validated_nonzero_float32(value: object, name: str) -> float:
    """Raise TypeError/ValueError unless `value` is a finite, nonzero real
    number safely representable as OpenCV's `float32`, else return its
    `float32` value as a plain `float`.

    For `tone_map_mantiuk`'s `scale` -- unlike `_validated_positive_float32`,
    both positive and negative values are legal (`TonemapMantiuk`'s own
    `signedPow` explicitly preserves the sign of its contrast values, and
    verified directly that a negative `scale` produces a normal, finite
    result), but zero is not: verified directly that `scale=0`
    deterministically produces a non-finite result (`NaN` or `inf`,
    depending on the OpenCV version) on every supported OpenCV version. A
    tiny nonzero value that underflows to positive or negative `0.0f` is
    rejected the same way, since Python's `==` treats `-0.0` and `0.0` as
    equal, making this check catch both signs of underflow.
    """
    require_finite(value, name)
    if value == 0:  # type: ignore[operator]
        raise ValueError(f"{name} must not be zero, got {value}")
    original = float(value)  # type: ignore[arg-type]
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        converted = np.float32(original)
    if not np.isfinite(converted):
        raise ValueError(
            f"{name} is too large to represent as OpenCV's float32 parameter, got {value}"
        )
    if converted == 0.0:
        raise ValueError(
            f"{name} must not be zero, got {value}, which is too small in magnitude to "
            "remain nonzero once converted to OpenCV's float32 parameter"
        )
    return float(converted)


def _require_mantiuk_min_size(hdr: np.ndarray) -> None:
    """Raise ValueError unless `hdr` is at least ``2x2`` for `tone_map_mantiuk`.

    Verified directly, in OpenCV's own C++ source, that `TonemapMantiuk`
    builds a multiresolution contrast pyramid with
    ``levels = int(log(min(rows, cols)) / log(2))`` levels; when the
    smaller spatial dimension is `1`, this is `0`, leaving the pyramid
    empty and producing a raw, low-level `cv2.error` from an unrelated
    internal matrix operation rather than a clean, attributable one.
    Confirmed identically on OpenCV 4.9.0, 4.13.0, and 5.0.0 for every
    combination of `1xN`/`Nx1`/`1x1` tested. No other tone-mapping function
    in this module has an equivalent minimum-size requirement.
    """
    height, width = hdr.shape[:2]
    if height < _MANTIUK_MIN_DIM or width < _MANTIUK_MIN_DIM:
        raise ValueError(
            f"tone_map_mantiuk requires both height and width to be at least "
            f"{_MANTIUK_MIN_DIM}, got shape {hdr.shape} -- verified directly, in OpenCV's "
            "own C++ source, that its internal contrast pyramid has zero levels when the "
            "smaller spatial dimension is 1, which otherwise surfaces as an unrelated, raw "
            "cv2.error instead of a clear one"
        )


def _require_not_spatially_constant(hdr: np.ndarray, operation: str) -> None:
    """Raise ValueError if every pixel of `hdr` is identical.

    Shared by `tone_map_reinhard` and `tone_map_mantiuk` -- verified
    directly, in OpenCV's own C++ source, that both compute a "key"/
    contrast quantity that divides by ``(log_max - log_min)`` (Reinhard) or
    builds a contrast pyramid that is entirely zero (Mantiuk) for a
    perfectly constant image, in both cases producing a non-finite result
    unconditionally -- verified for every constant value tested (`0`,
    `1e-6`, `1`, `100`, `1e6`), not just black. `tone_map`/`tone_map_drago`
    do not share this restriction: the base linear tonemap has a dedicated
    branch for a constant image (copying the value through unchanged
    rather than dividing by a zero range), and `tone_map_drago` only fails
    for the specific zero-luminance case handled by
    `_require_no_zero_luminance_pixel`.
    """
    if np.all(hdr == hdr[0, 0]):
        raise ValueError(
            f"{operation} does not support a spatially constant hdr image (every pixel "
            f"equal to {tuple(float(c) for c in hdr[0, 0])}) -- verified directly, in "
            "OpenCV's own C++ source, that its internal luminance-key computation divides "
            "by a quantity that is exactly zero for a constant image, producing a "
            "non-finite result unconditionally, regardless of parameters"
        )


def _require_no_zero_luminance_pixel(hdr: np.ndarray, operation: str) -> None:
    """Raise ValueError if OpenCV's base linear tonemap normalization would
    create a zero-luminance pixel in `hdr`.

    Shared by `tone_map_drago` and `tone_map_mantiuk` -- both call
    `hdr_common.cpp`'s `mapLuminance` internally, which divides each pixel's
    channels by that same pixel's own luminance with no protection against
    a zero denominator. Verified directly, in OpenCV's own C++ source and
    empirically (identically on OpenCV 4.9.0, 4.13.0, and 5.0.0): both
    operators first run `hdr` through the base linear tonemap, which
    normalizes the image's global minimum value to exactly `0.0` (when
    ``global_max - global_min > DBL_EPSILON``) or copies `hdr` through
    entirely unchanged, with no normalization at all, otherwise. A pixel
    becomes a true `(0, 0, 0)` black pixel -- zero luminance -- either when
    all three of its channels already equal `hdr`'s global minimum (in the
    normalizing case, since that channel's value maps to exactly `0.0`),
    or, in the copy-through case, when *any* pixel anywhere in `hdr` is
    itself exactly `(0, 0, 0)` -- not only `hdr[0, 0]`: OpenCV's
    ``max - min <= DBL_EPSILON`` check does not mean every pixel is
    bit-identical to `hdr[0, 0]`, only that they are all within
    `DBL_EPSILON` of each other, so a zero-luminance pixel elsewhere in an
    otherwise near-constant (but not exactly constant) `hdr` was
    previously missed entirely. Confirmed with a deterministic
    counterexample: ``hdr = np.full((4, 4, 3), 1e-20,
    dtype=np.float32); hdr[2, 2] = 0.0`` has `hdr[0, 0] == 1e-20`
    (nonzero) but a true zero-luminance pixel at `(2, 2)` -- the previous,
    single-pixel check let this reach OpenCV, which produced `NaN`
    (`TonemapDrago`) or a raw solver `cv2.error` (`TonemapMantiuk`),
    surfacing only as this module's own `RuntimeError` postcondition
    instead of being caught here first. Either way (normalizing or
    copy-through), a zero-luminance pixel produces a `NaN` at exactly that
    pixel, **independent of `bias`/`saturation`/`scale`** -- confirmed
    directly that every tested `bias` value (`-0.5` through `5.0`)
    reproduces the same `NaN` on the same zero-luminance pixel. Verified
    directly that a plain "does `hdr` contain a `(0, 0, 0)` pixel" test
    would be wrong for the *normalizing* case: with a negative global
    minimum, an already-`(0, 0, 0)` pixel in `hdr` does not necessarily
    end up at the global minimum after normalization, and so does not
    necessarily become zero-luminance.
    """
    global_min = float(hdr.min())
    global_max = float(hdr.max())
    if global_max - global_min > np.finfo(np.float64).eps:
        zero_luminance_pixel_exists = bool(np.any(np.all(hdr == global_min, axis=-1)))
    else:
        zero_luminance_pixel_exists = bool(np.any(np.all(hdr == 0.0, axis=-1)))
    if zero_luminance_pixel_exists:
        raise ValueError(
            f"{operation} would divide by zero luminance for this hdr image -- verified "
            "directly, in OpenCV's own C++ source, that its internal per-pixel luminance "
            "normalization divides each channel by that same pixel's own luminance with no "
            "protection against zero; OpenCV's base linear tonemap normalizes hdr's global "
            "minimum value to exactly 0.0 first (or leaves a spatially constant hdr "
            "unchanged), so a pixel whose three channels are all already at that global "
            "minimum (or a constant hdr that is itself exactly (0, 0, 0)) becomes a true "
            "zero-luminance black pixel, producing a NaN at exactly that pixel regardless "
            "of parameters"
        )


def _run_tonemap(factory: Callable[[], cv2.Tonemap], hdr: np.ndarray, operation: str) -> object:
    """Call ``factory().process(hdr)``, converting any raw `cv2.error` into a
    `RuntimeError`.

    Shared by all four tone-mapping functions, always given a freshly
    created operator object -- verified directly, in OpenCV's own C++
    source, that `TonemapReinhard.process()` mutates its own `intensity`
    member in place (`intensity = exp(-intensity)`), so reusing the same
    object across calls silently changes its behavior on every subsequent
    call; a fresh object per call neutralizes this. Also verified directly
    that each of the four OpenCV Tonemap classes can raise a raw, low-level
    `cv2.error` for a specific value combination that still passes every
    documented parameter/shape contract (e.g. an internal conjugate-
    gradient solver assertion in `TonemapMantiuk`, or a `max > 0` assertion
    in `TonemapDrago`) -- improcv only ever raises its own `ValueError`/
    `TypeError` for a validation failure, so any `cv2.error` surviving that
    validation is converted into a `RuntimeError` here instead of leaking
    as a raw OpenCV error. The original exception is preserved as
    `__cause__`.
    """
    try:
        operator = factory()
        return operator.process(hdr)
    except cv2.error as exc:
        raise RuntimeError(
            f"{operation} failed inside OpenCV for an HDR image and parameters that passed "
            "improcv's own validation"
        ) from exc


def tone_map(
    hdr: ImageFloat32,
    *,
    gamma: float = 1.0,
) -> ImageFloat32:
    """Apply a simple linear tone-mapping curve with gamma correction.

    Implemented via OpenCV's `cv2.createTonemap`. Normalizes `hdr` by its
    global min/max (pooled across all 3 channels together, not per-channel
    -- verified directly), then applies gamma correction. The other three
    tone-mapping functions in this module (`tone_map_drago`,
    `tone_map_reinhard`, `tone_map_mantiuk`) each call this exact operation
    internally as their own first processing step, with `gamma=1.0`
    (deferring gamma correction to their own final step) -- so this
    function's `hdr` contract is identical to theirs, and is the contract
    every tone-mapping function in this module enforces.

    Parameters
    ----------
    hdr : np.ndarray
        A non-empty `float32` array with shape ``(H, W, 3)`` (BGR). Values
        may be negative -- neither `merge_hdr_debevec` nor
        `merge_hdr_robertson` guarantees non-negative radiance, and
        verified directly that OpenCV's min/max normalization handles
        negative input numerically safely. All values must be finite.
        Grayscale, ``(H, W, 1)``, 2-channel, BGRA, and any dtype other than
        `float32` are all rejected, with no automatic conversion. Not
        modified; no defensive copy is made (verified directly that a
        non-contiguous, read-only, or Fortran-order array is handled
        safely).
    gamma : float, optional
        Gamma-correction exponent (the output is raised to ``1/gamma``).
        Must be a real number (Python or NumPy scalar, `bool` rejected),
        strictly positive, finite, and safely representable as `float32`.
        Verified directly that `gamma<=0` does not raise inside OpenCV, but
        produces a meaningless result (`gamma=0` reliably introduces a
        `NaN`, via `pow` applied to a tiny negative floating-point
        rounding artifact that even a well-behaved normalized image can
        contain; negative `gamma` silently produces enormous,
        non-physical values without ever becoming non-finite) -- both are
        rejected here rather than deferred to the output postcondition.
        No OpenCV-documented upper bound. Default `1.0`, OpenCV's own
        default (no correction).

    Returns
    -------
    np.ndarray
        Same shape as `hdr`, dtype `float32`. A new, independent array;
        never shares memory with `hdr`. **Not guaranteed to lie within
        ``[0, 1]``**, despite OpenCV's own documentation claiming that
        range unconditionally: verified directly, in OpenCV's own C++
        source, that a spatially constant `hdr` (``max - min`` within
        `DBL_EPSILON`) takes a dedicated branch that copies the input
        through unchanged rather than normalizing it, so e.g. a constant
        `hdr` of `1e6` produces a constant output of `1e6`, not `1.0`.
        Deterministic: verified directly that two calls with identical
        arguments are bit-for-bit identical, on OpenCV 4.9.0, 4.13.0, and
        5.0.0.

    Raises
    ------
    ValueError
        If `hdr` has an unsupported shape or channel count, is empty, or
        contains a non-finite value; if `gamma` is non-positive,
        non-finite, or too small/large to remain nonzero/finite once
        converted to `float32`.
    TypeError
        If `hdr` is not a NumPy array or does not have dtype `float32`; if
        `gamma` is not a real number (rejecting `bool`).
    RuntimeError
        If OpenCV's `Tonemap` does not return a finite `float32` array of
        `hdr`'s shape for the given inputs, or if it raises a raw
        `cv2.error` despite `hdr`/`gamma` passing improcv's own validation.
        **A finite result is not unconditionally guaranteed even for
        well-formed, non-degenerate `hdr`** whenever `1/gamma` is not an
        exact integer (`gamma != 1.0`, or more generally any `gamma` whose
        reciprocal has a fractional part): verified directly that OpenCV's
        min/max normalization can leave the pixel at `hdr`'s global
        minimum very slightly negative -- an ordinary floating-point
        rounding artifact, not a data problem -- and raising a negative
        base to a non-integer power is mathematically undefined,
        producing `NaN` at exactly that pixel. Whether this actually
        happens is CPU-architecture/SIMD-dispatch-dependent, not just
        data-dependent: confirmed directly that the identical seed and
        `gamma` value that tone-map finitely on Apple Silicon produced a
        non-finite result (caught cleanly by this `RuntimeError`) on
        x86_64 CI (both Linux and Windows).

    Notes
    -----
    This does not clip or quantize its output -- convert explicitly before
    saving as `uint8` (e.g. ``np.round(np.clip(result, 0.0, 1.0) *
    255.0).astype(np.uint8)``; see the README).
    """
    hdr = _require_valid_tonemap_hdr(hdr)
    gamma = _validated_positive_float32(gamma, "gamma")

    result = _run_tonemap(lambda: cv2.createTonemap(gamma), hdr, "Tonemap")
    return _validated_float32_result(result, hdr.shape, "Tonemap")


def tone_map_drago(
    hdr: ImageFloat32,
    *,
    gamma: float = 1.0,
    saturation: float = 1.0,
    bias: float = 0.85,
) -> ImageFloat32:
    """Apply Drago's adaptive logarithmic tone-mapping operator.

    Implemented via OpenCV's `cv2.createTonemapDrago`. Internally, this
    first runs `hdr` through the same linear normalization as `tone_map`
    (with `gamma=1.0`), then compresses the resulting luminance
    logarithmically (controlled by `bias`) before reapplying color via
    `saturation` and gamma correction (controlled by `gamma`).

    Parameters
    ----------
    hdr : np.ndarray
        See `tone_map`. Additionally, `hdr` must not contain a
        zero-luminance pixel once run through the same linear
        normalization `tone_map` performs -- see `Raises` below.
    gamma : float, optional
        See `tone_map`.
    saturation : float, optional
        Positive saturation enhancement; `1.0` preserves saturation,
        greater than `1.0` increases it, less than `1.0` decreases it.
        Must be a real number, strictly positive, finite, and safely
        representable as `float32` -- verified directly that an extreme
        value (e.g. `1e6`) can produce a partially non-finite result. No
        OpenCV-documented upper bound. Default `1.0`, OpenCV's own
        default.
    bias : float, optional
        Bias for Drago's logarithmic base function. Must be a real number
        within OpenCV's own documented ``[0, 1]`` range -- rejected
        outside it, even though this specific build was empirically
        observed to still return a finite result for some out-of-range
        values on the test images used during development; that is not
        treated as a stable, version-independent guarantee. `0` and `1`
        are both legal, including a positive value that underflows to
        `0.0f` once converted to `float32` (`0` is itself a legal
        endpoint, not a hidden contract violation). Default `0.85`,
        OpenCV's own default.

    Returns
    -------
    np.ndarray
        Same shape as `hdr`, dtype `float32`. A new, independent array;
        never shares memory with `hdr`. **Not guaranteed to lie within
        ``[0, 1]``** -- verified directly that a spatially constant,
        non-zero-luminance `hdr` (e.g. a constant positive color) produces
        an output scaled well outside ``[0, 1]`` (e.g. a constant `hdr` of
        `1e6` produces a constant output around `3 * 10**5`). Deterministic:
        verified directly that two calls with identical arguments are
        bit-for-bit identical, on OpenCV 4.9.0, 4.13.0, and 5.0.0.

    Raises
    ------
    ValueError
        As `tone_map`, plus: if `saturation` is non-positive, non-finite,
        or too small/large to remain nonzero/finite once converted to
        `float32`; if `bias` is outside ``[0, 1]``; if `hdr`, once run
        through OpenCV's base linear normalization, would contain a pixel
        with all three channels at exactly zero (a true zero-luminance
        black pixel) -- verified directly, in OpenCV's own C++ source,
        that `TonemapDrago` divides each pixel's channels by that same
        pixel's own luminance with no protection against a zero
        denominator, and confirmed empirically that this reproduces
        identically regardless of `bias`/`saturation`. This is a common,
        not merely hypothetical, case: any `hdr` whose darkest pixel
        (across all three channels combined) is a true black pixel
        triggers it, including a large flat background or shadow region
        at the image's darkest level.
    TypeError
        As `tone_map`, plus: if `saturation`/`bias` is not a real number
        (rejecting `bool`).
    RuntimeError
        As `tone_map`'s `gamma`-driven case. `bias` carries an
        independent, additional version of the same risk: `TonemapDrago`
        raises each pixel's normalized luminance to an exponent derived
        from `bias` (`log(bias) / log(0.5)`), which is only guaranteed an
        integer at `bias=0.5` -- any other `bias` value shares `gamma`'s
        architecture-dependent exposure to a `NaN` from a slightly
        negative floating-point base, independent of whatever `gamma`
        itself is set to.
    """
    hdr = _require_valid_tonemap_hdr(hdr)
    saturation = _validated_positive_float32(saturation, "saturation")
    bias = _validated_ranged_float32(bias, 0.0, 1.0, "bias")
    gamma = _validated_positive_float32(gamma, "gamma")
    _require_no_zero_luminance_pixel(hdr, "tone_map_drago")

    result = _run_tonemap(
        lambda: cv2.createTonemapDrago(gamma, saturation, bias), hdr, "TonemapDrago"
    )
    return _validated_float32_result(result, hdr.shape, "TonemapDrago")


def tone_map_reinhard(
    hdr: ImageFloat32,
    *,
    gamma: float = 1.0,
    intensity: float = 0.0,
    light_adaptation: float = 1.0,
    color_adaptation: float = 0.0,
) -> ImageFloat32:
    """Apply Reinhard's photographic tone-mapping operator.

    Implemented via OpenCV's `cv2.createTonemapReinhard`. Internally
    computes a local adaptation luminance for each pixel, blended between a
    purely per-pixel value and a global scene value by `light_adaptation`,
    and between per-channel and shared-across-channels statistics by
    `color_adaptation`, then compresses each channel against that
    adaptation level before reapplying gamma correction.

    OpenCV's own C++ API names the third and fourth parameters
    `light_adapt`/`color_adapt` (matching its own getter/setter method
    names, `getLightAdaptation`/`getColorAdaptation`); this wrapper spells
    them out in full since the mapping is unambiguous.

    Parameters
    ----------
    hdr : np.ndarray
        See `tone_map`. Additionally, `hdr` must not be spatially
        constant -- see `Raises` below.
    gamma : float, optional
        See `tone_map`.
    intensity : float, optional
        Result intensity; greater values produce a brighter result. Must
        be a real number within OpenCV's own documented ``[-8, 8]``
        range -- rejected outside it, even though this specific build was
        empirically observed to still return a finite result beyond that
        range on the test images used during development; that is not
        treated as a stable, version-independent guarantee. Default `0.0`,
        OpenCV's own default.
    light_adaptation : float, optional
        Light adaptation weight (OpenCV's own parameter name:
        `light_adapt`). Must be a real number within ``[0, 1]``: `1.0`
        adapts purely per-pixel, `0.0` purely globally, anything between
        is a weighted blend. Verified directly that a value outside
        ``[0, 1]`` is a genuine mathematical extrapolation of this blend
        and can produce a partially non-finite result. Default `1.0`,
        OpenCV's own default.
    color_adaptation : float, optional
        Chromatic adaptation weight (OpenCV's own parameter name:
        `color_adapt`). Must be a real number within ``[0, 1]``: `1.0`
        treats channels independently, `0.0` gives every channel the same
        adaptation level. Verified directly that a value outside
        ``[0, 1]`` can produce a partially non-finite result, the same as
        `light_adaptation`. Default `0.0`, OpenCV's own default.

    Returns
    -------
    np.ndarray
        Same shape as `hdr`, dtype `float32`. A new, independent array;
        never shares memory with `hdr`. Not guaranteed to lie within
        ``[0, 1]`` (see `tone_map`'s `Returns`), though no case producing a
        large excursion was found for this operator specifically.
        Deterministic **as called by this function**: verified directly
        that a fresh `cv2.TonemapReinhard` object is created on every call
        (see `Notes`), so two calls with identical arguments are
        bit-for-bit identical, on OpenCV 4.9.0, 4.13.0, and 5.0.0.

    Raises
    ------
    ValueError
        As `tone_map`, plus: if `intensity` is outside ``[-8, 8]`,
        `light_adaptation`/`color_adaptation` is outside ``[0, 1]``; if
        `hdr` is spatially constant (every pixel identical, including a
        constant non-black color) -- verified directly, in OpenCV's own
        C++ source, that `TonemapReinhard` computes a luminance "key" by
        dividing by ``(log_max - log_min)``, which is exactly zero for a
        constant image, producing a non-finite result unconditionally.
    TypeError
        As `tone_map`, plus: if `intensity`/`light_adaptation`/
        `color_adaptation` is not a real number (rejecting `bool`).
    RuntimeError
        As `tone_map`'s `gamma`-driven case, applied to this function's own
        final gamma-correction step. `TonemapReinhard`'s core computation
        also raises values to two further exponents unrelated to `gamma`
        (a fixed `1.4` power, and a data-derived "key" exponent) that are
        not part of this function's public parameters and cannot be tuned
        away -- a `NaN` from either is likewise architecture-dependent and
        surfaces as this same `RuntimeError`.

    Notes
    -----
    Verified directly, in OpenCV's own C++ source, that
    `cv2.TonemapReinhard.process()` mutates its own object's `intensity`
    field in place (``intensity = exp(-intensity)``) as a side effect --
    confirmed empirically that calling `process()` repeatedly on the same
    object silently changes its result on every call (`intensity` never
    stabilizes; it keeps oscillating). This function always constructs a
    fresh `cv2.TonemapReinhard` object for every call, which fully
    neutralizes this: never cache or reuse a `Tonemap` object across
    calls.
    """
    hdr = _require_valid_tonemap_hdr(hdr)
    gamma = _validated_positive_float32(gamma, "gamma")
    intensity = _validated_ranged_float32(intensity, -8.0, 8.0, "intensity")
    light_adaptation = _validated_ranged_float32(light_adaptation, 0.0, 1.0, "light_adaptation")
    color_adaptation = _validated_ranged_float32(color_adaptation, 0.0, 1.0, "color_adaptation")
    _require_not_spatially_constant(hdr, "tone_map_reinhard")

    result = _run_tonemap(
        lambda: cv2.createTonemapReinhard(gamma, intensity, light_adaptation, color_adaptation),
        hdr,
        "TonemapReinhard",
    )
    return _validated_float32_result(result, hdr.shape, "TonemapReinhard")


def tone_map_mantiuk(
    hdr: ImageFloat32,
    *,
    gamma: float = 1.0,
    scale: float = 0.7,
    saturation: float = 1.0,
) -> ImageFloat32:
    """Apply Mantiuk's contrast-mapping tone-mapping operator.

    Implemented via OpenCV's `cv2.createTonemapMantiuk`. Internally builds
    a multiresolution contrast pyramid of `hdr`'s log-luminance, compresses
    each level's contrast (scaled by `scale`), then reconstructs the image
    via an iterative conjugate-gradient solver before reapplying color via
    `saturation` and gamma correction. Markedly more expensive than the
    other three tone-mapping functions in this module -- measured roughly
    5-15x the cost of `tone_map_reinhard`/`tone_map_drago` and 20-100x the
    cost of `tone_map`, for the same image size (exact multipliers vary by
    OpenCV version and are not a contract).

    Parameters
    ----------
    hdr : np.ndarray
        See `tone_map`. Additionally: both `hdr.shape[0]` and
        `hdr.shape[1]` must be at least `2`, `hdr` must not be spatially
        constant, and `hdr` must not contain a zero-luminance pixel once
        run through OpenCV's base linear normalization -- see `Raises`
        below.
    gamma : float, optional
        See `tone_map`.
    scale : float, optional
        Contrast scale factor; the perceptual contrast response is
        multiplied by this value, compressing dynamic range. Must be a
        real number, finite, safely representable as `float32`, and
        nonzero -- both positive and negative values are legal (verified
        directly that `TonemapMantiuk`'s internal `signedPow` explicitly
        preserves sign, so a negative `scale` produces a normal, finite,
        if visually inverted-contrast, result), but `scale=0`
        deterministically produces a non-finite result on every supported
        OpenCV version, so it is rejected here along with any nonzero
        value that underflows to positive or negative `0.0f` once
        converted to `float32`. No OpenCV-documented upper bound, though
        an extreme value (e.g. `10.0`) was observed to produce a partially
        non-finite result. Default `0.7`, OpenCV's own default.
    saturation : float, optional
        See `tone_map_drago`'s `saturation`.

    Returns
    -------
    np.ndarray
        Same shape as `hdr`, dtype `float32`. A new, independent array;
        never shares memory with `hdr`. Not guaranteed to lie within
        ``[0, 1]`` (see `tone_map`'s `Returns`). Deterministic: verified
        directly that two calls with identical arguments are bit-for-bit
        identical, on OpenCV 4.9.0, 4.13.0, and 5.0.0.

    Raises
    ------
    ValueError
        As `tone_map`, plus: if `scale` is zero, non-finite, or too
        small/large in magnitude to remain nonzero/finite once converted
        to `float32`; if `saturation` is non-positive, non-finite, or too
        small/large to remain nonzero/finite once converted to `float32`;
        if `hdr.shape[0]` or `hdr.shape[1]` is `1` -- verified directly, in
        OpenCV's own C++ source, that `TonemapMantiuk`'s contrast pyramid
        has zero levels in that case, otherwise surfacing as an unrelated,
        raw `cv2.error`; if `hdr` is spatially constant, or would contain a
        zero-luminance pixel once run through OpenCV's base linear
        normalization -- see `tone_map_drago`'s `Raises` for the latter
        (the same underlying mechanism; `TonemapMantiuk` shares OpenCV's
        `mapLuminance` helper with `TonemapDrago`).
    TypeError
        As `tone_map`, plus: if `scale`/`saturation` is not a real number
        (rejecting `bool`).
    RuntimeError
        As `tone_map`'s `gamma`-driven case, applied to this function's own
        final gamma-correction step. `TonemapMantiuk`'s contrast-mapping
        itself explicitly preserves sign via its own `signedPow` (verified
        directly that a negative `scale` does not trigger this), so `scale`
        does not carry the same risk `bias` does for `tone_map_drago`.
    """
    hdr = _require_valid_tonemap_hdr(hdr)
    _require_mantiuk_min_size(hdr)
    gamma = _validated_positive_float32(gamma, "gamma")
    scale = _validated_nonzero_float32(scale, "scale")
    saturation = _validated_positive_float32(saturation, "saturation")
    _require_not_spatially_constant(hdr, "tone_map_mantiuk")
    _require_no_zero_luminance_pixel(hdr, "tone_map_mantiuk")

    result = _run_tonemap(
        lambda: cv2.createTonemapMantiuk(gamma, scale, saturation), hdr, "TonemapMantiuk"
    )
    return _validated_float32_result(result, hdr.shape, "TonemapMantiuk")
