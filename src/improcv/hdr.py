"""HDR-related operations: exposure fusion, radiance merging, and tone mapping."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal, cast

import cv2
import numpy as np

from improcv._compat.opencv import merge_hdr_supports_dtype
from improcv._validation import require_dtype, require_non_negative, require_positive
from improcv.types import Image, ImageFloat32, ImageU8

__all__ = [
    "fuse_exposures",
    "merge_hdr_debevec",
    "merge_hdr_robertson",
]

_MIN_EXPOSURES = 2
_LDR_LUT_LENGTH = 256
_HDR_LUT_LENGTH = 65536
_MERGE_DTYPES = (np.uint8, np.uint16, np.float32)


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


def _validated_exposure_time(value: object, index: int) -> float:
    """Raise TypeError/ValueError unless `value` is a valid single exposure
    time, else return its `float32` value as a plain `float`.

    Mirrors `_validated_weight`'s `float32`-safety pattern, but requires
    strictly positive (not just non-negative) -- verified directly that
    OpenCV performs no validation on exposure times at all: zero, negative,
    `NaN`, and `inf` are all silently accepted, producing numerically
    meaningless (though not always non-finite) results.
    """
    name = f"exposure_times[{index}]"
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


def _require_valid_response_curve(
    response_curve: object, image: np.ndarray, *, method: Literal["debevec", "robertson"]
) -> np.ndarray:
    """Raise ValueError/TypeError unless `response_curve` is a valid response
    curve for `image`'s dtype and channel count, else return it unchanged.

    Shape depends on `image`'s dtype (``256`` LUT entries for `uint8`,
    ``65536`` for `uint16`/`float32`) and channel count (``(length, 1)`` for
    grayscale, ``(length, 1, 3)`` for BGR). Neither continuity nor
    writeability is required -- verified directly that OpenCV safely
    accepts a non-contiguous or read-only curve without mutating it, so no
    defensive copy is made here.

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

    lut_length = _lut_length_for_dtype(image.dtype)
    channels = 1 if image.ndim == 2 else image.shape[2]
    expected_shape = (lut_length, 1) if channels == 1 else (lut_length, 1, channels)
    if response_curve.shape != expected_shape:
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
        generator/iterator are all rejected explicitly. Every image must be
        non-empty and have exactly the same shape and dtype as `images[0]`:
        either 2D grayscale ``(H, W)`` or 3D BGR ``(H, W, 3)`` -- ``(H, W, 1)``,
        2-channel, and BGRA are all rejected, with no automatic conversion,
        and mixing grayscale/BGR or mixing dtypes within one stack is
        rejected. dtype must be `uint8`, `uint16`, or `float32` -- for
        `float32`, every value must be finite and within ``[0, 1]``
        (verified directly that OpenCV silently clips values outside this
        range instead of rejecting them). `uint16`/`float32` additionally
        require an OpenCV build that supports them: verified directly that
        OpenCV `4.9.0` (this project's documented minimum) only supports
        `uint8` for HDR merge, raising a raw `CV_Assert` for anything else,
        while `4.13.0`/`5.0.0` support all three -- the exact version this
        was added is not pinned down, so the real capability is probed
        directly instead (see `improcv._compat.opencv.merge_hdr_supports_dtype`),
        raising a clear `ValueError` rather than letting that raw assertion
        surface on an older build. Not modified, nor is the `images`
        container itself.
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
        The camera's inverse response function, as returned by a future
        camera-response calibration step. If `None` (the default), **no
        calibration is performed**: OpenCV uses its own fixed linear
        response instead -- this is a concrete, deterministic fallback, not
        an implicit/unpredictable one. If given, must be `float32`, finite,
        and strictly positive everywhere (its logarithm is used internally;
        verified directly that a zero or negative entry can silently
        corrupt the result rather than raising an error), with shape
        depending on `images[0]`'s dtype and channel count: ``(256, 1)``/
        ``(256, 1, 3)`` for `uint8` (grayscale/BGR), ``(65536, 1)``/
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
        shape or dtype different from `images[0]`, an unsupported channel
        count, (for `float32`) a non-finite or out-of-``[0, 1]`` value, or a
        dtype the installed OpenCV build's HDR merge does not support; if
        `exposure_times` does not have exactly `len(images)` values, or a
        value is non-positive, non-finite, or too small to remain positive
        once converted to `float32`; if `response_curve` has the wrong
        shape for `images[0]`'s dtype/channels, contains a non-finite value,
        or contains a zero or negative value.
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
    A response curve produced by camera-response calibration (a future,
    separate function) can be passed via `response_curve`. Structurally,
    OpenCV accepts a curve regardless of which calibration algorithm
    produced it, as long as the shape/dtype match -- but pairing a
    Robertson-calibrated curve here is not recommended: the two calibration
    algorithms normalize their output differently, so cross-pairing remains
    numerically finite but is not verified to be physically meaningful.
    """
    normalized_images = _require_valid_exposure_stack(images, allowed_dtypes=_MERGE_DTYPES)
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
        See `merge_hdr_debevec`, with one additional restriction: **grayscale
        input is not supported**. Verified directly that `cv2.MergeRobertson`
        raises a raw, unhelpful `cv2.error` for any stack that is not
        3-channel BGR -- regardless of dtype, and regardless of whether
        `response_curve` is given -- so a grayscale stack is rejected here
        with a clear message before ever reaching OpenCV. Use
        `merge_hdr_debevec` for a grayscale stack instead.
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
            "count other than 3, regardless of dtype -- use merge_hdr_debevec for a "
            "grayscale stack instead"
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
