"""Image analysis: histograms, moments, template matching, and pixel statistics."""

from __future__ import annotations

from typing import Literal, NamedTuple, cast

import cv2
import numpy as np
import numpy.typing as npt

from improcv._compat.opencv import _normalize_calc_hist_output
from improcv._validation import (
    require_bool,
    require_channel_count,
    require_dtype,
    require_finite,
    require_image_ndim,
    require_integral,
    require_one_of,
    require_positive_integral,
    require_spatial_mask,
)
from improcv.contours import Contour, _require_contour
from improcv.types import Image, Mask

__all__ = [
    "histogram",
    "moments",
    "match_template",
    "min_max_loc",
    "mean_stddev",
    "Moments",
    "TemplateMatchMethod",
    "MinMaxResult",
    "MeanStdDevResult",
]

_HISTOGRAM_DTYPES = (np.uint8, np.uint16, np.float32)
_HISTOGRAM_MAX_CHANNELS = 128


def _require_value_range(value: object, name: str = "value_range") -> tuple[float, float]:
    """Raise ValueError unless `value` is a 2-tuple of finite reals with low < high."""
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{name} must be a 2-tuple, got {value!r}")
    low, high = value
    require_finite(low, f"{name}[0]")
    require_finite(high, f"{name}[1]")
    if not low < high:
        raise ValueError(f"{name} must have low < high, got ({low}, {high})")
    return (float(low), float(high))


def histogram(
    image: Image,
    channel: int = 0,
    bins: int = 256,
    value_range: tuple[float, float] = (0.0, 256.0),
    mask: Mask | None = None,
) -> npt.NDArray[np.float32]:
    """Compute the intensity histogram of a single selected channel.

    `image` may have one or more channels; exactly one channel, selected by
    `channel`, is used -- this is not a general wrapper around OpenCV's
    multi-dimensional/multi-channel `calcHist` API.

    Parameters
    ----------
    image : np.ndarray
        Input image with shape ``(H, W)`` or ``(H, W, C)``, dtype ``uint8``,
        ``uint16``, or ``float32``, 1-128 channels. The 128-channel ceiling
        is a common, cross-version-safe limit: verified directly that
        selecting a channel index near the top of a very-high-channel-count
        image works on OpenCV 4.13 well beyond 128 channels, but raises a
        raw ``cv2.error`` on OpenCV 5.0 above 128 -- so 128 is the largest
        count both bindings handle identically.
    channel : int, default 0
        Index of the channel to compute the histogram over, in
        ``[0, image_channel_count)``. Accepts any `numbers.Integral`
        (including NumPy integer scalars), not just plain `int`.
    bins : int, default 256
        Number of histogram bins. Must be positive -- verified directly
        that ``cv2.calcHist`` silently returns ``None`` for ``bins=0``
        rather than raising, so this is checked before the underlying call.
    value_range : tuple of float, default (0.0, 256.0)
        ``(low, high)`` value range, with ``low < high``. The lower bound is
        **inclusive**, the upper bound is **exclusive** -- verified
        directly. A pixel value outside this range entirely is simply not
        counted (no error, no clipping bin).
    mask : np.ndarray or None, default None
        Optional ``uint8`` mask, shape ``(H, W)`` matching `image`'s spatial
        size regardless of `image`'s own channel count or dtype. Any
        nonzero value marks a selected pixel.

    Returns
    -------
    np.ndarray
        Shape ``(bins,)``, dtype ``float32``. A new array; `image`/`mask`
        are never modified.

    Raises
    ------
    ValueError
        If `image` is not 2D/3D or is empty, does not have 1-128 channels,
        `channel` is out of range, `bins` is not positive, `value_range`
        is not a 2-tuple, does not have `low < high`, or contains a
        non-finite value, or `mask` does not match `image`'s spatial size.
    TypeError
        If `image` does not have dtype ``uint8``/``uint16``/``float32``,
        `channel`/`bins` is not `numbers.Integral` (rejecting `bool` and
        `float`), or `mask` does not have dtype ``uint8``.
    """
    require_image_ndim(image, ndims=(2, 3))
    require_dtype(image, _HISTOGRAM_DTYPES)
    num_channels = require_channel_count(image, 1, _HISTOGRAM_MAX_CHANNELS)
    require_integral(channel, "channel")
    channel_int = int(channel)
    if not (0 <= channel_int < num_channels):
        raise ValueError(f"channel must be in [0, {num_channels}), got {channel_int}")
    require_positive_integral(bins, "bins")
    bins_int = int(bins)
    low, high = _require_value_range(value_range)
    if mask is not None:
        require_spatial_mask(mask, image)

    raw = cv2.calcHist([image], [channel_int], mask, [bins_int], [float(low), float(high)])
    return _normalize_calc_hist_output(raw, bins_int)


class Moments(NamedTuple):
    """Image or spatial moments, matching ``cv2.moments()``'s 24 returned fields exactly."""

    m00: float
    m10: float
    m01: float
    m20: float
    m11: float
    m02: float
    m30: float
    m21: float
    m12: float
    m03: float
    mu20: float
    mu11: float
    mu02: float
    mu30: float
    mu21: float
    mu12: float
    mu03: float
    nu20: float
    nu11: float
    nu02: float
    nu30: float
    nu21: float
    nu12: float
    nu03: float


_MOMENTS_RASTER_DTYPES = (np.uint8, np.uint16, np.int16, np.float32, np.float64)


def _moments_from_dict(raw: dict[str, float]) -> Moments:
    """Build a `Moments` from `cv2.moments()`'s raw dict, by explicit key name.

    Never `Moments(*raw.values())` or `Moments(**raw)`: dict key/value order
    is not a documented guarantee, and this keeps the 24-field contract
    closed -- a future OpenCV adding an extra key would otherwise pass
    through silently instead of being explicitly ignored here.
    """
    return Moments(
        m00=raw["m00"],
        m10=raw["m10"],
        m01=raw["m01"],
        m20=raw["m20"],
        m11=raw["m11"],
        m02=raw["m02"],
        m30=raw["m30"],
        m21=raw["m21"],
        m12=raw["m12"],
        m03=raw["m03"],
        mu20=raw["mu20"],
        mu11=raw["mu11"],
        mu02=raw["mu02"],
        mu30=raw["mu30"],
        mu21=raw["mu21"],
        mu12=raw["mu12"],
        mu03=raw["mu03"],
        nu20=raw["nu20"],
        nu11=raw["nu11"],
        nu02=raw["nu02"],
        nu30=raw["nu30"],
        nu21=raw["nu21"],
        nu12=raw["nu12"],
        nu03=raw["nu03"],
    )


def moments(image_or_contour: Image | Contour, binary_image: bool = False) -> Moments:
    """Compute image or spatial moments from a raster image/mask or a contour.

    Parameters
    ----------
    image_or_contour : np.ndarray
        Either a 2D, single-channel, non-empty raster image (dtype one of
        ``uint8``, ``uint16``, ``int16``, ``float32``, ``float64``), or a
        `Contour` -- shape ``(N, 1, 2)``, dtype ``int32``, at least 1 point.
        Dispatched on ``ndim`` and ``dtype``: 3D, ``int32`` input is treated
        as a contour attempt; everything else goes through the 2D-only
        raster path (so a 3-channel raster image, also 3D but not
        ``int32``, still gets the clear "must have 2 dimensions" raster
        error instead of a confusing contour-shaped one).
    binary_image : bool, default False
        Raster input only: if ``True``, nonzero pixels are treated as ``1``
        (not ``255``) when computing raw moments -- verified directly.
        Combining ``binary_image=True`` with contour input raises
        `ValueError`: OpenCV silently ignores `binary_image` for contour
        input (verified directly), so accepting it here would misleadingly
        suggest it has an effect.

    Returns
    -------
    Moments
        24 fields, matching ``cv2.moments()``'s dict keys exactly.

    Raises
    ------
    ValueError
        For raster input: if it does not have exactly 2 dimensions or is
        empty. For contour input: if it does not have shape ``(N, 1, 2)``
        with at least 1 point, or if `binary_image` is ``True``.
    TypeError
        For raster input: if it does not have one of the accepted dtypes.
        For contour input: if it is not an `np.ndarray` of dtype ``int32``.
        In both cases, if `binary_image` is not an actual `bool`.
    """
    require_bool(binary_image, "binary_image")
    if (
        isinstance(image_or_contour, np.ndarray)
        and image_or_contour.ndim == 3
        and image_or_contour.dtype == np.int32
    ):
        _require_contour(image_or_contour, min_points=1)
        if binary_image:
            raise ValueError(
                "binary_image is not supported for contour input -- OpenCV silently "
                "ignores it, so improcv rejects the combination instead of accepting "
                "a parameter that has no effect"
            )
        raw = cv2.moments(image_or_contour)
        return _moments_from_dict(raw)
    require_image_ndim(image_or_contour, ndims=(2,))
    require_dtype(image_or_contour, _MOMENTS_RASTER_DTYPES)
    raw = cv2.moments(image_or_contour, binary_image)
    return _moments_from_dict(raw)


TemplateMatchMethod = Literal[
    "ccoeff", "ccoeff_normed", "ccorr", "ccorr_normed", "sqdiff", "sqdiff_normed"
]
_TEMPLATE_MATCH_METHODS: dict[TemplateMatchMethod, int] = {
    "ccoeff": cv2.TM_CCOEFF,
    "ccoeff_normed": cv2.TM_CCOEFF_NORMED,
    "ccorr": cv2.TM_CCORR,
    "ccorr_normed": cv2.TM_CCORR_NORMED,
    "sqdiff": cv2.TM_SQDIFF,
    "sqdiff_normed": cv2.TM_SQDIFF_NORMED,
}

_MATCH_TEMPLATE_DTYPES = (np.uint8, np.float32)


def _is_spatially_constant(template: np.ndarray) -> bool:
    """Return True if every channel of `template` is spatially constant.

    Checked per-channel, not globally: a per-channel-constant color
    template (e.g. BGR (0, 128, 255)) has nonzero *global* std but zero
    variance *within* each channel -- verified directly that a plain
    `template.std() == 0` check misses this case.
    """
    if template.ndim == 2:
        return bool(template.min() == template.max())
    channel_min = template.min(axis=(0, 1))
    channel_max = template.max(axis=(0, 1))
    return bool(np.all(channel_min == channel_max))


def match_template(
    image: Image, template: Image, method: TemplateMatchMethod
) -> npt.NDArray[np.float32]:
    """Locate a template within an image by sliding-window comparison.

    Parameters
    ----------
    image : np.ndarray
        Input image with shape ``(H, W)`` or ``(H, W, C)``, dtype ``uint8``
        or ``float32``, 1-4 channels.
    template : np.ndarray
        Template to search for, same dtype and channel count as `image`.
        Both spatial dimensions must be ``<=`` `image`'s. Some OpenCV
        implementation paths may accept swapped image/template roles when
        one array is larger than the other; improcv always enforces this
        documented size contract regardless of what OpenCV would otherwise
        tolerate.
    method : {"ccoeff", "ccoeff_normed", "ccorr", "ccorr_normed", "sqdiff", "sqdiff_normed"}
        Comparison method, matching ``cv2.TM_*``. For ``"sqdiff"``/
        ``"sqdiff_normed"``, the best match is the **minimum** of the
        result map; for every other method, the best match is the
        **maximum**.

    Returns
    -------
    np.ndarray
        Shape ``(image.height - template.height + 1, image.width -
        template.width + 1)``, dtype ``float32``. A new array; `image`/
        `template` are never modified.

    Raises
    ------
    ValueError
        If `image`/`template` is not 2D/3D or is empty, `template` does not
        fit within `image` spatially, `image`/`template` does not have 1-4
        channels or their channel counts differ, `method` is not one of
        the accepted values, `template` is spatially constant (per channel)
        and `method` is ``"ccoeff"``, ``"ccoeff_normed"``, or
        ``"sqdiff_normed"``, or `template` is all-zero (zero energy) and
        `method` is ``"ccorr"`` or ``"ccorr_normed"``.
    TypeError
        If `image`/`template` does not have dtype ``uint8``/``float32``, or
        their dtypes differ.
    """
    require_one_of(method, tuple(_TEMPLATE_MATCH_METHODS), "method")
    require_image_ndim(image, ndims=(2, 3))
    require_image_ndim(template, ndims=(2, 3))
    require_dtype(image, _MATCH_TEMPLATE_DTYPES)
    require_dtype(template, _MATCH_TEMPLATE_DTYPES, "template")
    if image.dtype != template.dtype:
        raise TypeError(
            f"image and template must have the same dtype, got {image.dtype} and {template.dtype}"
        )
    image_channels = 1 if image.ndim == 2 else image.shape[2]
    template_channels = 1 if template.ndim == 2 else template.shape[2]
    if image_channels not in (1, 2, 3, 4):
        raise ValueError(f"image must have 1-4 channels, got {image_channels}")
    if template_channels not in (1, 2, 3, 4):
        raise ValueError(f"template must have 1-4 channels, got {template_channels}")
    if image_channels != template_channels:
        raise ValueError(
            f"image and template must have the same channel count, got "
            f"{image_channels} and {template_channels}"
        )
    image_height, image_width = image.shape[:2]
    template_height, template_width = template.shape[:2]
    if template_height > image_height or template_width > image_width:
        raise ValueError(
            f"template ({template_height}x{template_width}) must fit within image "
            f"({image_height}x{image_width})"
        )
    if method in ("ccoeff", "ccoeff_normed", "sqdiff_normed") and _is_spatially_constant(template):
        # Two distinct reasons share this one check:
        #
        # - "ccoeff"/"ccoeff_normed" subtract the template's own mean before
        #   correlating. For a spatially constant template, the mean-centered
        #   template is deterministically all zero, so the numerator is zero
        #   at every position -- verified directly (a supposedly-exact-zero
        #   result shows only floating-point noise, e.g. min=-0.0625,
        #   max=0.0625 for a mid-gray template, never a real signal). Picking
        #   a max/min out of that noise is meaningless, not just unreliable.
        # - "sqdiff_normed" is a deliberate, conservative improcv choice, not
        #   a mathematical necessity: the formula is well-defined for a
        #   constant, nonzero template (its energy is well-defined). Verified
        #   directly, on both OpenCV 4.13 and 5.0, that the normalized result
        #   can still degenerate to a uniform 1.0 map depending on template
        #   size and pixel intensity in a way that isn't safely predictable
        #   (e.g. a mid-gray value is non-degenerate at 3x3 but fully
        #   degenerate at 10x10), so improcv rejects the whole
        #   spatially-constant category rather than trying to carve out a
        #   "safe" subset. This does not catch every possible degenerate case
        #   (a verified non-constant, very-low-energy template can still
        #   produce a uniform result) -- it is a narrow, deliberately limited
        #   guard against the clearest, most common one.
        raise ValueError(
            f"template must not be spatially constant (per channel) for method "
            f"{method!r} -- verified to produce a meaningless (mean-centered-to-zero "
            "or potentially uniform) result"
        )
    if method in ("ccorr", "ccorr_normed") and not np.any(template):
        # "ccorr" (unnormalized cross-correlation): numerator = sum(I * T);
        # an all-zero template makes this exactly zero at every position --
        # verified directly, a real, deterministic degeneracy, not numerical
        # noise. "ccorr_normed" hits the same zero-energy 0/0 case, resolved
        # by OpenCV to a clean 0.0 rather than NaN -- still uninformative.
        raise ValueError(f"template must not be all-zero (zero energy) for method {method!r}")

    result = cv2.matchTemplate(image, template, _TEMPLATE_MATCH_METHODS[method])
    # cv2's stubs type matchTemplate's result as the loose MatLike; it always
    # produces float32 in practice, verified directly for both uint8 and
    # float32 input.
    return cast(npt.NDArray[np.float32], result)


class MinMaxResult(NamedTuple):
    """Result of `min_max_loc`.

    `min_loc`/`max_loc` are ``(x, y)`` -- column, row -- matching
    ``cv2.minMaxLoc``'s own convention, not ``(row, column)``.
    """

    min_val: float
    max_val: float
    min_loc: tuple[int, int]
    max_loc: tuple[int, int]


_MIN_MAX_LOC_DTYPES = (np.uint8, np.uint16, np.int16, np.int32, np.float32, np.float64)


def min_max_loc(image: Image, mask: Mask | None = None) -> MinMaxResult:
    """Find the global minimum and maximum value and location in a single-channel image.

    Parameters
    ----------
    image : np.ndarray
        Input image with shape ``(H, W)``, dtype one of ``uint8``,
        ``uint16``, ``int16``, ``int32``, ``float32``, ``float64``.
    mask : np.ndarray or None, default None
        Optional ``uint8`` mask, shape ``(H, W)`` matching `image`. Must
        contain at least one nonzero pixel -- verified directly that
        ``cv2.minMaxLoc`` returns the sentinel result
        ``(0.0, 0.0, (-1, -1), (-1, -1))`` for an all-zero mask, which would
        otherwise look like a valid (if coincidentally zero) result.

    Returns
    -------
    MinMaxResult
        ``min_val``, ``max_val``, ``min_loc`` ``(x, y)``, ``max_loc``
        ``(x, y)``.

    Raises
    ------
    ValueError
        If `image` does not have exactly 2 dimensions or is empty, or
        `mask` does not match `image`'s spatial size or contains no
        nonzero pixel.
    TypeError
        If `image` does not have one of the accepted dtypes, or `mask`
        does not have dtype ``uint8``.

    Notes
    -----
    OpenCV's tie-breaking location when multiple pixels share the extreme
    value is unspecified -- do not rely on a particular one among ties.
    """
    require_image_ndim(image, ndims=(2,))
    require_dtype(image, _MIN_MAX_LOC_DTYPES)
    if mask is not None:
        require_spatial_mask(mask, image)
        if not np.any(mask):
            raise ValueError("mask must contain at least one nonzero pixel")
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(image, mask)
    # cv2's stubs type minMaxLoc's locations as the loose Point (Sequence[int]);
    # they are always a 2-tuple (x, y) in practice.
    return MinMaxResult(
        min_val, max_val, cast(tuple[int, int], min_loc), cast(tuple[int, int], max_loc)
    )


class MeanStdDevResult(NamedTuple):
    """Result of `mean_stddev`. One element per channel, not a covariance matrix."""

    mean: tuple[float, ...]
    stddev: tuple[float, ...]


_MEAN_STDDEV_DTYPES = (np.uint8, np.uint16, np.int16, np.int32, np.float32, np.float64)
_MEAN_STDDEV_MAX_CHANNELS = 128


def mean_stddev(image: Image, mask: Mask | None = None) -> MeanStdDevResult:
    """Compute the per-channel mean and population standard deviation of an image.

    Parameters
    ----------
    image : np.ndarray
        Input image with shape ``(H, W)`` or ``(H, W, C)``, dtype one of
        ``uint8``, ``uint16``, ``int16``, ``int32``, ``float32``,
        ``float64``, 1-128 channels. The 128-channel ceiling is a common,
        cross-version-safe limit, not an OpenCV 5.x-specific one: verified
        directly that ``cv2.meanStdDev`` handles up to 512 channels
        correctly on OpenCV 4.13, but silently collapses to a single
        aggregate mean/stddev above 128 channels on OpenCV 5.0 -- so 128 is
        the largest count both bindings compute correctly, and improcv
        rejects anything above it rather than silently corrupting the
        result depending on which OpenCV happens to be installed.
    mask : np.ndarray or None, default None
        Optional ``uint8`` mask, shape ``(H, W)`` matching `image`'s
        spatial size, applied identically to every channel. For an
        all-zero mask, the result is all-zeros for both `mean` and
        `stddev` (verified directly) -- this is accepted and returned
        as-is, not rejected.

    Returns
    -------
    MeanStdDevResult
        ``mean``/``stddev``, one element per channel. `stddev` is the
        **population** standard deviation (divided by ``N``, not
        ``N-1``). Channels are treated independently -- this is not a
        covariance matrix.

    Raises
    ------
    ValueError
        If `image` is not 2D/3D or is empty, does not have 1-128 channels,
        or `mask` does not match `image`'s spatial size.
    TypeError
        If `image` does not have one of the accepted dtypes, or `mask`
        does not have dtype ``uint8``.
    """
    require_image_ndim(image, ndims=(2, 3))
    require_dtype(image, _MEAN_STDDEV_DTYPES)
    require_channel_count(image, 1, _MEAN_STDDEV_MAX_CHANNELS)
    if mask is not None:
        require_spatial_mask(mask, image)
    mean, stddev = cv2.meanStdDev(image, mask=mask)
    return MeanStdDevResult(tuple(mean.ravel().tolist()), tuple(stddev.ravel().tolist()))
