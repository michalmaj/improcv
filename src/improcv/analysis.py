"""Image analysis: histograms, moments, template matching, and pixel statistics."""

from __future__ import annotations

from typing import Literal, NamedTuple, cast

import cv2
import numpy as np
import numpy.typing as npt

from improcv._compat.opencv import _normalize_calc_hist_output
from improcv._validation import (
    require_bool,
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
    "Moments",
    "TemplateMatchMethod",
]

_HISTOGRAM_DTYPES = (np.uint8, np.uint16, np.float32)


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
        ``uint16``, or ``float32``.
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
        If `image` is not 2D/3D or is empty, `channel` is out of range,
        `bins` is not positive, `value_range` does not have `low < high` or
        contains a non-finite value, or `mask` does not match `image`'s
        spatial size.
    TypeError
        If `image` does not have dtype ``uint8``/``uint16``/``float32``,
        `channel`/`bins` is not `numbers.Integral` (rejecting `bool` and
        `float`), or `mask` does not have dtype ``uint8``.
    """
    require_image_ndim(image, ndims=(2, 3))
    require_dtype(image, _HISTOGRAM_DTYPES)
    require_integral(channel, "channel")
    channel_int = int(channel)
    num_channels = 1 if image.ndim == 2 else image.shape[2]
    if not (0 <= channel_int < num_channels):
        raise ValueError(f"channel must be in [0, {num_channels}), got {channel_int}")
    require_positive_integral(bins, "bins")
    bins_int = int(bins)
    low, high = value_range
    require_finite(low, "value_range[0]")
    require_finite(high, "value_range[1]")
    if not low < high:
        raise ValueError(f"value_range must have low < high, got ({low}, {high})")
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
        return Moments(**raw)
    require_image_ndim(image_or_contour, ndims=(2,))
    require_dtype(image_or_contour, _MOMENTS_RASTER_DTYPES)
    raw = cv2.moments(image_or_contour, binary_image)
    return Moments(**raw)


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
        channels or their channel counts differ, or `method` is not one of
        the accepted values.
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
    if method in ("ccoeff_normed", "sqdiff_normed") and _is_spatially_constant(template):
        # A deliberate, conservative improcv choice -- not a claim that the
        # formula is mathematically undefined for a constant, nonzero
        # template (it isn't for sqdiff_normed; the energy is well-defined).
        # Verified directly, on both OpenCV 4.13 and 5.0: a spatially
        # constant template's normalized result can degenerate to a
        # uniform 1.0 map depending on template size and pixel intensity in
        # a way that isn't safely predictable (e.g. a mid-gray value is
        # non-degenerate at 3x3 but fully degenerate at 10x10), so improcv
        # rejects the whole spatially-constant category rather than trying
        # to carve out a "safe" subset. This does not catch every possible
        # degenerate case (a verified non-constant, very-low-energy
        # template can still produce a uniform result) -- it is a narrow,
        # deliberately limited guard against the clearest, most common one.
        raise ValueError(
            f"template must not be spatially constant (per channel) for method "
            f"{method!r} -- verified to potentially produce a uniform, uninformative "
            "result"
        )
    if method == "ccorr_normed" and not np.any(template):
        raise ValueError("template must not be all-zero (zero energy) for method 'ccorr_normed'")

    result = cv2.matchTemplate(image, template, _TEMPLATE_MATCH_METHODS[method])
    # cv2's stubs type matchTemplate's result as the loose MatLike; it always
    # produces float32 in practice, verified directly for both uint8 and
    # float32 input.
    return cast(npt.NDArray[np.float32], result)
