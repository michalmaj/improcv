"""Image quality metrics: MSE, PSNR, SSIM."""

from __future__ import annotations

import math

import cv2
import numpy as np

from improcv._validation import (
    require_channel_count,
    require_dtype,
    require_image_ndim,
    require_positive,
    require_same_shape_and_dtype,
)
from improcv.types import Image

__all__ = [
    "mse",
    "psnr",
    "ssim",
]

_QUALITY_DTYPES = (np.uint8, np.uint16, np.float32, np.float64)
_FLOAT_DTYPES = (np.float32, np.float64)

_SSIM_WINDOW_SIZE = 11
_SSIM_SIGMA = 1.5
_SSIM_K1 = 0.01
_SSIM_K2 = 0.03
_SSIM_BORDER_CROP = (_SSIM_WINDOW_SIZE - 1) // 2


def _require_finite_image(image: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(image)):
        raise ValueError(f"{name} must not contain NaN or infinity")


def _require_comparable_images(image1: np.ndarray, image2: np.ndarray) -> None:
    """Raise ValueError/TypeError unless `image1`/`image2` are comparable for a quality metric.

    Validation order: non-empty + ndim (cheapest, most fundamental) ->
    shape/dtype agreement between the two images -> dtype allow-list ->
    channel count -> finite-value check for float inputs (most expensive,
    only when actually needed). Each step assumes the previous ones already
    hold, so error messages stay specific rather than generic.
    """
    require_image_ndim(image1, ndims=(2, 3))
    require_image_ndim(image2, ndims=(2, 3))
    require_same_shape_and_dtype(image1, image2, "image1", "image2")
    require_dtype(image1, _QUALITY_DTYPES, "image1")
    require_channel_count(image1, 1, 4, "image1")
    if image1.dtype in _FLOAT_DTYPES:
        _require_finite_image(image1, "image1")
        _require_finite_image(image2, "image2")


def _normalize_data_range(data_range: object, dtype: np.dtype) -> float:
    """Resolve `data_range` to a float, inferring a default only for uint8/uint16.

    `uint8`/`uint16` have an unambiguous natural range; `float32`/`float64`
    images could plausibly be in `[0, 1]`, `[0, 255]`, or something else
    entirely -- guessing would silently produce a wrong metric, so an
    explicit `data_range` is required for float inputs instead.
    """
    if data_range is None:
        if dtype == np.uint8:
            return 255.0
        if dtype == np.uint16:
            return 65535.0
        raise ValueError(
            f"data_range must be provided explicitly for dtype {dtype} "
            "(only inferred automatically for uint8/uint16)"
        )
    require_positive(data_range, "data_range")
    return float(data_range)  # type: ignore[arg-type]


def _mse_value(image1: np.ndarray, image2: np.ndarray) -> float:
    a = image1.astype(np.float64)
    b = image2.astype(np.float64)
    # Extreme (but individually finite) float64 inputs can overflow this
    # arithmetic to inf/nan -- caught explicitly below via math.isfinite,
    # so numpy's own RuntimeWarning for the same condition is redundant.
    with np.errstate(over="ignore", invalid="ignore"):
        diff = a - b
        result = float(np.mean(diff * diff))
    if not math.isfinite(result):
        raise ValueError("mse computation overflowed to a non-finite value")
    return result


def mse(image1: Image, image2: Image) -> float:
    """Compute the Mean Squared Error between two images.

    Parameters
    ----------
    image1, image2 : np.ndarray
        Two images with identical shape and dtype -- `uint8`, `uint16`,
        `float32`, or `float64` -- grayscale (``(H, W)``) or with 1-4
        channels (``(H, W, C)``). Neither is modified; internally both are
        cast to `float64` copies before any arithmetic.

    Returns
    -------
    float
        The mean squared error over every element (including channels),
        always finite and non-negative.

    Raises
    ------
    ValueError
        If either image is empty, does not have 2 or 3 dimensions, the two
        images don't have the same shape, either has an unsupported channel
        count, a float image contains `NaN`/infinity, or the computed
        result itself overflows to a non-finite value.
    TypeError
        If the two images don't have the same dtype, or the dtype isn't one
        of `uint8`/`uint16`/`float32`/`float64`.
    """
    _require_comparable_images(image1, image2)
    return _mse_value(image1, image2)


def psnr(
    image1: Image,
    image2: Image,
    data_range: float | None = None,
) -> float:
    """Compute the Peak Signal-to-Noise Ratio between two images, in decibels.

    Computed independently from `mse`, using the stable, direct form
    ``20*log10(data_range) - 10*log10(mse)`` rather than
    ``10*log10(data_range**2/mse)`` (avoids squaring `data_range` before
    the log, which matters for large `data_range` values). This project's
    own implementation is used rather than `cv2.PSNR`: verified directly
    that `cv2.PSNR` returns a large-but-finite sentinel (`~361.2`) for
    identical images instead of the mathematically correct `inf`, and
    silently does not scale its default reference value with `uint16`'s
    actual range unless the caller passes it explicitly.

    Parameters
    ----------
    image1, image2 : np.ndarray
        Two images with identical shape and dtype -- `uint8`, `uint16`,
        `float32`, or `float64` -- grayscale or with 1-4 channels. Neither
        is modified.
    data_range : float or None, optional
        The possible value span of the image data, used to scale the
        metric. Defaults to `255.0` for `uint8` and `65535.0` for `uint16`;
        must be provided explicitly (a positive, finite, non-bool real
        number) for `float32`/`float64`, since their actual range isn't
        implied by the dtype. Input values are never clipped or rescaled
        to `data_range` -- it only parameterizes the formula.

    Returns
    -------
    float
        The PSNR in decibels. `math.inf` when the two images are
        pixel-for-pixel identical (`mse == 0.0`). Can be negative when the
        error is larger than `data_range` itself -- not clamped.

    Raises
    ------
    ValueError
        If either image is empty, does not have 2 or 3 dimensions, the two
        images don't have the same shape, either has an unsupported channel
        count, a float image contains `NaN`/infinity, `data_range` is
        `None` for a float image, `data_range` is not positive/finite, or
        the underlying MSE computation overflows to a non-finite value.
    TypeError
        If the two images don't have the same dtype, the dtype isn't one of
        `uint8`/`uint16`/`float32`/`float64`, or `data_range` is not a real
        number (a `bool` is rejected, not silently treated as `0`/`1`).
    """
    _require_comparable_images(image1, image2)
    resolved_range = _normalize_data_range(data_range, image1.dtype)
    error = _mse_value(image1, image2)
    if error == 0.0:
        return math.inf
    return 20.0 * math.log10(resolved_range) - 10.0 * math.log10(error)


def _gaussian_filter(x: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(
        x,
        (_SSIM_WINDOW_SIZE, _SSIM_WINDOW_SIZE),
        sigmaX=_SSIM_SIGMA,
        sigmaY=_SSIM_SIGMA,
        borderType=cv2.BORDER_REFLECT101,
    )


def _ssim_channel_map(a: np.ndarray, b: np.ndarray, c1: float, c2: float) -> np.ndarray:
    """Compute the (uncropped) per-pixel SSIM map for one 2D float64 channel.

    Uses population covariance (no ``N/(N-1)`` sample-variance correction):
    local mean/variance/covariance come directly from Gaussian-weighted
    averages, matching a continuous weighted statistic rather than an
    unbiased finite-sample estimator.
    """
    mu_a = _gaussian_filter(a)
    mu_b = _gaussian_filter(b)
    mu_a_sq = mu_a * mu_a
    mu_b_sq = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a_sq = _gaussian_filter(a * a) - mu_a_sq
    sigma_b_sq = _gaussian_filter(b * b) - mu_b_sq
    sigma_ab = _gaussian_filter(a * b) - mu_ab

    numerator = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2)
    return numerator / denominator


def ssim(
    image1: Image,
    image2: Image,
    data_range: float | None = None,
) -> float:
    """Compute the Structural Similarity Index (SSIM) between two images.

    Implements the windowed-Gaussian variant from Wang et al. (2004): an
    ``11x11`` Gaussian window (``sigma=1.5``), stabilizing constants
    ``C1 = (0.01 * data_range)**2`` and ``C2 = (0.03 * data_range)**2``, and
    population (not sample-corrected) local covariance. Cross-checked
    numerically against `scikit-image` 0.26.0's
    ``structural_similarity(..., gaussian_weights=True, sigma=1.5,
    use_sample_covariance=False)`` in an isolated environment -- agreement
    at the floating-point-precision level, not `scikit-image` as a
    dependency of this project.

    Border handling: the Gaussian filtering itself uses
    ``cv2.BORDER_REFLECT101`` internally (needed to produce a value at
    every pixel position), but the outermost 5-pixel band (``(11-1)//2``,
    where the ``11x11`` window would extend past the border) is excluded
    from the final scalar -- matching the standard "valid" convolution
    region rather than letting a border-extension choice affect the
    result. An input exactly ``11x11`` therefore has only its single
    center pixel contributing to the result.

    No color space conversion is performed: channel order (BGR vs RGB)
    never matters as long as both images use the same order, and an alpha
    channel in a BGRA image participates identically to any other channel.
    Multi-channel images have SSIM computed independently per channel
    (each with its own spatial windowing), then averaged across channels
    and the valid spatial region together.

    Parameters
    ----------
    image1, image2 : np.ndarray
        Two images with identical shape and dtype -- `uint8`, `uint16`,
        `float32`, or `float64` -- with spatial dimensions at least
        ``11x11``, grayscale or with 1-4 channels. Neither is modified.
    data_range : float or None, optional
        The possible value span of the image data. Defaults to `255.0` for
        `uint8` and `65535.0` for `uint16`; must be provided explicitly for
        `float32`/`float64`. Input values are never clipped or rescaled.

    Returns
    -------
    float
        The mean SSIM over the valid spatial region and all channels,
        always in a mathematically unclamped range (typically ``[-1, 1]``
        but not enforced as such).

    Raises
    ------
    ValueError
        If either image is empty, does not have 2 or 3 dimensions, the two
        images don't have the same shape, either has an unsupported channel
        count, either spatial dimension is smaller than 11 pixels, a float
        image contains `NaN`/infinity, `data_range` is `None` for a float
        image, `data_range` is not positive/finite, or the computed result
        is non-finite (e.g. from extreme input magnitudes overflowing
        `float64` arithmetic internally).
    TypeError
        If the two images don't have the same dtype, the dtype isn't one of
        `uint8`/`uint16`/`float32`/`float64`, or `data_range` is not a real
        number.
    """
    _require_comparable_images(image1, image2)
    height, width = image1.shape[:2]
    if height < _SSIM_WINDOW_SIZE or width < _SSIM_WINDOW_SIZE:
        raise ValueError(
            f"image spatial size must be at least {_SSIM_WINDOW_SIZE}x{_SSIM_WINDOW_SIZE} "
            f"for ssim, got {height}x{width}"
        )
    resolved_range = _normalize_data_range(data_range, image1.dtype)
    c1 = (_SSIM_K1 * resolved_range) ** 2
    c2 = (_SSIM_K2 * resolved_range) ** 2

    a = image1.astype(np.float64)
    b = image2.astype(np.float64)
    channels = 1 if a.ndim == 2 else a.shape[2]

    crop = _SSIM_BORDER_CROP
    channel_means: list[float] = []
    # Extreme (but individually finite) float64 inputs can overflow this
    # arithmetic to inf/nan -- caught explicitly below via math.isfinite,
    # so numpy's own RuntimeWarning for the same condition is redundant.
    with np.errstate(over="ignore", invalid="ignore"):
        for c in range(channels):
            a_c = a if channels == 1 else a[:, :, c]
            b_c = b if channels == 1 else b[:, :, c]
            ssim_map = _ssim_channel_map(a_c, b_c, c1, c2)
            valid = ssim_map[crop:-crop, crop:-crop] if crop > 0 else ssim_map
            channel_means.append(float(valid.mean()))

    result = float(np.mean(channel_means))
    if not math.isfinite(result):
        raise ValueError("ssim computation produced a non-finite result")
    return result
