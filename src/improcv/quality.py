"""Image quality metrics: MSE, PSNR, SSIM, GMSD."""

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
    "gmsd",
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
# Symmetric bounds (reciprocals of each other) on the largest relevant
# magnitude (data_range, or either image's own largest absolute value)
# outside of which images/data_range get rescaled before computing SSIM --
# see the detailed justification in `ssim`'s body, where the actual check
# happens. Both leave wide margin against float64's range (~1.8e308 max,
# ~4.9e-324 smallest positive subnormal) given the formula's numerator/
# denominator are each effectively a 4th power of this magnitude.
_SSIM_SAFE_MAGNITUDE_MAX = 1e75
_SSIM_SAFE_MAGNITUDE_MIN = 1e-75

# GMSD constants, verified against the reference MATLAB implementation shared
# by the original authors (Xue, Zhang, Mou, Bovik -- "Gradient Magnitude
# Similarity Deviation", IEEE TIP 2014; GMSD.m from
# www4.comp.polyu.edu.hk/~cslzhang/IQA/GMSD/GMSD.htm), not the paper's own
# rounded prose. The paper's text states c=0.0026 for images normalized to
# [0, 1]; the authors' own shipped code instead uses T=170 directly on 0-255
# data. These are NOT the same constant: 170 / 255**2 == 0.00261438..., not
# 0.0026 -- confirmed to produce measurably different GMSD scores (~1e-5 to
# 1e-4 absolute, cross-checked against the reference code in Octave). T=170
# is used here since it is what the authors' own code -- and therefore every
# benchmark actually run against it -- computes.
_GMSD_T_CONST = 170.0 / 255.0**2
_GMSD_AVERAGE_KERNEL = np.full((2, 2), 0.25, dtype=np.float64)
_GMSD_DX = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float64) / 3.0
_GMSD_DY = _GMSD_DX.T.copy()
# Reused numerically from SSIM's own safe-magnitude bounds, but justified
# differently: GMSD's quality map (2*g1*g2+T)/(g1**2+g2**2+T) is a ratio of
# terms that are each order (data_range)**2 throughout (T itself scales as
# data_range**2, and gradient products/squares are order data_range**2 too)
# -- a 2nd-degree relationship in data_range, not SSIM's 4th-degree one
# (SSIM's C1*C2 product is a product of two already-squared terms). At
# `1e75`, `(1e75)**2 == 1e150` stays far under float64's ~1.8e308 max; at
# `1e-75`, `(1e-75)**2 == 1e-150` stays far above the smallest positive
# subnormal (~4.9e-324) -- both with even more margin than strictly needed
# for a 2nd-degree expression, so reusing the same numeric bounds is safe.
_GMSD_SAFE_MAGNITUDE_MAX = 1e75
_GMSD_SAFE_MAGNITUDE_MIN = 1e-75


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


def _scaled_squared_error_stats(image1: np.ndarray, image2: np.ndarray) -> tuple[float, float]:
    """Return `(scale, mean_sq_normalized)` such that `mse == scale**2 * mean_sq_normalized`
    exactly (mathematically), computed by normalizing the difference by its own largest
    absolute value first.

    Squaring the raw difference directly underflows to exactly `0.0` for two images that
    are extremely close but not identical (e.g. a constant offset of `1e-162` in `float64`
    -- `(1e-162)**2` is below the smallest representable subnormal `float64` and rounds to
    `0.0`), which would silently misreport genuinely different images as identical.
    Dividing by `scale` first keeps every squared term of order 1 or less, so only the
    final `scale**2` multiplication -- the mathematically unavoidable point of underflow,
    if any -- can produce `0.0`. `scale == 0.0` iff the two images are exactly identical
    (every element of the difference is exactly zero).
    """
    a = image1.astype(np.float64)
    b = image2.astype(np.float64)
    # Extreme (but individually finite) float64 inputs can overflow/underflow this
    # arithmetic -- caught explicitly by callers via math.isfinite/exact-zero checks,
    # so numpy's own RuntimeWarning for the same condition is redundant.
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        diff = a - b
        scale = float(np.max(np.abs(diff)))
        if scale == 0.0:
            return 0.0, 0.0
        normalized = diff / scale
        mean_sq_normalized = float(np.mean(normalized * normalized))
    return scale, mean_sq_normalized


def _mse_value(image1: np.ndarray, image2: np.ndarray) -> float:
    scale, mean_sq_normalized = _scaled_squared_error_stats(image1, image2)
    if scale == 0.0:
        return 0.0
    with np.errstate(over="ignore", under="ignore"):
        result = (scale * scale) * mean_sq_normalized
    if not math.isfinite(result):
        raise ValueError("mse computation overflowed to a non-finite value")
    if result == 0.0:
        raise ValueError(
            "mse underflowed to zero: the true mean squared error between these "
            "non-identical images is too small to represent as a positive float64 value"
        )
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
        always finite and non-negative. Exactly `0.0` only when the two
        images are pixel-for-pixel identical.

    Raises
    ------
    ValueError
        If either image is empty, does not have 2 or 3 dimensions, the two
        images don't have the same shape, either has an unsupported channel
        count, a float image contains `NaN`/infinity, the computed result
        overflows to a non-finite value, or the two images are not
        identical but their true mean squared error is too small to
        represent as a positive `float64` value (e.g. two otherwise-equal
        `float64` images differing by `1e-162` at a single pixel) -- rather
        than silently reporting `0.0` and implying the images are
        identical. Use `psnr` instead if only the error's magnitude in
        decibels is needed: it remains well-defined in this situation.
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

    Computed independently from `mse` -- not by calling it and taking
    `log10` of the result -- since `mse` itself can legitimately raise for
    a true positive error too small to represent as a `float64` (see
    `mse`'s docstring), while `psnr` only ever needs that error's base-10
    logarithm, which stays finite and well-defined via the same
    scale-normalized decomposition (`log10(mse) = 2*log10(scale) +
    log10(mean_sq_normalized)`) even when the raw `mse` scalar itself
    would underflow to `0.0`. Uses the stable, direct form
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
        the underlying error computation overflows to a non-finite value.
    TypeError
        If the two images don't have the same dtype, the dtype isn't one of
        `uint8`/`uint16`/`float32`/`float64`, or `data_range` is not a real
        number (a `bool` is rejected, not silently treated as `0`/`1`).
    """
    _require_comparable_images(image1, image2)
    resolved_range = _normalize_data_range(data_range, image1.dtype)
    scale, mean_sq_normalized = _scaled_squared_error_stats(image1, image2)
    if scale == 0.0:
        return math.inf
    log10_mse = 2.0 * math.log10(scale) + math.log10(mean_sq_normalized)
    if not math.isfinite(log10_mse):
        raise ValueError("psnr computation overflowed to a non-finite value")
    return 20.0 * math.log10(resolved_range) - 10.0 * log10_mse


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

    # Exact-equality fast path: pixel-for-pixel identical images always
    # give SSIM == 1.0 by definition, regardless of dtype, magnitude, or
    # data_range -- placed after every validation above (dtype/shape/
    # spatial-size/data_range all still apply to identical inputs) so it
    # can never mask a real contract violation, only skip computation that
    # would otherwise need the magnitude-safety handling below.
    if np.array_equal(image1, image2):
        return 1.0

    a = image1.astype(np.float64)
    b = image2.astype(np.float64)

    # A common positive rescaling factor applied to both images and
    # data_range leaves SSIM mathematically unchanged (every term in the
    # formula scales by the same power of the factor and cancels in the
    # final ratio), so it's safe to divide by whatever keeps every
    # subsequent value comfortably representable. Skipped entirely
    # (factor stays 1.0, computation bit-identical to before) unless
    # something is actually large enough to risk it in *either* direction
    # -- ordinary uint8/uint16/small-float inputs never reach this branch,
    # verified against scikit-image's exact reference values after adding
    # this guard.
    #
    # Upper bound: the formula's numerator/denominator are each a product
    # of two already-squared terms (e.g. c1*c2, or mu_a_sq*mu_b_sq), so
    # effectively a 4th power of the working magnitude. `1e75**4 == 1e300`
    # stays comfortably under float64's ~1.8e308 max (~8 orders of
    # magnitude of margin).
    #
    # Lower bound: the same 4th-power relationship applies in the other
    # direction -- `1e-75**4 == 1e-300` stays far above float64's smallest
    # positive (subnormal) value (~4.9e-324, ~24 orders of magnitude of
    # margin), well past the point where a magnitude this small (e.g. two
    # otherwise-ordinary images with an astronomically small data_range,
    # or vice versa) would otherwise underflow c1*c2 or similar products
    # to exactly `0.0` and produce a `0/0` `NaN`. Chosen as the exact
    # reciprocal of the upper bound for a symmetric, easy-to-reason-about
    # safe zone.
    max_abs = max(resolved_range, float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    if max_abs > _SSIM_SAFE_MAGNITUDE_MAX or max_abs < _SSIM_SAFE_MAGNITUDE_MIN:
        a = a / max_abs
        b = b / max_abs
        resolved_range = resolved_range / max_abs

    c1 = (_SSIM_K1 * resolved_range) ** 2
    c2 = (_SSIM_K2 * resolved_range) ** 2

    channels = 1 if a.ndim == 2 else a.shape[2]

    crop = _SSIM_BORDER_CROP
    channel_means: list[float] = []
    # Extreme (but individually finite) float64 inputs can overflow this
    # arithmetic to inf/nan -- caught explicitly below via math.isfinite,
    # so numpy's own RuntimeWarning for the same condition is redundant.
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
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


def _require_valid_gmsd_images(image1: np.ndarray, image2: np.ndarray) -> None:
    """Raise ValueError/TypeError unless `image1`/`image2` are valid grayscale GMSD inputs.

    Validation order matches `mse`/`psnr`/`ssim`: non-empty + ndim -> shape/dtype
    agreement -> dtype allow-list -> grayscale-only channel constraint -> finite-value
    check for float inputs. GMSD is a luminance-only metric with no reference definition
    for color, and this project never hides an automatic color conversion -- a 3-channel
    or 4-channel image is rejected rather than silently converted.
    """
    require_image_ndim(image1, ndims=(2, 3))
    require_image_ndim(image2, ndims=(2, 3))
    require_same_shape_and_dtype(image1, image2, "image1", "image2")
    require_dtype(image1, _QUALITY_DTYPES, "image1")
    if image1.ndim == 3 and image1.shape[2] != 1:
        raise ValueError(
            f"gmsd requires a grayscale image (2D, or 3D with exactly 1 channel), got "
            f"{image1.shape[2]} channels -- convert first with improcv.ensure_gray"
        )
    if image1.dtype in _FLOAT_DTYPES:
        _require_finite_image(image1, "image1")
        _require_finite_image(image2, "image2")


def _gmsd_filter(image: np.ndarray, kernel: np.ndarray, anchor: tuple[int, int]) -> np.ndarray:
    return cv2.filter2D(
        image,
        ddepth=cv2.CV_64F,
        kernel=kernel,
        anchor=anchor,
        delta=0.0,
        borderType=cv2.BORDER_CONSTANT,
    )


def gmsd(
    image1: Image,
    image2: Image,
    data_range: float | None = None,
) -> float:
    """Compute the Gradient Magnitude Similarity Deviation between two grayscale images.

    Implements the algorithm from Xue, Zhang, Mou, and Bovik, "Gradient Magnitude
    Similarity Deviation: A Highly Efficient Perceptual Image Quality Index" (IEEE TIP,
    2014), matching the reference MATLAB implementation the authors shared (`GMSD.m`),
    not the paper's own rounded prose -- see `_GMSD_T_CONST`'s definition for exactly
    where these two diverge and why the code was preferred. Cross-checked numerically
    against that exact, unmodified MATLAB file (run via GNU Octave, an isolated,
    throwaway environment -- not a project dependency) across identical/constant/
    impulse/edge/noise/blur images and even/odd/mixed/small spatial sizes: agreement at
    floating-point precision (~1e-14 to exact) once two non-obvious implementation
    details were matched exactly (see below).

    Algorithm: both images are averaged with a 2x2 box filter and downsampled by 2
    (kept indices ``0, 2, 4, ...`` in each dimension, i.e. `ceil(size / 2)` samples per
    dimension); a horizontal and vertical Prewitt-style filter are applied to each to
    get a gradient magnitude image; a pixel-wise "gradient magnitude similarity" map is
    computed from the two gradient magnitude images; the final score is the *sample*
    standard deviation (``ddof=1``) of that map.

    Two non-obvious implementation details, verified necessary for exact agreement with
    the reference:

    - The 2x2 averaging filter has no single well-defined center (unlike the 3x3
      Prewitt filters). MATLAB's ``conv2(image, kernel, 'same')`` anchors an even-sized
      kernel at its top-left element; `cv2.filter2D`'s default anchor does not match
      this and silently gives a different (shifted) result. `anchor=(0, 0)` is required
      for the averaging step specifically.
    - `cv2.filter2D` computes correlation, while MATLAB's `conv2` computes convolution.
      The Prewitt kernels are anti-symmetric under 180-degree rotation
      (``rot180(kernel) == -kernel``), so correlation gives exactly the negative of what
      convolution gives -- which cancels out exactly once the gradient magnitude squares
      it, verified both mathematically and by the exact numerical match above. No
      kernel flip is needed.
    - Zero-padding (`cv2.BORDER_CONSTANT` with a default border value of `0`) is used
      throughout, matching `conv2`'s default boundary condition -- not
      `cv2.filter2D`'s own default border mode.

    Parameters
    ----------
    image1, image2 : np.ndarray
        Two images with identical shape and dtype -- `uint8`, `uint16`, `float32`, or
        `float64` -- either 2D (grayscale) or 3D with exactly 1 channel (`(H, W, 1)`,
        reduced internally to the same result as the equivalent 2D input). A 3-channel
        or 4-channel image is rejected: GMSD is a luminance-only metric with no
        reference definition for color, and this project never hides an automatic
        color conversion -- convert explicitly with `improcv.ensure_gray` first.
        Neither image is modified.
    data_range : float or None, optional
        The possible value span of the image data. Defaults to `255.0` for `uint8` and
        `65535.0` for `uint16`; must be provided explicitly (a positive, finite,
        non-bool number) for `float32`/`float64`. Input values are never clipped or
        rescaled to it.

    Returns
    -------
    float
        The GMSD score. Unlike `ssim`, **lower is better**: `0.0` exactly for
        pixel-for-pixel identical images (an explicit fast path, not a numerical
        coincidence), and larger values indicate more distortion. `0.0` is not general
        proof of pixel-for-pixel equality in the reverse direction -- it is only
        guaranteed to occur for identical inputs, not to occur *only* for them. GMSD is
        not guaranteed to be monotonic in every distortion type or severity. Two
        different *constant* images can give a non-zero score: `conv2`'s zero-padded
        border makes the gradient at the image edge artificially non-zero (the
        interior gradient of a constant image is exactly zero, but the border pixels
        see zero-padding as a discontinuity) -- this matches the reference
        implementation's own behavior and is not a bug in this port.

    Raises
    ------
    ValueError
        If either image is empty, does not have 2 or 3 dimensions, the two images
        don't have the same shape, either has more than 1 channel, a float image
        contains `NaN`/infinity, `data_range` is `None` for a float image,
        `data_range` is not positive/finite, the image's spatial size downsamples to
        fewer than 2 total samples (`ceil(H/2) * ceil(W/2) < 2` -- i.e. `1x1`, `1x2`,
        `2x1`, or `2x2` input; `ddof=1` pooling is undefined for a single sample; this
        is a deliberate, safer departure from the reference MATLAB implementation,
        which returns `0.0` for such degenerate inputs even when the two images are
        completely different, rather than raising), or the computation overflows or
        underflows to a non-finite intermediate value or final result.
    TypeError
        If the two images don't have the same dtype, the dtype isn't one of
        `uint8`/`uint16`/`float32`/`float64`, or `data_range` is not a real number.
    """
    _require_valid_gmsd_images(image1, image2)
    resolved_range = _normalize_data_range(data_range, image1.dtype)

    height, width = image1.shape[0], image1.shape[1]
    downsampled_height = (height + 1) // 2
    downsampled_width = (width + 1) // 2
    sample_count = downsampled_height * downsampled_width
    if sample_count < 2:
        raise ValueError(
            "gmsd requires at least 2 samples in the downsampled gradient magnitude "
            f"similarity map (ddof=1 pooling needs at least 2 values), got a "
            f"{height}x{width} image which downsamples to "
            f"{downsampled_height}x{downsampled_width} ({sample_count} sample(s))"
        )

    if np.array_equal(image1, image2):
        return 0.0

    a = image1[:, :, 0] if image1.ndim == 3 else image1
    b = image2[:, :, 0] if image2.ndim == 3 else image2
    a = a.astype(np.float64)
    b = b.astype(np.float64)

    # See _GMSD_SAFE_MAGNITUDE_MAX/MIN's definitions for the justification --
    # skipped entirely (factor stays 1.0, computation bit-identical to the
    # reference) unless something is actually large or small enough to risk it.
    max_abs = max(resolved_range, float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    if max_abs > _GMSD_SAFE_MAGNITUDE_MAX or max_abs < _GMSD_SAFE_MAGNITUDE_MIN:
        a = a / max_abs
        b = b / max_abs
        resolved_range = resolved_range / max_abs

    t = _GMSD_T_CONST * resolved_range**2

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        ave_a = _gmsd_filter(a, _GMSD_AVERAGE_KERNEL, anchor=(0, 0))
        ave_b = _gmsd_filter(b, _GMSD_AVERAGE_KERNEL, anchor=(0, 0))
        down_a = ave_a[0::2, 0::2]
        down_b = ave_b[0::2, 0::2]

        ix_a = _gmsd_filter(down_a, _GMSD_DX, anchor=(-1, -1))
        iy_a = _gmsd_filter(down_a, _GMSD_DY, anchor=(-1, -1))
        grad_a = np.sqrt(ix_a * ix_a + iy_a * iy_a)
        if not np.all(np.isfinite(grad_a)):
            raise ValueError(
                "gmsd: reference image's gradient magnitude overflowed to a non-finite value"
            )

        ix_b = _gmsd_filter(down_b, _GMSD_DX, anchor=(-1, -1))
        iy_b = _gmsd_filter(down_b, _GMSD_DY, anchor=(-1, -1))
        grad_b = np.sqrt(ix_b * ix_b + iy_b * iy_b)
        if not np.all(np.isfinite(grad_b)):
            raise ValueError(
                "gmsd: distorted image's gradient magnitude overflowed to a non-finite value"
            )

        gms_map = (2 * grad_a * grad_b + t) / (grad_a * grad_a + grad_b * grad_b + t)
        if not np.all(np.isfinite(gms_map)):
            raise ValueError(
                "gmsd: gradient magnitude similarity map computation produced a non-finite value"
            )

    result = float(np.std(gms_map, ddof=1))
    if not math.isfinite(result):
        raise ValueError("gmsd computation produced a non-finite result")
    return result
