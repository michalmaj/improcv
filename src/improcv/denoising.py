"""Non-local means image denoising."""

from __future__ import annotations

from typing import cast

import cv2
import numpy as np

from improcv._validation import (
    require_channels,
    require_dtype,
    require_fits_dtype,
    require_image_ndim,
    require_non_negative,
    require_odd,
    require_positive_integral,
)
from improcv.types import ImageU8

__all__ = [
    "nl_means_denoise",
    "nl_means_denoise_colored",
]


def _require_valid_grayscale_image(image: np.ndarray) -> None:
    """Raise ValueError/TypeError unless `image` is a valid 2D grayscale `uint8` image.

    `(H, W, 1)` is rejected -- unlike several other functions in this project,
    `nl_means_denoise` requires a genuine 2D array, not a 3D array with a
    trailing 1-channel axis. A 3D input gets a message tailored to what it
    actually looks like: a single trailing channel suggests dropping the
    axis directly (`image[..., 0]`), while any other channel count suggests
    a real grayscale conversion (`improcv.ensure_gray`) instead, since
    slicing an actual color image would silently keep only one color
    channel rather than a true luminance value.
    """
    if image.ndim == 3:
        if image.shape[2] == 1:
            raise ValueError(
                "nl_means_denoise requires a 2D grayscale image, got a single-channel "
                "image with an explicit trailing axis -- drop it first with "
                "image[..., 0]"
            )
        raise ValueError(
            f"nl_means_denoise requires a 2D grayscale image, got {image.shape[2]} "
            "channels -- convert first with improcv.ensure_gray if this is a color image"
        )
    require_image_ndim(image, ndims=(2,))
    require_dtype(image, (np.uint8,), "image")


def _require_valid_bgr_image(image: np.ndarray) -> None:
    """Raise ValueError/TypeError unless `image` is a valid 3-channel BGR `uint8` image.

    Grayscale, `(H, W, 1)`, 2-channel, and BGRA are all rejected. See
    `nl_means_denoise_colored`'s docstring for why BGRA specifically is
    rejected despite `cv2.fastNlMeansDenoisingColored` technically accepting
    it in some builds.
    """
    require_image_ndim(image, ndims=(3,))
    require_channels(image, 3)
    require_dtype(image, (np.uint8,), "image")


def _validated_h(value: object, name: str) -> float:
    """Raise TypeError/ValueError unless `value` is a valid `h`-style filter-strength
    parameter, else return its `float32` value as a plain `float` -- the exact value
    OpenCV will receive.

    Unlike `photo.py`'s `sigma_s`/`sigma_r`, `0` is a legal value here (a
    documented no-op for grayscale denoising), so this requires non-negative,
    not strictly positive. `h` has no OpenCV-documented upper bound, so
    (unlike `photo.py`'s parameters) an extreme value can overflow to `inf`
    once converted to `float32` -- both the conversion and the finiteness
    check are wrapped so this never raises an uncontrolled `RuntimeWarning`
    or lets a non-finite value reach OpenCV silently.
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


def _validated_window_size(value: object, name: str) -> int:
    """Raise TypeError/ValueError unless `value` is a valid window-size parameter,
    else return it as a plain `int`.

    Requires a positive, odd integer that fits in a C++ `int` -- OpenCV
    itself performs none of these checks (verified directly: an even size,
    or `search_window_size < template_window_size`, is silently accepted
    and produces a result identical to the input -- a silent no-op, not an
    error), so this project enforces them explicitly instead of forwarding
    a value that would otherwise waste computation for no effect, or (for a
    value too large for a C++ `int`) reach a raw, unfriendly `cv2.error`.

    No upper bound is enforced beyond the C++ `int` range: verified directly
    that larger search windows substantially increase execution time, but
    there is no OpenCV-documented maximum, and any threshold chosen from
    timing a single machine/image size would both reject legitimate uses
    and fail to bound the cost for a larger image anyway.
    """
    require_positive_integral(value, name)
    numeric = int(value)  # type: ignore[arg-type]
    require_odd(numeric, name)
    require_fits_dtype(numeric, np.int32, name)
    return numeric


def nl_means_denoise(
    image: ImageU8,
    h: float = 3.0,
    *,
    template_window_size: int = 7,
    search_window_size: int = 21,
) -> ImageU8:
    """Denoise a grayscale image with the Non-local Means algorithm.

    Unlike a local filter (`improcv.gaussian_blur`, `improcv.median_blur`),
    which only considers each pixel's immediate spatial neighborhood, and
    `improcv.bilateral_filter`, which weighs local neighbors by spatial
    distance and value similarity, Non-local Means compares small patches of
    the image against every other patch within `search_window_size` and
    averages pixels from patches that look similar -- wherever in that
    window they are. This can better preserve fine structure than local
    filters for a well-chosen `h`, but is substantially more expensive (see
    `search_window_size`), and is not guaranteed to outperform
    `improcv.bilateral_filter` for every image or noise type.

    Parameters
    ----------
    image : np.ndarray
        A `uint8` grayscale image, shape ``(H, W)``. `(H, W, 1)`, BGR, and
        BGRA are all rejected -- see `Raises`. Not modified.
    h : float, optional
        Filter strength: larger values remove more noise but also more
        image detail; smaller values preserve detail but also more noise.
        Must be finite and non-negative. `0` is a legal, documented no-op --
        verified directly that `h=0` returns a result identical to `image`.
        No upper bound is enforced (OpenCV documents none), only that the
        value must not overflow to `inf` once converted to `float32`.
        Default `3.0`, OpenCV's own default.
    template_window_size : int, optional
        Size in pixels of the patch used to compute similarity between
        regions. Must be a positive odd integer. Default `7`, OpenCV's own
        recommended value.
    search_window_size : int, optional
        Size in pixels of the window searched for similar patches. Must be
        a positive odd integer, and at least `template_window_size` --
        verified directly that violating either constraint is not an error
        in OpenCV, but silently returns `image` unchanged, wasting whatever
        computation still occurs. Larger search windows can substantially
        increase execution time. Default `21`, OpenCV's own recommended
        value.

    Returns
    -------
    np.ndarray
        Shape ``(H, W)``, dtype `uint8` -- same shape as `image`. A new,
        independent array; never shares memory with `image`.

    Raises
    ------
    ValueError
        If `image` is empty, is not 2D, `h` is negative, non-finite, or
        positive but too small to remain positive once converted to
        `float32`, or `template_window_size`/`search_window_size` is not a
        positive odd integer within the range of a C++ `int`, or
        `search_window_size < template_window_size`.
    TypeError
        If `image` does not have dtype ``uint8``, `h` is not a real number
        (rejecting `bool`), or a window size is not an integer (rejecting
        `bool`).
    """
    _require_valid_grayscale_image(image)
    h = _validated_h(h, "h")
    template_window_size = _validated_window_size(template_window_size, "template_window_size")
    search_window_size = _validated_window_size(search_window_size, "search_window_size")
    if search_window_size < template_window_size:
        raise ValueError(
            "search_window_size must be >= template_window_size, got "
            f"search_window_size={search_window_size} < "
            f"template_window_size={template_window_size}"
        )

    result = cv2.fastNlMeansDenoising(
        image,
        h=h,
        templateWindowSize=template_window_size,
        searchWindowSize=search_window_size,
    )
    return cast(ImageU8, result)


def nl_means_denoise_colored(
    image: ImageU8,
    h_luminance: float = 3.0,
    h_color: float = 3.0,
    *,
    template_window_size: int = 7,
    search_window_size: int = 21,
) -> ImageU8:
    """Denoise a BGR image with the Non-local Means algorithm.

    Internally converts `image` to CIELAB and runs Non-local Means on the
    luminance (L) and color (AB) components separately, with `h_luminance`
    and `h_color` respectively, before converting back to BGR -- the same
    approach as `cv2.fastNlMeansDenoisingColored`. See `nl_means_denoise`
    for how Non-local Means compares to `improcv.gaussian_blur`/
    `median_blur`/`bilateral_filter`.

    Parameters
    ----------
    image : np.ndarray
        A `uint8` BGR image, shape ``(H, W, 3)``. Grayscale, `(H, W, 1)`,
        2-channel, and BGRA are all rejected -- see `Raises`. Convert
        grayscale input first with `improcv.ensure_bgr`. `cv2.
        fastNlMeansDenoisingColored` technically accepts a 4-channel (BGRA)
        image in some builds, but verified directly that the output's alpha
        channel is not the input's alpha at all -- it is always a constant
        `255`, regardless of the input alpha's actual content. improcv
        rejects BGRA outright rather than silently discarding the original
        alpha channel this way. Not modified.
    h_luminance : float, optional
        Filter strength for the luminance (L) component. See
        `nl_means_denoise`'s `h` for the general contract (non-negative,
        `0` legal, no upper bound beyond `float32` overflow). Unlike
        grayscale denoising, `h_luminance=0` together with `h_color=0` does
        **not** guarantee a result identical to `image`: the BGR -> CIELAB
        -> BGR round trip alone can shift values by a small amount (verified
        directly: up to a few of the smallest bits, even for a constant
        input image), independent of how much filtering `h_luminance`/
        `h_color` actually request. Default `3.0`, OpenCV's own default.
    h_color : float, optional
        Filter strength for the color (AB) components. Same contract as
        `h_luminance`. For most images, OpenCV's own guidance is that `10`
        is enough to remove colored noise without visibly distorting
        colors. Default `3.0`, OpenCV's own default.
    template_window_size : int, optional
        See `nl_means_denoise`. Default `7`.
    search_window_size : int, optional
        See `nl_means_denoise`. Default `21`.

    Returns
    -------
    np.ndarray
        Shape ``(H, W, 3)``, dtype `uint8`, BGR channel order -- same shape
        as `image`. A new, independent array; never shares memory with
        `image`.

    Raises
    ------
    ValueError
        If `image` is empty, does not have exactly 3 dimensions, does not
        have exactly 3 channels, `h_luminance`/`h_color` is negative,
        non-finite, or positive but too small to remain positive once
        converted to `float32`, or `template_window_size`/
        `search_window_size` is not a positive odd integer within the range
        of a C++ `int`, or `search_window_size < template_window_size`.
    TypeError
        If `image` does not have dtype ``uint8``, `h_luminance`/`h_color`
        is not a real number (rejecting `bool`), or a window size is not an
        integer (rejecting `bool`).
    """
    _require_valid_bgr_image(image)
    h_luminance = _validated_h(h_luminance, "h_luminance")
    h_color = _validated_h(h_color, "h_color")
    template_window_size = _validated_window_size(template_window_size, "template_window_size")
    search_window_size = _validated_window_size(search_window_size, "search_window_size")
    if search_window_size < template_window_size:
        raise ValueError(
            "search_window_size must be >= template_window_size, got "
            f"search_window_size={search_window_size} < "
            f"template_window_size={template_window_size}"
        )

    result = cv2.fastNlMeansDenoisingColored(
        image,
        h=h_luminance,
        hColor=h_color,
        templateWindowSize=template_window_size,
        searchWindowSize=search_window_size,
    )
    return cast(ImageU8, result)
