"""Photo/creative single-image effects: pencil sketch, stylization, detail enhancement."""

from __future__ import annotations

from typing import NamedTuple, cast

import cv2
import numpy as np

from improcv._validation import (
    require_dtype,
    require_image_ndim,
    require_non_negative,
    require_positive,
)
from improcv.types import ImageU8

__all__ = [
    "detail_enhance",
    "pencil_sketch",
    "stylize",
    "PencilSketchResult",
]

# Documented, OpenCV-supported ranges (modules/photo/include/opencv2/photo.hpp) --
# not arbitrary improcv restrictions. Values outside these are unsupported by
# OpenCV's own API contract: the parameters are stored as C++ `float`, so an
# extreme-but-technically-finite Python value can degrade to `inf` after that
# conversion, silently producing a useless result rather than a clear error.
_SIGMA_S_MAX = 200.0
_SIGMA_R_MAX = 1.0
_SHADE_FACTOR_MAX = 0.1


def _require_valid_photo_image(image: np.ndarray) -> None:
    """Raise ValueError/TypeError unless `image` is a valid 3-channel BGR `uint8` image.

    Grayscale (2D or `(H, W, 1)`), 2-channel, and BGRA (`(H, W, 4)`) input are
    all rejected -- verified directly that `cv2.stylization`/`cv2.pencilSketch`
    do not validate channel count at all: grayscale/2-channel input is silently
    misinterpreted (producing an incorrect, non-error result), and BGRA input
    can crash the process outright (observed directly, on both OpenCV 4.13 and
    5.0, for `cv2.stylization`/`cv2.pencilSketch` at specific image sizes).
    Unsupported channel layouts are rejected before the OpenCV call to avoid
    raw errors and build-dependent unsafe behavior observed for BGRA inputs.
    """
    require_image_ndim(image, ndims=(3,))
    if image.shape[2] != 3:
        if image.shape[2] == 4:
            raise ValueError(
                "image must be a 3-channel BGR image, got a 4-channel (BGRA) image -- "
                "explicitly drop (e.g. image[..., :3]) or composite the alpha channel "
                "onto a chosen background before calling; improcv.ensure_bgr does not "
                "accept BGRA"
            )
        raise ValueError(
            f"image must be a 3-channel BGR image, got {image.shape[2]} channel(s) -- "
            "convert first with improcv.ensure_bgr if grayscale"
        )
    require_dtype(image, (np.uint8,), "image")


def _require_valid_sigma_s(value: object) -> None:
    require_positive(value, "sigma_s")
    if float(value) > _SIGMA_S_MAX:  # type: ignore[arg-type]
        raise ValueError(f"sigma_s must be <= {_SIGMA_S_MAX}, got {value}")


def _require_valid_sigma_r(value: object) -> None:
    require_positive(value, "sigma_r")
    if float(value) > _SIGMA_R_MAX:  # type: ignore[arg-type]
        raise ValueError(f"sigma_r must be <= {_SIGMA_R_MAX}, got {value}")


def _require_valid_shade_factor(value: object) -> None:
    require_non_negative(value, "shade_factor")
    if float(value) > _SHADE_FACTOR_MAX:  # type: ignore[arg-type]
        raise ValueError(f"shade_factor must be <= {_SHADE_FACTOR_MAX}, got {value}")


class PencilSketchResult(NamedTuple):
    """The two outputs of `pencil_sketch`, always computed together by OpenCV.

    ``grayscale`` and ``color`` are independent arrays -- neither shares
    memory with the other or with the input image.
    """

    grayscale: ImageU8
    color: ImageU8


def pencil_sketch(
    image: ImageU8,
    sigma_s: float = 60.0,
    sigma_r: float = 0.07,
    shade_factor: float = 0.02,
) -> PencilSketchResult:
    """Render a pencil-like non-photorealistic line drawing of `image`.

    Parameters
    ----------
    image : np.ndarray
        A `uint8` BGR image, shape ``(H, W, 3)``. Grayscale, `(H, W, 1)`,
        2-channel, and BGRA input are all rejected -- see the module's
        `Raises` section. Convert grayscale input first with
        `improcv.ensure_bgr`; for BGRA, explicitly drop or composite the
        alpha channel yourself first. Not modified.
    sigma_s : float, optional
        Spatial filter scale. Must satisfy ``0 < sigma_s <= 200`` -- the
        range OpenCV's own API documents; `sigma_s <= 0` is rejected since
        it is not a valid scale for this parameter (and internally leads to
        degenerate/undefined filtering), and values above `200` are outside
        OpenCV's own supported range. Default `60.0`.
    sigma_r : float, optional
        Range filter scale. Must satisfy ``0 < sigma_r <= 1`` -- `sigma_r`
        is used as a divisor inside the underlying filter, so `0` leads to
        division by zero; see `sigma_s` for why the upper bound is enforced.
        Default `0.07`.
    shade_factor : float, optional
        Brightness of the pencil strokes in `grayscale`/`color`. Must
        satisfy ``0 <= shade_factor <= 0.1`` -- unlike `sigma_s`/`sigma_r`,
        `0` is a valid, documented extreme (a black `grayscale` output), not
        a degenerate case. Default `0.02`.

    Returns
    -------
    PencilSketchResult
        ``grayscale``: shape ``(H, W)``, dtype `uint8`. ``color``: shape
        ``(H, W, 3)``, dtype `uint8`, BGR channel order. Both are new,
        independent arrays -- neither shares memory with `image` or with
        each other. Both are always computed: OpenCV's own implementation
        produces both outputs from a single internal pass, so there is no
        cost saved by offering a single-variant option. For a
        constant-valued `image`, both outputs are identical regardless of
        the constant's actual value (verified directly).

    Raises
    ------
    ValueError
        If `image` is empty, does not have exactly 3 dimensions, does not
        have exactly 3 channels, or `sigma_s`/`sigma_r`/`shade_factor` is
        outside its documented range (including non-finite).
    TypeError
        If `image` does not have dtype ``uint8``, or `sigma_s`/`sigma_r`/
        `shade_factor` is not a real number (rejecting `bool`).
    """
    _require_valid_photo_image(image)
    _require_valid_sigma_s(sigma_s)
    _require_valid_sigma_r(sigma_r)
    _require_valid_shade_factor(shade_factor)

    grayscale, color = cv2.pencilSketch(
        image,
        sigma_s=float(sigma_s),
        sigma_r=float(sigma_r),
        shade_factor=float(shade_factor),
    )
    return PencilSketchResult(grayscale=cast(ImageU8, grayscale), color=cast(ImageU8, color))


def stylize(image: ImageU8, sigma_s: float = 60.0, sigma_r: float = 0.45) -> ImageU8:
    """Apply an edge-aware stylization filter to `image`.

    Parameters
    ----------
    image : np.ndarray
        A `uint8` BGR image, shape ``(H, W, 3)``. Grayscale, `(H, W, 1)`,
        2-channel, and BGRA input are all rejected -- see the module's
        `Raises` section. Convert grayscale input first with
        `improcv.ensure_bgr`; for BGRA, explicitly drop or composite the
        alpha channel yourself first. Not modified.
    sigma_s : float, optional
        Spatial filter scale. Must satisfy ``0 < sigma_s <= 200`` -- see
        `pencil_sketch`'s `sigma_s` for the same reasoning. Default `60.0`.
    sigma_r : float, optional
        Range filter scale. Must satisfy ``0 < sigma_r <= 1`` -- see
        `pencil_sketch`'s `sigma_r` for the same reasoning. Default `0.45`.

    Returns
    -------
    np.ndarray
        Shape ``(H, W, 3)``, dtype `uint8`, BGR channel order -- same shape
        as `image`. A new, independent array; never shares memory with
        `image`. For a constant-valued `image`, the result is identical
        regardless of the constant's actual value (verified directly).

    Raises
    ------
    ValueError
        If `image` is empty, does not have exactly 3 dimensions, does not
        have exactly 3 channels, or `sigma_s`/`sigma_r` is outside its
        documented range (including non-finite).
    TypeError
        If `image` does not have dtype ``uint8``, or `sigma_s`/`sigma_r` is
        not a real number (rejecting `bool`).
    """
    _require_valid_photo_image(image)
    _require_valid_sigma_s(sigma_s)
    _require_valid_sigma_r(sigma_r)

    result = cv2.stylization(image, sigma_s=float(sigma_s), sigma_r=float(sigma_r))
    return cast(ImageU8, result)


def detail_enhance(image: ImageU8, sigma_s: float = 10.0, sigma_r: float = 0.15) -> ImageU8:
    """Enhance local detail/texture in `image` while preserving edges.

    Parameters
    ----------
    image : np.ndarray
        A `uint8` BGR image, shape ``(H, W, 3)``. Grayscale, `(H, W, 1)`,
        2-channel, and BGRA input are all rejected -- see the module's
        `Raises` section. Convert grayscale input first with
        `improcv.ensure_bgr`; for BGRA, explicitly drop or composite the
        alpha channel yourself first. Not modified.
    sigma_s : float, optional
        Spatial filter scale. Must satisfy ``0 < sigma_s <= 200`` -- see
        `pencil_sketch`'s `sigma_s` for the same reasoning. Default `10.0`.
    sigma_r : float, optional
        Range filter scale. Must satisfy ``0 < sigma_r <= 1`` -- see
        `pencil_sketch`'s `sigma_r` for the same reasoning. Default `0.15`.

    Returns
    -------
    np.ndarray
        Shape ``(H, W, 3)``, dtype `uint8`, BGR channel order -- same shape
        as `image`. A new, independent array; never shares memory with
        `image`. For a constant-valued `image`, the result is identical to
        `image` (verified directly: zero local detail means zero
        enhancement) -- unlike `pencil_sketch`/`stylize`, whose constant-
        image result does not simply reproduce the input.

    Raises
    ------
    ValueError
        If `image` is empty, does not have exactly 3 dimensions, does not
        have exactly 3 channels, or `sigma_s`/`sigma_r` is outside its
        documented range (including non-finite).
    TypeError
        If `image` does not have dtype ``uint8``, or `sigma_s`/`sigma_r` is
        not a real number (rejecting `bool`).
    """
    _require_valid_photo_image(image)
    _require_valid_sigma_s(sigma_s)
    _require_valid_sigma_r(sigma_r)

    result = cv2.detailEnhance(image, sigma_s=float(sigma_s), sigma_r=float(sigma_r))
    return cast(ImageU8, result)
