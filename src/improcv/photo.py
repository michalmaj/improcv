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
    if image.ndim == 2:
        raise ValueError(
            "image must be a 3-channel BGR image, got a 2D grayscale image -- convert "
            "first with improcv.ensure_bgr(image)"
        )
    require_image_ndim(image, ndims=(3,))
    if image.shape[2] != 3:
        if image.shape[2] == 4:
            raise ValueError(
                "image must be a 3-channel BGR image, got a 4-channel (BGRA) image -- "
                "explicitly drop (e.g. image[..., :3]) or composite the alpha channel "
                "onto a chosen background before calling; improcv.ensure_bgr does not "
                "accept BGRA"
            )
        if image.shape[2] == 1:
            raise ValueError(
                "image must be a 3-channel BGR image, got a single-channel (grayscale) "
                "image with an explicit trailing axis -- improcv.ensure_bgr itself does "
                "not accept (H, W, 1); drop the axis first with "
                "improcv.ensure_bgr(image[..., 0])"
            )
        raise ValueError(
            f"image must be a 3-channel BGR image, got {image.shape[2]} channels -- "
            "there is no supported conversion for this channel count"
        )
    require_dtype(image, (np.uint8,), "image")


def _validated_sigma_s(value: object) -> float:
    """Raise TypeError/ValueError unless `value` is a valid `sigma_s`, else return
    its `float32` value as a plain `float` -- the exact value OpenCV will receive.

    Validated on the `float32` value, not the original Python value: e.g.
    `1e-46` is a positive `float`, but `np.float32(1e-46) == 0.0` (underflows
    below `float32`'s smallest positive subnormal, `~1.4e-45`) -- silently
    passing such a value through would let it bypass the public `> 0`
    contract and reach OpenCV as `0.0`, exactly the degenerate case the
    contract exists to reject (`stylize`/`pencil_sketch` verified to produce
    an all-black result for `sigma_s=0`).
    """
    require_positive(value, "sigma_s")
    numeric = float(value)  # type: ignore[arg-type]
    if numeric > _SIGMA_S_MAX:
        raise ValueError(f"sigma_s must be <= {_SIGMA_S_MAX}, got {value}")
    converted = float(np.float32(numeric))
    if converted == 0.0:
        raise ValueError(
            f"sigma_s must be positive, got {value}, which is too small to remain "
            "positive once converted to OpenCV's float32 parameter"
        )
    return converted


def _validated_sigma_r(value: object) -> float:
    """Raise TypeError/ValueError unless `value` is a valid `sigma_r`, else return
    its `float32` value as a plain `float` -- the exact value OpenCV will receive.

    See `_validated_sigma_s` for why validation happens on the converted
    `float32` value rather than the original Python value.
    """
    require_positive(value, "sigma_r")
    numeric = float(value)  # type: ignore[arg-type]
    if numeric > _SIGMA_R_MAX:
        raise ValueError(f"sigma_r must be <= {_SIGMA_R_MAX}, got {value}")
    converted = float(np.float32(numeric))
    if converted == 0.0:
        raise ValueError(
            f"sigma_r must be positive, got {value}, which is too small to remain "
            "positive once converted to OpenCV's float32 parameter"
        )
    return converted


def _validated_shade_factor(value: object) -> float:
    """Raise TypeError/ValueError unless `value` is a valid `shade_factor`, else
    return its `float32` value as a plain `float` -- the exact value OpenCV
    will receive.

    Unlike `sigma_s`/`sigma_r`, a `float32`-converted value of `0.0` is not
    rejected here -- `shade_factor=0` is itself a valid, documented extreme,
    so a tiny positive value underflowing to `0.0` reaches the same
    already-legal value, not a hidden contract violation.
    """
    require_non_negative(value, "shade_factor")
    numeric = float(value)  # type: ignore[arg-type]
    if numeric > _SHADE_FACTOR_MAX:
        raise ValueError(f"shade_factor must be <= {_SHADE_FACTOR_MAX}, got {value}")
    return float(np.float32(numeric))


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
        have exactly 3 channels, `sigma_s`/`sigma_r`/`shade_factor` is
        outside its documented range (including non-finite), or `sigma_s`/
        `sigma_r` is positive but too small to remain positive once
        converted to OpenCV's `float32` parameter (e.g. `1e-46`).
    TypeError
        If `image` does not have dtype ``uint8``, or `sigma_s`/`sigma_r`/
        `shade_factor` is not a real number (rejecting `bool`).
    """
    _require_valid_photo_image(image)
    sigma_s = _validated_sigma_s(sigma_s)
    sigma_r = _validated_sigma_r(sigma_r)
    shade_factor = _validated_shade_factor(shade_factor)

    grayscale, color = cv2.pencilSketch(
        image,
        sigma_s=sigma_s,
        sigma_r=sigma_r,
        shade_factor=shade_factor,
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
        have exactly 3 channels, `sigma_s`/`sigma_r` is outside its
        documented range (including non-finite), or `sigma_s`/`sigma_r` is
        positive but too small to remain positive once converted to
        OpenCV's `float32` parameter (e.g. `1e-46`).
    TypeError
        If `image` does not have dtype ``uint8``, or `sigma_s`/`sigma_r` is
        not a real number (rejecting `bool`).
    """
    _require_valid_photo_image(image)
    sigma_s = _validated_sigma_s(sigma_s)
    sigma_r = _validated_sigma_r(sigma_r)

    result = cv2.stylization(image, sigma_s=sigma_s, sigma_r=sigma_r)
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
        have exactly 3 channels, `sigma_s`/`sigma_r` is outside its
        documented range (including non-finite), or `sigma_s`/`sigma_r` is
        positive but too small to remain positive once converted to
        OpenCV's `float32` parameter (e.g. `1e-46`).
    TypeError
        If `image` does not have dtype ``uint8``, or `sigma_s`/`sigma_r` is
        not a real number (rejecting `bool`).
    """
    _require_valid_photo_image(image)
    sigma_s = _validated_sigma_s(sigma_s)
    sigma_r = _validated_sigma_r(sigma_r)

    result = cv2.detailEnhance(image, sigma_s=sigma_s, sigma_r=sigma_r)
    return cast(ImageU8, result)
