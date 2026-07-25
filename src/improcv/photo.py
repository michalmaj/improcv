"""Photo and creative operations: stylization, enhancement, and image compositing."""

from __future__ import annotations

from typing import Literal, NamedTuple, cast

import cv2
import numpy as np

from improcv._validation import (
    require_dtype,
    require_fits_dtype,
    require_image_ndim,
    require_integral,
    require_non_negative,
    require_one_of,
    require_positive,
    require_spatial_mask,
)
from improcv.types import ImageU8, Mask

__all__ = [
    "detail_enhance",
    "pencil_sketch",
    "seamless_clone",
    "stylize",
    "PencilSketchResult",
    "SeamlessCloneMode",
]

# Documented, OpenCV-supported ranges (modules/photo/include/opencv2/photo.hpp) --
# not arbitrary improcv restrictions. Values outside these are unsupported by
# OpenCV's own API contract: the parameters are stored as C++ `float`, so an
# extreme-but-technically-finite Python value can degrade to `inf` after that
# conversion, silently producing a useless result rather than a clear error.
_SIGMA_S_MAX = 200.0
_SIGMA_R_MAX = 1.0
_SHADE_FACTOR_MAX = 0.1


def _require_valid_photo_image(image: np.ndarray, *, name: str = "image") -> None:
    """Raise ValueError/TypeError unless `image` is a valid 3-channel BGR `uint8` image.

    Grayscale (2D or `(H, W, 1)`), 2-channel, and BGRA (`(H, W, 4)`) input are
    all rejected -- verified directly that `cv2.stylization`/`cv2.pencilSketch`
    do not validate channel count at all: grayscale/2-channel input is silently
    misinterpreted (producing an incorrect, non-error result), and BGRA input
    can crash the process outright (observed directly, on both OpenCV 4.13 and
    5.0, for `cv2.stylization`/`cv2.pencilSketch` at specific image sizes).
    Unsupported channel layouts are rejected before the OpenCV call to avoid
    raw errors and build-dependent unsafe behavior observed for BGRA inputs.

    `name` only customizes the raised messages, for callers with more than
    one image-shaped parameter (e.g. `seamless_clone`'s `source`/
    `destination`) -- it does not change the validation itself.
    """
    if image.ndim == 2:
        raise ValueError(
            f"{name} must be a 3-channel BGR image, got a 2D grayscale image -- "
            f"convert first with improcv.ensure_bgr({name})"
        )
    require_image_ndim(image, ndims=(3,))
    if image.shape[2] != 3:
        if image.shape[2] == 4:
            raise ValueError(
                f"{name} must be a 3-channel BGR image, got a 4-channel (BGRA) image -- "
                f"explicitly drop (e.g. {name}[..., :3]) or composite the alpha channel "
                "onto a chosen background before calling; improcv.ensure_bgr does not "
                "accept BGRA"
            )
        if image.shape[2] == 1:
            raise ValueError(
                f"{name} must be a 3-channel BGR image, got a single-channel (grayscale) "
                f"image with an explicit trailing axis -- improcv.ensure_bgr itself does "
                f"not accept (H, W, 1); drop the axis first with "
                f"improcv.ensure_bgr({name}[..., 0])"
            )
        raise ValueError(
            f"{name} must be a 3-channel BGR image, got {image.shape[2]} channels -- "
            "there is no supported conversion for this channel count"
        )
    require_dtype(image, (np.uint8,), name)


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


SeamlessCloneMode = Literal["normal", "mixed", "monochrome_transfer"]
_SEAMLESS_CLONE_MODES: dict[SeamlessCloneMode, int] = {
    "normal": cv2.NORMAL_CLONE,
    "mixed": cv2.MIXED_CLONE,
    "monochrome_transfer": cv2.MONOCHROME_TRANSFER,
}

# Verified directly, on both OpenCV 4.13 and 5.0: a mask whose active
# (non-zero, post-border) bounding box has a width or height below this
# raises an opaque, unhelpful native exception from cv2.seamlessClone
# (sometimes just the string "vector", with no file/line information) --
# rejected explicitly instead.
_SEAMLESS_CLONE_MIN_ROI = 3


def _validated_clone_center(center: object) -> tuple[int, int]:
    """Raise TypeError/ValueError unless `center` is a valid `(x, y)` point for
    `seamless_clone`, else return it as a plain `(int, int)` tuple.

    Unlike `improcv._validation.require_point_2d` (used for float-accepting
    parameters like `rotate`'s `center`), this requires each coordinate to
    be integral -- rejecting `bool` and any `float`, including a whole
    number like `50.0` -- and to fit within a signed 32-bit `int`, matching
    `cv2.seamlessClone`'s own `cv2.Point` binding for this parameter
    (verified directly that the raw binding itself rejects a `float`/`bool`
    coordinate with an unhelpful `TypeError`-like `cv2.error`).
    """
    if not isinstance(center, tuple) or len(center) != 2:
        raise ValueError(f"center must be a 2-tuple, got {center!r}")
    x, y = center
    require_integral(x, "center[0]")
    require_integral(y, "center[1]")
    require_fits_dtype(x, np.int32, "center[0]")
    require_fits_dtype(y, np.int32, "center[1]")
    return int(x), int(y)


def seamless_clone(
    source: ImageU8,
    destination: ImageU8,
    mask: Mask,
    center: tuple[int, int],
    *,
    mode: SeamlessCloneMode = "normal",
) -> ImageU8:
    """Blend the masked region of `source` into `destination` with Poisson gradient-domain cloning.

    Unlike alpha blending or a plain copy-paste, this reconstructs the
    pasted region from *gradients*: it matches `source`'s gradients inside
    the mask while keeping `destination`'s own values as the boundary
    condition. A flat-colored `source` region pasted onto a flat-colored
    `destination` can therefore produce little or no visible change
    (verified directly: zero gradient in means zero gradient out), and even
    a fully ``255`` mask does not reproduce `source` exactly (verified
    directly: up to 1 level of difference from a plain copy, on both
    OpenCV versions). This is not alpha-blending or direct pixel
    replacement.

    Parameters
    ----------
    source : np.ndarray
        A `uint8` BGR image, shape ``(H, W, 3)``. Grayscale, `(H, W, 1)`,
        2-channel, and BGRA are all rejected -- see `Raises`. May be a
        different size than `destination`; only the region selected by
        `mask` matters, not the rest of `source`. Not modified.
    destination : np.ndarray
        A `uint8` BGR image, shape ``(H, W, 3)``. Same rejections as
        `source` -- notably, verified directly that a 1-channel
        `destination` can crash the OpenCV process outright (a raw,
        non-Python-catchable segfault on OpenCV 4.13; nondeterministically
        either a crash or a raw `cv2.error` on OpenCV 5.0), which is why
        this is rejected before ever reaching OpenCV rather than relying
        on a caught exception. Not modified.
    mask : np.ndarray
        A `uint8` mask, shape identical to `source.shape[:2]`. Only ``0``
        and ``255`` are accepted -- see `Raises`. OpenCV always ignores
        the outermost 1-pixel border of `mask`, regardless of its value
        (verified directly on both OpenCV versions); the bounding box of
        the remaining non-zero region (after that border is zeroed) must
        be at least ``3x3`` pixels, or this raises `ValueError` before
        calling OpenCV -- see `_SEAMLESS_CLONE_MIN_ROI`. Always passed to
        OpenCV as a copy: verified directly that OpenCV's own
        implementation mutates a single-channel mask array in place
        (zeroing its border, then eroding its active region), so
        `seamless_clone` never passes the caller's own `mask` array
        directly to it. Not modified.
    center : tuple[int, int]
        ``(x, y)`` in `destination`'s coordinate system. For all three
        supported modes, this is the point where the center of the
        bounding box of `mask`'s non-zero region (after its border is
        zeroed, see `mask`) is placed in `destination` -- not the center
        of all of `source`, and not the center of all of `mask`. For an
        even-sized bounding-box dimension, placement uses integer floor
        division (``roi_width // 2`` / ``roi_height // 2``), so the exact
        geometric center of the placed region can fall between pixels.
        Each coordinate must be a Python `int` or a NumPy integer scalar
        (rejecting `bool` and any `float`, even a whole number like
        `50.0`), within the range of a signed 32-bit `int`.
    mode : {"normal", "mixed", "monochrome_transfer"}, optional
        Cloning method:

        - ``"normal"``: reconstructs the region from `source`'s own
          gradients inside the mask.
        - ``"mixed"``: at each pixel, uses whichever of `source`'s or
          `destination`'s gradient is stronger -- OpenCV's "mixed seamless
          cloning", intended for masks that only loosely follow an
          object's outline.
        - ``"monochrome_transfer"``: transfers `source`'s luminance
          structure (converted to grayscale first) into `destination`,
          rather than its color.

        The `*_WIDE` variants (a different ROI-placement geometry, see
        `center`) are not exposed. Default `"normal"`.

    Returns
    -------
    np.ndarray
        Shape and dtype of `destination`. A new, independent array; never
        shares memory with `source`, `destination`, or `mask`. If `mask`
        has no non-zero pixels remaining once its border is zeroed
        (including an all-zero `mask`, or one active only on that border),
        this is an exact copy of `destination` (verified directly on both
        OpenCV versions), returned without calling OpenCV -- in that case,
        `center` is validated for type/range but not against
        `destination`'s bounds.

    Raises
    ------
    ValueError
        If `source`/`destination` is empty, does not have exactly 3
        dimensions, does not have exactly 3 channels, `mask` does not have
        exactly 2 dimensions, does not match `source`'s spatial shape, or
        contains a value other than `0`/`255`, a coordinate of `center` is
        outside the range of a signed 32-bit `int`, the bounding box of
        `mask`'s active region (after zeroing its border) is smaller than
        `3x3` pixels, placing that bounding box at `center` would put it
        partially or fully outside `destination`, or `mode` is not one of
        the accepted values.
    TypeError
        If `source`/`destination` does not have dtype ``uint8``, `mask`
        does not have dtype ``uint8``, or a coordinate of `center` is not
        an integer (rejecting `bool` and `float`).
    """
    require_one_of(mode, tuple(_SEAMLESS_CLONE_MODES), "mode")
    _require_valid_photo_image(source, name="source")
    _require_valid_photo_image(destination, name="destination")
    require_spatial_mask(mask, source, "mask")
    if np.any((mask != 0) & (mask != 255)):
        raise ValueError("mask must contain only 0 and 255")
    center_x, center_y = _validated_clone_center(center)

    effective_mask = mask.copy()
    effective_mask[0, :] = 0
    effective_mask[-1, :] = 0
    effective_mask[:, 0] = 0
    effective_mask[:, -1] = 0
    ys, xs = np.nonzero(effective_mask)
    if ys.size == 0:
        return cast(ImageU8, destination.copy())

    roi_x = int(xs.min())
    roi_y = int(ys.min())
    roi_width = int(xs.max()) - roi_x + 1
    roi_height = int(ys.max()) - roi_y + 1
    if roi_width < _SEAMLESS_CLONE_MIN_ROI or roi_height < _SEAMLESS_CLONE_MIN_ROI:
        raise ValueError(
            "mask's active region (after zeroing its 1-pixel border) must be at "
            f"least {_SEAMLESS_CLONE_MIN_ROI}x{_SEAMLESS_CLONE_MIN_ROI} pixels, "
            f"got {roi_width}x{roi_height}"
        )

    dst_x = center_x - roi_width // 2
    dst_y = center_y - roi_height // 2
    dst_height, dst_width = destination.shape[:2]
    if dst_x < 0:
        raise ValueError(
            f"center places the mask's active region {-dst_x} pixel(s) past destination's left edge"
        )
    if dst_y < 0:
        raise ValueError(
            f"center places the mask's active region {-dst_y} pixel(s) past destination's top edge"
        )
    if dst_x + roi_width > dst_width:
        raise ValueError(
            "center places the mask's active region "
            f"{dst_x + roi_width - dst_width} pixel(s) past destination's right edge"
        )
    if dst_y + roi_height > dst_height:
        raise ValueError(
            "center places the mask's active region "
            f"{dst_y + roi_height - dst_height} pixel(s) past destination's bottom edge"
        )

    result = cv2.seamlessClone(
        source,
        destination,
        mask.copy(),
        (center_x, center_y),
        _SEAMLESS_CLONE_MODES[mode],
    )
    return cast(ImageU8, result)
