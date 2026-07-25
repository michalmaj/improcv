"""HDR-related operations: exposure fusion, radiance merging, and tone mapping."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import cv2
import numpy as np

from improcv._validation import require_dtype, require_non_negative
from improcv.types import ImageFloat32, ImageU8

__all__ = [
    "fuse_exposures",
]

_MIN_EXPOSURES = 2


def _require_valid_exposure_image(image: np.ndarray, name: str) -> None:
    """Raise ValueError/TypeError unless `image` is a valid grayscale or BGR
    `uint8` exposure frame.

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
    require_dtype(image, (np.uint8,), name)


def _require_valid_exposure_stack(images: object) -> list[np.ndarray]:
    """Raise ValueError/TypeError unless `images` is a valid exposure stack,
    else return it as a plain `list`.

    `images` must be a real `collections.abc.Sequence` -- a single
    `np.ndarray` (including a 4D stack), a `str`/`bytes`, or a
    generator/iterator (none of which implement the `Sequence` protocol) are
    all rejected, even though OpenCV's own Python binding happens to accept
    a 4D array in place of a list. Every element must be a non-empty,
    `uint8`, 2D grayscale or 3D BGR `np.ndarray` with exactly the same shape
    as `images[0]` -- mixing grayscale and BGR frames in one stack is
    therefore rejected as a shape mismatch. Every error message names the
    offending index (e.g. ``images[3]``).
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
    _require_valid_exposure_image(first, "images[0]")

    for index, image in enumerate(normalized[1:], start=1):
        name = f"images[{index}]"
        if not isinstance(image, np.ndarray):
            raise TypeError(f"{name} must be a NumPy array, got {type(image).__name__}")
        if image.shape != first.shape:
            raise ValueError(
                f"{name} has shape {image.shape}, expected {first.shape} (matching images[0])"
            )
        require_dtype(image, (np.uint8,), name)

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


def fuse_exposures(
    images: Sequence[ImageU8],
    *,
    contrast_weight: float = 1.0,
    saturation_weight: float = 1.0,
    exposure_weight: float = 0.0,
) -> ImageFloat32:
    """Fuse a stack of differently-exposed images into one well-exposed image.

    Implemented via OpenCV's Mertens exposure fusion (`cv2.createMergeMertens`).
    Unlike a future radiance HDR merge (Debevec/Robertson), this does **not**
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
        If OpenCV's `MergeMertens` produces a non-finite (`NaN`/`inf`)
        result for the given images and weights -- verified directly that
        an extreme but individually representable weight (e.g.
        `contrast_weight=1e10`) can trigger this on both OpenCV 4.13 and
        5.0, with no warning or error from OpenCV itself.

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
    if not np.all(np.isfinite(result)):
        raise RuntimeError(
            "MergeMertens did not produce a finite result for the given images and weights"
        )
    return cast(ImageFloat32, result)
