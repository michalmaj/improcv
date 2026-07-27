"""Geometric augmentation: reproducible flip and crop sampling for image + mask pairs.

This module separates *sampling* random parameters from *applying* them:
`sample_flip`/`sample_crop` consume an explicit `np.random.Generator` once and
return a small, independent, replayable result (`FlipParameters`/
`CropParameters`); `apply_flip`/`apply_crop` are pure functions of that result
and never touch any RNG themselves. The same sampled parameters can be
applied to an image and its segmentation mask (or to a second image of the
same spatial size) any number of times, always producing the same result.

Only flip and crop are covered here. Affine transforms (rotation,
translation, scale, shear), perspective, resize, photometric augmentation,
bounding boxes/keypoints/polygons, and any `Compose`-style pipeline are
deliberately out of scope for this slice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import overload

import cv2
import numpy as np

from improcv._validation import (
    require_bool,
    require_dtype,
    require_image_ndim,
    require_int,
    require_positive_integral,
    require_range,
)
from improcv.transforms import FlipDirection
from improcv.transforms import crop as _crop
from improcv.transforms import flip as _flip
from improcv.types import Image

__all__ = [
    "AugmentedImageMask",
    "CropParameters",
    "FlipParameters",
    "apply_crop",
    "apply_flip",
    "sample_crop",
    "sample_flip",
]

# Segmentation masks get a deliberately narrower dtype contract than images:
# verified directly (source and empirically, across OpenCV 4.9/4.13/5.0) that
# cv2.warpAffine+INTER_NEAREST also accepts int32 (labels preserved exactly)
# but silently downcasts int64 to int32, and outright rejects bool -- this
# slice stays with the same 3 integer dtypes that comfortably cover real
# label counts (uint16 alone already covers up to 65535 classes) rather than
# threading a wider, version-verified set through every call site. Widening
# to int32 later is a compatible extension, not a breaking one.
_MASK_DTYPES = (np.uint8, np.uint16, np.int16)

# Mirrors transforms._GEOMETRIC_DTYPES exactly (uint8/uint16/int16/float32/
# float64) -- duplicated here, not imported, since transforms.py's constant is
# module-private and every other module in this project defines its own
# dtype tuple rather than reaching into a sibling module's internals. Kept in
# sync deliberately: both describe the same OpenCV-verified dtype contract
# for the warpAffine/flip family of operations.
_IMAGE_DTYPES = (np.uint8, np.uint16, np.int16, np.float32, np.float64)

_INTP_MAX = int(np.iinfo(np.intp).max)


@dataclass(frozen=True, slots=True)
class FlipParameters:
    """The result of `sample_flip`: which axes to flip.

    Both fields are always exactly `bool`. `horizontal=True, vertical=True`
    means both axes are flipped (equivalent to `cv2.flip`'s `flipCode=-1`,
    verified directly to be byte-identical to flipping each axis in
    sequence); both `False` means no flip -- `apply_flip` still returns an
    independent copy in that case, never the original array.
    """

    horizontal: bool
    vertical: bool


@dataclass(frozen=True, slots=True)
class CropParameters:
    """The result of `sample_crop`: a crop rectangle plus the source size it was sampled for.

    `source_size` is `(width, height)`, matching `crop_size`/`source_size`'s
    own parameter order in `sample_crop`. It exists to make replay safe:
    `apply_crop` requires the image (and mask, if given) to have exactly this
    spatial size, so parameters sampled for one image can't be silently
    misapplied to a differently-sized one.
    """

    x: int
    y: int
    width: int
    height: int
    source_size: tuple[int, int]


@dataclass(frozen=True, slots=True, eq=False)
class AugmentedImageMask:
    """The image+mask result of `apply_flip`/`apply_crop` when called with a `mask`.

    Equality (`==`) compares both fields by value via `np.array_equal`, never
    by identity -- the default dataclass-generated equality would compare
    `image`/`mask` with `==` directly and hit NumPy's "truth value of an
    array is ambiguous" error for any non-trivial array. Instances are
    unhashable (`hash()` raises `TypeError`). Unlike `ConfusionMatrixResult`/
    `ClassificationMetrics` in `improcv.evaluation`, `image`/`mask` here are
    ordinary, writeable pipeline data, not a final numeric report -- neither
    field is marked read-only.
    """

    image: Image
    mask: np.ndarray

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AugmentedImageMask):
            return NotImplemented
        return bool(np.array_equal(self.image, other.image)) and bool(
            np.array_equal(self.mask, other.mask)
        )

    __hash__ = None  # type: ignore[assignment]


def sample_flip(
    rng: np.random.Generator,
    *,
    horizontal_probability: float = 0.5,
    vertical_probability: float = 0.0,
) -> FlipParameters:
    """Sample independent horizontal/vertical flip decisions.

    `rng` must be an actual `numpy.random.Generator` instance (checked by
    `isinstance`, not duck-typed) -- a legacy `numpy.random.RandomState`, a
    bare integer seed, the `numpy.random` module itself, or any object that
    merely implements a compatible interface are all rejected. This function
    consumes the given `rng`; `horizontal` and `vertical` are sampled
    independently. The exact number and order of internal draws against
    `rng` is an implementation detail, not part of the public contract, and
    may change between releases without notice -- what's guaranteed is that
    replaying a saved `FlipParameters` (via `apply_flip`) reproduces the same
    result, not that any particular `rng` state does.

    `horizontal_probability`/`vertical_probability` must each be a finite
    Python or NumPy real scalar (not `bool`/`np.bool_`) in the closed range
    `[0.0, 1.0]`; `0.0` (never flip that axis) and `1.0` (always flip it) are
    both legal. The two axes are sampled independently -- both landing on
    `True` means both axes are flipped.

    Returns
    -------
    FlipParameters
        Independent of `rng`'s state after this call; reusable any number of
        times via `apply_flip`.

    Raises
    ------
    TypeError
        If `rng` is not a `numpy.random.Generator`, or a probability is not
        a real number or is a `bool`/`np.bool_`.
    ValueError
        If a probability is outside `[0.0, 1.0]` or is `NaN`/infinite.
    """
    _require_generator(rng)
    require_range(horizontal_probability, 0.0, 1.0, "horizontal_probability")
    require_range(vertical_probability, 0.0, 1.0, "vertical_probability")

    horizontal = bool(rng.random() < horizontal_probability)
    vertical = bool(rng.random() < vertical_probability)
    return FlipParameters(horizontal=horizontal, vertical=vertical)


@overload
def apply_flip(
    image: Image,
    params: FlipParameters,
    *,
    mask: None = None,
) -> Image: ...
@overload
def apply_flip(
    image: Image,
    params: FlipParameters,
    *,
    mask: np.ndarray,
) -> AugmentedImageMask: ...
def apply_flip(
    image: Image,
    params: FlipParameters,
    *,
    mask: np.ndarray | None = None,
) -> Image | AugmentedImageMask:
    """Apply a previously sampled flip to `image` (and optionally `mask`).

    `params` must be exactly a `FlipParameters` (its fields are re-validated
    here too, since a frozen dataclass can still be constructed by hand with
    invalid field values -- `dict`/`tuple`/another dataclass with similar
    fields are all rejected, never duck-typed). `image` follows the exact
    same contract as `improcv.transforms.flip` (dtype, shape, `cv2.error`
    mapping); when both `params.horizontal` and `params.vertical` are
    `False`, this still returns an independent copy of `image`, never
    `image` itself, and never applies two sequential flips when one call
    with `direction="both"` is equivalent (verified directly, byte-identical
    for both 2-D and multi-channel layouts).

    If `mask` is given, it must be a non-empty `ndarray` with shape
    `(H, W)` or `(H, W, 1)` matching `image`'s spatial size exactly, and
    dtype `uint8`, `uint16`, or `int16` -- `bool`, `int32`, `int64`, and any
    floating-point dtype are rejected in this slice (see `_MASK_DTYPES`).
    The same flip is applied to `mask` via the same
    `improcv.transforms.flip`, and the result is returned as an
    `AugmentedImageMask` instead of a bare `Image`.

    Returns
    -------
    Image or AugmentedImageMask
        A new, independent array (or pair of arrays) with the same shape
        and dtype as the input(s); never aliases `image` or `mask`.

    Raises
    ------
    TypeError
        If `params` is not a `FlipParameters`, if its fields are not
        `bool`, if `image`/`mask` is not dtype-compatible, or if `mask` is
        not an `ndarray`.
    ValueError
        If `image`/`mask` has an unsupported shape, or `mask`'s spatial
        size does not match `image`'s.
    RuntimeError
        If the underlying `cv2.error` occurs after full validation, or if
        this function's own postconditions are violated.
    """
    _require_flip_parameters(params)
    require_image_ndim(image, ndims=(2, 3))
    require_dtype(image, _IMAGE_DTYPES, "image")

    direction = _flip_direction(params)
    augmented_image = _apply_flip_preserving_shape(image, direction)
    _check_flip_postconditions(image, augmented_image, "image")

    if mask is None:
        return augmented_image

    _require_mask(mask, "mask")
    _require_matching_spatial_shape(mask, image, "mask", "image")
    augmented_mask = _apply_flip_preserving_shape(mask, direction)
    _check_flip_postconditions(mask, augmented_mask, "mask")

    return AugmentedImageMask(image=augmented_image, mask=augmented_mask)


def sample_crop(
    rng: np.random.Generator,
    source_size: tuple[int, int],
    crop_size: tuple[int, int],
) -> CropParameters:
    """Sample a uniformly random crop rectangle of `crop_size` within `source_size`.

    Both `source_size` and `crop_size` are `(width, height)`. Each component
    must be a positive Python or NumPy integral scalar (`bool` rejected)
    representable as `np.intp` on this platform. `crop_size` must not exceed
    `source_size` along either axis; a `crop_size` equal to `source_size`
    deterministically yields `x=0, y=0` (the only legal position).

    `rng` must be an actual `numpy.random.Generator` (same contract as
    `sample_flip`). The crop position is drawn so that both the right and
    bottom edges of `source_size` are reachable, not just a top-left-biased
    range. As with `sample_flip`, the exact number and order of internal
    draws against `rng` is an implementation detail, not part of the public
    contract, and may change between releases without notice.

    Returns
    -------
    CropParameters
        Independent of `rng`'s state after this call; `source_size` is
        stored (as plain Python `int`s) alongside the crop rectangle so
        `apply_crop` can refuse to replay it against a differently-sized
        image.

    Raises
    ------
    TypeError
        If `rng` is not a `numpy.random.Generator`, or `source_size`/
        `crop_size` is not a 2-tuple of positive integral (non-`bool`)
        values.
    ValueError
        If any dimension is not representable as `np.intp` on this
        platform, or `crop_size` exceeds `source_size` along either axis.
    """
    _require_generator(rng)
    source_width, source_height = _normalize_size(source_size, "source_size")
    crop_width, crop_height = _normalize_size(crop_size, "crop_size")

    if crop_width > source_width:
        raise ValueError(
            f"crop_size width ({crop_width}) must not exceed source_size width ({source_width})"
        )
    if crop_height > source_height:
        raise ValueError(
            f"crop_size height ({crop_height}) must not exceed source_size height ({source_height})"
        )

    max_x = source_width - crop_width
    max_y = source_height - crop_height
    x = int(rng.integers(0, max_x + 1))
    y = int(rng.integers(0, max_y + 1))

    return CropParameters(
        x=x,
        y=y,
        width=crop_width,
        height=crop_height,
        source_size=(source_width, source_height),
    )


@overload
def apply_crop(
    image: Image,
    params: CropParameters,
    *,
    mask: None = None,
) -> Image: ...
@overload
def apply_crop(
    image: Image,
    params: CropParameters,
    *,
    mask: np.ndarray,
) -> AugmentedImageMask: ...
def apply_crop(
    image: Image,
    params: CropParameters,
    *,
    mask: np.ndarray | None = None,
) -> Image | AugmentedImageMask:
    """Apply a previously sampled crop to `image` (and optionally `mask`).

    `params` must be exactly a `CropParameters` (fields re-validated here
    too, for the same reason as `apply_flip`). `image`'s spatial size
    (`(width, height)`) must equal `params.source_size` *exactly* -- this is
    the replay guard: parameters sampled for one image size are refused
    against any other size, rather than silently cropping the wrong region.
    Applies `improcv.transforms.crop` unmodified (no slicing logic is
    reimplemented here); `image`'s own dtype/shape contract is exactly
    `crop`'s (crop does not restrict dtype).

    If `mask` is given, it must satisfy the same contract as
    `apply_flip`'s `mask` (shape `(H, W)`/`(H, W, 1)`, dtype `uint8`/
    `uint16`/`int16`, spatial size matching `image`) and is cropped with the
    same rectangle via the same `improcv.transforms.crop`.

    Returns
    -------
    Image or AugmentedImageMask
        A new, independent array (or pair) of shape
        `(params.height, params.width, ...)`; never aliases `image` or
        `mask`. `improcv.transforms.crop` already always copies, including
        when the crop covers the entire source image.

    Raises
    ------
    TypeError
        If `params` is not a `CropParameters`, if its fields are not the
        expected types, or if `mask` is not an `ndarray` of an accepted
        dtype.
    ValueError
        If `image`/`mask` does not have 2 or 3 dimensions, if `image`'s (or
        `mask`'s) spatial size does not match `params.source_size` (or
        `image`'s), or if `params`' fields are inconsistent (out of range,
        or the crop rectangle does not fit `params.source_size`).
    RuntimeError
        If this function's own postconditions are violated.
    """
    _require_crop_parameters(params)
    require_image_ndim(image, ndims=(2, 3))
    _require_matches_source_size(image, params, "image")

    augmented_image = _crop(image, params.x, params.y, params.width, params.height)
    _check_crop_postconditions(augmented_image, params, image, "image")

    if mask is None:
        return augmented_image

    _require_mask(mask, "mask")
    _require_matching_spatial_shape(mask, image, "mask", "image")
    augmented_mask = _crop(mask, params.x, params.y, params.width, params.height)
    _check_crop_postconditions(augmented_mask, params, mask, "mask")

    return AugmentedImageMask(image=augmented_image, mask=augmented_mask)


def _require_generator(rng: object) -> None:
    if not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be a numpy.random.Generator, got {type(rng).__name__}")


def _normalize_size(value: object, name: str) -> tuple[int, int]:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple, got {type(value).__name__}")
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly 2 elements, got {len(value)}")
    width, height = value
    require_positive_integral(width, f"{name}[0]")
    require_positive_integral(height, f"{name}[1]")
    width_int, height_int = int(width), int(height)
    if width_int > _INTP_MAX or height_int > _INTP_MAX:
        raise ValueError(f"{name} dimensions must fit in a signed intp (<= {_INTP_MAX})")
    return (width_int, height_int)


def _apply_flip_preserving_shape(array: np.ndarray, direction: FlipDirection | None) -> np.ndarray:
    if direction is None:
        return array.copy()
    try:
        result = _flip(array, direction)
    except cv2.error as exc:
        raise RuntimeError("OpenCV failed to apply flip augmentation") from exc
    if array.ndim == 3 and array.shape[2] == 1 and result.shape == array.shape[:2]:
        # cv2.flip drops a trailing singleton channel dimension (verified
        # directly for (H, W, 1) input) -- restore it so the output shape
        # always matches the input shape exactly, per this module's own
        # shape-preservation postcondition. Restricted to exactly this known
        # squeeze (a singleton channel dim disappearing), not merely "same
        # total element count" -- a wrong-shaped result with a coincidentally
        # matching size (e.g. a (H, W*C) result for a (H, W, C) input) must
        # still be left for _check_flip_postconditions to reject with a clear
        # RuntimeError, not silently reshaped into something plausible-looking.
        result = result[:, :, None]
    return result


def _flip_direction(params: FlipParameters) -> FlipDirection | None:
    if params.horizontal and params.vertical:
        return "both"
    if params.horizontal:
        return "horizontal"
    if params.vertical:
        return "vertical"
    return None


def _require_flip_parameters(params: object) -> None:
    if not isinstance(params, FlipParameters):
        raise TypeError(f"params must be a FlipParameters, got {type(params).__name__}")
    require_bool(params.horizontal, "params.horizontal")
    require_bool(params.vertical, "params.vertical")


def _require_crop_parameters(params: object) -> None:
    if not isinstance(params, CropParameters):
        raise TypeError(f"params must be a CropParameters, got {type(params).__name__}")
    require_int(params.x, "params.x")
    require_int(params.y, "params.y")
    require_int(params.width, "params.width")
    require_int(params.height, "params.height")

    source_size = params.source_size
    if not (
        isinstance(source_size, tuple)
        and len(source_size) == 2
        and all(isinstance(v, int) and not isinstance(v, bool) for v in source_size)
    ):
        raise TypeError(f"params.source_size must be a 2-tuple of int, got {source_size!r}")

    if params.width <= 0:
        raise ValueError(f"params.width must be positive, got {params.width}")
    if params.height <= 0:
        raise ValueError(f"params.height must be positive, got {params.height}")
    if params.x < 0:
        raise ValueError(f"params.x must be non-negative, got {params.x}")
    if params.y < 0:
        raise ValueError(f"params.y must be non-negative, got {params.y}")

    source_width, source_height = source_size
    if source_width <= 0 or source_height <= 0:
        raise ValueError(f"params.source_size must be positive, got {source_size}")
    if params.x + params.width > source_width:
        raise ValueError(
            f"params.x + params.width ({params.x + params.width}) exceeds "
            f"params.source_size width ({source_width})"
        )
    if params.y + params.height > source_height:
        raise ValueError(
            f"params.y + params.height ({params.y + params.height}) exceeds "
            f"params.source_size height ({source_height})"
        )


def _require_matches_source_size(image: np.ndarray, params: CropParameters, name: str) -> None:
    source_width, source_height = params.source_size
    image_height, image_width = image.shape[:2]
    if (image_width, image_height) != (source_width, source_height):
        raise ValueError(
            f"{name} spatial size {(image_width, image_height)} does not match "
            f"params.source_size {(source_width, source_height)}"
        )


def _require_mask(mask: object, name: str) -> None:
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array, got {type(mask).__name__}")
    if mask.ndim not in (2, 3):
        raise ValueError(f"{name} must have 2 or 3 dimensions, got {mask.ndim}")
    if mask.ndim == 3 and mask.shape[2] != 1:
        raise ValueError(f"{name} must have shape (H, W) or (H, W, 1), got {mask.shape}")
    if mask.size == 0:
        raise ValueError(f"{name} must not be empty, got shape {mask.shape}")
    require_dtype(mask, _MASK_DTYPES, name)


def _require_matching_spatial_shape(a: np.ndarray, b: np.ndarray, name_a: str, name_b: str) -> None:
    if a.shape[:2] != b.shape[:2]:
        raise ValueError(
            f"{name_a} must have the same spatial size as {name_b}, got {a.shape[:2]} "
            f"vs {b.shape[:2]}"
        )


def _check_flip_postconditions(original: np.ndarray, result: np.ndarray, name: str) -> None:
    if result.shape != original.shape:
        raise RuntimeError(
            f"internal error: {name} shape changed from {original.shape} to {result.shape}"
        )
    if result.dtype != original.dtype:
        raise RuntimeError(
            f"internal error: {name} dtype changed from {original.dtype} to {result.dtype}"
        )
    if np.shares_memory(result, original):
        raise RuntimeError(f"internal error: {name} output aliases input")


def _check_crop_postconditions(
    result: np.ndarray, params: CropParameters, original: np.ndarray, name: str
) -> None:
    expected_shape = (params.height, params.width) + original.shape[2:]
    if result.shape != expected_shape:
        raise RuntimeError(
            f"internal error: {name} crop has shape {result.shape}, expected {expected_shape}"
        )
    if result.dtype != original.dtype:
        raise RuntimeError(
            f"internal error: {name} crop dtype changed from {original.dtype} to {result.dtype}"
        )
    if np.shares_memory(result, original):
        raise RuntimeError(f"internal error: {name} crop output aliases input")
