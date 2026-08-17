"""Geometric augmentation: reproducible flip, crop, affine, and perspective sampling for
image + mask pairs.

This module separates *sampling* random parameters from *applying* them:
`sample_flip`/`sample_crop`/`sample_affine`/`sample_perspective` consume an
explicit `np.random.Generator` and return an independent, replayable
parameter object (`FlipParameters`/`CropParameters`/`AffineParameters`/
`PerspectiveParameters`); `apply_flip`/`apply_crop`/`apply_affine`/
`apply_perspective` are pure functions of that result and never touch any
RNG themselves. The same sampled parameters can be applied to an image and
its segmentation mask (or to a second image of the same spatial size) any
number of times, always producing the same result.

Affine coverage is a stable subset of the general affine group: sequential
shear, anisotropic axis scale, rotation with isotropic scale, and
translation, all composed around the image center. Perspective coverage is
a single, replayable homography sampled by displacing each of the source
rectangle's four corners inward, independently, within a bound controlled by
`distortion_scale`. Both `apply_affine` and `apply_perspective` render to
the source size by default (no canvas expansion); `expand_affine_canvas` and
`expand_perspective_canvas` are separate, deterministic, RNG-free grow-only
conversions that each expand their respective parameters' stored output size
(and adjust the matrix accordingly) so no transformed content is cropped --
resize, photometric augmentation, bounding boxes/keypoints/polygons, and any
`Compose`-style pipeline remain out of scope for this slice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import overload

import cv2
import numpy as np
import numpy.typing as npt

from improcv._validation import (
    require_bool,
    require_dtype,
    require_finite,
    require_fits_dtype,
    require_image_ndim,
    require_int,
    require_integral,
    require_point_2d,
    require_positive,
    require_positive_integral,
    require_range,
    require_transform_matrix,
)
from improcv.transforms import FlipDirection
from improcv.transforms import crop as _crop
from improcv.transforms import flip as _flip
from improcv.transforms import warp_affine as _warp_affine
from improcv.transforms import warp_perspective as _warp_perspective
from improcv.types import Image

__all__ = [
    "AffineParameters",
    "AugmentedImageMask",
    "CropParameters",
    "FlipParameters",
    "PerspectiveParameters",
    "apply_affine",
    "apply_crop",
    "apply_flip",
    "apply_perspective",
    "expand_affine_canvas",
    "expand_perspective_canvas",
    "sample_affine",
    "sample_crop",
    "sample_flip",
    "sample_perspective",
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

# apply_perspective's own, narrower mask dtype contract -- excludes int16.
# Verified directly via this project's own CI: cv2.warpPerspective with an
# int16 mask raises "Unknown C++ exception from OpenCV code" on Windows
# (opencv-python-headless==4.14.0.94), while the identical wheel version
# works correctly for the same call on Linux and macOS -- a genuine,
# platform-specific upstream OpenCV limitation, not something this project
# can fix. warpAffine is unaffected by this (verified: the same int16 mask
# through apply_affine works on all three platforms), so only apply_
# perspective's mask contract is narrowed; _MASK_DTYPES above is unchanged
# and still applies to flip/crop/affine.
_PERSPECTIVE_MASK_DTYPES = (np.uint8, np.uint16)

# Mirrors transforms._GEOMETRIC_DTYPES exactly (uint8/uint16/int16/float32/
# float64) -- duplicated here, not imported, since transforms.py's constant is
# module-private and every other module in this project defines its own
# dtype tuple rather than reaching into a sibling module's internals. Kept in
# sync deliberately: both describe the same OpenCV-verified dtype contract
# for the warpAffine/flip family of operations.
_IMAGE_DTYPES = (np.uint8, np.uint16, np.int16, np.float32, np.float64)

_INTP_MAX = int(np.iinfo(np.intp).max)

# cv2.warpAffine's dsize is parsed as a cv::Size, whose standard OpenCV
# specialization uses `int` fields -- verified directly, identically, against
# both OpenCV 4.9.0 (this project's floor) and 5.0.0: a dsize width/height of
# 2**31-1 is accepted, 2**31 raises cv2.error ("Overload resolution failed").
# _INTP_MAX (the platform's signed intp max, 2**63-1 on 64-bit) is far looser
# than this and is not the right bound for output_size specifically.
_OPENCV_SIZE_MAX = int(np.iinfo(np.int32).max)


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
class AffineParameters:
    """The result of `sample_affine`: a replayable affine transform composed from shear,
    anisotropic axis scale, rotation with isotropic scale, and translation.

    `matrix` (shape ``(2, 3)``, dtype ``float64``, finite, a new read-only
    buffer) is the sole source of truth for replay -- `apply_affine` applies
    it directly. `angle`/`translation`/`scale`/`shear`/`axis_scale` are
    sampling metadata only, recorded to make debugging, logging, and a
    readable `repr` easier; `apply_affine` never reconstructs the matrix
    from them, nor cross-checks the matrix against them beyond each field's
    own basic validity (finite, `scale > 0`, `axis_scale` components `> 0`).
    `source_size` is `(width, height)`, matching `CropParameters`'s own
    convention, and exists for the same reason: to make replay safe by
    refusing to reapply these parameters to a differently-sized image.

    `shear` is `(shear_x, shear_y)`, keyword-only so that the pre-existing
    five-positional-argument construction (`AffineParameters(matrix,
    source_size, angle, translation, scale)`) keeps working unchanged and
    `__match_args__` -- used for positional pattern matching -- stays
    exactly the five original field names; `shear` defaults to `(0.0, 0.0)`
    when omitted.

    `axis_scale` is `(axis_scale_x, axis_scale_y)`, also keyword-only for the
    same compatibility reason, defaulting to `(1.0, 1.0)` (no anisotropic
    deformation). Each component is a positive, dimensionless multiplier
    applied *on top of* `scale`, not a final axis scale by itself -- the
    actual realized per-axis scale is `scale * axis_scale[0]` (x) and
    `scale * axis_scale[1]` (y). `axis_scale` is never used to derive those
    effective scales for validation or replay; it exists purely as sampling
    metadata, exactly like `shear`.

    `output_size` is `(width, height)` or `None` (the default), also
    keyword-only for the same compatibility reason. `None` means `apply_affine`
    renders to `source_size`, exactly as before this field existed. A
    non-`None` value -- set by `expand_affine_canvas`, never by `sample_affine`
    -- is the explicit warp destination size; unlike `angle`/`translation`/
    `scale`/`shear`/`axis_scale`, `output_size` is *not* mere sampling
    metadata kept only for debugging/logging/`repr`: together with `matrix`
    it is part of the full source of truth `apply_affine` replays, since
    `matrix` alone no longer determines the destination canvas size once it
    can differ from `source_size`.
    """

    matrix: npt.NDArray[np.float64]
    source_size: tuple[int, int]
    angle: float
    translation: tuple[float, float]
    scale: float
    shear: tuple[float, float] = field(default=(0.0, 0.0), kw_only=True)
    axis_scale: tuple[float, float] = field(default=(1.0, 1.0), kw_only=True)
    output_size: tuple[int, int] | None = field(default=None, kw_only=True)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AffineParameters):
            return NotImplemented
        return (
            bool(np.array_equal(self.matrix, other.matrix))
            and self.source_size == other.source_size
            and self.angle == other.angle
            and self.translation == other.translation
            and self.scale == other.scale
            and self.shear == other.shear
            and self.axis_scale == other.axis_scale
            and self.output_size == other.output_size
        )

    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True, eq=False)
class PerspectiveParameters:
    """The result of `sample_perspective`: a full `3x3` projective transform matrix.

    `matrix` (shape ``(3, 3)``, dtype ``float64``, finite, a new read-only buffer) together with
    `output_size` (when not `None`) is the source of truth for replay -- `apply_perspective`
    applies `matrix` directly and renders to `output_size` if set, or to `source_size` otherwise.
    `destination_points` is sampling/debug metadata only: the four `(x, y)` destination corners
    actually used to build `matrix` via `cv2.getPerspectiveTransform`, in `top-left, top-right,
    bottom-right, bottom-left` order, recorded *after* the `float32` quantization that OpenCV
    itself requires for its input points -- so this is exactly what the solver saw, not the
    pre-quantization draw. `apply_perspective` never reconstructs `matrix` from
    `destination_points`, nor cross-checks the two numerically, mirroring `AffineParameters`' own
    metadata-is-not-truth contract. For parameters returned by `expand_perspective_canvas`,
    `destination_points` continues to describe the original `sample_perspective` draw and does
    not reflect the canvas-origin translation `expand_perspective_canvas` applies -- exactly as
    `AffineParameters.translation` remains silent about `expand_affine_canvas`'s own shift. The
    corresponding source corners are never stored -- they are always deterministically `(0, 0)`,
    `(width - 1, 0)`, `(width - 1, height - 1)`, `(0, height - 1)` for `source_size == (width,
    height)`, in the same corner order.

    `source_size` is `(width, height)`, matching `AffineParameters`'/`CropParameters`' own
    convention, and exists for the same reason: `apply_perspective` refuses to replay these
    parameters against a differently-sized image.

    `output_size` is `(width, height)` or `None` (the default), keyword-only so the pre-existing
    three-positional-argument construction (`PerspectiveParameters(matrix, source_size,
    destination_points)`) keeps working unchanged and `__match_args__` stays exactly the three
    original field names. `None` means `apply_perspective` renders to `source_size`, exactly as
    before this field existed; `sample_perspective` always produces `output_size=None`.
    `expand_perspective_canvas` is the canonical library operation that computes and sets a
    non-`None` value -- but a manually constructed `PerspectiveParameters` may also legally supply
    a validated non-`None` `output_size` keyword-only, exactly like `AffineParameters.output_size`.
    Unlike `destination_points`/etc., a non-`None` `output_size` is *not* mere sampling metadata:
    together with `matrix` it is part of the full source of truth `apply_perspective` replays,
    since `matrix` alone no longer determines the destination canvas size once it can differ from
    `source_size`.
    """

    matrix: npt.NDArray[np.float64]
    source_size: tuple[int, int]
    destination_points: tuple[
        tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]
    ]
    output_size: tuple[int, int] | None = field(default=None, kw_only=True)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PerspectiveParameters):
            return NotImplemented
        return (
            bool(np.array_equal(self.matrix, other.matrix))
            and self.source_size == other.source_size
            and self.destination_points == other.destination_points
            and self.output_size == other.output_size
        )

    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True, eq=False)
class AugmentedImageMask:
    """The image+mask result of `apply_flip`/`apply_crop`/`apply_affine`/`apply_perspective` when
    called with a `mask`.

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
    _check_shape_preserving_postconditions(image, augmented_image, "image")

    if mask is None:
        return augmented_image

    _require_mask(mask, "mask")
    _require_matching_spatial_shape(mask, image, "mask", "image")
    augmented_mask = _apply_flip_preserving_shape(mask, direction)
    _check_shape_preserving_postconditions(mask, augmented_mask, "mask")

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
    _require_matches_source_size(image, params.source_size, "image")

    augmented_image = _crop(image, params.x, params.y, params.width, params.height)
    _check_crop_postconditions(augmented_image, params, image, "image")

    if mask is None:
        return augmented_image

    _require_mask(mask, "mask")
    _require_matching_spatial_shape(mask, image, "mask", "image")
    augmented_mask = _crop(mask, params.x, params.y, params.width, params.height)
    _check_crop_postconditions(augmented_mask, params, mask, "mask")

    return AugmentedImageMask(image=augmented_image, mask=augmented_mask)


def sample_affine(
    rng: np.random.Generator,
    source_size: tuple[int, int],
    *,
    angle_range: tuple[float, float] = (0.0, 0.0),
    translation_x_range: tuple[float, float] = (0.0, 0.0),
    translation_y_range: tuple[float, float] = (0.0, 0.0),
    scale_range: tuple[float, float] = (1.0, 1.0),
    axis_scale_x_range: tuple[float, float] = (1.0, 1.0),
    axis_scale_y_range: tuple[float, float] = (1.0, 1.0),
    shear_x_range: tuple[float, float] = (0.0, 0.0),
    shear_y_range: tuple[float, float] = (0.0, 0.0),
) -> AffineParameters:
    """Sample a replayable affine transform with shear, anisotropic axis scale, rotation,
    isotropic scale, and translation.

    `source_size` is `(width, height)`. `angle_range` is in degrees, with
    the same positive (counter-clockwise) direction and center convention
    (`((width - 1) / 2, (height - 1) / 2)`) as `improcv.transforms.rotate`;
    it is not normalized modulo 360, so a range like `(350.0, 370.0)` is
    legal and meaningful as-is. `translation_x_range`/`translation_y_range`
    are in pixels (a positive `x` shifts content right, a positive `y`
    shifts it down, matching `improcv.transforms.translate`); float
    (subpixel) values are legal. `scale_range` is a positive, dimensionless,
    isotropic multiplier (`1.0` is unchanged size) applied identically to
    both axes. `axis_scale_x_range`/`axis_scale_y_range` are positive,
    dimensionless *axis multipliers* layered on top of `scale`, not final
    axis scales by themselves: the actual realized scale along each axis is
    `effective_scale_x = scale * axis_scale_x` and `effective_scale_y =
    scale * axis_scale_y`. `1.0` for both (the default) means no
    anisotropic deformation -- the transform is then purely isotropic, as
    it was before this parameter existed. `shear_x_range`/
    `shear_y_range` are raw, dimensionless shear coefficients (not degrees):
    `shear_x` maps `x' = x + shear_x * y` and `shear_y` (applied after
    `shear_x`) maps `y' = y + shear_y * x'`, both in the coordinate system
    centered on the same pivot as rotation/scale. There is no `abs(shear)`
    limit and no forbidden angle (unlike a degrees-based shear
    parameterization, this raw-coefficient form has no domain to avoid).
    The underlying `[[1, shear_x], [shear_y, 1 + shear_x*shear_y]]` matrix
    has determinant `1` mathematically for any finite `shear_x`/`shear_y` --
    but that is a statement about the exact real-number parameterization,
    not a promise of infinite `float64` precision: when `shear_x * shear_y`
    is large enough (roughly `2**52` in magnitude) that `1.0 +
    shear_x*shear_y` rounds back down to exactly `shear_x*shear_y`, the
    unit term that keeps the matrix invertible would be silently discarded,
    so that specific combination is rejected (see Raises below) rather than
    stored as a matrix that no longer matches the shear it was sampled as.

    Each range is a `(low, high)` tuple: a Python or NumPy real scalar pair
    (`bool`/`np.bool_` rejected), both finite, with `low <= high` (equal
    endpoints are legal and always sample that exact constant); `scale_range`
    and `axis_scale_x_range`/`axis_scale_y_range` additionally require
    `low > 0` (`shear_x_range`/`shear_y_range` have no such restriction --
    negative, zero, and positive coefficients are all legal). Every range is
    sampled independently via `rng.uniform(low, high)` -- `low` itself is
    reachable, but for a non-degenerate range, sampling a value exactly
    equal to `high` is not guaranteed (a property of continuous
    floating-point sampling, not a bug). As a compatibility guarantee for
    code written before shear (and, later, anisotropic scale) existed, a
    singleton `shear_x_range`/`shear_y_range`/`axis_scale_x_range`/
    `axis_scale_y_range` (`low == high`, including each parameter's own
    default) does not consume any `rng` state at all -- it never calls
    `rng.uniform` -- so an existing call site that never set these
    parameters keeps sampling `angle`/`translation`/`scale` from exactly the
    same `rng` state, call after call, as it did before shear and
    anisotropic scale were added.

    The transform is built as: shear x, then shear y, then anisotropic axis
    scale (all three around `source_size`'s center), then rotation +
    isotropic scale around that same center (via `cv2.getRotationMatrix2D`),
    then translated by `(dx, dy)` in the destination coordinate system.
    Shear does not commute with axis scale or rotation, and translation does
    not commute with the linear part in general, so this composition order
    is a fixed, documented part of the contract, not an implementation
    detail. When `shear_x`, `shear_y` sample to exactly `0.0` and
    `axis_scale_x`, `axis_scale_y` both sample to exactly `1.0` (all true
    for the defaults), the matrix is built via the original pre-shear code
    path with no extra matrix multiplication at all, so it is bit-for-bit
    identical to what this function produced before shear or anisotropic
    scale existed.

    `rng` must be an actual `numpy.random.Generator` instance (same contract
    as `sample_flip`/`sample_crop`); the exact number and order of internal
    draws is an implementation detail, not part of the public contract.

    Returns
    -------
    AffineParameters
        Independent of `rng`'s state after this call; `matrix` is the sole
        source of truth for replay via `apply_affine`. `angle`/
        `translation`/`scale`/`shear`/`axis_scale` are sampling metadata for
        debugging/logging/`repr` only.

    Raises
    ------
    TypeError
        If `rng` is not a `numpy.random.Generator`, or `source_size`/any
        `*_range` is not a 2-tuple of the expected element types.
    ValueError
        If `source_size`/any `*_range` has the wrong length or an
        out-of-contract value (non-finite, `low > high`, non-positive
        `scale_range`/`axis_scale_x_range`/`axis_scale_y_range`), if the
        sampled, otherwise-legal parameters combine into a non-finite
        matrix (representable only as `inf`/`NaN`, e.g. from an
        astronomically large `scale`/`shear`/`source_size` combination --
        verified directly reachable from finite inputs), if `scale *
        axis_scale_x` or `scale * axis_scale_y` is not representable as a
        finite, strictly positive `float64` (e.g. it overflows to `inf` or
        underflows to exactly `0.0` even though `scale` and the axis
        multiplier are each individually finite and positive), or if
        `shear_x * shear_y` is large enough that `float64` can no longer
        distinguish `1.0 + shear_x*shear_y` from `shear_x*shear_y` itself,
        silently losing the unit term the sequential shear matrix depends
        on for invertibility. There is no other limit on shear magnitude:
        a large, but still representable, shear coefficient is accepted
        even though it can strongly deform the image or push its content
        outside the canvas entirely -- that is not guarded against, and
        neither is a merely poorly-conditioned (but still representable)
        matrix in general; `1` is the exact determinant of the real-number
        parameterization, not a guarantee about the numerical stability of
        every accepted matrix.
    """
    _require_generator(rng)
    source_width, source_height = _normalize_size(source_size, "source_size")
    angle_low, angle_high = _normalize_range(angle_range, "angle_range")
    tx_low, tx_high = _normalize_range(translation_x_range, "translation_x_range")
    ty_low, ty_high = _normalize_range(translation_y_range, "translation_y_range")
    scale_low, scale_high = _normalize_range(scale_range, "scale_range")
    if scale_low <= 0:
        raise ValueError(f"scale_range must be positive, got {scale_range}")
    axis_scale_x_low, axis_scale_x_high = _normalize_range(axis_scale_x_range, "axis_scale_x_range")
    if axis_scale_x_low <= 0:
        raise ValueError(f"axis_scale_x_range must be positive, got {axis_scale_x_range}")
    axis_scale_y_low, axis_scale_y_high = _normalize_range(axis_scale_y_range, "axis_scale_y_range")
    if axis_scale_y_low <= 0:
        raise ValueError(f"axis_scale_y_range must be positive, got {axis_scale_y_range}")
    shear_x_bounds = _normalize_range(shear_x_range, "shear_x_range")
    shear_y_bounds = _normalize_range(shear_y_range, "shear_y_range")

    angle = float(rng.uniform(angle_low, angle_high))
    dx = float(rng.uniform(tx_low, tx_high))
    dy = float(rng.uniform(ty_low, ty_high))
    scale = float(rng.uniform(scale_low, scale_high))
    shear_x = _sample_singleton_aware_range(rng, shear_x_bounds)
    shear_y = _sample_singleton_aware_range(rng, shear_y_bounds)
    axis_x = _sample_singleton_aware_range(rng, (axis_scale_x_low, axis_scale_x_high))
    axis_y = _sample_singleton_aware_range(rng, (axis_scale_y_low, axis_scale_y_high))

    center = ((source_width - 1) / 2.0, (source_height - 1) / 2.0)
    rs_matrix = cv2.getRotationMatrix2D(center, angle, scale)

    axis_identity = axis_x == 1.0 and axis_y == 1.0

    if axis_identity and shear_x == 0.0 and shear_y == 0.0:
        # Fast path: no shear or axis-scale multiplication at all, so the
        # result is bit-for-bit identical to what this function produced
        # before shear or anisotropic scale existed, not merely numerically
        # close to it.
        matrix = rs_matrix
        matrix[0, 2] += dx
        matrix[1, 2] += dy
        matrix = np.asarray(matrix, dtype=np.float64)
    elif axis_identity:
        # Existing shear-only path, numerically untouched: axis scale plays
        # no part here at all, so this branch must remain byte-for-byte the
        # same code that shipped before anisotropic scale existed.
        #
        # Extreme, but individually finite, shear/scale/center values can
        # overflow intermediate products (e.g. shear_x * shear_y) to inf --
        # expected and handled by the finite-matrix checks below, not a sign
        # of a bug, so it must not surface as a stray RuntimeWarning (which
        # would fail test suites run with warnings-as-errors).
        with np.errstate(over="ignore", invalid="ignore"):
            shear_product = shear_x * shear_y
            shear_diagonal = 1.0 + shear_product

        if not np.isfinite(shear_product) or not np.isfinite(shear_diagonal):
            raise ValueError("sampled affine parameters do not produce a finite transform matrix")
        if shear_diagonal == shear_product:
            # The sequential shear matrix [[1, shear_x], [shear_y, 1 +
            # shear_x*shear_y]] has determinant 1 mathematically, for any
            # finite coefficients -- but that guarantee is about the exact
            # real-number parameterization, not about float64. When
            # |shear_x * shear_y| exceeds float64's ~2**52 integer
            # precision, "1.0 + shear_product" rounds back down to exactly
            # shear_product, silently discarding the unit term that made
            # the matrix invertible in the first place -- the stored matrix
            # would then no longer be the shear it was sampled as.
            raise ValueError(
                "sampled shear coefficients are too large to preserve "
                "an invertible sequential shear matrix in float64"
            )

        with np.errstate(over="ignore", invalid="ignore"):
            rs_3x3 = np.eye(3, dtype=np.float64)
            rs_3x3[:2, :] = rs_matrix

            # Sequential, area-preserving shear (det == 1 for any finite
            # shear_x/shear_y): x shear first, then y shear using the
            # already-sheared x -- never the naive, sometimes-singular
            # [[1, shear_x], [shear_y, 1]].
            shear_3x3 = np.eye(3, dtype=np.float64)
            shear_3x3[0, 1] = shear_x
            shear_3x3[1, 0] = shear_y
            shear_3x3[1, 1] = shear_diagonal

            cx, cy = center
            to_origin = np.eye(3, dtype=np.float64)
            to_origin[0, 2] = -cx
            to_origin[1, 2] = -cy
            from_origin = np.eye(3, dtype=np.float64)
            from_origin[0, 2] = cx
            from_origin[1, 2] = cy
            shear_centered = from_origin @ shear_3x3 @ to_origin

            combined = (rs_3x3 @ shear_centered)[:2, :].copy()
            combined[0, 2] += dx
            combined[1, 2] += dy
            matrix = np.asarray(combined, dtype=np.float64)
    else:
        # Genuinely anisotropic path: at least one axis multiplier differs
        # from 1.0. Plain Python float multiplication (not a NumPy ufunc) is
        # used deliberately here -- np.errstate does not govern it, it never
        # raises for overflow/underflow, and it can legitimately produce
        # inf (overflow) or exactly 0.0 (underflow) from two individually
        # finite, positive operands. Checking these two scalars explicitly,
        # before any matrix is built, catches a case the final whole-matrix
        # np.isfinite check below cannot: an underflow to exactly 0.0 is
        # still "finite", so it would otherwise silently produce a
        # degenerate, zero-width-axis transform from two legal positive
        # inputs.
        effective_scale_x = scale * axis_x
        effective_scale_y = scale * axis_y
        if not math.isfinite(effective_scale_x) or effective_scale_x <= 0.0:
            raise ValueError(
                "sampled scale and axis_scale_x combine to a non-representable "
                f"positive float64 x scale (scale={scale!r}, axis_scale_x={axis_x!r}, "
                f"product={effective_scale_x!r})"
            )
        if not math.isfinite(effective_scale_y) or effective_scale_y <= 0.0:
            raise ValueError(
                "sampled scale and axis_scale_y combine to a non-representable "
                f"positive float64 y scale (scale={scale!r}, axis_scale_y={axis_y!r}, "
                f"product={effective_scale_y!r})"
            )

        if not (shear_x == 0.0 and shear_y == 0.0):
            with np.errstate(over="ignore", invalid="ignore"):
                shear_product = shear_x * shear_y
                shear_diagonal = 1.0 + shear_product

            if not np.isfinite(shear_product) or not np.isfinite(shear_diagonal):
                raise ValueError(
                    "sampled affine parameters do not produce a finite transform matrix"
                )
            if shear_diagonal == shear_product:
                raise ValueError(
                    "sampled shear coefficients are too large to preserve "
                    "an invertible sequential shear matrix in float64"
                )

        with np.errstate(over="ignore", invalid="ignore"):
            rs_3x3 = np.eye(3, dtype=np.float64)
            rs_3x3[:2, :] = rs_matrix

            cx, cy = center
            to_origin = np.eye(3, dtype=np.float64)
            to_origin[0, 2] = -cx
            to_origin[1, 2] = -cy
            from_origin = np.eye(3, dtype=np.float64)
            from_origin[0, 2] = cx
            from_origin[1, 2] = cy

            axis_3x3 = np.eye(3, dtype=np.float64)
            axis_3x3[0, 0] = axis_x
            axis_3x3[1, 1] = axis_y
            axis_centered = from_origin @ axis_3x3 @ to_origin

            if shear_x == 0.0 and shear_y == 0.0:
                combined = (rs_3x3 @ axis_centered)[:2, :].copy()
            else:
                shear_3x3 = np.eye(3, dtype=np.float64)
                shear_3x3[0, 1] = shear_x
                shear_3x3[1, 0] = shear_y
                shear_3x3[1, 1] = shear_diagonal
                shear_centered = from_origin @ shear_3x3 @ to_origin

                combined = (rs_3x3 @ axis_centered @ shear_centered)[:2, :].copy()

            combined[0, 2] += dx
            combined[1, 2] += dy
            matrix = np.asarray(combined, dtype=np.float64)

    if matrix.shape != (2, 3) or matrix.dtype != np.float64:
        raise RuntimeError(
            f"internal error: sampled affine matrix has shape {matrix.shape} and "
            f"dtype {matrix.dtype}, expected (2, 3) float64"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("sampled affine parameters do not produce a finite transform matrix")
    matrix.setflags(write=False)

    return AffineParameters(
        matrix=matrix,
        source_size=(source_width, source_height),
        angle=angle,
        translation=(dx, dy),
        scale=scale,
        shear=(shear_x, shear_y),
        axis_scale=(axis_x, axis_y),
    )


@overload
def apply_affine(
    image: Image,
    params: AffineParameters,
    *,
    mask: None = None,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: float | tuple[float, ...] = 0,
) -> Image: ...
@overload
def apply_affine(
    image: Image,
    params: AffineParameters,
    *,
    mask: np.ndarray,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: float | tuple[float, ...] = 0,
    mask_border_value: int = 0,
) -> AugmentedImageMask: ...
def apply_affine(
    image: Image,
    params: AffineParameters,
    *,
    mask: np.ndarray | None = None,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: float | tuple[float, ...] = 0,
    mask_border_value: int = 0,
) -> Image | AugmentedImageMask:
    """Apply a previously sampled affine transform to `image` (and optionally `mask`).

    `params` must be an `AffineParameters` instance (its fields are
    re-validated here too, since a frozen dataclass can still be constructed
    by hand with invalid field values). Only `params.matrix` is used to
    perform the transform; `angle`/`translation`/`scale`/`shear`/
    `axis_scale` are checked for basic internal consistency (each finite,
    `scale > 0`, `axis_scale` components `> 0`) but are never used to
    reconstruct or cross-check the matrix numerically -- in particular,
    `apply_affine` does not compute `scale * axis_scale` and never rejects
    metadata merely because that product would be non-representable; only
    `params.matrix` itself must be finite.

    `image`'s spatial size (`(width, height)`) must equal `params.source_size`
    *exactly* -- the same replay guard as `apply_crop`; this guard is about
    the *input*, not the output, so it is unaffected by `params.output_size`.
    Output spatial size equals `params.source_size` when `params.output_size`
    is `None` (the default, unchanged from before canvas expansion existed),
    or `params.output_size` itself when set -- typically by
    `expand_affine_canvas`, though any manually constructed, valid
    `output_size` is accepted identically; `apply_affine` never computes or
    adjusts bounds itself, it only reads whichever size is already stored.
    Applies `improcv.transforms.warp_affine` directly (never raw
    `cv2.warpAffine`); `image`'s dtype/shape contract is exactly
    `warp_affine`'s own.

    `interpolation` selects an OpenCV interpolation mode only (e.g.
    `cv2.INTER_LINEAR`, `cv2.INTER_NEAREST`) -- it does not accept
    `cv2.WARP_INVERSE_MAP` or any other warp-control flag bit. `params.matrix`
    is always applied as the forward mapping it was sampled as (a positive
    `dx`/`dy` in `params.translation` moves content right/down); allowing
    `WARP_INVERSE_MAP` through would silently apply the saved transform in
    the opposite direction, which is rejected here before `warp_affine` is
    ever called.

    If `mask` is given, it must satisfy the same shape/dtype contract as
    `apply_flip`'s/`apply_crop`'s `mask` (shape `(H, W)`/`(H, W, 1)`, dtype
    `uint8`/`uint16`/`int16`, spatial size matching `image`) and is always
    warped with `interpolation=cv2.INTER_NEAREST`,
    `border_mode=cv2.BORDER_CONSTANT`, and `border_value=mask_border_value`
    -- the caller cannot change the mask's interpolation or border mode,
    only the fill value. `mask_border_value` must fit within `mask`'s actual
    dtype (checked via the same range check `improcv` uses elsewhere for
    saturating fill values) and need not already occur in `mask`. Because
    nearest-neighbor sampling introduces no new intermediate values, the
    output mask contains only values already present in the input mask plus
    `mask_border_value` -- this is guaranteed by construction and covered by
    tests, not re-verified by an expensive full-array scan on every call.

    Returns
    -------
    Image or AugmentedImageMask
        A new, independent array (or pair) with `image`'s dtype and, unless
        `params.output_size` is set, `image`'s original spatial shape; never
        aliases `image` or `mask`.

    Raises
    ------
    TypeError
        If `params` is not an `AffineParameters`, if its fields are not the
        expected types, if `interpolation` is not an integral value, if
        `mask`/`mask_border_value` is not an `ndarray`/integral, or if
        `image`/`mask` is not dtype-compatible.
    ValueError
        If `image`/`mask` has an unsupported shape, `image`'s (or `mask`'s)
        spatial size does not match `params.source_size` (or `image`'s),
        `params.output_size` is set but not a 2-tuple of positive ints each
        representable as an OpenCV `int` destination size, `interpolation`
        includes `WARP_INVERSE_MAP` or any other non-interpolation flag bit,
        or `mask_border_value` does not fit `mask`'s dtype range.
    RuntimeError
        If the underlying `cv2.error` occurs after full validation (for
        either the image or the mask warp), or if this function's own
        postconditions are violated.
    """
    _require_affine_parameters(params)
    interpolation = _require_interpolation_mode(interpolation)
    require_image_ndim(image, ndims=(2, 3))
    _require_matches_source_size(image, params.source_size, "image")

    output_size = params.source_size if params.output_size is None else params.output_size

    augmented_image = _apply_affine_to_array(
        image, params, output_size, interpolation, border_mode, border_value
    )
    _check_warp_postconditions(image, augmented_image, output_size, "image")

    if mask is None:
        return augmented_image

    _require_mask(mask, "mask")
    _require_matching_spatial_shape(mask, image, "mask", "image")
    require_integral(mask_border_value, "mask_border_value")
    require_fits_dtype(mask_border_value, mask.dtype, "mask_border_value")

    augmented_mask = _apply_affine_to_array(
        mask, params, output_size, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT, mask_border_value
    )
    _check_warp_postconditions(mask, augmented_mask, output_size, "mask")

    return AugmentedImageMask(image=augmented_image, mask=augmented_mask)


def expand_affine_canvas(params: AffineParameters) -> AffineParameters:
    """Grow `params`' output canvas so `apply_affine` no longer crops any transformed content.

    A purely deterministic conversion -- it never touches any RNG, and never
    calls `sample_affine` internally. It transforms `params`' full
    ``(width, height)`` source *pixel-cell footprint* (the continuous region
    ``[-0.5, width - 0.5] x [-0.5, height - 0.5]``, not just the rectangle of
    pixel centers) through `params.matrix` directly -- never through
    `params.translation`/`.angle`/`.scale`/`.shear`/`.axis_scale`, which are
    sampling metadata that a hand-constructed `params` need not agree with
    `matrix` on (exactly as everywhere else in this module, `matrix` is the
    only source of geometric truth). The new output canvas is the smallest
    axis-aligned, integer-pixel region that contains the *union* of that
    transformed footprint and the original, untransformed source footprint:
    the result is never smaller than `source_size` in either dimension, and
    no part of the transformed content is cropped.

    This union-with-source contract means `expand_affine_canvas` does **not**
    promise the leaner, tight output `improcv.transforms.rotate_bound`
    produces: for a non-square `source_size` rotated by an angle at or near
    90/270 degrees, the tight rotated bounding box is narrower than the
    source in one dimension (verified directly: a ``(3, 2)`` source rotated
    exactly 90 degrees has a tight rotated footprint of ``(2, 3)`, narrower
    than the original width of 3) -- `expand_affine_canvas`'s grow-only
    contract instead keeps the original width, giving `(3, 3)` there. The two
    contracts (`rotate_bound`'s "no larger than strictly necessary" and this
    function's "never smaller than source") cannot both hold for every
    input; this function always honors the "never smaller than source" one.

    Because the whole source footprint (not merely a translated copy of the
    transformed one) is unioned in, a transform that pushes content up/left
    can have its translation partially absorbed by the resulting shift in
    the destination coordinate origin -- the full transform (translation
    included) is still applied exactly once, and content is never cropped,
    but the *visible offset of content relative to the new canvas origin* is
    not guaranteed to equal `params.translation` verbatim.

    Bounds/shift computation is done in `float64` and snapped to the nearest
    integer only within a few ULPs of floating-point noise (never a fixed
    decimal-place rounding like `rotate_bound`'s own `round(value, 6)`,
    which is far coarser than the noise this actually needs to absorb, and
    would risk erasing a deliberately sampled, sub-1e-6 translation) --
    verified directly that a 1e-6-degree angle perturbation away from a
    right angle changes the required output size for a large-enough source,
    and that this function does not treat `89.999999`, `90.0`, and
    `90.000001` degrees as equivalent.

    Returns
    -------
    AffineParameters
        A new instance with an adjusted, independent, read-only `matrix`
        (`params.matrix` itself is never modified) and `output_size` set to
        the computed ``(width, height)``; `source_size`/`angle`/
        `translation`/`scale`/`shear`/`axis_scale` are copied unchanged from
        `params` and are not cross-checked against the new matrix, exactly
        as `apply_affine` never cross-checks them against `matrix` today.

    Raises
    ------
    TypeError
        If `params` is not an `AffineParameters`, or if its fields are not
        the expected types (same validation as `apply_affine`).
    ValueError
        If `params.output_size` is already set (this function requires
        unexpanded parameters -- it is not idempotent, and a hand-set
        `output_size` is not guaranteed to be this function's own prior
        output), if transforming the source footprint produces a
        non-finite coordinate, bound, span, or shift (e.g. from an
        astronomically large but individually finite `matrix`), or if the
        computed output size is not representable as a positive OpenCV
        ``int`` destination size (``<= 2**31 - 1`` per dimension --
        verified directly against both OpenCV 4.9 and 5.0).
    RuntimeError
        If this function's own postconditions are violated.
    """
    _require_affine_parameters(params)
    if params.output_size is not None:
        raise ValueError(
            "params already define an output_size; expand_affine_canvas requires "
            "unexpanded parameters (it is not idempotent)"
        )

    width, height = params.source_size
    matrix_3x3 = np.eye(3, dtype=np.float64)
    matrix_3x3[:2, :] = params.matrix

    corners = np.array(
        [
            [-0.5, -0.5, 1.0],
            [width - 0.5, -0.5, 1.0],
            [width - 0.5, height - 0.5, 1.0],
            [-0.5, height - 0.5, 1.0],
        ],
        dtype=np.float64,
    )

    # matrix_3x3's bottom row is exactly [0, 0, 1] by construction (affine,
    # never perspective), so the transformed third coordinate is exactly
    # 1.0 for every corner with no floating-point risk -- no perspective
    # divide is needed or performed.
    with np.errstate(over="ignore", invalid="ignore"):
        transformed = corners @ matrix_3x3.T

    if not np.all(np.isfinite(transformed[:, :2])):
        raise ValueError(
            "expand_affine_canvas: transforming the source footprint through params.matrix "
            "does not produce finite coordinates"
        )

    source_left, source_top = -0.5, -0.5
    source_right, source_bottom = width - 0.5, height - 0.5
    transformed_left = float(np.min(transformed[:, 0]))
    transformed_top = float(np.min(transformed[:, 1]))
    transformed_right = float(np.max(transformed[:, 0]))
    transformed_bottom = float(np.max(transformed[:, 1]))

    left = min(source_left, transformed_left)
    top = min(source_top, transformed_top)
    right = max(source_right, transformed_right)
    bottom = max(source_bottom, transformed_bottom)
    if not all(math.isfinite(value) for value in (left, top, right, bottom)):
        raise ValueError("expand_affine_canvas: computed canvas bounds are not finite")

    magnitude_x = max(1.0, abs(left), abs(right), float(width))
    magnitude_y = max(1.0, abs(top), abs(bottom), float(height))

    span_x = _snap_near_integer(right - left, magnitude=magnitude_x)
    span_y = _snap_near_integer(bottom - top, magnitude=magnitude_y)
    if not (math.isfinite(span_x) and math.isfinite(span_y)):
        raise ValueError("expand_affine_canvas: computed canvas spans are not finite")
    if span_x <= 0.0 or span_y <= 0.0:
        raise RuntimeError(
            f"internal error: expand_affine_canvas computed a non-positive span "
            f"({span_x}, {span_y})"
        )

    output_width = math.ceil(span_x)
    output_height = math.ceil(span_y)
    if output_width > _OPENCV_SIZE_MAX or output_height > _OPENCV_SIZE_MAX:
        raise ValueError(
            "expand_affine_canvas: computed output_size exceeds OpenCV's int32 dsize "
            f"limit ({_OPENCV_SIZE_MAX} per dimension), got ({output_width}, {output_height})"
        )

    shift_x = _snap_near_integer(-0.5 - left, magnitude=magnitude_x)
    shift_y = _snap_near_integer(-0.5 - top, magnitude=magnitude_y)
    if not (math.isfinite(shift_x) and math.isfinite(shift_y)):
        raise ValueError("expand_affine_canvas: computed canvas shift is not finite")

    shift_matrix = np.array(
        [[1.0, 0.0, shift_x], [0.0, 1.0, shift_y], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    with np.errstate(over="ignore", invalid="ignore"):
        expanded_3x3 = shift_matrix @ matrix_3x3
    expanded_matrix = np.array(expanded_3x3[:2, :], dtype=np.float64, order="C", copy=True)

    if expanded_matrix.shape != (2, 3) or expanded_matrix.dtype != np.float64:
        raise RuntimeError(
            f"internal error: expanded affine matrix has shape {expanded_matrix.shape} and "
            f"dtype {expanded_matrix.dtype}, expected (2, 3) float64"
        )
    if not np.all(np.isfinite(expanded_matrix)):
        raise ValueError("expand_affine_canvas: adjusted matrix is not finite")
    expanded_matrix.setflags(write=False)

    return AffineParameters(
        matrix=expanded_matrix,
        source_size=params.source_size,
        angle=params.angle,
        translation=params.translation,
        scale=params.scale,
        shear=params.shear,
        axis_scale=params.axis_scale,
        output_size=(output_width, output_height),
    )


def _project_perspective_footprint(
    matrix: np.ndarray, footprint_corners: tuple[tuple[float, float], ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Project `footprint_corners` through `matrix`'s full homogeneous divide, in `float64`.

    A separate, narrowly-scoped function (rather than inlined in `expand_perspective_canvas`)
    purely so the projection step has its own testable seam -- see
    `docs/design/0.5.0a2-expand-perspective-canvas.md` §14.
    """
    corners_h = np.array([[x, y, 1.0] for x, y in footprint_corners], dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        transformed_h = corners_h @ matrix.T
    w = transformed_h[:, 2]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        transformed_x = transformed_h[:, 0] / w
        transformed_y = transformed_h[:, 1] / w
    return transformed_x, transformed_y


def expand_perspective_canvas(params: PerspectiveParameters) -> PerspectiveParameters:
    """Grow `params`' output canvas so `apply_perspective` no longer crops any transformed content.

    The perspective counterpart to `expand_affine_canvas`, mirroring its contract exactly except
    for the genuine projective divide a homography (unlike an affine matrix) requires. A purely
    deterministic conversion -- it never touches any RNG, and never calls `sample_perspective`
    internally. It transforms `params`' full ``(width, height)`` source *pixel-cell footprint*
    (the continuous region ``[-0.5, width - 0.5] x [-0.5, height - 0.5]``, not just the rectangle
    of pixel centers `_require_perspective_matrix_geometry` itself validates) through
    `params.matrix` directly -- never through `destination_points`, which is sampling metadata a
    hand-constructed `params` need not agree with `matrix` on (exactly as everywhere else in this
    module, `matrix` is the only source of geometric truth). The new output canvas is the
    smallest axis-aligned, integer-pixel region that contains the *union* of that transformed
    footprint and the original, untransformed source footprint -- the result is never smaller
    than `source_size` in either dimension, and no part of the transformed content is cropped,
    mirroring `expand_affine_canvas`'s own grow-only, union-with-source contract exactly.

    Before trusting the projective divide, this function performs its own, additional,
    *stricter* horizon check over the full pixel-cell footprint -- `_require_perspective_
    parameters` (called first, applying today's unchanged validation) only checks the narrower
    pixel-center rectangle, which is sufficient for `apply_perspective`'s own fixed-canvas
    contract but not for this function's wider footprint. This means a `params` for which
    `apply_perspective` already succeeds can still be legally rejected here, when its horizon
    lies only in the half-pixel fringe between the pixel-center rectangle and the full pixel-cell
    footprint -- this is intentional, not a bug, and never tightens `apply_perspective`'s own,
    separately validated acceptance domain. No epsilon margin is used: the same strict,
    scale-invariant sign check `_require_perspective_matrix_geometry` already uses.

    Bounds/shift computation is done in `float64` and snapped to the nearest integer only within
    a few ULPs of floating-point noise (via the same `_snap_near_integer` `expand_affine_canvas`
    uses), never a fixed decimal-place rounding.

    Returns
    -------
    PerspectiveParameters
        A new instance with an adjusted, independent, read-only `matrix` (`params.matrix` itself
        is never modified) and `output_size` set to the computed ``(width, height)``;
        `source_size`/`destination_points` are copied unchanged from `params` and are not
        cross-checked against the new matrix, exactly as `apply_perspective` never cross-checks
        them against `matrix` today. `destination_points` continues to describe the original
        `sample_perspective` draw and does not reflect the canvas-origin translation this
        function applies.

    Raises
    ------
    TypeError
        If `params` is not a `PerspectiveParameters`, or if its fields are not the expected
        types (same validation as `apply_perspective`).
    ValueError
        If `params.output_size` is already set (this function requires unexpanded parameters --
        it is not idempotent, and a hand-set `output_size` is not guaranteed to be this
        function's own prior output), if `params.matrix`'s projective horizon crosses the full
        pixel-cell footprint (even if it does not cross the narrower pixel-center rectangle
        `apply_perspective` itself validates), if transforming the footprint produces a
        non-finite coordinate, bound, span, or shift, or if the computed output size is not
        representable as a positive OpenCV ``int`` destination size (``<= 2**31 - 1`` per
        dimension).
    RuntimeError
        If this function's own postconditions are violated.
    """
    _require_perspective_parameters(params)
    if params.output_size is not None:
        raise ValueError(
            "params already define an output_size; expand_perspective_canvas requires "
            "unexpanded parameters (it is not idempotent)"
        )

    width, height = params.source_size
    footprint_corners = _perspective_pixel_cell_corners((width, height))

    _require_consistent_denominator_sign(
        params.matrix,
        footprint_corners,
        "expand_perspective_canvas: params.matrix",
        "source footprint",
    )

    transformed_x, transformed_y = _project_perspective_footprint(params.matrix, footprint_corners)

    if not (np.all(np.isfinite(transformed_x)) and np.all(np.isfinite(transformed_y))):
        raise ValueError(
            "expand_perspective_canvas: transforming the source footprint through "
            "params.matrix does not produce finite coordinates"
        )

    source_left, source_top = -0.5, -0.5
    source_right, source_bottom = width - 0.5, height - 0.5
    transformed_left = float(np.min(transformed_x))
    transformed_top = float(np.min(transformed_y))
    transformed_right = float(np.max(transformed_x))
    transformed_bottom = float(np.max(transformed_y))

    left = min(source_left, transformed_left)
    top = min(source_top, transformed_top)
    right = max(source_right, transformed_right)
    bottom = max(source_bottom, transformed_bottom)
    if not all(math.isfinite(value) for value in (left, top, right, bottom)):
        raise ValueError("expand_perspective_canvas: computed canvas bounds are not finite")

    magnitude_x = max(1.0, abs(left), abs(right), float(width))
    magnitude_y = max(1.0, abs(top), abs(bottom), float(height))

    span_x = _snap_near_integer(right - left, magnitude=magnitude_x)
    span_y = _snap_near_integer(bottom - top, magnitude=magnitude_y)
    if not (math.isfinite(span_x) and math.isfinite(span_y)):
        raise ValueError("expand_perspective_canvas: computed canvas spans are not finite")
    if span_x <= 0.0 or span_y <= 0.0:
        raise RuntimeError(
            f"internal error: expand_perspective_canvas computed a non-positive span "
            f"({span_x}, {span_y})"
        )

    output_width = math.ceil(span_x)
    output_height = math.ceil(span_y)
    if output_width > _OPENCV_SIZE_MAX or output_height > _OPENCV_SIZE_MAX:
        raise ValueError(
            "expand_perspective_canvas: computed output_size exceeds OpenCV's int32 dsize "
            f"limit ({_OPENCV_SIZE_MAX} per dimension), got ({output_width}, {output_height})"
        )

    shift_x = _snap_near_integer(-0.5 - left, magnitude=magnitude_x)
    shift_y = _snap_near_integer(-0.5 - top, magnitude=magnitude_y)
    if not (math.isfinite(shift_x) and math.isfinite(shift_y)):
        raise ValueError("expand_perspective_canvas: computed canvas shift is not finite")

    shift_matrix = np.array(
        [[1.0, 0.0, shift_x], [0.0, 1.0, shift_y], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    with np.errstate(over="ignore", invalid="ignore"):
        expanded_3x3 = shift_matrix @ params.matrix
    expanded_matrix = np.array(expanded_3x3, dtype=np.float64, order="C", copy=True)

    if expanded_matrix.shape != (3, 3) or expanded_matrix.dtype != np.float64:
        raise RuntimeError(
            f"internal error: expanded perspective matrix has shape {expanded_matrix.shape} and "
            f"dtype {expanded_matrix.dtype}, expected (3, 3) float64"
        )
    if not np.all(np.isfinite(expanded_matrix)):
        raise ValueError("expand_perspective_canvas: adjusted matrix is not finite")
    expanded_matrix.setflags(write=False)

    return PerspectiveParameters(
        matrix=expanded_matrix,
        source_size=params.source_size,
        destination_points=params.destination_points,
        output_size=(output_width, output_height),
    )


def sample_perspective(
    rng: np.random.Generator,
    source_size: tuple[int, int],
    *,
    distortion_scale: float = 0.5,
) -> PerspectiveParameters:
    """Sample a replayable perspective transform by displacing each source corner inward.

    `source_size` is `(width, height)`; both dimensions must be at least `2` -- a genuine
    four-corner correspondence is not well defined otherwise (verified directly: even
    `cv2.getPerspectiveTransform(src, src)` does not return identity for `width < 2` or
    `height < 2`, since the four "source corners" collapse to fewer than four distinct,
    non-collinear points). The four source corners are always the deterministic rectangle
    `(0, 0)`, `(width - 1, 0)`, `(width - 1, height - 1)`, `(0, height - 1)`, in `top-left,
    top-right, bottom-right, bottom-left` order.

    `distortion_scale` is a single value in `[0.0, 1/2]` (not a range) -- it is the maximum
    fraction of half the source width/height that each corner may be displaced inward, not a
    directly-sampled transform parameter the way `AffineParameters.angle`/`.scale` are: it only
    bounds the region each corner is drawn from,

    ```text
    max_dx = distortion_scale * (width - 1) / 2.0
    max_dy = distortion_scale * (height - 1) / 2.0

    top_left     ~ (Uniform[0, max_dx),                    Uniform[0, max_dy))
    top_right    ~ (Uniform[(width-1)-max_dx, width-1),    Uniform[0, max_dy))
    bottom_right ~ (Uniform[(width-1)-max_dx, width-1),    Uniform[(height-1)-max_dy, height-1))
    bottom_left  ~ (Uniform[0, max_dx),                    Uniform[(height-1)-max_dy, height-1))
    ```

    each independently via `rng.uniform` -- `low` is reachable, `high` is not guaranteed
    (an ordinary property of continuous floating-point sampling). The exact number and order of
    internal draws against `rng` is an implementation detail, not part of the public contract,
    and may change between releases without notice.

    `distortion_scale == 0.0` takes an explicit identity fast path: no `rng` draw happens at
    all (verified directly: the generator's `bit_generator.state` is unchanged), `matrix` is
    exactly `np.eye(3, dtype=np.float64)`, and `destination_points` are exactly the source
    corners.

    `distortion_scale <= 0.5` is not an arbitrary cap: after normalizing both axes to `[0, 1]`,
    each corner moves inward by at most `a = distortion_scale / 2 <= 1/4`. For two consecutive
    edges of the resulting quadrilateral, the signed turn at their shared corner is bounded
    below by `(1 - 2*a)**2 - a**2 = 1 - 4*a + 3*a**2`, which for `a <= 1/4` is at least `3/16 >
    0` -- so in exact real arithmetic, the destination quadrilateral is always strictly convex,
    non-self-intersecting, and keeps the same corner order (never mirrored). This is a
    geometric proof for the real-number construction, not a guarantee that survives every
    floating-point rounding step: the four destination points actually used are the ones
    quantized to `float32` (`cv2.getPerspectiveTransform`'s own required input dtype -- verified
    directly against OpenCV 4.9 and 5.0), so this function still checks, on those quantized
    points, that all four consecutive signed turns are strictly positive (no epsilon margin) --
    this is expected to always hold given the proof above, but for an extreme `source_size`
    (still representable, but pushing coordinates far enough that `float32` rounding matters)
    it is a real, tested safeguard, not a formality. It then also checks the constructed matrix
    itself is numerically full-rank and free of a projective horizon crossing the source
    rectangle (see `apply_perspective`) -- raising `ValueError`, never silently returning a
    degenerate transform or retrying with a new draw.

    Returns
    -------
    PerspectiveParameters
        Independent of `rng`'s state after this call; `matrix` is the sole source of truth for
        replay via `apply_perspective`. `destination_points` are sampling metadata for
        debugging/logging/`repr` only.

    Raises
    ------
    TypeError
        If `rng` is not a `numpy.random.Generator`, or `source_size` is not a 2-tuple of
        positive integral (non-`bool`) values, or `distortion_scale` is not a real number
        (`bool`/`np.bool_` rejected).
    ValueError
        If either `source_size` dimension is not representable as `np.intp` on this platform,
        if either dimension is less than `2`, if `distortion_scale` is outside `[0.0, 0.5]` or
        is `NaN`/infinite, or if the sampled corners do not combine into a strictly convex,
        consistently-oriented, numerically full-rank, horizon-free transform (see above).
    """
    _require_generator(rng)
    source_width, source_height = _normalize_size(source_size, "source_size")
    if source_width < 2 or source_height < 2:
        raise ValueError(
            "source_size must have both dimensions >= 2 for a well-defined perspective "
            f"transform (a 4-corner correspondence requires 4 distinct, non-collinear source "
            f"points), got {(source_width, source_height)}"
        )
    require_range(distortion_scale, 0.0, 0.5, "distortion_scale")

    source_points = _perspective_source_corners((source_width, source_height))

    if distortion_scale == 0.0:
        matrix = np.eye(3, dtype=np.float64)
        matrix.setflags(write=False)
        return PerspectiveParameters(
            matrix=matrix,
            source_size=(source_width, source_height),
            destination_points=source_points,
        )

    max_dx = distortion_scale * (source_width - 1) / 2.0
    max_dy = distortion_scale * (source_height - 1) / 2.0

    raw_destination_points = (
        (rng.uniform(0.0, max_dx), rng.uniform(0.0, max_dy)),
        (
            rng.uniform((source_width - 1) - max_dx, source_width - 1),
            rng.uniform(0.0, max_dy),
        ),
        (
            rng.uniform((source_width - 1) - max_dx, source_width - 1),
            rng.uniform((source_height - 1) - max_dy, source_height - 1),
        ),
        (
            rng.uniform(0.0, max_dx),
            rng.uniform((source_height - 1) - max_dy, source_height - 1),
        ),
    )

    source_array = np.array(source_points, dtype=np.float32)
    destination_array = np.array(raw_destination_points, dtype=np.float32)
    destination_points = (
        (float(destination_array[0, 0]), float(destination_array[0, 1])),
        (float(destination_array[1, 0]), float(destination_array[1, 1])),
        (float(destination_array[2, 0]), float(destination_array[2, 1])),
        (float(destination_array[3, 0]), float(destination_array[3, 1])),
    )

    _require_convex_quadrilateral(destination_points, "sampled destination_points")

    raw_matrix = cv2.getPerspectiveTransform(source_array, destination_array)
    matrix = np.array(raw_matrix, dtype=np.float64, order="C", copy=True)
    if matrix.shape != (3, 3) or matrix.dtype != np.float64:
        raise RuntimeError(
            f"internal error: sampled perspective matrix has shape {matrix.shape} and "
            f"dtype {matrix.dtype}, expected (3, 3) float64"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("sampled perspective parameters do not produce a finite transform matrix")

    _require_perspective_matrix_geometry(matrix, (source_width, source_height), "matrix")

    matrix.setflags(write=False)

    return PerspectiveParameters(
        matrix=matrix,
        source_size=(source_width, source_height),
        destination_points=destination_points,
    )


@overload
def apply_perspective(
    image: Image,
    params: PerspectiveParameters,
    *,
    mask: None = None,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: float | tuple[float, ...] = 0,
) -> Image: ...
@overload
def apply_perspective(
    image: Image,
    params: PerspectiveParameters,
    *,
    mask: np.ndarray,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: float | tuple[float, ...] = 0,
    mask_border_value: int = 0,
) -> AugmentedImageMask: ...
def apply_perspective(
    image: Image,
    params: PerspectiveParameters,
    *,
    mask: np.ndarray | None = None,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: float | tuple[float, ...] = 0,
    mask_border_value: int = 0,
) -> Image | AugmentedImageMask:
    """Apply a previously sampled perspective transform to `image` (and optionally `mask`).

    `params` must be a `PerspectiveParameters` instance (its fields are re-validated here too,
    since a frozen dataclass can still be constructed by hand with invalid field values). Only
    `params.matrix` is used to perform the transform; `destination_points` is checked for basic
    internal consistency (a 4-tuple of finite `(x, y)` pairs) but is never used to reconstruct
    or cross-check the matrix numerically. A hand-constructed `params` is also checked for the
    same numerical-full-rank and horizon-free geometry `sample_perspective` itself enforces
    (see `sample_perspective`'s docstring) -- `cv2.warpPerspective` does not raise for a
    singular or horizon-crossing matrix, verified directly: it silently returns a degenerate
    (typically all-border-fill) image instead, so this must be checked here rather than left to
    OpenCV.

    `image`'s spatial size (`(width, height)`) must equal `params.source_size` *exactly* -- the
    same replay guard as `apply_affine`/`apply_crop`; this guard is about the *input*, not the
    output, so it is unaffected by `params.output_size`. A `PerspectiveParameters` built by hand
    for a `source_size` with a dimension of `1` remains legal here (unlike `sample_perspective`,
    which refuses to construct one) as long as its `matrix` independently passes the checks
    above -- `apply_perspective` does not require the 4-corner correspondence `sample_
    perspective` needs, only a valid matrix and a matching image. Output spatial size equals
    `params.source_size` when `params.output_size` is `None` (the default, unchanged from before
    canvas expansion existed), or `params.output_size` itself when set -- typically by
    `expand_perspective_canvas`, though any manually constructed, valid `output_size` is accepted
    identically; `apply_perspective` never computes or adjusts bounds itself, it only reads
    whichever size is already stored. Applies `improcv.transforms.warp_perspective` directly
    (never raw `cv2.warpPerspective`); `image`'s dtype/shape contract is exactly
    `warp_perspective`'s own (identical to `warp_affine`'s, verified directly).

    `interpolation` selects an OpenCV interpolation mode only -- it does not accept
    `cv2.WARP_INVERSE_MAP` or any other warp-control flag bit, exactly as in `apply_affine`.

    If `mask` is given, it must satisfy the same shape contract as `apply_affine`'s `mask`
    (shape `(H, W)`/`(H, W, 1)`, spatial size matching `image`) but a *narrower* dtype contract:
    `uint8`/`uint16` only, not `int16` -- unlike every other dtype behavior in this module,
    verified directly to be identical between `warpPerspective` and `warpAffine` (silently
    downcasting an `int64` mask to `int32`, rejecting `bool`), an `int16` mask specifically was
    found (via this project's own CI) to make `cv2.warpPerspective` raise "Unknown C++ exception
    from OpenCV code" on Windows (`opencv-python-headless==4.14.0.94`) while the identical wheel
    version works correctly for the same call on Linux and macOS -- a genuine, platform-specific
    upstream OpenCV limitation affecting `warpPerspective` only (`warpAffine` with the same
    `int16` mask is unaffected on all three platforms), not something this project can fix, so
    `apply_perspective` does not accept `int16` masks at all rather than support them
    unreliably depending on the caller's platform. `mask` is always warped with
    `interpolation=cv2.INTER_NEAREST`, `border_mode=cv2.BORDER_CONSTANT`, and
    `border_value=mask_border_value` -- the caller cannot change the mask's interpolation or
    border mode, only the fill value.

    Returns
    -------
    Image or AugmentedImageMask
        A new, independent array (or pair) with `image`'s original shape and dtype; never
        aliases `image` or `mask`.

    Raises
    ------
    TypeError
        If `params` is not a `PerspectiveParameters`, if its fields are not the expected types,
        if `interpolation` is not an integral value, if `mask`/`mask_border_value` is not an
        `ndarray`/integral, or if `image`/`mask` is not dtype-compatible.
    ValueError
        If `image`/`mask` has an unsupported shape, `image`'s (or `mask`'s) spatial size does
        not match `params.source_size` (or `image`'s), `interpolation` includes
        `WARP_INVERSE_MAP` or any other non-interpolation flag bit, `mask_border_value` does not
        fit `mask`'s dtype range, `params.output_size` is set but not a 2-tuple of positive ints
        each representable as an OpenCV `int` destination size, or `params.matrix` is not
        numerically full-rank or has a projective horizon crossing `params.source_size`'s
        rectangle.
    RuntimeError
        If the underlying `cv2.error` occurs after full validation (for either the image or the
        mask warp), or if this function's own postconditions are violated.
    """
    _require_perspective_parameters(params)
    interpolation = _require_interpolation_mode(interpolation)
    require_image_ndim(image, ndims=(2, 3))
    _require_matches_source_size(image, params.source_size, "image")

    output_size = params.source_size if params.output_size is None else params.output_size

    augmented_image = _apply_perspective_to_array(
        image, params, output_size, interpolation, border_mode, border_value
    )
    _check_warp_postconditions(image, augmented_image, output_size, "image")

    if mask is None:
        return augmented_image

    _require_mask(mask, "mask", _PERSPECTIVE_MASK_DTYPES)
    _require_matching_spatial_shape(mask, image, "mask", "image")
    require_integral(mask_border_value, "mask_border_value")
    require_fits_dtype(mask_border_value, mask.dtype, "mask_border_value")

    augmented_mask = _apply_perspective_to_array(
        mask, params, output_size, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT, mask_border_value
    )
    _check_warp_postconditions(mask, augmented_mask, output_size, "mask")

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


def _normalize_range(value: object, name: str) -> tuple[float, float]:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple, got {type(value).__name__}")
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly 2 elements, got {len(value)}")
    low, high = value
    require_finite(low, f"{name}[0]")
    require_finite(high, f"{name}[1]")
    low_f, high_f = float(low), float(high)
    if low_f > high_f:
        raise ValueError(f"{name} low must be <= high, got {value}")
    return (low_f, high_f)


def _sample_singleton_aware_range(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    # A singleton range (including each parameter's own default) must not
    # consume any rng state at all -- existing call sites that never set
    # shear or axis-scale ranges must keep sampling angle/translation/scale
    # from exactly the same rng state, call after call, as they did before
    # shear/anisotropic scale existed. Shared by shear_x/shear_y and
    # axis_scale_x/axis_scale_y -- this body was already fully generic
    # before the rename, nothing else about its behavior changes.
    low, high = bounds
    if low == high:
        return low
    return float(rng.uniform(low, high))


def _snap_near_integer(value: float, *, magnitude: float) -> float:
    # Only meant to absorb a few ULPs of floating-point noise from the
    # handful of matrix multiplications used to build expand_affine_canvas's
    # bounds (e.g. cos(90 degrees) landing at ~6.12e-17 instead of exactly
    # 0.0) -- never a fixed decimal-place round like rotate_bound's own
    # round(value, 6), which is far coarser than float64 noise at realistic
    # image magnitudes and would risk silently destroying a genuinely
    # sampled, deliberate subpixel translation (verified directly: a
    # translation as small as 1e-7 must survive untouched, and round(x, 6)
    # would not preserve it). magnitude scales the tolerance to the actual
    # coordinates involved, not a single global constant.
    nearest = float(round(value))
    tolerance = 16.0 * math.ulp(max(1.0, magnitude))
    if abs(value - nearest) <= tolerance:
        return nearest
    return value


def _restore_singleton_channel(
    source: np.ndarray, result: np.ndarray, expected_spatial_shape: tuple[int, int]
) -> np.ndarray:
    # cv2.flip and cv2.warpAffine/warpPerspective all drop a trailing
    # singleton channel dimension (verified directly for (H, W, 1) input, on
    # all three) -- restore it so the output shape always matches the
    # expected (H, W, 1) shape exactly, per this module's own
    # shape-preservation postcondition. expected_spatial_shape is passed in
    # explicitly (rather than derived from source.shape[:2]) because affine
    # canvas expansion can make the output's spatial size differ from the
    # source's -- for flip/crop and for fixed-canvas affine/perspective,
    # callers pass exactly source.shape[:2], so behavior there is unchanged.
    # Restricted to exactly this known squeeze (a singleton channel dim
    # disappearing to precisely the expected spatial shape), not merely
    # "same total element count" or "any 2-D result" -- a wrong-shaped
    # result (e.g. a transposed (W, H) array, or a (H, W*C) result for a
    # (H, W, C) input) must still be left for the postcondition check to
    # reject with a clear RuntimeError, not silently reshaped into
    # something plausible-looking.
    if source.ndim == 3 and source.shape[2] == 1 and result.shape == expected_spatial_shape:
        return result[:, :, None]
    return result


def _apply_flip_preserving_shape(array: np.ndarray, direction: FlipDirection | None) -> np.ndarray:
    if direction is None:
        return array.copy()
    try:
        result = _flip(array, direction)
    except cv2.error as exc:
        raise RuntimeError("OpenCV failed to apply flip augmentation") from exc
    return _restore_singleton_channel(array, result, array.shape[:2])


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


def _require_affine_parameters(params: object) -> None:
    if not isinstance(params, AffineParameters):
        raise TypeError(f"params must be an AffineParameters, got {type(params).__name__}")

    if not isinstance(params.matrix, np.ndarray):
        raise TypeError(f"params.matrix must be a NumPy array, got {type(params.matrix).__name__}")
    if params.matrix.dtype != np.float64:
        raise TypeError(f"params.matrix must have dtype float64, got {params.matrix.dtype}")
    require_transform_matrix(params.matrix, (2, 3), "params.matrix")

    _require_source_size(params.source_size)

    require_finite(params.angle, "params.angle")
    _require_finite_pair(params.translation, "params.translation")
    require_positive(params.scale, "params.scale")
    _require_finite_pair(params.shear, "params.shear")
    _require_finite_pair(params.axis_scale, "params.axis_scale")
    require_positive(params.axis_scale[0], "params.axis_scale[0]")
    require_positive(params.axis_scale[1], "params.axis_scale[1]")
    _require_optional_output_size(params.output_size)


def _require_optional_output_size(output_size: object) -> None:
    if output_size is None:
        return
    if not isinstance(output_size, tuple):
        raise TypeError(
            f"params.output_size must be None or a tuple, got {type(output_size).__name__}"
        )
    if len(output_size) != 2:
        raise ValueError(
            f"params.output_size must contain exactly 2 elements, got {len(output_size)}"
        )
    for value in output_size:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"params.output_size elements must be int, got {type(value).__name__}")
    width, height = output_size
    if width <= 0 or height <= 0:
        raise ValueError(f"params.output_size must be positive, got {output_size}")
    if width > _OPENCV_SIZE_MAX or height > _OPENCV_SIZE_MAX:
        raise ValueError(
            f"params.output_size must fit in an OpenCV int32 dsize (<= {_OPENCV_SIZE_MAX}), "
            f"got {output_size}"
        )


def _require_source_size(source_size: object) -> None:
    if not isinstance(source_size, tuple):
        raise TypeError(f"params.source_size must be a tuple, got {type(source_size).__name__}")
    if len(source_size) != 2:
        raise ValueError(
            f"params.source_size must contain exactly 2 elements, got {len(source_size)}"
        )
    for value in source_size:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"params.source_size elements must be int, got {type(value).__name__}")
    width, height = source_size
    if width <= 0 or height <= 0:
        raise ValueError(f"params.source_size must be positive, got {source_size}")
    if width > _INTP_MAX or height > _INTP_MAX:
        raise ValueError(
            f"params.source_size must fit in a signed intp (<= {_INTP_MAX}), got {source_size}"
        )


def _require_finite_pair(value: object, name: str) -> None:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple, got {type(value).__name__}")
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly 2 elements, got {len(value)}")
    dx, dy = value
    require_finite(dx, f"{name}[0]")
    require_finite(dy, f"{name}[1]")


def _require_interpolation_mode(value: object) -> int:
    require_integral(value, "interpolation")
    normalized = int(value)  # type: ignore[arg-type]

    if not 0 <= normalized < int(cv2.INTER_MAX):
        raise ValueError(
            "interpolation must be an OpenCV interpolation mode without "
            "WARP_INVERSE_MAP or other warp modifier flags"
        )

    return normalized


def _apply_affine_to_array(
    array: np.ndarray,
    params: AffineParameters,
    output_size: tuple[int, int],
    interpolation: int,
    border_mode: int,
    border_value: float | tuple[float, ...],
) -> np.ndarray:
    try:
        result = _warp_affine(
            array,
            params.matrix,
            output_size,
            interpolation=interpolation,
            border_mode=border_mode,
            border_value=border_value,
        )
    except cv2.error as exc:
        raise RuntimeError("OpenCV failed to apply affine augmentation") from exc
    output_width, output_height = output_size
    return _restore_singleton_channel(array, result, (output_height, output_width))


def _perspective_source_corners(
    source_size: tuple[int, int],
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]:
    """The four deterministic source corners for `source_size`, in top-left, top-right,
    bottom-right, bottom-left order."""
    width, height = source_size
    return (
        (0.0, 0.0),
        (float(width - 1), 0.0),
        (float(width - 1), float(height - 1)),
        (0.0, float(height - 1)),
    )


def _perspective_pixel_cell_corners(
    source_size: tuple[int, int],
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]:
    """The four corners of `source_size`'s continuous pixel-cell footprint
    (``[-0.5, width - 0.5] x [-0.5, height - 0.5]``), in the same top-left, top-right,
    bottom-right, bottom-left order as `_perspective_source_corners`.

    Used only by `expand_perspective_canvas` -- the wider footprint `expand_affine_canvas`
    already uses for its own (affine) bounds, half a pixel larger on each side than
    `_perspective_source_corners`'s pixel-center rectangle."""
    width, height = source_size
    return (
        (-0.5, -0.5),
        (float(width) - 0.5, -0.5),
        (float(width) - 0.5, float(height) - 0.5),
        (-0.5, float(height) - 0.5),
    )


def _require_convex_quadrilateral(points: tuple[tuple[float, float], ...], name: str) -> None:
    """Raise ValueError unless `points` (in cyclic order) form a strictly convex,
    consistently-oriented quadrilateral.

    Checked on the actual points used (e.g. already quantized to `float32`), not on
    pre-quantization values -- rounding for an extreme `source_size` could otherwise turn a
    provably-safe real-number construction into a degenerate one. Every one of the four
    consecutive signed turns must be strictly positive (matching this module's fixed
    top-left/top-right/bottom-right/bottom-left orientation) -- no epsilon margin: a zero or
    negative turn means two points coincide, three are collinear, or the quadrilateral is
    self-intersecting or mirrored, none of which this module accepts from a sampler.
    """
    if len(set(points)) != 4:
        raise ValueError(f"{name} must be four distinct points, got {points}")

    turns = []
    count = len(points)
    for index in range(count):
        x0, y0 = points[index]
        x1, y1 = points[(index + 1) % count]
        x2, y2 = points[(index + 2) % count]
        v1x, v1y = x1 - x0, y1 - y0
        v2x, v2y = x2 - x1, y2 - y1
        turns.append(v1x * v2y - v1y * v2x)

    if not all(turn > 0.0 for turn in turns):
        raise ValueError(
            f"{name} do not form a strictly convex, consistently oriented quadrilateral "
            f"(signed turns: {turns})"
        )


def _require_perspective_matrix_geometry(
    matrix: np.ndarray, source_size: tuple[int, int], name: str
) -> None:
    """Raise ValueError unless `matrix` is numerically full-rank and free of a projective
    horizon crossing the `source_size` rectangle.

    A homography is only defined up to a nonzero scalar multiple, so both checks below are
    scale-invariant by construction (each normalizes by its own matrix's/row's largest absolute
    element before testing) -- verified directly that `matrix`, `matrix * 1e200`, and
    `matrix * 1e-200` all reach the same accept/reject decision, as long as every element stays
    finite. `np.linalg.det(matrix) != 0` is deliberately not used: verified directly (OpenCV
    5.0, degenerate input points) that a fully degenerate `getPerspectiveTransform` result can
    have a nonzero-but-astronomically-small determinant (e.g. ``-1.2e-31``) that a naive
    determinant check would accept -- `np.linalg.matrix_rank`'s own SVD-based tolerance does
    not.
    """
    max_abs = float(np.max(np.abs(matrix)))
    if max_abs == 0.0:
        raise ValueError(f"{name} must not be the zero matrix")

    with np.errstate(over="ignore", invalid="ignore"):
        scaled_matrix = matrix / max_abs
    if not np.all(np.isfinite(scaled_matrix)):
        raise ValueError(f"{name} could not be safely scaled for a numerical rank check")

    try:
        rank = int(np.linalg.matrix_rank(scaled_matrix))
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} numerical rank could not be determined") from exc
    if rank != 3:
        raise ValueError(
            f"{name} must be numerically full-rank and sufficiently separated from "
            f"singularity, got numerical rank {rank}"
        )

    _require_consistent_denominator_sign(
        matrix, _perspective_source_corners(source_size), name, "source rectangle"
    )


def _require_consistent_denominator_sign(
    matrix: np.ndarray,
    corners: tuple[tuple[float, float], ...],
    name: str,
    region: str,
) -> None:
    """Raise ValueError unless `matrix`'s homogeneous denominator keeps one strict sign across
    `corners` (each an `(x, y)` point).

    Shared, scale-invariant primitive behind both `_require_perspective_matrix_geometry`'s
    existing pixel-center-rectangle check (`region="source rectangle"`, reproducing its exact,
    unchanged message) and `expand_perspective_canvas`'s own, additional, stricter pixel-cell-
    footprint check (`region="source footprint"`) -- see `docs/design/
    0.5.0a2-expand-perspective-canvas.md` §11.
    """
    h20, h21, h22 = float(matrix[2, 0]), float(matrix[2, 1]), float(matrix[2, 2])
    row_scale = max(abs(h20), abs(h21), abs(h22))
    if row_scale == 0.0:
        raise ValueError(f"{name} third row must not be entirely zero")
    a, b, c = h20 / row_scale, h21 / row_scale, h22 / row_scale

    denominators = [math.fsum((a * x, b * y, c)) for x, y in corners]
    if not (
        all(value > 0.0 for value in denominators) or all(value < 0.0 for value in denominators)
    ):
        raise ValueError(
            f"{name}'s homogeneous denominator changes sign (or reaches zero) within the "
            f"{region} -- its projective horizon crosses the image"
        )


def _require_perspective_parameters(params: object) -> None:
    if not isinstance(params, PerspectiveParameters):
        raise TypeError(f"params must be a PerspectiveParameters, got {type(params).__name__}")

    if not isinstance(params.matrix, np.ndarray):
        raise TypeError(f"params.matrix must be a NumPy array, got {type(params.matrix).__name__}")
    if params.matrix.dtype != np.float64:
        raise TypeError(f"params.matrix must have dtype float64, got {params.matrix.dtype}")
    require_transform_matrix(params.matrix, (3, 3), "params.matrix")

    _require_source_size(params.source_size)

    destination_points = params.destination_points
    if not isinstance(destination_points, tuple):
        raise TypeError(
            f"params.destination_points must be a tuple, got {type(destination_points).__name__}"
        )
    if len(destination_points) != 4:
        raise ValueError(
            "params.destination_points must contain exactly 4 points, got "
            f"{len(destination_points)}"
        )
    for index, point in enumerate(destination_points):
        require_point_2d(point, f"params.destination_points[{index}]")

    _require_perspective_matrix_geometry(params.matrix, params.source_size, "params.matrix")
    _require_optional_output_size(params.output_size)


def _apply_perspective_to_array(
    array: np.ndarray,
    params: PerspectiveParameters,
    output_size: tuple[int, int],
    interpolation: int,
    border_mode: int,
    border_value: float | tuple[float, ...],
) -> np.ndarray:
    try:
        result = _warp_perspective(
            array,
            params.matrix,
            output_size,
            interpolation=interpolation,
            border_mode=border_mode,
            border_value=border_value,
        )
    except cv2.error as exc:
        raise RuntimeError("OpenCV failed to apply perspective augmentation") from exc
    output_width, output_height = output_size
    return _restore_singleton_channel(array, result, (output_height, output_width))


def _require_matches_source_size(
    image: np.ndarray, source_size: tuple[int, int], name: str
) -> None:
    source_width, source_height = source_size
    image_height, image_width = image.shape[:2]
    if (image_width, image_height) != (source_width, source_height):
        raise ValueError(
            f"{name} spatial size {(image_width, image_height)} does not match "
            f"params.source_size {(source_width, source_height)}"
        )


def _require_mask(mask: object, name: str, dtypes: tuple[type, ...] = _MASK_DTYPES) -> None:
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array, got {type(mask).__name__}")
    if mask.ndim not in (2, 3):
        raise ValueError(f"{name} must have 2 or 3 dimensions, got {mask.ndim}")
    if mask.ndim == 3 and mask.shape[2] != 1:
        raise ValueError(f"{name} must have shape (H, W) or (H, W, 1), got {mask.shape}")
    if mask.size == 0:
        raise ValueError(f"{name} must not be empty, got shape {mask.shape}")
    require_dtype(mask, dtypes, name)


def _require_matching_spatial_shape(a: np.ndarray, b: np.ndarray, name_a: str, name_b: str) -> None:
    if a.shape[:2] != b.shape[:2]:
        raise ValueError(
            f"{name_a} must have the same spatial size as {name_b}, got {a.shape[:2]} "
            f"vs {b.shape[:2]}"
        )


def _check_shape_preserving_postconditions(
    original: np.ndarray, result: np.ndarray, name: str
) -> None:
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


def _check_warp_postconditions(
    source: np.ndarray, result: np.ndarray, output_size: tuple[int, int], name: str
) -> None:
    # Generalizes _check_shape_preserving_postconditions for a warp whose
    # spatial output size may legitimately differ from the input's (affine
    # canvas expansion) -- dtype and channel shape must still be unchanged,
    # and the output must still be independent, but the spatial dimensions
    # are checked against the caller-supplied output_size instead of
    # source.shape. When output_size == source's own spatial size (the
    # fixed-canvas case, i.e. params.output_size is None), this reduces to
    # exactly the same check as _check_shape_preserving_postconditions.
    width, height = output_size
    expected_shape = (height, width) + source.shape[2:]
    if result.shape != expected_shape:
        raise RuntimeError(
            f"internal error: {name} shape is {result.shape}, expected {expected_shape}"
        )
    if result.dtype != source.dtype:
        raise RuntimeError(
            f"internal error: {name} dtype changed from {source.dtype} to {result.dtype}"
        )
    if np.shares_memory(result, source):
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
