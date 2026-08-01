"""Benchmarks for improcv's affine augmentation API against equivalent raw OpenCV calls.

Only ever collected by an explicit `uv run --group benchmark pytest benchmarks/` --
`benchmarks/` sits outside `[tool.pytest.ini_options].testpaths`, so a plain `uv run pytest`
never sees this file. See `benchmarks/README.md` for what these cases answer, how to run a
stable local baseline, and how to read the results.

Three `@pytest.mark.benchmark(group=...)` groups (a real `pytest-benchmark` grouping, reflected
in each benchmark entry's own `group` field, not just this comment):

- `affine-python-geometry` -- `sample_affine`/`expand_affine_canvas`, pure Python/NumPy geometry
  with no OpenCV kernel involved.
- `affine-image-only` -- `apply_affine` on an image alone vs. a single raw `cv2.warpAffine` call,
  at three sizes.
- `affine-image-mask` -- `apply_affine` on an image plus a segmentation mask vs. two raw
  `cv2.warpAffine` calls (the second forced to `INTER_NEAREST`, matching the mask's own
  label-safe contract), at three sizes.

The six correctness tests (`*_matches_raw_kernel(s)`) do not use the `benchmark` fixture and
carry no group marker -- they are plain assertions, never a benchmark entry in the JSON output.

Every raw/`improcv` pair shares the same image, mask, and `AffineParameters` object, and the raw
side always performs the same warp (same matrix, `dsize`, interpolation, and border handling) --
a dedicated, non-benchmark correctness test for each group asserts `np.array_equal` between the
two sides, so a ratio computed from these benchmarks is never comparing different semantics.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

import improcv as im

_SIZES: list[tuple[int, int]] = [(64, 64), (640, 480), (1920, 1080)]
_SIZE_IDS = ["64x64", "640x480", "1920x1080"]

_IMAGE_BORDER_VALUE = 0
_MASK_BORDER_VALUE = 255
_SEED = 20260801


def _build_image(width: int, height: int) -> np.ndarray:
    rng = np.random.default_rng(_SEED)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def _build_mask(width: int, height: int) -> np.ndarray:
    rng = np.random.default_rng(_SEED + 1)
    return rng.integers(0, 3, size=(height, width), dtype=np.uint8)


def _build_affine_params(width: int, height: int) -> im.AffineParameters:
    # A fixed, non-degenerate transform (not the identity) -- singleton ranges, so this is
    # deterministic regardless of the rng seed or its internal state.
    rng = np.random.default_rng(_SEED + 2)
    return im.sample_affine(
        rng,
        source_size=(width, height),
        angle_range=(15.0, 15.0),
        translation_x_range=(10.0, 10.0),
        translation_y_range=(-8.0, -8.0),
        scale_range=(1.0, 1.0),
        axis_scale_x_range=(1.0, 1.0),
        axis_scale_y_range=(1.0, 1.0),
        shear_x_range=(0.0, 0.0),
        shear_y_range=(0.0, 0.0),
    )


@pytest.fixture(params=_SIZES, ids=_SIZE_IDS)
def size(request: pytest.FixtureRequest) -> tuple[int, int]:
    return request.param


@pytest.fixture(params=["raw", "improcv"], ids=["raw", "improcv"])
def implementation(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def image(size: tuple[int, int]) -> np.ndarray:
    width, height = size
    return _build_image(width, height)


@pytest.fixture
def mask(size: tuple[int, int]) -> np.ndarray:
    width, height = size
    return _build_mask(width, height)


@pytest.fixture
def affine_params(size: tuple[int, int]) -> im.AffineParameters:
    width, height = size
    return _build_affine_params(width, height)


# --- affine-python-geometry --------------------------------------------------------------


@pytest.mark.benchmark(group="affine-python-geometry")
def test_sample_affine_random_ranges(benchmark: object) -> None:
    """`sample_affine` with real, non-degenerate ranges -- not the singleton fast path used
    elsewhere in this file to build a fixed transform for the apply benchmarks below.
    """
    rng = np.random.default_rng(_SEED + 3)
    source_size = (640, 480)

    def run() -> im.AffineParameters:
        return im.sample_affine(
            rng,
            source_size=source_size,
            angle_range=(-20.0, 20.0),
            translation_x_range=(-12.0, 12.0),
            translation_y_range=(-12.0, 12.0),
            scale_range=(0.9, 1.1),
            axis_scale_x_range=(0.9, 1.1),
            axis_scale_y_range=(0.9, 1.1),
            shear_x_range=(-5.0, 5.0),
            shear_y_range=(-5.0, 5.0),
        )

    result = benchmark(run)  # type: ignore[operator]

    assert result.matrix.shape == (2, 3)
    assert result.matrix.dtype == np.float64
    assert bool(np.all(np.isfinite(result.matrix)))
    assert result.source_size == source_size

    benchmark.extra_info.update(  # type: ignore[attr-defined]
        {"operation": "sample_affine_random_ranges", "implementation": "improcv"}
    )


@pytest.mark.benchmark(group="affine-python-geometry")
def test_expand_affine_canvas(benchmark: object) -> None:
    params = _build_affine_params(640, 480)

    result = benchmark(im.expand_affine_canvas, params)  # type: ignore[operator]

    assert result.output_size is not None
    assert result.output_size[0] >= params.source_size[0]
    assert result.output_size[1] >= params.source_size[1]

    benchmark.extra_info.update(  # type: ignore[attr-defined]
        {"operation": "expand_affine_canvas", "implementation": "improcv"}
    )


# --- affine-image-only ---------------------------------------------------------------------


def test_apply_affine_image_only_matches_raw_kernel(size: tuple[int, int]) -> None:
    """Correctness, not a benchmark: `improcv.apply_affine`'s image-only output must be
    bit-for-bit identical to the raw `cv2.warpAffine` call it wraps.
    """
    width, height = size
    image = _build_image(width, height)
    params = _build_affine_params(width, height)

    raw_result = cv2.warpAffine(
        image,
        params.matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    improcv_result = im.apply_affine(image, params)

    assert np.array_equal(raw_result, improcv_result)


@pytest.mark.benchmark(group="affine-image-only")
def test_apply_affine_image_only(
    benchmark: object,
    implementation: str,
    image: np.ndarray,
    affine_params: im.AffineParameters,
    size: tuple[int, int],
) -> None:
    width, height = size
    dsize = (width, height)

    def run_raw() -> np.ndarray:
        return cv2.warpAffine(
            image,
            affine_params.matrix,
            dsize,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=_IMAGE_BORDER_VALUE,
        )

    def run_improcv() -> np.ndarray:
        return im.apply_affine(image, affine_params)

    run = run_raw if implementation == "raw" else run_improcv
    result = benchmark(run)  # type: ignore[operator]

    assert result.shape == (height, width, 3)
    assert result.dtype == np.uint8

    benchmark.extra_info.update(  # type: ignore[attr-defined]
        {
            "operation": "apply_affine_image_only",
            "implementation": implementation,
            "width": width,
            "height": height,
            "image_dtype": "uint8",
            "mask_dtype": None,
            "interpolation": "INTER_LINEAR",
            "border_mode": "BORDER_CONSTANT",
            "image_border_value": _IMAGE_BORDER_VALUE,
            "mask_border_value": None,
        }
    )


# --- affine-image-mask ---------------------------------------------------------------------


def test_apply_affine_image_mask_matches_raw_kernels(size: tuple[int, int]) -> None:
    """Correctness, not a benchmark: `improcv.apply_affine`'s image and mask outputs must each
    be bit-for-bit identical to their own raw `cv2.warpAffine` call -- the mask call forced to
    `INTER_NEAREST`, exactly as `apply_affine` itself always forces internally.
    """
    width, height = size
    image = _build_image(width, height)
    mask = _build_mask(width, height)
    params = _build_affine_params(width, height)
    dsize = (width, height)

    raw_image = cv2.warpAffine(
        image,
        params.matrix,
        dsize,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    raw_mask = cv2.warpAffine(
        mask,
        params.matrix,
        dsize,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=_MASK_BORDER_VALUE,
    )

    wrapped = im.apply_affine(image, params, mask=mask, mask_border_value=_MASK_BORDER_VALUE)

    assert np.array_equal(raw_image, wrapped.image)
    assert np.array_equal(raw_mask, wrapped.mask)


@pytest.mark.benchmark(group="affine-image-mask")
def test_apply_affine_image_mask(
    benchmark: object,
    implementation: str,
    image: np.ndarray,
    mask: np.ndarray,
    affine_params: im.AffineParameters,
    size: tuple[int, int],
) -> None:
    width, height = size
    dsize = (width, height)

    def run_raw() -> tuple[np.ndarray, np.ndarray]:
        raw_image = cv2.warpAffine(
            image,
            affine_params.matrix,
            dsize,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=_IMAGE_BORDER_VALUE,
        )
        raw_mask = cv2.warpAffine(
            mask,
            affine_params.matrix,
            dsize,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=_MASK_BORDER_VALUE,
        )
        return raw_image, raw_mask

    def run_improcv() -> im.AugmentedImageMask:
        return im.apply_affine(
            image, affine_params, mask=mask, mask_border_value=_MASK_BORDER_VALUE
        )

    if implementation == "raw":
        result = benchmark(run_raw)  # type: ignore[operator]
        result_image, result_mask = result
    else:
        result = benchmark(run_improcv)  # type: ignore[operator]
        result_image, result_mask = result.image, result.mask

    assert result_image.shape == (height, width, 3)
    assert result_mask.shape == (height, width)
    assert result_mask.dtype == np.uint8

    benchmark.extra_info.update(  # type: ignore[attr-defined]
        {
            "operation": "apply_affine_image_mask",
            "implementation": implementation,
            "width": width,
            "height": height,
            "image_dtype": "uint8",
            "mask_dtype": "uint8",
            "interpolation": "INTER_LINEAR/INTER_NEAREST",
            "border_mode": "BORDER_CONSTANT",
            "image_border_value": _IMAGE_BORDER_VALUE,
            "mask_border_value": _MASK_BORDER_VALUE,
        }
    )
