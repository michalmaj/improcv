"""Benchmarks for improcv's perceptual hashing API (`average_hash`/`phash`), scaling with the
size of a BGR `uint8` input image.

Only ever collected by an explicit `uv run --group benchmark pytest benchmarks/` --
`benchmarks/` sits outside `[tool.pytest.ini_options].testpaths`, so a plain `uv run pytest`
never sees this file. See `benchmarks/README.md` for what these cases answer, how to run a
stable local baseline, and how to read the results.

Two `@pytest.mark.benchmark(group=...)` groups (a real `pytest-benchmark` grouping, reflected in
each benchmark entry's own `group` field, not just this comment):

- `hashing-average-hash` -- the complete public `average_hash(image, hash_size=8)` call, at three
  source image sizes.
- `hashing-phash` -- the complete public `phash(image, hash_size=8)` call, at the same three
  sizes.

`hash_size` is fixed at 8 throughout -- hash-size scaling is a distinct question, deliberately
out of scope here (see "Explicitly out of scope" below and `benchmarks/README.md`).

Every source image is a deterministic, seeded, C-contiguous `(height, width, 3)` `uint8` BGR
array, built once per size in a session-scoped fixture shared by both groups, entirely outside
the timed `benchmark(...)` call. No image is read from or written to disk, and no encoded image
format (PNG/JPEG) is involved anywhere in this file -- image decoding and dataset discovery are
a separate concern (see `benchmark_discovery.py`) and are not exercised here.

The timed region for each case is the *complete* public function call -- input validation,
`hash_size` validation, the resize, the BGR-to-grayscale conversion, every algorithm-specific
step (`average_hash`'s mean/threshold; `phash`'s float32 cast, `cv2.dct`, block selection, DC
zeroing, `cv2.mean`, threshold cast), bit packing, and `PerceptualHash` construction. None of
these steps is extracted or pre-computed outside the timed call -- see `test_average_hash`/
`test_phash` below for exactly what happens before vs. inside `benchmark(...)`.

There is no raw baseline here (no hand-written NumPy/OpenCV pipeline, no `cv2.img_hash`): no
single raw kernel corresponds to either function's complete public contract, since both
`average_hash` and `phash` are themselves multi-step pipelines (see `improcv/hashing.py`) --
reimplementing that pipeline by hand in a benchmark would just duplicate the implementation
under a different name, not provide an independent reference. `cv2.img_hash` (OpenCV's own
reference implementation both functions are bit-compatible with, for `uint8` input) additionally
requires `opencv-contrib-python`, which this project does not depend on, and returns its result
as a packed-byte array through a different API shape -- not an `improcv.PerceptualHash` -- so a
ratio against it would compare two different result types under two different dependency
footprints, not a same-contract, same-output raw/wrapper pair like the affine benchmarks'
raw/`improcv` comparisons. This first hashing slice measures how the public API itself scales
with source image size, not a ratio against a semantically different implementation.

Explicitly out of scope for this file (see `benchmarks/README.md`'s "Non-goals" for the full,
project-wide list):

- `find_similar_image_pairs` / pair-search scaling;
- Hamming distance (`PerceptualHash.distance`) as its own microbenchmark;
- dataset discovery or any filesystem access;
- image decoding (`cv2.imread`/`cv2.imencode`) of any kind;
- `hash_size` as a scaling axis (fixed at 8 throughout);
- grayscale/BGRA input as a separate axis (BGR only);
- any algorithm other than `average_hash`/`phash`;
- `opencv-contrib-python` / `cv2.img_hash`.

Results reported here describe wall-clock scaling of the public call, nothing about perceptual
hash quality, collision rate, or robustness.
"""

from __future__ import annotations

import time
from typing import NamedTuple

import numpy as np
import pytest

import improcv as im
from improcv.hashing import PerceptualHashAlgorithm

_SIZES: tuple[tuple[int, int], ...] = (
    (64, 64),
    (640, 480),
    (1920, 1080),
)

_SIZE_IDS = (
    "64x64",
    "640x480",
    "1920x1080",
)

_HASH_SIZE = 8
_BASE_SEED = 20260803


class HashingDataset(NamedTuple):
    """One deterministic BGR `uint8` source image per size, shared by both hashing benchmarks.

    `original` is an independent copy, kept solely to confirm that neither `average_hash` nor
    `phash` mutates `image`.
    """

    image: np.ndarray
    original: np.ndarray
    width: int
    height: int
    channels: int
    seed: int


def _seed_for(width: int, height: int) -> int:
    """Deterministically derive a seed from `(width, height)`.

    Never Python's `hash()`: it is randomized per-process for `str`, and is not a documented,
    stable derivation for other types across Python versions -- this needs a seed that is the
    same on every run and every machine.
    """
    return _BASE_SEED + width * 1_000 + height


def _build_hashing_dataset(width: int, height: int) -> HashingDataset:
    seed = _seed_for(width, height)
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)

    assert image.dtype == np.uint8
    assert image.shape == (height, width, 3)
    assert image.flags.c_contiguous
    assert image.nbytes == width * height * 3

    original = image.copy()
    return HashingDataset(
        image=image, original=original, width=width, height=height, channels=3, seed=seed
    )


@pytest.fixture(scope="session", params=_SIZES, ids=_SIZE_IDS)
def hashing_dataset(request: pytest.FixtureRequest) -> HashingDataset:
    """One deterministic BGR `uint8` dataset per source size, built once per session and shared
    by `average_hash` and `phash`."""
    width, height = request.param
    setup_start = time.perf_counter()
    dataset = _build_hashing_dataset(width, height)
    setup_elapsed = time.perf_counter() - setup_start
    # Diagnostic only -- never a benchmark result or a performance claim; visible with `-s`.
    print(
        f"\n[hashing benchmark setup] {width}x{height} BGR uint8 took {setup_elapsed:.3f}s, "
        f"image.nbytes={dataset.image.nbytes}"
    )
    return dataset


def _common_extra_info(
    dataset: HashingDataset, *, operation: str, algorithm: str
) -> dict[str, object]:
    """Metadata shared by both hashing benchmarks -- scalar/string/bool only, no arrays or paths."""
    return {
        "implementation": "improcv",
        "operation": operation,
        "algorithm": algorithm,
        "source_width": dataset.width,
        "source_height": dataset.height,
        "source_pixels": dataset.width * dataset.height,
        "source_channels": dataset.channels,
        "source_color_model": "BGR",
        "source_dtype": "uint8",
        "source_layout": "C",
        "source_nbytes": dataset.image.nbytes,
        "source_policy": "seeded-uniform-uint8",
        "source_seed": dataset.seed,
        "hash_size": _HASH_SIZE,
        "hash_bits": _HASH_SIZE**2,
        "output_type": "PerceptualHash",
        "image_mutated": False,
        "filesystem_io": False,
        "image_decoding": False,
        "opencv_contrib_required": False,
    }


# --- hashing-average-hash -----------------------------------------------------------------


@pytest.mark.benchmark(group="hashing-average-hash")
def test_average_hash(benchmark: object, hashing_dataset: HashingDataset) -> None:
    """The complete public `average_hash(image, hash_size=8)` call, at scale.

    Timed region: input/`hash_size` validation, resize to `8x8`, BGR-to-grayscale conversion,
    `cv2.mean`, round-half-to-even threshold, bit comparison, bit packing, and `PerceptualHash`
    construction -- nothing is extracted or pre-computed outside `benchmark(...)`.
    """
    expected = im.average_hash(hashing_dataset.image, hash_size=_HASH_SIZE)
    assert isinstance(expected, im.PerceptualHash)
    assert expected.algorithm == PerceptualHashAlgorithm.AVERAGE_HASH
    assert expected.hash_size == _HASH_SIZE
    assert len(expected) == 64
    assert len(str(expected)) == 16

    result = benchmark(  # type: ignore[operator]
        im.average_hash, hashing_dataset.image, hash_size=_HASH_SIZE
    )

    assert result == expected
    assert np.array_equal(hashing_dataset.image, hashing_dataset.original)

    extra_info = _common_extra_info(
        hashing_dataset, operation="average_hash", algorithm="average_hash"
    )
    extra_info.update(
        {
            "pipeline": "resize-then-grayscale-rounded-mean-threshold",
            "resize_interpolation": "INTER_LINEAR_EXACT",
            "target_grid_width": _HASH_SIZE,
            "target_grid_height": _HASH_SIZE,
            "threshold_statistic": "rounded-mean",
            "dct": False,
            "alpha_handling": "not-applicable-bgr-input",
        }
    )
    benchmark.extra_info.update(extra_info)  # type: ignore[attr-defined]


# --- hashing-phash --------------------------------------------------------------------------


@pytest.mark.benchmark(group="hashing-phash")
def test_phash(benchmark: object, hashing_dataset: HashingDataset) -> None:
    """The complete public `phash(image, hash_size=8)` call, at scale.

    Timed region: input/`hash_size` validation, resize to `32x32`, BGR-to-grayscale conversion,
    `float32` cast, `cv2.dct`, top-left `8x8` block selection/copy, DC zeroing, `cv2.mean`,
    `float32` threshold cast, bit comparison, bit packing, and `PerceptualHash` construction --
    nothing is extracted or pre-computed outside `benchmark(...)`.
    """
    expected = im.phash(hashing_dataset.image, hash_size=_HASH_SIZE)
    assert isinstance(expected, im.PerceptualHash)
    assert expected.algorithm == PerceptualHashAlgorithm.PHASH
    assert expected.hash_size == _HASH_SIZE
    assert len(expected) == 64
    assert len(str(expected)) == 16

    result = benchmark(  # type: ignore[operator]
        im.phash, hashing_dataset.image, hash_size=_HASH_SIZE
    )

    assert result == expected
    assert np.array_equal(hashing_dataset.image, hashing_dataset.original)

    extra_info = _common_extra_info(hashing_dataset, operation="phash", algorithm="phash")
    extra_info.update(
        {
            "pipeline": "resize-then-grayscale-float32-dct-mean-threshold",
            "resize_interpolation": "INTER_LINEAR_EXACT",
            "dct_input_width": _HASH_SIZE * 4,
            "dct_input_height": _HASH_SIZE * 4,
            "dct_dtype": "float32",
            "selected_block_width": _HASH_SIZE,
            "selected_block_height": _HASH_SIZE,
            "highfreq_factor": 4,
            "threshold_statistic": "cv2-mean-cast-float32",
            "dc_policy": "zeroed-before-threshold-and-output-bit",
            "dct": True,
            "alpha_handling": "not-applicable-bgr-input",
        }
    )
    benchmark.extra_info.update(extra_info)  # type: ignore[attr-defined]
