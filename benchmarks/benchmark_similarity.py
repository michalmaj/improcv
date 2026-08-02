"""Benchmarks for improcv's pairwise similarity search (`find_similar_image_pairs`), scaling
with the number of precomputed perceptual hashes, at two result-cardinality regimes.

Only ever collected by an explicit `uv run --group benchmark pytest benchmarks/` --
`benchmarks/` sits outside `[tool.pytest.ini_options].testpaths`, so a plain `uv run pytest`
never sees this file. See `benchmarks/README.md` for what these cases answer, how to run a
stable local baseline, and how to read the results.

Two `@pytest.mark.benchmark(group=...)` groups (a real `pytest-benchmark` grouping, reflected in
each benchmark entry's own `group` field, not just this comment):

- `similarity-no-matches` -- the complete public `find_similar_image_pairs(hashes,
  max_distance=0)` call, where every hash is unique and the result is always empty.
- `similarity-all-matches` -- the complete public `find_similar_image_pairs(hashes,
  max_distance=64)` call (`64 == hash_size**2`, the maximum legal threshold for `hash_size=8`),
  where every unordered pair is within the threshold and the result materializes all of them.

at three item counts (`30`, `100`, `300`), i.e. `435`, `4,950`, and `44,850` unordered pairs
respectively. Both regimes at a given item count share exactly the same input mapping, built
once per session -- this file measures how one public API scales at two very different result
cardinalities and branch outcomes, not two different inputs.

The input to `find_similar_image_pairs` is a `Mapping` of already-computed `PerceptualHash`
values -- this benchmark never reads a file, decodes an image, computes a real perceptual hash
(`average_hash`/`phash`), or calls `discover_images`. See `benchmark_hashing.py` for hash
*computation* cost; this file is exclusively about the pair-search step that consumes already-
computed hashes. Every hash value here is a synthetic, legal `PerceptualHash` object built from a
deterministic integer transform (see `_hash_for` below) -- it is not a claim about how real
perceptual hashes are distributed on any real dataset, only a cheap, reproducible way to obtain
`n` guaranteed-unique 64-bit hash values without touching NumPy, OpenCV, or the filesystem.

Path identifiers (`images/image_NNNNN.png`) are synthetic and are never opened, created, or
otherwise touched -- `find_similar_image_pairs` treats them as opaque identifiers and performs no
filesystem access at all (see `improcv/similarity.py`). The input mapping is built once per
session, entirely outside the timed `benchmark(...)` call; the *only* deliberate precomputation
here is the hash mapping itself, since the public API's contract starts from already-computed
hashes -- there would be no way to benchmark this function without first having some.

The timed region is the *complete* public function call: `max_distance` validation, `Mapping`
validation, `os.fspath()` plus `Path` normalization of every identifier, duplicate-normalized-key
detection, validation of every `PerceptualHash`, the shared `algorithm`/`hash_size` check, the
`max_distance` upper-bound check, sorting the normalized entries, enumerating every unordered
pair, every `PerceptualHash.distance` call, the threshold branch, `SimilarImagePair` construction
for every pair that matches, the final result sort, and the tuple conversion. None of these steps
is extracted or pre-computed outside `benchmark(...)` -- see `test_find_similar_image_pairs_no_
matches`/`test_find_similar_image_pairs_all_matches` below for exactly what happens before vs.
inside the timed call.

There is no raw baseline here (no hand-written `itertools.combinations` loop, no private access
to `PerceptualHash._value`, no manual XOR/`bit_count`): a hand-rolled version would duplicate a
fragment of `find_similar_image_pairs`'s own implementation, skip its path normalization and
`algorithm`/`hash_size` compatibility validation, and return raw tuples instead of
`SimilarImagePair` -- it would not be the same public workflow, so a ratio against it would not
be a meaningful comparison. The no-matches/all-matches contrast is a comparison *within* the same
public API at its two extreme result densities, not a raw/wrapper ratio.

**The all-matches/no-matches contrast compares two complete public workflows with different
result cardinalities and branch outcomes. It does not isolate an exact per-object materialization
cost** -- `all_matches - no_matches` is never computed anywhere in this file as an exact cost of
constructing `SimilarImagePair` objects.

Explicitly out of scope for this file (see `benchmarks/README.md`'s "Non-goals" for the full,
project-wide list): end-to-end discovery/decode/hash/search, perceptual hash computation, a
standalone `PerceptualHash.distance` microbenchmark, any intermediate match density between 0%
and 100%, any `max_distance` other than `0`/`64`, any `hash_size` other than `8`, any algorithm
other than `PHASH`, any concrete mapping type other than `dict`, a `str`-vs-`Path` key-type axis,
duplicate grouping/clustering, connected components, or any subquadratic/approximate search
structure (BK-tree, ANN, etc.).

Results reported here describe wall-clock scaling and Python-object materialization cost of the
public call, nothing about perceptual hash quality, collision rate, or real-world duplicate-
detection accuracy.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import NamedTuple

import pytest

import improcv as im

_ITEM_COUNTS: tuple[int, ...] = (
    30,
    100,
    300,
)

_ITEM_IDS = (
    "30-images",
    "100-images",
    "300-images",
)

_HASH_SIZE = 8
_HASH_BITS = 64

_HASH_VALUE_MULTIPLIER = 0x9E3779B97F4A7C15
_HASH_VALUE_MASK = (1 << _HASH_BITS) - 1


class SimilarityDataset(NamedTuple):
    """One precomputed-hash mapping, shared by both similarity benchmarks.

    `original_items` is an independent snapshot of `hashes.items()`, taken right after
    construction, kept solely to confirm that `find_similar_image_pairs` does not change the
    mapping's keys, values, or insertion order.
    """

    hashes: dict[str | os.PathLike[str], im.PerceptualHash]
    original_items: tuple[tuple[str | os.PathLike[str], im.PerceptualHash], ...]
    n_items: int
    unordered_pairs: int


def _path_for(index: int) -> Path:
    """A synthetic, never-opened relative path identifier for `index`."""
    return Path("images") / f"image_{index:05d}.png"


def _hash_for(index: int) -> im.PerceptualHash:
    """A synthetic, legal `PerceptualHash` for `index`, via a deterministic integer transform.

    `_HASH_VALUE_MULTIPLIER` is odd, so multiplication modulo `2**_HASH_BITS` is a bijection --
    for the small, distinct `index + 1` values used here (1..300), the resulting 64-bit values
    are guaranteed pairwise distinct. This is not `hash()`, not randomness, and not a real
    perceptual hash -- it exists only to produce cheap, reproducible, guaranteed-unique 64-bit
    hash values for benchmarking the pair-search step in isolation.
    """
    value = ((index + 1) * _HASH_VALUE_MULTIPLIER) & _HASH_VALUE_MASK
    text = f"{value:016x}"
    return im.PerceptualHash.from_hex(
        text, algorithm=im.PerceptualHashAlgorithm.PHASH, hash_size=_HASH_SIZE
    )


def _build_similarity_dataset(n_items: int) -> SimilarityDataset:
    """Build `n_items` synthetic (path, hash) entries and insert them in reverse canonical order.

    `_path_for` is zero-padded and monotonic in `index`, so `entries` below is already in
    canonical order (ascending `path.as_posix()`); inserting `reversed(entries)` means the timed
    `find_similar_image_pairs` call must actually normalize and sort the input to produce a
    canonically ordered result -- it cannot benefit from an already-sorted mapping.
    """
    entries: list[tuple[Path, im.PerceptualHash]] = [
        (_path_for(index), _hash_for(index)) for index in range(n_items)
    ]

    hash_values = [hash_value for _, hash_value in entries]
    assert len(set(hash_values)) == n_items, "synthetic hash values must all be unique"
    for hash_value in hash_values:
        assert hash_value.algorithm == im.PerceptualHashAlgorithm.PHASH
        assert hash_value.hash_size == _HASH_SIZE
        assert len(hash_value) == _HASH_BITS
        assert len(str(hash_value)) == 16

    posix_keys = [path.as_posix() for path, _ in entries]
    assert posix_keys == sorted(posix_keys), "entries must be built in canonical order"
    assert len(set(posix_keys)) == n_items, "no two identifiers may normalize to the same path"

    hashes: dict[str | os.PathLike[str], im.PerceptualHash] = dict(reversed(entries))
    reversed_keys = [key.as_posix() for key in hashes]  # type: ignore[union-attr]
    assert reversed_keys == list(reversed(posix_keys)), (
        "the mapping must be inserted in reverse canonical order"
    )

    original_items = tuple(hashes.items())
    unordered_pairs = n_items * (n_items - 1) // 2
    assert unordered_pairs == math.comb(n_items, 2)

    assert len(hashes) == n_items
    assert len(original_items) == n_items
    assert all(isinstance(key, Path) for key in hashes)
    assert all(isinstance(value, im.PerceptualHash) for value in hashes.values())

    return SimilarityDataset(
        hashes=hashes,
        original_items=original_items,
        n_items=n_items,
        unordered_pairs=unordered_pairs,
    )


@pytest.fixture(scope="session", params=_ITEM_COUNTS, ids=_ITEM_IDS)
def similarity_dataset(request: pytest.FixtureRequest) -> SimilarityDataset:
    """One precomputed-hash dataset per item count, built once per session and shared by both
    `similarity-no-matches` and `similarity-all-matches`."""
    n_items = request.param
    setup_start = time.perf_counter()
    dataset = _build_similarity_dataset(n_items)
    setup_elapsed = time.perf_counter() - setup_start
    # Diagnostic only -- never a benchmark result or a performance claim; visible with `-s`.
    print(
        f"\n[similarity benchmark setup] {n_items} precomputed phash values, "
        f"{dataset.unordered_pairs} unordered pairs, reverse insertion order, took "
        f"{setup_elapsed:.3f}s"
    )
    return dataset


def _common_extra_info(dataset: SimilarityDataset) -> dict[str, object]:
    """Metadata shared by both similarity benchmarks -- scalar/string/bool only, no collections."""
    return {
        "implementation": "improcv",
        "operation": "find_similar_image_pairs",
        "n_items": dataset.n_items,
        "unordered_pairs": dataset.unordered_pairs,
        "mapping_type": "dict",
        "mapping_key_type": "Path",
        "input_iteration_order": "reverse-canonical-path-order",
        "path_identifier_policy": "synthetic-zero-padded-relative-identifiers",
        "hash_algorithm": "phash",
        "hash_size": _HASH_SIZE,
        "hash_bits": _HASH_BITS,
        "hash_values_precomputed": True,
        "hash_values_unique": True,
        "hash_value_policy": "odd-multiplier-mod-2^64",
        "pair_enumeration": "every-unordered-pair-once",
        "result_type": "tuple[SimilarImagePair, ...]",
        "filesystem_io": False,
        "image_decoding": False,
        "image_hashing": False,
        "discover_images_called": False,
        "grouping_or_clustering": False,
        "parallelism": False,
        "input_mutated": False,
    }


# --- similarity-no-matches -----------------------------------------------------------------


@pytest.mark.benchmark(group="similarity-no-matches")
def test_find_similar_image_pairs_no_matches(
    benchmark: object, similarity_dataset: SimilarityDataset
) -> None:
    """The complete public `find_similar_image_pairs(hashes, max_distance=0)` call, at scale.

    Every hash is unique, so no pair can ever be within distance `0`: this exercises the full
    workflow (validation, normalization, sorting, every `n(n-1)/2` distance comparison, the
    threshold branch) down to sorting and tuple-converting an always-empty result -- without ever
    constructing a `SimilarImagePair`.
    """
    expected = im.find_similar_image_pairs(similarity_dataset.hashes, max_distance=0)
    assert expected == ()
    assert isinstance(expected, tuple)

    result = benchmark(  # type: ignore[operator]
        im.find_similar_image_pairs, similarity_dataset.hashes, max_distance=0
    )

    assert result == expected
    assert tuple(similarity_dataset.hashes.items()) == similarity_dataset.original_items

    extra_info = _common_extra_info(similarity_dataset)
    extra_info.update(
        {
            "result_regime": "no-matches",
            "max_distance": 0,
            "expected_matches": 0,
            "match_density": 0.0,
            "pair_objects_materialized": 0,
        }
    )
    benchmark.extra_info.update(extra_info)  # type: ignore[attr-defined]


# --- similarity-all-matches ----------------------------------------------------------------


@pytest.mark.benchmark(group="similarity-all-matches")
def test_find_similar_image_pairs_all_matches(
    benchmark: object, similarity_dataset: SimilarityDataset
) -> None:
    """The complete public `find_similar_image_pairs(hashes, max_distance=64)` call, at scale.

    `64 == hash_size**2` is the maximum legal threshold for `hash_size=8`, so every unordered
    pair is within it: this exercises the full workflow including constructing one
    `SimilarImagePair` per pair, collecting all of them, sorting the full result list, and
    converting it to a tuple.
    """
    expected = im.find_similar_image_pairs(similarity_dataset.hashes, max_distance=_HASH_BITS)
    assert isinstance(expected, tuple)
    assert len(expected) == similarity_dataset.unordered_pairs
    assert all(isinstance(pair, im.SimilarImagePair) for pair in expected)
    assert all(pair.first.as_posix() < pair.second.as_posix() for pair in expected)
    assert expected == tuple(
        sorted(
            expected,
            key=lambda pair: (pair.distance, pair.first.as_posix(), pair.second.as_posix()),
        )
    )

    result = benchmark(  # type: ignore[operator]
        im.find_similar_image_pairs, similarity_dataset.hashes, max_distance=_HASH_BITS
    )

    assert result == expected
    assert len(result) == similarity_dataset.unordered_pairs
    assert tuple(similarity_dataset.hashes.items()) == similarity_dataset.original_items

    extra_info = _common_extra_info(similarity_dataset)
    extra_info.update(
        {
            "result_regime": "all-matches",
            "max_distance": _HASH_BITS,
            "expected_matches": similarity_dataset.unordered_pairs,
            "match_density": 1.0,
            "pair_objects_materialized": similarity_dataset.unordered_pairs,
        }
    )
    benchmark.extra_info.update(extra_info)  # type: ignore[attr-defined]
