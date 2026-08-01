"""Benchmarks for improcv's dataset discovery API against a warm filesystem cache.

Only ever collected by an explicit `uv run --group benchmark pytest benchmarks/` --
`benchmarks/` sits outside `[tool.pytest.ini_options].testpaths`, so a plain `uv run pytest`
never sees this file. See `benchmarks/README.md` for what these cases answer, how to run a
stable local baseline, and how to read the results.

Two `@pytest.mark.benchmark(group=...)` groups (a real `pytest-benchmark` grouping, reflected
in each benchmark entry's own `group` field, not just this comment):

- `discovery-images` -- `discover_images` over a single root, at three sizes.
- `discovery-pairs` -- `discover_image_mask_pairs` over two roots, at three sizes.

There is no raw baseline here (no `os.walk`/`Path.rglob`/`glob` comparison, unlike the affine
benchmarks' raw/`improcv` pairing): `discover_images`'s contract -- a fresh
`os.stat(..., follow_symlinks=False)` per entry, symlink/reparse-point skipping, a hidden-file
policy, a deterministic global POSIX-relative sort, extension normalization -- has no simple
raw equivalent of matching strength. `discover_image_mask_pairs` additionally strips
extensions, groups by key, checks for duplicates, and validates a strict bijection before
pairing; there is no raw manual-pairing baseline either. This first discovery slice measures
how the public API itself scales with entry count, not a ratio against a semantically weaker
iterator.

Every dataset is a deterministic, sharded tree of zero-byte files, created with `Path.touch()`
-- never `cv2.imwrite`, a valid encoded image, or a random NumPy array. This is not a shortcut
that changes what is measured: `discover_images` finds files by extension only and never opens,
decodes, or otherwise inspects their content (see `improcv/discovery.py`), so a zero-byte file
with a matching extension is exactly the input its documented contract expects. File setup
happens entirely outside the timed region, in a session-scoped fixture shared by both
benchmarks below; naming these entries "extension-only discovery entries" throughout (never
"invalid images") reflects that discovery never looks past the filename.

Every timed call is preceded by one untimed, asserted call to the same public function on the
same dataset. This validates the dataset once and pre-warms the OS filesystem metadata cache
for its directory entries and inodes -- both traversal families below are warm-filesystem-cache
measurements only. Cold-cache traversal (post-boot, post-`sync && purge`, or on a freshly
mounted filesystem) is deliberately out of scope: it depends on OS, filesystem, disk hardware,
and machine state in ways this harness cannot control or reproduce, and there is no cache-drop
mechanism here (no `sync`, no `purge`, no sudo, no fresh tree per round).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import NamedTuple

import pytest

import improcv as im

_DISCOVERY_COUNTS: tuple[int, ...] = (100, 1_000, 10_000)
_SHARD_COUNT = 10


class DiscoveryDataset(NamedTuple):
    """One deterministic, sharded discovery fixture: `count` images paired with `count` masks."""

    image_root: Path
    mask_root: Path
    count: int
    shard_count: int


def _shard_name(shard_index: int) -> str:
    return f"shard_{shard_index:02d}"


def _sample_stem(index: int) -> str:
    return f"sample_{index:06d}"


def _pairing_key(index: int, shard_count: int) -> str:
    return f"{_shard_name(index % shard_count)}/{_sample_stem(index)}"


def _build_discovery_dataset(root: Path, count: int, shard_count: int) -> DiscoveryDataset:
    """Create `count` zero-byte image/mask file pairs under `root`, split into `shard_count` shards.

    Ten shard directories (`shard_00` .. `shard_09`) are created under each of `root/images`
    and `root/masks`; entry `index` is assigned to shard `index % shard_count`, giving evenly
    distributed, equal-sized shards for any `count` divisible by `shard_count`. Every file is
    created empty via `Path.touch()` -- see the module docstring for why this matches
    `discover_images`'s extension-only contract exactly.
    """
    image_root = root / "dataset" / "images"
    mask_root = root / "dataset" / "masks"
    image_shard_dirs = [image_root / _shard_name(shard) for shard in range(shard_count)]
    mask_shard_dirs = [mask_root / _shard_name(shard) for shard in range(shard_count)]
    for shard_dir in (*image_shard_dirs, *mask_shard_dirs):
        shard_dir.mkdir(parents=True)

    for index in range(count):
        shard = index % shard_count
        stem = _sample_stem(index)
        (image_shard_dirs[shard] / f"{stem}.jpg").touch()
        (mask_shard_dirs[shard] / f"{stem}.png").touch()

    return DiscoveryDataset(
        image_root=image_root, mask_root=mask_root, count=count, shard_count=shard_count
    )


@pytest.fixture(
    scope="session",
    params=_DISCOVERY_COUNTS,
    ids=lambda count: f"{count}-entries",
)
def discovery_dataset(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
) -> DiscoveryDataset:
    """Build one sharded dataset per size, once per session, shared by both benchmarks below."""
    count = request.param
    root = tmp_path_factory.mktemp(f"discovery-{count}")

    setup_start = time.perf_counter()
    dataset = _build_discovery_dataset(root, count, _SHARD_COUNT)
    setup_elapsed = time.perf_counter() - setup_start
    # Diagnostic only -- never a benchmark result or a performance claim; visible with `-s`.
    print(f"\n[discovery benchmark setup] {count} entries per side took {setup_elapsed:.3f}s")

    return dataset


# --- discovery-images ------------------------------------------------------------------------


@pytest.mark.benchmark(group="discovery-images")
def test_discover_images(benchmark: object, discovery_dataset: DiscoveryDataset) -> None:
    """Full recursive traversal + fresh stat + extension filter + global sort, at scale.

    Times the complete public `discover_images` call (root validation, default-extension
    handling, `scandir`, a fresh `os.stat` per entry, recursion, list materialization, and the
    final global sort) against `discovery_dataset.image_root`, over a warm filesystem cache.
    """
    expected = im.discover_images(
        discovery_dataset.image_root,
        recursive=True,
        extensions=None,
        include_hidden=False,
    )
    assert len(expected) == discovery_dataset.count

    result = benchmark(  # type: ignore[operator]
        im.discover_images,
        discovery_dataset.image_root,
        recursive=True,
        extensions=None,
        include_hidden=False,
    )

    assert isinstance(result, tuple)
    assert len(result) == discovery_dataset.count
    assert result == expected

    previous_key = ""
    for path in result:
        assert path.is_relative_to(discovery_dataset.image_root)
        assert path.name.endswith(".jpg")
        key = path.relative_to(discovery_dataset.image_root).as_posix()
        assert key > previous_key
        previous_key = key

    benchmark.extra_info.update(  # type: ignore[attr-defined]
        {
            "operation": "discover_images",
            "implementation": "improcv",
            "file_count": discovery_dataset.count,
            "pair_count": None,
            "root_count": 1,
            "shard_count": discovery_dataset.shard_count,
            "directory_depth": 1,
            "recursive": True,
            "include_hidden": False,
            "extensions": "default",
            "file_content_policy": "zero-byte-extension-only",
            "filesystem_cache": "warm",
            "file_contents_opened": False,
        }
    )


# --- discovery-pairs --------------------------------------------------------------------------


@pytest.mark.benchmark(group="discovery-pairs")
def test_discover_image_mask_pairs(benchmark: object, discovery_dataset: DiscoveryDataset) -> None:
    """Two-sided discovery + extension stripping + grouping + duplicate/key-set checks, at scale.

    Times the complete public `discover_image_mask_pairs` call (two full `discover_images`
    traversals, pairing-key construction via extension stripping, grouping, duplicate
    detection, key-set comparison, and the final sorted pairing) against
    `discovery_dataset.image_root`/`discovery_dataset.mask_root`, over a warm filesystem cache.
    """
    expected = im.discover_image_mask_pairs(
        discovery_dataset.image_root,
        discovery_dataset.mask_root,
        recursive=True,
        image_extensions=None,
        mask_extensions=None,
        include_hidden=False,
    )
    assert len(expected) == discovery_dataset.count

    result = benchmark(  # type: ignore[operator]
        im.discover_image_mask_pairs,
        discovery_dataset.image_root,
        discovery_dataset.mask_root,
        recursive=True,
        image_extensions=None,
        mask_extensions=None,
        include_hidden=False,
    )

    assert isinstance(result, tuple)
    assert len(result) == discovery_dataset.count
    assert result == expected

    for pair in result:
        assert pair.image.suffix == ".jpg"
        assert pair.mask.suffix == ".png"
        image_key = pair.image.relative_to(discovery_dataset.image_root).with_suffix("").as_posix()
        mask_key = pair.mask.relative_to(discovery_dataset.mask_root).with_suffix("").as_posix()
        assert image_key == mask_key

    first_key = _pairing_key(0, discovery_dataset.shard_count)
    last_key = _pairing_key(discovery_dataset.count - 1, discovery_dataset.shard_count)
    assert (
        result[0].image.relative_to(discovery_dataset.image_root).with_suffix("").as_posix()
        == first_key
    )
    assert (
        result[-1].image.relative_to(discovery_dataset.image_root).with_suffix("").as_posix()
        == last_key
    )

    benchmark.extra_info.update(  # type: ignore[attr-defined]
        {
            "operation": "discover_image_mask_pairs",
            "implementation": "improcv",
            "file_count": discovery_dataset.count * 2,
            "files_per_root": discovery_dataset.count,
            "pair_count": discovery_dataset.count,
            "root_count": 2,
            "shard_count_per_root": discovery_dataset.shard_count,
            "directory_depth": 1,
            "recursive": True,
            "include_hidden": False,
            "image_extensions": "default",
            "mask_extensions": "default",
            "file_content_policy": "zero-byte-extension-only",
            "filesystem_cache": "warm",
            "file_contents_opened": False,
        }
    )
