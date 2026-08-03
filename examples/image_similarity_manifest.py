"""Discover, hash, persist, move, and reload -- a complete manifest persistence workflow.

Builds the same four tiny synthetic 8x8 grayscale images as `image_similarity.py`, discovers them
with `improcv.discover_images`, and decodes and hashes each one exactly once with
`improcv.average_hash`. Unlike `image_similarity.py` (which keeps its hash mapping entirely
in-memory and re-searches it at a second, wider threshold), this recipe is about what happens
*after* hashing: the hashes are stored under keys relative to the dataset root, wrapped in a
`PerceptualHashManifest`, and written out as deterministic JSON with `manifest.save()`. The dataset
directory is then physically moved to a different location, the manifest is read back with
`PerceptualHashManifest.load()`, and a similarity search runs against the *reloaded* hashes --
without decoding a single image or computing a single hash again. The original in-memory hash
mapping is even deleted before that search, so there is no way for it to accidentally supply the
answer.

This is deliberately a separate script from `image_similarity.py`, not a rewrite of it: the two
recipes demonstrate different concerns and are meant to be read (and copied) independently.
`image_similarity.py` is about hand-verifiable hashes, distances, and re-searching one in-memory
mapping at a different threshold; it says nothing about persistence, portability, or relative
identifiers. This script says nothing new about hashing or thresholds -- it reuses the same
threshold-2 result and does not re-run the search at a second threshold -- and focuses entirely
on: relative, portable manifest identifiers; saving and reloading a manifest as deterministic
JSON; and the fact that moving the dataset root doesn't invalidate those identifiers, since they
were never tied to any absolute location in the first place.

A `PerceptualHashManifest` is a snapshot, not a cache: it does not check whether an image still
exists, does not detect that a dataset moved, and does not know or care whether a stored hash is
still valid for its image's current content -- reloading it and reusing its hashes here is a
deliberate choice made once, not something the manifest verifies or automates for you.

Run with:

    uv run python examples/image_similarity_manifest.py
"""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import cast

import cv2
import numpy as np

import improcv as im


def build_similarity_images() -> dict[str, np.ndarray]:
    """Build the four deterministic 8x8 uint8 grayscale images used by this recipe.

    Each image has exactly 32 white (255) and 32 black (0) pixels. Duplicated here rather than
    imported from `image_similarity.py`: each `examples/*.py` script is meant to be copied and
    read on its own, with no dependency on any other script in this directory.

    - ``a_base.png``: a checkerboard, ``base[y, x] = 255 if (x + y) % 2 else 0``.
    - ``b_variant.png``: a copy of A with the pixels at (row 0, col 0) and (row 0, col 1) swapped.
    - ``c_variant_more.png``: a copy of B with an additional swap at (row 1, col 0)/(row 1, col 1).
    - ``z_unrelated.png``: a vertical split -- columns 0-3 are 0, columns 4-7 are 255.
    """
    row, col = np.indices((8, 8))
    base = np.where((col + row) % 2 == 0, 0, 255).astype(np.uint8)

    variant = base.copy()
    variant[0, 0], variant[0, 1] = variant[0, 1], variant[0, 0]

    variant_more = variant.copy()
    variant_more[1, 0], variant_more[1, 1] = variant_more[1, 1], variant_more[1, 0]

    unrelated = np.zeros((8, 8), dtype=np.uint8)
    unrelated[:, 4:] = 255

    return {
        "a_base.png": base,
        "b_variant.png": variant,
        "c_variant_more.png": variant_more,
        "z_unrelated.png": unrelated,
    }


def write_dataset(root: Path, images: Mapping[str, np.ndarray]) -> Path:
    """Write `images` as PNGs under a fresh `dataset` directory inside `root`.

    Returns the created dataset directory.
    """
    dataset_dir = root / "dataset"
    dataset_dir.mkdir(parents=True)
    for name, image in images.items():
        destination = dataset_dir / name
        ok = cv2.imwrite(str(destination), image)
        if not ok:
            raise RuntimeError(f"failed to write {destination}")
    return dataset_dir


def main() -> None:
    images = build_similarity_images()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        dataset_dir = write_dataset(root, images)

        paths = im.discover_images(dataset_dir)
        assert len(paths) == 4, f"expected 4 discovered files, got {len(paths)}"
        assert [path.name for path in paths] == [
            "a_base.png",
            "b_variant.png",
            "c_variant_more.png",
            "z_unrelated.png",
        ], [path.name for path in paths]
        a_path, b_path, c_path, d_path = paths

        # Decode and hash each discovered file exactly once, storing each hash under a key
        # relative to the dataset root -- never the absolute path `discover_images` returned.
        hash_size = 8
        decode_count = 0
        hash_count = 0
        hashes: dict[Path, im.PerceptualHash] = {}
        for path in paths:
            decoded = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            decode_count += 1
            if decoded is None:
                raise RuntimeError(f"could not decode {path}")
            # cv2.imread's stubs type the result as the loose MatLike; IMREAD_GRAYSCALE always
            # produces a uint8 array in practice.
            hash_value = im.average_hash(cast(im.ImageU8, decoded), hash_size=hash_size)
            hash_count += 1
            hashes[path.relative_to(dataset_dir)] = hash_value

        assert decode_count == 4, decode_count
        assert hash_count == 4, hash_count

        a_rel = a_path.relative_to(dataset_dir)
        b_rel = b_path.relative_to(dataset_dir)
        c_rel = c_path.relative_to(dataset_dir)
        d_rel = d_path.relative_to(dataset_dir)
        assert str(hashes[a_rel]) == "55aa55aa55aa55aa"
        assert str(hashes[b_rel]) == "95aa55aa55aa55aa"
        assert str(hashes[c_rel]) == "956a55aa55aa55aa"
        assert str(hashes[d_rel]) == "0f0f0f0f0f0f0f0f"

        manifest = im.PerceptualHashManifest.from_hashes(
            hashes,
            algorithm=im.PerceptualHashAlgorithm.AVERAGE_HASH,
            hash_size=hash_size,
        )
        assert len(manifest.entries) == 4
        assert [entry.path.as_posix() for entry in manifest.entries] == [
            "a_base.png",
            "b_variant.png",
            "c_variant_more.png",
            "z_unrelated.png",
        ]
        for entry in manifest.entries:
            assert type(entry.path) is PurePosixPath
            assert not entry.path.is_absolute()
        assert manifest.to_hashes() == {entry.path: entry.hash for entry in manifest.entries}
        assert list(manifest.to_hashes().keys()) == [entry.path for entry in manifest.entries]

        # The manifest file lives next to the dataset, not inside it.
        manifest_path = root / "image-hashes.json"
        save_result = manifest.save(manifest_path)
        assert save_result is None
        assert manifest_path.read_bytes() == manifest.to_json().encode("utf-8")

        # Keep only what's needed to check the round trip later; the original in-memory mapping
        # is gone, so the upcoming pair search cannot accidentally fall back to it.
        expected_json = manifest.to_json()
        del hashes

        # Move the dataset root -- the manifest's relative identifiers don't reference it at all,
        # so nothing about the manifest needs to change.
        moved_dataset_dir = root / "moved-dataset"
        dataset_dir.rename(moved_dataset_dir)
        assert not dataset_dir.exists()
        assert moved_dataset_dir.exists()
        for entry in manifest.entries:
            moved_path = moved_dataset_dir.joinpath(*entry.path.parts)
            assert moved_path.is_file(), moved_path

        restored = im.PerceptualHashManifest.load(manifest_path)
        assert restored == manifest
        assert restored.to_json() == expected_json
        assert restored.algorithm == im.PerceptualHashAlgorithm.AVERAGE_HASH
        assert restored.hash_size == hash_size
        assert len(restored.entries) == 4
        for entry in restored.entries:
            assert type(entry.path) is PurePosixPath
            assert not entry.path.is_absolute()

        # Pair search uses only the reloaded hashes -- no decoding, no hashing happens here.
        threshold = 2
        pairs = im.find_similar_image_pairs(restored.to_hashes(), max_distance=threshold)
        found = tuple(
            (pair.first.as_posix(), pair.second.as_posix(), pair.distance) for pair in pairs
        )
        expected_pairs = (
            ("a_base.png", "b_variant.png", 2),
            ("b_variant.png", "c_variant_more.png", 2),
        )
        assert found == expected_pairs, found
        for pair in pairs:
            assert "z_unrelated.png" not in (pair.first.as_posix(), pair.second.as_posix())

        # Reusing the saved snapshot never touched cv2.imread or average_hash again.
        assert decode_count == 4, decode_count
        assert hash_count == 4, hash_count

        print(f"images discovered: {len(paths)}")
        print(f"images decoded: {decode_count}")
        print(f"hashes computed: {hash_count}")
        print("algorithm: average_hash")
        print(f"hash size: {hash_size} (64 bits)")
        print(f"manifest entries: {len(manifest.entries)}")
        print("manifest paths relative: yes")
        print("manifest round-trip identical: yes")
        print("dataset moved after save: yes")
        print("rehash after load: no")
        print(f"threshold: {threshold}")
        print(f"similar pairs: {len(pairs)}")
        for pair in pairs:
            print(f"{pair.distance}  {pair.first.as_posix()} <-> {pair.second.as_posix()}")


if __name__ == "__main__":
    main()
