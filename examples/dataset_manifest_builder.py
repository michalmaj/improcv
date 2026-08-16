"""Discover, decode, hash, persist, move, and reload -- the concise, builder-based equivalent.

`examples/image_similarity_manifest.py` shows, step by step, what a dataset-to-manifest workflow
actually does: call `discover_images`, decode each file, hash it, store the hash under an
identifier relative to the dataset root, and hand the resulting mapping to
`PerceptualHashManifest.from_hashes`. This script is that exact same workflow, but performed by
one public function instead: `improcv.build_perceptual_hash_manifest` runs discovery, sequential
fixed 8-bit grayscale decoding, hashing, and relative-identifier construction internally, and
returns the finished manifest directly -- there is no separate discovery loop, no manual
`path.relative_to(root)`, and no explicit call to `from_hashes` in this script at all.

The builder only ever returns a `PerceptualHashManifest` -- it does not save it. `save`/`load`
remain separate, explicit steps the caller performs afterwards, exactly as they do in
`image_similarity_manifest.py`. A `PerceptualHashManifest` is a snapshot, not a cache: it does not
check whether an image still exists, does not detect that a dataset moved, and does not know or
care whether a stored hash is still valid for its image's current content -- this script does not
check freshness, and it does not demonstrate parallelism or performance; the dataset below is tiny
and processed sequentially, same as the manual version.

This is deliberately a separate, fully self-contained script, not a rewrite of either
`image_similarity.py` or `image_similarity_manifest.py`: it duplicates the small amount of
fixture-building code those two scripts also duplicate from each other, rather than importing it,
so it can be read and copied entirely on its own. The dataset root here uses a Unicode directory
name (and a nested subdirectory) specifically to exercise the builder's cross-platform contract:
identifiers are computed relative to the dataset root, so they stay correct however that root is
named or spelled.

Run with:

    uv run python examples/dataset_manifest_builder.py
"""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

import numpy as np

import improcv as im


def build_similarity_images() -> dict[str, np.ndarray]:
    """Build the four deterministic 8x8 uint8 grayscale images used by this recipe.

    Each image has exactly 32 white (255) and 32 black (0) pixels. Duplicated here rather than
    imported from another `examples/*.py` script: each script in this directory is meant to be
    copied and read entirely on its own.

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
    """Write `images` as PNGs under ``<root>/żółw-dataset/obrazy/``.

    Writes via `im.save_image` rather than `cv2.imwrite`, since the dataset root's name contains
    characters outside ASCII, and `cv2.imwrite`'s filename-based handling is not reliable for such
    paths on Windows -- `im.save_image` owns that Unicode-safe write boundary (in-memory
    `cv2.imencode` + `Path.write_bytes`) so this script doesn't have to. `save_image` does not
    create parent directories itself, so `images_dir.mkdir(parents=True)` still happens here.
    Returns the created dataset root (the ``żółw-dataset`` directory, not the ``obrazy``
    subdirectory).
    """
    dataset_root = root / "żółw-dataset"
    images_dir = dataset_root / "obrazy"
    images_dir.mkdir(parents=True)
    for name, pixels in images.items():
        im.save_image(images_dir / name, pixels)
    return dataset_root


def main() -> None:
    images = build_similarity_images()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        dataset_root = write_dataset(root, images)

        # One call replaces discover_images + a decode loop + average_hash/phash + relative_to +
        # PerceptualHashManifest.from_hashes -- see image_similarity_manifest.py for that
        # manual, step-by-step equivalent.
        manifest = im.build_perceptual_hash_manifest(
            dataset_root,
            algorithm=im.PerceptualHashAlgorithm.AVERAGE_HASH,
            hash_size=8,
        )

        assert manifest.algorithm is im.PerceptualHashAlgorithm.AVERAGE_HASH
        assert manifest.hash_size == 8
        assert len(manifest.entries) == 4

        expected_identifiers = [
            "obrazy/a_base.png",
            "obrazy/b_variant.png",
            "obrazy/c_variant_more.png",
            "obrazy/z_unrelated.png",
        ]
        assert [entry.path.as_posix() for entry in manifest.entries] == expected_identifiers
        for entry in manifest.entries:
            assert type(entry.path) is PurePosixPath
            assert not entry.path.is_absolute()
            assert "żółw-dataset" not in entry.path.as_posix()

        hashes_by_name = {entry.path.name: entry.hash for entry in manifest.entries}
        assert str(hashes_by_name["a_base.png"]) == "55aa55aa55aa55aa"
        assert str(hashes_by_name["b_variant.png"]) == "95aa55aa55aa55aa"
        assert str(hashes_by_name["c_variant_more.png"]) == "956a55aa55aa55aa"
        assert str(hashes_by_name["z_unrelated.png"]) == "0f0f0f0f0f0f0f0f"

        # The manifest file lives next to the dataset root, not inside it.
        manifest_path = root / "image-hashes.json"
        manifest.save(manifest_path)
        assert manifest_path.read_bytes() == manifest.to_json().encode("utf-8")

        # Move the dataset root -- the manifest's relative identifiers don't reference it at all,
        # so nothing about the manifest needs to change.
        moved_root = root / "przeniesiony-dataset"
        dataset_root.rename(moved_root)
        assert not dataset_root.exists()
        assert moved_root.exists()
        for entry in manifest.entries:
            moved_path = moved_root.joinpath(*entry.path.parts)
            assert moved_path.is_file(), moved_path

        restored = im.PerceptualHashManifest.load(manifest_path)
        assert restored == manifest
        assert restored.to_json() == manifest.to_json()

        threshold = 2
        pairs = im.find_similar_image_pairs(restored.to_hashes(), max_distance=threshold)
        found = tuple(
            (pair.first.as_posix(), pair.second.as_posix(), pair.distance) for pair in pairs
        )
        expected_pairs = (
            ("obrazy/a_base.png", "obrazy/b_variant.png", 2),
            ("obrazy/b_variant.png", "obrazy/c_variant_more.png", 2),
        )
        assert found == expected_pairs, found
        for pair in pairs:
            assert "z_unrelated.png" not in (pair.first.name, pair.second.name)

        print("builder: build_perceptual_hash_manifest")
        print(f"algorithm: {manifest.algorithm.value}")
        print(f"hash size: {manifest.hash_size} (64 bits)")
        print(f"manifest entries: {len(manifest.entries)}")
        print("dataset root stored: no")
        print("manifest paths relative: yes")
        print("unicode dataset path: yes")
        print("manifest round-trip identical: yes")
        print("dataset moved after save: yes")
        print(f"threshold: {threshold}")
        print(f"similar pairs: {len(pairs)}")
        for pair in pairs:
            print(f"{pair.distance}  {pair.first.as_posix()} <-> {pair.second.as_posix()}")


if __name__ == "__main__":
    main()
