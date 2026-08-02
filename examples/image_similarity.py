"""Discover, hash, and pair up similar images -- a complete, copyable recipe.

Builds four tiny synthetic 8x8 grayscale images in a temporary directory, discovers them with
`improcv.discover_images`, decodes and hashes each one exactly once with `improcv.average_hash`,
then searches for similar pairs with `improcv.find_similar_image_pairs` at two different
thresholds using that same, already-computed hash mapping -- no redecoding, no rehashing.

Why synthetic, pixel-exact images: every pixel is either 0 or 255 and every image has exactly 32
of each, so the resulting hash bits (and the Hamming distances between them) can be verified by
hand, not just by running the script. Three of the four images (A, B, C) are built as small,
deliberate variations of one another; the fourth (D) is unrelated.

Why `average_hash` and not `phash` here: an 8x8 checkerboard makes `average_hash`'s bit decisions
easy to verify by hand (its default threshold is the image's own pixel mean), which is exactly
what this recipe needs to demonstrate the pairing behavior with real, checkable numbers. This is
not a claim that `average_hash` is a better algorithm than `phash` in general -- see this
project's main README for a `phash`-based version of the same workflow on realistic photos.
Whichever algorithm you pick, its hashes are only comparable to other hashes from that same
algorithm and `hash_size` (`PerceptualHash.distance` itself enforces this), and you must re-choose
an appropriate `max_distance` threshold for it -- the threshold used below is specific to this
tiny synthetic example, not a general recommendation.

The interesting result is that similarity here is not transitive: A is within the threshold of B,
and B is within the threshold of C, but A is not within the threshold of C.
`find_similar_image_pairs` reports this as two independent pairs, never as one merged group of
three.

Run with:

    uv run python examples/image_similarity.py
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import cv2
import numpy as np

import improcv as im


def build_similarity_images() -> dict[str, np.ndarray]:
    """Build the four deterministic 8x8 uint8 grayscale images used by this recipe.

    Each image has exactly 32 white (255) and 32 black (0) pixels.

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
        dataset_dir = write_dataset(Path(tmp), images)

        paths = im.discover_images(dataset_dir)
        assert len(paths) == 4, f"expected 4 discovered files, got {len(paths)}"
        assert [path.name for path in paths] == [
            "a_base.png",
            "b_variant.png",
            "c_variant_more.png",
            "z_unrelated.png",
        ], [path.name for path in paths]
        a_path, b_path, c_path, d_path = paths

        hash_size = 8
        hashes: dict[str | os.PathLike[str], im.PerceptualHash] = {}
        for path in paths:
            decoded = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if decoded is None:
                raise RuntimeError(f"could not decode {path}")
            # cv2.imread's stubs type the result as the loose MatLike; IMREAD_GRAYSCALE always
            # produces a uint8 array in practice.
            hashes[path] = im.average_hash(cast(im.ImageU8, decoded), hash_size=hash_size)

        assert str(hashes[a_path]) == "55aa55aa55aa55aa"
        assert str(hashes[b_path]) == "95aa55aa55aa55aa"
        assert str(hashes[c_path]) == "956a55aa55aa55aa"
        assert str(hashes[d_path]) == "0f0f0f0f0f0f0f0f"

        distance_ab = hashes[a_path].distance(hashes[b_path])
        distance_bc = hashes[b_path].distance(hashes[c_path])
        distance_ac = hashes[a_path].distance(hashes[c_path])
        assert distance_ab == 2, distance_ab
        assert distance_bc == 2, distance_bc
        assert distance_ac == 4, distance_ac
        for other_path in (a_path, b_path, c_path):
            distance_to_unrelated = hashes[d_path].distance(hashes[other_path])
            assert distance_to_unrelated > 2, distance_to_unrelated

        hashes_before_search = dict(hashes)

        threshold = 2
        pairs = im.find_similar_image_pairs(hashes, max_distance=threshold)
        found = tuple((pair.first.name, pair.second.name, pair.distance) for pair in pairs)
        expected = (
            ("a_base.png", "b_variant.png", 2),
            ("b_variant.png", "c_variant_more.png", 2),
        )
        assert found == expected, found

        # Re-searching the same, already-computed hash mapping at a wider threshold requires no
        # redecoding and no rehashing -- only find_similar_image_pairs is called again.
        wider_pairs = im.find_similar_image_pairs(hashes, max_distance=4)
        wider_found = tuple(
            (pair.first.name, pair.second.name, pair.distance) for pair in wider_pairs
        )
        expected_wider = (*expected, ("a_base.png", "c_variant_more.png", 4))
        assert wider_found == expected_wider, wider_found

        assert hashes == hashes_before_search, "the hash mapping must not change across a re-search"

        print(f"images discovered: {len(paths)}")
        print("algorithm: average_hash")
        print(f"hash size: {hash_size} (64 bits)")
        print(f"threshold: {threshold}")
        print(f"similar pairs: {len(pairs)}")
        for pair in pairs:
            print(f"{pair.distance}  {pair.first.name} <-> {pair.second.name}")
        print(f"non-transitive gap: {a_path.name} <-> {c_path.name} = {distance_ac}")
        print("rehash required for threshold 4: no")


if __name__ == "__main__":
    main()
