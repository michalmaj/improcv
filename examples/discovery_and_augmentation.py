"""Dataset discovery + replayable affine augmentation, end to end.

Builds a tiny, synthetic ``dataset/images/train`` + ``dataset/masks/train``
directory pair in a temporary directory, discovers and pairs the two images
with their masks via `improcv.discover_image_mask_pairs`, then samples one
replayable affine transform and applies it identically to an image and its
segmentation mask via `improcv.sample_affine`/`improcv.apply_affine`,
including replay and optional canvas expansion.

Run with:

    uv run python examples/discovery_and_augmentation.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np

import improcv as im


def _write_dataset(root: Path) -> None:
    images_dir = root / "dataset" / "images" / "train"
    masks_dir = root / "dataset" / "masks" / "train"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)

    for name, seed in (("sample_a", 1), ("sample_b", 2)):
        rng = np.random.default_rng(seed)
        image = rng.integers(0, 256, size=(40, 60, 3), dtype=np.uint8)
        # Three explicit integer labels (0, 1, 2), arranged as simple bands
        # so the augmentation step below has real structure to preserve.
        mask = np.zeros((40, 60), dtype=np.uint8)
        mask[:, 20:40] = 1
        mask[:, 40:] = 2

        cv2.imwrite(str(images_dir / f"{name}.png"), image)
        cv2.imwrite(str(masks_dir / f"{name}.png"), mask)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_dataset(root)

        pairs = im.discover_image_mask_pairs(
            root / "dataset" / "images",
            root / "dataset" / "masks",
        )

        assert len(pairs) == 2, f"expected 2 pairs, got {len(pairs)}"
        # discover_image_mask_pairs returns a globally-sorted tuple, so the
        # order below is deterministic, not incidental.
        assert [pair.image.stem for pair in pairs] == ["sample_a", "sample_b"]
        for pair in pairs:
            assert pair.image.parent.name == "train"
            assert pair.mask.parent.name == "train"
            assert pair.image.stem == pair.mask.stem

        first_pair = pairs[0]
        image = cv2.imread(str(first_pair.image), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(first_pair.mask), cv2.IMREAD_UNCHANGED)

        assert image is not None and mask is not None
        assert image.dtype == np.uint8 and mask.dtype == np.uint8
        assert image.shape[:2] == mask.shape[:2]

        source_size = (image.shape[1], image.shape[0])  # (width, height)
        rng = np.random.default_rng(42)
        affine_params = im.sample_affine(
            rng,
            source_size=source_size,
            angle_range=(-15.0, 15.0),
            translation_x_range=(-5.0, 5.0),
            translation_y_range=(-5.0, 5.0),
        )

        # A border value outside the mask's real labels (0, 1, 2) so it can
        # never be confused with genuine content revealed by the transform.
        mask_border_value = 255
        first = im.apply_affine(
            image, affine_params, mask=mask, mask_border_value=mask_border_value
        )
        replay = im.apply_affine(
            image, affine_params, mask=mask, mask_border_value=mask_border_value
        )

        replay_identical = bool(
            np.array_equal(first.image, replay.image) and np.array_equal(first.mask, replay.mask)
        )
        assert replay_identical, "replaying the same AffineParameters must be bit-for-bit identical"

        mask_labels = set(np.unique(first.mask).tolist())
        assert mask_labels <= {0, 1, 2, mask_border_value}, (
            f"unexpected mask values introduced by the warp: {mask_labels}"
        )

        expanded = im.expand_affine_canvas(affine_params)
        assert expanded.output_size is not None
        expanded_pair = im.apply_affine(
            image, expanded, mask=mask, mask_border_value=mask_border_value
        )
        expected_shape = (expanded.output_size[1], expanded.output_size[0])
        assert expanded_pair.image.shape[:2] == expected_shape
        assert expanded_pair.mask.shape[:2] == expanded_pair.image.shape[:2]
        assert expanded.output_size[0] >= source_size[0]
        assert expanded.output_size[1] >= source_size[1]

        print(f"pairs discovered: {len(pairs)}")
        print(f"replay identical: {'yes' if replay_identical else 'no'}")
        print(f"source shape: {image.shape}")
        print(f"expanded shape: {expanded_pair.image.shape}")
        print(f"mask labels: {sorted(mask_labels)}")


if __name__ == "__main__":
    main()
