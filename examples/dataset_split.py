"""Split a dataset into train/validation/test -- a deterministic, in-memory partition.

Like `manifest_comparison.py`, this script creates no files and needs no `cv2` -- `split_dataset`
operates entirely on an already-materialized `Sequence[T]` and an explicit `np.random.Generator`,
with no filesystem access anywhere in its contract.

Ten small, hand-listed `Path` values stand in for a discovered dataset (e.g. the output of
`improcv.discover_images`, which returns exactly this shape: `tuple[Path, ...]`). They are split
`train=0.7, validation=0.15` (`test` is always the exact remainder, `0.15`), using an explicit,
seeded `np.random.default_rng` -- the same generator convention `improcv.sample_flip`/
`sample_affine` already use, consumed (not cloned) by `split_dataset` exactly as those functions
consume it.

Run with:

    uv run python examples/dataset_split.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import improcv as im

_SEED = 20260807


def main() -> None:
    paths = tuple(Path(f"cats/{index:02d}.png") for index in range(10))

    rng = np.random.default_rng(_SEED)
    result = im.split_dataset(paths, train=0.7, validation=0.15, rng=rng)

    # Every input path appears in exactly one split; none is omitted or duplicated.
    combined = sorted(result.train) + sorted(result.validation) + sorted(result.test)
    assert sorted(combined) == sorted(paths)
    assert len(result.train) + len(result.validation) + len(result.test) == len(paths)

    # Split *sizes* are a pure function of len(paths)/train/validation -- reproducible even
    # without knowing rng's state (see docs/design/0.4.0a2-dataset-split.md section 7).
    assert (len(result.train), len(result.validation), len(result.test)) == (7, 1, 2)

    print("split: split_dataset")
    print("input: 10 synthetic Path values (no filesystem access)")
    print("ratios: train=0.7 validation=0.15 test=0.15 (exact remainder)")
    print(f"train: {len(result.train)}")
    for path in result.train:
        print(f"T  {path.as_posix()}")
    print(f"validation: {len(result.validation)}")
    for path in result.validation:
        print(f"V  {path.as_posix()}")
    print(f"test: {len(result.test)}")
    for path in result.test:
        print(f"E  {path.as_posix()}")
    print("occurrence overlap: none")
    print("semantic/subject/class leakage guarantee: none")

    # split_dataset treats each Sequence element as one atomic sample -- an ImageMaskPair from
    # discover_image_mask_pairs(...) composes directly, with no special-casing and no risk of its
    # image/mask fields landing in different splits, e.g.:
    #
    #     pairs = im.discover_image_mask_pairs(image_root, mask_root)
    #     pair_split = im.split_dataset(pairs, train=0.7, validation=0.15, rng=rng)


if __name__ == "__main__":
    main()
