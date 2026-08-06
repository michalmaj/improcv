"""Compare two PerceptualHashManifest snapshots -- a pure, in-memory, deterministic diff.

Unlike every other script in this directory, this one creates no files, uses no `cv2`, no NumPy,
and no temporary directory -- `compare_perceptual_hash_manifests` operates entirely on two
already-constructed `PerceptualHashManifest` values, and that absence of any filesystem or
decoding step is itself the point being demonstrated here.

Two small, hand-constructed manifests represent the same logical dataset ("cats/") at two points
in time:

- ``cats/a.png`` is unchanged.
- ``cats/b.png``'s hash changed (its content changed).
- ``cats/old_name.png`` was renamed to ``cats/new_name.png`` -- reported as one `removed` entry
  and one `added` entry, never specially detected as a rename, even though the hash under both
  names is identical.
- ``cats/c.png`` is a new file.

Run with:

    uv run python examples/manifest_comparison.py
"""

from __future__ import annotations

import improcv as im

_ALGORITHM = im.PerceptualHashAlgorithm.AVERAGE_HASH
_HASH_SIZE = 8


def _hash(hex_value: str) -> im.PerceptualHash:
    return im.PerceptualHash.from_hex(hex_value, algorithm=_ALGORITHM, hash_size=_HASH_SIZE)


def main() -> None:
    before = im.PerceptualHashManifest.from_hashes(
        {
            "cats/a.png": _hash("55aa55aa55aa55aa"),
            "cats/b.png": _hash("0f0f0f0f0f0f0f0f"),
            "cats/old_name.png": _hash("3333333333333333"),
        },
        algorithm=_ALGORITHM,
        hash_size=_HASH_SIZE,
    )
    after = im.PerceptualHashManifest.from_hashes(
        {
            "cats/a.png": _hash("55aa55aa55aa55aa"),
            "cats/b.png": _hash("8f0f0f0f0f0f0f0f"),
            "cats/c.png": _hash("aaaaaaaaaaaaaaaa"),
            "cats/new_name.png": _hash("3333333333333333"),
        },
        algorithm=_ALGORITHM,
        hash_size=_HASH_SIZE,
    )

    result = im.compare_perceptual_hash_manifests(before, after)

    assert tuple(entry.path.as_posix() for entry in result.added) == (
        "cats/c.png",
        "cats/new_name.png",
    )
    assert tuple(entry.path.as_posix() for entry in result.removed) == ("cats/old_name.png",)
    assert tuple(change.path.as_posix() for change in result.changed) == ("cats/b.png",)
    assert tuple(path.as_posix() for path in result.unchanged) == ("cats/a.png",)

    # The rename is never detected: the hash under the old and new name is identical, but it
    # still shows up once in `removed` and once in `added`, never merged into one record.
    renamed_hash = next(
        entry.hash for entry in result.removed if entry.path.as_posix() == "cats/old_name.png"
    )
    same_hash_reported_separately = any(
        entry.hash == renamed_hash
        for entry in result.added
        if entry.path.as_posix() == "cats/new_name.png"
    )
    assert same_hash_reported_separately

    print("comparison: compare_perceptual_hash_manifests")
    print("identity: canonical manifest path")
    print("filesystem access: no")
    print(f"added: {len(result.added)}")
    for entry in result.added:
        print(f"A  {entry.path.as_posix()}  {entry.hash}")
    print(f"removed: {len(result.removed)}")
    for entry in result.removed:
        print(f"R  {entry.path.as_posix()}  {entry.hash}")
    print(f"changed: {len(result.changed)}")
    for change in result.changed:
        print(f"C  {change.path.as_posix()}  {change.before} -> {change.after}")
    print(f"unchanged: {len(result.unchanged)}")
    for path in result.unchanged:
        print(f"U  {path.as_posix()}")
    print("rename detection: no")
    print("same hash old/new reported separately: yes")


if __name__ == "__main__":
    main()
