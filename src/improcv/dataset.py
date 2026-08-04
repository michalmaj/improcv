"""Sequential dataset-to-manifest workflow: discover, decode, hash, and build a manifest.

`build_perceptual_hash_manifest` is a thin orchestration layer over three already-independent
building blocks -- `improcv.discovery.discover_images`, `improcv.hashing.average_hash`/`phash`,
and `improcv.manifest.PerceptualHashManifest.from_hashes` -- wired together exactly the way a
caller would otherwise wire them by hand (see `improcv.manifest`'s own docstring, and
`examples/image_similarity_manifest.py`, for that manual equivalent). It adds no new semantics of
its own: every path, ordering, and validation rule is inherited unchanged from the module it
delegates to.

This module is deliberately the only one in `improcv` that imports across all three of
`discovery`, `hashing`, and `manifest` (plus `cv2` for decoding) -- none of those lower modules
import this one or each other for this purpose, so there is no import cycle. `discovery.py`,
`hashing.py`, and `manifest.py` remain exactly as decoupled from the filesystem/each other as
before this module existed.

Reading and decoding a discovered file are two deliberately separate steps, not one `cv2.imread`
call: each file's bytes are read through Python's own filesystem path handling
(`pathlib.Path.read_bytes`), then decoded from an in-memory buffer via `cv2.imdecode`. Python's
path handling still goes through the operating system's own filesystem APIs -- it does not avoid
them -- but on Windows it is Unicode-capable, whereas OpenCV's filename-based `cv2.imread`/
`cv2.imwrite` are not reliable for paths containing characters outside the active code page.
`cv2.imdecode` sidesteps that specific problem by never receiving a filename in the first place:
by the time it runs, the file has already been read, so it only ever sees the resulting in-memory
encoded byte buffer. This is not a new, general-purpose decoder: the decode policy is still
exactly fixed 8-bit grayscale, nothing about `average_hash`/`phash`'s input contract changes, and
a file whose bytes cannot be decoded as an image still raises the same `ValueError` as before;
only *how* the file's bytes reach OpenCV has changed. A native error reading the file itself
(missing file, permission denied, or another filesystem error) is a different failure mode from
an undecodable image and is never turned into that `ValueError` -- see
`build_perceptual_hash_manifest`'s own docstring.

`build_perceptual_hash_manifest` only ever returns a `PerceptualHashManifest` -- it never writes a
file, never reads an existing manifest, never checks whether a previously computed hash is still
valid for its image's current content, and never runs any work in parallel. Saving the result is
a separate, explicit step the caller performs afterwards with `PerceptualHashManifest.save`.
"""

from __future__ import annotations

import os
from collections.abc import Collection
from pathlib import Path
from typing import assert_never, cast

import cv2
import numpy as np

from improcv.discovery import discover_images
from improcv.hashing import PerceptualHash, PerceptualHashAlgorithm, average_hash, phash
from improcv.manifest import PerceptualHashManifest
from improcv.types import ImageU8

__all__ = [
    "build_perceptual_hash_manifest",
]


def _decode_grayscale(path: Path) -> ImageU8 | None:
    """Read `path`'s bytes through Python's own filesystem path handling, then decode them as an
    8-bit grayscale image via `cv2.imdecode` -- never `cv2.imread`.

    `path.read_bytes()` still goes through the operating system's own filesystem APIs to open and
    read the file -- it does not avoid them -- but on Windows that handling is Unicode-capable,
    unlike `cv2.imread`, whose filename-based handling is not reliable for paths containing
    characters outside the active code page. `cv2.imdecode` never receives a filename at all: by
    the time it runs, `path`'s bytes are already an in-memory buffer, so whichever filename-related
    limitation OpenCV's own handling might have on a given platform is simply never exercised. An
    empty file is treated as undecodable (`None`) without ever calling `cv2.imdecode` on an empty
    buffer. Any `OSError` raised by `path.read_bytes()` itself (e.g. the file was deleted after
    discovery, or is unreadable) propagates unchanged -- this function does not distinguish a
    read failure from a decode failure; that distinction is made by its caller.
    """
    encoded_bytes = path.read_bytes()
    if not encoded_bytes:
        return None
    encoded = np.frombuffer(encoded_bytes, dtype=np.uint8)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if decoded is None:
        return None
    # cv2.imdecode's stubs type the result as the loose MatLike | None; IMREAD_GRAYSCALE always
    # produces a uint8 array when decoding succeeds.
    return cast(ImageU8, decoded)


def _compute_perceptual_hash(
    image: ImageU8,
    *,
    algorithm: PerceptualHashAlgorithm,
    hash_size: int,
) -> PerceptualHash:
    """Dispatch to `average_hash`/`phash` by `algorithm`.

    `algorithm` is assumed already validated by the caller (via
    `PerceptualHashManifest.from_hashes`) -- every `PerceptualHashAlgorithm` member is handled
    explicitly above, so the `assert_never` below documents an unreachable branch rather than a
    real runtime check.
    """
    if algorithm is PerceptualHashAlgorithm.AVERAGE_HASH:
        return average_hash(image, hash_size=hash_size)
    if algorithm is PerceptualHashAlgorithm.PHASH:
        return phash(image, hash_size=hash_size)
    assert_never(algorithm)


def build_perceptual_hash_manifest(
    root: str | os.PathLike[str],
    *,
    algorithm: PerceptualHashAlgorithm,
    hash_size: int = 8,
    recursive: bool = True,
    extensions: Collection[str] | None = None,
    include_hidden: bool = False,
) -> PerceptualHashManifest:
    """Discover, decode, and hash a local image dataset into a portable `PerceptualHashManifest`.

    Runs exactly one deterministic, sequential workflow: `discover_images(root, recursive=
    recursive, extensions=extensions, include_hidden=include_hidden)` finds candidate image
    files (its full contract -- ordering, symlink policy, hidden-file policy, extension
    matching, and root-related errors -- is inherited unchanged, not reimplemented here); each
    discovered file's bytes are read exactly once through Python's own, Unicode-capable filesystem
    path handling and decoded exactly once as 8-bit grayscale via `cv2.imdecode` (fixed, not a
    parameter, in this first slice: one comparable decode policy, a stable input shape, no alpha
    channel, and no unsupported `uint16`/floating-point depths to reason about) -- see
    `_decode_grayscale`, which never calls `cv2.imread`/`cv2.imwrite` and so never routes a
    filename through OpenCV's own filename-based handling, which is not reliable on Windows for
    paths outside the active code page; each decoded image is hashed exactly once with the
    algorithm named by `algorithm`; and the resulting
    `{relative identifier: PerceptualHash}` mapping is handed to
    `PerceptualHashManifest.from_hashes`, which performs the actual path-canonicalization,
    sorting, and hash-space validation -- this function never constructs a
    `PerceptualHashManifestEntry` or a `PurePosixPath` itself.

    Each discovered path is converted to its manifest identifier via
    ``path.relative_to(root)`` -- the dataset root itself is never stored in the manifest and
    never appears as a path segment in any identifier (a file at ``<root>/cats/a.png`` becomes
    the identifier ``cats/a.png``, not ``<root's name>/cats/a.png``). `root` itself is only ever
    turned into a local `Path` for this purpose -- never expanded (``~``), resolved, made
    absolute, case-folded, or Unicode-normalized.

    This function only ever returns a `PerceptualHashManifest` -- it never writes a file (saving
    is a separate, explicit ``manifest.save(...)`` call the caller makes afterwards), never reads
    an existing manifest, never checks a previously computed hash's freshness against its image's
    current content, and never runs any work in parallel (images are decoded and hashed one at a
    time, in `discover_images`' own canonical order).

    Parameters
    ----------
    root : str | os.PathLike[str]
        Passed directly to `discover_images` as its own `root` argument; see that function for
        the exact contract (may be relative or absolute; must reference an existing directory).
    algorithm : PerceptualHashAlgorithm
        Which algorithm (`average_hash`/`phash`) to run on every decoded image. Must be supplied
        explicitly; there is no default, since guessing one silently would be worse than
        requiring it.
    hash_size : int, optional
        Passed to the selected hashing function and to `PerceptualHashManifest.from_hashes`.
        Must be a plain `int` in `PerceptualHash`'s own valid range (`[2, 256]`); validated once,
        up front, by attempting to build an empty manifest with `algorithm`/`hash_size` -- the
        same check `from_hashes` itself performs, not a separate copy of it. Default `8`,
        matching `average_hash`/`phash`'s own default.
    recursive : bool, optional
        Passed directly to `discover_images`. Default `True`.
    extensions : Collection[str] | None, optional
        Passed directly to `discover_images`. Default `None` (that function's own default
        extension set).
    include_hidden : bool, optional
        Passed directly to `discover_images`. Default `False`.

    Returns
    -------
    PerceptualHashManifest
        `algorithm`/`hash_size` are exactly the values passed in. `entries` are in canonical
        sorted order (via `from_hashes`), with relative `PurePosixPath` identifiers and no
        dataset-root segment. An empty dataset (no discovered files) returns a well-defined,
        empty manifest with the given `algorithm`/`hash_size` -- not an error -- and never reads
        or decodes any file. Two calls against the same files with the same parameters return
        equal manifests (`manifest_a == manifest_b`) that serialize identically
        (``manifest_a.to_json() == manifest_b.to_json()``); this does not extend across different
        OpenCV versions/builds, whose exact decode or resize output is not guaranteed to be
        bit-identical.

    Raises
    ------
    TypeError
        If `algorithm` is not exactly a `PerceptualHashAlgorithm` (a string, a different enum, an
        `int`, a `bool`, or `None` are all rejected); if `hash_size` is not a plain `int`
        (`bool` included); or any `TypeError` `discover_images` itself raises for `root`/
        `recursive`/`extensions`/`include_hidden`.
    ValueError
        If `hash_size` is out of `PerceptualHash`'s valid range; any `ValueError`
        `discover_images` itself raises (e.g. an empty `root`, illegal `extensions`); or if any
        discovered file's bytes cannot be decoded as an image (`cv2.imdecode` returns `None`,
        including an empty file, which is never even handed to `cv2.imdecode`) -- raised
        immediately, for the first such file in `discover_images`' own canonical order, naming
        that file's relative manifest identifier and its local source path. No later file is
        read or decoded once this happens, and no partial manifest is ever returned. This is
        distinct from a native error *reading* a file's bytes (see below), which is never turned
        into this `ValueError`.
    FileNotFoundError, NotADirectoryError, OSError
        Propagated unchanged from `discover_images` for problems with `root` itself or its
        traversal; also propagated unchanged (never wrapped) if reading a discovered file's
        bytes itself fails -- e.g. the file was deleted after discovery, or is unreadable -- as
        opposed to being read successfully but failing to decode as an image, which raises
        `ValueError` instead (see above).
    """
    if not isinstance(algorithm, PerceptualHashAlgorithm):
        raise TypeError(
            f"algorithm must be a PerceptualHashAlgorithm, got {type(algorithm).__name__}"
        )

    # Reuses PerceptualHashManifest's own hash_size validation (via an otherwise-legal empty
    # manifest) instead of a second, private copy of the [2, 256]/plain-int rule -- and this
    # empty manifest is also exactly the correct return value if discovery below finds nothing.
    empty_manifest = PerceptualHashManifest.from_hashes(
        {}, algorithm=algorithm, hash_size=hash_size
    )

    paths = discover_images(
        root,
        recursive=recursive,
        extensions=extensions,
        include_hidden=include_hidden,
    )
    if not paths:
        return empty_manifest

    root_path = Path(os.fspath(root))
    hashes: dict[Path, PerceptualHash] = {}
    for path in paths:
        identifier = path.relative_to(root_path)
        decoded = _decode_grayscale(path)
        if decoded is None:
            raise ValueError(f"failed to decode image {identifier.as_posix()!r} from {str(path)!r}")
        hashes[identifier] = _compute_perceptual_hash(
            decoded, algorithm=algorithm, hash_size=hash_size
        )

    return PerceptualHashManifest.from_hashes(hashes, algorithm=algorithm, hash_size=hash_size)
