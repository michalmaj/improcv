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

`split_dataset`/`DatasetSplit`, below, are an entirely separate, independent utility that happens
to share this module because both are dataset-workflow orchestration (see
`docs/design/0.4.0a2-dataset-split.md`): `split_dataset` never touches the filesystem, never
decodes an image, and has no relationship to `PerceptualHashManifest` beyond both being usable on
`discover_images`'/`discover_image_mask_pairs`' output.
"""

from __future__ import annotations

import math
import os
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar, assert_never, cast

import cv2
import numpy as np

from improcv._validation import require_range
from improcv.discovery import discover_images
from improcv.hashing import PerceptualHash, PerceptualHashAlgorithm, average_hash, phash
from improcv.manifest import PerceptualHashManifest
from improcv.types import ImageU8

__all__ = [
    "DatasetSplit",
    "build_perceptual_hash_manifest",
    "split_dataset",
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


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class DatasetSplit(Generic[T]):
    """The result of `split_dataset`: every input occurrence classified as `train`, `validation`,
    or `test`.

    `train`/`validation`/`test` are each a `tuple[T, ...]` in permutation order (see
    `split_dataset`'s own docstring) -- never re-sorted to the source `items`' original order and
    never sorted by value. Together, they contain every element of the `items` passed to
    `split_dataset`, each exactly once, counted by input position (occurrence), not by value: two
    equal values at two different input positions may land in two different fields here, and that
    is correct, not a defect (see `split_dataset`).

    Frozen and slotted, with default `dataclass` equality/`repr`/immutability -- no custom
    `__post_init__`, since there is no cross-field invariant to enforce at construction time (a
    manually constructed `DatasetSplit` is a plain, statically-typed value object, exactly like
    `improcv.manifest.PerceptualHashManifestDiff`).
    """

    train: tuple[T, ...]
    validation: tuple[T, ...]
    test: tuple[T, ...]


def _require_sequence(items: object) -> None:
    """Raise TypeError unless `items` is a `Sequence` and not one of the forms that would be
    silently misinterpreted as one.

    Mirrors `improcv.discovery._normalize_extensions`'s own validation shape: `str`/`bytes`/
    `bytearray` genuinely satisfy `isinstance(x, Sequence)` (a string is a sequence of its own
    characters) and get a dedicated message rather than being silently accepted as a "dataset" of
    single characters; `Mapping` fails the generic `Sequence` check anyway but gets its own
    dedicated message explaining that iterating a `Mapping` walks its keys only; `numpy.ndarray` is
    not registered as a `Sequence` either, but gets a dedicated message (rather than falling
    through to the generic one) because a NumPy-heavy caller is exactly the audience most likely to
    try it and be confused by an unexplained rejection -- see
    `docs/design/0.4.0a2-dataset-split.md` §3. This performs no iteration over `items` -- only
    `isinstance` checks.
    """
    if isinstance(items, (str, bytes, bytearray)):
        raise TypeError(
            "items must be a Sequence, not a single str/bytes/bytearray, got "
            f"{type(items).__name__}"
        )
    if isinstance(items, Mapping):
        raise TypeError(f"items must be a Sequence, not a Mapping, got {type(items).__name__}")
    if isinstance(items, np.ndarray):
        raise TypeError(
            "items must be a Sequence, not a numpy.ndarray -- convert via tuple(items) or "
            "list(items) first if you intend to split its elements as separate samples"
        )
    if not isinstance(items, Sequence):
        raise TypeError(f"items must be a Sequence, got {type(items).__name__}")


def _largest_remainder_counts(n: int, train: float, validation: float) -> tuple[int, int, int]:
    """Split `n` into `(n_train, n_validation, n_test)` via the Largest Remainder Method.

    `train`/`validation` must already be plain Python `float` -- normalized by `split_dataset`
    itself before calling this helper (see that function's own docstring). This is a private
    implementation detail, not a second public contract: the normalization guarantee callers
    actually rely on is the one `split_dataset` documents, not this helper's own parameter types.

    `test`'s ratio is `1.0 - train - validation`, computed exactly as `split_dataset`'s own
    docstring documents (by subtraction, not as an independent input). Each of the three ideal
    (real-valued) counts is floored; the `0`/`1`/`2` leftover occurrences implied by those floors
    summing to less than `n` are distributed one at a time to the split(s) with the largest
    fractional part, breaking an *exact* tie (bit-for-bit equal fractional parts) by the fixed
    priority `train` -> `validation` -> `test`. See `docs/design/0.4.0a2-dataset-split.md` §7-§8
    for the full rationale and worked examples, including two cases where floating-point
    subtraction makes an apparent tie resolve in a way that does not match a naive rational-
    arithmetic expectation -- this is correct, documented behavior, not a bug.
    """
    test = 1.0 - train - validation
    ideal_counts = {"train": n * train, "validation": n * validation, "test": n * test}
    floors = {key: math.floor(value) for key, value in ideal_counts.items()}
    remainder = n - sum(floors.values())
    fractions = {key: ideal_counts[key] - floors[key] for key in ideal_counts}
    priority = ("train", "validation", "test")
    order = sorted(priority, key=lambda key: (-fractions[key], priority.index(key)))

    counts = dict(floors)
    for key in order[:remainder]:
        counts[key] += 1
    return counts["train"], counts["validation"], counts["test"]


def split_dataset(
    items: Sequence[T],
    *,
    train: float,
    validation: float = 0.0,
    rng: np.random.Generator,
) -> DatasetSplit[T]:
    """Deterministically partition `items` into train/validation/test by input occurrence.

    Every input *position* (`0 <= i < len(items)`) is assigned to exactly one of `train`/
    `validation`/`test`; no position is omitted or assigned twice. This guarantee is about
    occurrences, not values: `split_dataset(["a", "a", "b"], ...)` treats the two `"a"` entries as
    two independent occurrences that may land in different splits -- `items` need not be hashable
    and its elements' `__eq__` (if any) is never consulted. Internally, this is implemented by
    permuting the integer indices `0 .. len(items) - 1` (`rng.permutation(len(items))`) and reading
    each element back via `items[i]` exactly once; no `set`/`dict` keyed by `items`' values is ever
    built.

    `items` must be a `Sequence[T]` -- `list[T]`, `tuple[T, ...]` (including the direct output of
    `discover_images`/`discover_image_mask_pairs`), or any other `Sequence` implementation. A bare
    `str`/`bytes`/`bytearray` is rejected (it would otherwise be silently split as a sequence of
    its own characters/bytes); a `Mapping` (e.g. `dict`) is rejected (iterating one walks its keys
    only, not samples); a `numpy.ndarray` is rejected (it is ambiguous whether it should be treated
    as an array of opaque samples or as stacked pixel data to split along axis 0 -- resolving that
    is out of scope for this function; convert via `tuple(items)`/`list(items)` first); a bare
    generator/iterator is rejected (it has no length and this function never mutates or exhausts
    its input); a single `pathlib.Path` is rejected (it is one sample, not a dataset -- pass
    `[path]` or a real collection of paths).

    Because each element of `items` is treated as one atomic, indivisible unit, a
    `discover_image_mask_pairs(...)` result (`tuple[ImageMaskPair, ...]`) composes directly and
    correctly with no special-casing anywhere in this function: an `ImageMaskPair`'s `image`/`mask`
    fields are never separated across splits, since `split_dataset` never looks inside its own `T`.

    `train`/`validation` are ratios in `[0.0, 1.0]` (a plain `int`, `float`, or NumPy real scalar;
    `bool` is rejected even though it is technically an `int` subclass, matching every other
    probability-shaped parameter in this codebase). Once validated, both are converted to plain
    Python `float` (binary64) -- `float(train)`/`float(validation)` -- and every composite ratio
    computation from that point on (the `train + validation` sum check, `test`'s ratio, the three
    ideal Largest Remainder counts, and their fractional parts) uses only those converted values,
    never the caller's original NumPy scalar dtype. This prevents split allocation from depending on
    NumPy scalar-promotion rules (e.g. `np.float16(0.999) + np.float16(0.001)` rounds to exactly
    `1.0` in `float16` arithmetic, while the actual binary64 values those two `float16`s represent
    sum to slightly more than `1.0`) while still honoring the scalar's own actually-represented
    numeric value -- `np.float16(0.8)` is treated as `float(np.float16(0.8))`
    (`0.7998046875...`), not as the literal Python `0.8`.

    `test`'s ratio is always the exact remainder, `1.0 - train - validation` (using the converted
    `float` values) -- there is no `test` parameter, and `train + validation` must not exceed `1.0`
    (checked by an *exact* comparison of the converted `float` sum, with no `math.isclose` tolerance
    anywhere in this function). Splits sizes are computed via the Largest Remainder Method: each of
    the three ideal, real-valued counts (`len(items) * ratio`) is floored, and the `0`/`1`/`2`
    leftover occurrences those floors omit are distributed to the split(s) with the largest
    fractional part, breaking an exact tie by the fixed priority `train` -> `validation` -> `test`.
    This guarantees `len(result.train) + len(result.validation) + len(result.test) ==
    len(items)` exactly, for every valid `(train, validation)` and every `len(items)`, unlike
    independently flooring or rounding each split (which can under- or over-allocate). Split
    *sizes* are a pure, deterministic function of `len(items)`/`train`/`validation` alone,
    independent of `rng` -- only *which* items land in which split depends on `rng`. Two documented
    cases, `train=0.7, validation=0.15` and `train=validation=1/3`, resolve an apparent tie in
    favor of `test` rather than `validation`/`train`, because `test`'s ratio is computed by
    subtraction and its floating-point value is not bit-identical to a naively expected exact
    fraction -- this is deterministic, reproducible, documented behavior (see
    `docs/design/0.4.0a2-dataset-split.md` §7), not a bug to "fix" by rounding differently.

    Within each of `train`/`validation`/`test`, elements appear in **permutation order** -- the
    order they occur in the shuffled index sequence -- never re-sorted back to `items`' own
    original order and never sorted by value.

    `rng` must be an actual `numpy.random.Generator` instance (`isinstance`-checked, not
    duck-typed; a legacy `numpy.random.RandomState`, a bare integer seed, the `numpy.random`
    module itself, or `None` are all rejected). This function **uses the given `rng` directly and
    never clones it** -- it calls `rng.permutation(len(items))`, and `rng` is left in whatever
    state that call leaves it, exactly like `improcv.augmentation.sample_flip`/`sample_affine`
    leave their own `rng` argument. For a non-trivial permutation, `rng`'s state normally advances
    -- but this is *not* a guarantee that two consecutive calls with the same `rng` must return
    different `DatasetSplit` values: an empty or single-element `items` may require no draw at
    all, and even two genuinely different `rng` states are not a mathematical guarantee of two
    different sampled permutations. Two independently constructed `Generator` instances in
    equivalent initial states (e.g. two separate `np.random.default_rng(0)` calls, each not yet
    consumed by anything else) passed to two separate calls with the same `items`/`train`/
    `validation` produce identical results -- that is the guarantee this function makes; its
    converse (different `rng` states imply different results) is not promised. No global or
    ambient RNG (`np.random.seed`, the free functions on the `numpy.random` module, Python's own
    `random` module) is read or written anywhere in this function.

    The determinism guarantee is: the same `items` (same positions, same values) plus a `rng` in
    the same internal state, on the currently supported NumPy/Python implementation stack, produce
    the same `DatasetSplit`. The exact number and order of internal draws against `rng`, and the
    specific permutation algorithm used, are implementation details, not part of the public
    contract, and may change between releases without notice -- mirroring `sample_flip`'s own
    documented policy. This is not a promise that today's exact split is frozen forever across
    every future NumPy/Python version.

    This function guarantees only occurrence-level non-overlap (every input position assigned to
    exactly one split). It does **not** guarantee: that the same real-world subject appears in only
    one split; that semantically or visually duplicate content is kept out of more than one split;
    class balance; stratification; or any grouped/subject-aware assignment. `items` carries no
    label, group ID, or subject ID concept to this function -- it is an opaque sample sequence, so
    no leakage guarantee beyond occurrence-level non-overlap can honestly be made.

    This function operates entirely in memory: it never accesses the filesystem, never decodes an
    image, and never mutates `items`.

    Complexity
    ----------
    Time: `O(n)`, where `n = len(items)` -- one `rng.permutation(n)` call plus one linear pass to
    build the three result tuples. Additional memory (beyond `items` and the result): `O(n)` --
    `rng.permutation(n)` allocates a new integer array of length `n`; no `O(1)` step exists in this
    function.

    Parameters
    ----------
    items : Sequence[T]
        The dataset to split, as a `Sequence`. Not mutated. See above for accepted/rejected forms.
    train : float
        The train split's ratio, in `[0.0, 1.0]`. No default -- there is no universally correct
        value to guess.
    validation : float, optional
        The validation split's ratio, in `[0.0, 1.0]`. Default `0.0` (no validation split).
    rng : numpy.random.Generator
        The source of randomness for the permutation. No default; consumed directly (see above).

    Returns
    -------
    DatasetSplit[T]
        `train`/`validation`/`test`, each a `tuple[T, ...]` in permutation order. Their lengths sum
        to exactly `len(items)`. `items = ()` returns `DatasetSplit(train=(), validation=(),
        test=())`, not an error, matching `discover_images`/`find_similar_image_pairs`/
        `build_perceptual_hash_manifest`'s own existing behavior for an empty collection.

    Raises
    ------
    TypeError
        If `items` is not a `Sequence` (including a bare `str`/`bytes`/`bytearray`, a `Mapping`, a
        `numpy.ndarray`, a generator/iterator, or a single `pathlib.Path` -- see above); if
        `train`/`validation` is not a real number, or is a `bool`; or if `rng` is not a
        `numpy.random.Generator`.
    ValueError
        If `train`/`validation` is `NaN`, `+inf`, `-inf`, or outside `[0.0, 1.0]`; or if
        `train + validation` exceeds `1.0` (exact comparison, no tolerance).
    """
    _require_sequence(items)
    require_range(train, 0.0, 1.0, "train")
    require_range(validation, 0.0, 1.0, "validation")

    # Normalize to plain Python (binary64) float *after* validating the caller's original value,
    # and use only these two values for every composite ratio computation from here on -- the sum
    # check, the Largest Remainder allocation, and its error message. This is what keeps allocation
    # from depending on the caller's NumPy scalar dtype/promotion rules (e.g. np.float16(0.999) +
    # np.float16(0.001) rounds to exactly 1.0 in float16 arithmetic, even though the binary64 values
    # those two float16s actually represent sum to slightly more than 1.0) while still honoring the
    # scalar's own actually-represented value, not a re-interpretation of it.
    train_value = float(train)
    validation_value = float(validation)
    ratio_sum = train_value + validation_value
    if ratio_sum > 1.0:
        raise ValueError(
            "train + validation must be at most 1.0, got "
            f"train={train_value!r}, validation={validation_value!r} (sum={ratio_sum!r})"
        )
    if not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be a numpy.random.Generator, got {type(rng).__name__}")

    n = len(items)
    train_count, validation_count, _test_count = _largest_remainder_counts(
        n, train_value, validation_value
    )

    indices = rng.permutation(n)
    train_indices = indices[:train_count]
    validation_indices = indices[train_count : train_count + validation_count]
    test_indices = indices[train_count + validation_count :]

    return DatasetSplit(
        train=tuple(items[int(i)] for i in train_indices),
        validation=tuple(items[int(i)] for i in validation_indices),
        test=tuple(items[int(i)] for i in test_indices),
    )
