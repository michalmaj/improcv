"""Deterministic pairwise similarity search over precomputed perceptual hashes.

This module never reads, decodes, or hashes an image itself -- it only compares
already-computed `PerceptualHash` values (see `improcv.hashing`). The caller is
responsible for deciding which files to hash, hashing them, and building the
`hashes` mapping passed to `find_similar_image_pairs`.
"""

from __future__ import annotations

import itertools
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from improcv._validation import require_non_negative_int
from improcv.hashing import PerceptualHash

__all__ = [
    "SimilarImagePair",
    "find_similar_image_pairs",
]


@dataclass(frozen=True, slots=True)
class SimilarImagePair:
    """One unordered pair of image identifiers found within a similarity search's
    `max_distance` threshold.

    `distance` is the Hamming distance between the two images' `PerceptualHash`
    values, as returned by `PerceptualHash.distance`. A smaller `distance`
    usually means the two images are more visually similar, but it does not by
    itself prove that they are duplicates -- two visually different images can
    have a small distance (perceptual hash collisions are expected, not a
    defect; see `PerceptualHash`), and the right cutoff is application-specific.

    `first`/`second` are always in canonical order (`first.as_posix() <
    second.as_posix()`), so one unordered pair has exactly one representation:
    `(a, b)` and `(b, a)` never both appear, and equality/hashing stay
    unambiguous. This class does not swap `first`/`second` into canonical order
    for you -- constructing one by hand with the wrong order raises `ValueError`
    rather than silently reordering; `find_similar_image_pairs` itself always
    constructs results correctly.

    The threshold (`max_distance`) used to find this pair is deliberately not
    stored here -- it is a property of a specific search, not of the pair
    itself; the same two images are equally "distance 3 apart" regardless of
    what threshold happened to be used to find them.
    """

    first: Path
    second: Path
    distance: int

    def __post_init__(self) -> None:
        if not isinstance(self.first, Path):
            raise TypeError(f"first must be a pathlib.Path, got {type(self.first).__name__}")
        if not isinstance(self.second, Path):
            raise TypeError(f"second must be a pathlib.Path, got {type(self.second).__name__}")
        if self.first.as_posix() == self.second.as_posix():
            raise ValueError(
                f"first and second must be different paths, got {self.first.as_posix()!r} for both"
            )
        require_non_negative_int(self.distance, "distance")
        if not (self.first.as_posix() < self.second.as_posix()):
            raise ValueError(
                "first and second must be in canonical order (first.as_posix() < "
                f"second.as_posix()), got first={self.first.as_posix()!r}, "
                f"second={self.second.as_posix()!r}"
            )


def _normalize_path_identifier(value: object, *, name: str) -> Path:
    """Raise TypeError/ValueError unless `value` is a valid path identifier; return as `Path`.

    Accepts a `str` or an `os.PathLike[str]` -- never a bare `bytes`, and
    never an `os.PathLike` whose `__fspath__()` returns `bytes`. Normalizes
    through `os.fspath()` only, never `str()`: the latter would silently
    accept arbitrary objects (e.g. turning an `int` key into `"3"`) instead
    of rejecting them. The result is never resolved, made absolute, or
    otherwise touched beyond `Path()` construction -- this is an identifier
    for an image, not a filesystem request.
    """
    try:
        raw_path = os.fspath(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(
            f"{name} must be a str or os.PathLike[str], got {type(value).__name__}"
        ) from exc
    if not isinstance(raw_path, str):
        raise TypeError(
            f"{name} must resolve to a str, got {type(raw_path).__name__} from os.fspath() "
            "-- a PathLike whose __fspath__() returns bytes is not accepted"
        )
    if raw_path == "":
        raise ValueError(f"{name} must not be an empty string")
    if "\x00" in raw_path:
        raise ValueError(f"{name} must not contain a NUL character")
    return Path(raw_path)


def _normalize_hash_mapping(hashes: object) -> list[tuple[str, Path, PerceptualHash]]:
    """Raise TypeError/ValueError unless `hashes` is a valid hash mapping; return normalized
    entries.

    Each returned entry is `(path.as_posix(), path, hash)`, in `hashes`'
    own iteration order. Two distinct keys that normalize to the same
    `Path.as_posix()` string (including via `Path`'s own collapsing of
    redundant separators or `.` components) raise `ValueError` naming the
    colliding normalized path -- never silently picked between or
    overwritten.
    """
    if not isinstance(hashes, Mapping):
        raise TypeError(f"hashes must be a Mapping, got {type(hashes).__name__}")

    seen: set[str] = set()
    entries: list[tuple[str, Path, PerceptualHash]] = []
    for key, value in hashes.items():
        path = _normalize_path_identifier(key, name="hashes key")
        posix = path.as_posix()
        if posix in seen:
            raise ValueError(
                f"hashes contains two different keys that normalize to the same path {posix!r}"
            )
        seen.add(posix)
        if not isinstance(value, PerceptualHash):
            raise TypeError(
                f"hashes[{posix!r}] must be a PerceptualHash, got {type(value).__name__}"
            )
        entries.append((posix, path, value))
    return entries


def _require_single_hash_space(entries: list[tuple[str, Path, PerceptualHash]]) -> None:
    """Raise ValueError unless every entry shares the same algorithm and hash_size.

    Checked explicitly, once, before any pairwise comparison -- not left to
    incidentally surface from the first `PerceptualHash.distance` call that
    happens to compare two incompatible entries, which could otherwise
    return a partial result before hitting the mismatch.
    """
    spaces = {(entry[2].algorithm, entry[2].hash_size) for entry in entries}
    if len(spaces) > 1:
        ordered = sorted(spaces, key=lambda space: (space[0].value, space[1]))
        described = "\n".join(
            f"{algorithm.value}(hash_size={hash_size})" for algorithm, hash_size in ordered
        )
        raise ValueError(
            "hashes must all share the same algorithm and hash_size, found multiple "
            f"comparison spaces:\n{described}"
        )


_PathIdentifierT = TypeVar("_PathIdentifierT", bound=str | os.PathLike[str])
"""Bound to `str | os.PathLike[str]` so a concrete mapping type (`dict[str, ...]`,
`dict[Path, ...]`, `dict[PurePosixPath, ...]`, a custom `PathLike[str]`, ...) is accepted
directly by `find_similar_image_pairs` -- `Mapping`'s key type is otherwise invariant, so a
`Mapping[str | os.PathLike[str], PerceptualHash]` parameter would reject every concrete mapping
type except that exact union. Not exported; purely a static-typing device with no runtime effect.
"""


def find_similar_image_pairs(
    hashes: Mapping[_PathIdentifierT, PerceptualHash],
    *,
    max_distance: int,
) -> tuple[SimilarImagePair, ...]:
    """Find every unordered pair of images whose perceptual hashes are within `max_distance`.

    This function returns matching pairs, not duplicate groups or clusters.
    Threshold similarity is not transitive: if `distance(A, B) <= max_distance`
    and `distance(B, C) <= max_distance`, it does not follow that
    `distance(A, C) <= max_distance`. `A`/`B` and `B`/`C` are both reported as
    independent pairs; no transitive closure, connected-component grouping, or
    deduplication across pairs is performed.

    This function never reads, decodes, or hashes an image itself -- `hashes`
    must already contain one precomputed `PerceptualHash` per image (see
    `improcv.hashing.average_hash`/`phash`). It performs no filesystem access
    at all: an identifier need not refer to an existing file.

    Complexity
    ----------
    Time: ``O(n**2)`` hash comparisons, where ``n = len(hashes)`` -- every
    unordered pair is compared exactly once via `PerceptualHash.distance`.
    Normalized input storage: ``O(n)``. Result storage: ``O(m)``, where ``m``
    is the number of returned pairs, which may be ``n * (n - 1) / 2`` in the
    worst case (every pair matches). This is a simple first slice, not a
    solution intended to scale to very large datasets -- bucketing, BK-trees,
    approximate nearest-neighbor search, and parallelism are all deliberately
    out of scope here.

    Parameters
    ----------
    hashes : Mapping[str | os.PathLike[str], PerceptualHash]
        Maps an image identifier (a path-like value; never read from disk,
        never checked for existence) to that image's already-computed
        `PerceptualHash`. Every value must share the same `algorithm` and
        `hash_size` -- comparing hashes from different algorithms or
        `hash_size`s is not meaningful (`PerceptualHash.distance` itself
        rejects it). Two different keys that normalize to the same path
        (e.g. via redundant separators or a leading `.`) are rejected, not
        silently merged. Not modified.
    max_distance : int
        The inclusive maximum Hamming distance for a pair to be reported: a
        pair with ``distance == max_distance`` is included, one with
        ``distance == max_distance + 1`` is not. Keyword-only, with no
        default -- an appropriate threshold depends on the algorithm and the
        application, and guessing one silently would be worse than requiring
        it explicitly. Must be a plain `int`, non-negative, and (for a
        non-empty `hashes`) at most ``hash_size ** 2`` -- the maximum
        Hamming distance two hashes of that size can have; a larger value is
        rejected rather than silently clamped, since it would otherwise
        misleadingly suggest a stricter search than ``hash_size ** 2``
        (return-everything) actually is.

    Returns
    -------
    tuple[SimilarImagePair, ...]
        Deterministically ordered by ``(distance, first.as_posix(),
        second.as_posix())``, ascending -- independent of `hashes`' own
        iteration order, independent of `hashes`' concrete `Mapping` type,
        and independent of the order pairs happen to be discovered while
        comparing them. Empty when `hashes` has fewer than two entries, or
        when no pair is within `max_distance`. A pair is never reported
        against itself, and two different identifiers with an identical hash
        (``distance == 0``) are a legal pair, never deduplicated away by
        their shared hash value.

    Raises
    ------
    TypeError
        If `hashes` is not a `Mapping` (a list/tuple of pairs, a generator,
        or any other non-`Mapping` object is rejected, not silently
        iterated); if any key is not a `str`/`os.PathLike[str]` (including a
        bare `bytes` or a `PathLike` whose `__fspath__()` returns `bytes`);
        if any value is not a `PerceptualHash`; or if `max_distance` is not
        a plain `int` (a `bool` or a NumPy integer scalar is rejected, not
        silently accepted).
    ValueError
        If any key normalizes to an empty string or contains a NUL
        character; if two different keys normalize to the same path; if
        `hashes` has two or more entries whose `PerceptualHash` values do
        not all share the same `algorithm` and `hash_size` (naming every
        distinct comparison space found, sorted for a deterministic
        message); if `max_distance` is negative; or if `max_distance`
        exceeds ``hash_size ** 2`` for a non-empty `hashes`.
    """
    require_non_negative_int(max_distance, "max_distance")
    entries = _normalize_hash_mapping(hashes)

    if entries:
        _require_single_hash_space(entries)
        hash_size = entries[0][2].hash_size
        bit_count = hash_size**2
        if max_distance > bit_count:
            raise ValueError(
                f"max_distance must be at most hash_size**2={bit_count} for hash_size="
                f"{hash_size}, got {max_distance}"
            )

    if len(entries) < 2:
        return ()

    sorted_entries = sorted(entries, key=lambda entry: entry[0])

    pairs: list[SimilarImagePair] = []
    for (_, path_a, hash_a), (_, path_b, hash_b) in itertools.combinations(sorted_entries, 2):
        distance = hash_a.distance(hash_b)
        if distance <= max_distance:
            pairs.append(SimilarImagePair(first=path_a, second=path_b, distance=distance))

    pairs.sort(key=lambda pair: (pair.distance, pair.first.as_posix(), pair.second.as_posix()))
    return tuple(pairs)
