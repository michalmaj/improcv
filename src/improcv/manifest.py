"""A deterministic, portable JSON manifest for precomputed perceptual hashes.

A `PerceptualHashManifest` is a snapshot of ``relative image identifier -> precomputed
PerceptualHash`` -- nothing more. It never reads, decodes, or hashes an image itself, never
touches the filesystem, never stores an absolute dataset root, and never records file size,
modification time, or content digest. It does not check whether an identifier still refers to
an existing file, and it does not know or care whether a hash is still "fresh" relative to the
image it was computed from -- that judgment belongs entirely to the caller. This is a manifest,
not a cache: there is no invalidation logic here, and none is planned for this module. File
save/load convenience (reading/writing an actual file on disk) is deliberately out of scope for
this slice and is planned as a separate follow-up.

Every stored identifier is a canonical, relative, POSIX-style `pathlib.PurePosixPath` -- never a
platform-specific `Path`, and never an absolute path. This is what makes a manifest portable: two
manifests built on different machines (Windows, Linux, macOS) for the same relative dataset
layout serialize to byte-identical JSON. See `PerceptualHashManifestEntry`/
`PerceptualHashManifest` for the exact path contract.

A manifest has exactly one hash space (one `algorithm`, one `hash_size`) for all of its entries --
the same constraint `improcv.similarity.find_similar_image_pairs` itself already enforces on any
mapping passed to it. `PerceptualHashManifest.to_hashes()` returns a mapping directly usable as
that function's `hashes` argument.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Self

from improcv._validation import require_int
from improcv.hashing import PerceptualHash, PerceptualHashAlgorithm

__all__ = [
    "PerceptualHashManifest",
    "PerceptualHashManifestEntry",
]

# The only schema version this slice understands. A manifest JSON document with any other
# `schema_version` is rejected outright -- there is no migration or best-effort compatibility
# logic here, deliberately: silently reinterpreting an unknown future schema would risk silently
# misreading data it was never designed to understand.
_SCHEMA_VERSION = 1

# Mirrors PerceptualHash's own [2, 256] range (see improcv.hashing) -- duplicated here, not
# imported from that module's private helper, so this module never reaches into hashing.py's
# internals; a manifest's hash_size must be valid for the exact same reason a PerceptualHash's
# own hash_size must be, since every entry's hash is itself a PerceptualHash.
_HASH_SIZE_MIN = 2
_HASH_SIZE_MAX = 256

_TOP_LEVEL_FIELDS = ("schema_version", "algorithm", "hash_size", "entries")
_ENTRY_FIELDS = ("path", "hash")


def _require_valid_manifest_hash_size(value: object) -> None:
    """Raise TypeError unless `value` is a plain int, then ValueError unless it's within
    [2, 256] -- the same range `PerceptualHash`/`average_hash`/`phash` themselves enforce."""
    require_int(value, "hash_size")
    assert isinstance(value, int)  # narrows for the type checker; require_int already enforced this
    if not (_HASH_SIZE_MIN <= value <= _HASH_SIZE_MAX):
        raise ValueError(
            f"hash_size must be between {_HASH_SIZE_MIN} and {_HASH_SIZE_MAX}, got {value}"
        )


def _validate_canonical_posix_identifier(text: str, *, name: str) -> PurePosixPath:
    """Raise ValueError unless `text` is already a canonical, relative POSIX path identifier.

    Canonical means: non-empty, no NUL, no backslash, not absolute (no leading ``/``), no
    Windows drive letter (``C:...``), no empty/``.``/``..`` path segment, and exactly equal to
    its own `PurePosixPath.as_posix()` round-trip (so no redundant or trailing separators either).
    This is the single source of truth for "is this a legal manifest path identifier" -- used both
    for identifiers built from a local `Path` (`_local_path_to_manifest_identifier`) and for
    identifiers read directly out of JSON text (`PerceptualHashManifest.from_json`), so a manifest
    round-tripped through JSON can never accept a path that direct construction would reject.
    """
    if text == "":
        raise ValueError(f"{name} must not be an empty string")
    if "\x00" in text:
        raise ValueError(f"{name} must not contain a NUL character")
    if "\\" in text:
        raise ValueError(f"{name} must not contain a backslash, got {text!r}")
    if text.startswith("/"):
        raise ValueError(f"{name} must be a relative path, got {text!r}")
    if len(text) >= 2 and text[1] == ":" and text[0].isalpha():
        raise ValueError(f"{name} must not contain a Windows drive, got {text!r}")

    segments = text.split("/")
    if any(segment in ("", ".", "..") for segment in segments):
        raise ValueError(
            f"{name} must not contain an empty, '.', or '..' path segment, got {text!r}"
        )

    candidate = PurePosixPath(text)
    if candidate.as_posix() != text:
        raise ValueError(f"{name} must be in canonical POSIX form, got {text!r}")
    return candidate


def _local_path_to_manifest_identifier(value: object, *, name: str) -> PurePosixPath:
    """Raise TypeError/ValueError unless `value` is a valid local path identifier; return the
    canonical `PurePosixPath` manifest identifier for it.

    Accepts a `str` or an `os.PathLike[str]` -- never a bare `bytes`, and never a `PathLike`
    whose `__fspath__()` returns `bytes`. The result is interpreted through the *local*
    platform's own `pathlib.Path` semantics (so a Windows caller's native backslash-separated
    relative path is accepted and normalized to forward slashes via `.as_posix()`, exactly as a
    real `Path` object from that caller's own filesystem would be) before being validated as a
    canonical manifest identifier. Never resolved, made absolute, or checked against the
    filesystem -- this is an identifier for an image, not a filesystem request.

    A side effect of this local-`Path` interpretation: a lone ``.`` path segment (e.g.
    ``"./image.png"`` or ``"a/./b.png"``) is silently collapsed away by `pathlib` itself before
    this function's own validation ever runs (`Path("./image.png").as_posix() ==
    "image.png"`) -- there is no way to preserve it while still routing through local `Path`
    semantics, and a leading/embedded ``./`` conventionally names the same location either way.
    A ``..`` segment is not collapsed by `pathlib` and is always rejected explicitly. This
    differs from `PerceptualHashManifest.from_json`, which validates raw JSON text directly and
    therefore does reject a literal ``"./image.png"`` string verbatim -- see that method.
    """
    try:
        raw = os.fspath(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(
            f"{name} must be a str or os.PathLike[str], got {type(value).__name__}"
        ) from exc
    if not isinstance(raw, str):
        raise TypeError(
            f"{name} must resolve to a str, got {type(raw).__name__} from os.fspath() -- "
            "a PathLike whose __fspath__() returns bytes is not accepted"
        )
    if raw == "":
        raise ValueError(f"{name} must not be an empty string")

    local_path = Path(raw)
    if local_path.is_absolute():
        raise ValueError(f"{name} must be a relative path, got {raw!r}")

    return _validate_canonical_posix_identifier(local_path.as_posix(), name=name)


def _json_kind(value: object) -> str:
    """A short, human-readable name for a JSON-decoded Python value's kind, for error messages.

    Never includes the value itself, only its kind -- callers may separately include a small,
    specific value (e.g. a field's own text) in their own message, but this helper never risks
    leaking an entire malformed document.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """`object_pairs_hook` for `json.loads` that raises ValueError on a duplicate object key.

    Applied uniformly by `json.loads` to every JSON object in the document, at every nesting
    level -- the top-level manifest object and every individual entry object alike -- so a
    duplicate key anywhere is rejected, never silently resolved to its last-seen value the way
    the stdlib `json` decoder does by default.
    """
    seen: set[str] = set()
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(f"duplicate JSON object key {key!r}")
        seen.add(key)
        result[key] = value
    return result


def _require_exact_keys(
    document: dict[str, Any], allowed: tuple[str, ...], *, context: str
) -> None:
    """Raise ValueError unless `document`'s keys are exactly `allowed` -- no missing, no extra."""
    allowed_set = set(allowed)
    present = set(document)
    missing = allowed_set - present
    if missing:
        raise ValueError(f"{context} is missing required field(s): {sorted(missing)!r}")
    extra = present - allowed_set
    if extra:
        raise ValueError(f"{context} has unexpected field(s): {sorted(extra)!r}")


@dataclass(frozen=True, slots=True)
class PerceptualHashManifestEntry:
    """One ``(path, hash)`` record inside a `PerceptualHashManifest`.

    `path` must be exactly a `pathlib.PurePosixPath` (never a platform `Path`, and never a
    subclass) holding a canonical, relative, POSIX-style identifier -- see
    `PerceptualHashManifest` for the full path policy. `hash` must be a `PerceptualHash`; this
    entry does not itself check that `hash`'s `algorithm`/`hash_size` match any particular
    manifest -- that cross-check belongs to `PerceptualHashManifest.__post_init__`, since a bare
    entry has no manifest to compare against yet.

    This class never sorts or otherwise changes its own fields -- constructing one with an
    illegal path or a non-`PerceptualHash` value raises immediately, rather than silently
    normalizing anything. Note that `pathlib.PurePosixPath` itself collapses a lone ``.``
    segment at construction time (`PurePosixPath("./a.png") == PurePosixPath("a.png")`) --
    `path` is validated as whatever `PurePosixPath` object is actually passed in, so this
    collapsing has already happened before `__post_init__` ever runs; a ``..`` segment is not
    collapsed by `pathlib` and is always rejected here.
    """

    path: PurePosixPath
    hash: PerceptualHash

    def __post_init__(self) -> None:
        if type(self.path) is not PurePosixPath:
            raise TypeError(
                f"path must be exactly a pathlib.PurePosixPath, got {type(self.path).__name__}"
            )
        _validate_canonical_posix_identifier(self.path.as_posix(), name="path")
        if not isinstance(self.hash, PerceptualHash):
            raise TypeError(f"hash must be a PerceptualHash, got {type(self.hash).__name__}")


@dataclass(frozen=True, slots=True)
class PerceptualHashManifest:
    """A deterministic, portable snapshot of precomputed `PerceptualHash` values.

    `algorithm`/`hash_size` are the manifest's own, single, explicit hash space -- every entry's
    `hash` must share exactly that `algorithm` and `hash_size` (the same one-hash-space
    constraint `improcv.similarity.find_similar_image_pairs` already enforces on any mapping
    passed to it). Requiring `algorithm`/`hash_size` explicitly, even for an empty manifest,
    means there is never an "unknown hash space" state to special-case: a manifest with zero
    entries still has a well-defined hash space, just no data in it yet.

    `entries` must already be a tuple of `PerceptualHashManifestEntry`, sorted ascending by
    `entry.path.as_posix()`, with no duplicate paths -- `__post_init__` checks this but never
    sorts or deduplicates for you; `from_hashes`/`from_json` always build a canonically sorted
    tuple themselves.

    This manifest is a snapshot, not a cache: it never checks whether a path still refers to an
    existing file, never stores an absolute dataset root, and never records file size,
    modification time, or content digest. The caller is entirely responsible for judging whether
    a stored hash is still valid for the current content of the image it names. There is no
    grouping/clustering here, and no pair-search result is ever stored in a manifest -- see
    `to_hashes` for handing a manifest's hashes to
    `improcv.similarity.find_similar_image_pairs`.
    """

    algorithm: PerceptualHashAlgorithm
    hash_size: int
    entries: tuple[PerceptualHashManifestEntry, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.algorithm, PerceptualHashAlgorithm):
            raise TypeError(
                f"algorithm must be a PerceptualHashAlgorithm, got {type(self.algorithm).__name__}"
            )
        _require_valid_manifest_hash_size(self.hash_size)
        if not isinstance(self.entries, tuple):
            raise TypeError(f"entries must be a tuple, got {type(self.entries).__name__}")

        for index, entry in enumerate(self.entries):
            if not isinstance(entry, PerceptualHashManifestEntry):
                raise TypeError(
                    f"entries[{index}] must be a PerceptualHashManifestEntry, got "
                    f"{type(entry).__name__}"
                )
            if entry.hash.algorithm != self.algorithm or entry.hash.hash_size != self.hash_size:
                raise ValueError(
                    f"entries[{index}]'s hash must be {self.algorithm.value}"
                    f"(hash_size={self.hash_size}), got {entry.hash.algorithm.value}"
                    f"(hash_size={entry.hash.hash_size})"
                )

        posix_paths = [entry.path.as_posix() for entry in self.entries]
        if len(set(posix_paths)) != len(posix_paths):
            seen: set[str] = set()
            duplicates: list[str] = []
            for posix in posix_paths:
                if posix in seen and posix not in duplicates:
                    duplicates.append(posix)
                seen.add(posix)
            raise ValueError(
                f"entries must have unique paths, found duplicate(s): {tuple(duplicates)!r}"
            )
        if posix_paths != sorted(posix_paths):
            raise ValueError(
                "entries must already be sorted ascending by path.as_posix() -- "
                "PerceptualHashManifest does not sort them for you"
            )

    @classmethod
    def from_hashes(
        cls,
        hashes: Mapping[str | os.PathLike[str], PerceptualHash],
        *,
        algorithm: PerceptualHashAlgorithm,
        hash_size: int,
    ) -> Self:
        """Build a `PerceptualHashManifest` from an already-computed hash mapping.

        `algorithm`/`hash_size` are required explicitly -- they are not inferred from `hashes`,
        so an empty mapping can still produce a well-defined (empty) manifest, and every value in
        `hashes` must match them exactly. Each key is interpreted through the local platform's
        own path semantics and converted to a canonical, relative, POSIX identifier (see
        `_local_path_to_manifest_identifier`); two different keys that normalize to the same
        identifier raise `ValueError`, exactly as
        `improcv.similarity.find_similar_image_pairs` itself rejects colliding keys. `hashes`
        and the `PerceptualHash` objects within it are never modified. The result's `entries` are
        sorted canonically by path, independent of `hashes`' own iteration order. Performs no
        filesystem access.

        Raises
        ------
        TypeError
            If `algorithm` is not a `PerceptualHashAlgorithm`, `hash_size` is not a plain int,
            `hashes` is not a `Mapping`, any key is not a `str`/`os.PathLike[str]` (including a
            bare `bytes` or a `PathLike` whose `__fspath__()` returns `bytes`), or any value is
            not a `PerceptualHash`.
        ValueError
            If `hash_size` is out of range; if any key normalizes to an illegal identifier (see
            `PerceptualHashManifest`'s path policy) or an empty string; if two different keys
            normalize to the same identifier; or if any value's `algorithm`/`hash_size` does not
            match the ones given here.
        """
        if not isinstance(algorithm, PerceptualHashAlgorithm):
            raise TypeError(
                f"algorithm must be a PerceptualHashAlgorithm, got {type(algorithm).__name__}"
            )
        _require_valid_manifest_hash_size(hash_size)
        if not isinstance(hashes, Mapping):
            raise TypeError(f"hashes must be a Mapping, got {type(hashes).__name__}")

        seen_paths: set[str] = set()
        entries: list[PerceptualHashManifestEntry] = []
        for key, value in hashes.items():
            path = _local_path_to_manifest_identifier(key, name="hashes key")
            posix = path.as_posix()
            if posix in seen_paths:
                raise ValueError(
                    f"hashes contains two different keys that normalize to the same path {posix!r}"
                )
            seen_paths.add(posix)

            if not isinstance(value, PerceptualHash):
                raise TypeError(
                    f"hashes[{posix!r}] must be a PerceptualHash, got {type(value).__name__}"
                )
            if value.algorithm != algorithm or value.hash_size != hash_size:
                raise ValueError(
                    f"hashes[{posix!r}] must be {algorithm.value}(hash_size={hash_size}), got "
                    f"{value.algorithm.value}(hash_size={value.hash_size})"
                )
            entries.append(PerceptualHashManifestEntry(path=path, hash=value))

        sorted_entries = tuple(sorted(entries, key=lambda entry: entry.path.as_posix()))
        return cls(algorithm=algorithm, hash_size=hash_size, entries=sorted_entries)

    def to_hashes(self) -> dict[PurePosixPath, PerceptualHash]:
        """Return a new ``dict[PurePosixPath, PerceptualHash]`` built from this manifest's entries.

        Insertion order matches `entries`' own canonical (sorted-by-path) order. This is a fresh
        dict, never a view over this manifest's internal storage -- mutating it never affects
        this `PerceptualHashManifest`. The result is directly usable as the `hashes` argument to
        `improcv.similarity.find_similar_image_pairs` (a `PurePosixPath` is a valid
        `os.PathLike[str]`).
        """
        return {entry.path: entry.hash for entry in self.entries}

    def to_json(self) -> str:
        """Serialize this manifest to deterministic, canonical schema-v1 JSON text.

        Top-level fields are always written in the fixed order ``schema_version``,
        ``algorithm``, ``hash_size``, ``entries``; each entry as ``path`` then ``hash``. Uses one
        fixed `json.dumps` configuration (``ensure_ascii=False, indent=2``) and appends exactly
        one trailing newline. Contains no timestamp, no environment information, no absolute
        path, and no library version -- two manifests with the same `algorithm`/`hash_size`/
        entries always serialize to byte-identical text, regardless of the machine, working
        directory, or `improcv` version that produced them, and regardless of the iteration order
        of whatever mapping `from_hashes` originally built them from (entries are already stored
        in canonical sorted order -- see `PerceptualHashManifest`).
        """
        document = {
            "schema_version": _SCHEMA_VERSION,
            "algorithm": self.algorithm.value,
            "hash_size": self.hash_size,
            "entries": [
                {"path": entry.path.as_posix(), "hash": str(entry.hash)} for entry in self.entries
            ],
        }
        return json.dumps(document, ensure_ascii=False, indent=2) + "\n"

    @classmethod
    def from_json(cls, text: str) -> Self:
        """Parse a `PerceptualHashManifest` from the exact JSON text produced by `to_json`.

        `text` must be a plain `str` -- a `bytes` value or an `os.PathLike` is rejected outright
        (this parses JSON text already in hand; it never reads a file). Every JSON object in the
        document (the top-level object and every entry object alike) is checked for duplicate
        keys; the top-level object must have exactly the fields ``schema_version``, ``algorithm``,
        ``hash_size``, ``entries`` (no more, no fewer), and each entry object exactly ``path``,
        ``hash``. `schema_version` must be the plain int `1` (a `bool`, `float`, `str`, or any
        other int value is rejected) -- there is no migration path for an unrecognized version.
        `algorithm` must name a real `PerceptualHashAlgorithm` member; `hash_size` must pass the
        same range check `PerceptualHash` itself enforces. Every entry's `hash` is parsed only
        through the public `PerceptualHash.from_hex(text, algorithm=algorithm,
        hash_size=hash_size)` (never through `PerceptualHash`'s private `_value`), using the
        manifest's own `algorithm`/`hash_size` for every entry -- so a mixed-hash-space document
        can never arise from this path. An entry's hex text must equal `str()` of the
        `PerceptualHash` it parses to exactly (lowercase, zero-padded, canonical width) -- an
        otherwise-valid but non-canonical spelling (e.g. uppercase) is rejected, not silently
        accepted. Every entry's `path` is validated as a canonical manifest identifier directly
        from its JSON text -- never passed through a platform `Path` first (unlike `from_hashes`,
        which does; see `PerceptualHashManifest`'s path policy). Duplicate paths across entries
        raise `ValueError` (via the same check `__post_init__` performs on any manifest); the
        returned manifest's `entries` are always in canonical sorted order, regardless of the
        entries' order in the JSON document.

        Every failure raises `ValueError` naming the specific field or entry index at fault (or
        the underlying `json.JSONDecodeError`'s own message for malformed JSON) -- never the
        full document. The sole exception is `text` itself not being a `str`, which raises
        `TypeError`.

        Raises
        ------
        TypeError
            If `text` is not a `str`.
        ValueError
            If `text` is not valid JSON; if the top-level value is not a JSON object; if any
            object in the document (top-level or an entry) has a duplicate key, a missing
            field, or an unexpected field; if `schema_version` is not the plain int `1`; if
            `algorithm` is not a string naming a real `PerceptualHashAlgorithm`; if `hash_size`
            is not a plain int in `PerceptualHash`'s valid range; if `entries` is not a list, or
            any element is not an object; if an entry's `path` is not a string or is not a legal,
            canonical manifest identifier; if an entry's `hash` is not a string, is not valid hex
            for `algorithm`/`hash_size`, or is not in canonical (lowercase, zero-padded) form;
            or if two entries share the same path.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be a str, got {type(text).__name__}")

        try:
            document = json.loads(text, object_pairs_hook=_reject_duplicate_json_keys)
        except json.JSONDecodeError as exc:
            raise ValueError(f"manifest text is not valid JSON: {exc}") from exc

        if not isinstance(document, dict):
            raise ValueError(
                f"manifest JSON must be a top-level object, got {_json_kind(document)}"
            )
        _require_exact_keys(document, _TOP_LEVEL_FIELDS, context="manifest")

        schema_version = document["schema_version"]
        if isinstance(schema_version, bool) or not isinstance(schema_version, int):
            raise ValueError(
                f"manifest schema_version must be a plain int, got {_json_kind(schema_version)}"
            )
        if schema_version != _SCHEMA_VERSION:
            raise ValueError(
                f"unsupported manifest schema_version {schema_version!r}, expected "
                f"{_SCHEMA_VERSION}"
            )

        algorithm_text = document["algorithm"]
        if not isinstance(algorithm_text, str):
            raise ValueError(
                f"manifest algorithm must be a string, got {_json_kind(algorithm_text)}"
            )
        try:
            algorithm = PerceptualHashAlgorithm(algorithm_text)
        except ValueError as exc:
            allowed = ", ".join(repr(member.value) for member in PerceptualHashAlgorithm)
            raise ValueError(
                f"manifest algorithm must be one of {allowed}, got {algorithm_text!r}"
            ) from exc

        hash_size = document["hash_size"]
        if isinstance(hash_size, bool) or not isinstance(hash_size, int):
            raise ValueError(f"manifest hash_size must be a plain int, got {_json_kind(hash_size)}")
        _require_valid_manifest_hash_size(hash_size)

        entries_raw = document["entries"]
        if not isinstance(entries_raw, list):
            raise ValueError(f"manifest entries must be a list, got {_json_kind(entries_raw)}")

        entries: list[PerceptualHashManifestEntry] = []
        for index, entry_raw in enumerate(entries_raw):
            if not isinstance(entry_raw, dict):
                raise ValueError(f"entries[{index}] must be an object, got {_json_kind(entry_raw)}")
            _require_exact_keys(entry_raw, _ENTRY_FIELDS, context=f"entries[{index}]")

            path_text = entry_raw["path"]
            if not isinstance(path_text, str):
                raise ValueError(
                    f"entries[{index}].path must be a string, got {_json_kind(path_text)}"
                )
            path = _validate_canonical_posix_identifier(path_text, name=f"entries[{index}].path")

            hash_text = entry_raw["hash"]
            if not isinstance(hash_text, str):
                raise ValueError(
                    f"entries[{index}].hash must be a string, got {_json_kind(hash_text)}"
                )
            try:
                hash_value = PerceptualHash.from_hex(
                    hash_text, algorithm=algorithm, hash_size=hash_size
                )
            except ValueError as exc:
                raise ValueError(f"entries[{index}].hash is invalid: {exc}") from exc
            if str(hash_value) != hash_text:
                raise ValueError(
                    f"entries[{index}].hash must be canonical lowercase hex "
                    f"({str(hash_value)!r}), got {hash_text!r}"
                )

            entries.append(PerceptualHashManifestEntry(path=path, hash=hash_value))

        sorted_entries = tuple(sorted(entries, key=lambda entry: entry.path.as_posix()))
        return cls(algorithm=algorithm, hash_size=hash_size, entries=sorted_entries)
