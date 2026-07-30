"""Deterministic, extension-based discovery of image files under a directory.

`discover_images` finds candidate image files by filename extension only --
it never opens, decodes, or otherwise inspects a file's content. This is a
read-only filesystem operation: it does not create, write, or touch
timestamps on anything, and it does not change the process's working
directory. `discover_image_mask_pairs` builds on it to pair up images and
masks discovered under two separate directories by a deterministic,
extension-stripped relative-path key -- neither function infers classes
from directory names, produces train/validation/test splits, or loads/
decodes any image; those remain out of scope for this slice.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

from improcv._validation import require_bool

__all__ = ["ImageMaskPair", "discover_image_mask_pairs", "discover_images"]

# Five common raster formats (JPEG and TIFF each have two conventional
# extensions) that make up the large majority of real-world image datasets.
# Deliberately not derived from what this OpenCV build happens to support
# (`cv2.haveImageReader` et al.) -- discovery is extension-based only and
# must give the same result regardless of which optional codecs the local
# OpenCV build was compiled with; see the module/function docstrings.
# Excluded on purpose: GIF (animated/palette format, rarely a single
# training image), JPEG2000/OpenEXR/Netpbm/HEIC/AVIF (all far less common in
# practice, and/or inconsistently supported) -- pass `extensions=` for any
# of these instead of expecting them by default.
_DEFAULT_IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
)

# Path separators (both, regardless of platform), NUL, and glob metacharacters
# -- none of these can legitimately appear in a bare filename extension.
_FORBIDDEN_EXTENSION_CHARS = frozenset("/\\\x00*?[]")


def discover_images(
    root: str | os.PathLike[str],
    *,
    recursive: bool = True,
    extensions: Collection[str] | None = None,
    include_hidden: bool = False,
) -> tuple[Path, ...]:
    """Find image files under `root` by filename extension, in a deterministic order.

    `root` must reference an existing directory -- a regular file, a FIFO/
    socket/device, or a path that doesn't exist at all all raise (see
    Raises below). `root` itself may be a symlink to a directory (its own
    validation uses `stat()`, which follows symlinks); a symlink or Windows
    reparse point (including a junction) *found while traversing* `root`'s
    contents is always skipped, along with anything under it -- there is no
    `follow_symlinks` option. Returned paths are anchored under `root`
    exactly as given (e.g. `discover_images("data")` returns paths like
    `Path("data/cat.jpg")`, not absolute ones) -- `root` is never resolved,
    made absolute, or otherwise normalized beyond `os.fspath()`.

    Discovery is extension-based only: a file's content is never opened,
    decoded, or otherwise inspected, so an empty, corrupted, or non-image
    file with a matching extension is still discovered. `extensions=None`
    (the default) uses a fixed list of seven common raster extensions
    (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, `.webp`) covering five
    widely-used formats; pass `extensions=` explicitly for anything else
    (e.g. `.gif`, `.jp2`, multi-part extensions like `.nii.gz`). Each
    element may omit its leading dot (`"jpg"` and `".jpg"` are equivalent)
    and is matched case-insensitively; matching is by `str.endswith` on the
    full lowercased filename, not by `Path.suffix`, so a multi-part
    extension like `.nii.gz` works as a single unit. A directory named like
    an image file (e.g. `photo.jpg/`), a non-regular entry (FIFO, socket,
    device), and a file whose name ends with an extra suffix (e.g.
    `photo.jpg.tmp`) are never returned.

    A descendant entry (file or directory) whose name starts with `.` is
    skipped by default, along with everything under a skipped directory;
    `include_hidden=True` disables this filter. This rule never applies to
    `root` itself -- an explicitly given `root` like `".dataset"` is always
    searched, regardless of `include_hidden`. `recursive=True` (the
    default) descends into every non-skipped subdirectory; `recursive=False`
    only considers `root`'s direct children.

    The result is a materialized, globally-sorted `tuple[Path, ...]` --
    sorted by each path's POSIX-style form relative to `root`
    (`path.relative_to(root).as_posix()`), case-sensitively, independent of
    the underlying filesystem's traversal order or the platform's path
    separator. Materializing costs `O(N)` memory (this is not a streaming
    indexer for an arbitrarily large tree) but is what makes global sorting,
    reporting an error before any result is used, and repeated iteration
    over the same result possible. An empty directory, or a directory with
    no matching files, returns `()`, not an error. No two returned paths
    are ever the same string, but two different paths may reference the
    same physical file via a hard link -- this function performs no
    identity-based deduplication.

    For every non-hidden descendant entry, discovery performs a fresh
    path-based stat with `follow_symlinks=False` before classifying it --
    never a cached result from directory enumeration itself (which some
    platforms, notably Windows, can otherwise return without a fresh system
    call). If an entry disappears between being listed and this fresh
    inspection, the native `FileNotFoundError` propagates (see Raises
    below), never silently skipped. An entry observed as a symlink or
    Windows reparse point at inspection time is skipped, per the policy
    above.

    This is a snapshot of one traversal, not an atomic view of the
    filesystem: this fresh-stat guarantee covers only the moment of
    inspection itself -- a file can still be created, modified, or deleted
    right after being classified (or after this function returns entirely),
    and nothing here detects that later change.

    Raises
    ------
    TypeError
        If `root` is not a `str` or `os.PathLike[str]` (including a
        `PathLike` whose `__fspath__()` returns `bytes`), if `recursive`/
        `include_hidden` is not exactly a `bool`, if `extensions` is not a
        `Collection[str]` (a bare `str`/`bytes`/`bytearray`, a `Mapping`, or
        a generator/iterator are all rejected, not silently misinterpreted
        as a collection of characters/keys/single-use values), or if any
        `extensions` element is not a `str`.
    ValueError
        If `root` is an empty string, if `extensions` is an empty
        collection, or if any `extensions` element is empty, is `"."`,
        contains whitespace, contains `/` or `\\`, contains a NUL byte, or
        contains a glob metacharacter (`*`, `?`, `[`, `]`).
    FileNotFoundError
        If `root` does not exist, or is a symlink whose target does not
        exist.
    NotADirectoryError
        If `root` exists but is not a directory (a regular file, FIFO,
        socket, device, or a symlink to any of those).
    OSError
        Any other native filesystem error (e.g. `PermissionError`) from
        checking `root` or from traversing its contents, propagated
        unmodified -- including one entry disappearing between being listed
        and being inspected, which is not silently skipped.
    """
    require_bool(recursive, "recursive")
    require_bool(include_hidden, "include_hidden")
    root_path = _normalize_root(root)
    normalized_extensions = _normalize_extensions(extensions)

    files: list[Path] = []
    pending: list[Path] = [root_path]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                if not include_hidden and entry.name.startswith("."):
                    continue

                entry_stat = os.stat(entry.path, follow_symlinks=False)
                mode = entry_stat.st_mode
                if stat.S_ISLNK(mode) or _is_reparse_point(entry_stat):
                    continue

                if stat.S_ISDIR(mode):
                    if recursive:
                        pending.append(Path(entry.path))
                elif stat.S_ISREG(mode) and entry.name.lower().endswith(normalized_extensions):
                    files.append(Path(entry.path))

    files.sort(key=lambda path: path.relative_to(root_path).as_posix())
    return tuple(files)


def _normalize_root(root: object, *, name: str = "root") -> Path:
    try:
        raw_root = os.fspath(root)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(
            f"{name} must be a str or os.PathLike[str], got {type(root).__name__}"
        ) from exc
    if not isinstance(raw_root, str):
        raise TypeError(
            f"{name} must resolve to a str, got {type(raw_root).__name__} from os.fspath() "
            "-- a PathLike whose __fspath__() returns bytes is not accepted"
        )
    if raw_root == "":
        raise ValueError(f"{name} must not be an empty string")

    root_path = Path(raw_root)
    root_stat = root_path.stat()
    if not stat.S_ISDIR(root_stat.st_mode):
        raise NotADirectoryError(f"{name} must reference a directory, got {raw_root!r}")
    return root_path


def _normalize_extensions(extensions: object, *, name: str = "extensions") -> tuple[str, ...]:
    if extensions is None:
        return _DEFAULT_IMAGE_EXTENSIONS

    if isinstance(extensions, (str, bytes, bytearray)):
        raise TypeError(
            f"{name} must be a Collection[str], not a single str/bytes/bytearray, "
            f"got {type(extensions).__name__}"
        )
    if isinstance(extensions, Mapping):
        raise TypeError(
            f"{name} must be a Collection[str], not a Mapping, got {type(extensions).__name__}"
        )
    if not isinstance(extensions, Collection):
        raise TypeError(f"{name} must be a Collection[str], got {type(extensions).__name__}")
    if len(extensions) == 0:
        raise ValueError(f"{name} must not be empty")

    normalized: list[str] = []
    seen: set[str] = set()
    for item in extensions:
        if not isinstance(item, str):
            raise TypeError(f"{name} elements must be str, got {type(item).__name__}")
        if any(ch.isspace() for ch in item):
            raise ValueError(f"{name} element must not contain whitespace, got {item!r}")
        if any(ch in _FORBIDDEN_EXTENSION_CHARS for ch in item):
            raise ValueError(
                f"{name} element must not contain a path separator, NUL, or glob "
                f"character, got {item!r}"
            )
        candidate = item[1:] if item.startswith(".") else item
        if candidate == "":
            raise ValueError(f"{name} element must not be empty, got {item!r}")

        normalized_item = "." + candidate.lower()
        if normalized_item not in seen:
            seen.add(normalized_item)
            normalized.append(normalized_item)

    return tuple(normalized)


def _is_reparse_point(entry_stat: os.stat_result) -> bool:
    # Windows-only in practice: a junction (and some other reparse points)
    # is not necessarily reported as a symlink by stat.S_ISLNK, but always
    # sets FILE_ATTRIBUTE_REPARSE_POINT. Both attributes are absent (default
    # to 0) on POSIX, so this is a no-op there.
    file_attributes = getattr(entry_stat, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(reparse_flag and file_attributes & reparse_flag)


class ImageMaskPair(NamedTuple):
    """One image matched to its mask by `discover_image_mask_pairs`.

    Both fields are plain `Path` objects, anchored under `image_root`/
    `mask_root` exactly as those were given (never resolved or made
    absolute) -- the same anchoring `discover_images` itself promises.
    """

    image: Path
    mask: Path


# Diagnostic messages for duplicate/unmatched pairing keys list every
# offending key, but a pathologically large dataset could otherwise produce
# an unusably huge exception; this caps how many already-sorted entries are
# shown before summarizing the rest by count.
_DIAGNOSTIC_PREVIEW_LIMIT = 10


def _format_diagnostic_preview(
    entries: Sequence[str], limit: int = _DIAGNOSTIC_PREVIEW_LIMIT
) -> str:
    """Join already-sorted diagnostic entries, truncating past `limit` with a summary count.

    Callers are responsible for sorting `entries` first -- this only joins
    and truncates, it never reorders.
    """
    if len(entries) <= limit:
        return "\n".join(entries)
    shown = "\n".join(entries[:limit])
    remaining = len(entries) - limit
    return f"{shown}\n... and {remaining} more"


def _strip_matched_extension_for_key(
    relative: str, normalized_extensions: tuple[str, ...], *, role: str
) -> str:
    """Strip the longest matching extension from `relative`, for use as a pairing key.

    `discover_images` only ever returns paths whose lowercased name ends
    with at least one of `normalized_extensions`, so `matched` below is
    never empty in practice -- the `RuntimeError` guards an internal
    invariant, not a reachable user input. When several configured
    extensions match (e.g. `.gz` and `.nii.gz` both matching
    `scan.nii.gz`), the longest one is removed, deterministically -- two
    *different* normalized extensions can never tie in length and both
    match the same suffix (that would require them to be the same string,
    which `_normalize_extensions` already deduplicates). Matching itself is
    case-insensitive, but the returned key keeps `relative`'s own casing
    for every character it retains.
    """
    lowered = relative.lower()
    matched = [ext for ext in normalized_extensions if lowered.endswith(ext)]
    if not matched:
        raise RuntimeError(
            f"internal error: {role} path {relative!r} does not end with any of the "
            f"configured {role} extensions {normalized_extensions!r}"
        )
    longest = max(matched, key=len)
    key = relative[: -len(longest)]
    if not key.rsplit("/", 1)[-1]:
        raise ValueError(
            f"{role} path {relative!r} does not define a non-empty pairing key after "
            f"removing extension {longest!r}"
        )
    return key


def _build_pairing_entries(
    paths: tuple[Path, ...],
    root: Path,
    normalized_extensions: tuple[str, ...],
    *,
    role: str,
) -> list[tuple[str, str, Path]]:
    """Compute `(key, relative_posix_path, path)` for every discovered path."""
    entries: list[tuple[str, str, Path]] = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        key = _strip_matched_extension_for_key(relative, normalized_extensions, role=role)
        entries.append((key, relative, path))
    return entries


def _group_pairing_entries(
    entries: list[tuple[str, str, Path]],
) -> tuple[dict[str, Path], dict[str, list[str]]]:
    """Split `entries` into a unique key->path map and a key->relative-paths duplicate map."""
    grouped: dict[str, list[tuple[str, Path]]] = {}
    for key, relative, path in entries:
        grouped.setdefault(key, []).append((relative, path))

    unique: dict[str, Path] = {}
    duplicates: dict[str, list[str]] = {}
    for key, items in grouped.items():
        if len(items) > 1:
            duplicates[key] = sorted(relative for relative, _ in items)
        else:
            ((_, path),) = items
            unique[key] = path
    return unique, duplicates


def _format_duplicate_error(
    image_duplicates: dict[str, list[str]], mask_duplicates: dict[str, list[str]]
) -> str:
    sections: list[str] = []
    if image_duplicates:
        entries = [f"{key!r}: {tuple(paths)!r}" for key, paths in sorted(image_duplicates.items())]
        sections.append("duplicate image pairing keys:\n" + _format_diagnostic_preview(entries))
    if mask_duplicates:
        entries = [f"{key!r}: {tuple(paths)!r}" for key, paths in sorted(mask_duplicates.items())]
        sections.append("duplicate mask pairing keys:\n" + _format_diagnostic_preview(entries))
    return "duplicate pairing keys found:\n\n" + "\n\n".join(sections)


def _format_unmatched_error(image_keys: set[str], mask_keys: set[str]) -> str:
    images_without_masks = sorted(image_keys - mask_keys)
    masks_without_images = sorted(mask_keys - image_keys)
    sections: list[str] = []
    if images_without_masks:
        entries = [repr(key) for key in images_without_masks]
        sections.append(
            "image keys without a matching mask:\n" + _format_diagnostic_preview(entries)
        )
    if masks_without_images:
        entries = [repr(key) for key in masks_without_images]
        sections.append(
            "mask keys without a matching image:\n" + _format_diagnostic_preview(entries)
        )
    return "image and mask pairing keys do not match:\n\n" + "\n\n".join(sections)


def discover_image_mask_pairs(
    image_root: str | os.PathLike[str],
    mask_root: str | os.PathLike[str],
    *,
    recursive: bool = True,
    image_extensions: Collection[str] | None = None,
    mask_extensions: Collection[str] | None = None,
    include_hidden: bool = False,
) -> tuple[ImageMaskPair, ...]:
    """Find images under `image_root` and masks under `mask_root`, paired by relative path.

    Runs `discover_images` once for each root (`image_root` with
    `image_extensions`, `mask_root` with `mask_extensions`), inheriting its
    full contract for both: extension-based only (no file is ever opened,
    decoded, or otherwise inspected), the same `recursive`/`include_hidden`
    policy for both roots, descendant symlinks/Windows reparse points always
    skipped, a fresh `os.stat(..., follow_symlinks=False)` per descendant,
    fail-fast on any native filesystem error, and paths anchored under their
    own root exactly as given (never resolved or made absolute). The two
    traversals are sequential, independent snapshots, not one atomic
    operation -- a file changed between the image traversal and the mask
    traversal (or after this function returns) is not detected, the same
    caveat `discover_images` itself already carries; no extra `stat` is
    performed here beyond what each `discover_images` call already does.

    Each discovered path is turned into a *pairing key*: its POSIX-style
    path relative to its own root, with exactly one matched extension
    removed from the end (the longest one, when several configured
    extensions could match, e.g. `.gz` and `.nii.gz` both matching
    `scan.nii.gz` strip to `scan`, not `scan.nii` -- see
    `_strip_matched_extension_for_key`). The relative directory is part of
    the key (`cats/001` and `dogs/001` are different keys, never merged by
    basename alone). Extension matching is case-insensitive; everything the
    key keeps beyond that keeps its original case and is never Unicode-
    normalized (`Cat/001` and `cat/001` are different keys; an NFC- and an
    NFD-normalized form of the same visual name are different keys). If
    removing the matched extension would leave an empty final path
    component (e.g. a file literally named `.jpg`, only reachable with
    `include_hidden=True`, or `subdir/.jpg`), this raises `ValueError`
    immediately rather than inventing a placeholder key.

    Pairing is a strict bijection: every image key must have exactly one
    mask with the same key, and vice versa, with no partial-result mode.
    Two different paths on the same side reducing to the same key is a
    duplicate and raises `ValueError` (listing every colliding key on
    either side, together with its colliding relative paths); an image
    key with no matching mask key, or a mask key with no matching image
    key, raises `ValueError` (listing every such key on either side). Both
    error messages truncate past 10 already-sorted entries with a trailing
    `"... and N more"` rather than dumping an unbounded diagnostic. `image_
    root == mask_root` is not itself rejected -- which physical files land
    on which side is governed entirely by `image_extensions`/`mask_
    extensions` (disjoint extension sets under a shared root never produce
    a collision); however, an image and a mask that resolve to the exact
    same path (only possible when the two extension sets overlap under a
    shared root) is rejected with `ValueError`, since an image and its own
    mask must be two distinct paths. None of this relies on `resolve()`,
    inode comparison, or an extra `stat` -- two syntactically different
    roots that happen to reference the same directory are not specially
    detected, and two different paths referencing the same file via a hard
    link remain a legal, distinct pair (this module never deduplicates by
    file identity).

    Both sides empty gives `()`. `average`-style partial modes
    (`strict=False`, `on_unmatched=...`) are not offered in this slice.

    This function only discovers and pairs *paths* -- it never opens,
    decodes, or inspects file contents, so it cannot and does not check
    image/mask dimensions, dtype, or any other semantic correspondence
    between a pair; that is left entirely to the caller (e.g. via
    `cv2.imread`). There is no suffix/prefix convention (`mask_suffix=`)
    in this slice -- an image and its mask must share the same relative
    path once their own extension is removed.

    Raises
    ------
    TypeError
        Same as `discover_images`, independently for `image_root`/`image_
        extensions` and `mask_root`/`mask_extensions` (error messages name
        the offending parameter as `image_root`/`mask_root`/`image_
        extensions`/`mask_extensions`, not the generic `root`/`extensions`
        `discover_images` itself uses), or if `recursive`/`include_hidden`
        is not exactly a `bool`.
    ValueError
        Same as `discover_images` for each root/extensions pair; if
        stripping a matched extension would leave an empty pairing key; if
        a pairing key collides within either side; if the image and mask
        key sets differ; or if an image and a mask resolve to the same
        path.
    FileNotFoundError, NotADirectoryError, OSError
        Same as `discover_images`, from either traversal.
    RuntimeError
        If a discovered path does not end with any configured extension
        for its own role -- an internal invariant `discover_images` itself
        already guarantees, never expected to trigger.
    """
    require_bool(recursive, "recursive")
    require_bool(include_hidden, "include_hidden")

    normalized_image_root = _normalize_root(image_root, name="image_root")
    normalized_image_extensions = _normalize_extensions(image_extensions, name="image_extensions")
    normalized_mask_root = _normalize_root(mask_root, name="mask_root")
    normalized_mask_extensions = _normalize_extensions(mask_extensions, name="mask_extensions")

    images = discover_images(
        normalized_image_root,
        recursive=recursive,
        extensions=normalized_image_extensions,
        include_hidden=include_hidden,
    )
    masks = discover_images(
        normalized_mask_root,
        recursive=recursive,
        extensions=normalized_mask_extensions,
        include_hidden=include_hidden,
    )

    image_entries = _build_pairing_entries(
        images, normalized_image_root, normalized_image_extensions, role="image"
    )
    mask_entries = _build_pairing_entries(
        masks, normalized_mask_root, normalized_mask_extensions, role="mask"
    )

    image_map, image_duplicates = _group_pairing_entries(image_entries)
    mask_map, mask_duplicates = _group_pairing_entries(mask_entries)

    if image_duplicates or mask_duplicates:
        raise ValueError(_format_duplicate_error(image_duplicates, mask_duplicates))

    image_keys = set(image_map)
    mask_keys = set(mask_map)
    if image_keys != mask_keys:
        raise ValueError(_format_unmatched_error(image_keys, mask_keys))

    self_paired = sorted(key for key in image_map if image_map[key] == mask_map[key])
    if self_paired:
        raise ValueError(
            f"image and mask paths must be distinct; self-paired keys: {tuple(self_paired)}"
        )

    return tuple(
        ImageMaskPair(image=image_map[key], mask=mask_map[key]) for key in sorted(image_map)
    )
