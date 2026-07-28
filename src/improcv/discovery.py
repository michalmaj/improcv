"""Deterministic, extension-based discovery of image files under a directory.

`discover_images` finds candidate image files by filename extension only --
it never opens, decodes, or otherwise inspects a file's content. This is a
read-only filesystem operation: it does not create, write, or touch
timestamps on anything, and it does not change the process's working
directory. It does not pair images with masks, infer classes from directory
names, produce train/validation/test splits, or load/decode any image --
those are all out of scope for this slice.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Collection, Mapping
from pathlib import Path

from improcv._validation import require_bool

__all__ = ["discover_images"]

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


def _normalize_root(root: object) -> Path:
    try:
        raw_root = os.fspath(root)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(
            f"root must be a str or os.PathLike[str], got {type(root).__name__}"
        ) from exc
    if not isinstance(raw_root, str):
        raise TypeError(
            f"root must resolve to a str, got {type(raw_root).__name__} from os.fspath() "
            "-- a PathLike whose __fspath__() returns bytes is not accepted"
        )
    if raw_root == "":
        raise ValueError("root must not be an empty string")

    root_path = Path(raw_root)
    root_stat = root_path.stat()
    if not stat.S_ISDIR(root_stat.st_mode):
        raise NotADirectoryError(f"root must reference a directory, got {raw_root!r}")
    return root_path


def _normalize_extensions(extensions: object) -> tuple[str, ...]:
    if extensions is None:
        return _DEFAULT_IMAGE_EXTENSIONS

    if isinstance(extensions, (str, bytes, bytearray)):
        raise TypeError(
            f"extensions must be a Collection[str], not a single str/bytes/bytearray, "
            f"got {type(extensions).__name__}"
        )
    if isinstance(extensions, Mapping):
        raise TypeError(
            f"extensions must be a Collection[str], not a Mapping, got {type(extensions).__name__}"
        )
    if not isinstance(extensions, Collection):
        raise TypeError(f"extensions must be a Collection[str], got {type(extensions).__name__}")
    if len(extensions) == 0:
        raise ValueError("extensions must not be empty")

    normalized: list[str] = []
    seen: set[str] = set()
    for item in extensions:
        if not isinstance(item, str):
            raise TypeError(f"extensions elements must be str, got {type(item).__name__}")
        if any(ch.isspace() for ch in item):
            raise ValueError(f"extensions element must not contain whitespace, got {item!r}")
        if any(ch in _FORBIDDEN_EXTENSION_CHARS for ch in item):
            raise ValueError(
                f"extensions element must not contain a path separator, NUL, or glob "
                f"character, got {item!r}"
            )
        candidate = item[1:] if item.startswith(".") else item
        if candidate == "":
            raise ValueError(f"extensions element must not be empty, got {item!r}")

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
