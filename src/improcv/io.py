"""Filesystem <-> decoded-array boundary: public, Unicode-safe single-image loading.

`load_image` reads a file's bytes through `Path.read_bytes()` and decodes them with
`cv2.imdecode` -- never `cv2.imread` -- so a path containing characters outside a Windows
machine's active code page still works, exactly like `dataset.py`'s private `_decode_grayscale`.
See `docs/design/0.4.0a4-load-image.md` for the full frozen contract this module implements.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, cast, overload

import cv2
import numpy as np

from improcv._validation import require_one_of
from improcv.types import Image, ImageU8

__all__ = ["ImageReadMode", "load_image"]

ImageReadMode = Literal["color", "grayscale", "unchanged"]
"""Which OpenCV decode policy `load_image` applies: `"color"` (default, 3-channel BGR `uint8`),
`"grayscale"` (2-D `uint8`, not promised equal to `ensure_gray(load_image(..., mode="color"))`),
or `"unchanged"` (OpenCV's own `IMREAD_UNCHANGED` decode; dtype/channel count vary by source).
"""

_FLAGS: dict[ImageReadMode, int] = {
    "color": cv2.IMREAD_COLOR,
    "grayscale": cv2.IMREAD_GRAYSCALE,
    "unchanged": cv2.IMREAD_UNCHANGED,
}


def _normalize_path(path: object) -> Path:
    try:
        raw_path = os.fspath(path)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(
            f"path must be a str or os.PathLike[str], got {type(path).__name__}"
        ) from exc

    if not isinstance(raw_path, str):
        raise TypeError(
            f"path must resolve to a str, got {type(raw_path).__name__} "
            "from os.fspath() -- a PathLike whose __fspath__() returns bytes "
            "is not accepted"
        )

    if raw_path == "":
        raise ValueError("path must not be an empty string")

    return Path(raw_path)


@overload
def load_image(
    path: str | os.PathLike[str],
    *,
    mode: Literal["color"] = "color",
) -> ImageU8: ...
@overload
def load_image(
    path: str | os.PathLike[str],
    *,
    mode: Literal["grayscale"],
) -> ImageU8: ...
@overload
def load_image(
    path: str | os.PathLike[str],
    *,
    mode: Literal["unchanged"],
) -> Image: ...
def load_image(
    path: str | os.PathLike[str],
    *,
    mode: ImageReadMode = "color",
) -> Image:
    """Read one local image file's bytes and decode it with OpenCV, Unicode-safe.

    Reads `path`'s bytes exactly once via `Path.read_bytes()` and decodes them with
    `cv2.imdecode` -- never `cv2.imread(str(path), ...)`, whose filename-based handling is not
    reliable on Windows for paths containing characters outside the active code page. By the time
    `cv2.imdecode` runs, it only ever sees an in-memory buffer; no filename is involved in the
    decode step at all. No stat-before-read, no extension validation, no format-guessing from the
    file suffix, no filesystem write, and no caching.

    `mode="color"` and `mode="grayscale"` apply OpenCV's own automatic JPEG EXIF-orientation
    handling; `mode="unchanged"` does not (a real OpenCV exception, not an oversight). No mode
    performs any transformation beyond what OpenCV's own decode produces: no resize, no color
    conversion beyond what `mode` itself selects, no normalization, no copy-with-modification. For
    the same bytes and the same OpenCV build/version, `load_image(path, mode=X)` is
    `np.array_equal` to `cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), FLAG(X))`
    computed against those same bytes.

    `load_image` has no mask-specific semantics for any mode -- `mode="unchanged"` is not a safe,
    general-purpose mask loader. It does not preserve a palette/indexed color table as an index
    map (OpenCV expands a palette PNG to BGR color); does not preserve any file metadata (EXIF,
    ICC profile, ...); and makes no promise about which frame an animated/multipage source yields.

    Parameters
    ----------
    path : str | os.PathLike[str]
        The image file to read. Never expanded (``~``), resolved, made absolute, Unicode-
        normalized, or case-folded. May be relative or absolute. A `PathLike` whose
        `__fspath__()` returns `bytes` is rejected (`TypeError`), not silently accepted.
    mode : ImageReadMode, optional
        `"color"` (default): always exactly 3 channels, BGR order, `uint8` -- a 16-bit or
        grayscale source is downcast/promoted; a BGRA source has its alpha dropped. `"grayscale"`:
        always exactly 2-D, `uint8`; this is OpenCV's own grayscale decode policy, not a promise
        of bit-for-bit equality with `ensure_gray(load_image(path, mode="color"))` -- the two can
        differ by up to 1 per pixel for a genuinely color source. `"unchanged"`: returns whatever
        `cv2.IMREAD_UNCHANGED` produces -- dtype/channel count vary by source (`uint8`/`uint16`,
        1/3/4 channels observed).

    Returns
    -------
    ImageU8
        For `mode="color"` (default) or `mode="grayscale"`.
    Image
        For `mode="unchanged"`, since its dtype genuinely varies by source. The static overloads
        narrow the two `uint8` modes to `ImageU8`; the runtime array itself is returned unchanged
        in every mode -- narrowing is typing-only, with no run-time effect on the array's data.

    Raises
    ------
    TypeError
        If `path` is not a `str`/`os.PathLike[str]`, or resolves via `os.fspath()` to something
        other than `str` (e.g. a `PathLike` returning `bytes`, or `bytes`/`bytearray` directly).
    ValueError
        If `path` is an empty string; if `mode` is not one of `"color"`/`"grayscale"`/
        `"unchanged"` (any other value, of any type, including a raw `cv2.IMREAD_COLOR` int); if
        `path`'s bytes are empty (`b""`) -- `cv2.imdecode` is never called on an empty buffer; or
        if `cv2.imdecode` fails to decode `path`'s bytes (a corrupt file, or a codec this OpenCV
        build does not support) -- both `ValueError` cases name `path` in their message.
    OSError
        `FileNotFoundError`, `PermissionError`, `IsADirectoryError`, or any other `OSError` raised
        by `Path.read_bytes()` itself propagates completely unwrapped -- distinct from the
        `ValueError` cases above, which are always about the bytes being unreadable as an image,
        never about the read itself failing.
    """
    source = _normalize_path(path)
    require_one_of(mode, ("color", "grayscale", "unchanged"), "mode")

    payload = source.read_bytes()
    if not payload:
        raise ValueError(f"failed to decode image from {str(source)!r}: file is empty")

    buffer = np.frombuffer(payload, dtype=np.uint8)
    decoded = cv2.imdecode(buffer, _FLAGS[mode])
    if decoded is None:
        raise ValueError(f"failed to decode image from {str(source)!r}")

    if mode == "unchanged":
        return decoded
    # color/grayscale both always decode to uint8 per OpenCV's own documented IMREAD_COLOR/
    # IMREAD_GRAYSCALE contract; this cast is typing-only and has no runtime effect.
    return cast(ImageU8, decoded)
