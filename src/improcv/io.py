"""Unicode-safe single-image loading: filesystem path in, decoded ndarray out."""

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

_MODES: tuple[ImageReadMode, ...] = ("color", "grayscale", "unchanged")

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
            f"path must resolve to a str, got {type(raw_path).__name__} from os.fspath() "
            "-- a PathLike whose __fspath__() returns bytes is not accepted"
        )
    if raw_path == "":
        raise ValueError("path must not be an empty string")
    return Path(raw_path)


def _validate_mode(mode: object) -> None:
    # `require_one_of`'s `value not in allowed` membership check is only safe once `value` is
    # known to be a plain `str` -- a non-str object (e.g. a NumPy array) can make `==`/`bool()`
    # raise or behave ambiguously (unhashable-type errors, "truth value of an array is ambiguous")
    # instead of the controlled `ValueError` this API promises. Gating on `isinstance(mode, str)`
    # first makes every runtime object -- comparable or not, hashable or not -- resolve to the
    # same controlled `ValueError`.
    if not isinstance(mode, str):
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
    require_one_of(mode, _MODES, "mode")


@overload
def load_image(path: str | os.PathLike[str], *, mode: Literal["color"] = "color") -> ImageU8: ...
@overload
def load_image(path: str | os.PathLike[str], *, mode: Literal["grayscale"]) -> ImageU8: ...
@overload
def load_image(path: str | os.PathLike[str], *, mode: Literal["unchanged"]) -> Image: ...
def load_image(path: str | os.PathLike[str], *, mode: ImageReadMode = "color") -> Image:
    """Read `path`'s bytes and decode them into a NumPy image, Unicode-safely.

    Reads `path` through Python's own filesystem path handling (`Path.read_bytes()`) and decodes
    the resulting bytes with `cv2.imdecode` -- never `cv2.imread`. `cv2.imread`'s filename-based
    handling is not reliable on Windows for paths containing characters outside the active code
    page; `Path.read_bytes()` does not have that limitation, and `cv2.imdecode` never receives a
    filename at all, only an in-memory buffer.

    For a successful call, this is exactly equivalent to calling
    ``cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), <flag for mode>)`` directly
    against the same bytes -- no additional transformation, resize, color conversion, or
    normalization is applied beyond what `mode` itself selects.

    Parameters
    ----------
    path : str | os.PathLike[str]
        A local filesystem path. Never resolved, expanded (``~``), made absolute, or
        Unicode-normalized -- used exactly as given. A symlink is followed via `Path.
        read_bytes()`'s ordinary semantics; this function performs no symlink-specific handling
        of its own. The file's extension/suffix is never inspected -- decode capability is
        determined entirely by the file's actual encoded content.
    mode : {"color", "grayscale", "unchanged"}, default "color"
        Decode policy, passed through to `cv2.imdecode` as the corresponding `cv2.IMREAD_*` flag:

        - ``"color"`` (`cv2.IMREAD_COLOR`): always 3 channels, BGR order, dtype `uint8`. A
          higher-bit-depth source is downcast, an alpha channel is dropped, and a grayscale
          source is promoted to 3 channels -- the exact downcast/promotion algorithm is OpenCV's
          own and is not reproduced here. EXIF orientation is applied automatically by OpenCV's
          decode path when present (verified for JPEG).
        - ``"grayscale"`` (`cv2.IMREAD_GRAYSCALE`): always 2-D, dtype `uint8`. This is the
          codec/OpenCV's own grayscale decode policy -- **not** guaranteed bit-identical to
          ``ensure_gray(load_image(path, mode="color"))``; the two are independent OpenCV code
          paths and have been verified to differ by up to 1 per pixel for a genuinely color
          source. EXIF orientation is applied automatically, identically to ``"color"``.
        - ``"unchanged"`` (`cv2.IMREAD_UNCHANGED`): returns the ndarray produced by OpenCV's
          `IMREAD_UNCHANGED` decode policy; this is still a decoded representation, not the
          encoded file's raw pixel/storage representation or metadata. Dtype and channel count
          **may vary with the encoded input and codec** -- not a fixed, universal guarantee for
          every codec/build. Verified directly for PNG: 8-bit and 16-bit grayscale/BGR/BGRA
          sources are preserved exactly, but a palette/indexed source is instead expanded by
          OpenCV to `uint8` BGR rather than preserved as an index map (do not assume index-map
          preservation for any other source/codec not verified here). EXIF orientation is **not**
          applied, even without `cv2.IMREAD_IGNORE_ORIENTATION` set --
          a real, documented OpenCV exception to the other two modes' default. No metadata (EXIF,
          ICC profile, or otherwise) is ever returned, for any mode.

        This function has no mask-specific semantics for any mode -- in particular,
        ``mode="unchanged"`` on a segmentation mask file does not guarantee palette class-index
        preservation, a single-channel result, a specific class-index dtype, or label validity.

    Returns
    -------
    np.ndarray
        `mode="color"`/`mode="grayscale"` always return `uint8`; `mode="unchanged"` returns
        whatever dtype OpenCV's decode produced for that source. A new, independent array --
        never a view over `path`'s bytes.

    Raises
    ------
    TypeError
        If `path` is not a `str` or `os.PathLike[str]`, or resolves (via `os.fspath()`) to
        something other than `str` (e.g. a `PathLike` whose ``__fspath__()`` returns `bytes`).
    ValueError
        If `path` is an empty string; if `mode` is not one of `"color"`/`"grayscale"`/
        `"unchanged"`; if the file at `path` is empty; or if its bytes cannot be decoded as an
        image for the requested `mode`. The last two cases both raise `ValueError` naming `path`
        -- an empty file is treated as undecodable without ever calling `cv2.imdecode` on an
        empty buffer.
    OSError
        A native filesystem error reading `path` (`FileNotFoundError`, `PermissionError`,
        `IsADirectoryError`, or any other `OSError`) propagates unchanged -- never wrapped into
        `ValueError`, since it is a different failure mode from an undecodable image.

    Notes
    -----
    Reads `path`'s bytes exactly once. Performs no filesystem write, no caching, and no directory
    traversal of any kind -- this is not a discovery/batch API (see `discover_images`/
    `discover_image_mask_pairs` for that). Decode time and memory are delegated entirely to the
    OpenCV codec and the decoded image's own size, not characterized here.
    """
    source = _normalize_path(path)
    _validate_mode(mode)

    payload = source.read_bytes()
    if not payload:
        raise ValueError(f"failed to decode image from {str(source)!r}: file is empty")

    buffer = np.frombuffer(payload, dtype=np.uint8)
    try:
        decoded = cv2.imdecode(buffer, _FLAGS[mode])
    except cv2.error as exc:
        raise ValueError(f"failed to decode image from {str(source)!r}") from exc
    if decoded is None:
        raise ValueError(f"failed to decode image from {str(source)!r}")

    if mode == "unchanged":
        return cast(Image, decoded)
    return cast(ImageU8, decoded)
