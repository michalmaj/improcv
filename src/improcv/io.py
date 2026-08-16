"""Unicode-safe single-image I/O: load a filesystem path into an ndarray, or save one back."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, cast, overload

import cv2
import numpy as np

from improcv._validation import require_dtype, require_image_ndim, require_one_of
from improcv.types import Image, ImageU8

__all__ = ["ImageReadMode", "load_image", "save_image"]

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


def save_image(path: str | os.PathLike[str], image: Image) -> None:
    """Encode `image` in memory and write the result to `path`, Unicode-safely.

    Encodes `image` with `cv2.imencode`, never `cv2.imwrite`, then writes the encoded bytes to
    `path` through Python's own filesystem path handling (`Path.write_bytes()`). `cv2.imwrite`'s
    filename-based handling is not reliable on Windows for paths containing characters outside
    the active code page; `cv2.imencode` never receives a filename at all, only an in-memory
    buffer and an extension string, and `Path.write_bytes()` does not have that limitation.

    For a successful call, this is exactly equivalent to calling
    ``Path(path).write_bytes(cv2.imencode(Path(path).suffix, image)[1].tobytes())`` directly --
    no additional transformation, resize, color conversion, or normalization is applied.

    Parameters
    ----------
    path : str | os.PathLike[str]
        A local filesystem destination. Never resolved, expanded (``~``), made absolute, or
        Unicode-normalized -- used exactly as given. The parent directory is never created -- a
        missing parent raises a native filesystem error (see Raises). An existing regular file at
        `path` is overwritten; there is no `overwrite=False`/no-clobber option.
    image : np.ndarray
        Must be `dtype=np.uint8`, 2-D or 3-D, and non-empty. No dtype conversion, clipping,
        normalization, or color conversion of any kind is applied -- channel order and layout are
        passed through to `cv2.imencode` exactly as given. Channel-count validity beyond
        dimensionality (e.g. 2 or 5 channels) is not pre-checked here -- an unsupported layout
        surfaces as an encode failure (see Raises). A non-contiguous array is accepted and encodes
        byte-identically to an equivalent contiguous copy.

    Returns
    -------
    None
        Never OpenCV's own success flag, a `Path`, `bytes`, or a result object.

    Raises
    ------
    TypeError
        If `path` is not a `str` or `os.PathLike[str]`, or resolves (via `os.fspath()`) to
        something other than `str` (e.g. a `PathLike` whose ``__fspath__()`` returns `bytes`); or
        if `image.dtype` is not `np.uint8`.
    ValueError
        If `path` is an empty string; if `image` does not have 2 or 3 dimensions or is empty; or
        if encoding fails -- either because `cv2.imencode` raises `cv2.error` (chained as
        ``__cause__``) or returns a false success flag. Both cases raise the same `ValueError`
        naming `path`. The encoder is selected entirely from `path`'s final suffix (`Path.suffix`,
        case-insensitive), with no allowlist, normalization, or content-based detection -- an
        unrecognized or missing suffix is an encode failure like any other. Codec/format support
        beyond what the installed OpenCV build provides is never promised by this function.

        A path containing an embedded NUL byte is a distinct case: `save_image` does not
        pre-validate for it, and it never reaches the encode step. Python's own path/filesystem
        layer raises a native `ValueError` (exact wording is platform-dependent, e.g. "embedded
        null byte" on POSIX) from within `Path.write_bytes()`, and it propagates unchanged --
        never renamed, wrapped, or normalized -- exactly like the native `OSError` cases below.
    OSError
        A native filesystem error writing `path` (`FileNotFoundError` for a missing parent,
        `PermissionError`, `IsADirectoryError`, or any other `OSError`) propagates unchanged --
        never wrapped into `ValueError`, since it is a different failure mode from an encode
        failure.

    Notes
    -----
    `Path.write_bytes()` is called only after `cv2.imencode` has already succeeded, so an
    **encode** failure leaves any pre-existing destination byte-for-byte unchanged. That guarantee
    stops there: once the filesystem write itself begins, no atomicity is guaranteed -- no
    temporary file, `fsync`, or atomic rename is used, so a filesystem-level write failure, crash,
    or interruption during that write can still leave an existing destination truncated or
    partially written. An ordinary successful overwrite is not atomic either -- it is a plain
    `Path.write_bytes()` call, nothing more. `image` is never mutated. No encoder parameters
    (quality, compression level, ...), metadata (EXIF, ICC profile, ...), or format-specific
    options are supported.
    """
    destination = _normalize_path(path)
    require_image_ndim(image, ndims=(2, 3))
    require_dtype(image, (np.uint8,), "image")

    suffix = destination.suffix
    try:
        ok, encoded = cv2.imencode(suffix, image)
    except cv2.error as exc:
        raise ValueError(f"failed to encode image for {str(destination)!r}") from exc
    if not ok:
        raise ValueError(f"failed to encode image for {str(destination)!r}")

    destination.write_bytes(encoded.tobytes())
