"""OpenCV DNN blob preprocessing: turn image(s) into NCHW `float32` blobs for `cv2.dnn`.

This module wraps `cv2.dnn.blobFromImage`/`blobFromImages` -- it does not load
models, create a `cv2.dnn.Net`, or run inference. Verified directly (source and
empirically, across OpenCV 4.9/4.13/5.0) that the raw OpenCV functions behave
inconsistently across that version range for several inputs that this module
restricts or validates explicitly; see each private helper's docstring for the
specific finding it exists to guard against.
"""

from __future__ import annotations

import numbers
import os
import stat
import warnings
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from improcv._compat.opencv import read_onnx_net_from_buffer, read_onnx_net_from_path
from improcv._validation import (
    require_bool,
    require_dtype,
    require_positive_integral,
    require_real_number,
)
from improcv.types import Image, ImageFloat32

__all__ = [
    "create_dnn_batch_blob",
    "create_dnn_blob",
    "load_onnx_network",
    "load_onnx_network_from_bytes",
]

StrPath = str | os.PathLike[str]
"""A file path given as `str` or an `os.PathLike[str]` (e.g. `pathlib.Path`).

Deliberately narrower than a hypothetical "any path-like" alias: a
`PathLike` whose `__fspath__()` returns `bytes` is not accepted here (see
`_normalize_onnx_path`), so this module's own type parameter is `str`, not
`str | bytes`, on the `os.PathLike` side too.
"""

_ALLOWED_DTYPES = (np.uint8, np.float32)
_ALLOWED_CHANNELS = (1, 3, 4)
_MAX_INT32 = int(np.iinfo(np.int32).max)


def create_dnn_blob(
    image: Image,
    *,
    size: tuple[int, int] | None = None,
    scale: float = 1.0,
    mean: float | tuple[float, ...] = 0.0,
    swap_rb: bool = False,
    crop: bool = False,
) -> ImageFloat32:
    """Convert a single image into a 4-D NCHW `float32` blob for `cv2.dnn`.

    `image` must be a non-empty `uint8` or `float32` array, grayscale
    (`(H, W)` or `(H, W, 1)`), BGR (`(H, W, 3)`), or BGRA (`(H, W, 4)`) --
    other dtypes and channel counts are rejected rather than silently
    converted. `float32` input must be entirely finite.

    `size` is `(width, height)`. `size=None` keeps `image`'s native spatial
    size (no resize). `crop=False` resizes directly to `size`, stretching
    without preserving aspect ratio; `crop=True` uniformly scales `image` so
    it covers `size` in both dimensions, then takes a centered crop --
    `crop=True` with `size=None` is rejected as a no-op that is almost
    certainly a caller mistake. Both modes use OpenCV's `INTER_LINEAR`;
    there is no `interpolation` parameter because the underlying OpenCV
    function does not expose one.

    `scale` may be zero or negative -- only non-finite values (`NaN`/an
    overflow-to-infinity) are rejected. `mean` is a single value (broadcast
    to every channel) or a tuple with exactly one element per channel; both
    forms are converted through `float32` and must remain finite. `mean`'s
    channel order matches the *output* channel order: for BGR/BGRA,
    `swap_rb=False` means `(B, G, R[, A])` and `swap_rb=True` means
    `(R, G, B[, A])` -- the alpha channel, if present, is never touched by
    `swap_rb`. `swap_rb=True` on a single-channel image is rejected (it
    would otherwise silently do nothing).

    The result is a new, independent array (never a view of `image`);
    `image` is never mutated. Output shape is always `(1, C, H, W)` and
    dtype is always `float32`. The returned blob is guaranteed finite: if
    a finite input combined with an extreme `scale` or `mean` overflows
    during OpenCV preprocessing, this function raises `RuntimeError`
    rather than returning a blob containing `NaN`/infinity -- it never
    clips or otherwise rescales the result to make that guarantee hold,
    it only refuses to return the non-finite value. There is no `[0, 1]`/
    `[0, 255]` range requirement on the output.

    Raises
    ------
    TypeError
        If `image` is not a NumPy array, has a dtype other than `uint8`/
        `float32`, or if `mean`/`scale`/`swap_rb`/`crop`/`size` have the
        wrong type (including a non-`bool` `swap_rb`/`crop`, or a `bool`
        passed where a number is expected).
    ValueError
        If `image` is empty, has an unsupported number of dimensions or
        channels, contains non-finite values (`float32` only), if `size`
        is invalid (non-positive, too large, or `None` with `crop=True`),
        or if `scale`/`mean` are non-finite once converted to `float32`.
    RuntimeError
        If OpenCV itself fails on input that passed all of the above
        validation, or returns a result that fails this function's own
        postconditions (shape/dtype/finiteness).
    """
    _require_dnn_image(image, "image")
    channels = _channel_count(image)
    require_bool(swap_rb, "swap_rb")
    require_bool(crop, "crop")
    normalized_size = _normalize_size(size, "size", required=False)
    if crop and normalized_size is None:
        raise ValueError(
            "size must not be None when crop=True -- cropping with no target size is a no-op"
        )
    _require_swap_rb_applicable(swap_rb, channels)
    normalized_scale = _normalize_scale(scale)
    normalized_mean = _normalize_mean(mean, channels)

    cv2_size = normalized_size if normalized_size is not None else (0, 0)
    try:
        result = cv2.dnn.blobFromImage(
            image,
            scalefactor=normalized_scale,
            size=cv2_size,
            mean=normalized_mean,
            swapRB=swap_rb,
            crop=crop,
            ddepth=cv2.CV_32F,
        )
    except cv2.error as exc:
        raise RuntimeError(
            "OpenCV DNN blob preprocessing failed for input that passed improcv validation"
        ) from exc

    if normalized_size is not None:
        expected_width, expected_height = normalized_size
    else:
        expected_height, expected_width = image.shape[0], image.shape[1]
    return _check_blob_postconditions(
        result,
        expected_batch=1,
        expected_channels=channels,
        expected_height=expected_height,
        expected_width=expected_width,
    )


def create_dnn_batch_blob(
    images: Sequence[Image],
    *,
    size: tuple[int, int],
    scale: float = 1.0,
    mean: float | tuple[float, ...] = 0.0,
    swap_rb: bool = False,
    crop: bool = False,
) -> ImageFloat32:
    """Convert a sequence of images into a single 4-D NCHW `float32` blob for `cv2.dnn`.

    `images` must be a real `collections.abc.Sequence` (e.g. a `list` or
    `tuple`) of at least one image -- a single `np.ndarray` (including a
    4-D stack), `str`/`bytes`/`bytearray`, and a generator/iterator are all
    rejected explicitly. Every element is validated exactly like
    `create_dnn_blob`'s `image` (dtype, channels, finiteness), with the
    offending index named in the error message; every element must also
    share the same dtype and channel count as `images[0]` (spatial size
    may differ between elements).

    Unlike `create_dnn_blob`, `size` is required here, not optional.
    Verified directly that without an explicit `size`, OpenCV silently
    resizes every image after the first to match the *first* image's
    native size -- a data-dependent, easy-to-miss footgun for a batch of
    differently-sized images -- so this function does not offer a "keep
    native size" default at all.

    See `create_dnn_blob` for the exact meaning of `size`/`scale`/`mean`/
    `swap_rb`/`crop`, all of which apply identically here, as one shared
    value for the whole batch (there is no per-image `mean`/`scale`).

    The result is a new, independent array; none of `images`' elements are
    mutated. Output shape is always `(N, C, H, W)` where `N == len(images)`,
    dtype always `float32`. The returned blob is guaranteed finite -- see
    `create_dnn_blob` for what happens if a finite input combined with an
    extreme `scale`/`mean` would otherwise overflow.

    Raises
    ------
    TypeError
        If `images` is not a `Sequence` (or is a `str`/`bytes`/`bytearray`/
        `np.ndarray`), if any element is not a NumPy array or has a dtype
        other than `uint8`/`float32`, if any element's dtype disagrees with
        `images[0]`'s, or for the same parameter-type errors as
        `create_dnn_blob`.
    ValueError
        For the same value errors as `create_dnn_blob` (now applied
        per-element where relevant), plus an empty `images`, a channel-count
        mismatch between elements, or more images than fit in a signed
        32-bit count.
    RuntimeError
        Same as `create_dnn_blob`.
    """
    normalized_images = _require_dnn_image_stack(images, "images")
    channels = _channel_count(normalized_images[0])
    require_bool(swap_rb, "swap_rb")
    require_bool(crop, "crop")
    normalized_size = _normalize_size(size, "size", required=True)
    assert normalized_size is not None  # required=True guarantees this
    _require_swap_rb_applicable(swap_rb, channels)
    normalized_scale = _normalize_scale(scale)
    normalized_mean = _normalize_mean(mean, channels)

    batch_size = len(normalized_images)
    if batch_size > _MAX_INT32:
        raise ValueError(
            f"images must contain at most {_MAX_INT32} images (a signed 32-bit count), "
            f"got {batch_size}"
        )

    try:
        result = cv2.dnn.blobFromImages(
            normalized_images,
            scalefactor=normalized_scale,
            size=normalized_size,
            mean=normalized_mean,
            swapRB=swap_rb,
            crop=crop,
            ddepth=cv2.CV_32F,
        )
    except cv2.error as exc:
        raise RuntimeError(
            "OpenCV DNN blob preprocessing failed for input that passed improcv validation"
        ) from exc

    expected_width, expected_height = normalized_size
    return _check_blob_postconditions(
        result,
        expected_batch=batch_size,
        expected_channels=channels,
        expected_height=expected_height,
        expected_width=expected_width,
    )


def load_onnx_network(path: StrPath) -> cv2.dnn.Net:
    """Load an ONNX network from a file into a `cv2.dnn.Net`.

    `path` must reference an existing, non-empty, regular file -- a
    directory, a special file (FIFO/socket/device), or an empty file are
    all rejected before OpenCV ever sees them. The file's *content* is
    what's parsed as ONNX; there is no extension check, and a valid ONNX
    file without a `.onnx` extension (or with an unusual one) is accepted
    -- verified directly that `cv2.dnn.readNetFromONNX` itself never
    inspects the extension. A symlink to a valid regular file is followed
    and accepted; a broken symlink raises `FileNotFoundError`, the same as
    a path that doesn't exist at all.

    **A path containing non-ASCII characters is not guaranteed to work on
    every platform.** Verified directly (via CI): the same accented-Unicode
    path that opens correctly on Linux/macOS makes OpenCV's own
    file-opening code fail on Windows, surfacing here as `RuntimeError`
    (OpenCV's `cv2.error`, not a bug in this wrapper's own validation,
    which has already confirmed the file exists, is a non-empty regular
    file). Prefer an ASCII-only path if you need this to work identically
    across platforms.

    This function only loads the network -- it never calls `setInput`,
    `forward`, reads layer names, inspects input/output shapes, or sets a
    backend/target. Every call parses the file again and returns a new,
    independent `cv2.dnn.Net`; nothing is cached.

    On OpenCV 5, improcv requests `ENGINE_CLASSIC` as the common behavior
    shared with OpenCV 4.x. OpenCV process configuration, including
    `OPENCV_FORCE_DNN_ENGINE`, may override that request.

    The returned `Net` is a stateful object: calling `setInput()` on it
    later mutates it, backend/target configuration is the caller's
    responsibility, and this function makes no thread-safety promise about
    it.

    Raises
    ------
    TypeError
        If `path` is not a `str` or `os.PathLike[str]` (including a
        `PathLike` whose `__fspath__()` returns `bytes`).
    ValueError
        If `path` is an empty string, or resolves to a file that is not a
        regular file (e.g. a FIFO or device), or is a regular file of size
        zero.
    FileNotFoundError
        If `path` does not exist, or is a symlink whose target does not
        exist.
    IsADirectoryError
        If `path` is a directory.
    PermissionError
        If checking `path` itself is denied (e.g. missing traversal
        permission on a parent directory). A file that exists and is
        stat-able but not readable is a different case -- verified
        directly that `stat()` needs no read permission on the file's own
        content, so that denial only surfaces once OpenCV itself tries to
        open the file, as `RuntimeError` (below), not `PermissionError`.
    RuntimeError
        If OpenCV fails to open or parse the file's content as ONNX
        (including a permission or ACL denial that only manifests when
        OpenCV opens the file, a TOCTOU race, or a network filesystem
        issue), or if the loaded network fails this function's own
        postconditions
        (wrong type, or an empty network).
    """
    normalized_path = _normalize_onnx_path(path)
    try:
        net = read_onnx_net_from_path(normalized_path)
    except cv2.error as exc:
        raise RuntimeError("OpenCV failed to load the ONNX network") from exc
    return _check_net_postconditions(net)


def load_onnx_network_from_bytes(data: bytes) -> cv2.dnn.Net:
    """Load an ONNX network from an in-memory buffer into a `cv2.dnn.Net`.

    `data` must be a `bytes` object -- `bytearray`, `memoryview`, an
    `ndarray`, a `list`/`tuple` of ints, and a `str`/`Path` are all
    rejected, even though the raw `cv2.dnn.readNetFromONNX` buffer overload
    happens to accept several of them. `bytes` is deliberately the only
    accepted type: it gives the simplest possible buffer-ownership
    contract (immutable, so there's nothing to reason about regarding a
    caller mutating it after the call), and -- verified directly -- OpenCV
    5.0.0's own type stubs for the buffer overload only declare
    `numpy.ndarray[Any, numpy.dtype[numpy.uint8]]`, never `bytes`, so this
    function converts internally rather than exposing that mismatch.

    Internally, `data` is wrapped (not copied) via
    `np.frombuffer(data, dtype=np.uint8)` and passed to OpenCV by the
    `buffer=` keyword, never positionally -- verified directly that a
    `bytes` object passed positionally as the sole argument is silently
    routed to the *path* overload rather than the buffer overload on
    OpenCV 4.13.0/5.0.0 (see `improcv._compat.opencv.read_onnx_net_from_buffer`).
    The returned `Net` does not depend on the local buffer or `data`
    staying alive -- OpenCV's ONNX parser consumes the buffer's content
    synchronously, inside the call.

    See `load_onnx_network` for what this function does and does not do
    beyond loading (no inference, no caching, best-effort `ENGINE_CLASSIC`
    on OpenCV 5, a stateful, non-thread-safe result).

    Raises
    ------
    TypeError
        If `data` is not exactly a `bytes` object.
    ValueError
        If `data` is empty.
    RuntimeError
        If OpenCV fails to parse `data` as ONNX, or if the loaded network
        fails this function's own postconditions (wrong type, or an empty
        network).
    """
    if not isinstance(data, bytes):
        raise TypeError(f"data must be bytes, got {type(data).__name__}")
    if len(data) == 0:
        raise ValueError("data must not be empty")

    buffer = np.frombuffer(data, dtype=np.uint8)
    try:
        net = read_onnx_net_from_buffer(buffer)
    except cv2.error as exc:
        raise RuntimeError("OpenCV failed to load the ONNX network") from exc
    return _check_net_postconditions(net)


def _channel_count(image: np.ndarray) -> int:
    return 1 if image.ndim == 2 else image.shape[2]


def _require_dnn_image(image: object, name: str) -> None:
    """Raise TypeError/ValueError unless `image` is a valid `create_dnn_blob` input.

    Order matches the rest of the project (see `stitching._require_valid_stitch_stack`):
    type, then shape/channels/emptiness, then dtype, then (for `float32`)
    finiteness. Only `uint8`/`float32` are accepted -- verified directly
    (source and empirically) that `int16`/`uint16`/`float64` are silently
    accepted and converted by `cv2.dnn.blobFromImage` on OpenCV 4.13/5.0 but
    raise a raw `cv2.error` on OpenCV 4.9 (the project's floor), which would
    make this function's behavior depend on the installed OpenCV version if
    those dtypes were allowed through.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array, got {type(image).__name__}")
    if image.ndim not in (2, 3):
        raise ValueError(f"{name} must have 2 or 3 dimensions, got {image.ndim}")
    if image.size == 0:
        raise ValueError(f"{name} must not be empty, got shape {image.shape}")
    if image.ndim == 3 and image.shape[2] not in _ALLOWED_CHANNELS:
        raise ValueError(
            f"{name} must have 1, 3, or 4 channels (grayscale, BGR, or BGRA), "
            f"got shape {image.shape}"
        )
    require_dtype(image, _ALLOWED_DTYPES, name)
    if image.dtype == np.float32 and not np.all(np.isfinite(image)):
        raise ValueError(f"{name} must contain only finite values")


def _require_dnn_image_stack(images: object, name: str) -> list[np.ndarray]:
    """Raise TypeError/ValueError unless `images` is a valid `create_dnn_batch_blob` input.

    Mirrors `stitching._require_valid_stitch_stack`'s container check: a
    real `Sequence`, not a `str`/`bytes`/`bytearray`, and -- verified
    directly -- not a bare `np.ndarray` either, since `isinstance(array,
    Sequence)` is `False` for NumPy arrays (they satisfy the sequence
    *protocol* structurally, which is enough for OpenCV's own pybind
    parameter parsing to silently treat a 4-D array as a stack of images,
    but not enough to pass `isinstance(..., collections.abc.Sequence)`).
    """
    if isinstance(images, (str, bytes, bytearray)) or not isinstance(images, Sequence):
        raise TypeError(
            f"{name} must be a Sequence of arrays (e.g. a list or tuple), not a single "
            f"array or {type(images).__name__}"
        )
    normalized = list(images)
    if len(normalized) == 0:
        raise ValueError(f"{name} must contain at least one image, got an empty sequence")

    for index, image in enumerate(normalized):
        _require_dnn_image(image, f"{name}[{index}]")

    first_dtype = normalized[0].dtype
    first_channels = _channel_count(normalized[0])
    for index, image in enumerate(normalized[1:], start=1):
        element_name = f"{name}[{index}]"
        if image.dtype != first_dtype:
            raise TypeError(
                f"{element_name} has dtype {image.dtype}, but {name}[0] has dtype "
                f"{first_dtype} -- every image in a batch must share the same dtype"
            )
        channels = _channel_count(image)
        if channels != first_channels:
            raise ValueError(
                f"{element_name} has {channels} channel(s), but {name}[0] has "
                f"{first_channels} -- every image in a batch must share the same channel count"
            )
    return normalized


def _require_swap_rb_applicable(swap_rb: bool, channels: int) -> None:
    """Raise ValueError if `swap_rb=True` is applied to a single-channel image.

    Verified directly that OpenCV itself accepts this combination and
    silently does nothing (only logging an internal C++ warning, invisible
    from Python) rather than erroring -- rejected here instead, since a
    caller passing `swap_rb=True` for a grayscale image almost certainly
    expected some effect.
    """
    if swap_rb and channels == 1:
        raise ValueError("swap_rb=True has no effect on a single-channel (grayscale) image")


def _normalize_size(value: object, name: str, *, required: bool) -> tuple[int, int] | None:
    """Raise TypeError/ValueError unless `value` is a valid `(width, height)` size.

    Accepts `None` (only when `required=False`), else a 2-tuple of
    positive Python/NumPy integral scalars (never `bool`) that individually
    fit in a signed 32-bit int. Additionally requires `width * height` to
    fit in a signed 32-bit int -- verified directly, in OpenCV's source,
    that the newer fast NCHW blob-construction path (OpenCV >= 4.13) stores
    a per-image element count as a C++ `int`, not just each dimension
    individually.
    """
    if value is None:
        if required:
            raise TypeError(f"{name} must be a 2-tuple of positive integers, got None")
        return None
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{name} must be a 2-tuple of positive integers, got {value!r}")
    width, height = value
    require_positive_integral(width, f"{name}[0]")
    require_positive_integral(height, f"{name}[1]")
    width_int, height_int = int(width), int(height)
    if width_int > _MAX_INT32 or height_int > _MAX_INT32:
        raise ValueError(
            f"{name}'s dimensions must each fit in a signed 32-bit int (<= {_MAX_INT32}), "
            f"got {(width_int, height_int)}"
        )
    if width_int * height_int > _MAX_INT32:
        raise ValueError(
            f"{name}'s width * height ({width_int * height_int}) must fit in a signed "
            f"32-bit int (<= {_MAX_INT32})"
        )
    return (width_int, height_int)


def _to_finite_float32(value: numbers.Real, name: str) -> float:
    """Convert `value` to the nearest `float32`, requiring the result stay finite.

    `np.float32(value)` raises a raw `OverflowError` for a Python `int`
    too large to convert at all (e.g. `10**400`), and can emit a
    `RuntimeWarning` for a large-but-representable-as-float64 value that
    overflows specifically on the narrower cast to `float32` (e.g. `1e300`)
    -- both are normalized here into a plain `ValueError` about
    non-finiteness, matching this module's convention of never letting a
    raw overflow (warning or exception) reach the caller.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            converted = np.float32(value)
    except OverflowError:
        converted = np.float32(np.inf if value > 0 else -np.inf)  # type: ignore[operator]
    if not np.isfinite(converted):
        raise ValueError(f"{name} must be representable as a finite float32 value, got {value!r}")
    return float(converted)


def _normalize_scale(value: object) -> float:
    """Raise TypeError/ValueError unless `value` is a finite (once cast to `float32`) real number.

    Zero and negative values are legal -- only `NaN` and infinity (including
    infinity produced by the `float32` cast itself) are rejected.
    """
    require_real_number(value, "scale")
    assert isinstance(value, numbers.Real)  # narrows for the type checker
    return _to_finite_float32(value, "scale")


def _normalize_mean(value: object, channels: int) -> tuple[float, ...]:
    """Raise TypeError/ValueError unless `value` is a valid scalar-or-tuple `mean`.

    A scalar is broadcast to every channel; a `tuple` must have exactly one
    element per channel -- for a single-channel (grayscale) image, that
    means a length-1 tuple is legal (and equivalent to the scalar form),
    but the same length-1 tuple is rejected for a 3- or 4-channel image,
    where a full-length tuple is required instead. `list` and
    `np.ndarray` are deliberately not accepted -- unlike raw
    `cv2.dnn.blobFromImage`, which accepts either but treats a bare scalar
    as only the *first* `cv::Scalar` element rather than broadcasting it,
    a foot-gun this function avoids by not offering that overload at all.
    """
    if isinstance(value, tuple):
        if len(value) != channels:
            raise ValueError(
                f"mean tuple must have exactly {channels} element(s) (matching the "
                f"image's channel count), got {len(value)}"
            )
        return tuple(
            _normalize_mean_element(element, f"mean[{index}]")
            for index, element in enumerate(value)
        )
    normalized = _normalize_mean_element(value, "mean")
    return (normalized,) * channels


def _normalize_mean_element(value: object, name: str) -> float:
    require_real_number(value, name)
    assert isinstance(value, numbers.Real)  # narrows for the type checker
    return _to_finite_float32(value, name)


def _check_blob_postconditions(
    result: object,
    *,
    expected_batch: int,
    expected_channels: int,
    expected_height: int,
    expected_width: int,
) -> ImageFloat32:
    """Raise RuntimeError unless `result` is a well-formed NCHW `float32` blob.

    Guards against OpenCV returning something other than what this module
    promises -- verified directly that `cv2.dnn.blobFromImage` silently
    returns `None` (no exception) for a `(0, 0, 3)` empty image on OpenCV
    4.13/5.0, which this function's own pre-validation already rejects, but
    a monkeypatched/future OpenCV misbehaving the same way for other input
    would otherwise surface as a confusing `AttributeError` deep in caller
    code instead of a clear `RuntimeError` here.
    """
    if not isinstance(result, np.ndarray):
        raise RuntimeError(
            "OpenCV DNN blob preprocessing returned "
            f"{type(result).__name__} instead of a NumPy array"
        )
    expected_shape = (expected_batch, expected_channels, expected_height, expected_width)
    if result.ndim != 4 or result.shape != expected_shape:
        raise RuntimeError(
            f"OpenCV DNN blob preprocessing returned shape {result.shape}, "
            f"expected {expected_shape}"
        )
    if result.dtype != np.float32:
        raise RuntimeError(
            f"OpenCV DNN blob preprocessing returned dtype {result.dtype}, expected float32"
        )
    if not np.all(np.isfinite(result)):
        raise RuntimeError("OpenCV DNN blob preprocessing returned a blob with NaN/Inf values")
    return result


def _normalize_onnx_path(path: object) -> str:
    """Raise TypeError/ValueError unless `path` is a valid ONNX file path, else return it as `str`.

    A single, explicit pass: `os.fspath()` (never a bare `str(path)`, which
    would silently stringify an arbitrary object instead of rejecting it),
    then exactly one `stat()` call, whose result decides between
    `IsADirectoryError`/`ValueError`(irregular file)/`ValueError`(empty
    file). `Path.stat()` follows symlinks by default, so a symlink to a
    valid regular file is accepted transparently, and a broken symlink
    raises `FileNotFoundError` the same way a missing path does --
    verified directly. `FileNotFoundError`/`PermissionError`/other `OSError`
    subclasses from `stat()` itself are allowed to propagate unmodified;
    this function does not attempt to guess every possible filesystem
    failure in advance (see `load_onnx_network`'s docstring on TOCTOU).
    """
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

    file_stat = Path(raw_path).stat()
    if stat.S_ISDIR(file_stat.st_mode):
        raise IsADirectoryError(f"path must reference a file, not a directory: {raw_path!r}")
    if not stat.S_ISREG(file_stat.st_mode):
        raise ValueError(f"path must reference a regular file, got {raw_path!r}")
    if file_stat.st_size == 0:
        raise ValueError(f"path must reference a non-empty file, got an empty file: {raw_path!r}")

    return raw_path


def _check_net_postconditions(net: object) -> cv2.dnn.Net:
    """Raise RuntimeError unless `net` is a non-empty `cv2.dnn.Net`.

    Guards against OpenCV returning something other than a working network
    -- an unexpected type, or a `Net` that loaded without raising but is
    still empty (`net.empty()` is a plain query, verified directly not to
    raise for a successfully-loaded network, but is still wrapped here in
    case a future/patched OpenCV build raises from it instead).
    """
    if not isinstance(net, cv2.dnn.Net):
        raise RuntimeError(
            f"OpenCV ONNX loading returned {type(net).__name__} instead of a cv2.dnn.Net"
        )
    try:
        is_empty = net.empty()
    except cv2.error as exc:
        raise RuntimeError("OpenCV failed to load the ONNX network") from exc
    if is_empty:
        raise RuntimeError("OpenCV loaded an empty ONNX network")
    return net
