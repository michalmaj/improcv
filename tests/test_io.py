from __future__ import annotations

import binascii
import inspect
import os
import platform
import re
import struct
import zlib
from enum import Enum
from pathlib import Path
from typing import assert_type

import cv2
import numpy as np
import pytest

import improcv as im
import improcv.io as io_module
from improcv.io import ImageReadMode, load_image, save_image
from improcv.types import Image, ImageU8

# =====================================================================================
# Fixture helpers (stdlib + cv2/numpy only -- no Pillow/piexif/imageio)
# =====================================================================================


def _write_image(path: Path, pixels: np.ndarray) -> None:
    # Path-safe write: encode in memory, then Path.write_bytes -- mirrors test_dataset.py's own
    # `_write_image` helper and load_image's own Unicode-safe read policy.
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", pixels)
    if not ok:
        raise RuntimeError(f"failed to encode test fixture {path}")
    path.write_bytes(encoded.tobytes())


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _png_bytes(pixels: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", pixels)
    assert ok
    return encoded.tobytes()


def _chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", binascii.crc32(tag + data))


def _build_indexed_png(indices: np.ndarray, palette: list[tuple[int, int, int]]) -> bytes:
    """Build a color-type-3 (palette/indexed) PNG by hand -- stdlib zlib/struct/binascii only.

    `cv2.imencode` has no option to write a palette PNG, so this constructs the PNG container
    (signature, IHDR/PLTE/IDAT/IEND chunks) directly to produce a genuine indexed-color source
    for the UNCHANGED palette-expansion contract test.
    """
    height, width = indices.shape
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 3, 0, 0, 0)  # 8-bit, color type 3 (indexed)
    plte = b"".join(struct.pack("BBB", r, g, b) for (r, g, b) in palette)
    raw = bytearray()
    for row in range(height):
        raw.append(0)  # filter type 0 (none)
        raw.extend(indices[row].astype(np.uint8).tobytes())
    idat = zlib.compress(bytes(raw), 9)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"PLTE", plte)
        + _chunk(b"IDAT", idat)
        + _chunk(b"IEND", b"")
    )


def _huge_declared_dimensions_png() -> bytes:
    """Build a tiny (~70-byte) PNG whose IHDR declares dimensions far beyond OpenCV's own
    `CV_IO_MAX_IMAGE_PIXELS` safety limit -- a real, deterministic, version-independent way to make
    `cv2.imdecode` *raise* `cv2.error` rather than return `None`. The IDAT payload itself is a
    handful of zero bytes; OpenCV's pixel-count assertion fires before any real decode work (and
    thus before any real allocation) is attempted, so this fixture is safe to construct and decode
    on every supported platform without allocating anything close to `100000 x 100000` pixels.
    """
    width, height = 100_000, 100_000
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit, color type 2 (RGB)
    idat = zlib.compress(b"\x00" * 10, 9)
    return (
        b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")
    )


def _exif_orientation_app1(orientation: int) -> bytes:
    """Build a minimal JPEG APP1/EXIF segment carrying a single Orientation tag -- stdlib only."""
    tiff_header = b"II" + struct.pack("<H", 42) + struct.pack("<I", 8)
    entry = struct.pack("<HHI", 0x0112, 3, 1) + struct.pack("<H", orientation) + b"\x00\x00"
    ifd0 = struct.pack("<H", 1) + entry + struct.pack("<I", 0)
    exif_payload = b"Exif\x00\x00" + tiff_header + ifd0
    return b"\xff\xe1" + struct.pack(">H", len(exif_payload) + 2) + exif_payload


def _jpeg_with_exif_orientation(pixels: np.ndarray, orientation: int) -> bytes:
    ok, encoded = cv2.imencode(".jpg", pixels)
    assert ok
    base = encoded.tobytes()
    assert base[:2] == b"\xff\xd8"
    return base[:2] + _exif_orientation_app1(orientation) + base[2:]


class _Unrelated(Enum):
    A = "a"


# =====================================================================================
# Exports and signature
# =====================================================================================


def test_top_level_export_is_the_same_object() -> None:
    assert im.load_image is load_image
    assert im.ImageReadMode is io_module.ImageReadMode
    assert im.save_image is save_image
    assert im.save_image is io_module.save_image


def test_module_all_contains_exactly_the_public_symbols() -> None:
    assert io_module.__all__ == ["ImageReadMode", "load_image", "save_image"]


def test_top_level_all_contains_new_symbols_without_duplicates() -> None:
    assert im.__all__.count("load_image") == 1
    assert im.__all__.count("ImageReadMode") == 1
    assert im.__all__.count("save_image") == 1
    assert len(im.__all__) == len(set(im.__all__))


def test_top_level_all_places_load_image_alphabetically() -> None:
    index = im.__all__.index("load_image")
    assert im.__all__[index - 1] == "laplacian_edge"
    assert im.__all__[index + 1] == "load_onnx_network"


def test_top_level_all_places_image_read_mode_alphabetically() -> None:
    index = im.__all__.index("ImageReadMode")
    assert im.__all__[index - 1] == "ImageMaskPair"
    assert im.__all__[index + 1] == "ImageU8"


def test_public_signature() -> None:
    signature = inspect.signature(load_image)
    parameters = signature.parameters
    assert list(parameters) == ["path", "mode"]
    assert parameters["path"].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameters["path"].default is inspect.Parameter.empty
    assert parameters["mode"].kind == inspect.Parameter.KEYWORD_ONLY
    assert parameters["mode"].default == "color"


def test_save_image_public_signature() -> None:
    signature = inspect.signature(save_image, eval_str=True)
    parameters = signature.parameters
    assert list(parameters) == ["path", "image"]
    assert parameters["path"].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameters["path"].default is inspect.Parameter.empty
    assert parameters["image"].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameters["image"].default is inspect.Parameter.empty
    assert signature.return_annotation is None


def test_top_level_all_places_save_image_alphabetically() -> None:
    index = im.__all__.index("save_image")
    assert im.__all__[index - 1] == "sample_perspective"
    assert im.__all__[index + 1] == "seamless_clone"


# =====================================================================================
# Typing regression (permanent Pyright coverage, checked via `uv run pyright tests`)
# =====================================================================================


def test_typing_default_and_explicit_modes(tmp_path: Path) -> None:
    p = tmp_path / "x.png"
    _write_image(p, np.zeros((4, 4, 3), dtype=np.uint8))

    default_result: ImageU8 = load_image(p)
    assert_type(default_result, ImageU8)

    color_result = load_image(p, mode="color")
    assert_type(color_result, ImageU8)

    grayscale_result = load_image(p, mode="grayscale")
    assert_type(grayscale_result, ImageU8)

    unchanged_result = load_image(p, mode="unchanged")
    assert_type(unchanged_result, Image)


def _load_with_runtime_mode(path: Path, mode: ImageReadMode) -> Image:
    # `mode` is a genuinely non-narrowed `ImageReadMode` here (a function parameter, never
    # reassigned to a single literal) -- this is the regression the design's own Pyright check
    # (a pre-typed *variable*) did not cover: a call site where the argument's static type is the
    # full three-member Literal union, not one member of it. Pyright must still accept this call
    # and resolve the union of the matching overloads' return types (ImageU8 | Image, which is
    # assignable to Image) via its overload-argument-type-expansion behavior.
    return load_image(path, mode=mode)


def test_typing_non_narrowed_mode_is_legal(tmp_path: Path) -> None:
    p = tmp_path / "x.png"
    _write_image(p, np.zeros((4, 4, 3), dtype=np.uint8))
    result: Image = _load_with_runtime_mode(p, "color")
    assert result is not None


def test_typing_save_image_returns_none(tmp_path: Path) -> None:
    p = tmp_path / "x.png"
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    result = save_image(p, image)
    assert_type(result, None)


# =====================================================================================
# Path contract
# =====================================================================================


def test_accepts_str_path(tmp_path: Path) -> None:
    p = tmp_path / "a.png"
    _write_image(p, np.zeros((4, 4), dtype=np.uint8))
    result = load_image(str(p))
    assert result.shape == (4, 4, 3)


def test_accepts_pathlib_path(tmp_path: Path) -> None:
    p = tmp_path / "a.png"
    _write_image(p, np.zeros((4, 4), dtype=np.uint8))
    result = load_image(p)
    assert result.shape == (4, 4, 3)


def test_accepts_custom_pathlike_returning_str(tmp_path: Path) -> None:
    p = tmp_path / "a.png"
    _write_image(p, np.zeros((4, 4), dtype=np.uint8))

    class _StrPathLike:
        def __fspath__(self) -> str:
            return str(p)

    result = load_image(_StrPathLike())
    assert result.shape == (4, 4, 3)


def test_rejects_custom_pathlike_returning_bytes(tmp_path: Path) -> None:
    p = tmp_path / "a.png"

    class _BytesPathLike:
        def __fspath__(self) -> bytes:
            return str(p).encode()

    with pytest.raises(TypeError, match="resolve to a str"):
        load_image(_BytesPathLike())  # type: ignore[arg-type]


def test_rejects_bytes_path() -> None:
    with pytest.raises(TypeError, match="resolve to a str"):
        load_image(b"a.png")  # type: ignore[arg-type]


def test_rejects_bytearray_path() -> None:
    with pytest.raises(TypeError, match="str or os.PathLike"):
        load_image(bytearray(b"a.png"))  # type: ignore[arg-type]


def test_rejects_int_path() -> None:
    with pytest.raises(TypeError, match="str or os.PathLike"):
        load_image(123)  # type: ignore[arg-type]


def test_rejects_none_path() -> None:
    with pytest.raises(TypeError, match="str or os.PathLike"):
        load_image(None)  # type: ignore[arg-type]


def test_rejects_empty_string_path() -> None:
    with pytest.raises(ValueError, match="must not be an empty string"):
        load_image("")


def test_rejects_nul_byte_path() -> None:
    # Python's own filesystem layer raises ValueError for an embedded NUL, but its exact wording
    # is platform-dependent -- verified via CI: "embedded null byte" on POSIX, "embedded null
    # character" on Windows. `load_image` does not special-case this (design doc §8), so the test
    # only pins the exception type, not platform-specific wording.
    with pytest.raises(ValueError, match="null"):
        load_image("a\x00b.png")


def test_accepts_relative_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_image(tmp_path / "a.png", np.zeros((4, 4), dtype=np.uint8))
    monkeypatch.chdir(tmp_path)
    result = load_image("a.png")
    assert result.shape == (4, 4, 3)


def test_accepts_absolute_path(tmp_path: Path) -> None:
    p = tmp_path / "a.png"
    _write_image(p, np.zeros((4, 4), dtype=np.uint8))
    assert p.is_absolute()
    result = load_image(p)
    assert result.shape == (4, 4, 3)


def test_missing_file_raises_file_not_found_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "does_not_exist.png")


def test_directory_path_raises_os_error(tmp_path: Path) -> None:
    # Which OSError subclass surfaces for a directory path is itself platform-dependent, verified
    # via CI: POSIX's Path.open() raises IsADirectoryError, but Windows has no equivalent errno and
    # raises PermissionError ([Errno 13]) instead. Both are real, native OSError subclasses from
    # Path.read_bytes() itself, propagated unwrapped -- design doc §8/§11 only promises "a native
    # OSError propagates unchanged", not one specific subclass across every platform.
    directory = tmp_path / "a_directory"
    directory.mkdir()
    with pytest.raises((IsADirectoryError, PermissionError)):
        load_image(directory)


def test_broken_symlink_raises_file_not_found_error(tmp_path: Path) -> None:
    link = tmp_path / "broken_link.png"
    try:
        link.symlink_to(tmp_path / "does_not_exist_target.png")
    except (OSError, NotImplementedError):
        pytest.skip("platform does not support creating symlinks")
    with pytest.raises(FileNotFoundError):
        load_image(link)


def test_valid_symlink_is_followed(tmp_path: Path) -> None:
    real = tmp_path / "real.png"
    _write_image(real, np.zeros((4, 4), dtype=np.uint8))
    link = tmp_path / "link.png"
    try:
        link.symlink_to(real)
    except (OSError, NotImplementedError):
        pytest.skip("platform does not support creating symlinks")
    result = load_image(link)
    assert result.shape == (4, 4, 3)


def test_propagates_synthetic_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(self: Path, *args: object, **kwargs: object) -> bytes:
        raise PermissionError("synthetic permission denial")

    monkeypatch.setattr(Path, "read_bytes", boom)

    with pytest.raises(PermissionError):
        load_image(tmp_path / "a.png")


@pytest.mark.skipif(
    platform.system() == "Windows" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="chmod-based permission denial is unreliable on Windows and as root",
)
def test_real_unreadable_file_raises_permission_error(tmp_path: Path) -> None:
    unreadable = tmp_path / "unreadable.png"
    _write_image(unreadable, np.zeros((4, 4), dtype=np.uint8))
    unreadable.chmod(0o000)
    try:
        with pytest.raises(PermissionError):
            load_image(unreadable)
    finally:
        unreadable.chmod(0o644)


def test_unicode_filename_str(tmp_path: Path) -> None:
    name = "zażółć_日本_🧪.png"
    p = tmp_path / name
    _write_image(p, np.zeros((4, 4), dtype=np.uint8))
    result = load_image(str(p))
    assert result.shape == (4, 4, 3)


def test_unicode_filename_path(tmp_path: Path) -> None:
    name = "zażółć_日本_🧪.png"
    p = tmp_path / name
    _write_image(p, np.zeros((4, 4), dtype=np.uint8))
    result = load_image(p)
    assert result.shape == (4, 4, 3)


def test_unicode_filename_custom_pathlike(tmp_path: Path) -> None:
    name = "zażółć_日本_🧪.png"
    p = tmp_path / name
    _write_image(p, np.zeros((4, 4), dtype=np.uint8))

    class _StrPathLike:
        def __fspath__(self) -> str:
            return str(p)

    result = load_image(_StrPathLike())
    assert result.shape == (4, 4, 3)


# =====================================================================================
# Mode validation
# =====================================================================================


@pytest.mark.parametrize("mode", ["color", "grayscale", "unchanged"])
def test_accepts_legal_modes(tmp_path: Path, mode: ImageReadMode) -> None:
    p = tmp_path / "a.png"
    _write_image(p, np.zeros((4, 4, 3), dtype=np.uint8))
    result = load_image(p, mode=mode)
    assert result is not None


_INVALID_RUNTIME_MODES = [
    pytest.param("bogus", id="bad_string"),
    pytest.param(1, id="int"),
    pytest.param(True, id="bool"),
    pytest.param(None, id="none"),
    pytest.param(cv2.IMREAD_COLOR, id="raw_cv2_flag"),
    # A 0-d ndarray compares equal (via __eq__) to a matching string, so it must be rejected by
    # type before any membership check runs, or it would silently be treated as a legal mode.
    pytest.param(np.array("color"), id="ndarray_scalar_matching_value"),
    # A multi-element ndarray makes `value in allowed`'s implicit bool() raise "truth value of an
    # array is ambiguous" if it ever reaches a raw membership check unguarded.
    pytest.param(np.array(["color", "x"]), id="ndarray_multi_element"),
    pytest.param(object(), id="object"),
    pytest.param(_Unrelated.A, id="unrelated_enum"),
]


@pytest.mark.parametrize("value", _INVALID_RUNTIME_MODES)
def test_rejects_invalid_runtime_mode(tmp_path: Path, value: object) -> None:
    p = tmp_path / "a.png"
    _write_image(p, np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="mode must be one of"):
        load_image(p, mode=value)  # type: ignore[arg-type]


def test_mode_validated_before_filesystem_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An invalid mode must fail fast without ever touching the filesystem, even for a path that
    # does not exist -- validation order is path type, then mode, then I/O (design doc §11).
    def boom(self: Path, *args: object, **kwargs: object) -> bytes:
        raise AssertionError("must not read the file before mode validation")

    monkeypatch.setattr(Path, "read_bytes", boom)
    with pytest.raises(ValueError, match="mode must be one of"):
        load_image(tmp_path / "does_not_exist.png", mode="bogus")  # type: ignore[arg-type]


# =====================================================================================
# Empty file / decode failure
# =====================================================================================


def test_empty_file_raises_value_error(tmp_path: Path) -> None:
    p = tmp_path / "empty.png"
    p.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        load_image(p)


def test_empty_file_never_calls_imdecode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / "empty.png"
    p.write_bytes(b"")

    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("cv2.imdecode must not be called for an empty file")

    monkeypatch.setattr(cv2, "imdecode", boom)
    with pytest.raises(ValueError):
        load_image(p)


def test_corrupt_file_raises_value_error(tmp_path: Path) -> None:
    p = tmp_path / "corrupt.png"
    p.write_bytes(b"not a real image, just garbage bytes" * 4)
    with pytest.raises(ValueError, match="failed to decode image"):
        load_image(p)


def test_truncated_jpeg_raises_value_error(tmp_path: Path) -> None:
    pixels = _rng(10).integers(0, 256, size=(20, 20, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", pixels)
    assert ok
    truncated = encoded.tobytes()[:20]
    p = tmp_path / "truncated.jpg"
    p.write_bytes(truncated)
    with pytest.raises(ValueError, match="failed to decode image"):
        load_image(p)


def test_cv2_error_during_decode_raises_value_error(tmp_path: Path) -> None:
    # cv2.imdecode does not always signal a decode failure by returning None -- it can also raise
    # cv2.error directly (e.g. OpenCV's own CV_IO_MAX_IMAGE_PIXELS safety assertion for a source
    # whose declared dimensions are absurd). Both are public decode failures and must normalize to
    # the same ValueError naming the source path, never leak a raw cv2.error.
    p = tmp_path / "huge_declared_dimensions.png"
    p.write_bytes(_huge_declared_dimensions_png())
    with pytest.raises(ValueError, match="failed to decode image"):
        load_image(p)


def test_cv2_error_during_decode_is_chained_as_cause(tmp_path: Path) -> None:
    p = tmp_path / "huge_declared_dimensions.png"
    p.write_bytes(_huge_declared_dimensions_png())
    with pytest.raises(ValueError) as exc_info:
        load_image(p)
    assert isinstance(exc_info.value.__cause__, cv2.error)


def test_valid_png_bytes_under_wrong_extension(tmp_path: Path) -> None:
    pixels = _rng(11).integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    p = tmp_path / "not_actually_a_jpeg.jpg"
    p.write_bytes(_png_bytes(pixels))
    result = load_image(p)
    assert result.shape == (8, 8, 3)


def test_valid_png_bytes_no_extension(tmp_path: Path) -> None:
    pixels = _rng(12).integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    p = tmp_path / "no_extension_at_all"
    p.write_bytes(_png_bytes(pixels))
    result = load_image(p)
    assert result.shape == (8, 8, 3)


# =====================================================================================
# Isolation: never cv2.imread, exactly one read
# =====================================================================================


def test_never_calls_cv2_imread(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / "a.png"
    _write_image(p, np.zeros((4, 4), dtype=np.uint8))

    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("load_image must never call cv2.imread")

    monkeypatch.setattr(cv2, "imread", boom)
    result = load_image(p)
    assert result.shape == (4, 4, 3)


def test_reads_file_exactly_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / "a.png"
    _write_image(p, np.zeros((4, 4), dtype=np.uint8))

    call_count = 0
    original_read_bytes = Path.read_bytes

    def spy(self: Path, *args: object, **kwargs: object) -> bytes:
        nonlocal call_count
        call_count += 1
        return original_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", spy)
    load_image(p)
    assert call_count == 1


def test_does_not_write_to_filesystem(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / "a.png"
    _write_image(p, np.zeros((4, 4), dtype=np.uint8))

    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("load_image must not write to the filesystem")

    monkeypatch.setattr(Path, "write_bytes", boom)
    monkeypatch.setattr(Path, "write_text", boom)
    load_image(p)


# =====================================================================================
# COLOR contract
# =====================================================================================


@pytest.mark.parametrize(
    ("shape", "dtype"),
    [
        ((10, 12), np.uint8),
        ((10, 12, 3), np.uint8),
        ((10, 12, 4), np.uint8),
        ((10, 12), np.uint16),
        ((10, 12, 3), np.uint16),
        ((10, 12, 4), np.uint16),
    ],
)
def test_color_always_returns_3_channel_bgr_uint8(
    tmp_path: Path, shape: tuple[int, ...], dtype: type
) -> None:
    pixels = _rng(hash(shape) & 0xFFFF).integers(0, np.iinfo(dtype).max, size=shape, dtype=dtype)
    p = tmp_path / "src.png"
    _write_image(p, pixels)

    result = load_image(p, mode="color")

    assert result.shape == (10, 12, 3)
    assert result.dtype == np.uint8


def test_color_is_default_mode(tmp_path: Path) -> None:
    pixels = _rng(20).integers(0, 256, size=(6, 6, 3), dtype=np.uint8)
    p = tmp_path / "a.png"
    _write_image(p, pixels)
    np.testing.assert_array_equal(load_image(p), load_image(p, mode="color"))


def test_color_matches_bgr_channel_order(tmp_path: Path) -> None:
    pixels = np.zeros((4, 4, 3), dtype=np.uint8)
    pixels[:, :, 0] = 10  # B
    pixels[:, :, 1] = 20  # G
    pixels[:, :, 2] = 30  # R
    p = tmp_path / "a.png"
    _write_image(p, pixels)
    result = load_image(p, mode="color")
    assert result[0, 0, 0] == 10
    assert result[0, 0, 1] == 20
    assert result[0, 0, 2] == 30


# =====================================================================================
# GRAYSCALE contract
# =====================================================================================


@pytest.mark.parametrize(
    ("shape", "dtype"),
    [
        ((10, 12), np.uint8),
        ((10, 12, 3), np.uint8),
        ((10, 12, 4), np.uint8),
        ((10, 12), np.uint16),
        ((10, 12, 3), np.uint16),
        ((10, 12, 4), np.uint16),
    ],
)
def test_grayscale_always_returns_2d_uint8(
    tmp_path: Path, shape: tuple[int, ...], dtype: type
) -> None:
    pixels = _rng(hash(shape) & 0xFFFF).integers(0, np.iinfo(dtype).max, size=shape, dtype=dtype)
    p = tmp_path / "src.png"
    _write_image(p, pixels)

    result = load_image(p, mode="grayscale")

    assert result.shape == (10, 12)
    assert result.dtype == np.uint8


def test_grayscale_not_guaranteed_equal_to_ensure_gray_of_color() -> None:
    # A genuinely color source where IMREAD_GRAYSCALE and IMREAD_COLOR + ensure_gray are known
    # (design doc §4) to be able to differ -- this test only asserts they *are able* to differ for
    # a deterministic fixture, not any specific number of differing pixels.
    pixels = _rng(99).integers(0, 256, size=(12, 12, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", pixels)
    assert ok
    encoded = np.frombuffer(buf.tobytes(), dtype=np.uint8)

    direct_grayscale = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    direct_color = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    assert direct_grayscale is not None
    assert direct_color is not None
    direct_color_then_gray = im.ensure_gray(direct_color)

    assert not np.array_equal(direct_grayscale, direct_color_then_gray), (
        "fixture must demonstrate a real difference; regenerate with a different seed if this "
        "assertion ever fails to hold for this exact array"
    )


# =====================================================================================
# UNCHANGED contract
# =====================================================================================


@pytest.mark.parametrize(
    ("shape", "dtype"),
    [
        ((9, 11), np.uint8),
        ((9, 11, 3), np.uint8),
        ((9, 11, 4), np.uint8),
        ((9, 11), np.uint16),
        ((9, 11, 3), np.uint16),
        ((9, 11, 4), np.uint16),
    ],
)
def test_unchanged_preserves_source_shape_and_dtype_exactly(
    tmp_path: Path, shape: tuple[int, ...], dtype: type
) -> None:
    pixels = _rng(hash(shape) & 0xFFFF).integers(0, np.iinfo(dtype).max, size=shape, dtype=dtype)
    p = tmp_path / "src.png"
    _write_image(p, pixels)

    result = load_image(p, mode="unchanged")

    assert result.shape == pixels.shape
    assert result.dtype == pixels.dtype
    np.testing.assert_array_equal(result, pixels)


def test_unchanged_expands_palette_png_to_bgr_uint8_not_index_map(tmp_path: Path) -> None:
    indices = np.zeros((8, 8), dtype=np.uint8)
    indices[:4, :] = 1
    indices[4:, :4] = 2
    palette = [(0, 0, 0), (255, 0, 0), (0, 255, 0)] + [(0, 0, 0)] * 253

    p = tmp_path / "indexed.png"
    p.write_bytes(_build_indexed_png(indices, palette))

    result = load_image(p, mode="unchanged")

    assert result.shape == (8, 8, 3)
    assert result.dtype == np.uint8
    # not a single-channel index map -- expanded to BGR colors from the palette
    assert result.ndim == 3 and result.shape[2] == 3


# =====================================================================================
# Direct cv2.imdecode equivalence invariant
# =====================================================================================


def test_direct_imdecode_equivalence_targeted(tmp_path: Path) -> None:
    pixels = _rng(42).integers(0, 256, size=(10, 10, 3), dtype=np.uint8)
    p = tmp_path / "a.png"
    _write_image(p, pixels)

    payload = p.read_bytes()
    buffer = np.frombuffer(payload, dtype=np.uint8)
    for mode, flag in (
        ("color", cv2.IMREAD_COLOR),
        ("grayscale", cv2.IMREAD_GRAYSCALE),
        ("unchanged", cv2.IMREAD_UNCHANGED),
    ):
        expected = cv2.imdecode(buffer, flag)
        actual = load_image(p, mode=mode)  # type: ignore[arg-type]
        np.testing.assert_array_equal(actual, expected)


def test_direct_imdecode_equivalence_differential(tmp_path: Path) -> None:
    """Property/differential test: >=250 generated PNG *source cases*, oracle is direct
    cv2.imdecode.

    Deterministic, fixed seed, no Hypothesis. Every generated (shape, dtype) PNG source case is
    exercised against all 3 modes; the oracle is always ``cv2.imdecode`` applied directly to the
    exact same in-memory encoded bytes with the corresponding flag -- never the pre-encode original
    array. JPEG/lossy formats are excluded (PNG only), per the design's own property-test scope
    (§24). The threshold is on the count of distinct *generated source cases*, not on the (larger)
    count of decode-mode comparisons derived from them.
    """
    rng = _rng(2024)
    shapes: list[tuple[int, ...]] = [
        (h, w) if c is None else (h, w, c)
        for h in (2, 3, 4, 5, 6, 7, 8)
        for w in (2, 4, 6, 8, 10, 12)
        for c in (None, 3, 4)
    ]
    dtypes: list[type] = [np.uint8, np.uint16]
    modes: tuple[ImageReadMode, ...] = ("color", "grayscale", "unchanged")
    flags = {
        "color": cv2.IMREAD_COLOR,
        "grayscale": cv2.IMREAD_GRAYSCALE,
        "unchanged": cv2.IMREAD_UNCHANGED,
    }

    generated_cases = 0
    comparisons = 0
    for index, (shape, dtype) in enumerate((shape, dtype) for shape in shapes for dtype in dtypes):
        pixels = rng.integers(0, np.iinfo(dtype).max, size=shape, dtype=dtype)
        p = tmp_path / f"case_{index}.png"
        _write_image(p, pixels)
        generated_cases += 1
        payload = p.read_bytes()
        buffer = np.frombuffer(payload, dtype=np.uint8)

        for mode in modes:
            expected = cv2.imdecode(buffer, flags[mode])
            actual = load_image(p, mode=mode)
            np.testing.assert_array_equal(actual, expected)
            comparisons += 1

    assert generated_cases >= 250, generated_cases
    print(f"generated PNG source cases: {generated_cases}")
    print(f"decode-mode comparisons: {comparisons}")


# =====================================================================================
# EXIF orientation
# =====================================================================================


def test_color_applies_exif_orientation(tmp_path: Path) -> None:
    pixels = _rng(7).integers(0, 256, size=(12, 20, 3), dtype=np.uint8)  # H=12, W=20
    p = tmp_path / "exif.jpg"
    p.write_bytes(_jpeg_with_exif_orientation(pixels, orientation=6))

    result = load_image(p, mode="color")

    assert result.shape[:2] == (20, 12)  # rotated: orientation applied


def test_grayscale_applies_exif_orientation(tmp_path: Path) -> None:
    pixels = _rng(8).integers(0, 256, size=(12, 20, 3), dtype=np.uint8)
    p = tmp_path / "exif.jpg"
    p.write_bytes(_jpeg_with_exif_orientation(pixels, orientation=6))

    result = load_image(p, mode="grayscale")

    assert result.shape[:2] == (20, 12)


def test_unchanged_does_not_apply_exif_orientation(tmp_path: Path) -> None:
    pixels = _rng(9).integers(0, 256, size=(12, 20, 3), dtype=np.uint8)
    p = tmp_path / "exif.jpg"
    p.write_bytes(_jpeg_with_exif_orientation(pixels, orientation=6))

    result = load_image(p, mode="unchanged")

    assert result.shape[:2] == (12, 20)  # not rotated: orientation ignored


# =====================================================================================
# save_image: path contract (reuses load_image's _normalize_path unchanged)
# =====================================================================================


def test_save_image_accepts_str_path(tmp_path: Path) -> None:
    p = tmp_path / "a.png"
    pixels = np.zeros((4, 4, 3), dtype=np.uint8)
    save_image(str(p), pixels)
    assert p.exists()


def test_save_image_accepts_pathlib_path(tmp_path: Path) -> None:
    p = tmp_path / "a.png"
    pixels = np.zeros((4, 4, 3), dtype=np.uint8)
    save_image(p, pixels)
    assert p.exists()


def test_save_image_accepts_custom_pathlike_returning_str(tmp_path: Path) -> None:
    p = tmp_path / "a.png"
    pixels = np.zeros((4, 4, 3), dtype=np.uint8)

    class _StrPathLike:
        def __fspath__(self) -> str:
            return str(p)

    save_image(_StrPathLike(), pixels)
    assert p.exists()


def test_save_image_rejects_custom_pathlike_returning_bytes(tmp_path: Path) -> None:
    p = tmp_path / "a.png"
    pixels = np.zeros((4, 4, 3), dtype=np.uint8)

    class _BytesPathLike:
        def __fspath__(self) -> bytes:
            return str(p).encode()

    with pytest.raises(TypeError, match="resolve to a str"):
        save_image(_BytesPathLike(), pixels)  # type: ignore[arg-type]
    assert not p.exists()


def test_save_image_rejects_bytes_path() -> None:
    pixels = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(TypeError, match="resolve to a str"):
        save_image(b"a.png", pixels)  # type: ignore[arg-type]


def test_save_image_rejects_empty_string_path() -> None:
    pixels = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="must not be an empty string"):
        save_image("", pixels)


def test_save_image_nul_byte_path_propagates_native_value_error() -> None:
    # Python's own filesystem layer raises ValueError for an embedded NUL, but its exact wording
    # is platform-dependent (e.g. "embedded null byte" on POSIX, "embedded null character" on
    # Windows). save_image does not special-case this (design doc §10), so the test only pins the
    # exception type, not platform-specific wording -- mirrors load_image's own
    # test_rejects_nul_byte_path. Uses a valid uint8 2-D image so the failure is genuinely from the
    # filesystem/path layer, not from image validation or encoding.
    pixels = np.zeros((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="null"):
        save_image("a\x00b.png", pixels)


def test_save_image_accepts_relative_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    pixels = np.zeros((4, 4, 3), dtype=np.uint8)
    save_image("a.png", pixels)
    assert (tmp_path / "a.png").exists()


def test_save_image_unicode_destination_matches_direct_imencode_oracle(tmp_path: Path) -> None:
    name = "zażółć_日本_🧪.png"
    p = tmp_path / name
    pixels = _rng(70).integers(0, 256, size=(6, 6, 3), dtype=np.uint8)

    save_image(p, pixels)

    ok, expected = cv2.imencode(p.suffix, pixels)
    assert ok
    assert p.read_bytes() == expected.tobytes()


def test_save_image_missing_parent_raises_os_error(tmp_path: Path) -> None:
    p = tmp_path / "missing_dir" / "a.png"
    pixels = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(OSError):
        save_image(p, pixels)


def test_save_image_directory_destination_raises_os_error(tmp_path: Path) -> None:
    # Named with a valid suffix so cv2.imencode succeeds and the failure is a genuine native
    # filesystem error from Path.write_bytes(), not an encode failure from an empty suffix.
    directory = tmp_path / "a_directory.png"
    directory.mkdir()
    pixels = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises((IsADirectoryError, PermissionError)):
        save_image(directory, pixels)


# =====================================================================================
# save_image: byte-level oracle (destination bytes == cv2.imencode(suffix, image))
# =====================================================================================


def _assert_byte_oracle(path: Path, image: np.ndarray) -> None:
    ok, expected = cv2.imencode(path.suffix, image)
    assert ok
    save_image(path, image)
    assert path.read_bytes() == expected.tobytes()


@pytest.mark.parametrize(
    ("shape", "dtype"),
    [
        ((10, 12), np.uint8),
        ((10, 12, 3), np.uint8),
        ((10, 12, 4), np.uint8),
        ((10, 12, 1), np.uint8),
    ],
    ids=["grayscale", "bgr", "bgra", "single_channel_3d"],
)
def test_save_image_png_byte_oracle(tmp_path: Path, shape: tuple[int, ...], dtype: type) -> None:
    pixels = _rng(hash(shape) & 0xFFFF).integers(0, 256, size=shape, dtype=dtype)
    _assert_byte_oracle(tmp_path / "out.png", pixels)


def test_save_image_jpeg_byte_oracle(tmp_path: Path) -> None:
    pixels = _rng(31).integers(0, 256, size=(20, 24, 3), dtype=np.uint8)
    _assert_byte_oracle(tmp_path / "out.jpg", pixels)


def test_save_image_jpeg_then_load_image_roundtrip_matches_shape_and_dtype(
    tmp_path: Path,
) -> None:
    # Secondary roundtrip check (legal per design) -- JPEG is lossy, so this checks structural
    # fidelity only, never pixel equality.
    pixels = _rng(32).integers(0, 256, size=(20, 24, 3), dtype=np.uint8)
    p = tmp_path / "out.jpg"
    save_image(p, pixels)
    result = load_image(p)
    assert result.shape == pixels.shape
    assert result.dtype == pixels.dtype


def test_save_image_png_roundtrip_via_load_image(tmp_path: Path) -> None:
    pixels = _rng(33).integers(0, 256, size=(10, 10, 3), dtype=np.uint8)
    p = tmp_path / "roundtrip.png"
    save_image(p, pixels)
    result = load_image(p, mode="unchanged")
    np.testing.assert_array_equal(result, pixels)


def test_save_image_uppercase_extension_byte_oracle(tmp_path: Path) -> None:
    pixels = _rng(34).integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    _assert_byte_oracle(tmp_path / "out.PNG", pixels)


def test_save_image_multi_dot_filename_byte_oracle(tmp_path: Path) -> None:
    pixels = _rng(35).integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    _assert_byte_oracle(tmp_path / "archive.v2.final.png", pixels)


# =====================================================================================
# save_image: rejected images (dtype/ndim validation, reused from _validation.py)
# =====================================================================================


def test_save_image_rejects_float32(tmp_path: Path) -> None:
    pixels = np.zeros((4, 4, 3), dtype=np.float32)
    p = tmp_path / "a.png"
    with pytest.raises(TypeError, match="dtype"):
        save_image(p, pixels)
    assert not p.exists()


def test_save_image_rejects_uint16(tmp_path: Path) -> None:
    pixels = np.zeros((4, 4, 3), dtype=np.uint16)
    p = tmp_path / "a.png"
    with pytest.raises(TypeError, match="dtype"):
        save_image(p, pixels)
    assert not p.exists()


def test_save_image_rejects_1d_array(tmp_path: Path) -> None:
    pixels = np.zeros((16,), dtype=np.uint8)
    p = tmp_path / "a.png"
    with pytest.raises(ValueError, match="dimensions"):
        save_image(p, pixels)
    assert not p.exists()


def test_save_image_rejects_0d_array(tmp_path: Path) -> None:
    pixels = np.array(5, dtype=np.uint8)
    p = tmp_path / "a.png"
    with pytest.raises(ValueError, match="dimensions"):
        save_image(p, pixels)
    assert not p.exists()


def test_save_image_rejects_empty_array(tmp_path: Path) -> None:
    pixels = np.zeros((0, 0), dtype=np.uint8)
    p = tmp_path / "a.png"
    with pytest.raises(ValueError, match="empty"):
        save_image(p, pixels)
    assert not p.exists()


def test_save_image_dtype_rejected_before_filesystem_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: object, **kwargs: object) -> int:
        raise AssertionError("must not write before dtype validation")

    monkeypatch.setattr(Path, "write_bytes", boom)
    pixels = np.zeros((4, 4, 3), dtype=np.float32)
    with pytest.raises(TypeError):
        save_image(tmp_path / "a.png", pixels)


# =====================================================================================
# save_image: encode failure mapping (cv2.error -> ValueError, ok=False -> ValueError)
# =====================================================================================


def test_save_image_unsupported_extension_raises_value_error(tmp_path: Path) -> None:
    pixels = np.zeros((4, 4, 3), dtype=np.uint8)
    p = tmp_path / "a.nosuchext"
    with pytest.raises(ValueError, match=re.escape(repr(str(p)))):
        save_image(p, pixels)
    assert not p.exists()


def test_save_image_cv2_error_is_chained_as_cause(tmp_path: Path) -> None:
    pixels = np.zeros((4, 4, 3), dtype=np.uint8)
    p = tmp_path / "a.nosuchext"
    with pytest.raises(ValueError) as exc_info:
        save_image(p, pixels)
    assert isinstance(exc_info.value.__cause__, cv2.error)


def test_save_image_invalid_channel_count_raises_value_error(tmp_path: Path) -> None:
    # 2 channels is not a layout any image codec supports -- a real cv2.error, not a synthetic one.
    pixels = np.zeros((4, 4, 2), dtype=np.uint8)
    p = tmp_path / "a.png"
    with pytest.raises(ValueError, match=re.escape(repr(str(p)))):
        save_image(p, pixels)
    assert not p.exists()


def test_save_image_synthetic_false_success_flag_raises_value_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_imencode(*args: object, **kwargs: object) -> tuple[bool, np.ndarray]:
        return False, np.zeros((1,), dtype=np.uint8)

    monkeypatch.setattr(cv2, "imencode", fake_imencode)
    pixels = np.zeros((4, 4, 3), dtype=np.uint8)
    p = tmp_path / "a.png"
    with pytest.raises(ValueError, match=re.escape(repr(str(p)))):
        save_image(p, pixels)
    assert not p.exists()


# =====================================================================================
# save_image: write-order invariant (encode fully before any filesystem write)
# =====================================================================================


def test_save_image_leaves_existing_destination_untouched_on_encode_failure(
    tmp_path: Path,
) -> None:
    p = tmp_path / "a.nosuchext"
    sentinel = b"sentinel-bytes-must-survive"
    p.write_bytes(sentinel)
    pixels = np.zeros((4, 4, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        save_image(p, pixels)

    assert p.read_bytes() == sentinel


def test_save_image_never_calls_write_bytes_on_encode_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: object, **kwargs: object) -> int:
        raise AssertionError("write_bytes must not be called when encoding fails")

    monkeypatch.setattr(Path, "write_bytes", boom)
    pixels = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        save_image(tmp_path / "a.nosuchext", pixels)


# =====================================================================================
# save_image: filesystem behavior (overwrite, native error propagation)
# =====================================================================================


def test_save_image_overwrites_existing_destination(tmp_path: Path) -> None:
    p = tmp_path / "a.png"
    old_pixels = _rng(40).integers(0, 256, size=(4, 4, 3), dtype=np.uint8)
    save_image(p, old_pixels)
    old_bytes = p.read_bytes()

    new_pixels = _rng(41).integers(0, 256, size=(6, 6, 3), dtype=np.uint8)
    save_image(p, new_pixels)

    ok, expected = cv2.imencode(".png", new_pixels)
    assert ok
    assert p.read_bytes() == expected.tobytes()
    assert p.read_bytes() != old_bytes


def test_save_image_propagates_synthetic_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(self: Path, *args: object, **kwargs: object) -> int:
        raise PermissionError("synthetic permission denial")

    monkeypatch.setattr(Path, "write_bytes", boom)
    pixels = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(PermissionError):
        save_image(tmp_path / "a.png", pixels)


# =====================================================================================
# save_image: isolation (cv2.imwrite never called; only Path.write_bytes touches disk)
# =====================================================================================


def test_save_image_never_calls_cv2_imwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("save_image must never call cv2.imwrite")

    monkeypatch.setattr(cv2, "imwrite", boom)
    pixels = _rng(50).integers(0, 256, size=(6, 6, 3), dtype=np.uint8)
    p = tmp_path / "a.png"
    save_image(p, pixels)
    assert p.exists()


def test_save_image_writes_full_encoded_bytes_via_write_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}
    original_write_bytes = Path.write_bytes

    def spy(self: Path, data: bytes, *args: object, **kwargs: object) -> int:
        captured["path"] = self
        captured["data"] = data
        return original_write_bytes(self, data)

    monkeypatch.setattr(Path, "write_bytes", spy)
    pixels = _rng(51).integers(0, 256, size=(6, 6, 3), dtype=np.uint8)
    p = tmp_path / "a.png"

    save_image(p, pixels)

    ok, expected = cv2.imencode(".png", pixels)
    assert ok
    assert captured["path"] == p
    assert captured["data"] == expected.tobytes()


# =====================================================================================
# save_image: no mutation, non-contiguous input
# =====================================================================================


def test_save_image_does_not_mutate_input(tmp_path: Path) -> None:
    pixels = _rng(60).integers(0, 256, size=(6, 6, 3), dtype=np.uint8)
    original = pixels.copy()
    save_image(tmp_path / "a.png", pixels)
    np.testing.assert_array_equal(pixels, original)


def test_save_image_accepts_non_contiguous_input_matching_direct_imencode(
    tmp_path: Path,
) -> None:
    base = _rng(61).integers(0, 256, size=(20, 20, 3), dtype=np.uint8)
    non_contiguous = base[::2, ::2, :]
    assert not non_contiguous.flags["C_CONTIGUOUS"]

    ok, expected = cv2.imencode(".png", non_contiguous)
    assert ok

    p = tmp_path / "a.png"
    save_image(p, non_contiguous)

    assert p.read_bytes() == expected.tobytes()
