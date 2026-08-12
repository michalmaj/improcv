from __future__ import annotations

import binascii
import inspect
import os
import platform
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
from improcv.io import ImageReadMode, load_image
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


def test_module_all_contains_exactly_the_public_symbols() -> None:
    assert io_module.__all__ == ["ImageReadMode", "load_image"]


def test_top_level_all_contains_new_symbols_without_duplicates() -> None:
    assert im.__all__.count("load_image") == 1
    assert im.__all__.count("ImageReadMode") == 1
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
    with pytest.raises(ValueError, match="null byte"):
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


def test_directory_path_raises_is_a_directory_error(tmp_path: Path) -> None:
    directory = tmp_path / "a_directory"
    directory.mkdir()
    with pytest.raises(IsADirectoryError):
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


def test_rejects_bad_mode_string(tmp_path: Path) -> None:
    p = tmp_path / "a.png"
    _write_image(p, np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="mode must be one of"):
        load_image(p, mode="bogus")  # type: ignore[arg-type]


def test_rejects_raw_cv2_flag_as_mode(tmp_path: Path) -> None:
    p = tmp_path / "a.png"
    _write_image(p, np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="mode must be one of"):
        load_image(p, mode=cv2.IMREAD_COLOR)  # type: ignore[arg-type]


def test_rejects_int_mode(tmp_path: Path) -> None:
    p = tmp_path / "a.png"
    _write_image(p, np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="mode must be one of"):
        load_image(p, mode=1)  # type: ignore[arg-type]


def test_rejects_bool_mode(tmp_path: Path) -> None:
    p = tmp_path / "a.png"
    _write_image(p, np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="mode must be one of"):
        load_image(p, mode=True)  # type: ignore[arg-type]


def test_rejects_none_mode(tmp_path: Path) -> None:
    p = tmp_path / "a.png"
    _write_image(p, np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="mode must be one of"):
        load_image(p, mode=None)  # type: ignore[arg-type]


def test_rejects_unrelated_enum_as_mode(tmp_path: Path) -> None:
    p = tmp_path / "a.png"
    _write_image(p, np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="mode must be one of"):
        load_image(p, mode=_Unrelated.A)  # type: ignore[arg-type]


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
    """Property/differential test: >=250 generated PNG cases, oracle is direct cv2.imdecode.

    Deterministic, fixed seed, no Hypothesis. Every legal (dtype, channel) combination is
    exercised against every mode where that combination is decodable; the oracle is always
    ``cv2.imdecode`` applied directly to the exact same in-memory encoded bytes with the
    corresponding flag -- never the pre-encode original array. JPEG/lossy formats are excluded
    (PNG only), per the design's own property-test scope (§24).
    """
    rng = _rng(2024)
    shapes: list[tuple[int, ...]] = [
        (h, w) if c is None else (h, w, c)
        for h in (3, 6, 9, 16)
        for w in (3, 8, 13, 16)
        for c in (None, 3, 4)
    ]
    dtypes: list[type] = [np.uint8, np.uint16]
    modes: tuple[ImageReadMode, ...] = ("color", "grayscale", "unchanged")

    comparisons = 0
    for index, (shape, dtype) in enumerate((shape, dtype) for shape in shapes for dtype in dtypes):
        pixels = rng.integers(0, np.iinfo(dtype).max, size=shape, dtype=dtype)
        p = tmp_path / f"case_{index}.png"
        _write_image(p, pixels)
        payload = p.read_bytes()
        buffer = np.frombuffer(payload, dtype=np.uint8)

        for mode in modes:
            flag = {
                "color": cv2.IMREAD_COLOR,
                "grayscale": cv2.IMREAD_GRAYSCALE,
                "unchanged": cv2.IMREAD_UNCHANGED,
            }[mode]
            expected = cv2.imdecode(buffer, flag)
            actual = load_image(p, mode=mode)
            np.testing.assert_array_equal(actual, expected)
            comparisons += 1

    assert comparisons >= 250, comparisons
    print(f"{comparisons}/{comparisons} generated decode-mode comparisons matched")


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
