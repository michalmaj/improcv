import ast
import inspect
import os
import struct
import sys
import zlib
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
# Fixture helpers -- all stdlib + NumPy + cv2, no Pillow/piexif/imageio, nothing committed
# as a binary file.
# =====================================================================================


def _encode_png(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    assert ok, "cv2.imencode(.png) failed for a test fixture"
    return encoded.tobytes()


def _uint8_gray(rng: np.random.Generator, h: int = 6, w: int = 5) -> np.ndarray:
    return rng.integers(0, 256, size=(h, w), dtype=np.uint8)


def _uint8_bgr(rng: np.random.Generator, h: int = 6, w: int = 5) -> np.ndarray:
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def _uint8_bgra(rng: np.random.Generator, h: int = 6, w: int = 5) -> np.ndarray:
    return rng.integers(0, 256, size=(h, w, 4), dtype=np.uint8)


def _uint16_gray(rng: np.random.Generator, h: int = 6, w: int = 5) -> np.ndarray:
    return rng.integers(0, 65536, size=(h, w), dtype=np.uint16)


def _uint16_bgr(rng: np.random.Generator, h: int = 6, w: int = 5) -> np.ndarray:
    return rng.integers(0, 65536, size=(h, w, 3), dtype=np.uint16)


def _uint16_bgra(rng: np.random.Generator, h: int = 6, w: int = 5) -> np.ndarray:
    return rng.integers(0, 65536, size=(h, w, 4), dtype=np.uint16)


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )


def _make_indexed_png(indices: np.ndarray, palette_bgr: list[tuple[int, int, int]]) -> bytes:
    """Hand-build a genuine palette/indexed (PNG color type 3) PNG, stdlib zlib only.

    `indices`: `(H, W)` `uint8` array of palette indices. `palette_bgr`: list of `(B, G, R)`
    tuples (PNG's own `PLTE` chunk stores RGB; this project's own convention is BGR, so the
    reordering happens here, once).
    """
    height, width = indices.shape
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 3, 0, 0, 0)
    plte_data = b"".join(bytes((r, g, b)) for (b, g, r) in palette_bgr)
    raw = bytearray()
    for row in indices:
        raw.append(0)  # filter type 0 (None), required per scanline
        raw.extend(row.tobytes())
    idat_data = zlib.compress(bytes(raw), level=9)
    return (
        _PNG_SIGNATURE
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"PLTE", plte_data)
        + _png_chunk(b"IDAT", idat_data)
        + _png_chunk(b"IEND", b"")
    )


def _exif_orientation_app1(orientation: int) -> bytes:
    """Build a minimal APP1/Exif JPEG segment carrying a single `Orientation` IFD entry."""
    tiff_header = b"II" + struct.pack("<H", 42) + struct.pack("<I", 8)
    entry = struct.pack("<HHI", 0x0112, 3, 1) + struct.pack("<H", orientation) + b"\x00\x00"
    ifd = struct.pack("<H", 1) + entry + struct.pack("<I", 0)
    payload = b"Exif\x00\x00" + tiff_header + ifd
    return b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload


def _make_exif_jpeg(image: np.ndarray, orientation: int) -> bytes:
    """Encode `image` as JPEG via `cv2.imencode`, then inject a real `Orientation` EXIF tag.

    Stdlib-only (`struct`); no Pillow/piexif. Verified directly (scratch, not committed) that
    OpenCV applies this exact hand-built tag identically to a `PIL`/`piexif`-authored one.
    """
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok, "cv2.imencode(.jpg) failed for a test fixture"
    raw = encoded.tobytes()
    assert raw[:2] == b"\xff\xd8"
    return raw[:2] + _exif_orientation_app1(orientation) + raw[2:]


class _CustomPathLikeStr:
    def __init__(self, path: str) -> None:
        self._path = path

    def __fspath__(self) -> str:
        return self._path


class _CustomPathLikeBytes:
    def __fspath__(self) -> bytes:
        return b"whatever"


# =====================================================================================
# Exports and signature
# =====================================================================================


def test_top_level_exports_are_the_same_object() -> None:
    assert im.load_image is load_image
    assert im.ImageReadMode is ImageReadMode


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


def test_default_mode_is_color(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    path = tmp_path / "default.png"
    path.write_bytes(_encode_png(_uint8_bgr(rng)))
    default_result = load_image(path)
    explicit_result = load_image(path, mode="color")
    assert np.array_equal(default_result, explicit_result)


# =====================================================================================
# Path contract (design doc §8)
# =====================================================================================


def test_accepts_str_path(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    path = tmp_path / "a.png"
    path.write_bytes(_encode_png(_uint8_gray(rng)))
    result = load_image(str(path), mode="grayscale")
    assert result.shape == (6, 5)


def test_accepts_path_object(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    path = tmp_path / "b.png"
    path.write_bytes(_encode_png(_uint8_gray(rng)))
    result = load_image(path, mode="grayscale")
    assert result.shape == (6, 5)


def test_accepts_custom_pathlike_str(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    path = tmp_path / "c.png"
    path.write_bytes(_encode_png(_uint8_gray(rng)))
    result = load_image(_CustomPathLikeStr(str(path)), mode="grayscale")
    assert result.shape == (6, 5)


def test_rejects_custom_pathlike_bytes() -> None:
    with pytest.raises(TypeError, match="path"):
        load_image(_CustomPathLikeBytes())  # type: ignore[arg-type]


def test_rejects_bytes_path() -> None:
    with pytest.raises(TypeError, match="path"):
        load_image(b"some/path.png")  # type: ignore[arg-type]


def test_rejects_bytearray_path() -> None:
    with pytest.raises(TypeError, match="path"):
        load_image(bytearray(b"some/path.png"))  # type: ignore[arg-type]


def test_rejects_empty_string_path() -> None:
    with pytest.raises(ValueError, match="path"):
        load_image("")


def test_rejects_embedded_nul_in_path() -> None:
    with pytest.raises(ValueError, match="null byte"):
        load_image("abc\x00def.png")


def test_accepts_relative_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(5)
    (tmp_path / "rel.png").write_bytes(_encode_png(_uint8_gray(rng)))
    monkeypatch.chdir(tmp_path)
    result = load_image("rel.png", mode="grayscale")
    assert result.shape == (6, 5)


def test_accepts_absolute_path(tmp_path: Path) -> None:
    rng = np.random.default_rng(6)
    path = tmp_path / "abs.png"
    path.write_bytes(_encode_png(_uint8_gray(rng)))
    assert path.is_absolute()
    result = load_image(path, mode="grayscale")
    assert result.shape == (6, 5)


def test_missing_file_raises_file_not_found_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "does-not-exist.png")


def test_directory_path_raises_is_a_directory_error(tmp_path: Path) -> None:
    with pytest.raises(IsADirectoryError):
        load_image(tmp_path)


def test_broken_symlink_raises_file_not_found_error(tmp_path: Path) -> None:
    link = tmp_path / "broken-link.png"
    try:
        link.symlink_to(tmp_path / "nowhere.png")
    except OSError:
        pytest.skip("this platform does not permit creating symlinks")
    with pytest.raises(FileNotFoundError):
        load_image(link)


def test_valid_symlink_is_followed(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    real = tmp_path / "real.png"
    real.write_bytes(_encode_png(_uint8_gray(rng)))
    link = tmp_path / "link.png"
    try:
        link.symlink_to(real)
    except OSError:
        pytest.skip("this platform does not permit creating symlinks")
    result = load_image(link, mode="grayscale")
    assert result.shape == (6, 5)


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX permission bits don't apply")
@pytest.mark.skipif(
    os.name == "posix" and os.geteuid() == 0, reason="root bypasses permission bits"
)
def test_permission_denied_raises_permission_error(tmp_path: Path) -> None:
    rng = np.random.default_rng(8)
    path = tmp_path / "no-permission.png"
    path.write_bytes(_encode_png(_uint8_gray(rng)))
    path.chmod(0o000)
    try:
        with pytest.raises(PermissionError):
            load_image(path)
    finally:
        path.chmod(0o644)


# =====================================================================================
# Mode validation (design doc §12)
# =====================================================================================


def test_rejects_unrelated_string_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mode"):
        load_image(tmp_path / "x.png", mode="bogus")  # type: ignore[arg-type]


def test_rejects_int_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mode"):
        load_image(tmp_path / "x.png", mode=1)  # type: ignore[arg-type]


def test_rejects_raw_cv2_flag_as_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mode"):
        load_image(tmp_path / "x.png", mode=cv2.IMREAD_COLOR)  # type: ignore[arg-type]


def test_rejects_bool_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mode"):
        load_image(tmp_path / "x.png", mode=True)  # type: ignore[arg-type]


def test_rejects_none_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mode"):
        load_image(tmp_path / "x.png", mode=None)  # type: ignore[arg-type]


def test_rejects_unrelated_enum_mode(tmp_path: Path) -> None:
    from improcv.hashing import PerceptualHashAlgorithm

    with pytest.raises(ValueError, match="mode"):
        load_image(tmp_path / "x.png", mode=PerceptualHashAlgorithm.AVERAGE_HASH)  # type: ignore[arg-type]


# =====================================================================================
# Validation order (design doc §11): path first, then mode, both before any filesystem I/O
# =====================================================================================


def test_path_validation_runs_before_mode_validation() -> None:
    # An invalid path AND an invalid mode together must fail on the path, not the mode --
    # path normalization is step 1, mode validation is step 2.
    with pytest.raises(TypeError, match="path"):
        load_image(b"bad", mode="bogus")  # type: ignore[arg-type]


def test_mode_validation_runs_before_filesystem_access(tmp_path: Path) -> None:
    # A missing file combined with an invalid mode must fail on mode -- mode validation (step 2)
    # is required to happen before the read attempt (step 3).
    with pytest.raises(ValueError, match="mode"):
        load_image(tmp_path / "does-not-exist.png", mode="bogus")  # type: ignore[arg-type]


def test_validation_never_touches_the_filesystem_for_a_bad_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("must not read the filesystem for an invalid mode")

    monkeypatch.setattr(Path, "read_bytes", _boom)
    with pytest.raises(ValueError, match="mode"):
        load_image(tmp_path / "does-not-exist.png", mode="bogus")  # type: ignore[arg-type]


# =====================================================================================
# Empty file / decode failure (design doc §11)
# =====================================================================================


def test_empty_file_raises_value_error_naming_path(tmp_path: Path) -> None:
    path = tmp_path / "empty.png"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match=r"empty\.png") as exc_info:
        load_image(path)
    assert str(path) in str(exc_info.value)


def test_empty_file_never_reaches_cv2_imdecode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "empty.png"
    path.write_bytes(b"")

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("cv2.imdecode must not be called on an empty buffer")

    monkeypatch.setattr(cv2, "imdecode", _boom)
    with pytest.raises(ValueError):
        load_image(path)


def test_corrupt_file_raises_value_error_naming_path(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.png"
    path.write_bytes(b"this is not a real image file, just some bytes")
    with pytest.raises(ValueError, match=r"corrupt\.png") as exc_info:
        load_image(path)
    assert str(path) in str(exc_info.value)


def test_empty_file_and_corrupt_file_share_the_same_semantic_message(tmp_path: Path) -> None:
    empty_path = tmp_path / "empty.png"
    empty_path.write_bytes(b"")
    corrupt_path = tmp_path / "corrupt.png"
    corrupt_path.write_bytes(b"not an image")

    with pytest.raises(ValueError) as empty_exc:
        load_image(empty_path)
    with pytest.raises(ValueError) as corrupt_exc:
        load_image(corrupt_path)

    assert "failed to decode image from" in str(empty_exc.value)
    assert "failed to decode image from" in str(corrupt_exc.value)


def test_both_failure_kinds_raise_exactly_value_error_not_a_custom_type(tmp_path: Path) -> None:
    empty_path = tmp_path / "empty.png"
    empty_path.write_bytes(b"")
    with pytest.raises(ValueError) as exc_info:
        load_image(empty_path)
    assert type(exc_info.value) is ValueError


# =====================================================================================
# Exactly-one-read (design doc §23 future plan; main-instructions §27)
# =====================================================================================


def test_reads_the_file_exactly_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(9)
    path = tmp_path / "once.png"
    path.write_bytes(_encode_png(_uint8_gray(rng)))

    call_count = 0
    original_read_bytes = Path.read_bytes

    def _counting_read_bytes(self: Path, *args: object, **kwargs: object) -> bytes:
        nonlocal call_count
        call_count += 1
        return original_read_bytes(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", _counting_read_bytes)
    load_image(path, mode="grayscale")
    assert call_count == 1


# =====================================================================================
# cv2.imread prohibition (design doc §10; main-instructions §28)
# =====================================================================================


def test_source_never_references_cv2_imread_as_a_call() -> None:
    """AST-level audit: `cv2.imread` must never appear as an actual call in `io.py`'s code --
    checked by parsing, not substring search, so prose mentioning "cv2.imread" in the module's
    own docstrings/comments (which explain what NOT to do) can never produce a false positive.
    """
    source = inspect.getsource(io_module)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "imread":
            raise AssertionError("src/improcv/io.py must never reference cv2.imread")


def test_runtime_never_calls_cv2_imread(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(10)
    path = tmp_path / "no-imread.png"
    path.write_bytes(_encode_png(_uint8_bgr(rng)))

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("load_image must never call cv2.imread")

    monkeypatch.setattr(cv2, "imread", _boom)
    result = load_image(path)
    assert result.shape == (6, 5, 3)


# =====================================================================================
# COLOR contract (design doc §5; main-instructions §18)
# =====================================================================================


_SOURCE_BUILDERS = [
    ("uint8_gray", _uint8_gray, 101),
    ("uint8_bgr", _uint8_bgr, 102),
    ("uint8_bgra", _uint8_bgra, 103),
    ("uint16_gray", _uint16_gray, 104),
    ("uint16_bgr", _uint16_bgr, 105),
    ("uint16_bgra", _uint16_bgra, 106),
]


@pytest.mark.parametrize(
    "name,builder,seed", _SOURCE_BUILDERS, ids=[c[0] for c in _SOURCE_BUILDERS]
)
def test_color_mode_always_3_channel_bgr_uint8(
    tmp_path: Path, name: str, builder: object, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    source = builder(rng)  # type: ignore[operator]
    path = tmp_path / "color.png"
    path.write_bytes(_encode_png(source))

    result = load_image(path, mode="color")
    assert result.dtype == np.uint8
    assert result.ndim == 3
    assert result.shape[2] == 3


def test_color_mode_drops_alpha(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    source = _uint8_bgra(rng)
    source[..., 3] = 0  # fully transparent -- if alpha leaked through as a 4th channel this
    # would be visible in the result's shape, not its values.
    path = tmp_path / "alpha.png"
    path.write_bytes(_encode_png(source))

    result = load_image(path, mode="color")
    assert result.shape == (6, 5, 3)


def test_color_mode_downcasts_high_bit_depth(tmp_path: Path) -> None:
    rng = np.random.default_rng(12)
    source = _uint16_bgr(rng)
    path = tmp_path / "u16.png"
    path.write_bytes(_encode_png(source))

    result = load_image(path, mode="color")
    assert result.dtype == np.uint8


# =====================================================================================
# GRAYSCALE contract (design doc §5; main-instructions §19)
# =====================================================================================


@pytest.mark.parametrize(
    "name,builder,seed", _SOURCE_BUILDERS, ids=[c[0] for c in _SOURCE_BUILDERS]
)
def test_grayscale_mode_always_2d_uint8(
    tmp_path: Path, name: str, builder: object, seed: int
) -> None:
    rng = np.random.default_rng(seed + 1000)
    source = builder(rng)  # type: ignore[operator]
    path = tmp_path / "gray.png"
    path.write_bytes(_encode_png(source))

    result = load_image(path, mode="grayscale")
    assert result.dtype == np.uint8
    assert result.ndim == 2


def test_grayscale_mode_is_not_promised_equal_to_ensure_gray_of_color(tmp_path: Path) -> None:
    """Deterministic regression for the design's central §4 finding: for a genuinely color
    source, `IMREAD_GRAYSCALE` and `IMREAD_COLOR` + `ensure_gray` are two independent OpenCV code
    paths that are demonstrably allowed to differ -- confirmed for this exact fixture/seed
    (max abs diff of 1, 64/144 pixels) before being committed here, not asserted blindly.
    """
    rng = np.random.default_rng(12345)
    source = rng.integers(0, 256, size=(12, 12, 3), dtype=np.uint8)
    path = tmp_path / "divergence.png"
    path.write_bytes(_encode_png(source))

    direct_gray = load_image(path, mode="grayscale")
    via_ensure_gray = im.ensure_gray(load_image(path, mode="color"))

    assert direct_gray.shape == via_ensure_gray.shape
    assert not np.array_equal(direct_gray, via_ensure_gray), (
        "this fixture is specifically chosen to demonstrate the two paths CAN differ -- "
        "if they now match exactly, the fixture (or OpenCV's behavior) has changed"
    )


# =====================================================================================
# UNCHANGED contract (design doc §5; main-instructions §20-21)
# =====================================================================================


@pytest.mark.parametrize(
    "name,builder,seed", _SOURCE_BUILDERS, ids=[c[0] for c in _SOURCE_BUILDERS]
)
def test_unchanged_mode_matches_source_exactly(
    tmp_path: Path, name: str, builder: object, seed: int
) -> None:
    rng = np.random.default_rng(seed + 2000)
    source = builder(rng)  # type: ignore[operator]
    path = tmp_path / "unchanged.png"
    path.write_bytes(_encode_png(source))

    result = load_image(path, mode="unchanged")
    assert result.dtype == source.dtype
    assert result.shape == source.shape
    assert np.array_equal(result, source)


def test_unchanged_mode_preserves_alpha(tmp_path: Path) -> None:
    rng = np.random.default_rng(13)
    source = _uint8_bgra(rng)
    source[..., 3] = 128
    path = tmp_path / "alpha_unchanged.png"
    path.write_bytes(_encode_png(source))

    result = load_image(path, mode="unchanged")
    assert result.shape[2] == 4
    assert np.array_equal(result[..., 3], source[..., 3])


def test_unchanged_mode_preserves_16_bit_depth(tmp_path: Path) -> None:
    rng = np.random.default_rng(14)
    source = _uint16_gray(rng)
    path = tmp_path / "u16_unchanged.png"
    path.write_bytes(_encode_png(source))

    result = load_image(path, mode="unchanged")
    assert result.dtype == np.uint16
    assert np.array_equal(result, source)


def test_unchanged_mode_expands_palette_png_to_bgr(tmp_path: Path) -> None:
    palette = [(10, 20, 200), (0, 255, 0), (255, 0, 0), (128, 128, 128)]
    indices = np.array([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=np.uint8)
    path = tmp_path / "palette.png"
    path.write_bytes(_make_indexed_png(indices, palette))

    result = load_image(path, mode="unchanged")
    assert result.dtype == np.uint8
    assert result.shape == (2, 4, 3)
    expected = np.array([[palette[i] for i in row] for row in indices], dtype=np.uint8)
    assert np.array_equal(result, expected)


# =====================================================================================
# EXIF orientation (design doc §9; main-instructions §22)
# =====================================================================================


def test_color_mode_applies_exif_orientation(tmp_path: Path) -> None:
    image = np.zeros((20, 10, 3), dtype=np.uint8)  # H=20, W=10
    path = tmp_path / "exif_color.jpg"
    path.write_bytes(_make_exif_jpeg(image, orientation=6))

    result = load_image(path, mode="color")
    assert result.shape[:2] == (10, 20)  # rotated: dimensions swapped


def test_grayscale_mode_applies_exif_orientation(tmp_path: Path) -> None:
    image = np.zeros((20, 10, 3), dtype=np.uint8)
    path = tmp_path / "exif_gray.jpg"
    path.write_bytes(_make_exif_jpeg(image, orientation=6))

    result = load_image(path, mode="grayscale")
    assert result.shape[:2] == (10, 20)


def test_unchanged_mode_does_not_apply_exif_orientation(tmp_path: Path) -> None:
    image = np.zeros((20, 10, 3), dtype=np.uint8)
    path = tmp_path / "exif_unchanged.jpg"
    path.write_bytes(_make_exif_jpeg(image, orientation=6))

    result = load_image(path, mode="unchanged")
    assert result.shape[:2] == (20, 10)  # not rotated


# =====================================================================================
# Direct cv2.imdecode equivalence -- central invariant (design doc §22; main-instructions §16)
# =====================================================================================


@pytest.mark.parametrize(
    "mode,flag",
    [
        ("color", cv2.IMREAD_COLOR),
        ("grayscale", cv2.IMREAD_GRAYSCALE),
        ("unchanged", cv2.IMREAD_UNCHANGED),
    ],
)
def test_matches_direct_cv2_imdecode_exactly(
    tmp_path: Path, mode: ImageReadMode, flag: int
) -> None:
    rng = np.random.default_rng(15)
    source = _uint8_bgra(rng)
    path = tmp_path / "invariant.png"
    encoded_bytes = _encode_png(source)
    path.write_bytes(encoded_bytes)

    result = load_image(path, mode=mode)
    oracle = cv2.imdecode(np.frombuffer(encoded_bytes, dtype=np.uint8), flag)
    assert np.array_equal(result, oracle)


# =====================================================================================
# Unicode path (main-instructions §25)
# =====================================================================================


_UNICODE_NAME = "zażółć_日本_🧪.png"


def test_unicode_str_path(tmp_path: Path) -> None:
    rng = np.random.default_rng(16)
    path = tmp_path / _UNICODE_NAME
    path.write_bytes(_encode_png(_uint8_bgr(rng)))

    # The oracle is direct in-memory cv2.imdecode, never cv2.imread -- cv2.imread's
    # filename-based handling is exactly what this whole design avoids.
    oracle = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    result = load_image(str(path))
    assert np.array_equal(result, oracle)


def test_unicode_path_object(tmp_path: Path) -> None:
    rng = np.random.default_rng(17)
    path = tmp_path / _UNICODE_NAME
    path.write_bytes(_encode_png(_uint8_bgr(rng)))

    oracle = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    result = load_image(path)
    assert np.array_equal(result, oracle)


def test_unicode_custom_pathlike(tmp_path: Path) -> None:
    rng = np.random.default_rng(18)
    path = tmp_path / _UNICODE_NAME
    path.write_bytes(_encode_png(_uint8_bgr(rng)))

    oracle = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    result = load_image(_CustomPathLikeStr(str(path)))
    assert np.array_equal(result, oracle)


# =====================================================================================
# Differential/property test -- deterministic, fixed seed, no Hypothesis (design doc §24;
# main-instructions §29)
# =====================================================================================


def test_differential_against_direct_cv2_imdecode(tmp_path: Path) -> None:
    rng = np.random.default_rng(2024)
    builders = [_uint8_gray, _uint8_bgr, _uint8_bgra, _uint16_gray, _uint16_bgr, _uint16_bgra]
    cases_per_builder = 50  # 6 builders * 50 = 300 >= 250 required cases
    flags_by_mode: dict[ImageReadMode, int] = {
        "color": cv2.IMREAD_COLOR,
        "grayscale": cv2.IMREAD_GRAYSCALE,
        "unchanged": cv2.IMREAD_UNCHANGED,
    }

    total_comparisons = 0
    matched_comparisons = 0
    case_count = 0

    for builder_index, builder in enumerate(builders):
        for case_index in range(cases_per_builder):
            height = int(rng.integers(1, 6))
            width = int(rng.integers(1, 6))
            source = builder(rng, h=height, w=width)
            encoded_bytes = _encode_png(source)
            case_count += 1
            path = tmp_path / f"case_{builder_index}_{case_index}.png"
            path.write_bytes(encoded_bytes)
            buffer = np.frombuffer(encoded_bytes, dtype=np.uint8)

            for mode, flag in flags_by_mode.items():
                total_comparisons += 1
                oracle = cv2.imdecode(buffer, flag)
                result = load_image(path, mode=mode)
                if np.array_equal(result, oracle):
                    matched_comparisons += 1

    assert case_count >= 250, f"expected >= 250 generated PNG cases, got {case_count}"
    print(f"{matched_comparisons}/{total_comparisons} generated decode-mode comparisons matched")
    assert matched_comparisons == total_comparisons, (
        f"{matched_comparisons}/{total_comparisons} generated decode-mode comparisons matched"
    )


# =====================================================================================
# Typing regression (main-instructions §9, §38) -- Pyright-only static coverage.
# `typing.assert_type` is a runtime no-op (it returns its first argument unchanged), so these
# functions are never called by any test_* function above; they exist solely to be
# type-checked by `uv run pyright` (which includes `tests/`, per [tool.pyright] in
# pyproject.toml), giving a permanent static regression for load_image's overload surface.
# =====================================================================================


def _load_with_runtime_mode(path: Path, mode: ImageReadMode) -> Image:
    """`mode` here is a genuinely non-narrowed `ImageReadMode` -- not a literal string constant
    -- the regression the original overload design needed but did not have (main-instructions
    §9): this is the case that actually exercises whether every member of the `Literal` union
    remains individually assignable to `load_image`'s `mode` parameter.
    """
    return load_image(path, mode=mode)


def _static_overload_resolution_cases(path: Path) -> None:
    assert_type(load_image(path), ImageU8)
    assert_type(load_image(path, mode="color"), ImageU8)
    assert_type(load_image(path, mode="grayscale"), ImageU8)
    assert_type(load_image(path, mode="unchanged"), Image)


def _non_narrowed_mode_is_assignable_to_image(path: Path, mode: ImageReadMode) -> Image:
    # ImageReadMode -> assignable to Image: exercised by _load_with_runtime_mode's own return
    # type above; restated here as its own explicit static case for direct traceability to
    # main-instructions §9's "ImageReadMode -> assignable to Image" requirement.
    return _load_with_runtime_mode(path, mode)
