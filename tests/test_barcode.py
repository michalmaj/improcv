import cv2
import numpy as np
import pytest

import improcv as im
from improcv.barcode import (
    _quadrangle_area,
    _require_valid_barcode_image,
    _require_valid_barcode_result,
)

# --- EAN-13/EAN-8 encoders (hand-built, no external dependency; self-verifying: if a
# constructed image decodes back to the exact expected digits/type, the tables were correct) ---

_L = {
    "0": "0001101",
    "1": "0011001",
    "2": "0010011",
    "3": "0111101",
    "4": "0100011",
    "5": "0110001",
    "6": "0101111",
    "7": "0111011",
    "8": "0110111",
    "9": "0001011",
}
_G = {
    "0": "0100111",
    "1": "0110011",
    "2": "0011011",
    "3": "0100001",
    "4": "0011101",
    "5": "0111001",
    "6": "0000101",
    "7": "0010001",
    "8": "0001001",
    "9": "0010111",
}
_R = {
    "0": "1110010",
    "1": "1100110",
    "2": "1101100",
    "3": "1000010",
    "4": "1011100",
    "5": "1001110",
    "6": "1010000",
    "7": "1000100",
    "8": "1001000",
    "9": "1110100",
}
_PARITY = {
    "0": "LLLLLL",
    "1": "LLGLGG",
    "2": "LLGGLG",
    "3": "LLGGGL",
    "4": "LGLLGG",
    "5": "LGGLLG",
    "6": "LGGGLL",
    "7": "LGLGLG",
    "8": "LGLGGL",
    "9": "LGGLGL",
}


def _ean13_checksum(digits12: str) -> str:
    total = sum(int(d) * (1 if i % 2 == 0 else 3) for i, d in enumerate(digits12))
    return str((10 - total % 10) % 10)


def _encode_ean13(digits12: str) -> tuple[str, str]:
    code = digits12 + _ean13_checksum(digits12)
    first, left, right = code[0], code[1:7], code[7:13]
    parity = _PARITY[first]
    bits = "101"
    for d, p in zip(left, parity, strict=True):
        bits += _L[d] if p == "L" else _G[d]
    bits += "01010"
    for d in right:
        bits += _R[d]
    bits += "101"
    return bits, code


def _ean8_checksum(digits7: str) -> str:
    total = sum(int(d) * (3 if i % 2 == 0 else 1) for i, d in enumerate(digits7))
    return str((10 - total % 10) % 10)


def _encode_ean8(digits7: str) -> tuple[str, str]:
    code = digits7 + _ean8_checksum(digits7)
    left, right = code[0:4], code[4:8]
    bits = "101"
    for d in left:
        bits += _L[d]
    bits += "01010"
    for d in right:
        bits += _R[d]
    bits += "101"
    return bits, code


def _bitstring_to_image(
    bits: str, module_width: int = 2, quiet_modules: int = 10, height: int = 150
) -> np.ndarray:
    quiet = quiet_modules * module_width
    width = len(bits) * module_width + 2 * quiet
    image = np.full((height, width), 255, dtype=np.uint8)
    for i, b in enumerate(bits):
        if b == "1":
            x = quiet + i * module_width
            image[10 : height - 10, x : x + module_width] = 0
    return cv2.GaussianBlur(image, (3, 3), 0)


def _corrupt_bits(bits: str, seed: int = 1, n: int = 15) -> str:
    bits_list = list(bits)
    rng = np.random.default_rng(seed)
    for idx in rng.integers(10, len(bits_list) - 10, size=n):
        bits_list[idx] = "1" if bits_list[idx] == "0" else "0"
    return "".join(bits_list)


def _ean13_image(digits12: str = "400638133393", *, corrupt: bool = False) -> np.ndarray:
    bits, _code = _encode_ean13(digits12)
    if corrupt:
        bits = _corrupt_bits(bits)
    return _bitstring_to_image(bits)


def _ean8_image(digits7: str = "1234567") -> np.ndarray:
    bits, _code = _encode_ean8(digits7)
    return _bitstring_to_image(bits)


def _place(canvas: np.ndarray, image: np.ndarray, top: int, left: int) -> None:
    canvas[top : top + image.shape[0], left : left + image.shape[1]] = image


# --- _require_valid_barcode_image ---


def test_require_valid_barcode_image_accepts_grayscale() -> None:
    _require_valid_barcode_image(_ean13_image())


def test_require_valid_barcode_image_accepts_bgr() -> None:
    _require_valid_barcode_image(cv2.cvtColor(_ean13_image(), cv2.COLOR_GRAY2BGR))


def test_require_valid_barcode_image_accepts_bgra() -> None:
    _require_valid_barcode_image(cv2.cvtColor(_ean13_image(), cv2.COLOR_GRAY2BGRA))


def test_require_valid_barcode_image_rejects_two_channels() -> None:
    with pytest.raises(ValueError, match="channel"):
        _require_valid_barcode_image(np.zeros((100, 100, 2), dtype=np.uint8))


def test_require_valid_barcode_image_rejects_non_uint8() -> None:
    with pytest.raises(TypeError, match="dtype"):
        _require_valid_barcode_image(_ean13_image().astype(np.float32))


@pytest.mark.parametrize("shape", [(40, 100), (100, 40)])
def test_require_valid_barcode_image_rejects_too_small(shape: tuple[int, int]) -> None:
    image = np.zeros(shape, dtype=np.uint8)
    with pytest.raises(ValueError, match="41x41"):
        _require_valid_barcode_image(image)


def test_require_valid_barcode_image_accepts_41x41() -> None:
    image = np.zeros((41, 41), dtype=np.uint8)
    _require_valid_barcode_image(image)


# --- _quadrangle_area ---


def test_quadrangle_area_of_a_square() -> None:
    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    assert _quadrangle_area(square) == pytest.approx(100.0)


def test_quadrangle_area_of_collapsed_points_is_zero() -> None:
    collapsed = np.array([[5, 5], [5, 5], [5, 5], [5, 5]], dtype=np.float32)
    assert _quadrangle_area(collapsed) == 0.0


def test_quadrangle_area_of_collinear_points_is_zero() -> None:
    collinear = np.array([[0, 0], [5, 0], [10, 0], [15, 0]], dtype=np.float32)
    assert _quadrangle_area(collinear) == 0.0


def test_quadrangle_area_with_large_offset_does_not_cancel_to_zero() -> None:
    offset_square = np.array(
        [[100000, 100000], [100020, 100000], [100020, 100020], [100000, 100020]],
        dtype=np.float32,
    )
    assert _quadrangle_area(offset_square) == pytest.approx(400.0)


# --- _require_valid_barcode_result: postconditions ---


def _valid_points(n: int) -> np.ndarray:
    base = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    return np.stack([base + i * 20 for i in range(n)]).astype(np.float32)


def test_require_valid_barcode_result_accepts_empty() -> None:
    assert _require_valid_barcode_result((False, (), (), None)) == []


def test_require_valid_barcode_result_rejects_wrong_length() -> None:
    with pytest.raises(RuntimeError, match="unexpected"):
        _require_valid_barcode_result((False, (), None))  # type: ignore[arg-type]


def test_require_valid_barcode_result_rejects_non_tuple() -> None:
    with pytest.raises(RuntimeError, match="unexpected"):
        _require_valid_barcode_result(None)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_retval", [1, np.bool_(True), "True", None])
def test_require_valid_barcode_result_rejects_non_bool_retval(bad_retval: object) -> None:
    with pytest.raises(RuntimeError, match="non-bool retval"):
        _require_valid_barcode_result((bad_retval, (), (), None))


def test_require_valid_barcode_result_rejects_true_retval_with_no_info() -> None:
    with pytest.raises(RuntimeError, match="unexpected"):
        _require_valid_barcode_result((True, (), (), None))


def test_require_valid_barcode_result_rejects_all_empty_with_true_retval() -> None:
    points = _valid_points(1)
    with pytest.raises(RuntimeError, match="inconsistent retval"):
        _require_valid_barcode_result((True, ("",), ("",), points))


def test_require_valid_barcode_result_rejects_some_nonempty_with_false_retval() -> None:
    points = _valid_points(1)
    with pytest.raises(RuntimeError, match="inconsistent retval"):
        _require_valid_barcode_result((False, ("123",), ("EAN_8",), points))


def test_require_valid_barcode_result_rejects_mismatched_info_type_lengths() -> None:
    points = _valid_points(1)
    with pytest.raises(RuntimeError, match="unexpected"):
        _require_valid_barcode_result((True, ("123",), (), points))


def test_require_valid_barcode_result_rejects_non_str_element() -> None:
    points = _valid_points(1)
    with pytest.raises(RuntimeError, match="unexpected"):
        _require_valid_barcode_result((True, (123,), ("EAN_8",), points))


def test_require_valid_barcode_result_rejects_inconsistent_info_type_pair() -> None:
    points = _valid_points(1)
    with pytest.raises(RuntimeError, match="unexpected"):
        _require_valid_barcode_result((True, ("123",), ("",), points))


def test_require_valid_barcode_result_rejects_bad_points_shape() -> None:
    with pytest.raises(RuntimeError, match="unexpected"):
        _require_valid_barcode_result((False, ("",), ("",), np.zeros((1, 3, 2), dtype=np.float32)))


def test_require_valid_barcode_result_rejects_points_count_mismatch() -> None:
    with pytest.raises(RuntimeError, match="unexpected"):
        _require_valid_barcode_result((False, ("", ""), ("", ""), _valid_points(1)))


def test_require_valid_barcode_result_rejects_non_finite_points() -> None:
    points = _valid_points(1)
    points[0, 0, 0] = np.nan
    with pytest.raises(RuntimeError, match="non-finite"):
        _require_valid_barcode_result((False, ("",), ("",), points))


def test_require_valid_barcode_result_rejects_degenerate_collapsed_quadrangle() -> None:
    points = np.zeros((1, 4, 2), dtype=np.float32)
    with pytest.raises(RuntimeError, match="degenerate"):
        _require_valid_barcode_result((False, ("",), ("",), points))


def test_require_valid_barcode_result_rejects_degenerate_collinear_quadrangle() -> None:
    points = np.array([[[0, 0], [5, 0], [10, 0], [15, 0]]], dtype=np.float32)
    with pytest.raises(RuntimeError, match="degenerate"):
        _require_valid_barcode_result((False, ("",), ("",), points))


def test_require_valid_barcode_result_accepts_valid_detected_undecodable() -> None:
    points = _valid_points(1)
    result = _require_valid_barcode_result((False, ("",), ("",), points))
    assert len(result) == 1
    assert result[0].data is None
    assert result[0].barcode_type is None
    assert np.array_equal(result[0].points, points[0])


def test_require_valid_barcode_result_accepts_valid_decoded() -> None:
    points = _valid_points(1)
    result = _require_valid_barcode_result((True, ("1234567890128",), ("EAN_13",), points))
    assert len(result) == 1
    assert result[0].data == "1234567890128"
    assert result[0].barcode_type == "EAN_13"
    assert np.array_equal(result[0].points, points[0])


def test_require_valid_barcode_result_points_are_independent_copies() -> None:
    points = _valid_points(1)
    result = _require_valid_barcode_result((True, ("1234567890128",), ("EAN_13",), points))
    assert not np.shares_memory(result[0].points, points)


# --- decode_barcodes ---


def test_decode_barcodes_decodes_ean13() -> None:
    canvas = np.full((170, 300), 255, dtype=np.uint8)
    image = _ean13_image("400638133393")
    _place(canvas, image, 5, 5)

    codes = im.decode_barcodes(canvas)

    assert len(codes) == 1
    assert codes[0].data == "4006381333931"
    assert codes[0].barcode_type == "EAN_13"


def test_decode_barcodes_decodes_ean8() -> None:
    canvas = np.full((170, 300), 255, dtype=np.uint8)
    image = _ean8_image("1234567")
    _place(canvas, image, 5, 5)

    codes = im.decode_barcodes(canvas)

    assert len(codes) == 1
    assert codes[0].data == "12345670"
    assert codes[0].barcode_type == "EAN_8"


def test_decode_barcodes_decodes_upc_a_from_leading_zero_ean13() -> None:
    canvas = np.full((170, 300), 255, dtype=np.uint8)
    image = _ean13_image("012345678905")
    _place(canvas, image, 5, 5)

    codes = im.decode_barcodes(canvas)

    assert len(codes) == 1
    assert codes[0].data == "123456789050"
    assert codes[0].barcode_type == "UPC_A"


def test_decode_barcodes_blank_image_returns_empty() -> None:
    image = np.full((200, 200), 255, dtype=np.uint8)

    assert im.decode_barcodes(image) == []


def test_decode_barcodes_matches_three_results_by_position() -> None:
    img_ean13 = _ean13_image("400638133393")
    img_upca = _ean13_image("012345678905")
    img_corrupt = _ean13_image("400638133393", corrupt=True)

    width = img_ean13.shape[1] + img_upca.shape[1] + img_corrupt.shape[1] + 80
    canvas = np.full((170, width), 255, dtype=np.uint8)
    x = 20
    _place(canvas, img_ean13, 0, x)
    x += img_ean13.shape[1] + 20
    _place(canvas, img_upca, 0, x)
    x += img_upca.shape[1] + 20
    _place(canvas, img_corrupt, 0, x)

    codes = im.decode_barcodes(canvas)

    assert len(codes) == 3
    by_center = {float(code.points[:, 0].mean()): code for code in codes}
    centers = sorted(by_center)
    results_by_center_order = [by_center[c] for c in centers]
    data_values = {code.data for code in results_by_center_order}
    assert "4006381333931" in data_values
    assert "123456789050" in data_values
    assert None in data_values
    types_for_data = {code.data: code.barcode_type for code in results_by_center_order}
    assert types_for_data["4006381333931"] == "EAN_13"
    assert types_for_data["123456789050"] == "UPC_A"
    assert types_for_data[None] is None


def test_decode_barcodes_returns_undecodable_entries_when_all_fail() -> None:
    img1 = _ean13_image("400638133393", corrupt=True)
    img2 = _ean13_image("012345678905", corrupt=True)

    width = img1.shape[1] + img2.shape[1] + 60
    canvas = np.full((170, width), 255, dtype=np.uint8)
    _place(canvas, img1, 0, 20)
    _place(canvas, img2, 0, img1.shape[1] + 40)

    codes = im.decode_barcodes(canvas)

    assert len(codes) == 2
    assert all(code.data is None and code.barcode_type is None for code in codes)


def test_decode_barcodes_rejects_two_channels() -> None:
    with pytest.raises(ValueError, match="channel"):
        im.decode_barcodes(np.zeros((100, 100, 2), dtype=np.uint8))


def test_decode_barcodes_rejects_non_uint8() -> None:
    with pytest.raises(TypeError, match="dtype"):
        im.decode_barcodes(_ean13_image().astype(np.float32))  # type: ignore[arg-type]


@pytest.mark.parametrize("shape", [(40, 100), (100, 40)])
def test_decode_barcodes_rejects_too_small_image(shape: tuple[int, int]) -> None:
    image = np.zeros(shape, dtype=np.uint8)
    with pytest.raises(ValueError, match="41x41"):
        im.decode_barcodes(image)


def test_decode_barcodes_does_not_mutate_input() -> None:
    canvas = np.full((170, 300), 255, dtype=np.uint8)
    _place(canvas, _ean13_image(), 5, 5)
    before = canvas.copy()

    im.decode_barcodes(canvas)

    assert np.array_equal(canvas, before)


def test_decode_barcodes_points_are_independent_copies() -> None:
    canvas = np.full((170, 300), 255, dtype=np.uint8)
    _place(canvas, _ean13_image(), 5, 5)

    codes = im.decode_barcodes(canvas)

    assert len(codes) == 1
    original = codes[0].points.copy()
    codes[0].points[:] = 0
    codes2 = im.decode_barcodes(canvas)
    assert np.array_equal(codes2[0].points, original)


def test_decode_barcodes_converts_unicode_decode_error(monkeypatch: pytest.MonkeyPatch) -> None:
    canvas = np.full((170, 300), 255, dtype=np.uint8)
    exc = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
    monkeypatch.setattr(
        cv2.barcode.BarcodeDetector,
        "detectAndDecodeWithType",
        lambda self, img: (_ for _ in ()).throw(exc),
    )

    with pytest.raises(ValueError, match="UTF-8"):
        im.decode_barcodes(canvas)
