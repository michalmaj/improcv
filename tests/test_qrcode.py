import cv2
import numpy as np
import pytest

import improcv as im
from improcv.qrcode import (
    _decode_one_quadrangle,
    _quadrangle_area,
    _require_valid_qr_detection,
)


def _make_qr_bitmap(text: str, *, corrupt: bool = False) -> np.ndarray:
    encoder = cv2.QRCodeEncoder.create()
    bitmap = encoder.encode(text)
    if corrupt:
        rng = np.random.default_rng(0)
        height, width = bitmap.shape
        for _ in range(200):
            y = rng.integers(8, height - 8)
            x = rng.integers(8, width - 8)
            bitmap[y, x] = 255 - bitmap[y, x]
    return bitmap


def _place_qr(canvas: np.ndarray, text: str, top: int, left: int, *, corrupt: bool = False) -> None:
    bitmap = _make_qr_bitmap(text, corrupt=corrupt)
    cell = 6
    size = bitmap.shape[0] * cell
    resized = cv2.resize(bitmap, (size, size), interpolation=cv2.INTER_NEAREST)
    canvas[top : top + size, left : left + size] = resized


def _single_qr_image(text: str = "A", *, corrupt: bool = False) -> np.ndarray:
    canvas = np.full((250, 250), 255, dtype=np.uint8)
    _place_qr(canvas, text, 20, 20, corrupt=corrupt)
    return canvas


def _two_qr_image() -> np.ndarray:
    canvas = np.full((250, 450), 255, dtype=np.uint8)
    _place_qr(canvas, "A", 20, 20)
    _place_qr(canvas, "B", 20, 220)
    return canvas


def _three_qr_image() -> np.ndarray:
    canvas = np.full((250, 650), 255, dtype=np.uint8)
    _place_qr(canvas, "A", 20, 20)
    _place_qr(canvas, "", 20, 220)
    _place_qr(canvas, "longer payload needed for corruption to stick", 20, 420, corrupt=True)
    return canvas


# --- decode_qr_code ---


def test_decode_qr_code_finds_known_payload() -> None:
    image = _single_qr_image("A")

    result = im.decode_qr_code(image)

    assert result is not None
    assert result.data == "A"
    assert result.points.shape == (4, 2)
    assert result.points.dtype == np.float32


def test_decode_qr_code_blank_image_returns_none() -> None:
    image = np.full((200, 200), 255, dtype=np.uint8)

    assert im.decode_qr_code(image) is None


def test_decode_qr_code_empty_payload_decodes_to_empty_string() -> None:
    image = _single_qr_image("")

    result = im.decode_qr_code(image)

    assert result is not None
    assert result.data == ""


def test_decode_qr_code_corrupted_code_decodes_to_none_data() -> None:
    image = _single_qr_image("longer payload needed for corruption to stick", corrupt=True)

    result = im.decode_qr_code(image)

    assert result is not None
    assert result.data is None


def test_decode_qr_code_returns_none_for_image_with_two_codes() -> None:
    image = _two_qr_image()

    assert im.decode_qr_code(image) is None


@pytest.mark.parametrize("channels", [3, 4])
def test_decode_qr_code_accepts_color_images(channels: int) -> None:
    gray = _single_qr_image("A")
    code = cv2.COLOR_GRAY2BGR if channels == 3 else cv2.COLOR_GRAY2BGRA
    image = cv2.cvtColor(gray, code)

    result = im.decode_qr_code(image)

    assert result is not None
    assert result.data == "A"


@pytest.mark.parametrize("channels", [2, 5])
def test_decode_qr_code_rejects_unsupported_channel_counts(channels: int) -> None:
    image = np.zeros((100, 100, channels), dtype=np.uint8)

    with pytest.raises(ValueError, match="channel"):
        im.decode_qr_code(image)


def test_decode_qr_code_rejects_non_uint8_dtype() -> None:
    image = _single_qr_image("A").astype(np.float32)

    with pytest.raises(TypeError, match="dtype"):
        im.decode_qr_code(image)  # type: ignore[arg-type]


def test_decode_qr_code_does_not_mutate_input() -> None:
    image = _single_qr_image("A")
    before = image.copy()

    im.decode_qr_code(image)

    assert np.array_equal(image, before)


def test_decode_qr_code_points_are_independent_copy() -> None:
    image = _single_qr_image("A")

    result = im.decode_qr_code(image)

    assert result is not None
    original = result.points.copy()
    result.points[:] = 0
    result2 = im.decode_qr_code(image)
    assert result2 is not None
    assert np.array_equal(result2.points, original)


def test_require_valid_qr_detection_accepts_single_detection_shape() -> None:
    points = np.zeros((1, 4, 2), dtype=np.float32)

    _require_valid_qr_detection(True, points)


# --- _decode_one_quadrangle postcondition tests ---


class _FakeDetector:
    def __init__(self, decode_return: object) -> None:
        self._decode_return = decode_return

    def decode(self, image: np.ndarray, points: np.ndarray) -> object:
        if isinstance(self._decode_return, Exception):
            raise self._decode_return
        return self._decode_return


def _quad() -> np.ndarray:
    return np.array([[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]], dtype=np.float32)


def test_decode_one_quadrangle_accepts_valid_non_empty_result() -> None:
    straight = np.zeros((21, 21), dtype=np.uint8)
    detector = _FakeDetector(("hello", straight))

    result = _decode_one_quadrangle(detector, np.zeros((50, 50), dtype=np.uint8), _quad())  # type: ignore[arg-type]

    assert result.data == "hello"
    assert np.array_equal(result.points, _quad()[0])


def test_decode_one_quadrangle_accepts_valid_empty_result() -> None:
    straight = np.zeros((21, 21), dtype=np.uint8)
    detector = _FakeDetector(("", straight))

    result = _decode_one_quadrangle(detector, np.zeros((50, 50), dtype=np.uint8), _quad())  # type: ignore[arg-type]

    assert result.data == ""


def test_decode_one_quadrangle_accepts_undecodable_result() -> None:
    detector = _FakeDetector(("", None))

    result = _decode_one_quadrangle(detector, np.zeros((50, 50), dtype=np.uint8), _quad())  # type: ignore[arg-type]

    assert result.data is None


def test_decode_one_quadrangle_rejects_non_str_decoded_text() -> None:
    detector = _FakeDetector((b"hello", None))

    with pytest.raises(RuntimeError, match="unexpected"):
        _decode_one_quadrangle(detector, np.zeros((50, 50), dtype=np.uint8), _quad())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad_straight_code",
    [
        np.zeros((21, 21), dtype=np.float32),  # wrong dtype
        np.zeros((21, 21, 1), dtype=np.uint8),  # wrong ndim
        np.zeros((0, 0), dtype=np.uint8),  # empty
    ],
)
def test_decode_one_quadrangle_rejects_bad_straight_code(bad_straight_code: np.ndarray) -> None:
    detector = _FakeDetector(("hello", bad_straight_code))

    with pytest.raises(RuntimeError, match="unexpected"):
        _decode_one_quadrangle(detector, np.zeros((50, 50), dtype=np.uint8), _quad())  # type: ignore[arg-type]


def test_decode_one_quadrangle_rejects_non_empty_text_with_no_straight_code() -> None:
    detector = _FakeDetector(("hello", None))

    with pytest.raises(RuntimeError, match="unexpected"):
        _decode_one_quadrangle(detector, np.zeros((50, 50), dtype=np.uint8), _quad())  # type: ignore[arg-type]


def test_decode_one_quadrangle_converts_unicode_decode_error() -> None:
    exc = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
    detector = _FakeDetector(exc)

    with pytest.raises(ValueError, match="UTF-8"):
        _decode_one_quadrangle(detector, np.zeros((50, 50), dtype=np.uint8), _quad())  # type: ignore[arg-type]


def test_decode_qr_code_rejects_zero_area_quadrangle(monkeypatch: pytest.MonkeyPatch) -> None:
    image = _single_qr_image("A")
    degenerate = np.zeros((1, 4, 2), dtype=np.float32)
    monkeypatch.setattr(cv2.QRCodeDetector, "detect", lambda self, img: (True, degenerate))

    with pytest.raises(RuntimeError, match="zero-area"):
        im.decode_qr_code(image)


def test_decode_qr_code_converts_unicode_decode_error(monkeypatch: pytest.MonkeyPatch) -> None:
    image = _single_qr_image("A")
    exc = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
    monkeypatch.setattr(
        cv2.QRCodeDetector, "decode", lambda self, img, points: (_ for _ in ()).throw(exc)
    )

    with pytest.raises(ValueError, match="UTF-8"):
        im.decode_qr_code(image)


# --- decode_qr_codes ---


def test_decode_qr_codes_matches_three_results_by_position() -> None:
    image = _three_qr_image()

    results = im.decode_qr_codes(image)

    assert len(results) == 3
    by_center = {float(result.points[:, 0].mean()): result for result in results}
    centers = sorted(by_center)
    assert by_center[centers[0]].data == "A"
    assert by_center[centers[1]].data == ""
    assert by_center[centers[2]].data is None


def test_decode_qr_codes_blank_image_returns_empty_list() -> None:
    image = np.full((200, 200), 255, dtype=np.uint8)

    assert im.decode_qr_codes(image) == []


@pytest.mark.parametrize("channels", [3, 4])
def test_decode_qr_codes_accepts_color_images(channels: int) -> None:
    gray = _two_qr_image()
    code = cv2.COLOR_GRAY2BGR if channels == 3 else cv2.COLOR_GRAY2BGRA
    image = cv2.cvtColor(gray, code)

    results = im.decode_qr_codes(image)

    assert len(results) == 2


@pytest.mark.parametrize("channels", [2, 5])
def test_decode_qr_codes_rejects_unsupported_channel_counts(channels: int) -> None:
    image = np.zeros((100, 100, channels), dtype=np.uint8)

    with pytest.raises(ValueError, match="channel"):
        im.decode_qr_codes(image)


def test_decode_qr_codes_rejects_non_uint8_dtype() -> None:
    image = _two_qr_image().astype(np.float32)

    with pytest.raises(TypeError, match="dtype"):
        im.decode_qr_codes(image)  # type: ignore[arg-type]


def test_decode_qr_codes_does_not_mutate_input() -> None:
    image = _two_qr_image()
    before = image.copy()

    im.decode_qr_codes(image)

    assert np.array_equal(image, before)


def test_decode_qr_codes_points_are_independent_copies() -> None:
    image = _two_qr_image()

    results = im.decode_qr_codes(image)
    originals = sorted((r.points.copy() for r in results), key=lambda p: float(p[:, 0].mean()))
    for r in results:
        r.points[:] = 0

    results2 = im.decode_qr_codes(image)
    results2_sorted = sorted(results2, key=lambda r: float(r.points[:, 0].mean()))
    for r2, orig in zip(results2_sorted, originals, strict=True):
        assert np.array_equal(r2.points, orig)


def test_decode_qr_codes_rejects_zero_area_quadrangle(monkeypatch: pytest.MonkeyPatch) -> None:
    image = _two_qr_image()
    degenerate = np.zeros((2, 4, 2), dtype=np.float32)
    monkeypatch.setattr(cv2.QRCodeDetector, "detectMulti", lambda self, img: (True, degenerate))

    with pytest.raises(RuntimeError, match="zero-area"):
        im.decode_qr_codes(image)


def test_decode_qr_codes_rejects_bad_detect_multi_result(monkeypatch: pytest.MonkeyPatch) -> None:
    image = _two_qr_image()
    monkeypatch.setattr(cv2.QRCodeDetector, "detectMulti", lambda self, img: (True, None))

    with pytest.raises(RuntimeError, match="unexpected"):
        im.decode_qr_codes(image)


# --- shared type/quadrangle tests ---


def test_quadrangle_area_of_a_square() -> None:
    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)

    assert _quadrangle_area(square) == pytest.approx(100.0)


def test_quadrangle_area_of_collapsed_points_is_zero() -> None:
    collapsed = np.array([[5, 5], [5, 5], [5, 5], [5, 5]], dtype=np.float32)

    assert _quadrangle_area(collapsed) == 0.0


def test_quadrangle_area_of_collinear_points_is_zero() -> None:
    collinear = np.array([[0, 0], [5, 0], [10, 0], [15, 0]], dtype=np.float32)

    assert _quadrangle_area(collinear) == 0.0


def test_require_valid_qr_detection_accepts_not_detected() -> None:
    _require_valid_qr_detection(False, None)


def test_require_valid_qr_detection_accepts_detected_with_valid_points() -> None:
    points = np.zeros((2, 4, 2), dtype=np.float32)

    _require_valid_qr_detection(True, points)


def test_require_valid_qr_detection_rejects_non_bool_detected() -> None:
    with pytest.raises(RuntimeError, match="bool"):
        _require_valid_qr_detection(1, None)  # type: ignore[arg-type]


def test_require_valid_qr_detection_rejects_detected_true_with_none_points() -> None:
    with pytest.raises(RuntimeError, match="unexpected"):
        _require_valid_qr_detection(True, None)


def test_require_valid_qr_detection_rejects_detected_false_with_real_points() -> None:
    points = np.zeros((1, 4, 2), dtype=np.float32)

    with pytest.raises(RuntimeError, match="unexpected"):
        _require_valid_qr_detection(False, points)


def test_require_valid_qr_detection_rejects_wrong_dtype() -> None:
    points = np.zeros((1, 4, 2), dtype=np.float64)

    with pytest.raises(RuntimeError, match="unexpected"):
        _require_valid_qr_detection(True, points)


def test_require_valid_qr_detection_rejects_wrong_shape() -> None:
    points = np.zeros((1, 3, 2), dtype=np.float32)

    with pytest.raises(RuntimeError, match="unexpected"):
        _require_valid_qr_detection(True, points)


def test_require_valid_qr_detection_rejects_empty_points_array() -> None:
    points = np.zeros((0, 4, 2), dtype=np.float32)

    with pytest.raises(RuntimeError, match="unexpected"):
        _require_valid_qr_detection(True, points)


def test_require_valid_qr_detection_rejects_non_finite_points() -> None:
    points = np.full((1, 4, 2), np.nan, dtype=np.float32)

    with pytest.raises(RuntimeError, match="non-finite"):
        _require_valid_qr_detection(True, points)
