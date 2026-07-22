import numpy as np
import pytest

from improcv.qrcode import _quadrangle_area, _require_valid_qr_detection


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
