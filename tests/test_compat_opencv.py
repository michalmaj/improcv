import numpy as np
import pytest

from improcv._compat.opencv import _normalize_calc_hist_output, _normalize_hough_lines_p_output


def test_normalize_calc_hist_output_from_column_shape() -> None:
    raw = np.array([[1.0], [2.0], [3.0]])

    result = _normalize_calc_hist_output(raw, bins=3)

    assert result.shape == (3,)
    np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])


def test_normalize_calc_hist_output_from_flat_shape() -> None:
    raw = np.array([1.0, 2.0, 3.0])

    result = _normalize_calc_hist_output(raw, bins=3)

    assert result.shape == (3,)
    np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])


def test_normalize_calc_hist_output_rejects_unexpected_size() -> None:
    raw = np.array([1.0, 2.0, 3.0, 4.0])

    with pytest.raises(RuntimeError, match="size"):
        _normalize_calc_hist_output(raw, bins=3)


def test_normalize_hough_lines_p_output_passes_through_flat_shape() -> None:
    raw = np.array([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=np.int32)

    result = _normalize_hough_lines_p_output(raw)

    assert result.shape == (2, 4)
    np.testing.assert_array_equal(result, raw)


def test_normalize_hough_lines_p_output_squeezes_middle_dimension() -> None:
    raw = np.array([[[10, 20, 30, 40]], [[50, 60, 70, 80]]], dtype=np.int32)

    result = _normalize_hough_lines_p_output(raw)

    assert result.shape == (2, 4)
    np.testing.assert_array_equal(result, [[10, 20, 30, 40], [50, 60, 70, 80]])


def test_normalize_hough_lines_p_output_rejects_wrong_dtype() -> None:
    raw = np.array([[10, 20, 30, 40]], dtype=np.float32)

    with pytest.raises(RuntimeError, match="int32"):
        _normalize_hough_lines_p_output(raw)


@pytest.mark.parametrize(
    "raw",
    [
        np.zeros((2, 3), dtype=np.int32),
        np.zeros((2, 1, 3), dtype=np.int32),
        np.zeros((2, 2, 4), dtype=np.int32),
    ],
)
def test_normalize_hough_lines_p_output_rejects_wrong_field_count(raw: np.ndarray) -> None:
    with pytest.raises(RuntimeError, match="shape"):
        _normalize_hough_lines_p_output(raw)
