import numpy as np
import pytest

from improcv._compat.opencv import _normalize_calc_hist_output


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
