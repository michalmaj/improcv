import numpy as np
import pytest

from improcv._validation import (
    require_channels,
    require_image_ndim,
    require_non_negative,
    require_one_of,
    require_positive,
    require_positive_int,
)


def test_require_image_ndim_accepts_allowed_ndim() -> None:
    require_image_ndim(np.zeros((4, 4)), ndims=(2, 3))
    require_image_ndim(np.zeros((4, 4, 3)), ndims=(2, 3))


def test_require_image_ndim_rejects_disallowed_ndim() -> None:
    with pytest.raises(ValueError, match="2 or 3 dimensions"):
        require_image_ndim(np.zeros(4), ndims=(2, 3))


def test_require_image_ndim_single_allowed_value_message() -> None:
    with pytest.raises(ValueError, match="2 dimensions"):
        require_image_ndim(np.zeros((4, 4, 3)), ndims=(2,))


def test_require_image_ndim_rejects_empty_image() -> None:
    with pytest.raises(ValueError, match="empty"):
        require_image_ndim(np.zeros((0, 10)))
    with pytest.raises(ValueError, match="empty"):
        require_image_ndim(np.zeros((10, 0)))


def test_require_positive_accepts_positive_value() -> None:
    require_positive(1, "width")


def test_require_positive_rejects_zero_and_negative() -> None:
    with pytest.raises(ValueError, match="positive"):
        require_positive(0, "width")
    with pytest.raises(ValueError, match="positive"):
        require_positive(-1, "width")


def test_require_positive_rejects_nan_and_infinity() -> None:
    with pytest.raises(ValueError, match="finite"):
        require_positive(float("nan"), "gamma")
    with pytest.raises(ValueError, match="finite"):
        require_positive(float("inf"), "gamma")


def test_require_non_negative_accepts_zero() -> None:
    require_non_negative(0, "top")


def test_require_non_negative_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        require_non_negative(-1, "top")


def test_require_non_negative_rejects_nan_and_infinity() -> None:
    with pytest.raises(ValueError, match="finite"):
        require_non_negative(float("nan"), "factor")
    with pytest.raises(ValueError, match="finite"):
        require_non_negative(float("inf"), "factor")


def test_require_channels_accepts_matching_channel_count() -> None:
    require_channels(np.zeros((4, 4, 3)), 3)


def test_require_channels_rejects_wrong_channel_count() -> None:
    with pytest.raises(ValueError, match="3 channels"):
        require_channels(np.zeros((4, 4)), 3)
    with pytest.raises(ValueError, match="3 channels"):
        require_channels(np.zeros((4, 4, 1)), 3)


def test_require_one_of_accepts_allowed_value() -> None:
    require_one_of("horizontal", ("horizontal", "vertical"), "direction")


def test_require_one_of_rejects_disallowed_value() -> None:
    with pytest.raises(ValueError, match="direction"):
        require_one_of("diagonal", ("horizontal", "vertical"), "direction")


def test_require_positive_int_accepts_positive_int() -> None:
    require_positive_int(3, "kernel_size")


def test_require_positive_int_rejects_zero_and_negative() -> None:
    with pytest.raises(ValueError, match="positive"):
        require_positive_int(0, "kernel_size")
    with pytest.raises(ValueError, match="positive"):
        require_positive_int(-3, "kernel_size")


def test_require_positive_int_rejects_bool() -> None:
    with pytest.raises(TypeError, match="int"):
        require_positive_int(True, "kernel_size")


def test_require_positive_int_rejects_float() -> None:
    with pytest.raises(TypeError, match="int"):
        require_positive_int(3.0, "kernel_size")


def test_require_positive_int_rejects_nan_and_infinity() -> None:
    with pytest.raises(TypeError, match="int"):
        require_positive_int(float("nan"), "kernel_size")
    with pytest.raises(TypeError, match="int"):
        require_positive_int(float("inf"), "kernel_size")
