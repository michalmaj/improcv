import numpy as np
import pytest

from improcv._validation import (
    require_channels,
    require_dtype,
    require_finite,
    require_image_ndim,
    require_non_negative,
    require_odd,
    require_one_of,
    require_positive,
    require_positive_int,
    require_range,
    require_same_shape_and_dtype,
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


def test_require_finite_accepts_finite_value() -> None:
    require_finite(-30.0, "delta")
    require_finite(0, "delta")
    require_finite(30.0, "delta")


def test_require_finite_rejects_nan_and_infinity() -> None:
    with pytest.raises(ValueError, match="finite"):
        require_finite(float("nan"), "delta")
    with pytest.raises(ValueError, match="finite"):
        require_finite(float("inf"), "delta")
    with pytest.raises(ValueError, match="finite"):
        require_finite(float("-inf"), "delta")


def test_require_dtype_accepts_allowed_dtype() -> None:
    require_dtype(np.zeros((4, 4), dtype=np.uint8), (np.uint8,))


def test_require_dtype_accepts_any_of_multiple_allowed_dtypes() -> None:
    require_dtype(np.zeros((4, 4), dtype=np.uint16), (np.uint8, np.uint16))


def test_require_dtype_rejects_disallowed_dtype() -> None:
    with pytest.raises(TypeError, match="uint8"):
        require_dtype(np.zeros((4, 4), dtype=np.float32), (np.uint8,))


def test_require_channels_accepts_matching_channel_count() -> None:
    require_channels(np.zeros((4, 4, 3)), 3)


def test_require_channels_rejects_wrong_channel_count() -> None:
    with pytest.raises(ValueError, match="3 channels"):
        require_channels(np.zeros((4, 4)), 3)
    with pytest.raises(ValueError, match="3 channels"):
        require_channels(np.zeros((4, 4, 1)), 3)


def test_require_odd_accepts_odd_value() -> None:
    require_odd(3, "kernel_size")


def test_require_odd_rejects_even_value() -> None:
    with pytest.raises(ValueError, match="odd"):
        require_odd(4, "kernel_size")


def test_require_range_accepts_value_within_bounds() -> None:
    require_range(0.5, 0.0, 1.0, "alpha")
    require_range(0.0, 0.0, 1.0, "alpha")
    require_range(1.0, 0.0, 1.0, "alpha")


def test_require_range_rejects_value_outside_bounds() -> None:
    with pytest.raises(ValueError, match="alpha"):
        require_range(1.5, 0.0, 1.0, "alpha")
    with pytest.raises(ValueError, match="alpha"):
        require_range(-0.5, 0.0, 1.0, "alpha")


def test_require_same_shape_and_dtype_accepts_matching_images() -> None:
    require_same_shape_and_dtype(np.zeros((4, 4), dtype=np.uint8), np.zeros((4, 4), dtype=np.uint8))


def test_require_same_shape_and_dtype_rejects_mismatched_shape() -> None:
    with pytest.raises(ValueError, match="same shape"):
        require_same_shape_and_dtype(
            np.zeros((4, 4), dtype=np.uint8), np.zeros((4, 5), dtype=np.uint8)
        )


def test_require_same_shape_and_dtype_rejects_mismatched_dtype() -> None:
    with pytest.raises(TypeError, match="same dtype"):
        require_same_shape_and_dtype(
            np.zeros((4, 4), dtype=np.uint8), np.zeros((4, 4), dtype=np.float32)
        )


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
