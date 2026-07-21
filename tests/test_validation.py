import numpy as np
import pytest

from improcv._validation import (
    require_bool,
    require_channel_count,
    require_channels,
    require_dtype,
    require_finite,
    require_fits_dtype,
    require_image_ndim,
    require_int,
    require_integral,
    require_non_negative,
    require_non_negative_int,
    require_odd,
    require_one_of,
    require_point_2d,
    require_positive,
    require_positive_int,
    require_positive_integral,
    require_range,
    require_real_number,
    require_same_shape_and_dtype,
    require_size_2d,
    require_spatial_mask,
    require_transform_matrix,
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


def test_require_image_ndim_rejects_zero_channel_image() -> None:
    # (H, W, 0) has nonzero height/width but is still empty -- verified
    # directly that at least one cv2.* call returns uninitialized-memory
    # garbage for this shape rather than raising.
    with pytest.raises(ValueError, match="empty"):
        require_image_ndim(np.zeros((10, 10, 0)), ndims=(2, 3))


def test_require_real_number_accepts_python_and_numpy_numbers() -> None:
    require_real_number(1, "x")
    require_real_number(1.5, "x")
    require_real_number(np.float32(1.5), "x")
    require_real_number(np.float64(1.5), "x")
    require_real_number(np.int32(1), "x")


def test_require_real_number_rejects_bool() -> None:
    with pytest.raises(TypeError, match="real number"):
        require_real_number(True, "x")


def test_require_real_number_rejects_string() -> None:
    with pytest.raises(TypeError, match="real number"):
        require_real_number("1.5", "x")


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


def test_require_positive_rejects_numpy_nan_and_infinity() -> None:
    with pytest.raises(ValueError, match="finite"):
        require_positive(np.float32(np.nan), "gamma")
    with pytest.raises(ValueError, match="finite"):
        require_positive(np.float32(np.inf), "gamma")
    with pytest.raises(ValueError, match="finite"):
        require_positive(np.float64(np.nan), "gamma")


def test_require_positive_rejects_non_numeric() -> None:
    with pytest.raises(TypeError, match="real number"):
        require_positive("1.5", "gamma")
    with pytest.raises(TypeError, match="real number"):
        require_positive(True, "gamma")


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


def test_require_non_negative_rejects_numpy_nan_and_infinity() -> None:
    with pytest.raises(ValueError, match="finite"):
        require_non_negative(np.float32(np.nan), "factor")
    with pytest.raises(ValueError, match="finite"):
        require_non_negative(np.float64(np.inf), "factor")


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


def test_require_finite_rejects_numpy_nan_and_infinity() -> None:
    with pytest.raises(ValueError, match="finite"):
        require_finite(np.float32(np.nan), "delta")
    with pytest.raises(ValueError, match="finite"):
        require_finite(np.float64(np.inf), "delta")


def test_require_finite_rejects_non_numeric() -> None:
    with pytest.raises(TypeError, match="real number"):
        require_finite("nan", "delta")


def test_numeric_validators_reject_huge_int_without_raw_overflowerror() -> None:
    # float(10**400) raises a raw OverflowError ("int too large to convert
    # to float") -- verified directly against Python itself. Every
    # validator that internally converts a numbers.Real to float must catch
    # that and surface a clear ValueError instead, shared root cause
    # (float() overflow), not specific to any one caller.
    huge = 10**400
    with pytest.raises(ValueError, match="finite"):
        require_finite(huge, "x")
    with pytest.raises(ValueError, match="finite"):
        require_positive(huge, "x")
    with pytest.raises(ValueError, match="finite"):
        require_non_negative(huge, "x")
    with pytest.raises(ValueError, match="between"):
        require_range(huge, 0.0, 1.0, "x")
    with pytest.raises(ValueError, match="fit within the range"):
        require_fits_dtype(huge, np.uint8, "x")


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


def test_require_channels_rejects_empty_image() -> None:
    with pytest.raises(ValueError, match="empty"):
        require_channels(np.zeros((0, 10, 3)), 3)
    with pytest.raises(ValueError, match="empty"):
        require_channels(np.zeros((10, 0, 3)), 3)


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


def test_require_range_rejects_numpy_nan_and_infinity() -> None:
    with pytest.raises(ValueError, match="alpha"):
        require_range(np.float32(np.nan), 0.0, 1.0, "alpha")
    with pytest.raises(ValueError, match="alpha"):
        require_range(np.float64(np.inf), 0.0, 1.0, "alpha")


def test_require_range_rejects_non_numeric() -> None:
    with pytest.raises(TypeError, match="real number"):
        require_range("0.5", 0.0, 1.0, "alpha")


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


def test_require_fits_dtype_accepts_value_within_range() -> None:
    require_fits_dtype(255, np.uint8, "max_value")


def test_require_fits_dtype_rejects_value_exceeding_dtype_max() -> None:
    with pytest.raises(ValueError, match="max_value"):
        require_fits_dtype(300, np.uint8, "max_value")


def test_require_fits_dtype_rejects_value_below_dtype_min() -> None:
    with pytest.raises(ValueError, match="max_value"):
        require_fits_dtype(-1, np.uint8, "max_value")


def test_require_fits_dtype_skips_bound_check_for_float_dtype() -> None:
    require_fits_dtype(1e10, np.float32, "max_value")


def test_require_fits_dtype_rejects_non_real_value() -> None:
    with pytest.raises(TypeError, match="real number"):
        require_fits_dtype("300", np.uint8, "max_value")  # type: ignore[arg-type]


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


def test_require_int_accepts_int() -> None:
    require_int(3, "x")
    require_int(-3, "x")
    require_int(0, "x")


def test_require_int_rejects_bool() -> None:
    with pytest.raises(TypeError, match="int"):
        require_int(True, "x")


def test_require_int_rejects_float() -> None:
    with pytest.raises(TypeError, match="int"):
        require_int(1.5, "x")


def test_require_integral_accepts_int_and_numpy_integer() -> None:
    require_integral(3, "x")
    require_integral(-3, "x")
    require_integral(np.int32(5), "x")
    require_integral(np.int64(-7), "x")


def test_require_integral_rejects_bool() -> None:
    # bool is technically an int subclass (and therefore registers as
    # numbers.Integral too), but accepting True/False here would silently
    # misinterpret a boolean argument as 1/0 -- same reasoning as require_int.
    with pytest.raises(TypeError, match="integer"):
        require_integral(True, "x")
    with pytest.raises(TypeError, match="integer"):
        require_integral(np.bool_(True), "x")  # type: ignore[arg-type]


def test_require_integral_rejects_float() -> None:
    with pytest.raises(TypeError, match="integer"):
        require_integral(1.5, "x")
    with pytest.raises(TypeError, match="integer"):
        require_integral(np.float32(1.0), "x")  # type: ignore[arg-type]


def test_require_bool_accepts_bool() -> None:
    require_bool(True, "closed")
    require_bool(False, "closed")


def test_require_bool_rejects_non_bool() -> None:
    # OpenCV's own cv2.approxPolyDP loosely coerces 0/1/None for a bool
    # parameter — verified directly — so this must reject before that.
    with pytest.raises(TypeError, match="bool"):
        require_bool(1, "closed")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="bool"):
        require_bool(None, "closed")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="bool"):
        require_bool("yes", "closed")  # type: ignore[arg-type]


def test_require_non_negative_int_accepts_zero_and_positive() -> None:
    require_non_negative_int(0, "top")
    require_non_negative_int(5, "top")


def test_require_non_negative_int_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        require_non_negative_int(-1, "top")


def test_require_non_negative_int_rejects_non_int() -> None:
    with pytest.raises(TypeError, match="int"):
        require_non_negative_int(1.5, "top")
    with pytest.raises(TypeError, match="int"):
        require_non_negative_int(True, "top")


def test_require_size_2d_accepts_2_tuple_of_positive_ints() -> None:
    require_size_2d((5, 10), "output_size")


def test_require_size_2d_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="2-tuple"):
        require_size_2d((5,), "output_size")
    with pytest.raises(ValueError, match="2-tuple"):
        require_size_2d((5, 5, 5), "output_size")


def test_require_size_2d_rejects_non_positive_element() -> None:
    with pytest.raises(ValueError, match=r"output_size\[0\]"):
        require_size_2d((0, 5), "output_size")
    with pytest.raises(ValueError, match=r"output_size\[1\]"):
        require_size_2d((5, -5), "output_size")


def test_require_transform_matrix_accepts_valid_matrix() -> None:
    require_transform_matrix(np.eye(2, 3, dtype=np.float32), (2, 3), "matrix")
    require_transform_matrix(np.eye(3, dtype=np.float64), (3, 3), "matrix")


def test_require_transform_matrix_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match=r"\(2, 3\)"):
        require_transform_matrix(np.eye(3, dtype=np.float32), (2, 3), "matrix")


def test_require_transform_matrix_rejects_non_float_dtype() -> None:
    with pytest.raises(TypeError, match="float32"):
        require_transform_matrix(np.eye(2, 3, dtype=np.int32), (2, 3), "matrix")


def test_require_transform_matrix_rejects_non_finite_values() -> None:
    matrix = np.eye(2, 3, dtype=np.float32)
    matrix[0, 2] = np.nan

    with pytest.raises(ValueError, match="finite"):
        require_transform_matrix(matrix, (2, 3), "matrix")


def test_require_point_2d_accepts_valid_point() -> None:
    require_point_2d((1.0, 2.0), "center")
    require_point_2d((1, 2), "center")


def test_require_point_2d_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="2-tuple"):
        require_point_2d((1.0,), "center")
    with pytest.raises(ValueError, match="2-tuple"):
        require_point_2d((1.0, 2.0, 3.0), "center")


def test_require_point_2d_rejects_non_finite_element() -> None:
    with pytest.raises(ValueError, match=r"center\[0\]"):
        require_point_2d((float("nan"), 2.0), "center")
    with pytest.raises(ValueError, match=r"center\[1\]"):
        require_point_2d((1.0, float("inf")), "center")


def test_require_point_2d_rejects_non_numeric_element() -> None:
    with pytest.raises(TypeError, match="real number"):
        require_point_2d(("a", 2.0), "center")
    with pytest.raises(TypeError, match="real number"):
        require_point_2d((1.0, True), "center")


def test_require_spatial_mask_accepts_matching_uint8_2d_mask() -> None:
    image = np.zeros((10, 12, 3), dtype=np.float32)
    mask = np.zeros((10, 12), dtype=np.uint8)

    require_spatial_mask(mask, image)  # must not raise


def test_require_spatial_mask_rejects_non_uint8_dtype() -> None:
    image = np.zeros((10, 12), dtype=np.uint8)
    mask = np.zeros((10, 12), dtype=np.float32)

    with pytest.raises(TypeError, match="uint8"):
        require_spatial_mask(mask, image)  # type: ignore[arg-type]


def test_require_spatial_mask_rejects_3d_mask() -> None:
    image = np.zeros((10, 12), dtype=np.uint8)
    mask = np.zeros((10, 12, 1), dtype=np.uint8)

    with pytest.raises(ValueError, match="2 dimensions"):
        require_spatial_mask(mask, image)


def test_require_spatial_mask_rejects_mismatched_spatial_shape() -> None:
    image = np.zeros((10, 12, 3), dtype=np.float32)
    mask = np.zeros((10, 13), dtype=np.uint8)

    with pytest.raises(ValueError, match="shape"):
        require_spatial_mask(mask, image)


def test_require_spatial_mask_ignores_image_channel_count_and_dtype() -> None:
    # A BGR float32 image with a plain uint8 2D mask must be accepted --
    # the mask only constrains spatial size, never the image's own
    # channel count or dtype.
    image = np.zeros((5, 5, 3), dtype=np.float32)
    mask = np.zeros((5, 5), dtype=np.uint8)

    require_spatial_mask(mask, image)  # must not raise


def test_require_positive_integral_accepts_int_and_numpy_integer() -> None:
    require_positive_integral(5, "bins")
    require_positive_integral(np.int32(5), "bins")


def test_require_positive_integral_rejects_bool() -> None:
    with pytest.raises(TypeError, match="integer"):
        require_positive_integral(True, "bins")


def test_require_positive_integral_rejects_float() -> None:
    with pytest.raises(TypeError, match="integer"):
        require_positive_integral(5.0, "bins")


def test_require_positive_integral_rejects_zero_and_negative() -> None:
    with pytest.raises(ValueError, match="positive"):
        require_positive_integral(0, "bins")
    with pytest.raises(ValueError, match="positive"):
        require_positive_integral(-3, "bins")


def test_require_channel_count_accepts_2d_image_as_one_channel() -> None:
    channels = require_channel_count(np.zeros((5, 5)), 1, 128)

    assert channels == 1


def test_require_channel_count_accepts_within_range() -> None:
    channels = require_channel_count(np.zeros((5, 5, 128)), 1, 128)

    assert channels == 128


def test_require_channel_count_rejects_above_max() -> None:
    with pytest.raises(ValueError, match="channels"):
        require_channel_count(np.zeros((5, 5, 129)), 1, 128)


def test_require_channel_count_rejects_below_min() -> None:
    with pytest.raises(ValueError, match="channels"):
        require_channel_count(np.zeros((5, 5, 0)), 1, 128)
