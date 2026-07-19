import numpy as np
import pytest

import improcv as im


def test_in_range_marks_pixels_within_bounds() -> None:
    image = np.array([[[0, 0, 0], [100, 100, 100], [255, 255, 255]]], dtype=np.uint8)

    result = im.in_range(image, lower=(50, 50, 50), upper=(150, 150, 150))

    assert result.dtype == np.uint8
    assert result.tolist() == [[0, 255, 0]]


def test_in_range_rejects_1d_array() -> None:
    image = np.zeros(10, dtype=np.uint8)

    with pytest.raises(ValueError, match="2 or 3 dimensions"):
        im.in_range(image, lower=(0,), upper=(10,))


def test_in_range_rejects_lower_upper_length_mismatched_with_channels() -> None:
    # cv2.inRange does not broadcast a shorter "scalar" bound across
    # channels the way one might expect — verified directly against cv2
    # before deciding to reject a mismatched length rather than support it.
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="3 element"):
        im.in_range(image, lower=(0, 0), upper=(255, 255, 255))
    with pytest.raises(ValueError, match="3 element"):
        im.in_range(image, lower=(0, 0, 0), upper=(255, 255))


def test_in_range_rejects_wrong_length_for_grayscale_image() -> None:
    image = np.zeros((4, 4), dtype=np.uint8)

    with pytest.raises(ValueError, match="1 element"):
        im.in_range(image, lower=(0, 0), upper=(255, 255))


def test_in_range_rejects_non_finite_bound() -> None:
    # cv2.inRange raises a raw, low-level cv2.error for NaN bounds
    # (a dtype-mismatch assertion inside inRange) — verified directly.
    image = np.zeros((4, 4), dtype=np.uint8)

    with pytest.raises(ValueError, match="finite"):
        im.in_range(image, lower=(float("nan"),), upper=(255,))
    with pytest.raises(ValueError, match="finite"):
        im.in_range(image, lower=(0,), upper=(float("inf"),))


def test_in_range_rejects_string_bound() -> None:
    # cv2.inRange raises a raw cv2.error ("data type = <U1 is not
    # supported") for a string bound — verified directly.
    image = np.zeros((4, 4), dtype=np.uint8)

    with pytest.raises(TypeError, match="real number"):
        im.in_range(image, lower=("0",), upper=(255,))  # type: ignore[arg-type]


def test_in_range_rejects_bool_bound() -> None:
    # cv2.inRange silently accepts a bool bound, reinterpreting it as 1/0
    # instead of raising — verified directly.
    image = np.zeros((4, 4), dtype=np.uint8)

    with pytest.raises(TypeError, match="real number"):
        im.in_range(image, lower=(True,), upper=(255,))  # type: ignore[arg-type]


def test_in_range_rejects_bool_dtype() -> None:
    # cv2.inRange segfaults the interpreter outright for a bool image on
    # OpenCV 5.0 (no exception to catch) — verified directly. Raises a
    # normal cv2.error on 4.13, but since improcv supports both lines,
    # bool must be rejected unconditionally.
    image = np.zeros((4, 4), dtype=bool)

    with pytest.raises(TypeError, match="dtype"):
        im.in_range(image, lower=(0,), upper=(1,))  # type: ignore[arg-type]


def test_in_range_rejects_float16_dtype() -> None:
    # cv2.inRange segfaults the interpreter outright for a float16 image
    # on OpenCV 4.13 (no exception to catch) — verified directly. Works
    # on 5.0, but since improcv supports both lines, float16 must be
    # rejected unconditionally.
    image = np.zeros((4, 4), dtype=np.float16)

    with pytest.raises(TypeError, match="dtype"):
        im.in_range(image, lower=(0,), upper=(1,))  # type: ignore[arg-type]


def test_in_range_rejects_int64_dtype() -> None:
    # cv2.inRange silently produces a wrong mask for a large-magnitude
    # int64 image (verified directly: [-5_000_000_000, 0, 5_000_000_000]
    # bounded by [-1_000_000_000, 1_000_000_000] should mask out both
    # extremes but marks all three pixels as in-range) rather than
    # raising, so there is no safe subrange to carve out.
    image = np.zeros((4, 4), dtype=np.int64)

    with pytest.raises(TypeError, match="dtype"):
        im.in_range(image, lower=(0,), upper=(1,))  # type: ignore[arg-type]


def test_in_range_produces_correct_mask_when_bounds_match_image_float_dtype() -> None:
    # cv2.inRange silently returns an all-zero mask when a float32 image
    # is paired with float32-dtype bounds specifically (works fine when
    # the bounds happen to be float64, e.g. plain Python floats via
    # np.array()) — verified directly. in_range must not be sensitive to
    # the exact NumPy dtype of the values inside the lower/upper tuples.
    image = np.array([[10.0, 50.0, 90.0]], dtype=np.float32)

    result = im.in_range(
        image,
        lower=(np.float32(20.0),),  # type: ignore[arg-type]
        upper=(np.float32(80.0),),  # type: ignore[arg-type]
    )

    np.testing.assert_array_equal(result, np.array([[0, 255, 0]], dtype=np.uint8))


def test_in_range_accepts_fractional_bounds_for_float32_image() -> None:
    image = np.array([[0.5, 1.5, 2.5]], dtype=np.float32)

    result = im.in_range(image, lower=(1.0,), upper=(2.0,))

    np.testing.assert_array_equal(result, np.array([[0, 255, 0]], dtype=np.uint8))


def test_invert_flips_pixel_values() -> None:
    image = np.array([[0, 100, 255]], dtype=np.uint8)

    result = im.invert(image)

    np.testing.assert_array_equal(result, np.array([[255, 155, 0]], dtype=np.uint8))


def test_invert_rejects_1d_array() -> None:
    image = np.zeros(10, dtype=np.uint8)

    with pytest.raises(ValueError, match="2 or 3 dimensions"):
        im.invert(image)


def test_invert_rejects_non_uint8_dtype() -> None:
    image = np.zeros((5, 5), dtype=np.float32)

    with pytest.raises(TypeError, match="uint8"):
        im.invert(image)  # type: ignore[arg-type]


def test_adjust_brightness_increases_pixel_values() -> None:
    image = np.full((5, 5), 100, dtype=np.uint8)

    result = im.adjust_brightness(image, delta=50)

    assert result[0, 0] == 150


def test_adjust_brightness_rounds_fractional_delta_symmetrically() -> None:
    # Truncating instead of rounding was asymmetric: +0.9 was truncated
    # away to no change (100 -> 100) while -0.9 kept its full effect
    # (100 -> 99), because astype(uint8) always truncates toward zero.
    image = np.full((5, 5), 100, dtype=np.uint8)

    assert im.adjust_brightness(image, delta=0.9)[0, 0] == 101
    assert im.adjust_brightness(image, delta=-0.9)[0, 0] == 99


def test_adjust_brightness_clamps_to_255() -> None:
    image = np.full((5, 5), 240, dtype=np.uint8)

    result = im.adjust_brightness(image, delta=50)

    assert result[0, 0] == 255


def test_adjust_brightness_decreases_pixel_values() -> None:
    image = np.full((5, 5), 100, dtype=np.uint8)

    result = im.adjust_brightness(image, delta=-30)

    assert result[0, 0] == 70


def test_adjust_brightness_negative_delta_clamps_to_zero_not_abs() -> None:
    image = np.full((5, 5), 10, dtype=np.uint8)

    result = im.adjust_brightness(image, delta=-50)

    assert result[0, 0] == 0


def test_adjust_brightness_rejects_non_uint8_dtype() -> None:
    image = np.array([[0.2, 0.8]], dtype=np.float32)

    with pytest.raises(TypeError, match="uint8"):
        im.adjust_brightness(image, delta=10)  # type: ignore[arg-type]


def test_adjust_brightness_rejects_non_finite_delta() -> None:
    image = np.full((5, 5), 10, dtype=np.uint8)

    with pytest.raises(ValueError, match="finite"):
        im.adjust_brightness(image, delta=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        im.adjust_brightness(image, delta=float("inf"))


def test_adjust_brightness_rejects_numpy_nan_delta() -> None:
    # _is_nan_or_inf previously only recognized builtin float, so a NumPy
    # scalar NaN/inf silently passed through and produced a garbage result.
    image = np.full((5, 5), 10, dtype=np.uint8)

    with pytest.raises(ValueError, match="finite"):
        im.adjust_brightness(image, delta=np.float32(np.nan))  # type: ignore[arg-type]


def test_adjust_contrast_expands_values_away_from_midpoint() -> None:
    image = np.array([[200, 100]], dtype=np.uint8)

    result = im.adjust_contrast(image, factor=2.0)

    assert result[0, 0] == 255  # (200-128)*2+128 = 272 -> clamps to 255
    assert result[0, 1] == 72  # (100-128)*2+128 = 72


def test_adjust_contrast_rounds_fractional_result_instead_of_truncating() -> None:
    # (138-128)*1.09+128 = 138.9 -> should round to 139, not truncate to 138.
    image_a = np.full((5, 5), 138, dtype=np.uint8)
    assert im.adjust_contrast(image_a, factor=1.09)[0, 0] == 139

    # (118-128)*0.91+128 = 118.9 -> should round to 119, not truncate to 118.
    image_b = np.full((5, 5), 118, dtype=np.uint8)
    assert im.adjust_contrast(image_b, factor=0.91)[0, 0] == 119


def test_adjust_contrast_preserves_midpoint() -> None:
    image = np.full((5, 5), 128, dtype=np.uint8)

    result = im.adjust_contrast(image, factor=3.0)

    assert result[0, 0] == 128


def test_adjust_contrast_rejects_negative_factor() -> None:
    image = np.full((5, 5), 100, dtype=np.uint8)

    with pytest.raises(ValueError, match="non-negative"):
        im.adjust_contrast(image, factor=-1.0)


def test_adjust_contrast_rejects_numpy_infinite_factor() -> None:
    image = np.full((5, 5), 100, dtype=np.uint8)

    with pytest.raises(ValueError, match="finite"):
        im.adjust_contrast(image, factor=np.float32(np.inf))  # type: ignore[arg-type]


def test_adjust_contrast_rejects_non_uint8_dtype() -> None:
    image = np.array([[0.2, 0.8]], dtype=np.float32)

    with pytest.raises(TypeError, match="uint8"):
        im.adjust_contrast(image, factor=2.0)  # type: ignore[arg-type]


def test_alpha_blend_averages_two_images() -> None:
    image_a = np.full((5, 5), 0, dtype=np.uint8)
    image_b = np.full((5, 5), 100, dtype=np.uint8)

    result = im.alpha_blend(image_a, image_b, alpha=0.5)

    assert result[0, 0] == 50


def test_alpha_blend_rejects_mismatched_shapes() -> None:
    image_a = np.zeros((5, 5), dtype=np.uint8)
    image_b = np.zeros((5, 6), dtype=np.uint8)

    with pytest.raises(ValueError, match="same shape"):
        im.alpha_blend(image_a, image_b, alpha=0.5)


def test_alpha_blend_rejects_alpha_outside_unit_range() -> None:
    image_a = np.zeros((5, 5), dtype=np.uint8)
    image_b = np.zeros((5, 5), dtype=np.uint8)

    with pytest.raises(ValueError, match="alpha"):
        im.alpha_blend(image_a, image_b, alpha=1.5)


def test_alpha_blend_rejects_mismatched_dtype() -> None:
    image_a = np.zeros((5, 5), dtype=np.uint8)
    image_b = np.zeros((5, 5), dtype=np.float32)

    with pytest.raises(TypeError, match="same dtype"):
        im.alpha_blend(image_a, image_b, alpha=0.5)


def test_alpha_blend_rejects_int64_dtype() -> None:
    # cv2.addWeighted silently downcasts int64 to int32 instead of
    # rejecting or preserving it — verified directly against cv2.
    image_a = np.zeros((5, 5), dtype=np.int64)
    image_b = np.zeros((5, 5), dtype=np.int64)

    with pytest.raises(TypeError, match="dtype"):
        im.alpha_blend(image_a, image_b, alpha=0.5)  # type: ignore[arg-type]


def test_alpha_blend_rejects_bool_dtype() -> None:
    image_a = np.zeros((5, 5), dtype=bool)
    image_b = np.zeros((5, 5), dtype=bool)

    with pytest.raises(TypeError, match="dtype"):
        im.alpha_blend(image_a, image_b, alpha=0.5)  # type: ignore[arg-type]


def test_bitwise_and_combines_masks() -> None:
    image_a = np.array([[255, 255, 0]], dtype=np.uint8)
    image_b = np.array([[255, 0, 255]], dtype=np.uint8)

    result = im.bitwise_and(image_a, image_b)

    np.testing.assert_array_equal(result, np.array([[255, 0, 0]], dtype=np.uint8))


def test_bitwise_or_combines_masks() -> None:
    image_a = np.array([[255, 255, 0]], dtype=np.uint8)
    image_b = np.array([[255, 0, 255]], dtype=np.uint8)

    result = im.bitwise_or(image_a, image_b)

    np.testing.assert_array_equal(result, np.array([[255, 255, 255]], dtype=np.uint8))


def test_bitwise_and_rejects_mismatched_shapes() -> None:
    image_a = np.zeros((5, 5), dtype=np.uint8)
    image_b = np.zeros((5, 6), dtype=np.uint8)

    with pytest.raises(ValueError, match="same shape"):
        im.bitwise_and(image_a, image_b)


def test_bitwise_and_rejects_mismatched_dtype() -> None:
    image_a = np.zeros((5, 5), dtype=np.uint8)
    image_b = np.zeros((5, 5), dtype=np.float32)

    with pytest.raises(TypeError, match="same dtype"):
        im.bitwise_and(image_a, image_b)  # type: ignore[arg-type]


def test_bitwise_and_rejects_non_uint8_dtype() -> None:
    image_a = np.zeros((5, 5), dtype=np.float32)
    image_b = np.zeros((5, 5), dtype=np.float32)

    with pytest.raises(TypeError, match="uint8"):
        im.bitwise_and(image_a, image_b)  # type: ignore[arg-type]


def test_bitwise_or_rejects_non_uint8_dtype() -> None:
    image_a = np.zeros((5, 5), dtype=np.float32)
    image_b = np.zeros((5, 5), dtype=np.float32)

    with pytest.raises(TypeError, match="uint8"):
        im.bitwise_or(image_a, image_b)  # type: ignore[arg-type]


def test_apply_lut_remaps_pixel_values() -> None:
    image = np.array([[0, 1, 2]], dtype=np.uint8)
    table = np.arange(256, dtype=np.uint8)
    table[1] = 200

    result = im.apply_lut(image, table)

    np.testing.assert_array_equal(result, np.array([[0, 200, 2]], dtype=np.uint8))


def test_apply_lut_rejects_wrong_table_shape() -> None:
    image = np.zeros((5, 5), dtype=np.uint8)
    table = np.arange(10, dtype=np.uint8)

    with pytest.raises(ValueError, match=r"\(256,\)"):
        im.apply_lut(image, table)


def test_apply_lut_rejects_non_uint8_image_dtype() -> None:
    image = np.zeros((5, 5), dtype=np.float32)
    table = np.arange(256, dtype=np.uint8)

    with pytest.raises(TypeError, match="uint8"):
        im.apply_lut(image, table)  # type: ignore[arg-type]


def test_apply_lut_rejects_non_uint8_table_dtype() -> None:
    image = np.zeros((5, 5), dtype=np.uint8)
    table = np.full(256, -1, dtype=np.int64)

    with pytest.raises(TypeError, match="uint8"):
        im.apply_lut(image, table)
