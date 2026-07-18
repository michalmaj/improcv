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


def test_adjust_contrast_expands_values_away_from_midpoint() -> None:
    image = np.array([[200, 100]], dtype=np.uint8)

    result = im.adjust_contrast(image, factor=2.0)

    assert result[0, 0] == 255  # (200-128)*2+128 = 272 -> clamps to 255
    assert result[0, 1] == 72  # (100-128)*2+128 = 72


def test_adjust_contrast_preserves_midpoint() -> None:
    image = np.full((5, 5), 128, dtype=np.uint8)

    result = im.adjust_contrast(image, factor=3.0)

    assert result[0, 0] == 128


def test_adjust_contrast_rejects_negative_factor() -> None:
    image = np.full((5, 5), 100, dtype=np.uint8)

    with pytest.raises(ValueError, match="non-negative"):
        im.adjust_contrast(image, factor=-1.0)


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
