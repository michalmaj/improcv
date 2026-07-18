import cv2
import numpy as np
import pytest

import improcv as im


def _make_image(height: int, width: int, channels: int | None = 3) -> np.ndarray:
    shape = (height, width) if channels is None else (height, width, channels)
    return (np.arange(int(np.prod(shape))) % 256).astype(np.uint8).reshape(shape)


def test_resize_by_width_preserves_aspect_ratio() -> None:
    image = _make_image(100, 200)

    result = im.resize(image, width=100)

    assert result.shape == (50, 100, 3)


def test_resize_by_height_preserves_aspect_ratio() -> None:
    image = _make_image(100, 200)

    result = im.resize(image, height=25)

    assert result.shape == (25, 50, 3)


def test_resize_with_both_dimensions_ignores_aspect_ratio() -> None:
    image = _make_image(100, 200)

    result = im.resize(image, width=10, height=10)

    assert result.shape == (10, 10, 3)


def test_resize_without_dimensions_raises_value_error() -> None:
    image = _make_image(100, 200)

    with pytest.raises(ValueError, match="width.*height"):
        im.resize(image)


@pytest.mark.parametrize("width, height", [(0, None), (-5, None), (None, 0), (None, -5)])
def test_resize_with_non_positive_dimension_raises_value_error(
    width: int | None, height: int | None
) -> None:
    image = _make_image(100, 200)

    with pytest.raises(ValueError, match="positive"):
        im.resize(image, width=width, height=height)


def test_resize_rejects_1d_array() -> None:
    image = np.zeros(10, dtype=np.uint8)

    with pytest.raises(ValueError, match="2 or 3 dimensions"):
        im.resize(image, width=5)


def test_resize_preserves_grayscale_shape() -> None:
    image = _make_image(100, 200, channels=None)

    result = im.resize(image, width=100)

    assert result.shape == (50, 100)


def test_resize_preserves_dtype() -> None:
    image = _make_image(100, 200)

    result = im.resize(image, width=100)

    assert result.dtype == image.dtype


def test_resize_does_not_mutate_input() -> None:
    image = _make_image(100, 200)
    original = image.copy()

    im.resize(image, width=50)

    np.testing.assert_array_equal(image, original)


def test_resize_returns_new_array_when_size_unchanged() -> None:
    image = _make_image(100, 200)
    original = image.copy()

    result = im.resize(image, width=200, height=100)
    result[0, 0, 0] = 255

    np.testing.assert_array_equal(image, original)


def test_translate_shifts_content_by_given_offset() -> None:
    image = np.zeros((20, 20), dtype=np.uint8)
    image[5, 5] = 255

    result = im.translate(image, x=3, y=2)

    assert result[7, 8] == 255
    assert result[5, 5] == 0


def test_translate_preserves_shape_and_dtype() -> None:
    image = _make_image(20, 20)

    result = im.translate(image, x=2, y=-3)

    assert result.shape == image.shape
    assert result.dtype == image.dtype


def test_translate_by_zero_preserves_content() -> None:
    image = _make_image(20, 20)

    result = im.translate(image, x=0, y=0)

    np.testing.assert_array_equal(result, image)


def test_translate_does_not_mutate_input() -> None:
    image = np.zeros((20, 20), dtype=np.uint8)
    image[5, 5] = 255
    original = image.copy()

    im.translate(image, x=3, y=2)

    np.testing.assert_array_equal(image, original)


def test_translate_rejects_1d_array() -> None:
    image = np.zeros(10, dtype=np.uint8)

    with pytest.raises(ValueError, match="2 or 3 dimensions"):
        im.translate(image, x=1, y=1)


def test_rotate_by_zero_degrees_preserves_content() -> None:
    image = _make_image(20, 20)

    result = im.rotate(image, angle=0)

    np.testing.assert_array_equal(result, image)


def test_rotate_by_180_degrees_flips_content_around_center() -> None:
    image = np.zeros((21, 21), dtype=np.uint8)
    image[5, 5] = 255

    result = im.rotate(image, angle=180, interpolation=cv2.INTER_NEAREST)

    assert result[16, 16] == 255


def test_rotate_preserves_shape_and_dtype() -> None:
    image = _make_image(20, 20)

    result = im.rotate(image, angle=37)

    assert result.shape == image.shape
    assert result.dtype == image.dtype


def test_rotate_does_not_mutate_input() -> None:
    image = np.zeros((21, 21), dtype=np.uint8)
    image[5, 5] = 255
    original = image.copy()

    im.rotate(image, angle=45)

    np.testing.assert_array_equal(image, original)


def test_rotate_rejects_1d_array() -> None:
    image = np.zeros(10, dtype=np.uint8)

    with pytest.raises(ValueError, match="2 or 3 dimensions"):
        im.rotate(image, angle=10)


def test_rotate_bound_by_zero_degrees_preserves_size_and_content() -> None:
    image = _make_image(20, 20)

    result = im.rotate_bound(image, angle=0)

    np.testing.assert_array_equal(result, image)


def test_rotate_bound_swaps_dimensions_at_90_degrees() -> None:
    image = _make_image(20, 40)  # height=20, width=40

    result = im.rotate_bound(image, angle=90)

    assert result.shape[:2] == (40, 20)


def test_rotate_bound_expands_canvas_for_non_axis_aligned_angle() -> None:
    image = _make_image(20, 20)

    result = im.rotate_bound(image, angle=45)

    assert result.shape[0] > image.shape[0]
    assert result.shape[1] > image.shape[1]


def test_rotate_bound_does_not_mutate_input() -> None:
    image = _make_image(20, 20)
    original = image.copy()

    im.rotate_bound(image, angle=45)

    np.testing.assert_array_equal(image, original)


def test_rotate_bound_rejects_1d_array() -> None:
    image = np.zeros(10, dtype=np.uint8)

    with pytest.raises(ValueError, match="2 or 3 dimensions"):
        im.rotate_bound(image, angle=10)


def test_flip_horizontal_mirrors_columns() -> None:
    image = np.zeros((10, 10), dtype=np.uint8)
    image[2, 3] = 255

    result = im.flip(image, direction="horizontal")

    assert result[2, 6] == 255
    assert result[2, 3] == 0


def test_flip_vertical_mirrors_rows() -> None:
    image = np.zeros((10, 10), dtype=np.uint8)
    image[2, 3] = 255

    result = im.flip(image, direction="vertical")

    assert result[7, 3] == 255
    assert result[2, 3] == 0


def test_flip_both_mirrors_rows_and_columns() -> None:
    image = np.zeros((10, 10), dtype=np.uint8)
    image[2, 3] = 255

    result = im.flip(image, direction="both")

    assert result[7, 6] == 255
    assert result[2, 3] == 0


def test_flip_rejects_invalid_direction() -> None:
    image = _make_image(10, 10)

    with pytest.raises(ValueError, match="direction"):
        im.flip(image, direction="diagonal")  # type: ignore[arg-type]


def test_flip_preserves_shape_and_dtype() -> None:
    image = _make_image(10, 10)

    result = im.flip(image, direction="horizontal")

    assert result.shape == image.shape
    assert result.dtype == image.dtype


def test_flip_does_not_mutate_input() -> None:
    image = np.zeros((10, 10), dtype=np.uint8)
    image[2, 3] = 255
    original = image.copy()

    im.flip(image, direction="horizontal")

    np.testing.assert_array_equal(image, original)


def test_flip_rejects_1d_array() -> None:
    image = np.zeros(10, dtype=np.uint8)

    with pytest.raises(ValueError, match="2 or 3 dimensions"):
        im.flip(image, direction="horizontal")


def test_crop_extracts_expected_region() -> None:
    image = _make_image(10, 10, channels=None)

    result = im.crop(image, x=2, y=3, width=4, height=2)

    np.testing.assert_array_equal(result, image[3:5, 2:6])


def test_crop_returns_copy_not_view() -> None:
    image = _make_image(10, 10, channels=None)
    original = image.copy()

    result = im.crop(image, x=0, y=0, width=4, height=4)
    result[0, 0] = 255

    np.testing.assert_array_equal(image, original)


def test_crop_rejects_region_exceeding_bounds() -> None:
    image = _make_image(10, 10, channels=None)

    with pytest.raises(ValueError, match="exceeds image bounds"):
        im.crop(image, x=8, y=8, width=4, height=4)


def test_crop_rejects_negative_origin() -> None:
    image = _make_image(10, 10, channels=None)

    with pytest.raises(ValueError, match="non-negative"):
        im.crop(image, x=-1, y=0, width=4, height=4)


@pytest.mark.parametrize("width, height", [(0, 4), (4, 0), (-1, 4)])
def test_crop_rejects_non_positive_size(width: int, height: int) -> None:
    image = _make_image(10, 10, channels=None)

    with pytest.raises(ValueError, match="positive"):
        im.crop(image, x=0, y=0, width=width, height=height)


def test_crop_rejects_1d_array() -> None:
    image = np.zeros(10, dtype=np.uint8)

    with pytest.raises(ValueError, match="2 or 3 dimensions"):
        im.crop(image, x=0, y=0, width=2, height=2)


def test_center_crop_extracts_centered_region() -> None:
    image = _make_image(10, 10, channels=None)

    result = im.center_crop(image, width=4, height=4)

    np.testing.assert_array_equal(result, image[3:7, 3:7])


def test_center_crop_returns_copy() -> None:
    image = _make_image(10, 10, channels=None)
    original = image.copy()

    result = im.center_crop(image, width=4, height=4)
    result[0, 0] = 255

    np.testing.assert_array_equal(image, original)


def test_center_crop_rejects_size_larger_than_image() -> None:
    image = _make_image(10, 10, channels=None)

    with pytest.raises(ValueError):
        im.center_crop(image, width=20, height=20)
