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
