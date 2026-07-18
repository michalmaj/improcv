import numpy as np
import pytest

import improcv as im


def test_bgr_to_rgb_swaps_channels() -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[:, :, 0] = 10  # B
    image[:, :, 1] = 20  # G
    image[:, :, 2] = 30  # R

    result = im.bgr_to_rgb(image)

    assert result[0, 0, 0] == 30
    assert result[0, 0, 1] == 20
    assert result[0, 0, 2] == 10


def test_bgr_to_rgb_rejects_non_3_channel_image() -> None:
    image = np.zeros((4, 4), dtype=np.uint8)

    with pytest.raises(ValueError, match="3 channels"):
        im.bgr_to_rgb(image)


def test_bgr_to_rgb_rejects_empty_image() -> None:
    image = np.zeros((0, 10, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="empty"):
        im.bgr_to_rgb(image)


def test_bgr_to_rgb_rejects_int64_dtype() -> None:
    image = np.zeros((4, 4, 3), dtype=np.int64)

    with pytest.raises(TypeError, match="uint8"):
        im.bgr_to_rgb(image)  # type: ignore[arg-type]


def test_rgb_to_bgr_swaps_channels() -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[:, :, 0] = 10  # R
    image[:, :, 1] = 20  # G
    image[:, :, 2] = 30  # B

    result = im.rgb_to_bgr(image)

    assert result[0, 0, 0] == 30
    assert result[0, 0, 1] == 20
    assert result[0, 0, 2] == 10


def test_ensure_gray_converts_color_image(make_image) -> None:
    image = make_image(10, 10, channels=3)

    result = im.ensure_gray(image)

    assert result.shape == (10, 10)
    assert result.dtype == image.dtype


def test_ensure_gray_passes_through_already_gray_image(make_image) -> None:
    image = make_image(10, 10, channels=None)

    result = im.ensure_gray(image)

    np.testing.assert_array_equal(result, image)


def test_ensure_gray_returns_copy_for_already_gray_image(make_image) -> None:
    image = make_image(10, 10, channels=None)
    original = image.copy()

    result = im.ensure_gray(image)
    result[0, 0] = 255

    np.testing.assert_array_equal(image, original)


def test_ensure_gray_rejects_1d_array() -> None:
    image = np.zeros(10, dtype=np.uint8)

    with pytest.raises(ValueError, match="2 or 3 dimensions"):
        im.ensure_gray(image)


def test_to_hsv_returns_3_channel_image(make_image) -> None:
    image = make_image(10, 10, channels=3)

    result = im.to_hsv(image)

    assert result.shape == image.shape


def test_to_hsv_rejects_non_3_channel_image() -> None:
    image = np.zeros((4, 4), dtype=np.uint8)

    with pytest.raises(ValueError, match="3 channels"):
        im.to_hsv(image)


def test_to_hsv_rejects_uint16_dtype() -> None:
    # Unlike bgr_to_rgb/to_ycrcb, cv2's HSV conversion does not support
    # uint16 — verified directly against cv2 before adding this check.
    image = np.zeros((4, 4, 3), dtype=np.uint16)

    with pytest.raises(TypeError, match="uint8"):
        im.to_hsv(image)  # type: ignore[arg-type]


def test_to_lab_returns_3_channel_image(make_image) -> None:
    image = make_image(10, 10, channels=3)

    result = im.to_lab(image)

    assert result.shape == image.shape


def test_to_ycrcb_returns_3_channel_image(make_image) -> None:
    image = make_image(10, 10, channels=3)

    result = im.to_ycrcb(image)

    assert result.shape == image.shape
