import numpy as np
import pytest

import improcv as im


def test_threshold_binary_produces_two_values() -> None:
    image = np.array([[50, 200], [10, 250]], dtype=np.uint8)

    result = im.threshold(image, value=100, max_value=255, method="binary")

    np.testing.assert_array_equal(result, np.array([[0, 255], [0, 255]], dtype=np.uint8))


def test_threshold_rejects_multichannel_image(make_image) -> None:
    image = make_image(10, 10, channels=3)

    with pytest.raises(ValueError, match="2 dimensions"):
        im.threshold(image)


def test_threshold_otsu_produces_binary_output() -> None:
    image = np.zeros((10, 10), dtype=np.uint8)
    image[:, 5:] = 255

    result = im.threshold(image, method="otsu")

    assert set(np.unique(result).tolist()) <= {0, 255}


def test_threshold_adaptive_mean_produces_binary_output(make_image) -> None:
    image = make_image(20, 20, channels=None)

    result = im.threshold(image, method="adaptive_mean", block_size=5)

    assert set(np.unique(result).tolist()) <= {0, 255}


def test_threshold_adaptive_rejects_even_block_size(make_image) -> None:
    image = make_image(20, 20, channels=None)

    with pytest.raises(ValueError, match="odd"):
        im.threshold(image, method="adaptive_mean", block_size=4)


def test_threshold_rejects_unknown_method() -> None:
    image = np.zeros((10, 10), dtype=np.uint8)

    with pytest.raises(ValueError, match="method"):
        im.threshold(image, method="cokolwiek")  # type: ignore[arg-type]


def test_threshold_rejects_non_finite_value() -> None:
    image = np.zeros((10, 10), dtype=np.uint8)

    with pytest.raises(ValueError, match="finite"):
        im.threshold(image, value=float("nan"))


def test_threshold_rejects_non_finite_max_value() -> None:
    image = np.zeros((10, 10), dtype=np.uint8)

    with pytest.raises(ValueError, match="finite"):
        im.threshold(image, max_value=float("inf"))


def test_threshold_rejects_non_finite_constant() -> None:
    image = np.zeros((10, 10), dtype=np.uint8)

    with pytest.raises(ValueError, match="finite"):
        im.threshold(image, method="adaptive_mean", constant=float("nan"))


def test_threshold_otsu_rejects_non_uint8_dtype() -> None:
    image = np.zeros((10, 10), dtype=np.float32)

    with pytest.raises(TypeError, match="uint8"):
        im.threshold(image, method="otsu")


def test_threshold_adaptive_rejects_non_uint8_dtype() -> None:
    image = np.zeros((10, 10), dtype=np.float32)

    with pytest.raises(TypeError, match="uint8"):
        im.threshold(image, method="adaptive_mean")


def test_threshold_binary_rejects_int32_dtype() -> None:
    # cv2.threshold raises a raw cv2.error for int32 input — verified
    # directly against cv2 (identical on OpenCV 4.13 and 5.0).
    image = np.zeros((10, 10), dtype=np.int32)

    with pytest.raises(TypeError, match="dtype"):
        im.threshold(image, method="binary")  # type: ignore[arg-type]


def test_threshold_binary_rejects_max_value_exceeding_dtype_range() -> None:
    # cv2.threshold silently saturates an out-of-range max_value to 255
    # for a uint8 image instead of raising — verified directly against
    # cv2 with a foreground pixel of value 200.
    image = np.full((10, 10), 200, dtype=np.uint8)

    with pytest.raises(ValueError, match="max_value"):
        im.threshold(image, value=100, max_value=300, method="binary")


def test_threshold_binary_does_not_follow_mask_convention() -> None:
    # threshold is deliberately not one of the uint8-{0,255}-mask functions
    # (in_range/auto_canny/harris_corner): "binary" mode preserves dtype
    # and honors a custom max_value, so its output isn't always a mask.
    image = np.array([[0.1, 0.9]], dtype=np.float32)

    result = im.threshold(image, value=0.5, max_value=1.0, method="binary")

    assert result.dtype == np.float32
    np.testing.assert_allclose(result, [[0.0, 1.0]])


def test_dilate_grows_bright_region() -> None:
    image = np.zeros((11, 11), dtype=np.uint8)
    image[5, 5] = 255

    result = im.dilate(image, kernel_size=3)

    assert result[5, 4] == 255
    assert result[5, 6] == 255
    assert int(np.count_nonzero(result)) > 1


def test_dilate_rejects_1d_array() -> None:
    image = np.zeros(10, dtype=np.uint8)

    with pytest.raises(ValueError, match="2 or 3 dimensions"):
        im.dilate(image)


def test_dilate_with_zero_iterations_is_a_no_op() -> None:
    # iterations=0 is a meaningful "do nothing" value, not an error case.
    image = np.zeros((11, 11), dtype=np.uint8)
    image[5, 5] = 255

    result = im.dilate(image, kernel_size=3, iterations=0)

    np.testing.assert_array_equal(result, image)


def test_dilate_rejects_negative_iterations() -> None:
    image = np.zeros((11, 11), dtype=np.uint8)

    with pytest.raises(ValueError, match="non-negative"):
        im.dilate(image, kernel_size=3, iterations=-1)


def test_dilate_rejects_int64_dtype() -> None:
    # cv2.dilate raises a raw cv2.error for int64 input — verified
    # directly against cv2 (identical on OpenCV 4.13 and 5.0).
    image = np.zeros((11, 11), dtype=np.int64)

    with pytest.raises(TypeError, match="dtype"):
        im.dilate(image)  # type: ignore[arg-type]


def test_erode_shrinks_bright_region() -> None:
    image = np.zeros((11, 11), dtype=np.uint8)
    image[3:8, 3:8] = 255

    result = im.erode(image, kernel_size=3)

    assert int(np.count_nonzero(result)) < int(np.count_nonzero(image))


def test_erode_rejects_negative_iterations() -> None:
    image = np.zeros((11, 11), dtype=np.uint8)

    with pytest.raises(ValueError, match="non-negative"):
        im.erode(image, kernel_size=3, iterations=-1)


def test_morph_open_removes_small_noise_speck() -> None:
    image = np.zeros((15, 15), dtype=np.uint8)
    image[7, 7] = 255

    result = im.morph_open(image, kernel_size=3)

    assert np.count_nonzero(result) == 0


def test_morph_close_fills_small_hole() -> None:
    image = np.full((15, 15), 255, dtype=np.uint8)
    image[7, 7] = 0

    result = im.morph_close(image, kernel_size=3)

    assert result[7, 7] == 255


def test_morph_gradient_highlights_edges() -> None:
    image = np.zeros((15, 15), dtype=np.uint8)
    image[5:10, 5:10] = 255

    result = im.morph_gradient(image, kernel_size=3)

    assert result[7, 7] == 0
    assert int(np.count_nonzero(result)) > 0


def test_tophat_highlights_small_bright_detail() -> None:
    image = np.zeros((25, 25), dtype=np.uint8)
    image[12, 12] = 255

    result = im.tophat(image, kernel_size=9)

    assert result[12, 12] > 0


def test_blackhat_highlights_small_dark_detail() -> None:
    image = np.full((25, 25), 255, dtype=np.uint8)
    image[12, 12] = 0

    result = im.blackhat(image, kernel_size=9)

    assert result[12, 12] > 0
