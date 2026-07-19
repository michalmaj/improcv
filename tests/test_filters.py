import cv2
import numpy as np
import pytest

import improcv as im


def test_gaussian_blur_reduces_noise_variance() -> None:
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(50, 50), dtype=np.uint8)

    result = im.gaussian_blur(image, kernel_size=5)

    assert result.astype(np.float64).std() < image.astype(np.float64).std()


def test_gaussian_blur_rejects_even_kernel_size(make_image) -> None:
    image = make_image(10, 10, channels=None)

    with pytest.raises(ValueError, match="odd"):
        im.gaussian_blur(image, kernel_size=4)


def test_gaussian_blur_rejects_non_positive_kernel_size(make_image) -> None:
    image = make_image(10, 10, channels=None)

    with pytest.raises(ValueError, match="positive"):
        im.gaussian_blur(image, kernel_size=0)


def test_gaussian_blur_preserves_shape_and_dtype(make_image) -> None:
    image = make_image(10, 10, channels=3)

    result = im.gaussian_blur(image, kernel_size=3)

    assert result.shape == image.shape
    assert result.dtype == image.dtype


def test_gaussian_blur_rejects_1d_array() -> None:
    image = np.zeros(10, dtype=np.uint8)

    with pytest.raises(ValueError, match="2 or 3 dimensions"):
        im.gaussian_blur(image, kernel_size=3)


def test_gaussian_blur_accepts_zero_sigma(make_image) -> None:
    # sigma=0.0 is the documented default meaning "derive from kernel_size".
    image = make_image(10, 10, channels=None)

    result = im.gaussian_blur(image, kernel_size=3, sigma=0.0)

    assert result.shape == image.shape


def test_gaussian_blur_rejects_negative_sigma(make_image) -> None:
    image = make_image(10, 10, channels=None)

    with pytest.raises(ValueError, match="non-negative"):
        im.gaussian_blur(image, kernel_size=3, sigma=-1.0)


def test_gaussian_blur_rejects_non_finite_sigma(make_image) -> None:
    image = make_image(10, 10, channels=None)

    with pytest.raises(ValueError, match="finite"):
        im.gaussian_blur(image, kernel_size=3, sigma=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        im.gaussian_blur(image, kernel_size=3, sigma=float("inf"))


def test_gaussian_blur_rejects_int32_dtype() -> None:
    # cv2.GaussianBlur raises a raw cv2.error for int32 input — verified
    # directly against cv2 (identical on OpenCV 4.13 and 5.0).
    image = np.zeros((10, 10), dtype=np.int32)

    with pytest.raises(TypeError, match="dtype"):
        im.gaussian_blur(image, kernel_size=3)  # type: ignore[arg-type]


def test_median_blur_removes_salt_and_pepper_outlier() -> None:
    image = np.full((11, 11), 100, dtype=np.uint8)
    image[5, 5] = 255

    result = im.median_blur(image, kernel_size=3)

    assert result[5, 5] == 100


def test_median_blur_rejects_even_kernel_size(make_image) -> None:
    image = make_image(10, 10, channels=None)

    with pytest.raises(ValueError, match="odd"):
        im.median_blur(image, kernel_size=4)


def test_median_blur_accepts_uint16_with_kernel_size_7() -> None:
    # cv2.medianBlur accepts uint16 at kernel_size=7 (verified directly on
    # both OpenCV 4.13 and 5.0) — only kernel_size > 7 restricts to uint8.
    image = np.zeros((10, 10), dtype=np.uint16)

    result = im.median_blur(image, kernel_size=7)

    assert result.dtype == np.uint16


def test_median_blur_rejects_non_uint8_dtype_with_kernel_size_above_7() -> None:
    # cv2.medianBlur raises a raw cv2.error for uint16 at kernel_size=9 —
    # verified directly (identical on OpenCV 4.13 and 5.0).
    image = np.zeros((10, 10), dtype=np.uint16)

    with pytest.raises(TypeError, match="uint8"):
        im.median_blur(image, kernel_size=9)  # type: ignore[arg-type]


def test_median_blur_accepts_uint8_with_kernel_size_above_7() -> None:
    image = np.zeros((10, 10), dtype=np.uint8)

    result = im.median_blur(image, kernel_size=9)

    assert result.dtype == np.uint8


def test_median_blur_rejects_five_channels_with_kernel_size_above_5() -> None:
    # cv2.medianBlur raises a raw cv2.error for a 5-channel image once
    # kernel_size exceeds 5, regardless of dtype — verified directly
    # (identical on OpenCV 4.13 and 5.0); channel counts 1-4 all work at
    # any kernel_size.
    image = np.zeros((10, 10, 5), dtype=np.uint8)

    with pytest.raises(ValueError, match="4 channels"):
        im.median_blur(image, kernel_size=7)


def test_median_blur_accepts_up_to_four_channels_with_large_kernel_size() -> None:
    image = np.zeros((10, 10, 4), dtype=np.uint8)

    result = im.median_blur(image, kernel_size=9)

    assert result.shape == image.shape


def test_bilateral_filter_preserves_shape_and_dtype(make_image) -> None:
    image = make_image(10, 10, channels=3)

    result = im.bilateral_filter(image, diameter=5, sigma_color=75.0, sigma_space=75.0)

    assert result.shape == image.shape
    assert result.dtype == image.dtype


def test_bilateral_filter_rejects_non_positive_diameter(make_image) -> None:
    image = make_image(10, 10, channels=3)

    with pytest.raises(ValueError, match="positive"):
        im.bilateral_filter(image, diameter=0, sigma_color=75.0, sigma_space=75.0)


def test_bilateral_filter_rejects_non_int_diameter(make_image) -> None:
    image = make_image(10, 10, channels=3)

    with pytest.raises(TypeError, match="int"):
        im.bilateral_filter(image, diameter=5.0, sigma_color=75.0, sigma_space=75.0)  # type: ignore[arg-type]


def test_bilateral_filter_rejects_1d_array() -> None:
    image = np.zeros(10, dtype=np.uint8)

    with pytest.raises(ValueError, match="2 or 3 dimensions"):
        im.bilateral_filter(image, diameter=5, sigma_color=75.0, sigma_space=75.0)


def test_bilateral_filter_rejects_uint16_dtype() -> None:
    # cv2.bilateralFilter raises a raw cv2.error for uint16 input — only
    # uint8 and float32 are supported, verified directly against cv2
    # (identical on OpenCV 4.13 and 5.0).
    image = np.zeros((10, 10), dtype=np.uint16)

    with pytest.raises(TypeError, match="dtype"):
        im.bilateral_filter(image, diameter=5, sigma_color=75.0, sigma_space=75.0)  # type: ignore[arg-type]


def test_clahe_preserves_shape_and_dtype(make_image) -> None:
    image = make_image(20, 20, channels=None)

    result = im.clahe(image)

    assert result.shape == image.shape
    assert result.dtype == image.dtype


def test_clahe_rejects_multichannel_image(make_image) -> None:
    image = make_image(20, 20, channels=3)

    with pytest.raises(ValueError, match="2 dimensions"):
        im.clahe(image)


def test_clahe_rejects_non_positive_tile_grid_size_without_calling_opencv(
    make_image, monkeypatch
) -> None:
    image = make_image(20, 20, channels=None)

    def _fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("cv2.createCLAHE must not be called for invalid input")

    monkeypatch.setattr(cv2, "createCLAHE", _fail_if_called)

    with pytest.raises(ValueError, match="positive"):
        im.clahe(image, tile_grid_size=(0, 8))


def test_clahe_rejects_wrong_length_tile_grid_size_without_calling_opencv(
    make_image, monkeypatch
) -> None:
    # Previously unpacked tile_grid_size before validating its length,
    # raising a raw "not enough values to unpack" instead of a clear error.
    image = make_image(20, 20, channels=None)

    def _fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("cv2.createCLAHE must not be called for invalid input")

    monkeypatch.setattr(cv2, "createCLAHE", _fail_if_called)

    with pytest.raises(ValueError, match="2-tuple"):
        im.clahe(image, tile_grid_size=(8,))  # type: ignore[arg-type]


def test_clahe_rejects_non_positive_clip_limit(make_image) -> None:
    image = make_image(20, 20, channels=None)

    with pytest.raises(ValueError, match="positive"):
        im.clahe(image, clip_limit=0)


def test_clahe_accepts_uint16(make_image) -> None:
    image = make_image(20, 20, channels=None).astype(np.uint16)

    result = im.clahe(image)

    assert result.dtype == np.uint16


def test_clahe_rejects_unsupported_dtype() -> None:
    image = np.zeros((10, 10), dtype=np.float32)

    with pytest.raises(TypeError, match="uint8"):
        im.clahe(image)


def test_gamma_correction_below_one_darkens_image() -> None:
    image = np.full((10, 10), 200, dtype=np.uint8)

    result = im.gamma_correction(image, gamma=0.5)

    assert result[0, 0] < image[0, 0]


def test_gamma_correction_of_one_preserves_image(make_image) -> None:
    image = make_image(10, 10, channels=None)

    result = im.gamma_correction(image, gamma=1.0)

    np.testing.assert_array_equal(result, image)


def test_gamma_correction_rejects_non_positive_gamma(make_image) -> None:
    image = make_image(10, 10, channels=None)

    with pytest.raises(ValueError, match="positive"):
        im.gamma_correction(image, gamma=0)


def test_gamma_correction_rejects_numpy_nan_gamma(make_image) -> None:
    image = make_image(10, 10, channels=None)

    with pytest.raises(ValueError, match="finite"):
        im.gamma_correction(image, gamma=np.float32(np.nan))  # type: ignore[arg-type]


def test_gamma_correction_rejects_non_uint8_dtype() -> None:
    image = np.zeros((10, 10), dtype=np.float32)

    with pytest.raises(TypeError, match="uint8"):
        im.gamma_correction(image, gamma=2.0)  # type: ignore[arg-type]


def test_histogram_equalization_spreads_intensity_range() -> None:
    image = np.zeros((20, 20), dtype=np.uint8)
    image[:, 10:] = 50

    result = im.histogram_equalization(image)

    assert int(result.max()) - int(result.min()) >= 200


def test_histogram_equalization_rejects_multichannel_image(make_image) -> None:
    image = make_image(20, 20, channels=3)

    with pytest.raises(ValueError, match="2 dimensions"):
        im.histogram_equalization(image)


def test_histogram_equalization_rejects_non_uint8_dtype() -> None:
    image = np.zeros((10, 10), dtype=np.float32)

    with pytest.raises(TypeError, match="uint8"):
        im.histogram_equalization(image)  # type: ignore[arg-type]
