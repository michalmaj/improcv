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


def test_median_blur_removes_salt_and_pepper_outlier() -> None:
    image = np.full((11, 11), 100, dtype=np.uint8)
    image[5, 5] = 255

    result = im.median_blur(image, kernel_size=3)

    assert result[5, 5] == 100


def test_median_blur_rejects_even_kernel_size(make_image) -> None:
    image = make_image(10, 10, channels=None)

    with pytest.raises(ValueError, match="odd"):
        im.median_blur(image, kernel_size=4)


def test_bilateral_filter_preserves_shape_and_dtype(make_image) -> None:
    image = make_image(10, 10, channels=3)

    result = im.bilateral_filter(image, diameter=5, sigma_color=75.0, sigma_space=75.0)

    assert result.shape == image.shape
    assert result.dtype == image.dtype


def test_bilateral_filter_rejects_non_positive_diameter(make_image) -> None:
    image = make_image(10, 10, channels=3)

    with pytest.raises(ValueError, match="positive"):
        im.bilateral_filter(image, diameter=0, sigma_color=75.0, sigma_space=75.0)


def test_bilateral_filter_rejects_1d_array() -> None:
    image = np.zeros(10, dtype=np.uint8)

    with pytest.raises(ValueError, match="2 or 3 dimensions"):
        im.bilateral_filter(image, diameter=5, sigma_color=75.0, sigma_space=75.0)


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


def test_clahe_rejects_non_positive_clip_limit(make_image) -> None:
    image = make_image(20, 20, channels=None)

    with pytest.raises(ValueError, match="positive"):
        im.clahe(image, clip_limit=0)


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


def test_histogram_equalization_spreads_intensity_range() -> None:
    image = np.zeros((20, 20), dtype=np.uint8)
    image[:, 10:] = 50

    result = im.histogram_equalization(image)

    assert int(result.max()) - int(result.min()) >= 200


def test_histogram_equalization_rejects_multichannel_image(make_image) -> None:
    image = make_image(20, 20, channels=3)

    with pytest.raises(ValueError, match="2 dimensions"):
        im.histogram_equalization(image)
