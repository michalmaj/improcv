import numpy as np
import pytest

import improcv as im


def test_auto_canny_returns_binary_edge_map() -> None:
    image = np.zeros((20, 20), dtype=np.uint8)
    image[:, 10:] = 255

    result = im.auto_canny(image)

    assert result.shape == image.shape
    assert set(np.unique(result).tolist()) <= {0, 255}
    assert np.count_nonzero(result) > 0


def test_auto_canny_rejects_multichannel_image(make_image) -> None:
    image = make_image(20, 20, channels=3)

    with pytest.raises(ValueError, match="2 dimensions"):
        im.auto_canny(image)


def test_auto_canny_rejects_non_uint8_dtype() -> None:
    image = np.zeros((10, 10), dtype=np.float32)

    with pytest.raises(TypeError, match="uint8"):
        im.auto_canny(image)  # type: ignore[arg-type]


def test_auto_canny_rejects_sigma_outside_unit_range() -> None:
    image = np.zeros((10, 10), dtype=np.uint8)

    with pytest.raises(ValueError, match="sigma"):
        im.auto_canny(image, sigma=-1)
    with pytest.raises(ValueError, match="sigma"):
        im.auto_canny(image, sigma=1.5)


def test_auto_canny_rejects_non_finite_sigma() -> None:
    image = np.zeros((10, 10), dtype=np.uint8)

    with pytest.raises(ValueError, match="sigma"):
        im.auto_canny(image, sigma=float("nan"))


def test_sobel_edge_detects_vertical_edge() -> None:
    image = np.zeros((20, 20), dtype=np.uint8)
    image[:, 10:] = 255

    result = im.sobel_edge(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert result[:, 10].max() > 0


def test_sobel_edge_rejects_multichannel_image(make_image) -> None:
    image = make_image(20, 20, channels=3)

    with pytest.raises(ValueError, match="2 dimensions"):
        im.sobel_edge(image)


def test_sobel_edge_rejects_even_kernel_size() -> None:
    image = np.zeros((20, 20), dtype=np.uint8)

    with pytest.raises(ValueError, match="odd"):
        im.sobel_edge(image, kernel_size=2)


def test_sobel_edge_rejects_kernel_size_above_31() -> None:
    image = np.zeros((40, 40), dtype=np.uint8)

    with pytest.raises(ValueError, match="kernel_size"):
        im.sobel_edge(image, kernel_size=33)


def test_laplacian_edge_detects_edge() -> None:
    image = np.zeros((20, 20), dtype=np.uint8)
    image[:, 10:] = 255

    result = im.laplacian_edge(image)

    assert result.shape == image.shape
    assert np.count_nonzero(result) > 0


def test_laplacian_edge_rejects_even_kernel_size() -> None:
    image = np.zeros((20, 20), dtype=np.uint8)

    with pytest.raises(ValueError, match="odd"):
        im.laplacian_edge(image, kernel_size=2)


def test_laplacian_edge_rejects_kernel_size_above_31() -> None:
    image = np.zeros((40, 40), dtype=np.uint8)

    with pytest.raises(ValueError, match="kernel_size"):
        im.laplacian_edge(image, kernel_size=33)


def test_laplacian_edge_rejects_float32_dtype() -> None:
    # cv2.Laplacian always requests a float64 destination here, and OpenCV
    # rejects the float32-source/float64-dest combination specifically —
    # verified directly against cv2 (identical on OpenCV 4.13 and 5.0),
    # even though float32 works for every other function in this module.
    image = np.zeros((20, 20), dtype=np.float32)

    with pytest.raises(TypeError, match="dtype"):
        im.laplacian_edge(image)  # type: ignore[arg-type]


def test_sobel_edge_accepts_float32_dtype() -> None:
    # Unlike laplacian_edge, cv2.Sobel accepts float32 source with a
    # float64 destination — verified directly against cv2.
    image = np.zeros((20, 20), dtype=np.float32)
    image[:, 10:] = 1.0

    result = im.sobel_edge(image)

    assert result.shape == image.shape


def test_harris_corner_detects_corner_of_square() -> None:
    image = np.zeros((30, 30), dtype=np.uint8)
    image[10:20, 10:20] = 255

    result = im.harris_corner(image)

    assert result.dtype == np.uint8
    assert result.shape == image.shape
    assert set(np.unique(result).tolist()) <= {0, 255}
    assert np.count_nonzero(result) > 0


def test_harris_corner_rejects_multichannel_image(make_image) -> None:
    image = make_image(20, 20, channels=3)

    with pytest.raises(ValueError, match="2 dimensions"):
        im.harris_corner(image)


def test_harris_corner_rejects_even_kernel_size() -> None:
    image = np.zeros((20, 20), dtype=np.uint8)

    with pytest.raises(ValueError, match="odd"):
        im.harris_corner(image, kernel_size=2)


def test_harris_corner_rejects_kernel_size_above_31() -> None:
    image = np.zeros((40, 40), dtype=np.uint8)

    with pytest.raises(ValueError, match="kernel_size"):
        im.harris_corner(image, kernel_size=33)


def test_harris_corner_rejects_non_positive_block_size() -> None:
    image = np.zeros((20, 20), dtype=np.uint8)

    with pytest.raises(ValueError, match="positive"):
        im.harris_corner(image, block_size=0)


def test_harris_corner_rejects_negative_threshold() -> None:
    image = np.zeros((20, 20), dtype=np.uint8)

    with pytest.raises(ValueError, match="non-negative"):
        im.harris_corner(image, threshold=-0.1)


def test_harris_corner_rejects_non_positive_k() -> None:
    image = np.zeros((20, 20), dtype=np.uint8)

    with pytest.raises(ValueError, match="positive"):
        im.harris_corner(image, k=0)
