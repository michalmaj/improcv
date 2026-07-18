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


def test_laplacian_edge_detects_edge() -> None:
    image = np.zeros((20, 20), dtype=np.uint8)
    image[:, 10:] = 255

    result = im.laplacian_edge(image)

    assert result.shape == image.shape
    assert np.count_nonzero(result) > 0


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
