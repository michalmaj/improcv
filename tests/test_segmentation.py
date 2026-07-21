import numpy as np
import pytest

import improcv as im


def _two_region_image() -> np.ndarray:
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    image[:, :10] = (0, 0, 255)  # left half: red (BGR)
    image[:, 10:] = (255, 0, 0)  # right half: blue
    return image


def test_watershed_labels_two_seeded_regions_with_a_boundary() -> None:
    image = _two_region_image()
    markers = np.zeros((10, 20), dtype=np.int32)
    markers[5, 2] = 2
    markers[5, 17] = 10

    result = im.watershed(image, markers)

    assert set(np.unique(result)) == {-1, 2, 10}
    assert result[5, 2] == 2
    assert result[5, 17] == 10


def test_watershed_accepts_a_single_seed() -> None:
    image = _two_region_image()
    markers = np.zeros((10, 20), dtype=np.int32)
    markers[5, 5] = 7

    result = im.watershed(image, markers)

    assert 7 in np.unique(result)


def test_watershed_accepts_non_contiguous_seed_labels() -> None:
    image = _two_region_image()
    markers = np.zeros((10, 20), dtype=np.int32)
    markers[5, 2] = 2
    markers[5, 17] = 10

    result = im.watershed(image, markers)

    assert set(np.unique(result)) == {-1, 2, 10}


def test_watershed_does_not_mutate_input() -> None:
    image = _two_region_image()
    markers = np.zeros((10, 20), dtype=np.int32)
    markers[5, 5] = 1
    original_image = image.copy()
    original_markers = markers.copy()

    im.watershed(image, markers)

    np.testing.assert_array_equal(image, original_image)
    np.testing.assert_array_equal(markers, original_markers)


def test_watershed_returns_a_fresh_array() -> None:
    image = _two_region_image()
    markers = np.zeros((10, 20), dtype=np.int32)
    markers[5, 5] = 1

    result = im.watershed(image, markers)

    assert not np.shares_memory(result, markers)


def test_watershed_rejects_negative_marker_value() -> None:
    image = _two_region_image()
    markers = np.zeros((10, 20), dtype=np.int32)
    markers[5, 5] = 1
    markers[3, 3] = -2

    with pytest.raises(ValueError, match="negative"):
        im.watershed(image, markers)


def test_watershed_rejects_all_zero_markers() -> None:
    image = _two_region_image()
    markers = np.zeros((10, 20), dtype=np.int32)

    with pytest.raises(ValueError, match="positive seed"):
        im.watershed(image, markers)


def test_watershed_rejects_non_int32_markers() -> None:
    image = _two_region_image()
    markers = np.zeros((10, 20), dtype=np.float32)
    markers[5, 5] = 1

    with pytest.raises(TypeError, match="dtype"):
        im.watershed(image, markers)  # type: ignore[arg-type]


def test_watershed_rejects_grayscale_image() -> None:
    image = np.zeros((10, 20), dtype=np.uint8)
    markers = np.zeros((10, 20), dtype=np.int32)
    markers[5, 5] = 1

    with pytest.raises(ValueError, match="channels"):
        im.watershed(image, markers)


def test_watershed_rejects_mismatched_markers_shape() -> None:
    image = _two_region_image()
    markers = np.zeros((5, 5), dtype=np.int32)
    markers[2, 2] = 1

    with pytest.raises(ValueError, match="shape"):
        im.watershed(image, markers)
