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


def _grabcut_test_image() -> np.ndarray:
    # A clearly-separable synthetic image: light gray background, a dark
    # red-ish square in the middle -- verified directly to give a stable,
    # obvious foreground/background split on both OpenCV 4.13 and 5.0.
    image = np.full((60, 60, 3), (200, 200, 200), dtype=np.uint8)
    image[20:40, 20:40] = (10, 10, 200)
    return image


def test_grabcut_rect_marks_far_corners_as_background() -> None:
    image = _grabcut_test_image()

    mask = im.grabcut_rect(image, im.BoundingBox(10, 10, 40, 40))

    assert mask.shape == (60, 60)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)) <= {0, 255}
    assert mask[0, 0] == 0
    assert mask[59, 59] == 0


def test_grabcut_rect_marks_obvious_foreground_square() -> None:
    # Verified directly stable on both OpenCV 4.13 and 5.0 for this exact
    # synthetic image/rect pair -- not asserted as a general guarantee for
    # every input (GrabCut can legitimately find no foreground at all).
    image = _grabcut_test_image()

    mask = im.grabcut_rect(image, im.BoundingBox(10, 10, 40, 40))

    assert mask[30, 30] == 255


def test_grabcut_rect_does_not_mutate_image() -> None:
    image = _grabcut_test_image()
    original = image.copy()

    im.grabcut_rect(image, im.BoundingBox(10, 10, 40, 40))

    np.testing.assert_array_equal(image, original)


def test_grabcut_rect_rejects_grayscale_image() -> None:
    image = np.zeros((60, 60), dtype=np.uint8)

    with pytest.raises(ValueError, match="channels"):
        im.grabcut_rect(image, im.BoundingBox(10, 10, 40, 40))


def test_grabcut_rect_rejects_non_uint8_dtype() -> None:
    image = _grabcut_test_image().astype(np.float32)

    with pytest.raises(TypeError, match="dtype"):
        im.grabcut_rect(image, im.BoundingBox(10, 10, 40, 40))  # type: ignore[arg-type]


def test_grabcut_rect_rejects_rect_covering_full_image() -> None:
    image = _grabcut_test_image()

    with pytest.raises(ValueError, match="entire image"):
        im.grabcut_rect(image, im.BoundingBox(0, 0, 60, 60))


def test_grabcut_rect_rejects_rect_extending_outside_image() -> None:
    image = _grabcut_test_image()

    with pytest.raises(ValueError, match="contained"):
        im.grabcut_rect(image, im.BoundingBox(-1, -1, 40, 40))
    with pytest.raises(ValueError, match="contained"):
        im.grabcut_rect(image, im.BoundingBox(10, 10, 100, 40))


def test_grabcut_rect_rejects_non_positive_rect_dimensions() -> None:
    image = _grabcut_test_image()

    with pytest.raises(ValueError, match="positive"):
        im.grabcut_rect(image, im.BoundingBox(10, 10, 0, 40))


def test_grabcut_rect_rejects_float_rect_fields() -> None:
    image = _grabcut_test_image()

    with pytest.raises(TypeError, match="integer"):
        im.grabcut_rect(image, (10.0, 10, 40, 40))  # type: ignore[arg-type]


def test_grabcut_rect_rejects_bool_rect_field() -> None:
    image = _grabcut_test_image()

    with pytest.raises(TypeError, match="integer"):
        im.grabcut_rect(image, (True, 10, 40, 40))  # type: ignore[arg-type]


def test_grabcut_rect_accepts_numpy_integer_rect_fields() -> None:
    image = _grabcut_test_image()

    mask = im.grabcut_rect(
        image,
        (np.int32(10), np.int32(10), np.int32(40), np.int32(40)),  # type: ignore[arg-type]
    )

    assert mask.shape == (60, 60)


def test_grabcut_rect_rejects_non_positive_iterations() -> None:
    image = _grabcut_test_image()

    with pytest.raises(ValueError, match="positive"):
        im.grabcut_rect(image, im.BoundingBox(10, 10, 40, 40), iterations=0)


def test_grabcut_rect_rejects_bool_iterations() -> None:
    image = _grabcut_test_image()

    with pytest.raises(TypeError, match="integer"):
        im.grabcut_rect(  # type: ignore[arg-type]
            image, im.BoundingBox(10, 10, 40, 40), iterations=True
        )


def test_grabcut_rect_accepts_minimal_rect_and_image_sizes() -> None:
    # Verified directly not to raise or produce an unstable result on
    # either OpenCV version: a 1x1 rect in a 5x5 image, and a rect leaving
    # only a 1px border in a 10x10 image.
    small_image = np.random.default_rng(0).integers(0, 255, (5, 5, 3), dtype=np.uint8)
    mask = im.grabcut_rect(small_image, im.BoundingBox(2, 2, 1, 1))
    assert mask.shape == (5, 5)

    bordered_image = np.random.default_rng(0).integers(0, 255, (10, 10, 3), dtype=np.uint8)
    mask2 = im.grabcut_rect(bordered_image, im.BoundingBox(1, 1, 8, 8))
    assert mask2.shape == (10, 10)


def test_grabcut_rect_accepts_constant_color_image() -> None:
    # Verified directly not to raise on either OpenCV version, despite no
    # texture variation to distinguish foreground from background.
    image = np.full((20, 20, 3), 128, dtype=np.uint8)

    mask = im.grabcut_rect(image, im.BoundingBox(5, 5, 10, 10))

    assert mask.shape == (20, 20)
