import numpy as np
import pytest

import improcv as im


def _rect_mask(y0: int, y1: int, x0: int, x1: int, shape: tuple[int, int] = (20, 20)) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255
    return mask


def test_connected_components_counts_background_and_components() -> None:
    mask = _rect_mask(2, 6, 2, 6)
    mask[10:14, 10:14] = 255

    num_labels, labels = im.connected_components(mask, connectivity=8)

    assert num_labels == 3  # background + 2 components
    assert labels.shape == mask.shape
    assert labels.dtype == np.int32
    assert labels[0, 0] == 0  # background


def test_connected_components_treats_any_nonzero_value_as_foreground() -> None:
    mask = _rect_mask(2, 6, 2, 6)
    mask[mask == 255] = 1

    num_labels, _ = im.connected_components(mask, connectivity=8)

    assert num_labels == 2


def test_connected_components_connectivity_4_vs_8_diagonal_touch() -> None:
    # Two pixels touching only at a corner -- verified directly against cv2:
    # connectivity=4 keeps them as 2 separate components (n=3 incl.
    # background); connectivity=8 merges them into 1 (n=2 incl. background).
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[3, 3] = 255
    mask[4, 4] = 255

    n4, _ = im.connected_components(mask, connectivity=4)
    n8, _ = im.connected_components(mask, connectivity=8)

    assert n4 == 3
    assert n8 == 2


def test_connected_components_does_not_mutate_input() -> None:
    mask = _rect_mask(2, 6, 2, 6)
    original = mask.copy()

    im.connected_components(mask)

    np.testing.assert_array_equal(mask, original)


def test_connected_components_rejects_non_uint8_dtype() -> None:
    mask = np.zeros((10, 10), dtype=np.float32)

    with pytest.raises(TypeError, match="uint8"):
        im.connected_components(mask)  # type: ignore[arg-type]


def test_connected_components_rejects_1d_array() -> None:
    mask = np.zeros(10, dtype=np.uint8)

    with pytest.raises(ValueError, match="2 dimensions"):
        im.connected_components(mask)


def test_connected_components_rejects_invalid_connectivity() -> None:
    mask = _rect_mask(2, 6, 2, 6)

    with pytest.raises(ValueError, match="connectivity"):
        im.connected_components(mask, connectivity=6)  # type: ignore[arg-type]


def test_connected_components_accepts_numpy_integer_connectivity() -> None:
    mask = _rect_mask(2, 6, 2, 6)

    n32, _ = im.connected_components(mask, connectivity=np.int32(4))  # type: ignore[arg-type]
    n64, _ = im.connected_components(mask, connectivity=np.int64(8))  # type: ignore[arg-type]

    assert n32 == 2
    assert n64 == 2


def test_connected_components_rejects_float_connectivity() -> None:
    # 4.0 == 4 in Python, so a bare require_one_of membership check would
    # silently accept it -- verified directly that this previously reached
    # a raw cv2.error instead of a clear validation error.
    mask = _rect_mask(2, 6, 2, 6)

    with pytest.raises(TypeError, match="integer"):
        im.connected_components(mask, connectivity=4.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="integer"):
        im.connected_components(mask, connectivity=np.float32(4))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="integer"):
        im.connected_components(mask, connectivity=True)  # type: ignore[arg-type]


def test_connected_components_with_stats_accepts_numpy_integer_connectivity() -> None:
    mask = _rect_mask(2, 6, 2, 6)

    n, *_ = im.connected_components_with_stats(mask, connectivity=np.int32(8))  # type: ignore[arg-type]

    assert n == 2


def test_connected_components_with_stats_rejects_float_connectivity() -> None:
    mask = _rect_mask(2, 6, 2, 6)

    with pytest.raises(TypeError, match="integer"):
        im.connected_components_with_stats(mask, connectivity=4.0)  # type: ignore[arg-type]


def test_connected_components_with_stats_reads_component_stats_by_label() -> None:
    # Two known squares. Read each component's label from a specific pixel
    # rather than assuming a fixed label numbering -- verified directly
    # against cv2 that these exact stats/centroids result from this mask.
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:4, 2:4] = 255
    mask[6:8, 6:8] = 255

    num_labels, labels, stats, centroids = im.connected_components_with_stats(mask, connectivity=8)

    assert num_labels == 3
    assert stats.shape == (3, 5)
    assert stats.dtype == np.int32
    assert centroids.shape == (3, 2)
    assert centroids.dtype == np.float64

    label_a = labels[3, 3]
    label_b = labels[7, 7]
    np.testing.assert_array_equal(stats[label_a], [2, 2, 2, 2, 4])
    np.testing.assert_array_equal(stats[label_b], [6, 6, 2, 2, 4])
    np.testing.assert_allclose(centroids[label_a], [2.5, 2.5])
    np.testing.assert_allclose(centroids[label_b], [6.5, 6.5])


def test_connected_components_with_stats_all_black_mask() -> None:
    # All background: stats[0] legitimately covers the whole image here --
    # but this is a consequence of every pixel being background, not a
    # general property of label 0 (see the mixed-mask and all-white tests).
    mask = np.zeros((10, 10), dtype=np.uint8)

    num_labels, labels, stats, centroids = im.connected_components_with_stats(mask)

    assert num_labels == 1
    np.testing.assert_array_equal(stats[0], [0, 0, 10, 10, 100])
    np.testing.assert_allclose(centroids[0], [4.5, 4.5])


def test_connected_components_with_stats_background_bbox_is_not_always_whole_image() -> None:
    # Background pixels confined near one corner: background bbox must NOT
    # be asserted as the whole image -- only as covering the specific
    # region the background pixels actually occupy.
    mask = np.full((10, 10), 255, dtype=np.uint8)
    mask[0:3, 0:3] = 0  # small background patch in the corner

    num_labels, labels, stats, _ = im.connected_components_with_stats(mask)

    assert num_labels == 2
    background_label = labels[0, 0]
    x, y, w, h, area = stats[background_label]
    assert (x, y, w, h) == (0, 0, 3, 3)
    assert area == 9


def test_connected_components_with_stats_all_white_mask() -> None:
    # No background pixels at all: OpenCV returns a degenerate sentinel box
    # and a NaN centroid for label 0 -- verified directly, identical on
    # OpenCV 4.13 and 5.0.
    mask = np.full((10, 10), 255, dtype=np.uint8)

    num_labels, labels, stats, centroids = im.connected_components_with_stats(mask)

    assert num_labels == 2
    assert stats[0, 4] == 0  # background area is zero
    assert np.all(np.isnan(centroids[0]))
    foreground_label = labels[0, 0]
    np.testing.assert_array_equal(stats[foreground_label], [0, 0, 10, 10, 100])


def test_connected_components_with_stats_does_not_mutate_input() -> None:
    mask = _rect_mask(2, 6, 2, 6)
    original = mask.copy()

    im.connected_components_with_stats(mask)

    np.testing.assert_array_equal(mask, original)


def test_connected_components_with_stats_rejects_non_uint8_dtype() -> None:
    mask = np.zeros((10, 10), dtype=np.float32)

    with pytest.raises(TypeError, match="uint8"):
        im.connected_components_with_stats(mask)  # type: ignore[arg-type]


def test_distance_transform_computes_distance_to_nearest_zero_pixel() -> None:
    mask = np.zeros((11, 11), dtype=np.uint8)
    mask[5, 5] = 255  # single foreground pixel, surrounded by background

    result = im.distance_transform(mask)

    assert result.shape == mask.shape
    assert result.dtype == np.float32
    # A single isolated foreground pixel is adjacent to background on every
    # side, so its distance to the nearest zero pixel is 1.0.
    assert result[5, 5] == pytest.approx(1.0)
    assert result[6, 6] == 0.0  # background pixels have distance 0


def test_distance_transform_l2_accepts_0_3_and_5() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[3:7, 3:7] = 255

    for mask_size in (0, 3, 5):
        result = im.distance_transform(mask, distance_type="l2", mask_size=mask_size)
        assert result.shape == mask.shape


def test_distance_transform_l1_and_c_accept_only_3() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[3:7, 3:7] = 255

    for distance_type in ("l1", "c"):
        result = im.distance_transform(mask, distance_type=distance_type, mask_size=3)  # type: ignore[arg-type]
        assert result.shape == mask.shape

        with pytest.raises(ValueError, match="mask_size"):
            im.distance_transform(mask, distance_type=distance_type, mask_size=5)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="mask_size"):
            im.distance_transform(mask, distance_type=distance_type, mask_size=0)  # type: ignore[arg-type]


def test_distance_transform_none_mask_size_resolves_per_distance_type() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[3:7, 3:7] = 255

    # "l2" with mask_size=None must match explicit mask_size=5.
    default_l2 = im.distance_transform(mask, distance_type="l2", mask_size=None)
    explicit_l2 = im.distance_transform(mask, distance_type="l2", mask_size=5)
    np.testing.assert_array_equal(default_l2, explicit_l2)

    # "l1"/"c" with mask_size=None must match explicit mask_size=3.
    default_l1 = im.distance_transform(mask, distance_type="l1", mask_size=None)
    explicit_l1 = im.distance_transform(mask, distance_type="l1", mask_size=3)
    np.testing.assert_array_equal(default_l1, explicit_l1)


def test_distance_transform_does_not_mutate_input() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[3:7, 3:7] = 255
    original = mask.copy()

    im.distance_transform(mask)

    np.testing.assert_array_equal(mask, original)


def test_distance_transform_rejects_non_uint8_dtype() -> None:
    mask = np.zeros((10, 10), dtype=np.float32)

    with pytest.raises(TypeError, match="uint8"):
        im.distance_transform(mask)  # type: ignore[arg-type]


def test_distance_transform_rejects_1d_array() -> None:
    mask = np.zeros(10, dtype=np.uint8)

    with pytest.raises(ValueError, match="2 dimensions"):
        im.distance_transform(mask)


def test_distance_transform_rejects_invalid_distance_type() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)

    with pytest.raises(ValueError, match="distance_type"):
        im.distance_transform(mask, distance_type="bogus")  # type: ignore[arg-type]


def test_distance_transform_accepts_numpy_integer_mask_size() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[3:7, 3:7] = 255

    result = im.distance_transform(mask, distance_type="l2", mask_size=np.int32(3))  # type: ignore[arg-type]

    assert result.shape == mask.shape


def test_distance_transform_rejects_float_mask_size() -> None:
    # 3.0 == 3 in Python, so a bare membership check would silently accept
    # it -- verified directly that this previously reached a raw cv2.error.
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[3:7, 3:7] = 255

    with pytest.raises(TypeError, match="integer"):
        im.distance_transform(mask, distance_type="l2", mask_size=3.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="integer"):
        im.distance_transform(mask, distance_type="l2", mask_size=np.float32(3))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="integer"):
        im.distance_transform(mask, distance_type="l2", mask_size=True)  # type: ignore[arg-type]


def test_flood_fill_does_not_mutate_input() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    original = image.copy()

    im.flood_fill(image, (0, 0), (255, 0, 0))

    np.testing.assert_array_equal(image, original)


def test_flood_fill_grayscale_scalar_values() -> None:
    image = np.zeros((10, 10), dtype=np.uint8)

    result = im.flood_fill(image, (0, 0), 200)

    assert result.filled_count == 100
    assert result.image[0, 0] == 200
    assert result.mask.shape == (10, 10)
    assert set(np.unique(result.mask).tolist()) <= {0, 255}
    assert result.bounding_box == im.BoundingBox(0, 0, 10, 10)


def test_flood_fill_bgr_per_channel_values() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    result = im.flood_fill(image, (0, 0), (10, 20, 30))

    assert result.filled_count == 100
    np.testing.assert_array_equal(result.image[0, 0], [10, 20, 30])


def test_flood_fill_result_invariants() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[:, 5:] = 100  # confine the fill to the left half

    result = im.flood_fill(image, (0, 0), (255, 0, 0))

    assert result.filled_count == np.count_nonzero(result.mask)
    assert set(np.unique(result.mask).tolist()) <= {0, 255}


def test_flood_fill_floating_range_vs_fixed_range_differ() -> None:
    # A step-5 gradient with loDiff=upDiff=10: floating range keeps
    # flooding because each consecutive step is within the diff, but fixed
    # range compares every pixel back to the seed's original value and
    # stops once the cumulative difference exceeds 10 -- verified directly,
    # identical on OpenCV 4.13 and 5.0 (floating fills all 20 pixels, fixed
    # fills only 3).
    image = np.zeros((1, 20), dtype=np.uint8)
    for i in range(20):
        image[0, i] = i * 5

    floating = im.flood_fill(image, (0, 0), 255, lo_diff=10, up_diff=10, fixed_range=False)
    fixed = im.flood_fill(image, (0, 0), 255, lo_diff=10, up_diff=10, fixed_range=True)

    assert floating.filled_count == 20
    assert fixed.filled_count == 3


def test_flood_fill_accepts_numpy_integer_seed_point() -> None:
    image = np.zeros((10, 10), dtype=np.uint8)

    result = im.flood_fill(image, (np.int32(0), np.int32(0)), 200)  # type: ignore[arg-type]

    assert result.filled_count == 100


def test_flood_fill_accepts_numpy_integer_connectivity() -> None:
    image = np.zeros((10, 10), dtype=np.uint8)

    result = im.flood_fill(image, (0, 0), 200, connectivity=np.int32(8))  # type: ignore[arg-type]

    assert result.filled_count == 100


def test_flood_fill_rejects_float_connectivity() -> None:
    # 4.0 == 4 in Python, so a bare membership check would silently accept
    # it and reach `connectivity | (255 << 8)` -- verified directly that
    # this previously raised an unrelated TypeError from the `|` operator.
    image = np.zeros((10, 10), dtype=np.uint8)

    with pytest.raises(TypeError, match="integer"):
        im.flood_fill(image, (0, 0), 200, connectivity=4.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="integer"):
        im.flood_fill(image, (0, 0), 200, connectivity=True)  # type: ignore[arg-type]


def test_flood_fill_rejects_out_of_bounds_seed() -> None:
    image = np.zeros((10, 10), dtype=np.uint8)

    with pytest.raises(ValueError, match="seed_point"):
        im.flood_fill(image, (100, 100), 200)


def test_flood_fill_rejects_bool_seed_point_element() -> None:
    image = np.zeros((10, 10), dtype=np.uint8)

    with pytest.raises(TypeError, match="integer"):
        im.flood_fill(image, (True, 0), 200)  # type: ignore[arg-type]


def test_flood_fill_rejects_wrong_new_value_element_count() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="new_value"):
        im.flood_fill(image, (0, 0), (255, 0))  # type: ignore[arg-type]


def test_flood_fill_rejects_bytes_new_value() -> None:
    # bytes/bytearray iterate to plain ints (b"x" -> 120), so without an
    # explicit rejection a bytes value of the right length would silently
    # pass through as if it were a sequence of numbers -- verified directly
    # that this previously filled the image with 120, no error at all.
    image = np.zeros((10, 10), dtype=np.uint8)

    with pytest.raises(TypeError, match="new_value"):
        im.flood_fill(image, (0, 0), b"x")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="new_value"):
        im.flood_fill(image, (0, 0), bytearray(b"x"))  # type: ignore[arg-type]


def test_flood_fill_rejects_str_new_value() -> None:
    image = np.zeros((10, 10), dtype=np.uint8)

    with pytest.raises(TypeError, match="new_value"):
        im.flood_fill(image, (0, 0), "1")  # type: ignore[arg-type]


def test_flood_fill_rejects_memoryview_new_value() -> None:
    image = np.zeros((10, 10), dtype=np.uint8)

    with pytest.raises(TypeError, match="new_value"):
        im.flood_fill(image, (0, 0), memoryview(b"x"))  # type: ignore[arg-type]


def test_flood_fill_rejects_bytes_lo_diff_and_up_diff() -> None:
    image = np.zeros((10, 10), dtype=np.uint8)

    with pytest.raises(TypeError, match="lo_diff"):
        im.flood_fill(image, (0, 0), 200, lo_diff=b"x")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="up_diff"):
        im.flood_fill(image, (0, 0), 200, up_diff=b"x")  # type: ignore[arg-type]


def test_flood_fill_rejects_out_of_range_new_value_for_uint8() -> None:
    image = np.zeros((10, 10), dtype=np.uint8)

    with pytest.raises(ValueError, match="new_value"):
        im.flood_fill(image, (0, 0), 300)


def test_flood_fill_accepts_integer_valued_float_new_value_for_uint8() -> None:
    image = np.zeros((10, 10), dtype=np.uint8)

    result = im.flood_fill(image, (0, 0), 7.0)

    assert result.image[0, 0] == 7


def test_flood_fill_rejects_fractional_new_value_for_uint8() -> None:
    # Verified directly: raw cv2.floodFill silently rounds a fractional
    # new_value instead of rejecting it (0.5 -> 0, 254.5 -> 254).
    image = np.zeros((10, 10), dtype=np.uint8)

    with pytest.raises(ValueError, match="new_value"):
        im.flood_fill(image, (0, 0), 0.5)
    with pytest.raises(ValueError, match="new_value"):
        im.flood_fill(image, (0, 0), 254.5)
    with pytest.raises(ValueError, match="new_value"):
        im.flood_fill(image, (0, 0), 7.5)


def test_flood_fill_accepts_float32_max_new_value() -> None:
    image = np.zeros((10, 10), dtype=np.float32)
    max_value = float(np.finfo(np.float32).max)

    result = im.flood_fill(image, (0, 0), max_value)

    assert np.isfinite(result.image[0, 0])


def test_flood_fill_rejects_new_value_overflowing_float32() -> None:
    # Verified directly: a finite Python float comfortably inside float64's
    # range (3.5e38 > float32's max of ~3.4028235e38) silently produces inf
    # in the filled result instead of raising.
    image = np.zeros((10, 10), dtype=np.float32)

    with pytest.raises(ValueError, match="new_value"):
        im.flood_fill(image, (0, 0), 3.5e38)


def test_flood_fill_rejects_negative_diff() -> None:
    image = np.zeros((10, 10), dtype=np.uint8)

    with pytest.raises(ValueError, match="non-negative"):
        im.flood_fill(image, (0, 0), 200, lo_diff=-1)


def test_flood_fill_rejects_unsupported_dtype() -> None:
    image = np.zeros((10, 10), dtype=np.float64)

    with pytest.raises(TypeError, match="dtype"):
        im.flood_fill(image, (0, 0), 200.0)  # type: ignore[arg-type]


def test_flood_fill_rejects_four_channel_image() -> None:
    image = np.zeros((10, 10, 4), dtype=np.uint8)

    with pytest.raises(ValueError, match="channels"):
        im.flood_fill(image, (0, 0), (255, 0, 0, 0))  # type: ignore[arg-type]
