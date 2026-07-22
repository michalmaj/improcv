import cv2
import numpy as np
import pytest

import improcv as im
from improcv.contours import BoundingBox
from improcv.detectors import (
    MSERRegion,
    _normalize_integral_param,
    _normalize_mser_bbox,
    _normalize_mser_region_points,
    _require_valid_detector_image,
    _require_valid_keypoints,
    _require_valid_mser_result,
)


def _noise(shape: tuple[int, int] = (100, 100), seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, shape, dtype=np.uint8)


def _mask_hit(mask: np.ndarray, kp: cv2.KeyPoint) -> bool:
    x = int(round(kp.pt[0]))
    y = int(round(kp.pt[1]))
    return bool(mask[y, x] != 0)


# --- _require_valid_detector_image ---


def test_require_valid_detector_image_accepts_grayscale() -> None:
    _require_valid_detector_image(_noise())


def test_require_valid_detector_image_accepts_bgr() -> None:
    _require_valid_detector_image(cv2.cvtColor(_noise(), cv2.COLOR_GRAY2BGR))


def test_require_valid_detector_image_accepts_bgra() -> None:
    _require_valid_detector_image(cv2.cvtColor(_noise(), cv2.COLOR_GRAY2BGRA))


def test_require_valid_detector_image_rejects_two_channels() -> None:
    with pytest.raises(ValueError, match="channel"):
        _require_valid_detector_image(np.zeros((10, 10, 2), dtype=np.uint8))


def test_require_valid_detector_image_rejects_non_uint8() -> None:
    with pytest.raises(TypeError, match="dtype"):
        _require_valid_detector_image(_noise().astype(np.float32))


# --- _normalize_integral_param ---


def test_normalize_integral_param_accepts_plain_int() -> None:
    assert _normalize_integral_param(5, "x") == 5


def test_normalize_integral_param_accepts_numpy_int() -> None:
    result = _normalize_integral_param(np.int32(5), "x")
    assert result == 5
    assert type(result) is int


def test_normalize_integral_param_rejects_bool() -> None:
    with pytest.raises(TypeError, match="integ"):
        _normalize_integral_param(True, "x")


def test_normalize_integral_param_rejects_float() -> None:
    with pytest.raises(TypeError, match="integ"):
        _normalize_integral_param(1.5, "x")


@pytest.mark.parametrize("value", [2**31, np.int64(2**31)])
def test_normalize_integral_param_rejects_outside_int32(value: object) -> None:
    with pytest.raises(ValueError, match="int32"):
        _normalize_integral_param(value, "x")


# --- _require_valid_keypoints ---


def test_require_valid_keypoints_accepts_real_tuple() -> None:
    detector = cv2.FastFeatureDetector.create()
    raw = detector.detect(_noise(), None)
    result = _require_valid_keypoints(raw)
    assert isinstance(result, list)
    assert all(isinstance(kp, cv2.KeyPoint) for kp in result)


def test_require_valid_keypoints_rejects_non_sequence() -> None:
    with pytest.raises(RuntimeError, match="unexpected"):
        _require_valid_keypoints(None)  # type: ignore[arg-type]


def test_require_valid_keypoints_rejects_wrong_element_type() -> None:
    with pytest.raises(RuntimeError, match="unexpected"):
        _require_valid_keypoints([1, 2, 3])  # type: ignore[list-item]


def test_require_valid_keypoints_rejects_non_finite_pt() -> None:
    bad_kp = cv2.KeyPoint(float("nan"), 0.0, 1.0)
    with pytest.raises(RuntimeError, match="unexpected"):
        _require_valid_keypoints([bad_kp])


def test_require_valid_keypoints_rejects_negative_size() -> None:
    bad_kp = cv2.KeyPoint(0.0, 0.0, -1.0)
    with pytest.raises(RuntimeError, match="unexpected"):
        _require_valid_keypoints([bad_kp])


# --- detect_fast_keypoints ---


def test_detect_fast_keypoints_finds_keypoints_in_noise() -> None:
    keypoints = im.detect_fast_keypoints(_noise())

    assert len(keypoints) > 0
    assert all(isinstance(kp, cv2.KeyPoint) for kp in keypoints)


def test_detect_fast_keypoints_blank_image_returns_empty() -> None:
    image = np.full((100, 100), 128, dtype=np.uint8)

    assert im.detect_fast_keypoints(image) == []


def test_detect_fast_keypoints_rejects_two_channels() -> None:
    with pytest.raises(ValueError, match="channel"):
        im.detect_fast_keypoints(np.zeros((10, 10, 2), dtype=np.uint8))


def test_detect_fast_keypoints_rejects_non_uint8() -> None:
    with pytest.raises(TypeError, match="dtype"):
        im.detect_fast_keypoints(_noise().astype(np.float32))  # type: ignore[arg-type]


@pytest.mark.parametrize("threshold", [-1, 256])
def test_detect_fast_keypoints_rejects_out_of_range_threshold(threshold: int) -> None:
    with pytest.raises(ValueError, match="threshold"):
        im.detect_fast_keypoints(_noise(), threshold=threshold)


@pytest.mark.parametrize("threshold", [2**31, np.int64(2**31)])
def test_detect_fast_keypoints_rejects_huge_threshold(threshold: object) -> None:
    with pytest.raises(ValueError, match="int32"):
        im.detect_fast_keypoints(_noise(), threshold=threshold)  # type: ignore[arg-type]


def test_detect_fast_keypoints_rejects_bool_threshold() -> None:
    with pytest.raises(TypeError, match="integ"):
        im.detect_fast_keypoints(_noise(), threshold=True)  # type: ignore[arg-type]


def test_detect_fast_keypoints_rejects_float_threshold() -> None:
    with pytest.raises(TypeError, match="integ"):
        im.detect_fast_keypoints(_noise(), threshold=1.5)  # type: ignore[arg-type]


def test_detect_fast_keypoints_rejects_non_bool_nonmax_suppression() -> None:
    with pytest.raises(TypeError, match="bool"):
        im.detect_fast_keypoints(_noise(), nonmax_suppression=1)  # type: ignore[arg-type]


def test_detect_fast_keypoints_rejects_invalid_fast_type() -> None:
    with pytest.raises(ValueError, match="fast_type"):
        im.detect_fast_keypoints(_noise(), fast_type="bogus")  # type: ignore[arg-type]


def test_detect_fast_keypoints_respects_mask() -> None:
    image = _noise()
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[:50, :] = 255

    keypoints = im.detect_fast_keypoints(image, mask=mask)

    assert len(keypoints) > 0
    assert all(_mask_hit(mask, kp) for kp in keypoints)


def test_detect_fast_keypoints_does_not_mutate_input() -> None:
    image = _noise()
    before = image.copy()

    im.detect_fast_keypoints(image)

    assert np.array_equal(image, before)


def test_detect_fast_keypoints_works_with_draw_keypoints() -> None:
    image = cv2.cvtColor(_noise(), cv2.COLOR_GRAY2BGR)
    keypoints = im.detect_fast_keypoints(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    annotated = cv2.drawKeypoints(image, keypoints, None)  # type: ignore[call-overload]

    assert annotated.shape == image.shape


# --- detect_blob_keypoints ---


def _circles_image(centers: list[tuple[int, int]], radius: int = 15) -> np.ndarray:
    image = np.full((200, 200), 255, dtype=np.uint8)
    for cx, cy in centers:
        cv2.circle(image, (cx, cy), radius, 0, -1)
    return image


def test_detect_blob_keypoints_finds_circles() -> None:
    centers = [(50, 50), (150, 150)]
    image = _circles_image(centers)

    keypoints = im.detect_blob_keypoints(image)

    assert len(keypoints) == 2
    found_centers = sorted((round(kp.pt[0]), round(kp.pt[1])) for kp in keypoints)
    assert found_centers == sorted(centers)


def test_detect_blob_keypoints_blank_image_returns_empty() -> None:
    image = np.full((100, 100), 255, dtype=np.uint8)

    assert im.detect_blob_keypoints(image) == []


def test_detect_blob_keypoints_rejects_two_channels() -> None:
    with pytest.raises(ValueError, match="channel"):
        im.detect_blob_keypoints(np.zeros((10, 10, 2), dtype=np.uint8))


def test_detect_blob_keypoints_rejects_non_uint8() -> None:
    with pytest.raises(TypeError, match="dtype"):
        im.detect_blob_keypoints(_circles_image([(50, 50)]).astype(np.float32))  # type: ignore[arg-type]


def test_detect_blob_keypoints_rejects_wrong_type_params() -> None:
    with pytest.raises(TypeError, match="Params"):
        im.detect_blob_keypoints(_circles_image([(50, 50)]), params="not params")  # type: ignore[arg-type]


def test_detect_blob_keypoints_rejects_invalid_params_configuration() -> None:
    params = cv2.SimpleBlobDetector.Params()
    params.thresholdStep = 0

    with pytest.raises(ValueError, match="invalid"):
        im.detect_blob_keypoints(_circles_image([(50, 50)]), params=params)


def test_detect_blob_keypoints_custom_params_changes_result() -> None:
    image = _circles_image([(50, 50), (150, 150)], radius=15)

    default_result = im.detect_blob_keypoints(image)

    params = cv2.SimpleBlobDetector.Params()
    params.filterByArea = True
    params.minArea = 100000  # larger than any blob present
    params.maxArea = 200000

    restricted_result = im.detect_blob_keypoints(image, params=params)

    assert len(restricted_result) < len(default_result)


def test_detect_blob_keypoints_respects_mask() -> None:
    image = _circles_image([(50, 50), (150, 150)])
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[:100, :] = 255

    keypoints = im.detect_blob_keypoints(image, mask=mask)

    assert len(keypoints) == 1
    assert _mask_hit(mask, keypoints[0])


def test_detect_blob_keypoints_does_not_mutate_input() -> None:
    image = _circles_image([(50, 50)])
    before = image.copy()

    im.detect_blob_keypoints(image)

    assert np.array_equal(image, before)


# --- MSER helpers: _normalize_mser_region_points ---


def test_normalize_mser_region_points_accepts_int32_array() -> None:
    region = np.array([[1, 2], [3, 4]], dtype=np.int32)

    result = _normalize_mser_region_points(region, 0)

    assert np.array_equal(result, region)
    assert result.dtype == np.int32


def test_normalize_mser_region_points_rejects_wrong_shape() -> None:
    region = np.array([1, 2, 3], dtype=np.int32)

    with pytest.raises(RuntimeError, match="unexpected"):
        _normalize_mser_region_points(region, 0)


def test_normalize_mser_region_points_rejects_empty() -> None:
    region = np.zeros((0, 2), dtype=np.int32)

    with pytest.raises(RuntimeError, match="unexpected"):
        _normalize_mser_region_points(region, 0)


def test_normalize_mser_region_points_rejects_unexpected_dtype() -> None:
    region = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)

    with pytest.raises(RuntimeError, match="unexpected"):
        _normalize_mser_region_points(region, 0)


def test_normalize_mser_region_points_accepts_valid_object_dtype() -> None:
    region = np.array([[1, 2], [3, 4]], dtype=object)

    result = _normalize_mser_region_points(region, 0)

    assert result.dtype == np.int32
    assert np.array_equal(result, np.array([[1, 2], [3, 4]], dtype=np.int32))


def test_normalize_mser_region_points_rejects_object_dtype_with_non_integral_element() -> None:
    region = np.array([[1, 2.5], [3, 4]], dtype=object)

    with pytest.raises(RuntimeError, match="unexpected"):
        _normalize_mser_region_points(region, 0)


def test_normalize_mser_region_points_rejects_object_dtype_with_bool_element() -> None:
    region = np.array([[1, True], [3, 4]], dtype=object)

    with pytest.raises(RuntimeError, match="unexpected"):
        _normalize_mser_region_points(region, 0)


def test_normalize_mser_region_points_rejects_object_dtype_out_of_int32_range() -> None:
    region = np.array([[1, 2**31], [3, 4]], dtype=object)

    with pytest.raises(RuntimeError, match="unexpected"):
        _normalize_mser_region_points(region, 0)


# --- MSER helpers: _normalize_mser_bbox ---


def test_normalize_mser_bbox_accepts_valid_row() -> None:
    row = np.array([1, 2, 3, 4], dtype=np.int32)

    result = _normalize_mser_bbox(row, 0)

    assert result == BoundingBox(1, 2, 3, 4)


def test_normalize_mser_bbox_rejects_non_positive_width() -> None:
    row = np.array([1, 2, 0, 4], dtype=np.int32)

    with pytest.raises(RuntimeError, match="unexpected"):
        _normalize_mser_bbox(row, 0)


def test_normalize_mser_bbox_rejects_non_positive_height() -> None:
    row = np.array([1, 2, 3, 0], dtype=np.int32)

    with pytest.raises(RuntimeError, match="unexpected"):
        _normalize_mser_bbox(row, 0)


# --- MSER helpers: _require_valid_mser_result ---


def test_require_valid_mser_result_accepts_empty_variant() -> None:
    assert _require_valid_mser_result((), ()) == []


def test_require_valid_mser_result_rejects_empty_regions_with_nonempty_bboxes() -> None:
    with pytest.raises(RuntimeError, match="unexpected"):
        _require_valid_mser_result((), np.zeros((1, 4), dtype=np.int32))


def test_require_valid_mser_result_rejects_mismatched_counts() -> None:
    regions = (np.array([[0, 0], [1, 1]], dtype=np.int32),)
    bboxes = np.zeros((2, 4), dtype=np.int32)

    with pytest.raises(RuntimeError, match="unexpected"):
        _require_valid_mser_result(regions, bboxes)


def test_require_valid_mser_result_rejects_non_sequence_regions() -> None:
    with pytest.raises(RuntimeError, match="unexpected"):
        _require_valid_mser_result(None, ())  # type: ignore[arg-type]


def test_require_valid_mser_result_rejects_bbox_not_matching_region() -> None:
    regions = (np.array([[0, 0], [10, 10]], dtype=np.int32),)
    bboxes = np.array([[0, 0, 999, 999]], dtype=np.int32)

    with pytest.raises(RuntimeError, match="unexpected"):
        _require_valid_mser_result(regions, bboxes)


def test_require_valid_mser_result_accepts_consistent_region_and_bbox() -> None:
    region = np.array([[2, 3], [5, 3], [5, 8], [2, 8]], dtype=np.int32)
    box = BoundingBox(2, 3, 4, 6)
    bboxes = np.array([[box.x, box.y, box.width, box.height]], dtype=np.int32)

    result = _require_valid_mser_result((region,), bboxes)

    assert result == [MSERRegion(points=region, bounding_box=box)]


# --- detect_mser_regions ---


def _duplicated_patch_image() -> np.ndarray:
    rng = np.random.default_rng(0)
    patch = rng.integers(0, 256, (60, 60), dtype=np.uint8)
    image = np.full((400, 400), 128, dtype=np.uint8)
    for y, x in [(20, 20), (20, 120), (20, 220), (120, 20), (120, 120), (220, 220)]:
        image[y : y + 60, x : x + 60] = patch
    return image


def test_detect_mser_regions_finds_regions_in_noise() -> None:
    image = _noise((300, 300))

    regions = im.detect_mser_regions(image)

    assert len(regions) > 0
    for region in regions:
        assert isinstance(region, im.MSERRegion)
        assert region.points.shape[1] == 2
        assert region.points.dtype == np.int32
        x_min, y_min = region.points[:, 0].min(), region.points[:, 1].min()
        x_max, y_max = region.points[:, 0].max(), region.points[:, 1].max()
        assert region.bounding_box == BoundingBox(
            int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)
        )


def test_detect_mser_regions_deterministic_fixture_is_non_empty_and_consistent() -> None:
    image = _duplicated_patch_image()

    regions = im.detect_mser_regions(image)

    assert len(regions) > 0
    for region in regions:
        x_min, y_min = region.points[:, 0].min(), region.points[:, 1].min()
        x_max, y_max = region.points[:, 0].max(), region.points[:, 1].max()
        assert region.bounding_box.x == x_min
        assert region.bounding_box.y == y_min
        assert region.bounding_box.x + region.bounding_box.width - 1 == x_max
        assert region.bounding_box.y + region.bounding_box.height - 1 == y_max


def test_detect_mser_regions_uniform_image_with_tiny_max_area_returns_empty() -> None:
    # A uniform image forms exactly one maximally-stable region: the whole
    # image (verified directly, and confirmed version-dependent otherwise --
    # OpenCV 4.13 finds this single region even for blurred noise where
    # OpenCV 5.0 finds none, so the empty case is pinned via max_area
    # instead of relying on incidental image content). max_area smaller
    # than the image's own area filters that one region out on both
    # versions.
    image = np.full((100, 100), 128, dtype=np.uint8)

    assert im.detect_mser_regions(image, min_area=1, max_area=2) == []


def test_detect_mser_regions_rejects_non_uint8() -> None:
    with pytest.raises(TypeError, match="dtype"):
        im.detect_mser_regions(_noise((50, 50)).astype(np.float32))  # type: ignore[arg-type]


def test_detect_mser_regions_rejects_two_channels() -> None:
    with pytest.raises(ValueError, match="channel"):
        im.detect_mser_regions(np.zeros((50, 50, 2), dtype=np.uint8))


@pytest.mark.parametrize("shape", [(1, 10), (2, 10), (10, 1), (10, 2)])
def test_detect_mser_regions_rejects_too_small_image(shape: tuple[int, int]) -> None:
    image = np.random.default_rng(0).integers(0, 256, shape, dtype=np.uint8)

    with pytest.raises(ValueError, match="3x3"):
        im.detect_mser_regions(image)


def test_detect_mser_regions_accepts_3x3_image() -> None:
    image = np.random.default_rng(0).integers(0, 256, (3, 3), dtype=np.uint8)

    assert im.detect_mser_regions(image) == []


@pytest.mark.parametrize("delta", [0, -1])
def test_detect_mser_regions_rejects_non_positive_delta(delta: int) -> None:
    with pytest.raises(ValueError, match="delta"):
        im.detect_mser_regions(_noise((50, 50)), delta=delta)


@pytest.mark.parametrize("min_area", [0, -1])
def test_detect_mser_regions_rejects_non_positive_min_area(min_area: int) -> None:
    with pytest.raises(ValueError, match="min_area"):
        im.detect_mser_regions(_noise((50, 50)), min_area=min_area)


@pytest.mark.parametrize("max_area", [0, -1])
def test_detect_mser_regions_rejects_non_positive_max_area(max_area: int) -> None:
    with pytest.raises(ValueError, match="max_area"):
        im.detect_mser_regions(_noise((50, 50)), max_area=max_area)


def test_detect_mser_regions_rejects_min_area_not_less_than_max_area() -> None:
    with pytest.raises(ValueError, match="min_area"):
        im.detect_mser_regions(_noise((50, 50)), min_area=100, max_area=100)


@pytest.mark.parametrize("value", [2**31, np.int64(2**31)])
def test_detect_mser_regions_rejects_huge_delta(value: object) -> None:
    with pytest.raises(ValueError, match="int32"):
        im.detect_mser_regions(_noise((50, 50)), delta=value)  # type: ignore[arg-type]


def test_detect_mser_regions_rejects_bool_min_area() -> None:
    with pytest.raises(TypeError, match="integ"):
        im.detect_mser_regions(_noise((50, 50)), min_area=True)  # type: ignore[arg-type]


def test_detect_mser_regions_does_not_mutate_input() -> None:
    image = _noise((300, 300))
    before = image.copy()

    im.detect_mser_regions(image)

    assert np.array_equal(image, before)


def test_detect_mser_regions_rejects_bad_raw_result(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeMSER:
        def detectRegions(self, image: np.ndarray) -> tuple[object, object]:
            return (np.array([[1, 2], [3, 4]], dtype=np.int32),), np.zeros((2, 4), dtype=np.int32)

    monkeypatch.setattr(cv2.MSER, "create", staticmethod(lambda **kwargs: _FakeMSER()))

    with pytest.raises(RuntimeError, match="unexpected"):
        im.detect_mser_regions(_noise((50, 50)))
