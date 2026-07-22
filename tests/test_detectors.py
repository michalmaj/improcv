import cv2
import numpy as np
import pytest

import improcv as im
from improcv.detectors import (
    _normalize_integral_param,
    _require_valid_detector_image,
    _require_valid_keypoints,
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
