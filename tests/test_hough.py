import cv2
import numpy as np
import pytest

import improcv as im


def _lines_image() -> np.ndarray:
    image = np.zeros((200, 200), dtype=np.uint8)
    cv2.line(image, (10, 10), (190, 10), 255, 2)
    cv2.line(image, (10, 10), (10, 190), 255, 2)
    cv2.line(image, (20, 180), (180, 20), 255, 2)
    return image


def test_hough_lines_finds_real_lines() -> None:
    image = _lines_image()

    lines = im.hough_lines(image, threshold=80)

    assert len(lines) > 0
    for line in lines:
        assert np.isfinite(line.rho)
        assert np.isfinite(line.theta)


def test_hough_lines_blank_image_returns_empty() -> None:
    image = np.zeros((200, 200), dtype=np.uint8)

    lines = im.hough_lines(image, threshold=80)

    assert lines == []


@pytest.mark.parametrize("rho", [0.0, -1.0])
def test_hough_lines_rejects_non_positive_rho(rho: float) -> None:
    image = _lines_image()

    with pytest.raises(ValueError, match="rho"):
        im.hough_lines(image, threshold=80, rho=rho)


@pytest.mark.parametrize("theta", [0.0, -np.pi / 180])
def test_hough_lines_rejects_non_positive_theta(theta: float) -> None:
    image = _lines_image()

    with pytest.raises(ValueError, match="theta"):
        im.hough_lines(image, threshold=80, theta=theta)


@pytest.mark.parametrize("threshold", [0, -5])
def test_hough_lines_rejects_non_positive_threshold(threshold: int) -> None:
    image = _lines_image()

    with pytest.raises(ValueError, match="positive"):
        im.hough_lines(image, threshold=threshold)


def test_hough_lines_rejects_float_threshold() -> None:
    image = _lines_image()

    with pytest.raises(TypeError, match="integer"):
        im.hough_lines(image, threshold=80.0)  # type: ignore[arg-type]


@pytest.mark.parametrize("threshold", [2**31, 2**63])
def test_hough_lines_rejects_threshold_above_int32(threshold: int) -> None:
    image = _lines_image()

    with pytest.raises(ValueError, match="int32"):
        im.hough_lines(image, threshold=threshold)


def test_hough_lines_rejects_non_uint8_dtype() -> None:
    image = _lines_image().astype(np.float32)

    with pytest.raises(TypeError, match="dtype"):
        im.hough_lines(image, threshold=80)  # type: ignore[arg-type]


def test_hough_lines_rejects_three_channel_image() -> None:
    image = cv2.cvtColor(_lines_image(), cv2.COLOR_GRAY2BGR)

    with pytest.raises(ValueError, match="dimensions"):
        im.hough_lines(image, threshold=80)  # type: ignore[arg-type]


def test_hough_lines_does_not_mutate_input() -> None:
    image = _lines_image()
    before = image.copy()

    im.hough_lines(image, threshold=80)

    assert np.array_equal(image, before)


def test_hough_lines_rejects_bad_raw_result_from_matcher(monkeypatch: pytest.MonkeyPatch) -> None:
    image = _lines_image()
    bad_result = np.zeros((3, 1, 3), dtype=np.float32)  # 3 fields instead of 2
    monkeypatch.setattr(cv2, "HoughLines", lambda *args, **kwargs: bad_result)

    with pytest.raises(RuntimeError, match="unexpected"):
        im.hough_lines(image, threshold=80)


def test_hough_line_segments_finds_real_segments() -> None:
    image = _lines_image()

    segments = im.hough_line_segments(image, threshold=50, min_line_length=30, max_line_gap=10)

    assert len(segments) > 0
    for segment in segments:
        assert isinstance(segment.x1, int)
        assert isinstance(segment.y1, int)
        assert isinstance(segment.x2, int)
        assert isinstance(segment.y2, int)


def test_hough_line_segments_blank_image_returns_empty() -> None:
    image = np.zeros((200, 200), dtype=np.uint8)

    segments = im.hough_line_segments(image, threshold=50)

    assert segments == []


@pytest.mark.parametrize("rho", [0.0, -1.0])
def test_hough_line_segments_rejects_non_positive_rho(rho: float) -> None:
    image = _lines_image()

    with pytest.raises(ValueError, match="rho"):
        im.hough_line_segments(image, threshold=50, rho=rho)


@pytest.mark.parametrize("theta", [0.0, -np.pi / 180])
def test_hough_line_segments_rejects_non_positive_theta(theta: float) -> None:
    image = _lines_image()

    with pytest.raises(ValueError, match="theta"):
        im.hough_line_segments(image, threshold=50, theta=theta)


@pytest.mark.parametrize("threshold", [0, -5])
def test_hough_line_segments_rejects_non_positive_threshold(threshold: int) -> None:
    image = _lines_image()

    with pytest.raises(ValueError, match="positive"):
        im.hough_line_segments(image, threshold=threshold)


def test_hough_line_segments_rejects_float_threshold() -> None:
    image = _lines_image()

    with pytest.raises(TypeError, match="integer"):
        im.hough_line_segments(image, threshold=50.0)  # type: ignore[arg-type]


@pytest.mark.parametrize("threshold", [2**31, 2**63])
def test_hough_line_segments_rejects_threshold_above_int32(threshold: int) -> None:
    image = _lines_image()

    with pytest.raises(ValueError, match="int32"):
        im.hough_line_segments(image, threshold=threshold)


@pytest.mark.parametrize("min_line_length", [-1.0, -0.001])
def test_hough_line_segments_rejects_negative_min_line_length(min_line_length: float) -> None:
    image = _lines_image()

    with pytest.raises(ValueError, match="min_line_length"):
        im.hough_line_segments(image, threshold=50, min_line_length=min_line_length)


@pytest.mark.parametrize("max_line_gap", [-1.0, -0.001])
def test_hough_line_segments_rejects_negative_max_line_gap(max_line_gap: float) -> None:
    image = _lines_image()

    with pytest.raises(ValueError, match="max_line_gap"):
        im.hough_line_segments(image, threshold=50, max_line_gap=max_line_gap)


def test_hough_line_segments_rejects_non_uint8_dtype() -> None:
    image = _lines_image().astype(np.float32)

    with pytest.raises(TypeError, match="dtype"):
        im.hough_line_segments(image, threshold=50)  # type: ignore[arg-type]


def test_hough_line_segments_rejects_three_channel_image() -> None:
    image = cv2.cvtColor(_lines_image(), cv2.COLOR_GRAY2BGR)

    with pytest.raises(ValueError, match="dimensions"):
        im.hough_line_segments(image, threshold=50)  # type: ignore[arg-type]


def test_hough_line_segments_does_not_mutate_input() -> None:
    image = _lines_image()
    before = image.copy()

    im.hough_line_segments(image, threshold=50)

    assert np.array_equal(image, before)


def _circles_image() -> np.ndarray:
    image = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(image, (60, 60), 30, 255, 2)
    cv2.circle(image, (140, 140), 20, 255, 2)
    return cv2.GaussianBlur(image, (9, 9), 2)


def test_hough_circles_finds_real_circles() -> None:
    image = _circles_image()

    circles = im.hough_circles(image, min_dist=20, param2=30)

    assert len(circles) > 0
    for circle in circles:
        assert np.isfinite(circle.x)
        assert np.isfinite(circle.y)
        assert np.isfinite(circle.radius)


def test_hough_circles_blank_image_returns_empty() -> None:
    image = np.zeros((200, 200), dtype=np.uint8)

    circles = im.hough_circles(image, min_dist=20, param2=30)

    assert circles == []


@pytest.mark.parametrize("dp", [0.0, -1.0])
def test_hough_circles_rejects_non_positive_dp(dp: float) -> None:
    image = _circles_image()

    with pytest.raises(ValueError, match="dp"):
        im.hough_circles(image, min_dist=20, dp=dp)


@pytest.mark.parametrize("min_dist", [0.0, -1.0])
def test_hough_circles_rejects_non_positive_min_dist(min_dist: float) -> None:
    image = _circles_image()

    with pytest.raises(ValueError, match="min_dist"):
        im.hough_circles(image, min_dist=min_dist)


@pytest.mark.parametrize("param1", [0.0, -1.0])
def test_hough_circles_rejects_non_positive_param1(param1: float) -> None:
    image = _circles_image()

    with pytest.raises(ValueError, match="param1"):
        im.hough_circles(image, min_dist=20, param1=param1)


def test_hough_circles_param2_none_matches_explicit_default_for_gradient() -> None:
    image = _circles_image()

    default_result = im.hough_circles(image, min_dist=20, method="gradient")
    explicit_result = im.hough_circles(image, min_dist=20, method="gradient", param2=100.0)

    assert default_result == explicit_result


def test_hough_circles_param2_none_matches_explicit_default_for_gradient_alt() -> None:
    image = _circles_image()

    default_result = im.hough_circles(image, min_dist=20, method="gradient_alt", dp=1.5, param1=300)
    explicit_result = im.hough_circles(
        image, min_dist=20, method="gradient_alt", dp=1.5, param1=300, param2=0.9
    )

    assert default_result == explicit_result


def test_hough_circles_rejects_gradient_default_param2_for_gradient_alt() -> None:
    image = _circles_image()

    with pytest.raises(ValueError, match="param2"):
        im.hough_circles(image, min_dist=20, method="gradient_alt", param2=100.0)


def test_hough_circles_accepts_small_param2_for_gradient() -> None:
    image = _circles_image()

    # No upper-bound-of-1.0 constraint for method="gradient".
    circles = im.hough_circles(image, min_dist=20, method="gradient", param2=0.5)

    assert isinstance(circles, list)


def test_hough_circles_rejects_negative_min_radius() -> None:
    image = _circles_image()

    with pytest.raises(ValueError, match="min_radius"):
        im.hough_circles(image, min_dist=20, param2=30, min_radius=-1)


def test_hough_circles_negative_max_radius_gradient_is_centers_only() -> None:
    image = _circles_image()

    circles = im.hough_circles(
        image, min_dist=20, method="gradient", param2=30, min_radius=10, max_radius=-1
    )

    assert len(circles) > 0
    for circle in circles:
        assert circle.radius == 0.0


def test_hough_circles_rejects_negative_max_radius_for_gradient_alt() -> None:
    image = _circles_image()

    with pytest.raises(ValueError, match="max_radius"):
        im.hough_circles(
            image, min_dist=20, method="gradient_alt", dp=1.5, param1=300, max_radius=-1
        )


@pytest.mark.parametrize("min_radius,max_radius", [(30, 30), (30, 10), (50, 5)])
@pytest.mark.parametrize("method", ["gradient", "gradient_alt"])
def test_hough_circles_rejects_max_radius_not_greater_than_min_radius(
    method: str, min_radius: int, max_radius: int
) -> None:
    image = _circles_image()
    kwargs = {"dp": 1.5, "param1": 300.0} if method == "gradient_alt" else {}

    with pytest.raises(ValueError, match="max_radius"):
        im.hough_circles(
            image,
            min_dist=20,
            method=method,  # type: ignore[arg-type]
            min_radius=min_radius,
            max_radius=max_radius,
            **kwargs,
        )


def test_hough_circles_max_radius_zero_means_automatic() -> None:
    image = _circles_image()

    circles = im.hough_circles(image, min_dist=20, param2=30, min_radius=10, max_radius=0)

    assert isinstance(circles, list)


@pytest.mark.parametrize("min_radius", [10.5, True])
def test_hough_circles_rejects_non_integral_min_radius(min_radius: object) -> None:
    image = _circles_image()

    with pytest.raises(TypeError, match="min_radius"):
        im.hough_circles(image, min_dist=20, param2=30, min_radius=min_radius)  # type: ignore[arg-type]


@pytest.mark.parametrize("max_radius", [50.5, True])
def test_hough_circles_rejects_non_integral_max_radius(max_radius: object) -> None:
    image = _circles_image()

    with pytest.raises(TypeError, match="max_radius"):
        im.hough_circles(image, min_dist=20, param2=30, max_radius=max_radius)  # type: ignore[arg-type]


@pytest.mark.parametrize("min_radius", [2**31, 2**63])
def test_hough_circles_rejects_min_radius_above_int32(min_radius: int) -> None:
    image = _circles_image()

    with pytest.raises(ValueError, match="int32"):
        im.hough_circles(image, min_dist=20, param2=30, min_radius=min_radius)


@pytest.mark.parametrize("max_radius", [2**31, 2**63, -(2**31) - 1])
def test_hough_circles_rejects_max_radius_outside_int32(max_radius: int) -> None:
    image = _circles_image()

    with pytest.raises(ValueError, match="int32"):
        im.hough_circles(image, min_dist=20, param2=30, max_radius=max_radius)


def test_hough_circles_rejects_huge_uint64_min_radius() -> None:
    image = _circles_image()

    with pytest.raises(ValueError, match="int32"):
        im.hough_circles(image, min_dist=20, param2=30, min_radius=np.uint64(2**64 - 1))  # type: ignore[arg-type]


def test_hough_circles_rejects_unrecognized_method() -> None:
    image = _circles_image()

    with pytest.raises(ValueError, match="method"):
        im.hough_circles(image, min_dist=20, method="invalid")  # type: ignore[arg-type]


def test_hough_circles_rejects_non_uint8_dtype() -> None:
    image = _circles_image().astype(np.float32)

    with pytest.raises(TypeError, match="dtype"):
        im.hough_circles(image, min_dist=20)  # type: ignore[arg-type]


def test_hough_circles_rejects_three_channel_image() -> None:
    image = cv2.cvtColor(_circles_image(), cv2.COLOR_GRAY2BGR)

    with pytest.raises(ValueError, match="dimensions"):
        im.hough_circles(image, min_dist=20)  # type: ignore[arg-type]


def test_hough_circles_does_not_mutate_input() -> None:
    image = _circles_image()
    before = image.copy()

    im.hough_circles(image, min_dist=20, param2=30)

    assert np.array_equal(image, before)


def test_hough_circles_rejects_bad_raw_shape_from_matcher(monkeypatch: pytest.MonkeyPatch) -> None:
    image = _circles_image()
    bad_result = np.zeros((1, 2, 5), dtype=np.float32)  # 5 fields, not 3 or 4
    monkeypatch.setattr(cv2, "HoughCircles", lambda *args, **kwargs: bad_result)

    with pytest.raises(RuntimeError, match="unexpected"):
        im.hough_circles(image, min_dist=20, param2=30)


def test_hough_circles_rejects_non_finite_circle_from_matcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = _circles_image()
    bad_result = np.array([[[10.0, 10.0, float("nan")]]], dtype=np.float32)
    monkeypatch.setattr(cv2, "HoughCircles", lambda *args, **kwargs: bad_result)

    with pytest.raises(RuntimeError, match="non-finite"):
        im.hough_circles(image, min_dist=20, param2=30)


def test_hough_circles_accepts_four_field_result_and_drops_votes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = _circles_image()
    good_result = np.array([[[10.0, 20.0, 5.0, 99.0]]], dtype=np.float32)
    monkeypatch.setattr(cv2, "HoughCircles", lambda *args, **kwargs: good_result)

    circles = im.hough_circles(image, min_dist=20, param2=30)

    assert circles == [im.Circle(x=10.0, y=20.0, radius=5.0)]
