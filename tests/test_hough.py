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
