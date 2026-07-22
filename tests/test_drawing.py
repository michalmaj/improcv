import numpy as np
import pytest

import improcv as im
from improcv.drawing import (
    _normalize_bgr_color,
    _normalize_thickness,
    _require_valid_contours,
)


def _contour(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    return np.array(
        [[[x0, y0]], [[x1, y0]], [[x1, y1]], [[x0, y1]]],
        dtype=np.int32,
    )


# --- _normalize_bgr_color ---


def test_normalize_bgr_color_accepts_valid_tuple() -> None:
    assert _normalize_bgr_color((0, 255, 128)) == (0, 255, 128)


def test_normalize_bgr_color_normalizes_numpy_scalars_to_plain_int() -> None:
    result = _normalize_bgr_color((np.int32(0), np.int32(255), np.int32(128)))
    assert result == (0, 255, 128)
    assert all(type(channel) is int for channel in result)


def test_normalize_bgr_color_rejects_non_tuple() -> None:
    with pytest.raises(TypeError, match="tuple"):
        _normalize_bgr_color([0, 255, 128])  # type: ignore[arg-type]


def test_normalize_bgr_color_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="3"):
        _normalize_bgr_color((0, 255))


@pytest.mark.parametrize("bad_color", [(0, 255, 1.5), (0, 255, "x"), (0, 255, None)])
def test_normalize_bgr_color_rejects_non_integral_channel(bad_color: object) -> None:
    with pytest.raises(TypeError, match="integ"):
        _normalize_bgr_color(bad_color)  # type: ignore[arg-type]


def test_normalize_bgr_color_rejects_bool_channel() -> None:
    with pytest.raises(TypeError, match="integ"):
        _normalize_bgr_color((0, 255, True))


@pytest.mark.parametrize("bad_color", [(-1, 0, 0), (0, 256, 0)])
def test_normalize_bgr_color_rejects_out_of_range_channel(bad_color: tuple[int, int, int]) -> None:
    with pytest.raises(ValueError, match="between"):
        _normalize_bgr_color(bad_color)


# --- _normalize_thickness ---


def test_normalize_thickness_accepts_positive_int() -> None:
    assert _normalize_thickness(2) == 2


def test_normalize_thickness_accepts_negative_int() -> None:
    assert _normalize_thickness(-1) == -1


def test_normalize_thickness_normalizes_numpy_scalar_to_plain_int() -> None:
    result = _normalize_thickness(np.int32(3))
    assert result == 3
    assert type(result) is int


def test_normalize_thickness_rejects_zero() -> None:
    with pytest.raises(ValueError, match="0"):
        _normalize_thickness(0)


def test_normalize_thickness_rejects_bool() -> None:
    with pytest.raises(TypeError, match="integ"):
        _normalize_thickness(True)


def test_normalize_thickness_rejects_float() -> None:
    with pytest.raises(TypeError, match="integ"):
        _normalize_thickness(1.5)


@pytest.mark.parametrize("value", [2**31, -(2**31) - 1])
def test_normalize_thickness_rejects_outside_int32(value: int) -> None:
    with pytest.raises(ValueError, match="int32"):
        _normalize_thickness(value)


# --- _require_valid_contours ---


def test_require_valid_contours_accepts_empty_list() -> None:
    assert _require_valid_contours([]) == []


def test_require_valid_contours_accepts_valid_contours() -> None:
    contour = _contour(0, 0, 10, 10)
    result = _require_valid_contours([contour])
    assert len(result) == 1
    assert np.array_equal(result[0], contour)


def test_require_valid_contours_rejects_single_ndarray_instead_of_sequence() -> None:
    contour = _contour(0, 0, 10, 10)
    with pytest.raises(TypeError, match="sequence"):
        _require_valid_contours(contour)  # type: ignore[arg-type]


def test_require_valid_contours_rejects_non_ndarray_element() -> None:
    with pytest.raises(TypeError, match="ndarray"):
        _require_valid_contours([[[0, 0]], [[1, 1]]])  # type: ignore[list-item]


def test_require_valid_contours_rejects_wrong_dtype() -> None:
    bad = _contour(0, 0, 10, 10).astype(np.float32)
    with pytest.raises(ValueError, match="int32"):
        _require_valid_contours([bad])


def test_require_valid_contours_rejects_wrong_shape() -> None:
    bad = np.array([[0, 0], [1, 1]], dtype=np.int32)
    with pytest.raises(ValueError, match="shape"):
        _require_valid_contours([bad])


def test_require_valid_contours_rejects_empty_contour() -> None:
    bad = np.zeros((0, 1, 2), dtype=np.int32)
    with pytest.raises(ValueError, match="shape"):
        _require_valid_contours([bad])


# --- draw_contours ---


def test_draw_contours_draws_filled_square_pixels() -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    contour = _contour(2, 3, 6, 8)  # x in [2,6), y in [3,8) per shoelace corners

    result = im.draw_contours(image, [contour], color=(0, 255, 0), thickness=-1)

    assert tuple(result[5, 4]) == (0, 255, 0)
    assert tuple(result[0, 0]) == (0, 0, 0)


def test_draw_contours_empty_list_is_noop() -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)

    result = im.draw_contours(image, [])

    assert np.array_equal(result, image)


def test_draw_contours_does_not_mutate_input() -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    before = image.copy()
    contour = _contour(2, 3, 15, 15)

    im.draw_contours(image, [contour], thickness=-1)

    assert np.array_equal(image, before)


def test_draw_contours_rejects_grayscale_image() -> None:
    image = np.zeros((20, 20), dtype=np.uint8)

    with pytest.raises(ValueError, match="channel"):
        im.draw_contours(image, [])  # type: ignore[arg-type]


def test_draw_contours_rejects_bgra_image() -> None:
    image = np.zeros((20, 20, 4), dtype=np.uint8)

    with pytest.raises(ValueError, match="channel"):
        im.draw_contours(image, [])  # type: ignore[arg-type]


def test_draw_contours_rejects_non_uint8_image() -> None:
    image = np.zeros((20, 20, 3), dtype=np.float32)

    with pytest.raises(TypeError, match="dtype"):
        im.draw_contours(image, [])  # type: ignore[arg-type]


def test_draw_contours_nested_contours_filled_together_leave_a_hole() -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    outer = _contour(0, 0, 19, 19)
    inner = _contour(5, 5, 14, 14)

    together = im.draw_contours(image, [outer, inner], color=(255, 255, 255), thickness=-1)

    assert tuple(together[9, 9]) == (0, 0, 0)


def test_draw_contours_nested_contours_filled_separately_do_not_leave_a_hole() -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    outer = _contour(0, 0, 19, 19)
    inner = _contour(5, 5, 14, 14)

    separate = im.draw_contours(image, [outer], color=(255, 255, 255), thickness=-1)
    separate = im.draw_contours(separate, [inner], color=(255, 255, 255), thickness=-1)

    assert tuple(separate[9, 9]) == (255, 255, 255)
