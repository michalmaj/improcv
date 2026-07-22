import cv2
import numpy as np
import pytest

import improcv as im
from improcv.contours import BoundingBox
from improcv.drawing import (
    _normalize_bgr_color,
    _normalize_thickness,
    _require_valid_boxes,
    _require_valid_contours,
    _require_valid_fill_value,
    _require_valid_montage_dim,
    _require_valid_montage_images,
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


def test_normalize_thickness_rejects_negative_outside_int32() -> None:
    with pytest.raises(ValueError, match="int32"):
        _normalize_thickness(-(2**31) - 1)


def test_normalize_thickness_accepts_max_positive_thickness() -> None:
    assert _normalize_thickness(32767) == 32767


def test_normalize_thickness_rejects_above_max_positive_thickness() -> None:
    with pytest.raises(ValueError, match="32767"):
        _normalize_thickness(32768)


def test_normalize_thickness_rejects_int32_max_as_thickness() -> None:
    with pytest.raises(ValueError, match="32767"):
        _normalize_thickness(2**31 - 1)


@pytest.mark.parametrize("value", [32768, 2**31 - 1])
def test_draw_contours_rejects_thickness_above_max(value: int) -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    contour = _contour(0, 0, 10, 10)

    with pytest.raises(ValueError, match="32767"):
        im.draw_contours(image, [contour], thickness=value)


@pytest.mark.parametrize("value", [32768, 2**31 - 1])
def test_draw_bounding_boxes_rejects_thickness_above_max(value: int) -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="32767"):
        im.draw_bounding_boxes(image, [BoundingBox(2, 3, 4, 5)], thickness=value)


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
    with pytest.raises(TypeError, match="int32"):
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


# --- _require_valid_boxes ---


def test_require_valid_boxes_accepts_empty_list() -> None:
    assert _require_valid_boxes([]) == []


def test_require_valid_boxes_accepts_valid_boxes() -> None:
    box = BoundingBox(2, 3, 4, 5)
    result = _require_valid_boxes([box])
    assert result == [box]


def test_require_valid_boxes_rejects_single_bounding_box_instead_of_sequence() -> None:
    box = BoundingBox(2, 3, 4, 5)
    with pytest.raises(TypeError, match="sequence"):
        _require_valid_boxes(box)  # type: ignore[arg-type]


def test_require_valid_boxes_rejects_non_bounding_box_element() -> None:
    with pytest.raises(TypeError, match="BoundingBox"):
        _require_valid_boxes([(2, 3, 4, 5)])  # type: ignore[list-item]


def test_require_valid_boxes_rejects_non_positive_width() -> None:
    with pytest.raises(ValueError, match="width"):
        _require_valid_boxes([BoundingBox(0, 0, 0, 5)])


def test_require_valid_boxes_rejects_non_positive_height() -> None:
    with pytest.raises(ValueError, match="height"):
        _require_valid_boxes([BoundingBox(0, 0, 5, 0)])


def test_require_valid_boxes_rejects_bool_field() -> None:
    with pytest.raises(TypeError, match="integ"):
        _require_valid_boxes([BoundingBox(True, 0, 5, 5)])  # type: ignore[arg-type]


def test_require_valid_boxes_rejects_field_outside_int32() -> None:
    with pytest.raises(ValueError, match="int32"):
        _require_valid_boxes([BoundingBox(0, 0, 2**31, 5)])


def test_require_valid_boxes_rejects_edge_sum_outside_int32() -> None:
    with pytest.raises(ValueError, match="int32"):
        _require_valid_boxes([BoundingBox(2**31 - 1, 0, 2**31 - 1, 5)])


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_require_valid_boxes_rejects_x_plus_width_overflow_with_numpy_int32_fields() -> None:
    box = BoundingBox(np.int32(2**31 - 10), np.int32(0), np.int32(20), np.int32(5))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="int32"):
        _require_valid_boxes([box])


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_require_valid_boxes_rejects_y_plus_height_overflow_with_numpy_int32_fields() -> None:
    box = BoundingBox(np.int32(0), np.int32(2**31 - 10), np.int32(5), np.int32(20))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="int32"):
        _require_valid_boxes([box])


def test_require_valid_boxes_returns_normalized_plain_int_fields() -> None:
    box = BoundingBox(np.int32(2), np.int32(3), np.int32(4), np.int32(5))  # type: ignore[arg-type]
    result = _require_valid_boxes([box])
    assert result == [BoundingBox(2, 3, 4, 5)]
    assert all(type(field) is int for field in result[0])


# --- draw_bounding_boxes ---


def test_draw_bounding_boxes_fills_exact_pixel_region() -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    box = BoundingBox(2, 3, 4, 5)

    result = im.draw_bounding_boxes(image, [box], color=(255, 255, 255), thickness=-1)

    ys, xs = np.where(result[:, :, 0] > 0)
    assert xs.min() == 2
    assert xs.max() == 5
    assert ys.min() == 3
    assert ys.max() == 7


def test_draw_bounding_boxes_empty_list_is_noop() -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)

    result = im.draw_bounding_boxes(image, [])

    assert np.array_equal(result, image)


def test_draw_bounding_boxes_does_not_mutate_input() -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    before = image.copy()

    im.draw_bounding_boxes(image, [BoundingBox(2, 3, 4, 5)], thickness=-1)

    assert np.array_equal(image, before)


def test_draw_bounding_boxes_rejects_grayscale_image() -> None:
    image = np.zeros((20, 20), dtype=np.uint8)

    with pytest.raises(ValueError, match="channel"):
        im.draw_bounding_boxes(image, [])  # type: ignore[arg-type]


def test_draw_bounding_boxes_rejects_non_uint8_image() -> None:
    image = np.zeros((20, 20, 3), dtype=np.float32)

    with pytest.raises(TypeError, match="dtype"):
        im.draw_bounding_boxes(image, [])  # type: ignore[arg-type]


def test_draw_bounding_boxes_accepts_box_partially_outside_image() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    box = BoundingBox(5, 5, 20, 20)

    result = im.draw_bounding_boxes(image, [box], color=(255, 255, 255), thickness=-1)

    assert tuple(result[9, 9]) == (255, 255, 255)


def test_draw_bounding_boxes_rejects_bad_color() -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="between"):
        im.draw_bounding_boxes(image, [BoundingBox(2, 3, 4, 5)], color=(0, 0, 256))


def test_draw_bounding_boxes_rejects_zero_thickness() -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="0"):
        im.draw_bounding_boxes(image, [BoundingBox(2, 3, 4, 5)], thickness=0)


# --- _require_valid_montage_images ---


def _solid(value: int, shape: tuple[int, ...] = (10, 10, 3)) -> np.ndarray:
    return np.full(shape, value, dtype=np.uint8)


def test_require_valid_montage_images_accepts_consistent_color_images() -> None:
    images = [_solid(1), _solid(2)]
    result = _require_valid_montage_images(images)
    assert len(result) == 2


def test_require_valid_montage_images_accepts_consistent_grayscale_images() -> None:
    images = [_solid(1, (10, 10)), _solid(2, (10, 10))]
    result = _require_valid_montage_images(images)
    assert len(result) == 2


def test_require_valid_montage_images_rejects_single_ndarray_instead_of_sequence() -> None:
    with pytest.raises(TypeError, match="sequence"):
        _require_valid_montage_images(_solid(1))  # type: ignore[arg-type]


def test_require_valid_montage_images_rejects_empty_sequence() -> None:
    with pytest.raises(ValueError, match="empty"):
        _require_valid_montage_images([])


def test_require_valid_montage_images_rejects_non_ndarray_element() -> None:
    with pytest.raises(TypeError, match="ndarray"):
        _require_valid_montage_images([_solid(1), [[1, 2], [3, 4]]])  # type: ignore[list-item]


def test_require_valid_montage_images_rejects_non_uint8_dtype() -> None:
    with pytest.raises(TypeError, match="uint8"):
        _require_valid_montage_images([_solid(1), _solid(2).astype(np.float32)])


def test_require_valid_montage_images_rejects_non_positive_shape() -> None:
    with pytest.raises(ValueError, match="empty"):
        _require_valid_montage_images([np.zeros((0, 10, 3), dtype=np.uint8)])


def test_require_valid_montage_images_rejects_disallowed_channel_count() -> None:
    with pytest.raises(ValueError, match="channel"):
        _require_valid_montage_images([_solid(1, (10, 10, 2))])


def test_require_valid_montage_images_rejects_mismatched_channel_counts() -> None:
    with pytest.raises(ValueError, match="channel"):
        _require_valid_montage_images([_solid(1, (10, 10, 3)), _solid(2, (10, 10, 4))])


def test_require_valid_montage_images_rejects_mismatched_ndim() -> None:
    with pytest.raises(ValueError, match="channel"):
        _require_valid_montage_images([_solid(1, (10, 10)), _solid(2, (10, 10, 3))])


# --- _require_valid_montage_dim ---


def test_require_valid_montage_dim_accepts_positive_int() -> None:
    assert _require_valid_montage_dim(5, "tile_width") == 5


def test_require_valid_montage_dim_rejects_bool() -> None:
    with pytest.raises(TypeError, match="integ"):
        _require_valid_montage_dim(True, "tile_width")


def test_require_valid_montage_dim_rejects_float() -> None:
    with pytest.raises(TypeError, match="integ"):
        _require_valid_montage_dim(1.5, "tile_width")


def test_require_valid_montage_dim_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        _require_valid_montage_dim(0, "tile_width")


def test_require_valid_montage_dim_rejects_outside_int32() -> None:
    with pytest.raises(ValueError, match="int32"):
        _require_valid_montage_dim(2**31, "tile_width")


# --- _require_valid_fill_value ---


def test_require_valid_fill_value_accepts_valid_value() -> None:
    assert _require_valid_fill_value(128) == 128


@pytest.mark.parametrize("value", [-1, 256])
def test_require_valid_fill_value_rejects_out_of_range(value: int) -> None:
    with pytest.raises(ValueError, match="between"):
        _require_valid_fill_value(value)


def test_require_valid_fill_value_rejects_float() -> None:
    with pytest.raises(TypeError, match="integ"):
        _require_valid_fill_value(1.5)


def test_require_valid_fill_value_rejects_bool() -> None:
    with pytest.raises(TypeError, match="integ"):
        _require_valid_fill_value(True)


# --- montage ---


def test_montage_tiles_four_images_into_default_2x2_grid() -> None:
    images = [_solid(1), _solid(2), _solid(3), _solid(4)]

    result = im.montage(images, tile_width=5, tile_height=5)

    assert result.shape == (10, 10, 3)
    assert tuple(result[0, 0]) == (1, 1, 1)
    assert tuple(result[0, 5]) == (2, 2, 2)
    assert tuple(result[5, 0]) == (3, 3, 3)
    assert tuple(result[5, 5]) == (4, 4, 4)


def test_montage_respects_explicit_columns() -> None:
    images = [_solid(1), _solid(2), _solid(3), _solid(4)]

    result = im.montage(images, tile_width=5, tile_height=5, columns=4)

    assert result.shape == (5, 20, 3)


def test_montage_fills_leftover_cells_with_fill_value() -> None:
    images = [_solid(1), _solid(2), _solid(3)]

    result = im.montage(images, tile_width=5, tile_height=5, columns=2, fill_value=9)

    assert result.shape == (10, 10, 3)
    assert tuple(result[5, 5]) == (9, 9, 9)


def test_montage_single_image_produces_one_tile() -> None:
    result = im.montage([_solid(7)], tile_width=5, tile_height=5)

    assert result.shape == (5, 5, 3)
    assert tuple(result[0, 0]) == (7, 7, 7)


def test_montage_grayscale_images() -> None:
    images = [np.full((10, 10), 1, dtype=np.uint8), np.full((10, 10), 2, dtype=np.uint8)]

    result = im.montage(images, tile_width=5, tile_height=5, columns=2)

    assert result.shape == (5, 10)


def test_montage_does_not_mutate_input_images() -> None:
    images = [_solid(1), _solid(2)]
    before = [image.copy() for image in images]

    im.montage(images, tile_width=5, tile_height=5)

    for image, image_before in zip(images, before, strict=True):
        assert np.array_equal(image, image_before)


def test_montage_placed_tile_has_correct_shape() -> None:
    images = [_solid(1, (20, 30, 3)), _solid(2, (20, 30, 3))]

    result = im.montage(images, tile_width=8, tile_height=6, columns=2)

    assert result.shape == (6, 16, 3)


def test_montage_rejects_single_ndarray_instead_of_sequence() -> None:
    with pytest.raises(TypeError, match="sequence"):
        im.montage(_solid(1), tile_width=5, tile_height=5)  # type: ignore[arg-type]


def test_montage_rejects_mismatched_channel_counts() -> None:
    with pytest.raises(ValueError, match="channel"):
        im.montage([_solid(1, (10, 10, 3)), _solid(2, (10, 10, 4))], tile_width=5, tile_height=5)


def test_montage_rejects_non_uint8_images() -> None:
    with pytest.raises(TypeError, match="uint8"):
        im.montage([_solid(1).astype(np.float32)], tile_width=5, tile_height=5)


@pytest.mark.parametrize("value", [-1, 256])
def test_montage_rejects_out_of_range_fill_value(value: int) -> None:
    with pytest.raises(ValueError, match="between"):
        im.montage([_solid(1)], tile_width=5, tile_height=5, fill_value=value)


def test_montage_rejects_float_fill_value() -> None:
    with pytest.raises(TypeError, match="integ"):
        im.montage([_solid(1)], tile_width=5, tile_height=5, fill_value=1.5)  # type: ignore[arg-type]


def test_montage_rejects_bool_fill_value() -> None:
    with pytest.raises(TypeError, match="integ"):
        im.montage([_solid(1)], tile_width=5, tile_height=5, fill_value=True)


def test_montage_rejects_non_positive_tile_dims() -> None:
    with pytest.raises(ValueError, match="positive"):
        im.montage([_solid(1)], tile_width=0, tile_height=5)


def test_montage_rejects_output_size_exceeding_safety_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    images = [_solid(1)]
    full_called = False
    resize_called = False

    def _spy_full(*args: object, **kwargs: object) -> np.ndarray:
        nonlocal full_called
        full_called = True
        raise AssertionError("np.full should not be called")

    def _spy_resize(*args: object, **kwargs: object) -> np.ndarray:
        nonlocal resize_called
        resize_called = True
        raise AssertionError("cv2.resize should not be called")

    monkeypatch.setattr(np, "full", _spy_full)
    monkeypatch.setattr(cv2, "resize", _spy_resize)

    with pytest.raises(ValueError, match="byte"):
        im.montage(images, tile_width=20_000, tile_height=20_000, columns=100)

    assert not full_called
    assert not resize_called


def test_montage_uses_inter_area_when_shrinking(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[int] = []
    real_resize = cv2.resize

    def _spy_resize(image: np.ndarray, size: tuple[int, int], interpolation: int) -> np.ndarray:
        captured.append(interpolation)
        return real_resize(image, size, interpolation=interpolation)

    monkeypatch.setattr(cv2, "resize", _spy_resize)

    im.montage([_solid(1, (100, 100, 3))], tile_width=10, tile_height=10)

    assert captured == [cv2.INTER_AREA]


def test_montage_uses_inter_linear_when_enlarging(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[int] = []
    real_resize = cv2.resize

    def _spy_resize(image: np.ndarray, size: tuple[int, int], interpolation: int) -> np.ndarray:
        captured.append(interpolation)
        return real_resize(image, size, interpolation=interpolation)

    monkeypatch.setattr(cv2, "resize", _spy_resize)

    im.montage([_solid(1, (10, 10, 3))], tile_width=100, tile_height=100)

    assert captured == [cv2.INTER_LINEAR]
