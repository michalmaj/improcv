import cv2
import numpy as np
import pytest

import improcv as im
from improcv.contours import BoundingBox


def _rect_mask(y0: int, y1: int, x0: int, x1: int, shape: tuple[int, int] = (20, 20)) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255
    return mask


def test_find_contours_finds_expected_contour_and_bounding_box() -> None:
    mask = _rect_mask(5, 15, 3, 10)  # verified: cv2.boundingRect gives (3, 5, 7, 10)

    contours, hierarchy = im.find_contours(mask)

    assert len(contours) == 1
    assert contours[0].shape == (4, 1, 2)
    assert contours[0].dtype == np.int32
    assert cv2.boundingRect(contours[0]) == (3, 5, 7, 10)
    assert hierarchy.shape == (1, 4)
    assert hierarchy.dtype == np.int32


def test_find_contours_treats_any_nonzero_value_as_foreground() -> None:
    # Verified directly against cv2.findContours: foreground is not only 255.
    mask = _rect_mask(5, 15, 3, 10)
    mask[mask == 255] = 1

    contours, _ = im.find_contours(mask)

    assert len(contours) == 1
    assert cv2.boundingRect(contours[0]) == (3, 5, 7, 10)


def test_find_contours_does_not_mutate_input() -> None:
    mask = _rect_mask(5, 15, 3, 10)
    original = mask.copy()

    im.find_contours(mask)

    np.testing.assert_array_equal(mask, original)


def test_find_contours_returns_empty_results_for_blank_mask() -> None:
    mask = np.zeros((20, 20), dtype=np.uint8)

    contours, hierarchy = im.find_contours(mask)

    assert contours == []
    assert hierarchy.shape == (0, 4)
    assert hierarchy.dtype == np.int32


def test_find_contours_hierarchy_is_a_fresh_independent_array() -> None:
    mask = _rect_mask(5, 15, 3, 10)

    _, hierarchy = im.find_contours(mask)

    assert hierarchy.flags["OWNDATA"]


def test_find_contours_hierarchy_values_for_external_ccomp_and_tree() -> None:
    # A filled square with a square hole: exactly one parent/child pair.
    # Verified directly against cv2.findContours.
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[5:25, 5:25] = 255
    mask[10:20, 10:20] = 0

    ext_contours, ext_hierarchy = im.find_contours(mask, retrieval_mode="external")
    assert len(ext_contours) == 1
    np.testing.assert_array_equal(ext_hierarchy, [[-1, -1, -1, -1]])

    for mode in ("ccomp", "tree"):
        contours, hierarchy = im.find_contours(mask, retrieval_mode=mode)  # type: ignore[arg-type]
        assert len(contours) == 2
        np.testing.assert_array_equal(hierarchy, [[-1, -1, 1, -1], [-1, -1, -1, 0]])


@pytest.mark.parametrize("mode", ["external", "list", "ccomp", "tree"])
def test_find_contours_accepts_every_retrieval_mode(mode: str) -> None:
    mask = _rect_mask(5, 15, 3, 10)

    contours, hierarchy = im.find_contours(mask, retrieval_mode=mode)  # type: ignore[arg-type]

    assert len(contours) == 1
    assert hierarchy.shape == (1, 4)


@pytest.mark.parametrize("method", ["none", "simple", "tc89_l1", "tc89_kcos"])
def test_find_contours_accepts_every_approximation_method(method: str) -> None:
    mask = _rect_mask(5, 15, 3, 10)

    contours, _ = im.find_contours(mask, approximation=method)  # type: ignore[arg-type]

    assert len(contours) == 1


def test_find_contours_rejects_non_uint8_dtype() -> None:
    mask = np.zeros((10, 10), dtype=np.float32)

    with pytest.raises(TypeError, match="uint8"):
        im.find_contours(mask)  # type: ignore[arg-type]


def test_find_contours_rejects_1d_array() -> None:
    mask = np.zeros(10, dtype=np.uint8)

    with pytest.raises(ValueError, match="2 dimensions"):
        im.find_contours(mask)


def test_find_contours_rejects_invalid_retrieval_mode() -> None:
    mask = _rect_mask(5, 15, 3, 10)

    with pytest.raises(ValueError, match="retrieval_mode"):
        im.find_contours(mask, retrieval_mode="bogus")  # type: ignore[arg-type]


def test_find_contours_rejects_invalid_approximation() -> None:
    mask = _rect_mask(5, 15, 3, 10)

    with pytest.raises(ValueError, match="approximation"):
        im.find_contours(mask, approximation="bogus")  # type: ignore[arg-type]


def _contour(points: list[list[int]]) -> np.ndarray:
    return np.array(points, dtype=np.int32).reshape(-1, 1, 2)


# A verified-exact rectangle contour: cv2.boundingRect gives (3, 5, 7, 10).
_RECT_CONTOUR = _contour([[3, 5], [3, 14], [9, 14], [9, 5]])


def test_bounding_boxes_computes_expected_box() -> None:
    result = im.bounding_boxes([_RECT_CONTOUR])

    assert result == [BoundingBox(3, 5, 7, 10)]


def test_bounding_boxes_accepts_empty_sequence() -> None:
    assert im.bounding_boxes([]) == []
    assert im.bounding_boxes(()) == []


def test_bounding_boxes_accepts_raw_cv2_findcontours_tuple_directly() -> None:
    mask = _rect_mask(5, 15, 3, 10)
    raw_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = im.bounding_boxes(raw_contours)  # type: ignore[arg-type]

    assert result == [BoundingBox(3, 5, 7, 10)]


def test_bounding_boxes_accepts_zero_point_contour() -> None:
    # cv2.boundingRect on a 0-point contour returns (0, 0, 0, 0) rather than
    # erroring — verified directly — so bounding_boxes must not reject it.
    empty_contour = np.zeros((0, 1, 2), dtype=np.int32)

    result = im.bounding_boxes([empty_contour])

    assert result == [BoundingBox(0, 0, 0, 0)]


def test_bounding_boxes_accepts_non_contiguous_contour() -> None:
    non_contiguous = _RECT_CONTOUR[::-1]
    assert not non_contiguous.flags["C_CONTIGUOUS"]

    result = im.bounding_boxes([non_contiguous])

    assert result == [BoundingBox(3, 5, 7, 10)]


def test_bounding_boxes_rejects_non_int32_dtype() -> None:
    bad_contour = _RECT_CONTOUR.astype(np.float32)

    with pytest.raises(TypeError, match="int32"):
        im.bounding_boxes([bad_contour])  # type: ignore[arg-type]


def test_bounding_boxes_rejects_wrong_shape() -> None:
    bad_contour = np.zeros((4, 2), dtype=np.int32)

    with pytest.raises(ValueError, match=r"\(N, 1, 2\)"):
        im.bounding_boxes([bad_contour])  # type: ignore[arg-type]
