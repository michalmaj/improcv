import cv2
import numpy as np
import pytest

import improcv as im


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
