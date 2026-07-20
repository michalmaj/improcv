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
