import numpy as np
import pytest

from improcv._validation import (
    require_channels,
    require_image_ndim,
    require_non_negative,
    require_positive,
)


def test_require_image_ndim_accepts_allowed_ndim() -> None:
    require_image_ndim(np.zeros((4, 4)), ndims=(2, 3))
    require_image_ndim(np.zeros((4, 4, 3)), ndims=(2, 3))


def test_require_image_ndim_rejects_disallowed_ndim() -> None:
    with pytest.raises(ValueError, match="2 or 3 dimensions"):
        require_image_ndim(np.zeros(4), ndims=(2, 3))


def test_require_image_ndim_single_allowed_value_message() -> None:
    with pytest.raises(ValueError, match="2 dimensions"):
        require_image_ndim(np.zeros((4, 4, 3)), ndims=(2,))


def test_require_positive_accepts_positive_value() -> None:
    require_positive(1, "width")


def test_require_positive_rejects_zero_and_negative() -> None:
    with pytest.raises(ValueError, match="positive"):
        require_positive(0, "width")
    with pytest.raises(ValueError, match="positive"):
        require_positive(-1, "width")


def test_require_non_negative_accepts_zero() -> None:
    require_non_negative(0, "top")


def test_require_non_negative_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        require_non_negative(-1, "top")


def test_require_channels_accepts_matching_channel_count() -> None:
    require_channels(np.zeros((4, 4, 3)), 3)


def test_require_channels_rejects_wrong_channel_count() -> None:
    with pytest.raises(ValueError, match="3 channels"):
        require_channels(np.zeros((4, 4)), 3)
    with pytest.raises(ValueError, match="3 channels"):
        require_channels(np.zeros((4, 4, 1)), 3)
