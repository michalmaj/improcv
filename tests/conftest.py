"""Shared pytest fixtures for improcv's test suite."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest


@pytest.fixture
def make_image() -> Callable[..., np.ndarray]:
    def _make_image(height: int, width: int, channels: int | None = 3) -> np.ndarray:
        shape = (height, width) if channels is None else (height, width, channels)
        return (np.arange(int(np.prod(shape))) % 256).astype(np.uint8).reshape(shape)

    return _make_image
