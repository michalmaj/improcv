"""Shared type aliases for improcv's public API.

These are intentionally coarse (no pixel-level shape typing) — the goal
right now is just to make dtype expectations visible in signatures
(e.g. a mask-producing function returns `Mask`, not a bare `np.ndarray`)
without blocking a later move to a stricter Pyright mode.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = ["Image", "ImageFloat32", "ImageU8", "Mask", "TransformMatrix"]

Image = npt.NDArray[Any]
"""A generic image array of any dtype, shape ``(H, W)`` or ``(H, W, C)``."""

ImageFloat32 = npt.NDArray[np.float32]
"""A float32 image array of any shape (no shape promise beyond dtype, matching `Image`/`ImageU8`).

A 2D-only or channel-count requirement, where one applies, belongs to the specific function's own
contract (e.g. `distance_transform`), not to this type.
"""

ImageU8 = npt.NDArray[np.uint8]
"""An 8-bit image array, shape ``(H, W)`` or ``(H, W, C)``."""

Mask = npt.NDArray[np.uint8]
"""A mask array with values ``0``/``255``, shape ``(H, W)`` (improcv's mask convention)."""

TransformMatrix = npt.NDArray[np.floating[Any]]
"""A ``(2, 3)`` affine or ``(3, 3)`` perspective transformation matrix."""
