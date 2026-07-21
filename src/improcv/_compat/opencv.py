"""Isolates OpenCV 4.x/5.x behavioral differences behind narrow, capability-detected helpers.

Nothing here branches on the OpenCV version number: each helper detects the
actual difference it's normalizing (a shape, a presence of an attribute) and
handles it directly.
"""

from __future__ import annotations

import numpy as np

__all__: list[str] = []


def _normalize_calc_hist_output(raw: np.ndarray, bins: int) -> np.ndarray:
    """Normalize cv2.calcHist's output to a flat ``(bins,)`` array.

    ``cv2.calcHist`` returns shape ``(bins, 1)`` on some OpenCV builds and
    ``(bins,)`` on others for identical 1D-histogram input (verified
    directly: ``(bins, 1)`` on OpenCV 4.13.0, ``(bins,)`` on OpenCV 5.0.0).
    Detected by the array's actual size, not by checking the OpenCV version.
    """
    if raw.size != bins:
        raise RuntimeError(
            f"cv2.calcHist returned an array of size {raw.size}, expected {bins} "
            "-- unexpected OpenCV output shape"
        )
    return raw.reshape(bins)
