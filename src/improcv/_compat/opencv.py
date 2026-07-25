"""Isolates OpenCV 4.x/5.x behavioral differences behind narrow, capability-detected helpers.

Nothing here branches on the OpenCV version number: each helper detects the
actual difference it's normalizing (a shape, a presence of an attribute) and
handles it directly.
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

__all__: list[str] = []

_hdr_merge_dtype_support: dict[tuple[int, type], bool] = {}


def merge_hdr_supports_dtype(merger_factory: Callable[[], object], dtype: type) -> bool:
    """Return whether the installed OpenCV build's HDR merge accepts `dtype`
    for its image stack.

    `uint8` is always supported -- it is the original, unconditional depth
    for `cv2.MergeDebevec`/`cv2.MergeRobertson`. `uint16`/`float32` support
    was added to OpenCV's HDR merge at some point after `4.9.0` (verified
    directly: raises `CV_Assert(images[0].depth() == CV_8U)` there) and
    before `4.13.0` (verified: both work) -- the exact version boundary was
    not pinned down, and `MergeDebevec`/`MergeRobertson` are not guaranteed
    to have gained the capability in the same release, so this probes the
    real capability directly (a minimal 2x2 merge call against the actual
    `merger_factory`) rather than guessing from a version number or
    assuming the two classes moved together. The result is cached per
    `(id(merger_factory), dtype)` for the life of the process, since the
    installed OpenCV build cannot change at runtime.
    """
    if dtype == np.uint8:
        return True
    cache_key = (id(merger_factory), dtype)
    cached = _hdr_merge_dtype_support.get(cache_key)
    if cached is not None:
        return cached

    probe_value = 0.5 if dtype == np.float32 else 1
    probe = np.full((2, 2, 3), probe_value, dtype=dtype)
    times = np.array([1.0, 2.0], dtype=np.float32)
    try:
        merger_factory().process([probe, probe], times)  # type: ignore[attr-defined]
        supported = True
    except cv2.error:
        supported = False
    _hdr_merge_dtype_support[cache_key] = supported
    return supported


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


def _normalize_hough_lines_p_output(raw: np.ndarray) -> np.ndarray:
    """Normalize cv2.HoughLinesP's output to a flat ``(N, 4)`` int32 array.

    ``cv2.HoughLinesP`` returns shape ``(N, 4)`` on some OpenCV builds and
    ``(N, 1, 4)`` on others for identical input (verified directly:
    ``(N, 1, 4)`` on OpenCV 4.13.0, ``(N, 4)`` on OpenCV 5.0.0). Detected by
    the array's actual shape, not by checking the OpenCV version. Accepts
    only exactly 4 fields per row (``x1, y1, x2, y2``) and ``int32`` --
    anything else is an internally inconsistent OpenCV result, not a shape
    this function knows how to normalize.
    """
    if not isinstance(raw, np.ndarray):
        raise RuntimeError(
            f"cv2.HoughLinesP returned a {type(raw).__name__}, expected an np.ndarray -- "
            "unexpected OpenCV output"
        )
    if raw.dtype != np.int32:
        raise RuntimeError(
            f"cv2.HoughLinesP returned dtype {raw.dtype}, expected int32 -- "
            "unexpected OpenCV output"
        )
    if raw.ndim == 2 and raw.shape[1] == 4:
        return raw
    if raw.ndim == 3 and raw.shape[1:] == (1, 4):
        return raw[:, 0, :]
    raise RuntimeError(
        f"cv2.HoughLinesP returned an array of shape {raw.shape} -- unexpected OpenCV output shape"
    )
