"""QR code detection and decoding."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import numpy.typing as npt

__all__ = [
    "QRCode",
]


class QRCode(NamedTuple):
    """A QR code found by `decode_qr_code`/`decode_qr_codes`.

    ``data`` is `None` when a QR-shaped quadrangle was detected but its
    content could not be decoded (damaged/corrupted/too-noisy code); it
    is ``""`` when a QR code was successfully decoded and genuinely
    encodes empty content; otherwise it's the decoded UTF-8 string. These
    are three distinct, verified outcomes -- not collapsed into one.
    ``points`` is the quadrangle's 4 corners, shape ``(4, 2)``, ``float32``,
    always an independent copy.
    """

    data: str | None
    points: npt.NDArray[np.float32]


def _quadrangle_area(corners: np.ndarray) -> float:
    """Compute a quadrangle's area (shoelace formula) from its 4 (x, y) corners, shape (4, 2).

    Used purely as a degenerate-detection guard -- a real QR detection
    can never legitimately have zero area -- not exposed publicly.
    """
    x = corners[:, 0]
    y = corners[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def _require_valid_qr_detection(detected: object, points: object) -> None:
    """Raise RuntimeError unless a raw detect()/detectMulti() result is internally consistent.

    `detected` must be an actual `bool`; when `True`, `points` must be a
    finite `float32` array shaped `(N, 4, 2)` with `N >= 1`; when `False`,
    `points` must be `None`. OpenCV's own detection call is never expected
    to violate this, but it's verified only for the specific inputs this
    project has tried -- this closes the gap defensively rather than
    trusting it by convention alone.
    """
    if not isinstance(detected, bool):
        raise RuntimeError(
            f"cv2.QRCodeDetector detection returned a {type(detected).__name__} for its boolean "
            "result, expected bool -- unexpected OpenCV output"
        )
    if detected:
        if (
            not isinstance(points, np.ndarray)
            or points.dtype != np.float32
            or points.ndim != 3
            or points.shape[1:] != (4, 2)
            or points.shape[0] == 0
        ):
            raise RuntimeError(
                f"cv2.QRCodeDetector detection reported success but returned points of shape "
                f"{getattr(points, 'shape', None)} dtype {getattr(points, 'dtype', None)} -- "
                "unexpected OpenCV output"
            )
        if not np.all(np.isfinite(points)):
            raise RuntimeError("cv2.QRCodeDetector detection returned non-finite points")
    else:
        if points is not None:
            raise RuntimeError(
                "cv2.QRCodeDetector detection reported failure but still returned points -- "
                "unexpected OpenCV output"
            )
