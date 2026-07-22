"""QR code detection and decoding."""

from __future__ import annotations

from typing import NamedTuple

import cv2
import numpy as np
import numpy.typing as npt

from improcv._validation import require_dtype, require_image_ndim, require_one_of
from improcv.types import Image

__all__ = [
    "decode_qr_code",
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


def _require_valid_image_for_qr(image: np.ndarray) -> None:
    """Raise TypeError/ValueError unless `image` is a uint8 image with 1, 3, or 4 channels.

    Verified directly, identically on both OpenCV versions, that
    `cv2.QRCodeDetector` accepts exactly 1 (grayscale), 3 (BGR), or 4
    (BGRA) channels -- confirmed via OpenCV's own error message for any
    other channel count (`'incn == 1 || incn == 3 || incn == 4'`). This
    validates the same contract itself, for a consistent error message.
    """
    require_image_ndim(image, ndims=(2, 3))
    require_dtype(image, (np.uint8,))
    channels = 1 if image.ndim == 2 else image.shape[2]
    require_one_of(channels, (1, 3, 4), "image channel count")


def _decode_one_quadrangle(
    detector: cv2.QRCodeDetector, image: np.ndarray, quad_points: np.ndarray
) -> QRCode:
    """Decode a single already-detected QR quadrangle, given its 4 corner points.

    `quad_points` must already be validated as a finite, non-degenerate
    ``(1, 4, 2)`` float32 array by the caller -- this helper only
    validates `decode`'s own output.
    """
    try:
        decoded_text, straight_code = detector.decode(image, quad_points)
    except UnicodeDecodeError as exc:
        # This module's decode functions return `str` (UTF-8 text), not
        # arbitrary binary payloads -- a QR code containing byte-mode
        # content that isn't valid UTF-8 fails at the point OpenCV's C++
        # binding converts the decoded bytes to a Python `str`.
        raise ValueError("QR code payload is not valid UTF-8") from exc

    if not isinstance(decoded_text, str):
        raise RuntimeError(
            f"cv2.QRCodeDetector.decode returned a {type(decoded_text).__name__} for the "
            "decoded text, expected str -- unexpected OpenCV output"
        )

    if straight_code is not None:
        if (
            not isinstance(straight_code, np.ndarray)
            or straight_code.dtype != np.uint8
            or straight_code.ndim != 2
            or straight_code.size == 0
        ):
            raise RuntimeError(
                f"cv2.QRCodeDetector.decode returned a straight_code of shape "
                f"{getattr(straight_code, 'shape', None)} dtype "
                f"{getattr(straight_code, 'dtype', None)} -- unexpected OpenCV output"
            )
        data = decoded_text
    else:
        if decoded_text != "":
            raise RuntimeError(
                f"cv2.QRCodeDetector.decode returned non-empty decoded text {decoded_text!r} "
                "with no straight_code -- unexpected OpenCV output"
            )
        data = None

    return QRCode(data=data, points=quad_points[0].copy())


def decode_qr_code(image: Image) -> QRCode | None:
    """Detect and decode a single QR code.

    Parameters
    ----------
    image : np.ndarray
        A `uint8` image, grayscale (``(H, W)``), BGR (``(H, W, 3)``), or
        BGRA (``(H, W, 4)``) -- all three are genuinely, natively
        supported by OpenCV's QR detector, unlike `hough_lines`/
        `hough_line_segments`, which want a binary edge mask specifically.

    Returns
    -------
    QRCode or None
        `None` if no QR code is detected at all. **If `image` contains
        more than one QR code, this function returns `None`** -- verified
        directly that OpenCV's single-quadrangle detector does not fall
        back to finding just one of several codes present; use
        `decode_qr_codes` for images that may contain multiple QR codes.
        Otherwise a `QRCode`; `data` is `None` if a quadrangle was
        detected but its content could not be decoded, `""` if it was
        decoded and genuinely encodes empty content, or the decoded UTF-8
        string. This function decodes a single physical QR symbol -- it
        does not detect or reassemble a Structured Append sequence (a
        logical message split across multiple physical QR codes) into
        one message.

    Raises
    ------
    ValueError
        If `image` does not have exactly 2 dimensions or is empty, has a
        channel count other than 1, 3, or 4, or the detected code's
        payload is not valid UTF-8.
    TypeError
        If `image` does not have dtype ``uint8``.
    RuntimeError
        If OpenCV's raw detection or decode result is internally
        inconsistent (wrong type/shape/dtype, non-finite points, a
        zero-area quadrangle, or a decoded-text/straight_code
        combination that doesn't make sense) -- rather than a valid
        result.
    """
    _require_valid_image_for_qr(image)

    detector = cv2.QRCodeDetector()
    detected, points = detector.detect(image)
    _require_valid_qr_detection(detected, points)
    if not detected:
        return None

    quad = points[0:1]
    if _quadrangle_area(quad[0]) == 0.0:
        raise RuntimeError("cv2.QRCodeDetector detected a zero-area quadrangle")

    return _decode_one_quadrangle(detector, image, quad)
