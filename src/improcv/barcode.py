"""Barcode detection and decoding."""

from __future__ import annotations

from typing import NamedTuple

import cv2
import numpy as np
import numpy.typing as npt

from improcv._validation import require_dtype, require_image_ndim, require_one_of
from improcv.types import Image

__all__ = [
    "decode_barcodes",
    "Barcode",
]


class Barcode(NamedTuple):
    """A barcode found by `decode_barcodes`.

    ``data`` is `None` when a barcode-shaped quadrangle was detected but
    its content could not be decoded; otherwise the decoded string.
    Unlike `improcv.QRCode`, there is no separate "detected but genuinely
    empty" state -- verified that OpenCV's `decoded_info`/`decoded_type`
    are always empty together or non-empty together, and barcode formats
    have no meaningful concept of empty content anyway. ``barcode_type``
    names the symbology (e.g. ``"EAN_13"``, ``"EAN_8"``, ``"UPC_A"``),
    `None` exactly when `data` is `None`. Typed as `str`, not a closed
    `Literal`, since OpenCV may add more decoders in a future version.
    ``points`` is the quadrangle's 4 corners, shape ``(4, 2)``, ``float32``,
    always an independent copy.
    """

    data: str | None
    barcode_type: str | None
    points: npt.NDArray[np.float32]


def _require_valid_barcode_image(image: np.ndarray) -> None:
    """Raise TypeError/ValueError unless `image` is a uint8 image with 1, 3, or 4
    channels, at least 41x41.

    Verified directly, identically on both OpenCV versions, that
    `cv2.barcode.BarcodeDetector` never attempts detection -- returning
    `False` with empty results, indistinguishable from "nothing found" --
    when `image` has `height <= 40` or `width <= 40`. Rejecting this
    upfront prevents a misleading empty `[]` for an image OpenCV never
    actually searched.
    """
    require_image_ndim(image, ndims=(2, 3))
    channels = 1 if image.ndim == 2 else image.shape[2]
    require_one_of(channels, (1, 3, 4), "image channel count")
    require_dtype(image, (np.uint8,))
    if image.shape[0] <= 40 or image.shape[1] <= 40:
        raise ValueError(
            "image must be at least 41x41 for barcode detection, "
            f"got spatial shape {image.shape[:2]}"
        )


def _quadrangle_area(corners: npt.NDArray[np.float32]) -> float:
    """Compute a quadrangle's area (shoelace formula) from its 4 (x, y) corners, shape (4, 2).

    Used purely as a degenerate-detection guard -- not exposed publicly.
    Computed in `float64` after shifting corners relative to the first
    corner, matching `improcv.qrcode`'s identical helper -- avoids
    `float32` shoelace-formula cancellation for quadrangles detected far
    from the origin.
    """
    corners_f64 = corners.astype(np.float64, copy=False)
    shifted = corners_f64 - corners_f64[0]
    x = shifted[:, 0]
    y = shifted[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def _require_valid_barcode_result(raw: object) -> list[Barcode]:
    """Raise RuntimeError unless the raw detectAndDecodeWithType output is consistent.

    Validates the whole 4-tuple (`retval`, `decoded_info`, `decoded_type`,
    `points`) together rather than trusting `retval` in isolation --
    verified that `retval` only means "at least one code decoded
    successfully", not "anything was detected": an all-corrupted
    multi-barcode image gives `retval=False` with non-empty
    `decoded_info`/`points`. Using `retval` to decide "return `[]`" would
    silently drop detected-but-undecodable results whenever every code in
    a batch fails to decode.
    """
    if not isinstance(raw, tuple) or len(raw) != 4:
        raise RuntimeError("detectAndDecodeWithType returned an unexpected result")

    retval, decoded_info, decoded_type, points = raw

    if type(retval) is not bool:
        raise RuntimeError("detectAndDecodeWithType returned a non-bool retval")
    if not isinstance(decoded_info, (list, tuple)) or not isinstance(decoded_type, (list, tuple)):
        raise RuntimeError(
            f"detectAndDecodeWithType returned {type(decoded_info).__name__}/"
            f"{type(decoded_type).__name__} for decoded_info/decoded_type, expected "
            "list/tuple -- unexpected OpenCV output"
        )
    if len(decoded_info) != len(decoded_type):
        raise RuntimeError(
            f"detectAndDecodeWithType returned {len(decoded_info)} decoded_info but "
            f"{len(decoded_type)} decoded_type -- unexpected OpenCV output"
        )

    if len(decoded_info) == 0:
        if points is not None:
            raise RuntimeError(
                "detectAndDecodeWithType returned no decoded_info but non-None points -- "
                "unexpected OpenCV output"
            )
        if retval:
            raise RuntimeError(
                "detectAndDecodeWithType returned retval=True with no decoded_info -- "
                "unexpected OpenCV output"
            )
        return []

    if (
        not isinstance(points, np.ndarray)
        or points.dtype != np.float32
        or points.ndim != 3
        or points.shape[1:] != (4, 2)
        or points.shape[0] != len(decoded_info)
    ):
        raise RuntimeError(
            f"detectAndDecodeWithType returned {len(decoded_info)} decoded_info but points "
            f"shape {getattr(points, 'shape', None)} -- unexpected OpenCV output"
        )
    if not np.all(np.isfinite(points)):
        raise RuntimeError("detectAndDecodeWithType returned non-finite points")

    codes: list[Barcode] = []
    for i, (info, btype) in enumerate(zip(decoded_info, decoded_type, strict=True)):
        if not isinstance(info, str) or not isinstance(btype, str):
            raise RuntimeError(
                f"barcode[{i}] has decoded_info/decoded_type of type "
                f"{type(info).__name__}/{type(btype).__name__}, expected str -- "
                "unexpected OpenCV output"
            )
        if (info == "") != (btype == ""):
            raise RuntimeError(
                f"barcode[{i}] has inconsistent decoded_info={info!r} decoded_type={btype!r} "
                "-- unexpected OpenCV output"
            )
        if _quadrangle_area(points[i]) <= 0.0:
            raise RuntimeError(f"barcode[{i}] has a degenerate quadrangle")
        data = info if info != "" else None
        barcode_type = btype if btype != "" else None
        codes.append(Barcode(data=data, barcode_type=barcode_type, points=points[i].copy()))

    decoded_any = any(info != "" for info in decoded_info)
    if retval != decoded_any:
        raise RuntimeError("detectAndDecodeWithType returned an inconsistent retval")

    return codes


def decode_barcodes(image: Image) -> list[Barcode]:
    """Detect and decode EAN-8, EAN-13, and UPC-A barcodes supported by OpenCV's
    built-in barcode decoder.

    Parameters
    ----------
    image : np.ndarray
        A `uint8` image, grayscale (``(H, W)``), BGR (``(H, W, 3)``), or
        BGRA (``(H, W, 4)``), with both spatial dimensions greater than 40
        pixels -- OpenCV never attempts detection below that size.

    Returns
    -------
    list of Barcode
        One `Barcode` per detected quadrangle, `[]` if none are found. The
        list order is **not** guaranteed to match spatial left-to-right/
        top-to-bottom placement -- match results by `points` if position
        matters. A detected quadrangle whose content could not be decoded
        still appears in the list, with `data` and `barcode_type` set to
        `None` -- it is not silently dropped.

    Raises
    ------
    ValueError
        If `image` does not have exactly 2 or 3 dimensions, is empty, has
        a channel count other than 1, 3, or 4, has either spatial
        dimension of 40 pixels or less, or a detected code's payload is
        not valid UTF-8.
    TypeError
        If `image` does not have dtype ``uint8``.
    RuntimeError
        If OpenCV's raw detection result is internally inconsistent (wrong
        type/shape/dtype, non-finite or degenerate points, mismatched
        `decoded_info`/`decoded_type`, or a `retval` inconsistent with
        whether any code actually decoded) -- rather than a valid result.
    """
    _require_valid_barcode_image(image)

    detector = cv2.barcode.BarcodeDetector()
    try:
        raw = detector.detectAndDecodeWithType(image)
    except UnicodeDecodeError as exc:
        raise ValueError("barcode payload is not valid UTF-8") from exc

    return _require_valid_barcode_result(raw)
