"""Seeded image segmentation: watershed and rectangle-initialized GrabCut."""

from __future__ import annotations

from typing import cast

import cv2
import numpy as np
import numpy.typing as npt

from improcv._validation import require_channels, require_dtype, require_integral
from improcv.types import BoundingBox, ImageU8, Mask

__all__ = [
    "watershed",
    "grabcut_rect",
]


def watershed(
    image: ImageU8,
    markers: npt.NDArray[np.int32],
) -> npt.NDArray[np.int32]:
    """Segment an image via marker-based watershed.

    Parameters
    ----------
    image : np.ndarray
        Input image with shape ``(H, W, 3)``, dtype ``uint8``.
    markers : np.ndarray
        Shape ``(H, W)`` matching `image`'s spatial size, dtype ``int32``.
        ``0`` marks the unknown region to be filled in by the algorithm;
        positive values mark seed regions (not required to be contiguous
        or to start at ``1`` -- verified directly that non-contiguous
        labels like ``2`` and ``10`` are accepted and preserved). Negative
        values are rejected on input -- ``-1`` is reserved for the
        algorithm's own boundary output.

    Returns
    -------
    np.ndarray
        Shape ``(H, W)``, dtype ``int32``. Positive values mark pixels
        assigned to a region (matching one of the input seed labels);
        ``-1`` marks watershed boundaries between regions. A fresh,
        independent array; `image`/`markers` are never modified. ``0`` is
        not a guaranteed output class -- it is simply whatever seed-growth
        left unassigned, and may not appear at all for some inputs.

    Raises
    ------
    ValueError
        If `image` does not have exactly 3 channels or is empty, `markers`
        does not have exactly 2 dimensions or is empty, `markers`'s shape
        does not match `image`'s spatial size, `markers` contains a
        negative value, or `markers` contains no positive seed.
    TypeError
        If `image` does not have dtype ``uint8``, or `markers` does not
        have dtype ``int32``.
    """
    require_channels(image, 3)
    require_dtype(image, (np.uint8,))
    if markers.ndim != 2 or markers.size == 0:
        raise ValueError(
            f"markers must have exactly 2 dimensions and be non-empty, got shape {markers.shape}"
        )
    require_dtype(markers, (np.int32,), "markers")
    if markers.shape != image.shape[:2]:
        raise ValueError(
            f"markers must have shape {image.shape[:2]} matching image's spatial size, "
            f"got {markers.shape}"
        )
    if np.any(markers < 0):
        raise ValueError("markers must not contain negative values")
    if not np.any(markers > 0):
        raise ValueError("markers must contain at least one positive seed")

    markers_copy = markers.copy()
    result = cv2.watershed(image, markers_copy)
    # cv2's stubs type watershed's result as the loose MatLike; it always
    # returns the same int32 array it was given (mutated in place).
    return cast(npt.NDArray[np.int32], result)


def _require_rect(rect: object, image_width: int, image_height: int) -> tuple[int, int, int, int]:
    """Validate `rect` as a fully-contained, positive-area, non-full-image rectangle.

    Never relies on the `BoundingBox` type annotation alone: every field is
    checked at runtime (accepting any `numbers.Integral`, including NumPy
    integer scalars, rejecting `bool`/`np.bool_`/`float`) and converted to
    plain `int` before use.
    """
    if not isinstance(rect, tuple) or len(rect) != 4:
        raise ValueError(f"rect must be a 4-tuple (x, y, width, height), got {rect!r}")
    x, y, width, height = rect
    require_integral(x, "rect.x")
    require_integral(y, "rect.y")
    require_integral(width, "rect.width")
    require_integral(height, "rect.height")
    x, y, width, height = int(x), int(y), int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError(f"rect width/height must be positive, got width={width}, height={height}")
    if x < 0 or y < 0 or x + width > image_width or y + height > image_height:
        raise ValueError(
            f"rect must be fully contained within the image ({image_width}x{image_height}), "
            f"got (x={x}, y={y}, width={width}, height={height})"
        )
    covers_full_image = x == 0 and y == 0 and width == image_width and height == image_height
    if covers_full_image:
        raise ValueError(
            "rect must not cover the entire image -- GrabCut needs at least one "
            "background pixel outside rect"
        )
    return x, y, width, height


def grabcut_rect(image: ImageU8, rect: BoundingBox, iterations: int = 5) -> Mask:
    """Segment a foreground object within a rectangle via one-shot, rect-initialized GrabCut.

    A high-level, one-shot wrapper: it does not expose GrabCut's mask-init
    mode, iterative refinement, or persisted internal model state. A
    fuller, stateful API is left to a possible future function.

    Parameters
    ----------
    image : np.ndarray
        Input image with shape ``(H, W, 3)``, dtype ``uint8``.
    rect : tuple of int
        ``(x, y, width, height)`` (a `BoundingBox` or a plain 4-tuple).
        Must be fully contained within `image`, have positive `width`/
        `height`, and not cover the entire image (GrabCut needs at least
        one background pixel outside `rect` -- verified directly that
        OpenCV itself silently clips an out-of-bounds/negative-origin
        `rect` instead of rejecting it; improcv validates this itself
        rather than relying on that silent clipping). Accepts any
        `numbers.Integral` field (including NumPy integer scalars),
        rejecting `bool`/`np.bool_`/`float`.
    iterations : int, default 5
        Number of GrabCut iterations. Must be a positive integer --
        verified directly that a non-positive value does not error but
        skips GrabCut's actual refinement entirely.

    Returns
    -------
    np.ndarray
        A `Mask`: shape ``(H, W)``, dtype ``uint8``, values ``{0, 255}``.
        ``255`` marks pixels GrabCut classified as definite or probable
        foreground; ``0`` marks definite or probable background. A fresh
        array; `image` is never modified (verified directly that
        `cv2.grabCut` does not mutate its image argument, only its
        internal mask/model buffers, which this function never exposes).

    Raises
    ------
    ValueError
        If `image` does not have exactly 3 channels or is empty, `rect` is
        not a 4-tuple, is not fully contained within `image`, does not
        have positive `width`/`height`, covers the entire image, or
        `iterations` is not positive.
    TypeError
        If `image` does not have dtype ``uint8``, any `rect` field is not
        `numbers.Integral` (rejecting `bool`/`float`), or `iterations` is
        not `numbers.Integral` (rejecting `bool`/`float`).

    Notes
    -----
    GrabCut has internal numerical initialization not promised bit-identical
    across OpenCV builds/versions -- do not rely on exact whole-mask output
    for a given input; verified directly that shape, dtype, mask classes,
    and obvious background/foreground regions are stable.
    """
    require_channels(image, 3)
    require_dtype(image, (np.uint8,))
    image_height, image_width = image.shape[:2]
    x, y, width, height = _require_rect(rect, image_width, image_height)
    require_integral(iterations, "iterations")
    iterations_int = int(iterations)
    if iterations_int <= 0:
        raise ValueError(f"iterations must be positive, got {iterations_int}")

    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(
        image,
        mask,
        (x, y, width, height),
        bgd_model,
        fgd_model,
        iterations_int,
        cv2.GC_INIT_WITH_RECT,
    )
    result = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return result
