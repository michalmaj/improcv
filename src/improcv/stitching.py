"""Panorama and flat-scan image stitching via OpenCV's high-level Stitcher."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, cast

import cv2
import numpy as np

from improcv._validation import require_dtype, require_one_of
from improcv.types import ImageU8

__all__ = ["stitch_images", "StitchMode"]

StitchMode = Literal["panorama", "scans"]

_MIN_IMAGES = 2

_STITCH_MODES: dict[StitchMode, int] = {
    "panorama": cv2.Stitcher_PANORAMA,
    "scans": cv2.Stitcher_SCANS,
}

_STITCH_STATUS_NAMES = {
    cv2.Stitcher_OK: "OK",
    cv2.Stitcher_ERR_NEED_MORE_IMGS: "ERR_NEED_MORE_IMGS",
    cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "ERR_HOMOGRAPHY_EST_FAIL",
    cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "ERR_CAMERA_PARAMS_ADJUST_FAIL",
}

_STITCH_STATUS_MESSAGES = {
    cv2.Stitcher_ERR_NEED_MORE_IMGS: "not enough usable images, features, or overlap",
    cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "homography estimation failed",
    cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "camera parameter adjustment failed",
}


def _require_valid_stitch_stack(images: object) -> list[np.ndarray]:
    """Raise TypeError/ValueError unless `images` is a valid image stack for
    `stitch_images`, else return it as a plain `list`.

    `images` must be a real `collections.abc.Sequence` -- a single
    `np.ndarray` (including a 4D stack), a `str`/`bytes`/`bytearray`, or a
    generator/iterator (none of which implement the `Sequence` protocol,
    or are explicitly excluded here even though they technically do) are
    all rejected. Every element must be a non-empty, `uint8`, 3-channel
    BGR `(H, W, 3)` `np.ndarray` -- verified directly, in OpenCV's own
    C++ source and empirically, that grayscale, `(H, W, 1)`, 2-channel,
    and BGRA each raise a different, unindexed, low-level `cv2.error`
    from deep inside the stitching pipeline, and that `uint16`/`float32`
    are silently accepted by the Python binding but produce a misleading
    generic "not enough images" failure rather than any error -- both are
    rejected here, with a message naming the offending index, before ever
    reaching OpenCV. Unlike `merge_hdr_debevec`/`_robertson`'s image
    stack, elements may have different spatial shapes -- verified
    directly that OpenCV's Stitcher stitches differently-sized images
    without complaint.
    """
    if isinstance(images, (str, bytes, bytearray)) or not isinstance(images, Sequence):
        raise TypeError(
            "images must be a Sequence of arrays (e.g. a list or tuple), not a single "
            f"array or {type(images).__name__}"
        )
    normalized = list(images)
    if len(normalized) < _MIN_IMAGES:
        raise ValueError(
            f"images must contain at least {_MIN_IMAGES} images, got {len(normalized)}"
        )

    for index, image in enumerate(normalized):
        name = f"images[{index}]"
        if not isinstance(image, np.ndarray):
            raise TypeError(f"{name} must be a NumPy array, got {type(image).__name__}")
        if image.size == 0:
            raise ValueError(f"{name} must not be empty, got shape {image.shape}")
        if image.ndim != 3 or image.shape[2] != 3:
            if image.ndim == 2:
                raise ValueError(
                    f"{name} must be 3-channel BGR (H, W, 3), got a 2D grayscale array "
                    f"with shape {image.shape} -- stitching does not support grayscale input"
                )
            if image.ndim == 3 and image.shape[2] == 1:
                raise ValueError(
                    f"{name} must be 3-channel BGR (H, W, 3), got a single-channel image "
                    f"with an explicit trailing axis, shape {image.shape} -- drop it first "
                    f"with {name}[..., 0], though stitching does not support a genuinely "
                    "grayscale image either"
                )
            if image.ndim == 3 and image.shape[2] == 4:
                raise ValueError(
                    f"{name} must be 3-channel BGR (H, W, 3), got a 4-channel (BGRA) image "
                    f"with shape {image.shape} -- explicitly drop or composite the alpha "
                    "channel first"
                )
            raise ValueError(f"{name} must have shape (H, W, 3), got {image.shape}")
        require_dtype(image, (np.uint8,), name)

    return normalized


def _run_stitch(mode: StitchMode, images: list[np.ndarray]) -> tuple[int, object]:
    """Call a freshly created `cv2.Stitcher`, converting any raw `cv2.error`
    into a `RuntimeError`.

    A fresh `Stitcher` object is created for every call -- never cached or
    reused. `MemoryError`/`KeyboardInterrupt` are not caught here: only a
    `cv2.error` (OpenCV's own controlled failure signal) is translated;
    everything else propagates unmodified. The original exception is
    preserved as `__cause__`.
    """
    try:
        stitcher = cv2.Stitcher_create(_STITCH_MODES[mode])  # type: ignore[attr-defined]
        status, panorama = stitcher.stitch(images)
    except cv2.error as exc:
        raise RuntimeError(
            "OpenCV Stitcher failed for images that passed improcv validation"
        ) from exc
    return int(status), panorama


def _raise_for_status(status: int) -> None:
    """Raise RuntimeError unless `status` is `cv2.Stitcher_OK`.

    The message names the symbolic status constant, its numeric value, and
    (for the three known error statuses) a short, human-readable category
    -- never classifying insufficient overlap as a caller mistake: the
    input images are structurally valid, the algorithm simply could not
    relate them, which is a runtime/algorithmic failure, not a validation
    one. `ERR_HOMOGRAPHY_EST_FAIL`/`ERR_CAMERA_PARAMS_ADJUST_FAIL` are
    mapped on the authority of OpenCV's own `Status` enum, even though no
    stable synthetic input was found during audit that actually triggers
    either -- only `ERR_NEED_MORE_IMGS` was empirically reproduced.
    """
    if status == cv2.Stitcher_OK:
        return
    name = _STITCH_STATUS_NAMES.get(status)
    if name is None:
        raise RuntimeError(f"OpenCV Stitcher failed with unknown status {status}")
    message = _STITCH_STATUS_MESSAGES[status]
    raise RuntimeError(f"OpenCV Stitcher failed with {name} (status {status}): {message}")


def _validated_stitch_result(panorama: object) -> ImageU8:
    """Raise RuntimeError unless `panorama` is a valid stitched image, else
    return it cast to `ImageU8`.

    Checked only after `_raise_for_status` has confirmed `status ==
    cv2.Stitcher_OK` -- any `panorama` content is ignored for a non-OK
    status. Does not check `np.isfinite` (trivially true for `uint8`) and
    does not require any specific shape, aspect ratio, or relationship to
    the input images' sizes -- verified directly that a successful
    panorama's dimensions are not a simple function of the inputs (can be
    smaller than either input, and are not exactly reproducible across
    repeated calls with identical arguments -- see `stitch_images`'
    `Notes`).
    """
    if not isinstance(panorama, np.ndarray):
        raise RuntimeError("OpenCV Stitcher reported success but did not return a NumPy array")
    if panorama.dtype != np.uint8:
        raise RuntimeError(
            f"OpenCV Stitcher reported success but returned dtype {panorama.dtype} instead of uint8"
        )
    if panorama.ndim != 3 or panorama.shape[2] != 3:
        raise RuntimeError(
            "OpenCV Stitcher reported success but returned shape "
            f"{panorama.shape} instead of (H, W, 3)"
        )
    if panorama.shape[0] == 0 or panorama.shape[1] == 0 or panorama.size == 0:
        raise RuntimeError(
            f"OpenCV Stitcher reported success but returned an empty image, shape {panorama.shape}"
        )
    return cast(ImageU8, panorama)


def stitch_images(
    images: Sequence[ImageU8],
    *,
    mode: StitchMode = "panorama",
) -> ImageU8:
    """Stitch a sequence of overlapping images into one panorama or flat scan.

    Implemented via OpenCV's high-level `cv2.Stitcher`. Internally detects
    and matches features across all images, estimates their relative
    geometry, then warps and blends them into a single output. This is a
    thin wrapper around OpenCV's own default pipeline -- it does not
    expose or override any of `cv2.Stitcher`'s internal registration/
    seam/compositing/confidence settings.

    Parameters
    ----------
    images : Sequence[np.ndarray]
        A real `Sequence` (e.g. a list or tuple, or another
        `collections.abc.Sequence` with a stable length) of at least 2
        images -- a single `np.ndarray` (including a 4D stack), a
        `str`/`bytes`/`bytearray`, and a generator/iterator are all
        rejected explicitly. Every image must be a non-empty, `uint8`,
        3-channel BGR ``(H, W, 3)`` array -- grayscale, ``(H, W, 1)``,
        2-channel, BGRA, `uint16`, `float32`, and `float64` are all
        rejected, with no automatic conversion, since OpenCV either
        raises an unindexed, low-level `cv2.error` or (for `uint16`/
        `float32`) silently misreports the failure as insufficient
        overlap. Images may have different spatial shapes -- there is no
        requirement that they match. Not modified, nor is the `images`
        container itself; no defensive copy is made (verified directly
        that a non-contiguous, read-only, or Fortran-order array is
        handled safely).
    mode : {"panorama", "scans"}, optional
        Which of OpenCV's two stitching scenarios to use. `"panorama"`
        (the default) uses a homography/perspective model and a
        spherical projection, suited to photos taken by rotating a
        camera. `"scans"` uses an affine model, suited to flat scans or
        documents captured under a simpler (not necessarily
        rotation-only) planar transformation -- `"scans"` is not limited
        to pure translation. Both modes share the identical input/output
        contract; they differ only in the geometric model and matcher
        OpenCV uses internally.

    Returns
    -------
    np.ndarray
        The stitched image, dtype `uint8`, 3-channel BGR. A new,
        independent array; never shares memory with any input. Not
        guaranteed to have any particular shape, aspect ratio, or
        relationship to the input images' sizes -- verified directly that
        it can be smaller than either input, and that repeated calls with
        identical arguments are not guaranteed to produce the same shape
        or pixel values (see `Notes`). Not cropped or otherwise modified
        beyond what OpenCV itself produces (e.g. black borders from the
        warp are not removed).

    Raises
    ------
    ValueError
        If `images` has fewer than 2 elements, an element is empty, or
        has an unsupported shape/channel count; if `mode` is not
        `"panorama"` or `"scans"`.
    TypeError
        If `images` is not a `Sequence` (rejecting a single array, `str`/
        `bytes`/`bytearray`, or a generator/iterator), an element is not
        a NumPy array, or an element does not have dtype `uint8`.
    RuntimeError
        If OpenCV's `Stitcher` does not report success (`cv2.Stitcher_OK`)
        for the given, structurally valid images -- this is **not** a
        `ValueError`, even when the cause is insufficient overlap: the
        images are structurally fine, the algorithm simply could not
        relate them. The message names the specific status
        (`ERR_NEED_MORE_IMGS`, `ERR_HOMOGRAPHY_EST_FAIL`, or
        `ERR_CAMERA_PARAMS_ADJUST_FAIL`) and its numeric code. Also
        raised if OpenCV reports success but the returned panorama fails
        the output postcondition (not a NumPy array, wrong dtype, wrong
        shape, or empty), or if OpenCV raises a raw `cv2.error` despite
        `images`/`mode` passing improcv's own validation.

    Notes
    -----
    **Not deterministic, even within a single process.** OpenCV's feature
    matching and geometry estimation use RANSAC internally, which draws
    from OpenCV's global RNG (`cv2.setRNGSeed`) -- verified directly that,
    for a borderline amount of overlap, the exact same input images can
    succeed on one call and fail with `ERR_NEED_MORE_IMGS` on the next,
    in the same process. Even when the outcome is reliably a success, the
    exact output shape and pixel values are not guaranteed identical
    across repeated calls with the same arguments. improcv never calls
    `cv2.setRNGSeed` internally -- doing so would silently change OpenCV's
    global RNG state for every other OpenCV call in the process, not just
    this one. If reproducibility matters for your own experiments, you
    can call `cv2.setRNGSeed` yourself before calling this function, but
    that is a process-global, not a per-call, setting, and does not
    promise identical results across different OpenCV builds or
    platforms.

    **Can be expensive, in both time and memory, in a way this wrapper
    cannot safely bound.** OpenCV allocates the panorama and its internal
    intermediate buffers before `stitch()` returns; verified directly that
    a poorly-conditioned geometry estimate (e.g. two images whose relative
    orientation does not match what `mode` expects) can make OpenCV
    produce -- and report as a *successful* `cv2.Stitcher_OK` -- a
    panorama, and intermediate buffers, many times larger than the inputs.
    Because the large allocation already happens inside OpenCV before
    `stitch()` returns control to Python, no check this wrapper could
    perform on the returned array would have prevented it, so none is
    attempted. For untrusted or unpredictable input sets, consider running
    this function in an isolated process with its own resource limits, or
    using OpenCV's lower-level stitching pipeline directly for finer
    control.

    Per-image feature masks supported by the lower-level OpenCV Stitcher
    API (`stitch(images, masks, pano)`) are not exposed by this wrapper.
    """
    normalized_images = _require_valid_stitch_stack(images)
    require_one_of(mode, tuple(_STITCH_MODES), "mode")

    status, panorama = _run_stitch(cast(StitchMode, mode), normalized_images)
    _raise_for_status(status)
    return _validated_stitch_result(panorama)
