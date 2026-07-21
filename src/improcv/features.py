"""Feature detection and description: ORB and SIFT keypoints/descriptors."""

from __future__ import annotations

from typing import Literal, NamedTuple

import cv2
import numpy as np

from improcv._validation import (
    require_dtype,
    require_image_ndim,
    require_one_of,
    require_positive_integral,
    require_spatial_mask,
)
from improcv.types import Image, Mask

__all__ = [
    "detect_and_compute",
    "FeatureMethod",
    "DescriptorNorm",
    "Features",
]

FeatureMethod = Literal["orb", "sift"]
DescriptorNorm = Literal["hamming", "l2"]

_DESCRIPTOR_DTYPES: dict[int, np.dtype] = {
    cv2.CV_8U: np.dtype(np.uint8),
    cv2.CV_32F: np.dtype(np.float32),
}
_DESCRIPTOR_NORMS: dict[int, DescriptorNorm] = {
    cv2.NORM_HAMMING: "hamming",
    cv2.NORM_L2: "l2",
}


class Features(NamedTuple):
    """Result of `detect_and_compute`.

    `keypoints` holds real `cv2.KeyPoint` objects (not a flattened type):
    `cv2.drawKeypoints`, `cv2.drawMatches`, and future matchers all consume
    `cv2.KeyPoint` directly, so wrapping them would force a caller to
    reconstruct real `cv2.KeyPoint` instances before every further OpenCV
    call. `norm` is read from the detector itself (never guessed from
    `method`'s name), so a future matcher can pick Hamming vs. L2 distance
    directly off this result.
    """

    method: FeatureMethod
    norm: DescriptorNorm
    keypoints: list[cv2.KeyPoint]
    descriptors: np.ndarray


def detect_and_compute(
    image: Image,
    method: FeatureMethod = "orb",
    nfeatures: int = 500,
    mask: Mask | None = None,
) -> Features:
    """Detect keypoints and compute their descriptors using ORB or SIFT.

    Parameters
    ----------
    image : np.ndarray
        Input image with shape ``(H, W)``, dtype ``uint8``. Multi-channel
        input is rejected explicitly (call `ensure_gray` first) -- verified
        directly that OpenCV itself accepts 3-/4-channel input and silently
        converts it to grayscale internally, but this project never hides
        a color-to-grayscale conversion.
    method : {"orb", "sift"}, default "orb"
        Detector/descriptor algorithm.
    nfeatures : int, default 500
        A positive integer controlling how many of the best-ranked features
        the detector retains. The actual number of features returned can be
        fewer (or, rarely, a little more) depending on the detector and the
        image -- this is not an exact count or a hard upper bound. Accepts
        any `numbers.Integral` (including NumPy integer scalars), rejecting
        `bool` and `float`.
    mask : np.ndarray or None, default None
        Optional ``uint8`` mask, shape ``(H, W)`` matching `image`. Any
        nonzero value marks a pixel where features may be detected.
        Verified directly that OpenCV's own `detectAndCompute` silently
        accepts a wrong-shaped mask instead of raising, producing a
        different (not obviously wrong) result -- this function validates
        the mask's shape itself rather than relying on that.

    Returns
    -------
    Features
        ``method``, ``norm`` (the matching distance metric to use for this
        method's descriptors -- ``"hamming"`` for ORB, ``"l2"`` for SIFT),
        ``keypoints`` (a fresh `list` of `cv2.KeyPoint`), ``descriptors``
        (shape ``(len(keypoints), descriptor_size)``, dtype matching the
        method). Never `None` for zero keypoints -- normalized to a
        correctly-shaped, correctly-typed empty array. `image` is never
        modified.

    Raises
    ------
    ValueError
        If `image` does not have exactly 2 dimensions or is empty, `method`
        is not one of the accepted values, `nfeatures` is not positive, or
        `mask` does not match `image`'s spatial size.
    TypeError
        If `image` does not have dtype ``uint8``, `nfeatures` is not
        `numbers.Integral` (rejecting `bool`/`float`), or `mask` does not
        have dtype ``uint8``.
    RuntimeError
        If `method` is ``"sift"`` but `cv2.SIFT` is unavailable in
        the installed OpenCV build, or OpenCV reports a descriptor type or
        default norm this function does not recognize, or the detector's
        output is internally inconsistent (wrong descriptor shape/dtype
        for the number of keypoints found).
    """
    require_one_of(method, ("orb", "sift"), "method")
    require_image_ndim(image, ndims=(2,))
    require_dtype(image, (np.uint8,))
    require_positive_integral(nfeatures, "nfeatures")
    nfeatures_int = int(nfeatures)
    if mask is not None:
        require_spatial_mask(mask, image)

    if method == "sift":
        if not hasattr(cv2, "SIFT"):
            raise RuntimeError("SIFT is not available in this OpenCV build")
        detector = cv2.SIFT.create(nfeatures=nfeatures_int)
    else:
        detector = cv2.ORB.create(nfeatures=nfeatures_int)

    keypoints, descriptors = detector.detectAndCompute(image, mask)
    keypoints = list(keypoints)

    descriptor_size = detector.descriptorSize()
    descriptor_cv_type = detector.descriptorType()
    dtype = _DESCRIPTOR_DTYPES.get(descriptor_cv_type)
    if dtype is None:
        raise RuntimeError(
            f"unsupported descriptor type {descriptor_cv_type} reported by {method!r}"
        )

    norm_cv_type = detector.defaultNorm()
    norm = _DESCRIPTOR_NORMS.get(norm_cv_type)
    if norm is None:
        raise RuntimeError(f"unsupported descriptor norm {norm_cv_type} reported by {method!r}")

    if descriptors is None:
        descriptors = np.empty((0, descriptor_size), dtype=dtype)

    expected_shape = (len(keypoints), descriptor_size)
    if descriptors.shape != expected_shape or descriptors.dtype != dtype:
        raise RuntimeError(
            f"OpenCV {method} detectAndCompute returned inconsistent output: expected "
            f"descriptors shape {expected_shape} dtype {dtype}, got shape {descriptors.shape} "
            f"dtype {descriptors.dtype}"
        )

    return Features(method=method, norm=norm, keypoints=keypoints, descriptors=descriptors)
