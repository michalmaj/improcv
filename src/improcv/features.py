"""Feature detection and description: ORB and SIFT keypoints/descriptors and matching."""

from __future__ import annotations

import math
from typing import Literal, NamedTuple, cast

import cv2
import numpy as np

from improcv._validation import (
    require_bool,
    require_dtype,
    require_finite,
    require_image_ndim,
    require_one_of,
    require_positive_integral,
    require_spatial_mask,
)
from improcv.types import Image, Mask

__all__ = [
    "detect_and_compute",
    "match_features",
    "match_features_ratio",
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
        Input image with shape ``(H, W)``, dtype ``uint8``, height and
        width each at least 2 pixels. Multi-channel input is rejected
        explicitly (call `ensure_gray` first) -- verified directly that
        OpenCV itself accepts 3-/4-channel input and silently converts it
        to grayscale internally, but this project never hides a
        color-to-grayscale conversion. The height/width-2 floor exists
        because ORB raises a raw `cv2.error` (from an internal `cv2.resize`
        call, `inv_scale_x > 0`) for a `(1, N)`/`(N, 1)`/`(1, 1)` image on
        both OpenCV versions, while SIFT accepts the same shapes and
        returns an empty result -- rather than a method-dependent geometry
        contract for one shared function, both methods reject it the same
        way; a `(2, 2)` image works for both.
    method : {"orb", "sift"}, default "orb"
        Detector/descriptor algorithm.
    nfeatures : int, default 500
        A positive integer, at most ``2**31 - 1`` (the range of a signed
        C ``int``), controlling how many of the best-ranked features the
        detector retains. The actual number of features returned can be
        fewer (or, rarely, a little more) depending on the detector and the
        image -- this is not an exact count or a hard upper bound. Accepts
        any `numbers.Integral` (including NumPy integer scalars), rejecting
        `bool` and `float`. Verified directly, identically on both OpenCV
        versions: OpenCV passes this value to a C ``int`` parameter, and a
        value above ``2**31 - 1`` raises one of several different raw
        exceptions depending on the exact value and method (`ValueError`
        for ORB, a raw `cv2.error` for SIFT, or `OverflowError` for very
        large values) rather than a single, clear library error -- this
        function raises its own `ValueError` before ever reaching OpenCV.
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
        If `image` does not have exactly 2 dimensions or is empty, `image`
        has a height or width below 2 pixels, `method` is not one of the
        accepted values, `nfeatures` is not positive or exceeds
        ``2**31 - 1``, or `mask` does not match `image`'s spatial size.
    TypeError
        If `image` does not have dtype ``uint8``, `nfeatures` is not
        `numbers.Integral` (rejecting `bool`/`float`), or `mask` does not
        have dtype ``uint8``.
    RuntimeError
        If `method` is ``"sift"`` but `cv2.SIFT_create` is unavailable in
        the installed OpenCV build, or OpenCV reports a descriptor type or
        default norm this function does not recognize, or the detector's
        output is internally inconsistent (wrong descriptor shape/dtype
        for the number of keypoints found).
    """
    require_one_of(method, ("orb", "sift"), "method")
    require_image_ndim(image, ndims=(2,))
    require_dtype(image, (np.uint8,))
    height, width = image.shape
    if height < 2 or width < 2:
        raise ValueError(
            "image height and width must each be at least 2 pixels for feature detection"
        )
    require_positive_integral(nfeatures, "nfeatures")
    nfeatures_int = int(nfeatures)
    max_nfeatures = int(np.iinfo(np.int32).max)
    if nfeatures_int > max_nfeatures:
        raise ValueError(
            f"nfeatures must fit within the range of int32 ([1, {max_nfeatures}]), got {nfeatures}"
        )
    if mask is not None:
        require_spatial_mask(mask, image)

    detector: cv2.Feature2D
    if method == "sift":
        sift_create = getattr(cv2, "SIFT_create", None)
        if not callable(sift_create):
            raise RuntimeError("SIFT is not available in this OpenCV build")
        detector = cast(cv2.Feature2D, sift_create(nfeatures=nfeatures_int))
    else:
        detector = cv2.ORB_create(nfeatures=nfeatures_int)  # type: ignore[attr-defined]

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


_METHOD_DESCRIPTOR_CONTRACTS: dict[FeatureMethod, tuple[DescriptorNorm, np.dtype, int]] = {
    "orb": ("hamming", np.dtype(np.uint8), 32),
    "sift": ("l2", np.dtype(np.float32), 128),
}
_NORM_CV_TYPES: dict[DescriptorNorm, int] = {"hamming": cv2.NORM_HAMMING, "l2": cv2.NORM_L2}


def _require_valid_features(value: Features, name: str) -> None:
    """Validate `value` against `Features`' full public contract, not just its labels.

    `Features` is a public `NamedTuple`, so a caller can construct one by
    hand or hand-edit an existing one's fields -- a value that passes every
    shape/dtype check below could still hold non-`cv2.KeyPoint` objects in
    `keypoints`, silently breaking the promised `cv2.drawMatches` interop.
    Structural type checks run first, before any method/norm/shape/dtype
    check.
    """
    if not isinstance(value, Features):
        raise TypeError(f"{name} must be a Features, got {type(value).__name__}")
    if not isinstance(value.keypoints, list):
        raise TypeError(f"{name}.keypoints must be a list, got {type(value.keypoints).__name__}")
    if not isinstance(value.descriptors, np.ndarray):
        raise TypeError(
            f"{name}.descriptors must be a numpy.ndarray, got {type(value.descriptors).__name__}"
        )
    if not all(isinstance(kp, cv2.KeyPoint) for kp in value.keypoints):
        raise TypeError(f"{name}.keypoints must contain only cv2.KeyPoint objects")

    if not isinstance(value.method, str):
        raise TypeError(f"{name}.method must be a str, got {type(value.method).__name__}")
    if value.method not in _METHOD_DESCRIPTOR_CONTRACTS:
        raise ValueError(
            f"{name}.method must be one of {tuple(_METHOD_DESCRIPTOR_CONTRACTS)}, "
            f"got {value.method!r}"
        )
    expected_norm, expected_dtype, expected_width = _METHOD_DESCRIPTOR_CONTRACTS[
        cast(FeatureMethod, value.method)
    ]

    if value.descriptors.ndim != 2:
        raise ValueError(
            f"{name}.descriptors must have exactly 2 dimensions, got {value.descriptors.ndim}"
        )
    if value.descriptors.shape[0] != len(value.keypoints):
        raise ValueError(
            f"{name}.descriptors must have one row per keypoint, got "
            f"{value.descriptors.shape[0]} rows for {len(value.keypoints)} keypoints"
        )
    if value.descriptors.dtype != expected_dtype:
        raise TypeError(
            f"{name}.descriptors must have dtype {expected_dtype} for method {value.method!r}, "
            f"got {value.descriptors.dtype}"
        )
    if value.descriptors.shape[1] != expected_width:
        raise ValueError(
            f"{name}.descriptors must have width {expected_width} for method {value.method!r}, "
            f"got {value.descriptors.shape[1]}"
        )
    if value.norm != expected_norm:
        raise ValueError(
            f"{name}.norm must be {expected_norm!r} for method {value.method!r}, got {value.norm!r}"
        )
    if expected_dtype == np.float32 and not np.all(np.isfinite(value.descriptors)):
        raise ValueError(f"{name}.descriptors must contain only finite values")


def _require_safe_l2_magnitude(query: Features, train: Features) -> None:
    """Raise ValueError if `"l2"`-norm descriptors are too large for safe float32 L2 distance.

    Only applies when `query.norm == "l2"` (SIFT). Verified directly,
    identically on both OpenCV versions: finite but extreme `float32`
    values (e.g. ``1e18``) overflow `BFMatcher`'s internal L2 distance
    computation, silently producing corrupted results (zero total matches
    from `match()`, or fewer neighbors than requested from `knnMatch()`)
    instead of raising.
    """
    if query.norm != "l2":
        return
    descriptor_width = query.descriptors.shape[1]
    max_abs_value = float(
        max(
            np.abs(query.descriptors).max(initial=0.0),
            np.abs(train.descriptors).max(initial=0.0),
        )
    )
    safe_abs_limit = math.sqrt(float(np.finfo(np.float32).max) / (4.0 * descriptor_width))
    if max_abs_value >= safe_abs_limit:
        raise ValueError(
            "SIFT descriptor values are too large for safe float32 L2 distance computation"
        )


def match_features(
    query: Features,
    train: Features,
    cross_check: bool = True,
) -> list[cv2.DMatch]:
    """Match `query`'s descriptors against `train`'s using brute-force nearest-neighbor.

    A single best match per query descriptor -- no ratio test, no KNN, no
    FLANN, no RANSAC, and no geometric filtering. `query`/`train` are
    typically produced by `detect_and_compute`.

    Parameters
    ----------
    query : Features
        The descriptors to find matches for. `queryIdx` on each returned
        `cv2.DMatch` indexes into `query.keypoints`/`query.descriptors`.
    train : Features
        The descriptors to search within. `trainIdx` on each returned
        `cv2.DMatch` indexes into `train.keypoints`/`train.descriptors`.
        Must use the same `method`/`norm` as `query` -- matching descriptors
        from different methods is not supported in this function.
    cross_check : bool, default True
        If ``True``, keeps only mutually-best matches: a match `(i, j)` is
        kept only if `train`'s `j`-th descriptor is also `query`'s `i`-th
        descriptor's best match and vice versa -- fewer, higher-confidence
        matches. If ``False``, every query descriptor is matched to its
        single nearest `train` descriptor regardless of whether that
        train descriptor's own nearest match points back (raw
        one-directional nearest-neighbor).

    Returns
    -------
    list of cv2.DMatch
        Sorted by ``distance`` ascending (best match first) -- verified
        directly that `cv2.BFMatcher.match()`'s raw output is not sorted.
        Lower ``distance`` means a better match. Python's sort is stable,
        but `match_features` intentionally guarantees only non-decreasing
        distance order; the relative order of equal-distance matches is
        not part of the public API. Empty if either `query` or `train` has
        no descriptors.

    Raises
    ------
    ValueError
        If `query`/`train` fails its own internal contract (unrecognized
        `method`, wrong `descriptors` dimensionality, a row count not
        matching its keypoint count, the wrong width for its `method`, or
        a `norm` inconsistent with its `method`), `query`/`train` are not
        pairwise compatible (different `method`, `norm`, dtype, or
        descriptor width), or (for ``"l2"``/SIFT descriptors) any
        descriptor value is too large for safe ``float32`` L2 distance
        computation.
    TypeError
        If `query`/`train` is not a `Features`, its `keypoints` is not a
        `list`, any element of `keypoints` is not a `cv2.KeyPoint`, its
        `descriptors` is not an `np.ndarray`, its `method` is not a `str`,
        `query`/`train` have mismatched descriptor dtypes, or `cross_check`
        is not an actual `bool`.
    RuntimeError
        If a returned `cv2.DMatch` has a non-finite `distance` or an
        out-of-range `queryIdx`/`trainIdx` -- OpenCV producing an
        internally inconsistent result rather than a valid match.
    """
    require_bool(cross_check, "cross_check")
    _require_valid_features(query, "query")
    _require_valid_features(train, "train")

    if query.method != train.method:
        raise ValueError(
            f"query and train must use the same method, got {query.method!r} and {train.method!r}"
        )
    if query.norm != train.norm:
        raise ValueError(
            f"query and train must use the same norm, got {query.norm!r} and {train.norm!r}"
        )
    if query.descriptors.dtype != train.descriptors.dtype:
        raise TypeError(
            f"query and train descriptors must have the same dtype, got "
            f"{query.descriptors.dtype} and {train.descriptors.dtype}"
        )
    if query.descriptors.shape[1] != train.descriptors.shape[1]:
        raise ValueError(
            f"query and train descriptors must have the same width, got "
            f"{query.descriptors.shape[1]} and {train.descriptors.shape[1]}"
        )

    _require_safe_l2_magnitude(query, train)

    if query.descriptors.shape[0] == 0 or train.descriptors.shape[0] == 0:
        return []

    matcher = cv2.BFMatcher(_NORM_CV_TYPES[query.norm], crossCheck=cross_check)
    matches = list(matcher.match(query.descriptors, train.descriptors))
    matches.sort(key=lambda match: match.distance)

    num_query = query.descriptors.shape[0]
    num_train = train.descriptors.shape[0]
    for match in matches:
        if not math.isfinite(match.distance):
            raise RuntimeError(f"BFMatcher produced a non-finite distance: {match.distance}")
        if not (0 <= match.queryIdx < num_query):
            raise RuntimeError(f"BFMatcher produced an out-of-range queryIdx: {match.queryIdx}")
        if not (0 <= match.trainIdx < num_train):
            raise RuntimeError(f"BFMatcher produced an out-of-range trainIdx: {match.trainIdx}")

    query_indices = [match.queryIdx for match in matches]
    if not cross_check and num_train > 0:
        if len(matches) != num_query:
            raise RuntimeError(
                f"BFMatcher produced {len(matches)} matches, expected exactly {num_query} "
                "(one per query descriptor) for cross_check=False"
            )
        if len(set(query_indices)) != len(query_indices):
            raise RuntimeError("BFMatcher produced a duplicate queryIdx for cross_check=False")
    elif cross_check:
        train_indices = [match.trainIdx for match in matches]
        if len(set(query_indices)) != len(query_indices):
            raise RuntimeError("BFMatcher produced a duplicate queryIdx for cross_check=True")
        if len(set(train_indices)) != len(train_indices):
            raise RuntimeError("BFMatcher produced a duplicate trainIdx for cross_check=True")

    return matches


def match_features_ratio(
    query: Features,
    train: Features,
    ratio: float = 0.75,
) -> list[cv2.DMatch]:
    """Match `query`'s descriptors against `train`'s using KNN and Lowe's ratio test.

    For each query descriptor, compares its two nearest `train` neighbors
    and keeps the match only if the best candidate is meaningfully closer
    than the second-best (``best.distance < ratio * second_best.distance``)
    -- this filters out ambiguous matches where the two best candidates are
    nearly equidistant (likely a false positive), rather than filtering by
    cross-matcher agreement like `match_features`'s `cross_check`.

    There is no `cross_check` parameter: OpenCV's built-in `crossCheck` only
    works for a single nearest neighbor and cannot be directly combined with
    the two-neighbor search this function needs -- verified directly that
    `cv2.BFMatcher(crossCheck=True).knnMatch(..., k=2)` raises a raw
    `cv2.error` assertion on both OpenCV versions. A bidirectional
    ratio-matching algorithm is possible in principle but out of scope
    here; this function is one-directional (query -> train) only. There is
    also no `k` parameter -- the ratio test is only defined for exactly two
    neighbors, so `k=2` is fixed internally.

    Parameters
    ----------
    query : Features
        The descriptors to find matches for. `queryIdx` on each returned
        `cv2.DMatch` indexes into `query.keypoints`/`query.descriptors`.
    train : Features
        The descriptors to search within. `trainIdx` on each returned
        `cv2.DMatch` indexes into `train.keypoints`/`train.descriptors`.
        Must use the same `method`/`norm` as `query`. If `train` has fewer
        than 2 descriptors, the ratio test is undefined for every query
        descriptor and this function returns ``[]`` without calling OpenCV
        at all -- this is expected, not an error.
    ratio : float, default 0.75
        A common, more conservative choice used in practice (e.g. OpenCV's
        own tutorials) -- not the literal value from Lowe's original paper,
        which used ``0.8``. Must be strictly between ``0.0`` and ``1.0``.
        ``1.0`` is rejected even though the strict ``<`` comparison would
        still reject an exact tie between the two candidate distances: it
        removes the intended margin of distinctiveness and so defeats the
        point of the test, rather than genuinely disabling filtering.

    Returns
    -------
    list of cv2.DMatch
        Sorted by ``distance`` ascending (best match first). Lower
        ``distance`` means a better match. A query descriptor is silently
        excluded from the result if `train` has fewer than 2 descriptors
        (see `train` above) or if it fails the ratio test -- both are
        expected outcomes, not errors. `queryIdx` values in the result are
        unique (at most one match survives per query descriptor);
        `trainIdx` duplicates across different queries are not, since
        there is no cross-check enforcing a one-to-one mapping on the
        `train` side.

    Raises
    ------
    ValueError
        If `ratio` is not finite (including a Python `int` magnitude too
        large to represent as `float`) or not strictly between ``0.0`` and
        ``1.0``, `query`/`train` fails its own internal contract
        (unrecognized `method`, wrong `descriptors` dimensionality, a row
        count not matching its keypoint count, the wrong width for its
        `method`, or a `norm` inconsistent with its `method`), `query`/
        `train` are not pairwise compatible (different `method`, `norm`,
        or descriptor width), or (for ``"l2"``/SIFT descriptors) any
        descriptor value is too large for safe ``float32`` L2 distance
        computation.
    TypeError
        If `ratio` is not a real number, `query`/`train` is not a
        `Features`, its `keypoints` is not a `list`, any element of
        `keypoints` is not a `cv2.KeyPoint`, its `descriptors` is not an
        `np.ndarray`, its `method` is not a `str`, or `query`/`train` have
        mismatched descriptor dtypes.
    RuntimeError
        If OpenCV's raw KNN result is internally inconsistent given
        `train` has at least 2 descriptors: the wrong number of neighbor
        lists overall, the wrong number of neighbors for a query, a
        neighbor that isn't a `cv2.DMatch`, a non-finite or negative
        `distance`, a `queryIdx` inconsistent with its position in the
        result, an out-of-range `trainIdx`, neighbors not sorted by
        distance, or both neighbors of one query sharing the same
        `trainIdx`.
    """
    require_finite(ratio, "ratio")
    ratio_float = float(ratio)
    if not (0.0 < ratio_float < 1.0):
        raise ValueError(f"ratio must be strictly between 0.0 and 1.0, got {ratio}")

    _require_valid_features(query, "query")
    _require_valid_features(train, "train")

    if query.method != train.method:
        raise ValueError(
            f"query and train must use the same method, got {query.method!r} and {train.method!r}"
        )
    if query.norm != train.norm:
        raise ValueError(
            f"query and train must use the same norm, got {query.norm!r} and {train.norm!r}"
        )
    if query.descriptors.dtype != train.descriptors.dtype:
        raise TypeError(
            f"query and train descriptors must have the same dtype, got "
            f"{query.descriptors.dtype} and {train.descriptors.dtype}"
        )
    if query.descriptors.shape[1] != train.descriptors.shape[1]:
        raise ValueError(
            f"query and train descriptors must have the same width, got "
            f"{query.descriptors.shape[1]} and {train.descriptors.shape[1]}"
        )

    _require_safe_l2_magnitude(query, train)

    if query.descriptors.shape[0] == 0 or train.descriptors.shape[0] < 2:
        return []

    matcher = cv2.BFMatcher(_NORM_CV_TYPES[query.norm], crossCheck=False)
    knn_matches = matcher.knnMatch(query.descriptors, train.descriptors, k=2)

    num_query = query.descriptors.shape[0]
    num_train = train.descriptors.shape[0]
    if len(knn_matches) != num_query:
        raise RuntimeError(
            f"BFMatcher.knnMatch returned {len(knn_matches)} neighbor lists, expected exactly "
            f"{num_query} (one per query descriptor)"
        )

    kept: list[cv2.DMatch] = []
    for query_index, neighbors in enumerate(knn_matches):
        if len(neighbors) != 2:
            raise RuntimeError(
                f"BFMatcher.knnMatch returned {len(neighbors)} neighbors for query descriptor "
                f"{query_index}, expected exactly 2 (train has {num_train} >= 2 descriptors)"
            )
        for match in neighbors:
            if not isinstance(match, cv2.DMatch):
                raise RuntimeError(f"BFMatcher.knnMatch returned a non-DMatch neighbor: {match!r}")
            if not math.isfinite(match.distance):
                raise RuntimeError(f"BFMatcher produced a non-finite distance: {match.distance}")
            if match.distance < 0.0:
                raise RuntimeError(f"BFMatcher produced a negative distance: {match.distance}")
            if match.queryIdx != query_index:
                raise RuntimeError(
                    f"BFMatcher returned queryIdx {match.queryIdx} for neighbor list {query_index}"
                )
            if not (0 <= match.trainIdx < num_train):
                raise RuntimeError(f"BFMatcher produced an out-of-range trainIdx: {match.trainIdx}")

        best, second_best = neighbors
        if not (best.distance <= second_best.distance):
            raise RuntimeError(
                f"BFMatcher's neighbors for query descriptor {query_index} are not sorted "
                "by distance"
            )
        if best.trainIdx == second_best.trainIdx:
            raise RuntimeError(
                f"BFMatcher returned the same trainIdx {best.trainIdx} for both neighbors of "
                f"query descriptor {query_index}"
            )

        if best.distance < ratio_float * second_best.distance:
            kept.append(best)

    kept.sort(key=lambda match: match.distance)
    return kept
