import cv2
import numpy as np
import pytest

import improcv as im


def _textured_image() -> np.ndarray:
    # A non-trivial grayscale test image with real texture, so ORB/SIFT
    # actually find keypoints -- pure noise has enough local contrast for
    # both algorithms, verified directly.
    return np.random.default_rng(0).integers(0, 255, (100, 100), dtype=np.uint8)


@pytest.mark.parametrize("method", ["orb", "sift"])
def test_detect_and_compute_finds_keypoints_and_descriptors(method: str) -> None:
    image = _textured_image()

    features = im.detect_and_compute(image, method=method)  # type: ignore[arg-type]

    assert len(features.keypoints) > 0
    assert features.descriptors.shape[0] == len(features.keypoints)
    assert features.method == method


def test_detect_and_compute_orb_descriptor_contract() -> None:
    image = _textured_image()
    reference = cv2.ORB_create()  # type: ignore[attr-defined]

    features = im.detect_and_compute(image, method="orb")

    assert features.descriptors.shape[1] == reference.descriptorSize()
    assert features.descriptors.dtype == np.uint8
    assert features.norm == "hamming"


def test_detect_and_compute_sift_descriptor_contract() -> None:
    image = _textured_image()
    reference = cv2.SIFT_create()  # type: ignore[attr-defined]

    features = im.detect_and_compute(image, method="sift")

    assert features.descriptors.shape[1] == reference.descriptorSize()
    assert features.descriptors.dtype == np.float32
    assert features.norm == "l2"


@pytest.mark.parametrize("method", ["orb", "sift"])
def test_detect_and_compute_normalizes_empty_result(method: str) -> None:
    # A constant-value (not empty-array) image: no texture, both algorithms
    # find zero keypoints -- verified directly. cv2 itself returns
    # descriptors=None in this case; detect_and_compute must not.
    image = np.full((50, 50), 128, dtype=np.uint8)
    reference = (
        cv2.ORB_create()  # type: ignore[attr-defined]
        if method == "orb"
        else cv2.SIFT_create()  # type: ignore[attr-defined]
    )
    expected_width = reference.descriptorSize()
    expected_dtype = np.uint8 if method == "orb" else np.float32

    features = im.detect_and_compute(image, method=method)  # type: ignore[arg-type]

    assert features.keypoints == []
    assert features.descriptors.shape == (0, expected_width)
    assert features.descriptors.dtype == expected_dtype


def test_detect_and_compute_mask_restricts_detection() -> None:
    image = _textured_image()
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[:40, :40] = 255

    features = im.detect_and_compute(image, method="orb", mask=mask)

    assert len(features.keypoints) > 0
    for kp in features.keypoints:
        x, y = kp.pt
        assert x < 40
        assert y < 40


def test_detect_and_compute_rejects_mismatched_mask_shape() -> None:
    image = _textured_image()
    mask = np.zeros((40, 40), dtype=np.uint8)

    with pytest.raises(ValueError, match="shape"):
        im.detect_and_compute(image, method="orb", mask=mask)


def test_detect_and_compute_accepts_numpy_integer_nfeatures() -> None:
    image = _textured_image()

    features = im.detect_and_compute(
        image,
        method="orb",
        nfeatures=np.int32(10),  # type: ignore[arg-type]
    )

    assert len(features.keypoints) > 0


def test_detect_and_compute_rejects_zero_nfeatures() -> None:
    image = _textured_image()

    with pytest.raises(ValueError, match="positive"):
        im.detect_and_compute(image, method="orb", nfeatures=0)


def test_detect_and_compute_rejects_negative_nfeatures() -> None:
    image = _textured_image()

    with pytest.raises(ValueError, match="positive"):
        im.detect_and_compute(image, method="orb", nfeatures=-5)


def test_detect_and_compute_rejects_bool_nfeatures() -> None:
    image = _textured_image()

    with pytest.raises(TypeError, match="integer"):
        im.detect_and_compute(image, method="orb", nfeatures=True)  # type: ignore[arg-type]


def test_detect_and_compute_rejects_float_nfeatures() -> None:
    image = _textured_image()

    with pytest.raises(TypeError, match="integer"):
        im.detect_and_compute(image, method="orb", nfeatures=10.0)  # type: ignore[arg-type]


@pytest.mark.parametrize("method", ["orb", "sift"])
def test_detect_and_compute_accepts_a_large_nfeatures_value(method: str) -> None:
    # Deliberately not the literal int32 boundary (2**31 - 1): verified
    # directly that ORB's internal allocation at that exact value succeeds
    # on some platforms (e.g. macOS, which overcommits memory) but fails
    # with a genuine std::bad_alloc on memory-constrained Linux CI runners
    # -- even for this same tiny 10x10 image. That is a real resource
    # question (OpenCV correctly reporting it cannot allocate that much
    # memory), not a validation-contract question -- the rejection tests
    # below already pin the exact boundary (2**31 is rejected, so
    # 2**31 - 1 is the largest value our own validation permits). A large
    # but unextreme value here exercises "a large nfeatures is accepted"
    # without depending on how much memory the environment happens to have.
    image = np.zeros((10, 10), dtype=np.uint8)

    features = im.detect_and_compute(image, method=method, nfeatures=100_000)  # type: ignore[arg-type]

    assert features.descriptors.shape[0] == len(features.keypoints)


@pytest.mark.parametrize("method", ["orb", "sift"])
@pytest.mark.parametrize("nfeatures", [2**31, 2**63])
def test_detect_and_compute_rejects_nfeatures_above_int32_range(
    nfeatures: int, method: str
) -> None:
    image = np.zeros((10, 10), dtype=np.uint8)

    with pytest.raises(ValueError, match="int32"):
        im.detect_and_compute(image, method=method, nfeatures=nfeatures)  # type: ignore[arg-type]


def test_detect_and_compute_rejects_numpy_uint64_max_nfeatures() -> None:
    image = np.zeros((10, 10), dtype=np.uint8)

    with pytest.raises(ValueError, match="int32"):
        im.detect_and_compute(
            image,
            method="orb",
            nfeatures=np.uint64(2**64 - 1),  # type: ignore[arg-type]
        )


def test_detect_and_compute_rejects_three_channel_image() -> None:
    image = cv2.cvtColor(_textured_image(), cv2.COLOR_GRAY2BGR)

    with pytest.raises(ValueError, match="dimensions"):
        im.detect_and_compute(image, method="orb")


def test_detect_and_compute_rejects_non_uint8_dtype() -> None:
    image = _textured_image().astype(np.float32)

    with pytest.raises(TypeError, match="dtype"):
        im.detect_and_compute(image, method="orb")  # type: ignore[arg-type]


def test_detect_and_compute_rejects_invalid_method() -> None:
    image = _textured_image()

    with pytest.raises(ValueError, match="method"):
        im.detect_and_compute(image, method="invalid")  # type: ignore[arg-type]


@pytest.mark.parametrize("method", ["orb", "sift"])
@pytest.mark.parametrize("shape", [(1, 20), (20, 1), (1, 1)])
def test_detect_and_compute_rejects_single_row_or_column_image(
    shape: tuple[int, int], method: str
) -> None:
    # ORB raises a raw cv2.error (from an internal cv2.resize call,
    # "inv_scale_x > 0") for these shapes on both OpenCV versions, while
    # SIFT accepts them and returns an empty result -- verified directly.
    # detect_and_compute rejects them uniformly for both methods with its
    # own ValueError, instead of a method-dependent geometry contract.
    image = np.zeros(shape, dtype=np.uint8)

    with pytest.raises(ValueError, match="at least 2 pixels"):
        im.detect_and_compute(image, method=method)  # type: ignore[arg-type]


@pytest.mark.parametrize("method", ["orb", "sift"])
def test_detect_and_compute_does_not_mutate_input(method: str) -> None:
    image = _textured_image()
    original = image.copy()

    im.detect_and_compute(image, method=method)  # type: ignore[arg-type]

    np.testing.assert_array_equal(image, original)


def test_detect_and_compute_keypoints_work_directly_with_cv2_drawKeypoints() -> None:
    image = _textured_image()

    features = im.detect_and_compute(image, method="orb")
    output = cv2.drawKeypoints(image, features.keypoints, None)  # type: ignore[call-overload]

    assert output.shape == (100, 100, 3)


def test_detect_and_compute_raises_when_sift_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    # Both verification environments actually have cv2.SIFT_create, so this
    # simulates its absence directly rather than depending on a specific
    # OpenCV build lacking it -- exercises the capability-detection branch
    # itself. Deletes the actual factory function detect_and_compute calls,
    # not merely the related but distinct cv2.SIFT class.
    monkeypatch.delattr(cv2, "SIFT_create")
    image = _textured_image()

    with pytest.raises(RuntimeError, match="SIFT"):
        im.detect_and_compute(image, method="sift")


def _two_textured_images() -> tuple[np.ndarray, np.ndarray]:
    image1 = np.random.default_rng(0).integers(0, 255, (100, 100), dtype=np.uint8)
    image2 = np.random.default_rng(1).integers(0, 255, (100, 100), dtype=np.uint8)
    return image1, image2


def _two_overlapping_images() -> tuple[np.ndarray, np.ndarray]:
    # Two crops of the same underlying texture, shifted relative to each
    # other -- unlike _two_textured_images (two fully independent noise
    # images, which share no real correspondences), this gives ORB/SIFT
    # genuine matching content. Verified directly that Lowe's ratio test
    # against two independent noise images legitimately finds zero
    # survivors (correctly rejecting coincidental NN matches as
    # ambiguous), which would make behavioral assertions about
    # match_features_ratio's output vacuously true.
    base = np.random.default_rng(0).integers(0, 255, (150, 150), dtype=np.uint8)
    image1 = base[:100, :100]
    image2 = base[20:120, 20:120]
    return image1, image2


def _empty_features(method: im.FeatureMethod) -> im.Features:
    image1, image2 = _two_textured_images()
    blank = np.full((50, 50), 128, dtype=np.uint8)
    query = im.detect_and_compute(blank, method=method)
    assert query.keypoints == []  # sanity check the fixture is actually empty
    return query


@pytest.mark.parametrize("method", ["orb", "sift"])
def test_match_features_matches_real_features(method: str) -> None:
    image1, image2 = _two_textured_images()
    query = im.detect_and_compute(image1, method=method)  # type: ignore[arg-type]
    train = im.detect_and_compute(image2, method=method)  # type: ignore[arg-type]

    matches = im.match_features(query, train)

    assert len(matches) > 0
    for match in matches:
        assert 0 <= match.queryIdx < len(query.keypoints)
        assert 0 <= match.trainIdx < len(train.keypoints)


def test_match_features_result_is_sorted_by_distance() -> None:
    image1, image2 = _two_textured_images()
    query = im.detect_and_compute(image1, method="orb", nfeatures=2000)
    train = im.detect_and_compute(image2, method="orb", nfeatures=2000)

    matches = im.match_features(query, train, cross_check=False)

    distances = [m.distance for m in matches]
    assert distances == sorted(distances)


def test_match_features_result_works_directly_with_cv2_drawMatches() -> None:
    image1, image2 = _two_textured_images()
    query = im.detect_and_compute(image1, method="orb")
    train = im.detect_and_compute(image2, method="orb")

    matches = im.match_features(query, train)
    output = cv2.drawMatches(
        image1,
        query.keypoints,
        image2,
        train.keypoints,
        matches,
        None,  # type: ignore[call-overload]
    )

    assert output.ndim == 3


@pytest.mark.parametrize("cross_check", [True, False])
@pytest.mark.parametrize("query_empty", [True, False])
@pytest.mark.parametrize("train_empty", [True, False])
def test_match_features_empty_combinations_return_empty_list(
    train_empty: bool, query_empty: bool, cross_check: bool
) -> None:
    if not query_empty and not train_empty:
        pytest.skip("both-populated case is covered by other tests")
    image1, image2 = _two_textured_images()
    query = _empty_features("orb") if query_empty else im.detect_and_compute(image1, method="orb")
    train = _empty_features("orb") if train_empty else im.detect_and_compute(image2, method="orb")

    matches = im.match_features(query, train, cross_check=cross_check)

    assert matches == []


def test_match_features_rejects_empty_but_incompatible_pair() -> None:
    query = _empty_features("orb")
    train = _empty_features("sift")

    with pytest.raises(ValueError, match="method"):
        im.match_features(query, train)


def test_match_features_rejects_mismatched_method() -> None:
    image1, image2 = _two_textured_images()
    query = im.detect_and_compute(image1, method="orb")
    train = im.detect_and_compute(image2, method="sift")

    with pytest.raises(ValueError, match="method"):
        im.match_features(query, train)


def test_match_features_rejects_mismatched_dtype() -> None:
    image1, image2 = _two_textured_images()
    query = im.detect_and_compute(image1, method="orb")
    train = im.detect_and_compute(image2, method="orb")
    bad_train = train._replace(descriptors=train.descriptors.astype(np.int16))

    with pytest.raises(TypeError, match="dtype"):
        im.match_features(query, bad_train)


def test_match_features_rejects_mismatched_width() -> None:
    image1, image2 = _two_textured_images()
    query = im.detect_and_compute(image1, method="orb")
    train = im.detect_and_compute(image2, method="orb")
    bad_train = train._replace(descriptors=train.descriptors[:, :16])

    with pytest.raises(ValueError, match="width"):
        im.match_features(query, bad_train)


def test_match_features_rejects_row_count_not_matching_keypoints() -> None:
    image1, image2 = _two_textured_images()
    query = im.detect_and_compute(image1, method="orb")
    train = im.detect_and_compute(image2, method="orb")
    bad_query = query._replace(descriptors=query.descriptors[:-1])

    with pytest.raises(ValueError, match="keypoint"):
        im.match_features(bad_query, train)


def test_match_features_rejects_nan_sift_descriptors() -> None:
    image1, image2 = _two_textured_images()
    query = im.detect_and_compute(image1, method="sift")
    train = im.detect_and_compute(image2, method="sift")
    bad_descriptors = query.descriptors.copy()
    bad_descriptors[0, 0] = np.nan
    bad_query = query._replace(descriptors=bad_descriptors)

    with pytest.raises(ValueError, match="finite"):
        im.match_features(bad_query, train)


def test_match_features_rejects_infinite_sift_descriptors() -> None:
    image1, image2 = _two_textured_images()
    query = im.detect_and_compute(image1, method="sift")
    train = im.detect_and_compute(image2, method="sift")
    bad_descriptors = query.descriptors.copy()
    bad_descriptors[0, 0] = np.inf
    bad_query = query._replace(descriptors=bad_descriptors)

    with pytest.raises(ValueError, match="finite"):
        im.match_features(bad_query, train)


def test_match_features_rejects_excessively_large_sift_descriptors() -> None:
    # Verified directly, identically on both OpenCV versions: these values
    # are finite, but overflow float32 during BFMatcher's internal L2
    # distance computation, silently producing zero matches instead of an
    # error -- rejected explicitly instead.
    query_descriptors = np.full((1, 128), 1e18, dtype=np.float32)
    train_descriptors = np.full((1, 128), -1e18, dtype=np.float32)
    query = im.Features(
        method="sift", norm="l2", keypoints=[cv2.KeyPoint(0, 0, 1)], descriptors=query_descriptors
    )
    train = im.Features(
        method="sift", norm="l2", keypoints=[cv2.KeyPoint(0, 0, 1)], descriptors=train_descriptors
    )

    with pytest.raises(ValueError, match="large"):
        im.match_features(query, train)


@pytest.mark.parametrize("cross_check", [1, 0, 1.0, "true"])
def test_match_features_rejects_non_bool_cross_check(cross_check: object) -> None:
    image1, image2 = _two_textured_images()
    query = im.detect_and_compute(image1, method="orb")
    train = im.detect_and_compute(image2, method="orb")

    with pytest.raises(TypeError, match="bool"):
        im.match_features(query, train, cross_check=cross_check)  # type: ignore[arg-type]


def test_match_features_rejects_non_features_argument() -> None:
    image1, image2 = _two_textured_images()
    train = im.detect_and_compute(image2, method="orb")

    with pytest.raises(TypeError, match="Features"):
        im.match_features("not features", train)  # type: ignore[arg-type]


def test_match_features_rejects_non_list_keypoints() -> None:
    image1, image2 = _two_textured_images()
    query = im.detect_and_compute(image1, method="orb")
    train = im.detect_and_compute(image2, method="orb")
    bad_query = query._replace(keypoints=tuple(query.keypoints))

    with pytest.raises(TypeError, match="list"):
        im.match_features(bad_query, train)  # type: ignore[arg-type]


def test_match_features_rejects_non_keypoint_element() -> None:
    image1, image2 = _two_textured_images()
    query = im.detect_and_compute(image1, method="orb")
    train = im.detect_and_compute(image2, method="orb")
    bad_query = query._replace(keypoints=[object() for _ in query.keypoints])

    with pytest.raises(TypeError, match="KeyPoint"):
        im.match_features(bad_query, train)  # type: ignore[arg-type]


def test_match_features_rejects_non_ndarray_descriptors() -> None:
    image1, image2 = _two_textured_images()
    query = im.detect_and_compute(image1, method="orb")
    train = im.detect_and_compute(image2, method="orb")
    bad_query = query._replace(descriptors=query.descriptors.tolist())

    with pytest.raises(TypeError, match="ndarray"):
        im.match_features(bad_query, train)  # type: ignore[arg-type]


def test_match_features_no_cross_check_matches_every_query_descriptor_once() -> None:
    image1, image2 = _two_textured_images()
    query = im.detect_and_compute(image1, method="orb")
    train = im.detect_and_compute(image2, method="orb")

    matches = im.match_features(query, train, cross_check=False)

    assert len(matches) == query.descriptors.shape[0]
    query_indices = [m.queryIdx for m in matches]
    assert len(set(query_indices)) == len(query_indices)


def test_match_features_cross_check_has_unique_indices() -> None:
    image1, image2 = _two_textured_images()
    query = im.detect_and_compute(image1, method="orb")
    train = im.detect_and_compute(image2, method="orb")

    matches = im.match_features(query, train, cross_check=True)

    query_indices = [m.queryIdx for m in matches]
    train_indices = [m.trainIdx for m in matches]
    assert len(set(query_indices)) == len(query_indices)
    assert len(set(train_indices)) == len(train_indices)


@pytest.mark.parametrize("bad_method", [[], {}, np.array(["orb"])])
def test_match_features_rejects_non_str_method(bad_method: object) -> None:
    bad_query = im.Features(
        method=bad_method,  # type: ignore[arg-type]
        norm="hamming",
        keypoints=[],
        descriptors=np.empty((0, 32), dtype=np.uint8),
    )
    image1, image2 = _two_textured_images()
    train = im.detect_and_compute(image2, method="orb")

    with pytest.raises(TypeError, match="str"):
        im.match_features(bad_query, train)


def test_match_features_rejects_norm_inconsistent_with_method() -> None:
    image1, image2 = _two_textured_images()
    query = im.Features(
        method="orb",
        norm="l2",
        keypoints=[cv2.KeyPoint(0, 0, 1)],
        descriptors=np.zeros((1, 32), dtype=np.uint8),
    )
    train = im.detect_and_compute(image2, method="orb")

    with pytest.raises(ValueError, match="norm"):
        im.match_features(query, train)


class _FakeBFMatcher:
    """Stand-in for cv2.BFMatcher returning a fixed, possibly-broken match list.

    Used to exercise match_features' postcondition checks on the raw
    BFMatcher result, which real BFMatcher output never violates and so
    can't otherwise be reached from real descriptors.
    """

    def __init__(self, fake_matches: list[cv2.DMatch]) -> None:
        self._fake_matches = fake_matches

    def __call__(self, norm_type: int, crossCheck: bool) -> "_FakeBFMatcher":
        return self

    def match(
        self, query_descriptors: np.ndarray, train_descriptors: np.ndarray
    ) -> list[cv2.DMatch]:
        return self._fake_matches


def _orb_query_and_train() -> tuple[im.Features, im.Features]:
    image1, image2 = _two_textured_images()
    query = im.detect_and_compute(image1, method="orb", nfeatures=10)
    train = im.detect_and_compute(image2, method="orb", nfeatures=10)
    return query, train


def test_match_features_rejects_non_finite_distance_from_matcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query, train = _orb_query_and_train()
    fake_matches = [cv2.DMatch(0, 0, float("nan"))]
    monkeypatch.setattr(cv2, "BFMatcher", _FakeBFMatcher(fake_matches))

    with pytest.raises(RuntimeError, match="distance"):
        im.match_features(query, train)


def test_match_features_rejects_out_of_range_query_idx_from_matcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query, train = _orb_query_and_train()
    fake_matches = [cv2.DMatch(query.descriptors.shape[0], 0, 1.0)]
    monkeypatch.setattr(cv2, "BFMatcher", _FakeBFMatcher(fake_matches))

    with pytest.raises(RuntimeError, match="queryIdx"):
        im.match_features(query, train)


def test_match_features_rejects_out_of_range_train_idx_from_matcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query, train = _orb_query_and_train()
    fake_matches = [cv2.DMatch(0, train.descriptors.shape[0], 1.0)]
    monkeypatch.setattr(cv2, "BFMatcher", _FakeBFMatcher(fake_matches))

    with pytest.raises(RuntimeError, match="trainIdx"):
        im.match_features(query, train)


def test_match_features_rejects_too_few_matches_without_cross_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query, train = _orb_query_and_train()
    fake_matches = [cv2.DMatch(0, 0, 1.0)]  # fewer than query.descriptors.shape[0]
    monkeypatch.setattr(cv2, "BFMatcher", _FakeBFMatcher(fake_matches))

    with pytest.raises(RuntimeError, match="expected exactly"):
        im.match_features(query, train, cross_check=False)


def test_match_features_rejects_duplicate_query_idx_without_cross_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query, train = _orb_query_and_train()
    num_query = query.descriptors.shape[0]
    fake_matches = [cv2.DMatch(0, i, 1.0) for i in range(num_query)]
    fake_matches[-1] = cv2.DMatch(0, num_query - 1, 1.0)  # duplicate queryIdx 0
    monkeypatch.setattr(cv2, "BFMatcher", _FakeBFMatcher(fake_matches))

    with pytest.raises(RuntimeError, match="duplicate queryIdx"):
        im.match_features(query, train, cross_check=False)


def test_match_features_rejects_duplicate_train_idx_with_cross_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query, train = _orb_query_and_train()
    fake_matches = [cv2.DMatch(0, 0, 1.0), cv2.DMatch(1, 0, 2.0)]
    monkeypatch.setattr(cv2, "BFMatcher", _FakeBFMatcher(fake_matches))

    with pytest.raises(RuntimeError, match="duplicate trainIdx"):
        im.match_features(query, train, cross_check=True)


def _orb_features_with_n_descriptors(n: int) -> im.Features:
    rng = np.random.default_rng(0)
    descriptors = rng.integers(0, 255, (n, 32), dtype=np.uint8)
    keypoints = [cv2.KeyPoint(float(i), float(i), 1) for i in range(n)]
    return im.Features(method="orb", norm="hamming", keypoints=keypoints, descriptors=descriptors)


@pytest.mark.parametrize("method", ["orb", "sift"])
def test_match_features_ratio_matches_real_features(method: str) -> None:
    image1, image2 = _two_overlapping_images()
    query = im.detect_and_compute(image1, method=method)  # type: ignore[arg-type]
    train = im.detect_and_compute(image2, method=method)  # type: ignore[arg-type]

    matches = im.match_features_ratio(query, train)

    assert len(matches) > 0
    for match in matches:
        assert 0 <= match.queryIdx < len(query.keypoints)
        assert 0 <= match.trainIdx < len(train.keypoints)


def test_match_features_ratio_result_is_sorted_by_distance() -> None:
    image1, image2 = _two_overlapping_images()
    query = im.detect_and_compute(image1, method="orb", nfeatures=2000)
    train = im.detect_and_compute(image2, method="orb", nfeatures=2000)

    matches = im.match_features_ratio(query, train)

    assert len(matches) > 0
    distances = [m.distance for m in matches]
    assert distances == sorted(distances)


def test_match_features_ratio_result_works_directly_with_cv2_drawMatches() -> None:
    image1, image2 = _two_overlapping_images()
    query = im.detect_and_compute(image1, method="orb")
    train = im.detect_and_compute(image2, method="orb")

    matches = im.match_features_ratio(query, train)
    output = cv2.drawMatches(
        image1,
        query.keypoints,
        image2,
        train.keypoints,
        matches,
        None,  # type: ignore[call-overload]
    )

    assert len(matches) > 0
    assert output.ndim == 3


def test_match_features_ratio_default_matches_explicit_default() -> None:
    image1, image2 = _two_overlapping_images()
    query = im.detect_and_compute(image1, method="orb", nfeatures=2000)
    train = im.detect_and_compute(image2, method="orb", nfeatures=2000)

    default_matches = im.match_features_ratio(query, train)
    explicit_matches = im.match_features_ratio(query, train, ratio=0.75)

    default_pairs = {(m.queryIdx, m.trainIdx) for m in default_matches}
    explicit_pairs = {(m.queryIdx, m.trainIdx) for m in explicit_matches}
    assert len(default_pairs) > 0
    assert default_pairs == explicit_pairs


def test_match_features_ratio_stricter_ratio_yields_subset() -> None:
    image1, image2 = _two_overlapping_images()
    query = im.detect_and_compute(image1, method="orb", nfeatures=2000)
    train = im.detect_and_compute(image2, method="orb", nfeatures=2000)

    strict_matches = im.match_features_ratio(query, train, ratio=0.5)
    loose_matches = im.match_features_ratio(query, train, ratio=0.9)

    strict_pairs = {(m.queryIdx, m.trainIdx) for m in strict_matches}
    loose_pairs = {(m.queryIdx, m.trainIdx) for m in loose_matches}
    assert len(loose_pairs) > 0
    assert strict_pairs <= loose_pairs


def test_match_features_ratio_query_idx_values_are_unique() -> None:
    image1, image2 = _two_overlapping_images()
    query = im.detect_and_compute(image1, method="orb", nfeatures=2000)
    train = im.detect_and_compute(image2, method="orb", nfeatures=2000)

    matches = im.match_features_ratio(query, train)

    assert len(matches) > 0
    query_indices = [m.queryIdx for m in matches]
    assert len(set(query_indices)) == len(query_indices)


def test_match_features_ratio_does_not_mutate_descriptors() -> None:
    image1, image2 = _two_overlapping_images()
    query = im.detect_and_compute(image1, method="orb")
    train = im.detect_and_compute(image2, method="orb")
    query_before = query.descriptors.copy()
    train_before = train.descriptors.copy()

    matches = im.match_features_ratio(query, train)

    assert len(matches) > 0
    assert np.array_equal(query.descriptors, query_before)
    assert np.array_equal(train.descriptors, train_before)


@pytest.mark.parametrize("train_size", [0, 1])
def test_match_features_ratio_too_small_train_returns_empty(train_size: int) -> None:
    image1, _ = _two_textured_images()
    query = im.detect_and_compute(image1, method="orb")
    train = _orb_features_with_n_descriptors(train_size)

    matches = im.match_features_ratio(query, train)

    assert matches == []


def test_match_features_ratio_empty_query_returns_empty() -> None:
    query = _empty_features("orb")
    _, image2 = _two_textured_images()
    train = im.detect_and_compute(image2, method="orb")

    matches = im.match_features_ratio(query, train)

    assert matches == []


def test_match_features_ratio_empty_query_and_empty_train_returns_empty() -> None:
    query = _empty_features("orb")
    train = _empty_features("orb")

    matches = im.match_features_ratio(query, train)

    assert matches == []


def test_match_features_ratio_empty_query_and_single_descriptor_train_returns_empty() -> None:
    query = _empty_features("orb")
    train = _orb_features_with_n_descriptors(1)

    matches = im.match_features_ratio(query, train)

    assert matches == []


def test_match_features_ratio_rejects_empty_but_incompatible_pair() -> None:
    query = _empty_features("orb")
    train = _empty_features("sift")

    with pytest.raises(ValueError, match="method"):
        im.match_features_ratio(query, train)


@pytest.mark.parametrize("ratio", [0.0, 1.0, -0.1, 1.5])
def test_match_features_ratio_rejects_out_of_range_ratio(ratio: float) -> None:
    query, train = _orb_query_and_train()

    with pytest.raises(ValueError, match="ratio"):
        im.match_features_ratio(query, train, ratio=ratio)


@pytest.mark.parametrize("ratio", [10**400, -(10**400)])
def test_match_features_ratio_rejects_huge_python_int_ratio(ratio: int) -> None:
    # A Python int magnitude too large to represent as float raises a raw
    # OverflowError from a naive float(ratio) call -- verified directly.
    # require_finite handles this internally (its _safe_float treats an
    # unconvertible magnitude as infinity), so this must surface as a
    # controlled ValueError, never OverflowError.
    query, train = _orb_query_and_train()

    with pytest.raises(ValueError, match="finite"):
        im.match_features_ratio(query, train, ratio=ratio)  # type: ignore[arg-type]


@pytest.mark.parametrize("ratio", [True, "0.75", None])
def test_match_features_ratio_rejects_non_real_ratio(ratio: object) -> None:
    query, train = _orb_query_and_train()

    with pytest.raises(TypeError, match="ratio"):
        im.match_features_ratio(query, train, ratio=ratio)  # type: ignore[arg-type]


@pytest.mark.parametrize("ratio", [np.float32(0.75), np.float64(0.75)])
def test_match_features_ratio_accepts_numpy_scalar_ratio(ratio: object) -> None:
    query, train = _orb_query_and_train()

    matches = im.match_features_ratio(query, train, ratio=ratio)  # type: ignore[arg-type]

    assert isinstance(matches, list)


def test_match_features_ratio_rejects_non_features_argument() -> None:
    train = _orb_features_with_n_descriptors(5)

    with pytest.raises(TypeError, match="Features"):
        im.match_features_ratio("not features", train)  # type: ignore[arg-type]


def test_match_features_ratio_rejects_mismatched_method() -> None:
    image1, image2 = _two_textured_images()
    query = im.detect_and_compute(image1, method="orb")
    train = im.detect_and_compute(image2, method="sift")

    with pytest.raises(ValueError, match="method"):
        im.match_features_ratio(query, train)


def test_match_features_ratio_rejects_mismatched_dtype() -> None:
    query = _orb_features_with_n_descriptors(5)
    train = _orb_features_with_n_descriptors(5)
    bad_train = train._replace(descriptors=train.descriptors.astype(np.int16))

    with pytest.raises(TypeError, match="dtype"):
        im.match_features_ratio(query, bad_train)


def test_match_features_ratio_rejects_excessively_large_sift_descriptors() -> None:
    query_descriptors = np.full((1, 128), 1e18, dtype=np.float32)
    train_descriptors = np.full((2, 128), -1e18, dtype=np.float32)
    query = im.Features(
        method="sift", norm="l2", keypoints=[cv2.KeyPoint(0, 0, 1)], descriptors=query_descriptors
    )
    train = im.Features(
        method="sift",
        norm="l2",
        keypoints=[cv2.KeyPoint(0, 0, 1), cv2.KeyPoint(1, 1, 1)],
        descriptors=train_descriptors,
    )

    with pytest.raises(ValueError, match="large"):
        im.match_features_ratio(query, train)


class _FakeKnnBFMatcher:
    """Stand-in for cv2.BFMatcher returning a fixed, possibly-broken knnMatch result.

    Used to exercise match_features_ratio's raw-result validation, which
    real BFMatcher output never violates and so can't otherwise be reached
    from real descriptors.
    """

    def __init__(self, fake_knn_matches: list[list[cv2.DMatch]]) -> None:
        self._fake_knn_matches = fake_knn_matches

    def __call__(self, norm_type: int, crossCheck: bool) -> "_FakeKnnBFMatcher":
        return self

    def knnMatch(
        self, query_descriptors: np.ndarray, train_descriptors: np.ndarray, k: int
    ) -> list[list[cv2.DMatch]]:
        return self._fake_knn_matches


def _orb_query_and_train_ratio(n: int = 2) -> tuple[im.Features, im.Features]:
    return _orb_features_with_n_descriptors(n), _orb_features_with_n_descriptors(n)


def test_match_features_ratio_rejects_wrong_outer_length_from_matcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query, train = _orb_query_and_train_ratio(2)
    fake_knn_matches = [[cv2.DMatch(0, 0, 1.0), cv2.DMatch(0, 1, 2.0)]]  # only 1, expected 2
    monkeypatch.setattr(cv2, "BFMatcher", _FakeKnnBFMatcher(fake_knn_matches))

    with pytest.raises(RuntimeError, match="neighbor lists"):
        im.match_features_ratio(query, train)


@pytest.mark.parametrize("neighbor_count", [0, 1])
def test_match_features_ratio_rejects_too_few_neighbors_from_matcher(
    neighbor_count: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    query, train = _orb_query_and_train_ratio(2)
    short_neighbors = [cv2.DMatch(0, 0, 1.0), cv2.DMatch(0, 1, 2.0)][:neighbor_count]
    fake_knn_matches = [short_neighbors, [cv2.DMatch(1, 0, 1.0), cv2.DMatch(1, 1, 2.0)]]
    monkeypatch.setattr(cv2, "BFMatcher", _FakeKnnBFMatcher(fake_knn_matches))

    with pytest.raises(RuntimeError, match="expected exactly 2"):
        im.match_features_ratio(query, train)


def test_match_features_ratio_rejects_too_many_neighbors_from_matcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query, train = _orb_query_and_train_ratio(2)
    fake_knn_matches = [
        [cv2.DMatch(0, 0, 1.0), cv2.DMatch(0, 1, 2.0), cv2.DMatch(0, 0, 3.0)],
        [cv2.DMatch(1, 0, 1.0), cv2.DMatch(1, 1, 2.0)],
    ]
    monkeypatch.setattr(cv2, "BFMatcher", _FakeKnnBFMatcher(fake_knn_matches))

    with pytest.raises(RuntimeError, match="expected exactly 2"):
        im.match_features_ratio(query, train)


def test_match_features_ratio_rejects_non_dmatch_neighbor_from_matcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query, train = _orb_query_and_train_ratio(2)
    fake_knn_matches = [
        [object(), cv2.DMatch(0, 1, 2.0)],  # type: ignore[list-item]
        [cv2.DMatch(1, 0, 1.0), cv2.DMatch(1, 1, 2.0)],
    ]
    monkeypatch.setattr(cv2, "BFMatcher", _FakeKnnBFMatcher(fake_knn_matches))

    with pytest.raises(RuntimeError, match="non-DMatch"):
        im.match_features_ratio(query, train)


def test_match_features_ratio_rejects_non_finite_distance_from_matcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query, train = _orb_query_and_train_ratio(2)
    fake_knn_matches = [
        [cv2.DMatch(0, 0, float("nan")), cv2.DMatch(0, 1, 2.0)],
        [cv2.DMatch(1, 0, 1.0), cv2.DMatch(1, 1, 2.0)],
    ]
    monkeypatch.setattr(cv2, "BFMatcher", _FakeKnnBFMatcher(fake_knn_matches))

    with pytest.raises(RuntimeError, match="non-finite"):
        im.match_features_ratio(query, train)


def test_match_features_ratio_rejects_non_finite_second_best_distance_from_matcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A malformed second-best candidate must not simply fail the ratio
    # comparison and vanish silently -- every neighbor is validated before
    # the ratio decision, not only the ones that end up kept.
    query, train = _orb_query_and_train_ratio(2)
    fake_knn_matches = [
        [cv2.DMatch(0, 0, 1.0), cv2.DMatch(0, 1, float("nan"))],
        [cv2.DMatch(1, 0, 1.0), cv2.DMatch(1, 1, 2.0)],
    ]
    monkeypatch.setattr(cv2, "BFMatcher", _FakeKnnBFMatcher(fake_knn_matches))

    with pytest.raises(RuntimeError, match="non-finite"):
        im.match_features_ratio(query, train)


def test_match_features_ratio_rejects_negative_distance_from_matcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query, train = _orb_query_and_train_ratio(2)
    fake_knn_matches = [
        [cv2.DMatch(0, 0, -1.0), cv2.DMatch(0, 1, 2.0)],
        [cv2.DMatch(1, 0, 1.0), cv2.DMatch(1, 1, 2.0)],
    ]
    monkeypatch.setattr(cv2, "BFMatcher", _FakeKnnBFMatcher(fake_knn_matches))

    with pytest.raises(RuntimeError, match="negative"):
        im.match_features_ratio(query, train)


def test_match_features_ratio_rejects_query_idx_inconsistent_with_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query, train = _orb_query_and_train_ratio(2)
    fake_knn_matches = [
        [cv2.DMatch(1, 0, 1.0), cv2.DMatch(1, 1, 2.0)],  # queryIdx 1, but this is list index 0
        [cv2.DMatch(1, 0, 1.0), cv2.DMatch(1, 1, 2.0)],
    ]
    monkeypatch.setattr(cv2, "BFMatcher", _FakeKnnBFMatcher(fake_knn_matches))

    with pytest.raises(RuntimeError, match="queryIdx"):
        im.match_features_ratio(query, train)


def test_match_features_ratio_rejects_out_of_range_train_idx_from_matcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query, train = _orb_query_and_train_ratio(2)
    fake_knn_matches = [
        [cv2.DMatch(0, 0, 1.0), cv2.DMatch(0, 99, 2.0)],
        [cv2.DMatch(1, 0, 1.0), cv2.DMatch(1, 1, 2.0)],
    ]
    monkeypatch.setattr(cv2, "BFMatcher", _FakeKnnBFMatcher(fake_knn_matches))

    with pytest.raises(RuntimeError, match="out-of-range trainIdx"):
        im.match_features_ratio(query, train)


def test_match_features_ratio_rejects_neighbors_not_sorted_by_distance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query, train = _orb_query_and_train_ratio(2)
    fake_knn_matches = [
        [cv2.DMatch(0, 0, 5.0), cv2.DMatch(0, 1, 1.0)],  # decreasing, not ascending
        [cv2.DMatch(1, 0, 1.0), cv2.DMatch(1, 1, 2.0)],
    ]
    monkeypatch.setattr(cv2, "BFMatcher", _FakeKnnBFMatcher(fake_knn_matches))

    with pytest.raises(RuntimeError, match="not sorted"):
        im.match_features_ratio(query, train)


def test_match_features_ratio_rejects_duplicate_train_idx_within_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query, train = _orb_query_and_train_ratio(2)
    fake_knn_matches = [
        [cv2.DMatch(0, 0, 1.0), cv2.DMatch(0, 0, 2.0)],  # both neighbors trainIdx=0
        [cv2.DMatch(1, 0, 1.0), cv2.DMatch(1, 1, 2.0)],
    ]
    monkeypatch.setattr(cv2, "BFMatcher", _FakeKnnBFMatcher(fake_knn_matches))

    with pytest.raises(RuntimeError, match="same trainIdx"):
        im.match_features_ratio(query, train)


def test_match_features_ratio_allows_duplicate_train_idx_across_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query, train = _orb_query_and_train_ratio(2)
    fake_knn_matches = [
        [cv2.DMatch(0, 0, 10.0), cv2.DMatch(0, 1, 100.0)],
        [cv2.DMatch(1, 0, 20.0), cv2.DMatch(1, 1, 100.0)],
    ]
    monkeypatch.setattr(cv2, "BFMatcher", _FakeKnnBFMatcher(fake_knn_matches))

    matches = im.match_features_ratio(query, train, ratio=0.9)

    train_indices = [m.trainIdx for m in matches]
    assert len(matches) == 2
    assert train_indices.count(0) == 2
