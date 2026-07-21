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
