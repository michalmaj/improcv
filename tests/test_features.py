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
