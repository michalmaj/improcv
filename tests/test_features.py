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
    reference = cv2.ORB.create()

    features = im.detect_and_compute(image, method="orb")

    assert features.descriptors.shape[1] == reference.descriptorSize()
    assert features.descriptors.dtype == np.uint8
    assert features.norm == "hamming"


def test_detect_and_compute_sift_descriptor_contract() -> None:
    image = _textured_image()
    reference = cv2.SIFT.create()

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
    reference = cv2.ORB.create() if method == "orb" else cv2.SIFT.create()
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


def test_detect_and_compute_does_not_mutate_input() -> None:
    image = _textured_image()
    original = image.copy()

    im.detect_and_compute(image, method="orb")

    np.testing.assert_array_equal(image, original)


def test_detect_and_compute_keypoints_work_directly_with_cv2_drawKeypoints() -> None:
    image = _textured_image()

    features = im.detect_and_compute(image, method="orb")
    output = cv2.drawKeypoints(image, features.keypoints, None)  # type: ignore[call-overload]

    assert output.shape == (100, 100, 3)


def test_detect_and_compute_raises_when_sift_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    # Both verification environments actually have cv2.SIFT, so this
    # simulates its absence directly rather than depending on a specific
    # OpenCV build lacking it -- exercises the capability-detection branch
    # itself.
    monkeypatch.delattr(cv2, "SIFT")
    image = _textured_image()

    with pytest.raises(RuntimeError, match="SIFT"):
        im.detect_and_compute(image, method="sift")
