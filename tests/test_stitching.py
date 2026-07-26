from collections.abc import Sequence

import cv2
import numpy as np
import pytest

import improcv as im
from improcv.stitching import StitchMode, stitch_images


def _make_scene(seed: int = 1, h: int = 300, w: int = 600) -> np.ndarray:
    """A deterministic, non-periodic synthetic scene for stitching tests.

    Verified directly that a regular checkerboard/periodic texture gives
    unstable (seed-dependent, sometimes-fails) feature matching -- smooth,
    non-repeating color blobs plus spatially unique circles/lines give a
    reliable, repeatable stitch success across many repeated calls without
    seeding OpenCV's RNG.
    """
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, (h // 8, w // 8, 3), dtype=np.uint8)
    img = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
    img = cv2.GaussianBlur(img, (15, 15), 0)

    for _ in range(200):
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = int(rng.integers(3, 20))
        color = rng.integers(0, 255, size=3).tolist()
        cv2.circle(img, (cx, cy), r, color, -1)

    for _ in range(100):
        p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        color = rng.integers(0, 255, size=3).tolist()
        cv2.line(img, p1, p2, color, int(rng.integers(1, 4)))

    return img


def _overlapping_pair(seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    scene = _make_scene(seed=seed)
    width = scene.shape[1]
    overlap = 80
    left = scene[:, : width // 2 + overlap]
    right = scene[:, width // 2 - overlap :]
    return left, right


class _CustomSequence(Sequence):
    """A minimal, real `collections.abc.Sequence` that is neither list nor tuple."""

    def __init__(self, items: list) -> None:
        self._items = items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index):
        return self._items[index]


class _FakeStitcher:
    def __init__(self, status: int, panorama: object) -> None:
        self._status = status
        self._panorama = panorama
        self.received_images: object = None

    def stitch(self, images: object) -> tuple[int, object]:
        self.received_images = images
        return self._status, self._panorama


def _forbid_opencv_call(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args, **kwargs):
        pytest.fail("OpenCV must not be called after a validation error")

    monkeypatch.setattr(cv2, "Stitcher_create", boom)


# --- real Stitcher: integration tests ---


def test_stitch_images_panorama_two_images() -> None:
    left, right = _overlapping_pair(seed=1)
    left_before, right_before = left.copy(), right.copy()

    result = stitch_images([left, right])

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint8
    assert result.ndim == 3
    assert result.shape[2] == 3
    assert result.size > 0
    np.testing.assert_array_equal(left, left_before)
    np.testing.assert_array_equal(right, right_before)
    assert not np.shares_memory(result, left)
    assert not np.shares_memory(result, right)


def test_stitch_images_panorama_three_images() -> None:
    scene = _make_scene(seed=2)
    width = scene.shape[1]
    third = width // 3
    o = 60
    images = [
        scene[:, : third + o],
        scene[:, third - o : 2 * third + o],
        scene[:, 2 * third - o :],
    ]

    result = stitch_images(images)

    assert result.dtype == np.uint8
    assert result.ndim == 3
    assert result.shape[2] == 3


def test_stitch_images_scans_two_images() -> None:
    left, right = _overlapping_pair(seed=1)

    result = stitch_images([left, right], mode="scans")

    assert result.dtype == np.uint8
    assert result.ndim == 3
    assert result.shape[2] == 3


def test_stitch_images_no_overlap_raises_runtime_error_with_status() -> None:
    scene = _make_scene(seed=3)
    width = scene.shape[1]
    q1 = scene[:, : width // 4]
    q2 = scene[:, 3 * width // 4 :]

    with pytest.raises(RuntimeError, match="status 1"):
        stitch_images([q1, q2])


def test_stitch_images_accepts_different_spatial_shapes() -> None:
    left, right = _overlapping_pair(seed=1)
    left_shorter = left[:250, :]

    result = stitch_images([left_shorter, right])

    assert result.dtype == np.uint8
    assert result.ndim == 3
    assert result.shape[2] == 3


def test_stitch_images_does_not_mutate_container() -> None:
    left, right = _overlapping_pair(seed=1)
    images = [left, right]

    stitch_images(images)

    assert len(images) == 2
    assert images[0] is left
    assert images[1] is right


def test_stitch_images_accepts_non_contiguous_view() -> None:
    left, right = _overlapping_pair(seed=1)
    left_nc = np.ascontiguousarray(left)[:, ::1]
    right_nc = np.ascontiguousarray(right)[:, ::1]

    result = stitch_images([left_nc, right_nc])

    assert result.dtype == np.uint8


def test_stitch_images_accepts_read_only_arrays() -> None:
    left, right = _overlapping_pair(seed=1)
    left.flags.writeable = False
    right.flags.writeable = False

    result = stitch_images([left, right])

    assert result.dtype == np.uint8


def test_stitch_images_accepts_fortran_order() -> None:
    left, right = _overlapping_pair(seed=1)
    left_f = np.asfortranarray(left)
    right_f = np.asfortranarray(right)

    result = stitch_images([left_f, right_f])

    assert result.dtype == np.uint8


# --- fake Stitcher: unit tests for mode mapping, status mapping, postconditions ---


@pytest.mark.parametrize(
    ("mode", "expected_cv2_mode"),
    [("panorama", cv2.Stitcher_PANORAMA), ("scans", cv2.Stitcher_SCANS)],
)
def test_stitch_images_maps_mode_correctly(
    mode: StitchMode, expected_cv2_mode: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, int] = {}
    valid_pano = np.zeros((4, 4, 3), dtype=np.uint8)

    def fake_factory(cv2_mode):
        captured["mode"] = cv2_mode
        return _FakeStitcher(cv2.Stitcher_OK, valid_pano)

    monkeypatch.setattr(cv2, "Stitcher_create", fake_factory)

    stitch_images([np.zeros((4, 4, 3), dtype=np.uint8)] * 2, mode=mode)

    assert captured["mode"] == expected_cv2_mode


def test_stitch_images_passes_materialized_list_to_stitch(monkeypatch: pytest.MonkeyPatch) -> None:
    valid_pano = np.zeros((4, 4, 3), dtype=np.uint8)
    fake = _FakeStitcher(cv2.Stitcher_OK, valid_pano)
    monkeypatch.setattr(cv2, "Stitcher_create", lambda mode: fake)

    a = np.zeros((4, 4, 3), dtype=np.uint8)
    b = np.zeros((4, 4, 3), dtype=np.uint8)
    stitch_images((a, b))  # a tuple, not a list

    assert isinstance(fake.received_images, list)
    assert fake.received_images == [a, b]


def test_stitch_images_ok_returns_the_panorama(monkeypatch: pytest.MonkeyPatch) -> None:
    valid_pano = np.zeros((5, 6, 3), dtype=np.uint8)
    monkeypatch.setattr(
        cv2, "Stitcher_create", lambda mode: _FakeStitcher(cv2.Stitcher_OK, valid_pano)
    )

    result = stitch_images([np.zeros((4, 4, 3), dtype=np.uint8)] * 2)

    assert result is valid_pano


@pytest.mark.parametrize(
    ("status", "match"),
    [
        (cv2.Stitcher_ERR_NEED_MORE_IMGS, "ERR_NEED_MORE_IMGS"),
        (cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL, "ERR_HOMOGRAPHY_EST_FAIL"),
        (cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL, "ERR_CAMERA_PARAMS_ADJUST_FAIL"),
    ],
)
def test_stitch_images_maps_known_error_statuses(
    status: int, match: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cv2, "Stitcher_create", lambda mode: _FakeStitcher(status, None))

    with pytest.raises(RuntimeError, match=match):
        stitch_images([np.zeros((4, 4, 3), dtype=np.uint8)] * 2)


def test_stitch_images_maps_unknown_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cv2, "Stitcher_create", lambda mode: _FakeStitcher(17, None))

    with pytest.raises(RuntimeError, match="unknown status 17"):
        stitch_images([np.zeros((4, 4, 3), dtype=np.uint8)] * 2)


def test_stitch_images_ignores_panorama_content_for_non_ok_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A non-OK status with a populated (but presumably partial/garbage)
    # panorama must still raise -- content is never inspected.
    populated = np.ones((10, 10, 3), dtype=np.uint8)
    status = cv2.Stitcher_ERR_NEED_MORE_IMGS
    monkeypatch.setattr(cv2, "Stitcher_create", lambda mode: _FakeStitcher(status, populated))

    with pytest.raises(RuntimeError, match="ERR_NEED_MORE_IMGS"):
        stitch_images([np.zeros((4, 4, 3), dtype=np.uint8)] * 2)


def test_stitch_images_factory_cv2_error_becomes_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(mode):
        raise cv2.error("synthetic factory failure")

    monkeypatch.setattr(cv2, "Stitcher_create", boom)

    with pytest.raises(RuntimeError) as exc_info:
        stitch_images([np.zeros((4, 4, 3), dtype=np.uint8)] * 2)
    assert isinstance(exc_info.value.__cause__, cv2.error)


def test_stitch_images_stitch_cv2_error_becomes_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BoomStitcher:
        def stitch(self, images):
            raise cv2.error("synthetic stitch failure")

    monkeypatch.setattr(cv2, "Stitcher_create", lambda mode: _BoomStitcher())

    with pytest.raises(RuntimeError) as exc_info:
        stitch_images([np.zeros((4, 4, 3), dtype=np.uint8)] * 2)
    assert isinstance(exc_info.value.__cause__, cv2.error)


def test_stitch_images_ok_with_none_panorama_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cv2, "Stitcher_create", lambda mode: _FakeStitcher(cv2.Stitcher_OK, None))

    with pytest.raises(RuntimeError, match="did not return a NumPy array"):
        stitch_images([np.zeros((4, 4, 3), dtype=np.uint8)] * 2)


def test_stitch_images_ok_with_empty_panorama_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    monkeypatch.setattr(cv2, "Stitcher_create", lambda mode: _FakeStitcher(cv2.Stitcher_OK, empty))

    with pytest.raises(RuntimeError, match="empty"):
        stitch_images([np.zeros((4, 4, 3), dtype=np.uint8)] * 2)


def test_stitch_images_ok_with_wrong_dtype_panorama_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrong_dtype = np.zeros((4, 4, 3), dtype=np.float32)
    monkeypatch.setattr(
        cv2, "Stitcher_create", lambda mode: _FakeStitcher(cv2.Stitcher_OK, wrong_dtype)
    )

    with pytest.raises(RuntimeError, match="instead of uint8"):
        stitch_images([np.zeros((4, 4, 3), dtype=np.uint8)] * 2)


def test_stitch_images_ok_with_grayscale_panorama_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grayscale = np.zeros((4, 4), dtype=np.uint8)
    monkeypatch.setattr(
        cv2, "Stitcher_create", lambda mode: _FakeStitcher(cv2.Stitcher_OK, grayscale)
    )

    with pytest.raises(RuntimeError, match=r"instead of \(H, W, 3\)"):
        stitch_images([np.zeros((4, 4, 3), dtype=np.uint8)] * 2)


def test_stitch_images_ok_with_bgra_panorama_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bgra = np.zeros((4, 4, 4), dtype=np.uint8)
    monkeypatch.setattr(cv2, "Stitcher_create", lambda mode: _FakeStitcher(cv2.Stitcher_OK, bgra))

    with pytest.raises(RuntimeError, match=r"instead of \(H, W, 3\)"):
        stitch_images([np.zeros((4, 4, 3), dtype=np.uint8)] * 2)


def test_stitch_images_creates_fresh_factory_object_per_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0
    valid_pano = np.zeros((4, 4, 3), dtype=np.uint8)

    def counting_factory(mode):
        nonlocal call_count
        call_count += 1
        return _FakeStitcher(cv2.Stitcher_OK, valid_pano)

    monkeypatch.setattr(cv2, "Stitcher_create", counting_factory)
    images = [np.zeros((4, 4, 3), dtype=np.uint8)] * 2

    for _ in range(3):
        stitch_images(images)

    assert call_count == 3


# --- validation: sequence-level ---


def test_stitch_images_accepts_tuple(monkeypatch: pytest.MonkeyPatch) -> None:
    valid_pano = np.zeros((4, 4, 3), dtype=np.uint8)
    monkeypatch.setattr(
        cv2, "Stitcher_create", lambda mode: _FakeStitcher(cv2.Stitcher_OK, valid_pano)
    )

    pair = (np.zeros((4, 4, 3), dtype=np.uint8), np.zeros((4, 4, 3), dtype=np.uint8))
    result = stitch_images(pair)

    assert result is valid_pano


def test_stitch_images_accepts_custom_sequence(monkeypatch: pytest.MonkeyPatch) -> None:
    valid_pano = np.zeros((4, 4, 3), dtype=np.uint8)
    monkeypatch.setattr(
        cv2, "Stitcher_create", lambda mode: _FakeStitcher(cv2.Stitcher_OK, valid_pano)
    )
    images = _CustomSequence([np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)])

    result = stitch_images(images)

    assert result is valid_pano


def test_stitch_images_rejects_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    _forbid_opencv_call(monkeypatch)
    images = (np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2))

    with pytest.raises(TypeError, match="Sequence"):
        stitch_images(images)  # type: ignore[arg-type]


def test_stitch_images_rejects_single_4d_ndarray(monkeypatch: pytest.MonkeyPatch) -> None:
    _forbid_opencv_call(monkeypatch)
    stack = np.zeros((2, 4, 4, 3), dtype=np.uint8)

    with pytest.raises(TypeError, match="Sequence"):
        stitch_images(stack)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_container", ["abcdef", b"abcdef", bytearray(b"abcdef")])
def test_stitch_images_rejects_str_bytes_bytearray(
    bad_container, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid_opencv_call(monkeypatch)

    with pytest.raises(TypeError, match="Sequence"):
        stitch_images(bad_container)


def test_stitch_images_rejects_empty_sequence(monkeypatch: pytest.MonkeyPatch) -> None:
    _forbid_opencv_call(monkeypatch)

    with pytest.raises(ValueError, match="at least 2"):
        stitch_images([])


def test_stitch_images_rejects_single_image(monkeypatch: pytest.MonkeyPatch) -> None:
    _forbid_opencv_call(monkeypatch)

    with pytest.raises(ValueError, match="at least 2"):
        stitch_images([np.zeros((4, 4, 3), dtype=np.uint8)])


# --- validation: per-element ---


def test_stitch_images_rejects_non_ndarray_element(monkeypatch: pytest.MonkeyPatch) -> None:
    _forbid_opencv_call(monkeypatch)

    with pytest.raises(TypeError, match=r"images\[1\] must be a NumPy array"):
        stitch_images([np.zeros((4, 4, 3), dtype=np.uint8), None])  # type: ignore[list-item]


def test_stitch_images_rejects_empty_image(monkeypatch: pytest.MonkeyPatch) -> None:
    _forbid_opencv_call(monkeypatch)
    empty = np.zeros((0, 4, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match=r"images\[1\] must not be empty"):
        stitch_images([np.zeros((4, 4, 3), dtype=np.uint8), empty])


@pytest.mark.parametrize("dtype", [np.uint16, np.float32, np.float64])
def test_stitch_images_rejects_wrong_dtype(dtype, monkeypatch: pytest.MonkeyPatch) -> None:
    _forbid_opencv_call(monkeypatch)
    bad = np.zeros((4, 4, 3), dtype=dtype)

    with pytest.raises(TypeError, match=r"images\[1\] must have dtype"):
        stitch_images([np.zeros((4, 4, 3), dtype=np.uint8), bad])  # type: ignore[list-item]


def test_stitch_images_rejects_grayscale(monkeypatch: pytest.MonkeyPatch) -> None:
    _forbid_opencv_call(monkeypatch)
    gray = np.zeros((4, 4), dtype=np.uint8)

    with pytest.raises(ValueError, match=r"images\[1\].*grayscale"):
        stitch_images([np.zeros((4, 4, 3), dtype=np.uint8), gray])


def test_stitch_images_rejects_single_channel_with_trailing_axis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_opencv_call(monkeypatch)
    hw1 = np.zeros((4, 4, 1), dtype=np.uint8)

    with pytest.raises(ValueError, match=r"images\[1\].*single-channel"):
        stitch_images([np.zeros((4, 4, 3), dtype=np.uint8), hw1])


def test_stitch_images_rejects_two_channels(monkeypatch: pytest.MonkeyPatch) -> None:
    _forbid_opencv_call(monkeypatch)
    two_channel = np.zeros((4, 4, 2), dtype=np.uint8)

    with pytest.raises(ValueError, match=r"images\[1\] must have shape \(H, W, 3\)"):
        stitch_images([np.zeros((4, 4, 3), dtype=np.uint8), two_channel])


def test_stitch_images_rejects_bgra(monkeypatch: pytest.MonkeyPatch) -> None:
    _forbid_opencv_call(monkeypatch)
    bgra = np.zeros((4, 4, 4), dtype=np.uint8)

    with pytest.raises(ValueError, match=r"images\[1\].*BGRA"):
        stitch_images([np.zeros((4, 4, 3), dtype=np.uint8), bgra])


def test_stitch_images_rejects_invalid_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    _forbid_opencv_call(monkeypatch)

    with pytest.raises(ValueError, match="mode must be one of"):
        stitch_images([np.zeros((4, 4, 3), dtype=np.uint8)] * 2, mode="bogus")  # type: ignore[arg-type]


# --- public exports ---


def test_public_exports() -> None:
    assert im.stitch_images is stitch_images
