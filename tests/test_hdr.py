import math
from collections.abc import Sequence

import cv2
import numpy as np
import pytest

import improcv as im
from improcv.hdr import fuse_exposures

_WEIGHT_NAMES = ["contrast_weight", "saturation_weight", "exposure_weight"]


def _make_stack(
    rng: np.random.Generator, count: int = 3, height: int = 20, width: int = 24
) -> list[np.ndarray]:
    return [rng.integers(0, 256, (height, width, 3), dtype=np.uint8) for _ in range(count)]


class _CustomSequence(Sequence):
    """A minimal, real `collections.abc.Sequence` that is neither list nor tuple."""

    def __init__(self, items: list[np.ndarray]) -> None:
        self._items = items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index):
        return self._items[index]


# --- basic behavior ---


def test_returns_expected_shape_and_dtype() -> None:
    rng = np.random.default_rng(0)
    images = _make_stack(rng)

    result = fuse_exposures(images)

    assert result.shape == images[0].shape
    assert result.dtype == np.float32


def test_result_is_finite() -> None:
    rng = np.random.default_rng(1)
    images = _make_stack(rng)

    result = fuse_exposures(images)

    assert np.all(np.isfinite(result))


def test_does_not_mutate_input_images() -> None:
    rng = np.random.default_rng(2)
    images = _make_stack(rng)
    before = [image.copy() for image in images]

    fuse_exposures(images)

    for image, original in zip(images, before, strict=True):
        np.testing.assert_array_equal(image, original)


def test_does_not_mutate_input_container() -> None:
    rng = np.random.default_rng(3)
    images = _make_stack(rng)
    container = list(images)
    before_ids = [id(image) for image in container]

    fuse_exposures(container)

    assert [id(image) for image in container] == before_ids
    assert len(container) == 3


def test_output_does_not_share_memory_with_inputs() -> None:
    rng = np.random.default_rng(4)
    images = _make_stack(rng)

    result = fuse_exposures(images)

    for image in images:
        assert not np.shares_memory(result, image)


def test_accepts_grayscale_stack() -> None:
    rng = np.random.default_rng(5)
    images = [rng.integers(0, 256, (20, 24), dtype=np.uint8) for _ in range(3)]

    result = fuse_exposures(images)

    assert result.shape == (20, 24)
    assert result.dtype == np.float32


def test_accepts_bgr_stack() -> None:
    rng = np.random.default_rng(6)
    images = _make_stack(rng)

    result = fuse_exposures(images)

    assert result.shape == images[0].shape


def test_result_may_lie_slightly_outside_zero_one() -> None:
    # Pinned, deterministic case demonstrating that the Laplacian-pyramid
    # reconstruction can under/overshoot -- not a universal guarantee for
    # every input, just evidence the contract must not promise [0, 1].
    # Verified directly: seed 0 reliably undershoots below 0 and overshoots
    # above 1 on both OpenCV 4.13 and 5.0.
    rng = np.random.default_rng(0)
    images = _make_stack(rng, height=32, width=32)

    result = fuse_exposures(images)

    assert result.min() < 0.0
    assert result.max() > 1.0


# --- container types ---


def test_accepts_list() -> None:
    rng = np.random.default_rng(8)
    images = _make_stack(rng)

    result = fuse_exposures(images)

    assert result is not None


def test_accepts_tuple() -> None:
    rng = np.random.default_rng(9)
    images = tuple(_make_stack(rng))

    result = fuse_exposures(images)

    assert result is not None


def test_accepts_custom_sequence() -> None:
    rng = np.random.default_rng(10)
    images = _CustomSequence(_make_stack(rng))

    result = fuse_exposures(images)

    assert result is not None


def test_rejects_single_4d_array() -> None:
    rng = np.random.default_rng(11)
    images = np.stack(_make_stack(rng))

    with pytest.raises(TypeError, match="Sequence"):
        fuse_exposures(images)  # type: ignore[arg-type]


def test_rejects_generator() -> None:
    rng = np.random.default_rng(12)
    images = _make_stack(rng)

    with pytest.raises(TypeError, match="Sequence"):
        fuse_exposures(image for image in images)  # type: ignore[arg-type]


def test_rejects_string() -> None:
    with pytest.raises(TypeError, match="Sequence"):
        fuse_exposures("not an image stack")  # type: ignore[arg-type]


def test_rejects_bytes() -> None:
    with pytest.raises(TypeError, match="Sequence"):
        fuse_exposures(b"not an image stack")  # type: ignore[arg-type]


# --- image count ---


def test_rejects_zero_images() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        fuse_exposures([])


def test_rejects_one_image() -> None:
    rng = np.random.default_rng(13)
    images = _make_stack(rng, count=1)

    with pytest.raises(ValueError, match="at least 2"):
        fuse_exposures(images)


def test_accepts_two_images() -> None:
    rng = np.random.default_rng(14)
    images = _make_stack(rng, count=2)

    result = fuse_exposures(images)

    assert result is not None


def test_accepts_more_than_two_images() -> None:
    rng = np.random.default_rng(15)
    images = _make_stack(rng, count=5)

    result = fuse_exposures(images)

    assert result is not None


# --- per-element validation ---


def test_rejects_non_ndarray_element_with_index() -> None:
    rng = np.random.default_rng(16)
    images = _make_stack(rng)
    images[2] = "not an array"  # type: ignore[list-item]

    with pytest.raises(TypeError, match=r"images\[2\]"):
        fuse_exposures(images)


def test_rejects_non_ndarray_first_element_with_index() -> None:
    with pytest.raises(TypeError, match=r"images\[0\]"):
        fuse_exposures(["not an array", "also not an array"])  # type: ignore[list-item]


def test_rejects_mismatched_shape_with_index() -> None:
    rng = np.random.default_rng(17)
    images = _make_stack(rng, count=3)
    images[2] = rng.integers(0, 256, (10, 10, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match=r"images\[2\]"):
        fuse_exposures(images)


def test_rejects_mismatched_dtype_with_index() -> None:
    rng = np.random.default_rng(18)
    images = _make_stack(rng, count=3)
    images[1] = images[1].astype(np.float32)  # type: ignore[assignment] # same shape, wrong dtype

    with pytest.raises(TypeError, match=r"images\[1\]"):
        fuse_exposures(images)


def test_rejects_empty_first_image() -> None:
    empty = np.zeros((0, 10, 3), dtype=np.uint8)
    other = np.zeros((0, 10, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match=r"images\[0\]"):
        fuse_exposures([empty, other])


@pytest.mark.parametrize(
    "bad_shape",
    [(20, 24, 1), (20, 24, 2), (20, 24, 4)],
    ids=["h_w_1", "2ch", "bgra"],
)
def test_rejects_unsupported_channel_counts(bad_shape: tuple[int, ...]) -> None:
    rng = np.random.default_rng(19)
    images = [rng.integers(0, 256, bad_shape, dtype=np.uint8) for _ in range(3)]

    with pytest.raises(ValueError, match=r"images\[0\]"):
        fuse_exposures(images)


def test_rejects_mixed_grayscale_and_bgr_stack() -> None:
    rng = np.random.default_rng(20)
    gray = rng.integers(0, 256, (20, 24), dtype=np.uint8)
    bgr = rng.integers(0, 256, (20, 24, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match=r"images\[1\]"):
        fuse_exposures([gray, bgr])


@pytest.mark.parametrize("dtype", [np.uint16, np.int32, np.float32, np.float64])
def test_rejects_non_uint8_first_image(dtype) -> None:
    rng = np.random.default_rng(21)
    images = [rng.integers(0, 256, (20, 24, 3)).astype(dtype) for _ in range(3)]

    with pytest.raises(TypeError, match=r"images\[0\]"):
        fuse_exposures(images)


@pytest.mark.parametrize("shape", [(1, 1, 3), (1, 24, 3), (24, 1, 3)])
def test_accepts_tiny_and_thin_images(shape: tuple[int, ...]) -> None:
    rng = np.random.default_rng(22)
    images = [rng.integers(0, 256, shape, dtype=np.uint8) for _ in range(3)]

    result = fuse_exposures(images)

    assert result.shape == shape


def test_accepts_non_contiguous_images() -> None:
    rng = np.random.default_rng(23)
    big = rng.integers(0, 256, (40, 48, 3), dtype=np.uint8)
    view = big[::2, ::2]
    assert not view.flags["C_CONTIGUOUS"]
    images = [view, view.copy(), view.copy()]

    result = fuse_exposures(images)

    assert result.shape == view.shape


# --- weights: type and range ---


@pytest.mark.parametrize("weight", _WEIGHT_NAMES)
def test_accepts_weight_zero(weight: str) -> None:
    rng = np.random.default_rng(24)
    images = _make_stack(rng)

    result = fuse_exposures(images, **{weight: 0.0})

    assert result is not None


def test_accepts_all_weights_zero() -> None:
    rng = np.random.default_rng(25)
    images = _make_stack(rng)

    result = fuse_exposures(images, contrast_weight=0.0, saturation_weight=0.0, exposure_weight=0.0)

    assert np.all(np.isfinite(result))


@pytest.mark.parametrize("weight", _WEIGHT_NAMES)
def test_rejects_negative_weight(weight: str) -> None:
    rng = np.random.default_rng(26)
    images = _make_stack(rng)

    with pytest.raises(ValueError, match="non-negative"):
        fuse_exposures(images, **{weight: -1.0})


@pytest.mark.parametrize("weight", _WEIGHT_NAMES)
def test_rejects_bool_weight(weight: str) -> None:
    rng = np.random.default_rng(27)
    images = _make_stack(rng)

    with pytest.raises(TypeError):
        fuse_exposures(images, **{weight: True})


@pytest.mark.parametrize("weight", _WEIGHT_NAMES)
def test_rejects_nan_weight(weight: str) -> None:
    rng = np.random.default_rng(28)
    images = _make_stack(rng)

    with pytest.raises(ValueError):
        fuse_exposures(images, **{weight: math.nan})


@pytest.mark.parametrize("weight", _WEIGHT_NAMES)
def test_rejects_inf_weight(weight: str) -> None:
    rng = np.random.default_rng(29)
    images = _make_stack(rng)

    with pytest.raises(ValueError):
        fuse_exposures(images, **{weight: math.inf})


@pytest.mark.parametrize("weight", _WEIGHT_NAMES)
def test_accepts_int_weight(weight: str) -> None:
    rng = np.random.default_rng(30)
    images = _make_stack(rng)

    result = fuse_exposures(images, **{weight: 2})

    assert result is not None


@pytest.mark.parametrize("weight", _WEIGHT_NAMES)
def test_accepts_numpy_real_scalar_weight(weight: str) -> None:
    rng = np.random.default_rng(31)
    images = _make_stack(rng)

    result = fuse_exposures(images, **{weight: np.float32(2.0)})  # type: ignore[arg-type]

    assert result is not None


@pytest.mark.parametrize("weight", _WEIGHT_NAMES)
@pytest.mark.parametrize("value", [1e-46, 1e-100, np.nextafter(0.0, 1.0)])
def test_rejects_weight_underflowing_to_zero_in_float32(weight: str, value: float) -> None:
    assert value > 0.0
    assert np.float32(value) == 0.0
    rng = np.random.default_rng(32)
    images = _make_stack(rng)

    with pytest.raises(ValueError, match="too small"):
        fuse_exposures(images, **{weight: value})


@pytest.mark.parametrize("weight", _WEIGHT_NAMES)
def test_rejects_weight_overflowing_to_inf_in_float32(weight: str) -> None:
    rng = np.random.default_rng(33)
    images = _make_stack(rng)

    with pytest.raises(ValueError, match="too large"):
        fuse_exposures(images, **{weight: 1e40})


@pytest.mark.parametrize("weight", _WEIGHT_NAMES)
def test_rejects_huge_int_weight_with_controlled_value_error(weight: str) -> None:
    rng = np.random.default_rng(34)
    images = _make_stack(rng)

    try:
        fuse_exposures(images, **{weight: 10**400})
    except OverflowError:
        pytest.fail(f"a raw OverflowError propagated for an oversized int {weight}")
    except ValueError:
        pass


def test_extreme_but_representable_weight_raises_runtime_error() -> None:
    rng = np.random.default_rng(35)
    images = _make_stack(rng)

    with pytest.raises(RuntimeError, match="finite"):
        fuse_exposures(images, contrast_weight=1e10)


def test_never_reaches_opencv_for_invalid_weight(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def boom(*args: object, **kwargs: object) -> None:
        nonlocal called
        called = True
        raise AssertionError("cv2 must not be called for an invalid weight")

    monkeypatch.setattr(cv2, "createMergeMertens", boom)

    rng = np.random.default_rng(36)
    images = _make_stack(rng)
    with pytest.raises(ValueError):
        fuse_exposures(images, contrast_weight=-1.0)
    assert not called


def test_never_reaches_opencv_for_invalid_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def boom(*args: object, **kwargs: object) -> None:
        nonlocal called
        called = True
        raise AssertionError("cv2 must not be called for an invalid stack")

    monkeypatch.setattr(cv2, "createMergeMertens", boom)

    with pytest.raises(ValueError):
        fuse_exposures([])
    assert not called


# --- validation order ---


def test_validation_order_stack_before_weights() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        fuse_exposures([], contrast_weight=-1.0)


# --- valid-path integration: wrapper vs direct cv2, same process ---


def test_matches_direct_cv2_call_with_defaults() -> None:
    rng = np.random.default_rng(37)
    images = _make_stack(rng, height=40, width=48)

    result = fuse_exposures(images)
    expected = cv2.createMergeMertens().process(images)

    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)


def test_matches_direct_cv2_call_with_explicit_weights() -> None:
    rng = np.random.default_rng(38)
    images = _make_stack(rng, height=40, width=48)

    result = fuse_exposures(images, contrast_weight=0.5, saturation_weight=2.0, exposure_weight=1.0)
    expected = cv2.createMergeMertens(0.5, 2.0, 1.0).process(images)

    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)


def test_two_independent_calls_are_not_guaranteed_bit_identical() -> None:
    # Documents non-determinism -- not something to rely on in either
    # direction, so this only demonstrates it is possible, without pinning
    # the exact difference.
    rng = np.random.default_rng(39)
    images = _make_stack(rng, height=64, width=64)

    first = fuse_exposures(images)
    second = fuse_exposures(images)

    np.testing.assert_allclose(first, second, rtol=1e-4, atol=1e-4)


# --- quality: fusion should meaningfully improve a known scenario, without
# a universal "always better" claim ---


def test_fuses_under_and_over_exposed_frames_into_a_better_exposed_image() -> None:
    rng = np.random.default_rng(40)
    scene = rng.integers(40, 216, (48, 48, 3), dtype=np.uint8).astype(np.float64)
    dark = np.clip(scene * 0.3, 0, 255).astype(np.uint8)
    bright = np.clip(scene * 2.2, 0, 255).astype(np.uint8)

    result = fuse_exposures([dark, bright])

    # Wide, platform-robust margin: the fused result's mid-gray closeness
    # should clearly beat either single frame's, not an exact value.
    mid_gray_error_dark = np.abs(dark.astype(np.float64) / 255.0 - 0.5).mean()
    mid_gray_error_bright = np.abs(bright.astype(np.float64) / 255.0 - 0.5).mean()
    mid_gray_error_fused = np.abs(result.astype(np.float64) - 0.5).mean()
    assert mid_gray_error_fused < 0.7 * min(mid_gray_error_dark, mid_gray_error_bright)


# --- public exports ---


def test_public_exports() -> None:
    assert im.fuse_exposures is fuse_exposures
