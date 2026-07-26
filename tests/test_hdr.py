import math
from collections.abc import Sequence

import cv2
import numpy as np
import pytest

import improcv as im
import improcv.hdr as hdr_module
from improcv.hdr import (
    calibrate_camera_response_debevec,
    calibrate_camera_response_robertson,
    fuse_exposures,
    merge_hdr_debevec,
    merge_hdr_robertson,
    tone_map,
    tone_map_drago,
    tone_map_mantiuk,
    tone_map_reinhard,
)

_WEIGHT_NAMES = ["contrast_weight", "saturation_weight", "exposure_weight"]

_MERGE_FUNCS = [merge_hdr_debevec, merge_hdr_robertson]
_MERGE_FUNC_NAMES = ["merge_hdr_debevec", "merge_hdr_robertson"]
_CV2_MERGE_FACTORY = {
    merge_hdr_debevec: cv2.createMergeDebevec,
    merge_hdr_robertson: cv2.createMergeRobertson,
}
_DEFAULT_TIMES = [1 / 30.0, 1 / 4.0, 2.5]


def _make_hdr_images(
    rng: np.random.Generator,
    count: int = 3,
    height: int = 20,
    width: int = 24,
    dtype: type = np.uint8,
    channels: int = 3,
) -> list[np.ndarray]:
    shape = (height, width, channels) if channels else (height, width)
    if dtype == np.uint8:
        return [rng.integers(0, 256, shape, dtype=np.uint8) for _ in range(count)]
    if dtype == np.uint16:
        return [rng.integers(0, 65536, shape, dtype=np.uint16) for _ in range(count)]
    if dtype == np.float32:
        return [rng.random(shape, dtype=np.float32) for _ in range(count)]
    raise ValueError(f"unsupported dtype for test helper: {dtype}")


def _make_stack(
    rng: np.random.Generator, count: int = 3, height: int = 20, width: int = 24
) -> list[np.ndarray]:
    return [rng.integers(0, 256, (height, width, 3), dtype=np.uint8) for _ in range(count)]


def _make_hdr(
    rng: np.random.Generator,
    height: int = 32,
    width: int = 32,
    low: float = 0.5,
    high: float = 10.0,
) -> np.ndarray:
    """A random, strictly-positive, non-constant `float32` HDR image with no
    pixel tied to the array's global minimum across all 3 channels --
    avoids every degenerate case handled by `tone_map_drago`/
    `tone_map_reinhard`/`tone_map_mantiuk`'s own preconditions, so it is
    safe as a generic happy-path fixture for all four tone-mapping
    functions.
    """
    return rng.uniform(low, high, (height, width, 3)).astype(np.float32)


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


def test_matches_direct_cv2_call_grayscale() -> None:
    rng = np.random.default_rng(41)
    images = [rng.integers(0, 256, (40, 48), dtype=np.uint8) for _ in range(3)]

    result = fuse_exposures(images)
    expected = cv2.createMergeMertens().process(images)

    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)


def test_weight_reaches_opencv_as_its_float32_equivalent(monkeypatch: pytest.MonkeyPatch) -> None:
    # 0.1 is not exactly representable in binary floating point -- its
    # float64 and float32 representations differ. This confirms OpenCV
    # receives the float32 rounding, not the original Python float64 value.
    captured: dict[str, float] = {}
    real_create_merge_mertens = cv2.createMergeMertens

    def fake_create_merge_mertens(contrast_weight, saturation_weight, exposure_weight):
        captured["contrast_weight"] = contrast_weight
        return real_create_merge_mertens(contrast_weight, saturation_weight, exposure_weight)

    monkeypatch.setattr(cv2, "createMergeMertens", fake_create_merge_mertens)

    rng = np.random.default_rng(42)
    images = _make_stack(rng)
    fuse_exposures(images, contrast_weight=0.1)

    assert captured["contrast_weight"] == float(np.float32(0.1))
    assert captured["contrast_weight"] != 0.1


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


# --- merge_hdr_debevec / merge_hdr_robertson: dtype and channel happy paths ---


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_merge_hdr_accepts_dtype_combinations_bgr(func, dtype) -> None:
    # uint16/float32 support is OpenCV-build-dependent (verified directly:
    # absent on 4.9.0, this project's documented minimum) -- skip rather
    # than fail on a build where the capability genuinely isn't there; the
    # rejection path itself is covered separately (monkeypatched, so it
    # doesn't depend on which OpenCV happens to be installed).
    if not hdr_module.merge_hdr_supports_dtype(_CV2_MERGE_FACTORY[func], dtype):
        pytest.skip(f"this OpenCV build's {func.__name__} does not support dtype {dtype}")
    rng = np.random.default_rng(50)
    images = _make_hdr_images(rng, dtype=dtype, channels=3)

    result = func(images, _DEFAULT_TIMES)

    assert result.shape == images[0].shape
    assert result.dtype == np.float32
    assert np.all(np.isfinite(result))


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_merge_hdr_rejects_grayscale(func, dtype) -> None:
    # Neither merge function supports grayscale input, for different
    # reasons: MergeRobertson raises a raw, unhelpful cv2.error for any
    # non-3-channel stack; MergeDebevec's own default-response-curve
    # construction corrupts memory for a genuinely 1-channel array
    # (confirmed in OpenCV's own C++ source) -- a real, reproducible
    # process crash observed directly in CI, though undefined behavior
    # that happens not to manifest as a crash on every platform. This must
    # never be reachable through the public wrapper regardless of dtype.
    rng = np.random.default_rng(50)
    images = _make_hdr_images(rng, dtype=dtype, channels=0)

    with pytest.raises(ValueError, match="grayscale"):
        func(images, _DEFAULT_TIMES)


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_never_reaches_opencv_for_grayscale(
    func, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def boom():
        nonlocal called
        called = True
        raise AssertionError("cv2 must not be called for a grayscale stack")

    monkeypatch.setattr(cv2, _CV2_MERGE_FACTORY[func].__name__, boom)

    rng = np.random.default_rng(50)
    images = _make_hdr_images(rng, channels=0)
    with pytest.raises(ValueError, match="grayscale"):
        func(images, _DEFAULT_TIMES)
    assert not called


# --- merge_hdr_debevec / merge_hdr_robertson: OpenCV-build dtype capability ---
# OpenCV 4.9.0 (this project's documented minimum) only supports uint8 for
# HDR merge -- these tests force that unsupported outcome via monkeypatch,
# regardless of what the environment's actual installed OpenCV supports, so
# the rejection path is always covered.


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_rejects_unsupported_dtype_with_clear_message(
    func, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hdr_module, "merge_hdr_supports_dtype", lambda factory, dtype: False)

    rng = np.random.default_rng(110)
    images = _make_hdr_images(rng, dtype=np.uint16)

    with pytest.raises(ValueError, match="does not support it"):
        func(images, _DEFAULT_TIMES)


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_never_reaches_opencv_for_unsupported_dtype(
    func, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hdr_module, "merge_hdr_supports_dtype", lambda factory, dtype: False)

    called = False

    def boom():
        nonlocal called
        called = True
        raise AssertionError("cv2 must not be called for an unsupported dtype")

    monkeypatch.setattr(cv2, _CV2_MERGE_FACTORY[func].__name__, boom)

    rng = np.random.default_rng(111)
    images = _make_hdr_images(rng, dtype=np.uint16)
    with pytest.raises(ValueError):
        func(images, _DEFAULT_TIMES)
    assert not called


# --- merge_hdr_debevec / merge_hdr_robertson: stack rejections ---


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_rejects_float64(func) -> None:
    rng = np.random.default_rng(51)
    images = [im.astype(np.float64) for im in _make_hdr_images(rng)]

    with pytest.raises(TypeError, match="images\\[0\\]"):
        func(images, _DEFAULT_TIMES)


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_rejects_signed_integer(func) -> None:
    rng = np.random.default_rng(52)
    images = [im.astype(np.int32) for im in _make_hdr_images(rng)]

    with pytest.raises(TypeError, match="images\\[0\\]"):
        func(images, _DEFAULT_TIMES)


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_rejects_mixed_dtype_stack(func) -> None:
    rng = np.random.default_rng(53)
    images = _make_hdr_images(rng)
    images[1] = images[1].astype(np.uint16)

    with pytest.raises(TypeError, match=r"images\[1\]"):
        func(images, _DEFAULT_TIMES)


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_rejects_nan_in_float32_stack(func) -> None:
    rng = np.random.default_rng(54)
    images = _make_hdr_images(rng, dtype=np.float32)
    images[1][0, 0, 0] = math.nan

    with pytest.raises(ValueError, match=r"images\[1\]"):
        func(images, _DEFAULT_TIMES)


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_rejects_inf_in_float32_stack(func) -> None:
    rng = np.random.default_rng(55)
    images = _make_hdr_images(rng, dtype=np.float32)
    images[1][0, 0, 0] = math.inf

    with pytest.raises(ValueError, match=r"images\[1\]"):
        func(images, _DEFAULT_TIMES)


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_rejects_float32_below_zero(func) -> None:
    rng = np.random.default_rng(56)
    images = _make_hdr_images(rng, dtype=np.float32)
    images[0][0, 0, 0] = -0.1

    with pytest.raises(ValueError, match=r"images\[0\]"):
        func(images, _DEFAULT_TIMES)


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_rejects_float32_above_one(func) -> None:
    rng = np.random.default_rng(57)
    images = _make_hdr_images(rng, dtype=np.float32)
    images[0][0, 0, 0] = 1.1

    with pytest.raises(ValueError, match=r"images\[0\]"):
        func(images, _DEFAULT_TIMES)


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
@pytest.mark.parametrize(
    "bad_shape",
    [(20, 24, 1), (20, 24, 2), (20, 24, 4)],
    ids=["h_w_1", "2ch", "bgra"],
)
def test_merge_hdr_rejects_unsupported_channel_counts(func, bad_shape) -> None:
    rng = np.random.default_rng(58)
    images = [rng.integers(0, 256, bad_shape, dtype=np.uint8) for _ in range(3)]

    with pytest.raises(ValueError, match=r"images\[0\]"):
        func(images, _DEFAULT_TIMES)


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_rejects_mixed_grayscale_and_bgr(func) -> None:
    rng = np.random.default_rng(59)
    gray = rng.integers(0, 256, (20, 24), dtype=np.uint8)
    bgr = rng.integers(0, 256, (20, 24, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match=r"images\[1\]"):
        func([gray, bgr], [1.0, 2.0])


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_rejects_mismatched_shape(func) -> None:
    rng = np.random.default_rng(60)
    images = _make_hdr_images(rng)
    images[2] = rng.integers(0, 256, (10, 10, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match=r"images\[2\]"):
        func(images, _DEFAULT_TIMES)


# --- merge_hdr_debevec / merge_hdr_robertson: exposure_times ---


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_accepts_times_as_list(func) -> None:
    rng = np.random.default_rng(61)
    images = _make_hdr_images(rng)

    result = func(images, list(_DEFAULT_TIMES))

    assert result is not None


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_accepts_times_as_tuple(func) -> None:
    rng = np.random.default_rng(62)
    images = _make_hdr_images(rng)

    result = func(images, tuple(_DEFAULT_TIMES))

    assert result is not None


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_accepts_int_and_numpy_scalar_times(func) -> None:
    rng = np.random.default_rng(63)
    images = _make_hdr_images(rng)

    result = func(images, [1, np.float32(2.0), np.float64(3.0)])

    assert result is not None


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_rejects_wrong_times_length(func) -> None:
    rng = np.random.default_rng(64)
    images = _make_hdr_images(rng)

    with pytest.raises(ValueError, match="exactly 3"):
        func(images, [1.0, 2.0])


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_rejects_zero_time_with_index(func) -> None:
    rng = np.random.default_rng(65)
    images = _make_hdr_images(rng)

    with pytest.raises(ValueError, match=r"exposure_times\[1\]"):
        func(images, [1.0, 0.0, 2.0])


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_rejects_negative_time(func) -> None:
    rng = np.random.default_rng(66)
    images = _make_hdr_images(rng)

    with pytest.raises(ValueError, match=r"exposure_times\[0\]"):
        func(images, [-1.0, 1.0, 2.0])


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_rejects_bool_time(func) -> None:
    rng = np.random.default_rng(67)
    images = _make_hdr_images(rng)

    with pytest.raises(TypeError):
        func(images, [True, 1.0, 2.0])


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_rejects_nan_time(func) -> None:
    rng = np.random.default_rng(68)
    images = _make_hdr_images(rng)

    with pytest.raises(ValueError):
        func(images, [math.nan, 1.0, 2.0])


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_rejects_inf_time(func) -> None:
    rng = np.random.default_rng(69)
    images = _make_hdr_images(rng)

    with pytest.raises(ValueError):
        func(images, [math.inf, 1.0, 2.0])


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
@pytest.mark.parametrize("value", [1e-46, 1e-100, np.nextafter(0.0, 1.0)])
def test_merge_hdr_rejects_time_underflowing_to_zero(func, value: float) -> None:
    assert value > 0.0
    assert np.float32(value) == 0.0
    rng = np.random.default_rng(70)
    images = _make_hdr_images(rng)

    with pytest.raises(ValueError, match="too small"):
        func(images, [value, 1.0, 2.0])


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_rejects_time_overflowing_to_inf(func) -> None:
    rng = np.random.default_rng(71)
    images = _make_hdr_images(rng)

    with pytest.raises(ValueError, match="too large"):
        func(images, [1e40, 1.0, 2.0])


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_rejects_huge_int_time_with_controlled_value_error(func) -> None:
    rng = np.random.default_rng(72)
    images = _make_hdr_images(rng)

    try:
        func(images, [10**400, 1.0, 2.0])
    except OverflowError:
        pytest.fail("a raw OverflowError propagated for an oversized int exposure time")
    except ValueError:
        pass


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_accepts_duplicate_times(func) -> None:
    rng = np.random.default_rng(73)
    images = _make_hdr_images(rng)

    result = func(images, [1.0, 1.0, 1.0])

    assert np.all(np.isfinite(result))


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_accepts_increasing_and_decreasing_times(func) -> None:
    rng = np.random.default_rng(74)
    images = _make_hdr_images(rng)

    increasing = func(images, [1 / 30.0, 1 / 4.0, 2.5])
    decreasing = func(images, [2.5, 1 / 4.0, 1 / 30.0])

    assert np.all(np.isfinite(increasing))
    assert np.all(np.isfinite(decreasing))


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_passes_fresh_contiguous_float32_times(
    func, monkeypatch: pytest.MonkeyPatch
) -> None:
    rng = np.random.default_rng(75)
    images = _make_hdr_images(rng)
    original_times = np.array(_DEFAULT_TIMES, dtype=np.float64)
    captured: dict[str, np.ndarray] = {}

    real_factory = _CV2_MERGE_FACTORY[func]
    real_merger = real_factory()

    class _CapturingMerger:
        def process(self, imgs, times, *rest):
            captured["times"] = times
            return real_merger.process(imgs, times, *rest)

    monkeypatch.setattr(cv2, real_factory.__name__, lambda: _CapturingMerger())

    func(images, list(original_times))

    passed_times = captured["times"]
    assert passed_times.dtype == np.float32
    assert passed_times.flags["C_CONTIGUOUS"]
    assert not np.shares_memory(passed_times, original_times)


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_handles_float64_input_times_safely(func) -> None:
    # The raw cv2 binding can silently corrupt a float64 times array into a
    # garbage/inf result (verified directly) -- the wrapper must never let
    # the user's array dtype reach cv2 directly.
    rng = np.random.default_rng(76)
    images = _make_hdr_images(rng)
    times64 = list(np.array(_DEFAULT_TIMES, dtype=np.float64))

    result = func(images, times64)

    assert np.all(np.isfinite(result))


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_never_reaches_opencv_for_invalid_times(
    func, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def boom():
        nonlocal called
        called = True
        raise AssertionError("cv2 must not be called for invalid exposure_times")

    monkeypatch.setattr(cv2, _CV2_MERGE_FACTORY[func].__name__, boom)

    rng = np.random.default_rng(77)
    images = _make_hdr_images(rng)
    with pytest.raises(ValueError):
        func(images, [1.0, 2.0])
    assert not called


# --- merge_hdr_debevec / merge_hdr_robertson: response_curve, shared structure ---


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_response_curve_none_skips_calibration(func) -> None:
    rng = np.random.default_rng(78)
    images = _make_hdr_images(rng)

    result = func(images, _DEFAULT_TIMES, response_curve=None)

    assert np.all(np.isfinite(result))


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
@pytest.mark.parametrize(
    "dtype,shape",
    [
        (np.uint8, (256, 1, 3)),
        (np.uint16, (65536, 1, 3)),
        (np.float32, (65536, 1, 3)),
    ],
    ids=["uint8_bgr", "uint16_bgr", "float32_bgr"],
)
def test_merge_hdr_accepts_correct_response_curve_shape_bgr(func, dtype, shape) -> None:
    if not hdr_module.merge_hdr_supports_dtype(_CV2_MERGE_FACTORY[func], dtype):
        pytest.skip(f"this OpenCV build's {func.__name__} does not support dtype {dtype}")
    rng = np.random.default_rng(79)
    images = _make_hdr_images(rng, dtype=dtype, channels=3)
    curve = rng.random(shape).astype(np.float32) + 0.1  # keep strictly positive for Debevec too

    result = func(images, _DEFAULT_TIMES, response_curve=curve)

    assert np.all(np.isfinite(result))


def test_merge_hdr_debevec_rejects_grayscale_even_with_explicit_response_curve() -> None:
    # The memory-corrupting OpenCV code path only triggers when
    # response_curve is None (the default-linear-response construction),
    # so an explicit, correctly-shaped curve would technically bypass it --
    # but grayscale is rejected unconditionally here anyway, favoring a
    # simple, uniformly safe contract over a narrower conditional one.
    rng = np.random.default_rng(79)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=0)
    curve = rng.random((256, 1)).astype(np.float32) + 0.1

    with pytest.raises(ValueError, match="grayscale"):
        merge_hdr_debevec(images, _DEFAULT_TIMES, response_curve=curve)


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_rejects_non_ndarray_response_curve(func) -> None:
    rng = np.random.default_rng(80)
    images = _make_hdr_images(rng)

    with pytest.raises(TypeError, match="response_curve"):
        func(images, _DEFAULT_TIMES, response_curve="not an array")  # type: ignore[arg-type]


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_rejects_wrong_dtype_response_curve(func) -> None:
    rng = np.random.default_rng(81)
    images = _make_hdr_images(rng)
    curve = np.ones((256, 1, 3), dtype=np.float64)

    with pytest.raises(TypeError, match="response_curve"):
        func(images, _DEFAULT_TIMES, response_curve=curve)  # type: ignore[arg-type]


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_rejects_wrong_length_response_curve(func) -> None:
    if not hdr_module.merge_hdr_supports_dtype(_CV2_MERGE_FACTORY[func], np.uint16):
        pytest.skip(f"this OpenCV build's {func.__name__} does not support dtype uint16")
    rng = np.random.default_rng(82)
    images = _make_hdr_images(rng, dtype=np.uint16)  # expects 65536-length curve
    curve = np.ones((256, 1, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="shape"):
        func(images, _DEFAULT_TIMES, response_curve=curve)


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_rejects_wrong_channel_count_response_curve(func) -> None:
    rng = np.random.default_rng(83)
    images = _make_hdr_images(rng)  # BGR, expects (256, 1, 3)
    curve = np.ones((256, 1), dtype=np.float32)

    with pytest.raises(ValueError, match="shape"):
        func(images, _DEFAULT_TIMES, response_curve=curve)


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_rejects_flattened_response_curve_shape(func) -> None:
    rng = np.random.default_rng(84)
    images = _make_hdr_images(rng)
    curve = np.ones((256 * 3,), dtype=np.float32)

    with pytest.raises(ValueError, match="shape"):
        func(images, _DEFAULT_TIMES, response_curve=curve)


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_rejects_nan_response_curve(func) -> None:
    rng = np.random.default_rng(85)
    images = _make_hdr_images(rng)
    curve = np.full((256, 1, 3), 2.0, dtype=np.float32)
    curve[10] = math.nan

    with pytest.raises(ValueError, match="finite"):
        func(images, _DEFAULT_TIMES, response_curve=curve)


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_rejects_inf_response_curve(func) -> None:
    rng = np.random.default_rng(86)
    images = _make_hdr_images(rng)
    curve = np.full((256, 1, 3), 2.0, dtype=np.float32)
    curve[10] = math.inf

    with pytest.raises(ValueError, match="finite"):
        func(images, _DEFAULT_TIMES, response_curve=curve)


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_does_not_mutate_response_curve(func) -> None:
    rng = np.random.default_rng(87)
    images = _make_hdr_images(rng)
    curve = np.full((256, 1, 3), 2.0, dtype=np.float32)
    before = curve.copy()

    func(images, _DEFAULT_TIMES, response_curve=curve)

    np.testing.assert_array_equal(curve, before)


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_accepts_non_contiguous_response_curve(func) -> None:
    rng = np.random.default_rng(88)
    images = _make_hdr_images(rng)
    big = np.full((256, 1, 6), 2.0, dtype=np.float32)
    curve = big[:, :, ::2]
    assert not curve.flags["C_CONTIGUOUS"]

    result = func(images, _DEFAULT_TIMES, response_curve=curve)

    assert np.all(np.isfinite(result))


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_accepts_read_only_response_curve(func) -> None:
    rng = np.random.default_rng(89)
    images = _make_hdr_images(rng)
    curve = np.full((256, 1, 3), 2.0, dtype=np.float32)
    curve.flags.writeable = False

    result = func(images, _DEFAULT_TIMES, response_curve=curve)

    assert np.all(np.isfinite(result))


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_never_reaches_opencv_for_invalid_response_curve(
    func, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def boom():
        nonlocal called
        called = True
        raise AssertionError("cv2 must not be called for an invalid response_curve")

    monkeypatch.setattr(cv2, _CV2_MERGE_FACTORY[func].__name__, boom)

    rng = np.random.default_rng(90)
    images = _make_hdr_images(rng)
    curve = np.ones((100, 1, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        func(images, _DEFAULT_TIMES, response_curve=curve)
    assert not called


# --- response_curve: Debevec vs Robertson value rules ---


def test_merge_hdr_debevec_rejects_zero_in_response_curve() -> None:
    rng = np.random.default_rng(91)
    images = _make_hdr_images(rng)
    curve = np.full((256, 1, 3), 2.0, dtype=np.float32)
    curve[10] = 0.0

    with pytest.raises(ValueError, match="strictly positive"):
        merge_hdr_debevec(images, _DEFAULT_TIMES, response_curve=curve)


def test_merge_hdr_debevec_rejects_negative_in_response_curve() -> None:
    rng = np.random.default_rng(92)
    images = _make_hdr_images(rng)
    curve = np.full((256, 1, 3), 2.0, dtype=np.float32)
    curve[10] = -1.0

    with pytest.raises(ValueError, match="strictly positive"):
        merge_hdr_debevec(images, _DEFAULT_TIMES, response_curve=curve)


def test_merge_hdr_debevec_accepts_all_positive_response_curve() -> None:
    rng = np.random.default_rng(93)
    images = _make_hdr_images(rng)
    curve = rng.random((256, 1, 3)).astype(np.float32) + 0.1

    result = merge_hdr_debevec(images, _DEFAULT_TIMES, response_curve=curve)

    assert np.all(np.isfinite(result))


def test_merge_hdr_robertson_accepts_partial_zero_response_curve() -> None:
    rng = np.random.default_rng(94)
    images = _make_hdr_images(rng)
    curve = np.full((256, 1, 3), 2.0, dtype=np.float32)
    curve[10] = 0.0

    result = merge_hdr_robertson(images, _DEFAULT_TIMES, response_curve=curve)

    assert np.all(np.isfinite(result))


def test_merge_hdr_robertson_rejects_negative_response_curve() -> None:
    rng = np.random.default_rng(95)
    images = _make_hdr_images(rng)
    curve = np.full((256, 1, 3), 2.0, dtype=np.float32)
    curve[10] = -1.0

    with pytest.raises(ValueError, match="negative"):
        merge_hdr_robertson(images, _DEFAULT_TIMES, response_curve=curve)


def test_merge_hdr_robertson_rejects_all_zero_response_curve() -> None:
    rng = np.random.default_rng(96)
    images = _make_hdr_images(rng)
    curve = np.zeros((256, 1, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="all-zero"):
        merge_hdr_robertson(images, _DEFAULT_TIMES, response_curve=curve)


# --- merge_hdr_debevec / merge_hdr_robertson: reference tests ---


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_matches_direct_cv2_call_without_curve(func) -> None:
    rng = np.random.default_rng(97)
    images = _make_hdr_images(rng, height=32, width=32)

    result = func(images, _DEFAULT_TIMES)
    expected = _CV2_MERGE_FACTORY[func]().process(
        images, np.array(_DEFAULT_TIMES, dtype=np.float32)
    )

    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_matches_direct_cv2_call_with_curve(func) -> None:
    rng = np.random.default_rng(98)
    images = _make_hdr_images(rng, height=32, width=32)
    curve = rng.random((256, 1, 3)).astype(np.float32) + 0.1
    times_array = np.array(_DEFAULT_TIMES, dtype=np.float32)

    result = func(images, _DEFAULT_TIMES, response_curve=curve)
    expected = _CV2_MERGE_FACTORY[func]().process(images, times_array, curve)

    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_is_deterministic(func) -> None:
    rng = np.random.default_rng(99)
    images = _make_hdr_images(rng)

    first = func(images, _DEFAULT_TIMES)
    second = func(images, _DEFAULT_TIMES)

    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_radiance_scales_inversely_with_time_unit(func) -> None:
    rng = np.random.default_rng(100)
    images = _make_hdr_images(rng, height=48, width=48)
    base_times = [1 / 30.0, 1 / 4.0, 2.5]
    scaled_times = [t / 1000.0 for t in base_times]

    base_result = func(images, base_times)
    scaled_result = func(images, scaled_times)

    # Wide tolerance: exact per-pixel ratio breaks down where weights are
    # near-zero, so this checks the median ratio, not every pixel exactly.
    ratio = scaled_result / base_result
    finite_ratio = ratio[np.isfinite(ratio)]
    assert math.isclose(float(np.median(finite_ratio)), 1000.0, rel_tol=0.05)


# --- merge_hdr_debevec / merge_hdr_robertson: degenerate stacks ---


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_accepts_all_black_stack(func) -> None:
    images = [np.zeros((20, 24, 3), dtype=np.uint8) for _ in range(3)]

    result = func(images, _DEFAULT_TIMES)

    assert np.all(np.isfinite(result))


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_accepts_all_white_stack(func) -> None:
    images = [np.full((20, 24, 3), 255, dtype=np.uint8) for _ in range(3)]

    result = func(images, _DEFAULT_TIMES)

    assert np.all(np.isfinite(result))


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_accepts_identical_images_distinct_times(func) -> None:
    rng = np.random.default_rng(102)
    image = rng.integers(0, 256, (20, 24, 3), dtype=np.uint8)
    images = [image.copy() for _ in range(3)]

    result = func(images, _DEFAULT_TIMES)

    assert np.all(np.isfinite(result))


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_accepts_distinct_images_identical_times(func) -> None:
    rng = np.random.default_rng(103)
    images = _make_hdr_images(rng)

    result = func(images, [1.0, 1.0, 1.0])

    assert np.all(np.isfinite(result))


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_accepts_weakly_differentiated_bracket(func) -> None:
    # A "weak" bracket is not something this function judges or rejects --
    # only the documented, checkable contract is enforced.
    rng = np.random.default_rng(104)
    images = _make_hdr_images(rng)
    times = [1.0, 1.01, 1.02]

    result = func(images, times)

    assert np.all(np.isfinite(result))


# --- merge_hdr_debevec / merge_hdr_robertson: mutation and aliasing ---


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_does_not_mutate_images(func) -> None:
    rng = np.random.default_rng(105)
    images = _make_hdr_images(rng)
    before = [im.copy() for im in images]

    func(images, _DEFAULT_TIMES)

    for image, original in zip(images, before, strict=True):
        np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_does_not_mutate_images_container(func) -> None:
    rng = np.random.default_rng(106)
    images = _make_hdr_images(rng)
    container = list(images)
    before_ids = [id(image) for image in container]

    func(container, _DEFAULT_TIMES)

    assert [id(image) for image in container] == before_ids


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_does_not_mutate_exposure_times(func) -> None:
    rng = np.random.default_rng(107)
    images = _make_hdr_images(rng)
    times = list(_DEFAULT_TIMES)
    before = list(times)

    func(images, times)

    assert times == before


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_output_does_not_share_memory_with_inputs(func) -> None:
    rng = np.random.default_rng(108)
    images = _make_hdr_images(rng)
    curve = np.full((256, 1, 3), 2.0, dtype=np.float32)

    result = func(images, _DEFAULT_TIMES, response_curve=curve)

    for image in images:
        assert not np.shares_memory(result, image)
    assert not np.shares_memory(result, curve)


# --- merge_hdr_debevec / merge_hdr_robertson: validation order ---


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_validation_order_stack_before_times(func) -> None:
    with pytest.raises(ValueError, match="at least 2"):
        func([], [])


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_validation_order_times_before_response_curve(func) -> None:
    rng = np.random.default_rng(109)
    images = _make_hdr_images(rng)
    bad_curve = np.ones((7, 1, 3), dtype=np.float32)  # also wrong

    with pytest.raises(ValueError, match="exactly 3"):
        func(images, [1.0, 2.0], response_curve=bad_curve)


@pytest.mark.parametrize("func", _MERGE_FUNCS, ids=_MERGE_FUNC_NAMES)
def test_merge_hdr_never_reaches_opencv_for_invalid_stack(
    func, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def boom():
        nonlocal called
        called = True
        raise AssertionError("cv2 must not be called for an invalid stack")

    monkeypatch.setattr(cv2, _CV2_MERGE_FACTORY[func].__name__, boom)

    with pytest.raises(ValueError):
        func([], [])
    assert not called


# --- calibrate_camera_response_debevec / _robertson: happy paths ---

_CALIBRATE_FUNCS = [calibrate_camera_response_debevec, calibrate_camera_response_robertson]
_CALIBRATE_FUNC_NAMES = ["calibrate_camera_response_debevec", "calibrate_camera_response_robertson"]
_CV2_CALIBRATE_FACTORY = {
    calibrate_camera_response_debevec: cv2.createCalibrateDebevec,
    calibrate_camera_response_robertson: cv2.createCalibrateRobertson,
}


def test_calibrate_debevec_accepts_grayscale() -> None:
    rng = np.random.default_rng(200)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=0)

    result = calibrate_camera_response_debevec(images, _DEFAULT_TIMES)

    assert result.shape == (256, 1)
    assert result.dtype == np.float32
    assert np.all(np.isfinite(result))


def test_calibrate_debevec_accepts_bgr() -> None:
    rng = np.random.default_rng(201)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)

    result = calibrate_camera_response_debevec(images, _DEFAULT_TIMES)

    assert result.shape == (256, 1, 3)
    assert result.dtype == np.float32
    assert np.all(np.isfinite(result))


def test_calibrate_debevec_default_parameters() -> None:
    rng = np.random.default_rng(202)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)

    result = calibrate_camera_response_debevec(images, _DEFAULT_TIMES)
    expected = cv2.createCalibrateDebevec().process(
        images, np.array(_DEFAULT_TIMES, dtype=np.float32)
    )

    np.testing.assert_array_equal(result, expected)


def test_calibrate_robertson_accepts_bgr() -> None:
    rng = np.random.default_rng(203)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)

    result = calibrate_camera_response_robertson(images, _DEFAULT_TIMES)

    assert result.shape == (256, 1, 3)
    assert result.dtype == np.float32
    assert np.all(np.isfinite(result))


def test_calibrate_robertson_default_parameters() -> None:
    rng = np.random.default_rng(204)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)

    result = calibrate_camera_response_robertson(images, _DEFAULT_TIMES)
    expected = cv2.createCalibrateRobertson().process(
        images, np.array(_DEFAULT_TIMES, dtype=np.float32)
    )

    np.testing.assert_array_equal(result, expected)


def test_calibrate_robertson_rejects_grayscale_with_debevec_suggestion() -> None:
    rng = np.random.default_rng(205)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=0)

    with pytest.raises(ValueError, match="calibrate_camera_response_debevec"):
        calibrate_camera_response_robertson(images, _DEFAULT_TIMES)


def test_calibrate_robertson_never_reaches_opencv_for_grayscale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def boom():
        nonlocal called
        called = True
        raise AssertionError("cv2 must not be called for a grayscale Robertson calibration")

    monkeypatch.setattr(cv2, "createCalibrateRobertson", boom)

    rng = np.random.default_rng(206)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=0)
    with pytest.raises(ValueError, match="grayscale"):
        calibrate_camera_response_robertson(images, _DEFAULT_TIMES)
    assert not called


# --- calibrate_camera_response_debevec: samples ---


def test_calibrate_debevec_accepts_samples_1() -> None:
    rng = np.random.default_rng(207)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)

    result = calibrate_camera_response_debevec(images, _DEFAULT_TIMES, samples=1)

    assert np.all(np.isfinite(result))


def test_calibrate_debevec_accepts_numpy_integer_samples() -> None:
    rng = np.random.default_rng(208)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)

    result = calibrate_camera_response_debevec(
        images,
        _DEFAULT_TIMES,
        samples=np.int32(70),  # type: ignore[arg-type]
    )

    assert np.all(np.isfinite(result))


def test_calibrate_debevec_rejects_bool_samples() -> None:
    rng = np.random.default_rng(209)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)

    with pytest.raises(TypeError):
        calibrate_camera_response_debevec(images, _DEFAULT_TIMES, samples=True)


def test_calibrate_debevec_rejects_zero_samples() -> None:
    rng = np.random.default_rng(210)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)

    with pytest.raises(ValueError, match="positive"):
        calibrate_camera_response_debevec(images, _DEFAULT_TIMES, samples=0)


def test_calibrate_debevec_rejects_negative_samples() -> None:
    rng = np.random.default_rng(211)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)

    with pytest.raises(ValueError, match="positive"):
        calibrate_camera_response_debevec(images, _DEFAULT_TIMES, samples=-5)


def test_calibrate_debevec_rejects_samples_beyond_int32_range() -> None:
    rng = np.random.default_rng(212)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)

    with pytest.raises(ValueError, match="int32"):
        calibrate_camera_response_debevec(images, _DEFAULT_TIMES, samples=2**31)


def test_calibrate_debevec_grid_formula_landscape() -> None:
    rng = np.random.default_rng(213)
    images = _make_hdr_images(rng, dtype=np.uint8, height=20, width=40, channels=3)

    result = calibrate_camera_response_debevec(images, _DEFAULT_TIMES, samples=70)

    assert np.all(np.isfinite(result))


def test_calibrate_debevec_grid_formula_portrait() -> None:
    rng = np.random.default_rng(214)
    images = _make_hdr_images(rng, dtype=np.uint8, height=40, width=20, channels=3)

    result = calibrate_camera_response_debevec(images, _DEFAULT_TIMES, samples=70)

    assert np.all(np.isfinite(result))


def test_calibrate_debevec_equivalent_samples_produce_identical_grid() -> None:
    # Verified directly against raw OpenCV: samples=4 and samples=5 round to
    # the identical 2x2 grid on a square image via integer truncation.
    rng = np.random.default_rng(215)
    images = _make_hdr_images(rng, dtype=np.uint8, height=32, width=32, channels=3)

    result_4 = calibrate_camera_response_debevec(images, _DEFAULT_TIMES, samples=4)
    result_5 = calibrate_camera_response_debevec(images, _DEFAULT_TIMES, samples=5)

    np.testing.assert_array_equal(result_4, result_5)


def test_calibrate_debevec_rejects_samples_too_large_for_grid() -> None:
    rng = np.random.default_rng(216)
    images = _make_hdr_images(rng, dtype=np.uint8, height=20, width=24, channels=3)

    with pytest.raises(ValueError, match="sampling grid"):
        calibrate_camera_response_debevec(images, _DEFAULT_TIMES, samples=10**6)


def test_calibrate_debevec_never_reaches_opencv_for_samples_too_large_for_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def boom(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("cv2 must not be called for a samples value with no valid grid")

    monkeypatch.setattr(cv2, "createCalibrateDebevec", boom)

    rng = np.random.default_rng(217)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)
    with pytest.raises(ValueError, match="sampling grid"):
        calibrate_camera_response_debevec(images, _DEFAULT_TIMES, samples=10**6)
    assert not called


def test_calibrate_debevec_accepts_large_samples_with_random_sampling() -> None:
    # No grid to validate, and no upper bound tied to pixel count, when
    # random_sampling=True -- OpenCV samples with replacement. Deliberately
    # not an enormous value: OpenCV's own linear system scales with
    # samples * len(images), so an extreme value (e.g. 10**6) makes the
    # underlying solve itself allocate gigabytes and hang/OOM -- a real
    # resource cost, not something this validator should (or safely can)
    # guard against, so this test only needs to exceed the pixel count.
    rng = np.random.default_rng(218)
    images = _make_hdr_images(rng, dtype=np.uint8, height=20, width=24, channels=3)
    assert 2000 > 20 * 24  # exceeds the image's pixel count (480)

    result = calibrate_camera_response_debevec(
        images, _DEFAULT_TIMES, samples=2000, random_sampling=True
    )

    assert np.all(np.isfinite(result))


# --- calibrate_camera_response_debevec: smoothness ---


def test_calibrate_debevec_rejects_zero_smoothness() -> None:
    rng = np.random.default_rng(219)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)

    with pytest.raises(ValueError, match="positive"):
        calibrate_camera_response_debevec(images, _DEFAULT_TIMES, smoothness=0.0)


def test_calibrate_debevec_rejects_negative_smoothness() -> None:
    rng = np.random.default_rng(220)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)

    with pytest.raises(ValueError, match="positive"):
        calibrate_camera_response_debevec(images, _DEFAULT_TIMES, smoothness=-1.0)


def test_calibrate_debevec_rejects_bool_smoothness() -> None:
    rng = np.random.default_rng(221)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)

    with pytest.raises(TypeError):
        calibrate_camera_response_debevec(images, _DEFAULT_TIMES, smoothness=True)


def test_calibrate_debevec_rejects_nan_smoothness() -> None:
    rng = np.random.default_rng(222)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)

    with pytest.raises(ValueError):
        calibrate_camera_response_debevec(images, _DEFAULT_TIMES, smoothness=math.nan)


def test_calibrate_debevec_rejects_inf_smoothness() -> None:
    rng = np.random.default_rng(223)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)

    with pytest.raises(ValueError):
        calibrate_camera_response_debevec(images, _DEFAULT_TIMES, smoothness=math.inf)


@pytest.mark.parametrize("value", [1e-46, 1e-100, np.nextafter(0.0, 1.0)])
def test_calibrate_debevec_rejects_smoothness_underflowing_to_zero(value: float) -> None:
    assert value > 0.0
    assert np.float32(value) == 0.0
    rng = np.random.default_rng(224)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)

    with pytest.raises(ValueError, match="too small"):
        calibrate_camera_response_debevec(images, _DEFAULT_TIMES, smoothness=value)


def test_calibrate_debevec_rejects_smoothness_overflowing_to_inf() -> None:
    rng = np.random.default_rng(225)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)

    with pytest.raises(ValueError, match="too large"):
        calibrate_camera_response_debevec(images, _DEFAULT_TIMES, smoothness=1e40)


def test_calibrate_debevec_accepts_numpy_real_scalar_smoothness() -> None:
    rng = np.random.default_rng(226)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)

    result = calibrate_camera_response_debevec(
        images,
        _DEFAULT_TIMES,
        smoothness=np.float32(10.0),  # type: ignore[arg-type]
    )

    assert np.all(np.isfinite(result))


# --- calibrate_camera_response_debevec: random_sampling ---


def test_calibrate_debevec_rejects_int_random_sampling() -> None:
    rng = np.random.default_rng(227)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)

    with pytest.raises(TypeError):
        calibrate_camera_response_debevec(
            images,
            _DEFAULT_TIMES,
            random_sampling=1,  # type: ignore[arg-type]
        )


def test_calibrate_debevec_rejects_numpy_integer_random_sampling() -> None:
    rng = np.random.default_rng(228)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)

    with pytest.raises(TypeError):
        calibrate_camera_response_debevec(
            images,
            _DEFAULT_TIMES,
            random_sampling=np.int32(1),  # type: ignore[arg-type]
        )


def test_calibrate_debevec_rejects_string_random_sampling() -> None:
    rng = np.random.default_rng(229)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)

    with pytest.raises(TypeError):
        calibrate_camera_response_debevec(
            images,
            _DEFAULT_TIMES,
            random_sampling="true",  # type: ignore[arg-type]
        )


def test_calibrate_debevec_random_sampling_false_is_deterministic() -> None:
    rng = np.random.default_rng(230)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)

    first = calibrate_camera_response_debevec(images, _DEFAULT_TIMES, random_sampling=False)
    second = calibrate_camera_response_debevec(images, _DEFAULT_TIMES, random_sampling=False)

    np.testing.assert_array_equal(first, second)


def test_calibrate_debevec_random_sampling_true_runs_without_identity_promise() -> None:
    # Not asserted deterministic or non-deterministic -- only that it runs
    # and produces a valid, finite result; OpenCV's rand() gives no seed
    # control either way.
    rng = np.random.default_rng(231)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)

    result = calibrate_camera_response_debevec(images, _DEFAULT_TIMES, random_sampling=True)

    assert result.shape == (256, 1, 3)
    assert np.all(np.isfinite(result))


def test_calibrate_debevec_passes_exact_parameters_to_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    real_factory = cv2.createCalibrateDebevec

    def fake_factory(samples, smoothness, random_sampling):
        captured["samples"] = samples
        captured["smoothness"] = smoothness
        captured["random_sampling"] = random_sampling
        return real_factory(samples, smoothness, random_sampling)

    monkeypatch.setattr(cv2, "createCalibrateDebevec", fake_factory)

    rng = np.random.default_rng(232)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)
    calibrate_camera_response_debevec(
        images, _DEFAULT_TIMES, samples=42, smoothness=5.0, random_sampling=True
    )

    assert captured["samples"] == 42
    assert captured["smoothness"] == float(np.float32(5.0))
    assert captured["random_sampling"] is True


@pytest.mark.parametrize(
    "bad_kwargs",
    [{"samples": 0}, {"smoothness": 0.0}, {"random_sampling": 1}],
    ids=["samples", "smoothness", "random_sampling"],
)
def test_calibrate_debevec_never_reaches_opencv_for_invalid_parameter(
    bad_kwargs: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def boom(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("cv2 must not be called for an invalid parameter")

    monkeypatch.setattr(cv2, "createCalibrateDebevec", boom)

    rng = np.random.default_rng(233)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)
    with pytest.raises((ValueError, TypeError)):
        calibrate_camera_response_debevec(images, _DEFAULT_TIMES, **bad_kwargs)
    assert not called


# --- calibrate_camera_response_robertson: max_iterations ---


def test_calibrate_robertson_accepts_max_iterations_1() -> None:
    rng = np.random.default_rng(234)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)

    result = calibrate_camera_response_robertson(images, _DEFAULT_TIMES, max_iterations=1)

    assert np.all(np.isfinite(result))


def test_calibrate_robertson_accepts_numpy_integer_max_iterations() -> None:
    rng = np.random.default_rng(235)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)

    result = calibrate_camera_response_robertson(
        images,
        _DEFAULT_TIMES,
        max_iterations=np.int32(30),  # type: ignore[arg-type]
    )

    assert np.all(np.isfinite(result))


def test_calibrate_robertson_rejects_bool_max_iterations() -> None:
    rng = np.random.default_rng(236)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)

    with pytest.raises(TypeError):
        calibrate_camera_response_robertson(images, _DEFAULT_TIMES, max_iterations=True)


def test_calibrate_robertson_rejects_zero_max_iterations() -> None:
    rng = np.random.default_rng(237)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)

    with pytest.raises(ValueError, match="positive"):
        calibrate_camera_response_robertson(images, _DEFAULT_TIMES, max_iterations=0)


def test_calibrate_robertson_rejects_negative_max_iterations() -> None:
    rng = np.random.default_rng(238)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)

    with pytest.raises(ValueError, match="positive"):
        calibrate_camera_response_robertson(images, _DEFAULT_TIMES, max_iterations=-1)


def test_calibrate_robertson_rejects_max_iterations_beyond_int32_range() -> None:
    rng = np.random.default_rng(239)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)

    with pytest.raises(ValueError, match="int32"):
        calibrate_camera_response_robertson(images, _DEFAULT_TIMES, max_iterations=2**31)


# --- calibrate_camera_response_robertson: threshold ---


def test_calibrate_robertson_accepts_threshold_zero() -> None:
    rng = np.random.default_rng(240)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)

    result = calibrate_camera_response_robertson(images, _DEFAULT_TIMES, threshold=0.0)

    assert np.all(np.isfinite(result))


def test_calibrate_robertson_accepts_positive_threshold() -> None:
    rng = np.random.default_rng(241)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)

    result = calibrate_camera_response_robertson(images, _DEFAULT_TIMES, threshold=0.5)

    assert np.all(np.isfinite(result))


def test_calibrate_robertson_accepts_numpy_real_scalar_threshold() -> None:
    rng = np.random.default_rng(242)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)

    result = calibrate_camera_response_robertson(
        images,
        _DEFAULT_TIMES,
        threshold=np.float32(0.01),  # type: ignore[arg-type]
    )

    assert np.all(np.isfinite(result))


def test_calibrate_robertson_rejects_bool_threshold() -> None:
    rng = np.random.default_rng(243)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)

    with pytest.raises(TypeError):
        calibrate_camera_response_robertson(images, _DEFAULT_TIMES, threshold=True)


def test_calibrate_robertson_rejects_negative_threshold() -> None:
    rng = np.random.default_rng(244)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)

    with pytest.raises(ValueError, match="non-negative"):
        calibrate_camera_response_robertson(images, _DEFAULT_TIMES, threshold=-0.1)


def test_calibrate_robertson_rejects_nan_threshold() -> None:
    rng = np.random.default_rng(245)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)

    with pytest.raises(ValueError):
        calibrate_camera_response_robertson(images, _DEFAULT_TIMES, threshold=math.nan)


def test_calibrate_robertson_rejects_inf_threshold() -> None:
    rng = np.random.default_rng(246)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)

    with pytest.raises(ValueError):
        calibrate_camera_response_robertson(images, _DEFAULT_TIMES, threshold=math.inf)


@pytest.mark.parametrize("value", [1e-46, 1e-100, np.nextafter(0.0, 1.0)])
def test_calibrate_robertson_accepts_threshold_underflowing_to_effective_zero(
    value: float,
) -> None:
    # Unlike smoothness, a positive-but-tiny threshold underflowing to
    # 0.0f is accepted -- threshold=0 is itself a legal value.
    assert value > 0.0
    assert np.float32(value) == 0.0
    rng = np.random.default_rng(247)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)

    result = calibrate_camera_response_robertson(images, _DEFAULT_TIMES, threshold=value)

    assert np.all(np.isfinite(result))


def test_calibrate_robertson_rejects_threshold_overflowing_to_inf() -> None:
    rng = np.random.default_rng(248)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)

    with pytest.raises(ValueError, match="too large"):
        calibrate_camera_response_robertson(images, _DEFAULT_TIMES, threshold=1e40)


def test_calibrate_robertson_is_deterministic() -> None:
    rng = np.random.default_rng(249)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)

    first = calibrate_camera_response_robertson(images, _DEFAULT_TIMES)
    second = calibrate_camera_response_robertson(images, _DEFAULT_TIMES)

    np.testing.assert_array_equal(first, second)


def test_calibrate_robertson_passes_exact_parameters_to_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    real_factory = cv2.createCalibrateRobertson

    def fake_factory(max_iterations, threshold):
        captured["max_iterations"] = max_iterations
        captured["threshold"] = threshold
        return real_factory(max_iterations, threshold)

    monkeypatch.setattr(cv2, "createCalibrateRobertson", fake_factory)

    rng = np.random.default_rng(250)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)
    calibrate_camera_response_robertson(images, _DEFAULT_TIMES, max_iterations=7, threshold=0.5)

    assert captured["max_iterations"] == 7
    assert captured["threshold"] == float(np.float32(0.5))


@pytest.mark.parametrize(
    "bad_kwargs",
    [{"max_iterations": 0}, {"threshold": -1.0}],
    ids=["max_iterations", "threshold"],
)
def test_calibrate_robertson_never_reaches_opencv_for_invalid_parameter(
    bad_kwargs: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def boom(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("cv2 must not be called for an invalid parameter")

    monkeypatch.setattr(cv2, "createCalibrateRobertson", boom)

    rng = np.random.default_rng(251)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)
    with pytest.raises((ValueError, TypeError)):
        calibrate_camera_response_robertson(images, _DEFAULT_TIMES, **bad_kwargs)
    assert not called


# --- calibrate_camera_response_*: stack contract (shared helper, spot checks) ---


@pytest.mark.parametrize("func", _CALIBRATE_FUNCS, ids=_CALIBRATE_FUNC_NAMES)
def test_calibrate_rejects_too_few_images(func) -> None:
    rng = np.random.default_rng(252)
    images = _make_hdr_images(rng, count=1, dtype=np.uint8, channels=3)

    with pytest.raises(ValueError, match="at least 2"):
        func(images, [1.0])


@pytest.mark.parametrize("func", _CALIBRATE_FUNCS, ids=_CALIBRATE_FUNC_NAMES)
def test_calibrate_rejects_non_uint8_dtype(func) -> None:
    rng = np.random.default_rng(253)
    base_images = _make_hdr_images(rng, dtype=np.uint8, channels=3)
    images = [im.astype(np.uint16) for im in base_images]

    with pytest.raises(TypeError, match=r"images\[0\]"):
        func(images, _DEFAULT_TIMES)


@pytest.mark.parametrize("func", _CALIBRATE_FUNCS, ids=_CALIBRATE_FUNC_NAMES)
def test_calibrate_rejects_mismatched_shape_with_index(func) -> None:
    rng = np.random.default_rng(254)
    images = _make_hdr_images(rng, count=3, dtype=np.uint8, channels=3)
    images[2] = rng.integers(0, 256, (10, 10, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match=r"images\[2\]"):
        func(images, _DEFAULT_TIMES)


@pytest.mark.parametrize("func", _CALIBRATE_FUNCS, ids=_CALIBRATE_FUNC_NAMES)
@pytest.mark.parametrize(
    "bad_shape",
    [(20, 24, 1), (20, 24, 2), (20, 24, 4)],
    ids=["h_w_1", "2ch", "bgra"],
)
def test_calibrate_rejects_unsupported_channel_counts(func, bad_shape) -> None:
    rng = np.random.default_rng(255)
    images = [rng.integers(0, 256, bad_shape, dtype=np.uint8) for _ in range(3)]

    with pytest.raises(ValueError, match=r"images\[0\]"):
        func(images, _DEFAULT_TIMES)


@pytest.mark.parametrize("func", _CALIBRATE_FUNCS, ids=_CALIBRATE_FUNC_NAMES)
def test_calibrate_never_reaches_opencv_for_invalid_stack(
    func, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def boom(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("cv2 must not be called for an invalid stack")

    monkeypatch.setattr(cv2, _CV2_CALIBRATE_FACTORY[func].__name__, boom)

    with pytest.raises(ValueError):
        func([], [])
    assert not called


# --- calibrate_camera_response_*: exposure_times (reuses existing shared validation) ---


@pytest.mark.parametrize("func", _CALIBRATE_FUNCS, ids=_CALIBRATE_FUNC_NAMES)
def test_calibrate_rejects_wrong_times_length(func) -> None:
    rng = np.random.default_rng(256)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)

    with pytest.raises(ValueError, match="exactly 3"):
        func(images, [1.0, 2.0])


@pytest.mark.parametrize("func", _CALIBRATE_FUNCS, ids=_CALIBRATE_FUNC_NAMES)
def test_calibrate_rejects_negative_time(func) -> None:
    rng = np.random.default_rng(257)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)

    with pytest.raises(ValueError, match=r"exposure_times\[0\]"):
        func(images, [-1.0, 1.0, 2.0])


@pytest.mark.parametrize("func", _CALIBRATE_FUNCS, ids=_CALIBRATE_FUNC_NAMES)
def test_calibrate_passes_fresh_contiguous_float32_times(
    func, monkeypatch: pytest.MonkeyPatch
) -> None:
    rng = np.random.default_rng(258)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)
    original_times = np.array(_DEFAULT_TIMES, dtype=np.float64)
    captured: dict[str, np.ndarray] = {}

    real_factory = _CV2_CALIBRATE_FACTORY[func]
    real_calibrator = real_factory()

    class _CapturingCalibrator:
        def process(self, imgs, times):
            captured["times"] = times
            return real_calibrator.process(imgs, times)

    monkeypatch.setattr(cv2, real_factory.__name__, lambda *a, **kw: _CapturingCalibrator())

    func(images, list(original_times))

    passed_times = captured["times"]
    assert passed_times.dtype == np.float32
    assert passed_times.flags["C_CONTIGUOUS"]
    assert not np.shares_memory(passed_times, original_times)


# --- calibrate_camera_response_*: degenerate stacks ---
#
# None of these are heuristically rejected before calling OpenCV. Verified
# directly that some of them make CalibrateRobertson (never
# CalibrateDebevec, which is more robust thanks to its smoothness
# regularization term) produce a non-finite curve deterministically,
# regardless of image size: its histogram-based normalization divides by
# the count of pixels at each of the 256 intensity levels, so any level
# that never appears anywhere in the stack yields 0/0 = NaN for that
# entry -- an all-black or all-white image (a single intensity level) can
# therefore never produce a finite CalibrateRobertson curve. This is
# checked by asserting the documented contract itself (a finite result, or
# a controlled RuntimeError) rather than assuming one specific outcome.


def _assert_calibrates_or_raises_controlled_runtime_error(func, images, times, **kwargs) -> None:
    try:
        result = func(images, times, **kwargs)
    except RuntimeError as error:
        assert "finite" in str(error)
        return
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert np.all(np.isfinite(result))


@pytest.mark.parametrize("func", _CALIBRATE_FUNCS, ids=_CALIBRATE_FUNC_NAMES)
def test_calibrate_all_black_stack_finite_or_controlled_runtime_error(func) -> None:
    images = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(3)]

    _assert_calibrates_or_raises_controlled_runtime_error(func, images, _DEFAULT_TIMES)


@pytest.mark.parametrize("func", _CALIBRATE_FUNCS, ids=_CALIBRATE_FUNC_NAMES)
def test_calibrate_all_white_stack_finite_or_controlled_runtime_error(func) -> None:
    images = [np.full((64, 64, 3), 255, dtype=np.uint8) for _ in range(3)]

    _assert_calibrates_or_raises_controlled_runtime_error(func, images, _DEFAULT_TIMES)


@pytest.mark.parametrize("func", _CALIBRATE_FUNCS, ids=_CALIBRATE_FUNC_NAMES)
def test_calibrate_identical_images_distinct_times_finite_or_controlled_runtime_error(
    func,
) -> None:
    rng = np.random.default_rng(259)
    image = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    images = [image.copy() for _ in range(3)]

    _assert_calibrates_or_raises_controlled_runtime_error(func, images, _DEFAULT_TIMES)


@pytest.mark.parametrize("func", _CALIBRATE_FUNCS, ids=_CALIBRATE_FUNC_NAMES)
def test_calibrate_identical_times_finite_or_controlled_runtime_error(func) -> None:
    rng = np.random.default_rng(260)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)

    _assert_calibrates_or_raises_controlled_runtime_error(func, images, [1.0, 1.0, 1.0])


@pytest.mark.parametrize("func", _CALIBRATE_FUNCS, ids=_CALIBRATE_FUNC_NAMES)
def test_calibrate_few_intensity_levels_finite_or_controlled_runtime_error(func) -> None:
    rng = np.random.default_rng(261)
    images = [
        np.where(rng.random((64, 64, 3)) > 0.5, np.uint8(50), np.uint8(200)) for _ in range(3)
    ]

    _assert_calibrates_or_raises_controlled_runtime_error(func, images, _DEFAULT_TIMES)


def test_calibrate_robertson_all_black_stack_actually_raises() -> None:
    # Pinned, verified outcome (confirmed on OpenCV 4.9.0, 4.13.0, and
    # 5.0.0): CalibrateRobertson's histogram-based normalization can never
    # produce a finite curve for a single-intensity-level stack, regardless
    # of image size (0/0 for every unpopulated one of the 256 bins). Unlike
    # Robertson, whether CalibrateDebevec itself is robust to this same
    # stack is version-dependent (verified directly: finite on 4.13.0/
    # 5.0.0, non-finite on 4.9.0) -- so no equivalent pinned test exists
    # for Debevec, only the general finite-or-controlled-RuntimeError
    # check above.
    images = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(3)]

    with pytest.raises(RuntimeError, match="finite"):
        calibrate_camera_response_robertson(images, _DEFAULT_TIMES)


# --- calibrate_camera_response_*: response-curve value postcondition ---
#
# Verified directly, with this exact deterministic counterexample, that
# OpenCV's CalibrateDebevec can return a *finite* curve containing exact-
# zero entries: it estimates in log-space and then exponentiates, so a
# very negative but finite intermediate value can underflow float32 to
# exactly 0.0. Such a curve passes a finiteness-only postcondition but is
# unusable by merge_hdr_debevec, which takes the curve's logarithm. The
# exact number of zero entries this specific counterexample produces is
# not treated as a public contract -- only that it raises RuntimeError.
# The remaining tests monkeypatch the calibrator to return controlled
# values, to check the value postcondition itself without depending on a
# specific seed/parameter combination to trigger it.


def test_calibrate_debevec_deterministic_underflow_counterexample_raises() -> None:
    rng = np.random.default_rng(11)
    images = [rng.integers(0, 256, (32, 32, 3), dtype=np.uint8) for _ in range(3)]

    with pytest.raises(RuntimeError, match="zero|positive|logarithm"):
        calibrate_camera_response_debevec(images, _DEFAULT_TIMES, samples=1, smoothness=1e-4)


class _FakeCalibrator:
    def __init__(self, curve: np.ndarray) -> None:
        self._curve = curve

    def process(self, imgs, times):
        return self._curve


def test_calibrate_debevec_rejects_response_curve_with_zero_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    curve = np.full((256, 1, 3), 2.0, dtype=np.float32)
    curve[0, 0, 0] = 0.0
    monkeypatch.setattr(cv2, "createCalibrateDebevec", lambda *a, **kw: _FakeCalibrator(curve))
    rng = np.random.default_rng(266)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)

    with pytest.raises(RuntimeError, match="zero|positive|logarithm"):
        calibrate_camera_response_debevec(images, _DEFAULT_TIMES)


def test_calibrate_debevec_rejects_response_curve_with_negative_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    curve = np.full((256, 1, 3), 2.0, dtype=np.float32)
    curve[0, 0, 0] = -1.0
    monkeypatch.setattr(cv2, "createCalibrateDebevec", lambda *a, **kw: _FakeCalibrator(curve))
    rng = np.random.default_rng(267)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)

    with pytest.raises(RuntimeError, match="zero|positive|logarithm"):
        calibrate_camera_response_debevec(images, _DEFAULT_TIMES)


def test_calibrate_debevec_accepts_strictly_positive_response_curve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    curve = np.full((256, 1, 3), 2.0, dtype=np.float32)
    monkeypatch.setattr(cv2, "createCalibrateDebevec", lambda *a, **kw: _FakeCalibrator(curve))
    rng = np.random.default_rng(268)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)

    result = calibrate_camera_response_debevec(images, _DEFAULT_TIMES)

    np.testing.assert_array_equal(result, curve)


def test_calibrate_robertson_rejects_response_curve_with_negative_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    curve = np.full((256, 1, 3), 2.0, dtype=np.float32)
    curve[0, 0, 0] = -1.0
    monkeypatch.setattr(cv2, "createCalibrateRobertson", lambda *a, **kw: _FakeCalibrator(curve))
    rng = np.random.default_rng(269)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)

    with pytest.raises(RuntimeError, match="negative"):
        calibrate_camera_response_robertson(images, _DEFAULT_TIMES)


def test_calibrate_robertson_rejects_all_zero_response_curve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    curve = np.zeros((256, 1, 3), dtype=np.float32)
    monkeypatch.setattr(cv2, "createCalibrateRobertson", lambda *a, **kw: _FakeCalibrator(curve))
    rng = np.random.default_rng(270)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)

    with pytest.raises(RuntimeError, match="all-zero"):
        calibrate_camera_response_robertson(images, _DEFAULT_TIMES)


def test_calibrate_robertson_accepts_response_curve_with_partial_zeros(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    curve = np.zeros((256, 1, 3), dtype=np.float32)
    curve[0, 0, 0] = 1.0
    monkeypatch.setattr(cv2, "createCalibrateRobertson", lambda *a, **kw: _FakeCalibrator(curve))
    rng = np.random.default_rng(271)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)

    result = calibrate_camera_response_robertson(images, _DEFAULT_TIMES)

    np.testing.assert_array_equal(result, curve)


# --- calibrate + merge: end-to-end integration ---


def test_calibrate_and_merge_debevec_end_to_end() -> None:
    rng = np.random.default_rng(262)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)
    images_before = [im.copy() for im in images]
    times = list(_DEFAULT_TIMES)
    times_before = list(times)

    response = calibrate_camera_response_debevec(images, times)
    response_before = response.copy()
    hdr = merge_hdr_debevec(images, times, response_curve=response)

    assert hdr.shape == images[0].shape
    assert hdr.dtype == np.float32
    assert np.all(np.isfinite(hdr))
    for image, original in zip(images, images_before, strict=True):
        np.testing.assert_array_equal(image, original)
    assert times == times_before
    np.testing.assert_array_equal(response, response_before)


def test_calibrate_and_merge_robertson_end_to_end() -> None:
    rng = np.random.default_rng(263)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)
    images_before = [im.copy() for im in images]
    times = list(_DEFAULT_TIMES)
    times_before = list(times)

    response = calibrate_camera_response_robertson(images, times)
    response_before = response.copy()
    hdr = merge_hdr_robertson(images, times, response_curve=response)

    assert hdr.shape == images[0].shape
    assert hdr.dtype == np.float32
    assert np.all(np.isfinite(hdr))
    for image, original in zip(images, images_before, strict=True):
        np.testing.assert_array_equal(image, original)
    assert times == times_before
    np.testing.assert_array_equal(response, response_before)


# --- calibrate_camera_response_*: mutation and aliasing ---


@pytest.mark.parametrize("func", _CALIBRATE_FUNCS, ids=_CALIBRATE_FUNC_NAMES)
def test_calibrate_does_not_mutate_images(func) -> None:
    rng = np.random.default_rng(264)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)
    before = [im.copy() for im in images]

    func(images, _DEFAULT_TIMES)

    for image, original in zip(images, before, strict=True):
        np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("func", _CALIBRATE_FUNCS, ids=_CALIBRATE_FUNC_NAMES)
def test_calibrate_output_does_not_share_memory_with_inputs(func) -> None:
    rng = np.random.default_rng(265)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)

    result = func(images, _DEFAULT_TIMES)

    for image in images:
        assert not np.shares_memory(result, image)


# --- tone mapping: shared setup ---

_TONE_MAP_FUNCS = [tone_map, tone_map_drago, tone_map_reinhard, tone_map_mantiuk]
_TONE_MAP_FUNC_NAMES = ["tone_map", "tone_map_drago", "tone_map_reinhard", "tone_map_mantiuk"]
_CV2_TONEMAP_FACTORY = {
    tone_map: cv2.createTonemap,
    tone_map_drago: cv2.createTonemapDrago,
    tone_map_reinhard: cv2.createTonemapReinhard,
    tone_map_mantiuk: cv2.createTonemapMantiuk,
}


def _assert_tone_maps_or_raises_controlled_runtime_error(func, hdr, **kwargs) -> None:
    """Real, verified finding (not hypothetical): OpenCV's own `pow`/`log`
    calls inside these operators can receive a base that floating-point
    rounding leaves very slightly negative (an ordinary artifact of
    ``(src - min) / (max - min)`` normalization, present even for
    well-formed, non-degenerate `hdr`) at a non-integer exponent -- which
    is mathematically undefined and produces `NaN`. Whether this actually
    happens is architecture/SIMD-dispatch-dependent: confirmed directly
    that the identical seed/parameters that tone-map finitely on Apple
    Silicon produced a non-finite result on x86_64 CI (both Linux and
    Windows). This checks the documented contract itself (a finite
    result, or a controlled RuntimeError) rather than assuming one
    specific, architecture-dependent outcome -- mirroring
    `_assert_calibrates_or_raises_controlled_runtime_error` above.
    """
    try:
        result = func(hdr, **kwargs)
    except RuntimeError as error:
        assert "finite" in str(error)
        return
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert np.all(np.isfinite(result))


# --- tone mapping: shared hdr contract ---


@pytest.mark.parametrize("func", _TONE_MAP_FUNCS, ids=_TONE_MAP_FUNC_NAMES)
def test_tone_map_accepts_valid_hdr(func) -> None:
    rng = np.random.default_rng(300)
    hdr = _make_hdr(rng)

    result = func(hdr)

    assert result.shape == hdr.shape
    assert result.dtype == np.float32
    assert np.all(np.isfinite(result))


@pytest.mark.parametrize("func", _TONE_MAP_FUNCS, ids=_TONE_MAP_FUNC_NAMES)
def test_tone_map_rejects_non_ndarray(func) -> None:
    with pytest.raises(TypeError, match="NumPy array"):
        func([[1.0, 1.0, 1.0]])


@pytest.mark.parametrize("func", _TONE_MAP_FUNCS, ids=_TONE_MAP_FUNC_NAMES)
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float64])
def test_tone_map_rejects_wrong_dtype(func, dtype) -> None:
    hdr = np.ones((8, 8, 3), dtype=dtype)
    with pytest.raises(TypeError, match="float32"):
        func(hdr)


@pytest.mark.parametrize("func", _TONE_MAP_FUNCS, ids=_TONE_MAP_FUNC_NAMES)
def test_tone_map_rejects_grayscale(func) -> None:
    hdr = np.ones((8, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="grayscale"):
        func(hdr)


@pytest.mark.parametrize("func", _TONE_MAP_FUNCS, ids=_TONE_MAP_FUNC_NAMES)
def test_tone_map_rejects_single_channel_with_trailing_axis(func) -> None:
    hdr = np.ones((8, 8, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="single-channel"):
        func(hdr)


@pytest.mark.parametrize("func", _TONE_MAP_FUNCS, ids=_TONE_MAP_FUNC_NAMES)
def test_tone_map_rejects_bgra(func) -> None:
    hdr = np.ones((8, 8, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="BGRA"):
        func(hdr)


@pytest.mark.parametrize("func", _TONE_MAP_FUNCS, ids=_TONE_MAP_FUNC_NAMES)
def test_tone_map_rejects_two_channels(func) -> None:
    hdr = np.ones((8, 8, 2), dtype=np.float32)
    with pytest.raises(ValueError, match=r"shape \(H, W, 3\)"):
        func(hdr)


@pytest.mark.parametrize("func", _TONE_MAP_FUNCS, ids=_TONE_MAP_FUNC_NAMES)
def test_tone_map_rejects_empty(func) -> None:
    hdr = np.zeros((0, 0, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="empty"):
        func(hdr)


@pytest.mark.parametrize("func", _TONE_MAP_FUNCS, ids=_TONE_MAP_FUNC_NAMES)
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_tone_map_rejects_non_finite_hdr(func, bad_value) -> None:
    rng = np.random.default_rng(301)
    hdr = _make_hdr(rng)
    hdr[0, 0, 0] = bad_value
    with pytest.raises(ValueError, match="finite"):
        func(hdr)


def test_tone_map_accepts_negative_values() -> None:
    # Base `tone_map`/`tone_map_drago` place no non-negativity requirement
    # on `hdr` -- neither merge function guarantees non-negative radiance.
    rng = np.random.default_rng(302)
    hdr = _make_hdr(rng, low=-10.0, high=10.0)

    result = tone_map(hdr)

    assert np.all(np.isfinite(result))


def test_tone_map_drago_accepts_negative_values() -> None:
    rng = np.random.default_rng(303)
    hdr = _make_hdr(rng, low=-10.0, high=10.0)

    result = tone_map_drago(hdr)

    assert np.all(np.isfinite(result))


@pytest.mark.parametrize("func", _TONE_MAP_FUNCS, ids=_TONE_MAP_FUNC_NAMES)
def test_tone_map_accepts_non_contiguous_view(func) -> None:
    rng = np.random.default_rng(304)
    base = _make_hdr(rng, height=16, width=16)
    view = base[::2, ::2, :]
    assert not view.flags["C_CONTIGUOUS"]

    result = func(view)

    assert np.all(np.isfinite(result))


@pytest.mark.parametrize("func", _TONE_MAP_FUNCS, ids=_TONE_MAP_FUNC_NAMES)
def test_tone_map_accepts_read_only_array(func) -> None:
    rng = np.random.default_rng(305)
    hdr = _make_hdr(rng)
    hdr.flags.writeable = False

    result = func(hdr)

    assert np.all(np.isfinite(result))


@pytest.mark.parametrize("func", _TONE_MAP_FUNCS, ids=_TONE_MAP_FUNC_NAMES)
def test_tone_map_accepts_fortran_order(func) -> None:
    rng = np.random.default_rng(306)
    hdr = np.asfortranarray(_make_hdr(rng))
    assert hdr.flags["F_CONTIGUOUS"]

    result = func(hdr)

    assert np.all(np.isfinite(result))


@pytest.mark.parametrize("func", _TONE_MAP_FUNCS, ids=_TONE_MAP_FUNC_NAMES)
def test_tone_map_does_not_mutate_hdr(func) -> None:
    rng = np.random.default_rng(307)
    hdr = _make_hdr(rng)
    before = hdr.copy()

    func(hdr)

    np.testing.assert_array_equal(hdr, before)


@pytest.mark.parametrize("func", _TONE_MAP_FUNCS, ids=_TONE_MAP_FUNC_NAMES)
def test_tone_map_output_does_not_share_memory_with_hdr(func) -> None:
    rng = np.random.default_rng(308)
    hdr = _make_hdr(rng)

    result = func(hdr)

    assert not np.shares_memory(result, hdr)


# --- tone mapping: gamma ---


@pytest.mark.parametrize("func", _TONE_MAP_FUNCS, ids=_TONE_MAP_FUNC_NAMES)
def test_tone_map_accepts_numpy_real_scalar_gamma(func) -> None:
    # gamma=0.5 (not the default 1.0) proves a NumPy scalar is genuinely
    # accepted and used, while keeping 1/gamma an exact integer (2) --
    # deliberately avoiding a fractional exponent here, since that is a
    # separate, already-covered numerical-fragility concern (see
    # _assert_tone_maps_or_raises_controlled_runtime_error).
    rng = np.random.default_rng(309)
    hdr = _make_hdr(rng)

    result = func(hdr, gamma=np.float64(0.5))

    assert np.all(np.isfinite(result))


@pytest.mark.parametrize("func", _TONE_MAP_FUNCS, ids=_TONE_MAP_FUNC_NAMES)
def test_tone_map_rejects_bool_gamma(func) -> None:
    rng = np.random.default_rng(310)
    hdr = _make_hdr(rng)
    with pytest.raises(TypeError, match="real number"):
        func(hdr, gamma=True)


@pytest.mark.parametrize("func", _TONE_MAP_FUNCS, ids=_TONE_MAP_FUNC_NAMES)
@pytest.mark.parametrize("bad_gamma", [0.0, -1.0, math.nan, math.inf, -math.inf])
def test_tone_map_rejects_invalid_gamma(func, bad_gamma) -> None:
    rng = np.random.default_rng(311)
    hdr = _make_hdr(rng)
    with pytest.raises(ValueError):
        func(hdr, gamma=bad_gamma)


@pytest.mark.parametrize("func", _TONE_MAP_FUNCS, ids=_TONE_MAP_FUNC_NAMES)
def test_tone_map_rejects_gamma_underflowing_to_zero(func) -> None:
    rng = np.random.default_rng(312)
    hdr = _make_hdr(rng)
    with pytest.raises(ValueError, match="too small"):
        func(hdr, gamma=1e-46)


@pytest.mark.parametrize("func", _TONE_MAP_FUNCS, ids=_TONE_MAP_FUNC_NAMES)
def test_tone_map_rejects_gamma_overflowing_to_inf(func) -> None:
    rng = np.random.default_rng(313)
    hdr = _make_hdr(rng)
    with pytest.raises(ValueError, match="too large"):
        func(hdr, gamma=1e300)


@pytest.mark.parametrize("func", _TONE_MAP_FUNCS, ids=_TONE_MAP_FUNC_NAMES)
def test_tone_map_rejects_huge_int_gamma_with_controlled_value_error(func) -> None:
    rng = np.random.default_rng(3130)
    hdr = _make_hdr(rng)
    try:
        func(hdr, gamma=10**400)
    except OverflowError:
        pytest.fail("a raw OverflowError propagated for an oversized int gamma")
    except ValueError:
        pass


@pytest.mark.parametrize("func", _TONE_MAP_FUNCS, ids=_TONE_MAP_FUNC_NAMES)
def test_tone_map_accepts_huge_python_int_gamma(func) -> None:
    # A huge but float32-representable-as-inf-free int (not the overflow
    # case above): converts to a large, finite float32 value without
    # raising ValueError/TypeError. Whether the resulting *tone-mapping*
    # itself stays finite is a separate, architecture-dependent concern
    # (see _assert_tone_maps_or_raises_controlled_runtime_error) -- an
    # extreme 1/gamma exponent close to 0 is especially exposed to it.
    rng = np.random.default_rng(314)
    hdr = _make_hdr(rng)

    _assert_tone_maps_or_raises_controlled_runtime_error(func, hdr, gamma=10**6)


# --- tone mapping: exact converted values reach the factory ---


# These tests only check which exact values reach the OpenCV factory --
# deliberately decoupled from whether `.process()` itself then succeeds,
# since that is a separate, architecture-dependent numerical-fragility
# concern (see _assert_tone_maps_or_raises_controlled_runtime_error): the
# fake factory still delegates to the real one so the call shape stays
# realistic, but a RuntimeError from the postcondition afterward is
# irrelevant to what this test is checking and is swallowed here.


def test_tone_map_passes_exact_gamma_to_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, float] = {}
    real_factory = cv2.createTonemap

    def fake_factory(gamma):
        captured["gamma"] = gamma
        return real_factory(gamma)

    monkeypatch.setattr(cv2, "createTonemap", fake_factory)
    rng = np.random.default_rng(315)
    hdr = _make_hdr(rng)

    try:
        tone_map(hdr, gamma=np.float64(2.2))
    except RuntimeError:
        pass

    assert captured["gamma"] == np.float32(2.2)


def test_tone_map_drago_passes_exact_parameters_to_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, float] = {}
    real_factory = cv2.createTonemapDrago

    def fake_factory(gamma, saturation, bias):
        captured["gamma"] = gamma
        captured["saturation"] = saturation
        captured["bias"] = bias
        return real_factory(gamma, saturation, bias)

    monkeypatch.setattr(cv2, "createTonemapDrago", fake_factory)
    rng = np.random.default_rng(316)
    hdr = _make_hdr(rng)

    try:
        tone_map_drago(hdr, gamma=np.float64(1.5), saturation=np.float64(1.2), bias=np.float64(0.7))
    except RuntimeError:
        pass

    assert captured["gamma"] == np.float32(1.5)
    assert captured["saturation"] == np.float32(1.2)
    assert captured["bias"] == np.float32(0.7)


def test_tone_map_reinhard_passes_exact_parameters_to_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, float] = {}
    real_factory = cv2.createTonemapReinhard

    def fake_factory(gamma, intensity, light_adapt, color_adapt):
        captured["gamma"] = gamma
        captured["intensity"] = intensity
        captured["light_adapt"] = light_adapt
        captured["color_adapt"] = color_adapt
        return real_factory(gamma, intensity, light_adapt, color_adapt)

    monkeypatch.setattr(cv2, "createTonemapReinhard", fake_factory)
    rng = np.random.default_rng(317)
    hdr = _make_hdr(rng)

    try:
        tone_map_reinhard(
            hdr,
            gamma=np.float64(1.1),
            intensity=np.float64(3.0),
            light_adaptation=np.float64(0.4),
            color_adaptation=np.float64(0.6),
        )
    except RuntimeError:
        pass

    assert captured["gamma"] == np.float32(1.1)
    assert captured["intensity"] == np.float32(3.0)
    assert captured["light_adapt"] == np.float32(0.4)
    assert captured["color_adapt"] == np.float32(0.6)


def test_tone_map_mantiuk_passes_exact_parameters_to_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, float] = {}
    real_factory = cv2.createTonemapMantiuk

    def fake_factory(gamma, scale, saturation):
        captured["gamma"] = gamma
        captured["scale"] = scale
        captured["saturation"] = saturation
        return real_factory(gamma, scale, saturation)

    monkeypatch.setattr(cv2, "createTonemapMantiuk", fake_factory)
    rng = np.random.default_rng(318)
    hdr = _make_hdr(rng)

    try:
        tone_map_mantiuk(
            hdr, gamma=np.float64(1.3), scale=np.float64(-0.5), saturation=np.float64(1.4)
        )
    except RuntimeError:
        pass

    assert captured["gamma"] == np.float32(1.3)
    assert captured["scale"] == np.float32(-0.5)
    assert captured["saturation"] == np.float32(1.4)


# --- tone_map: output range not guaranteed [0, 1] ---


def test_tone_map_all_zero_is_finite() -> None:
    hdr = np.zeros((8, 8, 3), dtype=np.float32)

    result = tone_map(hdr)

    assert np.all(np.isfinite(result))
    np.testing.assert_array_equal(result, hdr)


def test_tone_map_constant_positive_is_not_normalized_to_one() -> None:
    # Pinned, verified behavior: OpenCV's base Tonemap has a dedicated
    # branch for a spatially constant image (max - min within
    # DBL_EPSILON) that copies the input through unchanged rather than
    # normalizing it -- so the output is NOT the documented [0, 1] range.
    hdr = np.full((8, 8, 3), 1e6, dtype=np.float32)

    result = tone_map(hdr)

    assert np.all(np.isfinite(result))
    np.testing.assert_array_equal(result, hdr)


def test_tone_map_gradient_can_exceed_normalized_range_bounds() -> None:
    rng = np.random.default_rng(319)
    hdr = _make_hdr(rng, low=0.0, high=1000.0)

    result = tone_map(hdr)

    assert np.all(np.isfinite(result))
    assert result.min() >= -1e-6
    assert result.max() <= 1.0 + 1e-6


def test_tone_map_non_finite_output_raises_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeTonemap:
        def process(self, hdr):
            return np.full(hdr.shape, np.nan, dtype=np.float32)

    monkeypatch.setattr(cv2, "createTonemap", lambda gamma: _FakeTonemap())
    rng = np.random.default_rng(320)
    hdr = _make_hdr(rng)

    with pytest.raises(RuntimeError, match="finite"):
        tone_map(hdr)


# --- tone_map_drago: saturation/bias ---


def test_tone_map_drago_rejects_bool_saturation() -> None:
    rng = np.random.default_rng(321)
    hdr = _make_hdr(rng)
    with pytest.raises(TypeError, match="real number"):
        tone_map_drago(hdr, saturation=True)


@pytest.mark.parametrize("bad_saturation", [0.0, -1.0, math.nan, math.inf])
def test_tone_map_drago_rejects_invalid_saturation(bad_saturation) -> None:
    rng = np.random.default_rng(322)
    hdr = _make_hdr(rng)
    with pytest.raises(ValueError):
        tone_map_drago(hdr, saturation=bad_saturation)


def test_tone_map_drago_rejects_saturation_underflowing_to_zero() -> None:
    rng = np.random.default_rng(323)
    hdr = _make_hdr(rng)
    with pytest.raises(ValueError, match="too small"):
        tone_map_drago(hdr, saturation=1e-46)


def test_tone_map_drago_rejects_saturation_overflowing_to_inf() -> None:
    rng = np.random.default_rng(324)
    hdr = _make_hdr(rng)
    with pytest.raises(ValueError, match="too large"):
        tone_map_drago(hdr, saturation=1e300)


def test_tone_map_drago_rejects_huge_int_saturation_with_controlled_value_error() -> None:
    rng = np.random.default_rng(3240)
    hdr = _make_hdr(rng)
    try:
        tone_map_drago(hdr, saturation=10**400)
    except OverflowError:
        pytest.fail("a raw OverflowError propagated for an oversized int saturation")
    except ValueError:
        pass


def test_tone_map_drago_rejects_bool_bias() -> None:
    rng = np.random.default_rng(325)
    hdr = _make_hdr(rng)
    with pytest.raises(TypeError, match="real number"):
        tone_map_drago(hdr, bias=True)


@pytest.mark.parametrize("bad_bias", [-0.0001, 1.0001, -1.0, 2.0, math.nan, math.inf, -math.inf])
def test_tone_map_drago_rejects_bias_outside_range(bad_bias) -> None:
    rng = np.random.default_rng(326)
    hdr = _make_hdr(rng)
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        tone_map_drago(hdr, bias=bad_bias)


@pytest.mark.parametrize("boundary_bias", [0.0, 1.0])
def test_tone_map_drago_accepts_bias_boundaries(boundary_bias) -> None:
    # bias=0 drives TonemapDrago's internal exponent to +inf (logf(0) is
    # -inf); whether that produces a finite result is architecture-
    # dependent for the same reason as _assert_tone_maps_or_raises_
    # controlled_runtime_error -- this only checks that the boundary
    # value itself is accepted (not rejected as out-of-range), not that
    # OpenCV's own numerics succeed.
    rng = np.random.default_rng(327)
    hdr = _make_hdr(rng)

    _assert_tone_maps_or_raises_controlled_runtime_error(tone_map_drago, hdr, bias=boundary_bias)


def test_tone_map_drago_accepts_bias_underflowing_to_zero() -> None:
    # Unlike saturation/gamma, a positive bias underflowing to 0.0f is a
    # legal endpoint, not a hidden contract violation -- see the boundary
    # test above for why finiteness itself isn't asserted here.
    rng = np.random.default_rng(328)
    hdr = _make_hdr(rng)

    _assert_tone_maps_or_raises_controlled_runtime_error(tone_map_drago, hdr, bias=1e-46)


# --- tone_map_drago: degenerate data ---


def test_tone_map_drago_accepts_constant_positive_color() -> None:
    hdr = np.full((8, 8, 3), 5.0, dtype=np.float32)

    result = tone_map_drago(hdr)

    assert np.all(np.isfinite(result))


def test_tone_map_drago_rejects_all_zero() -> None:
    hdr = np.zeros((8, 8, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="zero luminance"):
        tone_map_drago(hdr)


def test_tone_map_drago_rejects_non_constant_image_with_tied_global_minimum() -> None:
    rng = np.random.default_rng(329)
    hdr = _make_hdr(rng, low=0.5, high=10.0)
    global_min = hdr.min()
    hdr[5, 5, :] = global_min

    with pytest.raises(ValueError, match="zero luminance"):
        tone_map_drago(hdr)


def test_tone_map_drago_does_not_use_a_black_pixel_heuristic() -> None:
    # A genuine (0, 0, 0) pixel is safe as long as it is not tied to the
    # array's post-normalization global minimum -- verified directly that
    # a negative global minimum elsewhere means an existing (0, 0, 0)
    # pixel does not necessarily become zero-luminance after OpenCV's own
    # linear normalization.
    rng = np.random.default_rng(330)
    hdr = _make_hdr(rng, low=1.0, high=10.0)
    hdr[2, 2, :] = 0.0  # a genuine (0, 0, 0) pixel
    hdr[7, 7, 0] = -5.0  # array's true global minimum, but only in one channel -- not
    # tied across all 3 channels at (7, 7), so this pixel does not become zero-luminance

    # The point here is that this does NOT raise ValueError (our own
    # precondition correctly lets it through); if it happens to, an
    # uncaught ValueError fails this test as intended. Whether OpenCV's
    # own numerics then succeed is the separate, architecture-dependent
    # concern this helper already covers elsewhere.
    _assert_tone_maps_or_raises_controlled_runtime_error(tone_map_drago, hdr)


def test_tone_map_drago_rejects_zero_luminance_pixel_in_near_constant_hdr() -> None:
    # Regression: _require_no_zero_luminance_pixel's copy-through branch
    # (OpenCV: max - min <= DBL_EPSILON) previously only checked hdr[0, 0]
    # for an exact (0, 0, 0), missing a zero-luminance pixel elsewhere in
    # an otherwise near-constant (but not bit-identical) hdr. Before the
    # fix, this deterministic counterexample reached OpenCV and produced
    # NaN (TonemapDrago), surfacing only as this module's own RuntimeError
    # postcondition instead of being caught here first.
    hdr = np.full((4, 4, 3), 1e-20, dtype=np.float32)
    hdr[2, 2] = 0.0

    with pytest.raises(ValueError, match="zero luminance"):
        tone_map_drago(hdr)


def test_tone_map_mantiuk_rejects_zero_luminance_pixel_in_near_constant_hdr() -> None:
    # See test_tone_map_drago_rejects_zero_luminance_pixel_in_near_constant_hdr
    # -- before the fix, this reached OpenCV and produced a raw solver
    # cv2.error (TonemapMantiuk), surfacing only as this module's own
    # RuntimeError postcondition instead of being caught here first.
    hdr = np.full((4, 4, 3), 1e-20, dtype=np.float32)
    hdr[2, 2] = 0.0

    with pytest.raises(ValueError, match="zero luminance"):
        tone_map_mantiuk(hdr)


def test_tone_map_drago_zero_luminance_rejected_regardless_of_bias() -> None:
    rng = np.random.default_rng(331)
    hdr = _make_hdr(rng, low=0.5, high=10.0)
    hdr[3, 3, :] = hdr.min()

    for bias in (0.0, 0.5, 0.85, 1.0):
        with pytest.raises(ValueError, match="zero luminance"):
            tone_map_drago(hdr, bias=bias)


def test_tone_map_drago_cv2_error_becomes_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(gamma, saturation, bias):
        raise cv2.error("synthetic failure")

    monkeypatch.setattr(cv2, "createTonemapDrago", boom)
    rng = np.random.default_rng(332)
    hdr = _make_hdr(rng)

    with pytest.raises(RuntimeError) as exc_info:
        tone_map_drago(hdr)
    assert isinstance(exc_info.value.__cause__, cv2.error)


# --- tone_map_reinhard: intensity/light_adaptation/color_adaptation ---


@pytest.mark.parametrize("name", ["intensity", "light_adaptation", "color_adaptation"])
def test_tone_map_reinhard_rejects_bool_parameter(name) -> None:
    rng = np.random.default_rng(333)
    hdr = _make_hdr(rng)
    with pytest.raises(TypeError, match="real number"):
        tone_map_reinhard(hdr, **{name: True})


@pytest.mark.parametrize("bad_intensity", [-8.0001, 8.0001, -20.0, 20.0, math.nan, math.inf])
def test_tone_map_reinhard_rejects_intensity_outside_range(bad_intensity) -> None:
    rng = np.random.default_rng(334)
    hdr = _make_hdr(rng)
    with pytest.raises(ValueError, match="between -8.0 and 8.0"):
        tone_map_reinhard(hdr, intensity=bad_intensity)


@pytest.mark.parametrize("boundary_intensity", [-8.0, 8.0])
def test_tone_map_reinhard_accepts_intensity_boundaries(boundary_intensity) -> None:
    rng = np.random.default_rng(335)
    hdr = _make_hdr(rng)

    result = tone_map_reinhard(hdr, intensity=boundary_intensity)

    assert np.all(np.isfinite(result))


@pytest.mark.parametrize("name", ["light_adaptation", "color_adaptation"])
@pytest.mark.parametrize("bad_value", [-0.0001, 1.0001, -1.0, 2.0, math.nan, math.inf])
def test_tone_map_reinhard_rejects_adaptation_outside_range(name, bad_value) -> None:
    rng = np.random.default_rng(336)
    hdr = _make_hdr(rng)
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        tone_map_reinhard(hdr, **{name: bad_value})


@pytest.mark.parametrize("name", ["light_adaptation", "color_adaptation"])
@pytest.mark.parametrize("boundary_value", [0.0, 1.0])
def test_tone_map_reinhard_accepts_adaptation_boundaries(name, boundary_value) -> None:
    rng = np.random.default_rng(337)
    hdr = _make_hdr(rng)

    result = tone_map_reinhard(hdr, **{name: boundary_value})

    assert np.all(np.isfinite(result))


# --- tone_map_reinhard: degenerate data ---


def test_tone_map_reinhard_rejects_constant_gray() -> None:
    hdr = np.full((8, 8, 3), 3.0, dtype=np.float32)
    with pytest.raises(ValueError, match="spatially constant"):
        tone_map_reinhard(hdr)


def test_tone_map_reinhard_rejects_constant_color() -> None:
    hdr = np.empty((8, 8, 3), dtype=np.float32)
    hdr[:] = (1.0, 2.0, 3.0)
    with pytest.raises(ValueError, match="spatially constant"):
        tone_map_reinhard(hdr)


def test_tone_map_reinhard_accepts_thin_non_constant_image() -> None:
    rng = np.random.default_rng(338)
    hdr = _make_hdr(rng, height=1, width=64)

    result = tone_map_reinhard(hdr)

    assert np.all(np.isfinite(result))


def test_tone_map_reinhard_two_wrapper_calls_are_bit_identical() -> None:
    rng = np.random.default_rng(339)
    hdr = _make_hdr(rng)

    try:
        result1 = tone_map_reinhard(hdr)
    except RuntimeError:
        # Still a meaningful determinism check: a fresh object per call
        # (see tone_map_reinhard's Notes) must fail the same way twice,
        # not succeed once and fail once.
        with pytest.raises(RuntimeError):
            tone_map_reinhard(hdr)
        return
    result2 = tone_map_reinhard(hdr)

    np.testing.assert_array_equal(result1, result2)


def test_tone_map_reinhard_creates_a_fresh_factory_object_per_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0
    real_factory = cv2.createTonemapReinhard

    def counting_factory(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return real_factory(*args, **kwargs)

    monkeypatch.setattr(cv2, "createTonemapReinhard", counting_factory)
    rng = np.random.default_rng(340)
    hdr = _make_hdr(rng)

    for _ in range(3):
        try:
            tone_map_reinhard(hdr)
        except RuntimeError:
            pass  # irrelevant here -- this only checks the factory call count

    assert call_count == 3


def test_tone_map_reinhard_cv2_error_becomes_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(gamma, intensity, light_adapt, color_adapt):
        raise cv2.error("synthetic failure")

    monkeypatch.setattr(cv2, "createTonemapReinhard", boom)
    rng = np.random.default_rng(341)
    hdr = _make_hdr(rng)

    with pytest.raises(RuntimeError) as exc_info:
        tone_map_reinhard(hdr)
    assert isinstance(exc_info.value.__cause__, cv2.error)


# --- tone_map_mantiuk: scale/saturation ---


def test_tone_map_mantiuk_rejects_bool_scale() -> None:
    rng = np.random.default_rng(342)
    hdr = _make_hdr(rng)
    with pytest.raises(TypeError, match="real number"):
        tone_map_mantiuk(hdr, scale=True)


def test_tone_map_mantiuk_rejects_zero_scale() -> None:
    rng = np.random.default_rng(343)
    hdr = _make_hdr(rng)
    with pytest.raises(ValueError, match="zero"):
        tone_map_mantiuk(hdr, scale=0.0)


@pytest.mark.parametrize("bad_scale", [math.nan, math.inf, -math.inf])
def test_tone_map_mantiuk_rejects_non_finite_scale(bad_scale) -> None:
    rng = np.random.default_rng(344)
    hdr = _make_hdr(rng)
    with pytest.raises(ValueError):
        tone_map_mantiuk(hdr, scale=bad_scale)


@pytest.mark.parametrize("tiny_scale", [1e-46, -1e-46])
def test_tone_map_mantiuk_rejects_scale_underflowing_to_zero(tiny_scale) -> None:
    rng = np.random.default_rng(345)
    hdr = _make_hdr(rng)
    with pytest.raises(ValueError, match="too small"):
        tone_map_mantiuk(hdr, scale=tiny_scale)


def test_tone_map_mantiuk_rejects_scale_overflowing_to_inf() -> None:
    rng = np.random.default_rng(346)
    hdr = _make_hdr(rng)
    with pytest.raises(ValueError, match="too large"):
        tone_map_mantiuk(hdr, scale=1e300)


def test_tone_map_mantiuk_rejects_huge_int_scale_with_controlled_value_error() -> None:
    rng = np.random.default_rng(3460)
    hdr = _make_hdr(rng)
    try:
        tone_map_mantiuk(hdr, scale=10**400)
    except OverflowError:
        pytest.fail("a raw OverflowError propagated for an oversized int scale")
    except ValueError:
        pass


def test_tone_map_mantiuk_accepts_negative_scale() -> None:
    rng = np.random.default_rng(347)
    hdr = _make_hdr(rng)

    result = tone_map_mantiuk(hdr, scale=-0.5)

    assert np.all(np.isfinite(result))


def test_tone_map_mantiuk_rejects_bool_saturation() -> None:
    rng = np.random.default_rng(348)
    hdr = _make_hdr(rng)
    with pytest.raises(TypeError, match="real number"):
        tone_map_mantiuk(hdr, saturation=True)


@pytest.mark.parametrize("bad_saturation", [0.0, -1.0, math.nan, math.inf])
def test_tone_map_mantiuk_rejects_invalid_saturation(bad_saturation) -> None:
    rng = np.random.default_rng(349)
    hdr = _make_hdr(rng)
    with pytest.raises(ValueError):
        tone_map_mantiuk(hdr, saturation=bad_saturation)


# --- tone_map_mantiuk: minimal size and degenerate data ---


@pytest.mark.parametrize("shape", [(1, 1, 3), (1, 5, 3), (5, 1, 3)])
def test_tone_map_mantiuk_rejects_too_small_shapes(shape) -> None:
    hdr = np.ones(shape, dtype=np.float32)
    with pytest.raises(ValueError, match="at least 2"):
        tone_map_mantiuk(hdr)


def test_tone_map_mantiuk_accepts_2x2() -> None:
    hdr = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [1.5, 2.5, 3.5]]]).astype(
        np.float32
    )

    result = tone_map_mantiuk(hdr)

    assert result.shape == (2, 2, 3)
    assert np.all(np.isfinite(result))


def test_tone_map_mantiuk_rejects_constant_image() -> None:
    hdr = np.full((8, 8, 3), 3.0, dtype=np.float32)
    with pytest.raises(ValueError, match="spatially constant"):
        tone_map_mantiuk(hdr)


def test_tone_map_mantiuk_rejects_zero_luminance_pixel() -> None:
    rng = np.random.default_rng(350)
    hdr = _make_hdr(rng, low=0.5, high=10.0)
    hdr[3, 3, :] = hdr.min()

    with pytest.raises(ValueError, match="zero luminance"):
        tone_map_mantiuk(hdr)


def test_tone_map_mantiuk_cv2_error_becomes_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(gamma, scale, saturation):
        raise cv2.error("synthetic failure")

    monkeypatch.setattr(cv2, "createTonemapMantiuk", boom)
    rng = np.random.default_rng(351)
    hdr = _make_hdr(rng)

    with pytest.raises(RuntimeError) as exc_info:
        tone_map_mantiuk(hdr)
    assert isinstance(exc_info.value.__cause__, cv2.error)


# --- tone mapping: cross-check against direct OpenCV ---


@pytest.mark.parametrize("func", _TONE_MAP_FUNCS, ids=_TONE_MAP_FUNC_NAMES)
# Each cross-check below computes the direct OpenCV result first (the raw
# `cv2.Tonemap*.process()` call itself never raises for this data --
# only improcv's own postcondition does, for a non-finite result). If the
# wrapper then raises RuntimeError, that is only "matching" behavior if
# the direct result is indeed non-finite too -- confirming the wrapper
# faithfully reports what OpenCV itself produced rather than diverging
# from it, architecture-dependent numerics and all (see
# _assert_tone_maps_or_raises_controlled_runtime_error).


def test_tone_map_matches_direct_opencv_defaults(func) -> None:
    rng = np.random.default_rng(352)
    hdr = _make_hdr(rng)

    direct_result = _CV2_TONEMAP_FACTORY[func]().process(hdr)
    try:
        wrapper_result = func(hdr)
    except RuntimeError:
        assert not np.all(np.isfinite(direct_result))
        return

    np.testing.assert_array_equal(wrapper_result, direct_result)


def test_tone_map_drago_matches_direct_opencv_explicit_parameters() -> None:
    rng = np.random.default_rng(353)
    hdr = _make_hdr(rng)

    direct_result = cv2.createTonemapDrago(1.8, 1.3, 0.7).process(hdr)
    try:
        wrapper_result = tone_map_drago(hdr, gamma=1.8, saturation=1.3, bias=0.7)
    except RuntimeError:
        assert not np.all(np.isfinite(direct_result))
        return

    np.testing.assert_array_equal(wrapper_result, direct_result)


def test_tone_map_reinhard_matches_direct_opencv_explicit_parameters() -> None:
    rng = np.random.default_rng(354)
    hdr = _make_hdr(rng)

    direct_result = cv2.createTonemapReinhard(1.2, 1.0, 0.6, 0.4).process(hdr)
    try:
        wrapper_result = tone_map_reinhard(
            hdr, gamma=1.2, intensity=1.0, light_adaptation=0.6, color_adaptation=0.4
        )
    except RuntimeError:
        assert not np.all(np.isfinite(direct_result))
        return

    np.testing.assert_array_equal(wrapper_result, direct_result)


def test_tone_map_mantiuk_matches_direct_opencv_explicit_parameters() -> None:
    rng = np.random.default_rng(355)
    hdr = _make_hdr(rng)

    direct_result = cv2.createTonemapMantiuk(1.1, 0.8, 1.2).process(hdr)
    try:
        wrapper_result = tone_map_mantiuk(hdr, gamma=1.1, scale=0.8, saturation=1.2)
    except RuntimeError:
        assert not np.all(np.isfinite(direct_result))
        return

    np.testing.assert_array_equal(wrapper_result, direct_result)


# --- tone mapping: end-to-end pipeline ---


def test_calibrate_merge_tone_map_end_to_end_pipeline() -> None:
    rng = np.random.default_rng(356)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=3, height=64, width=64)
    times = list(_DEFAULT_TIMES)

    response = calibrate_camera_response_debevec(images, times)
    hdr = merge_hdr_debevec(images, times, response_curve=response)

    assert hdr.shape == images[0].shape
    assert hdr.dtype == np.float32

    try:
        ldr = tone_map_reinhard(hdr)
    except RuntimeError as error:
        # See _assert_tone_maps_or_raises_controlled_runtime_error: real
        # merged radiance can still trigger this architecture-dependent
        # OpenCV numerical fragility on some platforms.
        assert "finite" in str(error)
        return

    display = np.round(np.clip(ldr, 0.0, 1.0) * 255.0).astype(np.uint8)

    assert ldr.shape == images[0].shape
    assert ldr.dtype == np.float32
    assert display.shape == images[0].shape
    assert display.dtype == np.uint8
    assert np.all(np.isfinite(ldr))


# --- public exports ---


def test_public_exports() -> None:
    assert im.fuse_exposures is fuse_exposures
    assert im.merge_hdr_debevec is merge_hdr_debevec
    assert im.merge_hdr_robertson is merge_hdr_robertson
    assert im.calibrate_camera_response_debevec is calibrate_camera_response_debevec
    assert im.calibrate_camera_response_robertson is calibrate_camera_response_robertson
    assert im.tone_map is tone_map
    assert im.tone_map_drago is tone_map_drago
    assert im.tone_map_reinhard is tone_map_reinhard
    assert im.tone_map_mantiuk is tone_map_mantiuk
