import math
from collections.abc import Sequence

import cv2
import numpy as np
import pytest

import improcv as im
import improcv.hdr as hdr_module
from improcv.hdr import fuse_exposures, merge_hdr_debevec, merge_hdr_robertson

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


def test_merge_hdr_debevec_accepts_uint8_grayscale() -> None:
    rng = np.random.default_rng(50)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=0)

    result = merge_hdr_debevec(images, _DEFAULT_TIMES)

    assert result.shape == images[0].shape
    assert result.dtype == np.float32
    assert np.all(np.isfinite(result))


@pytest.mark.parametrize("dtype", [np.uint16, np.float32])
def test_merge_hdr_debevec_rejects_non_uint8_grayscale(dtype) -> None:
    # A grayscale uint16/float32 stack was observed to crash the OpenCV
    # process outright (a non-catchable native abort, not a cv2.error) on
    # at least one supported platform/OpenCV build -- this must never be
    # reachable through the public wrapper, regardless of what any single
    # local environment's capability probe would otherwise report.
    rng = np.random.default_rng(50)
    images = _make_hdr_images(rng, dtype=dtype, channels=0)

    with pytest.raises(ValueError, match="grayscale"):
        merge_hdr_debevec(images, _DEFAULT_TIMES)


@pytest.mark.parametrize("dtype", [np.uint16, np.float32])
def test_merge_hdr_debevec_never_reaches_opencv_for_non_uint8_grayscale(
    dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def boom():
        nonlocal called
        called = True
        raise AssertionError("cv2 must not be called for a non-uint8 grayscale Debevec stack")

    monkeypatch.setattr(cv2, "createMergeDebevec", boom)

    rng = np.random.default_rng(50)
    images = _make_hdr_images(rng, dtype=dtype, channels=0)
    with pytest.raises(ValueError, match="grayscale"):
        merge_hdr_debevec(images, _DEFAULT_TIMES)
    assert not called


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_merge_hdr_robertson_rejects_grayscale(dtype) -> None:
    # Verified directly: cv2.MergeRobertson raises a raw, unhelpful error
    # for grayscale input regardless of dtype -- this must never be
    # reachable through the public wrapper.
    rng = np.random.default_rng(50)
    images = _make_hdr_images(rng, dtype=dtype, channels=0)

    with pytest.raises(ValueError, match="grayscale"):
        merge_hdr_robertson(images, _DEFAULT_TIMES)


def test_merge_hdr_robertson_never_reaches_opencv_for_grayscale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def boom():
        nonlocal called
        called = True
        raise AssertionError("cv2 must not be called for a grayscale Robertson stack")

    monkeypatch.setattr(cv2, "createMergeRobertson", boom)

    rng = np.random.default_rng(50)
    images = _make_hdr_images(rng, channels=0)
    with pytest.raises(ValueError, match="grayscale"):
        merge_hdr_robertson(images, _DEFAULT_TIMES)
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


def test_merge_hdr_debevec_accepts_correct_response_curve_shape_grayscale() -> None:
    rng = np.random.default_rng(79)
    images = _make_hdr_images(rng, dtype=np.uint8, channels=0)
    curve = rng.random((256, 1)).astype(np.float32) + 0.1

    result = merge_hdr_debevec(images, _DEFAULT_TIMES, response_curve=curve)

    assert np.all(np.isfinite(result))


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


# --- public exports ---


def test_public_exports() -> None:
    assert im.fuse_exposures is fuse_exposures
    assert im.merge_hdr_debevec is merge_hdr_debevec
    assert im.merge_hdr_robertson is merge_hdr_robertson
