import math

import cv2
import numpy as np
import pytest

import improcv as im
from improcv.photo import (
    PencilSketchResult,
    detail_enhance,
    pencil_sketch,
    seamless_clone,
    stylize,
)

_FUNCS = [pencil_sketch, stylize, detail_enhance]
_FUNC_NAMES = ["pencil_sketch", "stylize", "detail_enhance"]

# Each function's own (sigma_s, sigma_r) keyword names -- all three share the
# same contract, so parametrized tests below drive them generically.
_TWO_SIGMA_FUNCS = [stylize, detail_enhance]
_TWO_SIGMA_NAMES = ["stylize", "detail_enhance"]


def _make_bgr(rng: np.random.Generator, height: int = 32, width: int = 32) -> np.ndarray:
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


# --- basic behavior ---


def test_pencil_sketch_returns_expected_shapes_and_dtypes() -> None:
    rng = np.random.default_rng(0)
    image = _make_bgr(rng)

    result = pencil_sketch(image)

    assert isinstance(result, PencilSketchResult)
    assert result.grayscale.shape == (32, 32)
    assert result.grayscale.dtype == np.uint8
    assert result.color.shape == (32, 32, 3)
    assert result.color.dtype == np.uint8


@pytest.mark.parametrize("func", _TWO_SIGMA_FUNCS, ids=_TWO_SIGMA_NAMES)
def test_returns_same_shape_and_dtype_as_input(func) -> None:
    rng = np.random.default_rng(1)
    image = _make_bgr(rng, 20, 24)

    result = func(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8


@pytest.mark.parametrize("func", _FUNCS, ids=_FUNC_NAMES)
def test_does_not_mutate_input(func) -> None:
    rng = np.random.default_rng(2)
    image = _make_bgr(rng)
    before = image.copy()

    func(image)

    np.testing.assert_array_equal(image, before)


def test_pencil_sketch_outputs_do_not_share_memory_with_input_or_each_other() -> None:
    rng = np.random.default_rng(3)
    image = _make_bgr(rng)

    result = pencil_sketch(image)

    assert not np.shares_memory(result.grayscale, image)
    assert not np.shares_memory(result.color, image)
    assert not np.shares_memory(result.grayscale, result.color)


@pytest.mark.parametrize("func", _TWO_SIGMA_FUNCS, ids=_TWO_SIGMA_NAMES)
def test_output_does_not_share_memory_with_input(func) -> None:
    rng = np.random.default_rng(4)
    image = _make_bgr(rng)

    result = func(image)

    assert not np.shares_memory(result, image)


@pytest.mark.parametrize("func", _FUNCS, ids=_FUNC_NAMES)
def test_output_is_sensitive_to_channel_content(func) -> None:
    # A spatially constant color gives no gradient information regardless of
    # which channel holds it (verified: pencil_sketch/stylize give the same
    # degenerate result for any constant color, not just constant gray), so
    # this needs real spatial variation to actually exercise channel order:
    # compare an image against its B/R-swapped counterpart.
    rng = np.random.default_rng(200)
    image = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    swapped = image[:, :, ::-1].copy()  # BGR -> RGB-as-if-BGR (B and R swapped)
    assert not np.array_equal(image, swapped)

    if func is pencil_sketch:
        out_original = func(image).color
        out_swapped = func(swapped).color
    else:
        out_original = func(image)
        out_swapped = func(swapped)

    assert not np.array_equal(out_original, out_swapped)


# --- valid-path integration: wrapper vs direct cv2 call, same process ---


def test_pencil_sketch_matches_direct_cv2_call() -> None:
    rng = np.random.default_rng(5)
    image = _make_bgr(rng, 40, 48)

    result = pencil_sketch(image, sigma_s=45.0, sigma_r=0.2, shade_factor=0.05)
    expected_gray, expected_color = cv2.pencilSketch(
        image, sigma_s=45.0, sigma_r=0.2, shade_factor=0.05
    )

    np.testing.assert_array_equal(result.grayscale, expected_gray)
    np.testing.assert_array_equal(result.color, expected_color)


def test_pencil_sketch_matches_direct_cv2_call_with_defaults() -> None:
    rng = np.random.default_rng(6)
    image = _make_bgr(rng, 40, 48)

    result = pencil_sketch(image)
    expected_gray, expected_color = cv2.pencilSketch(image)

    np.testing.assert_array_equal(result.grayscale, expected_gray)
    np.testing.assert_array_equal(result.color, expected_color)


def test_stylize_matches_direct_cv2_call() -> None:
    rng = np.random.default_rng(7)
    image = _make_bgr(rng, 40, 48)

    result = stylize(image, sigma_s=30.0, sigma_r=0.6)
    expected = cv2.stylization(image, sigma_s=30.0, sigma_r=0.6)

    np.testing.assert_array_equal(result, expected)


def test_stylize_matches_direct_cv2_call_with_defaults() -> None:
    rng = np.random.default_rng(8)
    image = _make_bgr(rng, 40, 48)

    result = stylize(image)
    expected = cv2.stylization(image)

    np.testing.assert_array_equal(result, expected)


def test_detail_enhance_matches_direct_cv2_call() -> None:
    rng = np.random.default_rng(9)
    image = _make_bgr(rng, 40, 48)

    result = detail_enhance(image, sigma_s=5.0, sigma_r=0.3)
    expected = cv2.detailEnhance(image, sigma_s=5.0, sigma_r=0.3)

    np.testing.assert_array_equal(result, expected)


def test_detail_enhance_matches_direct_cv2_call_with_defaults() -> None:
    rng = np.random.default_rng(10)
    image = _make_bgr(rng, 40, 48)

    result = detail_enhance(image)
    expected = cv2.detailEnhance(image)

    np.testing.assert_array_equal(result, expected)


# --- image validation: ndim -> channels -> dtype ---


@pytest.mark.parametrize("func", _FUNCS, ids=_FUNC_NAMES)
def test_rejects_empty_image(func) -> None:
    image = np.zeros((0, 10, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="empty"):
        func(image)


@pytest.mark.parametrize("func", _FUNCS, ids=_FUNC_NAMES)
def test_rejects_2d_grayscale(func) -> None:
    # The message must point at improcv.ensure_bgr as the fix.
    image = np.zeros((10, 10), dtype=np.uint8)

    with pytest.raises(ValueError, match=r"ensure_bgr\(image\)"):
        func(image)


@pytest.mark.parametrize("func", _FUNCS, ids=_FUNC_NAMES)
def test_rejects_h_w_1_grayscale(func) -> None:
    # The message must not suggest calling improcv.ensure_bgr directly on
    # (H, W, 1) -- ensure_bgr itself rejects that shape. It should instead
    # point at dropping the trailing axis first.
    image = np.zeros((10, 10, 1), dtype=np.uint8)

    with pytest.raises(ValueError, match=r"ensure_bgr\(image\[\.\.\., 0\]\)") as exc_info:
        func(image)
    assert "ensure_bgr(image)" not in str(exc_info.value)


@pytest.mark.parametrize("func", _FUNCS, ids=_FUNC_NAMES)
def test_rejects_two_channels(func) -> None:
    # No conversion is suggested for 2-channel input -- there is none.
    image = np.zeros((10, 10, 2), dtype=np.uint8)

    with pytest.raises(ValueError, match="3-channel") as exc_info:
        func(image)
    assert "ensure_bgr" not in str(exc_info.value)


@pytest.mark.parametrize("func", _FUNCS, ids=_FUNC_NAMES)
def test_rejects_bgra_with_alpha_guidance(func) -> None:
    image = np.zeros((10, 10, 4), dtype=np.uint8)

    with pytest.raises(ValueError, match="alpha"):
        func(image)


@pytest.mark.parametrize("func", _FUNCS, ids=_FUNC_NAMES)
def test_rejects_4d_array(func) -> None:
    image = np.zeros((5, 5, 5, 5), dtype=np.uint8)

    with pytest.raises(ValueError, match="3 dimensions"):
        func(image)


@pytest.mark.parametrize("func", _FUNCS, ids=_FUNC_NAMES)
@pytest.mark.parametrize("dtype", [np.uint16, np.int32, np.float32, np.float64])
def test_rejects_non_uint8_dtype(func, dtype) -> None:
    rng = np.random.default_rng(11)
    image = rng.integers(0, 256, (10, 10, 3)).astype(dtype)

    with pytest.raises(TypeError, match="uint8"):
        func(image)


@pytest.mark.parametrize("func", _FUNCS, ids=_FUNC_NAMES)
def test_validation_order_shape_before_dtype(func) -> None:
    # A wrong channel count must be reported even when dtype is ALSO wrong --
    # shape/channel validation happens before dtype validation in this module.
    image = np.zeros((10, 10, 4), dtype=np.float64)

    with pytest.raises(ValueError, match="channel"):
        func(image)


@pytest.mark.parametrize("func", _FUNCS, ids=_FUNC_NAMES)
def test_validation_order_image_before_sigma_s(func) -> None:
    image = np.zeros((10, 10), dtype=np.uint8)

    with pytest.raises(ValueError, match="ensure_bgr"):
        func(image, sigma_s=-5.0)


def test_validation_order_sigma_s_before_sigma_r() -> None:
    rng = np.random.default_rng(12)
    image = _make_bgr(rng)

    with pytest.raises(ValueError, match="sigma_s"):
        stylize(image, sigma_s=-1.0, sigma_r=-1.0)


def test_validation_order_sigma_r_before_shade_factor() -> None:
    rng = np.random.default_rng(13)
    image = _make_bgr(rng)

    with pytest.raises(ValueError, match="sigma_r"):
        pencil_sketch(image, sigma_r=-1.0, shade_factor=-1.0)


# --- BGRA/grayscale/2-channel safety: improcv must never reach the OpenCV
# call for these inputs -- proven via monkeypatch, not by relying on the
# BGRA crash reproducing in CI. A one-off empirical reproduction of the
# crash (see the design discussion) is sufficient justification for this
# restrictive contract; this test protects the contract itself, not the
# upstream bug. ---

_CV2_FUNC_NAME = {
    pencil_sketch: "pencilSketch",
    stylize: "stylization",
    detail_enhance: "detailEnhance",
}


@pytest.mark.parametrize("func", _FUNCS, ids=_FUNC_NAMES)
@pytest.mark.parametrize(
    "bad_shape",
    [(16, 16), (16, 16, 1), (16, 16, 2), (16, 16, 4)],
    ids=["2d", "h_w_1", "2ch", "bgra"],
)
def test_never_reaches_opencv_for_unsupported_channel_layouts(
    func, bad_shape: tuple[int, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def boom(*args: object, **kwargs: object) -> None:
        nonlocal called
        called = True
        raise AssertionError("cv2 must not be called for an unsupported channel layout")

    monkeypatch.setattr(cv2, _CV2_FUNC_NAME[func], boom)

    image = np.zeros(bad_shape, dtype=np.uint8)
    with pytest.raises(ValueError):
        func(image)
    assert not called


# --- sigma_s / sigma_r / shade_factor: type and range contract ---


@pytest.mark.parametrize("func", _FUNCS, ids=_FUNC_NAMES)
def test_accepts_sigma_s_lower_boundary(func) -> None:
    rng = np.random.default_rng(14)
    image = _make_bgr(rng)

    result = func(image, sigma_s=1e-6)

    assert result is not None


@pytest.mark.parametrize("func", _FUNCS, ids=_FUNC_NAMES)
def test_accepts_sigma_s_upper_boundary(func) -> None:
    rng = np.random.default_rng(15)
    image = _make_bgr(rng)

    result = func(image, sigma_s=200.0)

    assert result is not None


@pytest.mark.parametrize("func", _FUNCS, ids=_FUNC_NAMES)
@pytest.mark.parametrize("bad_sigma_s", [0.0, -1.0, 200.0001, 201.0])
def test_rejects_sigma_s_outside_bounds(func, bad_sigma_s: float) -> None:
    rng = np.random.default_rng(16)
    image = _make_bgr(rng)

    with pytest.raises(ValueError, match="sigma_s"):
        func(image, sigma_s=bad_sigma_s)


@pytest.mark.parametrize("func", _FUNCS, ids=_FUNC_NAMES)
def test_accepts_sigma_r_lower_boundary(func) -> None:
    rng = np.random.default_rng(17)
    image = _make_bgr(rng)

    result = func(image, sigma_r=1e-6)

    assert result is not None


@pytest.mark.parametrize("func", _FUNCS, ids=_FUNC_NAMES)
def test_accepts_sigma_r_upper_boundary(func) -> None:
    rng = np.random.default_rng(18)
    image = _make_bgr(rng)

    result = func(image, sigma_r=1.0)

    assert result is not None


@pytest.mark.parametrize("func", _FUNCS, ids=_FUNC_NAMES)
@pytest.mark.parametrize("bad_sigma_r", [0.0, -1.0, 1.0001, 2.0])
def test_rejects_sigma_r_outside_bounds(func, bad_sigma_r: float) -> None:
    rng = np.random.default_rng(19)
    image = _make_bgr(rng)

    with pytest.raises(ValueError, match="sigma_r"):
        func(image, sigma_r=bad_sigma_r)


def test_accepts_shade_factor_zero() -> None:
    # 0 is a valid, documented extreme (a black grayscale sketch), not a
    # degenerate case -- must not be rejected, unlike sigma_s=0/sigma_r=0.
    rng = np.random.default_rng(20)
    image = _make_bgr(rng)

    result = pencil_sketch(image, shade_factor=0.0)

    assert np.all(result.grayscale == 0)


def test_accepts_shade_factor_upper_boundary() -> None:
    rng = np.random.default_rng(21)
    image = _make_bgr(rng)

    result = pencil_sketch(image, shade_factor=0.1)

    assert result is not None


@pytest.mark.parametrize("bad_shade", [-0.0001, -1.0, 0.1001, 1.0])
def test_rejects_shade_factor_outside_bounds(bad_shade: float) -> None:
    rng = np.random.default_rng(22)
    image = _make_bgr(rng)

    with pytest.raises(ValueError, match="shade_factor"):
        pencil_sketch(image, shade_factor=bad_shade)


@pytest.mark.parametrize("func", _FUNCS, ids=_FUNC_NAMES)
def test_accepts_int_sigma_s(func) -> None:
    rng = np.random.default_rng(23)
    image = _make_bgr(rng)

    result = func(image, sigma_s=60)

    assert result is not None


@pytest.mark.parametrize("func", _FUNCS, ids=_FUNC_NAMES)
def test_accepts_numpy_real_scalar_sigma_s(func) -> None:
    rng = np.random.default_rng(24)
    image = _make_bgr(rng)

    result = func(image, sigma_s=np.float32(60.0))

    assert result is not None


@pytest.mark.parametrize("func", _FUNCS, ids=_FUNC_NAMES)
def test_rejects_bool_sigma_s(func) -> None:
    rng = np.random.default_rng(25)
    image = _make_bgr(rng)

    with pytest.raises(TypeError):
        func(image, sigma_s=True)  # type: ignore[arg-type]


@pytest.mark.parametrize("func", _FUNCS, ids=_FUNC_NAMES)
def test_rejects_nan_sigma_s(func) -> None:
    rng = np.random.default_rng(26)
    image = _make_bgr(rng)

    with pytest.raises(ValueError, match="sigma_s"):
        func(image, sigma_s=math.nan)


@pytest.mark.parametrize("func", _FUNCS, ids=_FUNC_NAMES)
def test_rejects_inf_sigma_s(func) -> None:
    rng = np.random.default_rng(27)
    image = _make_bgr(rng)

    with pytest.raises(ValueError, match="sigma_s"):
        func(image, sigma_s=math.inf)


@pytest.mark.parametrize("func", _FUNCS, ids=_FUNC_NAMES)
def test_rejects_nan_sigma_r(func) -> None:
    rng = np.random.default_rng(28)
    image = _make_bgr(rng)

    with pytest.raises(ValueError, match="sigma_r"):
        func(image, sigma_r=math.nan)


@pytest.mark.parametrize("func", _FUNCS, ids=_FUNC_NAMES)
def test_rejects_inf_sigma_r(func) -> None:
    rng = np.random.default_rng(35)
    image = _make_bgr(rng)

    with pytest.raises(ValueError, match="sigma_r"):
        func(image, sigma_r=math.inf)


@pytest.mark.parametrize("func", _FUNCS, ids=_FUNC_NAMES)
def test_rejects_bool_sigma_r(func) -> None:
    rng = np.random.default_rng(36)
    image = _make_bgr(rng)

    with pytest.raises(TypeError):
        func(image, sigma_r=True)  # type: ignore[arg-type]


@pytest.mark.parametrize("func", _FUNCS, ids=_FUNC_NAMES)
def test_accepts_int_sigma_r(func) -> None:
    rng = np.random.default_rng(37)
    image = _make_bgr(rng)

    result = func(image, sigma_r=1)

    assert result is not None


@pytest.mark.parametrize("func", _FUNCS, ids=_FUNC_NAMES)
def test_accepts_numpy_real_scalar_sigma_r(func) -> None:
    rng = np.random.default_rng(38)
    image = _make_bgr(rng)

    result = func(image, sigma_r=np.float32(0.5))

    assert result is not None


@pytest.mark.parametrize("func", _FUNCS, ids=_FUNC_NAMES)
def test_rejects_huge_int_sigma_s_with_controlled_value_error(func) -> None:
    # require_positive's own overflow handling must surface as a controlled
    # ValueError ("must be finite"), never a raw OverflowError from float().
    rng = np.random.default_rng(39)
    image = _make_bgr(rng)

    try:
        func(image, sigma_s=10**400)
    except OverflowError:
        pytest.fail("a raw OverflowError propagated for an oversized int sigma_s")
    except ValueError:
        pass


@pytest.mark.parametrize("func", _FUNCS, ids=_FUNC_NAMES)
def test_rejects_huge_int_sigma_r_with_controlled_value_error(func) -> None:
    rng = np.random.default_rng(40)
    image = _make_bgr(rng)

    try:
        func(image, sigma_r=10**400)
    except OverflowError:
        pytest.fail("a raw OverflowError propagated for an oversized int sigma_r")
    except ValueError:
        pass


def test_rejects_bool_shade_factor() -> None:
    rng = np.random.default_rng(29)
    image = _make_bgr(rng)

    with pytest.raises(TypeError):
        pencil_sketch(image, shade_factor=True)  # type: ignore[arg-type]


def test_rejects_nan_shade_factor() -> None:
    rng = np.random.default_rng(30)
    image = _make_bgr(rng)

    with pytest.raises(ValueError, match="shade_factor"):
        pencil_sketch(image, shade_factor=math.nan)


def test_rejects_inf_shade_factor() -> None:
    rng = np.random.default_rng(41)
    image = _make_bgr(rng)

    with pytest.raises(ValueError, match="shade_factor"):
        pencil_sketch(image, shade_factor=math.inf)


def test_accepts_int_shade_factor() -> None:
    # 0 is the only int that fits shade_factor's [0, 0.1] range.
    rng = np.random.default_rng(42)
    image = _make_bgr(rng)

    result = pencil_sketch(image, shade_factor=0)

    assert result is not None


def test_accepts_numpy_real_scalar_shade_factor() -> None:
    rng = np.random.default_rng(43)
    image = _make_bgr(rng)

    result = pencil_sketch(image, shade_factor=np.float32(0.05))  # type: ignore[arg-type]

    assert result is not None


def test_rejects_negative_shade_factor() -> None:
    rng = np.random.default_rng(31)
    image = _make_bgr(rng)

    with pytest.raises(ValueError, match="shade_factor"):
        pencil_sketch(image, shade_factor=-0.01)


# --- sigma_s / sigma_r: rejected once the value underflows to exactly 0.0
# after conversion to OpenCV's float32 parameter, even though it is a
# positive Python float. Verified directly: np.float32(1e-46) == 0.0
# (below float32's smallest positive subnormal, ~1.4e-45). ---

_FLOAT32_UNDERFLOW_VALUES = [1e-46, 1e-100, np.nextafter(0.0, 1.0)]


@pytest.mark.parametrize("func", _FUNCS, ids=_FUNC_NAMES)
@pytest.mark.parametrize("value", _FLOAT32_UNDERFLOW_VALUES)
def test_rejects_sigma_s_underflowing_to_zero_in_float32(func, value: float) -> None:
    assert value > 0.0  # positive in float64 -- the whole point of this case
    assert np.float32(value) == 0.0  # but exactly zero once OpenCV would see it
    rng = np.random.default_rng(44)
    image = _make_bgr(rng)

    with pytest.raises(ValueError, match="sigma_s"):
        func(image, sigma_s=value)


@pytest.mark.parametrize("func", _FUNCS, ids=_FUNC_NAMES)
@pytest.mark.parametrize("value", _FLOAT32_UNDERFLOW_VALUES)
def test_rejects_sigma_r_underflowing_to_zero_in_float32(func, value: float) -> None:
    assert value > 0.0
    assert np.float32(value) == 0.0
    rng = np.random.default_rng(45)
    image = _make_bgr(rng)

    with pytest.raises(ValueError, match="sigma_r"):
        func(image, sigma_r=value)


@pytest.mark.parametrize("value", _FLOAT32_UNDERFLOW_VALUES)
def test_shade_factor_underflowing_to_zero_in_float32_is_still_accepted(value: float) -> None:
    # Unlike sigma_s/sigma_r, shade_factor=0.0 is itself a valid, documented
    # value -- underflowing to it is not a hidden contract violation.
    rng = np.random.default_rng(46)
    image = _make_bgr(rng)

    result = pencil_sketch(image, shade_factor=value)

    assert np.all(result.grayscale == 0)


@pytest.mark.parametrize("func", _FUNCS, ids=_FUNC_NAMES)
@pytest.mark.parametrize("value", _FLOAT32_UNDERFLOW_VALUES)
def test_never_reaches_opencv_for_sigma_s_underflowing_to_zero(
    func, value: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def boom(*args: object, **kwargs: object) -> None:
        nonlocal called
        called = True
        raise AssertionError("cv2 must not be called for a sigma_s that underflows to 0.0")

    monkeypatch.setattr(cv2, _CV2_FUNC_NAME[func], boom)

    rng = np.random.default_rng(47)
    image = _make_bgr(rng)
    with pytest.raises(ValueError):
        func(image, sigma_s=value)
    assert not called


def test_ordinary_sigma_s_reaches_opencv_as_its_float32_equivalent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    real_stylization = cv2.stylization

    def spy(image: np.ndarray, sigma_s: float = 60.0, sigma_r: float = 0.45):
        captured["sigma_s"] = sigma_s
        captured["sigma_r"] = sigma_r
        return real_stylization(image, sigma_s=sigma_s, sigma_r=sigma_r)

    monkeypatch.setattr(cv2, "stylization", spy)

    rng = np.random.default_rng(48)
    image = _make_bgr(rng)
    stylize(image, sigma_s=33.0, sigma_r=0.5)

    assert captured["sigma_s"] == float(np.float32(33.0))
    assert captured["sigma_r"] == float(np.float32(0.5))
    assert type(captured["sigma_s"]) is float


# --- small/degenerate/thin images ---


@pytest.mark.parametrize("func", _FUNCS, ids=_FUNC_NAMES)
@pytest.mark.parametrize("shape", [(1, 1, 3), (1, 50, 3), (50, 1, 3), (1, 2, 3), (2, 1, 3)])
def test_accepts_tiny_and_thin_images(func, shape: tuple[int, int, int]) -> None:
    rng = np.random.default_rng(32)
    image = rng.integers(0, 256, shape, dtype=np.uint8)

    result = func(image)

    if func is pencil_sketch:
        assert result.grayscale.shape[:2] == shape[:2]
        assert result.color.shape == shape
    else:
        assert result.shape == shape


# --- non-contiguous input ---


@pytest.mark.parametrize("func", _FUNCS, ids=_FUNC_NAMES)
def test_accepts_non_contiguous_view(func) -> None:
    rng = np.random.default_rng(33)
    big = _make_bgr(rng, 80, 80)
    view = big[::2, ::2, :]
    assert not view.flags["C_CONTIGUOUS"]
    before = big.copy()

    result = func(view)

    assert result is not None
    np.testing.assert_array_equal(big, before)


# --- constant images: documented, deterministic properties only ---


@pytest.mark.parametrize("value", [0, 100, 255])
def test_detail_enhance_of_constant_image_equals_input(value: int) -> None:
    image = np.full((16, 16, 3), value, dtype=np.uint8)

    result = detail_enhance(image)

    np.testing.assert_array_equal(result, image)


def test_stylize_of_constant_images_is_independent_of_the_constant() -> None:
    black = np.zeros((16, 16, 3), dtype=np.uint8)
    white = np.full((16, 16, 3), 255, dtype=np.uint8)
    mid = np.full((16, 16, 3), 128, dtype=np.uint8)

    result_black = stylize(black)
    result_white = stylize(white)
    result_mid = stylize(mid)

    np.testing.assert_array_equal(result_black, result_white)
    np.testing.assert_array_equal(result_black, result_mid)


def test_pencil_sketch_of_constant_images_is_independent_of_the_constant() -> None:
    black = np.zeros((16, 16, 3), dtype=np.uint8)
    white = np.full((16, 16, 3), 255, dtype=np.uint8)

    result_black = pencil_sketch(black)
    result_white = pencil_sketch(white)

    np.testing.assert_array_equal(result_black.grayscale, result_white.grayscale)
    np.testing.assert_array_equal(result_black.color, result_white.color)


# --- determinism ---


@pytest.mark.parametrize("func", _FUNCS, ids=_FUNC_NAMES)
def test_is_deterministic(func) -> None:
    rng = np.random.default_rng(34)
    image = _make_bgr(rng)

    first = func(image)
    second = func(image)

    if func is pencil_sketch:
        np.testing.assert_array_equal(first.grayscale, second.grayscale)
        np.testing.assert_array_equal(first.color, second.color)
    else:
        np.testing.assert_array_equal(first, second)


# --- seamless_clone ---


def _make_clone_inputs(
    rng: np.random.Generator,
    source_shape: tuple[int, int] = (50, 50),
    dest_shape: tuple[int, int] = (100, 100),
    box: tuple[int, int, int, int] = (10, 10, 30, 30),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A source/destination/mask trio with a real-textured, off-center active
    region -- a flat-colored region has zero internal gradient and produces no
    visible change under Poisson cloning (verified directly), so tests that
    need a detectable change use real per-pixel noise inside the mask.

    `box` is `(y0, x0, y1, x1)` (exclusive end), sized well above the 3x3
    minimum by default.
    """
    source = np.full((*source_shape, 3), 10, dtype=np.uint8)
    y0, x0, y1, x1 = box
    source[y0:y1, x0:x1] = rng.integers(0, 256, (y1 - y0, x1 - x0, 3), dtype=np.uint8)
    mask = np.zeros(source_shape, dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255
    destination = np.full((*dest_shape, 3), 100, dtype=np.uint8)
    return source, destination, mask


_SEAMLESS_CLONE_MODES = ["normal", "mixed", "monochrome_transfer"]


@pytest.mark.parametrize("mode", _SEAMLESS_CLONE_MODES)
def test_seamless_clone_all_modes_return_expected_shape_and_dtype(mode: str) -> None:
    rng = np.random.default_rng(100)
    source, destination, mask = _make_clone_inputs(rng)

    result = seamless_clone(source, destination, mask, center=(50, 50), mode=mode)  # type: ignore[arg-type]

    assert result.shape == destination.shape
    assert result.dtype == destination.dtype


def test_seamless_clone_rejects_invalid_mode() -> None:
    rng = np.random.default_rng(101)
    source, destination, mask = _make_clone_inputs(rng)

    with pytest.raises(ValueError, match="mode"):
        seamless_clone(source, destination, mask, center=(50, 50), mode="normal_wide")  # type: ignore[arg-type]


@pytest.mark.parametrize("mode", _SEAMLESS_CLONE_MODES)
def test_seamless_clone_matches_direct_cv2_call(mode: str) -> None:
    rng = np.random.default_rng(102)
    source, destination, mask = _make_clone_inputs(rng)
    flag = {
        "normal": cv2.NORMAL_CLONE,
        "mixed": cv2.MIXED_CLONE,
        "monochrome_transfer": cv2.MONOCHROME_TRANSFER,
    }[mode]

    result = seamless_clone(source, destination, mask, center=(50, 50), mode=mode)  # type: ignore[arg-type]
    expected = cv2.seamlessClone(source, destination, mask.copy(), (50, 50), flag)

    np.testing.assert_array_equal(result, expected)


# --- seamless_clone: source/destination validation ---


@pytest.mark.parametrize(
    "bad_shape",
    [(20, 20), (20, 20, 1), (20, 20, 2), (20, 20, 4)],
    ids=["2d", "h_w_1", "2ch", "bgra"],
)
def test_seamless_clone_rejects_bad_source_shape(bad_shape: tuple[int, ...]) -> None:
    rng = np.random.default_rng(103)
    _, destination, _ = _make_clone_inputs(rng)
    source = np.zeros(bad_shape, dtype=np.uint8)
    mask = np.zeros(bad_shape[:2], dtype=np.uint8)
    mask[2:8, 2:8] = 255

    with pytest.raises(ValueError, match="source"):
        seamless_clone(source, destination, mask, center=(50, 50))


@pytest.mark.parametrize(
    "bad_shape",
    [(100, 100), (100, 100, 1), (100, 100, 2), (100, 100, 4)],
    ids=["2d", "h_w_1", "2ch", "bgra"],
)
def test_seamless_clone_rejects_bad_destination_shape(bad_shape: tuple[int, ...]) -> None:
    rng = np.random.default_rng(104)
    source, _, mask = _make_clone_inputs(rng)
    destination = np.zeros(bad_shape, dtype=np.uint8)

    with pytest.raises(ValueError, match="destination"):
        seamless_clone(source, destination, mask, center=(50, 50))


@pytest.mark.parametrize("dtype", [np.uint16, np.int32, np.float32, np.float64])
def test_seamless_clone_rejects_non_uint8_source(dtype) -> None:
    rng = np.random.default_rng(105)
    source, destination, mask = _make_clone_inputs(rng)

    with pytest.raises(TypeError, match="source"):
        seamless_clone(source.astype(dtype), destination, mask, center=(50, 50))


@pytest.mark.parametrize("dtype", [np.uint16, np.int32, np.float32, np.float64])
def test_seamless_clone_rejects_non_uint8_destination(dtype) -> None:
    rng = np.random.default_rng(106)
    source, destination, mask = _make_clone_inputs(rng)

    with pytest.raises(TypeError, match="destination"):
        seamless_clone(source, destination.astype(dtype), mask, center=(50, 50))


def test_seamless_clone_never_reaches_opencv_for_1_channel_destination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A 1-channel destination is uniquely dangerous: verified directly, it
    # can crash the OpenCV process outright (a raw, non-catchable segfault
    # on OpenCV 4.13; nondeterministically a crash or cv2.error on 5.0).
    # This must never be reachable, regardless of what cv2 itself would do.
    called = False

    def boom(*args: object, **kwargs: object) -> None:
        nonlocal called
        called = True
        raise AssertionError("cv2 must not be called for a 1-channel destination")

    monkeypatch.setattr(cv2, "seamlessClone", boom)

    rng = np.random.default_rng(107)
    source, _, mask = _make_clone_inputs(rng)
    destination = np.full((100, 100), 100, dtype=np.uint8)

    with pytest.raises(ValueError, match="destination"):
        seamless_clone(source, destination, mask, center=(50, 50))
    assert not called


# --- seamless_clone: mask validation ---


@pytest.mark.parametrize("dtype", [np.uint16, np.int32, np.float32, bool])
def test_seamless_clone_rejects_non_uint8_mask(dtype) -> None:
    rng = np.random.default_rng(108)
    source, destination, mask = _make_clone_inputs(rng)

    with pytest.raises(TypeError, match="mask"):
        seamless_clone(source, destination, mask.astype(dtype), center=(50, 50))


def test_seamless_clone_rejects_3d_mask() -> None:
    rng = np.random.default_rng(109)
    source, destination, mask = _make_clone_inputs(rng)
    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    with pytest.raises(ValueError, match="2 dimensions"):
        seamless_clone(source, destination, mask_3d, center=(50, 50))


def test_seamless_clone_rejects_mask_spatial_shape_mismatch() -> None:
    rng = np.random.default_rng(110)
    source, destination, _ = _make_clone_inputs(rng)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 255

    with pytest.raises(ValueError, match="mask"):
        seamless_clone(source, destination, mask, center=(50, 50))


@pytest.mark.parametrize("value", [1, 127, 254])
def test_seamless_clone_rejects_intermediate_mask_values(value: int) -> None:
    rng = np.random.default_rng(111)
    source, destination, mask = _make_clone_inputs(rng)
    mask[15, 15] = value

    with pytest.raises(ValueError, match="0 and 255"):
        seamless_clone(source, destination, mask, center=(50, 50))


def test_seamless_clone_accepts_binary_0_255_mask() -> None:
    rng = np.random.default_rng(112)
    source, destination, mask = _make_clone_inputs(rng)
    assert set(np.unique(mask).tolist()) <= {0, 255}

    result = seamless_clone(source, destination, mask, center=(50, 50))

    assert result is not None


# --- seamless_clone: degenerate masks and the all-zero fast path ---


def test_seamless_clone_all_zero_mask_returns_exact_copy_of_destination() -> None:
    rng = np.random.default_rng(113)
    source, destination, mask = _make_clone_inputs(rng)
    mask[:] = 0

    result = seamless_clone(source, destination, mask, center=(50, 50))

    np.testing.assert_array_equal(result, destination)
    assert not np.shares_memory(result, destination)


def test_seamless_clone_border_only_mask_returns_exact_copy_of_destination() -> None:
    # Active only on the outermost 1px border, which OpenCV always zeroes --
    # so the effective mask is empty, same as an all-zero mask.
    rng = np.random.default_rng(114)
    source, destination, mask = _make_clone_inputs(rng)
    mask[:] = 0
    mask[0, :] = 255
    mask[-1, :] = 255
    mask[:, 0] = 255
    mask[:, -1] = 255

    result = seamless_clone(source, destination, mask, center=(50, 50))

    np.testing.assert_array_equal(result, destination)


def test_seamless_clone_all_zero_mask_does_not_reach_opencv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def boom(*args: object, **kwargs: object) -> None:
        nonlocal called
        called = True
        raise AssertionError("cv2 must not be called for an all-zero mask")

    monkeypatch.setattr(cv2, "seamlessClone", boom)

    rng = np.random.default_rng(115)
    source, destination, mask = _make_clone_inputs(rng)
    mask[:] = 0

    result = seamless_clone(source, destination, mask, center=(50, 50))

    assert not called
    np.testing.assert_array_equal(result, destination)


def test_seamless_clone_all_zero_mask_skips_center_bounds_check() -> None:
    # Documented: for an empty effective mask, `center` isn't validated
    # against destination's bounds (there is no ROI to place).
    rng = np.random.default_rng(116)
    source, destination, mask = _make_clone_inputs(rng)
    mask[:] = 0

    result = seamless_clone(source, destination, mask, center=(10_000, -10_000))

    np.testing.assert_array_equal(result, destination)


@pytest.mark.parametrize("size", [1, 2])
def test_seamless_clone_rejects_active_roi_below_minimum(size: int) -> None:
    rng = np.random.default_rng(117)
    source, destination, mask = _make_clone_inputs(rng)
    mask[:] = 0
    mask[20 : 20 + size, 20 : 20 + size] = 255

    with pytest.raises(ValueError, match="3x3"):
        seamless_clone(source, destination, mask, center=(50, 50))


def test_seamless_clone_accepts_active_roi_at_minimum() -> None:
    rng = np.random.default_rng(118)
    source, destination, mask = _make_clone_inputs(rng)
    mask[:] = 0
    mask[20:23, 20:23] = 255

    result = seamless_clone(source, destination, mask, center=(50, 50))

    assert result is not None


def test_seamless_clone_disjoint_regions_use_combined_bounding_box() -> None:
    rng = np.random.default_rng(119)
    source, destination, mask = _make_clone_inputs(rng)
    mask[:] = 0
    mask[5:8, 5:8] = 255
    mask[40:43, 40:43] = 255

    result = seamless_clone(source, destination, mask, center=(50, 50))

    assert result is not None


def test_seamless_clone_mask_touching_edge_still_works() -> None:
    rng = np.random.default_rng(120)
    source, destination, mask = _make_clone_inputs(rng)
    mask[:] = 255  # touches every edge; OpenCV zeroes the border internally

    result = seamless_clone(source, destination, mask, center=(50, 50))

    assert result is not None


def test_seamless_clone_never_reaches_opencv_for_active_roi_below_minimum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def boom(*args: object, **kwargs: object) -> None:
        nonlocal called
        called = True
        raise AssertionError("cv2 must not be called for a too-small active ROI")

    monkeypatch.setattr(cv2, "seamlessClone", boom)

    rng = np.random.default_rng(121)
    source, destination, mask = _make_clone_inputs(rng)
    mask[:] = 0
    mask[20:22, 20:22] = 255

    with pytest.raises(ValueError, match="3x3"):
        seamless_clone(source, destination, mask, center=(50, 50))
    assert not called


# --- seamless_clone: center validation ---


def test_seamless_clone_accepts_python_int_center() -> None:
    rng = np.random.default_rng(122)
    source, destination, mask = _make_clone_inputs(rng)

    result = seamless_clone(source, destination, mask, center=(50, 50))

    assert result is not None


def test_seamless_clone_accepts_numpy_integer_center() -> None:
    rng = np.random.default_rng(123)
    source, destination, mask = _make_clone_inputs(rng)

    result = seamless_clone(
        source,
        destination,
        mask,
        center=(np.int32(50), np.int64(50)),  # type: ignore[arg-type]
    )

    assert result is not None


def test_seamless_clone_rejects_bool_center() -> None:
    rng = np.random.default_rng(124)
    source, destination, mask = _make_clone_inputs(rng)

    with pytest.raises(TypeError):
        seamless_clone(source, destination, mask, center=(True, True))


@pytest.mark.parametrize("center", [(50.0, 50.0), (50.5, 50.5)])
def test_seamless_clone_rejects_float_center(center: tuple[float, float]) -> None:
    rng = np.random.default_rng(125)
    source, destination, mask = _make_clone_inputs(rng)

    with pytest.raises(TypeError):
        seamless_clone(source, destination, mask, center=center)  # type: ignore[arg-type]


@pytest.mark.parametrize("center", [(50,), (50, 50, 50), (50, 50, 50, 50)])
def test_seamless_clone_rejects_wrong_length_center(center: tuple[int, ...]) -> None:
    rng = np.random.default_rng(126)
    source, destination, mask = _make_clone_inputs(rng)

    with pytest.raises(ValueError, match="2-tuple"):
        seamless_clone(source, destination, mask, center=center)  # type: ignore[arg-type]


def test_seamless_clone_rejects_non_tuple_center() -> None:
    rng = np.random.default_rng(127)
    source, destination, mask = _make_clone_inputs(rng)

    with pytest.raises(ValueError, match="2-tuple"):
        seamless_clone(source, destination, mask, center=[50, 50])  # type: ignore[arg-type]


@pytest.mark.parametrize("center", [(2**31, 50), (50, 2**31), (-(2**31) - 1, 50)])
def test_seamless_clone_rejects_center_outside_int32_range(center: tuple[int, int]) -> None:
    rng = np.random.default_rng(128)
    source, destination, mask = _make_clone_inputs(rng)

    with pytest.raises(ValueError):
        seamless_clone(source, destination, mask, center=center)


# --- seamless_clone: geometry ---


def test_seamless_clone_rejects_roi_past_left_edge() -> None:
    rng = np.random.default_rng(129)
    source, destination, mask = _make_clone_inputs(rng)
    # box width=20, roi centered at center_x - 10; center_x=9 -> dst_x=-1
    with pytest.raises(ValueError, match="left edge"):
        seamless_clone(source, destination, mask, center=(9, 50))


def test_seamless_clone_rejects_roi_past_top_edge() -> None:
    rng = np.random.default_rng(130)
    source, destination, mask = _make_clone_inputs(rng)
    with pytest.raises(ValueError, match="top edge"):
        seamless_clone(source, destination, mask, center=(50, 9))


def test_seamless_clone_rejects_roi_past_right_edge() -> None:
    rng = np.random.default_rng(131)
    source, destination, mask = _make_clone_inputs(rng)
    # dest width=100, roi width=20; needs center_x - 10 + 20 > 100, i.e. center_x > 90
    with pytest.raises(ValueError, match="right edge"):
        seamless_clone(source, destination, mask, center=(91, 50))


def test_seamless_clone_rejects_roi_past_bottom_edge() -> None:
    rng = np.random.default_rng(132)
    source, destination, mask = _make_clone_inputs(rng)
    with pytest.raises(ValueError, match="bottom edge"):
        seamless_clone(source, destination, mask, center=(50, 91))


def test_seamless_clone_accepts_roi_exactly_at_each_edge() -> None:
    rng = np.random.default_rng(133)
    source, destination, mask = _make_clone_inputs(rng)
    # roi is 20x20; valid centers place dst_x/dst_y in [0, 80]
    for center in [(10, 50), (90, 50), (50, 10), (50, 90)]:
        result = seamless_clone(source, destination, mask, center=center)
        assert result is not None


def test_seamless_clone_never_reaches_opencv_for_geometry_out_of_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def boom(*args: object, **kwargs: object) -> None:
        nonlocal called
        called = True
        raise AssertionError("cv2 must not be called for out-of-bounds geometry")

    monkeypatch.setattr(cv2, "seamlessClone", boom)

    rng = np.random.default_rng(134)
    source, destination, mask = _make_clone_inputs(rng)

    with pytest.raises(ValueError):
        seamless_clone(source, destination, mask, center=(0, 0))
    assert not called


def test_seamless_clone_source_larger_than_destination() -> None:
    rng = np.random.default_rng(135)
    source, _, mask = _make_clone_inputs(rng, source_shape=(150, 150), box=(60, 60, 80, 80))
    destination = np.full((100, 100, 3), 100, dtype=np.uint8)

    result = seamless_clone(source, destination, mask, center=(50, 50))

    assert result.shape == destination.shape


def test_seamless_clone_source_smaller_than_destination() -> None:
    rng = np.random.default_rng(136)
    source, destination, mask = _make_clone_inputs(
        rng, source_shape=(30, 30), dest_shape=(200, 200), box=(5, 5, 25, 25)
    )

    result = seamless_clone(source, destination, mask, center=(100, 100))

    assert result.shape == destination.shape


# --- seamless_clone: stability, mutation, aliasing ---


def test_seamless_clone_does_not_mutate_inputs() -> None:
    rng = np.random.default_rng(137)
    source, destination, mask = _make_clone_inputs(rng)
    source_before = source.copy()
    destination_before = destination.copy()
    mask_before = mask.copy()

    seamless_clone(source, destination, mask, center=(50, 50))

    np.testing.assert_array_equal(source, source_before)
    np.testing.assert_array_equal(destination, destination_before)
    np.testing.assert_array_equal(mask, mask_before)


def test_seamless_clone_output_does_not_share_memory_with_inputs() -> None:
    rng = np.random.default_rng(138)
    source, destination, mask = _make_clone_inputs(rng)

    result = seamless_clone(source, destination, mask, center=(50, 50))

    assert not np.shares_memory(result, source)
    assert not np.shares_memory(result, destination)
    assert not np.shares_memory(result, mask)


def test_seamless_clone_is_deterministic() -> None:
    rng = np.random.default_rng(139)
    source, destination, mask = _make_clone_inputs(rng)

    first = seamless_clone(source, destination, mask, center=(50, 50))
    second = seamless_clone(source, destination, mask, center=(50, 50))

    np.testing.assert_array_equal(first, second)


def test_seamless_clone_accepts_non_contiguous_source() -> None:
    rng = np.random.default_rng(140)
    big = np.full((100, 100, 3), 10, dtype=np.uint8)
    big[20:60, 20:60] = rng.integers(0, 256, (40, 40, 3), dtype=np.uint8)
    source = big[::2, ::2]
    assert not source.flags["C_CONTIGUOUS"]
    mask = np.zeros(source.shape[:2], dtype=np.uint8)
    mask[10:30, 10:30] = 255
    destination = np.full((100, 100, 3), 100, dtype=np.uint8)

    result = seamless_clone(source, destination, mask, center=(50, 50))

    assert result.shape == destination.shape


def test_seamless_clone_accepts_non_contiguous_destination() -> None:
    rng = np.random.default_rng(141)
    source, _, mask = _make_clone_inputs(rng)
    big_dest = np.full((200, 200, 3), 100, dtype=np.uint8)
    destination = big_dest[::2, ::2]
    assert not destination.flags["C_CONTIGUOUS"]

    result = seamless_clone(source, destination, mask, center=(50, 50))

    assert result.shape == destination.shape


def test_seamless_clone_accepts_non_contiguous_mask() -> None:
    rng = np.random.default_rng(142)
    _, destination, _ = _make_clone_inputs(rng)

    big_source = np.full((200, 200, 3), 10, dtype=np.uint8)
    big_source[40:120, 40:120] = rng.integers(0, 256, (80, 80, 3), dtype=np.uint8)
    source = big_source[::2, ::2]
    big_mask = np.zeros((200, 200), dtype=np.uint8)
    big_mask[40:120, 40:120] = 255
    mask = big_mask[::2, ::2]
    assert not mask.flags["C_CONTIGUOUS"]

    result = seamless_clone(source, destination, mask, center=(50, 50))

    assert result.shape == destination.shape


# --- seamless_clone: validation order ---


def test_seamless_clone_validation_order_mode_before_source() -> None:
    bad_source = np.zeros((10,), dtype=np.uint8)
    destination = np.full((100, 100, 3), 100, dtype=np.uint8)
    mask = np.zeros((10,), dtype=np.uint8)

    with pytest.raises(ValueError, match="mode"):
        seamless_clone(bad_source, destination, mask, center=(50, 50), mode="bogus")  # type: ignore[arg-type]


def test_seamless_clone_validation_order_source_before_destination() -> None:
    # Both are invalid (2D grayscale) -- only source's message must surface.
    bad_source = np.zeros((10, 10), dtype=np.uint8)
    bad_destination = np.zeros((10, 10), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:8, 2:8] = 255

    with pytest.raises(ValueError, match="source"):
        seamless_clone(bad_source, bad_destination, mask, center=(5, 5))


def test_seamless_clone_validation_order_destination_before_mask() -> None:
    rng = np.random.default_rng(143)
    source, _, _ = _make_clone_inputs(rng)
    bad_destination = np.zeros((10, 10), dtype=np.uint8)  # 2D grayscale, invalid
    bad_mask = np.zeros((3, 3), dtype=np.float32)  # wrong dtype AND wrong shape

    with pytest.raises(ValueError, match="destination"):
        seamless_clone(source, bad_destination, bad_mask, center=(50, 50))  # type: ignore[arg-type]


def test_seamless_clone_validation_order_mask_dtype_before_shape() -> None:
    rng = np.random.default_rng(144)
    source, destination, _ = _make_clone_inputs(rng)
    bad_mask = np.zeros((3, 3), dtype=np.float32)  # wrong dtype AND wrong spatial shape

    with pytest.raises(TypeError, match="mask"):
        seamless_clone(source, destination, bad_mask, center=(50, 50))  # type: ignore[arg-type]


def test_seamless_clone_validation_order_mask_shape_before_values() -> None:
    rng = np.random.default_rng(145)
    source, destination, _ = _make_clone_inputs(rng)
    bad_mask = np.full((3, 3), 127, dtype=np.uint8)  # wrong shape AND bad values

    with pytest.raises(ValueError, match="mask"):
        seamless_clone(source, destination, bad_mask, center=(50, 50))


def test_seamless_clone_validation_order_mask_values_before_center() -> None:
    rng = np.random.default_rng(146)
    source, destination, mask = _make_clone_inputs(rng)
    mask[15, 15] = 127  # bad value

    with pytest.raises(ValueError, match="0 and 255"):
        seamless_clone(source, destination, mask, center=(50.0, 50.0))  # type: ignore[arg-type]


def test_seamless_clone_validation_order_center_before_geometry() -> None:
    rng = np.random.default_rng(147)
    source, destination, mask = _make_clone_inputs(rng)

    with pytest.raises(TypeError):
        seamless_clone(source, destination, mask, center=(True, 50))


def test_seamless_clone_validation_order_center_before_effective_mask() -> None:
    rng = np.random.default_rng(148)
    source, destination, mask = _make_clone_inputs(rng)
    mask[:] = 0  # would otherwise take the all-zero fast path

    with pytest.raises(TypeError):
        seamless_clone(source, destination, mask, center=(True, 50))


def test_seamless_clone_validation_order_minimum_roi_before_geometry() -> None:
    rng = np.random.default_rng(149)
    source, destination, mask = _make_clone_inputs(rng)
    mask[:] = 0
    mask[20:22, 20:22] = 255  # too-small ROI AND out-of-bounds center

    with pytest.raises(ValueError, match="3x3"):
        seamless_clone(source, destination, mask, center=(0, 0))


# --- public exports ---


def test_public_exports() -> None:
    assert im.pencil_sketch is pencil_sketch
    assert im.stylize is stylize
    assert im.detail_enhance is detail_enhance
    assert im.seamless_clone is seamless_clone
    assert im.PencilSketchResult is PencilSketchResult
