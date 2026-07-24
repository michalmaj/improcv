import math

import cv2
import numpy as np
import pytest

import improcv as im
from improcv.photo import PencilSketchResult, detail_enhance, pencil_sketch, stylize

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
    image = np.zeros((10, 10), dtype=np.uint8)

    with pytest.raises(ValueError, match="3 dimensions"):
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

    with pytest.raises(ValueError, match="3 dimensions"):
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


# --- public exports ---


def test_public_exports() -> None:
    assert im.pencil_sketch is pencil_sketch
    assert im.stylize is stylize
    assert im.detail_enhance is detail_enhance
    assert im.PencilSketchResult is PencilSketchResult
