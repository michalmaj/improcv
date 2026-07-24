import math

import cv2
import numpy as np
import pytest

import improcv as im
from improcv.denoising import nl_means_denoise, nl_means_denoise_colored

_ALL_FUNCS = [nl_means_denoise, nl_means_denoise_colored]
_ALL_FUNC_NAMES = ["nl_means_denoise", "nl_means_denoise_colored"]

_CV2_FUNC_NAME = {
    nl_means_denoise: "fastNlMeansDenoising",
    nl_means_denoise_colored: "fastNlMeansDenoisingColored",
}


def _make_gray(rng: np.random.Generator, height: int = 32, width: int = 32) -> np.ndarray:
    return rng.integers(0, 256, (height, width), dtype=np.uint8)


def _make_bgr(rng: np.random.Generator, height: int = 32, width: int = 32) -> np.ndarray:
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


# --- basic behavior ---


def test_nl_means_denoise_returns_same_shape_and_dtype() -> None:
    rng = np.random.default_rng(0)
    image = _make_gray(rng, 20, 24)

    result = nl_means_denoise(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_nl_means_denoise_colored_returns_same_shape_and_dtype() -> None:
    rng = np.random.default_rng(1)
    image = _make_bgr(rng, 20, 24)

    result = nl_means_denoise_colored(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8


@pytest.mark.parametrize("func", _ALL_FUNCS, ids=_ALL_FUNC_NAMES)
def test_does_not_mutate_input(func) -> None:
    rng = np.random.default_rng(2)
    image = _make_gray(rng) if func is nl_means_denoise else _make_bgr(rng)
    before = image.copy()

    func(image)

    np.testing.assert_array_equal(image, before)


@pytest.mark.parametrize("func", _ALL_FUNCS, ids=_ALL_FUNC_NAMES)
def test_output_does_not_share_memory_with_input(func) -> None:
    rng = np.random.default_rng(3)
    image = _make_gray(rng) if func is nl_means_denoise else _make_bgr(rng)

    result = func(image)

    assert not np.shares_memory(result, image)


@pytest.mark.parametrize("func", _ALL_FUNCS, ids=_ALL_FUNC_NAMES)
def test_is_deterministic(func) -> None:
    rng = np.random.default_rng(4)
    image = _make_gray(rng) if func is nl_means_denoise else _make_bgr(rng)

    first = func(image)
    second = func(image)

    np.testing.assert_array_equal(first, second)


# --- valid-path integration: wrapper vs direct cv2 call, same process ---


def test_nl_means_denoise_matches_direct_cv2_call_with_defaults() -> None:
    rng = np.random.default_rng(5)
    image = _make_gray(rng, 40, 48)

    result = nl_means_denoise(image)
    expected = cv2.fastNlMeansDenoising(image)

    np.testing.assert_array_equal(result, expected)


def test_nl_means_denoise_matches_direct_cv2_call_with_explicit_params() -> None:
    rng = np.random.default_rng(6)
    image = _make_gray(rng, 40, 48)

    result = nl_means_denoise(image, h=8.0, template_window_size=5, search_window_size=15)
    expected = cv2.fastNlMeansDenoising(image, h=8.0, templateWindowSize=5, searchWindowSize=15)

    np.testing.assert_array_equal(result, expected)


def test_nl_means_denoise_colored_matches_direct_cv2_call_with_defaults() -> None:
    rng = np.random.default_rng(7)
    image = _make_bgr(rng, 40, 48)

    result = nl_means_denoise_colored(image)
    expected = cv2.fastNlMeansDenoisingColored(image)

    np.testing.assert_array_equal(result, expected)


def test_nl_means_denoise_colored_matches_direct_cv2_call_with_explicit_params() -> None:
    rng = np.random.default_rng(8)
    image = _make_bgr(rng, 40, 48)

    result = nl_means_denoise_colored(
        image, h_luminance=8.0, h_color=12.0, template_window_size=5, search_window_size=15
    )
    expected = cv2.fastNlMeansDenoisingColored(
        image, h=8.0, hColor=12.0, templateWindowSize=5, searchWindowSize=15
    )

    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("shape", [(1, 1), (1, 50), (50, 1), (2, 2), (3, 3)])
def test_nl_means_denoise_matches_direct_cv2_call_for_small_and_thin_images(
    shape: tuple[int, int],
) -> None:
    rng = np.random.default_rng(9)
    image = rng.integers(0, 256, shape, dtype=np.uint8)

    result = nl_means_denoise(image)
    expected = cv2.fastNlMeansDenoising(image)

    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("shape", [(1, 1, 3), (1, 50, 3), (50, 1, 3), (2, 2, 3), (3, 3, 3)])
def test_nl_means_denoise_colored_matches_direct_cv2_call_for_small_and_thin_images(
    shape: tuple[int, int, int],
) -> None:
    rng = np.random.default_rng(10)
    image = rng.integers(0, 256, shape, dtype=np.uint8)

    result = nl_means_denoise_colored(image)
    expected = cv2.fastNlMeansDenoisingColored(image)

    np.testing.assert_array_equal(result, expected)


def test_nl_means_denoise_matches_direct_cv2_call_for_non_contiguous_view() -> None:
    rng = np.random.default_rng(11)
    big = rng.integers(0, 256, (80, 80), dtype=np.uint8)
    view = big[::2, ::2]
    assert not view.flags["C_CONTIGUOUS"]

    result = nl_means_denoise(view)
    expected = cv2.fastNlMeansDenoising(view)

    np.testing.assert_array_equal(result, expected)


def test_nl_means_denoise_colored_matches_direct_cv2_call_for_non_contiguous_view() -> None:
    rng = np.random.default_rng(12)
    big = rng.integers(0, 256, (80, 80, 3), dtype=np.uint8)
    view = big[::2, ::2, :]
    assert not view.flags["C_CONTIGUOUS"]

    result = nl_means_denoise_colored(view)
    expected = cv2.fastNlMeansDenoisingColored(view)

    np.testing.assert_array_equal(result, expected)


# --- image validation ---


def test_nl_means_denoise_rejects_empty_image() -> None:
    image = np.zeros((0, 10), dtype=np.uint8)

    with pytest.raises(ValueError, match="empty"):
        nl_means_denoise(image)


def test_nl_means_denoise_colored_rejects_empty_image() -> None:
    image = np.zeros((0, 10, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="empty"):
        nl_means_denoise_colored(image)


def test_nl_means_denoise_rejects_h_w_1_pointing_at_axis_drop() -> None:
    image = np.zeros((10, 10, 1), dtype=np.uint8)

    with pytest.raises(ValueError, match=r"image\[\.\.\., 0\]"):
        nl_means_denoise(image)


def test_nl_means_denoise_rejects_bgr_pointing_at_ensure_gray() -> None:
    rng = np.random.default_rng(13)
    image = _make_bgr(rng)

    with pytest.raises(ValueError, match="ensure_gray"):
        nl_means_denoise(image)


def test_nl_means_denoise_rejects_1d_array() -> None:
    image = np.zeros(10, dtype=np.uint8)

    with pytest.raises(ValueError):
        nl_means_denoise(image)


def test_nl_means_denoise_colored_rejects_grayscale() -> None:
    rng = np.random.default_rng(14)
    image = _make_gray(rng)

    with pytest.raises(ValueError):
        nl_means_denoise_colored(image)


def test_nl_means_denoise_colored_rejects_h_w_1() -> None:
    image = np.zeros((10, 10, 1), dtype=np.uint8)

    with pytest.raises(ValueError):
        nl_means_denoise_colored(image)


def test_nl_means_denoise_colored_rejects_two_channels() -> None:
    image = np.zeros((10, 10, 2), dtype=np.uint8)

    with pytest.raises(ValueError, match="3 channels"):
        nl_means_denoise_colored(image)


def test_nl_means_denoise_colored_rejects_bgra() -> None:
    # cv2.fastNlMeansDenoisingColored technically accepts BGRA in some
    # builds, but silently replaces the output alpha with a constant 255
    # regardless of input alpha -- improcv rejects it outright instead.
    image = np.zeros((10, 10, 4), dtype=np.uint8)

    with pytest.raises(ValueError, match="3 channels"):
        nl_means_denoise_colored(image)


@pytest.mark.parametrize("func", _ALL_FUNCS, ids=_ALL_FUNC_NAMES)
@pytest.mark.parametrize("dtype", [np.uint16, np.int32, np.float32, np.float64])
def test_rejects_non_uint8_dtype(func, dtype) -> None:
    rng = np.random.default_rng(15)
    if func is nl_means_denoise:
        image = rng.integers(0, 256, (10, 10)).astype(dtype)
    else:
        image = rng.integers(0, 256, (10, 10, 3)).astype(dtype)

    with pytest.raises(TypeError, match="uint8"):
        func(image)


def test_validation_order_shape_before_dtype_colored() -> None:
    # A wrong channel count must be reported even when dtype is ALSO wrong.
    image = np.zeros((10, 10, 4), dtype=np.float64)

    with pytest.raises(ValueError, match="channel"):
        nl_means_denoise_colored(image)  # type: ignore[arg-type]


@pytest.mark.parametrize("func", _ALL_FUNCS, ids=_ALL_FUNC_NAMES)
def test_validation_order_image_before_h(func) -> None:
    image = np.zeros((10,), dtype=np.uint8)

    with pytest.raises(ValueError):
        func(image, **({"h": -5.0} if func is nl_means_denoise else {"h_luminance": -5.0}))


# --- h / h_luminance / h_color ---

_H_KWARG = {nl_means_denoise: "h", nl_means_denoise_colored: "h_luminance"}


@pytest.mark.parametrize("func", _ALL_FUNCS, ids=_ALL_FUNC_NAMES)
def test_accepts_h_exactly_zero(func) -> None:
    rng = np.random.default_rng(16)
    image = _make_gray(rng) if func is nl_means_denoise else _make_bgr(rng)

    result = func(image, **{_H_KWARG[func]: 0.0})

    assert result is not None


def test_h_zero_gives_exact_identity_for_grayscale() -> None:
    rng = np.random.default_rng(17)
    image = _make_gray(rng)

    result = nl_means_denoise(image, h=0.0)

    np.testing.assert_array_equal(result, image)


def test_h_and_h_color_zero_matches_direct_cv2_call_for_colored() -> None:
    # Documented divergence: the BGR -> CIELAB -> BGR round trip alone can
    # shift values, independent of h_luminance/h_color -- but the exact size
    # of that shift is seed/image-dependent (verified directly: max diff 3
    # for most sampled seeds, but 4 for at least one other), not a fixed
    # universal bound. So this pins the wrapper's output to a direct cv2
    # call instead of to any specific difference threshold.
    rng = np.random.default_rng(18)
    image = _make_bgr(rng)
    before = image.copy()

    result = nl_means_denoise_colored(image, h_luminance=0.0, h_color=0.0)
    expected = cv2.fastNlMeansDenoisingColored(
        image, h=0.0, hColor=0.0, templateWindowSize=7, searchWindowSize=21
    )

    assert not np.array_equal(result, image)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(image, before)
    assert result.shape == image.shape
    assert result.dtype == image.dtype


@pytest.mark.parametrize("func", _ALL_FUNCS, ids=_ALL_FUNC_NAMES)
def test_rejects_negative_h(func) -> None:
    rng = np.random.default_rng(19)
    image = _make_gray(rng) if func is nl_means_denoise else _make_bgr(rng)

    with pytest.raises(ValueError, match="non-negative"):
        func(image, **{_H_KWARG[func]: -1.0})


@pytest.mark.parametrize("func", _ALL_FUNCS, ids=_ALL_FUNC_NAMES)
def test_rejects_bool_h(func) -> None:
    rng = np.random.default_rng(20)
    image = _make_gray(rng) if func is nl_means_denoise else _make_bgr(rng)

    with pytest.raises(TypeError):
        func(image, **{_H_KWARG[func]: True})


@pytest.mark.parametrize("func", _ALL_FUNCS, ids=_ALL_FUNC_NAMES)
def test_rejects_nan_h(func) -> None:
    rng = np.random.default_rng(21)
    image = _make_gray(rng) if func is nl_means_denoise else _make_bgr(rng)

    with pytest.raises(ValueError):
        func(image, **{_H_KWARG[func]: math.nan})


@pytest.mark.parametrize("func", _ALL_FUNCS, ids=_ALL_FUNC_NAMES)
def test_rejects_inf_h(func) -> None:
    rng = np.random.default_rng(22)
    image = _make_gray(rng) if func is nl_means_denoise else _make_bgr(rng)

    with pytest.raises(ValueError):
        func(image, **{_H_KWARG[func]: math.inf})


@pytest.mark.parametrize("func", _ALL_FUNCS, ids=_ALL_FUNC_NAMES)
def test_accepts_int_h(func) -> None:
    rng = np.random.default_rng(23)
    image = _make_gray(rng) if func is nl_means_denoise else _make_bgr(rng)

    result = func(image, **{_H_KWARG[func]: 5})

    assert result is not None


@pytest.mark.parametrize("func", _ALL_FUNCS, ids=_ALL_FUNC_NAMES)
def test_accepts_numpy_real_scalar_h(func) -> None:
    rng = np.random.default_rng(24)
    image = _make_gray(rng) if func is nl_means_denoise else _make_bgr(rng)

    result = func(image, **{_H_KWARG[func]: np.float32(5.0)})

    assert result is not None


@pytest.mark.parametrize("func", _ALL_FUNCS, ids=_ALL_FUNC_NAMES)
@pytest.mark.parametrize("value", [1e-46, 1e-100, np.nextafter(0.0, 1.0)])
def test_rejects_h_underflowing_to_zero_in_float32(func, value: float) -> None:
    assert value > 0.0
    assert np.float32(value) == 0.0
    rng = np.random.default_rng(25)
    image = _make_gray(rng) if func is nl_means_denoise else _make_bgr(rng)

    with pytest.raises(ValueError, match="too small"):
        func(image, **{_H_KWARG[func]: value})


@pytest.mark.parametrize("func", _ALL_FUNCS, ids=_ALL_FUNC_NAMES)
def test_rejects_h_overflowing_to_inf_in_float32(func) -> None:
    rng = np.random.default_rng(26)
    image = _make_gray(rng) if func is nl_means_denoise else _make_bgr(rng)

    with pytest.raises(ValueError, match="too large"):
        func(image, **{_H_KWARG[func]: 1e40})


@pytest.mark.parametrize("func", _ALL_FUNCS, ids=_ALL_FUNC_NAMES)
@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_h_overflow_never_raises_uncontrolled_runtime_warning(func) -> None:
    # With the filter above, an escaped RuntimeWarning would itself become
    # the raised exception -- so a ValueError (not a RuntimeWarning) is only
    # possible if the float32 overflow was properly contained internally.
    rng = np.random.default_rng(27)
    image = _make_gray(rng) if func is nl_means_denoise else _make_bgr(rng)

    with pytest.raises(ValueError, match="too large"):
        func(image, **{_H_KWARG[func]: 1e40})


@pytest.mark.parametrize("func", _ALL_FUNCS, ids=_ALL_FUNC_NAMES)
def test_rejects_huge_int_h_with_controlled_value_error(func) -> None:
    rng = np.random.default_rng(28)
    image = _make_gray(rng) if func is nl_means_denoise else _make_bgr(rng)

    try:
        func(image, **{_H_KWARG[func]: 10**400})
    except OverflowError:
        pytest.fail("a raw OverflowError propagated for an oversized int h")
    except ValueError:
        pass


def test_rejects_negative_h_color() -> None:
    rng = np.random.default_rng(29)
    image = _make_bgr(rng)

    with pytest.raises(ValueError, match="h_color"):
        nl_means_denoise_colored(image, h_color=-1.0)


def test_accepts_h_color_zero() -> None:
    rng = np.random.default_rng(30)
    image = _make_bgr(rng)

    result = nl_means_denoise_colored(image, h_color=0.0)

    assert result is not None


def test_rejects_bool_h_color() -> None:
    rng = np.random.default_rng(43)
    image = _make_bgr(rng)

    with pytest.raises(TypeError):
        nl_means_denoise_colored(image, h_color=True)


def test_rejects_nan_h_color() -> None:
    rng = np.random.default_rng(44)
    image = _make_bgr(rng)

    with pytest.raises(ValueError):
        nl_means_denoise_colored(image, h_color=math.nan)


def test_rejects_inf_h_color() -> None:
    rng = np.random.default_rng(45)
    image = _make_bgr(rng)

    with pytest.raises(ValueError):
        nl_means_denoise_colored(image, h_color=math.inf)


@pytest.mark.parametrize("value", [1e-46, 1e-100, np.nextafter(0.0, 1.0)])
def test_rejects_h_color_underflowing_to_zero_in_float32(value: float) -> None:
    assert value > 0.0
    assert np.float32(value) == 0.0
    rng = np.random.default_rng(46)
    image = _make_bgr(rng)

    with pytest.raises(ValueError, match="too small"):
        nl_means_denoise_colored(image, h_color=value)


def test_rejects_h_color_overflowing_to_inf_in_float32() -> None:
    rng = np.random.default_rng(47)
    image = _make_bgr(rng)

    with pytest.raises(ValueError, match="too large"):
        nl_means_denoise_colored(image, h_color=1e40)


def test_rejects_huge_int_h_color_with_controlled_value_error() -> None:
    rng = np.random.default_rng(48)
    image = _make_bgr(rng)

    try:
        nl_means_denoise_colored(image, h_color=10**400)
    except OverflowError:
        pytest.fail("a raw OverflowError propagated for an oversized int h_color")
    except ValueError:
        pass


def test_accepts_numpy_real_scalar_h_color() -> None:
    rng = np.random.default_rng(49)
    image = _make_bgr(rng)

    result = nl_means_denoise_colored(image, h_color=np.float32(5.0))  # type: ignore[arg-type]

    assert result is not None


# --- window sizes ---


@pytest.mark.parametrize("func", _ALL_FUNCS, ids=_ALL_FUNC_NAMES)
def test_accepts_window_size_1(func) -> None:
    rng = np.random.default_rng(31)
    image = _make_gray(rng) if func is nl_means_denoise else _make_bgr(rng)

    result = func(image, template_window_size=1, search_window_size=1)

    assert result is not None


@pytest.mark.parametrize("func", _ALL_FUNCS, ids=_ALL_FUNC_NAMES)
def test_rejects_even_template_window_size(func) -> None:
    rng = np.random.default_rng(32)
    image = _make_gray(rng) if func is nl_means_denoise else _make_bgr(rng)

    with pytest.raises(ValueError, match="odd"):
        func(image, template_window_size=2)


@pytest.mark.parametrize("func", _ALL_FUNCS, ids=_ALL_FUNC_NAMES)
def test_rejects_even_search_window_size(func) -> None:
    rng = np.random.default_rng(33)
    image = _make_gray(rng) if func is nl_means_denoise else _make_bgr(rng)

    with pytest.raises(ValueError, match="odd"):
        func(image, search_window_size=20)


@pytest.mark.parametrize("func", _ALL_FUNCS, ids=_ALL_FUNC_NAMES)
def test_rejects_zero_template_window_size(func) -> None:
    rng = np.random.default_rng(34)
    image = _make_gray(rng) if func is nl_means_denoise else _make_bgr(rng)

    with pytest.raises(ValueError):
        func(image, template_window_size=0)


@pytest.mark.parametrize("func", _ALL_FUNCS, ids=_ALL_FUNC_NAMES)
def test_rejects_negative_search_window_size(func) -> None:
    rng = np.random.default_rng(35)
    image = _make_gray(rng) if func is nl_means_denoise else _make_bgr(rng)

    with pytest.raises(ValueError):
        func(image, search_window_size=-21)


def test_search_smaller_than_template_matches_direct_cv2_call_grayscale() -> None:
    # search_window_size < template_window_size is not rejected: these are
    # two independent parameters (patch size for comparison vs. area
    # searched for similar patch centers), and OpenCV produces real,
    # different output for this combination -- verified directly it is not
    # a no-op (1019/1024 pixels differ from the input, max diff 156, and
    # differs from the search_window_size=7 result too).
    rng = np.random.default_rng(1)
    image = rng.integers(0, 256, (32, 32), dtype=np.uint8)

    result = nl_means_denoise(image, h=100, template_window_size=7, search_window_size=5)
    expected = cv2.fastNlMeansDenoising(image, h=100, templateWindowSize=7, searchWindowSize=5)

    np.testing.assert_array_equal(result, expected)


def test_search_smaller_than_template_matches_direct_cv2_call_colored() -> None:
    # Same relaxation as the grayscale case above, for the colored function.
    rng = np.random.default_rng(1)
    image = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)

    result = nl_means_denoise_colored(
        image, h_luminance=100, h_color=100, template_window_size=7, search_window_size=5
    )
    expected = cv2.fastNlMeansDenoisingColored(
        image, h=100, hColor=100, templateWindowSize=7, searchWindowSize=5
    )

    np.testing.assert_array_equal(result, expected)


def test_opencv_itself_canonicalizes_even_template_window_size_to_next_odd() -> None:
    # Reference test documenting raw OpenCV behavior, not the wrapper: an
    # even templateWindowSize is not a no-op, it is silently canonicalized
    # to the next odd value. The wrapper still rejects even sizes outright
    # -- see test_rejects_even_template_window_size -- because the value a
    # caller passed would not be the value actually used.
    rng = np.random.default_rng(43)
    image = rng.integers(0, 256, (32, 32), dtype=np.uint8)

    even = cv2.fastNlMeansDenoising(image, templateWindowSize=2)
    odd = cv2.fastNlMeansDenoising(image, templateWindowSize=3)

    np.testing.assert_array_equal(even, odd)


def test_opencv_itself_canonicalizes_even_search_window_size_to_next_odd() -> None:
    rng = np.random.default_rng(44)
    image = rng.integers(0, 256, (32, 32), dtype=np.uint8)

    even = cv2.fastNlMeansDenoising(image, searchWindowSize=20)
    odd = cv2.fastNlMeansDenoising(image, searchWindowSize=21)

    np.testing.assert_array_equal(even, odd)


@pytest.mark.parametrize("func", _ALL_FUNCS, ids=_ALL_FUNC_NAMES)
def test_rejects_bool_window_size(func) -> None:
    rng = np.random.default_rng(37)
    image = _make_gray(rng) if func is nl_means_denoise else _make_bgr(rng)

    with pytest.raises(TypeError):
        func(image, template_window_size=True)


@pytest.mark.parametrize("func", _ALL_FUNCS, ids=_ALL_FUNC_NAMES)
def test_rejects_non_integral_window_size(func) -> None:
    rng = np.random.default_rng(38)
    image = _make_gray(rng) if func is nl_means_denoise else _make_bgr(rng)

    with pytest.raises(TypeError):
        func(image, template_window_size=7.5)


@pytest.mark.parametrize("func", _ALL_FUNCS, ids=_ALL_FUNC_NAMES)
def test_accepts_numpy_integer_scalar_window_size(func) -> None:
    rng = np.random.default_rng(39)
    image = _make_gray(rng) if func is nl_means_denoise else _make_bgr(rng)

    result = func(image, template_window_size=np.int32(7), search_window_size=np.int32(21))

    assert result is not None


@pytest.mark.parametrize("func", _ALL_FUNCS, ids=_ALL_FUNC_NAMES)
def test_rejects_window_size_beyond_int32_range(func) -> None:
    rng = np.random.default_rng(40)
    image = _make_gray(rng) if func is nl_means_denoise else _make_bgr(rng)

    with pytest.raises(ValueError, match="int32"):
        func(image, template_window_size=2**31 + 1)  # odd, so this exercises the range check


@pytest.mark.parametrize("func", _ALL_FUNCS, ids=_ALL_FUNC_NAMES)
def test_rejects_huge_int_window_size_with_controlled_value_error(func) -> None:
    rng = np.random.default_rng(41)
    image = _make_gray(rng) if func is nl_means_denoise else _make_bgr(rng)

    try:
        func(image, template_window_size=2**200)
    except OverflowError:
        pytest.fail("a raw OverflowError propagated for an oversized int window size")
    except (ValueError, TypeError):
        pass


# --- BGRA/grayscale/2-channel/wrong-ndim safety: improcv must never reach
# the OpenCV call for a validation failure. ---


@pytest.mark.parametrize("bad_shape", [(16,), (16, 16, 1), (16, 16, 2), (16, 16, 3), (16, 16, 4)])
def test_nl_means_denoise_never_reaches_opencv_for_invalid_image(
    bad_shape: tuple[int, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def boom(*args: object, **kwargs: object) -> None:
        nonlocal called
        called = True
        raise AssertionError("cv2 must not be called for an invalid image")

    monkeypatch.setattr(cv2, "fastNlMeansDenoising", boom)

    image = np.zeros(bad_shape, dtype=np.uint8)
    with pytest.raises(ValueError):
        nl_means_denoise(image)
    assert not called


@pytest.mark.parametrize("bad_shape", [(16, 16), (16, 16, 1), (16, 16, 2), (16, 16, 4)])
def test_nl_means_denoise_colored_never_reaches_opencv_for_invalid_image(
    bad_shape: tuple[int, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def boom(*args: object, **kwargs: object) -> None:
        nonlocal called
        called = True
        raise AssertionError("cv2 must not be called for an invalid image")

    monkeypatch.setattr(cv2, "fastNlMeansDenoisingColored", boom)

    image = np.zeros(bad_shape, dtype=np.uint8)
    with pytest.raises(ValueError):
        nl_means_denoise_colored(image)
    assert not called


@pytest.mark.parametrize("func", _ALL_FUNCS, ids=_ALL_FUNC_NAMES)
def test_never_reaches_opencv_for_invalid_window_size(
    func, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def boom(*args: object, **kwargs: object) -> None:
        nonlocal called
        called = True
        raise AssertionError("cv2 must not be called for an invalid window size")

    monkeypatch.setattr(cv2, _CV2_FUNC_NAME[func], boom)

    rng = np.random.default_rng(42)
    image = _make_gray(rng) if func is nl_means_denoise else _make_bgr(rng)
    with pytest.raises(ValueError):
        func(image, template_window_size=2)
    assert not called


# --- quality: denoising must meaningfully reduce error for a known,
# pinned noise scenario -- not a claim that it improves for every image
# or parameter. ---


def test_nl_means_denoise_improves_mse_for_a_known_noisy_image() -> None:
    rng = np.random.default_rng(42)
    clean = np.zeros((64, 64), dtype=np.uint8)
    clean[16:48, 16:48] = 200
    clean[:16, :] = 50
    noise = rng.normal(0, 15, clean.shape)
    noisy = np.clip(clean.astype(np.float64) + noise, 0, 255).astype(np.uint8)

    denoised = nl_means_denoise(noisy, h=10.0, template_window_size=7, search_window_size=21)

    mse_before = im.mse(clean, noisy)
    mse_after = im.mse(clean, denoised)
    # Wide, platform-robust margin (empirically ~42 vs ~164, well under half)
    # rather than a brittle exact value or full-image snapshot.
    assert mse_after < 0.5 * mse_before


# --- public exports ---


def test_public_exports() -> None:
    assert im.nl_means_denoise is nl_means_denoise
    assert im.nl_means_denoise_colored is nl_means_denoise_colored
