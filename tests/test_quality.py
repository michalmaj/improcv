import math

import numpy as np
import pytest

import improcv as im

# --- mse ---


def test_mse_of_identical_images_is_zero() -> None:
    image = np.full((10, 10), 100, dtype=np.uint8)

    assert im.mse(image, image) == 0.0


def test_mse_of_identical_random_images_is_zero() -> None:
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (16, 16), dtype=np.uint8)

    assert im.mse(image, image) == 0.0


def test_mse_analytic_constant_difference() -> None:
    a = np.full((10, 10), 100, dtype=np.uint8)
    b = np.full((10, 10), 105, dtype=np.uint8)

    assert im.mse(a, b) == pytest.approx(25.0)


def test_mse_is_symmetric() -> None:
    rng = np.random.default_rng(1)
    a = rng.integers(0, 256, (16, 16), dtype=np.uint8)
    b = rng.integers(0, 256, (16, 16), dtype=np.uint8)

    assert im.mse(a, b) == im.mse(b, a)


def test_mse_does_not_mutate_inputs() -> None:
    a = np.full((10, 10), 100, dtype=np.uint8)
    b = np.full((10, 10), 105, dtype=np.uint8)
    a_before = a.copy()
    b_before = b.copy()

    im.mse(a, b)

    np.testing.assert_array_equal(a, a_before)
    np.testing.assert_array_equal(b, b_before)


def test_mse_returns_a_plain_float() -> None:
    a = np.full((10, 10), 100, dtype=np.uint8)
    b = np.full((10, 10), 105, dtype=np.uint8)

    result = im.mse(a, b)

    assert type(result) is float


def test_mse_raises_value_error_on_extreme_float_overflow() -> None:
    a = np.full((10, 10), 1e200, dtype=np.float64)
    b = np.full((10, 10), -1e200, dtype=np.float64)

    with pytest.raises(ValueError, match="non-finite"):
        im.mse(a, b)


# --- mse/psnr underflow: a squared tiny-but-nonzero difference must not be
# silently reported as "identical" (0.0/inf) just because it underflows ---


@pytest.mark.parametrize("exponent", [160, 161])
def test_mse_represents_small_but_representable_differences(exponent: int) -> None:
    a = np.zeros((1, 1), dtype=np.float64)
    b = np.full((1, 1), 10.0**-exponent, dtype=np.float64)

    result = im.mse(a, b)

    assert result > 0.0
    assert result == pytest.approx((10.0**-exponent) ** 2, rel=1e-9)


@pytest.mark.parametrize("exponent", [162, 163, 200])
def test_mse_raises_value_error_on_true_underflow(exponent: int) -> None:
    a = np.zeros((1, 1), dtype=np.float64)
    b = np.full((1, 1), 10.0**-exponent, dtype=np.float64)

    assert not np.array_equal(a, b)
    with pytest.raises(ValueError, match="underflow"):
        im.mse(a, b)


@pytest.mark.parametrize("exponent", [160, 161, 162, 163, 200])
def test_psnr_stays_finite_even_when_mse_would_underflow(exponent: int) -> None:
    a = np.zeros((1, 1), dtype=np.float64)
    b = np.full((1, 1), 10.0**-exponent, dtype=np.float64)

    result = im.psnr(a, b, data_range=1.0)

    assert math.isfinite(result)
    assert result == pytest.approx(20.0 * exponent, abs=1e-6)


def test_psnr_of_the_1e_minus_162_example_is_about_3240_db() -> None:
    a = np.zeros((1, 1), dtype=np.float64)
    b = np.full((1, 1), 1e-162, dtype=np.float64)

    result = im.psnr(a, b, data_range=1.0)

    assert result == pytest.approx(3240.0, abs=1.0)


def test_mse_true_underflow_does_not_report_images_as_identical() -> None:
    # The bug this guards against: reporting mse == 0.0 (and therefore
    # psnr == inf) for two images that are provably not equal, just
    # because the squared difference rounds to zero.
    a = np.zeros((1, 1), dtype=np.float64)
    b = np.full((1, 1), 1e-162, dtype=np.float64)

    assert not np.array_equal(a, b)
    with pytest.raises(ValueError):
        im.mse(a, b)
    assert im.psnr(a, b, data_range=1.0) != math.inf


# --- psnr ---


def test_psnr_of_identical_images_is_infinite() -> None:
    image = np.full((10, 10), 100, dtype=np.uint8)

    assert im.psnr(image, image) == math.inf


def test_psnr_of_identical_random_images_is_infinite() -> None:
    rng = np.random.default_rng(2)
    image = rng.integers(0, 256, (16, 16), dtype=np.uint8)

    assert im.psnr(image, image) == math.inf


def test_psnr_analytic_constant_difference() -> None:
    a = np.full((10, 10), 100, dtype=np.uint8)
    b = np.full((10, 10), 105, dtype=np.uint8)

    # mse = 25; psnr = 20*log10(255) - 10*log10(25)
    expected = 20.0 * math.log10(255.0) - 10.0 * math.log10(25.0)
    assert im.psnr(a, b) == pytest.approx(expected)


def test_psnr_is_symmetric() -> None:
    rng = np.random.default_rng(3)
    a = rng.integers(0, 256, (16, 16), dtype=np.uint8)
    b = rng.integers(0, 256, (16, 16), dtype=np.uint8)

    assert im.psnr(a, b) == im.psnr(b, a)


def test_psnr_allows_negative_result_for_very_different_images() -> None:
    a = np.zeros((10, 10), dtype=np.uint8)
    b = np.full((10, 10), 255, dtype=np.uint8)

    result = im.psnr(a, b, data_range=1.0)

    assert result < 0.0


def test_psnr_infers_data_range_for_uint8() -> None:
    a = np.full((10, 10), 100, dtype=np.uint8)
    b = np.full((10, 10), 105, dtype=np.uint8)

    expected = 20.0 * math.log10(255.0) - 10.0 * math.log10(25.0)
    assert im.psnr(a, b) == pytest.approx(expected)


def test_psnr_infers_data_range_for_uint16() -> None:
    a = np.full((10, 10), 1000, dtype=np.uint16)
    b = np.full((10, 10), 1005, dtype=np.uint16)

    expected = 20.0 * math.log10(65535.0) - 10.0 * math.log10(25.0)
    assert im.psnr(a, b) == pytest.approx(expected)


def test_psnr_requires_explicit_data_range_for_float32() -> None:
    a = np.full((10, 10), 0.5, dtype=np.float32)
    b = np.full((10, 10), 0.6, dtype=np.float32)

    with pytest.raises(ValueError, match="data_range"):
        im.psnr(a, b)


def test_psnr_requires_explicit_data_range_for_float64() -> None:
    a = np.full((10, 10), 0.5, dtype=np.float64)
    b = np.full((10, 10), 0.6, dtype=np.float64)

    with pytest.raises(ValueError, match="data_range"):
        im.psnr(a, b)


def test_psnr_accepts_explicit_data_range_for_float() -> None:
    a = np.full((10, 10), 0.5, dtype=np.float32)
    b = np.full((10, 10), 0.6, dtype=np.float32)

    result = im.psnr(a, b, data_range=1.0)

    assert math.isfinite(result)


@pytest.mark.parametrize("bad_range", [0.0, -1.0, math.inf, math.nan, True, False])
def test_psnr_rejects_invalid_data_range(bad_range: object) -> None:
    a = np.full((10, 10), 100, dtype=np.uint8)
    b = np.full((10, 10), 105, dtype=np.uint8)

    with pytest.raises((ValueError, TypeError)):
        im.psnr(a, b, data_range=bad_range)  # type: ignore[arg-type]


def test_psnr_does_not_mutate_inputs() -> None:
    a = np.full((10, 10), 100, dtype=np.uint8)
    b = np.full((10, 10), 105, dtype=np.uint8)
    a_before = a.copy()
    b_before = b.copy()

    im.psnr(a, b)

    np.testing.assert_array_equal(a, a_before)
    np.testing.assert_array_equal(b, b_before)


def test_psnr_does_not_call_cv2_psnr(monkeypatch: pytest.MonkeyPatch) -> None:
    import cv2

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("cv2.PSNR must not be called")

    monkeypatch.setattr(cv2, "PSNR", _boom)

    a = np.full((10, 10), 100, dtype=np.uint8)
    b = np.full((10, 10), 105, dtype=np.uint8)
    im.psnr(a, b)  # must not raise


# --- ssim ---


def test_ssim_of_identical_images_is_one() -> None:
    image = np.full((32, 32), 100, dtype=np.uint8)

    assert im.ssim(image, image) == pytest.approx(1.0)


def test_ssim_of_identical_random_images_is_one() -> None:
    rng = np.random.default_rng(4)
    image = rng.integers(0, 256, (32, 32), dtype=np.uint8)

    assert im.ssim(image, image) == pytest.approx(1.0)


def test_ssim_of_two_different_constant_images_matches_reference() -> None:
    a = np.full((32, 32), 100, dtype=np.uint8)
    b = np.full((32, 32), 150, dtype=np.uint8)

    # Reference value from scikit-image 0.26.0:
    # structural_similarity(a, b, data_range=255, gaussian_weights=True,
    #                       sigma=1.5, use_sample_covariance=False)
    assert im.ssim(a, b) == pytest.approx(0.9230923105307928, abs=1e-9)


def test_ssim_is_symmetric() -> None:
    rng = np.random.default_rng(5)
    a = rng.integers(0, 256, (32, 32), dtype=np.uint8)
    b = rng.integers(0, 256, (32, 32), dtype=np.uint8)

    assert im.ssim(a, b) == im.ssim(b, a)


def test_ssim_does_not_mutate_inputs() -> None:
    rng = np.random.default_rng(6)
    a = rng.integers(0, 256, (32, 32), dtype=np.uint8)
    b = rng.integers(0, 256, (32, 32), dtype=np.uint8)
    a_before = a.copy()
    b_before = b.copy()

    im.ssim(a, b)

    np.testing.assert_array_equal(a, a_before)
    np.testing.assert_array_equal(b, b_before)


def test_ssim_result_is_not_clipped_and_can_be_negative() -> None:
    rng = np.random.default_rng(7)
    a = rng.integers(0, 256, (32, 32), dtype=np.uint8)
    b = 255 - a

    result = im.ssim(a, b)

    assert result < 0.0


def test_ssim_requires_minimum_11x11_spatial_size() -> None:
    a = np.zeros((10, 11), dtype=np.uint8)
    b = np.zeros((10, 11), dtype=np.uint8)

    with pytest.raises(ValueError, match="11"):
        im.ssim(a, b)


def test_ssim_requires_minimum_11x11_spatial_size_other_axis() -> None:
    a = np.zeros((11, 10), dtype=np.uint8)
    b = np.zeros((11, 10), dtype=np.uint8)

    with pytest.raises(ValueError, match="11"):
        im.ssim(a, b)


def test_ssim_accepts_exactly_11x11() -> None:
    rng = np.random.default_rng(8)
    a = rng.integers(0, 256, (11, 11), dtype=np.uint8)
    b = rng.integers(0, 256, (11, 11), dtype=np.uint8)

    result = im.ssim(a, b)

    assert math.isfinite(result)


def test_ssim_requires_explicit_data_range_for_float() -> None:
    rng = np.random.default_rng(9)
    a = rng.random((32, 32), dtype=np.float32)
    b = rng.random((32, 32), dtype=np.float32)

    with pytest.raises(ValueError, match="data_range"):
        im.ssim(a, b)


def test_ssim_does_not_convert_color_space() -> None:
    # Same channel-order data compared against itself must give ssim == 1.0
    # regardless of whether the caller considers the layout BGR or RGB --
    # ssim never inspects or converts channel semantics.
    rng = np.random.default_rng(10)
    image = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)

    assert im.ssim(image, image) == pytest.approx(1.0)


def test_ssim_treats_alpha_channel_like_any_other_channel() -> None:
    rng = np.random.default_rng(11)
    a = rng.integers(0, 256, (32, 32, 4), dtype=np.uint8)
    b = a.copy()
    b[:, :, 3] = 255 - a[:, :, 3]  # perturb only the alpha channel

    result = im.ssim(a, b)

    assert result < 1.0  # alpha channel participates -- perturbing it changes the result


def test_ssim_raises_value_error_on_extreme_float_overflow() -> None:
    a = np.full((32, 32), 1e200, dtype=np.float64)
    b = np.full((32, 32), -1e200, dtype=np.float64)

    with pytest.raises(ValueError):
        im.ssim(a, b, data_range=1.0)


# --- ssim: extreme data_range must never raise a raw OverflowError ---


def test_ssim_identical_zeros_with_data_range_1e100() -> None:
    x = np.zeros((11, 11), dtype=np.float64)

    assert im.ssim(x, x, data_range=1e100) == pytest.approx(1.0)


def test_ssim_identical_zeros_with_data_range_1e156() -> None:
    # This exact call previously raised a raw OverflowError: (K2*1e156)**2
    # overflows float64 on its own, before any image data is involved.
    x = np.zeros((11, 11), dtype=np.float64)

    assert im.ssim(x, x, data_range=1e156) == pytest.approx(1.0)


def test_ssim_identical_zeros_with_data_range_float64_max() -> None:
    x = np.zeros((11, 11), dtype=np.float64)

    assert im.ssim(x, x, data_range=np.finfo(np.float64).max) == pytest.approx(1.0)


def test_ssim_with_very_small_positive_data_range_and_real_content_does_not_crash() -> None:
    rng = np.random.default_rng(21)
    a = rng.random((16, 16))
    b = rng.random((16, 16))

    result = im.ssim(a, b, data_range=1e-300)

    assert math.isfinite(result)


def test_ssim_tiny_data_range_with_constant_images_raises_controlled_error() -> None:
    # A data_range this mismatched with constant (zero-variance) image
    # content makes C1/C2 themselves underflow to 0.0, producing an actual
    # 0/0 -- must surface as the same controlled ValueError as any other
    # non-finite result, not propagate a NaN or crash.
    x = np.zeros((11, 11), dtype=np.float64)

    with pytest.raises(ValueError):
        im.ssim(x, x, data_range=1e-300)


def test_ssim_non_identical_images_with_huge_data_range_does_not_raise() -> None:
    rng = np.random.default_rng(22)
    a = rng.integers(0, 256, (16, 16)).astype(np.float64)
    b = rng.integers(0, 256, (16, 16)).astype(np.float64)

    result = im.ssim(a, b, data_range=1e200)

    # Not compared to the data_range=255 result: a data_range this wildly
    # mismatched with the images' actual scale makes C1/C2 dominate the
    # formula entirely, correctly (per the formula's own definition, not a
    # bug) driving the result toward 1.0 regardless of real content -- the
    # only property being verified here is that it stays finite.
    assert math.isfinite(result)


@pytest.mark.parametrize("bad_range", [1e100, 1e156, 1e-300, 1e200])
def test_ssim_extreme_data_range_never_raises_overflow_error(bad_range: float) -> None:
    rng = np.random.default_rng(23)
    a = rng.integers(0, 256, (16, 16)).astype(np.float64)
    b = rng.integers(0, 256, (16, 16)).astype(np.float64)

    try:
        im.ssim(a, b, data_range=bad_range)
    except OverflowError:
        pytest.fail(f"a raw OverflowError propagated for data_range={bad_range!r}")
    except ValueError:
        pass  # a controlled ValueError is an acceptable outcome


# --- ssim cross-reference regression values (scikit-image 0.26.0) ---
# Generated with:
#   structural_similarity(a, b, data_range=..., channel_axis=...,
#                          gaussian_weights=True, sigma=1.5,
#                          use_sample_covariance=False)
# in an isolated, throwaway virtual environment -- scikit-image is not a
# project dependency (runtime or dev). Each test uses its own independent
# np.random.default_rng seed, so fixtures don't depend on call order.


def test_ssim_matches_reference_grayscale_uint8() -> None:
    rng = np.random.default_rng(101)
    a = rng.integers(0, 256, (16, 16), dtype=np.uint8)
    b = np.clip(a.astype(np.int16) + rng.integers(-15, 15, (16, 16)), 0, 255).astype(np.uint8)

    assert im.ssim(a, b) == pytest.approx(0.9924735694938048, abs=1e-9)


def test_ssim_matches_reference_bgr_uint8() -> None:
    rng = np.random.default_rng(102)
    a3 = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    b3 = np.clip(a3.astype(np.int16) + rng.integers(-15, 15, (16, 16, 3)), 0, 255).astype(np.uint8)

    assert im.ssim(a3, b3) == pytest.approx(0.9937281656717323, abs=1e-9)


def test_ssim_matches_reference_bgra_uint8() -> None:
    rng = np.random.default_rng(103)
    a4 = rng.integers(0, 256, (16, 16, 4), dtype=np.uint8)
    b4 = np.clip(a4.astype(np.int16) + rng.integers(-15, 15, (16, 16, 4)), 0, 255).astype(np.uint8)

    assert im.ssim(a4, b4) == pytest.approx(0.99214629218644, abs=1e-9)


def test_ssim_matches_reference_uint16() -> None:
    rng = np.random.default_rng(104)
    a16 = rng.integers(0, 65536, (16, 16), dtype=np.uint16)
    b16 = np.clip(a16.astype(np.int32) + rng.integers(-2000, 2000, (16, 16)), 0, 65535).astype(
        np.uint16
    )

    assert im.ssim(a16, b16, data_range=65535.0) == pytest.approx(0.9979641058750508, abs=1e-9)


def test_ssim_matches_reference_float32() -> None:
    rng = np.random.default_rng(105)
    af = rng.random((16, 16), dtype=np.float32)
    bf = np.clip(af + rng.normal(0, 0.05, (16, 16)).astype(np.float32), 0, 1).astype(np.float32)

    assert im.ssim(af, bf, data_range=1.0) == pytest.approx(0.9819720536470413, abs=1e-6)


def test_ssim_matches_reference_11x11() -> None:
    rng = np.random.default_rng(106)
    a11 = rng.integers(0, 256, (11, 11), dtype=np.uint8)
    b11 = rng.integers(0, 256, (11, 11), dtype=np.uint8)

    assert im.ssim(a11, b11) == pytest.approx(0.007222447321097228, abs=1e-9)


def test_ssim_matches_reference_larger_image() -> None:
    rng = np.random.default_rng(107)
    a64 = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    b64 = np.clip(a64.astype(np.int16) + rng.integers(-15, 15, (64, 64)), 0, 255).astype(np.uint8)

    assert im.ssim(a64, b64) == pytest.approx(0.9930489890049938, abs=1e-9)


# --- shared validation across mse/psnr/ssim ---


@pytest.mark.parametrize("func_name", ["mse", "psnr", "ssim"])
def test_rejects_empty_image(func_name: str) -> None:
    func = getattr(im, func_name)
    a = np.zeros((0, 10), dtype=np.uint8)
    b = np.zeros((0, 10), dtype=np.uint8)

    with pytest.raises(ValueError):
        func(a, b)


@pytest.mark.parametrize("func_name", ["mse", "psnr", "ssim"])
def test_rejects_zero_channel_image(func_name: str) -> None:
    func = getattr(im, func_name)
    a = np.zeros((20, 20, 0), dtype=np.uint8)
    b = np.zeros((20, 20, 0), dtype=np.uint8)

    with pytest.raises(ValueError):
        func(a, b)


@pytest.mark.parametrize("func_name", ["mse", "psnr", "ssim"])
def test_rejects_wrong_ndim(func_name: str) -> None:
    func = getattr(im, func_name)
    a = np.zeros((5, 5, 5, 5), dtype=np.uint8)
    b = np.zeros((5, 5, 5, 5), dtype=np.uint8)

    with pytest.raises(ValueError):
        func(a, b)


@pytest.mark.parametrize("func_name", ["mse", "psnr", "ssim"])
def test_rejects_mismatched_shape(func_name: str) -> None:
    func = getattr(im, func_name)
    a = np.zeros((20, 20), dtype=np.uint8)
    b = np.zeros((20, 21), dtype=np.uint8)

    with pytest.raises(ValueError, match="shape"):
        func(a, b)


@pytest.mark.parametrize("func_name", ["mse", "psnr", "ssim"])
def test_rejects_mismatched_dtype(func_name: str) -> None:
    func = getattr(im, func_name)
    a = np.zeros((20, 20), dtype=np.uint8)
    b = np.zeros((20, 20), dtype=np.float32)

    with pytest.raises(TypeError, match="dtype"):
        func(a, b)


@pytest.mark.parametrize("func_name", ["mse", "psnr", "ssim"])
def test_rejects_unsupported_dtype(func_name: str) -> None:
    func = getattr(im, func_name)
    a = np.zeros((20, 20), dtype=np.int32)
    b = np.zeros((20, 20), dtype=np.int32)

    with pytest.raises(TypeError):
        func(a, b)


@pytest.mark.parametrize("func_name", ["mse", "psnr", "ssim"])
def test_rejects_more_than_4_channels(func_name: str) -> None:
    func = getattr(im, func_name)
    a = np.zeros((20, 20, 5), dtype=np.uint8)
    b = np.zeros((20, 20, 5), dtype=np.uint8)

    with pytest.raises(ValueError):
        func(a, b)


@pytest.mark.parametrize("func_name", ["mse", "psnr", "ssim"])
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_accepts_1_3_4_channels(func_name: str, channels: int) -> None:
    func = getattr(im, func_name)
    rng = np.random.default_rng(12)
    shape = (20, 20) if channels == 1 else (20, 20, channels)
    a = rng.integers(0, 256, shape, dtype=np.uint8)
    b = rng.integers(0, 256, shape, dtype=np.uint8)

    result = func(a, b)

    assert math.isfinite(result) or result == math.inf


@pytest.mark.parametrize("func_name", ["mse", "psnr", "ssim"])
def test_accepts_explicit_h_w_1_shape(func_name: str) -> None:
    func = getattr(im, func_name)
    rng = np.random.default_rng(13)
    a = rng.integers(0, 256, (20, 20, 1), dtype=np.uint8)
    b = rng.integers(0, 256, (20, 20, 1), dtype=np.uint8)

    result = func(a, b)

    assert math.isfinite(result) or result == math.inf


@pytest.mark.parametrize("func_name", ["mse", "psnr", "ssim"])
def test_accepts_2_channels(func_name: str) -> None:
    func = getattr(im, func_name)
    rng = np.random.default_rng(14)
    a = rng.integers(0, 256, (20, 20, 2), dtype=np.uint8)
    b = rng.integers(0, 256, (20, 20, 2), dtype=np.uint8)

    result = func(a, b)

    assert math.isfinite(result) or result == math.inf


@pytest.mark.parametrize("func_name", ["mse", "psnr", "ssim"])
def test_h_w_1_gives_the_same_result_as_h_w(func_name: str) -> None:
    func = getattr(im, func_name)
    rng = np.random.default_rng(15)
    a_2d = rng.integers(0, 256, (20, 20), dtype=np.uint8)
    b_2d = rng.integers(0, 256, (20, 20), dtype=np.uint8)
    a_3d = a_2d.reshape(20, 20, 1)
    b_3d = b_2d.reshape(20, 20, 1)

    assert func(a_2d, b_2d) == func(a_3d, b_3d)


@pytest.mark.parametrize("func_name", ["mse", "psnr", "ssim"])
def test_rejects_nan_in_first_float_image(func_name: str) -> None:
    func = getattr(im, func_name)
    a = np.full((20, 20), 0.5, dtype=np.float32)
    a[0, 0] = np.nan
    b = np.full((20, 20), 0.5, dtype=np.float32)
    kwargs = {} if func_name == "mse" else {"data_range": 1.0}

    with pytest.raises(ValueError):
        func(a, b, **kwargs)


@pytest.mark.parametrize("func_name", ["mse", "psnr", "ssim"])
def test_rejects_inf_in_second_float_image(func_name: str) -> None:
    func = getattr(im, func_name)
    a = np.full((20, 20), 0.5, dtype=np.float32)
    b = np.full((20, 20), 0.5, dtype=np.float32)
    b[0, 0] = np.inf
    kwargs = {} if func_name == "mse" else {"data_range": 1.0}

    with pytest.raises(ValueError):
        func(a, b, **kwargs)


def test_validation_order_shape_before_data_range() -> None:
    # A shape mismatch must be reported even when data_range is also garbage --
    # the more fundamental image-comparability error takes priority.
    a = np.zeros((20, 20), dtype=np.uint8)
    b = np.zeros((20, 21), dtype=np.uint8)

    with pytest.raises(ValueError, match="shape"):
        im.psnr(a, b, data_range=-5.0)
