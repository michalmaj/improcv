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


def test_ssim_tiny_data_range_with_identical_constant_images_is_exactly_one() -> None:
    # A data_range this mismatched with constant (zero-variance) image
    # content used to make C1/C2 themselves underflow to 0.0, producing an
    # actual 0/0 NaN -- fixed by both the exact-equality fast path (this
    # case) and, for non-identical-but-tiny inputs, the two-sided rescaling
    # below.
    x = np.zeros((11, 11), dtype=np.float64)

    assert im.ssim(x, x, data_range=1e-300) == 1.0


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


# --- ssim: extreme *small* data_range/image magnitude must not underflow
# the formula's internal products either (the symmetric counterpart to the
# large-magnitude fix above) ---


def test_ssim_identical_zeros_with_data_range_1e_minus_100_is_exactly_one() -> None:
    x = np.zeros((11, 11), dtype=np.float64)

    assert im.ssim(x, x, data_range=1e-100) == 1.0


def test_ssim_identical_zeros_with_data_range_1e_minus_300_is_exactly_one() -> None:
    x = np.zeros((11, 11), dtype=np.float64)

    assert im.ssim(x, x, data_range=1e-300) == 1.0


def test_ssim_identical_images_with_smallest_possible_data_range_is_exactly_one() -> None:
    rng = np.random.default_rng(24)
    x = rng.random((16, 16))

    assert im.ssim(x, x, data_range=np.nextafter(0.0, 1.0)) == 1.0


def test_ssim_constant_1e_minus_100_images_match_unit_scale_reference() -> None:
    a = np.full((11, 11), 1e-100, dtype=np.float64)
    b = np.full((11, 11), 2e-100, dtype=np.float64)
    reference_a = np.ones((11, 11), dtype=np.float64)
    reference_b = np.full((11, 11), 2.0, dtype=np.float64)

    result = im.ssim(a, b, data_range=1e-100)
    reference = im.ssim(reference_a, reference_b, data_range=1.0)

    assert result == pytest.approx(reference, abs=1e-9)


def test_ssim_constant_1e_minus_200_images_match_unit_scale_reference() -> None:
    a = np.full((11, 11), 1e-200, dtype=np.float64)
    b = np.full((11, 11), 2e-200, dtype=np.float64)
    reference_a = np.ones((11, 11), dtype=np.float64)
    reference_b = np.full((11, 11), 2.0, dtype=np.float64)

    result = im.ssim(a, b, data_range=1e-200)
    reference = im.ssim(reference_a, reference_b, data_range=1.0)

    assert result == pytest.approx(reference, abs=1e-9)


def test_ssim_random_images_scaled_down_by_1e_minus_100_match_unit_scale() -> None:
    rng = np.random.default_rng(25)
    a = rng.integers(0, 256, (16, 16)).astype(np.float64)
    b = rng.integers(0, 256, (16, 16)).astype(np.float64)

    scaled = im.ssim(a * 1e-100, b * 1e-100, data_range=255e-100)
    unit_scale = im.ssim(a, b, data_range=255.0)

    assert scaled == pytest.approx(unit_scale, abs=1e-9)


def test_ssim_with_very_small_positive_data_range_and_real_content_stays_finite() -> None:
    rng = np.random.default_rng(21)
    a = rng.random((16, 16))
    b = rng.random((16, 16))

    result = im.ssim(a, b, data_range=1e-300)

    assert math.isfinite(result)


@pytest.mark.parametrize("bad_range", [1e-100, 1e-200, 1e-300])
def test_ssim_small_data_range_never_raises_overflow_error(bad_range: float) -> None:
    rng = np.random.default_rng(26)
    a = rng.integers(0, 256, (16, 16)).astype(np.float64)
    b = rng.integers(0, 256, (16, 16)).astype(np.float64)

    try:
        result = im.ssim(a, b, data_range=bad_range)
    except OverflowError:
        pytest.fail(f"a raw OverflowError propagated for data_range={bad_range!r}")
    else:
        assert math.isfinite(result)


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


# --- gmsd ---
#
# Implements Xue, Zhang, Mou, Bovik, "Gradient Magnitude Similarity
# Deviation: A Highly Efficient Perceptual Image Quality Index" (IEEE TIP,
# 2014), matching the reference MATLAB implementation the authors shared
# (GMSD.m from www4.comp.polyu.edu.hk/~cslzhang/IQA/GMSD/GMSD.htm), not the
# paper's own rounded prose. Cross-checked against that exact, unmodified
# file run in GNU Octave 11.3.0 (an isolated, throwaway environment -- not a
# project dependency). Unlike mse/psnr/ssim, gmsd is grayscale-only (no
# multi-channel contract) and lower scores mean higher quality.


def test_gmsd_of_identical_images_is_zero() -> None:
    image = np.full((16, 16), 100, dtype=np.uint8)

    assert im.gmsd(image, image) == 0.0


def test_gmsd_of_identical_random_images_is_zero() -> None:
    rng = np.random.default_rng(200)
    image = rng.integers(0, 256, (16, 16), dtype=np.uint8)

    assert im.gmsd(image, image) == 0.0


def test_gmsd_is_symmetric() -> None:
    rng = np.random.default_rng(201)
    a = rng.integers(0, 256, (16, 16), dtype=np.uint8)
    b = rng.integers(0, 256, (16, 16), dtype=np.uint8)

    assert im.gmsd(a, b) == im.gmsd(b, a)


def test_gmsd_does_not_mutate_inputs() -> None:
    rng = np.random.default_rng(202)
    a = rng.integers(0, 256, (16, 16), dtype=np.uint8)
    b = rng.integers(0, 256, (16, 16), dtype=np.uint8)
    a_before = a.copy()
    b_before = b.copy()

    im.gmsd(a, b)

    np.testing.assert_array_equal(a, a_before)
    np.testing.assert_array_equal(b, b_before)


def test_gmsd_returns_a_plain_float() -> None:
    rng = np.random.default_rng(203)
    a = rng.integers(0, 256, (16, 16), dtype=np.uint8)
    b = rng.integers(0, 256, (16, 16), dtype=np.uint8)

    result = im.gmsd(a, b)

    assert type(result) is float


def test_gmsd_more_distortion_gives_a_higher_score() -> None:
    # gmsd is a *distortion* measure: lower is better, unlike ssim/psnr.
    rng = np.random.default_rng(204)
    a = rng.integers(0, 256, (32, 32), dtype=np.uint8)
    mild = np.clip(a.astype(np.int16) + rng.integers(-5, 5, (32, 32)), 0, 255).astype(np.uint8)
    strong = np.clip(a.astype(np.int16) + rng.integers(-60, 60, (32, 32)), 0, 255).astype(np.uint8)

    assert im.gmsd(a, mild) < im.gmsd(a, strong)


def test_gmsd_infers_data_range_for_uint8() -> None:
    rng = np.random.default_rng(205)
    a = rng.integers(0, 256, (16, 16), dtype=np.uint8)
    b = rng.integers(0, 256, (16, 16), dtype=np.uint8)

    assert im.gmsd(a, b) == im.gmsd(a, b, data_range=255.0)


def test_gmsd_infers_data_range_for_uint16() -> None:
    rng = np.random.default_rng(206)
    a = rng.integers(0, 65536, (16, 16), dtype=np.uint16)
    b = rng.integers(0, 65536, (16, 16), dtype=np.uint16)

    assert im.gmsd(a, b) == im.gmsd(a, b, data_range=65535.0)


def test_gmsd_requires_explicit_data_range_for_float32() -> None:
    rng = np.random.default_rng(207)
    a = rng.random((16, 16), dtype=np.float32)
    b = rng.random((16, 16), dtype=np.float32)

    with pytest.raises(ValueError, match="data_range"):
        im.gmsd(a, b)


def test_gmsd_requires_explicit_data_range_for_float64() -> None:
    rng = np.random.default_rng(208)
    a = rng.random((16, 16))
    b = rng.random((16, 16))

    with pytest.raises(ValueError, match="data_range"):
        im.gmsd(a, b)


def test_gmsd_accepts_explicit_data_range_for_float() -> None:
    rng = np.random.default_rng(209)
    a = rng.random((16, 16), dtype=np.float32)
    b = rng.random((16, 16), dtype=np.float32)

    result = im.gmsd(a, b, data_range=1.0)

    assert math.isfinite(result)


@pytest.mark.parametrize("bad_range", [0.0, -1.0, math.inf, math.nan, True, False])
def test_gmsd_rejects_invalid_data_range(bad_range: object) -> None:
    rng = np.random.default_rng(210)
    a = rng.integers(0, 256, (16, 16), dtype=np.uint8)
    b = rng.integers(0, 256, (16, 16), dtype=np.uint8)

    with pytest.raises((ValueError, TypeError)):
        im.gmsd(a, b, data_range=bad_range)  # type: ignore[arg-type]


# --- gmsd: shared-shape validation (mirrors mse/psnr/ssim's own checks) ---


def test_gmsd_rejects_empty_image() -> None:
    a = np.zeros((0, 10), dtype=np.uint8)
    b = np.zeros((0, 10), dtype=np.uint8)

    with pytest.raises(ValueError):
        im.gmsd(a, b)


def test_gmsd_rejects_wrong_ndim() -> None:
    a = np.zeros((5, 5, 5, 5), dtype=np.uint8)
    b = np.zeros((5, 5, 5, 5), dtype=np.uint8)

    with pytest.raises(ValueError):
        im.gmsd(a, b)


def test_gmsd_rejects_mismatched_shape() -> None:
    a = np.zeros((16, 16), dtype=np.uint8)
    b = np.zeros((16, 17), dtype=np.uint8)

    with pytest.raises(ValueError, match="shape"):
        im.gmsd(a, b)


def test_gmsd_rejects_mismatched_dtype() -> None:
    a = np.zeros((16, 16), dtype=np.uint8)
    b = np.zeros((16, 16), dtype=np.float32)

    with pytest.raises(TypeError, match="dtype"):
        im.gmsd(a, b)


def test_gmsd_rejects_unsupported_dtype() -> None:
    a = np.zeros((16, 16), dtype=np.int32)
    b = np.zeros((16, 16), dtype=np.int32)

    with pytest.raises(TypeError):
        im.gmsd(a, b)


def test_gmsd_rejects_nan_in_first_float_image() -> None:
    a = np.full((16, 16), 0.5, dtype=np.float32)
    a[0, 0] = np.nan
    b = np.full((16, 16), 0.5, dtype=np.float32)

    with pytest.raises(ValueError):
        im.gmsd(a, b, data_range=1.0)


def test_gmsd_rejects_inf_in_second_float_image() -> None:
    a = np.full((16, 16), 0.5, dtype=np.float32)
    b = np.full((16, 16), 0.5, dtype=np.float32)
    b[0, 0] = np.inf

    with pytest.raises(ValueError):
        im.gmsd(a, b, data_range=1.0)


# --- gmsd: grayscale-only contract (no 2/3/4-channel support, unlike
# mse/psnr/ssim) ---


def test_gmsd_accepts_explicit_h_w_1_shape() -> None:
    rng = np.random.default_rng(211)
    a = rng.integers(0, 256, (16, 16, 1), dtype=np.uint8)
    b = rng.integers(0, 256, (16, 16, 1), dtype=np.uint8)

    result = im.gmsd(a, b)

    assert math.isfinite(result)


def test_gmsd_h_w_1_gives_the_same_result_as_h_w() -> None:
    rng = np.random.default_rng(212)
    a_2d = rng.integers(0, 256, (16, 16), dtype=np.uint8)
    b_2d = rng.integers(0, 256, (16, 16), dtype=np.uint8)
    a_3d = a_2d.reshape(16, 16, 1)
    b_3d = b_2d.reshape(16, 16, 1)

    assert im.gmsd(a_2d, b_2d) == im.gmsd(a_3d, b_3d)


@pytest.mark.parametrize("channels", [2, 3, 4])
def test_gmsd_rejects_multi_channel_images(channels: int) -> None:
    rng = np.random.default_rng(213)
    a = rng.integers(0, 256, (16, 16, channels), dtype=np.uint8)
    b = rng.integers(0, 256, (16, 16, channels), dtype=np.uint8)

    with pytest.raises(ValueError, match="ensure_gray"):
        im.gmsd(a, b)


# --- gmsd: degenerate downsampled-map-size rejection -- a deliberate,
# safer departure from the reference GMSD.m, which returns 0.0 for these
# sizes (std of a single-element map) regardless of whether the two images
# are identical or completely different; see the reference cross-check
# tests below for that documented divergence. ---


@pytest.mark.parametrize("shape", [(1, 1), (1, 2), (2, 1), (2, 2)])
def test_gmsd_rejects_degenerate_downsampled_size(shape: tuple[int, int]) -> None:
    rng = np.random.default_rng(214)
    a = rng.integers(0, 256, shape, dtype=np.uint8)
    b = rng.integers(0, 256, shape, dtype=np.uint8)

    with pytest.raises(ValueError, match="at least 2"):
        im.gmsd(a, b)


def test_gmsd_rejects_degenerate_size_even_for_identical_images() -> None:
    # The degenerate-size check must run *before* the exact-equality fast
    # path -- an identical pair with degenerate geometry must still raise,
    # not silently return 0.0.
    image = np.full((2, 2), 100, dtype=np.uint8)

    with pytest.raises(ValueError, match="at least 2"):
        im.gmsd(image, image)


@pytest.mark.parametrize("shape", [(1, 3), (3, 1), (3, 3)])
def test_gmsd_accepts_smallest_valid_sizes(shape: tuple[int, int]) -> None:
    rng = np.random.default_rng(215)
    a = rng.integers(0, 256, shape, dtype=np.uint8)
    b = rng.integers(0, 256, shape, dtype=np.uint8)

    result = im.gmsd(a, b)

    assert math.isfinite(result)


# --- gmsd: validation order ---


def test_gmsd_validation_order_shape_before_data_range() -> None:
    a = np.zeros((16, 16), dtype=np.uint8)
    b = np.zeros((16, 17), dtype=np.uint8)

    with pytest.raises(ValueError, match="shape"):
        im.gmsd(a, b, data_range=-5.0)


def test_gmsd_validation_order_channel_count_before_finite_check() -> None:
    # A rejected channel count must be reported even when the (also invalid)
    # float data additionally contains NaN -- the more fundamental
    # grayscale-contract error takes priority.
    a = np.full((16, 16, 3), 0.5, dtype=np.float32)
    a[0, 0, 0] = np.nan
    b = np.full((16, 16, 3), 0.5, dtype=np.float32)

    with pytest.raises(ValueError, match="ensure_gray"):
        im.gmsd(a, b, data_range=1.0)


def test_gmsd_validation_order_data_range_before_sample_count() -> None:
    # An invalid data_range must be reported even for a degenerate-size
    # image pair -- data_range resolution happens before the sample-count
    # check in the documented validation order.
    a = np.zeros((2, 2), dtype=np.uint8)
    b = np.zeros((2, 2), dtype=np.uint8)

    with pytest.raises(ValueError, match="data_range"):
        im.gmsd(a, b, data_range=-5.0)


# --- gmsd: border-padding causes a nonzero score for two different
# constant images (each image's own interior gradient is exactly zero, but
# GMSD.m's zero-padded convolution creates a nonzero gradient exactly at
# the border) -- deliberate reference behavior, not a bug. ---


def test_gmsd_of_two_different_constant_images_is_nonzero() -> None:
    a = np.full((10, 10), 100, dtype=np.uint8)
    b = np.full((10, 10), 150, dtype=np.uint8)

    result = im.gmsd(a, b)

    assert result > 0.0


def test_gmsd_of_two_equal_constant_images_is_zero() -> None:
    image = np.full((10, 10), 100, dtype=np.uint8)

    assert im.gmsd(image, image) == 0.0


# --- gmsd: numerical stability (mirrors ssim's own extreme-magnitude
# tests; justified via 2nd-degree, not ssim's 4th-degree, reasoning -- see
# _GMSD_SAFE_MAGNITUDE_MAX/MIN's definitions in quality.py) ---


def test_gmsd_extreme_float_magnitude_does_not_raise_overflow_error() -> None:
    a = np.full((32, 32), 1e200, dtype=np.float64)
    b = np.full((32, 32), 1.5e200, dtype=np.float64)

    result = im.gmsd(a, b, data_range=1e200)

    assert math.isfinite(result)


@pytest.mark.parametrize("data_range", [1e100, 1e156, np.finfo(np.float64).max])
def test_gmsd_identical_zeros_with_huge_data_range_is_exactly_zero(data_range: float) -> None:
    x = np.zeros((16, 16), dtype=np.float64)

    assert im.gmsd(x, x, data_range=data_range) == 0.0


@pytest.mark.parametrize("data_range", [1e-100, 1e-300, np.nextafter(0.0, 1.0)])
def test_gmsd_identical_zeros_with_tiny_data_range_is_exactly_zero(data_range: float) -> None:
    x = np.zeros((16, 16), dtype=np.float64)

    assert im.gmsd(x, x, data_range=data_range) == 0.0


@pytest.mark.parametrize("bad_range", [1e300, 1e250, 1e200, 1e100])
def test_gmsd_large_mismatched_data_range_never_raises_overflow_error(bad_range: float) -> None:
    rng = np.random.default_rng(216)
    a = rng.integers(0, 256, (16, 16)).astype(np.float64)
    b = rng.integers(0, 256, (16, 16)).astype(np.float64)

    try:
        result = im.gmsd(a, b, data_range=bad_range)
    except OverflowError:
        pytest.fail(f"a raw OverflowError propagated for data_range={bad_range!r}")
    else:
        assert math.isfinite(result)


@pytest.mark.parametrize("bad_range", [1e-300, 5e-324, 1e-100])
def test_gmsd_small_mismatched_data_range_never_raises_overflow_error(bad_range: float) -> None:
    rng = np.random.default_rng(217)
    a = rng.integers(0, 256, (16, 16)).astype(np.float64)
    b = rng.integers(0, 256, (16, 16)).astype(np.float64)

    try:
        result = im.gmsd(a, b, data_range=bad_range)
    except OverflowError:
        pytest.fail(f"a raw OverflowError propagated for data_range={bad_range!r}")
    else:
        assert math.isfinite(result)


def test_gmsd_extreme_magnitude_images_never_raise_overflow_error() -> None:
    maxv = np.finfo(np.float64).max
    a = np.full((16, 16), maxv, dtype=np.float64)
    b = np.full((16, 16), maxv / 2.0, dtype=np.float64)

    try:
        result = im.gmsd(a, b, data_range=1.0)
    except OverflowError:
        pytest.fail("a raw OverflowError propagated for extreme-magnitude image content")
    else:
        assert math.isfinite(result)


def test_gmsd_random_images_scaled_down_by_1e_minus_100_match_unit_scale() -> None:
    rng = np.random.default_rng(218)
    a = rng.integers(0, 256, (16, 16)).astype(np.float64)
    b = rng.integers(0, 256, (16, 16)).astype(np.float64)

    scaled = im.gmsd(a * 1e-100, b * 1e-100, data_range=255e-100)
    unit_scale = im.gmsd(a, b, data_range=255.0)

    assert scaled == pytest.approx(unit_scale, abs=1e-9)


# --- gmsd: the 2x2 averaging filter's anchor and zero-padded border are
# not incidental defaults -- using cv2.filter2D's own default anchor/border
# for that specific filter does NOT reproduce the reference score. This
# locks in both non-obvious details against a silent regression. ---


def test_gmsd_default_anchor_and_border_do_not_match_the_reference() -> None:
    import cv2

    t_const = 170.0 / 255.0**2
    dx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float64) / 3.0
    dy = dx.T.copy()
    avg_kernel = np.full((2, 2), 0.25, dtype=np.float64)

    rng = np.random.default_rng(219)
    a_u8 = rng.integers(0, 256, (11, 11), dtype=np.uint8)
    b_u8 = rng.integers(0, 256, (11, 11), dtype=np.uint8)
    a = a_u8.astype(np.float64)
    b = b_u8.astype(np.float64)

    def wrong_border_gmsd(y1: np.ndarray, y2: np.ndarray) -> float:
        # Deliberately wrong: default anchor + default border for the 2x2
        # averaging step, instead of anchor=(0, 0) + BORDER_CONSTANT.
        ave_a = cv2.filter2D(y1, ddepth=cv2.CV_64F, kernel=avg_kernel)
        ave_b = cv2.filter2D(y2, ddepth=cv2.CV_64F, kernel=avg_kernel)
        down_a = ave_a[0::2, 0::2]
        down_b = ave_b[0::2, 0::2]
        ix_a = cv2.filter2D(down_a, ddepth=cv2.CV_64F, kernel=dx, borderType=cv2.BORDER_CONSTANT)
        iy_a = cv2.filter2D(down_a, ddepth=cv2.CV_64F, kernel=dy, borderType=cv2.BORDER_CONSTANT)
        grad_a = np.sqrt(ix_a * ix_a + iy_a * iy_a)
        ix_b = cv2.filter2D(down_b, ddepth=cv2.CV_64F, kernel=dx, borderType=cv2.BORDER_CONSTANT)
        iy_b = cv2.filter2D(down_b, ddepth=cv2.CV_64F, kernel=dy, borderType=cv2.BORDER_CONSTANT)
        grad_b = np.sqrt(ix_b * ix_b + iy_b * iy_b)
        gms_map = (2 * grad_a * grad_b + t_const * 255.0**2) / (
            grad_a * grad_a + grad_b * grad_b + t_const * 255.0**2
        )
        return float(np.std(gms_map, ddof=1))

    correct = im.gmsd(a_u8, b_u8)
    wrong = wrong_border_gmsd(a, b)

    assert correct != pytest.approx(wrong, abs=1e-9)


# --- gmsd cross-reference regression values (GMSD.m, the reference MATLAB
# implementation shared by the original authors, run unmodified in GNU
# Octave 11.3.0 -- Octave is not a project dependency, used only for this
# one-off verification). Parameters: T=170, ddof=1 (sample std, matching
# MATLAB's std2/std default), zero-padded ("same") convolution, anchor=(0,0)
# for the 2x2 averaging filter. Each test uses its own independent
# np.random.default_rng seed, so fixtures don't depend on call order. ---


def test_gmsd_matches_reference_noise_11x11() -> None:
    rng = np.random.default_rng(90001)
    a = rng.integers(0, 256, (11, 11), dtype=np.uint8)
    b = rng.integers(0, 256, (11, 11), dtype=np.uint8)

    assert im.gmsd(a, b) == pytest.approx(0.1596606810745852, abs=1e-9)


def test_gmsd_matches_reference_even_10x10() -> None:
    rng = np.random.default_rng(90002)
    a = rng.integers(0, 256, (10, 10), dtype=np.uint8)
    b = rng.integers(0, 256, (10, 10), dtype=np.uint8)

    assert im.gmsd(a, b) == pytest.approx(0.12466163314694197, abs=1e-9)


def test_gmsd_matches_reference_mixed_9x7() -> None:
    rng = np.random.default_rng(90003)
    a = rng.integers(0, 256, (9, 7), dtype=np.uint8)
    b = rng.integers(0, 256, (9, 7), dtype=np.uint8)

    assert im.gmsd(a, b) == pytest.approx(0.08925372152976777, abs=1e-9)


def test_gmsd_matches_reference_mixed_7x9() -> None:
    rng = np.random.default_rng(90004)
    a = rng.integers(0, 256, (7, 9), dtype=np.uint8)
    b = rng.integers(0, 256, (7, 9), dtype=np.uint8)

    assert im.gmsd(a, b) == pytest.approx(0.1023551589053105, abs=1e-9)


def test_gmsd_matches_reference_row_1x3() -> None:
    rng = np.random.default_rng(90005)
    a = rng.integers(0, 256, (1, 3), dtype=np.uint8)
    b = rng.integers(0, 256, (1, 3), dtype=np.uint8)

    assert im.gmsd(a, b) == pytest.approx(0.03337823653830409, abs=1e-9)


def test_gmsd_matches_reference_col_3x1() -> None:
    rng = np.random.default_rng(90006)
    a = rng.integers(0, 256, (3, 1), dtype=np.uint8)
    b = rng.integers(0, 256, (3, 1), dtype=np.uint8)

    assert im.gmsd(a, b) == pytest.approx(0.12182241457031197, abs=1e-9)


def test_gmsd_matches_reference_small_3x3() -> None:
    rng = np.random.default_rng(90007)
    a = rng.integers(0, 256, (3, 3), dtype=np.uint8)
    b = rng.integers(0, 256, (3, 3), dtype=np.uint8)

    assert im.gmsd(a, b) == pytest.approx(0.044742855478195426, abs=1e-9)


def test_gmsd_matches_reference_big_32x32() -> None:
    rng = np.random.default_rng(90008)
    a = rng.integers(0, 256, (32, 32), dtype=np.uint8)
    b = rng.integers(0, 256, (32, 32), dtype=np.uint8)

    assert im.gmsd(a, b) == pytest.approx(0.20281712958975784, abs=1e-9)


def test_gmsd_matches_reference_constant_images() -> None:
    a = np.full((10, 10), 100, dtype=np.uint8)
    b = np.full((10, 10), 150, dtype=np.uint8)

    assert im.gmsd(a, b) == pytest.approx(0.03748227666652311, abs=1e-8)


# --- gmsd: degenerate sizes are a deliberate, documented divergence from
# the reference. GMSD.m computes std() of a single-element quality map for
# these sizes, and MATLAB's std() has an undocumented-in-code special case
# returning 0.0 (not NaN) for N=1 -- confirmed directly against the
# unmodified reference in Octave, for both identical and different image
# content, since the reference's std2() cannot distinguish the two cases
# at N=1. improcv raises ValueError instead of silently reporting perfect
# quality for potentially very different images. ---


@pytest.mark.parametrize("shape", [(1, 1), (1, 2), (2, 1), (2, 2)])
def test_gmsd_degenerate_sizes_diverge_from_reference_by_design(shape: tuple[int, int]) -> None:
    rng = np.random.default_rng(90009)
    a = rng.integers(0, 256, shape, dtype=np.uint8)
    b = rng.integers(0, 256, shape, dtype=np.uint8)

    # GMSD.m (run unmodified in Octave) returns 0.0 here regardless of a/b's
    # content -- improcv instead raises, deliberately not matching this.
    with pytest.raises(ValueError, match="at least 2"):
        im.gmsd(a, b)
