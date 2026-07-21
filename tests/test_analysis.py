import numpy as np
import pytest

import improcv as im


def test_histogram_counts_pixel_values_into_bins() -> None:
    image = np.array([[0, 0, 100, 200]], dtype=np.uint8)

    hist = im.histogram(image, channel=0, bins=2, value_range=(0.0, 200.0))

    assert hist.shape == (2,)
    assert hist.dtype == np.float32
    # bin 0 is [0, 100): catches the two 0s. bin 1 is [100, 200): catches
    # the 100. The 200 falls exactly on the whole range's exclusive upper
    # bound, so it is not counted anywhere.
    np.testing.assert_array_equal(hist, [2.0, 1.0])


def test_histogram_upper_bound_is_exclusive() -> None:
    image = np.array([[200]], dtype=np.uint8)

    hist = im.histogram(image, channel=0, bins=1, value_range=(0.0, 200.0))

    assert hist[0] == 0.0


def test_histogram_selects_one_channel_of_a_multichannel_image() -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[:, :, 1] = 200  # only the green channel is nonzero, and >= the bin midpoint (128)

    hist_channel_0 = im.histogram(image, channel=0, bins=2, value_range=(0.0, 256.0))
    hist_channel_1 = im.histogram(image, channel=1, bins=2, value_range=(0.0, 256.0))

    assert hist_channel_0[0] == 16.0  # all 16 pixels (value 0) fall in the low bin [0, 128)
    assert hist_channel_1[1] == 16.0  # all 16 pixels (value 200) fall in the high bin [128, 256)


def test_histogram_applies_mask() -> None:
    image = np.array([[10, 20, 30, 40]], dtype=np.uint8)
    mask = np.array([[1, 1, 0, 0]], dtype=np.uint8)

    hist = im.histogram(image, channel=0, bins=1, value_range=(0.0, 256.0), mask=mask)

    assert hist[0] == 2.0  # only the two masked-in pixels counted


def test_histogram_uses_default_parameters() -> None:
    image = np.zeros((4, 4), dtype=np.uint8)

    hist = im.histogram(image)

    assert hist.shape == (256,)


def test_histogram_rejects_unsupported_dtype() -> None:
    image = np.zeros((4, 4), dtype=np.int32)

    with pytest.raises(TypeError, match="dtype"):
        im.histogram(image)  # type: ignore[arg-type]


def test_histogram_rejects_out_of_range_channel() -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="channel"):
        im.histogram(image, channel=3)


def test_histogram_rejects_non_integer_channel() -> None:
    image = np.zeros((4, 4), dtype=np.uint8)

    with pytest.raises(TypeError, match="integer"):
        im.histogram(image, channel=0.0)  # type: ignore[arg-type]


def test_histogram_rejects_zero_bins() -> None:
    image = np.zeros((4, 4), dtype=np.uint8)

    with pytest.raises(ValueError, match="positive"):
        im.histogram(image, bins=0)


def test_histogram_rejects_non_positive_range() -> None:
    image = np.zeros((4, 4), dtype=np.uint8)

    with pytest.raises(ValueError, match="value_range"):
        im.histogram(image, value_range=(200.0, 0.0))


def test_histogram_rejects_non_uint8_mask() -> None:
    image = np.zeros((4, 4), dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=np.float32)

    with pytest.raises(TypeError, match="uint8"):
        im.histogram(image, mask=mask)  # type: ignore[arg-type]


def test_histogram_does_not_mutate_input() -> None:
    image = np.array([[10, 20, 30]], dtype=np.uint8)
    original = image.copy()

    im.histogram(image)

    np.testing.assert_array_equal(image, original)


def test_moments_computes_raw_moments_for_a_filled_square() -> None:
    image = np.zeros((10, 10), dtype=np.uint8)
    image[2:8, 2:8] = 255  # 6x6 = 36 pixels, value 255

    result = im.moments(image)

    assert result.m00 == 36.0 * 255.0


def test_moments_binary_image_treats_nonzero_as_one_not_255() -> None:
    image = np.zeros((10, 10), dtype=np.uint8)
    image[2:8, 2:8] = 255

    result = im.moments(image, binary_image=True)

    assert result.m00 == 36.0


def test_moments_has_24_named_fields_matching_cv2() -> None:
    image = np.zeros((10, 10), dtype=np.uint8)
    image[2:8, 2:8] = 255

    result = im.moments(image)

    assert len(result._fields) == 24
    assert result._fields[:3] == ("m00", "m10", "m01")


def test_moments_rejects_multichannel_raster_image() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="dimensions"):
        im.moments(image)


def test_moments_rejects_unsupported_raster_dtype() -> None:
    image = np.zeros((10, 10), dtype=np.int32)

    with pytest.raises(TypeError, match="dtype"):
        im.moments(image)  # type: ignore[arg-type]


def test_moments_rejects_non_bool_binary_image() -> None:
    image = np.zeros((10, 10), dtype=np.uint8)

    with pytest.raises(TypeError, match="bool"):
        im.moments(image, binary_image=1)  # type: ignore[arg-type]


def test_moments_does_not_mutate_raster_input() -> None:
    image = np.zeros((10, 10), dtype=np.uint8)
    image[2:8, 2:8] = 255
    original = image.copy()

    im.moments(image)

    np.testing.assert_array_equal(image, original)


def test_moments_computes_moments_for_a_contour() -> None:
    # A square contour (corners only, matching cv2.findContours' "simple"
    # approximation output) spanning coordinates 0..9 -- a 9x9 span, so the
    # shoelace-formula area (m00) is 9*9 = 81, not 10*10.
    contour = np.array([[[0, 0]], [[9, 0]], [[9, 9]], [[0, 9]]], dtype=np.int32)

    result = im.moments(contour)

    assert result.m00 == 81.0


def test_moments_ignores_binary_image_is_rejected_for_contour() -> None:
    contour = np.array([[[0, 0]], [[9, 0]], [[9, 9]], [[0, 9]]], dtype=np.int32)

    with pytest.raises(ValueError, match="binary_image"):
        im.moments(contour, binary_image=True)


def test_moments_rejects_empty_contour() -> None:
    empty_contour = np.empty((0, 1, 2), dtype=np.int32)

    with pytest.raises(ValueError, match="point"):
        im.moments(empty_contour)


def test_moments_rejects_malformed_contour_shape() -> None:
    # ndim == 3 (so it dispatches to the contour path, not the raster path),
    # but the last dimension is 3 instead of 2 -- not a valid (N, 1, 2) Contour.
    bad_contour = np.zeros((5, 1, 3), dtype=np.int32)

    with pytest.raises(ValueError, match="shape"):
        im.moments(bad_contour)


def test_match_template_finds_exact_match_location() -> None:
    image = np.zeros((20, 20), dtype=np.uint8)
    image[5:10, 8:14] = np.arange(1, 31).reshape(5, 6)  # unique, non-constant patch
    template = image[5:10, 8:14].copy()

    result = im.match_template(image, template, "sqdiff")

    assert result.shape == (16, 15)
    assert result.dtype == np.float32
    min_loc = np.unravel_index(np.argmin(result), result.shape)
    assert min_loc == (5, 8)


def test_match_template_rejects_template_larger_than_image() -> None:
    image = np.zeros((10, 10), dtype=np.uint8)
    template = np.zeros((11, 10), dtype=np.uint8)

    with pytest.raises(ValueError, match="fit"):
        im.match_template(image, template, "ccorr")


def test_match_template_rejects_mismatched_dtype() -> None:
    image = np.zeros((10, 10), dtype=np.uint8)
    template = np.zeros((5, 5), dtype=np.float32)

    with pytest.raises(TypeError, match="dtype"):
        im.match_template(image, template, "ccorr")  # type: ignore[arg-type]


def test_match_template_rejects_unsupported_dtype() -> None:
    image = np.zeros((10, 10), dtype=np.int32)
    template = np.zeros((5, 5), dtype=np.int32)

    with pytest.raises(TypeError, match="dtype"):
        im.match_template(image, template, "ccorr")  # type: ignore[arg-type]


def test_match_template_rejects_mismatched_channel_count() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    template = np.zeros((5, 5), dtype=np.uint8)

    with pytest.raises(ValueError, match="channel"):
        im.match_template(image, template, "ccorr")


def test_match_template_rejects_more_than_four_channels() -> None:
    image = np.zeros((10, 10, 5), dtype=np.uint8)
    template = np.zeros((5, 5, 5), dtype=np.uint8)

    with pytest.raises(ValueError, match="channel"):
        im.match_template(image, template, "ccorr")


def test_match_template_rejects_invalid_method() -> None:
    image = np.zeros((10, 10), dtype=np.uint8)
    template = np.zeros((5, 5), dtype=np.uint8)

    with pytest.raises(ValueError, match="method"):
        im.match_template(image, template, "invalid")  # type: ignore[arg-type]


def test_match_template_does_not_mutate_input() -> None:
    image = np.zeros((10, 10), dtype=np.uint8)
    image[2:5, 2:5] = np.arange(1, 10).reshape(3, 3)
    template = image[2:5, 2:5].copy()
    original_image = image.copy()
    original_template = template.copy()

    im.match_template(image, template, "sqdiff")

    np.testing.assert_array_equal(image, original_image)
    np.testing.assert_array_equal(template, original_template)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("size", [5, 10])
def test_match_template_rejects_constant_grayscale_template_for_ccoeff_normed(
    dtype: type, size: int
) -> None:
    rng = np.random.default_rng(0)
    image = rng.uniform(0, 255, size=(20, 20)).astype(dtype)
    template = np.full((size, size), 50, dtype=dtype)

    with pytest.raises(ValueError, match="constant"):
        im.match_template(image, template, "ccoeff_normed")


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("size", [5, 10])
def test_match_template_rejects_constant_grayscale_template_for_sqdiff_normed(
    dtype: type, size: int
) -> None:
    rng = np.random.default_rng(0)
    image = rng.uniform(0, 255, size=(20, 20)).astype(dtype)
    template = np.full((size, size), 50, dtype=dtype)

    with pytest.raises(ValueError, match="constant"):
        im.match_template(image, template, "sqdiff_normed")


def test_match_template_allows_constant_nonzero_grayscale_template_for_ccorr_normed() -> None:
    rng = np.random.default_rng(0)
    image = rng.uniform(0, 255, size=(20, 20)).astype(np.uint8)
    template = np.full((5, 5), 50, dtype=np.uint8)

    result = im.match_template(image, template, "ccorr_normed")  # must not raise

    assert result.shape == (16, 16)


def test_match_template_rejects_constant_bgr_template_for_ccoeff_normed() -> None:
    # Per-channel constant (0, 128, 255) -- global std is nonzero, but each
    # channel individually has zero spatial variance. A naive
    # template.std() == 0 check would wrongly accept this.
    rng = np.random.default_rng(0)
    image = rng.uniform(0, 255, size=(20, 20, 3)).astype(np.uint8)
    template = np.zeros((5, 5, 3), dtype=np.uint8)
    template[:, :, 0] = 0
    template[:, :, 1] = 128
    template[:, :, 2] = 255

    with pytest.raises(ValueError, match="constant"):
        im.match_template(image, template, "ccoeff_normed")


def test_match_template_allows_one_constant_channel_with_one_varying_channel() -> None:
    rng = np.random.default_rng(0)
    image = rng.uniform(0, 255, size=(20, 20, 3)).astype(np.uint8)
    template = np.zeros((5, 5, 3), dtype=np.uint8)
    template[:, :, 0] = 100  # constant
    template[:, :, 1] = image[3:8, 3:8, 1]  # varying
    template[:, :, 2] = 50  # constant

    for method in ("ccoeff_normed", "sqdiff_normed", "ccorr_normed"):
        result = im.match_template(image, template, method)  # must not raise
        assert result.shape == (16, 16)


@pytest.mark.parametrize("method", ["ccoeff_normed", "sqdiff_normed", "ccorr_normed"])
def test_match_template_rejects_all_zero_template_for_every_normalized_method(
    method: str,
) -> None:
    rng = np.random.default_rng(0)
    image = rng.uniform(0, 255, size=(20, 20, 3)).astype(np.uint8)
    template = np.zeros((5, 5, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        im.match_template(image, template, method)  # type: ignore[arg-type]


def test_match_template_allows_constant_template_for_unnormalized_methods() -> None:
    rng = np.random.default_rng(0)
    image = rng.uniform(0, 255, size=(20, 20)).astype(np.uint8)
    template = np.full((5, 5), 50, dtype=np.uint8)

    for method in ("ccoeff", "ccorr", "sqdiff"):
        result = im.match_template(image, template, method)  # must not raise
        assert result.shape == (16, 16)
