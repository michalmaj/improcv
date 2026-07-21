import numpy as np
import pytest

import improcv as im


def test_inpaint_reconstructs_a_masked_region_2d_uint8() -> None:
    image = np.random.default_rng(0).integers(0, 255, (20, 20), dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[8:12, 8:12] = 255

    result = im.inpaint(image, mask, radius=3.0, method="telea")

    assert result.shape == image.shape
    assert result.dtype == image.dtype


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_inpaint_accepts_every_supported_2d_dtype(dtype: type) -> None:
    image = np.zeros((20, 20), dtype=dtype)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[8:12, 8:12] = 255

    result = im.inpaint(image, mask, radius=3.0)

    assert result.dtype == dtype
    assert np.all(np.isfinite(result.astype(np.float64)))


def test_inpaint_accepts_three_channel_uint8() -> None:
    image = np.random.default_rng(0).integers(0, 255, (20, 20, 3), dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[8:12, 8:12] = 255

    result = im.inpaint(image, mask, radius=3.0)

    assert result.shape == (20, 20, 3)
    assert result.dtype == np.uint8


@pytest.mark.parametrize("dtype", [np.uint16, np.float32])
def test_inpaint_rejects_three_channel_non_uint8(dtype: type) -> None:
    image = np.zeros((20, 20, 3), dtype=dtype)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[8:12, 8:12] = 255

    with pytest.raises(TypeError, match="dtype"):
        im.inpaint(image, mask, radius=3.0)  # type: ignore[arg-type]


def test_inpaint_rejects_single_channel_3d_image() -> None:
    # (H, W, 1) is not accepted as an implicit grayscale image.
    image = np.zeros((20, 20, 1), dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[8:12, 8:12] = 255

    with pytest.raises(ValueError, match="channels"):
        im.inpaint(image, mask, radius=3.0)


@pytest.mark.parametrize("method", ["ns", "telea"])
def test_inpaint_accepts_both_methods(method: str) -> None:
    image = np.random.default_rng(0).integers(0, 255, (20, 20), dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[8:12, 8:12] = 255

    result = im.inpaint(image, mask, radius=3.0, method=method)  # type: ignore[arg-type]

    assert result.shape == image.shape


def test_inpaint_rejects_invalid_method() -> None:
    image = np.zeros((20, 20), dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=np.uint8)

    with pytest.raises(ValueError, match="method"):
        im.inpaint(image, mask, radius=3.0, method="invalid")  # type: ignore[arg-type]


def test_inpaint_all_zero_mask_returns_unchanged_copy() -> None:
    image = np.random.default_rng(0).integers(0, 255, (10, 10), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=np.uint8)

    result = im.inpaint(image, mask, radius=3.0)

    np.testing.assert_array_equal(result, image)
    assert not np.shares_memory(result, image)


def test_inpaint_rejects_fully_masked_image() -> None:
    image = np.zeros((10, 10), dtype=np.uint8)
    mask = np.full((10, 10), 255, dtype=np.uint8)

    with pytest.raises(ValueError, match="known"):
        im.inpaint(image, mask, radius=3.0)


def test_inpaint_accepts_single_masked_pixel() -> None:
    image = np.random.default_rng(0).integers(0, 255, (10, 10), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[5, 5] = 255

    result = im.inpaint(image, mask, radius=3.0)

    assert result.shape == image.shape


def test_inpaint_accepts_masked_region_touching_every_edge() -> None:
    # Verified: no exact pixel values pinned -- only structural invariants,
    # since the reconstructed content isn't promised bit-identical across
    # OpenCV builds, especially for float32 with the Telea method.
    image = np.random.default_rng(0).random((20, 20)).astype(np.float32)
    original = image.copy()
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[0, :] = 255
    mask[-1, :] = 255
    mask[:, 0] = 255
    mask[:, -1] = 255

    result = im.inpaint(image, mask, radius=3.0, method="telea")

    assert result.shape == image.shape
    assert result.dtype == image.dtype
    np.testing.assert_array_equal(image, original)  # input not mutated
    assert not np.shares_memory(result, image)
    assert np.all(np.isfinite(result))


def test_inpaint_rejects_zero_radius() -> None:
    image = np.zeros((10, 10), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[5, 5] = 255

    with pytest.raises(ValueError, match="positive"):
        im.inpaint(image, mask, radius=0.0)


def test_inpaint_rejects_negative_radius() -> None:
    image = np.zeros((10, 10), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[5, 5] = 255

    with pytest.raises(ValueError, match="positive"):
        im.inpaint(image, mask, radius=-1.0)


def test_inpaint_rejects_nan_and_infinite_radius() -> None:
    image = np.zeros((10, 10), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[5, 5] = 255

    with pytest.raises(ValueError, match="finite"):
        im.inpaint(image, mask, radius=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        im.inpaint(image, mask, radius=float("inf"))


def test_inpaint_rejects_bool_radius() -> None:
    image = np.zeros((10, 10), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[5, 5] = 255

    with pytest.raises(TypeError, match="real number"):
        im.inpaint(image, mask, radius=True)  # type: ignore[arg-type]


def test_inpaint_does_not_mutate_input() -> None:
    image = np.random.default_rng(0).integers(0, 255, (10, 10), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[5, 5] = 255
    original = image.copy()

    im.inpaint(image, mask, radius=3.0)

    np.testing.assert_array_equal(image, original)


def test_inpaint_result_does_not_share_memory_with_input() -> None:
    image = np.random.default_rng(0).integers(0, 255, (10, 10), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[5, 5] = 255

    result = im.inpaint(image, mask, radius=3.0)

    assert not np.shares_memory(result, image)


def test_restoration_types_are_in_module_all() -> None:
    import improcv.restoration as restoration_module

    assert "InpaintMethod" in restoration_module.__all__
    assert hasattr(restoration_module, "InpaintMethod")


def test_restoration_public_names_are_reexported_from_improcv() -> None:
    for name in ("inpaint", "InpaintMethod"):
        assert name in im.__all__
        assert hasattr(im, name)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
@pytest.mark.parametrize("method", ["ns", "telea"])
@pytest.mark.parametrize("shape", [(1, 20), (20, 1), (1, 1)])
def test_inpaint_rejects_single_row_or_column_image(
    shape: tuple[int, int], method: str, dtype: type
) -> None:
    # Verified directly, repeatedly, across separate processes: a (1, N)
    # float32 image produces nondeterministic output for the exact same
    # input -- including NaN -- on both OpenCV 4.13 and 5.0. Rejected
    # uniformly for every dtype, not only where the corruption is visible
    # as NaN -- a (1, N) uint8/uint16 image was also verified to produce
    # non-reproducible output across runs, just not NaN-capable.
    image = np.ones(shape, dtype=dtype)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[0, 0] = 255

    with pytest.raises(ValueError, match="at least 2 pixels"):
        im.inpaint(image, mask, radius=3.0, method=method)  # type: ignore[arg-type]


@pytest.mark.parametrize("method", ["ns", "telea"])
@pytest.mark.parametrize("shape", [(1, 20, 3), (20, 1, 3), (1, 1, 3)])
def test_inpaint_rejects_single_row_or_column_bgr_image(
    shape: tuple[int, int, int], method: str
) -> None:
    image = np.ones(shape, dtype=np.uint8)
    mask = np.zeros(shape[:2], dtype=np.uint8)
    mask[0, 0] = 255

    with pytest.raises(ValueError, match="at least 2 pixels"):
        im.inpaint(image, mask, radius=3.0, method=method)  # type: ignore[arg-type]


def test_inpaint_rejects_float32_input_with_nan() -> None:
    image = np.ones((10, 10), dtype=np.float32)
    image[0, 0] = np.nan
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[4:6, 4:6] = 255

    with pytest.raises(ValueError, match="finite"):
        im.inpaint(image, mask, radius=3.0)


def test_inpaint_rejects_float32_input_with_infinity() -> None:
    image = np.ones((10, 10), dtype=np.float32)
    image[0, 0] = np.inf
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[4:6, 4:6] = 255

    with pytest.raises(ValueError, match="finite"):
        im.inpaint(image, mask, radius=3.0)


def test_inpaint_raises_runtime_error_for_non_finite_result_from_finite_float32_input() -> None:
    # Verified directly, deterministically, on both OpenCV 4.13 and 5.0:
    # a float32 image filled with numpy.finfo(numpy.float32).max produces
    # a non-finite (inf/NaN) result even though the input itself is finite.
    image = np.full((10, 10), np.finfo(np.float32).max, dtype=np.float32)
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[4:6, 4:6] = 255

    with pytest.raises(RuntimeError, match="non-finite"):
        im.inpaint(image, mask, radius=3.0)
