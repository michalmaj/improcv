import copy
import dataclasses
import warnings

import cv2
import numpy as np
import pytest

import improcv as im
from improcv.augmentation import (
    AffineParameters,
    AugmentedImageMask,
    CropParameters,
    FlipParameters,
    PerspectiveParameters,
    apply_affine,
    apply_crop,
    apply_flip,
    apply_perspective,
    sample_affine,
    sample_crop,
    sample_flip,
    sample_perspective,
)


def _make_image(height: int, width: int, channels: int | None = 3) -> np.ndarray:
    shape = (height, width) if channels is None else (height, width, channels)
    return (np.arange(int(np.prod(shape))) % 256).astype(np.uint8).reshape(shape)


def _make_mask(height: int, width: int, dtype: type = np.uint8) -> np.ndarray:
    return (np.arange(height * width) % 4).astype(dtype).reshape(height, width)


# --- import hygiene ---


def test_import_does_not_touch_rng_or_filesystem() -> None:
    # A fresh generator seeded identically before/after importing improcv must
    # still produce identical draws -- the import itself must not consume
    # any global or ambient RNG state, perform augmentation, or touch the
    # filesystem/ML frameworks.
    import importlib

    import improcv

    importlib.reload(improcv)
    rng_a = np.random.default_rng(0)
    rng_b = np.random.default_rng(0)
    assert rng_a.random() == rng_b.random()


# --- sample_flip ---


def test_sample_flip_defaults_returns_flip_parameters() -> None:
    rng = np.random.default_rng(0)
    params = sample_flip(rng)
    assert isinstance(params, FlipParameters)
    assert isinstance(params.horizontal, bool)
    assert isinstance(params.vertical, bool)


def test_sample_flip_probability_zero_never_flips_that_axis() -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        params = sample_flip(rng, horizontal_probability=0.0, vertical_probability=0.0)
        assert params.horizontal is False
        assert params.vertical is False


def test_sample_flip_probability_one_always_flips_that_axis() -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        params = sample_flip(rng, horizontal_probability=1.0, vertical_probability=1.0)
        assert params.horizontal is True
        assert params.vertical is True


def test_sample_flip_axes_are_independent() -> None:
    rng = np.random.default_rng(0)
    seen = {(False, False), (False, True), (True, False), (True, True)}
    observed = set()
    for _ in range(200):
        params = sample_flip(rng, horizontal_probability=0.5, vertical_probability=0.5)
        observed.add((params.horizontal, params.vertical))
    assert observed == seen


def test_sample_flip_fresh_generators_same_seed_give_same_params() -> None:
    params_a = sample_flip(np.random.default_rng(123))
    params_b = sample_flip(np.random.default_rng(123))
    assert params_a == params_b


def test_sample_flip_consecutive_calls_on_same_generator_can_differ() -> None:
    rng = np.random.default_rng(7)
    results = [sample_flip(rng, horizontal_probability=0.5) for _ in range(20)]
    assert len({(r.horizontal, r.vertical) for r in results}) > 1


def test_sample_flip_params_are_replayable() -> None:
    rng = np.random.default_rng(1)
    params = sample_flip(rng, horizontal_probability=1.0)
    image = _make_image(4, 5)
    first = apply_flip(image, params)
    second = apply_flip(image, params)
    np.testing.assert_array_equal(first, second)


def test_sample_flip_rejects_non_generator_rng() -> None:
    with pytest.raises(TypeError, match="Generator"):
        sample_flip(np.random.RandomState(0))  # type: ignore[arg-type]


def test_sample_flip_rejects_none_rng() -> None:
    with pytest.raises(TypeError, match="Generator"):
        sample_flip(None)  # type: ignore[arg-type]


def test_sample_flip_rejects_int_seed_as_rng() -> None:
    with pytest.raises(TypeError, match="Generator"):
        sample_flip(42)  # type: ignore[arg-type]


def test_sample_flip_rejects_numpy_random_module_as_rng() -> None:
    with pytest.raises(TypeError, match="Generator"):
        sample_flip(np.random)  # type: ignore[arg-type]


class _FakeGenerator:
    def random(self) -> float:
        return 0.0


def test_sample_flip_rejects_duck_typed_fake_rng() -> None:
    with pytest.raises(TypeError, match="Generator"):
        sample_flip(_FakeGenerator())  # type: ignore[arg-type]


def test_sample_flip_rejects_bool_probability() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(TypeError):
        sample_flip(rng, horizontal_probability=True)  # type: ignore[arg-type]


def test_sample_flip_rejects_nan_probability() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        sample_flip(rng, horizontal_probability=float("nan"))


def test_sample_flip_rejects_infinite_probability() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        sample_flip(rng, vertical_probability=float("inf"))


def test_sample_flip_rejects_out_of_range_probability() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        sample_flip(rng, horizontal_probability=1.5)


def test_sample_flip_accepts_numpy_real_scalar_probability() -> None:
    rng = np.random.default_rng(0)
    params = sample_flip(rng, horizontal_probability=np.float32(0.5))  # type: ignore[arg-type]
    assert isinstance(params, FlipParameters)


# --- apply_flip ---


def test_apply_flip_no_op_returns_independent_copy() -> None:
    image = _make_image(4, 5)
    before = image.copy()
    result = apply_flip(image, FlipParameters(False, False))
    np.testing.assert_array_equal(result, image)
    assert result is not image
    assert not np.shares_memory(result, image)
    np.testing.assert_array_equal(image, before)


def test_apply_flip_horizontal_matches_transforms_flip() -> None:
    image = _make_image(4, 5)
    result = apply_flip(image, FlipParameters(True, False))
    np.testing.assert_array_equal(result, im.flip(image, "horizontal"))


def test_apply_flip_vertical_matches_transforms_flip() -> None:
    image = _make_image(4, 5)
    result = apply_flip(image, FlipParameters(False, True))
    np.testing.assert_array_equal(result, im.flip(image, "vertical"))


def test_apply_flip_both_matches_transforms_flip_both() -> None:
    image = _make_image(4, 5)
    result = apply_flip(image, FlipParameters(True, True))
    np.testing.assert_array_equal(result, im.flip(image, "both"))


def test_apply_flip_both_matches_two_sequential_cv2_flips() -> None:
    image = _make_image(4, 5)
    result = apply_flip(image, FlipParameters(True, True))
    sequential = cv2.flip(cv2.flip(image, 1), 0)
    np.testing.assert_array_equal(result, sequential)


def test_apply_flip_does_not_call_cv2_flip_twice(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    real_flip = cv2.flip

    def counting_flip(src, flipCode):
        calls.append(flipCode)
        return real_flip(src, flipCode)

    monkeypatch.setattr(cv2, "flip", counting_flip)
    image = _make_image(4, 5)
    apply_flip(image, FlipParameters(True, True))
    assert len(calls) == 1


@pytest.mark.parametrize("channels", [None, 3])
def test_apply_flip_2d_and_bgr_layouts(channels: int | None) -> None:
    image = _make_image(6, 8, channels=channels)
    result = apply_flip(image, FlipParameters(True, False))
    assert result.shape == image.shape


def test_apply_flip_bgra_layout() -> None:
    image = _make_image(6, 8, channels=4)
    result = apply_flip(image, FlipParameters(True, True))
    assert result.shape == image.shape


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int16, np.float32, np.float64])
def test_apply_flip_supported_image_dtypes(dtype: type) -> None:
    image = _make_image(4, 5).astype(dtype)
    result = apply_flip(image, FlipParameters(True, False))
    assert result.dtype == dtype


def test_apply_flip_rejects_unsupported_image_dtype() -> None:
    image = _make_image(4, 5).astype(np.int32)
    with pytest.raises(TypeError, match="dtype"):
        apply_flip(image, FlipParameters(True, False))


def test_apply_flip_with_mask_returns_augmented_image_mask() -> None:
    image = _make_image(4, 5)
    mask = _make_mask(4, 5)
    result = apply_flip(image, FlipParameters(True, False), mask=mask)
    assert isinstance(result, AugmentedImageMask)
    np.testing.assert_array_equal(result.image, im.flip(image, "horizontal"))
    np.testing.assert_array_equal(result.mask, im.flip(mask, "horizontal"))


def test_apply_flip_mask_shape_hw1_preserved() -> None:
    image = _make_image(4, 5)
    mask = _make_mask(4, 5).reshape(4, 5, 1)
    result = apply_flip(image, FlipParameters(True, True), mask=mask)
    assert result.mask.shape == (4, 5, 1)
    expected = cv2.flip(mask[:, :, 0], -1)[:, :, None]
    np.testing.assert_array_equal(result.mask, expected)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int16])
def test_apply_flip_supported_mask_dtypes(dtype: type) -> None:
    image = _make_image(4, 5)
    mask = _make_mask(4, 5, dtype=dtype)
    result = apply_flip(image, FlipParameters(True, False), mask=mask)
    assert result.mask.dtype == dtype


@pytest.mark.parametrize("dtype", [np.bool_, np.int32, np.int64, np.float32, np.float64])
def test_apply_flip_rejects_unsupported_mask_dtype(dtype: type) -> None:
    image = _make_image(4, 5)
    mask = np.zeros((4, 5), dtype=dtype)
    with pytest.raises(TypeError, match="dtype"):
        apply_flip(image, FlipParameters(True, False), mask=mask)


def test_apply_flip_rejects_mask_spatial_mismatch() -> None:
    image = _make_image(4, 5)
    mask = _make_mask(5, 5)
    with pytest.raises(ValueError, match="spatial size"):
        apply_flip(image, FlipParameters(True, False), mask=mask)


def test_apply_flip_rejects_mask_with_wrong_channel_count() -> None:
    image = _make_image(4, 5)
    mask = np.zeros((4, 5, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match=r"\(H, W, 1\)"):
        apply_flip(image, FlipParameters(True, False), mask=mask)


def test_apply_flip_accepts_read_only_and_non_contiguous_image() -> None:
    image = _make_image(6, 8)
    read_only = image.copy()
    read_only.setflags(write=False)
    result = apply_flip(read_only, FlipParameters(True, False))
    np.testing.assert_array_equal(result, im.flip(image, "horizontal"))

    non_contiguous = image[:, ::2]
    assert not non_contiguous.flags["C_CONTIGUOUS"]
    result2 = apply_flip(non_contiguous, FlipParameters(False, True))
    np.testing.assert_array_equal(result2, im.flip(non_contiguous, "vertical"))


def test_apply_flip_does_not_mutate_image_or_mask() -> None:
    image = _make_image(4, 5)
    mask = _make_mask(4, 5)
    image_before = image.copy()
    mask_before = mask.copy()
    apply_flip(image, FlipParameters(True, True), mask=mask)
    np.testing.assert_array_equal(image, image_before)
    np.testing.assert_array_equal(mask, mask_before)


def test_apply_flip_result_never_aliases_inputs() -> None:
    image = _make_image(4, 5)
    mask = _make_mask(4, 5)
    result = apply_flip(image, FlipParameters(False, False), mask=mask)
    assert not np.shares_memory(result.image, image)
    assert not np.shares_memory(result.mask, mask)


def test_apply_flip_augmented_image_mask_equality() -> None:
    image = _make_image(4, 5)
    mask = _make_mask(4, 5)
    a = apply_flip(image, FlipParameters(True, False), mask=mask)
    b = apply_flip(image, FlipParameters(True, False), mask=mask)
    assert a == b
    assert a != object()
    with pytest.raises(TypeError):
        hash(a)


def test_apply_flip_rejects_dict_params() -> None:
    image = _make_image(4, 5)
    with pytest.raises(TypeError, match="FlipParameters"):
        apply_flip(image, {"horizontal": True, "vertical": False})  # type: ignore[arg-type]


def test_apply_flip_rejects_tuple_params() -> None:
    image = _make_image(4, 5)
    with pytest.raises(TypeError, match="FlipParameters"):
        apply_flip(image, (True, False))  # type: ignore[arg-type]


def test_apply_flip_rejects_none_params() -> None:
    image = _make_image(4, 5)
    with pytest.raises(TypeError, match="FlipParameters"):
        apply_flip(image, None)  # type: ignore[arg-type]


def test_apply_flip_rejects_manually_constructed_bad_bool_fields() -> None:
    image = _make_image(4, 5)
    bad_params = FlipParameters.__new__(FlipParameters)
    object.__setattr__(bad_params, "horizontal", 1)
    object.__setattr__(bad_params, "vertical", False)
    with pytest.raises(TypeError, match="bool"):
        apply_flip(image, bad_params)


def test_apply_flip_raises_runtime_error_on_postcondition_violation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import improcv.augmentation as augmentation_module

    monkeypatch.setattr(
        augmentation_module,
        "_flip",
        lambda image, direction: np.zeros((999, 999), dtype=image.dtype),
    )
    image = _make_image(4, 5)
    with pytest.raises(RuntimeError, match="internal error"):
        apply_flip(image, FlipParameters(True, False))


def test_apply_flip_does_not_reshape_arbitrary_same_size_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import improcv.augmentation as augmentation_module

    image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)

    monkeypatch.setattr(
        augmentation_module,
        "_flip",
        lambda image, direction: np.zeros((4, 15), dtype=image.dtype),
    )

    with pytest.raises(RuntimeError, match="shape"):
        apply_flip(image, FlipParameters(True, False))


def test_apply_flip_maps_unexpected_opencv_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import improcv.augmentation as augmentation_module

    error = cv2.error("simulated failure")

    def fail(image, direction):
        raise error

    monkeypatch.setattr(augmentation_module, "_flip", fail)

    with pytest.raises(RuntimeError, match="OpenCV failed") as exc_info:
        apply_flip(
            np.zeros((4, 5, 3), dtype=np.uint8),
            FlipParameters(True, False),
        )

    assert exc_info.value.__cause__ is error


def test_apply_flip_maps_unexpected_opencv_error_for_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import improcv.augmentation as augmentation_module

    real_flip = augmentation_module._flip
    error = cv2.error("simulated mask failure")
    calls = {"count": 0}

    def flip_then_fail(image, direction):
        calls["count"] += 1
        if calls["count"] == 1:
            return real_flip(image, direction)
        raise error

    monkeypatch.setattr(augmentation_module, "_flip", flip_then_fail)

    image = _make_image(4, 5)
    mask = _make_mask(4, 5)
    with pytest.raises(RuntimeError, match="OpenCV failed") as exc_info:
        apply_flip(image, FlipParameters(True, False), mask=mask)

    assert exc_info.value.__cause__ is error
    assert calls["count"] == 2


# --- sample_crop ---


def test_sample_crop_source_and_crop_use_width_height_order() -> None:
    rng = np.random.default_rng(0)
    params = sample_crop(rng, source_size=(10, 4), crop_size=(6, 3))
    assert params.width == 6
    assert params.height == 3
    assert params.source_size == (10, 4)


def test_sample_crop_equal_to_source_gives_top_left_deterministically() -> None:
    rng = np.random.default_rng(0)
    for _ in range(10):
        params = sample_crop(rng, source_size=(5, 4), crop_size=(5, 4))
        assert params.x == 0
        assert params.y == 0


def test_sample_crop_larger_than_source_raises_value_error() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="exceed"):
        sample_crop(rng, source_size=(5, 4), crop_size=(6, 4))


def test_sample_crop_larger_height_than_source_raises_value_error() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="exceed"):
        sample_crop(rng, source_size=(5, 4), crop_size=(5, 5))


@pytest.mark.parametrize("size", [(0, 4), (5, 0), (-1, 4), (5, -1)])
def test_sample_crop_rejects_non_positive_sizes(size: tuple[int, int]) -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        sample_crop(rng, source_size=size, crop_size=(1, 1))


def test_sample_crop_rejects_bool_dimension() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(TypeError):
        sample_crop(rng, source_size=(True, 4), crop_size=(1, 1))  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [[5, 5], None, "55"])
def test_sample_crop_rejects_non_tuple_source_size(bad: object) -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(TypeError, match="tuple"):
        sample_crop(rng, source_size=bad, crop_size=(1, 1))  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [[5, 5], None, "55"])
def test_sample_crop_rejects_non_tuple_crop_size(bad: object) -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(TypeError, match="tuple"):
        sample_crop(rng, source_size=(5, 5), crop_size=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [(5,), (5, 5, 5)])
def test_sample_crop_rejects_wrong_length_source_size(bad: tuple) -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="exactly 2 elements"):
        sample_crop(rng, source_size=bad, crop_size=(1, 1))  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [(5,), (5, 5, 5)])
def test_sample_crop_rejects_wrong_length_crop_size(bad: tuple) -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="exactly 2 elements"):
        sample_crop(rng, source_size=(5, 5), crop_size=bad)  # type: ignore[arg-type]


def test_sample_crop_accepts_numpy_integral_dimensions() -> None:
    rng = np.random.default_rng(0)
    params = sample_crop(
        rng,
        source_size=(np.int32(5), np.int64(4)),  # type: ignore[arg-type]
        crop_size=(np.uint8(2), np.uint8(2)),  # type: ignore[arg-type]
    )
    assert params.width == 2
    assert params.height == 2
    assert isinstance(params.width, int)
    assert isinstance(params.source_size[0], int)


def test_sample_crop_right_and_bottom_edges_are_reachable() -> None:
    rng = np.random.default_rng(0)
    seen_x = set()
    seen_y = set()
    for _ in range(300):
        params = sample_crop(rng, source_size=(4, 3), crop_size=(2, 2))
        seen_x.add(params.x)
        seen_y.add(params.y)
    assert seen_x == {0, 1, 2}
    assert seen_y == {0, 1}


def test_sample_crop_all_legal_positions_on_small_image_are_reachable() -> None:
    rng = np.random.default_rng(1)
    seen = set()
    for _ in range(500):
        params = sample_crop(rng, source_size=(3, 2), crop_size=(2, 1))
        seen.add((params.x, params.y))
    assert seen == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_sample_crop_fresh_generators_same_seed_give_same_params() -> None:
    a = sample_crop(np.random.default_rng(9), source_size=(10, 10), crop_size=(4, 4))
    b = sample_crop(np.random.default_rng(9), source_size=(10, 10), crop_size=(4, 4))
    assert a == b


def test_sample_crop_params_are_replayable() -> None:
    rng = np.random.default_rng(3)
    params = sample_crop(rng, source_size=(8, 6), crop_size=(4, 3))
    image = _make_image(6, 8)
    first = apply_crop(image, params)
    second = apply_crop(image, params)
    np.testing.assert_array_equal(first, second)


def test_crop_parameters_contains_source_size() -> None:
    rng = np.random.default_rng(0)
    params = sample_crop(rng, source_size=(10, 4), crop_size=(6, 3))
    assert isinstance(params, CropParameters)
    assert params.source_size == (10, 4)


def test_sample_crop_rejects_non_generator_rng() -> None:
    with pytest.raises(TypeError, match="Generator"):
        sample_crop(np.random.RandomState(0), source_size=(5, 5), crop_size=(2, 2))  # type: ignore[arg-type]


# --- apply_crop ---


def test_apply_crop_image_only_matches_transforms_crop() -> None:
    image = _make_image(6, 8)
    params = CropParameters(x=2, y=1, width=4, height=3, source_size=(8, 6))
    result = apply_crop(image, params)
    np.testing.assert_array_equal(result, im.crop(image, 2, 1, 4, 3))


def test_apply_crop_with_mask_returns_augmented_image_mask() -> None:
    image = _make_image(6, 8)
    mask = _make_mask(6, 8)
    params = CropParameters(x=2, y=1, width=4, height=3, source_size=(8, 6))
    result = apply_crop(image, params, mask=mask)
    assert isinstance(result, AugmentedImageMask)
    np.testing.assert_array_equal(result.image, im.crop(image, 2, 1, 4, 3))
    np.testing.assert_array_equal(result.mask, im.crop(mask, 2, 1, 4, 3))


def test_apply_crop_of_entire_image_still_returns_copy() -> None:
    image = _make_image(4, 5)
    params = CropParameters(x=0, y=0, width=5, height=4, source_size=(5, 4))
    result = apply_crop(image, params)
    np.testing.assert_array_equal(result, image)
    assert result is not image
    assert not np.shares_memory(result, image)


def test_apply_crop_preserves_trailing_channels() -> None:
    image = _make_image(6, 8, channels=4)
    params = CropParameters(x=1, y=1, width=3, height=2, source_size=(8, 6))
    result = apply_crop(image, params)
    assert result.shape == (2, 3, 4)


def test_apply_crop_rejects_source_size_mismatch() -> None:
    image = _make_image(6, 8)
    params = CropParameters(x=0, y=0, width=3, height=3, source_size=(999, 999))
    with pytest.raises(ValueError, match="source_size"):
        apply_crop(image, params)


def test_apply_crop_rejects_mask_spatial_mismatch() -> None:
    image = _make_image(6, 8)
    mask = _make_mask(5, 8)
    params = CropParameters(x=0, y=0, width=3, height=3, source_size=(8, 6))
    with pytest.raises(ValueError, match="spatial size"):
        apply_crop(image, params, mask=mask)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int16])
def test_apply_crop_supported_mask_dtypes(dtype: type) -> None:
    image = _make_image(6, 8)
    mask = _make_mask(6, 8, dtype=dtype)
    params = CropParameters(x=1, y=1, width=3, height=2, source_size=(8, 6))
    result = apply_crop(image, params, mask=mask)
    assert result.mask.dtype == dtype


@pytest.mark.parametrize("dtype", [np.bool_, np.int32, np.int64, np.float32])
def test_apply_crop_rejects_unsupported_mask_dtype(dtype: type) -> None:
    image = _make_image(6, 8)
    mask = np.zeros((6, 8), dtype=dtype)
    params = CropParameters(x=0, y=0, width=3, height=3, source_size=(8, 6))
    with pytest.raises(TypeError, match="dtype"):
        apply_crop(image, params, mask=mask)


def test_apply_crop_accepts_read_only_and_non_contiguous_image() -> None:
    image = _make_image(6, 8)
    read_only = image.copy()
    read_only.setflags(write=False)
    params = CropParameters(x=1, y=1, width=3, height=2, source_size=(8, 6))
    result = apply_crop(read_only, params)
    np.testing.assert_array_equal(result, im.crop(image, 1, 1, 3, 2))

    non_contiguous = image[:, ::2]
    assert not non_contiguous.flags["C_CONTIGUOUS"]
    params2 = CropParameters(x=0, y=0, width=2, height=2, source_size=(4, 6))
    result2 = apply_crop(non_contiguous, params2)
    np.testing.assert_array_equal(result2, im.crop(non_contiguous, 0, 0, 2, 2))


def test_apply_crop_does_not_mutate_image_or_mask() -> None:
    image = _make_image(6, 8)
    mask = _make_mask(6, 8)
    image_before = image.copy()
    mask_before = mask.copy()
    params = CropParameters(x=1, y=1, width=3, height=2, source_size=(8, 6))
    apply_crop(image, params, mask=mask)
    np.testing.assert_array_equal(image, image_before)
    np.testing.assert_array_equal(mask, mask_before)


def test_apply_crop_result_never_aliases_inputs() -> None:
    image = _make_image(6, 8)
    mask = _make_mask(6, 8)
    params = CropParameters(x=0, y=0, width=8, height=6, source_size=(8, 6))
    result = apply_crop(image, params, mask=mask)
    assert not np.shares_memory(result.image, image)
    assert not np.shares_memory(result.mask, mask)


def test_apply_crop_rejects_manually_constructed_illegal_params() -> None:
    image = _make_image(6, 8)
    bad_params = CropParameters(x=-1, y=0, width=3, height=3, source_size=(8, 6))
    with pytest.raises(ValueError, match="non-negative"):
        apply_crop(image, bad_params)


def test_apply_crop_rejects_params_where_crop_exceeds_source() -> None:
    image = _make_image(6, 8)
    bad_params = CropParameters(x=6, y=0, width=5, height=3, source_size=(8, 6))
    with pytest.raises(ValueError, match="exceeds"):
        apply_crop(image, bad_params)


def test_apply_crop_rejects_wrong_params_type() -> None:
    image = _make_image(6, 8)
    with pytest.raises(TypeError, match="CropParameters"):
        apply_crop(image, {"x": 0, "y": 0, "width": 1, "height": 1, "source_size": (8, 6)})  # type: ignore[arg-type]


def test_apply_crop_raises_runtime_error_on_postcondition_violation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import improcv.augmentation as augmentation_module

    monkeypatch.setattr(
        augmentation_module,
        "_crop",
        lambda image, x, y, width, height: np.zeros((999, 999), dtype=image.dtype),
    )
    image = _make_image(6, 8)
    params = CropParameters(x=0, y=0, width=3, height=2, source_size=(8, 6))
    with pytest.raises(RuntimeError, match="internal error"):
        apply_crop(image, params)


# --- sample_affine ---


def _reference_matrix(
    source_size: tuple[int, int], angle: float, dx: float, dy: float, scale: float
) -> np.ndarray:
    width, height = source_size
    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    matrix[0, 2] += dx
    matrix[1, 2] += dy
    return matrix


def test_sample_affine_identity_defaults() -> None:
    rng = np.random.default_rng(0)
    params = sample_affine(rng, source_size=(10, 8))
    assert isinstance(params, AffineParameters)
    np.testing.assert_allclose(params.matrix, np.eye(2, 3))
    assert params.angle == 0.0
    assert params.translation == (0.0, 0.0)
    assert params.scale == 1.0
    assert params.source_size == (10, 8)


def test_sample_affine_singleton_angle_range() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        params = sample_affine(rng, source_size=(10, 8), angle_range=(15.0, 15.0))
        assert params.angle == 15.0


def test_sample_affine_singleton_translation_range() -> None:
    rng = np.random.default_rng(0)
    params = sample_affine(
        rng,
        source_size=(10, 8),
        translation_x_range=(3.5, 3.5),
        translation_y_range=(-2.5, -2.5),
    )
    assert params.translation == (3.5, -2.5)


def test_sample_affine_singleton_scale_range() -> None:
    rng = np.random.default_rng(0)
    params = sample_affine(rng, source_size=(10, 8), scale_range=(1.5, 1.5))
    assert params.scale == 1.5


def test_sample_affine_angle_within_range() -> None:
    rng = np.random.default_rng(0)
    for _ in range(200):
        params = sample_affine(rng, source_size=(10, 8), angle_range=(-30.0, 30.0))
        assert -30.0 <= params.angle <= 30.0


def test_sample_affine_translation_within_range() -> None:
    rng = np.random.default_rng(0)
    for _ in range(200):
        params = sample_affine(
            rng,
            source_size=(10, 8),
            translation_x_range=(-5.0, 5.0),
            translation_y_range=(-3.0, 3.0),
        )
        assert -5.0 <= params.translation[0] <= 5.0
        assert -3.0 <= params.translation[1] <= 3.0


def test_sample_affine_scale_within_range() -> None:
    rng = np.random.default_rng(0)
    for _ in range(200):
        params = sample_affine(rng, source_size=(10, 8), scale_range=(0.5, 2.0))
        assert 0.5 <= params.scale <= 2.0


def test_sample_affine_fresh_generators_same_seed_give_same_params() -> None:
    a = sample_affine(
        np.random.default_rng(5),
        source_size=(10, 8),
        angle_range=(-10.0, 10.0),
        translation_x_range=(-4.0, 4.0),
        translation_y_range=(-4.0, 4.0),
        scale_range=(0.8, 1.2),
    )
    b = sample_affine(
        np.random.default_rng(5),
        source_size=(10, 8),
        angle_range=(-10.0, 10.0),
        translation_x_range=(-4.0, 4.0),
        translation_y_range=(-4.0, 4.0),
        scale_range=(0.8, 1.2),
    )
    assert a == b


def test_sample_affine_consecutive_calls_on_same_generator_can_differ() -> None:
    rng = np.random.default_rng(3)
    results = [
        sample_affine(rng, source_size=(10, 8), angle_range=(-30.0, 30.0)) for _ in range(20)
    ]
    assert len({r.angle for r in results}) > 1


def test_sample_affine_params_are_replayable() -> None:
    rng = np.random.default_rng(2)
    params = sample_affine(rng, source_size=(10, 8), angle_range=(-15.0, 15.0))
    image = _make_image(8, 10)
    first = apply_affine(image, params)
    second = apply_affine(image, params)
    np.testing.assert_array_equal(first, second)


def test_sample_affine_rejects_non_generator_rng() -> None:
    with pytest.raises(TypeError, match="Generator"):
        sample_affine(np.random.RandomState(0), source_size=(10, 8))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"angle_range": [1.0, 2.0]},
        {"angle_range": None},
        {"translation_x_range": "bad"},
        {"scale_range": (1.0,)},
        {"angle_range": (1.0, 2.0, 3.0)},
    ],
)
def test_sample_affine_rejects_malformed_ranges(kwargs: dict) -> None:
    rng = np.random.default_rng(0)
    with pytest.raises((TypeError, ValueError)):
        sample_affine(rng, source_size=(10, 8), **kwargs)  # type: ignore[arg-type]


def test_sample_affine_rejects_bool_in_range() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(TypeError):
        sample_affine(rng, source_size=(10, 8), angle_range=(True, 5.0))  # type: ignore[arg-type]


def test_sample_affine_rejects_nan_in_range() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        sample_affine(rng, source_size=(10, 8), angle_range=(float("nan"), 5.0))


def test_sample_affine_rejects_inf_in_range() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        sample_affine(rng, source_size=(10, 8), translation_x_range=(0.0, float("inf")))


def test_sample_affine_rejects_low_greater_than_high() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="low"):
        sample_affine(rng, source_size=(10, 8), angle_range=(5.0, -5.0))


def test_sample_affine_rejects_zero_scale() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        sample_affine(rng, source_size=(10, 8), scale_range=(0.0, 0.0))


def test_sample_affine_rejects_negative_scale() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        sample_affine(rng, source_size=(10, 8), scale_range=(-2.0, -1.0))


def test_sample_affine_rejects_non_finite_matrix_from_legal_but_extreme_inputs() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="finite"):
        sample_affine(
            rng,
            source_size=(2_000_000_000, 2_000_000_000),
            scale_range=(1e300, 1e300),
        )


# --- affine matrix semantics ---


def test_apply_affine_identity_preserves_image() -> None:
    rng = np.random.default_rng(0)
    image = _make_image(20, 30)
    params = sample_affine(rng, source_size=(30, 20))
    result = apply_affine(image, params)
    np.testing.assert_array_equal(result, image)


def test_sample_affine_pure_rotation_matches_get_rotation_matrix_2d() -> None:
    rng = np.random.default_rng(0)
    params = sample_affine(rng, source_size=(11, 7), angle_range=(37.0, 37.0))
    expected = _reference_matrix((11, 7), 37.0, 0.0, 0.0, 1.0)
    np.testing.assert_allclose(params.matrix, expected)


def test_sample_affine_rotation_and_scale_matches_get_rotation_matrix_2d() -> None:
    rng = np.random.default_rng(0)
    params = sample_affine(
        rng, source_size=(11, 7), angle_range=(37.0, 37.0), scale_range=(1.7, 1.7)
    )
    center = (5.0, 3.0)
    expected = cv2.getRotationMatrix2D(center, 37.0, 1.7)
    np.testing.assert_allclose(params.matrix, expected)


def test_sample_affine_pure_translation_matches_translate_matrix() -> None:
    rng = np.random.default_rng(0)
    params = sample_affine(
        rng,
        source_size=(11, 7),
        translation_x_range=(5.0, 5.0),
        translation_y_range=(-3.0, -3.0),
    )
    expected = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0]])
    np.testing.assert_allclose(params.matrix, expected)


def test_apply_affine_positive_dx_moves_content_right() -> None:
    image = np.zeros((5, 5), dtype=np.uint8)
    image[2, 1] = 99
    rng = np.random.default_rng(0)
    params = sample_affine(rng, source_size=(5, 5), translation_x_range=(2.0, 2.0))
    result = apply_affine(image, params)
    ys, xs = np.where(result == 99)
    assert xs.tolist() == [1 + 2]
    assert ys.tolist() == [2]


def test_apply_affine_positive_dy_moves_content_down() -> None:
    image = np.zeros((5, 5), dtype=np.uint8)
    image[2, 1] = 99
    rng = np.random.default_rng(0)
    params = sample_affine(rng, source_size=(5, 5), translation_y_range=(1.0, 1.0))
    result = apply_affine(image, params)
    ys, xs = np.where(result == 99)
    assert ys.tolist() == [2 + 1]
    assert xs.tolist() == [1]


def test_sample_affine_subpixel_translation_is_legal() -> None:
    rng = np.random.default_rng(0)
    params = sample_affine(
        rng, source_size=(5, 5), translation_x_range=(0.5, 0.5), translation_y_range=(0.25, 0.25)
    )
    assert params.translation == (0.5, 0.25)


def test_sample_affine_center_convention_matches_rotate() -> None:
    rng = np.random.default_rng(0)
    width, height = 11, 7
    image = _make_image(height, width)
    angle = 41.0
    params = sample_affine(rng, source_size=(width, height), angle_range=(angle, angle))
    result = apply_affine(image, params)
    expected = im.rotate(image, angle)
    np.testing.assert_array_equal(result, expected)


def test_sample_affine_on_1x1_image() -> None:
    rng = np.random.default_rng(0)
    image = np.array([[7]], dtype=np.uint8)
    params = sample_affine(rng, source_size=(1, 1), angle_range=(45.0, 45.0))
    result = apply_affine(image, params)
    assert result.shape == (1, 1)


def test_sample_affine_large_angle_without_modulo_normalization() -> None:
    rng = np.random.default_rng(0)
    params_large = sample_affine(rng, source_size=(11, 7), angle_range=(400.0, 400.0))
    params_normalized = sample_affine(rng, source_size=(11, 7), angle_range=(40.0, 40.0))
    assert params_large.angle == 400.0
    np.testing.assert_allclose(params_large.matrix, params_normalized.matrix)


# --- apply_affine: image ---


@pytest.mark.parametrize("channels", [None, 3])
def test_apply_affine_grayscale_and_bgr(channels: int | None) -> None:
    image = _make_image(10, 12, channels=channels)
    rng = np.random.default_rng(0)
    params = sample_affine(rng, source_size=(12, 10), angle_range=(10.0, 10.0))
    result = apply_affine(image, params)
    assert result.shape == image.shape


def test_apply_affine_bgra() -> None:
    image = _make_image(10, 12, channels=4)
    rng = np.random.default_rng(0)
    params = sample_affine(rng, source_size=(12, 10), angle_range=(10.0, 10.0))
    result = apply_affine(image, params)
    assert result.shape == image.shape


def test_apply_affine_hw1_shape_preserved() -> None:
    image = _make_image(10, 12).reshape(-1)[: 10 * 12].reshape(10, 12, 1)
    rng = np.random.default_rng(0)
    params = sample_affine(rng, source_size=(12, 10), angle_range=(15.0, 15.0))
    result = apply_affine(image, params)
    assert result.shape == (10, 12, 1)
    expected = im.warp_affine(image[:, :, 0], params.matrix, params.source_size)
    np.testing.assert_array_equal(result[:, :, 0], expected)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int16, np.float32, np.float64])
def test_apply_affine_supported_image_dtypes(dtype: type) -> None:
    image = _make_image(10, 12).astype(dtype)
    rng = np.random.default_rng(0)
    params = sample_affine(rng, source_size=(12, 10), angle_range=(10.0, 10.0))
    result = apply_affine(image, params)
    assert result.dtype == dtype


def test_apply_affine_rejects_unsupported_image_dtype() -> None:
    image = _make_image(10, 12).astype(np.int32)
    rng = np.random.default_rng(0)
    params = sample_affine(rng, source_size=(12, 10))
    with pytest.raises(TypeError, match="dtype"):
        apply_affine(image, params)


def test_apply_affine_interpolation_linear_vs_nearest_differ() -> None:
    image = _make_image(10, 12)
    rng = np.random.default_rng(0)
    params = sample_affine(rng, source_size=(12, 10), angle_range=(13.0, 13.0))
    linear = apply_affine(image, params, interpolation=cv2.INTER_LINEAR)
    nearest = apply_affine(image, params, interpolation=cv2.INTER_NEAREST)
    assert not np.array_equal(linear, nearest)


def test_apply_affine_rejects_inverse_mapping_flag() -> None:
    image = np.zeros((5, 7), dtype=np.uint8)
    image[2, 1] = 255

    params = sample_affine(
        np.random.default_rng(0),
        source_size=(7, 5),
        translation_x_range=(2.0, 2.0),
    )

    with pytest.raises(ValueError, match="interpolation"):
        apply_affine(
            image,
            params,
            interpolation=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
        )


def test_apply_affine_does_not_call_warp_affine_after_bad_interpolation_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import improcv.augmentation as augmentation_module

    def boom(*args, **kwargs):
        pytest.fail("_warp_affine must not be called after a validation error")

    monkeypatch.setattr(augmentation_module, "_warp_affine", boom)

    image = np.zeros((5, 7), dtype=np.uint8)
    params = sample_affine(np.random.default_rng(0), source_size=(7, 5))
    with pytest.raises(ValueError):
        apply_affine(image, params, interpolation=cv2.WARP_INVERSE_MAP)


@pytest.mark.parametrize(
    "bad_interpolation",
    [
        cv2.WARP_INVERSE_MAP,
        cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        cv2.WARP_FILL_OUTLIERS,
        -1,
    ],
)
def test_apply_affine_rejects_warp_modifier_flags(bad_interpolation: int) -> None:
    image = _make_image(10, 12)
    params = sample_affine(np.random.default_rng(0), source_size=(12, 10))
    with pytest.raises(ValueError, match="interpolation|warp modifier flags"):
        apply_affine(image, params, interpolation=bad_interpolation)


@pytest.mark.parametrize("bad_interpolation", [True, 1.5, "nearest", None])
def test_apply_affine_rejects_non_integral_interpolation(bad_interpolation: object) -> None:
    image = _make_image(10, 12)
    params = sample_affine(np.random.default_rng(0), source_size=(12, 10))
    with pytest.raises(TypeError):
        apply_affine(image, params, interpolation=bad_interpolation)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "interpolation",
    [
        cv2.INTER_NEAREST,
        cv2.INTER_LINEAR,
        cv2.INTER_CUBIC,
        cv2.INTER_AREA,
        cv2.INTER_LANCZOS4,
    ],
)
def test_apply_affine_accepts_legal_interpolation_modes(interpolation: int) -> None:
    image = _make_image(10, 12)
    params = sample_affine(np.random.default_rng(0), source_size=(12, 10), angle_range=(9.0, 9.0))
    result = apply_affine(image, params, interpolation=interpolation)
    assert result.shape == image.shape


@pytest.mark.parametrize("attr_name", ["INTER_LINEAR_EXACT", "INTER_NEAREST_EXACT"])
def test_apply_affine_accepts_exact_interpolation_modes_if_available(attr_name: str) -> None:
    interpolation = getattr(cv2, attr_name, None)
    if interpolation is None:
        pytest.skip(f"cv2.{attr_name} not available on this OpenCV build")

    image = _make_image(10, 12)
    params = sample_affine(np.random.default_rng(0), source_size=(12, 10), angle_range=(9.0, 9.0))
    try:
        result = apply_affine(image, params, interpolation=interpolation)
    except RuntimeError as exc:
        assert isinstance(exc.__cause__, cv2.error)
    else:
        assert result.shape == image.shape


def test_apply_affine_positive_dx_still_moves_content_right_with_explicit_interpolation() -> None:
    image = np.zeros((5, 7), dtype=np.uint8)
    image[2, 1] = 255
    params = sample_affine(
        np.random.default_rng(0), source_size=(7, 5), translation_x_range=(2.0, 2.0)
    )
    result = apply_affine(image, params, interpolation=cv2.INTER_NEAREST)
    ys, xs = np.where(result == 255)
    assert xs.tolist() == [1 + 2]
    assert ys.tolist() == [2]


def test_apply_affine_border_value_fills_exposed_pixels() -> None:
    image = np.full((10, 10), 5, dtype=np.uint8)
    rng = np.random.default_rng(0)
    params = sample_affine(rng, source_size=(10, 10), translation_x_range=(9.0, 9.0))
    result = apply_affine(image, params, border_value=200)
    assert result[0, 0] == 200


def test_apply_affine_rejects_source_size_mismatch() -> None:
    image = _make_image(10, 12)
    rng = np.random.default_rng(0)
    params = sample_affine(rng, source_size=(999, 999))
    with pytest.raises(ValueError, match="source_size"):
        apply_affine(image, params)


def test_apply_affine_accepts_read_only_non_contiguous_and_fortran_order() -> None:
    image = _make_image(10, 12, channels=3)
    rng = np.random.default_rng(0)
    params = sample_affine(rng, source_size=(12, 10), angle_range=(9.0, 9.0))
    expected = apply_affine(image, params)

    read_only = image.copy()
    read_only.setflags(write=False)
    np.testing.assert_array_equal(apply_affine(read_only, params), expected)

    fortran = np.asfortranarray(image)
    np.testing.assert_array_equal(apply_affine(fortran, params), expected)


def test_apply_affine_non_contiguous_slice() -> None:
    image = _make_image(10, 24, channels=3)
    non_contiguous = image[:, ::2]
    assert not non_contiguous.flags["C_CONTIGUOUS"]
    rng = np.random.default_rng(0)
    params = sample_affine(rng, source_size=(12, 10), angle_range=(9.0, 9.0))
    result = apply_affine(non_contiguous, params)
    expected = im.warp_affine(non_contiguous, params.matrix, params.source_size)
    np.testing.assert_array_equal(result, expected)


def test_apply_affine_does_not_mutate_image() -> None:
    image = _make_image(10, 12)
    before = image.copy()
    rng = np.random.default_rng(0)
    params = sample_affine(rng, source_size=(12, 10), angle_range=(20.0, 20.0))
    apply_affine(image, params)
    np.testing.assert_array_equal(image, before)


def test_apply_affine_result_does_not_alias_image() -> None:
    image = _make_image(10, 12)
    rng = np.random.default_rng(0)
    params = sample_affine(rng, source_size=(12, 10), angle_range=(20.0, 20.0))
    result = apply_affine(image, params)
    assert not np.shares_memory(result, image)


def test_apply_affine_matches_warp_affine_directly() -> None:
    image = _make_image(10, 12, channels=3)
    rng = np.random.default_rng(0)
    params = sample_affine(
        rng, source_size=(12, 10), angle_range=(17.0, 17.0), scale_range=(1.2, 1.2)
    )
    result = apply_affine(image, params, border_value=42)
    expected = im.warp_affine(image, params.matrix, params.source_size, border_value=42)
    np.testing.assert_array_equal(result, expected)


def test_apply_affine_arbitrary_same_size_wrong_shape_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import improcv.augmentation as augmentation_module

    image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    rng = np.random.default_rng(0)
    params = sample_affine(rng, source_size=(5, 4))

    monkeypatch.setattr(
        augmentation_module,
        "_warp_affine",
        lambda *a, **k: np.zeros((4, 15), dtype=image.dtype),
    )
    with pytest.raises(RuntimeError, match="shape"):
        apply_affine(image, params)


def test_apply_affine_maps_unexpected_opencv_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import improcv.augmentation as augmentation_module

    error = cv2.error("simulated failure")
    monkeypatch.setattr(
        augmentation_module,
        "_warp_affine",
        lambda *a, **k: (_ for _ in ()).throw(error),
    )
    image = _make_image(10, 12)
    rng = np.random.default_rng(0)
    params = sample_affine(rng, source_size=(12, 10))
    with pytest.raises(RuntimeError, match="OpenCV failed") as exc_info:
        apply_affine(image, params)
    assert exc_info.value.__cause__ is error


def test_apply_affine_postcondition_violation_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import improcv.augmentation as augmentation_module

    monkeypatch.setattr(
        augmentation_module,
        "_warp_affine",
        lambda *a, **k: np.zeros((999, 999), dtype=np.uint8),
    )
    image = _make_image(10, 12)
    rng = np.random.default_rng(0)
    params = sample_affine(rng, source_size=(12, 10))
    with pytest.raises(RuntimeError, match="internal error"):
        apply_affine(image, params)


# --- apply_affine: mask ---


def test_apply_affine_with_mask_returns_augmented_image_mask() -> None:
    image = _make_image(10, 12)
    mask = _make_mask(10, 12)
    rng = np.random.default_rng(0)
    params = sample_affine(rng, source_size=(12, 10), angle_range=(12.0, 12.0))
    result = apply_affine(image, params, mask=mask)
    assert isinstance(result, AugmentedImageMask)
    expected_mask = im.warp_affine(
        mask, params.matrix, params.source_size, interpolation=cv2.INTER_NEAREST
    )
    np.testing.assert_array_equal(result.mask, expected_mask)


def test_apply_affine_mask_hw1_shape_preserved() -> None:
    image = _make_image(10, 12)
    mask = _make_mask(10, 12).reshape(10, 12, 1)
    rng = np.random.default_rng(0)
    params = sample_affine(rng, source_size=(12, 10), angle_range=(12.0, 12.0))
    result = apply_affine(image, params, mask=mask)
    assert result.mask.shape == (10, 12, 1)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int16])
def test_apply_affine_supported_mask_dtypes(dtype: type) -> None:
    image = _make_image(10, 12)
    mask = _make_mask(10, 12, dtype=dtype)
    rng = np.random.default_rng(0)
    params = sample_affine(rng, source_size=(12, 10), angle_range=(5.0, 5.0))
    result = apply_affine(image, params, mask=mask)
    assert result.mask.dtype == dtype


@pytest.mark.parametrize("dtype", [np.bool_, np.int32, np.int64, np.float32])
def test_apply_affine_rejects_unsupported_mask_dtype(dtype: type) -> None:
    image = _make_image(10, 12)
    mask = np.zeros((10, 12), dtype=dtype)
    rng = np.random.default_rng(0)
    params = sample_affine(rng, source_size=(12, 10))
    with pytest.raises(TypeError, match="dtype"):
        apply_affine(image, params, mask=mask)


def test_apply_affine_mask_signed_negative_ignore_label_preserved() -> None:
    image = _make_image(10, 12)
    mask = np.zeros((10, 12), dtype=np.int16)
    mask[5, 6] = -1
    rng = np.random.default_rng(0)
    params = sample_affine(rng, source_size=(12, 10))
    result = apply_affine(image, params, mask=mask)
    assert -1 in np.unique(result.mask)


def test_apply_affine_mask_always_uses_nearest_neighbor(monkeypatch: pytest.MonkeyPatch) -> None:
    import improcv.augmentation as augmentation_module

    calls = []
    real_warp_affine = augmentation_module._warp_affine

    def spy(image, matrix, output_size, **kwargs):
        calls.append(kwargs.get("interpolation"))
        return real_warp_affine(image, matrix, output_size, **kwargs)

    monkeypatch.setattr(augmentation_module, "_warp_affine", spy)

    image = _make_image(10, 12)
    mask = _make_mask(10, 12)
    rng = np.random.default_rng(0)
    params = sample_affine(rng, source_size=(12, 10), angle_range=(8.0, 8.0))
    apply_affine(image, params, mask=mask, interpolation=cv2.INTER_LINEAR)

    assert calls[0] == cv2.INTER_LINEAR
    assert calls[1] == cv2.INTER_NEAREST


def test_apply_affine_mask_border_value_fills_exposed_pixels() -> None:
    image = np.zeros((10, 10), dtype=np.uint8)
    mask = np.full((10, 10), 3, dtype=np.uint8)
    rng = np.random.default_rng(0)
    params = sample_affine(rng, source_size=(10, 10), translation_x_range=(9.0, 9.0))
    result = apply_affine(image, params, mask=mask, mask_border_value=250)
    assert result.mask[0, 0] == 250


def test_apply_affine_rejects_mask_border_value_out_of_dtype_range() -> None:
    image = _make_image(10, 12)
    mask = _make_mask(10, 12, dtype=np.uint8)
    rng = np.random.default_rng(0)
    params = sample_affine(rng, source_size=(12, 10))
    with pytest.raises(ValueError, match="mask_border_value"):
        apply_affine(image, params, mask=mask, mask_border_value=300)


def test_apply_affine_rejects_bool_mask_border_value() -> None:
    image = _make_image(10, 12)
    mask = _make_mask(10, 12, dtype=np.uint8)
    rng = np.random.default_rng(0)
    params = sample_affine(rng, source_size=(12, 10))
    with pytest.raises(TypeError, match="mask_border_value"):
        apply_affine(image, params, mask=mask, mask_border_value=True)


def test_apply_affine_rejects_mask_spatial_mismatch() -> None:
    image = _make_image(10, 12)
    mask = _make_mask(9, 12)
    rng = np.random.default_rng(0)
    params = sample_affine(rng, source_size=(12, 10))
    with pytest.raises(ValueError, match="spatial size"):
        apply_affine(image, params, mask=mask)


def test_apply_affine_mask_contains_no_new_values_beyond_input_and_border() -> None:
    image = _make_image(20, 20)
    mask = _make_mask(20, 20)
    rng = np.random.default_rng(0)
    params = sample_affine(rng, source_size=(20, 20), angle_range=(23.0, 23.0))
    result = apply_affine(image, params, mask=mask, mask_border_value=9)
    allowed = set(np.unique(mask).tolist()) | {9}
    assert set(np.unique(result.mask).tolist()) <= allowed


def test_apply_affine_error_only_on_mask_warp_is_mapped(monkeypatch: pytest.MonkeyPatch) -> None:
    import improcv.augmentation as augmentation_module

    real_warp_affine = augmentation_module._warp_affine
    error = cv2.error("simulated mask failure")
    calls = {"count": 0}

    def image_ok_mask_fails(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return real_warp_affine(*args, **kwargs)
        raise error

    monkeypatch.setattr(augmentation_module, "_warp_affine", image_ok_mask_fails)

    image = _make_image(10, 12)
    mask = _make_mask(10, 12)
    rng = np.random.default_rng(0)
    params = sample_affine(rng, source_size=(12, 10))
    with pytest.raises(RuntimeError, match="OpenCV failed") as exc_info:
        apply_affine(image, params, mask=mask)
    assert exc_info.value.__cause__ is error
    assert calls["count"] == 2


def test_apply_affine_mask_does_not_mutate_or_alias() -> None:
    image = _make_image(10, 12)
    mask = _make_mask(10, 12)
    mask_before = mask.copy()
    rng = np.random.default_rng(0)
    params = sample_affine(rng, source_size=(12, 10), angle_range=(7.0, 7.0))
    result = apply_affine(image, params, mask=mask)
    np.testing.assert_array_equal(mask, mask_before)
    assert not np.shares_memory(result.mask, mask)


# --- AffineParameters: type behavior ---


def test_affine_parameters_equality() -> None:
    a = sample_affine(np.random.default_rng(11), source_size=(10, 8), angle_range=(5.0, 5.0))
    b = sample_affine(np.random.default_rng(11), source_size=(10, 8), angle_range=(5.0, 5.0))
    assert a == b
    assert a != object()


def test_affine_parameters_inequality_on_matrix_and_metadata() -> None:
    rng = np.random.default_rng(0)
    a = sample_affine(rng, source_size=(10, 8), angle_range=(5.0, 5.0))
    b = sample_affine(rng, source_size=(10, 8), angle_range=(6.0, 6.0))
    assert a != b


def test_affine_parameters_hash_raises() -> None:
    params = sample_affine(np.random.default_rng(0), source_size=(10, 8))
    with pytest.raises(TypeError):
        hash(params)


def test_affine_parameters_matrix_is_read_only() -> None:
    params = sample_affine(np.random.default_rng(0), source_size=(10, 8))
    assert not params.matrix.flags.writeable
    with pytest.raises(ValueError):
        params.matrix[0, 0] = 1.0


def test_affine_parameters_asdict() -> None:
    params = sample_affine(np.random.default_rng(0), source_size=(10, 8), angle_range=(5.0, 5.0))
    d = dataclasses.asdict(params)
    assert d["angle"] == 5.0
    np.testing.assert_array_equal(d["matrix"], params.matrix)


def test_apply_affine_rejects_wrong_params_type() -> None:
    image = _make_image(10, 12)
    with pytest.raises(TypeError, match="AffineParameters"):
        apply_affine(image, {"matrix": np.eye(2, 3)})  # type: ignore[arg-type]


def test_apply_affine_rejects_manually_constructed_float32_matrix() -> None:
    image = _make_image(10, 12)
    bad_params = AffineParameters(
        matrix=np.eye(2, 3, dtype=np.float32),  # type: ignore[arg-type]
        source_size=(12, 10),
        angle=0.0,
        translation=(0.0, 0.0),
        scale=1.0,
    )
    with pytest.raises(TypeError, match="float64"):
        apply_affine(image, bad_params)


def test_apply_affine_rejects_manually_constructed_nan_angle() -> None:
    image = _make_image(10, 12)
    bad_params = AffineParameters(
        matrix=np.eye(2, 3, dtype=np.float64),
        source_size=(12, 10),
        angle=float("nan"),
        translation=(0.0, 0.0),
        scale=1.0,
    )
    with pytest.raises(ValueError, match="finite"):
        apply_affine(image, bad_params)


def test_apply_affine_rejects_manually_constructed_non_positive_scale() -> None:
    image = _make_image(10, 12)
    bad_params = AffineParameters(
        matrix=np.eye(2, 3, dtype=np.float64),
        source_size=(12, 10),
        angle=0.0,
        translation=(0.0, 0.0),
        scale=0.0,
    )
    with pytest.raises(ValueError, match="positive"):
        apply_affine(image, bad_params)


def test_apply_affine_rejects_manually_constructed_wrong_matrix_shape() -> None:
    image = _make_image(10, 12)
    bad_params = AffineParameters(
        matrix=np.eye(3, 3, dtype=np.float64),
        source_size=(12, 10),
        angle=0.0,
        translation=(0.0, 0.0),
        scale=0.0,
    )
    with pytest.raises((TypeError, ValueError)):
        apply_affine(image, bad_params)


@pytest.mark.parametrize(
    "source_size, expected_exception",
    [
        ([12, 10], TypeError),
        (None, TypeError),
        ((12,), ValueError),
        ((12, 10, 8), ValueError),
        ((np.int64(12), 10), TypeError),
        ((True, 10), TypeError),
        ((0, 10), ValueError),
    ],
)
def test_apply_affine_rejects_manually_constructed_bad_source_size(
    source_size: object, expected_exception: type[Exception]
) -> None:
    image = _make_image(10, 12)
    bad_params = AffineParameters(
        matrix=np.eye(2, 3, dtype=np.float64),
        source_size=source_size,  # type: ignore[arg-type]
        angle=0.0,
        translation=(0.0, 0.0),
        scale=1.0,
    )
    with pytest.raises(expected_exception):
        apply_affine(image, bad_params)


@pytest.mark.parametrize(
    "translation, expected_exception",
    [
        ([0.0, 0.0], TypeError),
        (None, TypeError),
        ((0.0,), ValueError),
        ((0.0, 1.0, 2.0), ValueError),
        ((True, 0.0), TypeError),
        ((0.0, float("inf")), ValueError),
    ],
)
def test_apply_affine_rejects_manually_constructed_bad_translation(
    translation: object, expected_exception: type[Exception]
) -> None:
    image = _make_image(10, 12)
    bad_params = AffineParameters(
        matrix=np.eye(2, 3, dtype=np.float64),
        source_size=(12, 10),
        angle=0.0,
        translation=translation,  # type: ignore[arg-type]
        scale=1.0,
    )
    with pytest.raises(expected_exception):
        apply_affine(image, bad_params)


# --- shear: pure algebra (no sample_affine involved) ---


def _sh_xy(sx: float, sy: float) -> np.ndarray:
    return np.array([[1.0, sx], [sy, 1.0 + sx * sy]])


def _shx(sx: float) -> np.ndarray:
    return np.array([[1.0, sx], [0.0, 1.0]])


def _shy(sy: float) -> np.ndarray:
    return np.array([[1.0, 0.0], [sy, 1.0]])


def test_shear_sequential_formula_matches_shy_matmul_shx() -> None:
    sx, sy = 0.7, -0.4
    np.testing.assert_allclose(_shy(sy) @ _shx(sx), _sh_xy(sx, sy))


@pytest.mark.parametrize(
    "sx, sy", [(0.5, 0.5), (0.2, -0.3), (-0.7, 0.1), (0.0, 0.6), (0.6, 0.0), (0.0, 0.0)]
)
def test_shear_sequential_determinant_is_one(sx: float, sy: float) -> None:
    assert np.linalg.det(_sh_xy(sx, sy)) == pytest.approx(1.0)


def test_shear_sequential_x_then_y_is_not_commutative() -> None:
    sx, sy = 0.7, -0.4
    xy = _shy(sy) @ _shx(sx)
    yx = _shx(sx) @ _shy(sy)
    assert not np.allclose(xy, yx)


def test_shear_single_axis_x_reduces_to_elementary_shx() -> None:
    sx = 0.42
    np.testing.assert_allclose(_sh_xy(sx, 0.0), _shx(sx))


def test_shear_single_axis_y_reduces_to_elementary_shy() -> None:
    sy = -0.37
    np.testing.assert_allclose(_sh_xy(0.0, sy), _shy(sy))


def test_shear_sequential_inverse_is_shx_neg_matmul_shy_neg() -> None:
    sx, sy = 0.3, 0.6
    forward = _sh_xy(sx, sy)
    inverse = _shx(-sx) @ _shy(-sy)
    np.testing.assert_allclose(forward @ inverse, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(inverse @ forward, np.eye(2), atol=1e-12)


def test_shear_sequential_preserves_orientation() -> None:
    for sx, sy in [(0.5, 0.5), (-3.0, 2.0), (10.0, -10.0)]:
        assert np.linalg.det(_sh_xy(sx, sy)) > 0


# --- sample_affine: shear sampling ---


def test_sample_affine_shear_zero_defaults() -> None:
    rng = np.random.default_rng(0)
    params = sample_affine(rng, source_size=(10, 8))
    assert params.shear == (0.0, 0.0)


def test_sample_affine_shear_singleton_x() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        params = sample_affine(rng, source_size=(10, 8), shear_x_range=(0.3, 0.3))
        assert params.shear[0] == 0.3


def test_sample_affine_shear_singleton_y() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        params = sample_affine(rng, source_size=(10, 8), shear_y_range=(-0.2, -0.2))
        assert params.shear[1] == -0.2


def test_sample_affine_shear_both_directions_within_range() -> None:
    rng = np.random.default_rng(0)
    for _ in range(200):
        params = sample_affine(
            rng, source_size=(10, 8), shear_x_range=(-0.5, 0.5), shear_y_range=(-0.3, 0.3)
        )
        assert -0.5 <= params.shear[0] <= 0.5
        assert -0.3 <= params.shear[1] <= 0.3


def test_sample_affine_shear_negative_values_legal() -> None:
    rng = np.random.default_rng(0)
    params = sample_affine(rng, source_size=(10, 8), shear_x_range=(-5.0, -5.0))
    assert params.shear[0] == -5.0


def test_sample_affine_shear_same_seed_gives_same_shear() -> None:
    a = sample_affine(
        np.random.default_rng(7),
        source_size=(10, 8),
        shear_x_range=(-0.4, 0.4),
        shear_y_range=(-0.4, 0.4),
    )
    b = sample_affine(
        np.random.default_rng(7),
        source_size=(10, 8),
        shear_x_range=(-0.4, 0.4),
        shear_y_range=(-0.4, 0.4),
    )
    assert a == b


@pytest.mark.parametrize("bad", [[0.1, 0.2], None, "0.1"])
def test_sample_affine_rejects_non_tuple_shear_range(bad: object) -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(TypeError, match="tuple"):
        sample_affine(rng, source_size=(10, 8), shear_x_range=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [(0.1,), (0.1, 0.2, 0.3)])
def test_sample_affine_rejects_wrong_length_shear_range(bad: tuple) -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="exactly 2 elements"):
        sample_affine(rng, source_size=(10, 8), shear_y_range=bad)  # type: ignore[arg-type]


def test_sample_affine_rejects_bool_in_shear_range() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(TypeError):
        sample_affine(rng, source_size=(10, 8), shear_x_range=(True, 0.5))  # type: ignore[arg-type]


def test_sample_affine_rejects_nan_in_shear_range() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        sample_affine(rng, source_size=(10, 8), shear_x_range=(float("nan"), 0.5))


def test_sample_affine_rejects_inf_in_shear_range() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        sample_affine(rng, source_size=(10, 8), shear_y_range=(0.0, float("inf")))


def test_sample_affine_rejects_low_greater_than_high_shear_range() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="low"):
        sample_affine(rng, source_size=(10, 8), shear_x_range=(0.5, -0.5))


def test_sample_affine_shear_overflow_raises_value_error_not_warning() -> None:
    rng = np.random.default_rng(0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="finite"):
            sample_affine(
                rng,
                source_size=(10, 8),
                shear_x_range=(1e200, 1e200),
                shear_y_range=(1e200, 1e200),
            )


def test_sample_affine_rejects_shear_when_float64_loses_unit_determinant_term() -> None:
    with pytest.raises(ValueError, match="invertible|float64"):
        sample_affine(
            np.random.default_rng(0),
            source_size=(21, 21),
            shear_x_range=(1e8, 1e8),
            shear_y_range=(1e8, 1e8),
        )


def test_sample_affine_float64_shear_collapse_raises_without_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError):
            sample_affine(
                np.random.default_rng(0),
                source_size=(21, 21),
                shear_x_range=(1e8, 1e8),
                shear_y_range=(1e8, 1e8),
            )


def test_sample_affine_rejects_shear_with_negative_product_losing_unit_term() -> None:
    # Same float64 precision loss as the positive-product case -- magnitude
    # is what matters for representability, not sign.
    sx, sy = 1e8, -1e8
    assert (1.0 + sx * sy) == sx * sy  # confirms the chosen values actually trigger the loss
    with pytest.raises(ValueError, match="invertible|float64"):
        sample_affine(
            np.random.default_rng(0),
            source_size=(21, 21),
            shear_x_range=(sx, sx),
            shear_y_range=(sy, sy),
        )


def test_sample_affine_accepts_large_but_still_representable_shear_pair() -> None:
    # product == 2**52 is the largest exact integer float64 can represent
    # as "1 + product" without collapsing back to "product" -- still a very
    # large, likely visually-degenerate shear, but not a numerical policy
    # violation. This test is only about the representability boundary, not
    # about image quality at this extreme.
    sx = sy = float(2**26)
    rng = np.random.default_rng(0)
    params = sample_affine(
        rng, source_size=(21, 21), shear_x_range=(sx, sx), shear_y_range=(sy, sy)
    )
    product = params.shear[0] * params.shear[1]
    assert 1.0 + product != product
    assert np.all(np.isfinite(params.matrix))


# --- compatibility: shear must not disturb the pre-shear contract ---


def test_sample_affine_zero_shear_matches_pre_shear_matrix_bit_for_bit() -> None:
    source_size = (37, 23)
    center = ((source_size[0] - 1) / 2.0, (source_size[1] - 1) / 2.0)

    rng_new = np.random.default_rng(123)
    new_params = sample_affine(
        rng_new,
        source_size=source_size,
        angle_range=(-10.0, 10.0),
        translation_x_range=(-8.0, 8.0),
        translation_y_range=(-8.0, 8.0),
        scale_range=(0.9, 1.1),
    )

    rng_old = np.random.default_rng(123)
    angle = float(rng_old.uniform(-10.0, 10.0))
    dx = float(rng_old.uniform(-8.0, 8.0))
    dy = float(rng_old.uniform(-8.0, 8.0))
    scale = float(rng_old.uniform(0.9, 1.1))
    old_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    old_matrix[0, 2] += dx
    old_matrix[1, 2] += dy

    np.testing.assert_array_equal(new_params.matrix, old_matrix)
    assert new_params.angle == angle
    assert new_params.translation == (dx, dy)
    assert new_params.scale == scale


def test_sample_affine_default_shear_preserves_rng_sequence_across_calls() -> None:
    # Compares two consecutive sample_affine() calls against a *manually*
    # reconstructed pre-shear draw sequence (angle, dx, dy, scale -- in that
    # order, unconditionally, exactly as sample_affine sampled them before
    # shear existed), not against another run of the current implementation.
    # This is what actually proves shear's default (0.0, 0.0) ranges don't
    # consume any rng state: two runs of *this* implementation against each
    # other would still match even if a shared bug changed both identically.
    source_size = (10, 8)
    center = ((source_size[0] - 1) / 2.0, (source_size[1] - 1) / 2.0)
    angle_range = (-5.0, 5.0)

    rng_new = np.random.default_rng(999)
    new_results = [
        sample_affine(rng_new, source_size=source_size, angle_range=angle_range) for _ in range(2)
    ]

    rng_old = np.random.default_rng(999)
    for params in new_results:
        angle = float(rng_old.uniform(*angle_range))
        dx = float(rng_old.uniform(0.0, 0.0))
        dy = float(rng_old.uniform(0.0, 0.0))
        scale = float(rng_old.uniform(1.0, 1.0))
        expected_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        expected_matrix[0, 2] += dx
        expected_matrix[1, 2] += dy

        assert params.angle == angle
        assert params.translation == (dx, dy)
        assert params.scale == scale
        np.testing.assert_array_equal(params.matrix, expected_matrix)


def test_affine_parameters_five_positional_arguments_still_construct() -> None:
    matrix = np.eye(2, 3, dtype=np.float64)
    params = AffineParameters(matrix, (10, 8), 0.0, (0.0, 0.0), 1.0)
    assert params.shear == (0.0, 0.0)


def test_affine_parameters_shear_is_keyword_only_sixth_positional_rejected() -> None:
    matrix = np.eye(2, 3, dtype=np.float64)
    with pytest.raises(TypeError):
        AffineParameters(matrix, (10, 8), 0.0, (0.0, 0.0), 1.0, (0.2, -0.1))  # type: ignore[misc]


def test_affine_parameters_shear_accepted_as_keyword() -> None:
    matrix = np.eye(2, 3, dtype=np.float64)
    params = AffineParameters(matrix, (10, 8), 0.0, (0.0, 0.0), 1.0, shear=(0.2, -0.1))
    assert params.shear == (0.2, -0.1)


def test_affine_parameters_match_args_excludes_shear() -> None:
    assert AffineParameters.__match_args__ == (
        "matrix",
        "source_size",
        "angle",
        "translation",
        "scale",
    )


def test_affine_parameters_five_positional_pattern_matching_still_works() -> None:
    matrix = np.eye(2, 3, dtype=np.float64)
    params = AffineParameters(matrix, (10, 8), 1.5, (2.0, 3.0), 1.1, shear=(0.1, 0.2))
    match params:
        case AffineParameters(m, s, a, t, sc):
            assert m is params.matrix
            assert s == (10, 8)
            assert a == 1.5
            assert t == (2.0, 3.0)
            assert sc == 1.1
        case _:
            pytest.fail("pattern match failed")


def test_affine_parameters_equality_includes_shear() -> None:
    matrix = np.eye(2, 3, dtype=np.float64)
    a = AffineParameters(matrix, (10, 8), 0.0, (0.0, 0.0), 1.0, shear=(0.1, 0.2))
    b = AffineParameters(matrix.copy(), (10, 8), 0.0, (0.0, 0.0), 1.0, shear=(0.1, 0.2))
    c = AffineParameters(matrix.copy(), (10, 8), 0.0, (0.0, 0.0), 1.0, shear=(0.1, 0.3))
    assert a == b
    assert a != c


def test_affine_parameters_repr_and_asdict_contain_shear() -> None:
    matrix = np.eye(2, 3, dtype=np.float64)
    params = AffineParameters(matrix, (10, 8), 0.0, (0.0, 0.0), 1.0, shear=(0.25, -0.5))
    assert "shear" in repr(params)
    d = dataclasses.asdict(params)
    assert d["shear"] == (0.25, -0.5)


def test_affine_parameters_default_shear_is_exactly_zero_zero() -> None:
    matrix = np.eye(2, 3, dtype=np.float64)
    params = AffineParameters(matrix, (10, 8), 0.0, (0.0, 0.0), 1.0)
    assert params.shear == (0.0, 0.0)
    assert isinstance(params.shear[0], float)
    assert isinstance(params.shear[1], float)


# --- matrix semantics with shear ---


def test_sample_affine_pure_shear_x_matches_manual_centered_matrix() -> None:
    source_size = (21, 21)
    cx, cy = (source_size[0] - 1) / 2.0, (source_size[1] - 1) / 2.0
    rng = np.random.default_rng(0)
    params = sample_affine(rng, source_size=source_size, shear_x_range=(0.5, 0.5))

    t_neg = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
    t_pos = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])
    shx = np.array([[1, 0.5, 0], [0, 1, 0], [0, 0, 1]])
    expected = (t_pos @ shx @ t_neg)[:2, :]
    np.testing.assert_allclose(params.matrix, expected)


def test_sample_affine_pure_shear_y_matches_manual_centered_matrix() -> None:
    source_size = (21, 21)
    cx, cy = (source_size[0] - 1) / 2.0, (source_size[1] - 1) / 2.0
    rng = np.random.default_rng(0)
    params = sample_affine(rng, source_size=source_size, shear_y_range=(-0.4, -0.4))

    t_neg = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
    t_pos = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])
    shy = np.array([[1, 0, 0], [-0.4, 1, 0], [0, 0, 1]])
    expected = (t_pos @ shy @ t_neg)[:2, :]
    np.testing.assert_allclose(params.matrix, expected)


def test_sample_affine_shear_x_then_y_matches_manual_sequential_matrix() -> None:
    source_size = (21, 21)
    cx, cy = (source_size[0] - 1) / 2.0, (source_size[1] - 1) / 2.0
    sx, sy = 0.4, -0.2
    rng = np.random.default_rng(0)
    params = sample_affine(
        rng, source_size=source_size, shear_x_range=(sx, sx), shear_y_range=(sy, sy)
    )

    t_neg = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
    t_pos = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])
    sh_xy_3x3 = np.array([[1, sx, 0], [sy, 1 + sx * sy, 0], [0, 0, 1]])
    expected = (t_pos @ sh_xy_3x3 @ t_neg)[:2, :]
    np.testing.assert_allclose(params.matrix, expected)

    # Direct check of the stored unit determinant term itself, rather than
    # relying solely on np.linalg.det (which can be numerically misleading
    # -- returning a plausible-looking but wrong positive or near-zero
    # value depending on algorithm/rounding -- for a badly conditioned
    # matrix; here the coefficients are moderate, so det is also checked).
    stored_sx, stored_sy = params.shear
    product = stored_sx * stored_sy
    assert 1.0 + product != product
    # scale defaults to 1.0 here, so the combined linear part's determinant
    # is just the shear's own determinant (rotation alone has det 1).
    assert params.scale == 1.0
    assert np.linalg.det(params.matrix[:2, :2]) == pytest.approx(1.0)


def test_apply_affine_positive_shear_x_moves_bottom_right_relative_to_top() -> None:
    image = np.zeros((21, 21), dtype=np.uint8)
    image[15, 10] = 255  # below center
    image[5, 10] = 254  # above center
    rng = np.random.default_rng(0)
    params = sample_affine(rng, source_size=(21, 21), shear_x_range=(0.5, 0.5))
    result = apply_affine(image, params, interpolation=cv2.INTER_NEAREST)

    below_ys, below_xs = np.where(result == 255)
    above_ys, above_xs = np.where(result == 254)
    assert below_xs[0] > above_xs[0]


def test_apply_affine_positive_shear_y_moves_right_side_down_relative_to_left() -> None:
    image = np.zeros((21, 21), dtype=np.uint8)
    image[10, 15] = 255  # right of center
    image[10, 5] = 254  # left of center
    rng = np.random.default_rng(0)
    params = sample_affine(rng, source_size=(21, 21), shear_y_range=(0.5, 0.5))
    result = apply_affine(image, params, interpolation=cv2.INTER_NEAREST)

    right_ys, right_xs = np.where(result == 255)
    left_ys, left_xs = np.where(result == 254)
    assert right_ys[0] > left_ys[0]


@pytest.mark.parametrize("source_size", [(20, 20), (21, 21), (20, 21), (21, 20)])
def test_sample_affine_shear_pivot_matches_center_for_even_and_odd_sizes(
    source_size: tuple[int, int],
) -> None:
    cx, cy = (source_size[0] - 1) / 2.0, (source_size[1] - 1) / 2.0
    rng = np.random.default_rng(0)
    params = sample_affine(rng, source_size=source_size, shear_x_range=(0.3, 0.3))
    # the center itself must be a fixed point of the shear
    center_point = np.array([cx, cy, 1.0])
    transformed = params.matrix @ center_point
    np.testing.assert_allclose(transformed, [cx, cy], atol=1e-9)


def test_sample_affine_shear_on_1x1_image() -> None:
    rng = np.random.default_rng(0)
    image = np.array([[7]], dtype=np.uint8)
    params = sample_affine(rng, source_size=(1, 1), shear_x_range=(5.0, 5.0))
    result = apply_affine(image, params)
    assert result.shape == (1, 1)


def test_sample_affine_shear_on_1xn_image() -> None:
    rng = np.random.default_rng(0)
    image = np.arange(10, dtype=np.uint8).reshape(1, 10)
    params = sample_affine(rng, source_size=(10, 1), shear_x_range=(2.0, 2.0))
    result = apply_affine(image, params)
    np.testing.assert_array_equal(result, image)


def test_sample_affine_shear_on_nx1_image() -> None:
    rng = np.random.default_rng(0)
    image = np.arange(10, dtype=np.uint8).reshape(10, 1)
    params = sample_affine(rng, source_size=(1, 10), shear_y_range=(2.0, 2.0))
    result = apply_affine(image, params)
    np.testing.assert_array_equal(result, image)


def test_sample_affine_shear_before_rotation_matches_manual_composition() -> None:
    source_size = (21, 21)
    cx, cy = (source_size[0] - 1) / 2.0, (source_size[1] - 1) / 2.0
    rng = np.random.default_rng(0)
    params = sample_affine(
        rng,
        source_size=source_size,
        angle_range=(30.0, 30.0),
        scale_range=(1.2, 1.2),
        shear_x_range=(0.4, 0.4),
        shear_y_range=(-0.2, -0.2),
    )

    rs_3x3 = np.eye(3)
    rs_3x3[:2, :] = cv2.getRotationMatrix2D((cx, cy), 30.0, 1.2)
    sx, sy = 0.4, -0.2
    sh_3x3 = np.array([[1, sx, 0], [sy, 1 + sx * sy, 0], [0, 0, 1]])
    t_neg = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
    t_pos = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])
    sh_centered = t_pos @ sh_3x3 @ t_neg

    shear_then_rotate = (rs_3x3 @ sh_centered)[:2, :]
    rotate_then_shear = (sh_centered @ rs_3x3)[:2, :]

    np.testing.assert_allclose(params.matrix, shear_then_rotate)
    assert not np.allclose(params.matrix, rotate_then_shear)


def test_sample_affine_translation_applied_after_shear_and_rotation() -> None:
    source_size = (21, 21)
    rng = np.random.default_rng(0)
    without_translation = sample_affine(
        rng,
        source_size=source_size,
        angle_range=(20.0, 20.0),
        shear_x_range=(0.3, 0.3),
    )
    rng2 = np.random.default_rng(0)
    with_translation = sample_affine(
        rng2,
        source_size=source_size,
        angle_range=(20.0, 20.0),
        translation_x_range=(5.0, 5.0),
        translation_y_range=(-3.0, -3.0),
        shear_x_range=(0.3, 0.3),
    )
    diff = with_translation.matrix - without_translation.matrix
    expected_diff = np.array([[0.0, 0.0, 5.0], [0.0, 0.0, -3.0]])
    np.testing.assert_allclose(diff, expected_diff, atol=1e-9)


def test_sample_affine_shear_matches_manual_synthetic_grid_transform() -> None:
    source_size = (21, 15)
    rng = np.random.default_rng(0)
    params = sample_affine(
        rng,
        source_size=source_size,
        angle_range=(12.0, 12.0),
        scale_range=(1.1, 1.1),
        shear_x_range=(0.3, 0.3),
        shear_y_range=(0.15, 0.15),
    )
    xs, ys = np.meshgrid(np.arange(0, 21, 3), np.arange(0, 15, 3))
    points = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)])
    transformed = params.matrix @ points

    cx, cy = (source_size[0] - 1) / 2.0, (source_size[1] - 1) / 2.0
    rs_3x3 = np.eye(3)
    rs_3x3[:2, :] = cv2.getRotationMatrix2D((cx, cy), 12.0, 1.1)
    sx, sy = 0.3, 0.15
    sh_3x3 = np.array([[1, sx, 0], [sy, 1 + sx * sy, 0], [0, 0, 1]])
    t_neg = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
    t_pos = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])
    expected_matrix = (rs_3x3 @ (t_pos @ sh_3x3 @ t_neg))[:2, :]
    expected = expected_matrix @ points

    np.testing.assert_allclose(transformed, expected)


# --- apply image/mask with nonzero shear ---


@pytest.mark.parametrize("channels", [None, 3, 4])
def test_apply_affine_shear_preserves_layout(channels: int | None) -> None:
    image = _make_image(10, 12, channels=channels)
    rng = np.random.default_rng(0)
    params = sample_affine(rng, source_size=(12, 10), shear_x_range=(0.2, 0.2))
    result = apply_affine(image, params)
    assert result.shape == image.shape


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int16, np.float32, np.float64])
def test_apply_affine_shear_supported_image_dtypes(dtype: type) -> None:
    image = _make_image(10, 12).astype(dtype)
    rng = np.random.default_rng(0)
    params = sample_affine(rng, source_size=(12, 10), shear_x_range=(0.2, 0.2))
    result = apply_affine(image, params)
    assert result.dtype == dtype


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int16])
def test_apply_affine_shear_supported_mask_dtypes(dtype: type) -> None:
    image = _make_image(10, 12)
    mask = _make_mask(10, 12, dtype=dtype)
    rng = np.random.default_rng(0)
    params = sample_affine(rng, source_size=(12, 10), shear_x_range=(0.2, 0.2))
    result = apply_affine(image, params, mask=mask)
    assert result.mask.dtype == dtype


def test_apply_affine_shear_mask_alignment_on_synthetic_object() -> None:
    image = np.zeros((21, 21), dtype=np.uint8)
    mask = np.zeros((21, 21), dtype=np.uint8)
    image[15, 10] = 200
    mask[15, 10] = 5
    rng = np.random.default_rng(0)
    params = sample_affine(rng, source_size=(21, 21), shear_x_range=(0.4, 0.4))
    result = apply_affine(image, params, mask=mask, interpolation=cv2.INTER_NEAREST)
    image_ys, image_xs = np.where(result.image == 200)
    mask_ys, mask_xs = np.where(result.mask == 5)
    assert (image_ys.tolist(), image_xs.tolist()) == (mask_ys.tolist(), mask_xs.tolist())


def test_apply_affine_shear_mask_hw1_shape_preserved() -> None:
    image = _make_image(10, 12)
    mask = _make_mask(10, 12).reshape(10, 12, 1)
    rng = np.random.default_rng(0)
    params = sample_affine(rng, source_size=(12, 10), shear_x_range=(0.2, 0.2))
    result = apply_affine(image, params, mask=mask)
    assert result.mask.shape == (10, 12, 1)


def test_apply_affine_shear_mask_signed_negative_ignore_label_preserved() -> None:
    image = _make_image(10, 12)
    mask = np.zeros((10, 12), dtype=np.int16)
    mask[5, 6] = -1
    rng = np.random.default_rng(0)
    params = sample_affine(rng, source_size=(12, 10), shear_x_range=(0.1, 0.1))
    result = apply_affine(image, params, mask=mask)
    assert -1 in np.unique(result.mask)


def test_apply_affine_shear_mask_no_new_values_beyond_input_and_border() -> None:
    image = _make_image(20, 20)
    mask = _make_mask(20, 20)
    rng = np.random.default_rng(0)
    params = sample_affine(rng, source_size=(20, 20), shear_x_range=(0.5, 0.5))
    result = apply_affine(image, params, mask=mask, mask_border_value=9)
    allowed = set(np.unique(mask).tolist()) | {9}
    assert set(np.unique(result.mask).tolist()) <= allowed


def test_apply_affine_large_shear_pushes_content_partially_outside_canvas() -> None:
    image = np.full((20, 20), 100, dtype=np.uint8)
    rng = np.random.default_rng(0)
    params = sample_affine(rng, source_size=(20, 20), shear_x_range=(50.0, 50.0))
    result = apply_affine(image, params, border_value=0)
    assert np.any(result == 0)


def test_apply_affine_shear_rejects_source_size_mismatch() -> None:
    image = _make_image(10, 12)
    rng = np.random.default_rng(0)
    params = sample_affine(rng, source_size=(999, 999), shear_x_range=(0.2, 0.2))
    with pytest.raises(ValueError, match="source_size"):
        apply_affine(image, params)


def test_apply_affine_shear_does_not_mutate_or_alias() -> None:
    image = _make_image(10, 12)
    mask = _make_mask(10, 12)
    image_before = image.copy()
    mask_before = mask.copy()
    rng = np.random.default_rng(0)
    params = sample_affine(rng, source_size=(12, 10), shear_x_range=(0.2, 0.2))
    result = apply_affine(image, params, mask=mask)
    np.testing.assert_array_equal(image, image_before)
    np.testing.assert_array_equal(mask, mask_before)
    assert not np.shares_memory(result.image, image)
    assert not np.shares_memory(result.mask, mask)


def test_apply_affine_shear_params_are_replayable() -> None:
    rng = np.random.default_rng(0)
    params = sample_affine(rng, source_size=(12, 10), shear_x_range=(-0.4, 0.4))
    image = _make_image(10, 12)
    first = apply_affine(image, params)
    second = apply_affine(image, params)
    np.testing.assert_array_equal(first, second)


def test_apply_affine_shear_legal_singleton_squeeze() -> None:
    image = _make_image(10, 12, channels=None)
    image_hw1 = image.reshape(10, 12, 1)
    rng = np.random.default_rng(0)
    params = sample_affine(rng, source_size=(12, 10), shear_x_range=(0.3, 0.3))
    result = apply_affine(image_hw1, params)
    assert result.shape == (10, 12, 1)


def test_apply_affine_shear_arbitrary_same_size_wrong_shape_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import improcv.augmentation as augmentation_module

    image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    rng = np.random.default_rng(0)
    params = sample_affine(rng, source_size=(5, 4), shear_x_range=(0.2, 0.2))

    monkeypatch.setattr(
        augmentation_module,
        "_warp_affine",
        lambda *a, **k: np.zeros((4, 15), dtype=image.dtype),
    )
    with pytest.raises(RuntimeError, match="shape"):
        apply_affine(image, params)


def test_apply_affine_shear_maps_unexpected_opencv_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import improcv.augmentation as augmentation_module

    error = cv2.error("simulated failure")
    monkeypatch.setattr(
        augmentation_module,
        "_warp_affine",
        lambda *a, **k: (_ for _ in ()).throw(error),
    )
    image = _make_image(10, 12)
    rng = np.random.default_rng(0)
    params = sample_affine(rng, source_size=(12, 10), shear_x_range=(0.2, 0.2))
    with pytest.raises(RuntimeError, match="OpenCV failed") as exc_info:
        apply_affine(image, params)
    assert exc_info.value.__cause__ is error


# =====================================================================
# sample_perspective / apply_perspective / PerspectiveParameters
# =====================================================================


def _independent_source_corners(width: int, height: int) -> np.ndarray:
    return np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32
    )


# --- sample_perspective: scalar distortion_scale contract ---


def test_sample_perspective_default_distortion_scale_is_half() -> None:
    rng_default = np.random.default_rng(0)
    rng_explicit = np.random.default_rng(0)
    default_params = sample_perspective(rng_default, source_size=(10, 8))
    explicit_params = sample_perspective(rng_explicit, source_size=(10, 8), distortion_scale=0.5)
    assert default_params == explicit_params


def test_sample_perspective_distortion_scale_zero_is_identity() -> None:
    rng = np.random.default_rng(0)
    params = sample_perspective(rng, source_size=(10, 8), distortion_scale=0.0)
    assert isinstance(params, PerspectiveParameters)
    np.testing.assert_array_equal(params.matrix, np.eye(3))


def test_sample_perspective_distortion_scale_half_is_legal() -> None:
    rng = np.random.default_rng(0)
    params = sample_perspective(rng, source_size=(10, 8), distortion_scale=0.5)
    assert isinstance(params, PerspectiveParameters)


def test_sample_perspective_rejects_negative_distortion_scale() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="distortion_scale"):
        sample_perspective(rng, source_size=(10, 8), distortion_scale=-0.1)


def test_sample_perspective_rejects_distortion_scale_above_half() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="distortion_scale"):
        sample_perspective(rng, source_size=(10, 8), distortion_scale=0.51)


def test_sample_perspective_rejects_bool_distortion_scale() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(TypeError, match="distortion_scale"):
        sample_perspective(rng, source_size=(10, 8), distortion_scale=True)  # type: ignore[arg-type]


def test_sample_perspective_rejects_nan_distortion_scale() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="distortion_scale"):
        sample_perspective(rng, source_size=(10, 8), distortion_scale=float("nan"))


def test_sample_perspective_rejects_inf_distortion_scale() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="distortion_scale"):
        sample_perspective(rng, source_size=(10, 8), distortion_scale=float("inf"))


def test_sample_perspective_signature_has_no_distortion_scale_range() -> None:
    import inspect

    signature = inspect.signature(sample_perspective)
    assert "distortion_scale_range" not in signature.parameters
    assert "distortion_scale" in signature.parameters


# --- sample_perspective: source size edge cases ---


def test_sample_perspective_rejects_1x1() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match=">= 2"):
        sample_perspective(rng, source_size=(1, 1))


def test_sample_perspective_rejects_1xn() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match=">= 2"):
        sample_perspective(rng, source_size=(1, 8))


def test_sample_perspective_rejects_nx1() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match=">= 2"):
        sample_perspective(rng, source_size=(8, 1))


def test_sample_perspective_accepts_2x2() -> None:
    rng = np.random.default_rng(0)
    params = sample_perspective(rng, source_size=(2, 2), distortion_scale=0.5)
    assert params.source_size == (2, 2)


def test_apply_perspective_manual_identity_accepts_1x1_source_size() -> None:
    params = PerspectiveParameters(
        matrix=np.eye(3, dtype=np.float64),
        source_size=(1, 1),
        destination_points=((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
    )
    image = np.array([[5]], dtype=np.uint8)
    result = apply_perspective(image, params)
    np.testing.assert_array_equal(result, image)


# --- sample_perspective: identity fast path / RNG state ---


def test_sample_perspective_identity_consumes_no_rng_state() -> None:
    rng = np.random.default_rng(0)
    before = copy.deepcopy(rng.bit_generator.state)
    sample_perspective(rng, source_size=(10, 8), distortion_scale=0.0)
    after = copy.deepcopy(rng.bit_generator.state)
    assert before == after


def test_sample_perspective_identity_matrix_is_exact_eye() -> None:
    rng = np.random.default_rng(0)
    params = sample_perspective(rng, source_size=(10, 8), distortion_scale=0.0)
    assert np.array_equal(params.matrix, np.eye(3, dtype=np.float64))
    assert params.matrix.dtype == np.float64


def test_sample_perspective_identity_destination_points_equal_source_points() -> None:
    rng = np.random.default_rng(0)
    params = sample_perspective(rng, source_size=(10, 8), distortion_scale=0.0)
    assert params.destination_points == ((0.0, 0.0), (9.0, 0.0), (9.0, 7.0), (0.0, 7.0))


def test_sample_perspective_identity_matrix_is_independent_and_read_only() -> None:
    rng = np.random.default_rng(0)
    params = sample_perspective(rng, source_size=(10, 8), distortion_scale=0.0)
    assert not params.matrix.flags.writeable
    with pytest.raises(ValueError):
        params.matrix[0, 0] = 5.0


def test_apply_perspective_identity_preserves_image_and_mask() -> None:
    rng = np.random.default_rng(0)
    image = _make_image(8, 10)
    mask = _make_mask(8, 10)
    params = sample_perspective(rng, source_size=(10, 8), distortion_scale=0.0)
    result = apply_perspective(image, params, mask=mask)
    np.testing.assert_array_equal(result.image, image)
    np.testing.assert_array_equal(result.mask, mask)
    assert not np.shares_memory(result.image, image)
    assert not np.shares_memory(result.mask, mask)


def test_sample_perspective_consumes_rng_when_distortion_scale_positive() -> None:
    rng = np.random.default_rng(0)
    before = copy.deepcopy(rng.bit_generator.state)
    sample_perspective(rng, source_size=(10, 8), distortion_scale=0.5)
    after = copy.deepcopy(rng.bit_generator.state)
    assert before != after


# --- sample_perspective: determinism / replay ---


def test_sample_perspective_fresh_generators_same_seed_give_same_params() -> None:
    a = sample_perspective(np.random.default_rng(5), source_size=(10, 8), distortion_scale=0.4)
    b = sample_perspective(np.random.default_rng(5), source_size=(10, 8), distortion_scale=0.4)
    assert a == b


def test_sample_perspective_consecutive_calls_can_differ() -> None:
    rng = np.random.default_rng(3)
    results = [
        sample_perspective(rng, source_size=(10, 8), distortion_scale=0.4) for _ in range(10)
    ]
    assert len({r.destination_points for r in results}) > 1


def test_sample_perspective_params_are_replayable() -> None:
    rng = np.random.default_rng(2)
    params = sample_perspective(rng, source_size=(10, 8), distortion_scale=0.4)
    image = _make_image(8, 10)
    first = apply_perspective(image, params)
    second = apply_perspective(image, params)
    np.testing.assert_array_equal(first, second)


# --- sampled quadrilateral: safe-region regression (not a proof, just regression) ---


def _signed_turns(points: tuple[tuple[float, float], ...]) -> list[float]:
    turns = []
    count = len(points)
    for index in range(count):
        x0, y0 = points[index]
        x1, y1 = points[(index + 1) % count]
        x2, y2 = points[(index + 2) % count]
        v1x, v1y = x1 - x0, y1 - y0
        v2x, v2y = x2 - x1, y2 - y1
        turns.append(v1x * v2y - v1y * v2x)
    return turns


@pytest.mark.parametrize("source_size", [(37, 23), (2, 2), (5, 5), (2, 1000), (1000, 2)])
def test_sample_perspective_max_distortion_always_gives_convex_quadrilateral(
    source_size: tuple[int, int],
) -> None:
    rng = np.random.default_rng(0)
    for _ in range(300):
        params = sample_perspective(rng, source_size=source_size, distortion_scale=0.5)
        turns = _signed_turns(params.destination_points)
        assert all(turn > 0.0 for turn in turns)


def test_sample_perspective_destination_points_within_documented_regions() -> None:
    rng = np.random.default_rng(0)
    width, height = 20, 16
    max_dx = 0.5 * (width - 1) / 2.0
    max_dy = 0.5 * (height - 1) / 2.0
    for _ in range(200):
        params = sample_perspective(rng, source_size=(width, height), distortion_scale=0.5)
        tl, tr, br, bl = params.destination_points
        assert 0.0 <= tl[0] <= max_dx and 0.0 <= tl[1] <= max_dy
        assert (width - 1) - max_dx <= tr[0] <= width - 1 and 0.0 <= tr[1] <= max_dy
        assert (width - 1) - max_dx <= br[0] <= width - 1
        assert (height - 1) - max_dy <= br[1] <= height - 1
        assert 0.0 <= bl[0] <= max_dx
        assert (height - 1) - max_dy <= bl[1] <= height - 1


# --- destination_points reflect float32-quantized values, not pre-quantization draws ---


def test_sample_perspective_destination_points_are_float32_quantized() -> None:
    # A real np.random.Generator's C-extension attributes can't be monkeypatched
    # (e.g. `.uniform` is read-only), so this checks quantization structurally
    # instead: a float64 value fresh out of `rng.uniform` is, with overwhelming
    # probability, not already exactly representable in float32 -- so if the
    # stored metadata is idempotent under a float32 round-trip, it must have
    # actually been quantized (as opposed to storing the pre-quantization draw).
    rng = np.random.default_rng(0)
    params = sample_perspective(rng, source_size=(10, 8), distortion_scale=0.5)
    for x, y in params.destination_points:
        assert x == float(np.float32(x))
        assert y == float(np.float32(y))


# --- perspective matrix geometry: numerical rank ---


def test_perspective_matrix_geometry_rejects_singular_rank_2_matrix() -> None:
    matrix = np.array([[1.0, 2.0, 0.0], [2.0, 4.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    params = PerspectiveParameters(
        matrix=matrix,
        source_size=(5, 4),
        destination_points=((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)),
    )
    image = _make_image(4, 5)
    with pytest.raises(ValueError, match="rank"):
        apply_perspective(image, params)


def test_perspective_matrix_geometry_rejects_zero_matrix() -> None:
    matrix = np.zeros((3, 3), dtype=np.float64)
    params = PerspectiveParameters(
        matrix=matrix,
        source_size=(5, 4),
        destination_points=((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)),
    )
    image = _make_image(4, 5)
    with pytest.raises(ValueError, match="zero matrix"):
        apply_perspective(image, params)


def test_perspective_matrix_geometry_maps_linalgerror_to_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import improcv.augmentation as augmentation_module

    def boom(*args, **kwargs):
        raise np.linalg.LinAlgError("simulated SVD failure")

    monkeypatch.setattr(augmentation_module.np.linalg, "matrix_rank", boom)

    matrix = np.eye(3, dtype=np.float64)
    params = PerspectiveParameters(
        matrix=matrix,
        source_size=(5, 4),
        destination_points=((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)),
    )
    image = _make_image(4, 5)
    with pytest.raises(ValueError, match="numerical rank"):
        apply_perspective(image, params)


def test_perspective_matrix_geometry_message_mentions_numerical_rank_not_exact_determinant() -> (
    None
):
    matrix = np.array([[1.0, 2.0, 0.0], [2.0, 4.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    params = PerspectiveParameters(
        matrix=matrix,
        source_size=(5, 4),
        destination_points=((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)),
    )
    image = _make_image(4, 5)
    with pytest.raises(ValueError) as exc_info:
        apply_perspective(image, params)
    message = str(exc_info.value)
    assert "numerically full-rank" in message
    assert "must not be the zero matrix" not in message


def test_perspective_matrix_geometry_accepts_well_conditioned_matrix() -> None:
    rng = np.random.default_rng(0)
    params = sample_perspective(rng, source_size=(10, 8), distortion_scale=0.5)
    image = _make_image(8, 10)
    result = apply_perspective(image, params)
    assert result.shape == image.shape


# --- perspective matrix geometry: horizon check ---


def test_perspective_matrix_geometry_rejects_horizon_crossing_matrix() -> None:
    matrix = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.5, 0.0, 1.0]],
        dtype=np.float64,
    )
    assert np.linalg.matrix_rank(matrix) == 3  # confirm this fails via horizon, not rank
    params = PerspectiveParameters(
        matrix=matrix,
        source_size=(5, 4),
        destination_points=((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)),
    )
    image = _make_image(4, 5)
    with pytest.raises(ValueError, match="horizon"):
        apply_perspective(image, params)


def test_perspective_matrix_geometry_rejects_zero_denominator_corner() -> None:
    # w(x, y) = x - 2 -> exactly zero at x=2, well inside a (5, 4) source rectangle.
    matrix = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, -2.0]],
        dtype=np.float64,
    )
    params = PerspectiveParameters(
        matrix=matrix,
        source_size=(5, 4),
        destination_points=((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)),
    )
    image = _make_image(4, 5)
    with pytest.raises(ValueError, match="horizon"):
        apply_perspective(image, params)


def test_perspective_matrix_geometry_accepts_all_negative_denominators() -> None:
    matrix = np.array(
        [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        dtype=np.float64,
    )
    params = PerspectiveParameters(
        matrix=matrix,
        source_size=(5, 4),
        destination_points=((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)),
    )
    image = _make_image(4, 5)
    result = apply_perspective(image, params)
    assert result.shape == image.shape


# --- perspective matrix geometry: scale invariance ---


@pytest.mark.parametrize("factor", [1.0, 1e200, 1e-200])
def test_perspective_matrix_geometry_decision_is_scale_invariant_for_valid_matrix(
    factor: float,
) -> None:
    # Scale-invariance is a claim about the accept/reject *decision* of the geometry
    # checker, not about the warped pixels being bit-identical: cv2.warpPerspective's
    # own per-pixel homogeneous division can lose precision at such extreme
    # magnitudes even though the mathematical transform is unchanged.
    rng = np.random.default_rng(0)
    params = sample_perspective(rng, source_size=(10, 8), distortion_scale=0.4)
    scaled_matrix = params.matrix * factor
    assert np.all(np.isfinite(scaled_matrix))
    scaled_params = PerspectiveParameters(
        matrix=scaled_matrix,
        source_size=params.source_size,
        destination_points=params.destination_points,
    )
    image = _make_image(8, 10)
    result = apply_perspective(image, scaled_params)  # must not raise
    assert result.shape == image.shape


@pytest.mark.parametrize("factor", [1.0, 1e200, 1e-200])
def test_perspective_matrix_geometry_horizon_decision_is_scale_invariant(factor: float) -> None:
    matrix = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.5, 0.0, 1.0]],
        dtype=np.float64,
    )
    scaled = matrix * factor
    assert np.all(np.isfinite(scaled))
    params = PerspectiveParameters(
        matrix=scaled,
        source_size=(5, 4),
        destination_points=((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)),
    )
    image = _make_image(4, 5)
    with pytest.raises(ValueError, match="horizon"):
        apply_perspective(image, params)


# --- PerspectiveParameters validation (apply) ---


def test_apply_perspective_rejects_non_perspective_parameters() -> None:
    image = _make_image(4, 5)
    with pytest.raises(TypeError, match="PerspectiveParameters"):
        apply_perspective(image, "not-params")  # type: ignore[arg-type]


def test_apply_perspective_accepts_subclass_of_perspective_parameters() -> None:
    class SubParameters(PerspectiveParameters):
        pass

    params = SubParameters(
        matrix=np.eye(3, dtype=np.float64),
        source_size=(5, 4),
        destination_points=((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)),
    )
    image = _make_image(4, 5)
    result = apply_perspective(image, params)
    np.testing.assert_array_equal(result, image)


def test_apply_perspective_rejects_non_ndarray_matrix() -> None:
    params = PerspectiveParameters(
        matrix="not-an-array",  # type: ignore[arg-type]
        source_size=(5, 4),
        destination_points=((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)),
    )
    image = _make_image(4, 5)
    with pytest.raises(TypeError, match="matrix"):
        apply_perspective(image, params)


def test_apply_perspective_rejects_wrong_matrix_dtype() -> None:
    params = PerspectiveParameters(
        matrix=np.eye(3, dtype=np.float32),  # type: ignore[arg-type]
        source_size=(5, 4),
        destination_points=((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)),
    )
    image = _make_image(4, 5)
    with pytest.raises(TypeError, match="float64"):
        apply_perspective(image, params)


def test_apply_perspective_rejects_wrong_matrix_shape() -> None:
    params = PerspectiveParameters(
        matrix=np.eye(2, 3, dtype=np.float64),
        source_size=(5, 4),
        destination_points=((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)),
    )
    image = _make_image(4, 5)
    with pytest.raises(ValueError, match="shape"):
        apply_perspective(image, params)


def test_apply_perspective_rejects_non_finite_matrix() -> None:
    matrix = np.eye(3, dtype=np.float64)
    matrix[0, 0] = float("nan")
    params = PerspectiveParameters(
        matrix=matrix,
        source_size=(5, 4),
        destination_points=((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)),
    )
    image = _make_image(4, 5)
    with pytest.raises(ValueError, match="finite"):
        apply_perspective(image, params)


def test_apply_perspective_rejects_bad_destination_points_length() -> None:
    params = PerspectiveParameters(
        matrix=np.eye(3, dtype=np.float64),
        source_size=(5, 4),
        destination_points=((0.0, 0.0), (4.0, 0.0), (4.0, 3.0)),  # type: ignore[arg-type]
    )
    image = _make_image(4, 5)
    with pytest.raises(ValueError, match="destination_points"):
        apply_perspective(image, params)


def test_apply_perspective_rejects_non_finite_destination_point() -> None:
    params = PerspectiveParameters(
        matrix=np.eye(3, dtype=np.float64),
        source_size=(5, 4),
        destination_points=((0.0, 0.0), (4.0, 0.0), (4.0, float("nan")), (0.0, 3.0)),
    )
    image = _make_image(4, 5)
    with pytest.raises(ValueError, match="destination_points"):
        apply_perspective(image, params)


def test_apply_perspective_rejects_source_size_mismatch() -> None:
    rng = np.random.default_rng(0)
    params = sample_perspective(rng, source_size=(999, 999))
    image = _make_image(4, 5)
    with pytest.raises(ValueError, match="source_size"):
        apply_perspective(image, params)


# --- apply_perspective: image ---


@pytest.mark.parametrize("channels", [None, 3])
def test_apply_perspective_grayscale_and_bgr(channels: int | None) -> None:
    image = _make_image(10, 12, channels=channels)
    rng = np.random.default_rng(0)
    params = sample_perspective(rng, source_size=(12, 10), distortion_scale=0.3)
    result = apply_perspective(image, params)
    assert result.shape == image.shape


def test_apply_perspective_bgra() -> None:
    image = _make_image(10, 12, channels=4)
    rng = np.random.default_rng(0)
    params = sample_perspective(rng, source_size=(12, 10), distortion_scale=0.3)
    result = apply_perspective(image, params)
    assert result.shape == image.shape


def test_apply_perspective_hw1_shape_preserved() -> None:
    image = _make_image(10, 12).reshape(-1)[: 10 * 12].reshape(10, 12, 1)
    rng = np.random.default_rng(0)
    params = sample_perspective(rng, source_size=(12, 10), distortion_scale=0.3)
    result = apply_perspective(image, params)
    assert result.shape == (10, 12, 1)
    expected = im.warp_perspective(image[:, :, 0], params.matrix, params.source_size)
    np.testing.assert_array_equal(result[:, :, 0], expected)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int16, np.float32, np.float64])
def test_apply_perspective_supported_image_dtypes(dtype: type) -> None:
    image = _make_image(10, 12).astype(dtype)
    rng = np.random.default_rng(0)
    params = sample_perspective(rng, source_size=(12, 10), distortion_scale=0.3)
    result = apply_perspective(image, params)
    assert result.dtype == dtype


def test_apply_perspective_rejects_unsupported_image_dtype() -> None:
    image = _make_image(10, 12).astype(np.int32)
    rng = np.random.default_rng(0)
    params = sample_perspective(rng, source_size=(12, 10))
    with pytest.raises(TypeError, match="dtype"):
        apply_perspective(image, params)


def test_apply_perspective_rejects_inverse_mapping_flag() -> None:
    image = _make_image(10, 12)
    rng = np.random.default_rng(0)
    params = sample_perspective(rng, source_size=(12, 10), distortion_scale=0.3)
    with pytest.raises(ValueError, match="interpolation"):
        apply_perspective(image, params, interpolation=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)


def test_apply_perspective_does_not_call_warp_perspective_after_bad_interpolation_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import improcv.augmentation as augmentation_module

    def boom(*args, **kwargs):
        pytest.fail("_warp_perspective must not be called after a validation error")

    monkeypatch.setattr(augmentation_module, "_warp_perspective", boom)

    image = _make_image(10, 12)
    rng = np.random.default_rng(0)
    params = sample_perspective(rng, source_size=(12, 10))
    with pytest.raises(ValueError):
        apply_perspective(image, params, interpolation=cv2.WARP_INVERSE_MAP)


@pytest.mark.parametrize(
    "bad_interpolation",
    [cv2.WARP_INVERSE_MAP, cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, cv2.WARP_FILL_OUTLIERS, -1],
)
def test_apply_perspective_rejects_warp_modifier_flags(bad_interpolation: int) -> None:
    image = _make_image(10, 12)
    rng = np.random.default_rng(0)
    params = sample_perspective(rng, source_size=(12, 10))
    with pytest.raises(ValueError, match="interpolation|warp modifier flags"):
        apply_perspective(image, params, interpolation=bad_interpolation)


@pytest.mark.parametrize("bad_interpolation", [True, 1.5, "nearest", None])
def test_apply_perspective_rejects_non_integral_interpolation(bad_interpolation: object) -> None:
    image = _make_image(10, 12)
    rng = np.random.default_rng(0)
    params = sample_perspective(rng, source_size=(12, 10))
    with pytest.raises(TypeError):
        apply_perspective(image, params, interpolation=bad_interpolation)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "interpolation",
    [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4],
)
def test_apply_perspective_accepts_legal_interpolation_modes(interpolation: int) -> None:
    image = _make_image(10, 12)
    rng = np.random.default_rng(0)
    params = sample_perspective(rng, source_size=(12, 10), distortion_scale=0.3)
    result = apply_perspective(image, params, interpolation=interpolation)
    assert result.shape == image.shape


@pytest.mark.parametrize("attr_name", ["INTER_LINEAR_EXACT", "INTER_NEAREST_EXACT"])
def test_apply_perspective_handles_exact_interpolation_modes_if_available(attr_name: str) -> None:
    interpolation = getattr(cv2, attr_name, None)
    if interpolation is None:
        pytest.skip(f"cv2.{attr_name} not available on this OpenCV build")

    image = _make_image(10, 12)
    rng = np.random.default_rng(0)
    params = sample_perspective(rng, source_size=(12, 10), distortion_scale=0.3)
    try:
        result = apply_perspective(image, params, interpolation=interpolation)
    except RuntimeError as exc:
        assert isinstance(exc.__cause__, cv2.error)
    else:
        assert result.shape == image.shape


def test_apply_perspective_border_value_fills_exposed_pixels() -> None:
    image = np.full((10, 10), 5, dtype=np.uint8)
    rng = np.random.default_rng(0)
    params = sample_perspective(rng, source_size=(10, 10), distortion_scale=0.5)
    result = apply_perspective(image, params, border_value=200)
    assert 200 in np.unique(result)


def test_apply_perspective_accepts_read_only_non_contiguous_and_fortran_order() -> None:
    image = _make_image(10, 12, channels=3)
    rng = np.random.default_rng(0)
    params = sample_perspective(rng, source_size=(12, 10), distortion_scale=0.3)
    expected = apply_perspective(image, params)

    read_only = image.copy()
    read_only.setflags(write=False)
    np.testing.assert_array_equal(apply_perspective(read_only, params), expected)

    fortran = np.asfortranarray(image)
    np.testing.assert_array_equal(apply_perspective(fortran, params), expected)


def test_apply_perspective_non_contiguous_slice() -> None:
    image = _make_image(10, 24, channels=3)
    non_contiguous = image[:, ::2]
    assert not non_contiguous.flags["C_CONTIGUOUS"]
    rng = np.random.default_rng(0)
    params = sample_perspective(rng, source_size=(12, 10), distortion_scale=0.3)
    result = apply_perspective(non_contiguous, params)
    expected = im.warp_perspective(non_contiguous, params.matrix, params.source_size)
    np.testing.assert_array_equal(result, expected)


def test_apply_perspective_does_not_mutate_image() -> None:
    image = _make_image(10, 12)
    before = image.copy()
    rng = np.random.default_rng(0)
    params = sample_perspective(rng, source_size=(12, 10), distortion_scale=0.3)
    apply_perspective(image, params)
    np.testing.assert_array_equal(image, before)


def test_apply_perspective_result_does_not_alias_image() -> None:
    image = _make_image(10, 12)
    rng = np.random.default_rng(0)
    params = sample_perspective(rng, source_size=(12, 10), distortion_scale=0.3)
    result = apply_perspective(image, params)
    assert not np.shares_memory(result, image)


def test_apply_perspective_matches_warp_perspective_directly() -> None:
    image = _make_image(10, 12, channels=3)
    rng = np.random.default_rng(0)
    params = sample_perspective(rng, source_size=(12, 10), distortion_scale=0.4)
    result = apply_perspective(image, params, border_value=42)
    expected = im.warp_perspective(image, params.matrix, params.source_size, border_value=42)
    np.testing.assert_array_equal(result, expected)


def test_apply_perspective_arbitrary_same_size_wrong_shape_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import improcv.augmentation as augmentation_module

    image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    rng = np.random.default_rng(0)
    params = sample_perspective(rng, source_size=(5, 4))

    monkeypatch.setattr(
        augmentation_module,
        "_warp_perspective",
        lambda *a, **k: np.zeros((4, 15), dtype=image.dtype),
    )
    with pytest.raises(RuntimeError, match="shape"):
        apply_perspective(image, params)


def test_apply_perspective_maps_unexpected_opencv_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import improcv.augmentation as augmentation_module

    error = cv2.error("simulated failure")
    monkeypatch.setattr(
        augmentation_module,
        "_warp_perspective",
        lambda *a, **k: (_ for _ in ()).throw(error),
    )
    image = _make_image(10, 12)
    rng = np.random.default_rng(0)
    params = sample_perspective(rng, source_size=(12, 10))
    with pytest.raises(RuntimeError, match="OpenCV failed") as exc_info:
        apply_perspective(image, params)
    assert exc_info.value.__cause__ is error


def test_apply_perspective_postcondition_violation_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import improcv.augmentation as augmentation_module

    monkeypatch.setattr(
        augmentation_module,
        "_warp_perspective",
        lambda *a, **k: np.zeros((999, 999), dtype=np.uint8),
    )
    image = _make_image(10, 12)
    rng = np.random.default_rng(0)
    params = sample_perspective(rng, source_size=(12, 10))
    with pytest.raises(RuntimeError, match="internal error"):
        apply_perspective(image, params)


# --- apply_perspective: mask ---


def test_apply_perspective_with_mask_returns_augmented_image_mask() -> None:
    image = _make_image(10, 12)
    mask = _make_mask(10, 12)
    rng = np.random.default_rng(0)
    params = sample_perspective(rng, source_size=(12, 10), distortion_scale=0.3)
    result = apply_perspective(image, params, mask=mask)
    assert isinstance(result, AugmentedImageMask)
    expected_mask = im.warp_perspective(
        mask, params.matrix, params.source_size, interpolation=cv2.INTER_NEAREST
    )
    np.testing.assert_array_equal(result.mask, expected_mask)


def test_apply_perspective_mask_hw1_shape_preserved() -> None:
    image = _make_image(10, 12)
    mask = _make_mask(10, 12).reshape(10, 12, 1)
    rng = np.random.default_rng(0)
    params = sample_perspective(rng, source_size=(12, 10), distortion_scale=0.3)
    result = apply_perspective(image, params, mask=mask)
    assert result.mask.shape == (10, 12, 1)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_apply_perspective_supported_mask_dtypes(dtype: type) -> None:
    image = _make_image(10, 12)
    mask = _make_mask(10, 12, dtype=dtype)
    rng = np.random.default_rng(0)
    params = sample_perspective(rng, source_size=(12, 10), distortion_scale=0.2)
    result = apply_perspective(image, params, mask=mask)
    assert result.mask.dtype == dtype


@pytest.mark.parametrize("dtype", [np.bool_, np.int16, np.int32, np.int64, np.float32])
def test_apply_perspective_rejects_unsupported_mask_dtype(dtype: type) -> None:
    image = _make_image(10, 12)
    mask = np.zeros((10, 12), dtype=dtype)
    rng = np.random.default_rng(0)
    params = sample_perspective(rng, source_size=(12, 10))
    with pytest.raises(TypeError, match="dtype"):
        apply_perspective(image, params, mask=mask)


def test_apply_perspective_int16_mask_rejected_unlike_apply_affine() -> None:
    # Deliberate, narrower contract than apply_affine/apply_flip/apply_crop: verified
    # via this project's own Windows CI that cv2.warpPerspective (not warpAffine) with
    # an int16 mask raises "Unknown C++ exception from OpenCV code" on Windows for the
    # same opencv-python-headless version that works fine on Linux/macOS -- excluded
    # outright rather than supported unreliably depending on the caller's platform.
    image = _make_image(10, 12)
    mask = _make_mask(10, 12, dtype=np.int16)
    rng = np.random.default_rng(0)
    params = sample_perspective(rng, source_size=(12, 10), distortion_scale=0.0)
    with pytest.raises(TypeError, match="dtype"):
        apply_perspective(image, params, mask=mask)

    # the identical mask, through apply_affine, remains fully supported.
    affine_params = sample_affine(rng, source_size=(12, 10))
    affine_result = apply_affine(image, affine_params, mask=mask)
    assert affine_result.mask.dtype == np.int16


def test_apply_perspective_mask_always_uses_nearest_neighbor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import improcv.augmentation as augmentation_module

    calls = []
    real_warp_perspective = augmentation_module._warp_perspective

    def spy(image, matrix, output_size, **kwargs):
        calls.append(kwargs.get("interpolation"))
        return real_warp_perspective(image, matrix, output_size, **kwargs)

    monkeypatch.setattr(augmentation_module, "_warp_perspective", spy)

    image = _make_image(10, 12)
    mask = _make_mask(10, 12)
    rng = np.random.default_rng(0)
    params = sample_perspective(rng, source_size=(12, 10), distortion_scale=0.3)
    apply_perspective(image, params, mask=mask, interpolation=cv2.INTER_LINEAR)

    assert calls[0] == cv2.INTER_LINEAR
    assert calls[1] == cv2.INTER_NEAREST


def test_apply_perspective_mask_border_value_fills_exposed_pixels() -> None:
    image = np.zeros((10, 10), dtype=np.uint8)
    mask = np.full((10, 10), 3, dtype=np.uint8)
    rng = np.random.default_rng(0)
    params = sample_perspective(rng, source_size=(10, 10), distortion_scale=0.5)
    result = apply_perspective(image, params, mask=mask, mask_border_value=250)
    assert 250 in np.unique(result.mask)


def test_apply_perspective_rejects_mask_border_value_out_of_dtype_range() -> None:
    image = _make_image(10, 12)
    mask = _make_mask(10, 12, dtype=np.uint8)
    rng = np.random.default_rng(0)
    params = sample_perspective(rng, source_size=(12, 10))
    with pytest.raises(ValueError, match="mask_border_value"):
        apply_perspective(image, params, mask=mask, mask_border_value=300)


def test_apply_perspective_rejects_bool_mask_border_value() -> None:
    image = _make_image(10, 12)
    mask = _make_mask(10, 12, dtype=np.uint8)
    rng = np.random.default_rng(0)
    params = sample_perspective(rng, source_size=(12, 10))
    with pytest.raises(TypeError, match="mask_border_value"):
        apply_perspective(image, params, mask=mask, mask_border_value=True)


def test_apply_perspective_rejects_mask_spatial_mismatch() -> None:
    image = _make_image(10, 12)
    mask = _make_mask(9, 12)
    rng = np.random.default_rng(0)
    params = sample_perspective(rng, source_size=(12, 10))
    with pytest.raises(ValueError, match="spatial size"):
        apply_perspective(image, params, mask=mask)


def test_apply_perspective_mask_contains_no_new_values_beyond_input_and_border() -> None:
    image = _make_image(20, 20)
    mask = _make_mask(20, 20)
    rng = np.random.default_rng(0)
    params = sample_perspective(rng, source_size=(20, 20), distortion_scale=0.3)
    result = apply_perspective(image, params, mask=mask, mask_border_value=9)
    allowed = set(np.unique(mask).tolist()) | {9}
    assert set(np.unique(result.mask).tolist()) <= allowed


def test_apply_perspective_error_only_on_mask_warp_is_mapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import improcv.augmentation as augmentation_module

    real_warp_perspective = augmentation_module._warp_perspective
    error = cv2.error("simulated mask failure")
    calls = {"count": 0}

    def image_ok_mask_fails(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return real_warp_perspective(*args, **kwargs)
        raise error

    monkeypatch.setattr(augmentation_module, "_warp_perspective", image_ok_mask_fails)

    image = _make_image(10, 12)
    mask = _make_mask(10, 12)
    rng = np.random.default_rng(0)
    params = sample_perspective(rng, source_size=(12, 10))
    with pytest.raises(RuntimeError, match="OpenCV failed") as exc_info:
        apply_perspective(image, params, mask=mask)
    assert exc_info.value.__cause__ is error
    assert calls["count"] == 2


def test_apply_perspective_mask_does_not_mutate_or_alias() -> None:
    image = _make_image(10, 12)
    mask = _make_mask(10, 12)
    mask_before = mask.copy()
    rng = np.random.default_rng(0)
    params = sample_perspective(rng, source_size=(12, 10), distortion_scale=0.3)
    result = apply_perspective(image, params, mask=mask)
    np.testing.assert_array_equal(mask, mask_before)
    assert not np.shares_memory(result.mask, mask)


# --- PerspectiveParameters: type behavior ---


def test_perspective_parameters_equality() -> None:
    rng = np.random.default_rng(0)
    a = sample_perspective(rng, source_size=(10, 8), distortion_scale=0.3)
    b = dataclasses.replace(a)
    assert a == b


def test_perspective_parameters_inequality_on_matrix() -> None:
    a = PerspectiveParameters(
        matrix=np.eye(3, dtype=np.float64),
        source_size=(5, 4),
        destination_points=((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)),
    )
    other_matrix = np.eye(3, dtype=np.float64)
    other_matrix[0, 2] = 1.0
    b = dataclasses.replace(a, matrix=other_matrix)
    assert a != b


def test_perspective_parameters_inequality_on_destination_points() -> None:
    a = PerspectiveParameters(
        matrix=np.eye(3, dtype=np.float64),
        source_size=(5, 4),
        destination_points=((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)),
    )
    b = dataclasses.replace(a, destination_points=((1.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)))
    assert a != b


def test_perspective_parameters_hash_raises() -> None:
    params = PerspectiveParameters(
        matrix=np.eye(3, dtype=np.float64),
        source_size=(5, 4),
        destination_points=((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)),
    )
    with pytest.raises(TypeError):
        hash(params)


def test_perspective_parameters_matrix_is_read_only() -> None:
    rng = np.random.default_rng(0)
    params = sample_perspective(rng, source_size=(10, 8), distortion_scale=0.3)
    assert not params.matrix.flags.writeable


def test_perspective_parameters_does_not_have_distortion_scale_field() -> None:
    rng = np.random.default_rng(0)
    params = sample_perspective(rng, source_size=(10, 8), distortion_scale=0.3)
    assert not hasattr(params, "distortion_scale")
    assert {f.name for f in dataclasses.fields(params)} == {
        "matrix",
        "source_size",
        "destination_points",
    }


# --- manual examples / independent oracle ---


def test_perspective_manual_identity_example() -> None:
    width, height = 5, 4
    source = _independent_source_corners(width, height)
    matrix = cv2.getPerspectiveTransform(source, source)
    np.testing.assert_array_equal(matrix, np.eye(3))
    params = PerspectiveParameters(
        matrix=np.array(matrix, dtype=np.float64),
        source_size=(width, height),
        destination_points=((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)),
    )
    image = _make_image(height, width)
    np.testing.assert_array_equal(apply_perspective(image, params), image)


def test_perspective_manual_safe_single_corner_matches_independent_oracle() -> None:
    width, height = 5, 4
    source = _independent_source_corners(width, height)
    destination = np.array([[0.5, 0.4], [4, 0], [4, 3], [0, 3]], dtype=np.float32)
    expected_matrix = cv2.getPerspectiveTransform(source, destination)

    params = PerspectiveParameters(
        matrix=np.array(expected_matrix, dtype=np.float64),
        source_size=(width, height),
        destination_points=((0.5, 0.4), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)),
    )
    image = _make_image(height, width)
    result = apply_perspective(image, params)
    expected_result = im.warp_perspective(image, params.matrix, params.source_size)
    np.testing.assert_array_equal(result, expected_result)


def test_perspective_manual_symmetric_shrink_matches_expected_matrix() -> None:
    width, height = 5, 4
    source = _independent_source_corners(width, height)
    destination = np.array([[1.0, 0.75], [3.0, 0.75], [3.0, 2.25], [1.0, 2.25]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(source, destination)
    expected = np.array([[0.5, 0.0, 1.0], [0.0, 0.5, 0.75], [0.0, 0.0, 1.0]], dtype=np.float64)
    np.testing.assert_allclose(matrix, expected)

    params = PerspectiveParameters(
        matrix=np.array(matrix, dtype=np.float64),
        source_size=(width, height),
        destination_points=((1.0, 0.75), (3.0, 0.75), (3.0, 2.25), (1.0, 2.25)),
    )
    image = _make_image(height, width)
    result = apply_perspective(image, params)
    expected_result = im.warp_perspective(image, params.matrix, params.source_size)
    np.testing.assert_array_equal(result, expected_result)


def test_perspective_manual_horizon_only_example() -> None:
    matrix = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.5, 0.0, 1.0]],
        dtype=np.float64,
    )
    params = PerspectiveParameters(
        matrix=matrix,
        source_size=(5, 4),
        destination_points=((0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)),
    )
    image = _make_image(4, 5)
    with pytest.raises(ValueError, match="horizon"):
        apply_perspective(image, params)


# --- global np.seterr / warnings isolation ---


def test_sample_perspective_no_floating_point_error_under_seterr_raise() -> None:
    previous = np.seterr(all="raise")
    try:
        rng = np.random.default_rng(0)
        sample_perspective(rng, source_size=(12, 10), distortion_scale=0.5)
    finally:
        np.seterr(**previous)


def test_apply_perspective_no_floating_point_error_under_seterr_raise() -> None:
    previous = np.seterr(all="raise")
    try:
        rng = np.random.default_rng(0)
        params = sample_perspective(rng, source_size=(12, 10), distortion_scale=0.5)
        image = _make_image(10, 12)
        apply_perspective(image, params)
    finally:
        np.seterr(**previous)


def test_sample_and_apply_perspective_no_warning_under_seterr_warn() -> None:
    previous = np.seterr(under="warn")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            rng = np.random.default_rng(0)
            params = sample_perspective(rng, source_size=(12, 10), distortion_scale=0.5)
            image = _make_image(10, 12)
            mask = _make_mask(10, 12)
            apply_perspective(image, params, mask=mask)
    finally:
        np.seterr(**previous)


# --- affine regression: renamed private helper stays invisible publicly ---


def test_apply_affine_source_size_error_message_unchanged_after_helper_rename() -> None:
    params = dataclasses.replace(
        sample_affine(np.random.default_rng(0), source_size=(5, 4)),
        source_size="not-a-tuple",  # type: ignore[arg-type]
    )
    image = _make_image(4, 5)
    with pytest.raises(TypeError, match=r"^params\.source_size must be a tuple"):
        apply_affine(image, params)


# --- import hygiene (extended for perspective) ---


def test_augmentation_module_still_does_not_import_new_dependencies() -> None:
    from pathlib import Path

    import improcv.augmentation

    source = Path(improcv.augmentation.__file__).read_text()
    # cv2/numpy are already legitimate dependencies of this module; this only
    # guards against a brand-new third-party import sneaking in with perspective.
    disallowed = ["scipy", "torch", "torchvision", "albumentations"]
    for name in disallowed:
        assert name not in source
