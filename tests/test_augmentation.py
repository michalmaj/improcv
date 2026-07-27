import cv2
import numpy as np
import pytest

import improcv as im
from improcv.augmentation import (
    AugmentedImageMask,
    CropParameters,
    FlipParameters,
    apply_crop,
    apply_flip,
    sample_crop,
    sample_flip,
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
