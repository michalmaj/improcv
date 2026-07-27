from collections.abc import Iterator, Sequence

import cv2
import numpy as np
import pytest
from numpy.testing import assert_array_equal

import improcv as im
from improcv.dnn import create_dnn_batch_blob, create_dnn_blob


def _gradient_image(height: int, width: int, channels: int | None, dtype: type) -> np.ndarray:
    """A deterministic, non-periodic image encoding column/row/channel position in its values.

    Used for geometry tests (crop/resize) where a uniform-color image
    cannot distinguish "resized correctly" from "resized wrong" -- mirrors
    the reasoning in `test_stitching.py`'s `_make_scene` docstring, applied
    to a much simpler, fully deterministic pattern rather than a randomized
    scene (geometry tests here need exact, reproducible pixel values, not
    just "stitching succeeded").
    """
    shape = (height, width) if channels is None else (height, width, channels)
    image = np.zeros(shape, dtype=np.float32)
    cols = np.arange(width, dtype=np.float32)
    rows = np.arange(height, dtype=np.float32)
    if channels is None:
        image[:, :] = cols[np.newaxis, :] + rows[:, np.newaxis] * width
    else:
        for c in range(channels):
            image[:, :, c] = cols[np.newaxis, :] + rows[:, np.newaxis] * width + c
    if dtype == np.uint8:
        image = np.mod(image, 256.0)
    return image.astype(dtype)


def _non_contiguous_view(image: np.ndarray) -> np.ndarray:
    """A genuinely non-contiguous view with the same shape as `image`.

    See `test_stitching.py`'s identical helper: a step-1 slice of an
    already-contiguous array is still contiguous and would not exercise
    the non-contiguous-input contract at all.
    """
    storage = np.empty(
        (image.shape[0], image.shape[1] * 2, *image.shape[2:]),
        dtype=image.dtype,
    )
    storage[:, ::2] = image
    view = storage[:, ::2]
    assert view.shape == image.shape
    assert not view.flags.c_contiguous
    return view


class _CustomSequence(Sequence):
    """A minimal, real `collections.abc.Sequence` that is neither list nor tuple."""

    def __init__(self, items: list) -> None:
        self._items = items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index):
        return self._items[index]


def _forbid_blob_from_image(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args, **kwargs):
        pytest.fail("cv2.dnn.blobFromImage must not be called after a validation error")

    monkeypatch.setattr(cv2.dnn, "blobFromImage", boom)


def _forbid_blob_from_images(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args, **kwargs):
        pytest.fail("cv2.dnn.blobFromImages must not be called after a validation error")

    monkeypatch.setattr(cv2.dnn, "blobFromImages", boom)


# --- single image: happy paths ---


@pytest.mark.parametrize(
    ("channels", "dtype"),
    [
        (None, np.uint8),
        (1, np.uint8),
        (3, np.uint8),
        (4, np.uint8),
        (None, np.float32),
        (1, np.float32),
        (3, np.float32),
        (4, np.float32),
    ],
)
def test_create_dnn_blob_accepts_supported_dtype_and_channels(channels, dtype) -> None:
    image = _gradient_image(6, 8, channels, dtype)

    blob = create_dnn_blob(image)

    expected_channels = 1 if channels is None else channels
    assert blob.shape == (1, expected_channels, 6, 8)
    assert blob.dtype == np.float32


def test_create_dnn_blob_size_none_keeps_native_size() -> None:
    image = _gradient_image(5, 9, 3, np.uint8)

    blob = create_dnn_blob(image, size=None)

    assert blob.shape == (1, 3, 5, 9)


def test_create_dnn_blob_explicit_size_resizes() -> None:
    image = _gradient_image(10, 20, 3, np.uint8)

    blob = create_dnn_blob(image, size=(8, 4))

    assert blob.shape == (1, 3, 4, 8)


def test_create_dnn_blob_accepts_numpy_integer_scalar_in_size() -> None:
    image = _gradient_image(10, 10, 3, np.uint8)

    blob = create_dnn_blob(image, size=(np.int32(8), np.int64(8)))  # type: ignore[arg-type]

    assert blob.shape == (1, 3, 8, 8)


def test_create_dnn_blob_crop_false_stretches_without_aspect_ratio() -> None:
    image = _gradient_image(10, 20, 3, np.uint8)

    wrapper_blob = create_dnn_blob(image, size=(8, 8), crop=False)
    direct_blob = cv2.dnn.blobFromImage(image, size=(8, 8), crop=False, ddepth=cv2.CV_32F)

    assert_array_equal(wrapper_blob, direct_blob)


def test_create_dnn_blob_crop_true_preserves_aspect_ratio_then_center_crops() -> None:
    image = _gradient_image(10, 20, 3, np.uint8)

    wrapper_blob = create_dnn_blob(image, size=(8, 8), crop=True)
    direct_blob = cv2.dnn.blobFromImage(image, size=(8, 8), crop=True, ddepth=cv2.CV_32F)

    assert_array_equal(wrapper_blob, direct_blob)
    # crop=True and crop=False must disagree for a non-square resize -- otherwise this
    # test would not actually be exercising different geometry.
    stretched = create_dnn_blob(image, size=(8, 8), crop=False)
    assert not np.array_equal(wrapper_blob, stretched)


def test_create_dnn_blob_swap_rb_reorders_channels() -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[:, :, 0] = 10  # B
    image[:, :, 1] = 20  # G
    image[:, :, 2] = 30  # R

    no_swap = create_dnn_blob(image, swap_rb=False)
    swapped = create_dnn_blob(image, swap_rb=True)

    assert_array_equal(no_swap[0, :, 0, 0], [10, 20, 30])  # B, G, R
    assert_array_equal(swapped[0, :, 0, 0], [30, 20, 10])  # R, G, B


def test_create_dnn_blob_bgra_alpha_unaffected_by_swap_rb() -> None:
    image = np.zeros((4, 4, 4), dtype=np.uint8)
    image[:, :, 0] = 10  # B
    image[:, :, 1] = 20  # G
    image[:, :, 2] = 30  # R
    image[:, :, 3] = 40  # A

    blob = create_dnn_blob(image, swap_rb=True)

    assert_array_equal(blob[0, :, 0, 0], [30, 20, 10, 40])  # R, G, B, A


def test_create_dnn_blob_mean_scalar_broadcasts_to_all_channels() -> None:
    image = _gradient_image(4, 4, 3, np.uint8)

    wrapper_blob = create_dnn_blob(image, mean=1.0)
    direct_blob = cv2.dnn.blobFromImage(image, mean=(1.0, 1.0, 1.0), ddepth=cv2.CV_32F)

    assert_array_equal(wrapper_blob, direct_blob)


def test_create_dnn_blob_mean_tuple_order_follows_swap_rb() -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[:, :, 0] = 10  # B
    image[:, :, 1] = 20  # G
    image[:, :, 2] = 30  # R

    no_swap = create_dnn_blob(image, mean=(1.0, 2.0, 3.0), swap_rb=False)
    swapped = create_dnn_blob(image, mean=(1.0, 2.0, 3.0), swap_rb=True)

    assert_array_equal(no_swap[0, :, 0, 0], [9, 18, 27])  # (B-1, G-2, R-3)
    assert_array_equal(swapped[0, :, 0, 0], [29, 18, 7])  # (R-1, G-2, B-3)


@pytest.mark.parametrize("scale", [1.0, 0.0, -1.0, 2.5, np.float64(0.5)])
def test_create_dnn_blob_accepts_zero_positive_and_negative_scale(scale) -> None:
    image = np.full((4, 4, 3), 5, dtype=np.uint8)

    blob = create_dnn_blob(image, scale=scale)

    assert np.all(np.isfinite(blob))
    assert_array_equal(blob[0, :, 0, 0], [5.0 * float(scale)] * 3)


def test_create_dnn_blob_accepts_non_contiguous_view() -> None:
    image = _non_contiguous_view(_gradient_image(6, 6, 3, np.uint8))

    blob = create_dnn_blob(image)

    assert blob.shape == (1, 3, 6, 6)


def test_create_dnn_blob_accepts_read_only_array() -> None:
    image = _gradient_image(4, 4, 3, np.uint8)
    image.flags.writeable = False

    blob = create_dnn_blob(image)

    assert blob.shape == (1, 3, 4, 4)


def test_create_dnn_blob_accepts_fortran_order() -> None:
    image = np.asfortranarray(_gradient_image(4, 4, 3, np.uint8))

    blob = create_dnn_blob(image)

    assert blob.shape == (1, 3, 4, 4)


def test_create_dnn_blob_does_not_mutate_input() -> None:
    image = _gradient_image(4, 4, 3, np.uint8)
    before = image.copy()

    create_dnn_blob(image, scale=2.0, mean=1.0)

    assert_array_equal(image, before)


def test_create_dnn_blob_result_does_not_share_memory_with_input() -> None:
    image = _gradient_image(4, 4, 3, np.float32)

    blob = create_dnn_blob(image)

    assert not np.shares_memory(blob, image)


# --- single image: validation errors ---


def test_create_dnn_blob_rejects_non_ndarray() -> None:
    with pytest.raises(TypeError, match="image must be a NumPy array"):
        create_dnn_blob([[1, 2], [3, 4]])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "dtype",
    [bool, np.int8, np.int16, np.int32, np.uint16, np.float16, np.float64, np.complex64],
)
def test_create_dnn_blob_rejects_unsupported_dtype(dtype) -> None:
    image = np.zeros((4, 4, 3), dtype=dtype)

    with pytest.raises(TypeError, match="dtype"):
        create_dnn_blob(image)


@pytest.mark.parametrize("channels", [2, 5])
def test_create_dnn_blob_rejects_unsupported_channel_count(channels) -> None:
    image = np.zeros((4, 4, channels), dtype=np.uint8)

    with pytest.raises(ValueError, match="1, 3, or 4 channels"):
        create_dnn_blob(image)


@pytest.mark.parametrize("shape", [(4,), (2, 4, 4, 3)])
def test_create_dnn_blob_rejects_wrong_ndim(shape) -> None:
    image = np.zeros(shape, dtype=np.uint8)

    with pytest.raises(ValueError, match="dimensions"):
        create_dnn_blob(image)


def test_create_dnn_blob_rejects_empty_image() -> None:
    image = np.zeros((0, 0, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="empty"):
        create_dnn_blob(image)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_create_dnn_blob_rejects_non_finite_float32_input(bad_value) -> None:
    image = np.full((4, 4, 3), bad_value, dtype=np.float32)

    with pytest.raises(ValueError, match="finite"):
        create_dnn_blob(image)


@pytest.mark.parametrize("size", [(1, 2, 3), 8, "8x8"])
def test_create_dnn_blob_rejects_malformed_size(size) -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="2-tuple"):
        create_dnn_blob(image, size=size)  # type: ignore[arg-type]


def test_create_dnn_blob_rejects_bool_in_size() -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    with pytest.raises(TypeError):
        create_dnn_blob(image, size=(True, True))


@pytest.mark.parametrize("size", [(0, 8), (-1, 8), (8, 0)])
def test_create_dnn_blob_rejects_non_positive_size(size) -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        create_dnn_blob(image, size=size)


def test_create_dnn_blob_rejects_size_dimension_over_int32() -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    too_big = int(np.iinfo(np.int32).max) + 1

    with pytest.raises(ValueError, match="32-bit"):
        create_dnn_blob(image, size=(too_big, 8))


def test_create_dnn_blob_rejects_size_product_over_int32() -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    half = int(np.iinfo(np.int32).max // 2) + 2

    with pytest.raises(ValueError, match="width \\* height"):
        create_dnn_blob(image, size=(half, half))


def test_create_dnn_blob_rejects_crop_true_with_size_none() -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="no-op"):
        create_dnn_blob(image, size=None, crop=True)


@pytest.mark.parametrize("scale", [True, "1.0", None])
def test_create_dnn_blob_rejects_bad_scale_type(scale) -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    with pytest.raises(TypeError, match="scale"):
        create_dnn_blob(image, scale=scale)  # type: ignore[arg-type]


@pytest.mark.parametrize("scale", [float("nan"), float("inf"), float("-inf")])
def test_create_dnn_blob_rejects_non_finite_scale(scale) -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="finite"):
        create_dnn_blob(image, scale=scale)


def test_create_dnn_blob_rejects_scale_overflowing_float32() -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="finite"):
        create_dnn_blob(image, scale=1e300)


def test_create_dnn_blob_rejects_huge_python_int_scale() -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="finite"):
        create_dnn_blob(image, scale=10**400)


@pytest.mark.parametrize("mean", [True, "1.0", [1.0, 2.0, 3.0], np.array([1.0, 2.0, 3.0])])
def test_create_dnn_blob_rejects_bad_mean_type(mean) -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    with pytest.raises(TypeError, match="mean"):
        create_dnn_blob(image, mean=mean)  # type: ignore[arg-type]


@pytest.mark.parametrize("mean", [(1.0,), (1.0, 2.0), (1.0, 2.0, 3.0, 4.0)])
def test_create_dnn_blob_rejects_wrong_length_mean_tuple(mean) -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)  # 3 channels

    with pytest.raises(ValueError, match="3 element"):
        create_dnn_blob(image, mean=mean)


@pytest.mark.parametrize("shape", [(4, 4), (4, 4, 1)])
def test_create_dnn_blob_accepts_one_element_mean_tuple_for_grayscale(shape) -> None:
    image = np.full(shape, 5, dtype=np.uint8)

    blob = create_dnn_blob(image, mean=(2.0,))

    assert blob.shape == (1, 1, 4, 4)
    assert_array_equal(blob, np.full((1, 1, 4, 4), 3.0, dtype=np.float32))


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_create_dnn_blob_rejects_non_finite_mean_element(bad_value) -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="finite"):
        create_dnn_blob(image, mean=(1.0, bad_value, 3.0))


@pytest.mark.parametrize("swap_rb", [0, 1, np.bool_(True), "true", None])
def test_create_dnn_blob_rejects_non_bool_swap_rb(swap_rb) -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    with pytest.raises(TypeError, match="swap_rb"):
        create_dnn_blob(image, swap_rb=swap_rb)  # type: ignore[arg-type]


@pytest.mark.parametrize("crop", [0, 1, np.bool_(True), "true", None])
def test_create_dnn_blob_rejects_non_bool_crop(crop) -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    with pytest.raises(TypeError, match="crop"):
        create_dnn_blob(image, crop=crop)  # type: ignore[arg-type]


@pytest.mark.parametrize("channels", [None, 1])
def test_create_dnn_blob_rejects_swap_rb_on_grayscale(channels) -> None:
    image = _gradient_image(4, 4, channels, np.uint8)

    with pytest.raises(ValueError, match="single-channel"):
        create_dnn_blob(image, swap_rb=True)


@pytest.mark.parametrize(
    "make_invalid",
    [
        lambda: [1, 2, 3],
        lambda: np.zeros((4, 4, 2), dtype=np.uint8),
        lambda: np.zeros((0, 0, 3), dtype=np.uint8),
        lambda: np.full((4, 4, 3), np.nan, dtype=np.float32),
    ],
)
def test_create_dnn_blob_forbids_opencv_call_after_validation_error(
    make_invalid, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid_blob_from_image(monkeypatch)

    with pytest.raises((TypeError, ValueError)):
        create_dnn_blob(make_invalid())


# --- single image: cv2.error mapping and postconditions ---


def test_create_dnn_blob_wraps_cv2_error_as_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    original = cv2.error("synthetic failure")

    def boom(*args, **kwargs):
        raise original

    monkeypatch.setattr(cv2.dnn, "blobFromImage", boom)
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    with pytest.raises(RuntimeError) as exc_info:
        create_dnn_blob(image)

    assert exc_info.value.__cause__ is original


def test_create_dnn_blob_rejects_non_finite_result_from_numeric_overflow() -> None:
    """A real (non-monkeypatched) OpenCV call: finite input, finite `scale`, non-finite output.

    Verified directly, identically on OpenCV 4.9.0/4.13.0/5.0.0: multiplying
    `np.finfo(np.float32).max` by `scale=2.0` overflows to `inf` inside
    `cv2.dnn.blobFromImage` itself (no exception raised by OpenCV -- it
    just returns a blob containing `inf`). This is the real-world case the
    monkeypatched postcondition tests below stand in for; this test exists
    so at least one non-finite-output path is exercised through the actual
    OpenCV call, not only through a fake.
    """
    image = np.full((1, 1, 3), np.finfo(np.float32).max, dtype=np.float32)

    with pytest.raises(RuntimeError, match="NaN/Inf"):
        create_dnn_blob(image, scale=2.0)


@pytest.mark.parametrize(
    "fake_result",
    [
        None,
        np.zeros((1, 3, 4, 4), dtype=np.uint8),  # wrong dtype
        np.zeros((2, 3, 4, 4), dtype=np.float32),  # wrong batch size
        np.zeros((1, 4, 4, 4), dtype=np.float32),  # wrong channel count
        np.zeros((1, 3, 5, 5), dtype=np.float32),  # wrong spatial shape
        np.zeros((1, 3, 4), dtype=np.float32),  # wrong ndim
        np.full((1, 3, 4, 4), np.nan, dtype=np.float32),  # non-finite output
    ],
)
def test_create_dnn_blob_postcondition_failure_raises_runtime_error(
    fake_result, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cv2.dnn, "blobFromImage", lambda *a, **k: fake_result)
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    with pytest.raises(RuntimeError):
        create_dnn_blob(image)


# --- batch: happy paths ---


def test_create_dnn_batch_blob_single_image() -> None:
    image = _gradient_image(4, 4, 3, np.uint8)

    blob = create_dnn_batch_blob([image], size=(4, 4))

    assert blob.shape == (1, 3, 4, 4)


def test_create_dnn_batch_blob_two_images() -> None:
    image_a = _gradient_image(4, 4, 3, np.uint8)
    image_b = _gradient_image(4, 4, 3, np.uint8) + 1

    blob = create_dnn_batch_blob([image_a, image_b], size=(4, 4))

    assert blob.shape == (2, 3, 4, 4)


@pytest.mark.parametrize(
    "container_factory",
    [list, tuple, _CustomSequence],
)
def test_create_dnn_batch_blob_accepts_various_sequence_types(container_factory) -> None:
    images = container_factory(
        [_gradient_image(4, 4, 3, np.uint8), _gradient_image(4, 4, 3, np.uint8)]
    )

    blob = create_dnn_batch_blob(images, size=(4, 4))

    assert blob.shape == (2, 3, 4, 4)


def test_create_dnn_batch_blob_accepts_different_spatial_shapes_with_explicit_size() -> None:
    small = _gradient_image(4, 4, 3, np.uint8)
    large = _gradient_image(9, 12, 3, np.uint8)

    blob = create_dnn_batch_blob([small, large], size=(6, 6))

    assert blob.shape == (2, 3, 6, 6)


def test_create_dnn_batch_blob_does_not_mutate_elements() -> None:
    image_a = _gradient_image(4, 4, 3, np.uint8)
    image_b = _gradient_image(4, 4, 3, np.uint8) + 1
    before_a, before_b = image_a.copy(), image_b.copy()

    create_dnn_batch_blob([image_a, image_b], size=(4, 4), scale=2.0, mean=1.0)

    assert_array_equal(image_a, before_a)
    assert_array_equal(image_b, before_b)


def test_create_dnn_batch_blob_result_does_not_share_memory_with_elements() -> None:
    image_a = _gradient_image(4, 4, 3, np.float32)
    image_b = _gradient_image(4, 4, 3, np.float32) + 1.0

    blob = create_dnn_batch_blob([image_a, image_b], size=(4, 4))

    assert not np.shares_memory(blob, image_a)
    assert not np.shares_memory(blob, image_b)


def test_create_dnn_batch_blob_matches_direct_opencv_call() -> None:
    images = [_gradient_image(4, 4, 3, np.uint8), _gradient_image(4, 4, 3, np.uint8) + 1]
    # improcv normalizes `scale` through float32 before calling OpenCV (see
    # `_to_finite_float32`) -- comparing against a direct call must use that same
    # already-rounded value, or the two calls legitimately differ in the last bit
    # from a float64-vs-float32 rounding difference, not from a wrapper bug.
    normalized_scale = float(np.float32(1 / 255.0))

    wrapper_blob = create_dnn_batch_blob(images, size=(4, 4), scale=1 / 255.0, swap_rb=True)
    direct_blob = cv2.dnn.blobFromImages(
        images, scalefactor=normalized_scale, size=(4, 4), swapRB=True, ddepth=cv2.CV_32F
    )

    assert_array_equal(wrapper_blob, direct_blob)


# --- batch: validation errors ---


def test_create_dnn_batch_blob_requires_size() -> None:
    with pytest.raises(TypeError, match="size"):
        create_dnn_batch_blob([np.zeros((4, 4, 3), dtype=np.uint8)], size=None)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_container", ["abc", b"abc", bytearray(b"abc")])
def test_create_dnn_batch_blob_rejects_str_bytes_bytearray(bad_container) -> None:
    with pytest.raises(TypeError, match="Sequence"):
        create_dnn_batch_blob(bad_container, size=(4, 4))  # type: ignore[arg-type]


def test_create_dnn_batch_blob_rejects_single_ndarray_including_4d() -> None:
    stack = np.zeros((2, 4, 4, 3), dtype=np.uint8)

    with pytest.raises(TypeError, match="Sequence"):
        create_dnn_batch_blob(stack, size=(4, 4))  # type: ignore[arg-type]


def test_create_dnn_batch_blob_rejects_generator() -> None:
    def gen() -> Iterator[np.ndarray]:
        yield np.zeros((4, 4, 3), dtype=np.uint8)

    with pytest.raises(TypeError, match="Sequence"):
        create_dnn_batch_blob(gen(), size=(4, 4))  # type: ignore[arg-type]


def test_create_dnn_batch_blob_rejects_iterator() -> None:
    images = iter([np.zeros((4, 4, 3), dtype=np.uint8)])

    with pytest.raises(TypeError, match="Sequence"):
        create_dnn_batch_blob(images, size=(4, 4))  # type: ignore[arg-type]


def test_create_dnn_batch_blob_rejects_empty_sequence() -> None:
    with pytest.raises(ValueError, match="at least one image"):
        create_dnn_batch_blob([], size=(4, 4))


def test_create_dnn_batch_blob_rejects_indexed_empty_image() -> None:
    good = np.zeros((4, 4, 3), dtype=np.uint8)
    empty = np.zeros((0, 0, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match=r"images\[1\]"):
        create_dnn_batch_blob([good, empty], size=(4, 4))


def test_create_dnn_batch_blob_rejects_indexed_bad_dtype() -> None:
    good = np.zeros((4, 4, 3), dtype=np.uint8)
    bad = np.zeros((4, 4, 3), dtype=np.float64)

    with pytest.raises(TypeError, match=r"images\[1\]"):
        create_dnn_batch_blob([good, bad], size=(4, 4))


def test_create_dnn_batch_blob_rejects_mismatched_dtype_across_batch() -> None:
    uint8_image = np.zeros((4, 4, 3), dtype=np.uint8)
    float32_image = np.zeros((4, 4, 3), dtype=np.float32)

    with pytest.raises(TypeError, match=r"images\[1\].*dtype"):
        create_dnn_batch_blob([uint8_image, float32_image], size=(4, 4))


def test_create_dnn_batch_blob_rejects_mismatched_channel_count_across_batch() -> None:
    bgr = np.zeros((4, 4, 3), dtype=np.uint8)
    bgra = np.zeros((4, 4, 4), dtype=np.uint8)

    with pytest.raises(ValueError, match=r"images\[1\].*channel"):
        create_dnn_batch_blob([bgr, bgra], size=(4, 4))


def test_create_dnn_batch_blob_forbids_opencv_call_after_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_blob_from_images(monkeypatch)

    with pytest.raises(ValueError):
        create_dnn_batch_blob([], size=(4, 4))


# --- batch: cv2.error mapping and postconditions ---


def test_create_dnn_batch_blob_wraps_cv2_error_as_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = cv2.error("synthetic failure")

    def boom(*args, **kwargs):
        raise original

    monkeypatch.setattr(cv2.dnn, "blobFromImages", boom)
    images = [np.zeros((4, 4, 3), dtype=np.uint8)]

    with pytest.raises(RuntimeError) as exc_info:
        create_dnn_batch_blob(images, size=(4, 4))

    assert exc_info.value.__cause__ is original


def test_create_dnn_batch_blob_postcondition_wrong_batch_size_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cv2.dnn, "blobFromImages", lambda *a, **k: np.zeros((1, 3, 4, 4), dtype=np.float32)
    )
    images = [np.zeros((4, 4, 3), dtype=np.uint8), np.zeros((4, 4, 3), dtype=np.uint8)]

    with pytest.raises(RuntimeError):
        create_dnn_batch_blob(images, size=(4, 4))


# --- top-level export ---


def test_create_dnn_blob_and_batch_blob_exported_from_top_level_package() -> None:
    assert im.create_dnn_blob is create_dnn_blob
    assert im.create_dnn_batch_blob is create_dnn_batch_blob
