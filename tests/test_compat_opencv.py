from pathlib import Path

import cv2
import numpy as np
import pytest

from improcv._compat.opencv import (
    _normalize_calc_hist_output,
    _normalize_hough_lines_p_output,
    merge_hdr_supports_dtype,
    read_onnx_net_from_buffer,
    read_onnx_net_from_path,
)

_FIXTURE_PATH = str(Path(__file__).parent / "data" / "tiny_identity.onnx")


def test_normalize_calc_hist_output_from_column_shape() -> None:
    raw = np.array([[1.0], [2.0], [3.0]])

    result = _normalize_calc_hist_output(raw, bins=3)

    assert result.shape == (3,)
    np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])


def test_normalize_calc_hist_output_from_flat_shape() -> None:
    raw = np.array([1.0, 2.0, 3.0])

    result = _normalize_calc_hist_output(raw, bins=3)

    assert result.shape == (3,)
    np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])


def test_normalize_calc_hist_output_rejects_unexpected_size() -> None:
    raw = np.array([1.0, 2.0, 3.0, 4.0])

    with pytest.raises(RuntimeError, match="size"):
        _normalize_calc_hist_output(raw, bins=3)


def test_normalize_hough_lines_p_output_passes_through_flat_shape() -> None:
    raw = np.array([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=np.int32)

    result = _normalize_hough_lines_p_output(raw)

    assert result.shape == (2, 4)
    np.testing.assert_array_equal(result, raw)


def test_normalize_hough_lines_p_output_squeezes_middle_dimension() -> None:
    raw = np.array([[[10, 20, 30, 40]], [[50, 60, 70, 80]]], dtype=np.int32)

    result = _normalize_hough_lines_p_output(raw)

    assert result.shape == (2, 4)
    np.testing.assert_array_equal(result, [[10, 20, 30, 40], [50, 60, 70, 80]])


def test_normalize_hough_lines_p_output_rejects_wrong_dtype() -> None:
    raw = np.array([[10, 20, 30, 40]], dtype=np.float32)

    with pytest.raises(RuntimeError, match="int32"):
        _normalize_hough_lines_p_output(raw)


def test_normalize_hough_lines_p_output_rejects_non_ndarray() -> None:
    with pytest.raises(RuntimeError, match="ndarray"):
        _normalize_hough_lines_p_output(None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "raw",
    [
        np.zeros((2, 3), dtype=np.int32),
        np.zeros((2, 1, 3), dtype=np.int32),
        np.zeros((2, 2, 4), dtype=np.int32),
    ],
)
def test_normalize_hough_lines_p_output_rejects_wrong_field_count(raw: np.ndarray) -> None:
    with pytest.raises(RuntimeError, match="shape"):
        _normalize_hough_lines_p_output(raw)


# --- merge_hdr_supports_dtype ---
# The actual supported/unsupported outcome depends on the installed OpenCV
# build (verified directly: uint8-only on 4.9.0, uint8/uint16/float32 on
# 4.13.0/5.0.0) -- these tests never hardcode which is true for "the"
# installed OpenCV, only that the reported capability is self-consistent
# with what a direct cv2 call actually does.


def test_merge_hdr_supports_dtype_uint8_always_true() -> None:
    assert merge_hdr_supports_dtype(cv2.createMergeDebevec, np.uint8) is True
    assert merge_hdr_supports_dtype(cv2.createMergeRobertson, np.uint8) is True


@pytest.mark.parametrize("dtype", [np.uint16, np.float32])
def test_merge_hdr_supports_dtype_matches_direct_cv2_behavior(dtype) -> None:
    probe_value = 0.5 if dtype == np.float32 else 1
    probe = np.full((4, 4, 3), probe_value, dtype=dtype)
    times = np.array([1.0, 2.0], dtype=np.float32)

    reported = merge_hdr_supports_dtype(cv2.createMergeDebevec, dtype)

    try:
        cv2.createMergeDebevec().process([probe, probe], times)
        actually_works = True
    except cv2.error:
        actually_works = False

    assert reported is actually_works


def test_merge_hdr_supports_dtype_caches_per_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    call_count = 0
    real_factory = cv2.createMergeDebevec

    def counting_factory():
        nonlocal call_count
        call_count += 1
        return real_factory()

    first = merge_hdr_supports_dtype(counting_factory, np.uint16)
    calls_after_first = call_count
    second = merge_hdr_supports_dtype(counting_factory, np.uint16)

    assert first == second
    assert call_count == calls_after_first  # second call must not re-probe


def test_merge_hdr_supports_dtype_caches_debevec_and_robertson_independently() -> None:
    debevec_result = merge_hdr_supports_dtype(cv2.createMergeDebevec, np.float32)
    robertson_result = merge_hdr_supports_dtype(cv2.createMergeRobertson, np.float32)

    # Not asserted equal -- the two classes are probed and cached
    # independently, precisely because they are not assumed to move
    # together across OpenCV versions.
    assert isinstance(debevec_result, bool)
    assert isinstance(robertson_result, bool)


# --- read_onnx_net_from_path / read_onnx_net_from_buffer ---
# `ENGINE_CLASSIC` only exists on OpenCV >= 5.0 -- these tests force both
# branches via monkeypatch rather than depending on which OpenCV happens to
# be installed, so the capability-detection logic itself is verified on
# every supported version, not just whichever one CI happens to run this on.


def test_read_onnx_net_from_path_passes_engine_when_capability_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel_engine = object()
    captured: dict[str, object] = {}
    monkeypatch.setattr(cv2.dnn, "ENGINE_CLASSIC", sentinel_engine, raising=False)

    def fake_read(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(cv2.dnn, "readNetFromONNX", fake_read)

    read_onnx_net_from_path("some/path.onnx")

    assert captured["args"] == ("some/path.onnx",)
    assert captured["kwargs"] == {"engine": sentinel_engine}


def test_read_onnx_net_from_path_omits_engine_when_capability_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.delattr(cv2.dnn, "ENGINE_CLASSIC", raising=False)

    def fake_read(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(cv2.dnn, "readNetFromONNX", fake_read)

    read_onnx_net_from_path("some/path.onnx")

    assert captured["args"] == ("some/path.onnx",)
    assert captured["kwargs"] == {}


def test_read_onnx_net_from_buffer_passes_engine_when_capability_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel_engine = object()
    captured_args: tuple[object, ...] = ()
    captured_kwargs: dict[str, object] = {}
    buffer = np.zeros(8, dtype=np.uint8)
    monkeypatch.setattr(cv2.dnn, "ENGINE_CLASSIC", sentinel_engine, raising=False)

    def fake_read(*args, **kwargs):
        nonlocal captured_args
        captured_args = args
        captured_kwargs.update(kwargs)
        return object()

    monkeypatch.setattr(cv2.dnn, "readNetFromONNX", fake_read)

    read_onnx_net_from_buffer(buffer)

    assert captured_args == ()
    assert captured_kwargs["buffer"] is buffer
    assert captured_kwargs["engine"] is sentinel_engine


def test_read_onnx_net_from_buffer_omits_engine_when_capability_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    buffer = np.zeros(8, dtype=np.uint8)
    monkeypatch.delattr(cv2.dnn, "ENGINE_CLASSIC", raising=False)

    def fake_read(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(cv2.dnn, "readNetFromONNX", fake_read)

    read_onnx_net_from_buffer(buffer)

    assert captured["args"] == ()
    assert captured["kwargs"] == {"buffer": buffer}


def test_read_onnx_net_from_path_loads_real_fixture() -> None:
    net = read_onnx_net_from_path(_FIXTURE_PATH)

    assert isinstance(net, cv2.dnn.Net)
    assert not net.empty()


def test_read_onnx_net_from_buffer_loads_real_fixture() -> None:
    data = Path(_FIXTURE_PATH).read_bytes()
    buffer = np.frombuffer(data, dtype=np.uint8)

    net = read_onnx_net_from_buffer(buffer)

    assert isinstance(net, cv2.dnn.Net)
    assert not net.empty()
