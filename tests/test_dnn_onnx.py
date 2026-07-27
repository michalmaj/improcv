import gc
import os
import platform
from pathlib import Path

import cv2
import numpy as np
import pytest
from numpy.testing import assert_array_equal

import improcv as im
from improcv.dnn import load_onnx_network, load_onnx_network_from_bytes

FIXTURE_PATH = Path(__file__).parent / "data" / "tiny_identity.onnx"
FIXTURE_SHA256 = "2f9035bf781080e0ae154cf0c1c5dcec8b92aa0f3bd36caf73cae7aadd6219c3"
IDENTITY_SHAPE = (1, 3, 4, 4)


def _fixture_bytes() -> bytes:
    return FIXTURE_PATH.read_bytes()


class _CustomPathLike:
    """A minimal, real `os.PathLike[str]` that is neither `str` nor `pathlib.Path`."""

    def __init__(self, path: str) -> None:
        self._path = path

    def __fspath__(self) -> str:
        return self._path


class _BytesPathLike:
    """An `os.PathLike` whose `__fspath__()` returns `bytes`, not `str`."""

    def __init__(self, path: bytes) -> None:
        self._path = path

    def __fspath__(self) -> bytes:
        return self._path


def _forbid_read_net_from_onnx(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args, **kwargs):
        pytest.fail("cv2.dnn.readNetFromONNX must not be called after a validation error")

    monkeypatch.setattr(cv2.dnn, "readNetFromONNX", boom)


def _run_identity_forward(net: cv2.dnn.Net) -> None:
    blob = np.arange(48, dtype=np.float32).reshape(*IDENTITY_SHAPE)
    net.setInput(blob)
    output = net.forward()
    assert_array_equal(output, blob)


# --- fixture sanity ---


def test_fixture_matches_documented_sha256_and_size() -> None:
    data = _fixture_bytes()
    assert len(data) == 141
    import hashlib

    assert hashlib.sha256(data).hexdigest() == FIXTURE_SHA256


# --- path loader: happy paths ---


def test_load_onnx_network_accepts_str_path() -> None:
    net = load_onnx_network(str(FIXTURE_PATH))

    assert isinstance(net, cv2.dnn.Net)
    assert not net.empty()


def test_load_onnx_network_accepts_path_object() -> None:
    net = load_onnx_network(FIXTURE_PATH)

    assert not net.empty()


def test_load_onnx_network_accepts_custom_pathlike() -> None:
    net = load_onnx_network(_CustomPathLike(str(FIXTURE_PATH)))

    assert not net.empty()


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason=(
        "verified on CI: cv2.dnn.readNetFromONNX fails to open a path containing these "
        "non-ASCII characters on Windows (a cv2.error from OpenCV's own file-opening code, "
        "correctly mapped by this wrapper to RuntimeError -- not a bug in improcv, but not "
        "a cross-platform guarantee this test should assert either)"
    ),
)
def test_load_onnx_network_accepts_unicode_path(tmp_path: Path) -> None:
    unicode_dir = tmp_path / "onnx_uniçödé_tëst"
    unicode_dir.mkdir()
    unicode_path = unicode_dir / "mödel.onnx"
    unicode_path.write_bytes(_fixture_bytes())

    net = load_onnx_network(unicode_path)

    assert not net.empty()


def test_load_onnx_network_accepts_relative_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(FIXTURE_PATH.parent)

    net = load_onnx_network(FIXTURE_PATH.name)

    assert not net.empty()


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="symlink creation may require elevated privileges on Windows",
)
def test_load_onnx_network_accepts_valid_symlink(tmp_path: Path) -> None:
    link = tmp_path / "valid_link.onnx"
    try:
        link.symlink_to(FIXTURE_PATH)
    except OSError:
        pytest.skip("platform/environment does not allow creating symlinks")

    net = load_onnx_network(link)

    assert not net.empty()


def test_load_onnx_network_accepts_file_without_extension(tmp_path: Path) -> None:
    no_ext = tmp_path / "tiny_identity"
    no_ext.write_bytes(_fixture_bytes())

    net = load_onnx_network(no_ext)

    assert not net.empty()


def test_load_onnx_network_accepts_uppercase_onnx_extension(tmp_path: Path) -> None:
    upper = tmp_path / "tiny_identity.ONNX"
    upper.write_bytes(_fixture_bytes())

    net = load_onnx_network(upper)

    assert not net.empty()


def test_load_onnx_network_repeated_loads_return_distinct_objects() -> None:
    net1 = load_onnx_network(FIXTURE_PATH)
    net2 = load_onnx_network(FIXTURE_PATH)

    assert net1 is not net2


def test_load_onnx_network_integration_forward() -> None:
    net = load_onnx_network(FIXTURE_PATH)

    _run_identity_forward(net)


# --- path loader: validation errors ---


def test_load_onnx_network_rejects_bytes_pathlike() -> None:
    with pytest.raises(TypeError, match="path"):
        load_onnx_network(_BytesPathLike(str(FIXTURE_PATH).encode()))  # type: ignore[arg-type]


def test_load_onnx_network_rejects_non_pathlike_type() -> None:
    with pytest.raises(TypeError, match="path"):
        load_onnx_network(12345)  # type: ignore[arg-type]


def test_load_onnx_network_rejects_empty_string() -> None:
    with pytest.raises(ValueError, match="empty"):
        load_onnx_network("")


def test_load_onnx_network_rejects_nonexistent_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_onnx_network(tmp_path / "does_not_exist.onnx")


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="symlink creation may require elevated privileges on Windows",
)
def test_load_onnx_network_rejects_broken_symlink(tmp_path: Path) -> None:
    link = tmp_path / "broken_link.onnx"
    try:
        link.symlink_to(tmp_path / "no_such_target.onnx")
    except OSError:
        pytest.skip("platform/environment does not allow creating symlinks")

    with pytest.raises(FileNotFoundError):
        load_onnx_network(link)


def test_load_onnx_network_rejects_directory(tmp_path: Path) -> None:
    with pytest.raises(IsADirectoryError):
        load_onnx_network(tmp_path)


def test_load_onnx_network_rejects_empty_file(tmp_path: Path) -> None:
    empty_file = tmp_path / "empty.onnx"
    empty_file.write_bytes(b"")

    with pytest.raises(ValueError, match="empty"):
        load_onnx_network(empty_file)


def test_load_onnx_network_rejects_corrupt_random_bytes(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.onnx"
    corrupt.write_bytes(bytes([1, 2, 3, 4, 5]) * 20)

    with pytest.raises(RuntimeError):
        load_onnx_network(corrupt)


def test_load_onnx_network_rejects_truncated_fixture(tmp_path: Path) -> None:
    truncated = tmp_path / "truncated.onnx"
    data = _fixture_bytes()
    truncated.write_bytes(data[: len(data) // 2])

    with pytest.raises(RuntimeError):
        load_onnx_network(truncated)


def test_load_onnx_network_rejects_text_file_named_onnx(tmp_path: Path) -> None:
    text_file = tmp_path / "notes.onnx"
    text_file.write_text("this is not an onnx model at all")

    with pytest.raises(RuntimeError):
        load_onnx_network(text_file)


def test_load_onnx_network_propagates_permission_error_from_stat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(self, *args, **kwargs):
        raise PermissionError("synthetic permission denial")

    monkeypatch.setattr(Path, "stat", boom)

    with pytest.raises(PermissionError):
        load_onnx_network(FIXTURE_PATH)


@pytest.mark.skipif(
    platform.system() == "Windows" or os.geteuid() == 0,
    reason="chmod-based permission denial is unreliable on Windows and as root",
)
def test_load_onnx_network_maps_real_unreadable_file_to_runtime_error(tmp_path: Path) -> None:
    """A real (non-monkeypatched) chmod 000 file on POSIX.

    Verified directly: `Path.stat()` succeeds on a chmod 000 file (it only
    reads directory-entry metadata, not file content, so it needs no read
    permission on the file itself) -- the permission denial only surfaces
    later, when OpenCV itself tries to open the file, as a `cv2.error`
    mapped to `RuntimeError`. This is the documented, expected outcome (see
    `load_onnx_network`'s docstring), not a synthetic `PermissionError`
    from prevalidation -- that contract is covered separately above via a
    monkeypatched `Path.stat`.
    """
    unreadable = tmp_path / "unreadable.onnx"
    unreadable.write_bytes(_fixture_bytes())
    unreadable.chmod(0o000)
    try:
        with pytest.raises(RuntimeError):
            load_onnx_network(unreadable)
    finally:
        unreadable.chmod(0o644)


@pytest.mark.parametrize(
    "make_invalid",
    [
        lambda: "",
        lambda: 42,
    ],
)
def test_load_onnx_network_forbids_opencv_call_after_validation_error(
    make_invalid, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid_read_net_from_onnx(monkeypatch)

    with pytest.raises((TypeError, ValueError)):
        load_onnx_network(make_invalid())


# --- bytes loader: happy paths ---


def test_load_onnx_network_from_bytes_accepts_valid_bytes() -> None:
    net = load_onnx_network_from_bytes(_fixture_bytes())

    assert isinstance(net, cv2.dnn.Net)
    assert not net.empty()


def test_load_onnx_network_from_bytes_repeated_loads_return_distinct_objects() -> None:
    data = _fixture_bytes()

    net1 = load_onnx_network_from_bytes(data)
    net2 = load_onnx_network_from_bytes(data)

    assert net1 is not net2


def test_load_onnx_network_from_bytes_integration_forward() -> None:
    net = load_onnx_network_from_bytes(_fixture_bytes())

    _run_identity_forward(net)


def test_load_onnx_network_from_bytes_survives_buffer_deletion_and_gc() -> None:
    data = bytearray(_fixture_bytes())
    net = load_onnx_network_from_bytes(bytes(data))
    del data
    gc.collect()

    _run_identity_forward(net)


# --- bytes loader: validation errors ---


def test_load_onnx_network_from_bytes_rejects_empty_bytes() -> None:
    with pytest.raises(ValueError, match="empty"):
        load_onnx_network_from_bytes(b"")


def test_load_onnx_network_from_bytes_rejects_random_bytes() -> None:
    with pytest.raises(RuntimeError):
        load_onnx_network_from_bytes(bytes([1, 2, 3, 4, 5]) * 20)


def test_load_onnx_network_from_bytes_rejects_truncated_fixture() -> None:
    data = _fixture_bytes()

    with pytest.raises(RuntimeError):
        load_onnx_network_from_bytes(data[: len(data) // 2])


@pytest.mark.parametrize(
    "make_bad_input",
    [
        lambda data: bytearray(data),
        lambda data: memoryview(data),
        lambda data: np.frombuffer(data, dtype=np.uint8),
        lambda data: list(data),
        lambda data: tuple(data),
        lambda data: data.decode("latin1"),
        lambda data: FIXTURE_PATH,
    ],
)
def test_load_onnx_network_from_bytes_rejects_non_bytes_types(make_bad_input) -> None:
    data = _fixture_bytes()

    with pytest.raises(TypeError, match="data"):
        load_onnx_network_from_bytes(make_bad_input(data))  # type: ignore[arg-type]


def test_load_onnx_network_from_bytes_forbids_opencv_call_after_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_read_net_from_onnx(monkeypatch)

    with pytest.raises(ValueError):
        load_onnx_network_from_bytes(b"")


# --- postconditions (monkeypatched) ---


@pytest.mark.parametrize(
    "fake_result_factory",
    [
        lambda: None,
        lambda: 42,
        lambda: object(),
        lambda: cv2.dnn.Net(),  # a real but empty Net
    ],
)
def test_load_onnx_network_postcondition_failure_raises_runtime_error(
    fake_result_factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cv2.dnn, "readNetFromONNX", lambda *a, **k: fake_result_factory())

    with pytest.raises(RuntimeError):
        load_onnx_network(FIXTURE_PATH)


def test_load_onnx_network_postcondition_empty_raising_becomes_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BoomOnEmpty:
        def empty(self):
            raise cv2.error("synthetic empty() failure")

    monkeypatch.setattr(cv2.dnn, "readNetFromONNX", lambda *a, **k: _BoomOnEmpty())
    monkeypatch.setattr(cv2.dnn, "Net", _BoomOnEmpty)

    with pytest.raises(RuntimeError):
        load_onnx_network(FIXTURE_PATH)


def test_load_onnx_network_wraps_cv2_error_as_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = cv2.error("synthetic failure")

    def boom(*args, **kwargs):
        raise original

    monkeypatch.setattr(cv2.dnn, "readNetFromONNX", boom)

    with pytest.raises(RuntimeError) as exc_info:
        load_onnx_network(FIXTURE_PATH)

    assert exc_info.value.__cause__ is original


def test_load_onnx_network_from_bytes_wraps_cv2_error_as_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = cv2.error("synthetic failure")

    def boom(*args, **kwargs):
        raise original

    monkeypatch.setattr(cv2.dnn, "readNetFromONNX", boom)

    with pytest.raises(RuntimeError) as exc_info:
        load_onnx_network_from_bytes(_fixture_bytes())

    assert exc_info.value.__cause__ is original


# --- top-level export ---


def test_load_onnx_network_functions_exported_from_top_level_package() -> None:
    assert im.load_onnx_network is load_onnx_network
    assert im.load_onnx_network_from_bytes is load_onnx_network_from_bytes
