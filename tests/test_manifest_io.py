import os
from pathlib import Path
from typing import IO

import pytest

import improcv.manifest as manifest_module
from improcv.hashing import PerceptualHash, PerceptualHashAlgorithm
from improcv.manifest import PerceptualHashManifest

_PHASH = PerceptualHashAlgorithm.PHASH


def _hash(hex_value: str, *, hash_size: int = 2) -> PerceptualHash:
    """Build a `PerceptualHash` with an easily hand-verifiable value (see test_manifest.py)."""
    return PerceptualHash.from_hex(hex_value, algorithm=_PHASH, hash_size=hash_size)


def _manifest(hashes: dict[str, PerceptualHash] | None = None, *, hash_size: int = 2):
    return PerceptualHashManifest.from_hashes(hashes or {}, algorithm=_PHASH, hash_size=hash_size)


def _try_symlink(link_path: Path, target: Path) -> bool:
    """Attempt to create a symlink; return False (and create nothing) if the platform refuses."""
    try:
        link_path.symlink_to(target)
    except (OSError, NotImplementedError):
        return False
    return True


# =====================================================================================
# save/load -- basic round trip
# =====================================================================================


def test_round_trip_empty_manifest(tmp_path: Path) -> None:
    manifest = _manifest()
    dest = tmp_path / "m.json"
    result = manifest.save(dest)
    assert result is None
    assert PerceptualHashManifest.load(dest) == manifest


def test_round_trip_singleton_manifest(tmp_path: Path) -> None:
    manifest = _manifest({"a.png": _hash("0")})
    dest = tmp_path / "m.json"
    manifest.save(dest)
    assert PerceptualHashManifest.load(dest) == manifest


def test_round_trip_many_entries(tmp_path: Path) -> None:
    manifest = _manifest(
        {"z.png": _hash("0"), "a.png": _hash("1"), "m.png": _hash("2"), "b.png": _hash("3")}
    )
    dest = tmp_path / "m.json"
    manifest.save(dest)
    assert PerceptualHashManifest.load(dest) == manifest


def test_round_trip_unicode_path(tmp_path: Path) -> None:
    manifest = _manifest({"dane/żółw.png": _hash("0")})
    dest = tmp_path / "m.json"
    manifest.save(dest)
    assert PerceptualHashManifest.load(dest) == manifest


def test_save_returns_none(tmp_path: Path) -> None:
    manifest = _manifest({"a.png": _hash("0")})
    assert manifest.save(tmp_path / "m.json") is None


def test_file_bytes_are_exactly_to_json_encoded(tmp_path: Path) -> None:
    manifest = _manifest({"a.png": _hash("0"), "b.png": _hash("1")})
    dest = tmp_path / "m.json"
    manifest.save(dest)
    assert dest.read_bytes() == manifest.to_json().encode("utf-8")


def test_file_has_exactly_one_trailing_lf() -> None:
    manifest = _manifest({"a.png": _hash("0")})
    text = manifest.to_json()
    assert text.endswith("\n")
    assert not text.endswith("\n\n")


def test_file_has_no_bom(tmp_path: Path) -> None:
    manifest = _manifest({"a.png": _hash("0")})
    dest = tmp_path / "m.json"
    manifest.save(dest)
    assert not dest.read_bytes().startswith(b"\xef\xbb\xbf")


def test_repeated_overwrite_save_gives_identical_bytes(tmp_path: Path) -> None:
    manifest = _manifest({"a.png": _hash("0"), "b.png": _hash("1")})
    dest = tmp_path / "m.json"
    manifest.save(dest)
    first_bytes = dest.read_bytes()
    manifest.save(dest, overwrite=True)
    second_bytes = dest.read_bytes()
    assert first_bytes == second_bytes == manifest.to_json().encode("utf-8")


def test_load_to_json_preserves_canonical_output(tmp_path: Path) -> None:
    manifest = _manifest({"z.png": _hash("0"), "a.png": _hash("1")})
    dest = tmp_path / "m.json"
    manifest.save(dest)
    restored = PerceptualHashManifest.load(dest)
    assert restored.to_json() == manifest.to_json()


# =====================================================================================
# save/load -- file path argument types
# =====================================================================================


class _StrPath:
    def __init__(self, name: str) -> None:
        self._name = name

    def __fspath__(self) -> str:
        return self._name


class _BytesPath:
    def __fspath__(self) -> bytes:
        return b"m.json"


def test_save_accepts_str_path(tmp_path: Path) -> None:
    manifest = _manifest({"a.png": _hash("0")})
    dest = str(tmp_path / "m.json")
    manifest.save(dest)
    assert Path(dest).exists()


def test_save_accepts_path_object(tmp_path: Path) -> None:
    manifest = _manifest({"a.png": _hash("0")})
    dest = tmp_path / "m.json"
    manifest.save(dest)
    assert dest.exists()


def test_save_accepts_custom_pathlike(tmp_path: Path) -> None:
    manifest = _manifest({"a.png": _hash("0")})
    dest = _StrPath(str(tmp_path / "m.json"))
    manifest.save(dest)
    assert (tmp_path / "m.json").exists()


def test_save_accepts_absolute_path(tmp_path: Path) -> None:
    manifest = _manifest({"a.png": _hash("0")})
    dest = tmp_path / "m.json"
    assert dest.is_absolute()
    manifest.save(dest)
    assert dest.exists()


def test_save_accepts_relative_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = _manifest({"a.png": _hash("0")})
    manifest.save("m.json")
    assert (tmp_path / "m.json").exists()


def test_load_accepts_str_path(tmp_path: Path) -> None:
    manifest = _manifest({"a.png": _hash("0")})
    dest = tmp_path / "m.json"
    manifest.save(dest)
    assert PerceptualHashManifest.load(str(dest)) == manifest


def test_load_accepts_path_object(tmp_path: Path) -> None:
    manifest = _manifest({"a.png": _hash("0")})
    dest = tmp_path / "m.json"
    manifest.save(dest)
    assert PerceptualHashManifest.load(dest) == manifest


def test_load_accepts_custom_pathlike(tmp_path: Path) -> None:
    manifest = _manifest({"a.png": _hash("0")})
    dest = tmp_path / "m.json"
    manifest.save(dest)
    assert PerceptualHashManifest.load(_StrPath(str(dest))) == manifest


def test_load_accepts_absolute_path(tmp_path: Path) -> None:
    manifest = _manifest({"a.png": _hash("0")})
    dest = tmp_path / "m.json"
    manifest.save(dest)
    assert dest.is_absolute()
    assert PerceptualHashManifest.load(dest) == manifest


def test_load_accepts_relative_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _manifest({"a.png": _hash("0")})
    manifest.save(tmp_path / "m.json")
    monkeypatch.chdir(tmp_path)
    assert PerceptualHashManifest.load("m.json") == manifest


@pytest.mark.parametrize(
    ("bad_path", "expected_exception"),
    [
        (b"m.json", TypeError),
        (_BytesPath(), TypeError),
        (object(), TypeError),
        ("", ValueError),
        ("m\x00.json", ValueError),
    ],
)
def test_save_rejects_bad_path_argument(
    tmp_path: Path, bad_path: object, expected_exception: type[Exception]
) -> None:
    manifest = _manifest({"a.png": _hash("0")})
    with pytest.raises(expected_exception):
        manifest.save(bad_path)  # type: ignore[arg-type]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("bad_path", "expected_exception"),
    [
        (b"m.json", TypeError),
        (_BytesPath(), TypeError),
        (object(), TypeError),
        ("", ValueError),
        ("m\x00.json", ValueError),
    ],
)
def test_load_rejects_bad_path_argument(
    bad_path: object, expected_exception: type[Exception]
) -> None:
    with pytest.raises(expected_exception):
        PerceptualHashManifest.load(bad_path)  # type: ignore[arg-type]


def test_save_does_not_expanduser(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    manifest = _manifest({"a.png": _hash("0")})
    dest = tmp_path / "~unexpanded.json"
    manifest.save(dest)
    assert dest.exists()


def test_save_does_not_resolve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = _manifest({"a.png": _hash("0")})
    (tmp_path / "sub").mkdir()
    manifest.save("sub/../m.json")
    assert (tmp_path / "m.json").exists()


# =====================================================================================
# save -- overwrite=False (default)
# =====================================================================================


def test_overwrite_is_false_by_default(tmp_path: Path) -> None:
    manifest = _manifest({"a.png": _hash("0")})
    dest = tmp_path / "m.json"
    manifest.save(dest)
    with pytest.raises(FileExistsError):
        manifest.save(dest)


def test_no_overwrite_rejects_bad_overwrite_type() -> None:
    manifest = _manifest({"a.png": _hash("0")})
    with pytest.raises(TypeError):
        manifest.save("m.json", overwrite=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        manifest.save("m.json", overwrite="false")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        manifest.save("m.json", overwrite=None)  # type: ignore[arg-type]


def test_no_overwrite_existing_file_raises(tmp_path: Path) -> None:
    manifest = _manifest({"a.png": _hash("0")})
    other = _manifest({"b.png": _hash("1")})
    dest = tmp_path / "m.json"
    other.save(dest)
    with pytest.raises(FileExistsError):
        manifest.save(dest)


def test_no_overwrite_existing_file_stays_byte_identical(tmp_path: Path) -> None:
    manifest = _manifest({"a.png": _hash("0")})
    other = _manifest({"b.png": _hash("1")})
    dest = tmp_path / "m.json"
    other.save(dest)
    original_bytes = dest.read_bytes()
    with pytest.raises(FileExistsError):
        manifest.save(dest)
    assert dest.read_bytes() == original_bytes


def test_no_overwrite_existing_directory_raises(tmp_path: Path) -> None:
    manifest = _manifest({"a.png": _hash("0")})
    dest = tmp_path / "adir"
    dest.mkdir()
    with pytest.raises(FileExistsError):
        manifest.save(dest)
    assert dest.is_dir()


def test_no_overwrite_dangling_symlink_raises(tmp_path: Path) -> None:
    manifest = _manifest({"a.png": _hash("0")})
    dest = tmp_path / "link.json"
    if not _try_symlink(dest, tmp_path / "does-not-exist.json"):
        pytest.skip("platform does not support creating symlinks")
    with pytest.raises(FileExistsError):
        manifest.save(dest)


def test_no_overwrite_new_destination_writes_correct_file(tmp_path: Path) -> None:
    manifest = _manifest({"a.png": _hash("0")})
    dest = tmp_path / "m.json"
    manifest.save(dest)
    assert dest.read_bytes() == manifest.to_json().encode("utf-8")


def test_no_overwrite_leaves_no_temporary_sibling(tmp_path: Path) -> None:
    manifest = _manifest({"a.png": _hash("0")})
    dest = tmp_path / "m.json"
    manifest.save(dest)
    assert [p.name for p in tmp_path.iterdir()] == ["m.json"]


def test_no_overwrite_concurrent_create_race_does_not_clobber(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest({"a.png": _hash("0")})
    dest = tmp_path / "m.json"
    concurrent_bytes = b"written by a concurrent process, not a real manifest\n"

    real_link = os.link

    def fake_link(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        # Simulate another process winning the race and creating `dest` an instant before our
        # own publish call actually runs.
        Path(dst).write_bytes(concurrent_bytes)
        real_link(src, dst)

    monkeypatch.setattr(manifest_module.os, "link", fake_link)

    with pytest.raises(FileExistsError):
        manifest.save(dest)

    assert dest.read_bytes() == concurrent_bytes
    assert [p.name for p in tmp_path.iterdir()] == ["m.json"]


# =====================================================================================
# save -- overwrite=True
# =====================================================================================


def test_overwrite_true_replaces_existing_file(tmp_path: Path) -> None:
    manifest = _manifest({"a.png": _hash("0")})
    other = _manifest({"b.png": _hash("1")})
    dest = tmp_path / "m.json"
    other.save(dest)
    manifest.save(dest, overwrite=True)
    assert PerceptualHashManifest.load(dest) == manifest


def test_overwrite_true_new_file_contains_full_manifest(tmp_path: Path) -> None:
    manifest = _manifest({"a.png": _hash("0"), "b.png": _hash("1")})
    dest = tmp_path / "m.json"
    manifest.save(dest, overwrite=True)
    assert dest.read_bytes() == manifest.to_json().encode("utf-8")


def test_overwrite_true_old_destination_intact_before_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old_manifest = _manifest({"old.png": _hash("0")})
    new_manifest = _manifest({"new.png": _hash("1")})
    dest = tmp_path / "m.json"
    old_manifest.save(dest)
    old_bytes = dest.read_bytes()

    real_replace = os.replace

    def fake_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        # Immediately before the real atomic replace runs, the destination must still hold the
        # complete old content.
        assert Path(dst).read_bytes() == old_bytes
        real_replace(src, dst)

    monkeypatch.setattr(manifest_module.os, "replace", fake_replace)

    new_manifest.save(dest, overwrite=True)
    assert dest.read_bytes() == new_manifest.to_json().encode("utf-8")


def test_overwrite_true_temp_file_has_full_new_bytes_before_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest({"a.png": _hash("0")})
    dest = tmp_path / "m.json"
    expected_bytes = manifest.to_json().encode("utf-8")

    real_replace = os.replace

    def fake_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        assert Path(src).read_bytes() == expected_bytes
        real_replace(src, dst)

    monkeypatch.setattr(manifest_module.os, "replace", fake_replace)

    manifest.save(dest, overwrite=True)


def test_overwrite_true_destination_has_full_bytes_after_publish(tmp_path: Path) -> None:
    manifest = _manifest({"a.png": _hash("0"), "b.png": _hash("1")})
    dest = tmp_path / "m.json"
    manifest.save(dest, overwrite=True)
    assert dest.read_bytes() == manifest.to_json().encode("utf-8")


def test_overwrite_true_leaves_no_temporary_sibling(tmp_path: Path) -> None:
    manifest = _manifest({"a.png": _hash("0")})
    dest = tmp_path / "m.json"
    manifest.save(dest, overwrite=True)
    assert [p.name for p in tmp_path.iterdir()] == ["m.json"]


def test_overwrite_true_replaces_symlink_without_modifying_its_target(tmp_path: Path) -> None:
    manifest = _manifest({"a.png": _hash("0")})
    real_file = tmp_path / "real.json"
    real_file.write_text("untouched original content", encoding="utf-8")
    link_path = tmp_path / "link.json"
    if not _try_symlink(link_path, real_file):
        pytest.skip("platform does not support creating symlinks")

    manifest.save(link_path, overwrite=True)

    assert not link_path.is_symlink()
    assert link_path.read_bytes() == manifest.to_json().encode("utf-8")
    assert real_file.read_text(encoding="utf-8") == "untouched original content"


# =====================================================================================
# save -- errors and cleanup
# =====================================================================================


def test_save_missing_parent_directory_raises(tmp_path: Path) -> None:
    manifest = _manifest({"a.png": _hash("0")})
    dest = tmp_path / "does" / "not" / "exist" / "m.json"
    with pytest.raises(FileNotFoundError):
        manifest.save(dest)


def test_save_parent_is_a_file_raises(tmp_path: Path) -> None:
    manifest = _manifest({"a.png": _hash("0")})
    parent = tmp_path / "not_a_dir"
    parent.write_text("just a file", encoding="utf-8")
    dest = parent / "m.json"
    with pytest.raises(OSError):
        manifest.save(dest)


def test_save_destination_is_directory_with_overwrite_true_raises(tmp_path: Path) -> None:
    manifest = _manifest({"a.png": _hash("0")})
    dest = tmp_path / "adir"
    dest.mkdir()
    (dest / "keep.txt").write_text("keep me", encoding="utf-8")
    with pytest.raises(OSError):
        manifest.save(dest, overwrite=True)
    assert dest.is_dir()
    assert (dest / "keep.txt").read_text(encoding="utf-8") == "keep me"


def test_save_fsync_failure_leaves_no_temp_sibling_and_destination_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest({"a.png": _hash("0")})
    other = _manifest({"b.png": _hash("1")})
    dest = tmp_path / "m.json"
    other.save(dest)
    original_bytes = dest.read_bytes()

    def failing_fsync(fd: int) -> None:
        raise OSError("simulated fsync failure")

    monkeypatch.setattr(manifest_module.os, "fsync", failing_fsync)

    with pytest.raises(OSError):
        manifest.save(dest, overwrite=True)

    assert dest.read_bytes() == original_bytes
    assert [p.name for p in tmp_path.iterdir()] == ["m.json"]


def test_save_link_failure_leaves_no_temp_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest({"a.png": _hash("0")})
    dest = tmp_path / "m.json"

    def failing_link(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        raise OSError("simulated link failure")

    monkeypatch.setattr(manifest_module.os, "link", failing_link)

    with pytest.raises(OSError):
        manifest.save(dest)

    assert not dest.exists()
    assert list(tmp_path.iterdir()) == []


def test_save_replace_failure_leaves_no_temp_sibling_and_destination_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest({"a.png": _hash("0")})
    other = _manifest({"b.png": _hash("1")})
    dest = tmp_path / "m.json"
    other.save(dest)
    original_bytes = dest.read_bytes()

    def failing_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(manifest_module.os, "replace", failing_replace)

    with pytest.raises(OSError):
        manifest.save(dest, overwrite=True)

    assert dest.read_bytes() == original_bytes
    assert [p.name for p in tmp_path.iterdir()] == ["m.json"]


def test_save_write_failure_leaves_no_temp_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest({"a.png": _hash("0")})
    dest = tmp_path / "m.json"

    real_fdopen = os.fdopen

    class _FailingWriteFile:
        def __init__(self, real_file: IO[str]) -> None:
            self._real = real_file

        def write(self, _data: str) -> int:
            raise OSError("simulated write failure")

        def __getattr__(self, item: str) -> object:
            return getattr(self._real, item)

        def __enter__(self) -> "_FailingWriteFile":
            return self

        def __exit__(self, *exc_info: object) -> bool:
            self._real.close()
            return False

    def fake_fdopen(fd: int, mode: str, *, encoding: str, newline: str) -> _FailingWriteFile:
        return _FailingWriteFile(real_fdopen(fd, mode, encoding=encoding, newline=newline))

    monkeypatch.setattr(manifest_module.os, "fdopen", fake_fdopen)

    with pytest.raises(OSError):
        manifest.save(dest)

    assert not dest.exists()
    assert list(tmp_path.iterdir()) == []


# =====================================================================================
# load
# =====================================================================================


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        PerceptualHashManifest.load(tmp_path / "does-not-exist.json")


def test_load_invalid_utf8_raises(tmp_path: Path) -> None:
    dest = tmp_path / "m.json"
    dest.write_bytes(b"\xff\xfe not valid utf-8")
    with pytest.raises(UnicodeDecodeError):
        PerceptualHashManifest.load(dest)


def test_load_empty_file_raises_value_error(tmp_path: Path) -> None:
    dest = tmp_path / "m.json"
    dest.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        PerceptualHashManifest.load(dest)


def test_load_malformed_json_raises_value_error(tmp_path: Path) -> None:
    dest = tmp_path / "m.json"
    dest.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ValueError):
        PerceptualHashManifest.load(dest)


def test_load_unknown_schema_version_raises_value_error(tmp_path: Path) -> None:
    dest = tmp_path / "m.json"
    dest.write_text(
        '{"schema_version": 2, "algorithm": "phash", "hash_size": 2, "entries": []}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        PerceptualHashManifest.load(dest)


def test_load_invalid_entry_raises_value_error(tmp_path: Path) -> None:
    dest = tmp_path / "m.json"
    dest.write_text(
        '{"schema_version": 1, "algorithm": "phash", "hash_size": 2, '
        '"entries": [{"path": "a.png", "hash": "zz"}]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        PerceptualHashManifest.load(dest)


def test_load_crlf_file_parses_but_to_json_stays_canonical_lf(tmp_path: Path) -> None:
    manifest = _manifest({"a.png": _hash("0"), "b.png": _hash("1")})
    crlf_text = manifest.to_json().replace("\n", "\r\n")
    dest = tmp_path / "m.json"
    dest.write_bytes(crlf_text.encode("utf-8"))

    restored = PerceptualHashManifest.load(dest)

    assert restored == manifest
    assert restored.to_json() == manifest.to_json()
    assert "\r" not in restored.to_json()


def test_load_delegates_to_from_json_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest({"a.png": _hash("0")})
    dest = tmp_path / "m.json"
    manifest.save(dest)
    expected_text = manifest.to_json()

    real_from_json = PerceptualHashManifest.from_json.__func__
    calls: list[str] = []

    def fake_from_json(cls: type[PerceptualHashManifest], text: str) -> PerceptualHashManifest:
        calls.append(text)
        return real_from_json(cls, text)

    monkeypatch.setattr(PerceptualHashManifest, "from_json", classmethod(fake_from_json))

    restored = PerceptualHashManifest.load(dest)

    assert calls == [expected_text]
    assert restored == manifest


# =====================================================================================
# save -- atomic publication order
# =====================================================================================


def test_save_no_overwrite_publication_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest({"a.png": _hash("0")})
    dest = tmp_path / "m.json"
    expected_text = manifest.to_json()
    events: list[str] = []

    real_fsync = os.fsync

    def fake_fsync(fd: int) -> None:
        # A separate reader must already see the complete, flushed content before fsync runs --
        # this proves write() and flush() both completed first.
        candidates = list(tmp_path.glob(f".{dest.name}.*.tmp"))
        assert len(candidates) == 1
        assert candidates[0].read_text(encoding="utf-8") == expected_text
        events.append("fsync")
        real_fsync(fd)

    real_link = os.link

    def fake_link(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        events.append("publish")
        real_link(src, dst)

    real_unlink = Path.unlink

    def fake_unlink(self: Path, missing_ok: bool = False) -> None:
        events.append("cleanup")
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(manifest_module.os, "fsync", fake_fsync)
    monkeypatch.setattr(manifest_module.os, "link", fake_link)
    monkeypatch.setattr(Path, "unlink", fake_unlink)

    manifest.save(dest)

    assert events == ["fsync", "publish", "cleanup"]
    assert dest.read_bytes() == expected_text.encode("utf-8")


def test_save_overwrite_true_publication_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest({"a.png": _hash("0")})
    dest = tmp_path / "m.json"
    expected_text = manifest.to_json()
    events: list[str] = []

    real_fsync = os.fsync

    def fake_fsync(fd: int) -> None:
        candidates = list(tmp_path.glob(f".{dest.name}.*.tmp"))
        assert len(candidates) == 1
        assert candidates[0].read_text(encoding="utf-8") == expected_text
        events.append("fsync")
        real_fsync(fd)

    real_replace = os.replace

    def fake_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        events.append("publish")
        real_replace(src, dst)

    real_unlink = Path.unlink

    def fake_unlink(self: Path, missing_ok: bool = False) -> None:
        events.append("cleanup")
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(manifest_module.os, "fsync", fake_fsync)
    monkeypatch.setattr(manifest_module.os, "replace", fake_replace)
    monkeypatch.setattr(Path, "unlink", fake_unlink)

    manifest.save(dest, overwrite=True)

    assert events == ["fsync", "publish", "cleanup"]
    assert dest.read_bytes() == expected_text.encode("utf-8")


# =====================================================================================
# security and robustness
# =====================================================================================


def test_save_rejects_nul_in_path_before_creating_anything(tmp_path: Path) -> None:
    manifest = _manifest({"a.png": _hash("0")})
    with pytest.raises(ValueError):
        manifest.save(str(tmp_path / "m\x00.json"))
    assert list(tmp_path.iterdir()) == []


def test_no_overwrite_does_not_follow_destination_symlink(tmp_path: Path) -> None:
    manifest = _manifest({"a.png": _hash("0")})
    real_file = tmp_path / "real.json"
    real_file.write_text("original", encoding="utf-8")
    link_path = tmp_path / "link.json"
    if not _try_symlink(link_path, real_file):
        pytest.skip("platform does not support creating symlinks")

    with pytest.raises(FileExistsError):
        manifest.save(link_path)

    assert real_file.read_text(encoding="utf-8") == "original"
    assert link_path.is_symlink()


def test_temporary_file_stays_in_destination_directory(tmp_path: Path) -> None:
    manifest = _manifest({"a.png": _hash("0")})
    sub = tmp_path / "sub"
    sub.mkdir()
    dest = sub / "m.json"
    manifest.save(dest)
    assert [p.name for p in tmp_path.iterdir()] == ["sub"]
    assert [p.name for p in sub.iterdir()] == ["m.json"]


def test_temporary_filename_is_not_the_raw_full_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest({"a.png": _hash("0")})
    nested = tmp_path / "a" / "b" / "c"
    nested.mkdir(parents=True)
    dest = nested / "m.json"

    seen_candidates: list[str] = []
    real_open = os.open

    def recording_open(path: str | os.PathLike[str], flags: int, mode: int = 0o777) -> int:
        seen_candidates.append(os.fspath(path))
        return real_open(path, flags, mode)

    monkeypatch.setattr(manifest_module.os, "open", recording_open)

    manifest.save(dest)

    assert len(seen_candidates) == 1
    temp_name = Path(seen_candidates[0]).name
    full_path_str = str(dest)
    assert full_path_str not in temp_name
    assert temp_name.startswith(".m.json.")


def test_save_error_message_does_not_leak_manifest_content(tmp_path: Path) -> None:
    manifest = _manifest({"very-secret-image-name.png": _hash("0")})
    dest = tmp_path / "adir"
    dest.mkdir()
    with pytest.raises(FileExistsError) as exc_info:
        manifest.save(dest)
    assert "very-secret-image-name" not in str(exc_info.value)


def test_save_does_not_write_to_image_identifier_paths(tmp_path: Path) -> None:
    manifest = _manifest({"a.png": _hash("0")})
    manifest.save(tmp_path / "m.json")
    assert not (tmp_path / "a.png").exists()
