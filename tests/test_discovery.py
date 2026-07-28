import os
import platform
import stat as stat_module
from pathlib import Path

import pytest

from improcv.discovery import discover_images


def _touch(path: Path) -> None:
    path.write_bytes(b"")


# --- root ---


def test_discover_images_accepts_str(tmp_path: Path) -> None:
    _touch(tmp_path / "a.jpg")
    result = discover_images(str(tmp_path))
    assert result == (tmp_path / "a.jpg",)


def test_discover_images_accepts_path(tmp_path: Path) -> None:
    _touch(tmp_path / "a.jpg")
    result = discover_images(tmp_path)
    assert result == (tmp_path / "a.jpg",)


def test_discover_images_accepts_custom_pathlike(tmp_path: Path) -> None:
    class CustomPathLike:
        def __fspath__(self) -> str:
            return str(tmp_path)

    _touch(tmp_path / "a.jpg")
    result = discover_images(CustomPathLike())
    assert result == (tmp_path / "a.jpg",)


def test_discover_images_rejects_bytes(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="bytes"):
        discover_images(bytes(tmp_path))  # type: ignore[arg-type]


def test_discover_images_rejects_custom_pathlike_returning_bytes(tmp_path: Path) -> None:
    class BytesPathLike:
        def __fspath__(self) -> bytes:
            return bytes(str(tmp_path), "utf-8")

    with pytest.raises(TypeError, match="bytes"):
        discover_images(BytesPathLike())  # type: ignore[arg-type]


def test_discover_images_rejects_non_pathlike_type() -> None:
    with pytest.raises(TypeError, match="root"):
        discover_images(42)  # type: ignore[arg-type]


def test_discover_images_rejects_empty_string() -> None:
    with pytest.raises(ValueError, match="empty"):
        discover_images("")


def test_discover_images_accepts_relative_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "sub").mkdir()
    _touch(tmp_path / "sub" / "a.jpg")
    monkeypatch.chdir(tmp_path)
    result = discover_images("sub")
    assert result == (Path("sub") / "a.jpg",)


def test_discover_images_absolute_root_gives_absolute_results(tmp_path: Path) -> None:
    _touch(tmp_path / "a.jpg")
    result = discover_images(tmp_path)
    assert result[0].is_absolute()


def test_discover_images_accepts_unicode_directory_and_file_names(tmp_path: Path) -> None:
    unicode_dir = tmp_path / "zdjęcia_dataset"
    unicode_dir.mkdir()
    _touch(unicode_dir / "kot_ą.jpg")
    result = discover_images(unicode_dir)
    assert result == (unicode_dir / "kot_ą.jpg",)


def test_discover_images_empty_directory_returns_empty_tuple(tmp_path: Path) -> None:
    assert discover_images(tmp_path) == ()


def test_discover_images_rejects_missing_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_images(tmp_path / "does_not_exist")


def test_discover_images_rejects_regular_file_as_root(tmp_path: Path) -> None:
    file_path = tmp_path / "a.jpg"
    _touch(file_path)
    with pytest.raises(NotADirectoryError):
        discover_images(file_path)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFOs are not available on this platform")
def test_discover_images_rejects_fifo_as_root(tmp_path: Path) -> None:
    fifo_path = tmp_path / "stream"
    os.mkfifo(fifo_path)
    with pytest.raises(NotADirectoryError):
        discover_images(fifo_path)


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="os.symlink requires elevated privileges on Windows by default",
)
def test_discover_images_rejects_broken_root_symlink(tmp_path: Path) -> None:
    link = tmp_path / "broken_root"
    link.symlink_to(tmp_path / "does_not_exist")
    with pytest.raises(FileNotFoundError):
        discover_images(link)


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="os.symlink requires elevated privileges on Windows by default",
)
def test_discover_images_accepts_root_symlink_to_directory(tmp_path: Path) -> None:
    real_dir = tmp_path / "real_dir"
    real_dir.mkdir()
    _touch(real_dir / "a.jpg")
    link = tmp_path / "linked_root"
    link.symlink_to(real_dir, target_is_directory=True)

    result = discover_images(link)
    assert result == (link / "a.jpg",)


# --- extensions ---


def test_discover_images_default_extensions(tmp_path: Path) -> None:
    names = ["a.jpg", "b.jpeg", "c.png", "d.bmp", "e.tif", "f.tiff", "g.webp", "h.gif"]
    for name in names:
        _touch(tmp_path / name)
    result = discover_images(tmp_path, recursive=False)
    found_names = {p.name for p in result}
    assert found_names == set(names) - {"h.gif"}


@pytest.mark.parametrize("container", [list, tuple, set, frozenset])
def test_discover_images_accepts_various_collection_types(tmp_path: Path, container) -> None:
    _touch(tmp_path / "a.png")
    _touch(tmp_path / "b.jpg")
    result = discover_images(tmp_path, extensions=container([".png"]))
    assert result == (tmp_path / "a.png",)


def test_discover_images_accepts_custom_collection_class(tmp_path: Path) -> None:
    class CustomCollection:
        def __init__(self, items):
            self._items = list(items)

        def __len__(self):
            return len(self._items)

        def __iter__(self):
            return iter(self._items)

        def __contains__(self, item):
            return item in self._items

    _touch(tmp_path / "a.png")
    result = discover_images(tmp_path, extensions=CustomCollection([".png"]))
    assert result == (tmp_path / "a.png",)


def test_discover_images_extensions_none_uses_default(tmp_path: Path) -> None:
    _touch(tmp_path / "a.jpg")
    assert discover_images(tmp_path, extensions=None) == (tmp_path / "a.jpg",)


def test_discover_images_extensions_without_dot(tmp_path: Path) -> None:
    _touch(tmp_path / "a.jpg")
    assert discover_images(tmp_path, extensions=["jpg"]) == (tmp_path / "a.jpg",)


def test_discover_images_extensions_with_dot(tmp_path: Path) -> None:
    _touch(tmp_path / "a.jpg")
    assert discover_images(tmp_path, extensions=[".jpg"]) == (tmp_path / "a.jpg",)


def test_discover_images_extensions_uppercase_input(tmp_path: Path) -> None:
    _touch(tmp_path / "a.jpg")
    assert discover_images(tmp_path, extensions=[".JPG"]) == (tmp_path / "a.jpg",)


def test_discover_images_extensions_uppercase_filename(tmp_path: Path) -> None:
    _touch(tmp_path / "a.JPG")
    assert discover_images(tmp_path, extensions=[".jpg"]) == (tmp_path / "a.JPG",)


def test_discover_images_extensions_duplicates_deduplicated(tmp_path: Path) -> None:
    _touch(tmp_path / "a.jpg")
    result = discover_images(tmp_path, extensions=["jpg", ".jpg", ".JPG", "JPG"])
    assert result == (tmp_path / "a.jpg",)


def test_discover_images_rejects_empty_extensions_collection() -> None:
    with pytest.raises(ValueError, match="empty"):
        discover_images(".", extensions=[])


@pytest.mark.parametrize("bad", ["jpg", b"jpg", bytearray(b"jpg")])
def test_discover_images_rejects_bare_str_bytes_bytearray_extensions(bad) -> None:
    with pytest.raises(TypeError):
        discover_images(".", extensions=bad)  # type: ignore[arg-type]


def test_discover_images_rejects_mapping_extensions() -> None:
    with pytest.raises(TypeError, match="Mapping"):
        discover_images(".", extensions={"jpg": 1})  # type: ignore[arg-type]


def test_discover_images_rejects_generator_extensions() -> None:
    with pytest.raises(TypeError):
        discover_images(".", extensions=(e for e in ["jpg"]))  # type: ignore[arg-type]


def test_discover_images_rejects_non_str_extension_element() -> None:
    with pytest.raises(TypeError):
        discover_images(".", extensions=[123])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad_extension",
    ["", ".", " jpg", "jpg ", "j pg", "a/b", "a\\b", "a*b", "a?b", "a[b", "a]b", "a\x00b"],
)
def test_discover_images_rejects_illegal_extension_values(bad_extension: str) -> None:
    with pytest.raises(ValueError):
        discover_images(".", extensions=[bad_extension])


def test_discover_images_does_not_mutate_input_extensions_collection(tmp_path: Path) -> None:
    extensions = [".jpg", ".png"]
    before = list(extensions)
    discover_images(tmp_path, extensions=extensions)
    assert extensions == before


def test_discover_images_multi_segment_extension(tmp_path: Path) -> None:
    _touch(tmp_path / "scan.nii.gz")
    _touch(tmp_path / "scan.gz")
    result = discover_images(tmp_path, extensions=[".nii.gz"], recursive=False)
    assert result == (tmp_path / "scan.nii.gz",)


# --- bool contract ---


@pytest.mark.parametrize("bad", [1, 0, None, "true", 1.0])
def test_discover_images_rejects_non_bool_recursive(tmp_path: Path, bad) -> None:
    with pytest.raises(TypeError, match="recursive"):
        discover_images(tmp_path, recursive=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [1, 0, None, "true"])
def test_discover_images_rejects_non_bool_include_hidden(tmp_path: Path, bad) -> None:
    with pytest.raises(TypeError, match="include_hidden"):
        discover_images(tmp_path, include_hidden=bad)  # type: ignore[arg-type]


def test_discover_images_rejects_numpy_bool(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    with pytest.raises(TypeError, match="recursive"):
        discover_images(tmp_path, recursive=np.bool_(True))  # type: ignore[arg-type]


# --- discovery: recursion, hidden, layout ---


def test_discover_images_recursive_true_finds_nested_files(tmp_path: Path) -> None:
    (tmp_path / "sub" / "deeper").mkdir(parents=True)
    _touch(tmp_path / "a.jpg")
    _touch(tmp_path / "sub" / "b.jpg")
    _touch(tmp_path / "sub" / "deeper" / "c.jpg")

    result = discover_images(tmp_path, recursive=True)
    assert result == (
        tmp_path / "a.jpg",
        tmp_path / "sub" / "b.jpg",
        tmp_path / "sub" / "deeper" / "c.jpg",
    )


def test_discover_images_recursive_false_only_direct_children(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    _touch(tmp_path / "a.jpg")
    _touch(tmp_path / "sub" / "b.jpg")

    result = discover_images(tmp_path, recursive=False)
    assert result == (tmp_path / "a.jpg",)


def test_discover_images_hidden_file_skipped_by_default(tmp_path: Path) -> None:
    _touch(tmp_path / ".hidden.jpg")
    _touch(tmp_path / "visible.jpg")
    result = discover_images(tmp_path)
    assert result == (tmp_path / "visible.jpg",)


def test_discover_images_hidden_directory_skipped_by_default(tmp_path: Path) -> None:
    (tmp_path / ".hiddendir").mkdir()
    _touch(tmp_path / ".hiddendir" / "a.jpg")
    _touch(tmp_path / "visible.jpg")
    result = discover_images(tmp_path)
    assert result == (tmp_path / "visible.jpg",)


def test_discover_images_include_hidden_true_finds_hidden_entries(tmp_path: Path) -> None:
    (tmp_path / ".hiddendir").mkdir()
    _touch(tmp_path / ".hiddendir" / "a.jpg")
    _touch(tmp_path / ".hidden.jpg")
    _touch(tmp_path / "visible.jpg")

    result = discover_images(tmp_path, include_hidden=True)
    assert result == (
        tmp_path / ".hidden.jpg",
        tmp_path / ".hiddendir" / "a.jpg",
        tmp_path / "visible.jpg",
    )


def test_discover_images_root_dot_directory_searched_even_without_include_hidden(
    tmp_path: Path,
) -> None:
    dot_root = tmp_path / ".dataset"
    dot_root.mkdir()
    _touch(dot_root / "a.jpg")

    result = discover_images(dot_root, include_hidden=False)
    assert result == (dot_root / "a.jpg",)


def test_discover_images_directory_named_like_image_file_is_skipped(tmp_path: Path) -> None:
    (tmp_path / "photo.jpg").mkdir()
    _touch(tmp_path / "real.jpg")
    result = discover_images(tmp_path)
    assert result == (tmp_path / "real.jpg",)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFOs are not available on this platform")
def test_discover_images_non_regular_entry_is_skipped(tmp_path: Path) -> None:
    os.mkfifo(tmp_path / "stream.png")
    _touch(tmp_path / "real.png")
    result = discover_images(tmp_path)
    assert result == (tmp_path / "real.png",)


def test_discover_images_file_without_extension_skipped(tmp_path: Path) -> None:
    _touch(tmp_path / "noext")
    _touch(tmp_path / "real.jpg")
    result = discover_images(tmp_path)
    assert result == (tmp_path / "real.jpg",)


def test_discover_images_dotfile_extension_hidden_by_default_but_found_when_included(
    tmp_path: Path,
) -> None:
    _touch(tmp_path / ".jpg")
    assert discover_images(tmp_path) == ()
    assert discover_images(tmp_path, include_hidden=True) == (tmp_path / ".jpg",)


def test_discover_images_uppercase_extension_found(tmp_path: Path) -> None:
    _touch(tmp_path / "photo.JPG")
    assert discover_images(tmp_path) == (tmp_path / "photo.JPG",)


def test_discover_images_double_extension_found(tmp_path: Path) -> None:
    _touch(tmp_path / "double.tar.jpg")
    assert discover_images(tmp_path) == (tmp_path / "double.tar.jpg",)


def test_discover_images_temporary_suffix_skipped(tmp_path: Path) -> None:
    _touch(tmp_path / "photo.jpg.tmp")
    assert discover_images(tmp_path) == ()


def test_discover_images_fake_non_image_bytes_still_discovered(tmp_path: Path) -> None:
    (tmp_path / "not_really_an_image.jpg").write_bytes(b"this is not image data")
    assert discover_images(tmp_path) == (tmp_path / "not_really_an_image.jpg",)


def test_discover_images_empty_file_still_discovered(tmp_path: Path) -> None:
    _touch(tmp_path / "empty.jpg")
    assert discover_images(tmp_path) == (tmp_path / "empty.jpg",)


def test_discover_images_deterministic_global_order(tmp_path: Path) -> None:
    (tmp_path / "b_dir").mkdir()
    (tmp_path / "a_dir").mkdir()
    _touch(tmp_path / "z.jpg")
    _touch(tmp_path / "a_dir" / "y.jpg")
    _touch(tmp_path / "b_dir" / "x.jpg")

    expected = tuple(
        sorted(
            [
                tmp_path / "z.jpg",
                tmp_path / "a_dir" / "y.jpg",
                tmp_path / "b_dir" / "x.jpg",
            ],
            key=lambda p: p.relative_to(tmp_path).as_posix(),
        )
    )
    assert discover_images(tmp_path) == expected


def test_discover_images_order_independent_of_extensions_argument_order(tmp_path: Path) -> None:
    _touch(tmp_path / "a.png")
    _touch(tmp_path / "b.jpg")
    result_1 = discover_images(tmp_path, extensions=[".jpg", ".png"])
    result_2 = discover_images(tmp_path, extensions=[".png", ".jpg"])
    assert result_1 == result_2 == (tmp_path / "a.png", tmp_path / "b.jpg")


def test_discover_images_returns_tuple(tmp_path: Path) -> None:
    _touch(tmp_path / "a.jpg")
    result = discover_images(tmp_path)
    assert isinstance(result, tuple)


def test_discover_images_all_elements_are_path(tmp_path: Path) -> None:
    _touch(tmp_path / "a.jpg")
    _touch(tmp_path / "b.png")
    result = discover_images(tmp_path)
    assert all(isinstance(p, Path) for p in result)


def test_discover_images_relative_root_gives_relative_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _touch(tmp_path / "a.jpg")
    monkeypatch.chdir(tmp_path)
    result = discover_images(".")
    assert not result[0].is_absolute()
    assert result[0] == Path(".") / "a.jpg"


# --- symlinks / reparse points ---


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="os.symlink requires elevated privileges on Windows by default",
)
def test_discover_images_symlinked_file_skipped(tmp_path: Path) -> None:
    real = tmp_path / "real.jpg"
    _touch(real)
    link = tmp_path / "link.jpg"
    link.symlink_to(real)

    result = discover_images(tmp_path)
    assert result == (tmp_path / "real.jpg",)


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="os.symlink requires elevated privileges on Windows by default",
)
def test_discover_images_symlinked_directory_skipped(tmp_path: Path) -> None:
    real_dir = tmp_path / "real_dir"
    real_dir.mkdir()
    _touch(real_dir / "a.jpg")
    link_dir = tmp_path / "link_dir"
    link_dir.symlink_to(real_dir, target_is_directory=True)

    result = discover_images(tmp_path)
    assert result == (tmp_path / "real_dir" / "a.jpg",)


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="os.symlink requires elevated privileges on Windows by default",
)
def test_discover_images_broken_symlink_skipped(tmp_path: Path) -> None:
    link = tmp_path / "broken.jpg"
    link.symlink_to(tmp_path / "does_not_exist.jpg")
    _touch(tmp_path / "real.jpg")

    result = discover_images(tmp_path)
    assert result == (tmp_path / "real.jpg",)


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="os.symlink requires elevated privileges on Windows by default",
)
def test_discover_images_symlink_cycle_not_infinite(tmp_path: Path) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    _touch(sub / "a.jpg")
    cycle_link = sub / "back_to_root"
    cycle_link.symlink_to(tmp_path, target_is_directory=True)

    result = discover_images(tmp_path)
    assert result == (tmp_path / "sub" / "a.jpg",)


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="os.symlink requires elevated privileges on Windows by default",
)
def test_discover_images_symlink_pointing_outside_root_skipped(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"outside_{tmp_path.name}"
    outside.mkdir()
    _touch(outside / "outside.jpg")
    root = tmp_path / "root"
    root.mkdir()
    link = root / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
        result = discover_images(root)
        assert result == ()
    finally:
        import shutil

        shutil.rmtree(outside, ignore_errors=True)


def test_discover_images_windows_reparse_point_skipped_via_synthetic_attribute(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Simulates a Windows junction by monkeypatching os.stat's result.

    A real junction can't be created portably/reliably in CI without
    elevated privileges, so this exercises the `_is_reparse_point` branch
    directly via a monkeypatched fresh `os.stat(entry.path,
    follow_symlinks=False)` result (not `DirEntry.stat`, which production
    code must not use -- see the fail-fast tests below), independent of the
    host platform.
    """
    import improcv.discovery as discovery_module

    real_dir = tmp_path / "real_dir"
    real_dir.mkdir()
    _touch(real_dir / "a.jpg")
    fake_junction_dir = tmp_path / "junction"
    fake_junction_dir.mkdir()
    _touch(fake_junction_dir / "b.jpg")

    monkeypatch.setattr(stat_module, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400, raising=False)

    real_os_stat = discovery_module.os.stat

    class FakeStatResult:
        def __init__(self, real_stat, attributes):
            self._real_stat = real_stat
            self.st_file_attributes = attributes

        def __getattr__(self, item):
            return getattr(self._real_stat, item)

    def patched_stat(path, *args, **kwargs):
        real_stat = real_os_stat(path, *args, **kwargs)
        attributes = 0x400 if Path(path).name == "junction" else 0
        return FakeStatResult(real_stat, attributes)

    monkeypatch.setattr(discovery_module.os, "stat", patched_stat)

    result = discover_images(tmp_path)
    assert result == (tmp_path / "real_dir" / "a.jpg",)


# --- fail-fast ---


def test_discover_images_propagates_permission_error_from_scandir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import improcv.discovery as discovery_module

    original = PermissionError(13, "synthetic permission denial", str(tmp_path))

    def boom(path):
        raise original

    monkeypatch.setattr(discovery_module.os, "scandir", boom)

    with pytest.raises(PermissionError) as exc_info:
        discover_images(tmp_path)
    assert exc_info.value is original


def test_discover_images_propagates_permission_error_from_fresh_stat(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import improcv.discovery as discovery_module

    _touch(tmp_path / "a.jpg")
    original = PermissionError(13, "synthetic permission denial", str(tmp_path / "a.jpg"))

    def boom(path, *args, **kwargs):
        raise original

    monkeypatch.setattr(discovery_module.os, "stat", boom)

    with pytest.raises(PermissionError) as exc_info:
        discover_images(tmp_path)
    assert exc_info.value is original


def test_discover_images_hidden_entry_skips_stat_call_when_not_included(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import improcv.discovery as discovery_module

    _touch(tmp_path / ".hidden.jpg")
    _touch(tmp_path / "visible.jpg")

    real_stat = discovery_module.os.stat

    def spy_stat(path, *args, **kwargs):
        if Path(path).name == ".hidden.jpg":
            pytest.fail("hidden entries must be skipped before any stat call")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(discovery_module.os, "stat", spy_stat)

    result = discover_images(tmp_path, include_hidden=False)
    assert result == (tmp_path / "visible.jpg",)


def test_discover_images_never_calls_direntry_stat(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """DirEntry.stat may return cached metadata from directory enumeration
    (notably on Windows), not a fresh system call -- production code must
    classify every descendant via a fresh `os.stat(entry.path,
    follow_symlinks=False)` instead. This is a real `os.scandir` (not a
    fake entry), with only `DirEntry.stat` itself patched to fail the test
    if called at all.
    """
    _touch(tmp_path / "a.jpg")
    import os as os_module

    real_direntry_stat = os_module.DirEntry.stat

    def forbidden_stat(self, *args, **kwargs):
        pytest.fail("discover_images must use fresh os.stat, not cached DirEntry.stat")

    monkeypatch.setattr(os_module.DirEntry, "stat", forbidden_stat)
    try:
        result = discover_images(tmp_path)
    finally:
        monkeypatch.setattr(os_module.DirEntry, "stat", real_direntry_stat)
    assert result == (tmp_path / "a.jpg",)


def test_discover_images_propagates_error_from_fresh_stat_on_disappearing_entry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import improcv.discovery as discovery_module

    ghost_path = tmp_path / "ghost.jpg"
    real_stat = os.stat
    original = FileNotFoundError(2, "synthetic disappearing entry", os.fspath(ghost_path))

    def patched_stat(path, *args, **kwargs):
        if os.fspath(path) == os.fspath(ghost_path):
            raise original
        return real_stat(path, *args, **kwargs)

    class FakeEntry:
        name = "ghost.jpg"
        path = str(ghost_path)

        def stat(self, *args, **kwargs):
            pytest.fail("discover_images must use fresh os.stat, not cached DirEntry.stat")

    class FakeCM:
        def __enter__(self):
            return [FakeEntry()]

        def __exit__(self, *args):
            return False

    def patched_scandir(path):
        return FakeCM()

    monkeypatch.setattr(discovery_module.os, "scandir", patched_scandir)
    monkeypatch.setattr(discovery_module.os, "stat", patched_stat)

    with pytest.raises(FileNotFoundError) as exc_info:
        discover_images(tmp_path)
    assert exc_info.value is original


@pytest.mark.skipif(
    platform.system() == "Windows" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="chmod-based permission denial is unreliable on Windows and as root",
)
def test_discover_images_propagates_real_permission_error(tmp_path: Path) -> None:
    restricted = tmp_path / "restricted"
    restricted.mkdir()
    _touch(restricted / "a.jpg")
    restricted.chmod(0o000)
    try:
        with pytest.raises(PermissionError):
            discover_images(tmp_path)
    finally:
        restricted.chmod(0o755)


# --- import hygiene ---


def test_discover_images_exported_from_top_level_package(tmp_path: Path) -> None:
    import improcv as im

    _touch(tmp_path / "a.jpg")
    assert im.discover_images(tmp_path) == (tmp_path / "a.jpg",)


def test_discovery_module_does_not_import_cv2() -> None:
    import improcv.discovery

    # improcv's top-level __init__ already imports cv2, so checking
    # sys.modules would pass trivially regardless of what discovery.py
    # itself does -- inspect this module's own source instead. The
    # docstring legitimately mentions `cv2.haveImageReader` to explain why
    # it's deliberately *not* used, so only reject an actual import
    # statement, not the substring "cv2" anywhere in the file.
    source = Path(improcv.discovery.__file__).read_text()
    assert "import cv2" not in source
    assert not hasattr(improcv.discovery, "cv2")


def test_discovery_module_does_not_import_numpy() -> None:
    import improcv.discovery

    source = Path(improcv.discovery.__file__).read_text()
    assert "import numpy" not in source
    assert "numpy" not in source


def test_import_discovery_does_not_touch_filesystem_or_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    import improcv.discovery

    before_cwd = os.getcwd()
    importlib.reload(improcv.discovery)
    assert os.getcwd() == before_cwd


def test_import_discovery_does_not_run_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    import improcv.discovery as discovery_module

    calls = []
    monkeypatch.setattr(discovery_module.os, "scandir", lambda *a, **k: calls.append(1))
    importlib.reload(discovery_module)
    assert calls == []
