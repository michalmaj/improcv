import os
import platform
import stat as stat_module
from pathlib import Path

import pytest

from improcv.discovery import ImageMaskPair, discover_image_mask_pairs, discover_images


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


# =====================================================================
# discover_image_mask_pairs
# =====================================================================


def _make_pair_dirs(tmp_path: Path) -> tuple[Path, Path]:
    image_root = tmp_path / "images"
    mask_root = tmp_path / "masks"
    image_root.mkdir()
    mask_root.mkdir()
    return image_root, mask_root


# --- basic pairing ---


def test_pairs_basic_example(tmp_path: Path) -> None:
    image_root, mask_root = _make_pair_dirs(tmp_path)
    (image_root / "cats").mkdir()
    (image_root / "dogs").mkdir()
    (mask_root / "cats").mkdir()
    (mask_root / "dogs").mkdir()
    _touch(image_root / "cats" / "001.jpg")
    _touch(mask_root / "cats" / "001.png")
    _touch(image_root / "dogs" / "001.jpeg")
    _touch(mask_root / "dogs" / "001.tif")

    result = discover_image_mask_pairs(image_root, mask_root)
    assert result == (
        ImageMaskPair(image_root / "cats" / "001.jpg", mask_root / "cats" / "001.png"),
        ImageMaskPair(image_root / "dogs" / "001.jpeg", mask_root / "dogs" / "001.tif"),
    )


def test_pairs_returns_tuple(tmp_path: Path) -> None:
    # Looked up fresh (not the module-level import) since a couple of other
    # tests in this file legitimately `importlib.reload(improcv.discovery)`
    # as part of testing import hygiene, which replaces the module's
    # `ImageMaskPair` class object in place -- comparing against a stale,
    # pre-reload reference here would fail `isinstance` even though the
    # produced pairs are perfectly valid instances of the current class.
    import improcv.discovery as discovery_module

    image_root, mask_root = _make_pair_dirs(tmp_path)
    _touch(image_root / "a.jpg")
    _touch(mask_root / "a.png")
    result = discover_image_mask_pairs(image_root, mask_root)
    assert isinstance(result, tuple)
    assert all(isinstance(pair, discovery_module.ImageMaskPair) for pair in result)


# --- root forms ---


def test_pairs_both_roots_absolute(tmp_path: Path) -> None:
    image_root, mask_root = _make_pair_dirs(tmp_path)
    _touch(image_root / "a.jpg")
    _touch(mask_root / "a.png")
    result = discover_image_mask_pairs(image_root, mask_root)
    assert result[0].image.is_absolute()
    assert result[0].mask.is_absolute()


def test_pairs_both_roots_relative(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    image_root, mask_root = _make_pair_dirs(tmp_path)
    _touch(image_root / "a.jpg")
    _touch(mask_root / "a.png")
    monkeypatch.chdir(tmp_path)
    result = discover_image_mask_pairs("images", "masks")
    assert result == (ImageMaskPair(Path("images") / "a.jpg", Path("masks") / "a.png"),)
    assert not result[0].image.is_absolute()
    assert not result[0].mask.is_absolute()


def test_pairs_image_relative_mask_absolute(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    image_root, mask_root = _make_pair_dirs(tmp_path)
    _touch(image_root / "a.jpg")
    _touch(mask_root / "a.png")
    monkeypatch.chdir(tmp_path)
    result = discover_image_mask_pairs("images", mask_root)
    assert result[0].image == Path("images") / "a.jpg"
    assert result[0].mask == mask_root / "a.png"
    assert result[0].mask.is_absolute()


def test_pairs_image_absolute_mask_relative(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    image_root, mask_root = _make_pair_dirs(tmp_path)
    _touch(image_root / "a.jpg")
    _touch(mask_root / "a.png")
    monkeypatch.chdir(tmp_path)
    result = discover_image_mask_pairs(image_root, "masks")
    assert result[0].image == image_root / "a.jpg"
    assert result[0].mask == Path("masks") / "a.png"


def test_pairs_accepts_custom_pathlike(tmp_path: Path) -> None:
    image_root, mask_root = _make_pair_dirs(tmp_path)
    _touch(image_root / "a.jpg")
    _touch(mask_root / "a.png")

    class CustomPathLike:
        def __init__(self, target: Path) -> None:
            self._target = target

        def __fspath__(self) -> str:
            return str(self._target)

    result = discover_image_mask_pairs(CustomPathLike(image_root), CustomPathLike(mask_root))
    assert result == (ImageMaskPair(image_root / "a.jpg", mask_root / "a.png"),)


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="os.symlink requires elevated privileges on Windows by default",
)
def test_pairs_root_symlink_to_directory(tmp_path: Path) -> None:
    image_root, mask_root = _make_pair_dirs(tmp_path)
    _touch(image_root / "a.jpg")
    _touch(mask_root / "a.png")
    linked_image_root = tmp_path / "linked_images"
    linked_image_root.symlink_to(image_root, target_is_directory=True)

    result = discover_image_mask_pairs(linked_image_root, mask_root)
    assert result == (ImageMaskPair(linked_image_root / "a.jpg", mask_root / "a.png"),)


# --- recursion / hidden ---


def test_pairs_nested_directories(tmp_path: Path) -> None:
    image_root, mask_root = _make_pair_dirs(tmp_path)
    (image_root / "sub").mkdir()
    (mask_root / "sub").mkdir()
    _touch(image_root / "sub" / "a.jpg")
    _touch(mask_root / "sub" / "a.png")

    result = discover_image_mask_pairs(image_root, mask_root)
    assert result == (ImageMaskPair(image_root / "sub" / "a.jpg", mask_root / "sub" / "a.png"),)


def test_pairs_recursive_false_ignores_nested(tmp_path: Path) -> None:
    image_root, mask_root = _make_pair_dirs(tmp_path)
    (image_root / "sub").mkdir()
    (mask_root / "sub").mkdir()
    _touch(image_root / "a.jpg")
    _touch(mask_root / "a.png")
    _touch(image_root / "sub" / "b.jpg")
    _touch(mask_root / "sub" / "b.png")

    result = discover_image_mask_pairs(image_root, mask_root, recursive=False)
    assert result == (ImageMaskPair(image_root / "a.jpg", mask_root / "a.png"),)


def test_pairs_hidden_image_skipped_by_default(tmp_path: Path) -> None:
    image_root, mask_root = _make_pair_dirs(tmp_path)
    _touch(image_root / ".hidden.jpg")
    _touch(mask_root / ".hidden.png")
    _touch(image_root / "visible.jpg")
    _touch(mask_root / "visible.png")

    result = discover_image_mask_pairs(image_root, mask_root)
    assert result == (ImageMaskPair(image_root / "visible.jpg", mask_root / "visible.png"),)


def test_pairs_hidden_directory_skipped_on_both_sides(tmp_path: Path) -> None:
    image_root, mask_root = _make_pair_dirs(tmp_path)
    (image_root / ".hiddendir").mkdir()
    (mask_root / ".hiddendir").mkdir()
    _touch(image_root / ".hiddendir" / "a.jpg")
    _touch(mask_root / ".hiddendir" / "a.png")
    _touch(image_root / "visible.jpg")
    _touch(mask_root / "visible.png")

    # both hidden -> both skipped -> still a full bijection over the visible file only
    result = discover_image_mask_pairs(image_root, mask_root)
    assert result == (ImageMaskPair(image_root / "visible.jpg", mask_root / "visible.png"),)


def test_pairs_hidden_directory_only_on_image_side_causes_unmatched(tmp_path: Path) -> None:
    image_root, mask_root = _make_pair_dirs(tmp_path)
    (image_root / ".hiddendir").mkdir()
    _touch(image_root / ".hiddendir" / "a.jpg")
    # the mask counterpart lives in a non-hidden directory of the same name, so
    # it's still discovered -- creating a genuine mismatch: mask key
    # "hiddendir/a" has no matching image key (the image is hidden and skipped).
    (mask_root / "hiddendir").mkdir()
    _touch(mask_root / "hiddendir" / "a.png")

    with pytest.raises(ValueError, match="mask keys without a matching image"):
        discover_image_mask_pairs(image_root, mask_root)


def test_pairs_include_hidden_true_finds_hidden_entries(tmp_path: Path) -> None:
    image_root, mask_root = _make_pair_dirs(tmp_path)
    _touch(image_root / ".hidden.jpg")
    _touch(mask_root / ".hidden.png")

    result = discover_image_mask_pairs(image_root, mask_root, include_hidden=True)
    assert result == (ImageMaskPair(image_root / ".hidden.jpg", mask_root / ".hidden.png"),)


def test_pairs_hidden_root_still_searched(tmp_path: Path) -> None:
    dot_image_root = tmp_path / ".images"
    dot_mask_root = tmp_path / ".masks"
    dot_image_root.mkdir()
    dot_mask_root.mkdir()
    _touch(dot_image_root / "a.jpg")
    _touch(dot_mask_root / "a.png")

    result = discover_image_mask_pairs(dot_image_root, dot_mask_root, include_hidden=False)
    assert result == (ImageMaskPair(dot_image_root / "a.jpg", dot_mask_root / "a.png"),)


# --- extensions ---


def test_pairs_separate_image_and_mask_extensions(tmp_path: Path) -> None:
    image_root, mask_root = _make_pair_dirs(tmp_path)
    _touch(image_root / "a.jpg")
    _touch(mask_root / "a.tif")

    result = discover_image_mask_pairs(
        image_root, mask_root, image_extensions=[".jpg"], mask_extensions=[".tif"]
    )
    assert result == (ImageMaskPair(image_root / "a.jpg", mask_root / "a.tif"),)


def test_pairs_uppercase_filenames(tmp_path: Path) -> None:
    image_root, mask_root = _make_pair_dirs(tmp_path)
    _touch(image_root / "a.JPG")
    _touch(mask_root / "a.PNG")

    result = discover_image_mask_pairs(image_root, mask_root)
    assert result == (ImageMaskPair(image_root / "a.JPG", mask_root / "a.PNG"),)


def test_pairs_uppercase_extension_arguments(tmp_path: Path) -> None:
    image_root, mask_root = _make_pair_dirs(tmp_path)
    _touch(image_root / "a.jpg")
    _touch(mask_root / "a.png")

    result = discover_image_mask_pairs(
        image_root, mask_root, image_extensions=[".JPG"], mask_extensions=[".PNG"]
    )
    assert result == (ImageMaskPair(image_root / "a.jpg", mask_root / "a.png"),)


def test_pairs_extensions_without_leading_dot(tmp_path: Path) -> None:
    image_root, mask_root = _make_pair_dirs(tmp_path)
    _touch(image_root / "a.jpg")
    _touch(mask_root / "a.png")

    result = discover_image_mask_pairs(
        image_root, mask_root, image_extensions=["jpg"], mask_extensions=["png"]
    )
    assert result == (ImageMaskPair(image_root / "a.jpg", mask_root / "a.png"),)


def test_pairs_multi_part_extension_longest_match(tmp_path: Path) -> None:
    image_root, mask_root = _make_pair_dirs(tmp_path)
    _touch(image_root / "scan.nii.gz")
    _touch(mask_root / "scan.seg")

    result = discover_image_mask_pairs(
        image_root,
        mask_root,
        image_extensions=[".gz", ".nii.gz"],
        mask_extensions=[".seg"],
    )
    assert result == (ImageMaskPair(image_root / "scan.nii.gz", mask_root / "scan.seg"),)


def test_pairs_extension_argument_order_invariant(tmp_path: Path) -> None:
    image_root, mask_root = _make_pair_dirs(tmp_path)
    _touch(image_root / "a.jpg")
    _touch(image_root / "b.png")
    _touch(mask_root / "a.tif")
    _touch(mask_root / "b.tif")

    result_1 = discover_image_mask_pairs(
        image_root, mask_root, image_extensions=[".jpg", ".png"], mask_extensions=[".tif"]
    )
    result_2 = discover_image_mask_pairs(
        image_root, mask_root, image_extensions=[".png", ".jpg"], mask_extensions=[".tif"]
    )
    assert result_1 == result_2


def test_pairs_duplicate_extensions_deduplicated(tmp_path: Path) -> None:
    image_root, mask_root = _make_pair_dirs(tmp_path)
    _touch(image_root / "a.jpg")
    _touch(mask_root / "a.png")

    result = discover_image_mask_pairs(
        image_root, mask_root, image_extensions=["jpg", ".jpg", ".JPG"]
    )
    assert result == (ImageMaskPair(image_root / "a.jpg", mask_root / "a.png"),)


def test_pairs_rejects_bad_image_extensions_collection() -> None:
    with pytest.raises(TypeError, match="image_extensions"):
        discover_image_mask_pairs(".", ".", image_extensions="jpg")  # type: ignore[arg-type]


def test_pairs_rejects_bad_mask_extensions_collection() -> None:
    with pytest.raises(TypeError, match="mask_extensions"):
        discover_image_mask_pairs(".", ".", mask_extensions="png")  # type: ignore[arg-type]


def test_pairs_rejects_empty_image_extensions() -> None:
    with pytest.raises(ValueError, match="image_extensions"):
        discover_image_mask_pairs(".", ".", image_extensions=[])


def test_pairs_rejects_empty_mask_extensions() -> None:
    with pytest.raises(ValueError, match="mask_extensions"):
        discover_image_mask_pairs(".", ".", mask_extensions=[])


# --- pairing key ---


def test_pairs_relative_directory_is_part_of_key(tmp_path: Path) -> None:
    image_root, mask_root = _make_pair_dirs(tmp_path)
    (image_root / "cats").mkdir()
    (image_root / "dogs").mkdir()
    (mask_root / "cats").mkdir()
    (mask_root / "dogs").mkdir()
    _touch(image_root / "cats" / "001.jpg")
    _touch(mask_root / "cats" / "001.png")
    _touch(image_root / "dogs" / "001.jpg")
    _touch(mask_root / "dogs" / "001.png")

    result = discover_image_mask_pairs(image_root, mask_root)
    assert result == (
        ImageMaskPair(image_root / "cats" / "001.jpg", mask_root / "cats" / "001.png"),
        ImageMaskPair(image_root / "dogs" / "001.jpg", mask_root / "dogs" / "001.png"),
    )


def test_pairs_case_difference_does_not_pair(tmp_path: Path) -> None:
    image_root, mask_root = _make_pair_dirs(tmp_path)
    (image_root / "Cat").mkdir()
    (mask_root / "cat").mkdir()
    _touch(image_root / "Cat" / "001.jpg")
    _touch(mask_root / "cat" / "001.png")

    with pytest.raises(ValueError, match="do not match"):
        discover_image_mask_pairs(image_root, mask_root)


def test_pairs_unicode_filenames(tmp_path: Path) -> None:
    image_root, mask_root = _make_pair_dirs(tmp_path)
    _touch(image_root / "kot_ą.jpg")
    _touch(mask_root / "kot_ą.png")

    result = discover_image_mask_pairs(image_root, mask_root)
    assert result == (ImageMaskPair(image_root / "kot_ą.jpg", mask_root / "kot_ą.png"),)


def test_pairs_nfc_and_nfd_do_not_merge(tmp_path: Path) -> None:
    import unicodedata

    nfc_name = unicodedata.normalize("NFC", "kot_ą") + ".jpg"
    nfd_name = unicodedata.normalize("NFD", "kot_ą") + ".png"
    assert nfc_name != nfd_name  # the two forms are genuinely different code point sequences

    image_root, mask_root = _make_pair_dirs(tmp_path)
    _touch(image_root / nfc_name)
    _touch(mask_root / nfd_name)

    with pytest.raises(ValueError, match="do not match"):
        discover_image_mask_pairs(image_root, mask_root)


def test_pairs_filename_with_multiple_dots(tmp_path: Path) -> None:
    image_root, mask_root = _make_pair_dirs(tmp_path)
    _touch(image_root / "v1.2.final.jpg")
    _touch(mask_root / "v1.2.final.png")

    result = discover_image_mask_pairs(image_root, mask_root)
    assert result == (ImageMaskPair(image_root / "v1.2.final.jpg", mask_root / "v1.2.final.png"),)


def test_pairs_empty_key_at_root_raises(tmp_path: Path) -> None:
    image_root, mask_root = _make_pair_dirs(tmp_path)
    _touch(image_root / ".jpg")
    _touch(mask_root / ".png")

    with pytest.raises(ValueError, match=r"non-empty pairing key") as exc_info:
        discover_image_mask_pairs(image_root, mask_root, include_hidden=True)
    assert "image" in str(exc_info.value)
    assert ".jpg" in str(exc_info.value)


def test_pairs_empty_key_in_subdirectory_raises(tmp_path: Path) -> None:
    image_root, mask_root = _make_pair_dirs(tmp_path)
    (image_root / "subdir").mkdir()
    (mask_root / "subdir").mkdir()
    _touch(image_root / "subdir" / ".jpg")
    _touch(mask_root / "subdir" / ".png")

    with pytest.raises(ValueError, match=r"non-empty pairing key"):
        discover_image_mask_pairs(image_root, mask_root, include_hidden=True)


def test_pairs_empty_key_reports_mask_role(tmp_path: Path) -> None:
    image_root, mask_root = _make_pair_dirs(tmp_path)
    _touch(image_root / "a.jpg")
    _touch(mask_root / ".png")

    with pytest.raises(ValueError, match=r"non-empty pairing key") as exc_info:
        discover_image_mask_pairs(image_root, mask_root, include_hidden=True)
    assert "mask" in str(exc_info.value)


# --- duplicates ---


def test_pairs_duplicate_image_key_different_extensions(tmp_path: Path) -> None:
    image_root, mask_root = _make_pair_dirs(tmp_path)
    _touch(image_root / "a.jpg")
    _touch(image_root / "a.png")
    _touch(mask_root / "a.tif")

    with pytest.raises(ValueError, match="duplicate image pairing keys"):
        discover_image_mask_pairs(image_root, mask_root, image_extensions=[".jpg", ".png"])


def test_pairs_duplicate_mask_key_different_extensions(tmp_path: Path) -> None:
    image_root, mask_root = _make_pair_dirs(tmp_path)
    _touch(image_root / "a.jpg")
    _touch(mask_root / "a.png")
    _touch(mask_root / "a.tif")

    with pytest.raises(ValueError, match="duplicate mask pairing keys"):
        discover_image_mask_pairs(image_root, mask_root, mask_extensions=[".png", ".tif"])


def test_pairs_duplicate_after_longest_extension_stripping(tmp_path: Path) -> None:
    image_root, mask_root = _make_pair_dirs(tmp_path)
    _touch(image_root / "scan.nii.gz")
    _touch(image_root / "scan.gz")
    _touch(mask_root / "scan.seg")

    with pytest.raises(ValueError, match="duplicate image pairing keys"):
        discover_image_mask_pairs(
            image_root,
            mask_root,
            image_extensions=[".gz", ".nii.gz"],
            mask_extensions=[".seg"],
        )


def test_pairs_duplicate_diagnostics_content(tmp_path: Path) -> None:
    image_root, mask_root = _make_pair_dirs(tmp_path)
    _touch(image_root / "a.jpg")
    _touch(image_root / "a.png")
    _touch(mask_root / "a.tif")

    with pytest.raises(ValueError) as exc_info:
        discover_image_mask_pairs(image_root, mask_root, image_extensions=[".jpg", ".png"])
    message = str(exc_info.value)
    assert "duplicate image pairing keys" in message
    assert "'a'" in message
    assert "a.jpg" in message
    assert "a.png" in message


def test_pairs_several_duplicate_keys_reported_together(tmp_path: Path) -> None:
    image_root, mask_root = _make_pair_dirs(tmp_path)
    for name in ("a", "b"):
        _touch(image_root / f"{name}.jpg")
        _touch(image_root / f"{name}.png")
    _touch(mask_root / "a.tif")
    _touch(mask_root / "b.tif")

    with pytest.raises(ValueError) as exc_info:
        discover_image_mask_pairs(image_root, mask_root, image_extensions=[".jpg", ".png"])
    message = str(exc_info.value)
    assert "'a'" in message
    assert "'b'" in message


def test_pairs_duplicate_takes_priority_over_unmatched(tmp_path: Path) -> None:
    image_root, mask_root = _make_pair_dirs(tmp_path)
    _touch(image_root / "a.jpg")
    _touch(image_root / "a.png")
    # "b" is unmatched (image only, no mask) -- but the "a" duplicate must be reported first.
    _touch(image_root / "b.jpg")
    _touch(mask_root / "a.tif")

    with pytest.raises(ValueError, match="duplicate image pairing keys"):
        discover_image_mask_pairs(image_root, mask_root, image_extensions=[".jpg", ".png"])


# --- unmatched ---


def test_pairs_image_without_mask_raises(tmp_path: Path) -> None:
    image_root, mask_root = _make_pair_dirs(tmp_path)
    _touch(image_root / "a.jpg")

    with pytest.raises(ValueError, match="image keys without a matching mask"):
        discover_image_mask_pairs(image_root, mask_root)


def test_pairs_mask_without_image_raises(tmp_path: Path) -> None:
    image_root, mask_root = _make_pair_dirs(tmp_path)
    _touch(mask_root / "a.png")

    with pytest.raises(ValueError, match="mask keys without a matching image"):
        discover_image_mask_pairs(image_root, mask_root)


def test_pairs_both_unmatched_kinds_reported_together(tmp_path: Path) -> None:
    image_root, mask_root = _make_pair_dirs(tmp_path)
    _touch(image_root / "a.jpg")
    _touch(mask_root / "b.png")

    with pytest.raises(ValueError) as exc_info:
        discover_image_mask_pairs(image_root, mask_root)
    message = str(exc_info.value)
    assert "image keys without a matching mask" in message
    assert "mask keys without a matching image" in message
    assert "'a'" in message
    assert "'b'" in message


def test_pairs_one_side_empty_raises(tmp_path: Path) -> None:
    image_root, mask_root = _make_pair_dirs(tmp_path)
    _touch(image_root / "a.jpg")

    with pytest.raises(ValueError, match="image keys without a matching mask"):
        discover_image_mask_pairs(image_root, mask_root)


def test_pairs_both_sides_empty_returns_empty_tuple(tmp_path: Path) -> None:
    image_root, mask_root = _make_pair_dirs(tmp_path)
    assert discover_image_mask_pairs(image_root, mask_root) == ()


# --- same root / self-pair ---


def test_pairs_same_root_disjoint_extensions_legal(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    _touch(root / "a.jpg")
    _touch(root / "a.png")

    result = discover_image_mask_pairs(
        root, root, image_extensions=[".jpg"], mask_extensions=[".png"]
    )
    assert result == (ImageMaskPair(root / "a.jpg", root / "a.png"),)


def test_pairs_self_pair_raises(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    _touch(root / "a.png")

    with pytest.raises(ValueError, match="self-paired keys"):
        discover_image_mask_pairs(root, root, image_extensions=[".png"], mask_extensions=[".png"])


# --- filesystem semantics ---


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="os.symlink requires elevated privileges on Windows by default",
)
def test_pairs_symlinked_descendant_skipped(tmp_path: Path) -> None:
    image_root, mask_root = _make_pair_dirs(tmp_path)
    real = image_root / "real.jpg"
    _touch(real)
    link = image_root / "link.jpg"
    link.symlink_to(real)
    _touch(mask_root / "real.png")

    result = discover_image_mask_pairs(image_root, mask_root)
    assert result == (ImageMaskPair(image_root / "real.jpg", mask_root / "real.png"),)


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="os.symlink requires elevated privileges on Windows by default",
)
def test_pairs_broken_symlink_descendant_skipped(tmp_path: Path) -> None:
    image_root, mask_root = _make_pair_dirs(tmp_path)
    link = image_root / "broken.jpg"
    link.symlink_to(image_root / "does_not_exist.jpg")
    _touch(image_root / "real.jpg")
    _touch(mask_root / "real.png")

    result = discover_image_mask_pairs(image_root, mask_root)
    assert result == (ImageMaskPair(image_root / "real.jpg", mask_root / "real.png"),)


def test_pairs_windows_reparse_point_skipped_via_synthetic_attribute(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import improcv.discovery as discovery_module

    image_root, mask_root = _make_pair_dirs(tmp_path)
    junction_dir = image_root / "junction"
    junction_dir.mkdir()
    _touch(junction_dir / "b.jpg")
    _touch(image_root / "real.jpg")
    _touch(mask_root / "real.png")

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

    result = discover_image_mask_pairs(image_root, mask_root)
    assert result == (ImageMaskPair(image_root / "real.jpg", mask_root / "real.png"),)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFOs are not available on this platform")
def test_pairs_non_regular_entry_skipped(tmp_path: Path) -> None:
    image_root, mask_root = _make_pair_dirs(tmp_path)
    os.mkfifo(image_root / "stream.jpg")
    _touch(image_root / "real.jpg")
    _touch(mask_root / "real.png")

    result = discover_image_mask_pairs(image_root, mask_root)
    assert result == (ImageMaskPair(image_root / "real.jpg", mask_root / "real.png"),)


def test_pairs_permission_error_from_image_traversal_propagates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import improcv.discovery as discovery_module

    image_root, mask_root = _make_pair_dirs(tmp_path)
    original = PermissionError(13, "synthetic permission denial", str(image_root))
    real_scandir = discovery_module.os.scandir

    def patched_scandir(path):
        if os.fspath(path) == os.fspath(image_root):
            raise original
        return real_scandir(path)

    monkeypatch.setattr(discovery_module.os, "scandir", patched_scandir)

    with pytest.raises(PermissionError) as exc_info:
        discover_image_mask_pairs(image_root, mask_root)
    assert exc_info.value is original


def test_pairs_permission_error_from_mask_traversal_propagates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import improcv.discovery as discovery_module

    image_root, mask_root = _make_pair_dirs(tmp_path)
    _touch(image_root / "a.jpg")
    original = PermissionError(13, "synthetic permission denial", str(mask_root))
    real_scandir = discovery_module.os.scandir

    def patched_scandir(path):
        if os.fspath(path) == os.fspath(mask_root):
            raise original
        return real_scandir(path)

    monkeypatch.setattr(discovery_module.os, "scandir", patched_scandir)

    with pytest.raises(PermissionError) as exc_info:
        discover_image_mask_pairs(image_root, mask_root)
    assert exc_info.value is original


def test_pairs_disappearing_entry_propagates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import improcv.discovery as discovery_module

    image_root, mask_root = _make_pair_dirs(tmp_path)
    ghost_path = image_root / "ghost.jpg"
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
            pytest.fail("must use fresh os.stat, not cached DirEntry.stat")

    class FakeCM:
        def __enter__(self):
            return [FakeEntry()]

        def __exit__(self, *args):
            return False

    real_scandir = discovery_module.os.scandir

    def patched_scandir(path):
        if os.fspath(path) == os.fspath(image_root):
            return FakeCM()
        return real_scandir(path)

    monkeypatch.setattr(discovery_module.os, "scandir", patched_scandir)
    monkeypatch.setattr(discovery_module.os, "stat", patched_stat)

    with pytest.raises(FileNotFoundError) as exc_info:
        discover_image_mask_pairs(image_root, mask_root)
    assert exc_info.value is original


def test_pairs_fake_non_image_bytes_still_paired(tmp_path: Path) -> None:
    image_root, mask_root = _make_pair_dirs(tmp_path)
    (image_root / "a.jpg").write_bytes(b"this is not image data")
    (mask_root / "a.png").write_bytes(b"")

    result = discover_image_mask_pairs(image_root, mask_root)
    assert result == (ImageMaskPair(image_root / "a.jpg", mask_root / "a.png"),)


# --- determinism ---


@pytest.mark.parametrize("container", [list, tuple, set, frozenset])
def test_pairs_extensions_container_type_independence(tmp_path: Path, container) -> None:
    image_root, mask_root = _make_pair_dirs(tmp_path)
    _touch(image_root / "a.jpg")
    _touch(mask_root / "a.png")

    result = discover_image_mask_pairs(
        image_root,
        mask_root,
        image_extensions=container([".jpg"]),
        mask_extensions=container([".png"]),
    )
    assert result == (ImageMaskPair(image_root / "a.jpg", mask_root / "a.png"),)


def test_pairs_deterministic_key_order_independent_of_creation_order(tmp_path: Path) -> None:
    image_root, mask_root = _make_pair_dirs(tmp_path)
    for name in ("z", "a", "m"):
        _touch(image_root / f"{name}.jpg")
        _touch(mask_root / f"{name}.png")

    result = discover_image_mask_pairs(image_root, mask_root)
    assert [pair.image.stem for pair in result] == ["a", "m", "z"]


# --- normalized values passed to discover_images ---


def test_pairs_does_not_reread_original_extensions_collection(tmp_path: Path) -> None:
    image_root, mask_root = _make_pair_dirs(tmp_path)
    _touch(image_root / "a.jpg")
    _touch(mask_root / "a.png")

    class SpyOnceCollection:
        """A Collection whose second full iteration would return something different.

        `discover_image_mask_pairs` must normalize this once and pass the
        already-normalized tuple to `discover_images`, never re-iterating
        this object a second time expecting the same content.
        """

        def __init__(self, items: list[str]) -> None:
            self._items = list(items)
            self._iterations = 0

        def __len__(self) -> int:
            return len(self._items)

        def __iter__(self):
            self._iterations += 1
            if self._iterations > 1:
                return iter([".this_extension_does_not_exist_and_would_break_matching"])
            return iter(self._items)

        def __contains__(self, item: object) -> bool:
            return item in self._items

    result = discover_image_mask_pairs(
        image_root, mask_root, image_extensions=SpyOnceCollection([".jpg"])
    )
    assert result == (ImageMaskPair(image_root / "a.jpg", mask_root / "a.png"),)


# --- existing discover_images error messages stay unchanged ---


def test_discover_images_still_uses_bare_root_in_messages() -> None:
    with pytest.raises(ValueError, match=r"^root must not be an empty string$"):
        discover_images("")


def test_discover_images_still_uses_bare_extensions_in_messages() -> None:
    with pytest.raises(ValueError, match=r"^extensions must not be empty$"):
        discover_images(".", extensions=[])


def test_pairs_uses_role_specific_root_and_extensions_names(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"^image_root must not be an empty string$"):
        discover_image_mask_pairs("", str(tmp_path))
    with pytest.raises(ValueError, match=r"^mask_root must not be an empty string$"):
        discover_image_mask_pairs(str(tmp_path), "")


# --- public type: unpacking, equality, hashing, tuple semantics ---


def test_image_mask_pair_unpacking() -> None:
    pair = ImageMaskPair(Path("a.jpg"), Path("a.png"))
    image, mask = pair
    assert image == Path("a.jpg")
    assert mask == Path("a.png")


def test_image_mask_pair_fields() -> None:
    pair = ImageMaskPair(Path("a.jpg"), Path("a.png"))
    assert pair.image == Path("a.jpg")
    assert pair.mask == Path("a.png")


def test_image_mask_pair_equality() -> None:
    assert ImageMaskPair(Path("a.jpg"), Path("a.png")) == ImageMaskPair(
        Path("a.jpg"), Path("a.png")
    )
    assert ImageMaskPair(Path("a.jpg"), Path("a.png")) != ImageMaskPair(
        Path("b.jpg"), Path("b.png")
    )


def test_image_mask_pair_hashable() -> None:
    pair = ImageMaskPair(Path("a.jpg"), Path("a.png"))
    assert hash(pair) == hash(ImageMaskPair(Path("a.jpg"), Path("a.png")))
    assert {pair} == {ImageMaskPair(Path("a.jpg"), Path("a.png"))}


def test_image_mask_pair_is_tuple() -> None:
    pair = ImageMaskPair(Path("a.jpg"), Path("a.png"))
    assert isinstance(pair, tuple)
    assert tuple(pair) == (Path("a.jpg"), Path("a.png"))


# --- exports ---


def test_image_mask_pair_exported_from_top_level_package() -> None:
    import improcv as im

    assert im.ImageMaskPair is ImageMaskPair


def test_discover_image_mask_pairs_exported_from_top_level_package(tmp_path: Path) -> None:
    import improcv as im

    image_root, mask_root = _make_pair_dirs(tmp_path)
    _touch(image_root / "a.jpg")
    _touch(mask_root / "a.png")
    assert im.discover_image_mask_pairs(image_root, mask_root) == (
        ImageMaskPair(image_root / "a.jpg", mask_root / "a.png"),
    )


def test_image_mask_pair_in_discovery_all() -> None:
    import improcv.discovery as discovery_module

    assert "ImageMaskPair" in discovery_module.__all__
    assert "discover_image_mask_pairs" in discovery_module.__all__


def test_image_mask_pair_in_top_level_all() -> None:
    import improcv as im

    assert "ImageMaskPair" in im.__all__
    assert "discover_image_mask_pairs" in im.__all__


# --- diagnostic preview helper boundary tests ---


def _preview_entries(count: int) -> list[str]:
    return [f"entry_{index:02d}" for index in range(count)]


def test_diagnostic_preview_empty() -> None:
    from improcv.discovery import _format_diagnostic_preview

    assert _format_diagnostic_preview([]) == ""


def test_diagnostic_preview_single_entry() -> None:
    from improcv.discovery import _format_diagnostic_preview

    assert _format_diagnostic_preview(["entry_00"]) == "entry_00"


def test_diagnostic_preview_exactly_at_limit() -> None:
    from improcv.discovery import _format_diagnostic_preview

    entries = _preview_entries(10)
    result = _format_diagnostic_preview(entries)
    assert result == "\n".join(entries)
    assert "more" not in result


def test_diagnostic_preview_one_over_limit() -> None:
    from improcv.discovery import _format_diagnostic_preview

    entries = _preview_entries(11)
    result = _format_diagnostic_preview(entries)
    assert result == "\n".join(entries[:10]) + "\n... and 1 more"


def test_diagnostic_preview_well_over_limit() -> None:
    from improcv.discovery import _format_diagnostic_preview

    entries = _preview_entries(25)
    result = _format_diagnostic_preview(entries)
    assert result == "\n".join(entries[:10]) + "\n... and 15 more"


# --- import hygiene (extended for the new symbols) ---


def test_discovery_module_still_does_not_import_cv2_or_numpy() -> None:
    import improcv.discovery

    source = Path(improcv.discovery.__file__).read_text()
    assert "import cv2" not in source
    assert "import numpy" not in source
    assert "numpy" not in source
