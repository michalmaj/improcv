import inspect
import os
import sys
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pytest

import improcv as im
import improcv.dataset as dataset_module
from improcv.dataset import build_perceptual_hash_manifest
from improcv.hashing import PerceptualHash, PerceptualHashAlgorithm
from improcv.manifest import PerceptualHashManifest

_PHASH = PerceptualHashAlgorithm.PHASH
_AVERAGE_HASH = PerceptualHashAlgorithm.AVERAGE_HASH


def _checkerboard(size: int = 8) -> np.ndarray:
    row, col = np.indices((size, size))
    return np.where((row + col) % 2 == 0, 0, 255).astype(np.uint8)


def _vertical_split(size: int = 8) -> np.ndarray:
    image = np.zeros((size, size), dtype=np.uint8)
    image[:, size // 2 :] = 255
    return image


def _write_image(path: Path, pixels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), pixels)
    if not ok:
        raise RuntimeError(f"failed to write test fixture {path}")


class _Boom:
    def __call__(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("build_perceptual_hash_manifest must not perform this operation")


# =====================================================================================
# Exports and signature
# =====================================================================================


def test_top_level_export_is_the_same_object() -> None:
    assert im.build_perceptual_hash_manifest is build_perceptual_hash_manifest


def test_module_all_contains_exactly_the_public_symbol() -> None:
    assert dataset_module.__all__ == ["build_perceptual_hash_manifest"]


def test_top_level_all_contains_symbol_without_duplicates() -> None:
    assert im.__all__.count("build_perceptual_hash_manifest") == 1
    assert len(im.__all__) == len(set(im.__all__))


def test_top_level_all_places_symbol_alphabetically() -> None:
    index = im.__all__.index("build_perceptual_hash_manifest")
    assert im.__all__[index - 1] == "bounding_boxes"
    assert im.__all__[index + 1] == "calibrate_camera_response_debevec"


def test_public_signature() -> None:
    signature = inspect.signature(build_perceptual_hash_manifest)
    parameters = signature.parameters
    assert list(parameters) == [
        "root",
        "algorithm",
        "hash_size",
        "recursive",
        "extensions",
        "include_hidden",
    ]
    assert parameters["root"].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameters["algorithm"].kind == inspect.Parameter.KEYWORD_ONLY
    assert parameters["algorithm"].default is inspect.Parameter.empty
    assert parameters["hash_size"].kind == inspect.Parameter.KEYWORD_ONLY
    assert parameters["hash_size"].default == 8
    assert parameters["recursive"].kind == inspect.Parameter.KEYWORD_ONLY
    assert parameters["recursive"].default is True
    assert parameters["extensions"].kind == inspect.Parameter.KEYWORD_ONLY
    assert parameters["extensions"].default is None
    assert parameters["include_hidden"].kind == inspect.Parameter.KEYWORD_ONLY
    assert parameters["include_hidden"].default is False
    assert signature.return_annotation in (PerceptualHashManifest, "PerceptualHashManifest")


def test_module_does_not_export_extra_symbols() -> None:
    forbidden = {
        "Dataset",
        "DatasetBuilder",
        "PerceptualHashManifestBuilder",
        "BuildResult",
        "perceptual_hash",
    }
    assert not forbidden & set(dataset_module.__all__)
    assert not forbidden & set(im.__all__)


# =====================================================================================
# Argument validation
# =====================================================================================


def test_rejects_string_algorithm(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="algorithm"):
        build_perceptual_hash_manifest(tmp_path, algorithm="phash")  # type: ignore[arg-type]


def test_rejects_int_algorithm(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="algorithm"):
        build_perceptual_hash_manifest(tmp_path, algorithm=1)  # type: ignore[arg-type]


def test_rejects_bool_algorithm(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="algorithm"):
        build_perceptual_hash_manifest(tmp_path, algorithm=True)  # type: ignore[arg-type]


def test_rejects_none_algorithm(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="algorithm"):
        build_perceptual_hash_manifest(tmp_path, algorithm=None)  # type: ignore[arg-type]


def test_rejects_unrelated_enum_as_algorithm(tmp_path: Path) -> None:
    from enum import Enum

    class _NotAlgorithm(Enum):
        PHASH = "phash"

    with pytest.raises(TypeError, match="algorithm"):
        build_perceptual_hash_manifest(tmp_path, algorithm=_NotAlgorithm.PHASH)  # type: ignore[arg-type]


def test_rejects_bad_hash_size_type(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        build_perceptual_hash_manifest(tmp_path, algorithm=_PHASH, hash_size="8")  # type: ignore[arg-type]


def test_rejects_bool_hash_size(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        build_perceptual_hash_manifest(tmp_path, algorithm=_PHASH, hash_size=True)  # type: ignore[arg-type]


def test_rejects_hash_size_below_minimum(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="hash_size"):
        build_perceptual_hash_manifest(tmp_path, algorithm=_PHASH, hash_size=1)


def test_rejects_hash_size_above_maximum(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="hash_size"):
        build_perceptual_hash_manifest(tmp_path, algorithm=_PHASH, hash_size=257)


def test_argument_validation_runs_before_touching_filesystem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    boom = _Boom()
    monkeypatch.setattr(cv2, "imread", boom)
    # A missing root would otherwise raise inside discover_images -- confirms algorithm/hash_size
    # are checked first, before any discovery/decoding is attempted.
    with pytest.raises(TypeError, match="algorithm"):
        build_perceptual_hash_manifest(tmp_path / "does-not-exist", algorithm="phash")  # type: ignore[arg-type]


def test_delegates_recursive_validation_to_discover_images(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        build_perceptual_hash_manifest(tmp_path, algorithm=_PHASH, recursive=1)  # type: ignore[arg-type]


def test_delegates_extensions_validation_to_discover_images(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        build_perceptual_hash_manifest(tmp_path, algorithm=_PHASH, extensions=[])


def test_delegates_include_hidden_validation_to_discover_images(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        build_perceptual_hash_manifest(tmp_path, algorithm=_PHASH, include_hidden="yes")  # type: ignore[arg-type]


# =====================================================================================
# Empty dataset
# =====================================================================================


def test_empty_directory_returns_empty_manifest(tmp_path: Path) -> None:
    manifest = build_perceptual_hash_manifest(tmp_path, algorithm=_AVERAGE_HASH, hash_size=8)
    assert manifest.entries == ()
    assert manifest.algorithm is _AVERAGE_HASH
    assert manifest.hash_size == 8


def test_empty_directory_with_average_hash(tmp_path: Path) -> None:
    manifest = build_perceptual_hash_manifest(tmp_path, algorithm=_AVERAGE_HASH)
    assert manifest == PerceptualHashManifest.from_hashes({}, algorithm=_AVERAGE_HASH, hash_size=8)


def test_empty_directory_with_phash(tmp_path: Path) -> None:
    manifest = build_perceptual_hash_manifest(tmp_path, algorithm=_PHASH)
    assert manifest == PerceptualHashManifest.from_hashes({}, algorithm=_PHASH, hash_size=8)


def test_empty_directory_with_other_legal_hash_size(tmp_path: Path) -> None:
    manifest = build_perceptual_hash_manifest(tmp_path, algorithm=_PHASH, hash_size=16)
    assert manifest.hash_size == 16
    assert manifest.entries == ()


def test_directory_with_no_matching_extensions_returns_empty_manifest(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("not an image", encoding="utf-8")
    manifest = build_perceptual_hash_manifest(tmp_path, algorithm=_PHASH)
    assert manifest.entries == ()


def test_empty_dataset_never_calls_cv2_imread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cv2, "imread", _Boom())
    manifest = build_perceptual_hash_manifest(tmp_path, algorithm=_PHASH)
    assert manifest.entries == ()


# =====================================================================================
# Discovery delegation and path identifiers
# =====================================================================================


def test_singleton_dataset(tmp_path: Path) -> None:
    _write_image(tmp_path / "a.png", _checkerboard())
    manifest = build_perceptual_hash_manifest(tmp_path, algorithm=_AVERAGE_HASH)
    assert [entry.path.as_posix() for entry in manifest.entries] == ["a.png"]


def test_multiple_images(tmp_path: Path) -> None:
    _write_image(tmp_path / "a.png", _checkerboard())
    _write_image(tmp_path / "b.png", _vertical_split())
    manifest = build_perceptual_hash_manifest(tmp_path, algorithm=_AVERAGE_HASH)
    assert [entry.path.as_posix() for entry in manifest.entries] == ["a.png", "b.png"]


def test_nested_images_recursive_true(tmp_path: Path) -> None:
    _write_image(tmp_path / "a.png", _checkerboard())
    _write_image(tmp_path / "cats" / "b.png", _vertical_split())
    manifest = build_perceptual_hash_manifest(tmp_path, algorithm=_AVERAGE_HASH, recursive=True)
    assert [entry.path.as_posix() for entry in manifest.entries] == ["a.png", "cats/b.png"]


def test_no_dataset_root_name_in_entry_paths(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "my-dataset-name"
    _write_image(dataset_dir / "cats" / "a.png", _checkerboard())
    manifest = build_perceptual_hash_manifest(dataset_dir, algorithm=_AVERAGE_HASH)
    assert manifest.entries[0].path.as_posix() == "cats/a.png"
    assert "my-dataset-name" not in manifest.entries[0].path.as_posix()


def test_no_absolute_paths_in_manifest(tmp_path: Path) -> None:
    _write_image(tmp_path / "a.png", _checkerboard())
    manifest = build_perceptual_hash_manifest(tmp_path, algorithm=_AVERAGE_HASH)
    for entry in manifest.entries:
        assert not entry.path.is_absolute()


def test_recursive_false_only_direct_children(tmp_path: Path) -> None:
    _write_image(tmp_path / "a.png", _checkerboard())
    _write_image(tmp_path / "cats" / "b.png", _vertical_split())
    manifest = build_perceptual_hash_manifest(tmp_path, algorithm=_AVERAGE_HASH, recursive=False)
    assert [entry.path.as_posix() for entry in manifest.entries] == ["a.png"]


def test_custom_extensions(tmp_path: Path) -> None:
    _write_image(tmp_path / "a.bmp", _checkerboard())
    _write_image(tmp_path / "b.png", _vertical_split())
    manifest = build_perceptual_hash_manifest(tmp_path, algorithm=_AVERAGE_HASH, extensions=["bmp"])
    assert [entry.path.as_posix() for entry in manifest.entries] == ["a.bmp"]


def test_uppercase_extension_matched(tmp_path: Path) -> None:
    _write_image(tmp_path / "A.PNG", _checkerboard())
    manifest = build_perceptual_hash_manifest(tmp_path, algorithm=_AVERAGE_HASH)
    assert [entry.path.as_posix() for entry in manifest.entries] == ["A.PNG"]


def test_hidden_file_skipped_by_default(tmp_path: Path) -> None:
    _write_image(tmp_path / ".hidden.png", _checkerboard())
    _write_image(tmp_path / "visible.png", _vertical_split())
    manifest = build_perceptual_hash_manifest(tmp_path, algorithm=_AVERAGE_HASH)
    assert [entry.path.as_posix() for entry in manifest.entries] == ["visible.png"]


def test_include_hidden_true_finds_hidden_files(tmp_path: Path) -> None:
    _write_image(tmp_path / ".hidden.png", _checkerboard())
    manifest = build_perceptual_hash_manifest(
        tmp_path, algorithm=_AVERAGE_HASH, include_hidden=True
    )
    assert [entry.path.as_posix() for entry in manifest.entries] == [".hidden.png"]


def test_absolute_root(tmp_path: Path) -> None:
    _write_image(tmp_path / "a.png", _checkerboard())
    assert tmp_path.is_absolute()
    manifest = build_perceptual_hash_manifest(tmp_path, algorithm=_AVERAGE_HASH)
    assert [entry.path.as_posix() for entry in manifest.entries] == ["a.png"]


def test_relative_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_image(tmp_path / "dataset" / "a.png", _checkerboard())
    monkeypatch.chdir(tmp_path)
    manifest = build_perceptual_hash_manifest("dataset", algorithm=_AVERAGE_HASH)
    assert [entry.path.as_posix() for entry in manifest.entries] == ["a.png"]


def test_root_dot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_image(tmp_path / "a.png", _checkerboard())
    monkeypatch.chdir(tmp_path)
    manifest = build_perceptual_hash_manifest(".", algorithm=_AVERAGE_HASH)
    assert [entry.path.as_posix() for entry in manifest.entries] == ["a.png"]


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="cv2.imread/imwrite do not reliably support non-ASCII paths on Windows",
)
def test_root_with_unicode(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "żółw"
    _write_image(dataset_dir / "a.png", _checkerboard())
    manifest = build_perceptual_hash_manifest(dataset_dir, algorithm=_AVERAGE_HASH)
    assert [entry.path.as_posix() for entry in manifest.entries] == ["a.png"]


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="cv2.imread/imwrite do not reliably support non-ASCII paths on Windows",
)
def test_nested_unicode_identifier(tmp_path: Path) -> None:
    _write_image(tmp_path / "dane" / "żółw.png", _checkerboard())
    manifest = build_perceptual_hash_manifest(tmp_path, algorithm=_AVERAGE_HASH)
    assert [entry.path.as_posix() for entry in manifest.entries] == ["dane/żółw.png"]


def test_exact_canonical_entry_ordering(tmp_path: Path) -> None:
    _write_image(tmp_path / "z.png", _checkerboard())
    _write_image(tmp_path / "a.png", _vertical_split())
    _write_image(tmp_path / "m.png", _checkerboard())
    manifest = build_perceptual_hash_manifest(tmp_path, algorithm=_AVERAGE_HASH)
    assert [entry.path.as_posix() for entry in manifest.entries] == ["a.png", "m.png", "z.png"]


def test_missing_root_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_perceptual_hash_manifest(tmp_path / "does-not-exist", algorithm=_PHASH)


def test_root_is_a_file_raises(tmp_path: Path) -> None:
    file_root = tmp_path / "not_a_dir.png"
    _write_image(file_root, _checkerboard())
    with pytest.raises(NotADirectoryError):
        build_perceptual_hash_manifest(file_root, algorithm=_PHASH)


def test_symlinked_file_skipped(tmp_path: Path) -> None:
    _write_image(tmp_path / "real.png", _checkerboard())
    link = tmp_path / "link.png"
    try:
        link.symlink_to(tmp_path / "real.png")
    except (OSError, NotImplementedError):
        pytest.skip("platform does not support creating symlinks")
    manifest = build_perceptual_hash_manifest(tmp_path, algorithm=_AVERAGE_HASH)
    assert [entry.path.as_posix() for entry in manifest.entries] == ["real.png"]


# =====================================================================================
# Decode policy
# =====================================================================================


def test_uses_imread_grayscale_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_image(tmp_path / "a.png", _checkerboard())
    seen_flags: list[int] = []
    real_imread = cv2.imread

    def spy(filename: str, flags: int) -> np.ndarray | None:
        seen_flags.append(flags)
        return real_imread(filename, flags)

    monkeypatch.setattr(cv2, "imread", spy)
    build_perceptual_hash_manifest(tmp_path, algorithm=_AVERAGE_HASH)
    assert seen_flags == [cv2.IMREAD_GRAYSCALE]


def test_each_file_decoded_exactly_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_image(tmp_path / "a.png", _checkerboard())
    _write_image(tmp_path / "b.png", _vertical_split())
    call_count = 0
    real_imread = cv2.imread

    def spy(filename: str, flags: int) -> np.ndarray | None:
        nonlocal call_count
        call_count += 1
        return real_imread(filename, flags)

    monkeypatch.setattr(cv2, "imread", spy)
    build_perceptual_hash_manifest(tmp_path, algorithm=_AVERAGE_HASH)
    assert call_count == 2


def test_undecodable_file_raises_value_error(tmp_path: Path) -> None:
    (tmp_path / "broken.png").write_bytes(b"not a real png")
    with pytest.raises(ValueError, match="broken.png"):
        build_perceptual_hash_manifest(tmp_path, algorithm=_PHASH)


def test_decode_error_message_contains_relative_identifier(tmp_path: Path) -> None:
    (tmp_path / "sub" / "broken.png").parent.mkdir(parents=True)
    (tmp_path / "sub" / "broken.png").write_bytes(b"not a real png")
    with pytest.raises(ValueError, match="sub/broken.png"):
        build_perceptual_hash_manifest(tmp_path, algorithm=_PHASH)


def test_decode_error_message_contains_source_path(tmp_path: Path) -> None:
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not a real png")
    with pytest.raises(ValueError) as exc_info:
        build_perceptual_hash_manifest(tmp_path, algorithm=_PHASH)
    # The message embeds the source path via `!r` (see dataset.py), which backslash-escapes it
    # in the rendered text -- comparing against repr(str(broken)) directly (a plain substring
    # check) avoids re-deriving that escaping as a regex pattern.
    assert repr(str(broken)) in str(exc_info.value)


def test_multiple_broken_files_reports_first_in_canonical_order(tmp_path: Path) -> None:
    (tmp_path / "a_broken.png").write_bytes(b"not a real png")
    (tmp_path / "z_broken.png").write_bytes(b"also not a real png")
    with pytest.raises(ValueError, match="a_broken.png"):
        build_perceptual_hash_manifest(tmp_path, algorithm=_PHASH)


def test_no_decode_after_first_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "a_broken.png").write_bytes(b"not a real png")
    _write_image(tmp_path / "z_ok.png", _checkerboard())
    call_count = 0
    real_imread = cv2.imread

    def spy(filename: str, flags: int) -> np.ndarray | None:
        nonlocal call_count
        call_count += 1
        return real_imread(filename, flags)

    monkeypatch.setattr(cv2, "imread", spy)
    with pytest.raises(ValueError):
        build_perceptual_hash_manifest(tmp_path, algorithm=_PHASH)
    assert call_count == 1


def test_no_partial_manifest_on_decode_failure(tmp_path: Path) -> None:
    _write_image(tmp_path / "a_ok.png", _checkerboard())
    (tmp_path / "z_broken.png").write_bytes(b"not a real png")
    try:
        build_perceptual_hash_manifest(tmp_path, algorithm=_PHASH)
    except ValueError:
        pass
    else:
        raise AssertionError("expected a ValueError")
    # No manifest object is ever returned on this path -- nothing further to assert on a
    # non-existent return value; the absence of a returned manifest is the assertion itself.


def test_valid_extension_but_not_an_image_fails_to_decode(tmp_path: Path) -> None:
    (tmp_path / "fake.jpg").write_bytes(os.urandom(64))
    with pytest.raises(ValueError, match="fake.jpg"):
        build_perceptual_hash_manifest(tmp_path, algorithm=_PHASH)


def test_unicode_source_path_in_decode_error(tmp_path: Path) -> None:
    (tmp_path / "żółw.png").write_bytes(b"not a real png")
    with pytest.raises(ValueError, match="żółw.png"):
        build_perceptual_hash_manifest(tmp_path, algorithm=_PHASH)


def test_cv2_imread_exception_propagates_unwrapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_image(tmp_path / "a.png", _checkerboard())

    class _CustomError(RuntimeError):
        pass

    def failing_imread(filename: str, flags: int) -> np.ndarray | None:
        raise _CustomError("simulated decode crash")

    monkeypatch.setattr(cv2, "imread", failing_imread)
    with pytest.raises(_CustomError, match="simulated decode crash"):
        build_perceptual_hash_manifest(tmp_path, algorithm=_PHASH)


# =====================================================================================
# Algorithm dispatch
# =====================================================================================


def test_average_hash_dispatch_matches_direct_call(tmp_path: Path) -> None:
    pixels = _checkerboard()
    _write_image(tmp_path / "a.png", pixels)
    manifest = build_perceptual_hash_manifest(tmp_path, algorithm=_AVERAGE_HASH, hash_size=8)
    decoded = cv2.imread(str(tmp_path / "a.png"), cv2.IMREAD_GRAYSCALE)
    expected = im.average_hash(cast(im.ImageU8, decoded), hash_size=8)
    assert manifest.entries[0].hash == expected


def test_phash_dispatch_matches_direct_call(tmp_path: Path) -> None:
    pixels = _checkerboard()
    _write_image(tmp_path / "a.png", pixels)
    manifest = build_perceptual_hash_manifest(tmp_path, algorithm=_PHASH, hash_size=8)
    decoded = cv2.imread(str(tmp_path / "a.png"), cv2.IMREAD_GRAYSCALE)
    expected = im.phash(cast(im.ImageU8, decoded), hash_size=8)
    assert manifest.entries[0].hash == expected


@pytest.mark.parametrize("hash_size", [2, 8])
@pytest.mark.parametrize("algorithm", [_AVERAGE_HASH, _PHASH])
def test_dispatch_matches_direct_call_across_hash_sizes(
    tmp_path: Path, algorithm: PerceptualHashAlgorithm, hash_size: int
) -> None:
    pixels = _checkerboard()
    _write_image(tmp_path / "a.png", pixels)
    manifest = build_perceptual_hash_manifest(tmp_path, algorithm=algorithm, hash_size=hash_size)
    decoded = cv2.imread(str(tmp_path / "a.png"), cv2.IMREAD_GRAYSCALE)
    direct_function = im.average_hash if algorithm is _AVERAGE_HASH else im.phash
    expected = direct_function(cast(im.ImageU8, decoded), hash_size=hash_size)
    assert manifest.entries[0].hash == expected


def test_dispatch_across_multiple_images(tmp_path: Path) -> None:
    _write_image(tmp_path / "a.png", _checkerboard())
    _write_image(tmp_path / "b.png", _vertical_split())
    manifest = build_perceptual_hash_manifest(tmp_path, algorithm=_PHASH, hash_size=4)
    for entry in manifest.entries:
        decoded = cv2.imread(str(tmp_path / entry.path.as_posix()), cv2.IMREAD_GRAYSCALE)
        assert entry.hash == im.phash(cast(im.ImageU8, decoded), hash_size=4)


def test_every_entry_hash_matches_manifest_hash_space(tmp_path: Path) -> None:
    _write_image(tmp_path / "a.png", _checkerboard())
    _write_image(tmp_path / "b.png", _vertical_split())
    manifest = build_perceptual_hash_manifest(tmp_path, algorithm=_PHASH, hash_size=8)
    for entry in manifest.entries:
        assert entry.hash.algorithm is _PHASH
        assert entry.hash.hash_size == 8
    assert manifest.algorithm is _PHASH
    assert manifest.hash_size == 8


# =====================================================================================
# Manual-pipeline equivalence (load-bearing regression test)
# =====================================================================================


def test_matches_manual_pipeline(tmp_path: Path) -> None:
    _write_image(tmp_path / "a.png", _checkerboard())
    _write_image(tmp_path / "cats" / "b.png", _vertical_split())

    manual_paths = im.discover_images(tmp_path)
    manual_hashes: dict[Path, PerceptualHash] = {}
    for path in manual_paths:
        decoded_image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if decoded_image is None:
            raise RuntimeError(f"could not decode {path}")
        manual_hashes[path.relative_to(tmp_path)] = im.average_hash(
            cast(im.ImageU8, decoded_image), hash_size=8
        )

    manual_manifest = im.PerceptualHashManifest.from_hashes(
        manual_hashes,
        algorithm=im.PerceptualHashAlgorithm.AVERAGE_HASH,
        hash_size=8,
    )

    built_manifest = im.build_perceptual_hash_manifest(
        tmp_path,
        algorithm=im.PerceptualHashAlgorithm.AVERAGE_HASH,
        hash_size=8,
    )

    assert built_manifest == manual_manifest
    assert built_manifest.to_json() == manual_manifest.to_json()


# =====================================================================================
# No manifest file I/O
# =====================================================================================


def test_does_not_call_manifest_save_or_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_image(tmp_path / "a.png", _checkerboard())
    monkeypatch.setattr(PerceptualHashManifest, "save", _Boom())
    monkeypatch.setattr(PerceptualHashManifest, "load", classmethod(lambda cls, path: _Boom()()))
    build_perceptual_hash_manifest(tmp_path, algorithm=_AVERAGE_HASH)


def test_does_not_call_path_write_text_or_write_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_image(tmp_path / "a.png", _checkerboard())
    monkeypatch.setattr(Path, "write_text", _Boom())
    monkeypatch.setattr(Path, "write_bytes", _Boom())
    build_perceptual_hash_manifest(tmp_path, algorithm=_AVERAGE_HASH)


def test_dataset_root_contents_unchanged(tmp_path: Path) -> None:
    _write_image(tmp_path / "a.png", _checkerboard())
    _write_image(tmp_path / "cats" / "b.png", _vertical_split())

    before = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in sorted(tmp_path.rglob("*"))
        if path.is_file()
    }

    build_perceptual_hash_manifest(tmp_path, algorithm=_AVERAGE_HASH)

    after = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in sorted(tmp_path.rglob("*"))
        if path.is_file()
    }
    assert after == before


def test_leaves_no_new_files_in_dataset_root(tmp_path: Path) -> None:
    _write_image(tmp_path / "a.png", _checkerboard())
    before = {path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*")}
    build_perceptual_hash_manifest(tmp_path, algorithm=_AVERAGE_HASH)
    after = {path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*")}
    assert after == before
