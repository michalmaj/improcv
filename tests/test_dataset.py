import copy
import dataclasses
import inspect
import os
import random
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pytest

import improcv as im
import improcv.dataset as dataset_module
from improcv.dataset import DatasetSplit, build_perceptual_hash_manifest, split_dataset
from improcv.discovery import ImageMaskPair, discover_image_mask_pairs
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
    # Writes path-safely (encode in memory, then `Path.write_bytes`) rather than via
    # `cv2.imwrite`, which on Windows does not reliably support paths containing characters
    # outside the active code page -- this fixture helper is used for both ASCII and Unicode
    # paths so every test exercises the same, always-Unicode-safe write.
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", pixels)
    if not ok:
        raise RuntimeError(f"failed to encode test fixture {path}")
    path.write_bytes(encoded.tobytes())


class _Boom:
    def __call__(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("build_perceptual_hash_manifest must not perform this operation")


# =====================================================================================
# Exports and signature
# =====================================================================================


def test_top_level_export_is_the_same_object() -> None:
    assert im.build_perceptual_hash_manifest is build_perceptual_hash_manifest
    assert im.split_dataset is split_dataset
    assert im.DatasetSplit is DatasetSplit


def test_module_all_contains_exactly_the_public_symbols() -> None:
    assert dataset_module.__all__ == [
        "DatasetSplit",
        "build_perceptual_hash_manifest",
        "split_dataset",
    ]


def test_top_level_all_contains_new_symbols_without_duplicates() -> None:
    assert im.__all__.count("build_perceptual_hash_manifest") == 1
    assert im.__all__.count("split_dataset") == 1
    assert im.__all__.count("DatasetSplit") == 1
    assert len(im.__all__) == len(set(im.__all__))


def test_top_level_all_places_build_perceptual_hash_manifest_alphabetically() -> None:
    index = im.__all__.index("build_perceptual_hash_manifest")
    assert im.__all__[index - 1] == "bounding_boxes"
    assert im.__all__[index + 1] == "calibrate_camera_response_debevec"


def test_top_level_all_places_dataset_split_alphabetically() -> None:
    index = im.__all__.index("DatasetSplit")
    assert im.__all__[index - 1] == "CropParameters"
    assert im.__all__[index + 1] == "DescriptorNorm"


def test_top_level_all_places_split_dataset_alphabetically() -> None:
    index = im.__all__.index("split_dataset")
    assert im.__all__[index - 1] == "sort_contours"
    assert im.__all__[index + 1] == "ssim"


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
    monkeypatch.setattr(Path, "read_bytes", _Boom())
    monkeypatch.setattr(cv2, "imdecode", _Boom())
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


def test_empty_dataset_never_calls_cv2_imdecode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cv2, "imdecode", _Boom())
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


def test_root_with_unicode(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "żółw"
    _write_image(dataset_dir / "a.png", _checkerboard())
    manifest = build_perceptual_hash_manifest(dataset_dir, algorithm=_AVERAGE_HASH)
    assert [entry.path.as_posix() for entry in manifest.entries] == ["a.png"]


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


def test_uses_imdecode_grayscale_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_image(tmp_path / "a.png", _checkerboard())
    seen_flags: list[int] = []
    real_imdecode = cv2.imdecode

    def spy(buffer: np.ndarray, flags: int) -> np.ndarray | None:
        seen_flags.append(flags)
        return real_imdecode(buffer, flags)

    monkeypatch.setattr(cv2, "imdecode", spy)
    build_perceptual_hash_manifest(tmp_path, algorithm=_AVERAGE_HASH)
    assert seen_flags == [cv2.IMREAD_GRAYSCALE]


def test_each_file_decoded_exactly_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_image(tmp_path / "a.png", _checkerboard())
    _write_image(tmp_path / "b.png", _vertical_split())
    call_count = 0
    real_imdecode = cv2.imdecode

    def spy(buffer: np.ndarray, flags: int) -> np.ndarray | None:
        nonlocal call_count
        call_count += 1
        return real_imdecode(buffer, flags)

    monkeypatch.setattr(cv2, "imdecode", spy)
    build_perceptual_hash_manifest(tmp_path, algorithm=_AVERAGE_HASH)
    assert call_count == 2


def test_imdecode_receives_exact_file_bytes_as_1d_uint8_buffer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "a.png"
    _write_image(target, _checkerboard())
    expected_bytes = target.read_bytes()
    real_imdecode = cv2.imdecode
    seen_buffers: list[np.ndarray] = []

    def spy(buffer: np.ndarray, flags: int) -> np.ndarray | None:
        seen_buffers.append(buffer)
        return real_imdecode(buffer, flags)

    monkeypatch.setattr(cv2, "imdecode", spy)
    build_perceptual_hash_manifest(tmp_path, algorithm=_AVERAGE_HASH)

    assert len(seen_buffers) == 1
    buffer = seen_buffers[0]
    assert buffer.ndim == 1
    assert buffer.dtype == np.uint8
    assert buffer.tobytes() == expected_bytes


def test_empty_file_does_not_call_cv2_imdecode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "empty.png").write_bytes(b"")
    monkeypatch.setattr(cv2, "imdecode", _Boom())
    with pytest.raises(ValueError, match="empty.png"):
        build_perceptual_hash_manifest(tmp_path, algorithm=_PHASH)


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
    real_imdecode = cv2.imdecode

    def spy(buffer: np.ndarray, flags: int) -> np.ndarray | None:
        nonlocal call_count
        call_count += 1
        return real_imdecode(buffer, flags)

    monkeypatch.setattr(cv2, "imdecode", spy)
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


def test_cv2_imdecode_exception_propagates_unwrapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_image(tmp_path / "a.png", _checkerboard())

    class _CustomError(RuntimeError):
        pass

    def failing_imdecode(buffer: np.ndarray, flags: int) -> np.ndarray | None:
        raise _CustomError("simulated decode crash")

    monkeypatch.setattr(cv2, "imdecode", failing_imdecode)
    with pytest.raises(_CustomError, match="simulated decode crash"):
        build_perceptual_hash_manifest(tmp_path, algorithm=_PHASH)


def test_file_deleted_between_discovery_and_read_raises_file_not_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "a.png"
    _write_image(target, _checkerboard())
    real_discover_images = dataset_module.discover_images

    def vanishing_discover_images(
        root: str | os.PathLike[str], **kwargs: object
    ) -> tuple[Path, ...]:
        paths = real_discover_images(root, **kwargs)  # type: ignore[arg-type]
        target.unlink()
        return paths

    monkeypatch.setattr(dataset_module, "discover_images", vanishing_discover_images)
    with pytest.raises(FileNotFoundError):
        build_perceptual_hash_manifest(tmp_path, algorithm=_PHASH)


def test_artificial_read_error_propagates_unwrapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "a.png"
    _write_image(target, _checkerboard())

    class _CustomOSError(OSError):
        pass

    real_read_bytes = Path.read_bytes

    def failing_read_bytes(self: Path) -> bytes:
        if self == target:
            raise _CustomOSError("simulated read failure")
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", failing_read_bytes)
    monkeypatch.setattr(cv2, "imdecode", _Boom())

    before = {path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*")}
    with pytest.raises(_CustomOSError, match="simulated read failure"):
        build_perceptual_hash_manifest(tmp_path, algorithm=_PHASH)
    after = {path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*")}
    assert after == before


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


# =====================================================================================
# split_dataset / DatasetSplit
# =====================================================================================
#
# split_dataset/DatasetSplit are an entirely separate, independent utility from
# build_perceptual_hash_manifest above -- they share this module and this test file only
# because both are dataset-workflow orchestration (see docs/design/0.4.0a2-dataset-split.md).


class _CustomSequence(Sequence):
    """Minimal, real `collections.abc.Sequence` subclass -- not a list/tuple."""

    def __init__(self, data: Sequence[object]) -> None:
        self._data = tuple(data)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index):  # type: ignore[no-untyped-def]
        return self._data[index]


class _AlwaysEqual:
    """An object with a permissive `__eq__` and (as a consequence) no `__hash__` -- Python sets
    `__hash__` to `None` automatically for any class that defines `__eq__` without also defining
    `__hash__`, making this class both unhashable and always "equal" to anything."""

    def __eq__(self, other: object) -> bool:
        return True


def _make_generator(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


# --- signature ---


def test_split_dataset_signature() -> None:
    signature = inspect.signature(split_dataset)
    parameters = signature.parameters
    assert list(parameters) == ["items", "train", "validation", "rng"]
    assert parameters["items"].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameters["items"].default is inspect.Parameter.empty
    assert parameters["train"].kind == inspect.Parameter.KEYWORD_ONLY
    assert parameters["train"].default is inspect.Parameter.empty
    assert parameters["validation"].kind == inspect.Parameter.KEYWORD_ONLY
    assert parameters["validation"].default == 0.0
    assert parameters["rng"].kind == inspect.Parameter.KEYWORD_ONLY
    assert parameters["rng"].default is inspect.Parameter.empty


# --- accepted input forms ---


def test_accepts_list() -> None:
    result = split_dataset(["a", "b", "c"], train=0.7, rng=_make_generator(0))
    assert len(result.train) + len(result.validation) + len(result.test) == 3


def test_accepts_tuple() -> None:
    result = split_dataset(("a", "b", "c"), train=0.7, rng=_make_generator(0))
    assert len(result.train) + len(result.validation) + len(result.test) == 3


def test_accepts_custom_sequence() -> None:
    result = split_dataset(_CustomSequence(["a", "b", "c"]), train=0.7, rng=_make_generator(0))
    assert len(result.train) + len(result.validation) + len(result.test) == 3


def test_accepts_range() -> None:
    result = split_dataset(range(10), train=0.7, validation=0.15, rng=_make_generator(0))
    assert len(result.train) + len(result.validation) + len(result.test) == 10


def test_accepts_discover_images_result(tmp_path: Path) -> None:
    _write_image(tmp_path / "a.png", _checkerboard())
    _write_image(tmp_path / "b.png", _checkerboard())
    from improcv.discovery import discover_images

    paths = discover_images(tmp_path)
    result = split_dataset(paths, train=0.5, validation=0.5, rng=_make_generator(0))
    assert set(result.train) | set(result.validation) | set(result.test) == set(paths)


def test_accepts_discover_image_mask_pairs_result(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    mask_root = tmp_path / "masks"
    image_root.mkdir()
    mask_root.mkdir()
    (image_root / "a.jpg").write_bytes(b"")
    (mask_root / "a.png").write_bytes(b"")
    (image_root / "b.jpg").write_bytes(b"")
    (mask_root / "b.png").write_bytes(b"")

    pairs = discover_image_mask_pairs(image_root, mask_root)
    result = split_dataset(pairs, train=0.5, validation=0.5, rng=_make_generator(0))
    assert set(result.train) | set(result.validation) | set(result.test) == set(pairs)


# --- rejected input forms ---


def test_rejects_str_items() -> None:
    with pytest.raises(TypeError, match="Sequence"):
        split_dataset("images", train=0.5, rng=_make_generator(0))


def test_rejects_bytes_items() -> None:
    with pytest.raises(TypeError, match="Sequence"):
        split_dataset(b"images", train=0.5, rng=_make_generator(0))


def test_rejects_bytearray_items() -> None:
    with pytest.raises(TypeError, match="Sequence"):
        split_dataset(bytearray(b"images"), train=0.5, rng=_make_generator(0))


def test_rejects_bare_path_items() -> None:
    with pytest.raises(TypeError, match="Sequence"):
        split_dataset(Path("images"), train=0.5, rng=_make_generator(0))  # type: ignore[arg-type]


def test_rejects_mapping_items() -> None:
    with pytest.raises(TypeError, match="Mapping"):
        split_dataset({"a": 1, "b": 2}, train=0.5, rng=_make_generator(0))  # type: ignore[arg-type]


def test_rejects_generator_items() -> None:
    with pytest.raises(TypeError, match="Sequence"):
        split_dataset((x for x in range(3)), train=0.5, rng=_make_generator(0))  # type: ignore[arg-type]


def test_rejects_iterator_items() -> None:
    with pytest.raises(TypeError, match="Sequence"):
        split_dataset(iter([1, 2, 3]), train=0.5, rng=_make_generator(0))  # type: ignore[arg-type]


def test_rejects_ndarray_items() -> None:
    with pytest.raises(TypeError, match="numpy.ndarray"):
        split_dataset(np.array([1, 2, 3]), train=0.5, rng=_make_generator(0))  # type: ignore[arg-type]


def test_rejects_int_items() -> None:
    with pytest.raises(TypeError, match="Sequence"):
        split_dataset(3, train=0.5, rng=_make_generator(0))  # type: ignore[arg-type]


def test_rejects_none_items() -> None:
    with pytest.raises(TypeError, match="Sequence"):
        split_dataset(None, train=0.5, rng=_make_generator(0))  # type: ignore[arg-type]


def test_items_error_message_names_the_type() -> None:
    with pytest.raises(TypeError, match="items"):
        split_dataset("images", train=0.5, rng=_make_generator(0))
    with pytest.raises(TypeError) as exc_info:
        split_dataset(3, train=0.5, rng=_make_generator(0))  # type: ignore[arg-type]
    assert "int" in str(exc_info.value)


def test_does_not_iterate_items_during_validation() -> None:
    class _ExplodingIterAttempt(Sequence):
        def __len__(self) -> int:
            return 3

        def __getitem__(self, index):  # type: ignore[no-untyped-def]
            return index

        def __iter__(self):  # type: ignore[no-untyped-def]
            raise AssertionError("split_dataset must not iterate items during validation")

    # A real Sequence still must not be iterated (only indexed via permutation) --
    # this exercises the whole call, not just the type-check stage.
    result = split_dataset(_ExplodingIterAttempt(), train=1.0, rng=_make_generator(0))
    assert len(result.train) == 3


# --- ratio validation ---


def test_rejects_string_train() -> None:
    with pytest.raises(TypeError):
        split_dataset([1, 2, 3], train="0.5", rng=_make_generator(0))  # type: ignore[arg-type]


def test_rejects_none_train() -> None:
    with pytest.raises(TypeError):
        split_dataset([1, 2, 3], train=None, rng=_make_generator(0))  # type: ignore[arg-type]


def test_rejects_complex_train() -> None:
    with pytest.raises(TypeError):
        split_dataset([1, 2, 3], train=complex(0.5, 0.0), rng=_make_generator(0))  # type: ignore[arg-type]


def test_rejects_bool_train() -> None:
    with pytest.raises(TypeError):
        split_dataset([1, 2, 3], train=True, rng=_make_generator(0))  # type: ignore[arg-type]


def test_rejects_bool_validation() -> None:
    with pytest.raises(TypeError):
        split_dataset([1, 2, 3], train=0.5, validation=False, rng=_make_generator(0))  # type: ignore[arg-type]


def test_accepts_int_train() -> None:
    result = split_dataset([1, 2, 3], train=1, rng=_make_generator(0))
    assert len(result.train) == 3


def test_rejects_nan_train() -> None:
    with pytest.raises(ValueError):
        split_dataset([1, 2, 3], train=float("nan"), rng=_make_generator(0))


def test_rejects_positive_infinity_train() -> None:
    with pytest.raises(ValueError):
        split_dataset([1, 2, 3], train=float("inf"), rng=_make_generator(0))


def test_rejects_negative_infinity_train() -> None:
    with pytest.raises(ValueError):
        split_dataset([1, 2, 3], train=float("-inf"), rng=_make_generator(0))


def test_rejects_nan_validation() -> None:
    with pytest.raises(ValueError):
        split_dataset([1, 2, 3], train=0.5, validation=float("nan"), rng=_make_generator(0))


def test_rejects_train_below_zero() -> None:
    with pytest.raises(ValueError):
        split_dataset([1, 2, 3], train=-0.1, rng=_make_generator(0))


def test_rejects_train_above_one() -> None:
    with pytest.raises(ValueError):
        split_dataset([1, 2, 3], train=1.1, rng=_make_generator(0))


def test_rejects_validation_below_zero() -> None:
    with pytest.raises(ValueError):
        split_dataset([1, 2, 3], train=0.5, validation=-0.1, rng=_make_generator(0))


def test_rejects_validation_above_one() -> None:
    with pytest.raises(ValueError):
        split_dataset([1, 2, 3], train=0.0, validation=1.1, rng=_make_generator(0))


def test_accepts_train_exactly_zero() -> None:
    result = split_dataset([1, 2, 3], train=0.0, validation=1.0, rng=_make_generator(0))
    assert result.train == ()


def test_accepts_train_exactly_one() -> None:
    result = split_dataset([1, 2, 3], train=1.0, rng=_make_generator(0))
    assert result.validation == ()
    assert result.test == ()


def test_rejects_sum_above_one() -> None:
    with pytest.raises(ValueError):
        split_dataset([1, 2, 3], train=0.7, validation=0.4, rng=_make_generator(0))


def test_accepts_sum_exactly_one() -> None:
    result = split_dataset([1, 2, 3, 4], train=0.5, validation=0.5, rng=_make_generator(0))
    assert result.test == ()


def test_train_validated_before_validation() -> None:
    # train's error must win even when validation is also invalid -- exact validation order
    # is part of the frozen design contract (items -> train -> validation -> sum -> rng).
    with pytest.raises(TypeError) as exc_info:
        split_dataset([1, 2, 3], train="bad", validation="also bad", rng=_make_generator(0))  # type: ignore[arg-type]
    assert "train" in str(exc_info.value)


def test_validation_validated_before_sum_check() -> None:
    with pytest.raises(ValueError) as exc_info:
        split_dataset([1, 2, 3], train=0.9, validation=1.5, rng=_make_generator(0))
    assert "validation" in str(exc_info.value)


def test_rng_validated_last() -> None:
    # A bad rng only surfaces after every ratio check has already passed -- train/validation here
    # are both valid, so the TypeError below can only come from the rng check itself.
    with pytest.raises(TypeError, match="rng"):
        split_dataset([1, 2, 3], train=0.5, validation=0.3, rng="not-a-generator")  # type: ignore[arg-type]
    # And an invalid ratio is still reported even with an otherwise-valid rng.
    with pytest.raises(ValueError):
        split_dataset([1, 2, 3], train=2.0, rng=_make_generator(0))


# --- Largest Remainder Method: exact counts table (design doc section 8) ---


@pytest.mark.parametrize(
    ("n", "train", "validation", "expected"),
    [
        (0, 0.70, 0.15, (0, 0, 0)),
        (1, 0.70, 0.15, (1, 0, 0)),
        (2, 0.70, 0.15, (2, 0, 0)),
        (3, 0.70, 0.15, (2, 0, 1)),
        (10, 0.70, 0.15, (7, 1, 2)),
        (11, 0.70, 0.15, (8, 1, 2)),
        (10, 0.80, 0.10, (8, 1, 1)),
        (10, 0.50, 0.25, (5, 3, 2)),
        (3, 1 / 3, 1 / 3, (1, 1, 1)),
        (10, 1 / 3, 1 / 3, (3, 3, 4)),
        (2, 0.50, 0.50, (1, 1, 0)),
        (1, 1.00, 0.00, (1, 0, 0)),
        (5, 1.00, 0.00, (5, 0, 0)),
        (5, 0.00, 1.00, (0, 5, 0)),
    ],
)
def test_split_counts_match_design_table(
    n: int, train: float, validation: float, expected: tuple[int, int, int]
) -> None:
    items = list(range(n))
    result = split_dataset(items, train=train, validation=validation, rng=_make_generator(0))
    assert (len(result.train), len(result.validation), len(result.test)) == expected
    assert len(result.train) + len(result.validation) + len(result.test) == n


def test_rounding_protects_0_7_0_15_against_naive_tie_break(tmp_path: Path) -> None:
    """Regression guard: a 'simplifying' refactor that assumed train=0.7/validation=0.15 gives an
    exact validation/test tie at n=10 (and resolves it toward validation) would silently break
    this -- the true IEEE 754 tie-break goes to test. See design doc section 7."""
    items = list(range(10))
    result = split_dataset(items, train=0.70, validation=0.15, rng=_make_generator(0))
    assert (len(result.train), len(result.validation), len(result.test)) == (7, 1, 2)


def test_rounding_protects_one_third_one_third_against_naive_tie_break() -> None:
    """Regression guard: a 'simplifying' refactor that assumed train=validation=1/3 gives an exact
    3-way tie at n=10 (and resolves it toward train) would silently break this -- the true IEEE 754
    tie-break goes to test. See design doc section 7."""
    items = list(range(10))
    result = split_dataset(items, train=1 / 3, validation=1 / 3, rng=_make_generator(0))
    assert (len(result.train), len(result.validation), len(result.test)) == (3, 3, 4)


def test_split_sizes_are_independent_of_rng() -> None:
    items = list(range(37))
    sizes = {
        (len(r.train), len(r.validation), len(r.test))
        for r in (
            split_dataset(items, train=0.6, validation=0.2, rng=_make_generator(seed))
            for seed in range(20)
        )
    }
    assert sizes == {(22, 8, 7)}


# --- NumPy scalar ratio normalization (post-merge corrective fix) ---
#
# Accepted NumPy real scalar ratios (np.float16/np.float32/np.float64/NumPy integer scalars) must
# be converted to plain Python float *after* validation and *before* every composite ratio
# computation (the train + validation sum check, test's ratio, and the Largest Remainder
# allocation) -- never let the caller's NumPy scalar dtype/promotion rules leak into that
# arithmetic. See docs/design/0.4.0a2-dataset-split.md sections 6-9.


def _reference_counts_from_promoted_floats(
    n: int, train: float, validation: float
) -> tuple[int, int, int]:
    """Independent reference for the Largest Remainder Method, computed explicitly on
    `float(train)`/`float(validation)` -- the exact contract this fix restores."""
    train_f = float(train)
    validation_f = float(validation)
    test_f = 1.0 - train_f - validation_f
    ideal = [n * train_f, n * validation_f, n * test_f]
    floors = [int(value // 1) for value in ideal]
    remainder = n - sum(floors)
    fractions = [ideal[index] - floors[index] for index in range(3)]
    order = sorted(range(3), key=lambda index: (-fractions[index], index))
    counts = list(floors)
    for index in order[:remainder]:
        counts[index] += 1
    return counts[0], counts[1], counts[2]


def test_float16_allocation_uses_promoted_represented_value() -> None:
    # np.float16("0.8") represents approximately 0.7998046875, not exactly 0.8 -- the expected
    # counts below come from float(np.float16("0.8")), not from a hand-typed "0.8" literal, per
    # the task's own explicit warning against hardcoding 4000/500/500.
    n = 5000
    train = np.float16("0.8")
    validation = np.float16("0.1")
    expected = _reference_counts_from_promoted_floats(n, float(train), float(validation))
    assert expected == (3999, 500, 501)

    result = split_dataset(
        list(range(n)),
        train=train,  # type: ignore[arg-type]
        validation=validation,  # type: ignore[arg-type]
        rng=_make_generator(0),
    )
    assert (len(result.train), len(result.validation), len(result.test)) == expected


def test_float32_allocation_uses_promoted_represented_value() -> None:
    n = 100
    train = np.float32("0.22593926")
    validation = np.float32("0.04812143")
    expected = _reference_counts_from_promoted_floats(n, float(train), float(validation))
    assert expected == (22, 5, 73)

    result = split_dataset(
        list(range(n)),
        train=train,  # type: ignore[arg-type]
        validation=validation,  # type: ignore[arg-type]
        rng=_make_generator(0),
    )
    assert (len(result.train), len(result.validation), len(result.test)) == expected


def test_float64_allocation_matches_plain_python_float() -> None:
    n = 2048
    train = np.float64(0.6)
    validation = np.float64(0.25)
    expected = _reference_counts_from_promoted_floats(n, float(train), float(validation))
    result = split_dataset(
        list(range(n)),
        train=train,  # type: ignore[arg-type]
        validation=validation,  # type: ignore[arg-type]
        rng=_make_generator(0),
    )
    assert (len(result.train), len(result.validation), len(result.test)) == expected
    # float64 already is binary64 -- promotion is a no-op, so this must also match the plain-float
    # call with the same numeric value.
    plain_result = split_dataset(list(range(n)), train=0.6, validation=0.25, rng=_make_generator(0))
    assert (
        len(plain_result.train),
        len(plain_result.validation),
        len(plain_result.test),
    ) == expected


def test_numpy_integer_scalar_train_is_legal() -> None:
    # A NumPy integer scalar is numbers.Real (via numbers.Integral) and not bool -- legal wherever
    # a plain int is legal, matching require_range's existing acceptance of any numbers.Real.
    result = split_dataset(list(range(10)), train=np.int64(1), rng=_make_generator(0))  # type: ignore[arg-type]
    assert len(result.train) == 10


def test_low_precision_sum_is_rejected_after_promotion() -> None:
    # np.float16("0.001") + np.float16("0.999") rounds to exactly float16(1.0) in float16
    # arithmetic, but the binary64 values those two float16s actually represent sum to slightly
    # more than 1.0 -- the promoted sum, not the native float16 sum, must govern validation.
    train = np.float16("0.001")
    validation = np.float16("0.999")
    assert train + validation == np.float16(1.0)  # native float16 arithmetic: sanity check
    assert float(train) + float(validation) > 1.0  # binary64-promoted: what must be rejected

    with pytest.raises(ValueError, match="train \\+ validation"):
        split_dataset(
            list(range(10)),
            train=train,  # type: ignore[arg-type]
            validation=validation,  # type: ignore[arg-type]
            rng=_make_generator(0),
        )


def test_low_precision_sum_error_message_reports_promoted_values() -> None:
    train = np.float16("0.001")
    validation = np.float16("0.999")
    promoted_train = float(train)
    promoted_validation = float(validation)
    promoted_sum = promoted_train + promoted_validation
    assert promoted_sum > 1.0  # sanity: this is the promoted value that must be rejected

    with pytest.raises(ValueError) as exc_info:
        split_dataset(
            list(range(10)),
            train=train,  # type: ignore[arg-type]
            validation=validation,  # type: ignore[arg-type]
            rng=_make_generator(0),
        )
    message = str(exc_info.value)
    # design doc section 20's exception contract: the message "names both values and their exact
    # sum" -- checked here against the *promoted* (binary64) values actually used for the
    # rejection decision, not the native float16 values/sum (which would render as exactly "1.0"
    # and must not appear as the reported sum).
    assert repr(promoted_train) in message
    assert repr(promoted_validation) in message
    assert repr(promoted_sum) in message


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("n", [100, 2048, 5000, 10000])
def test_scalar_dtype_differential_matches_promoted_float_reference(dtype: type, n: int) -> None:
    """For a deterministic set of legal ratio/dataset-size combinations, split_dataset's counts
    for a NumPy scalar dtype must equal the reference computed from float(train)/float(validation)
    -- no Hypothesis dependency, plain fixed-seed stdlib random, matching this project's own
    differential-testing convention elsewhere."""
    rng_for_cases = random.Random(20260807 + n)
    executed = 0
    for _ in range(10):
        train_raw = rng_for_cases.uniform(0.0, 1.0)
        validation_raw = rng_for_cases.uniform(0.0, max(0.0, 1.0 - train_raw))
        train = dtype(train_raw)
        validation = dtype(validation_raw)
        if float(train) + float(validation) > 1.0:
            # A dtype with enough rounding error to violate the sum contract after promotion is
            # itself covered by test_low_precision_sum_is_rejected_after_promotion -- skip here to
            # keep this test focused on the allocation-matches-reference property.
            continue

        expected = _reference_counts_from_promoted_floats(n, float(train), float(validation))
        result = split_dataset(
            list(range(n)),
            train=train,  # type: ignore[arg-type]
            validation=validation,  # type: ignore[arg-type]
            rng=_make_generator(0),
        )
        actual = (len(result.train), len(result.validation), len(result.test))
        assert actual == expected, (
            f"dtype={dtype.__name__} n={n} train={train!r} validation={validation!r} "
            f"expected={expected} actual={actual}"
        )
        assert sum(actual) == n
        executed += 1

    # This fixed seed, for every (dtype, n) parametrization, is verified to keep all 10 draws
    # within the sum<=1.0 contract after promotion (the sampling range for validation_raw already
    # makes train_raw + validation_raw <= 1.0 in exact real arithmetic; only dtype rounding could
    # ever push the promoted sum above 1.0, and it does not for this seed) -- assert the exact
    # count so a future change to the seed, the draw logic, or dtype rounding that starts skipping
    # cases fails loudly here instead of silently verifying fewer than 10 comparisons.
    assert executed == 10


# --- empty input ---


def test_empty_list_returns_empty_split() -> None:
    result = split_dataset([], train=0.7, validation=0.15, rng=_make_generator(0))
    assert result == DatasetSplit(train=(), validation=(), test=())


def test_empty_tuple_returns_empty_split() -> None:
    result = split_dataset((), train=0.7, rng=_make_generator(0))
    assert result == DatasetSplit(train=(), validation=(), test=())


def test_empty_input_still_validates_bad_train() -> None:
    with pytest.raises(ValueError):
        split_dataset([], train=1.5, rng=_make_generator(0))


def test_empty_input_still_validates_bad_rng() -> None:
    with pytest.raises(TypeError):
        split_dataset([], train=0.5, rng="nope")  # type: ignore[arg-type]


# --- occurrence semantics ---


def test_duplicate_values_are_two_independent_occurrences() -> None:
    items = ["a", "a", "b"]
    result = split_dataset(items, train=1.0, rng=_make_generator(0))
    assert sorted(result.train) == ["a", "a", "b"]


def test_duplicate_values_split_across_train_and_test() -> None:
    items = ["a", "a", "b", "c"]
    result = split_dataset(items, train=0.5, rng=_make_generator(0))
    combined = list(result.train) + list(result.validation) + list(result.test)
    assert sorted(combined) == sorted(items)
    assert len(combined) == len(items)


def test_unhashable_values_do_not_raise() -> None:
    items = [_AlwaysEqual(), _AlwaysEqual(), _AlwaysEqual()]
    with pytest.raises(TypeError):
        hash(items[0])  # sanity: really unhashable
    result = split_dataset(items, train=1.0, rng=_make_generator(0))
    assert len(result.train) == 3


def test_always_equal_objects_are_not_deduplicated() -> None:
    items = [_AlwaysEqual() for _ in range(5)]
    result = split_dataset(items, train=1.0, rng=_make_generator(0))
    assert len(result.train) == 5


def test_every_input_position_appears_exactly_once() -> None:
    n = 50
    items = list(range(n))
    result = split_dataset(items, train=0.6, validation=0.2, rng=_make_generator(0))
    combined = sorted(list(result.train) + list(result.validation) + list(result.test))
    assert combined == list(range(n))


def test_no_position_is_omitted_or_duplicated_across_many_seeds() -> None:
    n = 30
    items = list(range(n))
    for seed in range(50):
        result = split_dataset(items, train=0.5, validation=0.3, rng=_make_generator(seed))
        combined = sorted(list(result.train) + list(result.validation) + list(result.test))
        assert combined == list(range(n))


# --- RNG contract ---


def test_rejects_int_seed() -> None:
    with pytest.raises(TypeError, match="numpy.random.Generator"):
        split_dataset([1, 2, 3], train=0.5, rng=0)  # type: ignore[arg-type]


def test_rejects_random_state() -> None:
    with pytest.raises(TypeError, match="numpy.random.Generator"):
        split_dataset([1, 2, 3], train=0.5, rng=np.random.RandomState(0))  # type: ignore[arg-type]


def test_rejects_none_rng() -> None:
    with pytest.raises(TypeError, match="numpy.random.Generator"):
        split_dataset([1, 2, 3], train=0.5, rng=None)  # type: ignore[arg-type]


def test_rejects_numpy_random_module_itself() -> None:
    with pytest.raises(TypeError, match="numpy.random.Generator"):
        split_dataset([1, 2, 3], train=0.5, rng=np.random)  # type: ignore[arg-type]


def test_rejects_duck_typed_rng() -> None:
    class _FakeGenerator:
        def permutation(self, n: int):  # type: ignore[no-untyped-def]
            return np.arange(n)

    with pytest.raises(TypeError, match="numpy.random.Generator"):
        split_dataset([1, 2, 3], train=0.5, rng=_FakeGenerator())  # type: ignore[arg-type]


def test_two_independent_generators_same_seed_produce_identical_results() -> None:
    items = [f"item{i}" for i in range(41)]
    result_a = split_dataset(items, train=0.6, validation=0.2, rng=np.random.default_rng(2026))
    result_b = split_dataset(items, train=0.6, validation=0.2, rng=np.random.default_rng(2026))
    assert result_a == result_b


def test_rng_state_advances_for_a_nontrivial_permutation() -> None:
    """`split_dataset` uses the caller's `rng` directly and never clones it, so a nontrivial
    permutation (n > 1) is expected to advance its state. This checks state consumption directly,
    not via the probabilistic (and not actually guaranteed) proxy of "the two results differ" --
    see docs/design/0.4.0a2-dataset-split.md section 12/section "RNG contract correction" for why
    result inequality is not something this function promises."""
    items = [f"item{i}" for i in range(41)]
    rng = np.random.default_rng(2026)
    state_before = copy.deepcopy(rng.bit_generator.state)
    split_dataset(items, train=0.6, validation=0.2, rng=rng)
    state_after = copy.deepcopy(rng.bit_generator.state)
    assert state_before != state_after


def test_split_dataset_succeeds_and_is_deterministic_for_empty_items() -> None:
    """`n=0` is a legal, well-defined edge case (see the "empty input" tests above). This confirms
    execution succeeds, the result has the documented empty-split shape, and the determinism
    guarantee -- same items + equivalent initial rng state -> same result -- holds at this
    boundary. Deliberately makes no assertion about whether `rng`'s own state advances for a
    trivial permutation: `split_dataset`'s own docstring promises neither that it does nor that it
    doesn't, so a test asserting either direction would freeze a NumPy implementation detail that
    isn't part of the public contract (see docs/design/0.4.0a2-dataset-split.md's RNG contract
    correction)."""
    result_a = split_dataset([], train=0.6, validation=0.2, rng=np.random.default_rng(2026))
    result_b = split_dataset([], train=0.6, validation=0.2, rng=np.random.default_rng(2026))
    assert result_a == result_b == DatasetSplit(train=(), validation=(), test=())


def test_split_dataset_succeeds_and_is_deterministic_for_single_item() -> None:
    """Same rationale as the empty-input case above, for a single-element permutation."""
    result_a = split_dataset(["only"], train=1.0, rng=np.random.default_rng(2026))
    result_b = split_dataset(["only"], train=1.0, rng=np.random.default_rng(2026))
    assert result_a == result_b == DatasetSplit(train=("only",), validation=(), test=())


def test_two_different_rng_states_are_not_guaranteed_to_differ_in_result() -> None:
    """The converse of determinism is explicitly not promised: this test documents, rather than
    silently assumes, that split_dataset never asserts result_a != result_b for two independently
    seeded generators -- it only asserts equality for equivalent initial states (see
    test_two_independent_generators_same_seed_produce_identical_results above). No assertion here
    claims two different seeds must produce two different results."""
    items = [f"item{i}" for i in range(41)]
    result_a = split_dataset(items, train=0.6, validation=0.2, rng=np.random.default_rng(1))
    result_b = split_dataset(items, train=0.6, validation=0.2, rng=np.random.default_rng(2))
    # Deliberately no assertion on result_a vs result_b -- both outcomes (equal or different) are
    # legal under the documented contract. This test exists to make the absence of that assertion
    # an explicit, reviewable decision rather than a silent gap.
    assert isinstance(result_a, DatasetSplit)
    assert isinstance(result_b, DatasetSplit)


def test_restored_generator_state_reproduces_identical_result() -> None:
    items = [f"item{i}" for i in range(41)]
    rng = np.random.default_rng(77)
    saved_state = rng.bit_generator.state
    result_a = split_dataset(items, train=0.5, validation=0.3, rng=rng)

    replay_rng = np.random.default_rng(0)
    replay_rng.bit_generator.state = saved_state
    result_b = split_dataset(items, train=0.5, validation=0.3, rng=replay_rng)
    assert result_a == result_b


def test_does_not_read_or_write_global_numpy_random_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("split_dataset must not touch the global numpy.random state")

    monkeypatch.setattr(np.random, "seed", _boom)
    monkeypatch.setattr(np.random, "random", _boom)
    monkeypatch.setattr(np.random, "randint", _boom)

    result = split_dataset([1, 2, 3, 4], train=0.5, validation=0.25, rng=_make_generator(0))
    assert len(result.train) + len(result.validation) + len(result.test) == 4


def test_does_not_use_python_stdlib_random(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("split_dataset must not touch Python's stdlib random module")

    monkeypatch.setattr(random, "shuffle", _boom)
    monkeypatch.setattr(random, "random", _boom)

    result = split_dataset([1, 2, 3, 4], train=0.5, validation=0.25, rng=_make_generator(0))
    assert len(result.train) + len(result.validation) + len(result.test) == 4


# --- ordering ---


def test_ordering_matches_permutation_for_a_frozen_seed() -> None:
    items = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    rng = np.random.default_rng(12345)
    expected_indices = np.random.default_rng(12345).permutation(len(items))
    result = split_dataset(items, train=0.6, validation=0.2, rng=rng)

    expected_train = tuple(items[int(i)] for i in expected_indices[:6])
    expected_validation = tuple(items[int(i)] for i in expected_indices[6:8])
    expected_test = tuple(items[int(i)] for i in expected_indices[8:])
    assert result.train == expected_train
    assert result.validation == expected_validation
    assert result.test == expected_test


def test_ordering_is_not_resorted_to_source_order() -> None:
    # Seed-independent check that does not assume any specific permutation outcome: if the
    # implementation ever resorted to source order, train would equal list(range(30)) for every
    # seed -- assert that at least one of several seeds disagrees.
    items = list(range(30))
    outcomes = {
        split_dataset(items, train=1.0, rng=_make_generator(seed)).train for seed in range(10)
    }
    assert any(outcome != tuple(range(30)) for outcome in outcomes)


# --- ImageMaskPair composability ---


def test_image_mask_pair_never_separated_across_splits() -> None:
    pairs = tuple(ImageMaskPair(image=Path(f"{i}.jpg"), mask=Path(f"{i}.png")) for i in range(20))
    result = split_dataset(pairs, train=0.6, validation=0.2, rng=_make_generator(0))
    for pair in list(result.train) + list(result.validation) + list(result.test):
        assert isinstance(pair, ImageMaskPair)
        # Each pair is drawn as one atomic unit -- its own image/mask fields are always
        # consistent with each other, never recombined with a different pair's fields.
        stem = pair.image.stem
        assert pair.mask.stem == stem


def test_no_split_image_mask_pairs_function_exists() -> None:
    assert not hasattr(dataset_module, "split_image_mask_pairs")
    assert not hasattr(im, "split_image_mask_pairs")


# --- result type ---


def test_dataset_split_is_frozen() -> None:
    result = DatasetSplit(train=(1,), validation=(), test=())
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.train = (2,)  # type: ignore[misc]


def test_dataset_split_uses_slots() -> None:
    assert DatasetSplit.__slots__ == ("train", "validation", "test")
    result = DatasetSplit(train=(), validation=(), test=())
    assert not hasattr(result, "__dict__")


def test_dataset_split_equality() -> None:
    a = DatasetSplit(train=(1, 2), validation=(3,), test=())
    b = DatasetSplit(train=(1, 2), validation=(3,), test=())
    c = DatasetSplit(train=(1,), validation=(3,), test=())
    assert a == b
    assert a != c


def test_dataset_split_repr_is_default_dataclass_repr() -> None:
    result = DatasetSplit(train=(1,), validation=(2,), test=(3,))
    assert repr(result) == "DatasetSplit(train=(1,), validation=(2,), test=(3,))"


def test_dataset_split_field_order() -> None:
    fields = dataclasses.fields(DatasetSplit)
    assert [field.name for field in fields] == ["train", "validation", "test"]


def test_dataset_split_has_no_custom_post_init() -> None:
    assert "__post_init__" not in DatasetSplit.__dict__


def test_dataset_split_manual_construction() -> None:
    result = DatasetSplit(train=("a",), validation=("b",), test=("c",))
    assert result.train == ("a",)
    assert result.validation == ("b",)
    assert result.test == ("c",)


def test_dataset_split_is_generic() -> None:
    # Runtime smoke: subscripting a Generic dataclass must not raise.
    DatasetSplit[int]
    DatasetSplit[Path]


# --- purity: no filesystem I/O, no mutation ---


def test_split_dataset_performs_no_filesystem_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("split_dataset must not touch the filesystem")

    monkeypatch.setattr(os, "stat", _boom)
    monkeypatch.setattr(os, "scandir", _boom)
    monkeypatch.setattr(Path, "open", _boom)
    monkeypatch.setattr("builtins.open", _boom)

    non_existent_paths = tuple(Path(f"/does/not/exist/{i}.png") for i in range(5))
    result = split_dataset(non_existent_paths, train=0.6, validation=0.2, rng=_make_generator(0))
    assert len(result.train) + len(result.validation) + len(result.test) == 5


def test_split_dataset_does_not_mutate_items() -> None:
    items = [1, 2, 3, 4, 5]
    original = list(items)
    split_dataset(items, train=0.6, validation=0.2, rng=_make_generator(0))
    assert items == original


def test_split_dataset_does_not_mutate_a_tuple_of_paths() -> None:
    items = tuple(Path(f"{i}.png") for i in range(5))
    original = tuple(items)
    split_dataset(items, train=0.6, rng=_make_generator(0))
    assert items == original


# --- randomized / differential property test (no Hypothesis dependency) ---


class _Sample:
    """A plain object with no `__hash__`/meaningful `__eq__`, used to prove the property test
    does not rely on hashability or value equality -- only index identity."""

    __slots__ = ("tag",)

    def __init__(self, tag: int) -> None:
        self.tag = tag


def _reference_largest_remainder(n: int, train: float, validation: float) -> tuple[int, int, int]:
    """Independent reference implementation of the Largest Remainder Method (design doc section 7),
    written separately from `improcv.dataset._largest_remainder_counts` so the property test below
    checks the function under test against independently-derived logic, not its own internals."""
    import math as _math

    test_ratio = 1.0 - train - validation
    ideal = {"train": n * train, "validation": n * validation, "test": n * test_ratio}
    floors = {key: _math.floor(value) for key, value in ideal.items()}
    remaining = n - sum(floors.values())
    fracs = {key: ideal[key] - floors[key] for key in ideal}
    priority = ["train", "validation", "test"]
    order = sorted(priority, key=lambda key: (-fracs[key], priority.index(key)))
    counts = dict(floors)
    for key in order[:remaining]:
        counts[key] += 1
    return counts["train"], counts["validation"], counts["test"]


def test_split_dataset_differential_property() -> None:
    seed = 20260807
    rng_for_cases = random.Random(seed)
    n_cases = 2000

    matched = 0
    for case_index in range(n_cases):
        n = rng_for_cases.choice([0, 1, 2, 3, 5, 10, 11, 25, 50, 100])
        train = rng_for_cases.choice(
            [0.0, 0.1, 0.15, 1 / 3, 0.5, 0.7, 0.8, 0.9999, 1.0, rng_for_cases.random()]
        )
        remaining_budget = max(0.0, 1.0 - train)
        validation = rng_for_cases.choice(
            [0.0, min(0.1, remaining_budget), min(1 / 3, remaining_budget), remaining_budget]
        )
        validation = min(validation, remaining_budget)

        use_duplicates = rng_for_cases.random() < 0.3
        if use_duplicates and n > 0:
            distinct_count = max(1, n // 2)
            values = [rng_for_cases.randrange(distinct_count) for _ in range(n)]
        else:
            values = list(range(n))
        items: list[object] = (
            [_Sample(tag) for tag in values] if case_index % 5 == 0 else list(values)
        )

        split_rng = np.random.default_rng(case_index)
        result = split_dataset(items, train=train, validation=validation, rng=split_rng)

        expected_counts = _reference_largest_remainder(n, train, validation)
        actual_counts = (len(result.train), len(result.validation), len(result.test))
        assert actual_counts == expected_counts, (
            f"case {case_index}: n={n} train={train} validation={validation} "
            f"expected={expected_counts} actual={actual_counts}"
        )
        assert sum(actual_counts) == n

        # Occurrence coverage, checked by identity (works for both plain ints, which have
        # meaningful equality, and _Sample instances, which do not).
        combined = list(result.train) + list(result.validation) + list(result.test)
        assert len(combined) == n
        combined_ids = sorted(id(element) for element in combined)
        source_ids = sorted(id(element) for element in items)
        assert combined_ids == source_ids

        matched += 1

    assert matched == n_cases
