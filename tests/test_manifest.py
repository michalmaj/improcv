import dataclasses
from pathlib import Path, PurePosixPath

import pytest

import improcv as im
import improcv.manifest as manifest_module
from improcv.hashing import PerceptualHash, PerceptualHashAlgorithm
from improcv.manifest import PerceptualHashManifest, PerceptualHashManifestEntry

_PHASH = PerceptualHashAlgorithm.PHASH
_AVERAGE_HASH = PerceptualHashAlgorithm.AVERAGE_HASH


def _hash(
    hex_value: str,
    *,
    algorithm: PerceptualHashAlgorithm = _PHASH,
    hash_size: int = 2,
) -> PerceptualHash:
    """Build a `PerceptualHash` with an easily hand-verifiable value (see test_similarity.py)."""
    return PerceptualHash.from_hex(hex_value, algorithm=algorithm, hash_size=hash_size)


class _Boom:
    def __call__(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("manifest model/serialization must not perform this operation")


# =====================================================================================
# PerceptualHashManifestEntry
# =====================================================================================


def test_entry_valid_construction() -> None:
    entry = PerceptualHashManifestEntry(path=PurePosixPath("a.png"), hash=_hash("0"))
    assert entry.path == PurePosixPath("a.png")
    assert entry.hash == _hash("0")


def test_entry_equality_and_hashability() -> None:
    entry1 = PerceptualHashManifestEntry(path=PurePosixPath("a.png"), hash=_hash("0"))
    entry2 = PerceptualHashManifestEntry(path=PurePosixPath("a.png"), hash=_hash("0"))
    assert entry1 == entry2
    assert hash(entry1) == hash(entry2)
    assert {entry1, entry2} == {entry1}


def test_entry_is_frozen() -> None:
    entry = PerceptualHashManifestEntry(path=PurePosixPath("a.png"), hash=_hash("0"))
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.hash = _hash("1")  # type: ignore[misc]


def test_entry_rejects_str_path() -> None:
    with pytest.raises(TypeError, match="path"):
        PerceptualHashManifestEntry(path="a.png", hash=_hash("0"))  # type: ignore[arg-type]


def test_entry_rejects_platform_path_even_on_posix() -> None:
    # A concrete Path (PosixPath on POSIX) is not "exactly" a PurePosixPath, even though
    # PosixPath happens to inherit from PurePosixPath -- the entry requires the exact type.
    with pytest.raises(TypeError, match="path"):
        PerceptualHashManifestEntry(path=Path("a.png"), hash=_hash("0"))  # type: ignore[arg-type]


def test_entry_rejects_non_perceptual_hash() -> None:
    with pytest.raises(TypeError, match="hash"):
        PerceptualHashManifestEntry(path=PurePosixPath("a.png"), hash="0")  # type: ignore[arg-type]


def test_entry_rejects_absolute_path() -> None:
    with pytest.raises(ValueError, match="relative"):
        PerceptualHashManifestEntry(path=PurePosixPath("/images/a.png"), hash=_hash("0"))


def test_entry_rejects_empty_path() -> None:
    with pytest.raises(ValueError):
        PerceptualHashManifestEntry(path=PurePosixPath(""), hash=_hash("0"))


def test_entry_rejects_dot_dot_segment() -> None:
    with pytest.raises(ValueError, match="segment"):
        PerceptualHashManifestEntry(path=PurePosixPath("../image.png"), hash=_hash("0"))


def test_entry_rejects_dot_dot_in_middle() -> None:
    with pytest.raises(ValueError, match="segment"):
        PerceptualHashManifestEntry(path=PurePosixPath("a/../image.png"), hash=_hash("0"))


def test_entry_rejects_windows_drive() -> None:
    with pytest.raises(ValueError, match="drive"):
        PerceptualHashManifestEntry(path=PurePosixPath("C:/images/a.png"), hash=_hash("0"))


def test_entry_accepts_valid_unicode_path() -> None:
    entry = PerceptualHashManifestEntry(path=PurePosixPath("dane/żółw.png"), hash=_hash("0"))
    assert entry.path.as_posix() == "dane/żółw.png"


def test_entry_distinguishes_nfc_and_nfd_unicode() -> None:
    nfc = "café.png"  # "café.png", precomposed é
    nfd = "café.png"  # "café.png", combining acute accent
    assert nfc != nfd
    entry_nfc = PerceptualHashManifestEntry(path=PurePosixPath(nfc), hash=_hash("0"))
    entry_nfd = PerceptualHashManifestEntry(path=PurePosixPath(nfd), hash=_hash("0"))
    assert entry_nfc.path.as_posix() == nfc
    assert entry_nfd.path.as_posix() == nfd
    assert entry_nfc != entry_nfd


# =====================================================================================
# PerceptualHashManifest -- construction contract
# =====================================================================================


def test_empty_manifest_is_legal() -> None:
    manifest = PerceptualHashManifest(algorithm=_PHASH, hash_size=8, entries=())
    assert manifest.algorithm == _PHASH
    assert manifest.hash_size == 8
    assert manifest.entries == ()


def test_singleton_manifest() -> None:
    entry = PerceptualHashManifestEntry(path=PurePosixPath("a.png"), hash=_hash("0"))
    manifest = PerceptualHashManifest(algorithm=_PHASH, hash_size=2, entries=(entry,))
    assert manifest.entries == (entry,)


def test_multi_entry_manifest_in_canonical_order() -> None:
    entry_a = PerceptualHashManifestEntry(path=PurePosixPath("a.png"), hash=_hash("0"))
    entry_b = PerceptualHashManifestEntry(path=PurePosixPath("b.png"), hash=_hash("1"))
    manifest = PerceptualHashManifest(algorithm=_PHASH, hash_size=2, entries=(entry_a, entry_b))
    assert manifest.entries == (entry_a, entry_b)


def test_manifest_equality_and_hashability() -> None:
    entry = PerceptualHashManifestEntry(path=PurePosixPath("a.png"), hash=_hash("0"))
    manifest1 = PerceptualHashManifest(algorithm=_PHASH, hash_size=2, entries=(entry,))
    manifest2 = PerceptualHashManifest(algorithm=_PHASH, hash_size=2, entries=(entry,))
    assert manifest1 == manifest2
    assert hash(manifest1) == hash(manifest2)
    assert {manifest1, manifest2} == {manifest1}


def test_manifest_is_frozen() -> None:
    manifest = PerceptualHashManifest(algorithm=_PHASH, hash_size=2, entries=())
    with pytest.raises(dataclasses.FrozenInstanceError):
        manifest.hash_size = 4  # type: ignore[misc]


def test_manifest_rejects_bad_algorithm_type() -> None:
    with pytest.raises(TypeError, match="algorithm"):
        PerceptualHashManifest(algorithm="phash", hash_size=8, entries=())  # type: ignore[arg-type]


def test_manifest_rejects_bad_hash_size_type() -> None:
    with pytest.raises(TypeError):
        PerceptualHashManifest(algorithm=_PHASH, hash_size="8", entries=())  # type: ignore[arg-type]


def test_manifest_rejects_hash_size_below_minimum() -> None:
    with pytest.raises(ValueError, match="hash_size"):
        PerceptualHashManifest(algorithm=_PHASH, hash_size=1, entries=())


def test_manifest_rejects_hash_size_above_maximum() -> None:
    with pytest.raises(ValueError, match="hash_size"):
        PerceptualHashManifest(algorithm=_PHASH, hash_size=257, entries=())


def test_manifest_rejects_list_instead_of_tuple() -> None:
    entry = PerceptualHashManifestEntry(path=PurePosixPath("a.png"), hash=_hash("0"))
    with pytest.raises(TypeError, match="tuple"):
        PerceptualHashManifest(algorithm=_PHASH, hash_size=2, entries=[entry])  # type: ignore[arg-type]


def test_manifest_rejects_wrong_entry_type() -> None:
    with pytest.raises(TypeError, match="entries"):
        PerceptualHashManifest(algorithm=_PHASH, hash_size=2, entries=("not an entry",))  # type: ignore[arg-type]


def test_manifest_rejects_mixed_algorithm() -> None:
    entry_a = PerceptualHashManifestEntry(
        path=PurePosixPath("a.png"), hash=_hash("0", algorithm=_PHASH)
    )
    entry_b = PerceptualHashManifestEntry(
        path=PurePosixPath("b.png"), hash=_hash("1", algorithm=_AVERAGE_HASH)
    )
    with pytest.raises(ValueError, match="phash"):
        PerceptualHashManifest(algorithm=_PHASH, hash_size=2, entries=(entry_a, entry_b))


def test_manifest_rejects_mixed_hash_size() -> None:
    entry_a = PerceptualHashManifestEntry(path=PurePosixPath("a.png"), hash=_hash("0", hash_size=2))
    entry_b = PerceptualHashManifestEntry(
        path=PurePosixPath("b.png"), hash=_hash("001", hash_size=3)
    )
    with pytest.raises(ValueError, match="hash_size"):
        PerceptualHashManifest(algorithm=_PHASH, hash_size=2, entries=(entry_a, entry_b))


def test_manifest_rejects_duplicate_path() -> None:
    entry_a = PerceptualHashManifestEntry(path=PurePosixPath("a.png"), hash=_hash("0"))
    entry_b = PerceptualHashManifestEntry(path=PurePosixPath("a.png"), hash=_hash("1"))
    with pytest.raises(ValueError, match="unique"):
        PerceptualHashManifest(algorithm=_PHASH, hash_size=2, entries=(entry_a, entry_b))


def test_manifest_rejects_non_canonical_order() -> None:
    entry_a = PerceptualHashManifestEntry(path=PurePosixPath("z.png"), hash=_hash("0"))
    entry_b = PerceptualHashManifestEntry(path=PurePosixPath("a.png"), hash=_hash("1"))
    with pytest.raises(ValueError, match="sorted"):
        PerceptualHashManifest(algorithm=_PHASH, hash_size=2, entries=(entry_a, entry_b))


def test_manifest_does_not_sort_for_you() -> None:
    # Direct construction never sorts -- only from_hashes/from_json do.
    entry_a = PerceptualHashManifestEntry(path=PurePosixPath("a.png"), hash=_hash("0"))
    entry_b = PerceptualHashManifestEntry(path=PurePosixPath("z.png"), hash=_hash("1"))
    manifest = PerceptualHashManifest(algorithm=_PHASH, hash_size=2, entries=(entry_a, entry_b))
    assert manifest.entries == (entry_a, entry_b)


# =====================================================================================
# from_hashes
# =====================================================================================


def test_from_hashes_empty_mapping_with_explicit_hash_space() -> None:
    manifest = PerceptualHashManifest.from_hashes({}, algorithm=_PHASH, hash_size=8)
    assert manifest.algorithm == _PHASH
    assert manifest.hash_size == 8
    assert manifest.entries == ()


def test_from_hashes_singleton() -> None:
    manifest = PerceptualHashManifest.from_hashes(
        {"a.png": _hash("0")}, algorithm=_PHASH, hash_size=2
    )
    assert len(manifest.entries) == 1
    assert manifest.entries[0].path == PurePosixPath("a.png")


def test_from_hashes_many_entries() -> None:
    manifest = PerceptualHashManifest.from_hashes(
        {"z.png": _hash("0"), "a.png": _hash("1"), "m.png": _hash("2")},
        algorithm=_PHASH,
        hash_size=2,
    )
    assert [entry.path.as_posix() for entry in manifest.entries] == ["a.png", "m.png", "z.png"]


def test_from_hashes_insertion_order_does_not_affect_result() -> None:
    forward = PerceptualHashManifest.from_hashes(
        {"a.png": _hash("0"), "b.png": _hash("1")}, algorithm=_PHASH, hash_size=2
    )
    backward = PerceptualHashManifest.from_hashes(
        {"b.png": _hash("1"), "a.png": _hash("0")}, algorithm=_PHASH, hash_size=2
    )
    assert forward == backward


def test_from_hashes_accepts_path_keys() -> None:
    manifest = PerceptualHashManifest.from_hashes(
        {Path("a.png"): _hash("0")}, algorithm=_PHASH, hash_size=2
    )
    assert manifest.entries[0].path == PurePosixPath("a.png")


def test_from_hashes_accepts_string_keys() -> None:
    manifest = PerceptualHashManifest.from_hashes(
        {"a.png": _hash("0")}, algorithm=_PHASH, hash_size=2
    )
    assert manifest.entries[0].path == PurePosixPath("a.png")


def test_from_hashes_accepts_custom_pathlike_returning_str() -> None:
    class _StrPath:
        def __fspath__(self) -> str:
            return "a.png"

    manifest = PerceptualHashManifest.from_hashes(
        {_StrPath(): _hash("0")}, algorithm=_PHASH, hash_size=2
    )
    assert manifest.entries[0].path == PurePosixPath("a.png")


def test_from_hashes_accepts_dict_str_keys_variable() -> None:
    hashes: dict[str, PerceptualHash] = {"a.png": _hash("0"), "b.png": _hash("1")}
    manifest = PerceptualHashManifest.from_hashes(hashes, algorithm=_PHASH, hash_size=2)
    assert len(manifest.entries) == 2


def test_from_hashes_accepts_dict_path_keys_variable() -> None:
    path_hashes: dict[Path, PerceptualHash] = {
        Path("a.png"): _hash("0"),
        Path("b.png"): _hash("1"),
    }
    manifest = PerceptualHashManifest.from_hashes(path_hashes, algorithm=_PHASH, hash_size=2)
    assert len(manifest.entries) == 2


def test_from_hashes_accepts_dict_pure_posix_path_keys_variable() -> None:
    pure_hashes: dict[PurePosixPath, PerceptualHash] = {
        PurePosixPath("a.png"): _hash("0"),
        PurePosixPath("b.png"): _hash("1"),
    }
    manifest = PerceptualHashManifest.from_hashes(pure_hashes, algorithm=_PHASH, hash_size=2)
    assert len(manifest.entries) == 2


def test_from_hashes_accepts_dict_custom_pathlike_keys_variable() -> None:
    class _StrPath:
        def __init__(self, name: str) -> None:
            self._name = name

        def __fspath__(self) -> str:
            return self._name

    custom_hashes: dict[_StrPath, PerceptualHash] = {
        _StrPath("a.png"): _hash("0"),
        _StrPath("b.png"): _hash("1"),
    }
    manifest = PerceptualHashManifest.from_hashes(custom_hashes, algorithm=_PHASH, hash_size=2)
    assert len(manifest.entries) == 2


def test_from_hashes_rejects_bytes_key() -> None:
    with pytest.raises(TypeError):
        PerceptualHashManifest.from_hashes(
            {b"a.png": _hash("0")},  # type: ignore[dict-item]
            algorithm=_PHASH,
            hash_size=2,
        )


def test_from_hashes_rejects_pathlike_returning_bytes() -> None:
    class _BytesPath:
        def __fspath__(self) -> bytes:
            return b"a.png"

    with pytest.raises(TypeError):
        PerceptualHashManifest.from_hashes(
            {_BytesPath(): _hash("0")},  # type: ignore[dict-item]
            algorithm=_PHASH,
            hash_size=2,
        )


def test_from_hashes_rejects_absolute_key() -> None:
    with pytest.raises(ValueError, match="relative"):
        PerceptualHashManifest.from_hashes(
            {"/images/a.png": _hash("0")}, algorithm=_PHASH, hash_size=2
        )


def test_from_hashes_rejects_dot_dot_key() -> None:
    with pytest.raises(ValueError):
        PerceptualHashManifest.from_hashes({"../a.png": _hash("0")}, algorithm=_PHASH, hash_size=2)


def test_from_hashes_rejects_lone_dot_key() -> None:
    with pytest.raises(ValueError):
        PerceptualHashManifest.from_hashes({".": _hash("0")}, algorithm=_PHASH, hash_size=2)


def test_from_hashes_collision_after_normalization_raises() -> None:
    with pytest.raises(ValueError, match="a.png"):
        PerceptualHashManifest.from_hashes(
            {"./a.png": _hash("0"), "a.png": _hash("1")}, algorithm=_PHASH, hash_size=2
        )


def test_from_hashes_rejects_wrong_value_type() -> None:
    with pytest.raises(TypeError, match="PerceptualHash"):
        PerceptualHashManifest.from_hashes(
            {"a.png": "0"},  # type: ignore[dict-item]
            algorithm=_PHASH,
            hash_size=2,
        )


def test_from_hashes_rejects_mixed_hash_spaces_in_input() -> None:
    with pytest.raises(ValueError):
        PerceptualHashManifest.from_hashes(
            {"a.png": _hash("0", algorithm=_PHASH), "b.png": _hash("1", algorithm=_AVERAGE_HASH)},
            algorithm=_PHASH,
            hash_size=2,
        )


def test_from_hashes_rejects_mismatch_with_explicit_algorithm() -> None:
    with pytest.raises(ValueError, match="average_hash"):
        PerceptualHashManifest.from_hashes(
            {"a.png": _hash("0", algorithm=_PHASH)}, algorithm=_AVERAGE_HASH, hash_size=2
        )


def test_from_hashes_rejects_mismatch_with_explicit_hash_size() -> None:
    with pytest.raises(ValueError, match="hash_size"):
        PerceptualHashManifest.from_hashes(
            {"a.png": _hash("0", hash_size=2)}, algorithm=_PHASH, hash_size=3
        )


def test_from_hashes_does_not_modify_input_mapping() -> None:
    hash_a = _hash("0")
    hashes = {"a.png": hash_a}
    snapshot = dict(hashes)

    PerceptualHashManifest.from_hashes(hashes, algorithm=_PHASH, hash_size=2)

    assert hashes == snapshot
    assert hashes["a.png"] is hash_a


def test_from_hashes_does_not_touch_filesystem(monkeypatch: pytest.MonkeyPatch) -> None:
    boom = _Boom()
    monkeypatch.setattr(Path, "exists", boom)
    monkeypatch.setattr(Path, "stat", boom)
    monkeypatch.setattr(Path, "is_file", boom)
    monkeypatch.setattr("builtins.open", boom)

    manifest = PerceptualHashManifest.from_hashes(
        {"does/not/exist/a.png": _hash("0")}, algorithm=_PHASH, hash_size=2
    )
    assert len(manifest.entries) == 1


# =====================================================================================
# to_hashes
# =====================================================================================


def test_to_hashes_returns_expected_mapping() -> None:
    manifest = PerceptualHashManifest.from_hashes(
        {"a.png": _hash("0"), "b.png": _hash("1")}, algorithm=_PHASH, hash_size=2
    )
    hashes = manifest.to_hashes()
    assert hashes == {PurePosixPath("a.png"): _hash("0"), PurePosixPath("b.png"): _hash("1")}


def test_to_hashes_insertion_order_matches_canonical_order() -> None:
    manifest = PerceptualHashManifest.from_hashes(
        {"z.png": _hash("0"), "a.png": _hash("1")}, algorithm=_PHASH, hash_size=2
    )
    assert list(manifest.to_hashes().keys()) == [PurePosixPath("a.png"), PurePosixPath("z.png")]


def test_to_hashes_returns_a_new_dict_each_time() -> None:
    manifest = PerceptualHashManifest.from_hashes(
        {"a.png": _hash("0")}, algorithm=_PHASH, hash_size=2
    )
    first = manifest.to_hashes()
    first[PurePosixPath("mutated.png")] = _hash("1")
    second = manifest.to_hashes()
    assert PurePosixPath("mutated.png") not in second


def test_to_hashes_empty_manifest() -> None:
    manifest = PerceptualHashManifest.from_hashes({}, algorithm=_PHASH, hash_size=8)
    assert manifest.to_hashes() == {}


# =====================================================================================
# to_json -- deterministic serialization
# =====================================================================================


def test_to_json_empty_manifest_exact_text() -> None:
    manifest = PerceptualHashManifest.from_hashes({}, algorithm=_PHASH, hash_size=8)
    expected = (
        "{\n"
        '  "schema_version": 1,\n'
        '  "algorithm": "phash",\n'
        '  "hash_size": 8,\n'
        '  "entries": []\n'
        "}\n"
    )
    assert manifest.to_json() == expected


def test_to_json_known_singleton_exact_text() -> None:
    known_hash = PerceptualHash.from_hex("0123456789abcdef", algorithm=_PHASH, hash_size=8)
    manifest = PerceptualHashManifest.from_hashes(
        {"images/a.png": known_hash},
        algorithm=_PHASH,
        hash_size=8,
    )
    expected = (
        "{\n"
        '  "schema_version": 1,\n'
        '  "algorithm": "phash",\n'
        '  "hash_size": 8,\n'
        '  "entries": [\n'
        "    {\n"
        '      "path": "images/a.png",\n'
        '      "hash": "0123456789abcdef"\n'
        "    }\n"
        "  ]\n"
        "}\n"
    )
    assert manifest.to_json() == expected


def test_to_json_ends_with_exactly_one_trailing_newline() -> None:
    manifest = PerceptualHashManifest.from_hashes({}, algorithm=_PHASH, hash_size=8)
    text = manifest.to_json()
    assert text.endswith("\n")
    assert not text.endswith("\n\n")


def test_to_json_is_ensure_ascii_false() -> None:
    manifest = PerceptualHashManifest.from_hashes(
        {"dane/żółw.png": _hash("0")}, algorithm=_PHASH, hash_size=2
    )
    text = manifest.to_json()
    assert "żółw" in text
    assert "\\u" not in text


def test_to_json_is_a_str() -> None:
    manifest = PerceptualHashManifest.from_hashes({}, algorithm=_PHASH, hash_size=8)
    assert isinstance(manifest.to_json(), str)


def test_to_json_is_deterministic_across_calls() -> None:
    manifest = PerceptualHashManifest.from_hashes(
        {"a.png": _hash("0"), "b.png": _hash("1")}, algorithm=_PHASH, hash_size=2
    )
    assert manifest.to_json() == manifest.to_json()


def test_to_json_independent_of_input_mapping_order() -> None:
    forward = PerceptualHashManifest.from_hashes(
        {"a.png": _hash("0"), "b.png": _hash("1")}, algorithm=_PHASH, hash_size=2
    )
    backward = PerceptualHashManifest.from_hashes(
        {"b.png": _hash("1"), "a.png": _hash("0")}, algorithm=_PHASH, hash_size=2
    )
    assert forward.to_json() == backward.to_json()


def test_to_json_does_not_touch_filesystem(monkeypatch: pytest.MonkeyPatch) -> None:
    boom = _Boom()
    monkeypatch.setattr(Path, "exists", boom)
    monkeypatch.setattr(Path, "stat", boom)
    monkeypatch.setattr("builtins.open", boom)

    manifest = PerceptualHashManifest.from_hashes(
        {"a.png": _hash("0")}, algorithm=_PHASH, hash_size=2
    )
    assert isinstance(manifest.to_json(), str)


# =====================================================================================
# from_json -- round trips
# =====================================================================================


def test_from_json_empty_manifest_round_trip() -> None:
    manifest = PerceptualHashManifest.from_hashes({}, algorithm=_PHASH, hash_size=8)
    restored = PerceptualHashManifest.from_json(manifest.to_json())
    assert restored == manifest


def test_from_json_singleton_round_trip() -> None:
    manifest = PerceptualHashManifest.from_hashes(
        {"a.png": _hash("0")}, algorithm=_PHASH, hash_size=2
    )
    restored = PerceptualHashManifest.from_json(manifest.to_json())
    assert restored == manifest


def test_from_json_many_entry_round_trip() -> None:
    manifest = PerceptualHashManifest.from_hashes(
        {
            "z.png": _hash("0", algorithm=_AVERAGE_HASH),
            "a.png": _hash("1", algorithm=_AVERAGE_HASH),
            "m.png": _hash("2", algorithm=_AVERAGE_HASH),
            "b.png": _hash("3", algorithm=_AVERAGE_HASH),
        },
        algorithm=_AVERAGE_HASH,
        hash_size=2,
    )
    restored = PerceptualHashManifest.from_json(manifest.to_json())
    assert restored == manifest


def test_from_json_exact_algorithm_hash_size_hex_preserved() -> None:
    manifest = PerceptualHashManifest.from_hashes(
        {"a.png": PerceptualHash.from_hex("0123456789abcdef", algorithm=_PHASH, hash_size=8)},
        algorithm=_PHASH,
        hash_size=8,
    )
    restored = PerceptualHashManifest.from_json(manifest.to_json())
    assert restored.algorithm == _PHASH
    assert restored.hash_size == 8
    assert str(restored.entries[0].hash) == "0123456789abcdef"


def test_from_json_rejects_text_that_is_not_str() -> None:
    with pytest.raises(TypeError):
        PerceptualHashManifest.from_json(b"{}")  # type: ignore[arg-type]


def test_from_json_rejects_pathlike_text() -> None:
    class _StrPath:
        def __fspath__(self) -> str:
            return "manifest.json"

    with pytest.raises(TypeError):
        PerceptualHashManifest.from_json(_StrPath())  # type: ignore[arg-type]


def test_from_json_does_not_touch_filesystem(monkeypatch: pytest.MonkeyPatch) -> None:
    boom = _Boom()
    monkeypatch.setattr(Path, "exists", boom)
    monkeypatch.setattr(Path, "stat", boom)
    monkeypatch.setattr("builtins.open", boom)

    manifest = PerceptualHashManifest.from_hashes(
        {"a.png": _hash("0")}, algorithm=_PHASH, hash_size=2
    )
    restored = PerceptualHashManifest.from_json(manifest.to_json())
    assert restored == manifest


# =====================================================================================
# from_json -- malformed JSON and structural errors
# =====================================================================================

_VALID_HEADER = '"schema_version": 1, "algorithm": "phash", "hash_size": 2'


def test_from_json_rejects_malformed_json() -> None:
    with pytest.raises(ValueError):
        PerceptualHashManifest.from_json("{not valid json")


def test_from_json_rejects_top_level_array() -> None:
    with pytest.raises(ValueError, match="object"):
        PerceptualHashManifest.from_json("[]")


def test_from_json_rejects_top_level_string() -> None:
    with pytest.raises(ValueError, match="object"):
        PerceptualHashManifest.from_json('"hello"')


def test_from_json_rejects_top_level_null() -> None:
    with pytest.raises(ValueError, match="object"):
        PerceptualHashManifest.from_json("null")


def test_from_json_rejects_duplicate_top_level_field() -> None:
    text = (
        '{"schema_version": 1, "schema_version": 1, "algorithm": "phash", '
        '"hash_size": 2, "entries": []}'
    )
    with pytest.raises(ValueError, match="duplicate"):
        PerceptualHashManifest.from_json(text)


def test_from_json_rejects_duplicate_entry_field() -> None:
    text = "{" + _VALID_HEADER + ', "entries": [{"path": "a.png", "path": "b.png", "hash": "0"}]}'
    with pytest.raises(ValueError, match="duplicate"):
        PerceptualHashManifest.from_json(text)


def test_from_json_rejects_missing_top_level_field() -> None:
    text = '{"schema_version": 1, "algorithm": "phash", "hash_size": 2}'
    with pytest.raises(ValueError, match="missing"):
        PerceptualHashManifest.from_json(text)


def test_from_json_rejects_extra_top_level_field() -> None:
    text = "{" + _VALID_HEADER + ', "entries": [], "dataset_root": "/data"}'
    with pytest.raises(ValueError, match="unexpected"):
        PerceptualHashManifest.from_json(text)


def test_from_json_rejects_wrong_schema_version_type_bool() -> None:
    text = '{"schema_version": true, "algorithm": "phash", "hash_size": 2, "entries": []}'
    with pytest.raises(ValueError, match="schema_version"):
        PerceptualHashManifest.from_json(text)


def test_from_json_rejects_wrong_schema_version_type_float() -> None:
    text = '{"schema_version": 1.0, "algorithm": "phash", "hash_size": 2, "entries": []}'
    with pytest.raises(ValueError, match="schema_version"):
        PerceptualHashManifest.from_json(text)


def test_from_json_rejects_wrong_schema_version_type_string() -> None:
    text = '{"schema_version": "1", "algorithm": "phash", "hash_size": 2, "entries": []}'
    with pytest.raises(ValueError, match="schema_version"):
        PerceptualHashManifest.from_json(text)


def test_from_json_rejects_unknown_schema_version() -> None:
    text = '{"schema_version": 2, "algorithm": "phash", "hash_size": 2, "entries": []}'
    with pytest.raises(ValueError, match="schema_version"):
        PerceptualHashManifest.from_json(text)


def test_from_json_rejects_unknown_algorithm() -> None:
    text = '{"schema_version": 1, "algorithm": "sha256", "hash_size": 2, "entries": []}'
    with pytest.raises(ValueError, match="algorithm"):
        PerceptualHashManifest.from_json(text)


def test_from_json_rejects_wrong_algorithm_type() -> None:
    text = '{"schema_version": 1, "algorithm": 1, "hash_size": 2, "entries": []}'
    with pytest.raises(ValueError, match="algorithm"):
        PerceptualHashManifest.from_json(text)


def test_from_json_rejects_wrong_hash_size_type() -> None:
    text = '{"schema_version": 1, "algorithm": "phash", "hash_size": "2", "entries": []}'
    with pytest.raises(ValueError, match="hash_size"):
        PerceptualHashManifest.from_json(text)


def test_from_json_rejects_hash_size_out_of_range() -> None:
    text = '{"schema_version": 1, "algorithm": "phash", "hash_size": 1, "entries": []}'
    with pytest.raises(ValueError, match="hash_size"):
        PerceptualHashManifest.from_json(text)


def test_from_json_rejects_entries_not_a_list() -> None:
    text = "{" + _VALID_HEADER + ', "entries": {}}'
    with pytest.raises(ValueError, match="list"):
        PerceptualHashManifest.from_json(text)


def test_from_json_rejects_entry_not_an_object() -> None:
    text = "{" + _VALID_HEADER + ', "entries": ["a.png"]}'
    with pytest.raises(ValueError, match="object"):
        PerceptualHashManifest.from_json(text)


def test_from_json_rejects_entry_missing_field() -> None:
    text = "{" + _VALID_HEADER + ', "entries": [{"path": "a.png"}]}'
    with pytest.raises(ValueError, match="missing"):
        PerceptualHashManifest.from_json(text)


def test_from_json_rejects_entry_extra_field() -> None:
    text = "{" + _VALID_HEADER + ', "entries": [{"path": "a.png", "hash": "0", "mtime_ns": 1}]}'
    with pytest.raises(ValueError, match="unexpected"):
        PerceptualHashManifest.from_json(text)


def test_from_json_rejects_wrong_path_type() -> None:
    text = "{" + _VALID_HEADER + ', "entries": [{"path": 1, "hash": "0"}]}'
    with pytest.raises(ValueError, match="path"):
        PerceptualHashManifest.from_json(text)


def test_from_json_rejects_deeply_nested_path_value() -> None:
    text = "{" + _VALID_HEADER + ', "entries": [{"path": {"a": {"b": ["c"]}}, "hash": "0"}]}'
    with pytest.raises(ValueError, match="path"):
        PerceptualHashManifest.from_json(text)


def test_from_json_rejects_wrong_hash_type() -> None:
    text = "{" + _VALID_HEADER + ', "entries": [{"path": "a.png", "hash": 0}]}'
    with pytest.raises(ValueError, match="hash"):
        PerceptualHashManifest.from_json(text)


def test_from_json_rejects_malformed_hex() -> None:
    text = "{" + _VALID_HEADER + ', "entries": [{"path": "a.png", "hash": "zz"}]}'
    with pytest.raises(ValueError, match="hash"):
        PerceptualHashManifest.from_json(text)


_VALID_HEADER_HASH_SIZE_8 = '"schema_version": 1, "algorithm": "phash", "hash_size": 8'


def test_from_json_rejects_uppercase_hex() -> None:
    text = (
        "{"
        + _VALID_HEADER_HASH_SIZE_8
        + ', "entries": [{"path": "a.png", "hash": "0123456789ABCDEF"}]}'
    )
    with pytest.raises(ValueError, match="canonical"):
        PerceptualHashManifest.from_json(text)


def test_from_json_rejects_hex_too_short() -> None:
    text = "{" + _VALID_HEADER_HASH_SIZE_8 + ', "entries": [{"path": "a.png", "hash": "0123"}]}'
    with pytest.raises(ValueError, match="hash"):
        PerceptualHashManifest.from_json(text)


def test_from_json_rejects_hex_too_long() -> None:
    text = (
        "{"
        + _VALID_HEADER_HASH_SIZE_8
        + ', "entries": [{"path": "a.png", "hash": "0123456789abcdef00"}]}'
    )
    with pytest.raises(ValueError, match="hash"):
        PerceptualHashManifest.from_json(text)


def test_from_json_rejects_leading_dot_slash_path() -> None:
    # Unlike from_hashes (which normalizes "./a.png" away via a local Path), from_json
    # validates the raw JSON text directly and rejects it verbatim.
    text = "{" + _VALID_HEADER + ', "entries": [{"path": "./a.png", "hash": "0"}]}'
    with pytest.raises(ValueError, match="segment"):
        PerceptualHashManifest.from_json(text)


def test_from_json_rejects_absolute_path() -> None:
    text = "{" + _VALID_HEADER + ', "entries": [{"path": "/a.png", "hash": "0"}]}'
    with pytest.raises(ValueError, match="relative"):
        PerceptualHashManifest.from_json(text)


def test_from_json_rejects_backslash_path() -> None:
    text = "{" + _VALID_HEADER + r', "entries": [{"path": "a\\b.png", "hash": "0"}]}'
    with pytest.raises(ValueError, match="backslash"):
        PerceptualHashManifest.from_json(text)


def test_from_json_rejects_windows_drive_path() -> None:
    text = "{" + _VALID_HEADER + ', "entries": [{"path": "C:/a.png", "hash": "0"}]}'
    with pytest.raises(ValueError, match="drive"):
        PerceptualHashManifest.from_json(text)


def test_from_json_rejects_duplicate_canonical_entry_path() -> None:
    text = (
        "{"
        + _VALID_HEADER
        + ', "entries": [{"path": "a.png", "hash": "0"}, {"path": "a.png", "hash": "1"}]}'
    )
    with pytest.raises(ValueError, match="unique"):
        PerceptualHashManifest.from_json(text)


def test_from_json_error_does_not_contain_whole_document() -> None:
    # A large, otherwise-valid document with one invalid entry near the end: the error must
    # name that entry specifically, not dump the (much larger) full document/entry list.
    good_entries = ", ".join(f'{{"path": "img_{i}.png", "hash": "0"}}' for i in range(500))
    text = (
        "{" + _VALID_HEADER + f', "entries": [{good_entries}, {{"path": "bad.png", "hash": 0}}]}}'
    )
    with pytest.raises(ValueError) as exc_info:
        PerceptualHashManifest.from_json(text)
    message = str(exc_info.value)
    assert len(message) < len(text) / 10
    assert "img_0.png" not in message
    assert "500" in message or "bad.png" in message


# =====================================================================================
# Integration -- similarity search
# =====================================================================================


def test_manifest_round_trip_preserves_find_similar_image_pairs_result() -> None:
    hashes = {
        "a.png": _hash("0"),
        "b.png": _hash("1"),
        "c.png": _hash("3"),
    }
    manifest = PerceptualHashManifest.from_hashes(hashes, algorithm=_PHASH, hash_size=2)
    restored = PerceptualHashManifest.from_json(manifest.to_json())

    assert restored == manifest

    pairs_before = im.find_similar_image_pairs(hashes, max_distance=1)
    # `to_hashes()` returns `dict[PurePosixPath, PerceptualHash]`, directly accepted here since
    # `find_similar_image_pairs`'s `hashes` parameter is generic over any `str | os.PathLike[str]`
    # key type (see `_PathIdentifierT` in `similarity.py`), not fixed to one concrete union.
    pairs_after = im.find_similar_image_pairs(restored.to_hashes(), max_distance=1)

    assert pairs_after == pairs_before
    assert len(pairs_after) == 2


def test_manifest_to_hashes_is_directly_usable_by_find_similar_image_pairs() -> None:
    manifest = PerceptualHashManifest.from_hashes(
        {"a.png": _hash("0"), "b.png": _hash("1")}, algorithm=_PHASH, hash_size=2
    )
    pairs = im.find_similar_image_pairs(manifest.to_hashes(), max_distance=4)
    assert len(pairs) == 1
    assert pairs[0].distance == 1


# =====================================================================================
# Exports
# =====================================================================================


def test_top_level_exports_are_the_same_objects() -> None:
    assert im.PerceptualHashManifest is PerceptualHashManifest
    assert im.PerceptualHashManifestEntry is PerceptualHashManifestEntry


def test_module_all_contains_exactly_the_public_symbols() -> None:
    assert manifest_module.__all__ == ["PerceptualHashManifest", "PerceptualHashManifestEntry"]


def test_top_level_all_contains_new_symbols_without_duplicates() -> None:
    assert im.__all__.count("PerceptualHashManifest") == 1
    assert im.__all__.count("PerceptualHashManifestEntry") == 1
    assert len(im.__all__) == len(set(im.__all__))


def test_top_level_all_places_new_symbols_alphabetically() -> None:
    index = im.__all__.index("PerceptualHashManifest")
    assert im.__all__[index - 1] == "PerceptualHashAlgorithm"
    assert im.__all__[index + 1] == "PerceptualHashManifestEntry"
    assert im.__all__[index + 2] == "PerspectiveParameters"


def test_manifest_module_does_not_import_cv2_or_numpy() -> None:
    """Static architecture-boundary check: `manifest.py` never references cv2 or NumPy --
    the manifest model and its JSON string transport have no dependency on either. This no
    longer claims the whole module is free of file I/O: `save`/`load` are its one explicit,
    intentional filesystem-touching surface (see tests/test_manifest_io.py).
    """
    source = Path(manifest_module.__file__).read_text()
    assert "cv2" not in source
    assert "numpy" not in source
