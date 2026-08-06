import dataclasses
from pathlib import Path, PurePosixPath

import pytest

import improcv as im
import improcv.manifest as manifest_module
from improcv.hashing import PerceptualHash, PerceptualHashAlgorithm
from improcv.manifest import (
    PerceptualHashManifest,
    PerceptualHashManifestChange,
    PerceptualHashManifestDiff,
    compare_perceptual_hash_manifests,
)

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


def _manifest(
    mapping: dict[str, PerceptualHash],
    *,
    algorithm: PerceptualHashAlgorithm = _PHASH,
    hash_size: int = 2,
) -> PerceptualHashManifest:
    return PerceptualHashManifest.from_hashes(mapping, algorithm=algorithm, hash_size=hash_size)


class _Boom:
    def __call__(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("compare_perceptual_hash_manifests must not perform this operation")


# =====================================================================================
# Basic classification
# =====================================================================================


def test_empty_vs_empty() -> None:
    before = _manifest({})
    after = _manifest({})
    result = compare_perceptual_hash_manifests(before, after)
    assert result == PerceptualHashManifestDiff(added=(), removed=(), changed=(), unchanged=())


def test_empty_vs_populated() -> None:
    before = _manifest({})
    after = _manifest({"a.png": _hash("0"), "b.png": _hash("1")})
    result = compare_perceptual_hash_manifests(before, after)
    assert result.added == after.entries
    assert result.removed == ()
    assert result.changed == ()
    assert result.unchanged == ()


def test_populated_vs_empty() -> None:
    before = _manifest({"a.png": _hash("0"), "b.png": _hash("1")})
    after = _manifest({})
    result = compare_perceptual_hash_manifests(before, after)
    assert result.removed == before.entries
    assert result.added == ()
    assert result.changed == ()
    assert result.unchanged == ()


def test_independently_built_identical_manifests() -> None:
    before = _manifest({"a.png": _hash("0"), "b.png": _hash("1")})
    after = _manifest({"b.png": _hash("1"), "a.png": _hash("0")})
    result = compare_perceptual_hash_manifests(before, after)
    assert result.added == ()
    assert result.removed == ()
    assert result.changed == ()
    assert result.unchanged == (PurePosixPath("a.png"), PurePosixPath("b.png"))


def test_only_added() -> None:
    before = _manifest({"a.png": _hash("0"), "b.png": _hash("1")})
    after = _manifest({"a.png": _hash("0"), "b.png": _hash("1"), "c.png": _hash("2")})
    result = compare_perceptual_hash_manifests(before, after)
    assert tuple(entry.path for entry in result.added) == (PurePosixPath("c.png"),)
    assert result.removed == ()
    assert result.changed == ()
    assert result.unchanged == (PurePosixPath("a.png"), PurePosixPath("b.png"))


def test_only_removed() -> None:
    before = _manifest({"a.png": _hash("0"), "b.png": _hash("1"), "c.png": _hash("2")})
    after = _manifest({"a.png": _hash("0"), "b.png": _hash("1")})
    result = compare_perceptual_hash_manifests(before, after)
    assert tuple(entry.path for entry in result.removed) == (PurePosixPath("c.png"),)
    assert result.added == ()
    assert result.changed == ()
    assert result.unchanged == (PurePosixPath("a.png"), PurePosixPath("b.png"))


def test_only_changed() -> None:
    before = _manifest({"a.png": _hash("0"), "b.png": _hash("1")})
    after = _manifest({"a.png": _hash("0"), "b.png": _hash("2")})
    result = compare_perceptual_hash_manifests(before, after)
    assert result.changed == (
        PerceptualHashManifestChange(
            path=PurePosixPath("b.png"), before=_hash("1"), after=_hash("2")
        ),
    )
    assert result.added == ()
    assert result.removed == ()
    assert result.unchanged == (PurePosixPath("a.png"),)


def test_mixture_of_all_four_categories() -> None:
    before = _manifest(
        {
            "a.png": _hash("0"),
            "b.png": _hash("1"),
            "c.png": _hash("2"),
            "d.png": _hash("3"),
        }
    )
    after = _manifest(
        {
            "b.png": _hash("9"),
            "c.png": _hash("2"),
            "e.png": _hash("4"),
        }
    )
    result = compare_perceptual_hash_manifests(before, after)
    assert tuple(entry.path for entry in result.removed) == (
        PurePosixPath("a.png"),
        PurePosixPath("d.png"),
    )
    assert result.changed == (
        PerceptualHashManifestChange(
            path=PurePosixPath("b.png"), before=_hash("1"), after=_hash("9")
        ),
    )
    assert result.unchanged == (PurePosixPath("c.png"),)
    assert tuple(entry.path for entry in result.added) == (PurePosixPath("e.png"),)


def test_left_tail_drain_when_before_is_longer() -> None:
    before = _manifest({"a.png": _hash("0"), "b.png": _hash("1"), "c.png": _hash("2")})
    after = _manifest({"a.png": _hash("0")})
    result = compare_perceptual_hash_manifests(before, after)
    assert tuple(entry.path for entry in result.removed) == (
        PurePosixPath("b.png"),
        PurePosixPath("c.png"),
    )


def test_right_tail_drain_when_after_is_longer() -> None:
    before = _manifest({"a.png": _hash("0")})
    after = _manifest({"a.png": _hash("0"), "b.png": _hash("1"), "c.png": _hash("2")})
    result = compare_perceptual_hash_manifests(before, after)
    assert tuple(entry.path for entry in result.added) == (
        PurePosixPath("b.png"),
        PurePosixPath("c.png"),
    )


# =====================================================================================
# Ordering
# =====================================================================================


def test_result_independent_of_insertion_order() -> None:
    before_forward = _manifest({"a.png": _hash("0"), "b.png": _hash("1"), "c.png": _hash("2")})
    before_backward = _manifest({"c.png": _hash("2"), "b.png": _hash("1"), "a.png": _hash("0")})
    after_forward = _manifest({"b.png": _hash("9"), "d.png": _hash("3")})
    after_backward = _manifest({"d.png": _hash("3"), "b.png": _hash("9")})

    result_forward = compare_perceptual_hash_manifests(before_forward, after_forward)
    result_backward = compare_perceptual_hash_manifests(before_backward, after_backward)
    assert result_forward == result_backward


def test_unicode_identifiers() -> None:
    before = _manifest({"dane/żółw.png": _hash("0")})
    after = _manifest({"dane/żółw.png": _hash("1")})
    result = compare_perceptual_hash_manifests(before, after)
    assert result.changed == (
        PerceptualHashManifestChange(
            path=PurePosixPath("dane/żółw.png"), before=_hash("0"), after=_hash("1")
        ),
    )


def test_nested_paths() -> None:
    before = _manifest(
        {
            "a.png": _hash("0"),
            "a/b.png": _hash("1"),
            "a/b/c.png": _hash("2"),
        }
    )
    after = _manifest(
        {
            "a.png": _hash("0"),
            "a/b.png": _hash("9"),
            "a/b/c.png": _hash("2"),
        }
    )
    result = compare_perceptual_hash_manifests(before, after)
    assert result.changed == (
        PerceptualHashManifestChange(
            path=PurePosixPath("a/b.png"), before=_hash("1"), after=_hash("9")
        ),
    )
    assert result.unchanged == (PurePosixPath("a.png"), PurePosixPath("a/b/c.png"))


def test_hyphen_versus_slash_ordering() -> None:
    # "a-b" < "a/b" as raw strings ('-' is code point 45, '/' is 47), the opposite of
    # PurePosixPath's own parts-tuple ordering -- see docs/design/0.4.0a1-manifest-comparison.md
    # section 9. This exercises that the merge-join compares path.as_posix() strings, matching
    # PerceptualHashManifest's own sort-order invariant, not PurePosixPath.__lt__.
    before = _manifest({"a-b.png": _hash("0"), "a/b.png": _hash("1")})
    after = _manifest({"a-b.png": _hash("9"), "a/b.png": _hash("8")})
    result = compare_perceptual_hash_manifests(before, after)
    assert tuple(change.path for change in result.changed) == (
        PurePosixPath("a-b.png"),
        PurePosixPath("a/b.png"),
    )


def test_all_result_fields_sorted_ascending_by_posix_path() -> None:
    before = _manifest(
        {
            "z.png": _hash("0"),
            "m.png": _hash("1"),
            "a.png": _hash("2"),
            "u.png": _hash("3"),
        }
    )
    after = _manifest(
        {
            "z.png": _hash("9"),
            "m.png": _hash("1"),
            "b.png": _hash("4"),
            "n.png": _hash("5"),
        }
    )
    result = compare_perceptual_hash_manifests(before, after)

    added_paths = [entry.path.as_posix() for entry in result.added]
    removed_paths = [entry.path.as_posix() for entry in result.removed]
    changed_paths = [change.path.as_posix() for change in result.changed]
    unchanged_paths = [path.as_posix() for path in result.unchanged]

    assert added_paths == sorted(added_paths)
    assert removed_paths == sorted(removed_paths)
    assert changed_paths == sorted(changed_paths)
    assert unchanged_paths == sorted(unchanged_paths)
    assert added_paths == ["b.png", "n.png"]
    assert removed_paths == ["a.png", "u.png"]
    assert changed_paths == ["z.png"]
    assert unchanged_paths == ["m.png"]


# =====================================================================================
# Validation
# =====================================================================================


def test_mismatched_algorithm_raises_value_error() -> None:
    before = _manifest({"a.png": _hash("0", algorithm=_PHASH)}, algorithm=_PHASH)
    after = _manifest({"a.png": _hash("0", algorithm=_AVERAGE_HASH)}, algorithm=_AVERAGE_HASH)
    with pytest.raises(ValueError, match="algorithm and hash_size"):
        compare_perceptual_hash_manifests(before, after)


def test_mismatched_hash_size_raises_value_error() -> None:
    before = _manifest({"a.png": _hash("0", hash_size=2)}, hash_size=2)
    after = _manifest({"a.png": _hash("0000", hash_size=4)}, hash_size=4)
    with pytest.raises(ValueError, match="algorithm and hash_size"):
        compare_perceptual_hash_manifests(before, after)


def test_mismatch_message_names_both_algorithms_and_hash_sizes() -> None:
    before = _manifest({}, algorithm=_PHASH, hash_size=2)
    after = _manifest({}, algorithm=_AVERAGE_HASH, hash_size=4)
    with pytest.raises(ValueError) as excinfo:
        compare_perceptual_hash_manifests(before, after)
    message = str(excinfo.value)
    assert "phash(hash_size=2)" in message
    assert "average_hash(hash_size=4)" in message


def test_rejects_bad_type_for_before() -> None:
    after = _manifest({})
    with pytest.raises(TypeError, match="before"):
        compare_perceptual_hash_manifests({}, after)  # type: ignore[arg-type]


def test_rejects_bad_type_for_after() -> None:
    before = _manifest({})
    with pytest.raises(TypeError, match="after"):
        compare_perceptual_hash_manifests(before, "not a manifest")  # type: ignore[arg-type]


def test_before_validated_before_after() -> None:
    # Both arguments are wrong; the error must name "before", proving it is checked first.
    with pytest.raises(TypeError, match="before"):
        compare_perceptual_hash_manifests(None, None)  # type: ignore[arg-type]


def test_type_validation_precedes_hash_space_check() -> None:
    # A bad `after` type must surface as TypeError, never as the hash-space ValueError, even
    # though `before`/`after` here would also mismatch on hash space if both were manifests.
    before = _manifest({}, algorithm=_PHASH)
    with pytest.raises(TypeError, match="after"):
        compare_perceptual_hash_manifests(before, object())  # type: ignore[arg-type]


class _ManifestSubclass(PerceptualHashManifest):
    """A trivial subclass used only to confirm `isinstance` acceptance, not `type(...) is`."""


def test_subclass_of_manifest_is_accepted_for_before_and_after() -> None:
    before = _ManifestSubclass(algorithm=_PHASH, hash_size=2, entries=())
    after = _ManifestSubclass(algorithm=_PHASH, hash_size=2, entries=())
    result = compare_perceptual_hash_manifests(before, after)
    assert result == PerceptualHashManifestDiff(added=(), removed=(), changed=(), unchanged=())


def test_subclass_before_plain_manifest_after_is_accepted() -> None:
    before = _ManifestSubclass(algorithm=_PHASH, hash_size=2, entries=())
    after = _manifest({})
    result = compare_perceptual_hash_manifests(before, after)
    assert result == PerceptualHashManifestDiff(added=(), removed=(), changed=(), unchanged=())


# =====================================================================================
# Result types
# =====================================================================================


def test_change_is_frozen() -> None:
    change = PerceptualHashManifestChange(
        path=PurePosixPath("a.png"), before=_hash("0"), after=_hash("1")
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        change.before = _hash("2")  # type: ignore[misc]


def test_diff_is_frozen() -> None:
    diff = PerceptualHashManifestDiff(added=(), removed=(), changed=(), unchanged=())
    with pytest.raises(dataclasses.FrozenInstanceError):
        diff.added = ()  # type: ignore[misc]


def test_change_has_no_dict() -> None:
    change = PerceptualHashManifestChange(
        path=PurePosixPath("a.png"), before=_hash("0"), after=_hash("1")
    )
    assert not hasattr(change, "__dict__")
    assert PerceptualHashManifestChange.__slots__ == ("path", "before", "after")


def test_diff_has_no_dict() -> None:
    diff = PerceptualHashManifestDiff(added=(), removed=(), changed=(), unchanged=())
    assert not hasattr(diff, "__dict__")
    assert PerceptualHashManifestDiff.__slots__ == ("added", "removed", "changed", "unchanged")


def test_change_equality() -> None:
    change1 = PerceptualHashManifestChange(
        path=PurePosixPath("a.png"), before=_hash("0"), after=_hash("1")
    )
    change2 = PerceptualHashManifestChange(
        path=PurePosixPath("a.png"), before=_hash("0"), after=_hash("1")
    )
    assert change1 == change2
    assert hash(change1) == hash(change2)


def test_diff_equality() -> None:
    diff1 = PerceptualHashManifestDiff(
        added=(), removed=(), changed=(), unchanged=(PurePosixPath("a.png"),)
    )
    diff2 = PerceptualHashManifestDiff(
        added=(), removed=(), changed=(), unchanged=(PurePosixPath("a.png"),)
    )
    assert diff1 == diff2


def test_change_repr_contains_class_name_and_fields() -> None:
    change = PerceptualHashManifestChange(
        path=PurePosixPath("a.png"), before=_hash("0"), after=_hash("1")
    )
    text = repr(change)
    assert "PerceptualHashManifestChange" in text
    assert "a.png" in text


def test_diff_repr_contains_class_name() -> None:
    diff = PerceptualHashManifestDiff(added=(), removed=(), changed=(), unchanged=())
    assert "PerceptualHashManifestDiff" in repr(diff)


def test_change_field_order_and_types() -> None:
    fields = dataclasses.fields(PerceptualHashManifestChange)
    assert [field.name for field in fields] == ["path", "before", "after"]


def test_diff_field_order_and_types() -> None:
    fields = dataclasses.fields(PerceptualHashManifestDiff)
    assert [field.name for field in fields] == ["added", "removed", "changed", "unchanged"]


def test_change_has_no_custom_post_init() -> None:
    assert "__post_init__" not in PerceptualHashManifestChange.__dict__


def test_diff_has_no_custom_post_init() -> None:
    assert "__post_init__" not in PerceptualHashManifestDiff.__dict__


def test_change_manual_construction_accepts_mismatched_path_without_validation() -> None:
    # No custom __post_init__: manual construction keeps ordinary, statically-typed dataclass
    # semantics -- it does not re-validate that before/after actually differ, or that the
    # dataclass was produced by compare_perceptual_hash_manifests itself.
    change = PerceptualHashManifestChange(
        path=PurePosixPath("x.png"), before=_hash("0"), after=_hash("0")
    )
    assert change.before == change.after


# =====================================================================================
# Public API
# =====================================================================================


def test_exports_from_improcv_manifest() -> None:
    assert manifest_module.compare_perceptual_hash_manifests is compare_perceptual_hash_manifests
    assert manifest_module.PerceptualHashManifestDiff is PerceptualHashManifestDiff
    assert manifest_module.PerceptualHashManifestChange is PerceptualHashManifestChange


def test_top_level_exports() -> None:
    assert im.compare_perceptual_hash_manifests is compare_perceptual_hash_manifests
    assert im.PerceptualHashManifestDiff is PerceptualHashManifestDiff
    assert im.PerceptualHashManifestChange is PerceptualHashManifestChange


def test_manifest_module_all_is_exact() -> None:
    assert manifest_module.__all__ == [
        "PerceptualHashManifest",
        "PerceptualHashManifestChange",
        "PerceptualHashManifestDiff",
        "PerceptualHashManifestEntry",
        "compare_perceptual_hash_manifests",
    ]


def test_top_level_all_contains_new_symbols_at_sorted_positions() -> None:
    names = im.__all__
    assert names.index("PerceptualHashManifest") < names.index("PerceptualHashManifestChange")
    assert names.index("PerceptualHashManifestChange") < names.index("PerceptualHashManifestDiff")
    assert names.index("PerceptualHashManifestDiff") < names.index("PerceptualHashManifestEntry")
    assert (
        names.index("classification_metrics_from_confusion_matrix")
        < names.index("compare_perceptual_hash_manifests")
        < names.index("confusion_matrix")
    )


def test_manifest_module_all_exports_no_private_helpers() -> None:
    for name in manifest_module.__all__:
        assert not name.startswith("_")


# =====================================================================================
# Behavior
# =====================================================================================


def test_deterministic_result_across_repeated_calls() -> None:
    before = _manifest({"a.png": _hash("0"), "b.png": _hash("1")})
    after = _manifest({"a.png": _hash("0"), "b.png": _hash("9"), "c.png": _hash("2")})
    result1 = compare_perceptual_hash_manifests(before, after)
    result2 = compare_perceptual_hash_manifests(before, after)
    assert result1 == result2


def test_rename_is_always_reported_as_remove_plus_add() -> None:
    before = _manifest({"cats/old_name.png": _hash("0")})
    after = _manifest({"cats/new_name.png": _hash("0")})
    result = compare_perceptual_hash_manifests(before, after)
    assert tuple(entry.path for entry in result.removed) == (PurePosixPath("cats/old_name.png"),)
    assert tuple(entry.path for entry in result.added) == (PurePosixPath("cats/new_name.png"),)
    assert result.changed == ()
    assert result.unchanged == ()
    assert result.removed[0].hash == result.added[0].hash


def test_identical_hash_under_different_path_is_neither_changed_nor_merged() -> None:
    before = _manifest({"a.png": _hash("0"), "old.png": _hash("5")})
    after = _manifest({"a.png": _hash("0"), "new.png": _hash("5")})
    result = compare_perceptual_hash_manifests(before, after)
    # "old.png" and "new.png" share a hash, but neither counts as a rename or a change of
    # some third identity -- they are plain, independent removed/added entries.
    assert tuple(entry.path for entry in result.removed) == (PurePosixPath("old.png"),)
    assert tuple(entry.path for entry in result.added) == (PurePosixPath("new.png"),)
    assert result.changed == ()
    assert result.unchanged == (PurePosixPath("a.png"),)


def test_does_not_call_find_similar_image_pairs(monkeypatch: pytest.MonkeyPatch) -> None:
    import improcv.similarity as similarity_module

    monkeypatch.setattr(similarity_module, "find_similar_image_pairs", _Boom())

    before = _manifest({"a.png": _hash("0"), "old.png": _hash("5")})
    after = _manifest({"a.png": _hash("0"), "new.png": _hash("5")})
    result = compare_perceptual_hash_manifests(before, after)

    assert tuple(entry.path for entry in result.removed) == (PurePosixPath("old.png"),)
    assert tuple(entry.path for entry in result.added) == (PurePosixPath("new.png"),)


def test_does_not_touch_filesystem(monkeypatch: pytest.MonkeyPatch) -> None:
    boom = _Boom()
    monkeypatch.setattr(Path, "exists", boom)
    monkeypatch.setattr(Path, "stat", boom)
    monkeypatch.setattr(Path, "is_file", boom)
    monkeypatch.setattr("builtins.open", boom)

    before = _manifest({"nonexistent/a.png": _hash("0")})
    after = _manifest({"nonexistent/a.png": _hash("1")})
    result = compare_perceptual_hash_manifests(before, after)
    assert len(result.changed) == 1


def test_does_not_mutate_input_manifests() -> None:
    before = _manifest({"a.png": _hash("0"), "b.png": _hash("1")})
    after = _manifest({"a.png": _hash("9"), "c.png": _hash("2")})
    before_copy = _manifest({"a.png": _hash("0"), "b.png": _hash("1")})
    after_copy = _manifest({"a.png": _hash("9"), "c.png": _hash("2")})

    compare_perceptual_hash_manifests(before, after)

    assert before == before_copy
    assert after == after_copy
