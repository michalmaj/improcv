import dataclasses
import os
from pathlib import Path

import numpy as np
import pytest

import improcv as im
import improcv.similarity as similarity_module
from improcv.hashing import PerceptualHash, PerceptualHashAlgorithm
from improcv.similarity import SimilarImagePair, find_similar_image_pairs

_PHASH = PerceptualHashAlgorithm.PHASH
_AVERAGE_HASH = PerceptualHashAlgorithm.AVERAGE_HASH


def _hash(
    hex_value: str,
    *,
    algorithm: PerceptualHashAlgorithm = _PHASH,
    hash_size: int = 2,
) -> PerceptualHash:
    """Build a `PerceptualHash` with an easily hand-verifiable value.

    `hash_size=2` gives 4 bits (a single hex digit), so the distances between
    small hex values can be checked by hand: e.g. "0" (0000) vs "1" (0001) is
    Hamming distance 1, "0" vs "3" (0011) is distance 2.
    """
    return PerceptualHash.from_hex(hex_value, algorithm=algorithm, hash_size=hash_size)


def _hashes(
    mapping: dict[str | os.PathLike[str], PerceptualHash],
) -> dict[str | os.PathLike[str], PerceptualHash]:
    """Identity wrapper that widens a dict literal's inferred key type to match
    `find_similar_image_pairs`' own `Mapping[str | os.PathLike[str], ...]`
    parameter type.

    `Mapping`'s key type is invariant in the type system, so a plain
    `dict[str, PerceptualHash]` literal is not otherwise assignable where
    `Mapping[str | os.PathLike[str], PerceptualHash]` is expected -- passing
    the literal through this function's annotated parameter instead makes
    Pyright infer the literal directly against the wider key type.
    """
    return mapping


class _Boom:
    def __call__(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("find_similar_image_pairs must not perform this operation")


# =====================================================================================
# SimilarImagePair
# =====================================================================================


def test_pair_valid_construction() -> None:
    pair = SimilarImagePair(first=Path("a.png"), second=Path("b.png"), distance=3)
    assert pair.first == Path("a.png")
    assert pair.second == Path("b.png")
    assert pair.distance == 3


def test_pair_equality() -> None:
    pair1 = SimilarImagePair(first=Path("a.png"), second=Path("b.png"), distance=3)
    pair2 = SimilarImagePair(first=Path("a.png"), second=Path("b.png"), distance=3)
    assert pair1 == pair2


def test_pair_hashability() -> None:
    pair1 = SimilarImagePair(first=Path("a.png"), second=Path("b.png"), distance=3)
    pair2 = SimilarImagePair(first=Path("a.png"), second=Path("b.png"), distance=3)
    assert hash(pair1) == hash(pair2)
    assert {pair1, pair2} == {pair1}


def test_pair_repr_is_readable() -> None:
    pair = SimilarImagePair(first=Path("a.png"), second=Path("b.png"), distance=3)
    text = repr(pair)
    assert "a.png" in text
    assert "b.png" in text
    assert "3" in text


def test_pair_is_frozen() -> None:
    pair = SimilarImagePair(first=Path("a.png"), second=Path("b.png"), distance=3)
    with pytest.raises(dataclasses.FrozenInstanceError):
        pair.distance = 5  # type: ignore[misc]


def test_pair_rejects_str_as_first() -> None:
    with pytest.raises(TypeError, match="first"):
        SimilarImagePair(first="a.png", second=Path("b.png"), distance=1)  # type: ignore[arg-type]


def test_pair_rejects_str_as_second() -> None:
    with pytest.raises(TypeError, match="second"):
        SimilarImagePair(first=Path("a.png"), second="b.png", distance=1)  # type: ignore[arg-type]


def test_pair_rejects_identical_paths() -> None:
    with pytest.raises(ValueError, match="different"):
        SimilarImagePair(first=Path("a.png"), second=Path("a.png"), distance=0)


def test_pair_rejects_reverse_canonical_order() -> None:
    with pytest.raises(ValueError, match="canonical"):
        SimilarImagePair(first=Path("b.png"), second=Path("a.png"), distance=1)


def test_pair_rejects_negative_distance() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        SimilarImagePair(first=Path("a.png"), second=Path("b.png"), distance=-1)


def test_pair_rejects_bool_distance() -> None:
    with pytest.raises(TypeError):
        SimilarImagePair(first=Path("a.png"), second=Path("b.png"), distance=True)  # type: ignore[arg-type]


def test_pair_rejects_float_distance() -> None:
    with pytest.raises(TypeError):
        SimilarImagePair(first=Path("a.png"), second=Path("b.png"), distance=1.0)  # type: ignore[arg-type]


def test_pair_accepts_zero_distance() -> None:
    pair = SimilarImagePair(first=Path("a.png"), second=Path("b.png"), distance=0)
    assert pair.distance == 0


# =====================================================================================
# find_similar_image_pairs -- basics
# =====================================================================================


def test_empty_mapping_returns_empty_tuple() -> None:
    assert find_similar_image_pairs({}, max_distance=0) == ()


def test_single_entry_returns_empty_tuple() -> None:
    result = find_similar_image_pairs({"a.png": _hash("0")}, max_distance=4)
    assert result == ()


def test_two_entries_outside_threshold_returns_empty_tuple() -> None:
    result = find_similar_image_pairs({"a.png": _hash("0"), "b.png": _hash("F")}, max_distance=1)
    assert result == ()


def test_exact_match_at_threshold_boundary_is_included() -> None:
    result = find_similar_image_pairs({"a.png": _hash("0"), "b.png": _hash("1")}, max_distance=1)
    assert result == (SimilarImagePair(first=Path("a.png"), second=Path("b.png"), distance=1),)


def test_excluded_one_past_threshold_boundary() -> None:
    result = find_similar_image_pairs({"a.png": _hash("0"), "b.png": _hash("1")}, max_distance=0)
    assert result == ()


def test_zero_threshold_matches_identical_hashes() -> None:
    result = find_similar_image_pairs({"a.png": _hash("0"), "b.png": _hash("0")}, max_distance=0)
    assert result == (SimilarImagePair(first=Path("a.png"), second=Path("b.png"), distance=0),)


def test_two_different_paths_with_identical_hash_is_a_legal_pair() -> None:
    result = find_similar_image_pairs({"a.png": _hash("5"), "b.png": _hash("5")}, max_distance=0)
    assert len(result) == 1
    assert result[0].distance == 0


def test_maximum_legal_threshold_returns_all_pairs() -> None:
    hashes = _hashes({"a.png": _hash("0"), "b.png": _hash("1"), "c.png": _hash("3")})
    result = find_similar_image_pairs(hashes, max_distance=4)
    assert len(result) == 3


def test_every_pair_checked_exactly_once() -> None:
    hashes = _hashes(
        {"a.png": _hash("0"), "b.png": _hash("1"), "c.png": _hash("3"), "d.png": _hash("F")}
    )
    result = find_similar_image_pairs(hashes, max_distance=4)
    seen_keys = {frozenset((p.first, p.second)) for p in result}
    assert len(seen_keys) == len(result) == 6


def test_no_self_pairs() -> None:
    hashes = _hashes({"a.png": _hash("0"), "b.png": _hash("1"), "c.png": _hash("3")})
    result = find_similar_image_pairs(hashes, max_distance=4)
    assert all(pair.first != pair.second for pair in result)


def test_result_is_a_tuple() -> None:
    result = find_similar_image_pairs({"a.png": _hash("0")}, max_distance=0)
    assert isinstance(result, tuple)


def test_result_elements_are_similar_image_pair() -> None:
    result = find_similar_image_pairs({"a.png": _hash("0"), "b.png": _hash("1")}, max_distance=1)
    assert all(isinstance(pair, SimilarImagePair) for pair in result)


# =====================================================================================
# find_similar_image_pairs -- determinism and ordering
# =====================================================================================


def test_insertion_order_does_not_affect_result() -> None:
    forward = _hashes({"a.png": _hash("0"), "b.png": _hash("1"), "c.png": _hash("3")})
    backward = _hashes({"c.png": _hash("3"), "b.png": _hash("1"), "a.png": _hash("0")})
    assert find_similar_image_pairs(forward, max_distance=4) == find_similar_image_pairs(
        backward, max_distance=4
    )


def test_result_sorted_by_distance_primarily() -> None:
    hashes = _hashes({"a.png": _hash("0"), "b.png": _hash("1"), "c.png": _hash("3")})
    result = find_similar_image_pairs(hashes, max_distance=4)
    distances = [pair.distance for pair in result]
    assert distances == sorted(distances)


def test_ties_in_distance_broken_by_first_path() -> None:
    # a<->m: distance(3, 1) = 1; a<->z: distance(3, 0) = 2; m<->z: distance(1, 0) = 1
    hashes = _hashes({"z.png": _hash("0"), "m.png": _hash("1"), "a.png": _hash("3")})
    result = find_similar_image_pairs(hashes, max_distance=4)
    assert [(p.first.name, p.second.name, p.distance) for p in result] == [
        ("a.png", "m.png", 1),
        ("m.png", "z.png", 1),
        ("a.png", "z.png", 2),
    ]


def test_ties_in_distance_and_first_broken_by_second_path() -> None:
    # a<->b: distance(0, 1) = 1; a<->c: distance(0, 2) = 1; b<->c: distance(1, 2) = 2
    hashes = _hashes({"a.png": _hash("0"), "b.png": _hash("1"), "c.png": _hash("2")})
    result = find_similar_image_pairs(hashes, max_distance=4)
    assert [(p.first.name, p.second.name, p.distance) for p in result] == [
        ("a.png", "b.png", 1),
        ("a.png", "c.png", 1),
        ("b.png", "c.png", 2),
    ]


def test_every_pair_is_in_canonical_order() -> None:
    hashes = _hashes({"z.png": _hash("0"), "m.png": _hash("1"), "a.png": _hash("3")})
    result = find_similar_image_pairs(hashes, max_distance=4)
    assert all(pair.first.as_posix() < pair.second.as_posix() for pair in result)


def test_three_entries_give_exactly_three_pairs_at_max_threshold() -> None:
    hashes = _hashes({"a.png": _hash("0"), "b.png": _hash("1"), "c.png": _hash("3")})
    result = find_similar_image_pairs(hashes, max_distance=4)
    assert len(result) == 3


def test_four_entries_give_exactly_six_pairs_at_max_threshold() -> None:
    hashes = _hashes(
        {"a.png": _hash("0"), "b.png": _hash("1"), "c.png": _hash("3"), "d.png": _hash("F")}
    )
    result = find_similar_image_pairs(hashes, max_distance=4)
    assert len(result) == 6


# =====================================================================================
# find_similar_image_pairs -- non-transitivity (load-bearing semantics test)
# =====================================================================================


def test_returns_pairs_without_transitive_grouping() -> None:
    """A = 0000, B = 0001, C = 0011; max_distance=1.

    distance(A, B) = 1, distance(B, C) = 1, distance(A, C) = 2 -- the result
    must contain exactly (A, B) and (B, C), never a merged group {A, B, C},
    never only one of the two pairs, and never (A, C).
    """
    hashes = _hashes({"A.png": _hash("0"), "B.png": _hash("1"), "C.png": _hash("3")})
    result = find_similar_image_pairs(hashes, max_distance=1)
    assert result == (
        SimilarImagePair(first=Path("A.png"), second=Path("B.png"), distance=1),
        SimilarImagePair(first=Path("B.png"), second=Path("C.png"), distance=1),
    )


# =====================================================================================
# find_similar_image_pairs -- hashes mapping validation
# =====================================================================================


def test_rejects_list_of_pairs() -> None:
    with pytest.raises(TypeError, match="Mapping"):
        find_similar_image_pairs([("a.png", _hash("0"))], max_distance=0)  # type: ignore[arg-type]


def test_rejects_tuple_of_pairs() -> None:
    with pytest.raises(TypeError, match="Mapping"):
        find_similar_image_pairs((("a.png", _hash("0")),), max_distance=0)  # type: ignore[arg-type]


def test_rejects_generator() -> None:
    with pytest.raises(TypeError, match="Mapping"):
        find_similar_image_pairs((x for x in ()), max_distance=0)  # type: ignore[arg-type]


def test_rejects_string() -> None:
    with pytest.raises(TypeError, match="Mapping"):
        find_similar_image_pairs("a.png", max_distance=0)  # type: ignore[arg-type]


def test_rejects_none_as_hashes() -> None:
    with pytest.raises(TypeError, match="Mapping"):
        find_similar_image_pairs(None, max_distance=0)  # type: ignore[arg-type]


def test_rejects_int_key() -> None:
    with pytest.raises(TypeError):
        find_similar_image_pairs({5: _hash("0")}, max_distance=0)  # type: ignore[dict-item]


def test_rejects_bytes_key() -> None:
    with pytest.raises(TypeError):
        find_similar_image_pairs({b"a.png": _hash("0")}, max_distance=0)  # type: ignore[dict-item]


def test_rejects_pathlike_that_returns_bytes() -> None:
    class _BytesPath:
        def __fspath__(self) -> bytes:
            return b"a.png"

    with pytest.raises(TypeError):
        find_similar_image_pairs({_BytesPath(): _hash("0")}, max_distance=0)  # type: ignore[dict-item]


def test_rejects_empty_string_key() -> None:
    with pytest.raises(ValueError, match="empty"):
        find_similar_image_pairs({"": _hash("0")}, max_distance=0)


def test_rejects_nul_in_path() -> None:
    with pytest.raises(ValueError, match="NUL"):
        find_similar_image_pairs({"a\x00.png": _hash("0")}, max_distance=0)


def test_rejects_int_value() -> None:
    with pytest.raises(TypeError, match="PerceptualHash"):
        find_similar_image_pairs({"a.png": 5}, max_distance=0)  # type: ignore[dict-item]


def test_rejects_string_value() -> None:
    with pytest.raises(TypeError, match="PerceptualHash"):
        find_similar_image_pairs({"a.png": "0"}, max_distance=0)  # type: ignore[dict-item]


def test_rejects_none_value() -> None:
    with pytest.raises(TypeError, match="PerceptualHash"):
        find_similar_image_pairs({"a.png": None}, max_distance=0)  # type: ignore[dict-item]


def test_rejects_colliding_keys_str_and_path() -> None:
    with pytest.raises(ValueError, match="a.png"):
        find_similar_image_pairs({"a.png": _hash("0"), Path("a.png"): _hash("1")}, max_distance=4)


def test_rejects_colliding_keys_after_dot_and_separator_normalization() -> None:
    with pytest.raises(ValueError):
        find_similar_image_pairs({"./a.png": _hash("0"), "a.png": _hash("1")}, max_distance=4)


def test_nonexistent_paths_are_legal() -> None:
    result = find_similar_image_pairs(
        {"does/not/exist/a.png": _hash("0"), "does/not/exist/b.png": _hash("1")},
        max_distance=1,
    )
    assert len(result) == 1


def test_relative_and_absolute_identifiers_are_legal() -> None:
    result = find_similar_image_pairs(
        {"relative/a.png": _hash("0"), "/absolute/b.png": _hash("1")},
        max_distance=1,
    )
    assert len(result) == 1


def test_does_not_expand_user_or_resolve() -> None:
    result = find_similar_image_pairs(
        {"~/a.png": _hash("0"), "~/b.png": _hash("1")}, max_distance=1
    )
    assert len(result) == 1
    pair = result[0]
    assert "~" in pair.first.as_posix()
    assert "~" in pair.second.as_posix()


# =====================================================================================
# find_similar_image_pairs -- hash-space compatibility
# =====================================================================================


def test_same_phash_same_hash_size_works() -> None:
    result = find_similar_image_pairs(
        {"a.png": _hash("0", algorithm=_PHASH), "b.png": _hash("1", algorithm=_PHASH)},
        max_distance=1,
    )
    assert len(result) == 1


def test_same_average_hash_same_hash_size_works() -> None:
    result = find_similar_image_pairs(
        {
            "a.png": _hash("0", algorithm=_AVERAGE_HASH),
            "b.png": _hash("1", algorithm=_AVERAGE_HASH),
        },
        max_distance=1,
    )
    assert len(result) == 1


def test_mixed_algorithms_raise_value_error() -> None:
    with pytest.raises(ValueError, match="algorithm and hash_size"):
        find_similar_image_pairs(
            {
                "a.png": _hash("0", algorithm=_PHASH),
                "b.png": _hash("1", algorithm=_AVERAGE_HASH),
            },
            max_distance=1,
        )


def test_mixed_hash_sizes_raise_value_error() -> None:
    with pytest.raises(ValueError, match="algorithm and hash_size"):
        find_similar_image_pairs(
            {
                "a.png": _hash("0", hash_size=2),
                "b.png": _hash("001", hash_size=3),
            },
            max_distance=1,
        )


def test_three_distinct_spaces_named_in_one_deterministic_error() -> None:
    with pytest.raises(ValueError) as exc_info:
        find_similar_image_pairs(
            {
                "a.png": _hash("0", algorithm=_AVERAGE_HASH, hash_size=2),
                "b.png": _hash("0", algorithm=_PHASH, hash_size=2),
                "c.png": _hash("111", algorithm=_PHASH, hash_size=3),
            },
            max_distance=1,
        )
    message = str(exc_info.value)
    assert "average_hash(hash_size=2)" in message
    assert "phash(hash_size=2)" in message
    assert "phash(hash_size=3)" in message
    # Deterministic ordering: sorted by (algorithm, hash_size).
    assert message.index("average_hash(hash_size=2)") < message.index("phash(hash_size=2)")
    assert message.index("phash(hash_size=2)") < message.index("phash(hash_size=3)")


def test_incompatibility_detected_before_returning_any_result() -> None:
    with pytest.raises(ValueError):
        find_similar_image_pairs(
            {
                "a.png": _hash("0", algorithm=_PHASH),
                "b.png": _hash("0", algorithm=_PHASH),
                "c.png": _hash("0", algorithm=_AVERAGE_HASH),
            },
            max_distance=4,
        )


def test_max_distance_upper_bound_from_hash_size_squared() -> None:
    result = find_similar_image_pairs({"a.png": _hash("0"), "b.png": _hash("F")}, max_distance=4)
    assert len(result) == 1


def test_max_distance_above_bit_count_raises() -> None:
    with pytest.raises(ValueError, match="hash_size"):
        find_similar_image_pairs({"a.png": _hash("0"), "b.png": _hash("F")}, max_distance=5)


def test_singleton_still_validates_upper_bound() -> None:
    with pytest.raises(ValueError, match="hash_size"):
        find_similar_image_pairs({"a.png": _hash("0")}, max_distance=5)


def test_empty_mapping_has_no_upper_bound_but_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        find_similar_image_pairs({}, max_distance=-1)


def test_empty_mapping_has_no_upper_bound_but_rejects_bad_type() -> None:
    with pytest.raises(TypeError):
        find_similar_image_pairs({}, max_distance=1.5)  # type: ignore[arg-type]


# =====================================================================================
# find_similar_image_pairs -- max_distance validation
# =====================================================================================


def test_rejects_bool_max_distance() -> None:
    with pytest.raises(TypeError):
        find_similar_image_pairs({"a.png": _hash("0")}, max_distance=True)  # type: ignore[arg-type]


def test_rejects_float_max_distance() -> None:
    with pytest.raises(TypeError):
        find_similar_image_pairs({"a.png": _hash("0")}, max_distance=1.0)  # type: ignore[arg-type]


def test_rejects_string_max_distance() -> None:
    with pytest.raises(TypeError):
        find_similar_image_pairs({"a.png": _hash("0")}, max_distance="1")  # type: ignore[arg-type]


def test_rejects_numpy_integer_max_distance() -> None:
    with pytest.raises(TypeError):
        find_similar_image_pairs({"a.png": _hash("0")}, max_distance=np.int64(1))  # type: ignore[arg-type]


def test_rejects_none_max_distance() -> None:
    with pytest.raises(TypeError):
        find_similar_image_pairs({"a.png": _hash("0")}, max_distance=None)  # type: ignore[arg-type]


def test_rejects_negative_max_distance() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        find_similar_image_pairs({"a.png": _hash("0"), "b.png": _hash("1")}, max_distance=-1)


# =====================================================================================
# find_similar_image_pairs -- no I/O, no mutation
# =====================================================================================


def test_does_not_touch_filesystem(monkeypatch: pytest.MonkeyPatch) -> None:
    boom = _Boom()
    monkeypatch.setattr(Path, "exists", boom)
    monkeypatch.setattr(Path, "stat", boom)
    monkeypatch.setattr(Path, "is_file", boom)
    monkeypatch.setattr("builtins.open", boom)

    result = find_similar_image_pairs(
        {"nonexistent/a.png": _hash("0"), "nonexistent/b.png": _hash("1")}, max_distance=1
    )
    assert len(result) == 1


def test_does_not_import_cv2_or_discovery() -> None:
    """Static architecture-boundary check: `similarity.py` never references cv2,
    image decoding, or `discover_images` -- comparing precomputed hashes is its
    entire responsibility.
    """
    source = Path(similarity_module.__file__).read_text()
    assert "cv2" not in source
    assert "imread" not in source
    assert "discover_images" not in source


def test_does_not_modify_the_input_mapping() -> None:
    hash_a = _hash("0")
    hash_b = _hash("1")
    hashes = _hashes({"a.png": hash_a, "b.png": hash_b})
    snapshot = dict(hashes)

    find_similar_image_pairs(hashes, max_distance=1)

    assert hashes == snapshot
    assert hashes["a.png"] is hash_a
    assert hashes["b.png"] is hash_b


def test_does_not_modify_hash_objects() -> None:
    hash_a = _hash("0")
    hash_b = _hash("1")
    before_a, before_b = repr(hash_a), repr(hash_b)

    find_similar_image_pairs({"a.png": hash_a, "b.png": hash_b}, max_distance=1)

    assert repr(hash_a) == before_a
    assert repr(hash_b) == before_b


def test_does_not_change_cwd() -> None:
    before = os.getcwd()
    find_similar_image_pairs({"a.png": _hash("0"), "b.png": _hash("1")}, max_distance=1)
    assert os.getcwd() == before


# =====================================================================================
# Exports
# =====================================================================================


def test_top_level_exports_are_the_same_objects() -> None:
    assert im.SimilarImagePair is SimilarImagePair
    assert im.find_similar_image_pairs is find_similar_image_pairs


def test_module_all_contains_exactly_the_public_symbols() -> None:
    assert similarity_module.__all__ == ["SimilarImagePair", "find_similar_image_pairs"]


def test_top_level_all_contains_new_symbols_without_duplicates() -> None:
    assert im.__all__.count("SimilarImagePair") == 1
    assert im.__all__.count("find_similar_image_pairs") == 1
    assert len(im.__all__) == len(set(im.__all__))


def test_top_level_all_places_new_symbols_alphabetically() -> None:
    # "SimilarImagePair" belongs alphabetically between "SeamlessCloneMode" and
    # "SortOrder"; "find_similar_image_pairs" belongs between "find_homography"
    # and "flip". Checked against immediate neighbors rather than asserting the
    # whole list is one global sort, since improcv.__all__ predates this change
    # with its own established (and not perfectly sorted -- e.g. "Mask" before
    # "MSERRegion") ordering that this PR must not disturb.
    index = im.__all__.index("SimilarImagePair")
    assert im.__all__[index - 1] == "SeamlessCloneMode"
    assert im.__all__[index + 1] == "SortOrder"

    index = im.__all__.index("find_similar_image_pairs")
    assert im.__all__[index - 1] == "find_homography"
    assert im.__all__[index + 1] == "flip"
