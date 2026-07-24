import dataclasses
import math

import cv2
import numpy as np
import pytest

import improcv as im
from improcv.hashing import (
    PerceptualHash,
    PerceptualHashAlgorithm,
    average_hash,
    phash,
)

_FUNCS = [average_hash, phash]
_FUNC_NAMES = ["average_hash", "phash"]


def _algorithm_for(func) -> PerceptualHashAlgorithm:
    return {
        average_hash: PerceptualHashAlgorithm.AVERAGE_HASH,
        phash: PerceptualHashAlgorithm.PHASH,
    }[func]


# --- basic behavior ---


@pytest.mark.parametrize("func", _FUNCS, ids=_FUNC_NAMES)
def test_returns_a_perceptual_hash(func) -> None:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (32, 32), dtype=np.uint8)

    result = func(img)

    assert isinstance(result, PerceptualHash)
    assert result.algorithm == _algorithm_for(func)
    assert result.hash_size == 8
    assert len(result) == 64


@pytest.mark.parametrize("func", _FUNCS, ids=_FUNC_NAMES)
def test_is_deterministic(func) -> None:
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (32, 32), dtype=np.uint8)

    assert func(img) == func(img.copy())


@pytest.mark.parametrize("func", _FUNCS, ids=_FUNC_NAMES)
def test_does_not_mutate_input(func) -> None:
    rng = np.random.default_rng(2)
    img = rng.integers(0, 256, (32, 32), dtype=np.uint8)
    before = img.copy()

    func(img)

    np.testing.assert_array_equal(img, before)


@pytest.mark.parametrize("func", _FUNCS, ids=_FUNC_NAMES)
def test_constant_image_gives_all_zero_hash(func) -> None:
    img = np.full((32, 32), 100, dtype=np.uint8)

    result = func(img)

    assert result.distance(func(img)) == 0
    assert str(result) == "0" * (len(result) // 4)


@pytest.mark.parametrize("func", _FUNCS, ids=_FUNC_NAMES)
@pytest.mark.parametrize("hash_size", [2, 3, 8, 16, 256])
def test_accepts_valid_hash_sizes(func, hash_size: int) -> None:
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, (64, 64), dtype=np.uint8)

    result = func(img, hash_size=hash_size)

    assert result.hash_size == hash_size
    assert len(result) == hash_size**2


@pytest.mark.parametrize("func", _FUNCS, ids=_FUNC_NAMES)
@pytest.mark.parametrize("hash_size", [0, 1, 257, -1])
def test_rejects_out_of_range_hash_size(func, hash_size: int) -> None:
    rng = np.random.default_rng(4)
    img = rng.integers(0, 256, (64, 64), dtype=np.uint8)

    with pytest.raises(ValueError):
        func(img, hash_size=hash_size)


@pytest.mark.parametrize("func", _FUNCS, ids=_FUNC_NAMES)
@pytest.mark.parametrize("bad_hash_size", [8.0, True, False, "8", None])
def test_rejects_non_int_hash_size(func, bad_hash_size: object) -> None:
    rng = np.random.default_rng(5)
    img = rng.integers(0, 256, (64, 64), dtype=np.uint8)

    with pytest.raises(TypeError):
        func(img, hash_size=bad_hash_size)


@pytest.mark.parametrize("func", _FUNCS, ids=_FUNC_NAMES)
def test_accepts_image_smaller_than_target_size(func) -> None:
    rng = np.random.default_rng(6)
    img = rng.integers(0, 256, (3, 3), dtype=np.uint8)

    result = func(img, hash_size=8)

    assert math.isfinite(len(result))  # just must not raise/crash


# --- validation ---


@pytest.mark.parametrize("func", _FUNCS, ids=_FUNC_NAMES)
def test_rejects_empty_image(func) -> None:
    img = np.zeros((0, 10), dtype=np.uint8)

    with pytest.raises(ValueError):
        func(img)


@pytest.mark.parametrize("func", _FUNCS, ids=_FUNC_NAMES)
def test_rejects_wrong_ndim(func) -> None:
    img = np.zeros((5, 5, 5, 5), dtype=np.uint8)

    with pytest.raises(ValueError):
        func(img)


@pytest.mark.parametrize("func", _FUNCS, ids=_FUNC_NAMES)
def test_rejects_two_channel_image(func) -> None:
    rng = np.random.default_rng(7)
    img = rng.integers(0, 256, (20, 20, 2), dtype=np.uint8)

    with pytest.raises(ValueError):
        func(img)


@pytest.mark.parametrize("func", _FUNCS, ids=_FUNC_NAMES)
def test_rejects_more_than_four_channels(func) -> None:
    rng = np.random.default_rng(8)
    img = rng.integers(0, 256, (20, 20, 5), dtype=np.uint8)

    with pytest.raises(ValueError):
        func(img)


@pytest.mark.parametrize("func", _FUNCS, ids=_FUNC_NAMES)
@pytest.mark.parametrize("dtype", [np.uint16, np.int32, np.float32, np.float64, np.bool_])
def test_rejects_non_uint8_dtype(func, dtype) -> None:
    rng = np.random.default_rng(9)
    img = rng.integers(0, 2, (20, 20)).astype(dtype)

    with pytest.raises(TypeError):
        func(img)


@pytest.mark.parametrize("func", _FUNCS, ids=_FUNC_NAMES)
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_accepts_1_3_4_channels(func, channels: int) -> None:
    rng = np.random.default_rng(10)
    shape = (20, 20) if channels == 1 else (20, 20, channels)
    img = rng.integers(0, 256, shape, dtype=np.uint8)

    result = func(img)

    assert len(result) == 64


@pytest.mark.parametrize("func", _FUNCS, ids=_FUNC_NAMES)
def test_h_w_1_reduces_to_h_w(func) -> None:
    rng = np.random.default_rng(11)
    img_2d = rng.integers(0, 256, (32, 32), dtype=np.uint8)
    img_3d = img_2d.reshape(32, 32, 1)

    assert func(img_2d) == func(img_3d)


def test_validation_order_image_before_hash_size() -> None:
    img = np.zeros((0, 10), dtype=np.uint8)

    with pytest.raises(ValueError, match="empty"):
        average_hash(img, hash_size=-5)


# --- color: order matters, alpha is ignored not composited ---


@pytest.mark.parametrize("func", _FUNCS, ids=_FUNC_NAMES)
def test_bgra_ignores_alpha_value(func) -> None:
    rng = np.random.default_rng(12)
    bgr = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    opaque = np.dstack([bgr, np.full((32, 32), 255, dtype=np.uint8)])
    transparent = np.dstack([bgr, np.full((32, 32), 0, dtype=np.uint8)])
    half = np.dstack([bgr, np.full((32, 32), 128, dtype=np.uint8)])

    assert func(opaque) == func(transparent) == func(half)


@pytest.mark.parametrize("func", _FUNCS, ids=_FUNC_NAMES)
def test_bgra_hidden_bgr_can_differ_at_alpha_zero(func) -> None:
    rng = np.random.default_rng(13)
    bgr_a = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    bgr_b = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    transparent_a = np.dstack([bgr_a, np.zeros((32, 32), dtype=np.uint8)])
    transparent_b = np.dstack([bgr_b, np.zeros((32, 32), dtype=np.uint8)])

    assert func(transparent_a) != func(transparent_b)


def test_resize_before_grayscale_matters_for_average_hash() -> None:
    # Regression guard: grayscale-then-resize (e.g. via improcv.ensure_gray
    # first) does NOT reproduce the algorithm's actual resize-then-grayscale
    # order for color input -- uint8 rounding at each stage does not commute.
    # Seed 1 is pinned because it's confirmed (not every seed) to produce a
    # bit difference between the two orderings for this image size.
    import cv2

    rng = np.random.default_rng(1)
    bgr = rng.integers(0, 256, (37, 41, 3), dtype=np.uint8)

    correct = average_hash(bgr, hash_size=8)

    gray_first = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray_first, (8, 8), interpolation=cv2.INTER_LINEAR_EXACT)
    mean = cv2.mean(resized)[0]
    threshold = round(mean)
    bits = resized > threshold
    bit_string = "".join("1" if b else "0" for b in bits.flatten())
    wrong_order_value = int(bit_string, 2)

    assert correct._value != wrong_order_value


def test_resize_before_grayscale_matters_for_phash() -> None:
    # Seed 1 is pinned because it's confirmed (not every seed) to produce a
    # bit difference between the two orderings for this image size.
    import cv2

    rng = np.random.default_rng(1)
    bgr = rng.integers(0, 256, (37, 41, 3), dtype=np.uint8)

    correct = phash(bgr, hash_size=8)

    gray_first = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray_first, (32, 32), interpolation=cv2.INTER_LINEAR_EXACT)
    grayf = resized.astype(np.float32)
    dct = cv2.dct(grayf)
    block = dct[:8, :8].copy()
    block[0, 0] = 0.0
    threshold = np.float32(np.mean(block, dtype=np.float64))
    bits = block > threshold
    bit_string = "".join("1" if b else "0" for b in bits.flatten())
    wrong_order_value = int(bit_string, 2)

    assert correct._value != wrong_order_value


# --- average_hash: nearest-even threshold ---


def test_average_hash_mean_ending_in_half_rounds_down_to_even() -> None:
    # mean = 10.5 exactly; nearest-even rounds DOWN to 10 (10 is even).
    img = np.zeros((8, 8), dtype=np.uint8)
    flat = img.reshape(-1)
    flat[:32] = 10
    flat[32:] = 11

    result = average_hash(img, hash_size=8)

    # threshold=10 (round-half-to-even(10.5)); first 4 rows (value 10) are
    # not > 10 -> all-zero bits; last 4 rows (value 11) are > 10 -> all-one.
    expected = int("0" * 32 + "1" * 32, 2)
    assert result._value == expected


def test_average_hash_mean_ending_in_half_rounds_up_to_even() -> None:
    # mean = 11.5 exactly; nearest-even rounds UP to 12 (12 is even) -- so
    # even the pixels valued 12 are NOT strictly greater than the threshold.
    img = np.zeros((8, 8), dtype=np.uint8)
    flat = img.reshape(-1)
    flat[:32] = 11
    flat[32:] = 12

    result = average_hash(img, hash_size=8)

    assert result._value == 0  # nothing is strictly greater than 12


def test_average_hash_naive_round_half_up_would_disagree() -> None:
    # Confirms the two rounding conventions actually diverge on this input --
    # otherwise the test above wouldn't be pinning down round-to-even at all.
    img = np.zeros((8, 8), dtype=np.uint8)
    flat = img.reshape(-1)
    flat[:32] = 10
    flat[32:] = 11

    result = average_hash(img, hash_size=8)

    naive_threshold = 11  # round-half-up(10.5)
    naive_bits = img > naive_threshold
    naive_value = int("".join("1" if b else "0" for b in naive_bits.flatten()), 2)
    assert result._value != naive_value


# --- phash: threshold computation regression, exercised through the actual
# public phash() call via monkeypatched cv2.dct/cv2.mean (not a standalone
# numeric assertion disconnected from production code -- an earlier version
# of this test only checked the arithmetic in isolation and passed unchanged
# even after removing the float32 cast from phash() itself). ---


def test_phash_threshold_cast_is_load_bearing(monkeypatch: pytest.MonkeyPatch) -> None:
    # coefficient sits exactly at mean64's nearest float32 representative,
    # with mean64 approaching it from below (rounds UP when cast) -- the only
    # configuration where comparing against the uncast float64 mean vs. the
    # float32-cast mean can differ for a `>` comparison against that exact
    # value. These two scalar comparisons demonstrate that directly: they are
    # ordinary numpy scalar-to-scalar comparisons (not array-to-scalar), so
    # unlike `block > threshold` below they aren't affected by numpy's
    # array/scalar type-promotion rules (which changed between numpy's
    # "legacy" and NEP-50 promotion modes) -- confirmed identical on both.
    coefficient = np.float32(1.0)
    mean64 = np.nextafter(np.float64(1.0), np.float64(0.0))
    assert coefficient > mean64  # without the cast: true, at float64 precision
    assert not coefficient > np.float32(mean64)  # with the cast: false (now equal)

    hash_size = 8
    dct_size = hash_size * 4  # matches phash's internal _PHASH_HIGHFREQ_FACTOR

    # A synthetic "DCT output": every AC coefficient is 0 except the one at
    # the bottom-right corner of the hash_size x hash_size block, set to
    # `coefficient`. cv2.dct's real behavior doesn't matter here -- it's
    # replaced outright, so the actual pixel content of `image` below is
    # irrelevant too.
    fake_dct = np.zeros((dct_size, dct_size), dtype=np.float32)
    fake_dct[hash_size - 1, hash_size - 1] = coefficient
    monkeypatch.setattr(cv2, "dct", lambda _src: fake_dct)
    # cv2.mean's real return value doesn't matter either -- forcing it to
    # `mean64` regardless of the (fake) block's actual content isolates the
    # cast/comparison behavior precisely at the boundary constructed above.
    monkeypatch.setattr(cv2, "mean", lambda _src: (mean64, 0.0, 0.0, 0.0))

    image = np.zeros((dct_size, dct_size), dtype=np.uint8)
    result = phash(image, hash_size=hash_size)

    # The bottom-right block position is the *last* element in row-major
    # flattening, which PerceptualHash's own bit-packing convention (see
    # _bits_to_value) maps to the least significant bit of the value.
    assert result._value & 1 == 0


# --- PerceptualHash / PerceptualHashAlgorithm ---


def test_perceptual_hash_is_really_immutable() -> None:
    h = PerceptualHash(algorithm=PerceptualHashAlgorithm.AVERAGE_HASH, hash_size=8, _value=0)

    with pytest.raises(dataclasses.FrozenInstanceError):
        h.hash_size = 16  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        h.algorithm = PerceptualHashAlgorithm.PHASH  # type: ignore[misc]


def test_perceptual_hash_post_init_rejects_bad_algorithm() -> None:
    with pytest.raises(TypeError):
        PerceptualHash(algorithm="average_hash", hash_size=8, _value=0)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_hash_size", [1, 257, 0, -1])
def test_perceptual_hash_post_init_rejects_bad_hash_size(bad_hash_size: int) -> None:
    with pytest.raises(ValueError):
        PerceptualHash(
            algorithm=PerceptualHashAlgorithm.AVERAGE_HASH, hash_size=bad_hash_size, _value=0
        )


def test_perceptual_hash_post_init_rejects_bool_hash_size() -> None:
    with pytest.raises(TypeError):
        PerceptualHash(algorithm=PerceptualHashAlgorithm.AVERAGE_HASH, hash_size=True, _value=0)  # type: ignore[arg-type]


def test_perceptual_hash_post_init_rejects_out_of_range_value() -> None:
    with pytest.raises(ValueError):
        PerceptualHash(algorithm=PerceptualHashAlgorithm.AVERAGE_HASH, hash_size=8, _value=2**64)


def test_perceptual_hash_post_init_rejects_negative_value() -> None:
    with pytest.raises(ValueError):
        PerceptualHash(algorithm=PerceptualHashAlgorithm.AVERAGE_HASH, hash_size=8, _value=-1)


def test_perceptual_hash_post_init_rejects_bool_value() -> None:
    with pytest.raises(TypeError):
        PerceptualHash(algorithm=PerceptualHashAlgorithm.AVERAGE_HASH, hash_size=8, _value=True)  # type: ignore[arg-type]


def test_equality_and_hashability() -> None:
    a = PerceptualHash(algorithm=PerceptualHashAlgorithm.AVERAGE_HASH, hash_size=8, _value=42)
    b = PerceptualHash(algorithm=PerceptualHashAlgorithm.AVERAGE_HASH, hash_size=8, _value=42)
    c = PerceptualHash(algorithm=PerceptualHashAlgorithm.PHASH, hash_size=8, _value=42)
    d = PerceptualHash(algorithm=PerceptualHashAlgorithm.AVERAGE_HASH, hash_size=16, _value=42)

    assert a == b
    assert hash(a) == hash(b)
    assert a != c
    assert a != d
    assert {a, b, c, d} == {a, c, d}  # a and b collapse in a set


def test_distance_to_self_is_zero() -> None:
    rng = np.random.default_rng(16)
    img = rng.integers(0, 256, (32, 32), dtype=np.uint8)
    h = average_hash(img)

    assert h.distance(h) == 0


def test_distance_is_symmetric() -> None:
    rng = np.random.default_rng(17)
    a = average_hash(rng.integers(0, 256, (32, 32), dtype=np.uint8))
    b = average_hash(rng.integers(0, 256, (32, 32), dtype=np.uint8))

    assert a.distance(b) == b.distance(a)


def test_distance_known_values() -> None:
    a = PerceptualHash(algorithm=PerceptualHashAlgorithm.AVERAGE_HASH, hash_size=8, _value=0b0000)
    b = PerceptualHash(algorithm=PerceptualHashAlgorithm.AVERAGE_HASH, hash_size=8, _value=0b1010)

    assert a.distance(b) == 2


def test_distance_rejects_non_perceptual_hash() -> None:
    h = PerceptualHash(algorithm=PerceptualHashAlgorithm.AVERAGE_HASH, hash_size=8, _value=0)

    with pytest.raises(TypeError):
        h.distance("not a hash")  # type: ignore[arg-type]


def test_distance_rejects_different_algorithm() -> None:
    a = PerceptualHash(algorithm=PerceptualHashAlgorithm.AVERAGE_HASH, hash_size=8, _value=0)
    b = PerceptualHash(algorithm=PerceptualHashAlgorithm.PHASH, hash_size=8, _value=0)

    with pytest.raises(ValueError):
        a.distance(b)


def test_distance_rejects_different_hash_size() -> None:
    a = PerceptualHash(algorithm=PerceptualHashAlgorithm.AVERAGE_HASH, hash_size=8, _value=0)
    b = PerceptualHash(algorithm=PerceptualHashAlgorithm.AVERAGE_HASH, hash_size=16, _value=0)

    with pytest.raises(ValueError):
        a.distance(b)


def test_distance_same_bit_length_different_algorithm_still_rejected() -> None:
    # Same hash_size (and therefore same bit length) is not enough --
    # different algorithms are different, non-comparable feature spaces.
    a = PerceptualHash(algorithm=PerceptualHashAlgorithm.AVERAGE_HASH, hash_size=8, _value=0)
    b = PerceptualHash(algorithm=PerceptualHashAlgorithm.PHASH, hash_size=8, _value=0)

    assert len(a) == len(b)
    with pytest.raises(ValueError):
        a.distance(b)


# --- str() / hex serialization ---


def test_str_is_lowercase_fixed_width_hex() -> None:
    h = PerceptualHash(algorithm=PerceptualHashAlgorithm.AVERAGE_HASH, hash_size=8, _value=255)

    assert str(h) == "00000000000000ff"
    assert len(str(h)) == 16


def test_str_preserves_leading_zeros() -> None:
    h = PerceptualHash(algorithm=PerceptualHashAlgorithm.AVERAGE_HASH, hash_size=8, _value=1)

    assert str(h) == "0000000000000001"


def test_repr_is_readable() -> None:
    h = PerceptualHash(algorithm=PerceptualHashAlgorithm.AVERAGE_HASH, hash_size=8, _value=1)

    text = repr(h)
    assert "PerceptualHash" in text
    assert "average_hash" in text
    assert "8" in text
    assert "0000000000000001" in text


def test_from_hex_round_trip() -> None:
    rng = np.random.default_rng(18)
    img = rng.integers(0, 256, (32, 32), dtype=np.uint8)
    h = phash(img, hash_size=8)

    restored = PerceptualHash.from_hex(str(h), algorithm=PerceptualHashAlgorithm.PHASH, hash_size=8)

    assert restored == h


def test_from_hex_hash_size_3_full_9_bit_range() -> None:
    # hash_size=3 -> 9 bits -> ceil(9/4)=3 hex chars, capacity 12 bits --
    # the top 3 bits of that capacity must always be zero.
    max_valid = 2**9 - 1  # 0x1ff
    h = PerceptualHash.from_hex(
        format(max_valid, "03x"), algorithm=PerceptualHashAlgorithm.AVERAGE_HASH, hash_size=3
    )
    assert h._value == max_valid
    assert str(h) == "1ff"


def test_from_hex_rejects_invalid_padding_bits() -> None:
    # 0x200 has bit 9 set (value 512 >= 2**9=512), which is outside the
    # 9 valid bits for hash_size=3 -- must be rejected, not silently accepted.
    with pytest.raises(ValueError):
        PerceptualHash.from_hex("200", algorithm=PerceptualHashAlgorithm.AVERAGE_HASH, hash_size=3)


def test_from_hex_rejects_wrong_length() -> None:
    with pytest.raises(ValueError):
        PerceptualHash.from_hex(
            "ff", algorithm=PerceptualHashAlgorithm.AVERAGE_HASH, hash_size=8
        )  # 16 expected, got 2


def test_from_hex_rejects_0x_prefix() -> None:
    with pytest.raises(ValueError):
        PerceptualHash.from_hex(
            "0x00000000000000", algorithm=PerceptualHashAlgorithm.AVERAGE_HASH, hash_size=8
        )


def test_from_hex_rejects_whitespace() -> None:
    with pytest.raises(ValueError):
        PerceptualHash.from_hex(
            " 000000000000000", algorithm=PerceptualHashAlgorithm.AVERAGE_HASH, hash_size=8
        )


def test_from_hex_accepts_uppercase() -> None:
    h = PerceptualHash.from_hex(
        "00000000000000FF", algorithm=PerceptualHashAlgorithm.AVERAGE_HASH, hash_size=8
    )
    assert h._value == 255
    assert str(h) == "00000000000000ff"  # str() always lowercase, regardless of input case


def test_from_hex_rejects_non_hex_characters() -> None:
    with pytest.raises(ValueError):
        PerceptualHash.from_hex(
            "000000000000000g", algorithm=PerceptualHashAlgorithm.AVERAGE_HASH, hash_size=8
        )


def test_from_hex_rejects_non_str_text() -> None:
    with pytest.raises(TypeError):
        PerceptualHash.from_hex(12345, algorithm=PerceptualHashAlgorithm.AVERAGE_HASH, hash_size=8)  # type: ignore[arg-type]


def test_from_hex_rejects_bad_algorithm() -> None:
    with pytest.raises(TypeError):
        PerceptualHash.from_hex("00", algorithm="average_hash", hash_size=8)  # type: ignore[arg-type]


def test_from_hex_rejects_bad_hash_size() -> None:
    with pytest.raises(ValueError):
        PerceptualHash.from_hex("00", algorithm=PerceptualHashAlgorithm.AVERAGE_HASH, hash_size=1)


# --- hash_size=8 bit-grid parity with cv2.img_hash (reference: opencv-contrib-python
# 4.13.0.90 and 5.0.0.93, GNU Octave not involved here). Values below are the improcv
# hex output for hand-picked deterministic images, independently cross-checked in a
# throwaway venv against cv2.img_hash.AverageHash/PHash's raw packed-byte output
# (translated bit-by-bit, not byte-by-byte, since improcv's serialization order
# differs from cv2.img_hash's -- see PerceptualHash's docstring). ---


def test_average_hash_matches_opencv_bit_decisions_grayscale() -> None:
    rng = np.random.default_rng(90101)
    img = rng.integers(0, 256, (37, 41), dtype=np.uint8)

    assert str(average_hash(img, hash_size=8)) == "a6de3730d4ff5909"


def test_average_hash_matches_opencv_bit_decisions_bgr() -> None:
    rng = np.random.default_rng(90102)
    img = rng.integers(0, 256, (37, 41, 3), dtype=np.uint8)

    assert str(average_hash(img, hash_size=8)) == "3c1d94cf886d4f32"


def test_phash_matches_opencv_bit_decisions_grayscale() -> None:
    rng = np.random.default_rng(90103)
    img = rng.integers(0, 256, (37, 41), dtype=np.uint8)

    assert str(phash(img, hash_size=8)) == "ea5415bf9270a9b3"


def test_phash_matches_opencv_bit_decisions_bgra() -> None:
    rng = np.random.default_rng(90104)
    img = rng.integers(0, 256, (37, 41, 4), dtype=np.uint8)

    assert str(phash(img, hash_size=8)) == "47924ed66e35357a"


# --- serialization differs from cv2.img_hash's own packed-byte layout ---


def test_serialization_is_not_cv2_img_hash_packed_bytes() -> None:
    # For the same bit *decisions* (verified equal above), improcv's hex
    # string is not byte-equal to cv2.img_hash's raw packed-byte output --
    # improcv is MSB-first row-major over the whole value; cv2.img_hash packs
    # one 8-pixel row per byte, LSB-first within each byte.
    rng = np.random.default_rng(90105)
    img = rng.integers(0, 256, (32, 32), dtype=np.uint8)

    h = average_hash(img, hash_size=8)
    # cv2.img_hash's packed-byte convention for the same bit grid: reverse
    # the bit order within each of the 8 row-bytes.
    value = h._value
    row_bytes = [(value >> (8 * (7 - r))) & 0xFF for r in range(8)]
    cv2_style_bytes = [int(f"{b:08b}"[::-1], 2) for b in row_bytes]
    cv2_style_hex = "".join(f"{b:02x}" for b in cv2_style_bytes)

    assert str(h) != cv2_style_hex


# --- OpenCV 4/5 consistency is exercised implicitly: this whole file is run
# against both the project's OpenCV 5.0.0 dev environment and a pinned
# OpenCV 4.13.0 scratchpad venv as part of the standard verification
# procedure (see the PR description) -- no separate skip-conditional test
# is needed since the algorithm has no version-dependent branch.


def test_public_exports() -> None:
    assert im.average_hash is average_hash
    assert im.phash is phash
    assert im.PerceptualHash is PerceptualHash
    assert im.PerceptualHashAlgorithm is PerceptualHashAlgorithm
