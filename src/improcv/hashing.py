"""Perceptual image hashing: average hash and DCT-based pHash."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import cv2
import numpy as np

from improcv._validation import require_dtype, require_image_ndim, require_int
from improcv.types import ImageU8

__all__ = [
    "average_hash",
    "phash",
    "PerceptualHash",
    "PerceptualHashAlgorithm",
]

_HASH_SIZE_MIN = 2
_HASH_SIZE_MAX = 256
# Fixed, not a public parameter in this first slice: matches both cv2.img_hash's
# own hardcoded 32x32 DCT input (hash_size=8 * 4) and ImageHash's default. See
# PHash's docstring for why this can't be safely exposed yet (cv2.dct requires
# an even-sized input; a non-fixed, user-supplied highfreq_factor could break
# that invariant for some hash_size values).
_PHASH_HIGHFREQ_FACTOR = 4


def _require_valid_hash_size(hash_size: object, name: str = "hash_size") -> None:
    """Raise TypeError unless `hash_size` is a plain int, then ValueError unless
    it's within [2, 256].

    The lower bound excludes a degenerate 1x1 hash: verified directly that both
    `average_hash` and `phash` always produce an all-zero hash at `hash_size=1`
    regardless of image content (a single-element mean/DCT-mean-block has no
    discriminative power). The upper bound is an explicit resource-usage guard,
    particularly for `phash`, whose DCT input is `(hash_size * 4)` per side.
    """
    require_int(hash_size, name)
    assert isinstance(hash_size, int)  # narrows for the type checker
    if not (_HASH_SIZE_MIN <= hash_size <= _HASH_SIZE_MAX):
        raise ValueError(
            f"{name} must be between {_HASH_SIZE_MIN} and {_HASH_SIZE_MAX}, got {hash_size}"
        )


def _require_valid_hash_image(image: np.ndarray) -> None:
    """Raise ValueError/TypeError unless `image` is a valid grayscale/BGR/BGRA `uint8` image.

    Accepts 2D grayscale or 3D with exactly 1, 3, or 4 channels -- a 2-channel
    image has no defined grayscale conversion and is rejected. Only `uint8` is
    supported in this first slice: OpenCV's own reference hashing
    implementations (`cv2.img_hash`) operate exclusively on 8-bit data, and
    generalizing to `uint16`/`float32`/`float64` would need its own
    normalization/rounding contract -- deferred to a later slice.
    """
    require_image_ndim(image, ndims=(2, 3))
    require_dtype(image, (np.uint8,), "image")
    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise ValueError(
            f"image must have 1, 3, or 4 channels (grayscale, BGR, or BGRA), got {image.shape[2]}"
        )


def _resize_then_grayscale(image: np.ndarray, size: int) -> np.ndarray:
    """Resize `image` to `(size, size)`, converting BGR/BGRA to grayscale *after* resizing.

    This order (resize first, grayscale second) is not arbitrary: it matches
    `cv2.img_hash`'s own pipeline and is required for bit-for-bit compatibility
    with it on color input -- converting to grayscale *before* resizing
    (e.g. via `improcv.ensure_gray`) does not reproduce the same result, since
    `uint8` rounding at each stage does not commute (verified directly: up to
    ~28% of random BGR/BGRA images differ by at least one bit under the
    opposite ordering). For BGRA input, the alpha channel is ignored
    entirely -- not composited against any background -- so the BGR values of
    a fully transparent pixel still influence the hash.
    """
    working = image[:, :, 0] if image.ndim == 3 and image.shape[2] == 1 else image
    resized = cv2.resize(working, (size, size), interpolation=cv2.INTER_LINEAR_EXACT)
    if resized.ndim == 3:
        return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return resized


def _bits_to_value(bits: np.ndarray) -> int:
    """Pack a row-major boolean grid into a single int: the first element (top-left,
    reading left-to-right then top-to-bottom) becomes the most significant bit.

    This is improcv's own serialization contract -- it matches neither
    `cv2.img_hash`'s packed-byte layout (LSB-first per byte, one 8-pixel image
    row per byte) nor any third-party library's convention. See `PerceptualHash`
    for the full contract.
    """
    bit_string = "".join("1" if b else "0" for b in bits.flatten())
    return int(bit_string, 2)


class PerceptualHashAlgorithm(StrEnum):
    """Which perceptual hashing algorithm produced a `PerceptualHash`."""

    AVERAGE_HASH = "average_hash"
    PHASH = "phash"


@dataclass(frozen=True, slots=True)
class PerceptualHash:
    """An immutable perceptual hash value, as produced by `average_hash`/`phash`.

    Comparable (`==`, `distance`) only to another `PerceptualHash` with the same
    `algorithm` and `hash_size` -- two hashes of the same bit length but
    different algorithms (or different `hash_size`) represent different,
    non-comparable feature spaces, even though nothing about their bit length
    alone would reveal that.

    Bit/hex serialization is improcv's own contract, not a reproduction of any
    third-party library's: the hash's `hash_size x hash_size` bit grid is
    read row-major (top-to-bottom, left-to-right), with the first bit as the
    most significant bit of both the internal value and the `str()` hex
    representation. For `uint8` input and `hash_size=8`, the underlying bit
    *decisions* (which pixel/DCT coefficient is "above" the threshold) match
    `cv2.img_hash.AverageHash`/`PHash` exactly -- but the serialized hex string
    does not match `cv2.img_hash`'s own packed-byte layout, and raw byte
    import/export is not offered.

    A perceptual hash is not a cryptographic hash: collisions (two visually
    different images producing the same or a very similar hash) are expected
    and are not a defect. A smaller `distance()` usually means more visually
    similar images, but no fixed threshold is universally correct -- an
    appropriate cutoff depends on the application. Other libraries (e.g.
    `ImageHash`) implement genuinely different, bit-incompatible variants of
    both average hash and pHash -- do not assume hash values are portable
    across libraries.
    """

    algorithm: PerceptualHashAlgorithm
    hash_size: int
    _value: int = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.algorithm, PerceptualHashAlgorithm):
            raise TypeError(
                f"algorithm must be a PerceptualHashAlgorithm, got {type(self.algorithm).__name__}"
            )
        _require_valid_hash_size(self.hash_size)
        require_int(self._value, "value")
        bit_count = self.hash_size**2
        if not (0 <= self._value < 2**bit_count):
            raise ValueError(
                f"value must satisfy 0 <= value < 2**{bit_count} for hash_size="
                f"{self.hash_size}, got {self._value}"
            )

    def __len__(self) -> int:
        """The number of bits in this hash (``hash_size ** 2``)."""
        return self.hash_size**2

    def __str__(self) -> str:
        """Fixed-width, lowercase, zero-padded hex -- width is always ``ceil(hash_size**2 / 4)``."""
        width = -(-len(self) // 4)
        return format(self._value, f"0{width}x")

    def __repr__(self) -> str:
        return (
            f"PerceptualHash(algorithm={self.algorithm!r}, hash_size={self.hash_size}, "
            f"value='{self}')"
        )

    def distance(self, other: PerceptualHash) -> int:
        """Compute the Hamming distance to `other`.

        Raises
        ------
        TypeError
            If `other` is not a `PerceptualHash`.
        ValueError
            If `other` has a different `algorithm` or `hash_size` -- these are
            different, non-comparable feature spaces even when the resulting
            bit length happens to match.
        """
        if not isinstance(other, PerceptualHash):
            raise TypeError(f"other must be a PerceptualHash, got {type(other).__name__}")
        if self.algorithm != other.algorithm or self.hash_size != other.hash_size:
            raise ValueError(
                f"cannot compare {self.algorithm.value}(hash_size={self.hash_size}) with "
                f"{other.algorithm.value}(hash_size={other.hash_size}) -- distance requires "
                "the same algorithm and hash_size"
            )
        return (self._value ^ other._value).bit_count()

    @classmethod
    def from_hex(
        cls,
        text: str,
        *,
        algorithm: PerceptualHashAlgorithm,
        hash_size: int = 8,
    ) -> PerceptualHash:
        """Parse a `PerceptualHash` from the exact hex format produced by `str()`.

        Parameters
        ----------
        text : str
            Exactly ``ceil(hash_size**2 / 4)`` hex characters (lowercase or
            uppercase), no ``0x`` prefix, no whitespace.
        algorithm : PerceptualHashAlgorithm
            Must be supplied explicitly -- a hex string alone cannot reveal
            which algorithm produced it.
        hash_size : int, optional
            Must match the `hash_size` the hash was produced with. Defaults
            to `8`, matching `average_hash`/`phash`'s own default.

        Raises
        ------
        TypeError
            If `algorithm` is not a `PerceptualHashAlgorithm`, `hash_size` is
            not a plain int, or `text` is not a `str`.
        ValueError
            If `hash_size` is out of range, `text` is not exactly the
            expected length, contains non-hex characters, or encodes a value
            with nonzero bits above `hash_size**2` (invalid padding).
        """
        if not isinstance(algorithm, PerceptualHashAlgorithm):
            raise TypeError(
                f"algorithm must be a PerceptualHashAlgorithm, got {type(algorithm).__name__}"
            )
        _require_valid_hash_size(hash_size)
        if not isinstance(text, str):
            raise TypeError(f"text must be a str, got {type(text).__name__}")

        bit_count = hash_size**2
        width = -(-bit_count // 4)
        if len(text) != width:
            raise ValueError(
                f"text must be exactly {width} hex characters for hash_size={hash_size}, "
                f"got {len(text)}"
            )
        if not all(c in "0123456789abcdefABCDEF" for c in text):
            raise ValueError(
                f"text must contain only hex digits (no '0x', no whitespace): {text!r}"
            )

        value = int(text, 16)
        if value >= 2**bit_count:
            raise ValueError(
                f"text encodes a value with nonzero bits above hash_size**2={bit_count} bits "
                "-- invalid padding"
            )
        return cls(algorithm=algorithm, hash_size=hash_size, _value=value)


def average_hash(image: ImageU8, hash_size: int = 8) -> PerceptualHash:
    """Compute the average hash of `image`.

    Algorithm (matching `cv2.img_hash.AverageHash`'s own bit decisions for
    `uint8` input at `hash_size=8`): resize to ``hash_size x hash_size`` with
    `cv2.INTER_LINEAR_EXACT`, convert BGR/BGRA to grayscale *after* resizing
    (see `_resize_then_grayscale`), compute the mean pixel value, round it to
    the nearest integer with round-half-to-even (matching `cv::cvRound`), and
    set each bit to whether the corresponding (still-`uint8`) pixel is
    strictly greater than that rounded threshold.

    Parameters
    ----------
    image : np.ndarray
        A `uint8` image: grayscale (``(H, W)``), ``(H, W, 1)``, BGR
        (``(H, W, 3)``), or BGRA (``(H, W, 4)``). For BGRA, the alpha channel
        is ignored entirely (not composited) -- the BGR values of a fully
        transparent pixel still influence the hash. Not modified.
    hash_size : int, optional
        Side length of the hash grid; the result has ``hash_size ** 2`` bits.
        Must be between 2 and 256. Default `8`, matching `cv2.img_hash`.

    Returns
    -------
    PerceptualHash
        `algorithm` is `PerceptualHashAlgorithm.AVERAGE_HASH`.

    Raises
    ------
    ValueError
        If `image` is empty, doesn't have 2 or 3 dimensions, has a channel
        count other than 1, 3, or 4, or `hash_size` is out of range.
    TypeError
        If `image` doesn't have dtype ``uint8``, or `hash_size` isn't a
        plain int.
    """
    _require_valid_hash_image(image)
    _require_valid_hash_size(hash_size)

    gray = _resize_then_grayscale(image, hash_size)
    mean = cv2.mean(gray)[0]
    threshold = round(mean)
    bits = gray > threshold

    value = _bits_to_value(bits)
    return PerceptualHash(
        algorithm=PerceptualHashAlgorithm.AVERAGE_HASH, hash_size=hash_size, _value=value
    )


def phash(image: ImageU8, hash_size: int = 8) -> PerceptualHash:
    """Compute the DCT-based perceptual hash of `image`.

    Algorithm (matching `cv2.img_hash.PHash`'s own bit decisions for `uint8`
    input at `hash_size=8`): resize to ``(hash_size * 4) x (hash_size * 4)``
    with `cv2.INTER_LINEAR_EXACT`, convert BGR/BGRA to grayscale *after*
    resizing (see `_resize_then_grayscale`), convert to `float32`, compute the
    2D DCT (`cv2.dct`), take the top-left ``hash_size x hash_size`` block, zero
    its DC term (position ``[0, 0]``), compute the mean of that block as a
    `float64`, cast the mean to `float32` (matching the reference
    implementation's own `float` storage), and set each bit to whether the
    corresponding block value is strictly greater than that `float32`
    threshold. The DC position still gets an output bit, from comparing its
    forced-zero value against the threshold.

    This is one of several genuinely different, non-interchangeable variants
    of "pHash" in circulation -- notably not the same as `ImageHash.phash`
    (which uses the median of the *unmodified* block, DC included) or the
    original hackerfactor.com blog description (mean of the 63 AC
    coefficients, DC excluded from both the sum and the divisor). This
    function reproduces `cv2.img_hash.PHash` specifically.

    `highfreq_factor` (the resize-to-hash_size ratio) is fixed at `4` and is
    not a public parameter in this first slice.

    Parameters
    ----------
    image : np.ndarray
        A `uint8` image: grayscale (``(H, W)``), ``(H, W, 1)``, BGR
        (``(H, W, 3)``), or BGRA (``(H, W, 4)``). For BGRA, the alpha channel
        is ignored entirely (not composited). Not modified.
    hash_size : int, optional
        Side length of the hash grid; the result has ``hash_size ** 2`` bits.
        Must be between 2 and 256 (the DCT input is ``(hash_size * 4)`` per
        side). Default `8`, matching `cv2.img_hash`.

    Returns
    -------
    PerceptualHash
        `algorithm` is `PerceptualHashAlgorithm.PHASH`.

    Raises
    ------
    ValueError
        If `image` is empty, doesn't have 2 or 3 dimensions, has a channel
        count other than 1, 3, or 4, or `hash_size` is out of range.
    TypeError
        If `image` doesn't have dtype ``uint8``, or `hash_size` isn't a
        plain int.
    """
    _require_valid_hash_image(image)
    _require_valid_hash_size(hash_size)

    dct_size = hash_size * _PHASH_HIGHFREQ_FACTOR
    gray = _resize_then_grayscale(image, dct_size)

    grayf = gray.astype(np.float32)
    dct = cv2.dct(grayf)
    block = dct[:hash_size, :hash_size].copy()
    block[0, 0] = 0.0
    # cv::mean always accumulates in double, but the reference implementation
    # stores the result back into a C++ float before comparing -- this cast
    # is part of bit-for-bit compatibility with it, not incidental precision
    # loss: see tests/test_hashing.py's dedicated regression test, which
    # constructs a block where comparing against the uncast float64 mean
    # flips a bit relative to comparing against this float32-cast one.
    threshold = np.float32(np.mean(block, dtype=np.float64))
    bits = block > threshold

    value = _bits_to_value(bits)
    return PerceptualHash(
        algorithm=PerceptualHashAlgorithm.PHASH, hash_size=hash_size, _value=value
    )
