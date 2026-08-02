"""Generate the image similarity demo PNG.

Builds four small, deterministic 8x8 grayscale images (A, B, C, D -- see
`build_similarity_inputs`), hashes each one once with `improcv.average_hash`, then computes the
full pairwise Hamming-distance matrix and the real `improcv.find_similar_image_pairs` result at a
fixed threshold (`compute_similarity_report`). Every number shown in the rendered figure -- every
hash, every matrix cell, every reported pair -- comes from those two real API calls; nothing here
is hand-typed into the figure.

A, B, and C are close variants of one another (single-pixel-pair swaps); D is unrelated. At
`max_distance=2`, A-B and B-C both match but A-C (distance 4) does not -- the figure's own point is
that `find_similar_image_pairs` reports this as two independent pairs, never one merged group of
three: threshold similarity is not transitive.

This script does not share its image-building code with `examples/image_similarity.py` -- both
independently construct the same four logical images so each stays self-contained and either can
be read on its own.

This script is the source of truth for `docs/assets/image-similarity-gallery.png`; the PNG must
never be edited by hand. Run with:

    uv run python demos/image_similarity_gallery.py

See `demos/README.md` for the full regeneration contract.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

import improcv as im

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.gridspec import GridSpecFromSubplotSpec

_FIGURE_SIZE_INCHES = (12.8, 7.2)
_FIGURE_DPI = 150

_NEUTRAL_ACCENT = "#4C72B0"
_HIT_ACCENT = "#009E73"
_UNRELATED_ACCENT = "#D55E00"

_NAMES = ("a_base.png", "b_variant.png", "c_variant_more.png", "z_unrelated.png")
_EXPECTED_HEX = (
    "55aa55aa55aa55aa",
    "95aa55aa55aa55aa",
    "956a55aa55aa55aa",
    "0f0f0f0f0f0f0f0f",
)
_HASH_SIZE = 8
_DEFAULT_THRESHOLD = 2


class SimilarityInputs(NamedTuple):
    """The four synthetic images and their already-computed `average_hash` hashes."""

    names: tuple[str, ...]
    images: tuple[np.ndarray, ...]
    hashes: tuple[im.PerceptualHash, ...]


class SimilarityReport(NamedTuple):
    """The real distance matrix and `find_similar_image_pairs` result for `SimilarityInputs`."""

    distances: np.ndarray
    pairs: tuple[im.SimilarImagePair, ...]
    threshold: int


def build_similarity_inputs() -> SimilarityInputs:
    """Build the four deterministic 8x8 uint8 grayscale images and hash each one once.

    - ``a_base.png``: a checkerboard, ``base[y, x] = 255 if (x + y) % 2 else 0``.
    - ``b_variant.png``: a copy of A with the pixels at (row 0, col 0) and (row 0, col 1) swapped.
    - ``c_variant_more.png``: a copy of B with an additional swap at (row 1, col 0)/(row 1, col 1).
    - ``z_unrelated.png``: a vertical split -- columns 0-3 are 0, columns 4-7 are 255.

    Each image has exactly 32 white (255) and 32 black (0) pixels.
    """
    row, col = np.indices((8, 8))
    base = np.where((col + row) % 2 == 0, 0, 255).astype(np.uint8)

    variant = base.copy()
    variant[0, 0], variant[0, 1] = variant[0, 1], variant[0, 0]

    variant_more = variant.copy()
    variant_more[1, 0], variant_more[1, 1] = variant_more[1, 1], variant_more[1, 0]

    unrelated = np.zeros((8, 8), dtype=np.uint8)
    unrelated[:, 4:] = 255

    images = (base, variant, variant_more, unrelated)
    for image in images:
        assert image.shape == (8, 8), image.shape
        assert image.dtype == np.uint8, image.dtype
        assert int(np.count_nonzero(image == 255)) == 32, int(np.count_nonzero(image == 255))

    hashes = tuple(im.average_hash(image, hash_size=_HASH_SIZE) for image in images)
    for name, hash_value, expected_hex in zip(_NAMES, hashes, _EXPECTED_HEX, strict=True):
        assert str(hash_value) == expected_hex, (name, str(hash_value))
        assert hash_value.algorithm == im.PerceptualHashAlgorithm.AVERAGE_HASH
        assert hash_value.hash_size == _HASH_SIZE

    return SimilarityInputs(names=_NAMES, images=images, hashes=hashes)


def compute_similarity_report(
    inputs: SimilarityInputs, *, threshold: int = _DEFAULT_THRESHOLD
) -> SimilarityReport:
    """Compute the full pairwise distance matrix and the real similar-pairs result for `inputs`."""
    count = len(inputs.names)
    distances = np.zeros((count, count), dtype=np.int64)
    for row in range(count):
        for col in range(count):
            distances[row, col] = inputs.hashes[row].distance(inputs.hashes[col])

    assert distances.shape == (count, count)
    assert np.array_equal(distances, distances.T), "distance matrix must be symmetric"
    assert np.all(np.diag(distances) == 0), "diagonal must be zero"

    index = {name: position for position, name in enumerate(inputs.names)}
    distance_ab = int(distances[index["a_base.png"], index["b_variant.png"]])
    distance_bc = int(distances[index["b_variant.png"], index["c_variant_more.png"]])
    distance_ac = int(distances[index["a_base.png"], index["c_variant_more.png"]])
    assert distance_ab == 2, distance_ab
    assert distance_bc == 2, distance_bc
    assert distance_ac == 4, distance_ac
    for other in ("a_base.png", "b_variant.png", "c_variant_more.png"):
        distance_to_unrelated = int(distances[index["z_unrelated.png"], index[other]])
        assert distance_to_unrelated > 2, distance_to_unrelated

    hashes_by_name: dict[str | os.PathLike[str], im.PerceptualHash] = dict(
        zip(inputs.names, inputs.hashes, strict=True)
    )
    pairs = im.find_similar_image_pairs(hashes_by_name, max_distance=threshold)
    found = tuple((pair.first.name, pair.second.name, pair.distance) for pair in pairs)
    expected = (
        ("a_base.png", "b_variant.png", 2),
        ("b_variant.png", "c_variant_more.png", 2),
    )
    assert found == expected, found

    return SimilarityReport(distances=distances, pairs=pairs, threshold=threshold)


def _find_pair(
    pairs: tuple[im.SimilarImagePair, ...], name_a: str, name_b: str
) -> im.SimilarImagePair | None:
    """Return the real pair matching the unordered `{name_a, name_b}`, or `None` if absent."""
    key_a, key_b = sorted((name_a, name_b))
    for pair in pairs:
        if pair.first.name == key_a and pair.second.name == key_b:
            return pair
    return None


def _draw_images_row(
    fig: Figure, grid_spec: GridSpecFromSubplotSpec, inputs: SimilarityInputs
) -> None:
    for position, (name, image, hash_value) in enumerate(
        zip(inputs.names, inputs.images, inputs.hashes, strict=True)
    ):
        ax = fig.add_subplot(grid_spec[0, position])
        ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(f"{name}\n{hash_value}\naverage_hash, 64 bits", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        accent = _UNRELATED_ACCENT if name == "z_unrelated.png" else _NEUTRAL_ACCENT
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(accent)
            spine.set_linewidth(1.5)


def _draw_distance_matrix(ax: Axes, inputs: SimilarityInputs, report: SimilarityReport) -> None:
    from matplotlib.patches import Rectangle

    count = len(inputs.names)
    vmax = max(int(report.distances.max()), 1)
    ax.imshow(report.distances, cmap="YlOrRd", vmin=0, vmax=vmax)
    ax.set_xticks(range(count))
    ax.set_yticks(range(count))
    ax.set_xticklabels(inputs.names, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(inputs.names, fontsize=8)
    ax.set_title("Hamming distance matrix (average_hash, 64 bits)", fontsize=10, loc="left")

    for row in range(count):
        for col in range(count):
            value = int(report.distances[row, col])
            ax.text(
                col,
                row,
                str(value),
                ha="center",
                va="center",
                fontsize=9,
                color="black",
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.0},
            )
            if row != col and value <= report.threshold:
                ax.add_patch(
                    Rectangle(
                        (col - 0.5, row - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor=_HIT_ACCENT,
                        linewidth=2.5,
                    )
                )


def _draw_pair_result(ax: Axes, inputs: SimilarityInputs, report: SimilarityReport) -> None:
    from matplotlib.patches import Rectangle

    ax.axis("off")
    index = {name: position for position, name in enumerate(inputs.names)}
    ab_pair = _find_pair(report.pairs, "a_base.png", "b_variant.png")
    bc_pair = _find_pair(report.pairs, "b_variant.png", "c_variant_more.png")
    assert ab_pair is not None
    assert bc_pair is not None
    distance_ac = int(report.distances[index["a_base.png"], index["c_variant_more.png"]])

    lines = [f"max_distance = {report.threshold}", f"pairs found: {len(report.pairs)}", ""]
    for pair in report.pairs:
        lines.append(f"{pair.distance}  {pair.first.name} <-> {pair.second.name}")
    lines += [
        "",
        f"a_base.png <-> b_variant.png: {ab_pair.distance} <= {report.threshold} -> match",
        f"b_variant.png <-> c_variant_more.png: {bc_pair.distance} <= {report.threshold} -> match",
        f"a_base.png <-> c_variant_more.png: {distance_ac} > {report.threshold} -> no match",
        "",
        "find_similar_image_pairs returns pairs, not duplicate groups:",
        "A~B and B~C does not imply A~C",
        "(non-transitive threshold similarity).",
    ]

    ax.text(
        0.03,
        0.95,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=9,
        fontfamily="monospace",
        linespacing=1.4,
        va="top",
        ha="left",
    )
    ax.set_title("find_similar_image_pairs (public API)", fontsize=10, loc="left")
    border = Rectangle(
        (0.0, 0.0),
        1.0,
        1.0,
        transform=ax.transAxes,
        fill=False,
        edgecolor=_NEUTRAL_ACCENT,
        linewidth=1.5,
    )
    ax.add_patch(border)


def build_similarity_figure(inputs: SimilarityInputs, report: SimilarityReport) -> Figure:
    """Build the image-similarity gallery figure without saving or closing it."""
    import matplotlib

    matplotlib.use("Agg", force=True)

    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=_FIGURE_SIZE_INCHES, dpi=_FIGURE_DPI)
    outer = fig.add_gridspec(
        2, 1, left=0.04, right=0.97, top=0.86, bottom=0.12, height_ratios=(0.38, 0.62), hspace=0.35
    )

    image_grid = outer[0].subgridspec(1, 4, wspace=0.3)
    _draw_images_row(fig, image_grid, inputs)

    lower_grid = outer[1].subgridspec(1, 2, width_ratios=(0.55, 0.45), wspace=0.3)
    matrix_ax = fig.add_subplot(lower_grid[0, 0])
    _draw_distance_matrix(matrix_ax, inputs, report)
    result_ax = fig.add_subplot(lower_grid[0, 1])
    _draw_pair_result(result_ax, inputs, report)

    fig.suptitle(
        "improcv image similarity: average_hash distances and threshold pairing", fontsize=13
    )

    return fig


def render_similarity_gallery(
    inputs: SimilarityInputs, report: SimilarityReport, output: Path
) -> tuple[int, int]:
    """Render the image-similarity gallery layout to `output` as a PNG.

    Returns the written image's `(width, height)` in pixels.
    """
    fig = build_similarity_figure(inputs, report)

    from matplotlib import pyplot as plt

    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=_FIGURE_DPI)
    finally:
        plt.close(fig)

    written_width = round(_FIGURE_SIZE_INCHES[0] * _FIGURE_DPI)
    written_height = round(_FIGURE_SIZE_INCHES[1] * _FIGURE_DPI)
    return written_width, written_height


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the generator.

    `--output` must end in `.png` (case-insensitive); this is checked here, via
    `parser.error` (standard argparse exit code `2`), before any work happens.
    """
    parser = argparse.ArgumentParser(description="Generate the improcv image similarity demo PNG.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/assets/image-similarity-gallery.png"),
        help="Output PNG path (default: docs/assets/image-similarity-gallery.png).",
    )
    args = parser.parse_args(argv)
    if args.output.suffix.lower() != ".png":
        parser.error(f"--output must end in .png, got {args.output}")
    return args


def main() -> None:
    args = parse_args()

    inputs = build_similarity_inputs()
    report = compute_similarity_report(inputs, threshold=_DEFAULT_THRESHOLD)

    try:
        width, height = render_similarity_gallery(inputs, report, args.output)
    except OSError as error:
        print(f"error: could not write {args.output}: {error}", file=sys.stderr)
        raise SystemExit(1) from None

    print(f"wrote {args.output} ({width}x{height})")


if __name__ == "__main__":
    main()
