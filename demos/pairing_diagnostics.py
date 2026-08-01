"""Generate the pairing diagnostics demo PNG.

Builds three tiny synthetic datasets under one shared temporary directory and
pairs each with `improcv.discover_image_mask_pairs`, comparing the real
result against a naive positional `zip(sorted(images), sorted(masks))`:

- a success case where both approaches happen to agree;
- a missing-pair case where the naive zip silently produces a wrong pair,
  while `discover_image_mask_pairs` raises a `ValueError` naming exactly
  which keys are unmatched;
- a duplicate-pairing-key case where two different image files reduce to the
  same key, and `discover_image_mask_pairs` raises a `ValueError` naming the
  colliding key and both competing paths.

Every directory tree, naive-zip result, and exception message shown in the
rendered image is computed from a real filesystem and a real call to
`improcv`'s public API -- nothing here is hand-typed.

This script is the source of truth for `docs/assets/pairing-diagnostics.png`;
the PNG must never be edited by hand. Run with:

    uv run python demos/pairing_diagnostics.py

See `demos/README.md` for the full regeneration contract.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import cv2
import numpy as np

import improcv as im

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.backend_bases import RendererBase
    from matplotlib.figure import Figure
    from matplotlib.text import Text

_FIGURE_SIZE_INCHES = (12.8, 7.2)
_FIGURE_DPI = 150

# The missing-pair scenario's real diagnostic (naive zip + a real ValueError, see
# build_missing_pair_scenario) needs noticeably more lines than the other two rows; giving it a
# taller share of the figure is what keeps its final lines from overflowing past the panel.
_ROW_HEIGHT_RATIOS = (0.9, 1.3, 0.9)

_TREE_ACCENT = "#4C72B0"
_SUCCESS_ACCENT = "#009E73"
_SILENT_ACCENT = "#E69F00"
_REJECTED_ACCENT = "#D55E00"

_MAX_DIAGNOSTIC_LINES = 8

# A panel with more than this many lines gets a slightly smaller font (still legible at an
# 880px README width) instead of being allowed to overflow its axes.
_COMPACT_FONT_LINE_THRESHOLD = 8
_DEFAULT_FONT_SIZE = 8.0
_COMPACT_FONT_SIZE = 7.0

# Minimum on-screen gap (in pixels, at the figure's own render resolution) required between a
# panel's text bounding box and its axes' bounding box on every side; see find_text_overflow.
_OVERFLOW_MARGIN_PX = 4.0


class Scenario(NamedTuple):
    """Everything `render_diagnostics` needs to draw one row of the layout."""

    title: str
    tree_text: str
    result_text: str
    accent: str
    raw_diagnostic: str | None = None


def _write_tiny_image(path: Path) -> None:
    """Write a small, valid, single-channel uint8 PNG/JPEG -- content is irrelevant to pairing."""
    image = np.zeros((8, 8), dtype=np.uint8)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise RuntimeError(f"internal error: cv2.imwrite failed for {path}")


def render_directory_tree(root: Path) -> str:
    """Render `root`'s contents as a stable, sorted, POSIX-style indented tree.

    Built from an actual `Path.iterdir()` walk (never hand-typed), listing only entries
    that exist under `root`; only each entry's own name is used, never an absolute path,
    so no temporary-directory prefix can leak into the result.
    """

    def _walk(directory: Path, prefix: str) -> list[str]:
        lines: list[str] = []
        for entry in sorted(directory.iterdir(), key=lambda entry: entry.name):
            if entry.is_dir():
                lines.append(f"{prefix}{entry.name}/")
                lines.extend(_walk(entry, prefix + "  "))
            else:
                lines.append(f"{prefix}{entry.name}")
        return lines

    return "\n".join([f"{root.name}/", *_walk(root, "  ")])


def _normalize_diagnostic_text(text: str, dataset_root: Path) -> str:
    """Replace this scenario's own absolute dataset root with `dataset`, for rendering only.

    The original exception text is never modified -- this produces a separate copy used
    solely for the figure. Never touches pairing keys or filenames themselves, only a
    literal occurrence of the absolute root path, if present.
    """
    normalized = text.replace(str(dataset_root), "dataset")
    normalized = normalized.replace(dataset_root.as_posix(), "dataset")
    return normalized.replace("\\", "/")


def _truncate_lines(text: str, max_lines: int = _MAX_DIAGNOSTIC_LINES) -> str:
    """Cap a diagnostic message at `max_lines`, summarizing the rest by count."""
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    remaining = len(lines) - max_lines
    return "\n".join(lines[:max_lines]) + f"\n... ({remaining} more lines)"


def _pairing_key(pair: im.ImageMaskPair, image_root: Path) -> str:
    """The display key for `pair`, derived from its actual `image` path, not hand-typed."""
    relative = pair.image.relative_to(image_root).as_posix()
    return relative.rsplit(".", 1)[0]


def build_success_scenario(root: Path) -> Scenario:
    """A dataset where every image has exactly one matching mask."""
    dataset = root / "success" / "dataset"
    images_train = dataset / "images" / "train"
    masks_train = dataset / "masks" / "train"
    images_train.mkdir(parents=True)
    masks_train.mkdir(parents=True)

    for name in ("cat_001", "dog_001"):
        _write_tiny_image(images_train / f"{name}.jpg")
        _write_tiny_image(masks_train / f"{name}.png")

    image_root = dataset / "images"
    pairs = im.discover_image_mask_pairs(image_root, dataset / "masks")
    assert len(pairs) == 2, f"expected 2 pairs, got {len(pairs)}"

    keys = [_pairing_key(pair, image_root) for pair in pairs]
    assert keys == sorted(keys), "pairs must come back in deterministic sorted order"
    assert keys == ["train/cat_001", "train/dog_001"], keys

    result_lines = ["PAIRED", "", f"paired: {len(pairs)}", *keys]

    return Scenario(
        title="1. Deterministic success",
        tree_text=render_directory_tree(dataset),
        result_text="\n".join(result_lines),
        accent=_SUCCESS_ACCENT,
    )


def build_missing_pair_scenario(root: Path) -> Scenario:
    """A dataset with one unmatched image and one unmatched mask.

    Shows a real naive `zip(sorted(images), sorted(masks))` silently producing a wrong
    pair next to the real `ValueError` `discover_image_mask_pairs` raises instead.
    """
    dataset = root / "missing" / "dataset"
    images_train = dataset / "images" / "train"
    masks_train = dataset / "masks" / "train"
    images_train.mkdir(parents=True)
    masks_train.mkdir(parents=True)

    _write_tiny_image(images_train / "cat_001.jpg")
    _write_tiny_image(images_train / "lonely.jpg")
    _write_tiny_image(masks_train / "cat_001.png")
    _write_tiny_image(masks_train / "orphan.png")

    # The naive approach: sort filenames independently on each side and zip by position.
    # Never fed into any further pairing logic -- shown only to demonstrate the risk.
    image_names = sorted(entry.name for entry in images_train.iterdir())
    mask_names = sorted(entry.name for entry in masks_train.iterdir())
    naive_pairs = list(zip(image_names, mask_names, strict=True))
    assert naive_pairs == [("cat_001.jpg", "cat_001.png"), ("lonely.jpg", "orphan.png")]

    naive_lines = ["naive positional zip:"]
    for image_name, mask_name in naive_pairs:
        matches = Path(image_name).stem == Path(mask_name).stem
        marker = "" if matches else "   <- WRONG BUT SILENT"
        naive_lines.append(f"{image_name} <-> {mask_name}{marker}")

    try:
        im.discover_image_mask_pairs(dataset / "images", dataset / "masks")
    except ValueError as error:
        raw_message = str(error)
    else:
        raise AssertionError("expected discover_image_mask_pairs to reject a missing pair")

    assert "lonely" in raw_message, raw_message
    assert "orphan" in raw_message, raw_message

    # The raw message's own leading "image and mask pairing keys do not match:" header is
    # redundant with the "discover_image_mask_pairs: REJECTED" line already shown below it --
    # this keeps an exact, contiguous slice of the real message starting at the first line that
    # actually names an unmatched key, never a paraphrase or a hand-typed replacement.
    fragment_start = "image keys without a matching mask:"
    assert fragment_start in raw_message, raw_message
    real_fragment = raw_message[raw_message.index(fragment_start) :]

    preview = _truncate_lines(_normalize_diagnostic_text(real_fragment, dataset))
    result_lines = [
        *naive_lines,
        "",
        "discover_image_mask_pairs: REJECTED",
        preview,
    ]

    return Scenario(
        title="2. Silent positional mispair avoided",
        tree_text=render_directory_tree(dataset),
        result_text="\n".join(result_lines),
        accent=_SILENT_ACCENT,
        raw_diagnostic=raw_message,
    )


def build_duplicate_key_scenario(root: Path) -> Scenario:
    """A dataset where two image files reduce to the same pairing key."""
    dataset = root / "duplicate" / "dataset"
    images_train = dataset / "images" / "train"
    masks_train = dataset / "masks" / "train"
    images_train.mkdir(parents=True)
    masks_train.mkdir(parents=True)

    _write_tiny_image(images_train / "cat_001.jpg")
    _write_tiny_image(images_train / "cat_001.png")
    _write_tiny_image(masks_train / "cat_001.png")

    try:
        im.discover_image_mask_pairs(dataset / "images", dataset / "masks")
    except ValueError as error:
        raw_message = str(error)
    else:
        raise AssertionError("expected discover_image_mask_pairs to reject a duplicate key")

    assert "cat_001" in raw_message, raw_message
    assert "train/cat_001.jpg" in raw_message, raw_message
    assert "train/cat_001.png" in raw_message, raw_message

    preview = _truncate_lines(_normalize_diagnostic_text(raw_message, dataset))
    result_lines = ["discover_image_mask_pairs: REJECTED", preview]

    return Scenario(
        title="3. Duplicate key rejected",
        tree_text=render_directory_tree(dataset),
        result_text="\n".join(result_lines),
        accent=_REJECTED_ACCENT,
        raw_diagnostic=raw_message,
    )


def _font_size_for(text: str) -> float:
    """A slightly smaller font for a panel with more than `_COMPACT_FONT_LINE_THRESHOLD` lines.

    Keeps a longer panel's text from needing more vertical space than its row can spare,
    without shrinking every panel uniformly.
    """
    line_count = text.count("\n") + 1
    return _COMPACT_FONT_SIZE if line_count > _COMPACT_FONT_LINE_THRESHOLD else _DEFAULT_FONT_SIZE


def find_text_overflow(
    panel_texts: Sequence[tuple[Axes, Text]],
    renderer: RendererBase,
    margin: float = _OVERFLOW_MARGIN_PX,
) -> list[str]:
    """Return one description per `(axes, text)` pair whose rendered bbox overflows its axes.

    `fig.canvas.draw()` must already have happened so `renderer` reflects final positions.
    Compares each text artist's `get_window_extent()` (its actual rendered pixel bbox) against
    its own axes' `get_window_extent()`, on all four sides, with a small pixel margin -- this
    is what actually caught the original bug (the missing-pair panel's last two lines rendering
    below its axes), not a visual inspection. Empty return means no overflow was found.
    """
    problems: list[str] = []
    for ax, text_artist in panel_texts:
        text_bbox = text_artist.get_window_extent(renderer=renderer)
        axes_bbox = ax.get_window_extent(renderer=renderer)
        label = text_artist.get_text().splitlines()[0] if text_artist.get_text() else "<empty>"

        if text_bbox.y0 < axes_bbox.y0 + margin:
            problems.append(
                f"{label!r}: text bottom {text_bbox.y0:.1f} overflows axes bottom "
                f"{axes_bbox.y0 + margin:.1f} (margin {margin}px)"
            )
        if text_bbox.y1 > axes_bbox.y1 - margin:
            problems.append(
                f"{label!r}: text top {text_bbox.y1:.1f} overflows axes top "
                f"{axes_bbox.y1 - margin:.1f} (margin {margin}px)"
            )
        if text_bbox.x0 < axes_bbox.x0 + margin:
            problems.append(
                f"{label!r}: text left {text_bbox.x0:.1f} overflows axes left "
                f"{axes_bbox.x0 + margin:.1f} (margin {margin}px)"
            )
        if text_bbox.x1 > axes_bbox.x1 - margin:
            problems.append(
                f"{label!r}: text right {text_bbox.x1:.1f} overflows axes right "
                f"{axes_bbox.x1 - margin:.1f} (margin {margin}px)"
            )
    return problems


def build_diagnostics_figure(
    scenarios: tuple[Scenario, Scenario, Scenario],
) -> tuple[Figure, tuple[tuple[Axes, Text], ...]]:
    """Build the 3x2 pairing diagnostics figure without saving or closing it.

    Matplotlib is imported and configured here, inside the build path, not at module import
    time. Returns the figure alongside every `(axes, text)` pair placed in the six tree/result
    panels (not row titles, which sit outside the panel's own text area) -- `render_diagnostics`
    uses this to save the figure, and `tests/test_demos.py` uses it directly to check for text
    overflow without ever writing a PNG or reading one back.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)

    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle

    fig = plt.figure(figsize=_FIGURE_SIZE_INCHES, dpi=_FIGURE_DPI)
    grid = fig.add_gridspec(
        3,
        2,
        left=0.02,
        right=0.99,
        top=0.87,
        bottom=0.08,
        wspace=0.05,
        hspace=0.35,
        height_ratios=_ROW_HEIGHT_RATIOS,
    )

    panel_texts: list[tuple[Axes, Text]] = []
    for row_index, scenario in enumerate(scenarios):
        tree_ax = fig.add_subplot(grid[row_index, 0])
        result_ax = fig.add_subplot(grid[row_index, 1])

        tree_ax.set_title(scenario.title, fontsize=10, loc="left")
        for ax, text, accent in (
            (tree_ax, scenario.tree_text, _TREE_ACCENT),
            (result_ax, scenario.result_text, scenario.accent),
        ):
            ax.axis("off")
            text_artist = ax.text(
                0.03,
                0.95,
                text,
                transform=ax.transAxes,
                fontsize=_font_size_for(text),
                fontfamily="monospace",
                linespacing=1.15,
                va="top",
                ha="left",
            )
            panel_texts.append((ax, text_artist))
            border = Rectangle(
                (0.0, 0.0),
                1.0,
                1.0,
                transform=ax.transAxes,
                fill=False,
                edgecolor=accent,
                linewidth=1.5,
            )
            ax.add_patch(border)

    fig.suptitle("improcv image/mask pairing diagnostics", fontsize=13)
    fig.text(
        0.5,
        0.02,
        "Pair by deterministic relative-path keys, not by list position.",
        ha="center",
        fontsize=9,
    )

    return fig, tuple(panel_texts)


def render_diagnostics(
    scenarios: tuple[Scenario, Scenario, Scenario], output: Path
) -> tuple[int, int]:
    """Render the 3x2 pairing diagnostics layout to `output` as a PNG.

    Returns the written image's `(width, height)` in pixels.
    """
    fig, panel_texts = build_diagnostics_figure(scenarios)

    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    try:
        fig.canvas.draw()
        assert isinstance(fig.canvas, FigureCanvasAgg)
        renderer = fig.canvas.get_renderer()
        overflow = find_text_overflow(panel_texts, renderer)
        if overflow:
            raise RuntimeError(
                "internal error: pairing diagnostics text overflows its panel:\n"
                + "\n".join(overflow)
            )

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
    parser = argparse.ArgumentParser(
        description="Generate the improcv pairing diagnostics demo PNG."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/assets/pairing-diagnostics.png"),
        help="Output PNG path (default: docs/assets/pairing-diagnostics.png).",
    )
    args = parser.parse_args(argv)
    if args.output.suffix.lower() != ".png":
        parser.error(f"--output must end in .png, got {args.output}")
    return args


def main() -> None:
    args = parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        scenarios = (
            build_success_scenario(root),
            build_missing_pair_scenario(root),
            build_duplicate_key_scenario(root),
        )

        try:
            width, height = render_diagnostics(scenarios, args.output)
        except OSError as error:
            print(f"error: could not write {args.output}: {error}", file=sys.stderr)
            raise SystemExit(1) from None

    print(f"wrote {args.output} ({width}x{height})")


if __name__ == "__main__":
    main()
