"""Generate the augmentation demo gallery PNG.

Builds a small synthetic BGR image and matching integer-label segmentation
mask, applies `improcv`'s replayable affine (fixed and expanded canvas) and
perspective transforms identically to both, and renders a 2x4 comparison
layout -- one parameter object per transform, always applied to the
image and mask together, with the mask always warped using nearest-neighbor
interpolation so its discrete labels stay legal.

This script is the source of truth for `docs/assets/augmentation-gallery.png`;
the PNG must never be edited by hand. Run with:

    uv run python demos/augmentation_gallery.py

See `demos/README.md` for the full regeneration contract.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

import improcv as im

_SOURCE_SIZE = (160, 120)  # (width, height)
_MASK_BORDER_VALUE = 255

# Okabe-Ito colorblind-safe palette, indexed to match _mask_indices' output:
# 0=background, 1=rectangle, 2=circle, 3=transform border.
_LEGEND_COLORS = ("#3B3B3B", "#56B4E9", "#E69F00", "#CC79A7")
_LEGEND_LABELS = ("0 background", "1 rectangle", "2 circle", "255 transform border")

_FIGURE_SIZE_INCHES = (12.8, 6.4)
_FIGURE_DPI = 150


class GalleryResults(NamedTuple):
    """Everything `render_gallery` needs to draw the gallery figure."""

    source_image: np.ndarray
    source_mask: np.ndarray
    affine_fixed: im.AugmentedImageMask
    affine_expanded: im.AugmentedImageMask
    perspective_fixed: im.AugmentedImageMask


def build_source_scene() -> tuple[np.ndarray, np.ndarray]:
    """Build the synthetic source image and its matching label mask.

    Returns a `(120, 160, 3)` uint8 BGR image and a `(120, 160)` uint8 mask
    with labels ``{0, 1, 2}`` (background, rectangle, circle). The image
    additionally carries an orientation grid, a directional arrow, and four
    distinctly colored corner markers that exist only in the image, never in
    the mask, and are positioned so they never overlap the rectangle, the
    circle, or the arrow.
    """
    width, height = _SOURCE_SIZE
    image = np.full((height, width, 3), 235, dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)

    grid_color = (200, 200, 200)
    for x in range(0, width, 20):
        cv2.line(image, (x, 0), (x, height - 1), grid_color, 1)
    for y in range(0, height, 20):
        cv2.line(image, (0, y), (width - 1, y), grid_color, 1)

    rectangle_top_left = (20, 20)
    rectangle_bottom_right = (60, 50)
    cv2.rectangle(image, rectangle_top_left, rectangle_bottom_right, (235, 150, 50), thickness=-1)
    cv2.rectangle(mask, rectangle_top_left, rectangle_bottom_right, 1, thickness=-1)

    circle_center = (120, 85)
    circle_radius = 20
    cv2.circle(image, circle_center, circle_radius, (60, 150, 235), thickness=-1)
    cv2.circle(mask, circle_center, circle_radius, 2, thickness=-1)

    cv2.arrowedLine(image, (15, 100), (75, 108), (0, 0, 0), thickness=2, tipLength=0.3)

    marker_size = 6
    corner_markers = (
        ((0, 0), (0, 0, 255)),  # top-left: red
        ((width - marker_size, 0), (0, 255, 0)),  # top-right: green
        ((width - marker_size, height - marker_size), (0, 255, 255)),  # bottom-right: yellow
        ((0, height - marker_size), (255, 0, 255)),  # bottom-left: magenta
    )
    for (marker_x, marker_y), color in corner_markers:
        corner = (marker_x + marker_size, marker_y + marker_size)
        cv2.rectangle(image, (marker_x, marker_y), corner, color, thickness=-1)

    return image, mask


def build_demo_results(image: np.ndarray, mask: np.ndarray) -> GalleryResults:
    """Sample and apply the affine and perspective transforms shown in the gallery.

    Asserts the geometric contract the gallery exists to demonstrate: replay
    equality, shared image/mask geometry, legal mask labels, and the fixed
    versus expanded affine canvas shapes. These are real checks against
    `improcv`'s public API, not illustrative comments.
    """
    source_size = (image.shape[1], image.shape[0])

    affine_rng = np.random.default_rng(7)
    affine_params = im.sample_affine(
        affine_rng,
        source_size=source_size,
        angle_range=(15.0, 15.0),
        translation_x_range=(10.0, 10.0),
        translation_y_range=(-8.0, -8.0),
        scale_range=(1.0, 1.0),
        axis_scale_x_range=(1.0, 1.0),
        axis_scale_y_range=(1.0, 1.0),
        shear_x_range=(0.0, 0.0),
        shear_y_range=(0.0, 0.0),
    )

    fixed = im.apply_affine(image, affine_params, mask=mask, mask_border_value=_MASK_BORDER_VALUE)
    replay = im.apply_affine(image, affine_params, mask=mask, mask_border_value=_MASK_BORDER_VALUE)
    assert np.array_equal(fixed.image, replay.image), "affine replay must be identical"
    assert np.array_equal(fixed.mask, replay.mask), "affine mask replay must be identical"
    assert fixed.image.shape[:2] == image.shape[:2], "fixed affine must keep the source shape"
    assert fixed.mask.shape == mask.shape, "fixed affine mask must keep the source shape"

    fixed_mask_labels = set(np.unique(fixed.mask).tolist())
    assert fixed_mask_labels <= {0, 1, 2, _MASK_BORDER_VALUE}, fixed_mask_labels
    assert _MASK_BORDER_VALUE in fixed_mask_labels, "fixed-canvas affine should reveal a border"

    expanded_params = im.expand_affine_canvas(affine_params)
    assert expanded_params.output_size is not None
    expanded = im.apply_affine(
        image, expanded_params, mask=mask, mask_border_value=_MASK_BORDER_VALUE
    )
    expected_shape = (expanded_params.output_size[1], expanded_params.output_size[0])
    assert expanded.image.shape[:2] == expected_shape
    assert expanded.mask.shape == expanded.image.shape[:2]
    assert expanded.image.shape[0] >= image.shape[0]
    assert expanded.image.shape[1] >= image.shape[1]

    expanded_mask_labels = set(np.unique(expanded.mask).tolist())
    assert expanded_mask_labels <= {0, 1, 2, _MASK_BORDER_VALUE}, expanded_mask_labels
    assert _MASK_BORDER_VALUE in expanded_mask_labels, "expanded affine should reveal a border"

    perspective_rng = np.random.default_rng(11)
    perspective_params = im.sample_perspective(
        perspective_rng, source_size=source_size, distortion_scale=0.3
    )
    perspective = im.apply_perspective(
        image, perspective_params, mask=mask, mask_border_value=_MASK_BORDER_VALUE
    )
    assert perspective.image.shape[:2] == image.shape[:2], "perspective must keep the source shape"
    assert perspective.mask.shape == mask.shape, "perspective mask must keep the source shape"

    perspective_mask_labels = set(np.unique(perspective.mask).tolist())
    assert perspective_mask_labels <= {0, 1, 2, _MASK_BORDER_VALUE}, perspective_mask_labels
    assert _MASK_BORDER_VALUE in perspective_mask_labels, "perspective should reveal a border"

    return GalleryResults(
        source_image=image,
        source_mask=mask,
        affine_fixed=fixed,
        affine_expanded=expanded,
        perspective_fixed=perspective,
    )


def _mask_indices(mask: np.ndarray) -> np.ndarray:
    """Map label values ``{0, 1, 2, 255}`` to dense colormap indices ``{0, 1, 2, 3}``."""
    indices = np.zeros_like(mask, dtype=np.uint8)
    indices[mask == 1] = 1
    indices[mask == 2] = 2
    indices[mask == _MASK_BORDER_VALUE] = 3
    return indices


def render_gallery(results: GalleryResults, output: Path) -> tuple[int, int]:
    """Render the 2x4 augmentation gallery image to `output` as a PNG.

    Matplotlib is imported and configured here, inside the render path, not
    at module import time -- importing this module never selects a backend,
    creates a figure, or writes a file. Returns the written image's
    `(width, height)` in pixels.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)

    from matplotlib import pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    cmap = ListedColormap(_LEGEND_COLORS)

    affine_fixed, affine_expanded, perspective_fixed = (
        results.affine_fixed,
        results.affine_expanded,
        results.perspective_fixed,
    )
    columns = (
        ("source", results.source_image, results.source_mask),
        ("affine -- fixed canvas", affine_fixed.image, affine_fixed.mask),
        ("affine -- expanded canvas", affine_expanded.image, affine_expanded.mask),
        ("perspective -- fixed canvas", perspective_fixed.image, perspective_fixed.mask),
    )

    fig, axes = plt.subplots(2, 4, figsize=_FIGURE_SIZE_INCHES, dpi=_FIGURE_DPI)
    plt.subplots_adjust(left=0.03, right=0.99, top=0.86, bottom=0.16, wspace=0.10, hspace=0.14)

    for column_index, (title, panel_image, panel_mask) in enumerate(columns):
        image_ax = axes[0, column_index]
        mask_ax = axes[1, column_index]

        panel_height, panel_width = panel_image.shape[:2]
        image_ax.imshow(im.bgr_to_rgb(panel_image))
        image_ax.set_title(f"{title}\n({panel_width}x{panel_height})", fontsize=9)

        mask_ax.imshow(
            _mask_indices(panel_mask), cmap=cmap, vmin=0, vmax=3, interpolation="nearest"
        )

        for ax in (image_ax, mask_ax):
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#888888")
                spine.set_linewidth(0.8)

    axes[0, 0].set_ylabel("image", fontsize=9)
    axes[1, 0].set_ylabel("mask", fontsize=9)

    legend_handles = [
        Patch(facecolor=color, edgecolor="#333333", label=label)
        for color, label in zip(_LEGEND_COLORS, _LEGEND_LABELS, strict=True)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.06),
    )
    fig.text(
        0.5,
        0.015,
        "One parameter object is applied to both the image and mask. "
        "Masks are always warped with nearest-neighbor interpolation.",
        ha="center",
        fontsize=8,
    )
    fig.suptitle("improcv augmentation gallery", fontsize=12)

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

    `--output` must end in `.png` (case-insensitive); this is checked here,
    via `parser.error` (standard argparse exit code `2`), before any
    rendering work happens.
    """
    parser = argparse.ArgumentParser(
        description="Generate the improcv augmentation demo gallery PNG."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/assets/augmentation-gallery.png"),
        help="Output PNG path (default: docs/assets/augmentation-gallery.png).",
    )
    args = parser.parse_args(argv)
    if args.output.suffix.lower() != ".png":
        parser.error(f"--output must end in .png, got {args.output}")
    return args


def main() -> None:
    args = parse_args()
    image, mask = build_source_scene()
    results = build_demo_results(image, mask)

    try:
        width, height = render_gallery(results, args.output)
    except OSError as error:
        print(f"error: could not write {args.output}: {error}", file=sys.stderr)
        raise SystemExit(1) from None

    print(f"wrote {args.output} ({width}x{height})")


if __name__ == "__main__":
    main()
