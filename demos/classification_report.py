"""Generate the classification report demo PNG.

Runs the existing public classification workflow --

    y_true + y_pred
    -> classification_report (confusion matrix + per-class/macro metrics,
       composed internally from confusion_matrix and
       classification_metrics_from_confusion_matrix)

    y_true + y_score
    -> multiclass_roc_auc_score
    -> multiclass_average_precision_score
    -> multiclass_roc_curve

on a small, fully explicit, hand-written multiclass example (the same one
used by `examples/classification_evaluation.py`, duplicated here so this
script stays self-contained), and renders a report showing three contracts
this workflow depends on: rows/columns of the confusion matrix are
`(true labels, predicted labels)`, `y_score[:, i]` corresponds to
`labels[i]` -- in exactly the (deliberately unsorted) order `labels` is
given, never a sorted one -- and `multiclass_roc_curve`'s own `result.curves[i]`
follows that same order, rendered here as real per-class ROC curves.

The prediction-level portion (confusion matrix, per-class metrics, macro
metrics) comes from the public `improcv.classification_report`/
`ClassificationReport`, introduced in `0.5.0a3` -- this demo no longer builds
that part itself. The ranking-level portion (AUC/AP/ROC curves, from
`y_score`) stays out of `ClassificationReport` by design (see
`docs/design/0.5.0a3-classification-report.md`) and remains demo-local,
computed via the existing multiclass ranking functions and bundled into the
demo-local `RankingSummary` below.

This script does not call `multiclass_precision_recall_curve` or render a
precision-recall curve -- one rendered curve type is enough to demonstrate the
new API without turning this demo into a general plotting surface; the
distinction between a PR curve's trapezoidal area and average precision is
already demonstrated in `examples/classification_evaluation.py`. This remains
a demo script rendering one fixed report figure, not a public plotting API --
`improcv.visualization` is not extended by this file.

This script is the source of truth for `docs/assets/classification-report.png`;
the PNG must never be edited by hand. Run with:

    uv run python demos/classification_report.py

See `demos/README.md` for the full regeneration contract.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

import improcv as im

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.backend_bases import RendererBase
    from matplotlib.figure import Figure
    from matplotlib.text import Text

_FIGURE_SIZE_INCHES = (12.8, 7.2)
_FIGURE_DPI = 150

_NEUTRAL_ACCENT = "#4C72B0"
_CLASSIFICATION_ACCENT = "#009E73"
_RANKING_ACCENT = "#CC79A7"
_ROC_ACCENT = "#D55E00"

# A small, fixed, deterministic color cycle for per-class ROC curves -- not Matplotlib's own
# default cycle, whose exact colors are not part of this module's own contract. Reused (wrapping
# around) if there are ever more labels than colors, a legibility trade-off, not a bug.
_ROC_CURVE_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9")

# Minimum on-screen gap (in pixels, at the figure's own render resolution) required between a
# panel's text bounding box and its axes' bounding box on every side; see find_text_overflow.
_OVERFLOW_MARGIN_PX = 4.0

_CONFUSION_MATRIX_CAPTION = "rows = true labels, columns = predicted labels"


class ClassificationInputs(NamedTuple):
    """The fixed, explicit multiclass example this demo evaluates."""

    labels: tuple[int, ...]
    y_true: tuple[int, ...]
    y_pred: tuple[int, ...]
    y_score: np.ndarray


class RankingSummary(NamedTuple):
    """Ranking-only evaluation results this demo renders, computed from `y_score`.

    Kept explicitly separate from the public, prediction-only `improcv.ClassificationReport`
    (which has no `y_score`/ranking data of any kind, by design -- see
    `docs/design/0.5.0a3-classification-report.md`). This is demo-local rendering input, not a
    public type.
    """

    auc_per_class: np.ndarray
    auc_macro: float
    auc_weighted: float
    auc_micro: float
    ap_per_class: np.ndarray
    ap_macro: float
    ap_weighted: float
    ap_micro: float
    roc_curves: im.MulticlassRocCurve


def build_classification_inputs() -> ClassificationInputs:
    """The same small, explicit multiclass example as `examples/classification_evaluation.py`.

    `labels` is deliberately unsorted -- it controls confusion-matrix row/column order and
    `y_score` column order alike, not sorted order. `y_score`'s rows are asserted to not sum to
    `1.0`: `improcv`'s multiclass ranking functions never require a probability simplex.
    """
    labels = (20, 10, 30)
    y_true = (20, 10, 30, 20, 10, 30, 20, 10, 30)
    y_pred = (20, 10, 30, 20, 30, 30, 20, 10, 10)
    y_score = np.array(
        [
            [0.75, 0.40, 0.35],
            [0.30, 0.70, 0.35],
            [0.20, 0.30, 0.80],
            [0.60, 0.50, 0.55],
            [0.55, 0.45, 0.40],
            [0.25, 0.35, 0.65],
            [0.70, 0.30, 0.45],
            [0.30, 0.65, 0.30],
            [0.20, 0.30, 0.60],
        ],
        dtype=np.float64,
    )

    assert y_score.shape == (len(y_true), len(labels))
    assert not np.allclose(y_score.sum(axis=1), 1.0), (
        "this demo exists to show that rows need not sum to 1.0 -- if this ever fails, the "
        "fixed example data above no longer demonstrates that contract"
    )

    return ClassificationInputs(labels=labels, y_true=y_true, y_pred=y_pred, y_score=y_score)


def compute_classification_report(inputs: ClassificationInputs) -> im.ClassificationReport:
    """Run the public `classification_report` and assert its documented contract before rendering.

    Only a handful of exactly-known classification values (this fixed example's confusion matrix,
    precision/recall/F1, accuracy, and macro F1 are all simple rationals) are asserted here -- this
    demo never independently calls `confusion_matrix`/`classification_metrics`/
    `classification_metrics_from_confusion_matrix`: `report.confusion`/`report.per_class`/
    `report.macro` are the single source of truth for those values.
    """
    labels = list(inputs.labels)

    report = im.classification_report(y_true=inputs.y_true, y_pred=inputs.y_pred, labels=labels)

    assert report.confusion.labels == inputs.labels
    assert report.confusion.matrix.shape == (3, 3)
    assert np.array_equal(
        report.confusion.matrix,
        np.array([[3, 0, 0], [0, 2, 1], [0, 1, 2]], dtype=np.int64),
    )

    assert report.per_class.labels == inputs.labels
    assert report.per_class.average is None
    assert isinstance(report.per_class.precision, np.ndarray)
    assert isinstance(report.per_class.recall, np.ndarray)
    assert isinstance(report.per_class.f1, np.ndarray)
    assert report.per_class.precision.shape == (3,)
    assert report.per_class.recall.shape == (3,)
    assert report.per_class.f1.shape == (3,)
    assert report.per_class.support.tolist() == [3, 3, 3]
    assert np.allclose(report.per_class.precision, [1.0, 2 / 3, 2 / 3])
    assert np.allclose(report.per_class.recall, [1.0, 2 / 3, 2 / 3])
    assert np.allclose(report.per_class.f1, [1.0, 2 / 3, 2 / 3])

    assert report.macro.average == "macro"
    assert np.isclose(report.macro.accuracy, 7 / 9)
    assert np.isclose(report.macro.f1, 7 / 9)

    return report


def compute_ranking_summary(inputs: ClassificationInputs) -> RankingSummary:
    """Compute ranking-only values from `y_score` and assert their documented contract.

    Kept independent of `compute_classification_report`: ranking evaluation stays out of the
    public `ClassificationReport` by design, so this demo computes it separately via the existing
    multiclass ranking functions, exactly as before this module's `classification_report`
    migration -- only its container type changed (`RankingSummary` instead of the old private
    `ClassificationReport`), not its computation.
    """
    labels = list(inputs.labels)

    auc_per_class = im.multiclass_roc_auc_score(
        inputs.y_true, inputs.y_score, labels=labels, average=None
    )
    auc_macro = im.multiclass_roc_auc_score(
        inputs.y_true, inputs.y_score, labels=labels, average="macro"
    )
    auc_weighted = im.multiclass_roc_auc_score(
        inputs.y_true, inputs.y_score, labels=labels, average="weighted"
    )
    auc_micro = im.multiclass_roc_auc_score(
        inputs.y_true, inputs.y_score, labels=labels, average="micro"
    )

    ap_per_class = im.multiclass_average_precision_score(
        inputs.y_true, inputs.y_score, labels=labels, average=None
    )
    ap_macro = im.multiclass_average_precision_score(
        inputs.y_true, inputs.y_score, labels=labels, average="macro"
    )
    ap_weighted = im.multiclass_average_precision_score(
        inputs.y_true, inputs.y_score, labels=labels, average="weighted"
    )
    ap_micro = im.multiclass_average_precision_score(
        inputs.y_true, inputs.y_score, labels=labels, average="micro"
    )

    roc_curves = im.multiclass_roc_curve(inputs.y_true, inputs.y_score, labels=labels)

    assert isinstance(auc_per_class, np.ndarray) and auc_per_class.shape == (3,)
    assert isinstance(ap_per_class, np.ndarray) and ap_per_class.shape == (3,)
    assert all(np.isfinite(auc_per_class))
    assert all(np.isfinite(ap_per_class))
    for scalar in (auc_macro, auc_weighted, auc_micro, ap_macro, ap_weighted, ap_micro):
        assert np.isfinite(scalar)

    # multiclass_roc_curve preserves labels' own (unsorted) order, and each curve's trapezoidal
    # area is bit-for-bit identical to that same class's multiclass_roc_auc_score(average=None)
    # entry -- both real, load-bearing contracts of the new API, not just rendering conveniences.
    assert roc_curves.labels == inputs.labels
    assert len(roc_curves.curves) == 3
    for index, curve in enumerate(roc_curves.curves):
        assert curve.positive_label == inputs.labels[index]
        area = im.auc(curve.false_positive_rate, curve.true_positive_rate)
        assert area == auc_per_class[index]

    return RankingSummary(
        auc_per_class=auc_per_class,
        auc_macro=float(auc_macro),
        auc_weighted=float(auc_weighted),
        auc_micro=float(auc_micro),
        ap_per_class=ap_per_class,
        ap_macro=float(ap_macro),
        ap_weighted=float(ap_weighted),
        ap_micro=float(ap_micro),
        roc_curves=roc_curves,
    )


def _score_column_lines(labels: Sequence[int]) -> list[str]:
    """`column i -> label labels[i]`, one line per label, from a real enumeration of `labels`."""
    return [f"column {index} -> label {label}" for index, label in enumerate(labels)]


def _top_contract_text(labels: Sequence[int]) -> str:
    """A compact, two-line version of the label/score-column contract for the top strip.

    The report column's own contract panel (`_report_contract_text`) restates this with headers
    and the "rows need not sum to 1" note; this strip is meant to stay readable at a glance.
    """
    mapping = "   ".join(
        f"y_score[:, {index}] -> label {label}" for index, label in enumerate(labels)
    )
    return f"labels = {list(labels)}\n{mapping}"


def _report_contract_text(labels: Sequence[int]) -> str:
    lines = [
        "LABEL ORDER",
        f"labels = {list(labels)}",
        "",
        "SCORE COLUMNS",
        *_score_column_lines(labels),
        "",
        "Rows of y_score do not need to sum to 1.",
    ]
    return "\n".join(lines)


def _classification_table_text(
    labels: Sequence[int], per_class: im.ClassificationMetrics, macro: im.ClassificationMetrics
) -> str:
    # per_class was computed with average=None, so precision/recall/f1 are always arrays here
    # (never the scalar form classification_metrics returns for a non-None average) -- asserted
    # for real, not just to satisfy the type checker.
    precision, recall, f1 = per_class.precision, per_class.recall, per_class.f1
    assert isinstance(precision, np.ndarray)
    assert isinstance(recall, np.ndarray)
    assert isinstance(f1, np.ndarray)

    header = f"{'label':>6} | {'precision':>9} | {'recall':>6} | {'F1':>5} | {'support':>7}"
    separator = "-" * len(header)
    rows = [
        f"{label:>6} | {row_precision:>9.3f} | {row_recall:>6.3f} | {row_f1:>5.3f} | "
        f"{int(support):>7}"
        for label, row_precision, row_recall, row_f1, support in zip(
            labels,
            precision,
            recall,
            f1,
            per_class.support,
            strict=True,
        )
    ]
    lines = [
        "CLASSIFICATION METRICS",
        "",
        header,
        separator,
        *rows,
        "",
        f"accuracy: {macro.accuracy:.3f}",
        f"macro F1: {macro.f1:.3f}",
    ]
    return "\n".join(lines)


def _ranking_table_text(
    labels: Sequence[int], auc_per_class: np.ndarray, ap_per_class: np.ndarray
) -> str:
    header = f"{'label':>6} | {'ROC AUC':>7} | {'average precision':>18}"
    separator = "-" * len(header)
    rows = [
        f"{label:>6} | {auc:>7.3f} | {ap:>18.3f}"
        for label, auc, ap in zip(labels, auc_per_class, ap_per_class, strict=True)
    ]
    lines = ["RANKING METRICS", "", header, separator, *rows]
    return "\n".join(lines)


def _ranking_summary_text(
    auc_macro: float,
    auc_weighted: float,
    auc_micro: float,
    ap_macro: float,
    ap_weighted: float,
    ap_micro: float,
) -> str:
    label_width = 9
    column_width = 10
    header = (
        f"{'':<{label_width}}{'macro':>{column_width}}"
        f"{'weighted':>{column_width}}{'micro':>{column_width}}"
    )
    auc_row = (
        f"{'ROC AUC':<{label_width}}{auc_macro:>{column_width}.3f}"
        f"{auc_weighted:>{column_width}.3f}{auc_micro:>{column_width}.3f}"
    )
    ap_row = (
        f"{'AP':<{label_width}}{ap_macro:>{column_width}.3f}"
        f"{ap_weighted:>{column_width}.3f}{ap_micro:>{column_width}.3f}"
    )
    return "\n".join(["SUMMARY", "", header, auc_row, ap_row])


def _add_border(ax: Axes, color: str) -> None:
    from matplotlib.patches import Rectangle

    border = Rectangle(
        (0.0, 0.0), 1.0, 1.0, transform=ax.transAxes, fill=False, edgecolor=color, linewidth=1.3
    )
    ax.add_patch(border)


def _render_confusion_matrix(
    ax: Axes, confusion: im.ConfusionMatrixResult, labels: Sequence[int]
) -> None:
    """Draw the confusion matrix heatmap directly with Matplotlib -- never through
    `improcv.visualization`, which this demo does not extend.
    """
    matrix = confusion.matrix
    positions = list(range(len(labels)))

    ax.imshow(matrix, cmap="Blues", aspect="equal")
    ax.set_xticks(positions)
    ax.set_xticklabels([str(label) for label in labels])
    ax.set_yticks(positions)
    ax.set_yticklabels([str(label) for label in labels])
    ax.set_xlabel("predicted label")
    ax.set_ylabel("true label")
    ax.set_title("Confusion matrix", fontsize=10, loc="left")

    max_value = float(matrix.max()) if matrix.size else 0.0
    for row in positions:
        for column in positions:
            value = int(matrix[row, column])
            normalized = value / max_value if max_value > 0 else 0.0
            # Color is an addition, not the only signal -- every cell's count is always
            # written as text, with its color chosen for contrast against that cell's fill.
            text_color = "white" if normalized > 0.5 else "black"
            ax.text(
                column, row, str(value), ha="center", va="center", color=text_color, fontsize=12
            )

    ax.text(
        0.5,
        -0.16,
        _CONFUSION_MATRIX_CAPTION,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
    )


def _render_roc_curves(ax: Axes, roc_curves: im.MulticlassRocCurve) -> None:
    """Draw one one-vs-rest ROC curve per label, directly from `roc_curves.curves`.

    Curves are drawn in `roc_curves.labels`' own order (never sorted) -- the same order
    `multiclass_roc_curve` itself preserves from the caller's `labels` argument. Never goes
    through `improcv.visualization`, which this demo does not extend.
    """
    for index, curve in enumerate(roc_curves.curves):
        color = _ROC_CURVE_COLORS[index % len(_ROC_CURVE_COLORS)]
        ax.plot(
            curve.false_positive_rate,
            curve.true_positive_rate,
            color=color,
            linewidth=1.6,
            label=f"label {roc_curves.labels[index]}",
        )
    ax.plot([0.0, 1.0], [0.0, 1.0], color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("false positive rate", fontsize=8)
    ax.set_ylabel("true positive rate", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title("Per-class ROC curves", fontsize=10, loc="left")
    ax.legend(loc="lower right", fontsize=7, framealpha=0.9)


def find_text_overflow(
    panel_texts: Sequence[tuple[Axes, Text]],
    renderer: RendererBase,
    margin: float = _OVERFLOW_MARGIN_PX,
) -> list[str]:
    """Return one description per `(axes, text)` pair whose rendered bbox overflows its axes.

    `fig.canvas.draw()` must already have happened so `renderer` reflects final positions.
    Compares each text artist's `get_window_extent()` against its own axes' `get_window_extent()`
    on all four sides, with a small pixel margin -- the same check `demos/pairing_diagnostics.py`
    uses, duplicated locally rather than shared, since this demo does not depend on that one.
    Empty return means no overflow was found.
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


def build_classification_figure(
    inputs: ClassificationInputs, report: im.ClassificationReport, ranking: RankingSummary
) -> tuple[Figure, tuple[tuple[Axes, Text], ...]]:
    """Build the classification report figure without saving or closing it.

    `report` is the public, prediction-only `improcv.ClassificationReport`; `ranking` is this
    demo's own ranking-only bundle -- kept as two separate parameters, matching the same
    architectural boundary the public API itself draws (see `docs/design/0.5.0a3-classification-
    report.md`), rather than merged back into one object here.

    Matplotlib is imported and configured here, inside the build path, not at module import
    time. Returns the figure alongside every `(axes, text)` pair placed in a controlled panel
    (the top contract strip, the report column's four sections, and the bottom caption) --
    `render_classification_report` uses this to save the figure, and `tests/test_demos.py` uses
    it directly to check for text overflow without ever writing a PNG or reading one back.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)

    from matplotlib import pyplot as plt

    labels = inputs.labels
    panel_texts: list[tuple[Axes, Text]] = []

    fig = plt.figure(figsize=_FIGURE_SIZE_INCHES, dpi=_FIGURE_DPI)
    outer = fig.add_gridspec(
        3,
        1,
        left=0.055,
        right=0.98,
        top=0.92,
        bottom=0.06,
        height_ratios=(0.08, 0.82, 0.07),
        hspace=0.12,
    )

    contract_ax = fig.add_subplot(outer[0])
    contract_ax.axis("off")
    contract_text = contract_ax.text(
        0.01,
        0.5,
        _top_contract_text(labels),
        transform=contract_ax.transAxes,
        fontsize=9,
        fontfamily="monospace",
        va="center",
        ha="left",
    )
    panel_texts.append((contract_ax, contract_text))
    _add_border(contract_ax, _NEUTRAL_ACCENT)

    # 3 columns instead of the original 2: confusion matrix, the new per-class ROC curve panel,
    # and the report column -- widened overall (0.9 + 0.95 + 1.25 vs. the original 1.0 + 1.35) so
    # the confusion matrix and report column keep essentially their original on-figure width at
    # 1920x1080, with the new ROC panel fit into the freed middle space rather than shrinking
    # either existing panel.
    main = outer[1].subgridspec(1, 3, width_ratios=(0.9, 0.95, 1.25), wspace=0.16)
    cm_ax = fig.add_subplot(main[0])
    _render_confusion_matrix(cm_ax, report.confusion, labels)

    roc_ax = fig.add_subplot(main[1])
    _render_roc_curves(roc_ax, ranking.roc_curves)
    _add_border(roc_ax, _ROC_ACCENT)

    report_grid = main[2].subgridspec(4, 1, height_ratios=(0.28, 0.30, 0.22, 0.20), hspace=0.12)

    report_contract_ax = fig.add_subplot(report_grid[0])
    report_contract_ax.axis("off")
    report_contract_text = report_contract_ax.text(
        0.02,
        0.94,
        _report_contract_text(labels),
        transform=report_contract_ax.transAxes,
        fontsize=7.5,
        fontfamily="monospace",
        linespacing=1.1,
        va="top",
        ha="left",
    )
    panel_texts.append((report_contract_ax, report_contract_text))
    _add_border(report_contract_ax, _NEUTRAL_ACCENT)

    classification_ax = fig.add_subplot(report_grid[1])
    classification_ax.axis("off")
    classification_text = classification_ax.text(
        0.02,
        0.96,
        _classification_table_text(labels, report.per_class, report.macro),
        transform=classification_ax.transAxes,
        fontsize=7.5,
        fontfamily="monospace",
        linespacing=1.1,
        va="top",
        ha="left",
    )
    panel_texts.append((classification_ax, classification_text))
    _add_border(classification_ax, _CLASSIFICATION_ACCENT)

    ranking_table_ax = fig.add_subplot(report_grid[2])
    ranking_table_ax.axis("off")
    ranking_table_text = ranking_table_ax.text(
        0.02,
        0.94,
        _ranking_table_text(labels, ranking.auc_per_class, ranking.ap_per_class),
        transform=ranking_table_ax.transAxes,
        fontsize=7.5,
        fontfamily="monospace",
        linespacing=1.1,
        va="top",
        ha="left",
    )
    panel_texts.append((ranking_table_ax, ranking_table_text))
    _add_border(ranking_table_ax, _RANKING_ACCENT)

    ranking_summary_ax = fig.add_subplot(report_grid[3])
    ranking_summary_ax.axis("off")
    ranking_summary_text = ranking_summary_ax.text(
        0.02,
        0.90,
        _ranking_summary_text(
            ranking.auc_macro,
            ranking.auc_weighted,
            ranking.auc_micro,
            ranking.ap_macro,
            ranking.ap_weighted,
            ranking.ap_micro,
        ),
        transform=ranking_summary_ax.transAxes,
        fontsize=8.5,
        fontfamily="monospace",
        va="top",
        ha="left",
    )
    panel_texts.append((ranking_summary_ax, ranking_summary_text))
    _add_border(ranking_summary_ax, _RANKING_ACCENT)

    caption_ax = fig.add_subplot(outer[2])
    caption_ax.axis("off")
    caption_text = caption_ax.text(
        0.5,
        0.5,
        "Explicit labels control both confusion-matrix order and y_score column meaning.",
        transform=caption_ax.transAxes,
        fontsize=9,
        ha="center",
        va="center",
    )
    panel_texts.append((caption_ax, caption_text))

    fig.suptitle("improcv multiclass classification report", fontsize=13)

    return fig, tuple(panel_texts)


def render_classification_report(
    inputs: ClassificationInputs,
    report: im.ClassificationReport,
    ranking: RankingSummary,
    output: Path,
) -> tuple[int, int]:
    """Render the classification report to `output` as a PNG.

    Returns the written image's `(width, height)` in pixels.
    """
    fig, panel_texts = build_classification_figure(inputs, report, ranking)

    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    try:
        fig.canvas.draw()
        assert isinstance(fig.canvas, FigureCanvasAgg)
        renderer = fig.canvas.get_renderer()
        overflow = find_text_overflow(panel_texts, renderer)
        if overflow:
            raise RuntimeError(
                "internal error: classification report text overflows its panel:\n"
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
        description="Generate the improcv classification report demo PNG."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/assets/classification-report.png"),
        help="Output PNG path (default: docs/assets/classification-report.png).",
    )
    args = parser.parse_args(argv)
    if args.output.suffix.lower() != ".png":
        parser.error(f"--output must end in .png, got {args.output}")
    return args


def main() -> None:
    args = parse_args()
    inputs = build_classification_inputs()
    report = compute_classification_report(inputs)
    ranking = compute_ranking_summary(inputs)

    try:
        width, height = render_classification_report(inputs, report, ranking, args.output)
    except OSError as error:
        print(f"error: could not write {args.output}: {error}", file=sys.stderr)
        raise SystemExit(1) from None

    print(f"wrote {args.output} ({width}x{height})")


if __name__ == "__main__":
    main()
