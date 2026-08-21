"""Classification-evaluation-result display helpers built on matplotlib."""

from __future__ import annotations

from typing import cast

import numpy as np
from matplotlib.axes import Axes

from improcv.evaluation import ConfusionMatrixResult, _require_confusion_matrix_result
from improcv.visualization._axes import _require_valid_axes, _resolve_axes

__all__ = [
    "plot_confusion_matrix",
]


def _format_cell_value(value: np.integer | np.floating, *, is_int64: bool) -> str:
    """Format one `ConfusionMatrixResult.matrix` cell for its annotation text.

    An `int64` matrix's cell formats as a plain, exact integer (`str(int(value))`); a `float64`
    matrix's cell formats via `str(float(value))` (identical to `repr(float(value))` in Python 3)
    -- the shortest decimal string that round-trips to the exact same `float64` value, at every
    magnitude, with no configurable precision and no silent rounding. `f"{value:g}"` is
    deliberately not used here: it rounds to 6 significant figures and can silently lose
    meaningful digits at large magnitude (e.g. `123456.789` -> `"123457"`), which would
    misrepresent a weighted confusion matrix's effective weight. `int(value)` is also
    deliberately not used: it truncates a `float64` value's fractional part entirely (e.g.
    `1.5` -> `"1"`), losing it completely.
    """
    if is_int64:
        return str(int(value))
    return str(float(value))


def plot_confusion_matrix(confusion: ConfusionMatrixResult, ax: Axes | None = None) -> Axes:
    """Display a confusion matrix as a heatmap, with per-cell value annotations.

    Parameters
    ----------
    confusion : ConfusionMatrixResult
        The result of `confusion_matrix` (or a hand-built equivalent). `confusion.matrix` must be
        structurally valid -- see Raises below -- but need not have a non-zero total: a
        structurally valid, non-empty, all-zero matrix (a legal output of `confusion_matrix`
        itself, e.g. for empty observations with explicit `labels`) is accepted and rendered.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If `None`, a new figure and axes are created.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the confusion matrix was drawn on (either `ax` or a newly created one) -- never
        calls `plt.show()` or saves to a file; displaying or saving is left to the caller.

    Raises
    ------
    TypeError
        If `confusion` is not a `ConfusionMatrixResult`, if `confusion.matrix` is not an
        `ndarray`, if `confusion.matrix`'s dtype is not exactly `int64` or exactly `float64`, if
        `confusion.labels` is not a `tuple`, if any label is not a plain (non-`bool`) `int`, or if
        `ax` is not a `matplotlib.axes.Axes` or `None`.
    ValueError
        If `confusion.matrix` does not have exactly 2 dimensions, is not square, is empty
        (``0x0``), contains a negative value, contains a non-finite value (`float64` only), or if
        `confusion.labels`'s length does not match `confusion.matrix`'s side length or contains a
        duplicate value.

    Notes
    -----
    Rows are true labels, columns are predicted labels, matching `confusion_matrix`'s own
    documented contract exactly -- the x/y axis labels ("predicted label"/"true label") state
    this directly. The colormap is fixed (`"Blues"`); there is no `cmap`, `annotate`, `title`, or
    numeric-format parameter in this first slice. Color normalization is fixed at
    ``vmin=0``/``vmax=float(confusion.matrix.max())`` -- `0` is a meaningful floor for a
    confusion matrix (no observations/no effective weight in that cell), not an arbitrary choice,
    and an all-zero matrix (``vmin == vmax == 0``) renders without error. Per-cell annotation text
    color is derived from the actual displayed image's own normalization
    (``image.norm(value)``), not a separately hand-computed formula, so it always matches the
    real displayed color. `confusion.matrix`'s dtype decides each cell's annotation format: an
    `int64` matrix (unweighted) annotates as a plain integer (e.g. `"2"`); a `float64` matrix
    (weighted, via `sample_weight`) annotates via `str(float(value))` (e.g. `"2.0"`,
    `"123456.789"`) -- exact at every magnitude, never rounded or truncated. This dtype-dependent
    formatting difference is intentional: a trailing `.0` signals a weighted cell.
    """
    _require_valid_axes(ax)
    _require_confusion_matrix_result(confusion)

    axes = _resolve_axes(ax)

    matrix = confusion.matrix
    labels = confusion.labels
    positions = list(range(len(labels)))
    is_int64 = matrix.dtype == np.int64

    image = axes.imshow(matrix, cmap="Blues", vmin=0, vmax=float(matrix.max()), aspect="equal")
    axes.set_xticks(positions)
    axes.set_xticklabels([str(label) for label in labels])
    axes.set_yticks(positions)
    axes.set_yticklabels([str(label) for label in labels])
    axes.set_xlabel("predicted label")
    axes.set_ylabel("true label")

    for row in positions:
        for column in positions:
            value = matrix[row, column]
            # `Normalize.__call__`'s type stub is typed for the general array-in/array-out case;
            # called with a NumPy scalar (as here), it returns a scalar, comparable directly.
            normalized = cast(float, image.norm(value))
            text_color = "white" if normalized > 0.5 else "black"
            axes.text(
                column,
                row,
                _format_cell_value(value, is_int64=is_int64),
                ha="center",
                va="center",
                color=text_color,
            )

    return axes
