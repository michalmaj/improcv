"""Shared, private `Axes`-handling helpers for every `improcv.visualization` module."""

from __future__ import annotations

from matplotlib.axes import Axes


def _require_valid_axes(ax: object) -> None:
    """Raise TypeError unless `ax` is a `matplotlib.axes.Axes` or `None`.

    Unlike `_resolve_axes`, this never creates a figure -- it's used to
    validate `ax` *before* any other, possibly-raising validation, so a
    caller error (a bad dtype, `bins`, `value_range`, or `mask`) can't
    leave behind an orphaned figure that `_resolve_axes` would have
    created before that later validation ran.
    """
    if ax is not None and not isinstance(ax, Axes):
        raise TypeError(f"ax must be a matplotlib.axes.Axes or None, got {type(ax).__name__}")


def _resolve_axes(ax: object) -> Axes:
    """Return `ax` if it's a valid `Axes`, else create and return a new one.

    Raises TypeError for anything else -- a wrong-type `ax` would
    otherwise fail deep inside matplotlib with a confusing `AttributeError`
    rather than a clear, attributable message.

    `pyplot` is imported here, not at module level, so that a caller who
    always passes their own `ax` never triggers pyplot's backend
    resolution merely by importing `improcv.visualization`.
    """
    if ax is not None:
        if not isinstance(ax, Axes):
            raise TypeError(f"ax must be a matplotlib.axes.Axes or None, got {type(ax).__name__}")
        return ax

    from matplotlib import pyplot as plt

    _, new_ax = plt.subplots()
    return new_ax
