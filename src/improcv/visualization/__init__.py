"""Matplotlib-based visualization helpers (requires the ``improcv[viz]`` extra).

``import improcv`` never imports this subpackage or matplotlib -- import
``improcv.visualization`` explicitly to use it.
"""

from __future__ import annotations

try:
    import matplotlib  # noqa: F401
    from matplotlib.axes import Axes  # noqa: F401
except ModuleNotFoundError as _exc:
    if _exc.name != "matplotlib":
        raise
    raise ImportError(
        "improcv.visualization requires matplotlib, which is not "
        'installed by default. Install it with `pip install "improcv[viz]"`.'
    ) from _exc

from improcv.visualization.image import plot_histogram, show_image  # noqa: E402

__all__ = [
    "plot_histogram",
    "show_image",
]
