"""Image and histogram display helpers built on matplotlib."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes

from improcv._validation import require_channels, require_dtype, require_image_ndim
from improcv.analysis import histogram
from improcv.color import bgr_to_rgb
from improcv.types import Image, ImageU8, Mask

__all__ = [
    "plot_histogram",
    "show_image",
]


def _require_grayscale_or_bgr(image: np.ndarray) -> None:
    """Raise ValueError unless `image` is 2D grayscale or 3-channel BGR.

    Shared by every function in this module -- BGRA and any other
    channel count are out of scope for this first version (YAGNI).
    """
    require_image_ndim(image, ndims=(2, 3))
    if image.ndim == 3:
        require_channels(image, 3)


def _require_valid_title(title: object) -> None:
    """Raise TypeError unless `title` is a `str` or `None`."""
    if title is not None and not isinstance(title, str):
        raise TypeError(f"title must be a str or None, got {type(title).__name__}")


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


def show_image(image: ImageU8, title: str | None = None, ax: Axes | None = None) -> Axes:
    """Display an image using matplotlib, handling BGR-to-RGB conversion.

    Parameters
    ----------
    image : np.ndarray
        A `uint8` image, grayscale (``(H, W)``) or BGR (``(H, W, 3)``).
        Other channel counts (e.g. BGRA) are rejected explicitly.
    title : str, optional
        If given, set as the axes' title.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If `None`, a new figure and axes are created.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the image was drawn on (either `ax` or a newly created
        one) -- never calls `plt.show()` or saves to a file; displaying or
        saving is left to the caller.

    Raises
    ------
    ValueError
        If `image` does not have exactly 2 dimensions or 3 channels, or is
        empty.
    TypeError
        If `image` does not have dtype ``uint8``, `title` is not a `str`
        or `None`, or `ax` is not a `matplotlib.axes.Axes` or `None`.

    Notes
    -----
    A grayscale image is shown with ``cmap="gray"`` and a fixed
    ``vmin=0``/``vmax=255`` range -- matplotlib's own default for 2D data
    is both ``cmap="viridis"`` (not grayscale) and a per-image-normalized
    range (a uniform mid-gray image would otherwise render identically to
    a uniform white or black image, since all three would normalize to the
    same single color without a fixed range) -- both verified directly. A
    3-channel image is converted from BGR to RGB via `bgr_to_rgb` before
    display -- matplotlib interprets channel 0 as red, so an unconverted
    BGR array displays with red and blue visually swapped (verified).
    Axes are hidden by default (``ax.axis("off")``); call ``ax.axis("on")``
    on the returned axes to restore them.
    """
    _require_grayscale_or_bgr(image)
    require_dtype(image, (np.uint8,))
    _require_valid_title(title)
    axes = _resolve_axes(ax)

    if image.ndim == 2:
        axes.imshow(image, cmap="gray", vmin=0, vmax=255)
    else:
        axes.imshow(bgr_to_rgb(image))
    axes.axis("off")
    if title is not None:
        axes.set_title(title)
    return axes


def plot_histogram(
    image: Image,
    bins: int = 256,
    value_range: tuple[float, float] = (0.0, 256.0),
    mask: Mask | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot the intensity histogram of every channel of an image.

    Parameters
    ----------
    image : np.ndarray
        Grayscale (``(H, W)``) or BGR (``(H, W, 3)``) image. Other channel
        counts are rejected explicitly. `dtype` is whatever `histogram`
        itself accepts (``uint8``, ``uint16``, ``float32``) -- not
        separately restricted here.
    bins, value_range, mask
        Passed straight through to `histogram` for each channel; see its
        own docstring for the exact contract. `value_range` is also used
        as the plotted x-axis range, converted to `float` only after
        `histogram` itself has validated it.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If `None`, a new figure and axes are created.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the histogram was plotted on -- never calls `plt.show()`.
        A grayscale image gets one black line; a BGR image gets three
        lines colored ``"b"``, ``"g"``, ``"r"`` -- matplotlib's own named
        colors happen to match OpenCV's channel-order letters exactly.
        Each line is plotted against its bin's center value (not the bin
        index), so the x-axis reflects `value_range` directly.

    Raises
    ------
    ValueError
        If `image` does not have exactly 2 dimensions or 3 channels, or is
        empty; anything `histogram` itself raises as `ValueError`
        propagates unchanged.
    TypeError
        If `ax` is not a `matplotlib.axes.Axes` or `None`; anything
        `histogram` itself raises as `TypeError` (e.g. an unsupported
        dtype) propagates unchanged.
    """
    _require_grayscale_or_bgr(image)
    axes = _resolve_axes(ax)

    channels = 1 if image.ndim == 2 else 3
    colors = ("k",) if channels == 1 else ("b", "g", "r")
    histograms = [
        histogram(image, channel=i, bins=bins, value_range=value_range, mask=mask)
        for i in range(channels)
    ]

    low = float(value_range[0])
    high = float(value_range[1])
    edges = np.linspace(low, high, histograms[0].shape[0] + 1, dtype=np.float64)
    centers = (edges[:-1] + edges[1:]) / 2.0

    for hist, color in zip(histograms, colors, strict=True):
        axes.plot(centers, hist, color=color)

    axes.set_xlim(low, high)
    axes.set_xlabel("pixel value")
    axes.set_ylabel("count")
    return axes
