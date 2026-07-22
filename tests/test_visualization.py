import subprocess
import sys
from collections.abc import Iterator

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

import improcv.visualization as viz
from improcv.color import bgr_to_rgb
from improcv.visualization.image import (
    _require_grayscale_or_bgr,
    _require_valid_title,
    _resolve_axes,
)


@pytest.fixture(autouse=True)
def _close_figures() -> Iterator[None]:
    yield
    plt.close("all")


def _gray(value: int = 128, shape: tuple[int, int] = (10, 10)) -> np.ndarray:
    return np.full(shape, value, dtype=np.uint8)


def _bgr(shape: tuple[int, int, int] = (10, 10, 3)) -> np.ndarray:
    image = np.zeros(shape, dtype=np.uint8)
    image[:, :, 0] = 255  # pure blue in BGR
    return image


# --- import guard behavior ---


def test_importing_visualization_does_not_import_pyplot() -> None:
    script = """
import sys

import improcv.visualization

assert "matplotlib.pyplot" not in sys.modules
print("ok")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr.decode()
    assert b"ok" in completed.stdout


# --- _require_grayscale_or_bgr ---


def test_require_grayscale_or_bgr_accepts_2d() -> None:
    _require_grayscale_or_bgr(_gray())


def test_require_grayscale_or_bgr_accepts_bgr() -> None:
    _require_grayscale_or_bgr(_bgr())


def test_require_grayscale_or_bgr_rejects_bgra() -> None:
    with pytest.raises(ValueError, match="channel"):
        _require_grayscale_or_bgr(np.zeros((10, 10, 4), dtype=np.uint8))


def test_require_grayscale_or_bgr_rejects_two_channels() -> None:
    with pytest.raises(ValueError, match="channel"):
        _require_grayscale_or_bgr(np.zeros((10, 10, 2), dtype=np.uint8))


# --- _require_valid_title ---


def test_require_valid_title_accepts_none() -> None:
    _require_valid_title(None)


def test_require_valid_title_accepts_str() -> None:
    _require_valid_title("hello")


def test_require_valid_title_rejects_non_str() -> None:
    with pytest.raises(TypeError, match="title"):
        _require_valid_title(123)


# --- _resolve_axes ---


def test_resolve_axes_returns_passed_axes() -> None:
    fig, ax = plt.subplots()
    assert _resolve_axes(ax) is ax


def test_resolve_axes_creates_new_axes_when_none() -> None:
    ax = _resolve_axes(None)
    assert isinstance(ax, Axes)


def test_resolve_axes_rejects_non_axes() -> None:
    with pytest.raises(TypeError, match="ax"):
        _resolve_axes("not an axes")  # type: ignore[arg-type]


# --- show_image ---


def test_show_image_grayscale_uses_gray_cmap_and_fixed_range() -> None:
    image = _gray(128)

    ax = viz.show_image(image)

    axes_image = ax.images[0]
    assert axes_image.get_cmap().name == "gray"
    assert axes_image.get_clim() == (0, 255)


def test_show_image_bgr_converts_to_rgb() -> None:
    image = _bgr()

    ax = viz.show_image(image)

    expected = bgr_to_rgb(image)
    assert np.array_equal(np.array(ax.images[0].get_array()), expected)


def test_show_image_hides_axes_by_default() -> None:
    ax = viz.show_image(_gray())

    assert ax.axison is False


def test_show_image_sets_title() -> None:
    ax = viz.show_image(_gray(), title="hello")

    assert ax.get_title() == "hello"


def test_show_image_uses_passed_axes() -> None:
    fig, ax = plt.subplots()

    result = viz.show_image(_gray(), ax=ax)

    assert result is ax


def test_show_image_does_not_mutate_input() -> None:
    image = _bgr()
    before = image.copy()

    viz.show_image(image)

    assert np.array_equal(image, before)


def test_show_image_rejects_bgra() -> None:
    with pytest.raises(ValueError, match="channel"):
        viz.show_image(np.zeros((10, 10, 4), dtype=np.uint8))


def test_show_image_rejects_non_uint8() -> None:
    with pytest.raises(TypeError, match="dtype"):
        viz.show_image(_gray().astype(np.float32))  # type: ignore[arg-type]


def test_show_image_rejects_bad_title() -> None:
    with pytest.raises(TypeError, match="title"):
        viz.show_image(_gray(), title=123)  # type: ignore[arg-type]


def test_show_image_rejects_bad_ax() -> None:
    with pytest.raises(TypeError, match="ax"):
        viz.show_image(_gray(), ax="not an axes")  # type: ignore[arg-type]


# --- plot_histogram ---


def test_plot_histogram_grayscale_produces_one_black_line() -> None:
    ax = viz.plot_histogram(_gray())

    lines = ax.get_lines()
    assert len(lines) == 1
    assert lines[0].get_color() == "k"


def test_plot_histogram_bgr_produces_three_colored_lines() -> None:
    ax = viz.plot_histogram(_bgr())

    lines = ax.get_lines()
    assert len(lines) == 3
    assert [line.get_color() for line in lines] == ["b", "g", "r"]


def test_plot_histogram_rejects_bgra() -> None:
    with pytest.raises(ValueError, match="channel"):
        viz.plot_histogram(np.zeros((10, 10, 4), dtype=np.uint8))


def test_plot_histogram_rejects_two_channels() -> None:
    with pytest.raises(ValueError, match="channel"):
        viz.plot_histogram(np.zeros((10, 10, 2), dtype=np.uint8))


def test_plot_histogram_mask_changes_result() -> None:
    image = _gray(128)
    image[:5, :] = 200  # top half a different value

    mask_top = np.zeros((10, 10), dtype=np.uint8)
    mask_top[:5, :] = 255
    mask_bottom = np.zeros((10, 10), dtype=np.uint8)
    mask_bottom[5:, :] = 255

    ax_top = viz.plot_histogram(image, mask=mask_top)
    ax_bottom = viz.plot_histogram(image, mask=mask_bottom)

    top_hist = ax_top.get_lines()[0].get_ydata()
    bottom_hist = ax_bottom.get_lines()[0].get_ydata()
    assert not np.array_equal(top_hist, bottom_hist)


def test_plot_histogram_uses_passed_axes() -> None:
    fig, ax = plt.subplots()

    result = viz.plot_histogram(_gray(), ax=ax)

    assert result is ax


def test_plot_histogram_does_not_mutate_input() -> None:
    image = _bgr()
    before = image.copy()

    viz.plot_histogram(image)

    assert np.array_equal(image, before)


def test_plot_histogram_propagates_histogram_dtype_error() -> None:
    with pytest.raises(TypeError):
        viz.plot_histogram(_gray().astype(np.float16))  # type: ignore[arg-type]


def test_plot_histogram_x_axis_reflects_value_range() -> None:
    image = _gray(150)

    ax = viz.plot_histogram(image, bins=4, value_range=(100.0, 200.0))

    x_data = ax.get_lines()[0].get_xdata()
    assert np.allclose(x_data, [112.5, 137.5, 162.5, 187.5])
    assert ax.get_xlim() == (100.0, 200.0)


def test_plot_histogram_does_not_leave_a_figure_open_on_dtype_error() -> None:
    before = tuple(plt.get_fignums())

    with pytest.raises(TypeError):
        viz.plot_histogram(np.zeros((10, 10), dtype=np.float16))  # type: ignore[arg-type]

    assert tuple(plt.get_fignums()) == before


def test_plot_histogram_does_not_leave_a_figure_open_on_bad_bins() -> None:
    before = tuple(plt.get_fignums())

    with pytest.raises(ValueError, match="bins"):
        viz.plot_histogram(_gray(), bins=0)

    assert tuple(plt.get_fignums()) == before
