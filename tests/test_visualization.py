import subprocess
import sys
import warnings
from collections.abc import Iterator

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

import improcv.visualization as viz
from improcv.color import bgr_to_rgb
from improcv.evaluation import ConfusionMatrixResult
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


# --- plot_confusion_matrix ---


def _int_confusion(matrix, labels) -> ConfusionMatrixResult:
    return ConfusionMatrixResult(matrix=np.array(matrix, dtype=np.int64), labels=tuple(labels))


def _float_confusion(matrix, labels) -> ConfusionMatrixResult:
    return ConfusionMatrixResult(matrix=np.array(matrix, dtype=np.float64), labels=tuple(labels))


def test_plot_confusion_matrix_returns_axes() -> None:
    confusion = _int_confusion([[3, 1], [0, 2]], (0, 1))
    ax = viz.plot_confusion_matrix(confusion)
    assert isinstance(ax, Axes)


def test_plot_confusion_matrix_uses_passed_axes() -> None:
    fig, ax = plt.subplots()
    confusion = _int_confusion([[3, 1], [0, 2]], (0, 1))

    result = viz.plot_confusion_matrix(confusion, ax=ax)

    assert result is ax


def test_plot_confusion_matrix_creates_new_axes_when_none() -> None:
    confusion = _int_confusion([[3, 1], [0, 2]], (0, 1))
    ax = viz.plot_confusion_matrix(confusion)
    assert isinstance(ax, Axes)


def test_plot_confusion_matrix_draws_exactly_one_image_artist() -> None:
    confusion = _int_confusion([[3, 1], [0, 2]], (0, 1))
    ax = viz.plot_confusion_matrix(confusion)
    assert len(ax.images) == 1


def test_plot_confusion_matrix_artist_array_equals_matrix() -> None:
    confusion = _int_confusion([[3, 1], [0, 2]], (0, 1))
    ax = viz.plot_confusion_matrix(confusion)
    assert np.array_equal(np.array(ax.images[0].get_array()), confusion.matrix)


def test_plot_confusion_matrix_uses_blues_cmap() -> None:
    confusion = _int_confusion([[3, 1], [0, 2]], (0, 1))
    ax = viz.plot_confusion_matrix(confusion)
    assert ax.images[0].get_cmap().name == "Blues"


def test_plot_confusion_matrix_clim_is_vmin_zero_vmax_matrix_max() -> None:
    confusion = _int_confusion([[3, 1], [0, 2]], (0, 1))
    ax = viz.plot_confusion_matrix(confusion)
    assert ax.images[0].get_clim() == (0.0, 3.0)


def test_plot_confusion_matrix_clim_uses_float_matrix_max_for_weighted() -> None:
    confusion = _float_confusion([[1.5, 0.25], [123456.789, 2.0]], (0, 1))
    ax = viz.plot_confusion_matrix(confusion)
    assert ax.images[0].get_clim() == (0.0, 123456.789)


def test_plot_confusion_matrix_tick_positions_and_count() -> None:
    confusion = _int_confusion([[1, 0, 0], [0, 1, 0], [0, 0, 1]], (10, 20, 30))
    ax = viz.plot_confusion_matrix(confusion)
    assert list(ax.get_xticks()) == [0, 1, 2]
    assert list(ax.get_yticks()) == [0, 1, 2]


def test_plot_confusion_matrix_tick_labels_match_confusion_labels_order() -> None:
    confusion = _int_confusion([[1, 0, 0], [0, 1, 0], [0, 0, 1]], (100, 5, 42))
    ax = viz.plot_confusion_matrix(confusion)
    x_labels = [t.get_text() for t in ax.get_xticklabels()]
    y_labels = [t.get_text() for t in ax.get_yticklabels()]
    assert x_labels == ["100", "5", "42"]
    assert y_labels == ["100", "5", "42"]


def test_plot_confusion_matrix_sets_axis_labels() -> None:
    confusion = _int_confusion([[3, 1], [0, 2]], (0, 1))
    ax = viz.plot_confusion_matrix(confusion)
    assert ax.get_xlabel() == "predicted label"
    assert ax.get_ylabel() == "true label"


def test_plot_confusion_matrix_int64_annotation_strings() -> None:
    confusion = _int_confusion([[0, 2], [123456, 1]], (0, 1))
    ax = viz.plot_confusion_matrix(confusion)
    texts = {text.get_text() for text in ax.texts}
    assert texts == {"0", "2", "123456", "1"}


def test_plot_confusion_matrix_float64_annotation_strings_exact() -> None:
    confusion = _float_confusion([[1.5, 0.25], [123456.789, 2.0]], (0, 1))
    ax = viz.plot_confusion_matrix(confusion)
    texts = {text.get_text() for text in ax.texts}
    assert texts == {"1.5", "0.25", "123456.789", "2.0"}


def test_plot_confusion_matrix_float64_annotation_does_not_use_g_format() -> None:
    """Regression lock for PR #196's correction: 123456.789 must render exactly, never rounded
    to the 6-significant-figure "123457" that `f"{value:g}"` would produce.
    """
    confusion = _float_confusion([[123456.789, 0.0], [0.0, 1.0]], (0, 1))
    ax = viz.plot_confusion_matrix(confusion)
    texts = {text.get_text() for text in ax.texts}
    assert "123456.789" in texts
    assert "123457" not in texts


def test_plot_confusion_matrix_float64_annotation_small_magnitude() -> None:
    confusion = _float_confusion([[1e-8, 3.75], [0.0, 5.0]], (0, 1))
    ax = viz.plot_confusion_matrix(confusion)
    texts = {text.get_text() for text in ax.texts}
    assert texts == {"1e-08", "3.75", "0.0", "5.0"}


def test_plot_confusion_matrix_float64_zero_is_distinguishable_from_int64_zero() -> None:
    int_confusion = _int_confusion([[0, 0], [0, 0]], (0, 1))
    float_confusion = _float_confusion([[0.0, 0.0], [0.0, 0.0]], (0, 1))

    int_ax = viz.plot_confusion_matrix(int_confusion)
    float_ax = viz.plot_confusion_matrix(float_confusion)

    assert {text.get_text() for text in int_ax.texts} == {"0"}
    assert {text.get_text() for text in float_ax.texts} == {"0.0"}


def test_plot_confusion_matrix_text_contrast_uses_artist_norm() -> None:
    confusion = _int_confusion([[0, 5], [10, 10]], (0, 1))
    ax = viz.plot_confusion_matrix(confusion)
    image = ax.images[0]

    by_text = {text.get_text(): text for text in ax.texts}
    # norm(0) == 0.0 -> not > 0.5 -> black
    assert image.norm(np.array(0)) == pytest.approx(0.0)
    assert by_text["0"].get_color() == "black"
    # norm(10) == 1.0 -> > 0.5 -> white
    assert image.norm(np.array(10)) == pytest.approx(1.0)
    assert by_text["10"].get_color() == "white"
    # norm(5) == 0.5 exactly -> threshold is strictly ">", so 0.5 stays black
    assert image.norm(np.array(5)) == pytest.approx(0.5)
    assert by_text["5"].get_color() == "black"


def test_plot_confusion_matrix_text_contrast_matches_norm_when_min_greater_than_zero() -> None:
    """Regression lock for the design's own demonstrated defect: contrast must be derived from
    the real AxesImage.norm, not from `value / matrix.max()`, which disagrees whenever the
    matrix's minimum is greater than zero.
    """
    confusion = _float_confusion([[1.5, 0.25], [2.0, 3.75]], (0, 1))
    ax = viz.plot_confusion_matrix(confusion)
    image = ax.images[0]

    by_text = {text.get_text(): text for text in ax.texts}
    # vmin=0 (frozen), vmax=3.75 -> norm(0.25) = 0.25/3.75 ≈ 0.0667, not the wrong
    # value/max_value=0.25/3.75 coincidence -- distinguished by checking against image.norm.
    assert image.norm(np.array(0.25)) == pytest.approx(0.25 / 3.75)
    assert by_text["0.25"].get_color() == "black"


def test_plot_confusion_matrix_all_zero_int_matrix_renders() -> None:
    confusion = _int_confusion([[0, 0], [0, 0]], (0, 1))
    ax = viz.plot_confusion_matrix(confusion)
    assert ax.images[0].get_clim() == (0.0, 0.0)
    assert {text.get_text() for text in ax.texts} == {"0"}


def test_plot_confusion_matrix_all_zero_float_matrix_renders_without_warning() -> None:
    confusion = _float_confusion([[0.0, 0.0], [0.0, 0.0]], (0, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ax = viz.plot_confusion_matrix(confusion)
    assert ax.images[0].get_clim() == (0.0, 0.0)


def test_plot_confusion_matrix_1x1_matrix_renders() -> None:
    confusion = _int_confusion([[5]], (7,))
    ax = viz.plot_confusion_matrix(confusion)
    assert ax.images[0].get_clim() == (0.0, 5.0)
    assert [t.get_text() for t in ax.get_xticklabels()] == ["7"]
    assert {text.get_text() for text in ax.texts} == {"5"}


def test_plot_confusion_matrix_reordered_noncontiguous_labels() -> None:
    confusion = _int_confusion([[1, 0], [0, 1]], (30, -5))
    ax = viz.plot_confusion_matrix(confusion)
    assert [t.get_text() for t in ax.get_xticklabels()] == ["30", "-5"]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["30", "-5"]


# --- plot_confusion_matrix: validation, reusing evaluation._require_confusion_matrix_result ---


def test_plot_confusion_matrix_rejects_wrong_object_type() -> None:
    with pytest.raises(TypeError, match="ConfusionMatrixResult"):
        viz.plot_confusion_matrix("not a confusion matrix")  # type: ignore[arg-type]


def test_plot_confusion_matrix_rejects_non_ndarray_matrix() -> None:
    bad = ConfusionMatrixResult(matrix=[[1, 2], [3, 4]], labels=(0, 1))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ndarray"):
        viz.plot_confusion_matrix(bad)


def test_plot_confusion_matrix_rejects_wrong_ndim() -> None:
    bad = ConfusionMatrixResult(matrix=np.array([1, 2, 3], dtype=np.int64), labels=(0, 1, 2))
    with pytest.raises(ValueError, match="2-D"):
        viz.plot_confusion_matrix(bad)


def test_plot_confusion_matrix_rejects_non_square() -> None:
    bad = ConfusionMatrixResult(matrix=np.zeros((2, 3), dtype=np.int64), labels=(0, 1))
    with pytest.raises(ValueError, match="square"):
        viz.plot_confusion_matrix(bad)


def test_plot_confusion_matrix_rejects_empty_matrix() -> None:
    bad = ConfusionMatrixResult(matrix=np.zeros((0, 0), dtype=np.int64), labels=())
    with pytest.raises(ValueError, match="empty"):
        viz.plot_confusion_matrix(bad)


def test_plot_confusion_matrix_rejects_unsupported_dtype() -> None:
    bad = ConfusionMatrixResult(
        matrix=np.zeros((2, 2), dtype=np.float32),  # type: ignore[arg-type]
        labels=(0, 1),
    )
    with pytest.raises(TypeError, match="int64 or float64"):
        viz.plot_confusion_matrix(bad)


def test_plot_confusion_matrix_rejects_negative_int64_values() -> None:
    bad = ConfusionMatrixResult(matrix=np.array([[-1, 0], [0, 1]], dtype=np.int64), labels=(0, 1))
    with pytest.raises(ValueError, match="negative"):
        viz.plot_confusion_matrix(bad)


def test_plot_confusion_matrix_rejects_negative_float64_values() -> None:
    bad = ConfusionMatrixResult(
        matrix=np.array([[-1.0, 0.0], [0.0, 1.0]], dtype=np.float64), labels=(0, 1)
    )
    with pytest.raises(ValueError, match="negative"):
        viz.plot_confusion_matrix(bad)


def test_plot_confusion_matrix_rejects_non_finite_float64_values() -> None:
    bad = ConfusionMatrixResult(
        matrix=np.array([[np.nan, 0.0], [0.0, 1.0]], dtype=np.float64), labels=(0, 1)
    )
    with pytest.raises(ValueError, match="finite"):
        viz.plot_confusion_matrix(bad)


def test_plot_confusion_matrix_rejects_labels_not_a_tuple() -> None:
    bad = ConfusionMatrixResult(matrix=np.zeros((2, 2), dtype=np.int64), labels=[0, 1])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="tuple"):
        viz.plot_confusion_matrix(bad)


def test_plot_confusion_matrix_rejects_non_int_label() -> None:
    bad = ConfusionMatrixResult(
        matrix=np.zeros((2, 2), dtype=np.int64),
        labels=(0, "a"),  # type: ignore[arg-type]
    )
    with pytest.raises(TypeError, match="int"):
        viz.plot_confusion_matrix(bad)


def test_plot_confusion_matrix_rejects_label_length_mismatch() -> None:
    bad = ConfusionMatrixResult(matrix=np.zeros((2, 2), dtype=np.int64), labels=(0, 1, 2))
    with pytest.raises(ValueError, match="entries"):
        viz.plot_confusion_matrix(bad)


def test_plot_confusion_matrix_rejects_duplicate_labels() -> None:
    bad = ConfusionMatrixResult(matrix=np.zeros((2, 2), dtype=np.int64), labels=(0, 0))
    with pytest.raises(ValueError, match="duplicate"):
        viz.plot_confusion_matrix(bad)


def test_plot_confusion_matrix_rejects_bad_ax() -> None:
    confusion = _int_confusion([[3, 1], [0, 2]], (0, 1))
    with pytest.raises(TypeError, match="ax"):
        viz.plot_confusion_matrix(confusion, ax="not an axes")  # type: ignore[arg-type]


def test_plot_confusion_matrix_does_not_leak_a_figure_on_invalid_confusion() -> None:
    before = tuple(plt.get_fignums())
    bad = ConfusionMatrixResult(matrix=np.zeros((0, 0), dtype=np.int64), labels=())

    with pytest.raises(ValueError):
        viz.plot_confusion_matrix(bad)

    assert tuple(plt.get_fignums()) == before


def test_plot_confusion_matrix_does_not_leak_a_figure_on_bad_ax() -> None:
    before = tuple(plt.get_fignums())
    confusion = _int_confusion([[3, 1], [0, 2]], (0, 1))

    with pytest.raises(TypeError):
        viz.plot_confusion_matrix(confusion, ax="not an axes")  # type: ignore[arg-type]

    assert tuple(plt.get_fignums()) == before


def test_plot_confusion_matrix_bad_ax_is_validated_before_confusion() -> None:
    """ax validation must happen first -- an invalid confusion combined with a bad ax must raise
    the ax TypeError, not a confusion-related error, matching the frozen validation ordering.
    """
    bad = ConfusionMatrixResult(matrix=np.zeros((0, 0), dtype=np.int64), labels=())
    with pytest.raises(TypeError, match="ax"):
        viz.plot_confusion_matrix(bad, ax="not an axes")  # type: ignore[arg-type]


def test_plot_confusion_matrix_does_not_call_plt_show(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def fake_show(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(plt, "show", fake_show)
    confusion = _int_confusion([[3, 1], [0, 2]], (0, 1))
    viz.plot_confusion_matrix(confusion)
    assert called is False


def test_plot_confusion_matrix_is_exported_from_visualization_package() -> None:
    assert "plot_confusion_matrix" in viz.__all__
    assert viz.plot_confusion_matrix is not None


def test_plot_confusion_matrix_is_not_a_top_level_improcv_symbol() -> None:
    import improcv as im

    assert not hasattr(im, "plot_confusion_matrix")
    assert "plot_confusion_matrix" not in im.__all__


def test_visualization_all_contains_exactly_three_names() -> None:
    assert sorted(viz.__all__) == ["plot_confusion_matrix", "plot_histogram", "show_image"]
