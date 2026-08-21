"""Runs demos/*.py as real subprocesses -- they are executable documentation.

Mirrors tests/test_examples.py's pattern: each generator is invoked with the current interpreter
(`sys.executable`), exactly as `python demos/whatever.py --output ...` would be run by a user,
never imported and called in-process for the subprocess checks -- importing would let a script
accidentally skip its own `if __name__ == "__main__":` guard. Each generator's own build_*
functions already assert their contract (replay equality, shared image/mask geometry, legal mask
labels, real pairing results/exceptions) before anything is rendered; this file additionally
loads each module directly (via `importlib`, not a package import -- `demos/` is not a package)
to exercise those functions as unit-level semantic checks, independent of rendering.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

import improcv as im

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEMOS_DIR = _REPO_ROOT / "demos"
_AUGMENTATION_SCRIPT = _DEMOS_DIR / "augmentation_gallery.py"
_CLASSIFICATION_SCRIPT = _DEMOS_DIR / "classification_report.py"
_PAIRING_SCRIPT = _DEMOS_DIR / "pairing_diagnostics.py"
_SIMILARITY_SCRIPT = _DEMOS_DIR / "image_similarity_gallery.py"
_DEMO_SCRIPTS = (_AUGMENTATION_SCRIPT, _CLASSIFICATION_SCRIPT, _PAIRING_SCRIPT, _SIMILARITY_SCRIPT)


def _run_demo(
    script: Path, args: list[str], env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=_REPO_ROOT,
        env=env,
    )


def _load_module(script: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(f"{script.stem}_demo", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def augmentation_module() -> types.ModuleType:
    return _load_module(_AUGMENTATION_SCRIPT)


@pytest.fixture(scope="module")
def pairing_module() -> types.ModuleType:
    return _load_module(_PAIRING_SCRIPT)


@pytest.fixture(scope="module")
def classification_module() -> types.ModuleType:
    return _load_module(_CLASSIFICATION_SCRIPT)


@pytest.fixture(scope="module")
def similarity_module() -> types.ModuleType:
    return _load_module(_SIMILARITY_SCRIPT)


def test_demos_directory_has_the_expected_generators() -> None:
    scripts = sorted(_DEMOS_DIR.glob("*.py"))
    assert [script.name for script in scripts] == [
        "augmentation_gallery.py",
        "classification_report.py",
        "image_similarity_gallery.py",
        "pairing_diagnostics.py",
    ]


@pytest.mark.parametrize("script", _DEMO_SCRIPTS, ids=lambda script: script.name)
def test_demo_is_importable_without_executing_main_or_touching_matplotlib(script: Path) -> None:
    # runpy.run_path with run_name="not_main" executes the module's top-level code (imports,
    # function/constant definitions) under a __name__ other than "__main__", so the
    # `if __name__ == "__main__":` guard never fires and main() is never called here.
    code = (
        "import runpy, sys\n"
        "runpy.run_path(sys.argv[1], run_name='not_main')\n"
        "assert 'matplotlib' not in sys.modules, "
        "'importing the demo must not import matplotlib'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(script)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize("script", _DEMO_SCRIPTS, ids=lambda script: script.name)
def test_demo_rejects_a_non_png_output_extension(script: Path, tmp_path: Path) -> None:
    output = tmp_path / "out.txt"

    result = _run_demo(script, ["--output", str(output)])

    assert result.returncode == 2
    assert not output.exists()
    assert "--output must end in .png" in result.stderr


@pytest.mark.parametrize("script", _DEMO_SCRIPTS, ids=lambda script: script.name)
def test_demo_reports_a_clean_error_when_the_output_parent_is_a_file(
    script: Path, tmp_path: Path
) -> None:
    # A parent path component that is a regular file (not a directory) fails the same way on
    # Windows, macOS, and Linux -- unlike a permissions-based failure, which varies by platform.
    blocking_file = tmp_path / "not_a_directory"
    blocking_file.write_text("")
    output = blocking_file / "out.png"

    result = _run_demo(script, ["--output", str(output)])

    assert result.returncode == 1
    assert result.stdout == ""
    assert not output.exists()
    assert "error: could not write" in result.stderr


@pytest.mark.parametrize("script", _DEMO_SCRIPTS, ids=lambda script: script.name)
def test_demos_leave_no_files_behind_in_the_repository(script: Path) -> None:
    # Mirrors tests/test_examples.py's shallow, non-recursive repo-cleanliness check: each
    # generator's default --output lands under docs/assets/, which is intentionally excluded
    # here since that committed asset is expected to exist; this only guards against a generator
    # writing anything unexpected next to its own source when given an explicit, non-default
    # --output (as every other test in this file does).
    before = {path for path in _DEMOS_DIR.iterdir() if path.name != "__pycache__"}

    with tempfile.TemporaryDirectory() as tmp:
        result = _run_demo(script, ["--output", str(Path(tmp) / "out.png")])
        assert result.returncode == 0, result.stderr

    after = {path for path in _DEMOS_DIR.iterdir() if path.name != "__pycache__"}
    assert after == before, f"{script.name} left new files behind: {after - before}"


# --- augmentation_gallery.py -------------------------------------------------------------


def test_augmentation_gallery_generates_the_expected_png_via_subprocess(tmp_path: Path) -> None:
    output = tmp_path / "gallery.png"
    before = set(tmp_path.iterdir())

    result = _run_demo(
        _AUGMENTATION_SCRIPT, ["--output", str(output)], env={**os.environ, "MPLBACKEND": "Agg"}
    )

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert result.stdout == f"wrote {output} (1920x960)\n"

    after = set(tmp_path.iterdir())
    assert after - before == {output}, f"expected exactly one new file, got {after - before}"

    image = cv2.imread(str(output), cv2.IMREAD_UNCHANGED)
    assert image is not None, "the written PNG must be readable by OpenCV"
    assert image.shape[0] == 960
    assert image.shape[1] == 1920
    assert image.shape[2] in (3, 4)
    assert output.stat().st_size > 0
    assert image.std() > 1.0, "the written PNG must not be a uniform, blank image"


def test_build_source_scene_returns_the_documented_contract(
    augmentation_module: types.ModuleType,
) -> None:
    image, mask = augmentation_module.build_source_scene()

    assert image.shape == (120, 160, 3)
    assert image.dtype == np.uint8
    assert mask.shape == (120, 160)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask).tolist()) <= {0, 1, 2}


def test_build_demo_results_satisfies_the_geometric_contract(
    augmentation_module: types.ModuleType,
) -> None:
    image, mask = augmentation_module.build_source_scene()

    # build_demo_results asserts replay equality, shared image/mask geometry, and legal mask
    # labels internally -- a successful call is itself part of the contract under test.
    results = augmentation_module.build_demo_results(image, mask)

    assert results.affine_fixed.image.shape[:2] == image.shape[:2]
    assert results.affine_fixed.mask.shape == mask.shape
    assert 255 in np.unique(results.affine_fixed.mask)

    assert results.affine_expanded.image.shape[0] >= image.shape[0]
    assert results.affine_expanded.image.shape[1] >= image.shape[1]
    assert results.affine_expanded.mask.shape == results.affine_expanded.image.shape[:2]
    assert 255 in np.unique(results.affine_expanded.mask)

    assert results.perspective_fixed.image.shape[:2] == image.shape[:2]
    assert results.perspective_fixed.mask.shape == mask.shape
    assert 255 in np.unique(results.perspective_fixed.mask)


def test_affine_replay_is_bit_for_bit_identical_via_the_public_api() -> None:
    # Independent of the demo's own internals: replays the exact sampling recipe the demo uses
    # directly against improcv's public API.
    image, mask = np.zeros((120, 160, 3), dtype=np.uint8), np.zeros((120, 160), dtype=np.uint8)
    rng = np.random.default_rng(7)
    params = im.sample_affine(
        rng,
        source_size=(160, 120),
        angle_range=(15.0, 15.0),
        translation_x_range=(10.0, 10.0),
        translation_y_range=(-8.0, -8.0),
    )

    first = im.apply_affine(image, params, mask=mask, mask_border_value=255)
    second = im.apply_affine(image, params, mask=mask, mask_border_value=255)

    assert np.array_equal(first.image, second.image)
    assert np.array_equal(first.mask, second.mask)


# --- classification_report.py -----------------------------------------------------------


def test_classification_report_generates_the_expected_png_via_subprocess(tmp_path: Path) -> None:
    output = tmp_path / "report.png"
    before = set(tmp_path.iterdir())

    result = _run_demo(
        _CLASSIFICATION_SCRIPT,
        ["--output", str(output)],
        env={**os.environ, "MPLBACKEND": "Agg"},
    )

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert result.stdout == f"wrote {output} (1920x1080)\n"

    after = set(tmp_path.iterdir())
    assert after - before == {output}, f"expected exactly one new file, got {after - before}"

    image = cv2.imread(str(output), cv2.IMREAD_UNCHANGED)
    assert image is not None, "the written PNG must be readable by OpenCV"
    assert image.shape[0] == 1080
    assert image.shape[1] == 1920
    assert image.shape[2] in (3, 4)
    assert output.stat().st_size > 0
    assert image.std() > 1.0, "the written PNG must not be a uniform, blank image"


def test_build_classification_inputs_returns_the_documented_contract(
    classification_module: types.ModuleType,
) -> None:
    inputs = classification_module.build_classification_inputs()

    assert inputs.labels == (20, 10, 30)
    assert inputs.y_score.shape == (9, 3)
    assert not np.allclose(inputs.y_score.sum(axis=1), 1.0)


def test_compute_classification_report_returns_the_public_report_type(
    classification_module: types.ModuleType,
) -> None:
    """Regression: the demo's hard-prediction portion must be the public `im.ClassificationReport`
    -- not a demo-local reimplementation -- and its fields must match a direct call to the public
    `im.classification_report(...)` exactly.
    """
    inputs = classification_module.build_classification_inputs()

    report = classification_module.compute_classification_report(inputs)

    assert type(report) is im.ClassificationReport
    assert not hasattr(classification_module, "ClassificationReport"), (
        "the demo must not define its own private ClassificationReport any more -- the public "
        "im.ClassificationReport is used directly"
    )

    direct = im.classification_report(
        y_true=inputs.y_true, y_pred=inputs.y_pred, labels=list(inputs.labels)
    )
    assert report.confusion == direct.confusion
    assert report.per_class == direct.per_class
    assert report.macro == direct.macro


def test_compute_classification_report_confusion_matrix(
    classification_module: types.ModuleType,
) -> None:
    inputs = classification_module.build_classification_inputs()

    # compute_classification_report asserts the exact confusion matrix and known classification
    # values internally -- a successful call is itself part of the contract under test.
    report = classification_module.compute_classification_report(inputs)

    assert report.confusion.labels == (20, 10, 30)
    assert report.confusion.matrix.shape == (3, 3)
    assert np.array_equal(
        report.confusion.matrix,
        np.array([[3, 0, 0], [0, 2, 1], [0, 1, 2]], dtype=np.int64),
    )


def test_compute_classification_report_classification_metrics(
    classification_module: types.ModuleType,
) -> None:
    inputs = classification_module.build_classification_inputs()
    report = classification_module.compute_classification_report(inputs)

    per_class = report.per_class
    assert per_class.labels == (20, 10, 30)
    assert per_class.average is None
    assert isinstance(per_class.precision, np.ndarray)
    assert isinstance(per_class.recall, np.ndarray)
    assert isinstance(per_class.f1, np.ndarray)
    assert per_class.precision.shape == (3,)
    assert per_class.recall.shape == (3,)
    assert per_class.f1.shape == (3,)
    assert per_class.support.tolist() == [3, 3, 3]

    assert per_class.precision.tolist() == pytest.approx([1.0, 2 / 3, 2 / 3])
    assert per_class.recall.tolist() == pytest.approx([1.0, 2 / 3, 2 / 3])
    assert per_class.f1.tolist() == pytest.approx([1.0, 2 / 3, 2 / 3])

    assert report.macro.accuracy == pytest.approx(7 / 9)
    assert report.macro.f1 == pytest.approx(7 / 9)


def test_compute_ranking_summary_is_demo_local_and_independent_of_the_public_report(
    classification_module: types.ModuleType,
) -> None:
    """Regression: ranking evaluation stays out of the public `ClassificationReport` -- it is a
    separate, demo-local `RankingSummary`, not a field bundled onto the public report.
    """
    inputs = classification_module.build_classification_inputs()

    ranking = classification_module.compute_ranking_summary(inputs)

    assert type(ranking) is classification_module.RankingSummary
    assert not hasattr(im, "RankingSummary"), "RankingSummary must never leak into public API"
    assert "RankingSummary" not in im.__all__


def test_compute_classification_report_ranking_metrics(
    classification_module: types.ModuleType,
) -> None:
    inputs = classification_module.build_classification_inputs()
    ranking = classification_module.compute_ranking_summary(inputs)

    assert isinstance(ranking.auc_per_class, np.ndarray)
    assert isinstance(ranking.ap_per_class, np.ndarray)
    assert ranking.auc_per_class.shape == (3,)
    assert ranking.ap_per_class.shape == (3,)
    assert all(np.isfinite(ranking.auc_per_class))
    assert all(np.isfinite(ranking.ap_per_class))
    for scalar in (
        ranking.auc_macro,
        ranking.auc_weighted,
        ranking.auc_micro,
        ranking.ap_macro,
        ranking.ap_weighted,
        ranking.ap_micro,
    ):
        assert np.isfinite(scalar)

    # Matches examples/classification_evaluation.py's printed output for this exact fixed
    # example, with a reasonable tolerance -- not hand-copied into the generator itself.
    assert ranking.auc_per_class.tolist() == pytest.approx([1.0, 0.9444444, 1.0], abs=1e-6)
    assert ranking.auc_macro == pytest.approx(0.9814815, abs=1e-6)
    assert ranking.auc_weighted == pytest.approx(0.9814815, abs=1e-6)
    assert ranking.auc_micro == pytest.approx(0.9783951, abs=1e-6)
    assert ranking.ap_per_class.tolist() == pytest.approx([1.0, 0.9166667, 1.0], abs=1e-6)
    assert ranking.ap_macro == pytest.approx(0.9722222, abs=1e-6)
    assert ranking.ap_weighted == pytest.approx(0.9722222, abs=1e-6)
    assert ranking.ap_micro == pytest.approx(0.9658120, abs=1e-6)


def test_compute_classification_report_roc_curves(
    classification_module: types.ModuleType,
) -> None:
    inputs = classification_module.build_classification_inputs()
    ranking = classification_module.compute_ranking_summary(inputs)

    roc_curves = ranking.roc_curves
    assert isinstance(roc_curves, im.MulticlassRocCurve)
    assert roc_curves.labels == (20, 10, 30)
    assert len(roc_curves.curves) == 3
    for index, curve in enumerate(roc_curves.curves):
        assert isinstance(curve, im.RocCurve)
        assert curve.positive_label == inputs.labels[index]

    # ROC AUC from the curves themselves is bit-for-bit identical to the already-verified
    # per-class multiclass_roc_auc_score(average=None) entries.
    for index, curve in enumerate(roc_curves.curves):
        area = im.auc(curve.false_positive_rate, curve.true_positive_rate)
        assert area == ranking.auc_per_class[index]


def test_score_column_mapping_comes_from_a_real_enumeration_of_labels(
    classification_module: types.ModuleType,
) -> None:
    labels = (20, 10, 30)

    lines = classification_module._score_column_lines(labels)

    assert lines == [
        "column 0 -> label 20",
        "column 1 -> label 10",
        "column 2 -> label 30",
    ]
    # Built from a real enumerate(labels), not three independently hand-typed lines: reordering
    # labels must reorder the rendered lines identically, not just their label numbers.
    reordered = classification_module._score_column_lines((10, 30, 20))
    assert reordered == [
        "column 0 -> label 10",
        "column 1 -> label 30",
        "column 2 -> label 20",
    ]


def test_classification_report_text_contains_the_required_terms(
    classification_module: types.ModuleType,
) -> None:
    inputs = classification_module.build_classification_inputs()
    report = classification_module.compute_classification_report(inputs)
    ranking = classification_module.compute_ranking_summary(inputs)

    contract_text = classification_module._report_contract_text(inputs.labels)
    classification_text = classification_module._classification_table_text(
        inputs.labels, report.per_class, report.macro
    )
    ranking_table_text = classification_module._ranking_table_text(
        inputs.labels, ranking.auc_per_class, ranking.ap_per_class
    )
    ranking_summary_text = classification_module._ranking_summary_text(
        ranking.auc_macro,
        ranking.auc_weighted,
        ranking.auc_micro,
        ranking.ap_macro,
        ranking.ap_weighted,
        ranking.ap_micro,
    )
    combined = "\n".join(
        [
            classification_module._CONFUSION_MATRIX_CAPTION,
            contract_text,
            classification_text,
            ranking_table_text,
            ranking_summary_text,
        ]
    )

    for expected in (
        "rows",
        "true",
        "columns",
        "predicted",
        "accuracy",
        "macro F1",
        "ROC AUC",
        "average precision",
        "macro",
        "weighted",
        "micro",
    ):
        assert expected in combined, expected


def test_classification_report_figure_renders_real_per_class_roc_curves(
    classification_module: types.ModuleType,
) -> None:
    inputs = classification_module.build_classification_inputs()
    report = classification_module.compute_classification_report(inputs)
    ranking = classification_module.compute_ranking_summary(inputs)

    fig, _panel_texts = classification_module.build_classification_figure(inputs, report, ranking)
    try:
        roc_axes = fig.axes[2]  # confusion matrix, ROC curves, then the report column's axes
        assert roc_axes.get_title(loc="left") == "Per-class ROC curves"

        # One line per label plus the diagonal reference line -- drawn directly from
        # ranking.roc_curves.curves, never through improcv.visualization.
        lines = roc_axes.get_lines()
        assert len(lines) == len(ranking.roc_curves.curves) + 1

        legend_labels = [text.get_text() for text in roc_axes.get_legend().get_texts()]
        assert legend_labels == [f"label {label}" for label in ranking.roc_curves.labels]

        for index, curve in enumerate(ranking.roc_curves.curves):
            line = lines[index]
            np.testing.assert_array_equal(line.get_xdata(), curve.false_positive_rate)
            np.testing.assert_array_equal(line.get_ydata(), curve.true_positive_rate)
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_render_confusion_matrix_delegates_to_public_plot_confusion_matrix(
    classification_module: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_render_confusion_matrix` must not reimplement the heatmap itself -- it must call the
    public `improcv.visualization.plot_confusion_matrix` exactly once, with the exact
    `ConfusionMatrixResult`/`Axes` it was given, and only add its own decoration on top.
    """
    import improcv.visualization as viz

    real_plot_confusion_matrix = viz.plot_confusion_matrix
    calls: list[tuple[object, object]] = []

    def spy(confusion, ax=None):
        calls.append((confusion, ax))
        return real_plot_confusion_matrix(confusion, ax=ax)

    monkeypatch.setattr(viz, "plot_confusion_matrix", spy)

    inputs = classification_module.build_classification_inputs()
    report = classification_module.compute_classification_report(inputs)
    ranking = classification_module.compute_ranking_summary(inputs)

    fig, _panel_texts = classification_module.build_classification_figure(inputs, report, ranking)
    try:
        assert len(calls) == 1, "plot_confusion_matrix must be called exactly once"
        called_confusion, called_ax = calls[0]
        assert called_confusion is report.confusion
        assert called_ax is fig.axes[1]  # contract strip, then confusion matrix, then ROC curves
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_classification_report_panel_text_never_overflows_its_axes(
    classification_module: types.ModuleType,
) -> None:
    # Regression-style check inspired by the same bug class fixed in PR #113 for
    # pairing_diagnostics.py: builds the actual figure, forces a draw so Matplotlib finalizes
    # every artist's on-screen position, and compares each panel's text bounding box against its
    # own axes' bounding box directly via find_text_overflow. No OCR, no PNG involved.
    inputs = classification_module.build_classification_inputs()
    report = classification_module.compute_classification_report(inputs)
    ranking = classification_module.compute_ranking_summary(inputs)

    fig, panel_texts = classification_module.build_classification_figure(inputs, report, ranking)
    try:
        assert len(panel_texts) == 6, (
            "expected the top contract strip, the report column's four sections, and the "
            "bottom caption"
        )

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        overflow = classification_module.find_text_overflow(panel_texts, renderer)

        assert overflow == [], "\n".join(overflow)
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_classification_report_demo_migration_leaks_no_new_public_symbols(
    classification_module: types.ModuleType,
) -> None:
    """Regression: neither `RankingSummary` nor `ClassificationInputs` (this demo's own private
    types) may appear in `improcv`'s public API -- the migration to the public
    `classification_report` must not accidentally widen it.
    """
    assert len(im.__all__) == 198
    assert len(im.__all__) == len(set(im.__all__))
    for private_name in ("RankingSummary", "ClassificationInputs"):
        assert not hasattr(im, private_name)
        assert private_name not in im.__all__
        assert hasattr(classification_module, private_name), (
            f"{private_name} should still exist as a demo-local type"
        )


def test_classification_report_demo_is_deterministic_across_repeated_execution(
    classification_module: types.ModuleType,
) -> None:
    inputs = classification_module.build_classification_inputs()

    report_a = classification_module.compute_classification_report(inputs)
    ranking_a = classification_module.compute_ranking_summary(inputs)
    report_b = classification_module.compute_classification_report(inputs)
    ranking_b = classification_module.compute_ranking_summary(inputs)

    assert report_a == report_b

    # RankingSummary is a plain NamedTuple holding raw ndarray fields, so field-by-field
    # comparison is required -- unlike ClassificationReport, whose ConfusionMatrixResult/
    # ClassificationMetrics fields already define a safe, array-aware __eq__.
    assert np.array_equal(ranking_a.auc_per_class, ranking_b.auc_per_class)
    assert ranking_a.auc_macro == ranking_b.auc_macro
    assert ranking_a.auc_weighted == ranking_b.auc_weighted
    assert ranking_a.auc_micro == ranking_b.auc_micro
    assert np.array_equal(ranking_a.ap_per_class, ranking_b.ap_per_class)
    assert ranking_a.ap_macro == ranking_b.ap_macro
    assert ranking_a.ap_weighted == ranking_b.ap_weighted
    assert ranking_a.ap_micro == ranking_b.ap_micro
    assert ranking_a.roc_curves == ranking_b.roc_curves


# --- pairing_diagnostics.py --------------------------------------------------------------


def test_pairing_diagnostics_generates_the_expected_png_via_subprocess(tmp_path: Path) -> None:
    output = tmp_path / "diagnostics.png"
    before = set(tmp_path.iterdir())

    result = _run_demo(
        _PAIRING_SCRIPT, ["--output", str(output)], env={**os.environ, "MPLBACKEND": "Agg"}
    )

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert result.stdout == f"wrote {output} (1920x1080)\n"

    after = set(tmp_path.iterdir())
    assert after - before == {output}, f"expected exactly one new file, got {after - before}"

    image = cv2.imread(str(output), cv2.IMREAD_UNCHANGED)
    assert image is not None, "the written PNG must be readable by OpenCV"
    assert image.shape[0] == 1080
    assert image.shape[1] == 1920
    assert image.shape[2] in (3, 4)
    assert output.stat().st_size > 0
    assert image.std() > 1.0, "the written PNG must not be a uniform, blank image"


def test_build_success_scenario_returns_two_deterministic_pairs(
    pairing_module: types.ModuleType,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        scenario = pairing_module.build_success_scenario(Path(tmp))

    assert scenario.raw_diagnostic is None
    assert "PAIRED" in scenario.result_text
    assert "paired: 2" in scenario.result_text
    assert "train/cat_001" in scenario.result_text
    assert "train/dog_001" in scenario.result_text


def test_build_missing_pair_scenario_naive_zip_produces_a_wrong_pair(
    pairing_module: types.ModuleType,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        scenario = pairing_module.build_missing_pair_scenario(Path(tmp))

    assert "cat_001.jpg <-> cat_001.png" in scenario.result_text
    assert "lonely.jpg <-> orphan.png" in scenario.result_text
    assert "WRONG BUT SILENT" in scenario.result_text


def test_build_missing_pair_scenario_raises_and_identifies_both_keys(
    pairing_module: types.ModuleType,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        scenario = pairing_module.build_missing_pair_scenario(root)

        assert scenario.raw_diagnostic is not None
        assert "lonely" in scenario.raw_diagnostic
        assert "orphan" in scenario.raw_diagnostic
        assert str(root) not in scenario.raw_diagnostic

    for expected in (
        "WRONG BUT SILENT",
        "discover_image_mask_pairs",
        "REJECTED",
        "image keys without a matching mask",
        "train/lonely",
        "mask keys without a matching image",
        "train/orphan",
    ):
        assert expected in scenario.result_text, expected

    # The final line naming the unmatched mask key must survive intact, not be replaced by a
    # "... (N more lines)" truncation summary.
    assert scenario.result_text.rstrip().endswith("'train/orphan'")
    assert "..." not in scenario.result_text

    # The rendered fragment naming both keys must be an exact, contiguous slice of the real
    # captured exception text, not a paraphrase or hand-typed replacement.
    assert scenario.raw_diagnostic is not None
    real_fragment = scenario.raw_diagnostic[
        scenario.raw_diagnostic.index("image keys without a matching mask:") :
    ]
    assert real_fragment in scenario.result_text

    assert str(root) not in scenario.result_text
    assert "\\" not in scenario.result_text


def test_build_duplicate_key_scenario_raises_and_identifies_both_paths(
    pairing_module: types.ModuleType,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        scenario = pairing_module.build_duplicate_key_scenario(Path(tmp))

    assert scenario.raw_diagnostic is not None
    assert "cat_001" in scenario.raw_diagnostic
    assert "train/cat_001.jpg" in scenario.raw_diagnostic
    assert "train/cat_001.png" in scenario.raw_diagnostic
    assert "REJECTED" in scenario.result_text


@pytest.mark.parametrize(
    "scenario_builder",
    ["build_success_scenario", "build_missing_pair_scenario", "build_duplicate_key_scenario"],
)
def test_pairing_rendered_text_has_no_temporary_root(
    pairing_module: types.ModuleType, scenario_builder: str
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        scenario = getattr(pairing_module, scenario_builder)(root)

        assert str(root) not in scenario.tree_text
        assert str(root) not in scenario.result_text
        assert "\\" not in scenario.result_text
        assert "\\" not in scenario.tree_text


def test_render_directory_tree_is_sorted_relative_and_posix_style(
    pairing_module: types.ModuleType,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        dataset = Path(tmp) / "dataset"
        (dataset / "images" / "train").mkdir(parents=True)
        (dataset / "masks" / "train").mkdir(parents=True)
        (dataset / "images" / "train" / "b.jpg").write_bytes(b"")
        (dataset / "images" / "train" / "a.jpg").write_bytes(b"")

        tree = pairing_module.render_directory_tree(dataset)

        assert str(dataset) not in tree

    assert tree == (
        "dataset/\n  images/\n    train/\n      a.jpg\n      b.jpg\n  masks/\n    train/"
    )


def test_pairing_diagnostics_panel_text_never_overflows_its_axes(
    pairing_module: types.ModuleType,
) -> None:
    # Regression test for a real bug: the missing-pair scenario's result panel used to render
    # its final lines below its own axes' bottom edge. No OCR, no reading pixels back from a
    # PNG -- this builds the actual figure, forces a draw so Matplotlib finalizes every
    # artist's on-screen position, and compares each panel's text bounding box against its own
    # axes' bounding box directly via find_text_overflow.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        scenarios = (
            pairing_module.build_success_scenario(root),
            pairing_module.build_missing_pair_scenario(root),
            pairing_module.build_duplicate_key_scenario(root),
        )

    fig, panel_texts = pairing_module.build_diagnostics_figure(scenarios)
    try:
        assert len(panel_texts) == 6, "expected one tree panel and one result panel per row"

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        overflow = pairing_module.find_text_overflow(panel_texts, renderer)

        assert overflow == [], "\n".join(overflow)
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


# --- image_similarity_gallery.py ---------------------------------------------------------


def test_image_similarity_gallery_generates_the_expected_png_via_subprocess(
    tmp_path: Path,
) -> None:
    output = tmp_path / "gallery.png"
    before = set(tmp_path.iterdir())

    result = _run_demo(
        _SIMILARITY_SCRIPT, ["--output", str(output)], env={**os.environ, "MPLBACKEND": "Agg"}
    )

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert result.stdout == f"wrote {output} (1920x1080)\n"

    after = set(tmp_path.iterdir())
    assert after - before == {output}, f"expected exactly one new file, got {after - before}"

    image = cv2.imread(str(output), cv2.IMREAD_UNCHANGED)
    assert image is not None, "the written PNG must be readable by OpenCV"
    assert image.shape[0] == 1080
    assert image.shape[1] == 1920
    assert image.shape[2] in (3, 4)
    assert output.stat().st_size > 0
    assert image.std() > 1.0, "the written PNG must not be a uniform, blank image"


def test_build_similarity_inputs_returns_the_documented_contract(
    similarity_module: types.ModuleType,
) -> None:
    inputs = similarity_module.build_similarity_inputs()

    assert inputs.names == (
        "a_base.png",
        "b_variant.png",
        "c_variant_more.png",
        "z_unrelated.png",
    )
    assert len(inputs.images) == 4
    for image in inputs.images:
        assert image.shape == (8, 8)
        assert image.dtype == np.uint8
        assert int(np.count_nonzero(image == 255)) == 32

    expected_hex = (
        "55aa55aa55aa55aa",
        "95aa55aa55aa55aa",
        "956a55aa55aa55aa",
        "0f0f0f0f0f0f0f0f",
    )
    for hash_value, hex_value in zip(inputs.hashes, expected_hex, strict=True):
        assert str(hash_value) == hex_value
        assert hash_value.algorithm == im.PerceptualHashAlgorithm.AVERAGE_HASH
        assert hash_value.hash_size == 8


def test_compute_similarity_report_returns_the_documented_contract(
    similarity_module: types.ModuleType,
) -> None:
    inputs = similarity_module.build_similarity_inputs()

    # compute_similarity_report asserts the exact distance values and pair result internally --
    # a successful call is itself part of the contract under test.
    report = similarity_module.compute_similarity_report(inputs, threshold=2)

    assert report.threshold == 2
    assert report.distances.shape == (4, 4)
    assert report.distances.dtype == np.int64
    assert np.array_equal(report.distances, report.distances.T)
    assert np.all(np.diag(report.distances) == 0)

    index = {name: position for position, name in enumerate(inputs.names)}
    assert report.distances[index["a_base.png"], index["b_variant.png"]] == 2
    assert report.distances[index["b_variant.png"], index["c_variant_more.png"]] == 2
    assert report.distances[index["a_base.png"], index["c_variant_more.png"]] == 4
    for other in ("a_base.png", "b_variant.png", "c_variant_more.png"):
        assert report.distances[index["z_unrelated.png"], index[other]] > 2

    found = tuple((pair.first.name, pair.second.name, pair.distance) for pair in report.pairs)
    assert found == (
        ("a_base.png", "b_variant.png", 2),
        ("b_variant.png", "c_variant_more.png", 2),
    ), found


def test_image_similarity_gallery_matches_the_public_api_independently(
    similarity_module: types.ModuleType,
) -> None:
    # Independent of the demo's own internals: recomputes one distance and one pair-search
    # result directly against improcv's public API, using the same images the demo builds.
    inputs = similarity_module.build_similarity_inputs()
    report = similarity_module.compute_similarity_report(inputs, threshold=2)

    index = {name: position for position, name in enumerate(inputs.names)}
    a_hash = im.average_hash(inputs.images[index["a_base.png"]], hash_size=8)
    b_hash = im.average_hash(inputs.images[index["b_variant.png"]], hash_size=8)
    assert a_hash.distance(b_hash) == report.distances[index["a_base.png"], index["b_variant.png"]]

    hashes_by_name = {
        name: im.average_hash(image, hash_size=8)
        for name, image in zip(inputs.names, inputs.images, strict=True)
    }
    pairs = im.find_similar_image_pairs(hashes_by_name, max_distance=2)
    found = tuple((pair.first.name, pair.second.name, pair.distance) for pair in pairs)
    assert found == tuple(
        (pair.first.name, pair.second.name, pair.distance) for pair in report.pairs
    )
