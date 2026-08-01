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
_PAIRING_SCRIPT = _DEMOS_DIR / "pairing_diagnostics.py"
_DEMO_SCRIPTS = (_AUGMENTATION_SCRIPT, _PAIRING_SCRIPT)


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


def test_demos_directory_has_the_expected_generators() -> None:
    scripts = sorted(_DEMOS_DIR.glob("*.py"))
    assert [script.name for script in scripts] == [
        "augmentation_gallery.py",
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

    assert "REJECTED" in scenario.result_text
    assert "lonely" in scenario.result_text
    assert "orphan" in scenario.result_text


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
