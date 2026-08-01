"""Runs demos/augmentation_gallery.py as a real subprocess -- it is executable documentation.

Mirrors tests/test_examples.py's pattern: the generator is invoked with the current interpreter
(`sys.executable`), exactly as `python demos/augmentation_gallery.py --output ...` would be run
by a user, never imported and called in-process for the subprocess checks -- importing would let
the script accidentally skip its own `if __name__ == "__main__":` guard. The demo's own
`build_demo_results` already asserts its geometric contract (replay equality, shared image/mask
geometry, legal mask labels, fixed vs. expanded canvas shape) before anything is rendered; this
file additionally loads the module directly (via `importlib`, not a package import -- `demos/`
is not a package) to exercise `build_source_scene`/`build_demo_results` as unit-level semantic
checks, independent of rendering.
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
_DEMO_SCRIPT = _DEMOS_DIR / "augmentation_gallery.py"


def _run_demo(
    args: list[str], env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_DEMO_SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=_REPO_ROOT,
        env=env,
    )


@pytest.fixture(scope="module")
def demo_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("augmentation_gallery_demo", _DEMO_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_demos_directory_has_the_expected_generator() -> None:
    scripts = sorted(_DEMOS_DIR.glob("*.py"))
    assert [script.name for script in scripts] == ["augmentation_gallery.py"]


def test_demo_is_importable_without_executing_main_or_touching_matplotlib() -> None:
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
        [sys.executable, "-c", code, str(_DEMO_SCRIPT)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


def test_demo_generates_the_expected_png_via_subprocess(tmp_path: Path) -> None:
    output = tmp_path / "gallery.png"
    before = set(tmp_path.iterdir())

    result = _run_demo(["--output", str(output)], env={**os.environ, "MPLBACKEND": "Agg"})

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


def test_demo_rejects_a_non_png_output_extension(tmp_path: Path) -> None:
    output = tmp_path / "gallery.txt"

    result = _run_demo(["--output", str(output)])

    assert result.returncode == 2
    assert not output.exists()
    assert "--output must end in .png" in result.stderr


def test_demo_reports_a_clean_error_when_the_output_parent_is_a_file(tmp_path: Path) -> None:
    # A parent path component that is a regular file (not a directory) fails the same way on
    # Windows, macOS, and Linux -- unlike a permissions-based failure, which varies by platform.
    blocking_file = tmp_path / "not_a_directory"
    blocking_file.write_text("")
    output = blocking_file / "gallery.png"

    result = _run_demo(["--output", str(output)])

    assert result.returncode == 1
    assert result.stdout == ""
    assert not output.exists()
    assert "error: could not write" in result.stderr


def test_build_source_scene_returns_the_documented_contract(demo_module: types.ModuleType) -> None:
    image, mask = demo_module.build_source_scene()

    assert image.shape == (120, 160, 3)
    assert image.dtype == np.uint8
    assert mask.shape == (120, 160)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask).tolist()) <= {0, 1, 2}


def test_build_demo_results_satisfies_the_geometric_contract(demo_module: types.ModuleType) -> None:
    image, mask = demo_module.build_source_scene()

    # build_demo_results asserts replay equality, shared image/mask geometry, and legal mask
    # labels internally -- a successful call is itself part of the contract under test.
    results = demo_module.build_demo_results(image, mask)

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


def test_demos_leave_no_files_behind_in_the_repository() -> None:
    # Mirrors tests/test_examples.py's shallow, non-recursive repo-cleanliness check: the
    # generator's default --output lands under docs/assets/, which is intentionally excluded
    # here since that committed asset is expected to exist; this only guards against the
    # generator writing anything unexpected next to its own source when given an explicit,
    # non-default --output (as every other test in this file does).
    before = {path for path in _DEMOS_DIR.iterdir() if path.name != "__pycache__"}

    with tempfile.TemporaryDirectory() as tmp:
        result = _run_demo(["--output", str(Path(tmp) / "gallery.png")])
        assert result.returncode == 0, result.stderr

    after = {path for path in _DEMOS_DIR.iterdir() if path.name != "__pycache__"}
    assert after == before, f"demo left new files behind: {after - before}"
