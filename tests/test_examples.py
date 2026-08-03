"""Runs the examples/ scripts as real subprocesses -- they are executable documentation.

Each example is invoked with the current interpreter (`sys.executable`), exactly as a user running
`python examples/whatever.py` would, rather than imported and called in-process -- importing would
let a script accidentally skip its own `if __name__ == "__main__":` guard or rely on test-only
state that a real run never has. Exact floating-point metric values are intentionally not asserted
here by parsing stdout -- each script already asserts its own contract (shapes, finiteness,
replay equality, label sets) before printing anything, so these tests only check that the process
succeeded cleanly and printed the expected stable markers.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
_EXAMPLE_SCRIPTS = sorted(_EXAMPLES_DIR.glob("*.py"))


def _run_example(script: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_examples_directory_has_the_expected_scripts() -> None:
    names = {script.name for script in _EXAMPLE_SCRIPTS}
    assert names == {
        "discovery_and_augmentation.py",
        "classification_evaluation.py",
        "image_similarity.py",
        "image_similarity_manifest.py",
    }


def test_discovery_and_augmentation_example_runs_cleanly() -> None:
    result = _run_example(_EXAMPLES_DIR / "discovery_and_augmentation.py")

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert "pairs discovered: 2" in result.stdout
    assert "replay identical: yes" in result.stdout
    assert "source shape:" in result.stdout
    assert "expanded shape:" in result.stdout
    assert "mask labels:" in result.stdout


def test_classification_evaluation_example_runs_cleanly() -> None:
    result = _run_example(_EXAMPLES_DIR / "classification_evaluation.py")

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert "labels: [20, 10, 30]" in result.stdout
    assert "confusion matrix (rows=true, columns=predicted):" in result.stdout
    assert "accuracy:" in result.stdout
    assert "macro F1:" in result.stdout
    assert "ROC AUC per class:" in result.stdout
    assert "ROC AUC macro/weighted/micro:" in result.stdout
    assert "AP per class:" in result.stdout
    assert "AP macro/weighted/micro:" in result.stdout


def test_image_similarity_example_runs_cleanly() -> None:
    result = _run_example(_EXAMPLES_DIR / "image_similarity.py")

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert "images discovered: 4" in result.stdout
    assert "algorithm: average_hash" in result.stdout
    assert "hash size: 8 (64 bits)" in result.stdout
    assert "threshold: 2" in result.stdout
    assert "similar pairs: 2" in result.stdout
    assert "a_base.png" in result.stdout
    assert "b_variant.png" in result.stdout
    assert "c_variant_more.png" in result.stdout
    assert "non-transitive gap:" in result.stdout
    assert "= 4" in result.stdout
    assert "rehash required for threshold 4: no" in result.stdout

    # The unrelated image must never appear alongside a reported similar pair.
    pair_lines = [line for line in result.stdout.splitlines() if "<->" in line]
    assert not any("z_unrelated.png" in line for line in pair_lines), pair_lines


def test_image_similarity_manifest_example_runs_cleanly() -> None:
    result = _run_example(_EXAMPLES_DIR / "image_similarity_manifest.py")

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert "images discovered: 4" in result.stdout
    assert "images decoded: 4" in result.stdout
    assert "hashes computed: 4" in result.stdout
    assert "algorithm: average_hash" in result.stdout
    assert "hash size: 8 (64 bits)" in result.stdout
    assert "manifest entries: 4" in result.stdout
    assert "manifest paths relative: yes" in result.stdout
    assert "manifest round-trip identical: yes" in result.stdout
    assert "dataset moved after save: yes" in result.stdout
    assert "rehash after load: no" in result.stdout
    assert "threshold: 2" in result.stdout
    assert "similar pairs: 2" in result.stdout

    pair_lines = [line for line in result.stdout.splitlines() if "<->" in line]
    assert len(pair_lines) == 2, pair_lines
    assert any(
        "a_base.png" in line and "b_variant.png" in line and line.startswith("2 ")
        for line in pair_lines
    ), pair_lines
    assert any(
        "b_variant.png" in line and "c_variant_more.png" in line and line.startswith("2 ")
        for line in pair_lines
    ), pair_lines
    assert not any("z_unrelated.png" in line for line in pair_lines), pair_lines


def test_examples_leave_no_files_behind_in_the_repository() -> None:
    # Examples write only inside their own tempfile.TemporaryDirectory, never
    # into the repository -- a shallow, non-recursive snapshot of the repo
    # root and of examples/ itself is enough to catch a script that
    # accidentally creates something next to its own source (e.g. a stray
    # "dataset/" directory), without walking .git.
    repo_root = _EXAMPLES_DIR.parent
    before_root = {path for path in repo_root.iterdir() if path.name != "__pycache__"}
    before_examples = {path for path in _EXAMPLES_DIR.iterdir() if path.name != "__pycache__"}

    for script in _EXAMPLE_SCRIPTS:
        result = _run_example(script)
        assert result.returncode == 0, result.stderr

    after_root = {path for path in repo_root.iterdir() if path.name != "__pycache__"}
    after_examples = {path for path in _EXAMPLES_DIR.iterdir() if path.name != "__pycache__"}
    assert after_root == before_root, f"examples left new files behind: {after_root - before_root}"
    assert after_examples == before_examples, (
        f"examples left new files behind: {after_examples - before_examples}"
    )


@pytest.mark.parametrize("script", _EXAMPLE_SCRIPTS, ids=lambda script: script.name)
def test_example_is_importable_without_executing_main(script: Path) -> None:
    # A separate, lightweight sanity check that each script is syntactically
    # valid and loads without error on its own -- the real behavioral test is
    # the subprocess run above, which is what actually exercises `main()`.
    # runpy.run_path with run_name="not_main" executes the module's top-level
    # code (imports, function/constant definitions) under a __name__ other
    # than "__main__", so the `if __name__ == "__main__":` guard never fires
    # and main() is never called here.
    code = "import runpy, sys; runpy.run_path(sys.argv[1], run_name='not_main')"
    result = subprocess.run(
        [sys.executable, "-c", code, str(script)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
