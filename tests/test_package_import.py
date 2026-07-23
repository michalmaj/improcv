import subprocess
import sys

import improcv


def test_version_matches_installed_package_metadata() -> None:
    from importlib.metadata import version

    assert improcv.__version__ == version("improcv")


def test_missing_cv2_raises_friendly_error() -> None:
    # sys.modules[name] = None is a documented Python import mechanism:
    # `import cv2` then raises ImportError immediately, without needing to
    # actually uninstall cv2 from the environment.
    script = "import sys\nsys.modules['cv2'] = None\nimport improcv\n"
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode != 0
    assert 'pip install "improcv[cv]"' in result.stderr
    last_line = result.stderr.strip().splitlines()[-1]
    assert last_line.startswith("ImportError:"), f"unexpected final line: {last_line!r}"


def test_broken_cv2_propagates_the_real_error_unmasked() -> None:
    # A present-but-broken cv2 (ABI mismatch, missing shared library,
    # corrupted install) raises a plain ImportError, not
    # ModuleNotFoundError -- improcv must not mask that with the "please
    # install one of these extras" message, since cv2 IS installed here;
    # the real underlying error is what the user needs to see.
    # A meta-path finder that raises during find_spec is closer to a real
    # broken native-extension load than monkeypatching sys.modules after
    # the fact.
    script = (
        "import sys\n"
        "import importlib.abc\n"
        "class _BrokenFinder(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path, target=None):\n"
        "        if name == 'cv2':\n"
        "            raise ImportError('simulated ABI mismatch: undefined symbol foo')\n"
        "        return None\n"
        "sys.meta_path.insert(0, _BrokenFinder())\n"
        "import improcv\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode != 0
    assert "pip install improcv[cv]" not in result.stderr
    assert "simulated ABI mismatch: undefined symbol foo" in result.stderr
