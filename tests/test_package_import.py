import subprocess
import sys


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
    assert "pip install improcv[cv]" in result.stderr
    last_line = result.stderr.strip().splitlines()[-1]
    assert last_line.startswith("ImportError:"), f"unexpected final line: {last_line!r}"
