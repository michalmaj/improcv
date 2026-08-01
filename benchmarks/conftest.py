"""Pytest configuration for benchmarks/, loaded only when pytest is pointed at this directory.

`benchmarks/` sits outside `[tool.pytest.ini_options].testpaths`, so a plain `uv run pytest`
never collects it and never imports this file -- this is exactly why it is safe for this file,
and only this file, to set OpenCV's process-wide thread/OpenCL state: that only ever happens in
a dedicated benchmark process, never as a side effect of `import improcv` or of running the
normal test suite.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

import improcv

# Single-threaded, no-OpenCL baseline: eliminates variance from the number of cores available
# on whichever machine runs the benchmark. Read back immediately below rather than assumed --
# a build without OpenCL support, for example, would otherwise silently report a request that
# was never actually honored.
cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)

_OPENCV_NUM_THREADS = cv2.getNumThreads()
_OPENCV_OPENCL_ENABLED = cv2.ocl.useOpenCL()


def pytest_benchmark_update_machine_info(config: object, machine_info: dict[str, object]) -> None:
    """Add improcv-specific environment facts to pytest-benchmark's own `machine_info`.

    pytest-benchmark already records Python version/implementation/build, platform, and CPU
    (via `py-cpuinfo`) in `machine_info`, and commit/dirty state separately in `commit_info` --
    none of that is duplicated here. `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS`
    are recorded as the values actually found in the environment (`None` if unset), documenting
    intent -- not a claim that any particular NumPy/OpenCV build actually honors them.
    """
    machine_info["numpy_version"] = np.__version__
    machine_info["opencv_version"] = cv2.__version__
    machine_info["improcv_version"] = improcv.__version__
    machine_info["opencv_num_threads"] = _OPENCV_NUM_THREADS
    machine_info["opencv_opencl_enabled"] = _OPENCV_OPENCL_ENABLED
    machine_info["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS")
    machine_info["OPENBLAS_NUM_THREADS"] = os.environ.get("OPENBLAS_NUM_THREADS")
    machine_info["MKL_NUM_THREADS"] = os.environ.get("MKL_NUM_THREADS")
