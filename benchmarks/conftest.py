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

# Request one OpenCV thread and disable OpenCL, then record the values actually reported by the
# active OpenCV backend -- not every OpenCV backend honors a thread-count request (some parallel
# frameworks manage their own thread pool outside this API), so "requested" and "observed" are
# tracked as two separate facts, never conflated into a single "single-threaded baseline" claim.
# Raw and improcv calls within one run still share the same process and the same actual active
# configuration, whatever it turns out to be -- that comparability holds regardless of whether
# the request was actually honored. Both requested and observed values are recorded in
# machine_info below so a committed result is always interpretable on its own terms.
_OPENCV_REQUESTED_NUM_THREADS = 1
_OPENCV_REQUESTED_OPENCL_ENABLED = False

cv2.setNumThreads(_OPENCV_REQUESTED_NUM_THREADS)
cv2.ocl.setUseOpenCL(_OPENCV_REQUESTED_OPENCL_ENABLED)

_OPENCV_OBSERVED_NUM_THREADS = cv2.getNumThreads()
_OPENCV_OBSERVED_OPENCL_ENABLED = cv2.ocl.useOpenCL()


def pytest_benchmark_update_machine_info(config: object, machine_info: dict[str, object]) -> None:
    """Add improcv-specific environment facts to pytest-benchmark's own `machine_info`.

    pytest-benchmark already records Python version/implementation/build, platform, and CPU
    (via `py-cpuinfo`) in `machine_info`, and commit/dirty state separately in `commit_info` --
    none of that is duplicated here. `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS`
    are recorded as the values actually found in the environment (`None` if unset), documenting
    intent -- not a claim that any particular NumPy/OpenCV build actually honors them. Likewise,
    `opencv_requested_*`/`opencv_*` record what was asked for and what the backend actually did
    with it -- they are not guaranteed to match.
    """
    machine_info["numpy_version"] = np.__version__
    machine_info["opencv_version"] = cv2.__version__
    machine_info["improcv_version"] = improcv.__version__
    machine_info["opencv_requested_num_threads"] = _OPENCV_REQUESTED_NUM_THREADS
    machine_info["opencv_num_threads"] = _OPENCV_OBSERVED_NUM_THREADS
    machine_info["opencv_requested_opencl_enabled"] = _OPENCV_REQUESTED_OPENCL_ENABLED
    machine_info["opencv_opencl_enabled"] = _OPENCV_OBSERVED_OPENCL_ENABLED
    machine_info["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS")
    machine_info["OPENBLAS_NUM_THREADS"] = os.environ.get("OPENBLAS_NUM_THREADS")
    machine_info["MKL_NUM_THREADS"] = os.environ.get("MKL_NUM_THREADS")
