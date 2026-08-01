# Benchmarks

Opt-in, developer-only performance measurements for `improcv`'s affine augmentation API. Never
part of the normal test suite, never a runtime dependency, never a gate on ordinary PRs.

## Purpose

These benchmarks answer:

- What does `improcv`'s validation/contract cost on top of an equivalent raw OpenCV kernel call?
- How does that cost scale with image size?
- What does the second, label-safe warp of a segmentation mask cost on top of the image warp?
- What does the pure-Python geometry (`sample_affine`, `expand_affine_canvas`) cost, independent
  of any OpenCV kernel?

## Non-goals

These benchmarks are **not**:

- a guarantee of performance on hardware other than the one they were run on;
- a marketing race against OpenCV;
- a comparison against scikit-learn (not a dependency of this project, and its contracts differ);
- a timing gate on ordinary pull requests;
- a measurement of inference or DNN workloads;
- an end-to-end dataset-workflow benchmark (discover -> load -> augment).

## Installation and smoke

`pytest-benchmark` lives in its own opt-in dependency group, never in `dev` and never a runtime
dependency or public extra:

```bash
uv sync --group benchmark
```

To confirm the harness collects and runs without measuring anything (no statistics, no JSON, each
case executed exactly once):

```bash
uv run --group benchmark pytest benchmarks/ --benchmark-disable
```

This is also what CI runs (see "CI" below) -- it is a correctness/collection smoke check, not a
timing run.

## Stable local run

A baseline worth recording uses an explicit warm-up phase and at least 20 rounds per case, with
the JSON written outside the repository. `benchmarks/conftest.py` requests a single-threaded,
OpenCL-disabled OpenCV at session start and records what was *actually* achieved, not just what
was requested -- on at least one real development machine (macOS, OpenCV 5.0.0, `Parallel
framework: GCD`), `cv2.setNumThreads(1)` is a documented no-op and `cv2.getNumThreads()` still
reports the full core count afterward, while `cv2.ocl.setUseOpenCL(False)` does take effect. Both
observed values end up in the committed JSON's `machine_info` regardless of which way they go:

```bash
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
uv run --group benchmark pytest benchmarks/ \
  --benchmark-warmup=on \
  --benchmark-min-rounds=20 \
  --benchmark-json=/tmp/improcv-augmentation-benchmark.json
```

The three thread-count environment variables document *intent*, not a guarantee that every
NumPy/OpenCV build actually honors them -- the real, observed OpenCV thread count and OpenCL
state are recorded in the JSON regardless (see `benchmarks/conftest.py`). `NUMEXPR_NUM_THREADS`
is deliberately not set: this project does not depend on NumExpr.

Run on a laptop, this also means: plugged into power, without heavy background load, and ideally
without other CPU-bound processes competing for the same cores -- none of this can be enforced by
the harness itself.

## Interpretation

```text
ratio = median(improcv) / median(raw)
```

- The **median** across rounds is the primary statistic (robust to a single anomalous round);
  IQR/standard deviation (both reported by `pytest-benchmark`) describe the spread around it.
- The ratio includes *everything* `improcv` does beyond the bare kernel call: parameter/shape/
  dtype validation, Python-level dispatch, and construction of the returned value -- it is not a
  narrower "validation-only" cost, and is never reported as one.
- Every raw/`improcv` pair in this file shares the same image, mask, and `AffineParameters`
  object, and the raw side always performs the *same* warp (matrix, `dsize`, interpolation,
  border mode and value) -- see the correctness tests in `benchmark_augmentation.py`, which
  assert `np.array_equal` between the two sides for every case.
- Results from different machines are **not** directly comparable to each other -- CPU, OS,
  OpenCV build, and background load all differ. Compare a machine only against its own earlier
  results.

## Current scope

Three groups, all affine, at three sizes (`64x64`, `640x480`, `1920x1080`) unless noted:

- `affine-python-geometry` -- `sample_affine` (non-degenerate ranges) and
  `expand_affine_canvas`, measured on their own; there is no equivalent raw OpenCV operation to
  compare either against.
- `affine-image-only` -- `apply_affine` (image only) vs. a single raw `cv2.warpAffine` call, at
  all three sizes.
- `affine-image-mask` -- `apply_affine` (image + mask) vs. two raw `cv2.warpAffine` calls (the
  mask call forced to `INTER_NEAREST`, matching `apply_affine`'s own contract), at all three
  sizes.

Perspective (`sample_perspective`/`apply_perspective`/`cv2.warpPerspective`) is deliberately out
of scope for this first slice -- affine already exercises the tool, the parameter contract, the
raw/wrapper grouping, scaling, and both image-only and image+mask cases. It is expected to follow
as a small, separate extension once this baseline has been reviewed.

## Results

No committed performance baseline yet. The first reviewed baseline (raw JSON under
`benchmarks/results/`, named `YYYY-MM-DD-augmentation-baseline.json`) will be added in a
follow-up PR, after this harness itself has been reviewed. See `benchmarks/results/README.md`
for the policy that will govern it.

## Future engineering stories

Once a real optimization is made based on these benchmarks, it will be documented here in the
same shape used elsewhere in this project's C++ sibling's own performance notes:

```text
problem -> measurement -> optimization -> result
```

No examples or numbers are invented ahead of an actual case.

## CI

CI runs exactly one non-timing smoke job: `uv run --group benchmark pytest benchmarks/
--benchmark-disable` on a single platform/Python/OpenCV combination. It checks that the harness
imports, collects, and its fixtures and correctness assertions still pass -- it asserts nothing
about timing, and produces no JSON. The normal `uv run pytest` used everywhere else never touches
`benchmarks/` at all.
