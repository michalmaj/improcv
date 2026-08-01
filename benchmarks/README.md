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
the JSON written outside the repository:

```bash
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
uv run --group benchmark pytest benchmarks/ \
  --benchmark-warmup=on \
  --benchmark-min-rounds=20 \
  --benchmark-json=/tmp/improcv-augmentation-benchmark.json
```

The three thread-count environment variables, together with `benchmarks/conftest.py`'s own
`cv2.setNumThreads(1)`/`cv2.ocl.setUseOpenCL(False)` calls at session start, document *intent* --
they are not a guarantee that this environment's active OpenCV backend actually honors them. The
harness requests one OpenCV thread and disables OpenCL, then records the state *actually*
reported by the active backend afterward (`cv2.getNumThreads()`/`cv2.ocl.useOpenCL()`), and writes
both the request and the observation into the JSON's `machine_info`
(`opencv_requested_num_threads` vs. `opencv_num_threads`,
`opencv_requested_opencl_enabled` vs. `opencv_opencl_enabled`) -- never only one of the two.

On the documented macOS/OpenCV 5 GCD build (`Parallel framework: GCD` in
`cv2.getBuildInformation()`), the thread request was ignored and `cv2.getNumThreads()` still
reported 12 after the call, while the OpenCL request did take effect. This is not a bug in the
harness -- it is a real property of that OpenCV build's parallel framework, and it is exactly why
request and observation are tracked separately instead of asserted equal. A result like this one
is a valid baseline **for the observed 12-thread/GCD configuration**, not for a single-threaded
one; it should only be compared later against results captured under a matching observed
configuration, not assumed comparable to a result from a build where the thread request actually
succeeded. Raw and `improcv` calls within the *same* run remain both semantically and
environmentally comparable to each other regardless -- they run in the same process against
whatever the actual active OpenCV configuration turns out to be.
`NUMEXPR_NUM_THREADS` is deliberately not set: this project does not depend on NumExpr.

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
- `machine_info` records both what was *requested* (`opencv_requested_num_threads`,
  `opencv_requested_opencl_enabled`) and what was *observed* (`opencv_num_threads`,
  `opencv_opencl_enabled`) -- a result is interpreted and compared using the observed values, not
  the requested ones, since the two are not guaranteed to match (see "Stable local run" above).
- Each benchmark entry's `group` field (`affine-python-geometry`/`affine-image-only`/
  `affine-image-mask`) is a real `pytest-benchmark` grouping, not just a naming convention -- raw
  and `improcv` cases for the same size and operation always share one group.

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

The first reviewed baseline is captured:

- Raw JSON: [`results/2026-08-01-augmentation-baseline.json`](results/2026-08-01-augmentation-baseline.json)
- Reviewed report: [`results/2026-08-01-augmentation-baseline.md`](results/2026-08-01-augmentation-baseline.md)

The reviewed narrative report contains the interpretation and tables; the JSON remains the raw
source of truth. See `benchmarks/results/README.md` for the policy governing committed results.

Three observations from that specific machine and run, elaborated on in the report itself:

- The observed wrapper/raw ratio was largest at `64x64` (where the raw kernel itself is only a
  few microseconds) and close to 1.0 at `640x480`/`1920x1080` (where the kernel dominates).
- The image+mask cases, which perform two warps instead of one, had correspondingly higher raw
  and `improcv` medians than the matching image-only case at every size.
- One case (`apply_affine_image_mask[raw-64x64]`) had a much higher coefficient of variation than
  the rest, traced to a small fraction (~0.1%) of rounds affected by ordinary OS scheduling
  jitter -- its median/IQR (this project's primary statistics) were unremarkable.

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
