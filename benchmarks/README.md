# Benchmarks

Opt-in, developer-only performance measurements for `improcv`. Currently covers three families --
affine augmentation, dataset discovery, and multiclass evaluation -- never part of the normal
test suite, never a runtime dependency, never a gate on ordinary PRs.

## Purpose

These benchmarks answer:

- What does `improcv`'s validation/contract cost on top of an equivalent raw OpenCV kernel call?
- How does that cost scale with image size?
- What does the second, label-safe warp of a segmentation mask cost on top of the image warp?
- What does the pure-Python geometry (`sample_affine`, `expand_affine_canvas`) cost, independent
  of any OpenCV kernel?
- How does `discover_images`'s full recursive traversal (fresh stat, extension filter, global
  sort) scale with file count?
- What does `discover_image_mask_pairs`'s additional pairing work (extension stripping,
  grouping, duplicate/key-set checks) cost on top of two plain traversals?
- How should a warm-filesystem-cache measurement be interpreted, given that it deliberately
  excludes cold-start cost?
- How does label-based evaluation (`confusion_matrix`, `classification_metrics`) scale with
  sample count?
- How does one-vs-rest ranking (`multiclass_roc_auc_score`, `multiclass_average_precision_score`)
  scale with sample count?
- How does that same ranking scale with class count, independent of sample count?
- What cost does each function's full public contract (explicit unsorted labels, type/value
  validation, no probability-simplex requirement, canonical order-independent reductions,
  read-only results) add, and does that cost remain part of every timed call?

## Non-goals

These benchmarks are **not**:

- a guarantee of performance on hardware other than the one they were run on;
- a marketing race against OpenCV;
- a comparison against scikit-learn (not a dependency of this project, and its contracts differ);
- a timing gate on ordinary pull requests;
- a measurement of inference or DNN workloads;
- an end-to-end dataset-workflow benchmark (discover -> load -> augment);
- a cold filesystem-cache measurement;
- a filesystem comparison between APFS/ext4/NTFS or any other pair of filesystems;
- a measurement of image decoding or dataset loading;
- a raw comparison against `os.walk`/`Path.rglob`/`glob` (see "Current scope" below for why
  discovery has no raw baseline);
- coverage of every `average` mode (`micro`/`weighted`/`None`) -- only `average="macro"` is
  measured in this first evaluation slice;
- a measurement with `sample_weight`;
- a measurement of binary curves (`roc_curve`/`precision_recall_curve`) or binary
  `roc_auc_score`/`average_precision_score`;
- a model-inference or end-to-end model-evaluation benchmark;
- a claim of proven asymptotic complexity from a handful of measured points.

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

## Running one family

Any family can be pointed at directly instead of all of `benchmarks/`. For discovery:

```bash
uv run --group benchmark pytest benchmarks/benchmark_discovery.py --benchmark-disable
```

For evaluation:

```bash
uv run --group benchmark pytest \
  benchmarks/benchmark_evaluation.py \
  --benchmark-disable
```

A short, non-baseline functional run (a handful of rounds, useful for a quick local sanity
check that timings scale sensibly, never for recording or comparing numbers) -- for evaluation:

```bash
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
uv run --group benchmark pytest \
  benchmarks/benchmark_evaluation.py \
  --benchmark-warmup=on \
  --benchmark-min-rounds=2 \
  --benchmark-max-time=0.05
```

Numbers from this short command are **not** a baseline -- too few rounds, no thread/OpenCL
pinning, no storage. Use the "Stable local run" commands below (pointed at the specific file)
for anything worth recording.

## Stable local run

A baseline worth recording uses an explicit warm-up phase and at least 20 rounds per case, run on
a laptop plugged into power, without heavy background load, and ideally without other CPU-bound
processes competing for the same cores -- none of this can be enforced by the harness itself.

There are two distinct native `pytest-benchmark` output modes for this, and they are not
interchangeable.

### Reviewed baseline candidate (default)

```bash
rm -rf /tmp/improcv-benchmark-storage

OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
uv run --group benchmark pytest benchmarks/ \
  --benchmark-warmup=on \
  --benchmark-min-rounds=20 \
  --benchmark-storage=file:///tmp/improcv-benchmark-storage \
  --benchmark-save=augmentation-baseline
```

This writes a **saved run** under
`/tmp/improcv-benchmark-storage/<platform-tag>/NNNN_augmentation-baseline.json` (the exact
subdirectory name depends on platform/Python; find it with `find /tmp/improcv-benchmark-storage
-type f`). It is a native, unedited `pytest-benchmark` output -- not something this project
post-processes or reduces -- that, by default, contains `machine_info`, `commit_info`, per-case
`group`/`params`/`extra_info`, and summary statistics (`median`/`mean`/`stddev`/`iqr`/`rounds`/
etc.) for every case, but **not** the full array of every individual round's timing
(`stats.data` is absent from a plain `--benchmark-save`, confirmed directly against this
project's `pytest-benchmark` version). This is the default candidate for committing after review:
copy the generated file byte-for-byte into `benchmarks/results/` (never edited, never manually
stripped of anything) once its exact commit SHA is clean (`dirty: false`), its machine metadata is
captured, and it has an accompanying reviewed Markdown report -- the same immutable-results policy
already in effect (see `benchmarks/results/README.md`).

### Full-data diagnostic capture (optional)

```bash
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
uv run --group benchmark pytest benchmarks/ \
  --benchmark-warmup=on \
  --benchmark-min-rounds=20 \
  --benchmark-json=/tmp/improcv-augmentation-full-data.json
```

`--benchmark-json` writes the *complete* per-round timing array (`stats.data`) for every case,
alongside the same machine/commit/per-case metadata as the saved run above. For a fast case run
tens of thousands of times, this can be tens of megabytes -- `results/2026-08-01-augmentation-
baseline.json` (the first-ever baseline, kept as a historical exception; see "Results" below) is
one such file, at ~20.9 MB. It is genuinely useful for diagnosing outliers and the shape of the
per-round distribution, but it stays **outside the repository by default**, and is only committed
when there is a specific, reviewed diagnostic reason to keep the full distribution rather than its
summary statistics.

Passing `--benchmark-save-data` alongside `--benchmark-save`/`--benchmark-storage` produces the
same full-data effect *inside* a saved run -- confirmed directly: a saved run captured with
`--benchmark-save-data` is the same size class as a plain `--benchmark-json` capture, not the
compact saved run above. For that reason, `--benchmark-save-data` is deliberately absent from the
default baseline command.

The three thread-count environment variables, together with `benchmarks/conftest.py`'s own
`cv2.setNumThreads(1)`/`cv2.ocl.setUseOpenCL(False)` calls at session start, document *intent* --
they are not a guarantee that this environment's active OpenCV backend actually honors them. The
harness requests one OpenCV thread and disables OpenCL, then records the state *actually*
reported by the active backend afterward (`cv2.getNumThreads()`/`cv2.ocl.useOpenCL()`), and writes
both the request and the observation into `machine_info`
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

### Affine augmentation

Three groups, all affine, at three sizes (`64x64`, `640x480`, `1920x1080`) unless noted:

- `affine-python-geometry` -- `sample_affine` (non-degenerate ranges) and
  `expand_affine_canvas`, measured on their own; there is no equivalent raw OpenCV operation to
  compare either against.
- `affine-image-only` -- `apply_affine` (image only) vs. a single raw `cv2.warpAffine` call, at
  all three sizes.
- `affine-image-mask` -- `apply_affine` (image + mask) vs. two raw `cv2.warpAffine` calls (the
  mask call forced to `INTER_NEAREST`, matching `apply_affine`'s own contract), at all three
  sizes.

Perspective (`sample_perspective`/`apply_perspective`/`cv2.warpPerspective`) remains a possible
later extension, but the next added family was discovery scaling (below) instead -- it exercises
filesystem traversal, sorting, and pairing behavior not represented by the affine baseline at
all, which perspective would not have added.

### Dataset discovery

Two groups, at three entry counts (`100`, `1,000`, `10,000`), defined in
`benchmark_discovery.py`:

- `discovery-images` -- `discover_images` over a single root of zero-byte, extension-only
  discovery entries, split across 10 shard directories.
- `discovery-pairs` -- `discover_image_mask_pairs` over a matching pair of image/mask roots
  (each with the same shard layout), producing the same entry count in pairs.

Every entry is a zero-byte file created with `Path.touch()`, split across exactly 10 shard
directories per root (`shard_00` .. `shard_09`) with `sample_NNNNNN.jpg`/`.png` filenames --
never a valid encoded image, since `discover_images` finds files by extension only and never
opens, decodes, or otherwise inspects their content. Both benchmarks perform one untimed,
asserted preflight call before the timed one, to validate the dataset and warm the filesystem
metadata cache -- these are **warm filesystem-cache** measurements only, never cold-start ones
(see "Non-goals" above).

There is no raw `os.walk`/`Path.rglob`/`glob` baseline for either group: `discover_images`'s
contract (fresh per-entry `os.stat(..., follow_symlinks=False)`, symlink/reparse-point
skipping, a hidden-file policy, a deterministic global POSIX-relative sort, extension
normalization) and `discover_image_mask_pairs`'s additional strict-bijection pairing have no
raw equivalent of matching strength -- a ratio against a semantically weaker iterator would not
be a meaningful comparison. This first discovery slice measures how the public API itself
scales with entry count.

### Multiclass evaluation

Four groups, defined in `benchmark_evaluation.py`, along two independent scaling axes:

- `evaluation-confusion-matrix` -- `confusion_matrix`, at 1,000/10,000/100,000 samples, fixed 10
  classes.
- `evaluation-classification-metrics` -- `classification_metrics(..., average="macro")`, the
  same sample-count axis, sharing its dataset with `evaluation-confusion-matrix`.
- `evaluation-roc-auc-macro` -- `multiclass_roc_auc_score(..., average="macro")`, at the same
  three sample counts (fixed 10 classes) plus 3/10/100 classes (fixed 10,000 samples) -- five
  scenarios total, `(10_000, 10)` shared between the two axes rather than duplicated.
- `evaluation-average-precision-macro` -- `multiclass_average_precision_score(...,
  average="macro")`, the same five scenarios, sharing its dataset with `evaluation-roc-auc-macro`.

Every dataset uses an explicit, deliberately **unsorted** `labels` tuple
(`(1, 2, ..., n_classes - 1, 0)`) -- this exercises the real explicit-label contract, not a
sorted-by-coincidence shortcut. `y_true`/`y_pred` are `int64` ndarrays from a balanced-by-
construction class assignment with a deterministic every-fifth-sample error policy (no
randomness). `y_score` (ranking scenarios only) is a seeded, `float64`, C-contiguous
`(n_samples, n_classes)` matrix with a `+0.75` boost on each row's true-class column -- rows are
deliberately **not** probability-normalized, since neither ranking function requires a
probability simplex.

There is no raw NumPy or scikit-learn baseline: no single reference implementation shares this
API's full contract (explicit unsorted `labels`, full validation, no probability-simplex
requirement, canonical order-independent reductions, read-only results) -- `sklearn.metrics.
roc_auc_score`'s multiclass mode, for comparison, requires row-normalized probabilities and a
pre-sorted explicit `labels`, two real contract differences that would make any ratio compare
different semantics rather than the same workflow. This first evaluation slice measures how the
public API itself scales with sample and class count. Only `average="macro"` is measured; no
`sample_weight`.

## Results

### Affine augmentation

The first reviewed baseline is captured:

- Raw JSON: [`results/2026-08-01-augmentation-baseline.json`](results/2026-08-01-augmentation-baseline.json)
- Reviewed report: [`results/2026-08-01-augmentation-baseline.md`](results/2026-08-01-augmentation-baseline.md)

The reviewed narrative report contains the interpretation and tables; the JSON remains the raw
source of truth. See `benchmarks/results/README.md` for the policy governing committed results.

That first JSON is a **full-data capture** (~20.9 MB, generated with `--benchmark-json`,
including the complete per-round timing array for all 14 cases) -- kept exactly as it is, as a
historical exception documenting the very first run in maximum detail and enabling per-round
outlier analysis (see the report's own "Measurement spread" section). It is not a mistake and
will not be replaced or trimmed. It is also not the default going forward: subsequent baselines
are expected to use the compact, stats-only saved-run format described in "Stable local run"
above, reserving a full-data capture for cases with a specific, reviewed diagnostic need.

Three observations from that specific machine and run, elaborated on in the report itself:

- The observed wrapper/raw ratio was largest at `64x64` (where the raw kernel itself is only a
  few microseconds) and close to 1.0 at `640x480`/`1920x1080` (where the kernel dominates).
- The image+mask cases, which perform two warps instead of one, had correspondingly higher raw
  and `improcv` medians than the matching image-only case at every size.
- One case (`apply_affine_image_mask[raw-64x64]`) had a much higher coefficient of variation than
  the rest, traced to a small fraction (~0.1%) of rounds affected by ordinary OS scheduling
  jitter -- its median/IQR (this project's primary statistics) were unremarkable.

### Dataset discovery

The first reviewed baseline is captured:

- Compact JSON: [`results/2026-08-01-discovery-baseline.json`](results/2026-08-01-discovery-baseline.json)
- Reviewed report: [`results/2026-08-01-discovery-baseline.md`](results/2026-08-01-discovery-baseline.md)

Captured at the exact harness commit `50a0a2bc48e8c49b9a26b3f7c8284107e6f5bfce`, with a clean
working tree (`commit_info.dirty: false`). This is a **stats-only saved run**
(`--benchmark-save`/`--benchmark-storage`, no `--benchmark-save-data`) -- `stats.data` (the full
per-round timing array) is confirmed absent from every entry; the reviewed Markdown report
contains the per-case tables and scaling interpretation, the JSON remains the raw source of
truth. See `benchmarks/results/README.md` for the policy governing committed results.

Three observations from that specific machine and run, elaborated on in the report itself:

- Both `discovery-images` and `discovery-pairs` medians increased monotonically across all three
  measured sizes (100/1,000/10,000).
- Each 10x increase in entry count produced an observed median growth close to 10x for both
  groups (`discover_images`: 9.62x then 10.3x; `discover_image_mask_pairs`: 9.86x then 10.4x) --
  approximately proportional over the measured range, not a proven asymptotic complexity claim
  from three data points.
- `discover_image_mask_pairs` traverses two roots and additionally builds/validates/sorts strict
  pairing keys; its median was consistently a few times higher than `discover_images`'s median at
  the matching per-root count on this run, but that is not treated as an isolated "pairing
  overhead" figure (see the report's "Interpretation" section for why).

### Multiclass evaluation

No committed evaluation baseline yet -- this PR adds only the harness, data, grouping, and this
documentation. A first evaluation baseline will follow in a separate PR, using the same compact,
stats-only saved-run format described above (`--benchmark-save`/`--benchmark-storage`, no
`stats.data`); a full-data capture would only be used again for a specific, reviewed diagnostic
need, exactly as for the affine and discovery families.

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
about timing, and produces no JSON. This collects all three families (20 affine cases + 6
discovery cases + 16 evaluation cases = 42). The normal `uv run pytest` used everywhere else
never touches `benchmarks/` at all.
