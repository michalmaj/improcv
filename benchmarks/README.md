# Benchmarks

Opt-in, developer-only performance measurements for `improcv`. Currently covers five families --
affine augmentation, perceptual hashing, pairwise image similarity, dataset discovery, and
multiclass evaluation -- never part of the normal test suite, never a runtime dependency, never a
gate on ordinary PRs.

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
- How does `average_hash` scale with the size of a BGR `uint8` source image, at a fixed
  `hash_size=8`?
- How does `phash` scale on that same source-image-size axis, at the same fixed `hash_size`?
- What does each hashing function's full public contract cost -- resize, grayscale conversion,
  threshold computation, bit packing, and `PerceptualHash` construction -- as one measured call,
  not a sum of hand-isolated steps?
- How should a fixed, small target hash grid (`8x8`, or `32x32` before pHash's DCT) be
  interpreted against a growing source image -- does hashing cost track source pixels, or is it
  dominated by the fixed-size resize target?
- How does the complete public `find_similar_image_pairs` call scale with the number of
  precomputed hashes it searches?
- How does the number of unordered comparisons (`n(n-1)/2`) drive that cost?
- How does the workflow behave when the search result is empty (no pair within threshold)?
- How does the workflow behave when every unordered pair is materialized into the result?
- What cost remains in the complete public call -- path normalization, `algorithm`/`hash_size`
  compatibility validation, input sorting, every Hamming comparison, and result
  construction/sorting -- as one measured call, not a sum of hand-isolated steps?

This does not measure how a perceptual hash is computed (see "Perceptual hashing" above) --
`find_similar_image_pairs` only ever consumes already-computed `PerceptualHash` values.

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
- a claim of proven asymptotic complexity from a handful of measured points;
- a measurement of image decoding;
- a measurement of Hamming distance (`PerceptualHash.distance`) on its own, as a standalone
  microbenchmark;
- a measurement of dataset discovery combined with hashing (an end-to-end hashing workflow);
- coverage of `hash_size` as a scaling axis -- only the default `hash_size=8` is measured;
- a grayscale/BGRA/channel-count comparison, or a dtype comparison beyond `uint8`;
- a comparison against `cv2.img_hash` or any other `opencv-contrib-python`-gated implementation;
- a measurement of perceptual robustness, collision rate, or hash-quality/accuracy;
- a recommendation of a universal similarity threshold or distance cutoff;
- an end-to-end discovery/decode/hash/search workflow benchmark;
- a measurement of perceptual hash computation (`average_hash`/`phash`) as part of the
  similarity-search cases -- `find_similar_image_pairs` only ever consumes already-computed
  hashes;
- any match-density scenario between the two measured extremes (`0%` and `100%`);
- coverage of any `max_distance` other than `0` and `64` (the maximum legal threshold for
  `hash_size=8`);
- coverage of any hashing algorithm other than `PHASH`, or any `hash_size` other than `8`, in the
  similarity-search cases;
- a comparison of concrete `Mapping` types, or a `str`-vs-`Path` key-type axis;
- duplicate grouping, clustering, or connected-component analysis;
- a BK-tree, approximate-nearest-neighbor, or any other subquadratic/indexed search structure;
- parallelism of any kind;
- a measurement of real-world duplicate-detection accuracy.

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

For perceptual hashing:

```bash
uv run --group benchmark pytest \
  benchmarks/benchmark_hashing.py \
  --benchmark-disable
```

For pairwise image similarity:

```bash
uv run --group benchmark pytest \
  benchmarks/benchmark_similarity.py \
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

A reviewed baseline is captured **one family at a time**, with the saved run's name matching
that family -- `augmentation`, `discovery`, `evaluation`, `hashing`, and `similarity` are the
current values. Running the whole `benchmarks/` directory (as in "Installation and smoke" above)
is still correct
for a quick local review across families, but it is not how a committed baseline is produced: a
single saved run named after all of `benchmarks/` at once would be ambiguous about which family
(or families) it actually captured, now that there is more than one.

```bash
FAMILY=evaluation

rm -rf "/tmp/improcv-${FAMILY}-benchmark-storage"

OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
uv run --group benchmark pytest \
  "benchmarks/benchmark_${FAMILY}.py" \
  --benchmark-warmup=on \
  --benchmark-min-rounds=20 \
  --benchmark-storage="file:///tmp/improcv-${FAMILY}-benchmark-storage" \
  --benchmark-save="${FAMILY}-baseline"
```

This writes a **saved run** under
`/tmp/improcv-${FAMILY}-benchmark-storage/<platform-tag>/NNNN_${FAMILY}-baseline.json` (the exact
subdirectory name depends on platform/Python; find it with `find /tmp/improcv-${FAMILY}-
benchmark-storage -type f`). It is a native, unedited `pytest-benchmark` output -- not something
this project post-processes or reduces -- that, by default, contains `machine_info`,
`commit_info`, per-case `group`/`params`/`extra_info`, and summary statistics (`median`/`mean`/
`stddev`/`iqr`/`rounds`/etc.) for every case, but **not** the full array of every individual
round's timing (`stats.data` is absent from a plain `--benchmark-save`, confirmed directly
against this project's `pytest-benchmark` version). This is the default candidate for committing
after review: copy the generated file byte-for-byte into `benchmarks/results/` (never edited,
never manually stripped of anything) once its exact commit SHA is clean (`dirty: false`), its
machine metadata is captured, and it has an accompanying reviewed Markdown report -- the same
immutable-results policy already in effect (see `benchmarks/results/README.md`).

### Full-data diagnostic capture (optional)

Also family-specific, for the same reason as above:

```bash
FAMILY=augmentation

OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
uv run --group benchmark pytest \
  "benchmarks/benchmark_${FAMILY}.py" \
  --benchmark-warmup=on \
  --benchmark-min-rounds=20 \
  --benchmark-json="/tmp/improcv-${FAMILY}-full-data.json"
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

### Perceptual hashing

Two groups, defined in `benchmark_hashing.py`, at three source image sizes (`64x64`, `640x480`,
`1920x1080`):

- `hashing-average-hash` -- the complete public `average_hash(image, hash_size=8)` call.
- `hashing-phash` -- the complete public `phash(image, hash_size=8)` call.

`hash_size` is fixed at `8` throughout -- hash-size scaling is a distinct question, out of scope
for this first hashing slice (see "Non-goals" above). Every source image is a deterministic,
seeded, C-contiguous `(height, width, 3)` `uint8` BGR array, built once per size in a
session-scoped fixture shared by both groups -- never read from or written to disk, and no
encoded image format is involved anywhere in this file. The timed region for each case is the
complete public function call: input/`hash_size` validation, resize, BGR-to-grayscale
conversion, every algorithm-specific step (`average_hash`'s mean/threshold; `phash`'s DCT/block
selection/threshold), bit packing, and `PerceptualHash` construction -- nothing is extracted or
pre-computed outside the timed call.

There is no raw baseline for either group (no hand-written NumPy/OpenCV pipeline, no
`cv2.img_hash`): no single raw kernel corresponds to either function's complete public contract,
since both `average_hash` and `phash` are themselves multi-step pipelines -- reimplementing that
pipeline by hand in a benchmark would just duplicate the implementation, not provide an
independent reference. `cv2.img_hash` additionally requires `opencv-contrib-python` (not a
dependency of this project) and returns a packed-byte result through a different API shape, not
an `improcv.PerceptualHash` -- a ratio against it would compare two different result types under
two different dependency footprints, not a same-contract raw/wrapper pair. This first hashing
slice measures how the public API itself scales with source image size.

`find_similar_image_pairs` -- the pair-search step that consumes already-computed hashes -- is a
distinct scaling question, benchmarked separately below.

### Pairwise image similarity

Two groups, defined in `benchmark_similarity.py`, at three item counts (`30`, `100`, `300`
precomputed hashes, i.e. `435`/`4,950`/`44,850` unordered pairs):

- `similarity-no-matches` -- the complete public `find_similar_image_pairs(hashes,
  max_distance=0)` call, where every hash is unique and the result is always empty.
- `similarity-all-matches` -- the complete public `find_similar_image_pairs(hashes,
  max_distance=64)` call (`64 == hash_size**2`, the maximum legal threshold for `hash_size=8`),
  where every unordered pair is materialized into the result.

Both regimes at a given item count share exactly the same input mapping, built once per session
and inserted in reverse canonical (`path.as_posix()`) order, so the timed call must actually
normalize and sort the input rather than benefit from an already-sorted mapping. Every hash value
is a synthetic, legal `PHASH` object built from a deterministic, guaranteed-unique integer
transform (see the module docstring) -- `average_hash`/`phash` are never called, no image or
NumPy array exists anywhere in this file, and `discover_images` is never called. Path identifiers
(`images/image_NNNNN.png`) are synthetic and never opened, created, or otherwise touched --
`find_similar_image_pairs` performs no filesystem access at all. The timed region for each case
is the complete public function call: `max_distance` validation, `Mapping` validation, path
normalization, duplicate-key detection, per-hash validation, the shared `algorithm`/`hash_size`
check, the threshold upper-bound check, input sorting, every unordered-pair enumeration, every
`PerceptualHash.distance` call, the threshold branch, `SimilarImagePair` construction where it
matches, the final result sort, and the tuple conversion -- nothing is extracted or pre-computed
outside the timed call except the hash mapping itself (unavoidable, since the public API's
contract starts from already-computed hashes).

There is no raw baseline here (no hand-written `itertools.combinations` loop, no private access
to `PerceptualHash._value`): such a version would duplicate a fragment of `find_similar_image_
pairs`'s own implementation, skip its path normalization and compatibility validation, and return
raw tuples instead of `SimilarImagePair` -- not the same public workflow. **The all-matches/
no-matches contrast compares two complete public workflows with different result cardinalities
and branch outcomes. It does not isolate an exact per-object materialization cost.** There is no
grouping/clustering, and no intermediate match-density scenario between the two measured
extremes -- see "Non-goals" above.

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

### Perceptual hashing

The first reviewed baseline is captured:

- Compact JSON: [`results/2026-08-02-hashing-baseline.json`](results/2026-08-02-hashing-baseline.json)
- Reviewed report: [`results/2026-08-02-hashing-baseline.md`](results/2026-08-02-hashing-baseline.md)

Captured at the exact harness commit `c0a07a5a506b20aea15a8572c68a22d9eea641ce`, with a clean
working tree (`commit_info.dirty: false`). This is a **stats-only saved run**
(`--benchmark-save`/`--benchmark-storage`, no `--benchmark-save-data`) -- `stats.data` (the full
per-round timing array) is confirmed absent from every entry; the reviewed Markdown report
contains the per-case tables and scaling interpretation, the JSON remains the raw source of
truth. See `benchmarks/results/README.md` for the policy governing committed results.

Three observations from that specific machine and run, elaborated on in the report itself:

- Both `average_hash` and `phash` medians increased monotonically across the three measured
  source sizes, but far from proportionally to source pixel count: `average_hash`'s median grew
  only ~1.01x-1.05x and `phash`'s only ~1.06x-1.37x across two pixel-growth steps of 75x and
  6.75x, respectively.
- `median/pixel` fell steeply for both functions as source size grew, consistent with each
  function reducing its input to a small, fixed target grid (`8x8` for `average_hash`, `32x32`
  for `phash`) before doing its algorithm-specific work.
- `phash`'s median was consistently 2.0x-2.7x `average_hash`'s median across all three sizes --
  reported only as an observed ratio between two complete, different workflows, not as an
  isolated DCT cost or a claim that either algorithm is generally faster or better.

### Pairwise image similarity

No committed similarity baseline yet. This harness (`benchmark_similarity.py`) is newly added; a
short local smoke run was used only to confirm the harness collects, runs, and produces the
expected stats-only shape -- that smoke capture is not a reviewed baseline and is not committed.
A reviewed, stats-only baseline capture (committed JSON plus its accompanying Markdown report,
from a clean, finalized harness commit, following the same policy as the other families here)
will follow as a separate PR once this harness has been merged.

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

The first reviewed baseline is captured:

- Compact JSON: [`results/2026-08-02-evaluation-baseline.json`](results/2026-08-02-evaluation-baseline.json)
- Reviewed report: [`results/2026-08-02-evaluation-baseline.md`](results/2026-08-02-evaluation-baseline.md)

Captured at the exact harness commit `658a6bcc6b943a1f9e232be51149b2d52d1a08d2`, with a clean
working tree (`commit_info.dirty: false`). This is a **stats-only saved run**
(`--benchmark-save`/`--benchmark-storage`, no `--benchmark-save-data`) -- `stats.data` (the full
per-round timing array) is confirmed absent from every entry; the reviewed Markdown report
contains the per-case tables and scaling interpretation, the JSON remains the raw source of
truth. See `benchmarks/results/README.md` for the policy governing committed results.

Three observations from that specific machine and run, elaborated on in the report itself:

- All four operations' medians increased monotonically across the measured sample counts
  (1,000/10,000/100,000), and both ranking operations' medians increased monotonically across
  the measured class counts (3/10/100).
- Each 10x increase in sample count produced an observed median growth in the 8.3x-10.7x range
  across all four operations -- approximately proportional over the measured range, not a proven
  asymptotic complexity claim from three data points; the class-count axis showed a similar
  approximately-proportional pattern for both ranking functions.
- The label-based workflows (`confusion_matrix`, `classification_metrics`) ran two to three
  orders of magnitude faster than the ranking workflows at matching sample counts on this run --
  expected, since the ranking functions perform one binary ranking per class plus `y_score`'s own
  copy/validation.

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
about timing, and produces no JSON. This collects all five families (20 affine cases + 6 hashing
cases + 6 similarity cases + 6 discovery cases + 16 evaluation cases = 54). The normal `uv run
pytest` used everywhere else never touches `benchmarks/` at all.
