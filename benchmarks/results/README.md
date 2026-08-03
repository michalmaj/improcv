# Benchmark results

Native `pytest-benchmark` JSON output, committed selectively -- not every local run, only
deliberately reviewed baselines.

## Policy

- Only consciously reviewed baseline JSON files are committed here, produced with a "Stable local
  run" command in `benchmarks/README.md` -- a run produced after requesting one OpenCV thread and
  disabling OpenCL, with both the requested and the observed state recorded in `machine_info`,
  plus an explicit warm-up phase and at least 20 rounds.
- A new run creates a **new** file. Existing result files are never overwritten.
- A run later found to be flawed is either replaced by a new file (with a note in this directory,
  or in an accompanying Markdown file, explaining why) or superseded by a newer dated file --
  never silently amended or deleted.
- Every committed result must be traceable to an environment: the JSON itself carries
  `machine_info` (including the `improcv`/NumPy/OpenCV versions, and both the requested and the
  actually observed OpenCV thread count and OpenCL state, added by `benchmarks/conftest.py`) and
  `commit_info` (including the commit SHA and whether the working tree was dirty at capture time).
- A result whose observed thread/OpenCL state differs from the requested state is still valid,
  but it must be interpreted and compared using the *observed* state, not the requested one.
- Results captured on different machines are not directly comparable to each other -- only to
  other results captured on the *same* machine under a matching observed configuration.

## Two kinds of committed JSON

Both are native, unedited `pytest-benchmark` output -- the difference is which of the tool's own
output modes produced the file, not any post-processing this project applies afterward.

**Reviewed compact baseline (default).** Produced by `--benchmark-save`/`--benchmark-storage`
without `--benchmark-save-data`. Contains `machine_info`, `commit_info`, and, per case, `group`,
`params`, `extra_info`, and summary statistics (`median`/`mean`/`stddev`/`iqr`/`rounds`/etc.) --
but not the array of every individual round's timing. This is the expected format for every
committed result going forward, each with an accompanying reviewed Markdown report.

**Full-data capture (optional).** Produced by `--benchmark-json`, or by adding
`--benchmark-save-data` to the saved-run command above. Contains everything the compact baseline
does, plus the complete per-round timing array (`stats.data`) for every case -- confirmed
directly to be tens of times larger even for a short smoke run, and dramatically larger for a
real, many-thousand-round baseline. Only committed when there is a specific, reviewed diagnostic
reason (e.g. investigating an outlier or a distribution shape) to keep the full per-round
distribution rather than its summary statistics.

**Do not manually strip `stats.data` from a JSON file.** Choose the appropriate native
`pytest-benchmark` output mode *before* capture instead -- a hand-edited "reduced" JSON would no
longer be an unedited tool output, breaking the guarantee every committed result here relies on.

## Filename format

```text
YYYY-MM-DD-<family>-baseline.json
```

`<family>` is the benchmark family the file belongs to (e.g. `augmentation`, `discovery`) -- one
result file and one accompanying report per family per dated capture, never a mix of families in
a single file. An optional, hand-written narrative Markdown file with the same date/topic prefix
may accompany a result to explain something not obvious from the raw numbers alone (mirroring
this project's C++ sibling's own dated result notes) -- not required for every result.

## Current contents

- [`2026-08-01-augmentation-baseline.json`](2026-08-01-augmentation-baseline.json) -- a **full-data
  capture** (generated with `--benchmark-json`, ~20,873,463 bytes, including the complete
  per-round timing array for all 14 cases). This size is a direct consequence of that output mode
  for tens of thousands of rounds, not a mistake or something to fix retroactively -- it is kept
  exactly as captured, as a historical exception documenting the first-ever run in maximum detail.
  It remains the raw, unedited `pytest-benchmark` output and the source of truth for every number
  in the accompanying report; not meant to be read directly.
- [`2026-08-01-augmentation-baseline.md`](2026-08-01-augmentation-baseline.md) -- the reviewed
  narrative interpretation of that JSON: environment, integrity checks, per-case tables,
  observations, measurement spread, and limitations.
- [`2026-08-01-discovery-baseline.json`](2026-08-01-discovery-baseline.json) -- a **compact,
  stats-only native saved run** (produced with `--benchmark-save`/`--benchmark-storage`, no
  `--benchmark-save-data`; `stats.data` confirmed absent from every entry), covering
  `discover_images` and `discover_image_mask_pairs` at 100/1,000/10,000 entries, at commit
  `50a0a2bc48e8c49b9a26b3f7c8284107e6f5bfce`. This is the default format described above, not an
  exception.
- [`2026-08-01-discovery-baseline.md`](2026-08-01-discovery-baseline.md) -- the reviewed narrative
  interpretation of that JSON: environment, integrity checks, per-case tables, scaling
  observations, measurement spread, and limitations.
- [`2026-08-02-evaluation-baseline.json`](2026-08-02-evaluation-baseline.json) -- a **compact,
  stats-only native saved run** (produced with `--benchmark-save`/`--benchmark-storage`, no
  `--benchmark-save-data`; `stats.data` confirmed absent from every entry), covering
  `confusion_matrix`, `classification_metrics`, `multiclass_roc_auc_score`, and
  `multiclass_average_precision_score` (the latter three at `average="macro"`) across sample scaling
  (1,000/10,000/100,000, fixed 10 classes) and ranking class scaling (3/10/100, fixed 10,000
  samples), at commit `658a6bcc6b943a1f9e232be51149b2d52d1a08d2`. This is the default format
  described above, not an exception.
- [`2026-08-02-evaluation-baseline.md`](2026-08-02-evaluation-baseline.md) -- the reviewed
  narrative interpretation of that JSON: environment, integrity checks, per-case tables, scaling
  observations, measurement spread, and limitations.
- [`2026-08-02-hashing-baseline.json`](2026-08-02-hashing-baseline.json) -- a **compact,
  stats-only native saved run** (produced with `--benchmark-save`/`--benchmark-storage`, no
  `--benchmark-save-data`; `stats.data` confirmed absent from every entry), covering the complete
  public `average_hash` and `phash` calls (fixed `hash_size=8`) at three BGR `uint8` source image
  sizes (`64x64`/`640x480`/`1920x1080`), at the exact harness commit
  `c0a07a5a506b20aea15a8572c68a22d9eea641ce`. This is the default format described above, not an
  exception.
- [`2026-08-02-hashing-baseline.md`](2026-08-02-hashing-baseline.md) -- the reviewed narrative
  interpretation of that JSON: environment, integrity checks, per-case tables, scaling
  observations, measurement spread, and limitations.
- [`2026-08-02-similarity-baseline.json`](2026-08-02-similarity-baseline.json) -- a **compact,
  stats-only native saved run** (produced with `--benchmark-save`/`--benchmark-storage`, no
  `--benchmark-save-data`; `stats.data` confirmed absent from every entry), covering the complete
  public `find_similar_image_pairs` call against precomputed `PHASH` values (`hash_size=8`) at
  `30`/`100`/`300` items (`435`/`4,950`/`44,850` unordered pairs), in two groups
  (`similarity-no-matches`: 3 entries, `similarity-all-matches`: 3 entries), at the exact harness
  commit `72a2e01594f0ad1e2c569a74a2a2600992d0fef6`. This is the default format described above,
  not an exception.
- [`2026-08-02-similarity-baseline.md`](2026-08-02-similarity-baseline.md) -- the reviewed
  narrative interpretation of that JSON: environment, integrity checks, per-case tables, scaling
  observations, measurement spread, and limitations.

The augmentation pair, the discovery pair, the evaluation pair, the hashing pair, and the
similarity pair each describe their own single capture, at their own commit
(`55ed9b6d92942b35319b13faf95938c51bc4cbc9` for augmentation,
`50a0a2bc48e8c49b9a26b3f7c8284107e6f5bfce` for discovery,
`658a6bcc6b943a1f9e232be51149b2d52d1a08d2` for evaluation,
`c0a07a5a506b20aea15a8572c68a22d9eea641ce` for hashing,
`72a2e01594f0ad1e2c569a74a2a2600992d0fef6` for similarity), all five with
`commit_info.dirty: false`. Per the immutable-results policy above, none of the five pairs is
ever overwritten by a future result -- a later baseline for any family gets its own new dated
filename. The augmentation full-data capture remains a historical exception specific to that
first-ever run; the discovery, evaluation, hashing, and similarity baselines use -- and future
baselines are expected to keep using -- the compact, stats-only saved-run format, reserving a
full-data capture for a specific, reviewed diagnostic need.
