# Affine augmentation baseline — 2026-08-01

## Scope

This is the first consciously reviewed baseline captured from the affine augmentation benchmark
harness (`benchmarks/benchmark_augmentation.py`). It measures:

- pure Python/NumPy geometry (`sample_affine` with non-degenerate ranges, `expand_affine_canvas`);
- `apply_affine`, image only, vs. a raw `cv2.warpAffine` call, at three sizes;
- `apply_affine`, image + mask, vs. two raw `cv2.warpAffine` calls, at three sizes.

It does **not** cover perspective transforms, dataset discovery, classification evaluation, or
any end-to-end workflow -- those remain out of scope for this harness as it exists today. Every
number below describes **this one machine, in this one captured run** -- it is not a claim about
any other hardware, OpenCV build, or workload.

## Source

```text
commit:      55ed9b6d92942b35319b13faf95938c51bc4cbc9
raw JSON:    2026-08-01-augmentation-baseline.json
JSON SHA-256: a0207a1981e51030ab3abaa46df7b37fdb465e9a8559c2fdf379ab2f415e9d6e
```

## Environment

All values below are read directly from the captured JSON's `machine_info`/`commit_info`, not
from memory.

| Field | Value |
|---|---|
| Timestamp (UTC) | 2026-08-01T14:09:53.445535+00:00 |
| OS | Darwin, release 25.5.0 |
| Architecture | arm64 |
| CPU | Apple M4 Pro (12 logical cores; physical core count not reported by the tool) |
| Python | CPython 3.12.7 |
| NumPy | 2.5.1 |
| OpenCV | 5.0.0 |
| improcv | 0.2.0a3.dev0 |
| pytest-benchmark | 5.2.3 |
| Requested OpenCV threads | 1 |
| Observed OpenCV threads | 12 |
| Requested OpenCL | False |
| Observed OpenCL | False |
| `OMP_NUM_THREADS` | "1" |
| `OPENBLAS_NUM_THREADS` | "1" |
| `MKL_NUM_THREADS` | "1" |
| Commit dirty state | `false` |

Procedural information, not part of the JSON: captured on AC power, battery at 100%, with `pmset
-g batt`/`uptime` checked immediately before, and `uptime` checked immediately after, capture.
Load averages were stable across the run (approximately 3.5/3.4/3.0 before, 3.3/3.4/3.1 after, on
a 12-core machine) -- ordinary desktop background load (browser, editor, IDE), no build, test, or
container process running concurrently.

## Command

```bash
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
uv run --group benchmark pytest benchmarks/ \
  --benchmark-warmup=on \
  --benchmark-min-rounds=20 \
  --benchmark-json=/tmp/2026-08-01-augmentation-baseline.json
```

## Integrity checks

- `commit_info.dirty`: `false`
- `commit_info.id`: `55ed9b6d92942b35319b13faf95938c51bc4cbc9` (exact harness commit)
- 14 benchmark entries, all with a non-empty `group`, `stats.data`, finite `median`/`mean`/
  `stddev`/`iqr`, and `extra_info`
- Group counts: `affine-python-geometry: 2`, `affine-image-only: 6`, `affine-image-mask: 6`
- All 14 entries: `rounds >= 20` (in practice, far more -- see the per-case table below; the
  fastest cases ran well over 100,000 rounds under `pytest-benchmark`'s own calibration)
- Raw/wrapper correctness smoke (`--benchmark-disable`, run immediately before this capture):
  `20 passed`
- Raw JSON copied into `benchmarks/results/` byte-for-byte, unedited; SHA-256 of the copy matches
  the SHA-256 of the file captured in `/tmp` (see "Source" above)

## Python-side geometry

No raw OpenCV baseline exists for either operation.

| operation | median | IQR | relative IQR | rounds | iterations |
|---|---|---|---|---|---|
| `sample_affine_random_ranges` | 24.4 µs | 2.62 µs | 0.108 | 43,011 | 1 |
| `expand_affine_canvas` | 17.8 µs | 1.92 µs | 0.108 | 59,553 | 1 |

## Image-only results

| size | raw median | improcv median | median delta | wrapper/raw ratio | raw relative IQR | improcv relative IQR |
|---|---|---|---|---|---|---|
| 64x64 | 7.00 µs | 15.8 µs | 8.79 µs | 2.26 | 0.018 | 0.113 |
| 640x480 | 433 µs | 448 µs | 14.7 µs | 1.03 | 0.062 | 0.064 |
| 1920x1080 | 3.19 ms | 3.24 ms | 48.7 µs | 1.02 | 0.035 | 0.033 |

## Image+mask results

The raw image+mask baseline performs two `cv2.warpAffine` calls: `INTER_LINEAR` for the image and
`INTER_NEAREST` for the mask. The raw median below is the time for *both* calls together, never a
single `warpAffine`.

| size | raw median | improcv median | median delta | wrapper/raw ratio | raw relative IQR | improcv relative IQR |
|---|---|---|---|---|---|---|
| 64x64 | 9.25 µs | 24.8 µs | 15.5 µs | 2.68 | 0.113 | 0.103 |
| 640x480 | 554 µs | 588 µs | 34.1 µs | 1.06 | 0.056 | 0.070 |
| 1920x1080 | 4.09 ms | 4.11 ms | 27.0 µs | 1.01 | 0.078 | 0.045 |

## Interpretation

Observations supported directly by the table above, for this machine and this run only:

- At `64x64`, the observed wrapper/raw ratio is largest (2.26 for image-only, 2.68 for
  image+mask) and the observed median delta is smallest in absolute terms (8.79 µs and 15.5 µs).
  At this size, the fixed per-call cost `improcv` adds (parameter/shape/dtype validation, Python
  dispatch, result construction) is a large fraction of the total call time, because the raw
  kernel itself is only a few microseconds.
- At `640x480` and `1920x1080`, the observed wrapper/raw ratio is close to 1.0 (1.01-1.06) even
  though the observed median delta in absolute terms is comparable to or larger than at `64x64`
  (14.7-48.7 µs). At these sizes, the `cv2.warpAffine` kernel itself dominates total call time, so
  the same roughly constant per-call overhead becomes a small fraction of it.
- The image+mask cases perform a second warp (`INTER_NEAREST` on the mask) in addition to the
  image warp; their raw and improcv medians are correspondingly higher than the matching
  image-only row at every size, consistent with two kernel calls instead of one.
- The two Python-side geometry cases (`sample_affine_random_ranges`, `expand_affine_canvas`) sit
  in the same low-tens-of-microseconds range as the smallest `apply_affine` overhead observed
  above, and have no OpenCV kernel involved at all -- their cost is a separate concern from the
  image-warp cases, not directly comparable to a "wrapper/raw" ratio.

No claim is made here about any other machine, OpenCV build, Python version, or production
workload.

## Measurement spread

Relative IQR (`iqr / median`) was consistently low across all 14 entries (0.018-0.113), with no
case standing out as unusually spread relative to the others by this measure. Coefficient of
variation (`stddev / mean`) tells a different story for exactly one case:

Sorted by coefficient of variation, descending (all 14 entries):

| case | relative IQR | coefficient of variation |
|---|---|---|
| `apply_affine_image_mask[raw-64x64]` | 0.113 | **1.992** |
| `apply_affine_image_only[raw-640x480]` | 0.062 | 0.405 |
| `expand_affine_canvas` | 0.108 | 0.350 |
| `apply_affine_image_mask[improcv-64x64]` | 0.103 | 0.292 |
| `apply_affine_image_mask[raw-1920x1080]` | 0.078 | 0.142 |
| `apply_affine_image_only[improcv-64x64]` | 0.113 | 0.132 |
| `sample_affine_random_ranges` | 0.108 | 0.107 |
| `apply_affine_image_only[raw-64x64]` | 0.018 | 0.097 |
| `apply_affine_image_mask[improcv-1920x1080]` | 0.045 | 0.093 |
| `apply_affine_image_mask[raw-640x480]` | 0.056 | 0.093 |
| `apply_affine_image_only[improcv-640x480]` | 0.064 | 0.086 |
| `apply_affine_image_mask[improcv-640x480]` | 0.070 | 0.072 |
| `apply_affine_image_only[raw-1920x1080]` | 0.035 | 0.071 |
| `apply_affine_image_only[improcv-1920x1080]` | 0.033 | 0.068 |

`apply_affine_image_mask[raw-64x64]` (median 9.25 µs, 111,114 rounds) is a real outlier by
coefficient of variation: of its 111,114 individual round timings, 112 (0.10%) exceeded 5x the
median, with the single largest at 6.41 ms -- about 690x the median. This inflates `mean`/`stddev`
heavily while leaving `median`/`IQR` unremarkable (0.113 relative IQR, in line with every other
case). The same pattern, at a smaller scale, appears in every other fast (single-digit-to-tens-
of-microseconds) case in this run: `expand_affine_canvas` had 62 of 59,553 rounds (0.10%) exceed
5x its median, and `apply_affine_image_only[raw-640x480]` had 6 of 2,334 (0.26%). This is
consistent with occasional OS scheduler preemption hitting an individual round out of tens or
hundreds of thousands, on a general-purpose desktop OS with ordinary background load -- not with
a systemic problem in the harness or in the code being measured. Load averages were stable
(neither rising nor spiking) across the ~46-second capture window, and the run produced no
`pytest-benchmark` stability warnings and no test failures (`20 passed`).

The measurement is judged stable on the basis of the median/IQR statistics this project treats as
primary (see `benchmarks/README.md`'s "Interpretation" section) -- coefficient of variation is
reported here for transparency, not as the deciding statistic, and no observation was removed or
a second run captured to make this number look different.

## Limitations

- Single machine, single run -- not repeated, not averaged across sessions.
- Reflects this process's specific warm-cache/OS-scheduling state at capture time, not a
  cold-start or long-running-service condition.
- No perspective transform, discovery, evaluation, or end-to-end workflow coverage.
- No `float32` image dtype coverage (only `uint8`).
- No thread-scaling study -- the requested single-thread OpenCV configuration was not actually
  observed (12 threads reported), and this baseline does not attempt to characterize behavior at
  any other explicit thread count.
- The requested OpenCV thread count is not guaranteed to match the observed one, as documented in
  `benchmarks/README.md` -- this run is an example of exactly that mismatch.
- No comparison against any other machine, OS, OpenCV build, or Python/NumPy version.
