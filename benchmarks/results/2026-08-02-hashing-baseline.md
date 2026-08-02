# Perceptual hashing baseline — 2026-08-02

## Scope

This is the first consciously reviewed baseline captured from the perceptual hashing benchmark
harness (`benchmarks/benchmark_hashing.py`). It covers two groups:

- `hashing-average-hash` -- the complete public `average_hash(image, hash_size=8)` call;
- `hashing-phash` -- the complete public `phash(image, hash_size=8)` call.

at three source image sizes (`64x64`, `640x480`, `1920x1080`). Every source image is a
deterministic, seeded, C-contiguous `(height, width, 3)` BGR `uint8` array, built once per size
in a session-scoped fixture, entirely **before** the timed `benchmark(...)` call -- never inside
the timed closure. `hash_size` is fixed at `8` throughout.

The timed region for each case is the *complete* public function call: input/`hash_size`
validation, resize, BGR-to-grayscale conversion, every algorithm-specific step (`average_hash`'s
mean/threshold; `phash`'s `float32` cast, `cv2.dct`, block selection, DC zeroing, `cv2.mean`,
threshold cast), bit packing, and `PerceptualHash` construction. No image is decoded, read from,
or written to disk anywhere in this harness.

There is no raw or `opencv-contrib` baseline here: neither function has a single raw kernel
matching its complete public contract (both are themselves multi-step pipelines) -- see
`benchmarks/README.md`/`benchmarks/benchmark_hashing.py` for the detailed justification. This
baseline measures how the public API itself scales with source image size, not a ratio against a
semantically different implementation.

This baseline does **not** cover `find_similar_image_pairs`/pair-search scaling, Hamming
distance in isolation, `hash_size` as a scaling axis, grayscale/BGRA input, any dtype other than
`uint8`, or any comparison with another library or machine -- those remain out of scope for this
harness as it exists today. Every number below describes **this one machine, in this one
captured run** -- it is not a claim about any other hardware or workload.

## Source

```text
harness commit:      c0a07a5a506b20aea15a8572c68a22d9eea641ce
raw stats-only JSON: 2026-08-02-hashing-baseline.json
JSON SHA-256:        19cb19474252cc6c824ee2c72fa9f1bd198ef3b4b1bc4ddf45540ca526d2a35b
```

## Environment

All values below are read directly from the captured JSON's `machine_info`/`commit_info`.

| Field | Value |
|---|---|
| Timestamp (UTC) | 2026-08-02T17:34:04.437497+00:00 |
| OS | Darwin, release 25.5.0 |
| Architecture | arm64 |
| CPU | Apple M4 Pro (12 logical cores; physical core count not reported by the tool) |
| Python | CPython 3.12.7 |
| NumPy | 2.5.1 |
| OpenCV | 5.0.0 |
| improcv | 0.3.0a1.dev0 |
| pytest-benchmark | 5.2.3 |
| Requested OpenCV threads | 1 |
| Observed OpenCV threads | 12 |
| Requested OpenCL | False |
| Observed OpenCL | False |
| `OMP_NUM_THREADS` | "1" |
| `OPENBLAS_NUM_THREADS` | "1" |
| `MKL_NUM_THREADS` | "1" |
| Commit dirty state | `false` |

Requested and observed OpenCV thread/OpenCL state are tracked separately, per
`benchmarks/conftest.py`'s own policy -- not every OpenCV build honors a thread-count request,
and this run's observed 12 threads is recorded as a fact about this build, not treated as a
harness defect.

## Data scenarios

Setup and input allocation happen entirely in a session-scoped fixture, **before** the timed
`benchmark(...)` call.

| size | shape (H, W, C) | pixels | bytes | seed | dtype | layout | channels/color |
|---|---|---|---|---|---|---|---|
| 64x64 | (64, 64, 3) | 4,096 | 12,288 | 20324867 | uint8 | C | 3 / BGR |
| 640x480 | (480, 640, 3) | 307,200 | 921,600 | 20901283 | uint8 | C | 3 / BGR |
| 1920x1080 | (1080, 1920, 3) | 2,073,600 | 6,220,800 | 22181883 | uint8 | C | 3 / BGR |

Diagnostic setup times (never a benchmark result, from the preflight correctness smoke run):
`64x64` took 1.161 s -- first-use/allocation noise outside the timed region (the process's first
call to `np.random.default_rng` in this session), not representative of per-scenario cost.
`640x480` took 0.001 s and `1920x1080` took 0.004 s.

## Command

```bash
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
uv run --group benchmark pytest benchmarks/benchmark_hashing.py \
  --benchmark-warmup=on \
  --benchmark-min-rounds=20 \
  --benchmark-storage=file:///tmp/improcv-hashing-benchmark-storage \
  --benchmark-save=hashing-baseline
```

## Integrity checks

- `commit_info.dirty`: `false`
- `commit_info.id`: `c0a07a5a506b20aea15a8572c68a22d9eea641ce` (exact harness commit)
- 6 benchmark entries, all with a non-empty `group`, `params`, `extra_info`, and finite
  `median`/`mean`/`stddev`/`iqr`
- Group counts: `hashing-average-hash: 3`, `hashing-phash: 3`
- All 6 entries: `rounds >= 20` (`average_hash`: 176,494/179,077/177,780 at
  64x64/640x480/1920x1080; `phash`: 94,860/63,328/59,703 at the same three sizes)
- `stats.data` (the full per-round timing array) confirmed **absent** from every entry -- this is
  a compact, stats-only saved run, not a full-data capture
- Correctness/collection smoke (`--benchmark-disable -s`, run immediately before this capture):
  `6 passed`
- Raw JSON copied into `benchmarks/results/` byte-for-byte, unedited; SHA-256 of the copy matches
  the SHA-256 of the file captured in `/tmp` (see "Source" above)
- No affine, discovery, or evaluation benchmark entries present in this file (hashing-only
  capture)

## Average hash results

| width | height | source pixels | source bytes | median | IQR | relative IQR | CV | median/pixel | rounds | iterations |
|---|---|---|---|---|---|---|---|---|---|---|
| 64 | 64 | 4,096 | 12,288 | 6.042 µs | 0.667 µs | 11.0% | 16.5% | 1.475 ns | 176,494 | 1 |
| 640 | 480 | 307,200 | 921,600 | 6.125 µs | 0.667 µs | 10.9% | 13.9% | 0.0199 ns | 179,077 | 1 |
| 1,920 | 1,080 | 2,073,600 | 6,220,800 | 6.416 µs | 0.708 µs | 11.0% | 37.8% | 0.0031 ns | 177,780 | 1 |

Observed median growth: `64x64 -> 640x480` (75x pixel growth) = **1.01x**; `640x480 -> 1920x1080`
(6.75x pixel growth) = **1.05x**.

## pHash results

| width | height | source pixels | source bytes | median | IQR | relative IQR | CV | median/pixel | rounds | iterations |
|---|---|---|---|---|---|---|---|---|---|---|
| 64 | 64 | 4,096 | 12,288 | 11.959 µs | 1.209 µs | 10.1% | 15.1% | 2.920 ns | 94,860 | 1 |
| 640 | 480 | 307,200 | 921,600 | 16.375 µs | 1.833 µs | 11.2% | 12.7% | 0.0533 ns | 63,328 | 1 |
| 1,920 | 1,080 | 2,073,600 | 6,220,800 | 17.375 µs | 1.958 µs | 11.3% | 13.9% | 0.0084 ns | 59,703 | 1 |

Observed median growth: `64x64 -> 640x480` (75x pixel growth) = **1.37x**; `640x480 -> 1920x1080`
(6.75x pixel growth) = **1.06x**.

## Workflow comparison

| source size | `average_hash` median | `phash` median | phash/average_hash |
|---|---|---|---|
| 64x64 | 6.042 µs | 11.959 µs | 1.98x |
| 640x480 | 6.125 µs | 16.375 µs | 2.67x |
| 1920x1080 | 6.416 µs | 17.375 µs | 2.71x |

**This compares two complete public workflows with different algorithms, target grids, and
result semantics. It does not isolate the exact cost of DCT or prove that one perceptual hashing
algorithm is generally faster or better.** `phash`'s median is not decomposed into an
"`average_hash`-equivalent part" plus a "DCT part" -- `phash - average_hash` is not computed as an
exact DCT cost anywhere in this report.

## Interpretation

Observations supported directly by the tables above, for this machine and this run only:

- Both `average_hash` and `phash` medians increased monotonically across the three measured
  source sizes (`64x64 < 640x480 < 1920x1080`).
- For `average_hash`, a 75x increase in source pixels (`64x64 -> 640x480`) produced only a 1.01x
  median increase, and a further 6.75x pixel increase (`640x480 -> 1920x1080`) produced only a
  1.05x median increase -- the observed public-call median stayed close to flat across a combined
  ~506x increase in source pixels on this machine.
- For `phash`, the same two pixel-growth steps produced 1.37x and 1.06x median growth
  respectively -- larger than `average_hash`'s but still far below proportional to pixel count.
- `median/pixel` fell steeply for both functions as source size grew (`average_hash`: 1.475 ns ->
  0.0199 ns -> 0.0031 ns; `phash`: 2.920 ns -> 0.0533 ns -> 0.0084 ns), consistent with each
  function reducing its input to a fixed, small target grid before doing its algorithm-specific
  work: `average_hash` resizes directly to `8x8` (64 output pixels regardless of source size),
  while `phash` resizes to `32x32` (1,024 output pixels) before its DCT and block selection. A
  near-flat median across a large source-pixel range is consistent with the resize step (and the
  fixed-size work after it) dominating total cost for both functions in this measured range, on
  this OpenCV build -- not evidence of a particular resize implementation detail beyond that.
- `phash`'s median was consistently 2.0x-2.7x `average_hash`'s median across all three sizes, and
  that ratio itself grew somewhat with source size (1.98x -> 2.67x -> 2.71x) -- consistent with
  `phash`'s larger `32x32` target grid and its additional DCT/block-selection work, but this is
  reported only as an observed ratio between two full workflows (see "Workflow comparison" above),
  not as an isolated per-step cost.

Three measured source-size points are not enough to establish a proven asymptotic complexity
class for either function. The observed public-call median changed by roughly 1.01x-1.05x
(`average_hash`) and 1.06x-1.37x (`phash`) across the two measured pixel-growth steps on this
machine -- this is reported as a measured fact about this run, not as `O(1)`, "constant time," or
"independent of input size." No claim is made here about perceptual quality, collision
resistance, robustness, or general algorithm superiority, and no claim is made about any other
machine or workload.

## Measurement spread

All 6 entries, sorted by relative IQR, descending:

| case | relative IQR | coefficient of variation | median | rounds |
|---|---|---|---|---|
| `test_phash[1920x1080]` | 11.3% | 13.9% | 17.375 µs | 59,703 |
| `test_phash[640x480]` | 11.2% | 12.7% | 16.375 µs | 63,328 |
| `test_average_hash[64x64]` | 11.0% | 16.5% | 6.042 µs | 176,494 |
| `test_average_hash[1920x1080]` | 11.0% | 37.8% | 6.416 µs | 177,780 |
| `test_average_hash[640x480]` | 10.9% | 13.9% | 6.125 µs | 179,077 |
| `test_phash[64x64]` | 10.1% | 15.1% | 11.959 µs | 94,860 |

Relative IQR stayed in a narrow 10.1%-11.3% band across all six entries, with no case standing
out as unusually spread relative to the others by this measure. Coefficient of variation tells a
different story for one case: `test_average_hash[1920x1080]` (CV 37.8%), well above the rest
(next-highest is 16.5%). Its saved `min`/`max`/`median` (`min`=5.583 µs, `max`=716.167 µs,
`median`=6.416 µs, over 177,780 rounds) show `max` at roughly 112x the median, while
`median`/IQR (this project's primary statistics; see `benchmarks/README.md`'s "Interpretation"
section) stayed low and unremarkable -- the same signature as the high-CV cases already
documented in the affine and evaluation baselines, consistent with rare OS-scheduler preemption
hitting a small fraction of a very large number of rounds in a fast case, not a systemic harness
or code problem. The `pytest-benchmark` plugin reported no stability warnings for this run.

**This compact saved run contains summary statistics, not full per-round data** -- there is no
`stats.data` array here to confirm the exact fraction of affected rounds or the single largest
outlier's cause; the interpretation above is based on `min`/`max`/`mean`/`stddev`/`median`/`iqr`
alone, per this baseline's deliberate stats-only format (see `benchmarks/results/README.md`). No
observation was removed and no second run was captured to make these numbers look different.

## Limitations

- Single machine, single run -- not repeated, not averaged across sessions.
- Seeded, synthetic BGR `uint8` inputs only -- no real photographs or encoded images were used.
- Only C-contiguous input arrays were measured.
- Only `hash_size=8` was measured -- hash-size scaling is out of scope for this harness today.
- Only three source sizes (`64x64`/`640x480`/`1920x1080`) were measured.
- No grayscale or BGRA input was measured -- only 3-channel BGR.
- No dtype other than `uint8` was measured.
- No image decoding or filesystem I/O of any kind is exercised by this harness.
- No raw NumPy/OpenCV or `cv2.img_hash`/`opencv-contrib` comparison, by design (see "Scope"
  above).
- No `find_similar_image_pairs` or pair-search scaling was measured.
- No test of perceptual quality, collision resistance, or robustness was performed or implied.
- Three measured source-size points are not enough to prove an asymptotic complexity class for
  either function.
