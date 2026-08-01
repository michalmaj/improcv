# Dataset discovery baseline — 2026-08-01

## Scope

This is the first consciously reviewed baseline captured from the dataset discovery benchmark
harness (`benchmarks/benchmark_discovery.py`). It measures, over a **warm filesystem cache**:

- `discover_images`, a single recursive traversal with a fresh per-entry stat, extension
  filtering, and a global sort, at 100/1,000/10,000 entries;
- `discover_image_mask_pairs`, two such traversals plus extension-stripped key construction,
  grouping, duplicate/key-set validation, and a final sorted pairing, at 100/1,000/10,000 pairs.

Every dataset is a deterministic, sharded tree (10 shard directories per root) of zero-byte,
extension-only entries created with `Path.touch()` -- discovery finds files by extension alone
and never opens, decodes, or inspects content, so a zero-byte file is exactly the input this
contract expects. There is no raw `os.walk`/`Path.rglob`/`glob`/manual-pairing baseline: neither
function has a raw equivalent of matching contract strength (see `benchmarks/README.md`).

This baseline does **not** cover cold-cache traversal, image decoding, dataset loading, failure
paths (unmatched/duplicate/hidden/symlink/permission), custom extensions, 100,000 entries, or any
end-to-end workflow -- those remain out of scope for this harness as it exists today. Every
number below describes **this one machine, in this one captured run** -- it is not a claim about
any other hardware, filesystem, or workload.

## Source

```text
commit:       50a0a2bc48e8c49b9a26b3f7c8284107e6f5bfce
raw stats-only JSON: 2026-08-01-discovery-baseline.json
JSON SHA-256: eb847b2348e5234bc0213e493f000ef0a2ae32ab068e353434cd74c05cc6a3c7
```

## Environment

All values below are read directly from the captured JSON's `machine_info`/`commit_info`, except
where noted as procedural (recorded from the preflight checks, not present in the JSON itself).

| Field | Value |
|---|---|
| Timestamp (UTC) | 2026-08-01T18:21:57.772026+00:00 |
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
| Power source (procedural) | AC power, battery 88%, charging |
| Load averages (procedural) | approximately 3.6 / 2.6 / 2.7 on a 12-core machine, immediately before capture -- ordinary desktop background load, no build/test/container process running concurrently |
| Temporary-directory filesystem (procedural) | APFS; benchmark trees were created on the volume backing the process's temporary directory (absolute path not recorded here) |

Discovery does not invoke any OpenCV image kernel, but the shared harness (`benchmarks/
conftest.py`) still records OpenCV/thread/OpenCL request-vs-observed state as part of every run's
`machine_info`, for consistency with the affine baseline.

## Dataset

| count per root | image files | mask files | total files | shards per root | directory depth | content policy |
|---|---|---|---|---|---|---|
| 100 | 100 | 100 | 200 | 10 | 1 | zero-byte, extension-only |
| 1,000 | 1,000 | 1,000 | 2,000 | 10 | 1 | zero-byte, extension-only |
| 10,000 | 10,000 | 10,000 | 20,000 | 10 | 1 | zero-byte, extension-only |

Diagnostic dataset setup time (session-scoped fixture, measured once per size, **outside** the
timed region and never part of any benchmark result): 100 entries per side took 0.010 s; 1,000
took 0.096 s; 10,000 took 1.113 s. File creation and cleanup are both outside the timed region --
cleanup itself is pytest's own temp-directory teardown, not measured at all here.

## Command

```bash
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
uv run --group benchmark pytest benchmarks/benchmark_discovery.py \
  --benchmark-warmup=on \
  --benchmark-min-rounds=20 \
  --benchmark-storage=file:///tmp/improcv-discovery-benchmark-storage \
  --benchmark-save=discovery-baseline
```

## Integrity checks

- `commit_info.dirty`: `false`
- `commit_info.id`: `50a0a2bc48e8c49b9a26b3f7c8284107e6f5bfce` (exact harness commit)
- 6 benchmark entries, all with a non-empty `group`, `params`, `extra_info`, and finite
  `median`/`mean`/`stddev`/`iqr`
- Group counts: `discovery-images: 3`, `discovery-pairs: 3`
- All 6 entries: `rounds >= 20` (100/1,000/10,000 for `discover_images`: 486/48/20 rounds;
  for `discover_image_mask_pairs`: 129/20/20 rounds)
- `stats.data` (the full per-round timing array) confirmed **absent** from every entry -- this is
  a compact, stats-only saved run, not a full-data capture
- Correctness/collection smoke (`--benchmark-disable -s`, run immediately before this capture):
  `6 passed`
- Raw JSON copied into `benchmarks/results/` byte-for-byte, unedited; SHA-256 of the copy matches
  the SHA-256 of the file captured in `/tmp` (see "Source" above)
- No affine benchmark entries present in this file (discovery-only capture)

## `discover_images`

| files | median | IQR | relative IQR | CV | median/file | growth vs. previous count | rounds | iterations |
|---|---|---|---|---|---|---|---|---|
| 100 | 2.24 ms | 54.3 µs | 2.4% | 2.8% | 22.4 µs | -- | 486 | 1 |
| 1,000 | 21.5 ms | 661 µs | 3.1% | 3.5% | 21.5 µs | 9.62x | 48 | 1 |
| 10,000 | 221 ms | 8.97 ms | 4.1% | 3.5% | 22.1 µs | 10.3x | 20 | 1 |

## `discover_image_mask_pairs`

| pairs | total files | median | IQR | relative IQR | CV | median/pair | median/visited-file | growth vs. previous count | rounds | iterations |
|---|---|---|---|---|---|---|---|---|---|---|
| 100 | 200 | 8.18 ms | 241 µs | 3.0% | 4.2% | 81.8 µs | 40.9 µs | -- | 129 | 1 |
| 1,000 | 2,000 | 80.6 ms | 731 µs | 0.9% | 1.7% | 80.6 µs | 40.3 µs | 9.86x | 20 | 1 |
| 10,000 | 20,000 | 840 ms | 15.2 ms | 1.8% | 1.2% | 84.0 µs | 42.0 µs | 10.4x | 20 | 1 |

## Interpretation

Observations supported directly by the tables above, for this machine and this run only:

- Both groups' observed medians increased monotonically across all three sizes.
- For `discover_images`, the two observed 10x-input-count steps produced 9.62x and 10.3x median
  growth respectively -- approximately proportional over the measured range. `median/file`
  stayed essentially flat (22.4 -> 21.5 -> 22.1 µs), consistent with that near-proportional
  growth.
- For `discover_image_mask_pairs`, the two 10x-input-count steps produced 9.86x and 10.4x median
  growth -- also approximately proportional over the measured range. `median/pair` was flat from
  100 to 1,000 (81.8 -> 80.6 µs) and rose slightly at 10,000 (84.0 µs, +4% vs. the 1,000-pair
  value), a small enough change that it is reported as an observation, not attributed to a cause,
  with only three measured sizes.
- `discover_image_mask_pairs`'s median was 3.66x-3.81x `discover_images`'s median at the matching
  per-root count (3.66x at 100, 3.75x at 1,000, 3.81x at 10,000). **This is not a raw-wrapper
  overhead ratio.** The pairing call traverses two roots and additionally builds, validates, and
  sorts strict pairing keys, whereas `discover_images` traverses one root. No attempt is made
  here to isolate "pure pairing cost" as `pairing - 2 x discover_images`: two separately measured
  benchmarks do not reliably isolate a difference this small, and the image and mask traversals
  are never measured together outside the pairing function itself.
- Three measured sizes are not enough to establish a proven asymptotic complexity class; the
  results above are described as "approximately proportional over the measured range," not as a
  demonstrated `O(n)`.

No claim is made here about any other machine, filesystem, or workload.

## Measurement spread

All 6 entries, sorted by relative IQR, descending:

| case | relative IQR | coefficient of variation | median | rounds |
|---|---|---|---|---|
| `test_discover_images[10000-entries]` | 4.1% | 3.5% | 221 ms | 20 |
| `test_discover_images[1000-entries]` | 3.1% | 3.5% | 21.5 ms | 48 |
| `test_discover_image_mask_pairs[100-entries]` | 3.0% | 4.2% | 8.18 ms | 129 |
| `test_discover_images[100-entries]` | 2.4% | 2.8% | 2.24 ms | 486 |
| `test_discover_image_mask_pairs[10000-entries]` | 1.8% | 1.2% | 840 ms | 20 |
| `test_discover_image_mask_pairs[1000-entries]` | 0.9% | 1.7% | 80.6 ms | 20 |

This stats-only capture looks stable: relative IQR stays under 5% and coefficient of variation
under 5% for every one of the 6 entries, with no case standing out as a radical outlier by either
measure -- unlike the affine baseline's one high-CV case, nothing here required deeper
per-round investigation. The plugin reported no stability warnings, and load averages did not
rise noticeably over the ~66-second capture window (checked via `uptime` immediately before and
after).

Because this is the default compact, stats-only format (no `stats.data`), there is no per-round
timing array to inspect further here -- this is a deliberate storage-policy tradeoff (see
`benchmarks/results/README.md`), not an oversight; a full-data diagnostic capture remains
available later if a specific case ever needs closer investigation.

## Limitations

- Single machine, single run -- not repeated, not averaged across sessions.
- Single filesystem (APFS, via the process's temporary directory) -- no comparison against any
  other filesystem.
- Warm filesystem cache only -- no cold-cache measurement.
- Fixed 10-shard-per-root topology at exactly three sizes (100/1,000/10,000) -- no other shard
  count or directory depth was measured.
- Zero-byte, extension-only entries -- no image decoding, no real file content of any size.
- No symlinks, hidden files, or failure paths (unmatched/duplicate/permission-error) measured.
- No 100,000-entry case.
- No raw `os.walk`/`Path.rglob`/`glob`/manual-pairing baseline, by design (see "Scope" above).
- No comparison against any other machine, OS, or filesystem.
