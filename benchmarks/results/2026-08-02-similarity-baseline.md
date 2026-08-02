# Pairwise image similarity baseline — 2026-08-02

## Scope

This is the first consciously reviewed baseline captured from the pairwise similarity benchmark
harness (`benchmarks/benchmark_similarity.py`). It measures the complete public
`improcv.find_similar_image_pairs` call against **precomputed `PHASH` values**
(`hash_size=8`) at three item counts (`30`/`100`/`300`, i.e. `435`/`4,950`/`44,850` unordered
pairs), in two extreme result-cardinality regimes:

- `similarity-no-matches` -- `max_distance=0`; every hash is unique, so the result is always
  empty;
- `similarity-all-matches` -- `max_distance=64` (`= hash_size**2`, the maximum legal threshold),
  so every unordered pair is materialized into the result.

Both regimes at a given item count share exactly the same input mapping, built once per session
and inserted in **reverse canonical** (`path.as_posix()`) order, so the timed call must actually
normalize and sort the input rather than benefit from an already-sorted mapping. Setup (building
the synthetic hash mapping) happens entirely in a session-scoped fixture, **before** the timed
`benchmark(...)` call; the timed region itself is the complete public call -- `max_distance`
validation, `Mapping` validation, path normalization, duplicate-key detection, per-hash
validation, the shared `algorithm`/`hash_size` check, the threshold upper-bound check, input
sorting, every unordered-pair enumeration, every `PerceptualHash.distance` call, the threshold
branch, `SimilarImagePair` construction where it matches, the final result sort, and the tuple
conversion.

There is no filesystem access, image decoding, hash computation, or call to `discover_images`
anywhere in this harness -- the input is exclusively already-computed `PerceptualHash` objects,
built from a deterministic integer transform, never `im.phash()`/`im.average_hash()`. There is no
duplicate grouping or clustering: the function returns pairs, never clusters. There is no raw
baseline: no single raw kernel matches `find_similar_image_pairs`'s complete public contract (see
`benchmarks/README.md`/`benchmarks/benchmark_similarity.py` for the detailed justification).

This baseline does **not** cover any intermediate match density between the two measured
extremes, any `max_distance` other than `0`/`64`, any `hash_size` other than `8`, any algorithm
other than `PHASH`, memory usage, or a comparison with another library or machine -- those remain
out of scope for this harness as it exists today. Every number below describes **this one
machine, in this one captured run** -- it is not a claim about any other hardware or workload.

## Source

```text
harness commit:      72a2e01594f0ad1e2c569a74a2a2600992d0fef6
raw stats-only JSON: 2026-08-02-similarity-baseline.json
JSON SHA-256:        3dcc639548da19a4fb70b0baf2442c5886967fde55126ec05410d0b70865226a
```

## Environment

All values below are read directly from the captured JSON's `machine_info`/`commit_info`.

| Field | Value |
|---|---|
| Timestamp (UTC) | 2026-08-02T19:34:43.156482+00:00 |
| OS | Darwin, release 25.5.0 |
| Architecture | arm64 |
| CPU | Apple M4 Pro (12 logical cores; physical core count not reported by the tool) |
| Python | CPython 3.12.7 |
| pytest | not reported by the tool (only `pytest-benchmark`'s own version is recorded) |
| pytest-benchmark | 5.2.3 |
| NumPy | 2.5.1 |
| OpenCV | 5.0.0 |
| improcv | 0.3.0a1.dev0 |
| Requested OpenCV threads | 1 |
| Observed OpenCV threads | 12 |
| Requested OpenCL | False |
| Observed OpenCL | False |
| `OMP_NUM_THREADS` | "1" |
| `OPENBLAS_NUM_THREADS` | "1" |
| `MKL_NUM_THREADS` | "1" |
| Commit dirty state | `false` |

The code under test never calls an OpenCV kernel, but the shared harness (`benchmarks/
conftest.py`) still records OpenCV/thread/OpenCL request-vs-observed state as part of every run's
`machine_info`, for consistency with the other benchmark families.

## Input scenarios

| n items | unordered pairs | path policy | insertion order | hash algorithm | hash size | hash bits | hash-value policy |
|---|---|---|---|---|---|---|---|
| 30 | 435 | synthetic, zero-padded, relative (`images/image_NNNNN.png`) | reverse canonical (`path.as_posix()`) | PHASH | 8 | 64 | odd-multiplier-mod-2^64 |
| 100 | 4,950 | synthetic, zero-padded, relative | reverse canonical | PHASH | 8 | 64 | odd-multiplier-mod-2^64 |
| 300 | 44,850 | synthetic, zero-padded, relative | reverse canonical | PHASH | 8 | 64 | odd-multiplier-mod-2^64 |

## Result regimes

| group | max_distance | expected matches | match density | materialized pair objects |
|---|---|---|---|---|
| `similarity-no-matches` | 0 | 0 | 0.0 | 0 |
| `similarity-all-matches` | 64 | `unordered_pairs` (435 / 4,950 / 44,850 for n=30/100/300) | 1.0 | `unordered_pairs` (435 / 4,950 / 44,850) |

## Command

```bash
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
uv run --group benchmark pytest benchmarks/benchmark_similarity.py \
  --benchmark-warmup=on \
  --benchmark-min-rounds=20 \
  --benchmark-storage=file:///tmp/improcv-similarity-benchmark-storage \
  --benchmark-save=similarity-baseline
```

## Integrity checks

- `commit_info.id`: `72a2e01594f0ad1e2c569a74a2a2600992d0fef6` (exact harness commit)
- `commit_info.dirty`: `false`
- 6 benchmark entries, all with a non-empty `group`, `params`, `extra_info`, and finite
  `median`/`mean`/`stddev`/`iqr`
- Group counts: `similarity-no-matches: 3`, `similarity-all-matches: 3`
- All 6 entries: `rounds >= 20` (`no-matches`: 12,481/1,871/242 at n=30/100/300; `all-matches`:
  2,346/198/21 at n=30/100/300)
- `stats.data` (the full per-round timing array) confirmed **absent** from every entry -- this is
  a compact, stats-only saved run, not a full-data capture
- Correctness/collection smoke (`--benchmark-disable -s`, run immediately before this capture):
  `6 passed`
- Raw JSON copied into `benchmarks/results/` byte-for-byte, unedited; SHA-256 of the copy matches
  the SHA-256 of the file captured in `/tmp` (see "Source" above)
- Both regimes at a given `n` share one identical session-scoped input mapping (confirmed by the
  harness's own fixture-sharing and input-immutability assertions, not re-derived here)
- No affine, hashing, discovery, or evaluation benchmark entries present in this file
  (similarity-only capture)

## No-matches results

| n items | unordered pairs | median | IQR | relative IQR | CV | median/pair | rounds | iterations |
|---|---|---|---|---|---|---|---|---|
| 30 | 435 | 89.71 µs | 8.46 µs | 9.4% | 6.6% | 206.2 ns | 12,481 | 1 |
| 100 | 4,950 | 594.25 µs | 18.57 µs | 3.1% | 4.2% | 120.1 ns | 1,871 | 1 |
| 300 | 44,850 | 4,389.58 µs | 125.71 µs | 2.9% | 7.8% | 97.9 ns | 242 | 1 |

Observed median growth: `30 -> 100` (pairs grew 11.379x) = **6.62x**; `100 -> 300` (pairs grew
9.061x) = **7.39x**.

## All-matches results

| n items | unordered pairs | median | IQR | relative IQR | CV | median/pair | rounds | iterations |
|---|---|---|---|---|---|---|---|---|
| 30 | 435 | 467.48 µs | 16.63 µs | 3.6% | 4.2% | 1,074.7 ns | 2,346 | 1 |
| 100 | 4,950 | 5,398.06 µs | 196.92 µs | 3.7% | 19.0% | 1,090.5 ns | 198 | 1 |
| 300 | 44,850 | 58,759.79 µs | 2,503.85 µs | 4.3% | 4.0% | 1,310.1 ns | 21 | 1 |

Observed median growth: `30 -> 100` (pairs grew 11.379x) = **11.55x**; `100 -> 300` (pairs grew
9.061x) = **10.89x**.

`median/pair` above is the **observed median normalized by the number of unordered input
pairs**, not an exact per-comparison or per-object cost: `no-matches` still includes the fixed
cost of validating, normalizing, and sorting the input regardless of pair count, and
`all-matches` additionally includes constructing, collecting, and sorting the full result list.

## Regime comparison

| n items | unordered pairs | no-matches median | all-matches median | all/no ratio |
|---|---|---|---|---|
| 30 | 435 | 89.71 µs | 467.48 µs | 5.21x |
| 100 | 4,950 | 594.25 µs | 5,398.06 µs | 9.08x |
| 300 | 44,850 | 4,389.58 µs | 58,759.79 µs | 13.39x |

**The all-matches/no-matches ratio compares two complete public workflows with different result
cardinalities and branch outcomes. It does not isolate an exact per-object materialization or
sorting cost.** `all_matches - no_matches` is not computed anywhere in this report as an exact
cost of constructing or sorting `SimilarImagePair` objects, and this difference is not divided by
the pair count.

## Scaling interpretation

Observations supported directly by the tables above, for this machine and this run only:

- Both regimes' medians increased monotonically across the three measured item counts
  (`30 < 100 < 300`).
- For `no-matches`, the two measured pair-count growth steps (11.379x, then 9.061x) produced
  observed median growth of 6.62x and 7.39x respectively -- sub-proportional to pair-count growth
  on this run. For `all-matches`, the same two steps produced 11.55x and 10.89x median growth --
  close to, and in the first step slightly above, proportional to pair-count growth.
- `median/pair` fell for `no-matches` as `n` grew (206.2 ns -> 120.1 ns -> 97.9 ns) but stayed
  roughly flat, with a slight upward drift, for `all-matches` (1,074.7 ns -> 1,090.5 ns ->
  1,310.1 ns). This is consistent with `no-matches`'s fixed validation/normalization/sorting
  overhead being amortized over more pairs as `n` grows, while `all-matches` additionally
  constructs, collects, and sorts a result list whose own size grows with the pair count.
- The all/no ratio itself grew with `n` (5.21x -> 9.08x -> 13.39x), consistent with
  `all-matches`'s additional per-pair work (constructing a `SimilarImagePair`, appending it, and
  sorting a growing result list) mattering more as the number of materialized pairs increases --
  reported only as an observed ratio between two complete workflows, not as an isolated
  materialization or sorting cost (see "Regime comparison" above).
- The code under test performs all `n(n-1)/2` unordered-pair comparisons regardless of regime,
  but three measured `n` values are not enough to prove or claim `O(n**2)`, exactly quadratic, or
  guaranteed quadratic scaling for the complete public workflow -- the growth figures above are
  reported as measured facts about this run, not as a proven complexity class. No claim is made
  about the representativeness of this synthetic, guaranteed-unique hash distribution for any
  real dataset, about real-world duplicate-detection accuracy, or about a universal similarity
  threshold.

## Measurement spread

All 6 entries, sorted by relative IQR, descending:

| case | relative IQR | coefficient of variation | median | rounds |
|---|---|---|---|---|
| `test_find_similar_image_pairs_no_matches[30-images]` | 9.4% | 6.6% | 89.71 µs | 12,481 |
| `test_find_similar_image_pairs_all_matches[300-images]` | 4.3% | 4.0% | 58,759.79 µs | 21 |
| `test_find_similar_image_pairs_all_matches[100-images]` | 3.7% | 19.0% | 5,398.06 µs | 198 |
| `test_find_similar_image_pairs_all_matches[30-images]` | 3.6% | 4.2% | 467.48 µs | 2,346 |
| `test_find_similar_image_pairs_no_matches[100-images]` | 3.1% | 4.2% | 594.25 µs | 1,871 |
| `test_find_similar_image_pairs_no_matches[300-images]` | 2.9% | 7.8% | 4,389.58 µs | 242 |

Relative IQR stayed in a narrow 2.9%-9.4% band across all six entries, with no case standing out
as unusually spread relative to the others by this measure. Coefficient of variation tells a
different story for one case: `test_find_similar_image_pairs_all_matches[100-images]` (CV 19.0%),
well above the rest (next-highest is 7.8%). Its saved `min`/`max`/`median` (`min`=5,097.71 µs,
`max`=9,815.96 µs, `median`=5,398.06 µs, over only 198 rounds) show `max` at roughly 1.82x the
median -- a smaller absolute spread than the high-CV cases seen in prior baselines, but with more
proportional impact here because this case has far fewer rounds (198) than the fast, many-round
cases elsewhere in this project's benchmarks; a handful of slower rounds move `stddev`/`mean`
noticeably in a small sample, while `median`/IQR (this project's primary statistics; see
`benchmarks/README.md`'s "Interpretation" section) stayed low and unremarkable. This is
consistent with ordinary OS-scheduler jitter affecting a small fraction of rounds, not a
systemic harness or code problem. The `pytest-benchmark` plugin reported no stability warnings
for this run.

**This compact saved run contains summary statistics, not full per-round data.**

## Limitations

- Single machine, single capture -- not repeated, not averaged across sessions.
- Three measured item counts (`30`/`100`/`300`) only.
- Only `PHASH` was measured -- no other perceptual hashing algorithm.
- Only `hash_size=8` was measured -- hash-size scaling is out of scope for this harness today.
- Hash values are synthetic and guaranteed unique by construction (an odd-multiplier transform
  modulo `2**64`), not real perceptual hashes of real images.
- Only a `dict` mapping type and `Path` keys were measured -- no other concrete mapping type or
  key type.
- Input mappings were always inserted in reverse canonical order -- no other insertion order was
  measured.
- Only the two extreme result densities (`0%` and `100%` match density) were measured -- no
  intermediate density.
- Only `max_distance` values of `0` and `64` were measured -- no other threshold.
- No filesystem access, image decoding, hash computation, or `discover_images` call is exercised
  anywhere in this harness.
- No duplicate grouping, clustering, or connected-component analysis was measured.
- No parallelism of any kind was measured.
- No memory measurement of any kind (no `sys.getsizeof`, no `tracemalloc`, no RAM estimation, no
  claim about `SimilarImagePair`'s size) -- this baseline measures time, not memory.
- No real-world duplicate-detection accuracy was measured or implied.
- Three measured item counts are not enough to prove an asymptotic complexity class for the
  complete public workflow.
- No comparison against any other machine, OS, or workload.
