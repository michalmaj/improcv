# Multiclass evaluation baseline — 2026-08-02

## Scope

This is the first consciously reviewed baseline captured from the multiclass evaluation
benchmark harness (`benchmarks/benchmark_evaluation.py`). It covers four groups:

- `confusion_matrix`;
- `classification_metrics(..., average="macro")`;
- `multiclass_roc_auc_score(..., average="macro")`;
- `multiclass_average_precision_score(..., average="macro")`.

along two independent scaling axes: sample count (1,000/10,000/100,000, fixed 10 classes) for
all four functions, and ranking class count (3/10/100, fixed 10,000 samples) for the two ranking
functions on top of their own sample-count axis. Every dataset is deterministic and synthetic --
an explicit, deliberately **unsorted** `labels` tuple, `int64` `y_true`/`y_pred` from a
balanced-by-construction assignment with a fixed every-fifth-sample error policy, and (for
ranking) a seeded `float64` score matrix intentionally **not** row-normalized to a probability
simplex. Only `average="macro"` is measured; there is no `sample_weight` anywhere in this slice.

There is no raw NumPy or scikit-learn baseline: no single reference implementation shares this
API's full contract (explicit unsorted `labels`, full validation, no probability-simplex
requirement, canonical order-independent reductions, read-only results) -- see
`benchmarks/README.md`/`benchmarks/benchmark_evaluation.py` for the detailed justification. This
baseline measures how the public API itself scales, not a ratio against a semantically different
implementation.

This baseline does **not** cover `average="micro"`/`"weighted"`/`None`, `sample_weight`, binary
curves/metrics, `classification_metrics_from_confusion_matrix`, score ties as an axis,
`float16`/`float32` scores, or any comparison with another library or machine -- those remain
out of scope for this harness as it exists today. Every number below describes **this one
machine, in this one captured run** -- it is not a claim about any other hardware or workload.

## Source

```text
commit:              658a6bcc6b943a1f9e232be51149b2d52d1a08d2
raw stats-only JSON: 2026-08-02-evaluation-baseline.json
JSON SHA-256:        5d1beb309d6f8d94d60dc955514b1ccc9dd626100d9d9239d08899349b923f2f
```

## Environment

All values below are read directly from the captured JSON's `machine_info`/`commit_info`, except
where noted as procedural (recorded from the preflight checks, not present in the JSON itself).

| Field | Value |
|---|---|
| Timestamp (UTC) | 2026-08-02T03:57:33.761483+00:00 |
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
| Power source (procedural) | AC power, battery 79%, charging |
| Load averages (procedural) | approximately 2.6 / 3.9 / 4.6 on a 12-core machine, immediately before capture -- ordinary desktop background load (browser, IDEs), no build/test/container process running concurrently |

Evaluation does not invoke any OpenCV image kernel, but the shared harness (`benchmarks/
conftest.py`) still records OpenCV/thread/OpenCL request-vs-observed state as part of every run's
`machine_info`, for consistency with the affine and discovery baselines.

## Data scenarios

Setup and input allocation happen entirely in a session-scoped fixture, **before** the timed
`benchmark(...)` call -- never inside the timed closure.

### Label datasets

| samples | classes | labels policy | `y_true` dtype | `y_pred` dtype | prediction policy |
|---|---|---|---|---|---|
| 1,000 | 10 | explicit, unsorted (`1,2,...,9,0`) | int64 | int64 | every-fifth-next-label |
| 10,000 | 10 | explicit, unsorted (`1,2,...,9,0`) | int64 | int64 | every-fifth-next-label |
| 100,000 | 10 | explicit, unsorted (`1,2,...,9,0`) | int64 | int64 | every-fifth-next-label |

Diagnostic setup times (never a benchmark result): all three label datasets built in under 1 ms
each.

### Ranking datasets

| samples | classes | `y_score` shape | `y_score` bytes | score dtype/layout | score policy | rows sum to one |
|---|---|---|---|---|---|---|
| 1,000 | 10 | (1000, 10) | 80,000 | float64 / C | seeded-uniform-plus-true-class-boost | false |
| 10,000 | 10 | (10000, 10) | 800,000 | float64 / C | seeded-uniform-plus-true-class-boost | false |
| 100,000 | 10 | (100000, 10) | 8,000,000 | float64 / C | seeded-uniform-plus-true-class-boost | false |
| 10,000 | 3 | (10000, 3) | 240,000 | float64 / C | seeded-uniform-plus-true-class-boost | false |
| 10,000 | 100 | (10000, 100) | 8,000,000 | float64 / C | seeded-uniform-plus-true-class-boost | false |

Diagnostic setup times (never a benchmark result): the first-built ranking dataset (1000x10)
took 1.895 s -- first-use/allocation noise outside the timed region (the process's first call to
`np.random.default_rng` in this session), not representative of per-scenario cost. The remaining
four ranking datasets each built in under 6 ms (10000x10: 1 ms; 100000x10: 5 ms; 10000x3: <1 ms;
10000x100: 3 ms).

## Command

```bash
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
uv run --group benchmark pytest benchmarks/benchmark_evaluation.py \
  --benchmark-warmup=on \
  --benchmark-min-rounds=20 \
  --benchmark-storage=file:///tmp/improcv-evaluation-benchmark-storage \
  --benchmark-save=evaluation-baseline
```

## Integrity checks

- `commit_info.dirty`: `false`
- `commit_info.id`: `658a6bcc6b943a1f9e232be51149b2d52d1a08d2` (exact harness commit)
- 16 benchmark entries, all with a non-empty `group`, `params`, `extra_info`, and finite
  `median`/`mean`/`stddev`/`iqr`
- Group counts: `evaluation-confusion-matrix: 3`, `evaluation-classification-metrics: 3`,
  `evaluation-roc-auc-macro: 5`, `evaluation-average-precision-macro: 5`
- All 16 entries: `rounds >= 20` (label scenarios: 7004/716/70 for confusion_matrix at
  1,000/10,000/100,000, 6162/716/70 for classification_metrics; ranking scenarios: 313/29/20 for
  1,000/10,000/100,000 samples at 10 classes, 94/20 for 3/100 classes at 10,000 samples, for both
  ranking functions)
- `stats.data` (the full per-round timing array) confirmed **absent** from every entry -- this is
  a compact, stats-only saved run, not a full-data capture
- Correctness/collection smoke (`--benchmark-disable -s`, run immediately before this capture):
  `16 passed`
- Raw JSON copied into `benchmarks/results/` byte-for-byte, unedited; SHA-256 of the copy matches
  the SHA-256 of the file captured in `/tmp` (see "Source" above)
- No affine or discovery benchmark entries present in this file (evaluation-only capture)

## Confusion matrix

| samples | classes | median | IQR | relative IQR | CV | median/sample | growth vs. previous | rounds | iterations |
|---|---|---|---|---|---|---|---|---|---|
| 1,000 | 10 | 160.2 µs | 10.4 µs | 6.5% | 10.2% | 0.160 µs | -- | 7,004 | 1 |
| 10,000 | 10 | 1.490 ms | 36.5 µs | 2.5% | 2.9% | 0.149 µs | 9.30x | 729 | 1 |
| 100,000 | 10 | 14.87 ms | 202.9 µs | 1.4% | 1.1% | 0.149 µs | 9.98x | 69 | 1 |

## Classification metrics

| samples | classes | median | IQR | relative IQR | CV | median/sample | growth vs. previous | rounds | iterations |
|---|---|---|---|---|---|---|---|---|---|
| 1,000 | 10 | 182.5 µs | 8.08 µs | 4.4% | 28.7% | 0.183 µs | -- | 6,162 | 1 |
| 10,000 | 10 | 1.520 ms | 31.4 µs | 2.1% | 35.9% | 0.152 µs | 8.33x | 716 | 1 |
| 100,000 | 10 | 14.95 ms | 190.6 µs | 1.3% | 1.2% | 0.150 µs | 9.84x | 70 | 1 |

**Workflow ratio** (`classification_metrics` median / `confusion_matrix` median, same
`n_samples`): 1.14x at 1,000, 1.02x at 10,000, 1.01x at 100,000. **This is a ratio between two
complete, independently measured public workflows. It does not isolate the exact incremental
cost of reducing an already-computed confusion matrix** -- `classification_metrics - confusion_
matrix` is not computed here as an exact precision/recall/F1 cost, since two separately measured
benchmarks do not reliably isolate a difference this small.

## ROC AUC — sample scaling

Fixed 10 classes.

| samples | classes | median | IQR | relative IQR | CV | median/sample | median/(sample×class) | growth vs. previous | rounds | iterations |
|---|---|---|---|---|---|---|---|---|---|---|
| 1,000 | 10 | 3.392 ms | 98.0 µs | 2.9% | 2.2% | 3.392 µs | 339.2 ns | -- | 313 | 1 |
| 10,000 | 10 | 34.82 ms | 354.2 µs | 1.0% | 0.8% | 3.482 µs | 348.2 ns | 10.26x | 29 | 1 |
| 100,000 | 10 | 371.98 ms | 4.494 ms | 1.2% | 1.0% | 3.720 µs | 372.0 ns | 10.68x | 20 | 1 |

## Average precision — sample scaling

Fixed 10 classes.

| samples | classes | median | IQR | relative IQR | CV | median/sample | median/(sample×class) | growth vs. previous | rounds | iterations |
|---|---|---|---|---|---|---|---|---|---|---|
| 1,000 | 10 | 3.336 ms | 75.5 µs | 2.3% | 8.7% | 3.336 µs | 333.6 ns | -- | 320 | 1 |
| 10,000 | 10 | 35.28 ms | 548.0 µs | 1.6% | 1.7% | 3.528 µs | 352.8 ns | 10.58x | 30 | 1 |
| 100,000 | 10 | 371.54 ms | 2.419 ms | 0.7% | 0.6% | 3.715 µs | 371.5 ns | 10.53x | 20 | 1 |

## ROC AUC — class scaling

Fixed 10,000 samples.

| samples | classes | score matrix size | median | IQR | relative IQR | CV | median/class | median/(sample×class) | class growth vs. previous | median growth vs. previous | rounds | iterations |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10,000 | 3 | 240,000 B | 11.12 ms | 360.5 µs | 3.2% | 11.4% | 3,706.6 µs | 370.7 ns | -- | -- | 94 | 1 |
| 10,000 | 10 | 800,000 B | 34.82 ms | 354.2 µs | 1.0% | 0.8% | 3,481.7 µs | 348.2 ns | 3.33x | 3.13x | 29 | 1 |
| 10,000 | 100 | 8,000,000 B | 358.75 ms | 9.367 ms | 2.6% | 2.1% | 3,587.5 µs | 358.7 ns | 10.0x | 10.30x | 20 | 1 |

## Average precision — class scaling

Fixed 10,000 samples.

| samples | classes | score matrix size | median | IQR | relative IQR | CV | median/class | median/(sample×class) | class growth vs. previous | median growth vs. previous | rounds | iterations |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10,000 | 3 | 240,000 B | 10.99 ms | 234.5 µs | 2.1% | 3.1% | 3,663.4 µs | 366.3 ns | -- | -- | 94 | 1 |
| 10,000 | 10 | 800,000 B | 35.28 ms | 548.0 µs | 1.6% | 1.7% | 3,528.0 µs | 352.8 ns | 3.33x | 3.21x | 30 | 1 |
| 10,000 | 100 | 8,000,000 B | 350.59 ms | 4.024 ms | 1.1% | 0.7% | 3,505.9 µs | 350.6 ns | 10.0x | 9.94x | 20 | 1 |

## ROC AUC vs. average precision

On this machine and this captured dataset, their observed medians at matching scenarios were
close in every case (ratio AP/ROC-AUC between 0.977 and 1.013): `1000x10` 3.392 ms vs. 3.336 ms;
`10000x10` 34.82 ms vs. 35.28 ms; `100000x10` 371.98 ms vs. 371.54 ms; `10000x3` 11.12 ms vs.
10.99 ms; `10000x100` 358.75 ms vs. 350.59 ms. No general claim is made that either metric is
"faster" or "slower" -- the two implementations share most of their per-class one-vs-rest
machinery, and the observed differences here are within the kind of run-to-run variation already
visible in each metric's own IQR.

## Interpretation

Observations supported directly by the tables above, for this machine and this run only:

- All four operations' medians increased monotonically across their measured sample counts
  (1,000 < 10,000 < 100,000), and both ranking operations' medians increased monotonically
  across their measured class counts (3 < 10 < 100).
- For `confusion_matrix` and `classification_metrics`, the two observed 10x-sample-count steps
  produced growth in the 8.3x-10.0x range for both -- approximately proportional over the
  measured range. `median/sample` stayed essentially flat for both (`confusion_matrix`:
  0.160 -> 0.149 -> 0.149 µs; `classification_metrics`: 0.183 -> 0.152 -> 0.150 µs), consistent
  with that near-proportional growth.
- For both ranking functions' sample-count axis (fixed 10 classes), the two 10x-sample-count
  steps produced growth of 10.26x/10.68x (ROC AUC) and 10.58x/10.53x (average precision) --
  also approximately proportional. `median/sample` and `median/(sample×class)` both stayed within
  a narrow band across all three sizes for each function (ROC AUC: 3.39-3.72 µs per sample;
  average precision: 3.34-3.72 µs per sample), with a small upward drift at 100,000 samples that
  is reported as an observation, not attributed to a specific cause, from three measured points.
- For both ranking functions' class-count axis (fixed 10,000 samples): the 3->10 step (3.33x
  class-count growth) produced 3.13x (ROC AUC) / 3.21x (average precision) median growth, and the
  10->100 step (10x class-count growth) produced 10.30x (ROC AUC) / 9.94x (average precision)
  median growth -- both approximately proportional to class count over the measured range.
  `median/class` stayed within a narrow band across all three class counts for both functions
  (ROC AUC: 3,481.7-3,706.6 µs per class; average precision: 3,505.9-3,663.4 µs per class).
- The label-based workflows (`confusion_matrix`, `classification_metrics`) are two to three
  orders of magnitude faster than the ranking workflows at matching sample counts on this run
  (e.g. at 10,000 samples: ~1.5 ms vs. ~35 ms) -- expected, since the ranking functions perform
  one binary ranking computation per class plus `y_score`'s own copy/validation, while the
  label-based functions only build and reduce a `(10, 10)` confusion matrix.
- `classification_metrics`'s median was only 1.0x-1.1x `confusion_matrix`'s median at matching
  sample counts, smallest at the two larger sizes -- consistent with the confusion-matrix-building
  step dominating both functions' total cost at these sizes, though this ratio is explicitly not
  treated as an isolated "metrics reduction" cost (see "Classification metrics" above).

Three measured sample-count points and three measured class-count points are not enough to
establish a proven asymptotic complexity class for any of the four operations; every growth
figure above is reported as "approximately proportional over the measured range," not as a
demonstrated `O(n)` or `O(k)`. No claim is made here about any other machine or workload.

## Measurement spread

All 16 entries, sorted by relative IQR, descending:

| case | relative IQR | coefficient of variation | median | rounds |
|---|---|---|---|---|
| `test_confusion_matrix[1000x10]` | 6.5% | 10.2% | 160.2 µs | 7,004 |
| `test_classification_metrics_macro[1000x10]` | 4.4% | 28.7% | 182.5 µs | 6,162 |
| `test_multiclass_roc_auc_macro[10000x3]` | 3.2% | 11.4% | 11.12 ms | 94 |
| `test_multiclass_roc_auc_macro[1000x10]` | 2.9% | 2.2% | 3.392 ms | 313 |
| `test_multiclass_roc_auc_macro[10000x100]` | 2.6% | 2.1% | 358.75 ms | 20 |
| `test_confusion_matrix[10000x10]` | 2.5% | 2.9% | 1.490 ms | 729 |
| `test_multiclass_average_precision_macro[1000x10]` | 2.3% | 8.7% | 3.336 ms | 320 |
| `test_multiclass_average_precision_macro[10000x3]` | 2.1% | 3.1% | 10.99 ms | 94 |
| `test_classification_metrics_macro[10000x10]` | 2.1% | 35.9% | 1.520 ms | 716 |
| `test_multiclass_average_precision_macro[10000x10]` | 1.6% | 1.7% | 35.28 ms | 30 |
| `test_confusion_matrix[100000x10]` | 1.4% | 1.1% | 14.87 ms | 69 |
| `test_classification_metrics_macro[100000x10]` | 1.3% | 1.2% | 14.95 ms | 70 |
| `test_multiclass_roc_auc_macro[100000x10]` | 1.2% | 1.0% | 371.98 ms | 20 |
| `test_multiclass_average_precision_macro[10000x100]` | 1.1% | 0.7% | 350.59 ms | 20 |
| `test_multiclass_roc_auc_macro[10000x10]` | 1.0% | 0.8% | 34.82 ms | 29 |
| `test_multiclass_average_precision_macro[100000x10]` | 0.7% | 0.6% | 371.54 ms | 20 |

Relative IQR stayed under 7% for every entry, with no case standing out as unusually spread
relative to the others by this measure. Coefficient of variation tells a different story for two
cases: `test_classification_metrics_macro[1000x10]` (CV 28.7%) and
`test_classification_metrics_macro[10000x10]` (CV 35.9%), both far above the rest (next-highest
is 11.4%). Their `min`/`max`/`median` from the saved stats (`min`=162.5 µs/`max`=1,192.4 µs/
`median`=182.5 µs for 1,000 samples over 6,162 rounds; `min`=1,371.8 µs/`max`=16,102.9 µs/
`median`=1,520.3 µs for 10,000 samples over 716 rounds) show `max` at 6.5x and 10.6x the median
respectively, while `median`/IQR (this project's primary statistics; see `benchmarks/README.md`'s
"Interpretation" section) stayed low and unremarkable for both -- the same signature as the one
high-CV case already documented in the affine baseline (`apply_affine_image_mask[raw-64x64]`),
consistent with rare OS-scheduler preemption hitting a small fraction of rounds in a fast,
many-round case, not a systemic harness or code problem. `test_confusion_matrix[1000x10]` and
`test_multiclass_roc_auc_macro[10000x3]` show a smaller version of the same pattern (CV
10.2%/11.4% against relative IQR 6.5%/3.2%). The `pytest-benchmark` plugin reported no stability
warnings, and load averages did not rise noticeably over the ~107-second capture window.

**This compact saved run contains summary statistics, not full per-round data** -- unlike the
affine baseline's full-data capture, there is no `stats.data` array here to confirm the exact
fraction of affected rounds or the single largest outlier's value; the interpretation above is
based on `min`/`max`/`mean`/`stddev`/`median`/`iqr` alone, per this baseline's deliberate
stats-only format (see `benchmarks/results/README.md`). No observation was removed and no second
run was captured to make these numbers look different.

## Limitations

- Single machine, single run -- not repeated, not averaged across sessions.
- Synthetic, balanced-by-construction labels only -- no class imbalance was measured.
- One deterministic prediction-error policy (every fifth sample) -- no other error rate or
  pattern was measured.
- One score generator (seeded uniform plus a fixed true-class boost) -- no other score
  distribution or difficulty level was measured.
- Only `average="macro"` -- `"micro"`, `"weighted"`, and `None` are not covered.
- No `sample_weight` coverage.
- No score-tie handling measured as a separate axis.
- No binary-metric (`roc_curve`, `precision_recall_curve`, binary `roc_auc_score`/`average_
  precision_score`) coverage.
- No `float32`/`float16` score-matrix coverage (only `float64`).
- No list-vs-ndarray input comparison (`y_true`/`y_pred` are always ndarrays here).
- No raw NumPy or scikit-learn baseline, by design (see "Scope" above).
- Three measured sample-scaling points (1,000/10,000/100,000) and three measured class-scaling
  points (3/10/100) -- not enough to prove an asymptotic complexity class for any operation.
- No comparison against any other machine, OS, or workload.
