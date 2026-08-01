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
YYYY-MM-DD-augmentation-baseline.json
```

An optional, hand-written narrative Markdown file with the same date/topic prefix may accompany a
result to explain something not obvious from the raw numbers alone (mirroring this project's C++
sibling's own dated result notes) -- not required for every result.

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

Both files describe the same capture, at commit `55ed9b6d92942b35319b13faf95938c51bc4cbc9`, with
`commit_info.dirty: false`. Per the immutable-results policy above, this pair is never overwritten
by a future result -- a later baseline gets its own new dated filename. Going forward, new
baselines are expected to use the compact, stats-only saved-run format described above unless a
specific diagnostic need calls for a full-data capture again.
