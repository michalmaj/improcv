# Benchmark results

Raw `pytest-benchmark` JSON output, committed selectively -- not every local run, only
deliberately reviewed baselines.

## Policy

- Only consciously reviewed baseline JSON files are committed here, produced with the "Stable
  local run" command in `benchmarks/README.md` -- a run produced after requesting one OpenCV
  thread and disabling OpenCL, with both the requested and the observed state recorded in
  `machine_info`, plus an explicit warm-up phase and at least 20 rounds.
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

## Filename format

```text
YYYY-MM-DD-augmentation-baseline.json
```

An optional, hand-written narrative Markdown file with the same date/topic prefix may accompany a
result to explain something not obvious from the raw numbers alone (mirroring this project's C++
sibling's own dated result notes) -- not required for every result.

## Current contents

- [`2026-08-01-augmentation-baseline.json`](2026-08-01-augmentation-baseline.json) -- the raw,
  unedited `pytest-benchmark` output. Source of truth for every number; not meant to be read
  directly.
- [`2026-08-01-augmentation-baseline.md`](2026-08-01-augmentation-baseline.md) -- the reviewed
  narrative interpretation of that JSON: environment, integrity checks, per-case tables,
  observations, measurement spread, and limitations.

Both files describe the same capture, at commit `55ed9b6d92942b35319b13faf95938c51bc4cbc9`, with
`commit_info.dirty: false`. Per the immutable-results policy above, this pair is never overwritten
by a future result -- a later baseline gets its own new dated filename.
