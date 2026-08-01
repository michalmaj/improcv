# Benchmark results

Raw `pytest-benchmark` JSON output, committed selectively -- not every local run, only
deliberately reviewed baselines.

## Policy

- Only consciously reviewed baseline JSON files are committed here, produced with the "Stable
  local run" command in `benchmarks/README.md` (single-threaded OpenCV, explicit warm-up, at
  least 20 rounds).
- A new run creates a **new** file. Existing result files are never overwritten.
- A run later found to be flawed is either replaced by a new file (with a note in this directory,
  or in an accompanying Markdown file, explaining why) or superseded by a newer dated file --
  never silently amended or deleted.
- Every committed result must be traceable to an environment: the JSON itself carries
  `machine_info` (including the `improcv`/NumPy/OpenCV versions and the actual observed OpenCV
  thread count and OpenCL state, added by `benchmarks/conftest.py`) and `commit_info` (including
  the commit SHA and whether the working tree was dirty at capture time).
- Results captured on different machines are not directly comparable to each other -- only to
  other results captured on the *same* machine.

## Filename format

```text
YYYY-MM-DD-augmentation-baseline.json
```

An optional, hand-written narrative Markdown file with the same date/topic prefix may accompany a
result to explain something not obvious from the raw numbers alone (mirroring this project's C++
sibling's own dated result notes) -- not required for every result.

## Current contents

No baseline JSON has been committed yet. The first one will be added in a follow-up PR, after
this benchmark harness itself has been reviewed -- see `benchmarks/README.md`'s "Results" section.
