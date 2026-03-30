# Phase E Proof Bundle

This folder preserves the runtime artifacts for the March 30, 2026 Phase E milestone runs.

## Included Evidence

- `live_arc3/arc3_live_ls20_level1_20260330.jsonl`
  The successful live ARC3 run on `ls20-9607627b`. This is the run that reached `levels_completed=1`.

- `benchmark_20x/`
  The full 20-question-per-suite benchmark artifacts from `phase_e_20260330_143210`, including:
  - per-suite result JSONL files
  - per-suite progress JSONL files
  - `summary.json`
  - `summary.partial.json`
  - `full_results.json`

## Why This Is In-Repo

These files are the proof bundle for the Phase E claims:

- first live ARC3 level completion
- full benchmark pipeline with streamed evidence
- local ARC3 benchmark success

They are copied from `Knowledge3D.local/logs/` into the tracked repository so the milestone survives local runtime cleanup.
