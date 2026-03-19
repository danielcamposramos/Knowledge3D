# Full Benchmark Run Status

Date: 2026-03-18
Status: In progress

## Active run

- Command:
  - `CUDA_VISIBLE_DEVICES=0 /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -u scripts/run_enriched_benchmarks.py --full`
- Session id:
  - `full-736b4175b3a7`
- Stdout log:
  - `/tmp/k3d_full_benchmark_20260318.log`
- Resume state:
  - `/K3D/Knowledge3D.local/logs/health_log.full.run_state.json`
- Health log:
  - `/K3D/Knowledge3D.local/logs/health_log.jsonl`

## Completed so far

- ARC:
  - `10/120`
  - `37.144s`
  - `0.3095s/q`
- Math:
  - `20/20`
  - `32.421s`
  - `1.6211s/q`

## Important live finding

The requested `Math 500` scope is not being honored by the current benchmark stack.

Root cause:
- [math_competitions.py](/mnt/arquivos/EchoSystems%20AI%20Studios/Knowledge%203D%20Standard/GitHub/Knowledge3D/benchmarks/math_competitions.py) defaults to `dataset_mode="synthetic"` when `dataset_path` is not explicitly passed.
- [benchmark_health_check.py](/mnt/arquivos/EchoSystems%20AI%20Studios/Knowledge%203D%20Standard/GitHub/Knowledge3D/knowledge3d/tools/benchmark_health_check.py) instantiates `MathCompetitionBenchmark(knowledgeverse=object(), max_problems=count)` without forcing present-data mode.
- Result: the benchmark only runs the synthetic 20-problem guard set, not the real train set at `/K3D/K3D_llama_cpp/datasets/math/data/train.jsonl`.

## Confirmed post-run fixes

These should be applied after the current run finishes:

1. Remove the H19 meaning-star cap.
   - Load all quality-filtered meaning stars.
   - Keep only:
     - `min_languages >= 5`
     - stopword removal
     - foundation dedup with Math protection
   - No `max_stars` cap in Python.

2. Fix Math full-data loading.
   - Preferred fix: make `MathCompetitionBenchmark` try present-data mode first when the known dataset exists.
   - Acceptable alternative: make `benchmark_health_check.py` pass `dataset_mode="present"` for Math loading.

## Resume design now in place

- `scripts/run_enriched_benchmarks.py --full` writes:
  - per-suite completion summaries
  - session id
  - suite counts
  - completion state
- If the process crashes after a suite boundary, rerunning the same command should resume from the next suite using:
  - `/K3D/Knowledge3D.local/logs/health_log.full.run_state.json`
  - `session_id = full-736b4175b3a7`

## Current observation

- The run is currently inside the next large suite after Math.
- No additional suite boundary has been logged yet at the time this status file was written.
