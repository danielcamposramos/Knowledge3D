# Codex Week 19.6 Progress Report (Benchmark Universe + ARC Wiring)

Date: 2026-02-08

## Implemented

1. Benchmark universe downloader:
- Added `scripts/download_all_benchmarks.py`.
- Supports benchmark manifest by tier (`prize`, `standard`, `specialized`).
- Supports `git`, direct file, archive extraction, and `manual_required` artifacts.
- Writes reproducible manifest to `download_manifest.json`.

2. Unified global runner:
- Added `scripts/run_all_global_benchmarks.py`.
- Runs integrated K3D benchmarks (ARC, Math, LHE) with empty vs enriched isolation.
- Inventories external benchmark assets (presence/file counts/sample paths).
- Optional proxy runs (`--run-proxy`) for GSM8K and MMLU to track readiness/perf trends.
- Writes `global_benchmark_summary.json` plus integrated benchmark JSON outputs.

3. ARC autonomous generation wiring:
- Updated `benchmarks/arc_agi_2_adapter.py` to attach generation telemetry:
  - `generated_pattern_count`
  - `generated_pattern_confidence_mean`
  - detailed `generated_patterns`
- Updated `benchmarks/arc_agi_2.py` summary to aggregate:
  - `generated_pattern_total`
  - `tasks_with_generated_patterns`

4. Tests:
- Added `tests/test_global_benchmark_scripts.py`.
- Coverage includes downloader list mode, dry-run manifest output, and global runner summary generation.

## Validation

### Unit/Integration tests
- Command:
  - `env PYTHONPATH=. pytest -q tests/test_navigator_specialist.py tests/test_week19_autonomous_generation.py tests/test_benchmarks.py tests/test_global_benchmark_scripts.py`
- Result: `21 passed`.

### Real benchmark download execution
- Command:
  - `python3 scripts/download_all_benchmarks.py --root ../Knowledge3D.local/datasets/global_benchmarks --benchmarks gpqa,mmlu,gsm8k,humaneval --timeout 120`
- Result:
  - `gpqa`: cloned
  - `gsm8k`: cloned
  - `humaneval`: cloned
  - `mmlu`: downloaded + extracted

### Full benchmark-universe download pass
- Command:
  - `python3 scripts/download_all_benchmarks.py --root ../Knowledge3D.local/datasets/global_benchmarks --timeout 120`
- Coverage:
  - 15 benchmark specs tracked
  - 12 artifacts already present/downloaded
  - 3 marked `manual_required` (ARC prize bundle, IMO challenge assets, PIQA source mirror)

### Unified global benchmark execution
- Command:
  - `python3 scripts/run_all_global_benchmarks.py --output-dir ../Knowledge3D.local/results/week19_6_global_benchmarks --storage-root ../Knowledge3D.local --dataset-root ../Knowledge3D.local/datasets/global_benchmarks --max-arc-tasks 20 --max-math-problems 20 --max-lhe-questions 20 --run-proxy --max-proxy-questions 20`
- Result:
  - ARC-AGI 2: `25.00% -> 25.00%` (`+0.00%`)
  - Math: `0.00% -> 33.33%` (`+33.33%`)
  - LHE: `50.00% -> 100.00%` (`+50.00%`)
  - GSM8K proxy: `0/20`
  - MMLU proxy: `4/20` (20%)

### Unified global benchmark execution (latest quick pass)
- Command:
  - `python3 scripts/run_all_global_benchmarks.py --output-dir ../Knowledge3D.local/results/week19_6_global_benchmarks --storage-root ../Knowledge3D.local --dataset-root ../Knowledge3D.local/datasets/global_benchmarks --max-arc-tasks 10 --max-math-problems 10 --max-lhe-questions 10 --run-proxy --max-proxy-questions 10`
- Result:
  - ARC-AGI 2: `20.00% -> 20.00%` (`+0.00%`)
  - Math: `0.00% -> 40.00%` (`+40.00%`)
  - LHE: `50.00% -> 100.00%` (`+50.00%`)
  - GSM8K proxy: `0/10`
  - MMLU proxy: `2/10` (20%)
  - Global inventory readiness: `13/15` benchmark directories detected with assets

### Galaxy growth check
- Command:
  - `python3 scripts/monitor_galaxy_growth.py --storage-root ../Knowledge3D.local/galaxies_enriched`
- Snapshot:
  - Grammar generated entries: `58` (11.6%)
  - Drawing generated entries: `1`
  - 3DObjects generated entries: `1`

## Notes

- Sovereign hot path remains unchanged (no external dependency inserted into PTX/RPN execution path).
- New downloader/runner are ingestion/evaluation tooling only.
- ARC autonomous generation is now measurable at benchmark-output level through generated-pattern telemetry.
