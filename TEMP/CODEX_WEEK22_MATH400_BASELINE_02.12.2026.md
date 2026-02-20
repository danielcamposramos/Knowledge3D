# Week 22 Math Baseline (400 tasks) - 2026-02-12

## Command
- `scripts/run_all_benchmarks.py`
- Mode: `unified`
- Enabled benchmark: Math only (`--max-math-problems 400`, ARC/LHE/MMLU skipped)
- Sovereignty enforcement: `--enforce-sovereignty`
- CUDA include env set for NVRTC (`CPATH`, `CPLUS_INCLUDE_PATH`, `CUDA_PATH`)

## Result
- Empty mind: `0/400` (`0.0%`)
- Enriched: `0/400` (`0.0%`)
- Delta: `+0.0%`

## Diagnostics
- `predicted_none_count`: `400`
- `predicted_none_rate`: `1.0`
- `predicted_numeric_count`: `0`
- `expected_numeric_count`: `400`
- `route_specialist_counts`: `{"math": 400}`

## Sovereignty Summary (Runner)
- Enforcement active: `true`
- Solved tasks: `0`
- GPU-verified solved tasks: `0`
- Fallback triggered: `0`
- Compliance: `100%` (trivially, because no solved tasks)

## Notes
- Infrastructure is stable and run completed at 400 scale.
- The current blocker is solver capability/composition (`predicted_none_rate=1.0`), not routing.
- If we need non-trivial sovereignty proof inside this benchmark path (not just daemon sender path), we must produce solved tasks first or wire per-task GPU telemetry directly into benchmark result rows for attempted solves.
