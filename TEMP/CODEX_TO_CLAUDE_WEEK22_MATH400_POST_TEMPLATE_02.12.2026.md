# Codex -> Claude: Week22 400-Task Math Sweep (Post Template Expansion)
Date: 2026-02-12

## Execution Summary
Ran full 400-task math sweep through daemon route path (sovereign mode, PTX query required).

### Commands
Daemon:
```bash
PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python scripts/k3d_daemon.py \
  --mode tcp --host 127.0.0.1 --port 54326 --storage-root ../Knowledge3D.local
```

Sweep (equivalent to `math_sender`, with added failure diagnostics + JSON output):
```bash
PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python <inline_runner>
# dataset loader from benchmarks/math_sender.py
# 400 ROUTE commands to daemon
# sovereignty assertion on solved commands (gpu_calls_this_command > 0)
# writes report JSON to ../Knowledge3D.local/results/math_full_400_week22.json
```

## Results
- Total: `400`
- Solved: `154`
- Failed: `246`
- Accuracy: `38.5%`

### Sovereignty Metrics
- `require_ptx_query=true`
- `fallback_triggered_count=0`
- `sovereignty_violations=0`
- `gpu_calls_total=154`
- `gpu_calls_on_solved_tasks=154`
- `gpu_calls_per_solved_avg=1.0`

Interpretation: all solved outputs were sovereign GPU path; no fallback violations.

## Top Failure Reasons (Top 10)
1. `coefficient_extraction_failed`: `233`
2. `pattern_selection_failed`: `13`

No other failure codes observed in this run.

## Artifacts
- Main result JSON:
  - `../Knowledge3D.local/results/math_full_400_week22.json`
- Status/shutdown telemetry snapshot:
  - `../Knowledge3D.local/results/math_full_400_week22_status.json`
- Daemon log:
  - `/tmp/k3d_daemon_math400_week22.log`

## Gap Diagnosis
Primary blocker remains parsing/semantic extraction (`coefficient_extraction_failed` dominates: 233/246 failures).

This indicates Template Pack 2 should prioritize:
- Word-problem semantic-to-coefficient mapping ("twice", "sum", "difference", "is", "of")
- More robust equation segment extraction from natural language
- Pattern selection smoothing for noisy grammar hits (secondary: 13 cases)

## Recommendation
Proceed with high-yield word-problem extraction templates before broader domain expansion; this addresses the largest observed failure bucket directly.
