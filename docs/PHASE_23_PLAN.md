# PHASE 23: AUTO-CURATED ARC/HLE BENCHMARK

## GOAL
- 50 zero-shot questions — multi-modal (text+image+audio+3D).
- Measure accuracy, honesty, cross-modal consistency.
- No training — pure geometric reasoning.

## COMMAND
```bash
conda activate k3d-cranium
PYTHONPATH=. python -m knowledge3d.tools.phase23.benchmark_runner --questions 50
```

## OUTPUT
- `logs/phase23_benchmark_report.json`
- `docs/PHASE_23_RESULTS.md`
