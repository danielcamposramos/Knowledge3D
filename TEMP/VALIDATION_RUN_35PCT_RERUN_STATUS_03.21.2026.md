# 35% Validation Rerun Status

**Date:** 2026-03-21
**Purpose:** Rerun the 35% validation slice after:

- non-finite numeric crash guards in `knowledgeverse.py`
- full meaning-layer load correction in `ingest_meaning_layer.py`

## Command

```bash
export CUDA_VISIBLE_DEVICES=0
/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python scripts/run_enriched_benchmarks.py \
  --full \
  --storage-root /K3D/Knowledge3D.local \
  --arc-max 42 \
  --math-max 500 \
  --gsm8k-max 462 \
  --lhe-max 35 \
  --mmlu-max 4915 \
  2>&1 | tee /tmp/k3d_validation_35pct_rerun_03.21.2026.log
```

## Pre-run validation

- `python3 -m compileall` on touched files: passed
- meaning-layer selection check: `117497 / 117497`
- focused LHE smoke with full ingest: completed `2` questions without the infinity crash

## Artifacts

- live log: `/tmp/k3d_validation_35pct_rerun_03.21.2026.log`
- run state: `/K3D/Knowledge3D.local/logs/health_log.full.run_state.json`
