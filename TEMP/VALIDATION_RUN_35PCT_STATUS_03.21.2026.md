# 35% Stratified Validation Run Status

**Date:** 2026-03-21
**Status:** Running

## Command

```bash
CUDA_VISIBLE_DEVICES=0 /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
  scripts/run_enriched_benchmarks.py \
  --full \
  --storage-root /K3D/Knowledge3D.local \
  --arc-max 42 \
  --math-max 500 \
  --gsm8k-max 462 \
  --lhe-max 35 \
  --mmlu-max 4915
```

## Counts

| Suite | Count |
|------|------:|
| ARC | 42 |
| Math | 500 |
| GSM8K | 462 |
| LHE | 35 |
| MMLU | 4915 |

## Live Artifacts

- Stdout log: `/tmp/k3d_validation_35pct_03.21.2026.log`
- Health log: `/K3D/Knowledge3D.local/logs/health_log.jsonl`
- Run state: `/K3D/Knowledge3D.local/logs/health_log.full.run_state.json`

## Confirmed Startup

- Meaning layer selected `35579 / 117497` stars
- Meaning layer staged `227` language-to-math symlinks
- Math rules ingest completed with `1199` total entries (`483` inserted, `716` updated)
- Benchmark phase entered `ARC`

## First Live Progress

- `ARC`: `10/42`, `2 correct`, `20.00%` running accuracy
