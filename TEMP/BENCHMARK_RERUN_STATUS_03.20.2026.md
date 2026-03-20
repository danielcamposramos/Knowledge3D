# Benchmark Rerun Status — 2026-03-20

## Outcome So Far

The benchmark rerun hang was diagnosed and fixed.

What looked like a "full-load ingest hang" was actually quadratic-style write behavior in the meaning-layer ingestion path:

- `scripts/ingest_math_rules.py` completed normally on its own
- `scripts/ingest_meaning_layer.py --full-load` selected about `71k` stars quickly, but the old per-entry `GalaxyManager.upsert_entry()` path degraded badly at full-load scale
- full-load ingest is now batched and observable

The later math crash was a separate issue:

- `knowledge3d/knowledgeverse/knowledgeverse.py`
- `_select_composed_head_candidate()`
- `subject_label` could be referenced before assignment on the real `MATH_TASK` path

That crash is fixed.

## Fixes Landed

### 1. Meaning-layer ingest no longer stalls at full-load

Files:

- `scripts/ingest_meaning_layer.py`

Changes:

- staged per-galaxy bulk writes instead of repeated linear replacement scans
- added explicit progress logging for load, selection, purge, staging, and write phases

Validation:

- standalone `--full-load` ingest completed in about `52.9s`
- selected set during validation: `72177 / 117497`

### 2. Math rules ingest now reports progress

Files:

- `scripts/ingest_math_rules.py`

Changes:

- progress callback support
- explicit staging / completion output

Validation:

- standalone math-rules ingest completed
- `695` entries ingested

### 3. Math benchmark crash fixed

Files:

- `knowledge3d/knowledgeverse/knowledgeverse.py`

Changes:

- initialized `subject_label` safely for math-path candidate selection
- preserved algebra signal when present

Validation:

- 1-question real-math smoke run completed without the prior `UnboundLocalError`

### 4. Long suites now emit periodic progress

Files:

- `benchmarks/math_competitions.py`
- `benchmarks/gsm8k.py`
- `benchmarks/mmlu.py`
- `benchmarks/last_humanity_exam.py`
- `benchmarks/arc_agi_2.py`
- `knowledge3d/tools/benchmark_health_check.py`
- `scripts/run_enriched_benchmarks.py`

Changes:

- per-suite progress callbacks
- suite-specific progress cadence:
  - `math`: every 10
  - `gsm8k`: every 25
  - `mmlu`: every 100
  - `lhe`: every 10
  - `arc`: every 10
- runner prints normalized progress lines with completion %, running accuracy, elapsed time, and subject/domain markers when available

## Current Live Rerun

The current rerun is active and healthy.

Command:

```bash
export CUDA_VISIBLE_DEVICES=0
/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -u scripts/run_enriched_benchmarks.py \
  --full --full-load \
  --log /K3D/Knowledge3D.local/logs/health_log.jsonl \
  --journal /K3D/Knowledge3D.local/logs/sleeptime_journal.jsonl \
  2>&1 | tee /tmp/k3d_benchmark_rerun_postfix_03.20.2026.log
```

Evidence paths:

- stdout log: `/tmp/k3d_benchmark_rerun_postfix_03.20.2026.log`
- health log: `/K3D/Knowledge3D.local/logs/health_log.jsonl`
- run state: `/K3D/Knowledge3D.local/logs/health_log.full.run_state.json`

Session state:

- `session_id`: `full-90b2ae46bc08`
- resume behavior: `arc` is skipped/resumed from existing session rows

Latest observed live output:

```text
Starting math benchmark (500 questions)...
[math] 10/500 (2.00%) correct=0 running_acc=0.00% elapsed=51.5s
[math] 20/500 (4.00%) correct=0 running_acc=0.00% elapsed=74.8s
[math] 30/500 (6.00%) correct=0 running_acc=0.00% elapsed=100.3s
[math] 40/500 (8.00%) correct=0 running_acc=0.00% elapsed=123.6s
[math] 50/500 (10.00%) correct=0 running_acc=0.00% elapsed=146.7s
[math] 60/500 (12.00%) correct=0 running_acc=0.00% elapsed=168.9s
```

That confirms:

- the run is no longer silent during long suites
- the rerun progressed past the previous math crash point
- full-load ingest + math-rules ingest both completed successfully inside the rerun

## Notes

- `health_log.jsonl` is still accumulative by design
- suite resume remains suite-granular, not question-granular
- during benchmark-backed suites, rows are appended after suite completion; progress now comes from stdout rather than partial health-log writes

## Next Expected Milestones

If the run continues normally, the next visible checkpoints should be:

1. additional `math` progress lines
2. final `math` suite summary
3. live `gsm8k`, `lhe`, and `mmlu` progress lines
4. sleep-time consolidation output at the end
