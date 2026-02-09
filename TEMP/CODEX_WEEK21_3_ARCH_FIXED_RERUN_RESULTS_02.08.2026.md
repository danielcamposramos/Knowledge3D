# Week 21.3 Re-run (Architecture-Fixed) Results

## Command
```bash
env PYTHONPATH=. python3 scripts/run_all_benchmarks.py \
  --max-arc-tasks 100 \
  --max-math-problems 100 \
  --max-lhe-questions 50 \
  --arc-enable-contrastive-learning \
  --arc-enable-validity-gates \
  --arc-enable-fuzzy-oracle \
  --arc-fuzzy-oracle-threshold 0.95 \
  --output-dir ../Knowledge3D.local/results/week21_3_architecture_fixed \
  --storage-root ../Knowledge3D.local
```

## Main Metrics
- ARC empty: **0.32**
- ARC enriched: **0.28**
- ARC delta (enriched-empty): **-0.04**
- Math empty: **0.00**
- Math enriched: **0.3333**
- Math delta: **+0.3333**
- LHE empty: **0.50**
- LHE enriched: **1.00**
- LHE delta: **+0.50**

## ARC Oracle/Generation Diagnostics (Enriched)
- `generated_pattern_total`: **686**
- `tasks_with_generated_patterns`: **100/100**
- `oracle_at_3`: **0.0**
- `oracle_at_10`: **0.0**
- `oracle_at_all`: **0.0**
- `fuzzy_oracle_at_all`: **0.05**
- `generation_failure_rate`: **1.0**
- `ranking_change_rate`: **0.45**

Pattern source accuracy:
- `legacy_pipeline`: **0.45** (9/20)
- `autonomous_generation`: **0.1940** (13/67)
- `contrastive_anti`: **0.4615** (6/13)

## Continuity / Runtime Signals
- `runtime_seed_knowledge`: **False** (as intended)
- Enriched galaxy counts:
  - Start: `Drawing 3055`, `Grammar 11915`, `Math 507`, `Reality 1584`, `3DObjects 434`
  - End:   `Drawing 3132`, `Grammar 12601`, `Math 507`, `Reality 1584`, `3DObjects 434`

## Log Signals
- `run.log` contains **214** `GALAXY LAZY` events.
- `run.log` contains **0** literal `Computing missing embeddings` phrase outside ARC lazy messages.
- Interpretation: benchmark-level reload/orchestration fix is active; remaining `GALAXY LAZY` is from per-task ARC candidate embedding misses in legacy generator flow.

## Historical Comparison (to previous run)
- ARC enriched: **0.28 -> 0.28** (delta 0.00)
- Math enriched: **0.3333 -> 0.3333** (delta 0.00)
- LHE enriched: **1.00 -> 1.00** (delta 0.00)

## Artifacts
- Summary JSON: `../Knowledge3D.local/results/week21_3_architecture_fixed/week14_benchmark_summary.json`
- Full log: `../Knowledge3D.local/results/week21_3_architecture_fixed/run.log`
- History log: `../Knowledge3D.local/benchmarks/run_all_benchmarks_history.jsonl`
- Usage log: `../Knowledge3D.local/logs/benchmark_usage_metrics.jsonl`
