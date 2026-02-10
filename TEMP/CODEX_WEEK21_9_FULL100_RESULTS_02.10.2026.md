# Week 21.9 Full100 Results (GPU Migration + Negative Forms)

Date: 2026-02-10

## Run command
`/mnt/anaconda3/bin/conda run -n k3d-cranium env PYTHONPATH=. python scripts/run_all_benchmarks.py --model-persistence-mode unified --max-arc-tasks 100 --max-math-problems 100 --max-lhe-questions 50 --arc-enable-full-ptx --arc-enable-contrastive-learning --arc-enable-validity-gates --arc-enable-fuzzy-oracle --arc-fuzzy-oracle-threshold 0.95 --arc-constraint-mode penalty --arc-enable-negative-forms --arc-embedding-lazy-mode skip --track-curriculum-coverage --require-min-galaxies-per-block 5 --output-dir ../Knowledge3D.local/results/week21_9_full100_gpu_migration --storage-root ../Knowledge3D.local`

## Artifacts
- Summary: `../Knowledge3D.local/results/week21_9_full100_gpu_migration/week14_benchmark_summary.json`
- Usage metrics: `../Knowledge3D.local/logs/benchmark_usage_metrics.jsonl`
- History: `../Knowledge3D.local/benchmarks/run_all_benchmarks_history.jsonl`

## Core benchmark outcomes (enriched)
- ARC: `0.06` (6/100)
- Math: `0.3333`
- LHE: `1.0000`

## ARC diagnostics
- `generated_pattern_total`: `1000`
- `tasks_with_generated_patterns`: `100`
- `oracle_at_all`: `0.01`  **(first exact oracle unlock)**
- `fuzzy_oracle_at_all`: `0.06`
- `oracle_fuzzy_0_90`: `0.13`
- `oracle_fuzzy_0_95`: `0.06`
- `generation_failure_rate`: `0.99`

## Constraint component means
- `family`: `0.7920`
- `shape`: `0.9640`
- `palette`: `0.7391`  **(up strongly)**
- `object`: `0.9600`

## Filter telemetry
- `generation_filter_generated_total`: `4996`
- `generation_filter_accept_rate_mean`: `0.4153`
- `generation_filter_reject_rate_mean`: `0.5847`
- `rejected_was_better_count`: `11`
- `rejected_was_better_rate`: `0.11`

## Oracle failure mode counts
- `palette`: `66`
- `object_count`: `48`
- `shape`: `34`
- `family`: `0`
- `generation_gap`: `4`
- `near_miss`: `3`

## Architecture and persistence validation
- Solver path: `arc_ptx_ops`
- PTX usage rates:
  - `ptx_full_used_rate`: `1.0`
  - `ptx_ranking_used_rate`: `1.0`
  - `ptx_oracle_used_rate`: `1.0`
- Persistence:
  - `shared_instance=true`
  - same instance id in empty/enriched block
  - `arc_embedding_lazy_mode=skip`
- World continuity:
  - `runtime_seed_knowledge=false` (not a clean world run)
  - storage root reused: `../Knowledge3D.local/galaxies_enriched`
  - Grammar growth in run: `30539 -> 31539` (`+1000`)

## Runtime
- Total elapsed: `976.00s` (`16.27 min`) for ARC(100)+Math(100)+LHE(50)
- ARC enriched block: `338.99s`

## Comparison vs Week 21.8 full100
- ARC accuracy: `0.05 -> 0.06` (+0.01)
- `oracle_at_all`: `0.00 -> 0.01` (+0.01)
- `fuzzy_oracle_at_all`: `0.05 -> 0.06` (+0.01)
- `oracle_fuzzy_0_90`: `0.12 -> 0.13` (+0.01)
- palette score: `0.6356 -> 0.7391` (+0.1035)

## Interpretation
- PTX contract + unified persistence are stable and correct.
- Week 21.9 changes produced measurable but modest oracle/accuracy gains.
- Largest quality gain is palette consistency.
- Remaining blocker is candidate quality in object-count + palette exactness, not routing.

## Suggested Week 21.9b next patch
1. Add object-count constrained generation templates (pre-ranking).
2. Move constraint-penalty fusion to PTX (eliminate Python post-score multiplier).
3. Add explicit top-k rescue lane (keep top 8-16 pre-oracle) to reduce generation_gap.
4. Add kernel duty-cycle telemetry (sum kernel-time / stage-time), since coarse GPU snapshots under-report utilization.

