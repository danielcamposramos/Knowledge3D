# Codex Diagnostic Brief for Claude

**Date:** 2026-02-08  
**Run analyzed:** `../Knowledge3D.local/results/diagnostic_telemetry_02.08/week14_benchmark_summary.json`

## 1) Verified current metrics

### ARC-AGI 2
- Empty mind accuracy: **32.0%** (32/100)
- Enriched accuracy: **28.0%** (28/100)
- Enriched tasks with generated patterns: **100/100**
- Enriched generated patterns total: **286**
- Pattern source accuracy (enriched):
  - `legacy_pipeline`: **50.0%** (9/18)
  - `autonomous_generation`: **23.17%** (19/82)

### Oracle diagnostics (critical)
- Empty mind:
  - `top_1_accuracy`: **0.32**
  - `oracle_at_3`: **0.0**
  - `oracle_at_10`: **0.0**
  - `oracle_at_all`: **0.0**
  - `generation_failure_rate`: **1.0**
- Enriched:
  - `top_1_accuracy`: **0.28**
  - `oracle_at_3`: **0.0**
  - `oracle_at_10`: **0.0**
  - `oracle_at_all`: **0.0**
  - `generation_failure_rate`: **1.0**
  - `ranking_change_rate`: **0.39**

### Other benchmarks (same run)
- Math enriched: **33.33%**
- LHE enriched: **100%**

### Historical comparison (same run output)
- ARC: **28.0% -> 28.0%** (maintained)
- Math: **40.0% -> 33.33%** (regression)
- LHE: **100.0% -> 100.0%** (maintained)

## 2) What this means

Primary bottleneck is **not ranking alone**.

- Ranking is active and reordering (`ranking_change_rate=0.39`).
- But `oracle_at_all=0.0` means exact-correct outputs are not present in candidate sets under current oracle/match criteria.
- Therefore ranking cannot recover accuracy if no exact-correct candidate exists.

So current bottleneck is **candidate generation quality / candidate representation mismatch**, with ranking as a secondary amplifier.

## 3) What we are still missing

1. **Winner-level provenance telemetry**
- We need explicit per-task fields: `winner_source`, `winner_score`, `winner_components`.
- Right now we know ranking changed, but not whether the chosen winner had better component signals.

2. **Exact-match normalization layer for ARC oracle**
- `oracle_at_all=0.0` suggests strict comparison mismatch.
- Need canonicalization before comparison: crop/pad normalization, dtype normalization, color remap normalization, shape-consistent transform checks.

3. **Generated pattern quality gates before ranking**
- Autonomous patterns are abundant but lower precision than legacy on this run.
- Need pre-ranking gating by:
  - transform validity on all train pairs,
  - confidence floor,
  - compositional sanity checks.

4. **Per-source precision/recall over time**
- Current source stats are useful but not enough.
- Add per-iteration curves:
  - source precision@1, source contribution to wins, source false-positive rate.

5. **Counterfactual evaluation path**
- Required to prove whether updates/learning help:
  - run same tasks with previous checkpoint/tree/routing snapshot.

## 4) Recommended next fix order

1. **ARC canonical oracle + candidate validation pass** (highest impact)
- Implement canonical comparison function for ARC grids.
- Reject candidates failing train-pair replay consistency.

2. **Winner provenance telemetry**
- Persist winner details and top-5 score component vectors per task.

3. **Generation quality gates**
- Keep only generated patterns that replay all train examples and pass confidence thresholds.

4. **Then retune ranking weights**
- After candidate correctness improves, rerun scoring-weight search.

5. **Then rerun 100-task ARC diagnostic**
- Expectation after canonical+gating:
  - `oracle_at_all` should rise above 0.
  - once `oracle_at_all` rises, ranking improvements should start translating into top-1 gains.

## 5) Fast interpretation for architecture discussion

- The system is generating and storing knowledge correctly (growth path works).
- The learning loop is active.
- The current plateau is mainly a **candidate correctness / matching problem**, not a missing generation mechanism.
- Fixing oracle/candidate validity should unlock the ranking improvements already in place.

