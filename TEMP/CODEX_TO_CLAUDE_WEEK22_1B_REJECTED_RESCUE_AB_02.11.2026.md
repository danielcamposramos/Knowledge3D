# Codex -> Claude: Week 22.1b Oracle-Rejected-Rescue A/B

**Date:** 2026-02-11  
**Goal:** Implement oracle-only rejected-candidate rescue and run controlled A/B.

## Implemented

### 1) Oracle-only rejected rescue (bounded top-k)
- Added adapter flags:
  - `enable_oracle_rejected_rescue`
  - `oracle_rejected_rescue_size`
- Added generation-stage rejected pool builder:
  - `_build_oracle_rejected_rescue_candidates(...)`
- Added oracle metrics augmentation:
  - `_augment_oracle_metrics_with_rejected_rescue(...)`
- Strict contract preserved:
  - rescue path augments oracle diagnostics only
  - top-1 prediction selection unchanged

### 2) Runner/benchmark wiring
- `scripts/run_all_benchmarks.py` new CLI flags:
  - `--arc-enable-oracle-rejected-rescue`
  - `--arc-oracle-rejected-rescue-size`
- Propagated through:
  - runner -> `ARCAGI2Benchmark` -> `ArcAgi2Adapter`
- Runtime config + console reporting added.

### 3) Tests
- `tests/test_arc_agi_2_adapter.py`
  - added rescue-metric augmentation test
  - added rejected-rescue candidate dedupe test
- Result: `19 passed`

## Controlled A/B Runs (ARC-only, 100 tasks)

### Baseline (control)
- Output: `../Knowledge3D.local/results/week22_1b_baseline_arc100_control/week14_benchmark_summary.json`
- Config: week22_1b forced navigation, no adaptive, no oracle rejected rescue

### Variant (rescue enabled)
- Output: `../Knowledge3D.local/results/week22_1b_oracle_rejected_rescue_arc100/week14_benchmark_summary.json`
- Same config + `--arc-enable-oracle-rejected-rescue --arc-oracle-rejected-rescue-size 16`

## A/B Metrics (enriched ARC)

No delta observed:
- `top_1_accuracy`: `0.06 -> 0.06`
- `oracle_at_all`: `0.01 -> 0.01`
- `oracle_at_10`: `0.01 -> 0.01`
- `fuzzy_oracle_at_all`: `0.06 -> 0.06`
- `oracle_fuzzy_0_90`: `0.13 -> 0.13`
- `generation_failure_rate`: `0.99 -> 0.99`
- `rejected_was_better_count`: `13 -> 13`

## Key Diagnostic Finding

Rescue lane had **no candidates** to add under current flow:
- `oracle_rejected_rescue_candidate_count` (mean): `0.0`
- `oracle_rejected_rescue_exact` tasks: `0`
- `oracle_rejected_rescue_fuzzy` tasks: `0`

Interpretation:
- In current `constraint_mode=penalty`, rejected candidates are already present in ranked pool (or deduped into existing signatures), so the additional oracle-rejected pool contributes nothing.
- Therefore this patch is mechanically correct but not leverageful in the current pipeline shape.

## Recommendation

Given this result, the next leverage is not this rescue lane itself; it is either:
1. **Generation-side quality improvement** (reduce `generation_failure_rate=0.99`), or
2. **Constraint/rejection flow change** where rejected candidates are actually excluded from ranking and can then be rescued for oracle-only diagnostics.

