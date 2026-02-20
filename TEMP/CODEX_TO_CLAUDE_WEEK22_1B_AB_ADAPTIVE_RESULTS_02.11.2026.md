# Codex → Claude: Week 22.1b A/B (Adaptive Penalties) Results

**Date:** 2026-02-11  
**Baseline:** `../Knowledge3D.local/results/week22_1b_forced_nav_full100/week14_benchmark_summary.json`  
**Adaptive Variant:** `../Knowledge3D.local/results/week22_1b_adaptive_penalties_full100/week14_benchmark_summary.json`

## 1) A/B Outcome Summary

Adaptive penalties were applied successfully, but produced **no top-line metric gain** in this run.

### Benchmark deltas (enriched)
- ARC: `0.06 -> 0.06` (delta `+0.00`)
- Math: `0.00 -> 0.00` (delta `+0.00`)
- LHE: `0.14 -> 0.14` (delta `+0.00`)
- MMLU: skipped both (`max-mmlu-questions=0`)

### ARC oracle/generation deltas
- `oracle_at_all`: `0.01 -> 0.01`
- `oracle_at_10`: `0.01 -> 0.01`
- `fuzzy_oracle_at_all`: `0.06 -> 0.06`
- `oracle_fuzzy_0_90`: `0.13 -> 0.13`
- `generation_failure_rate`: `0.99 -> 0.99`
- `generation_filter_accept_rate_mean`: `0.4253 -> 0.4253`
- `generation_filter_generated_total`: `6869 -> 6869`

### Secondary movement
- `ranking_palette_score_mean`: `0.7543 -> 0.7579` (+0.0036)
- `rejected_was_better_count`: `13 -> 14` (+1)

## 2) Adaptive Penalty Validation (Mechanism Check)

Adaptive mode is functioning and applying non-default weights:

- baseline: `enabled=false`, `applied=false`
- adaptive: `enabled=true`, `applied=true`
- adaptive weights used:
  - `family_penalty_weight: 0.5`
  - `shape_penalty_weight: 0.934`
  - `palette_penalty_weight: 2.1171`
  - `object_penalty_weight: 1.4944`

Conclusion: **no bug in adaptive toggle**; the bottleneck is upstream candidate quality / oracle reachability, not static-vs-adaptive weight selection.

## 3) Curriculum / Coverage / Sovereignty Status

### Forced-navigation curriculum
- Coverage remained strong in both runs:
  - unique queried galaxies: `[3DObjects, Drawing, Grammar, Math, Reality]`
  - avg queried galaxies per ARC task: `5.0`
  - cross-galaxy navigation rate: `1.0`

### PTX and persistence
- PTX rates stable at 1.0 (`full`, `ranking`, `oracle`)
- unified persistence stable (`shared_instance=true`, same root for empty/enriched)
- lazy embedding mode remained `skip`

## 4) Continuity / World State Note

Both A/B runs used the persistent unified world (`../Knowledge3D.local/galaxies_enriched`) and did **not** start clean.

- Baseline grammar growth within run: `43901 -> 45901` (+2000)
- Adaptive grammar growth within run: `47905 -> 49905` (+2000)

## 5) Architectural Readout

- Week 22.1b routing objective remains met (coverage + navigation gates pass).
- Adaptive penalties alone do not unlock oracle/generation under current candidate pipeline.
- Most likely immediate leverage: bounded rejected-candidate oracle rescue + generation-side quality changes (not ranking-only).

## 6) Recommended Immediate Next Test

Run a controlled variant with:
1. Oracle-only rejected rescue pass (top-16 rejected candidates)
2. Keep top-1 prediction lane unchanged (integrity-safe)
3. Same forced-navigation setup for comparability

Success criteria for next run:
- `oracle_at_all > 0.01`
- `fuzzy_oracle_at_all > 0.06`
- `generation_failure_rate < 0.99`

