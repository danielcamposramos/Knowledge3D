# Week 21.3 Oracle Unlock V2 Report

Date: 2026-02-08
Run: `../Knowledge3D.local/results/week21_3_oracle_unlock_v2/week14_benchmark_summary.json`

## What was implemented
- `benchmarks/arc_agi_2_adapter.py`
  - Strict train-pair profile built before ranking and reused by gates + diagnostics.
  - Added inferred transform family and family compatibility gating.
  - Added source-precision + ternary quality + train-pair similarity + novelty scoring.
  - Added stratified fuzzy oracle metrics (`0.80/0.85/0.90/0.95` + exact).
  - Added family-reject accounting and diagnostics.
- `benchmarks/arc_agi_2.py`
  - Aggregates stratified fuzzy oracle metrics and family-reject means.
- `tests/test_arc_agi_2_adapter.py`
  - Added coverage for inferred family profile, family mismatch rejection, and stratified oracle keys.

## Validation
- Tests:
  - `pytest -q tests/test_arc_agi_2_adapter.py` -> 10 passed
  - `pytest -q tests/test_benchmarks.py -k arc` -> 1 passed
  - `pytest -q tests/test_arc_agi_2_adapter.py tests/test_benchmarks.py -k "arc"` -> 11 passed

## Full benchmark results (100/100/50)
- ARC empty: `0.32`
- ARC enriched: `0.28`
- ARC delta (same-run): `-0.04`
- Historical enriched ARC delta: `0.00` (`0.28 -> 0.28`)

### ARC enriched diagnostics
- `generated_pattern_total`: `686`
- `tasks_with_generated_patterns`: `100/100`
- `oracle_at_all`: `0.0`
- `oracle_at_3`: `0.0`
- `oracle_at_10`: `0.0`
- `oracle_fuzzy_0_80`: `0.31`
- `oracle_fuzzy_0_85`: `0.20`
- `oracle_fuzzy_0_90`: `0.12`
- `oracle_fuzzy_0_95`: `0.05`
- `fuzzy_best_score_mean`: `0.6250`
- `validity_reject_rate_mean`: `0.7967`
- `family_rejects_mean`: `1.05`
- `generation_failure_rate`: `1.0`
- `ranking_change_rate`: `0.33`

### ARC enriched source precision observed
- `contrastive_anti`: `15/41 = 0.3659`
- `autonomous_generation`: `6/23 = 0.2609`
- `legacy_pipeline`: `7/36 = 0.1944`

## Interpretation
- Oracle is still locked at exact-match (`oracle_at_all=0.0`), but stratified fuzzy now shows a strong near-miss band (`0.31` at `0.80`).
- Current strict gates are likely over-pruning valid-but-near candidates (`~79.7%` rejected).
- Source accuracy now clearly favors `contrastive_anti` over both `autonomous_generation` and `legacy_pipeline`.

## Recommended next pass
1. Relax strict gates from hard reject to soft penalty for family/object mismatches when fuzzy >= `0.80`.
2. Increase ranking weight for `contrastive_anti` and reduce legacy prior further.
3. Add train-pair family confidence to avoid overconfident hard family rejection on mixed tasks.
4. Keep all default galaxies loaded as is (already eager by `Knowledgeverse` default) and continue persistence in single enriched storage root.
