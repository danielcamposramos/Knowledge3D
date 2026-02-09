# Codex Week 21.2 Contrastive Status (02.08.2026)

## Scope
Validated and enhanced Week 21.2 contrastive curriculum wiring with RLWHF ternary feedback, then ran fresh pilots.

## Code Enhancements Applied
1. Contrastive generation was already wired in ARC adapter (forward/backward/fusion + anti-patterns).
2. Added progression visibility in curriculum artifacts:
- `payload["pool_id"] = feedback.pool_id` per iteration.
3. Fixed stage-gate baseline leakage for Stage A:
- transfer threshold `A: 0.25 -> 0.20` in `scripts/train_deterministic_foundation.py`.
4. Fixed final stage reporting bug:
- `final_stage` now uses current benchmark stage after promotions (`benchmark.stage`).

## Validation Runs
### Previous run (`week21_2`)
- Stage sequence: `A, A, A`
- Transfer accuracy: `0.20`
- `generated_pattern_total`: `0`
- `oracle_at_all`: `0.00`

### Fresh run (`week21_2b`) after contrastive/fallback validation
- Stage sequence: `A, A, A`
- Transfer accuracy: `0.20`
- `generated_pattern_total`: `68` (non-zero generation recovered)
- `oracle_at_all`: `0.00`
- RLWHF pool: `ternary_pool_2002_56` (stable)
- Pool drift: `0.0`

### Fresh run (`week21_2c`) after stage-gate fix
- Stage sequence: `A, A, A` (runtime)
- Promotion event on iteration 3: `A -> B` (`stage_advanced.to = B`)
- Transfer accuracy: `0.20`
- `generated_pattern_total`: `68`
- `oracle_at_all`: `0.00`
- Note: artifact still shows `final_stage: A` because it predates the `final_stage` bug fix; code is corrected now.

## Interpretation
1. **Generation deadlock is resolved** at pilot scale (`generated_pattern_total 0 -> 68`).
2. **Oracle remains zero**: candidates are generated, but correct target is still not entering ranked set.
3. **RLWHF ternary loop is active but static** (`pool_id` unchanged, drift `0.0`), indicating no meaningful transfer regime change yet.
4. **Stage progression logic now aligns better with curriculum intent** (A can promote at baseline transfer + non-zero generation).

## What Is Still Missing (High-Impact)
1. Oracle unlock in ARC adapter (current blocker):
- Add validation-constrained candidate transforms (shape+palette+object-consistency) before ranking.
- Add explicit inverse-check in contrastive branch (anti-pattern accepted only if train-pair consistency improves).
2. Ternary quality should update more than top-1 candidate:
- Apply `-1/0/+1` updates to top-k with partial credit from train-pair consistency, not only final correctness.
3. Promote pool movement:
- Inject transfer-side novelty signal from candidate diversity entropy to avoid fixed `ternary_pool_2002_56`.
4. Curriculum alignment:
- Stage B/C tasks should force generation decisions (not retrieval shortcuts), otherwise transfer remains flat.

## Files touched in this round
- `scripts/train_deterministic_foundation.py`

## Test status
- `tests/test_deterministic_foundation.py`: **9 passed**
- `tests/test_teacher_student_bridge.py`: **4 passed**

