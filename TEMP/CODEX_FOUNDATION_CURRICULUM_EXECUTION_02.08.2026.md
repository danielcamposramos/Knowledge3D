# Codex Foundation Curriculum Execution (Week 20 Pivot)

## Implemented

- Added deterministic curriculum task generators:
  - `benchmarks/tasks/geometric_tasks.py`
  - `benchmarks/tasks/arithmetic_tasks.py`
  - `benchmarks/tasks/pattern_tasks.py`
  - `benchmarks/tasks/compositional_tasks.py`
  - `benchmarks/tasks/rpn_tasks.py`
  - `benchmarks/tasks/__init__.py`
- Added benchmark suite:
  - `benchmarks/deterministic_foundation.py`
- Added foundational operation bootstrap (idempotent, persistent):
  - `knowledge3d/knowledgeverse/foundational_operations_bootstrap.py`
- Added training driver:
  - `scripts/train_deterministic_foundation.py`
- Added validation tests:
  - `tests/test_deterministic_foundation.py`
- Exported bootstrap utilities in:
  - `knowledge3d/knowledgeverse/__init__.py`

## Validation

- Tests:
  - `pytest -q tests/test_deterministic_foundation.py` -> `4 passed`
  - `pytest -q tests/test_arc_agi_2_adapter.py tests/test_benchmarks.py` -> `9 passed`
- Clean bootstrap probe:
  - Inserted foundational ops: `63` (`35` Grammar + `28` Math)
- Smoke training run:
  - Command: `python3 scripts/train_deterministic_foundation.py --iterations 1 --tasks-per-category 10 --storage-root ../Knowledge3D.local/foundation_curriculum_world --output-dir ../Knowledge3D.local/results/foundation_training_smoke_clean`
  - Output summary: `initial=1.000 final=1.000 status=SUCCESS`
  - Artifact: `../Knowledge3D.local/results/foundation_training_smoke_clean/training_history.json`

## Notes

- Persistence and single-world evolution are preserved:
  - Training writes to the provided `storage_root` and keeps events/weights/history there.
  - Bootstrap is idempotent; re-runs do not duplicate the same operation IDs.
- The current deterministic suite is intentionally exact and yields high score immediately after bootstrap.

## Recommended Next Enhancement (to match "child learning to walk")

- Introduce staged curriculum difficulty so TRM must learn progressively instead of immediately saturating:
  - Stage A: direct op tokens
  - Stage B: alias-only prompts
  - Stage C: mixed distractors + longer compositional chains
  - Stage D: sparse context and noisy phrasing
- Keep deterministic ground truth, but require stronger routing/composition behavior before ARC handoff.

