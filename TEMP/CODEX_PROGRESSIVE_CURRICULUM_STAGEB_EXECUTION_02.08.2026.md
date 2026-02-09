# Codex Progressive Curriculum Execution (Stage A->B)

## Scope completed

- Added stage-aware deterministic curriculum with development stages:
  - Stage A: direct operation prompts
  - Stage B: alias-only prompts (no direct `operation` field)
  - Stage C: alias + distractors
  - Stage D: sparse few-shot/noisy phrasing
- Added automatic gate advancement in trainer:
  - Gate window: 3 consecutive iterations
  - Thresholds: A=0.95, B=0.85, C=0.75, D=0.65
  - Advancement: A->B->C->D

## Files changed

- `benchmarks/deterministic_foundation.py`
  - Stage adaptation pipeline (`_adapt_tasks_for_stage`)
  - Alias inference (`_infer_operation_from_query`)
  - Stage-aware operation resolver (`_resolve_task_operation`)
  - Stage tag included in benchmark output and Shadow Copy events
- `scripts/train_deterministic_foundation.py`
  - Gate-based automatic stage advancement
  - Stage metadata in per-iteration artifacts
  - Final payload includes `final_stage`, `stage_thresholds`, `gate_window`
- `tests/test_deterministic_foundation.py`
  - Added Stage B alias-task validation
  - Added stage advancement test

## Validation

- Tests:
  - `pytest -q tests/test_deterministic_foundation.py` -> `6 passed`
  - `pytest -q tests/test_arc_agi_2_adapter.py tests/test_benchmarks.py` -> `9 passed`
- Stage probe run:
  - Command:
    - `python3 scripts/train_deterministic_foundation.py --iterations 4 --tasks-per-category 10 --storage-root ../Knowledge3D.local/foundation_curriculum_world_stageprobe --output-dir ../Knowledge3D.local/results/foundation_training_stageprobe`
  - Result:
    - Iteration 1: Stage A, overall 1.00
    - Iteration 2: Stage A, overall 1.00
    - Iteration 3: Stage A, overall 1.00 -> **advance to Stage B**
    - Iteration 4: Stage B, overall 0.98
  - `final_stage`: `B`
  - Artifact: `../Knowledge3D.local/results/foundation_training_stageprobe/training_history.json`

## Interpretation

- Curriculum progression now behaves as intended:
  - Stage A saturates quickly.
  - System auto-promotes to harder Stage B.
  - Stage B introduces measurable difficulty drop (good signal for learning runway).

## Recommended next step

- Run full Week-21 Stage B block:
  - `python3 scripts/train_deterministic_foundation.py --iterations 10 --tasks-per-category 100 --storage-root ../Knowledge3D.local/foundation_curriculum_world --output-dir ../Knowledge3D.local/results/foundation_training_week21`
- Then evaluate ARC with same persistent world to measure transfer uplift.

