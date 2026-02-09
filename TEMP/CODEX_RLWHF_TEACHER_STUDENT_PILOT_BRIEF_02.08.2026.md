# Codex RLWHF Teacher/Student + Ternary Pilot Brief

Date: 2026-02-08
Run: Week 21.1 pilot (3 iterations)

## What was implemented
- RLWHF teacher/student bridge with non-binary ternary + multi-level pools:
  - `knowledge3d/training/rlwhf/teacher_student_bridge.py`
  - 4-axis ternary pooling -> 81 pools (`3^4`) via `pool_id`.
- Persistent ternary quality memory for runtime ranking priors:
  - `knowledge3d/knowledgeverse/ternary_quality_memory.py`
  - 3-axis ternary pooling -> 27 pools (`3^3`) + EMA prior in `[-1,+1]`.
- Integrated into ARC ranking path:
  - `benchmarks/arc_agi_2_adapter.py`
  - ranking score now includes `quality_prior`, with online updates after each task.
- Integrated into curriculum training loop:
  - `scripts/train_deterministic_foundation.py`
  - transfer-aware gates + RLWHF feedback persistence + ternary quality updates.
- Added system-structure literacy support and optional Ollama augmentation:
  - `benchmarks/deterministic_foundation.py`
  - `benchmarks/tasks/system_literacy_tasks.py`
  - `knowledge3d/augmentation/ollama_curriculum_augmenter.py`

## Validation status
- Focused tests passed:
  - `pytest -q tests/test_teacher_student_bridge.py tests/test_ternary_quality_memory.py tests/test_deterministic_foundation.py tests/test_arc_agi_2_adapter.py`
  - Result: `20 passed`.

## Pilot command
```bash
env PYTHONPATH=. python3 scripts/train_deterministic_foundation.py \
  --iterations 3 \
  --tasks-per-category 50 \
  --enable-transfer-gates \
  --enable-ternary-quality \
  --enable-ollama-augmentation \
  --ollama-vision-model llava \
  --ollama-language-model llama3.2 \
  --ollama-multimodal-model llava \
  --include-system-literacy \
  --storage-root ../Knowledge3D.local/foundation_curriculum_world_21_1 \
  --output-dir ../Knowledge3D.local/results/foundation_training_week21_1
```

## Pilot metrics
Source: `../Knowledge3D.local/results/foundation_training_week21_1/training_history.json`

- Iteration 1: stage A, train 1.00, transfer 0.20, oracle_at_all 0.00, generated 0
- Iteration 2: stage A, train 1.00, transfer 0.20, oracle_at_all 0.00, generated 0
- Iteration 3: stage A, train 1.00, transfer 0.20, oracle_at_all 0.00, generated 0

Summary:
- final_stage: A (no promotion)
- transfer_gates_enabled: true
- ternary_quality_enabled: true
- ollama_augmentation_enabled: true
- train delta: +0.00 (1.00 -> 1.00)

RLWHF teacher/student signal:
- pool_id each iteration: `ternary_pool_2000_54`
- teacher_label: `uncertain`
- teacher_rating: `0`

Ternary quality memory snapshot:
- file: `../Knowledge3D.local/foundation_curriculum_world_21_1/checkpoints/curriculum_quality_memory.json`
- stage entries persisted with pool `pool_222_26` and prior `0.1`.

## Interpretation
- Gate leakage is now blocked correctly: no stage promotion while transfer/generation health is below thresholds.
- Teacher/student + ternary memory wiring is functioning and persistent.
- The core bottleneck remains generation transfer (oracle/generation still zero in this pilot).

## Codex recommendations (next pass)
1. Add Stage-1 ARC-shaped mini-generation tasks into the curriculum loop itself (not only transfer probe), so `generated_pattern_total` can rise before Stage B.
2. Add per-iteration pool drift metric (pool transitions across iterations) to detect non-binary learning movement even when top-line accuracy is flat.
3. Add domain-specific transfer probes (`visual`, `math`, `system`) and gate against the relevant one per stage to avoid over-penalizing early structure-literacy stages.
4. Keep ternary priors but raise quality-memory weight adaptively only when oracle_at_all > 0 to avoid reinforcing neutral/wrong early candidates.
