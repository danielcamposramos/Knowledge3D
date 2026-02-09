# Codex Benchmark Architecture Fix Results

**Date:** 2026-02-08  
**Scope:** Remove benchmark-side orchestration, keep single-world continuity, add runtime metrics logging.

## 1. What Was Fixed

### Benchmark-layer orchestration removed (already applied and validated)
- `benchmarks/arc_agi_2.py`
  - No benchmark-level default-galaxy forcing.
  - Runtime seeding now opt-in via `runtime_seed_knowledge=False` (default).
- `benchmarks/arc_agi_2_adapter.py`
  - Removed benchmark-side `ensure_default_galaxies_loaded()` call.
- `benchmarks/math_competitions.py`
  - Runtime seeding guarded by `runtime_seed_knowledge` (default off).
- `benchmarks/last_humanity_exam.py`
  - Runtime seeding guarded by `runtime_seed_knowledge` (default off).

### Runner-layer metrics + continuity
- `scripts/run_all_benchmarks.py` (already done in prior pass)
  - Per-run CPU/GPU runtime metrics.
  - Galaxy counts at start/end.
  - Usage log: `../Knowledge3D.local/logs/benchmark_usage_metrics.jsonl`.
- `scripts/run_all_global_benchmarks.py` (completed now)
  - Added same runtime telemetry pattern.
  - Added `--benchmark-runtime-seeding` (default off).
  - Reuses the **same enriched Knowledgeverse** for proxy runs (`gsm8k`, `mmlu`) to preserve single-world continuity.
  - Persists usage telemetry to `../Knowledge3D.local/logs/global_benchmark_usage_metrics.jsonl`.

### ARC lazy-embedding continuity fix (new)
- `Old_Attempts/curriculum_specific_training/arc_agi/sovereign_pipeline.py`
  - Added persistent/shared ARC embedding cache backing `embedding_galaxy`.
  - Cache path per world and dim:
    - `.../checkpoints/arc_embedding_galaxy_d128.json`
    - `.../checkpoints/arc_embedding_galaxy_d512.json`
  - Reuses cache within the same Knowledgeverse instance and across reruns on same storage root.

## 2. Validation Results

### Test suite
- `tests/test_run_all_global_benchmarks_history.py` ✅
- `tests/test_global_benchmark_scripts.py` ✅
- `tests/test_run_all_benchmarks_history.py` ✅
- `tests/test_arc_agi_2_adapter.py` ✅
- `tests/test_navigator_specialist.py` ✅
- `tests/test_teacher_student_bridge.py` ✅

### Runtime continuity check (global runner, same storage root)
- Run 1: `GALAXY LAZY` occurrences = **4**
- Run 3 (same root, after cache persisted): `GALAXY LAZY` occurrences = **0**

Interpretation:
- The prior visible recomputation was mostly from ARC candidate embedding cache warm-up, not default-galaxy reload.
- With persisted cache, subsequent runs avoid those lazy embedding batches.

### Usage telemetry artifact checks
- `global_benchmark_summary.json` now includes `runtime_usage` with:
  - `runs` (elapsed, rss before/after, gpu before/after)
  - `proxy_runs`
  - `galaxy_counts` (empty/enriched start/end)
- Usage log JSONL entries appended under:
  - `../Knowledge3D.local/logs/global_benchmark_usage_metrics.jsonl`

## 3. Architectural Notes

- This aligns with the project directive that benchmarks should be adapters (K3D in/out), not orchestrators.
- Default galaxies are loaded centrally by `Knowledgeverse` (`eager_load_default_galaxies=True`) and now benchmark scripts no longer force reload behavior.
- ARC legacy path still prints many diagnostic lines by design; with cache persistence, the expensive missing-embedding phase is materially reduced after first warm-up.

## 4. Next Recommended Patch (high leverage)

To make RLWHF + ternary diagnostics visible at system level:
1. Add per-task `embedding_cache_size_before/after` counters into ARC task result telemetry.
2. Emit `pool_drift_entropy` and `topk_ternary_quality_distribution` in benchmark summaries.
3. Add `--strict-no-lazy-embeddings` mode (fail-fast after warm-up) for controlled experiments.

