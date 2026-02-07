# Week 17.1 ARC Worker Context Fix Report (2026-02-07)

## Objective
Remove ARC worker/context conflicts and keep ARC benchmark on sovereign legacy pipeline (no TRM fallback), while preserving Knowledgeverse integration.

## Root Causes Confirmed
1. `ArcAgi2Adapter` instantiated `SovereignAIPipeline` without Knowledgeverse context.
2. Legacy `ParallelCandidateGenerator` defaulted to process workers, re-initializing ARC runtime per worker.
3. In Knowledgeverse-integrated mode, grammar wrapper lacked compatibility method `get_high_confidence_rules` used by ARC sequential refinement.
4. Legacy sequential refiner had an unbound local (`refined`) in tier-2/tier-3 paths.

## Implemented Fixes

### 1) Adapter injects Knowledgeverse into legacy pipeline
- File: `benchmarks/arc_agi_2_adapter.py`
- Added `knowledgeverse` parameter and passed it into `SovereignAIPipeline(...)`.

### 2) ARC benchmark passes shared Knowledgeverse instance
- File: `benchmarks/arc_agi_2.py`
- `ArcAgi2Adapter(..., knowledgeverse=self.kv)` now used.

### 3) Legacy pipeline supports Knowledgeverse-aware initialization
- File: `Old_Attempts/curriculum_specific_training/arc_agi/sovereign_pipeline.py`
- Added `knowledgeverse` parameter.
- When present, pipeline uses:
  - `knowledgeverse.galaxy_manager.get_galaxy("Drawing")`
  - `knowledgeverse.galaxy_manager.get_galaxy("Grammar")`
- Default `embedding_galaxy` now initialized to `{}` if unset.
- Parallel candidate generator call now receives full dependencies and explicit process-worker policy.

### 4) Parallel generator supports shared-context local mode
- File: `Old_Attempts/curriculum_specific_training/arc_agi/parallel_candidate_generator.py`
- Added constructor support for:
  - `knowledgeverse`, `use_process_workers`
  - `shadow_copy`, `drawing_galaxy`, `codec_embedder`, `embedding_galaxy`, `cosine_bridge`
- When Knowledgeverse is provided, defaults to local shared-context worker execution (no process forking).
- Kept process path available for non-Knowledgeverse mode.

### 5) Grammar compatibility for ARC refinement
- File: `knowledge3d/knowledgeverse/grammar_galaxy.py`
- Added `get_high_confidence_rules(min_score=...)` compatibility method.

### 6) Sequential refiner robustness fix
- File: `Old_Attempts/curriculum_specific_training/arc_agi/sequential_refiner.py`
- Fixed tier-2/tier-3 `refined` unbound local usage.
- Improvement checks are now safely scoped per loop iteration.

## Validation

### Unit/Integration Tests
- Command:
  - `pytest -q tests/test_arc_agi_2_adapter.py tests/test_benchmarks.py tests/test_week15_galaxy_integration.py tests/test_knowledgeverse_integration.py tests/test_navigator_specialist.py tests/test_specialist_router.py`
- Result:
  - `24 passed`

### Benchmark Smoke (Week 14 script)
- Command:
  - `env PYTHONPATH=. python3 scripts/run_all_benchmarks.py --max-arc-tasks 2 --max-math-problems 4 --max-lhe-questions 4 --output-dir /tmp/k3d_week14_smoke_meta_fix2`
- Result snapshot:
  - ARC-AGI 2: `0.00% -> 0.00%`
  - Math: `0.00% -> 50.00%`
  - LHE: `50.00% -> 100.00%`

### Key ARC Correctness Signal (post-fix)
- Adapter output now uses `solver=legacy_sovereign_pipeline` (not fallback).
- No more worker loader/context errors observed in run logs.
- No fallback reason present on ARC tasks.
- Fuzzy scores observed around `0.70-0.80` on sampled ARC tasks, indicating candidate quality without exact match yet.

## Status
- Week 17.1 context/worker conflict: **resolved**.
- ARC now runs through integrated legacy sovereign path within Knowledgeverse context.
- Next performance step: improve exact-match selection/ranking for ARC (candidate scoring/composition), not context plumbing.
