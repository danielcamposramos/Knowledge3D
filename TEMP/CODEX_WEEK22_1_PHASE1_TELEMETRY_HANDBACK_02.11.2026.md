# Codex Week 22.1 Phase 1 Telemetry Handoff

**Date:** February 11, 2026  
**Status:** Implemented + smoke validated in `k3d-cranium`

## What Was Implemented

### 1) Query-based coverage telemetry (ARC task-level)
- Added per-task query participation fields derived from discovered pattern provenance:
  - `queried_galaxies`
  - `queried_galaxy_count`
  - `source_galaxy_counts`
  - `target_galaxy_counts`
  - `cross_galaxy_composition_count`
- Added ARC diagnostics aggregation:
  - `query_based_coverage.avg_queried_galaxies_per_task`
  - `query_based_coverage.unique_queried_galaxies`
  - `query_based_coverage.queried_galaxy_task_counts`
  - `query_based_coverage.source_galaxy_query_counts`
  - `query_based_coverage.target_galaxy_query_counts`
  - `query_based_coverage.cross_galaxy_navigation_rate`

### 2) Oracle search lane separated from prediction lane
- New ARC argument: `oracle_search_lane_size` (default `32`), independent from `rescue_lane_size`.
- Selection function now reports:
  - `oracle_search_lane_size`
  - `oracle_lane_size`
  - `oracle_probe_exact`
  - `oracle_probe_exact_rank`
  - `oracle_probe_fuzzy_score`
  - `oracle_probe_fuzzy_rank`
- Prediction behavior remains rescue-lane based; oracle probing is widened for learning diagnostics.

### 3) Generation diagnostics hardening
- Added `generated_pattern_total` alias in task outputs (in addition to `generated_pattern_count`).
- Added generation object-count distributions:
  - `generation_object_count_distribution`
  - `generation_object_count_distribution_accepted`
  - `generation_object_count_distribution_rejected`
- Aggregated these distributions in ARC benchmark diagnostics.

### 4) Coverage gate moved beyond count deltas
- `scripts/run_all_benchmarks.py` now emits both:
  - `storage_growth_touched` (existing count-delta style)
  - `query_participation` (new actual-use style)
- Soft gate now reports pass/fail for both growth-touch and query-touch counts.

### 5) Generation-failure trend telemetry
- Added history-aware trend block in summary runtime usage:
  - `arc_generation_failure_trend.points`
  - `arc_generation_failure_trend.delta_latest`
  - `arc_generation_failure_trend.flat_last_3_runs`

## Files Updated
- `benchmarks/arc_agi_2_adapter.py`
- `benchmarks/arc_agi_2.py`
- `scripts/run_all_benchmarks.py`

## Validation Run

Command (smoke, unified, PTX full):

```bash
/home/daniel/miniforge/bin/conda run -n k3d-cranium env PYTHONPATH=. python scripts/run_all_benchmarks.py \
  --model-persistence-mode unified \
  --max-arc-tasks 5 --max-math-problems 5 --max-lhe-questions 5 \
  --arc-enable-full-ptx --arc-enable-contrastive-learning --arc-enable-validity-gates \
  --arc-constraint-mode penalty --arc-enable-figure-ground-reversal \
  --arc-enable-object-aware-generation --arc-enable-rescue-lane --arc-rescue-lane-size 16 \
  --arc-oracle-search-lane-size 32 --arc-enable-dual-track-oracle \
  --arc-enable-fuzzy-oracle --arc-fuzzy-oracle-threshold 0.95 \
  --arc-embedding-lazy-mode skip --track-curriculum-coverage \
  --require-min-galaxies-per-block 5 \
  --output-dir ../Knowledge3D.local/results/week22_1_phase1_smoke \
  --storage-root ../Knowledge3D.local
```

Summary path:
- `../Knowledge3D.local/results/week22_1_phase1_smoke/week14_benchmark_summary.json`

## Smoke Results (telemetry-focused)

ARC enriched diagnostics now include query-participation signal:
- `query_based_coverage.avg_queried_galaxies_per_task = 3.0`
- `query_based_coverage.unique_queried_galaxies = ["3DObjects", "Drawing", "Grammar"]`
- `query_based_coverage.cross_galaxy_navigation_rate = 1.0`

Runtime coverage block now includes query-participation section:
- `runtime_usage.curriculum_coverage.query_participation.enriched_block.unique_galaxies = ["3DObjects", "Drawing", "Grammar"]`

Oracle lane telemetry appears in per-task rows:
- `oracle_search_lane_size = 32`
- `oracle_lane_size` present
- `oracle_probe_exact` / `oracle_probe_fuzzy_score` present

Generation object distribution appears:
- Example task: `generation_object_count_distribution = {"1": 64}`

## Current Interpretation

- Phase 1 telemetry foundation is working.
- We now can measure *accessibility* (actual query participation) directly instead of inferring from entry-count growth.
- In this smoke, ARC queried only 3 galaxies (`Drawing`, `Grammar`, `3DObjects`) despite expanded universe, supporting the Week 22.1 diagnosis that navigation breadth is still constrained.

## Suggested Next Steps (for Claude review)

1. **Promote query-based coverage to primary gate** for ARC stage checks (`>=5` unique queried galaxies over enriched block), keep count-delta gate as secondary.
2. **Micro-curriculum routing target:** explicitly force or reward retrieval from `Math` and `Reality` in palette/object/shape subsets.
3. **Add per-failure-mode query participation split**:
   - For palette failures: which galaxies were queried?
   - For object failures: which galaxies were queried?
   - For shape failures: which galaxies were queried?
4. **Run 100-task ARC-only diagnostic pass** with the new telemetry before broader benchmark loop to avoid confounding from Math/LHE.

## Questions for Claude

1. Should Week 22.1 gate require **both** `query>=5 galaxies` and `cross_galaxy_navigation_rate >= 0.6` before stage promotion?  
2. For palette/object/shape micro-curriculum, should we enforce hard routing priors (must touch target galaxies) or soft reward in ranking only?  
3. Should `oracle_search_lane_size` be staged by curriculum difficulty (e.g., 32 in foundation, 16 later)?
