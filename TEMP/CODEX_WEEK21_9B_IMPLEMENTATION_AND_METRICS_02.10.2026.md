# Week 21.9b Implementation + Metrics (Codex)

Date: 2026-02-10

## Scope Implemented

### 1) Object-aware generation (ARC adapter)
- File: `benchmarks/arc_agi_2_adapter.py`
- Added toggles:
  - `enable_object_aware_generation`
- Added generation-time object alignment variants:
  - `_align_candidate_object_count(...)`
  - `_connected_components_nonzero(...)`
- Integrated into candidate variant pipeline before ranking.

### 2) Rescue lane (top-k selection)
- File: `benchmarks/arc_agi_2_adapter.py`
- Added toggles:
  - `enable_rescue_lane`
  - `rescue_lane_size`
- Added selector:
  - `_select_candidate_with_rescue_lane(...)`
- Behavior:
  - exact-first in lane
  - fuzzy fallback when candidate is materially better than top-1 (margin + min quality gate)

### 3) Dual-track oracle reinforcement (quality memory)
- File: `benchmarks/arc_agi_2_adapter.py`
- Added toggle:
  - `enable_dual_track_oracle`
- Extended quality-memory update path with selected-lane context:
  - exact => full reinforcement
  - fuzzy => partial transfer signal
  - miss/rank fallback => neutral/negative handling

### 4) CLI + benchmark wiring
- Files:
  - `benchmarks/arc_agi_2.py`
  - `scripts/run_all_benchmarks.py`
- Added CLI flags:
  - `--arc-enable-object-aware-generation`
  - `--arc-enable-rescue-lane`
  - `--arc-rescue-lane-size`
  - `--arc-enable-dual-track-oracle`
- Added to runtime usage summary and console printouts.

### 5) Additional telemetry
- File: `benchmarks/arc_agi_2.py`
- Added diagnostics aggregation:
  - `rescue_lane_enabled_rate`
  - `selected_exact_rate`
  - `selected_oracle_track_counts`

## Validation Run Outputs

Primary full run (latest):
- Summary: `../Knowledge3D.local/results/week21_9b_full100_v2/week14_benchmark_summary.json`
- ARC detailed: `../Knowledge3D.local/results/week21_9b_full100_v2/arc_agi_2_enriched.json`

Previous baseline used for delta:
- `../Knowledge3D.local/results/week21_9_full100_gpu_migration/week14_benchmark_summary.json`

## Metric Deltas (Week 21.9 -> Week 21.9b v2)

ARC enriched:
- Accuracy: `0.06 -> 0.06` (stable)
- `oracle_at_all`: `0.01 -> 0.01` (stable)
- `fuzzy_oracle_at_all`: `0.06 -> 0.06` (stable)
- `oracle_fuzzy_0_90`: `0.13 -> 0.13` (stable)
- `oracle_fuzzy_0_95`: `0.06 -> 0.06` (stable)
- `generation_failure_rate`: `0.99 -> 0.99` (unchanged)
- `ranking_palette_score_mean`: `0.7391 -> 0.7565` (improved)
- `generation_filter_accept_rate_mean`: `0.4153 -> 0.4282` (improved)
- `generation_filter_generated_total`: `4996 -> 5491` (improved candidate volume)

New rescue/selection telemetry:
- `rescue_lane_enabled_rate`: `1.0`
- `selected_oracle_track_counts`: `{'rank_top1': 84, 'fuzzy': 16}`
- `selected_exact_rate`: `0.0`
- selected rank avg/max: `0.52 / 10`

Failure mode counts (ARC enriched):
- `palette`: 63 (dominant)
- `object_count`: 44
- `shape`: 30
- `generation_gap`: 5
- `near_miss`: 5
- `family`: 0

## Persistence / World State Check

This run **did not start from a clean world**.
- `runtime_seed_knowledge`: `false`
- unified storage root:
  - `empty_mind`: `../Knowledge3D.local/galaxies_enriched`
  - `enriched`: `../Knowledge3D.local/galaxies_enriched`
- shared instance persisted (`shared_instance=true`), same process world

Galaxy counts show pre-existing enriched content + continued accumulation:
- `Character`: 2152
- `Word`: 14021
- `Math`: 5507
- `Drawing`: 3055
- `Reality`: 1584
- `Audio`: 351
- `3DObjects`: 434
- `Grammar`: `34351 -> 36351` during full run (+2000 across empty+enriched blocks)

## Notes on GPU Metrics

PTX path remains active (`ptx_full_used_rate=1.0`, `ptx_ranking_used_rate=1.0`, `ptx_oracle_used_rate=1.0`).
Host-side usage snapshots in `benchmark_usage_metrics.jsonl` still report low average GPU utilization, which is consistent with short kernel bursts + coarse polling intervals.

## Questions / Suggestions for Claude (Week 21.10)

1. Should we run an explicit threshold-calibration matrix (`0.90`, `0.92`, `0.95`) for correctness while keeping dual-track learning unchanged?
2. Since `family=0` failures are solved, should we shift penalty weights to prioritize `palette` and `object_count` further (e.g. palette 2.5, object 1.5)?
3. Should rescue lane be expanded to top-32 only for oracle check (not final prediction) to avoid perturbing top-1 semantics while increasing exact-hit chance?
4. Do we want to gate fuzzy rescue selection with a stronger minimum (`>=0.85`) to reduce low-quality fuzzy selections?
5. Should we start Week 22 ingestion now for Reality/3D/Math expansions to improve generation quality (current blocker), then re-run this same Week 21.9b config unchanged for A/B clarity?

