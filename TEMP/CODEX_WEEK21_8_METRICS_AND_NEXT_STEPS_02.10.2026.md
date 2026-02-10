# Week 21.8 Metrics + Next Steps (PTX Path Enforced)

Date: 2026-02-10

## Authoritative artifacts
- Summary: `../Knowledge3D.local/results/week21_8_full100_figground_palette/week14_benchmark_summary.json`
- History: `../Knowledge3D.local/benchmarks/run_all_benchmarks_history.jsonl`
- Usage metrics: `../Knowledge3D.local/logs/benchmark_usage_metrics.jsonl`

## Core run status
- `solver`: `arc_ptx_ops` on all 100 ARC tasks
- `ptx_full_used_rate`: `1.0`
- `ptx_ranking_used_rate`: `1.0`
- `ptx_oracle_used_rate`: `1.0`
- `shared_instance`: `true`
- `arc_embedding_lazy_mode`: `skip`

## Benchmark outcomes (enriched)
- ARC: `0.05` (5/100)
- Math: `0.3333333333`
- LHE: `1.0`

## ARC oracle diagnostics (enriched)
- `oracle_at_all`: `0.0`
- `fuzzy_oracle_at_all`: `0.05`
- `oracle_fuzzy_0_90`: `0.12`
- `generation_failure_rate`: `1.0`
- `generated_pattern_total`: `700`
- `tasks_with_generated_patterns`: `100`
- `rejected_was_better_count`: `10`
- `rejected_was_better_rate`: `0.10`
- `rejected_fuzzy_delta_mean`: `0.0940483458`

## Constraint score means (top-ranked candidates)
- `family`: `0.7855`
- `shape`: `0.9656359494`
- `palette`: `0.6356127989`  <-- weakest signal
- `object`: `0.92`

## Filter telemetry
- `generation_filter_generated_total`: `1000`
- `generation_filter_accept_rate_mean`: `0.549`
- `generation_filter_reject_rate_mean`: `0.451`

## Failure mode counts (oracle)
- `palette`: `58`
- `object_count`: `54`
- `shape`: `32`
- `family`: `0`
- `generation_gap`: `10`
- `near_miss`: `3`

## World continuity / persistence status
- This was **not** a clean-world run.
- `runtime_seed_knowledge`: `false`
- Unified storage root: `../Knowledge3D.local/galaxies_enriched`
- Enriched start counts:
  - Drawing `3055`, Character `2152`, Word `14021`, Grammar `28330`, Math `5507`, Reality `1584`, Audio `351`, 3DObjects `434`
- Enriched end counts:
  - Drawing `3055`, Character `2152`, Word `14021`, Grammar `29030`, Math `5507`, Reality `1584`, Audio `351`, 3DObjects `434`
- Net growth in run: Grammar `+700`

## Practical interpretation
- PTX routing/contract is now correct.
- Main bottleneck remains **candidate quality**, especially palette/object consistency.
- Family is no longer dominant blocker (`family` failures = 0).

## Proposed next patch set (Week 21.9 candidate-quality pass)
1. Palette-first generation constraints
   - Promote palette-distribution matching to generator stage (not only rank penalty).
   - Reject or strongly penalize candidates that violate train-output palette histogram bands.
2. Object-count structural priors
   - Add connected-component delta constraints into generation templates.
3. GPU-side feature extraction completion
   - Move remaining Python-side feature extraction/filtering loops to kernels (`extract_pattern_features`, `check_grid_validity`, `compare_grids`, `filter_by_threshold`) to remove hybrid bottleneck.
4. Coverage gate wiring
   - Current coverage union is only `Grammar`; enforce explicit multi-galaxy retrieval participation during ARC solving to satisfy docs/vocabulary multi-curriculum intent.

## Questions for Claude
1. Should we prioritize **palette histogram constraints** over current scalar palette overlap in generation (hard clamp + soft penalty)?
2. For object-count failures, do we add a hard cap (strict mode only) or weighted penalty in all modes?
3. Should Stage gate continue requiring exact oracle (`oracle_at_all`) at this phase, or permit fuzzy progression threshold first (e.g., `oracle_fuzzy_0_90 >= 0.20`)?
4. Do we lock ARC to PTX-only and postpone Math/LHE PTX expansion until ARC reaches `oracle_at_all >= 0.10`?
5. Do we want an explicit “rejected-was-better rescue” lane in final top-k selection for near-miss recovery?

