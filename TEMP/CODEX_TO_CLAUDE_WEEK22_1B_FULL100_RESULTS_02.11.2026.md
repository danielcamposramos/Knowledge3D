# Codex → Claude: Week 22.1b Forced-Navigation Full100 Results

**Date:** 2026-02-11
**Run Type:** Unified persistent world, forced-navigation curriculum (`week22_1b`)
**Summary Path:** `../Knowledge3D.local/results/week22_1b_forced_nav_full100/week14_benchmark_summary.json`
**Usage Log Path:** `../Knowledge3D.local/logs/benchmark_usage_metrics.jsonl`

## 1) Command Executed

```bash
/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python scripts/run_all_benchmarks.py \
  --model-persistence-mode unified \
  --max-arc-tasks 100 \
  --max-math-problems 100 \
  --max-lhe-questions 50 \
  --max-mmlu-questions 0 \
  --arc-enable-full-ptx \
  --arc-enable-contrastive-learning \
  --arc-enable-validity-gates \
  --arc-constraint-mode penalty \
  --arc-enable-figure-ground-reversal \
  --arc-enable-object-aware-generation \
  --arc-enable-rescue-lane --arc-rescue-lane-size 16 \
  --arc-oracle-search-lane-size 64 \
  --arc-enable-dual-track-oracle \
  --arc-enable-fuzzy-oracle --arc-fuzzy-oracle-threshold 0.95 \
  --arc-embedding-lazy-mode skip \
  --arc-curriculum-stage week22_1b \
  --track-curriculum-coverage \
  --output-dir ../Knowledge3D.local/results/week22_1b_forced_nav_full100 \
  --storage-root ../Knowledge3D.local
```

## 2) Core Benchmark Metrics (Enriched)

- ARC accuracy: **0.06** (6/100)
- Math accuracy: **0.00** (0/100)
- LHE accuracy: **0.14** (7/50)
- MMLU: **skipped** (`max-mmlu-questions=0` semantics working)

## 3) ARC Oracle/Generation Diagnostics

From `benchmarks.arc_agi_2.enriched.oracle_diagnostics`:

- `top_1_accuracy`: **0.06**
- `oracle_at_all`: **0.01**
- `oracle_at_10`: **0.01**
- `fuzzy_oracle_at_all`: **0.06**
- `oracle_fuzzy_0_90`: **0.13**
- `oracle_fuzzy_0_95`: **0.06**
- `generation_failure_rate`: **0.99**
- `generation_filter_generated_total`: **6869**
- `generation_filter_accept_rate_mean`: **0.4253**
- `generation_filter_reject_rate_mean`: **0.5747**
- `rejected_was_better_count`: **13**
- `rejected_fuzzy_delta_mean`: **0.1634**

Ranking component means:
- `ranking_palette_score_mean`: **0.7543**
- `ranking_shape_score_mean`: **0.9640**
- `ranking_family_score_mean`: **0.7920**
- `ranking_object_score_mean`: **1.0000**

## 4) Forced-Navigation Curriculum Telemetry

- Curriculum enabled: **yes** (`week22_1b`)
- Forced pattern source usage (per-task results):
  - `pattern_source=curriculum_forced_navigation`: **91 / 100 tasks**
- Aggregated forced patterns in ARC result entries:
  - `forced_navigation_pattern_count` total: **400**

Query-based coverage (enriched ARC block):
- `unique_galaxies`: **[3DObjects, Drawing, Grammar, Math, Reality]**
- `avg_queried_galaxies_per_task`: **5.0**
- `cross_galaxy_navigation_rate`: **1.0**

Coverage soft gate:
- `min_required=5`, `min_cross_rate=0.5`
- `enriched_query_passed=true`
- `enriched_cross_rate_passed=true`

## 5) Sovereignty / Persistence Validation

- PTX path:
  - `ptx_full_used_rate=1.0`
  - `ptx_ranking_used_rate=1.0`
  - `ptx_oracle_used_rate=1.0`
- Embedding mode: `arc_embedding_lazy_mode=skip`
- Persistence mode: `shared_instance=true`
- Storage roots:
  - Empty: `../Knowledge3D.local/galaxies_enriched`
  - Enriched: `../Knowledge3D.local/galaxies_enriched`

### Important continuity note
This run **did not start from a clean world**. It ran against the existing persistent enriched world.

- `runtime_seed_knowledge=false`
- Grammar growth during this run (same unified root, both blocks):
  - Start empty block: `Grammar=43901`
  - End empty block: `Grammar=44901`
  - End enriched block: `Grammar=45901`
  - Net run growth: **+2000 Grammar entries**

## 6) Math Diagnostic Status

Math routing is correct but solver capability still weak:
- `route_specialist_counts`: `{ "math": 100 }`
- `predicted_none_rate`: **0.76**
- `predicted_numeric_count`: **24**
- `expected_numeric_count`: **100**

## 7) Architectural Readout

- Navigation bottleneck (coverage) is now **partially broken**:
  - ARC is touching 5 galaxies consistently.
- Main blocker remains **candidate quality / generation failure**:
  - `generation_failure_rate=0.99` still dominates.
- PTX-only sovereignty and unified persistence are stable.

## 8) Suggested Next Step Questions for Claude

1. Should we tighten curriculum success around `oracle_at_10` / `oracle_at_all` progression now that coverage targets are met?
2. Given `rejected_was_better_count=13`, should we add a bounded rejected-candidate rescue pass for oracle lane only (not top-1 prediction)?
3. Should we switch from static penalty weights to adaptive weights per run (from current oracle failure mode distribution)?
4. For Math, do we prioritize `_solve_math` extractor enhancement now, or keep focus on ARC oracle unlock first?
5. Do we want an A/B rerun with `--arc-enable-adaptive-penalties` immediately after this baseline for controlled comparison?

