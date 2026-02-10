# Week 21.9 Progress: GPU Migration + Negative Forms (In Progress)

Date: 2026-02-10

## Implemented in this pass

### 1) Negative-form duality documented (spec-level)
- Updated: `docs/vocabulary/DUAL_CLIENT_CONTRACT_SPECIFICATION.md`
- Added Section 1.7: Positive/Negative Form Duality
  - Ternary mapping: `+1` positive, `0` neutral, `-1` negative
  - Zero-cost derivation: `negative = canvas - positive`
  - Cross-galaxy implications (Drawing/Character/3D/ARC)

### 2) ARC PTX ops: reduced CPU hot-loop overhead
- Updated: `knowledge3d/cranium/ptx/arc_ops.py`
- `discover_patterns_ptx()`:
  - top_k default raised `64 -> 256`
  - threads increased `128 -> 256`
- `apply_validity_gates_relaxed_ptx()`:
  - Added fast-path that reuses precomputed candidate component scores
    (`family_score`, `shape_score`, `palette_score`, `object_score`) instead of recomputing validity per-grid on CPU
  - Fallback to old per-grid CPU computation remains for compatibility
  - threads increased `128 -> 256`

### 3) ARC adapter: larger discovery pool + palette-aware candidate variants
- Updated: `benchmarks/arc_agi_2_adapter.py`
- Discovery expansion:
  - PTX discovery top_k `64 -> 256`
  - Traditional/autonomous train-pair seed limit `3 -> 6`
  - Cross-modal candidates `5 -> 12`
- Generation variants:
  - For each generated candidate, now includes:
    - `base`
    - `palette_aligned`
    - `negative_form` (when enabled)
    - `negative_palette_aligned` (when enabled)
  - Added metadata tag: `form_variant`
- Palette-aware alignment helper:
  - Added `_align_candidate_palette()`
  - Remaps candidate palette toward train output distribution before ranking
- Stronger palette scoring in generation constraints:
  - Increased distribution influence and added invalid-color penalty

### 4) Runner UX alias for architecture language
- Updated: `scripts/run_all_benchmarks.py`
- Added CLI alias:
  - `--arc-enable-negative-forms` (mapped to existing figure-ground reversal path)
- Kept compatibility with `--arc-enable-figure-ground-reversal`
- Benchmark default in runner remains palette-emphasized:
  - `--arc-palette-penalty-weight` default = `2.0`

## Validation executed

### Tests
- `tests/test_arc_agi_2_adapter.py`: **15 passed**

### Smoke run (5 ARC)
- Output: `../Knowledge3D.local/results/week21_9_smoke5_gpu_migration/week14_benchmark_summary.json`
- ARC sample was noisy (0/5), not reliable for trajectory decisions.

### Medium run (20 ARC)
- Output: `../Knowledge3D.local/results/week21_9_medium20_gpu_migration/week14_benchmark_summary.json`
- Enriched ARC:
  - accuracy: `0.05` (1/20)
  - oracle_at_all: `0.0`
  - fuzzy_oracle_at_all: `0.05`
  - oracle_fuzzy_0_90: `0.10`
  - generation_failure_rate: `1.0`
- Quality components:
  - family: `0.7075`
  - shape: `0.9790`
  - palette: `0.7000`  (**up vs week21_8_full100: 0.6356**)
  - object: `1.0`
- Generation/filter telemetry:
  - generated_pattern_total: `200`
  - generation_filter_generated_total: `1022`
  - accept_rate_mean: `0.3614`
  - reject_rate_mean: `0.6386`
- Failure modes (20-task sample):
  - palette: `15`
  - object_count: `11`
  - shape: `5`
  - family: `0`

## Architecture checks (still green)
- Unified persistence:
  - `shared_instance = true`
  - same instance id empty/enriched
- Sovereignty:
  - `arc_embedding_lazy_mode = skip`
- Persistent world (not clean run):
  - `runtime_seed_knowledge = false`
  - storage root: `../Knowledge3D.local/galaxies_enriched`
  - Grammar grows during run (continuity intact)

## Interpretation
- ✅ Palette quality signal improved significantly (0.6356 -> ~0.70 in medium run).
- ✅ PTX-only contract and unified persistence remain intact.
- ⚠️ Oracle exact unlock still blocked (`oracle_at_all=0.0`), and generation failure remains high.
- ⚠️ GPU utilization summary from run-level sampler still appears flat (~0.5%), likely due low-duty-cycle kernels + coarse snapshot sampling.

## Proposed next patch block (Week 21.9b)
1. Move remaining ranking-constraint penalty math to GPU kernel stage (not Python post-process).
2. Add PTX-side batch telemetry counters (`n_candidates`, `n_patterns`, kernel duty-time accumulation) to compute true GPU duty cycle.
3. Add object-count-aware generation templates (current palette fixes alone are insufficient).
4. Add top-k candidate expansion before final validity/oracle (keep best 8-16 instead of very narrow top-1 pressure).
5. Run full 100-task validation once Week 21.9b is in.

## Questions for Claude
1. Should we prioritize object-count constrained generation next (given failures: object_count remains high), before additional palette tuning?
2. For Week 21.9b, do we enforce `top_k >= 8` pre-oracle to improve exact-hit odds?
3. Do we want a temporary fuzzy-first stage gate (e.g., `oracle_fuzzy_0_90 >= 0.20`) before exact oracle gate to keep iteration moving?
4. Should we add a dedicated PTX kernel for component-penalty application so score post-processing is fully GPU-resident?

