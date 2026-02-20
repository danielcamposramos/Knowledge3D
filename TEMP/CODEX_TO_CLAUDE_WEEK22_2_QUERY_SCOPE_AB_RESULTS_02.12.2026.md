# Week 22.2 Query Scope A/B Results (PTX Validation)

**Date:** 2026-02-12  
**Executor:** Codex  
**Context:** A/B validation requested by Claude (`baseline unscoped` vs `scoped query`)

## Runs Executed

### A) Baseline (true unscoped)
```bash
/home/daniel/miniforge/bin/conda run -p /K3D/Knowledge3D.local/envs/k3d-cranium env PYTHONPATH=. python scripts/run_all_benchmarks.py \
  --model-persistence-mode unified \
  --max-arc-tasks 20 \
  --max-math-problems 20 \
  --max-lhe-questions 20 \
  --max-mmlu-questions 0 \
  --arc-enable-full-ptx \
  --arc-query-scope-galaxies "" \
  --math-query-scope-galaxies "" \
  --lhe-query-scope-galaxies "" \
  --mmlu-query-scope-galaxies "" \
  --output-dir ../Knowledge3D.local/results/week22_2_baseline_unscoped_true \
  --storage-root ../Knowledge3D.local
```

Summary: `../Knowledge3D.local/results/week22_2_baseline_unscoped_true/week14_benchmark_summary.json`

### B) Scoped
```bash
/home/daniel/miniforge/bin/conda run -p /K3D/Knowledge3D.local/envs/k3d-cranium env PYTHONPATH=. python scripts/run_all_benchmarks.py \
  --model-persistence-mode unified \
  --max-arc-tasks 20 \
  --max-math-problems 20 \
  --max-lhe-questions 20 \
  --max-mmlu-questions 0 \
  --arc-enable-full-ptx \
  --arc-query-scope-galaxies "Drawing,Geometry,Pattern,3DObjects" \
  --math-query-scope-galaxies "Math,Algebra,Geometry" \
  --lhe-query-scope-galaxies "Drawing,Math,Grammar,Word,Reality" \
  --output-dir ../Knowledge3D.local/results/week22_2_scoped_query \
  --storage-root ../Knowledge3D.local
```

Summary: `../Knowledge3D.local/results/week22_2_scoped_query/week14_benchmark_summary.json`

## A/B Comparison

| Metric | Baseline (unscoped) | Scoped | Delta |
|---|---:|---:|---:|
| ARC accuracy (enriched) | 0.00 | 0.00 | +0.00 |
| ARC oracle@all (enriched) | 0.00 | 0.00 | +0.00 |
| ARC fuzzy oracle@all (enriched) | 0.05 | 0.05 | +0.00 |
| ARC generation failure rate (enriched) | 1.00 | 1.00 | +0.00 |
| ARC queried galaxies (unique) | 3DObjects, Drawing, Grammar | 3DObjects, Drawing, Grammar | same |
| ARC avg queried galaxies/task | 3.0 | 3.0 | +0.0 |
| Math accuracy (enriched) | 0.00 | 0.00 | +0.00 |
| LHE accuracy (enriched) | 0.15 | 0.15 | +0.00 |
| MMLU accuracy (enriched) | 0.00 (skipped) | 0.00 (skipped) | +0.00 |
| Total run elapsed sum (s, all stages) | 687.02 | 666.53 | -20.49 |
| Speedup factor (baseline/scoped) | 1.00x | 1.03x | +3.1% |

### Per-stage elapsed (s)

| Stage | Baseline | Scoped | Delta |
|---|---:|---:|---:|
| arc_empty_mind | 37.18 | 37.73 | +0.55 |
| arc_enriched | 39.12 | 41.77 | +2.65 |
| math_empty_mind | 207.96 | 194.22 | -13.74 |
| math_enriched | 333.34 | 307.88 | -25.46 |
| lhe_empty_mind | 35.39 | 48.85 | +13.46 |
| lhe_enriched | 34.03 | 36.07 | +2.04 |

## Operational Notes

- Both runs maintained sovereignty settings:
  - `shared_instance=true`
  - `arc_enable_full_ptx=true`
  - `arc_embedding_lazy_mode=skip`
- `max-mmlu-questions=0` skip semantics worked correctly (MMLU skipped intentionally).

## Architectural Conclusion

1. Query scope implementation is correctly wired and stable.
2. In this A/B window, query scoping did **not** improve oracle or accuracy metrics.
3. Runtime improved only marginally (~3.1% overall), with mixed per-stage effects.
4. ARC still effectively queries only `Drawing/Grammar/3DObjects`, so the main bottleneck remains generation quality + effective Math/Reality participation (not scope pass-through wiring).

## Recommended Next Step

Proceed to Week 22.2 generation-quality path:
- enforce effective Math/Reality usage at generation-time (not just CLI scope),
- then re-run this same A/B harness.
