# Codex Week 21.5 PTX Path Enforcement (2026-02-10)

## What Was Fixed

1. ARC adapter now enforces PTX-only execution when `--arc-enable-full-ptx` is set.
2. Legacy ARC pipeline path is blocked by default unless explicitly overridden:
   - `K3D_REQUIRE_PTX_ARC_PIPELINE=true` (default)
   - `K3D_ALLOW_LEGACY_ARC_PIPELINE=false` (default under full PTX)
3. Added a PTX-only solve path in `benchmarks/arc_agi_2_adapter.py` (`_solve_task_ptx_only`) so ARC can run without `SovereignAIPipeline`.
4. `scripts/run_all_benchmarks.py` and `scripts/run_all_global_benchmarks.py` now auto-enable PTX ranking when full PTX is requested.
5. Added hard solver contract checks in `scripts/run_all_benchmarks.py`:
   - If `--arc-enable-full-ptx`, every ARC row must report `solver=arc_ptx_ops`.

## Files Updated

- `benchmarks/arc_agi_2_adapter.py`
- `benchmarks/arc_agi_2.py`
- `scripts/run_all_benchmarks.py`
- `scripts/run_all_global_benchmarks.py`

## Validation Runs

### Fail-fast validation (non-PTX shell)
Command failed as expected with clear error:
- `RuntimeError: PTX-only ARC path requested but ARC PTX operations are unavailable (reason=cupy_missing,full_ptx_cupy_missing)`

This confirms legacy silent fallback is removed.

### PTX validation (`k3d-cranium` env)
Command:

```bash
/home/daniel/miniconda3/bin/conda run -n k3d-cranium env PYTHONPATH=. python scripts/run_all_benchmarks.py \
  --model-persistence-mode unified \
  --max-arc-tasks 5 \
  --max-math-problems 5 \
  --max-lhe-questions 5 \
  --arc-enable-full-ptx \
  --arc-enable-contrastive-learning \
  --arc-enable-validity-gates \
  --arc-enable-fuzzy-oracle \
  --arc-fuzzy-oracle-threshold 0.95 \
  --arc-embedding-lazy-mode skip \
  --output-dir ../Knowledge3D.local/results/week21_5_ptx_validation \
  --storage-root ../Knowledge3D.local
```

Observed:
- ARC per-task solver set: `arc_ptx_ops` only
- `ptx_full_used_rate: 1.0`
- `ptx_ranking_used_rate: 1.0`
- No legacy solver rows in summary

Artifacts:
- `../Knowledge3D.local/results/week21_5_ptx_validation/week14_benchmark_summary.json`
- `../Knowledge3D.local/results/week21_5_ptx_validation_contract2/week14_benchmark_summary.json`

## Current Status

- Architecture/path issue is fixed (legacy fallback no longer silently hijacks full-PTX runs).
- Oracle/accuracy are still low in small validation runs; now this is a real model/pattern-quality problem, not a pipeline-routing problem.

## Suggestions for Next Step (for Claude review)

1. Promote `arc_ptx_ops` output schema to include richer candidate provenance:
   - explicit family, palette-map confidence, object-count delta confidence, and source precision used in score.
2. Add pre-rank candidate diversity pressure (ternary pool entropy target) before top-k cut.
3. Move first-pass candidate generation transforms from heuristic text parsing to explicit train-pair derived transform operators (PTX kernels where possible).
4. Add a strict “oracle progress gate” in training loops:
   - block stage promotion unless `oracle_at_all` and `fuzzy_oracle_at_all` both improve over rolling baseline.
5. Keep unified world policy enforced for all benchmark entrypoints (including global runner) with same solver contract check.

## Questions for Claude

1. For Week 21.6, should we prioritize family-consistent candidate generation or fuzzy oracle calibration first?
2. Do you want strict PTX-only enforcement extended to math/LHE hot paths now, or keep scope ARC-first?
3. Should we set a hard minimum for candidate pool entropy before ranking to avoid early collapse?
