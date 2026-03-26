# Codex: Fix DefeasibleResolver.resolve() + Rerun Warm 35%

**Date:** 2026-03-25
**Priority:** IMMEDIATE — This is the ONLY blocker. Fix it, rerun.
**Binding spec:** `docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md` §4.1 — fail-fast, no silent fallbacks. The crash IS the spec working correctly — it surfaced a real bug instead of hiding behind numpy's permissive type coercion.

---

## Part A: The Bug (1-Line Fix)

**File:** `knowledge3d/cranium/bridges/sovereign_bridges.py`
**Function:** `DefeasibleResolver.resolve()` — lines 1169-1182

**Root cause:** `superiority_arr` is ONLY assigned inside the `if len(superiority_shape) == 1:` branch. When superiority input is rank-2 (which is the COMMON case from `_apply_intra_path_defeasible`), the variable is never assigned, and line 1178 crashes with `UnboundLocalError`.

**Current code (broken):**
```python
superiority_shape = _shape_of(superiority)
if len(superiority_shape) == 1:                    # rank-1: reshape
    if max_superiors is None:
        raise ValueError("max_superiors_required_for_flat_superiority")
    superiority_arr = HostTensorF32.from_array_like(
        [int(value) for value in superiority],
        rows=worker_count,
        cols=int(max_superiors),
    )
if superiority_arr.ndim != 2 or ...:              # ← CRASH: unbound for rank-2 input
```

**Fix:** Add the `else` branch for rank-2 input:
```python
superiority_shape = _shape_of(superiority)
if len(superiority_shape) == 1:
    if max_superiors is None:
        raise ValueError("max_superiors_required_for_flat_superiority")
    superiority_arr = HostTensorF32.from_array_like(
        [int(value) for value in superiority],
        rows=worker_count,
        cols=int(max_superiors),
    )
else:
    superiority_arr = _f32_matrix(superiority)
if superiority_arr.ndim != 2 or superiority_arr.shape[0] != worker_count:
    raise ValueError("superiority_shape_mismatch")
```

This mirrors EXACTLY the pattern used for `conclusions` on lines 1149-1154 (rank-1 gets reshaped, rank-2 goes through `_f32_matrix`).

**That is the ENTIRE fix.** One `else` clause.

---

## Part B: Validate the Fix

```bash
# 1. Compile check
python3 -m compileall knowledge3d/cranium/bridges/sovereign_bridges.py

# 2. Focused test
pytest -q tests/test_trm_game_loop.py tests/test_routing_contrastive_multihop.py

# 3. Quick defeasible smoke — call resolve() directly with rank-2 superiority
python3 -c "
from knowledge3d.cranium.bridges.sovereign_bridges import DefeasibleResolver
resolver = DefeasibleResolver()

# Rank-2 superiority (the case that crashed)
conclusions = [[0.8, 0.3], [0.4, 0.9], [0.6, 0.5]]  # 3 workers, 2 candidates
rule_strengths = [1, -1, 1]                             # 3 workers
superiority = [[0, 1], [2, 0], [1, 0]]                 # 3 workers x 2 max_superiors

verdicts, proof_tags = resolver.resolve(
    conclusions,
    rule_strengths,
    superiority,
    num_workers=3,
    num_candidates=2,
    max_superiors=2,
)
print(f'verdicts: {list(verdicts)}')
print(f'proof_tags: {list(proof_tags)}')
print('DefeasibleResolver rank-2 path: PASSED')
"
```

---

## Part C: Rerun Warm 35% Benchmark

After the fix validates:

```bash
cd "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D"
export CUDA_VISIBLE_DEVICES=0
conda activate k3d-cranium

nohup python3 -u benchmarks/run_all.py \
  --warm --sample-rate 0.35 \
  > /tmp/k3d_phase3c_fixed_warm_35pct_03.25.log 2>&1 &

echo "Warm 35% benchmark launched. PID: $!"
echo "Log: /tmp/k3d_phase3c_fixed_warm_35pct_03.25.log"
```

**While the benchmark runs:** Start a live monitor capture (60s sample) so we can compare GPU utilization before and after the Phase 3 migration.

---

## Part D: After Benchmark Completes

Write the report to `TEMP/CLAUDE_PHASE3C_FIXED_WARM_35PCT_REPORT_03.25.2026.md` with:

1. **All 5 suite scores** (ARC, Math, GSM8K, LHE, MMLU, Combined)
2. **Contrastive/sleep-time outcome** (trained: true/false per specialist, checkpoint status)
3. **Live monitor snapshot** (GPU util avg/max, CPU%, VRAM usage)
4. **Throughput per suite** (seconds per question — compare with pre-Phase-3 baseline: ARC was ~16s/q, Math was ~3.3s/q)
5. **NumPy count** in cranium package: `rg "import numpy|from numpy" knowledge3d/cranium/ --type py -c`

**Comparison baselines:**
| Metric | Pre-Phase-3 (03.24 AM) | Post-Phase-1 (03.24 noon) | Post-Phase-3C (this run) |
|--------|----------------------|--------------------------|------------------------|
| GPU util | 1.25% | 0% | ? |
| CPU% | 149% | 146% | ? |
| VRAM | 306 MB | 1190 MB | ? |
| Combined score | 19.21% | 18.66% | ? |
| Contrastive | all failed | all trained:true | ? |

---

## WHY THIS MATTERS

The crash was GOOD. Under old numpy, `superiority` would silently coerce through `np.asarray()` regardless of rank, masking the shape logic bug. The sovereign `HostTensorF32` path is STRICT — it surfaced a real bug that numpy was hiding. This is EXACTLY what `docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md` §4.1 means by "fail-fast, no silent fallbacks."
