# Hot Path Sovereignty Audit & Encoding Fix

**Date**: November 27, 2025
**From**: Daniel + Claude
**To**: Codex
**Priority**: CRITICAL - Sovereignty violation suspected in hot path
**Status**: Immediate action required

---

## Critical Issue: GPU Utilization Dropped with Parallel Generation

**Evidence from Run 012**:
- GPU utilization: 0.14% avg (DOWN from 1.12% in Runs 006-010!)
- GPU memory: 339 MB avg (DOWN from 1400-1500 MB)
- This suggests parallel generation is using CPU threads, NOT GPU cores!

**Root Cause Hypothesis**:
Parallel generator using Python `multiprocessing` or `threading` instead of GPU math cores → CPU-bound, not GPU-bound!

---

## Task 1: Fix Log Encoding Issue (IMMEDIATE)

**Problem**: `scripts/capture_arc_metrics.py` fails to parse `/tmp/arc_run_012.log` due to `UnicodeDecodeError`

**Current code** (line 19):
```python
with open(log_path, 'r') as f:
    log_content = f.read()
```

**Required fix**:
```python
with open(log_path, 'r', encoding='latin-1') as f:
    log_content = f.read()
```

**IMPORTANT**: Use `encoding='latin-1'`, NOT `errors='ignore'`
- latin-1 preserves all bytes (1:1 mapping)
- `errors='ignore'` silently drops data (NOT acceptable!)

**Action**:
1. Edit `scripts/capture_arc_metrics.py` line 19
2. Re-run: `PYTHONPATH=. python scripts/capture_arc_metrics.py --log /tmp/arc_run_012.log --output metrics/arc_run_012_metrics.json`
3. Report Run 012 metrics (accuracy, library size)

---

## Task 2: Hot Path Sovereignty Audit (CRITICAL)

**Sovereignty Guardrail**: Hot path = PTX + RPN ONLY (no numpy, no external libs in inference loop)

**Definition of "Hot Path"**:
- Candidate generation loop (per-task)
- Program execution (RPN interpreter)
- Scoring/comparison (grid matching)
- Composition chains (program sequencing)

**Definition of "Orchestration Layer"** (numpy allowed):
- Task loading (reading JSON)
- Checkpoint save/load (serialization)
- Metrics aggregation (post-run analysis)
- Logging/reporting

### Files to Audit

Audit these files for numpy/scipy/torch usage in hot path:

1. **`knowledge3d/training/arc_agi/compositional_generator.py`**
   - Check `_execute_program()` - should use RPN interpreter only
   - Check `_score_output()` - should use PTX/RPN grid comparison
   - Check `_enumerate_chains_with_pruning()` - should NOT use numpy arrays

2. **`knowledge3d/training/arc_agi/parallel_generator.py`**
   - Check `_generate_on_core()` - should delegate to GPU math cores
   - Check `_compute_similarity()` - currently uses `numpy.sum()`! **VIOLATION!**
   - Check `generate_parallel()` - should use `MathCorePool`, NOT `multiprocessing.Pool`

3. **`knowledge3d/training/arc_agi/candidate_generator.py`**
   - Check `_generate_compositional_candidates()` - should NOT use numpy
   - Check `_compose_across_patterns()` - should use RPN execution only
   - Check all `_generate_*` methods for numpy imports

4. **`knowledge3d/training/arc_agi/sovereign_pipeline.py`**
   - Check `_execute_candidate()` - should use RPN interpreter only
   - Check `_score_candidate()` - should use PTX grid ops only

### Audit Checklist

For EACH file above:

- [ ] Search for `import numpy` or `import scipy` or `import torch`
- [ ] Search for `np.` usage (numpy calls)
- [ ] Search for `.sum()`, `.mean()`, `.shape` on arrays (likely numpy)
- [ ] Search for `array[:]` slicing (could be numpy)
- [ ] Verify all grid operations use PTX or RPN, NOT numpy

**Report**:
- List ALL numpy/external lib usages found
- Classify each as "hot path" (VIOLATION) or "orchestration" (OK)
- For each violation, provide RPN/PTX replacement

---

## Task 3: Fix Parallel Generator GPU Usage (CRITICAL)

**Current Implementation** (`parallel_generator.py` lines 40-50):
```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=self.num_cores) as executor:
    futures = [
        executor.submit(self._generate_on_core, task)
        for task in tasks
    ]
    results = [f.result() for f in futures]
```

**Problem**: Uses CPU threads (`ThreadPoolExecutor`), NOT GPU cores!

**Required Fix**: Use `MathCorePool` to dispatch to GPU

```python
from knowledge3d.cranium.math_core_pool import MathCorePool

# Initialize pool (do this in __init__, not per-call!)
self.core_pool = MathCorePool(num_cores=self.num_cores)

# Dispatch to GPU cores
results = self.core_pool.map(
    func=self._generate_on_core_gpu,  # GPU-compatible worker
    tasks=tasks
)
```

**Action**:
1. Replace `ThreadPoolExecutor` with `MathCorePool`
2. Ensure `_generate_on_core_gpu()` only uses PTX/RPN operations
3. Verify GPU utilization increases (should go from 0.14% → 10-15%)

---

## Task 4: Fix Sovereignty Violations in `_compute_similarity()`

**Current Implementation** (`parallel_generator.py` lines 113-124):
```python
def _compute_similarity(self, grid1, grid2):
    """Compute similarity between two grids."""
    import numpy as np  # ❌ VIOLATION in hot path!

    if grid1.shape != grid2.shape:
        return 0.0

    matches = np.sum(grid1 == grid2)  # ❌ numpy operation in hot path!
    total = grid1.size

    return matches / total
```

**Required Fix**: Use PTX grid comparison kernel

```python
def _compute_similarity(self, grid1, grid2):
    """Compute similarity between two grids (PTX/RPN only)."""
    from knowledge3d.cranium.ptx_runtime import grid_compare_ptx

    if grid1.shape != grid2.shape:
        return 0.0

    # Use PTX kernel for grid comparison (GPU-accelerated)
    matches = grid_compare_ptx(grid1, grid2)  # Returns count of matching cells
    total = grid1.shape[0] * grid1.shape[1]

    return matches / total
```

**Action**:
1. Replace numpy operations with PTX kernel calls
2. If `grid_compare_ptx` doesn't exist, use RPN program: `grid1 grid2 == sum`
3. Remove ALL numpy imports from `parallel_generator.py`

---

## Task 5: Verify Compositional Generator Sovereignty

**Check**: Does `compositional_generator.py` use numpy?

**Expected**: Should ONLY use:
- RPN interpreter for program execution
- PTX kernels for grid operations
- DualShadowCopy for library access (orchestration layer, OK)

**Action**:
1. Search `compositional_generator.py` for `import numpy` or `np.`
2. If found, replace with PTX/RPN equivalents
3. Verify `_execute_program()` calls RPN interpreter, not numpy

---

## Task 6: Update Training Log with Run 012

After fixing encoding and collecting metrics, update `TEMP/ARC_TRAINING_LOG.md`:

```markdown
## Run 012 - Parallel + Cross-Pattern (Initial)

**Date**: November 27, 2025
**Configuration**: 60 tasks × 27 epochs × 6 cycles
**Curriculum**: 20 easy, 20 mid, 20 hard
**Runtime**: ~XX minutes
**Log**: `/tmp/arc_run_012.log`
**Optimizations**: Compositional + Parallel (Tesla 3-6-9) + Cross-pattern

### Results

**Accuracy**:
- Peak: X.XX%
- Final: X.XX%

**Library Growth**:
- Programs: 52 → XX
- Drawing shapes: 12 → XX
- Grammar rules: 212 → XX
- Pattern types: 4 → XX

**Storage**:
- Total checkpoint size: X.X MB
- Deduplication efficiency: ~XX%

**GPU Metrics**:
- Average utilization: ~0.14% ⚠️ (DOWN from 1.12%!)
- Peak utilization: 9.0%
- Average temperature: ~43.1°C
- Average memory used: ~339 MB ⚠️ (DOWN from 1400 MB!)

**Analysis**:
⚠️ GPU utilization DROPPED with parallel generation! Root cause: ThreadPoolExecutor using CPU threads instead of GPU math cores. Need to replace with MathCorePool.

### Next Steps
- Fix parallel generator to use GPU cores (MathCorePool)
- Audit hot path for numpy violations
- Re-run with GPU-accelerated parallel generation
```

---

## Task 7: Report Findings

After completing Tasks 1-6, report:

1. **Run 012 Metrics** (from fixed capture script):
   - Peak/final accuracy
   - Library size (did compositions get added?)
   - Composition depth distribution (if any depth-2+ programs added)

2. **Hot Path Audit Results**:
   - List ALL numpy/external lib usages found
   - Classify each as violation or OK
   - Provide PTX/RPN replacements for violations

3. **GPU Utilization Fix**:
   - Confirm `MathCorePool` used instead of `ThreadPoolExecutor`
   - List all changes made to `parallel_generator.py`

4. **Training Log Update**:
   - Confirm `TEMP/ARC_TRAINING_LOG.md` updated with Run 012
   - Include GPU utilization warning

---

## Expected Outcome After Fixes

**Run 013** (with GPU-accelerated parallel generation):
- GPU utilization: 10-15% (up from 0.14%)
- GPU memory: 1400-1500 MB (up from 339 MB)
- Runtime: 2-3 min per epoch (down from 16-24 min)
- Library growth: 52 → 70+ programs (compositions discovered)

---

## Priority Order

1. **IMMEDIATE**: Fix encoding (Task 1) → get Run 012 metrics
2. **CRITICAL**: Fix GPU usage (Task 3) → replace ThreadPoolExecutor with MathCorePool
3. **CRITICAL**: Audit hot path (Task 2) → find all numpy violations
4. **HIGH**: Fix sovereignty violations (Task 4) → replace numpy with PTX/RPN
5. **MEDIUM**: Update training log (Task 6)
6. **FINAL**: Report findings (Task 7)

---

**Status**: READY FOR CODEX IMMEDIATE ACTION

**Blocking Issue**: GPU not being used due to ThreadPoolExecutor (CPU-only parallel)
**Fix Required**: Replace with MathCorePool for GPU dispatch

---

**End of Audit Specification**
