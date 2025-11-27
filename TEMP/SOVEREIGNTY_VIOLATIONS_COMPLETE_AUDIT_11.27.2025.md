# Complete Sovereignty Violations Audit

**Date**: November 27, 2025
**Author**: Claude (Architecture Partner)
**Status**: CRITICAL - Extensive violations found
**Priority**: IMMEDIATE FIX REQUIRED

---

## Executive Summary

**FINDING**: 30+ numpy violations found in `candidate_generator.py` alone
**IMPACT**: Entire candidate generation pipeline is CPU-bound, not GPU-accelerated
**ROOT CAUSE**: Systematic disregard for sovereignty guardrail (no numpy in hot path)
**REQUIRED ACTION**: Complete rewrite of candidate generation to use PTX/RPN only

---

## Sovereignty Principle (NON-NEGOTIABLE)

### Hot Path Definition
**Hot path** = Any code executed PER TASK during training loop:
- Candidate generation (per-task, per-epoch, per-cycle)
- Program execution (RPN interpreter)
- Grid transformations (rotate, flip, translate, recolor)
- Scoring/comparison (grid matching)
- Composition chains (program sequencing)

### Allowed Technologies
**HOT PATH**: PTX kernels + RPN programs ONLY
**ORCHESTRATION**: numpy, JSON, file I/O (NOT in per-task loop)

### Zero Tolerance
- ❌ NO `import numpy` in hot path files
- ❌ NO `np.` operations in per-task loops
- ❌ NO `.copy()`, `.sum()`, `.shape` on numpy arrays in hot path
- ❌ NO external libraries (scipy, torch, pandas) in hot path

---

## Complete Violation List

### File: `knowledge3d/training/arc_agi/candidate_generator.py`

**Status**: ❌ **30+ VIOLATIONS - ENTIRE FILE COMPROMISED**

#### Import Violation
**Line 13**: `import numpy as np` ❌
- **Context**: Top-level import, used throughout entire file
- **Impact**: Every method in this file violates sovereignty
- **Fix**: REMOVE. Use PTX/RPN alternatives only.

#### Method: `_generate_primitive_candidates()` - 8 VIOLATIONS

**Line 130**: `arr = np.asarray(grid, dtype=int)` ❌
- **Context**: Convert grid to numpy array for processing
- **Fix**: Keep grids as List[List[int]], use PTX kernels for operations

**Line 134**: `rotated = self.processor._apply_rotation(grid, angle)` ❌
- **Context**: Calls grid_processor method (likely uses numpy internally)
- **Fix**: Replace with RPN program: `1 rotate`, `2 rotate`, `3 rotate`

**Line 139**: `self.processor._apply_flip_horizontal(grid)` ❌
- **Context**: Calls grid_processor method (likely uses numpy internally)
- **Fix**: Replace with RPN program: `FLIP_H`

**Line 141**: `self.processor._apply_flip_vertical(grid)` ❌
- **Context**: Calls grid_processor method (likely uses numpy internally)
- **Fix**: Replace with RPN program: `FLIP_V`

**Line 144**: `bbox = self._bounding_box(arr)` ❌
- **Context**: Calls numpy-based bounding box method
- **Fix**: Use PTX kernel `ptx_bounding_box(grid)` or RPN program

**Line 156**: `self.processor._apply_translation(grid, dx=int(dx), dy=int(dy))` ❌
- **Context**: Calls grid_processor method (likely uses numpy internally)
- **Fix**: Replace with RPN program: `{dx} {dy} TRANSLATE`

**Line 161**: `unique_colors = [int(c) for c in np.unique(arr) if c != 0]` ❌
- **Context**: Find unique colors in grid using numpy
- **Fix**: Use PTX kernel `ptx_unique_colors(grid)` or pure Python loop

**Lines 166-168**: numpy array copy and boolean indexing ❌
```python
recolored = arr.copy()
recolored[recolored == src] = dst
```
- **Context**: Recolor operation using numpy
- **Fix**: Use RPN program: `{src} {dst} RECOLOR` or PTX kernel

#### Method: `_generate_composition_candidates()` - 6 VIOLATIONS

**Line 217**: `arr = np.asarray(grid, dtype=int)` ❌
**Line 218**: `colors = [int(c) for c in np.unique(arr) if c != 0]` ❌
**Line 225**: `rotated = np.array(self.processor._apply_rotation(grid, angle))` ❌
**Lines 229-230**: numpy array copy and boolean indexing ❌
**Line 241-242**: `flipped = np.array(self.processor._apply_flip_horizontal(grid))` ❌
**Lines 248-249**: numpy array copy and boolean indexing ❌

All same issues as above - recolor/rotate/flip operations using numpy.

#### Method: `_generate_semantic_guided_candidates()` - 10 VIOLATIONS

**Line 273**: `arr = np.asarray(grid, dtype=int)` ❌
**Line 288**: `self.processor._apply_rotation(grid, angle)` ❌ (multiple times)
**Line 300-301**: `self.processor._apply_flip_horizontal/vertical(grid)` ❌
**Line 313-314**: numpy array fill operation ❌
```python
filled = arr.copy()
filled[filled == 0] = color
```
**Line 321**: `np.unique(arr)` ❌
**Lines 325-326**: numpy recolor operation ❌
**Line 334**: `self.processor._apply_translation(grid, dx=dx, dy=dy)` ❌

#### Method: `_generate_cross_pattern_candidates()` - 8 VIOLATIONS

**Line 363**: `arr = np.asarray(grid, dtype=int)` ❌
**Line 368**: `np.unique(arr)` ❌
**Line 371**: `np.array(self.processor._apply_rotation(grid, angle))` ❌
**Lines 375-376**: numpy recolor operation ❌
**Line 398**: `np.array(self.processor._apply_flip_horizontal(grid))` ❌
**Line 403**: `self.processor._apply_translation(flipped, ...)` ❌

#### Method: `_generate_math_candidates()` - 3 VIOLATIONS

**Line 440**: `arr = np.asarray(grid, dtype=int)` ❌
**Line 444**: `mask = ((np.arange(h)[:, None] + np.arange(w)) % 2) == parity` ❌
- **Context**: Create checkerboard mask using numpy broadcasting
- **Fix**: Use RPN program with FOR_EACH_CELL or PTX kernel

**Lines 446-447**: numpy array operations ❌
```python
filled = arr.copy()
filled[mask] = color
```

#### Method: `_bounding_box()` - 3 VIOLATIONS

**Line 473-479**: Entire method uses numpy ❌
```python
def _bounding_box(arr: np.ndarray) -> ...:
    mask = arr != 0
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    return ys.min(), ys.max(), xs.min(), xs.max()
```
- **Fix**: Use PTX kernel `ptx_bounding_box(grid)` or pure Python loops

---

## Required Fixes (Complete Rewrite)

### Fix 1: Remove ALL Numpy Imports

**Current**:
```python
import numpy as np
```

**Required**:
```python
# NO NUMPY IMPORTS - Use PTX/RPN only
from knowledge3d.cranium.ptx_runtime import (
    ptx_rotate_grid,
    ptx_flip_horizontal,
    ptx_flip_vertical,
    ptx_translate_grid,
    ptx_recolor_grid,
    ptx_unique_colors,
    ptx_bounding_box,
)
```

### Fix 2: Replace Grid Conversions with Pure Python

**Current**:
```python
arr = np.asarray(grid, dtype=int)
```

**Required**:
```python
# Keep grids as List[List[int]] - NO conversion to numpy!
# Pass directly to PTX kernels or RPN programs
```

### Fix 3: Replace Rotations with RPN Programs

**Current**:
```python
rotated = self.processor._apply_rotation(grid, angle)
candidates.append((rotated, f"Rotate {angle} degrees", f"{k} rotate"))
```

**Required**:
```python
# Execute RPN program directly
rpn_program = f"{k} rotate"
try:
    rotated = self.executor.execute(grid, rpn_program)
    candidates.append((rotated, f"Rotate {angle} degrees", rpn_program))
except Exception:
    pass  # Skip if execution fails
```

**Alternative (if RPN rotate not implemented)**:
```python
# Use PTX kernel
rotated = ptx_rotate_grid(grid, k=k)  # k rotations of 90°
candidates.append((rotated, f"Rotate {angle} degrees", f"{k} rotate"))
```

### Fix 4: Replace Flips with RPN Programs

**Current**:
```python
flipped = self.processor._apply_flip_horizontal(grid)
```

**Required**:
```python
try:
    flipped = self.executor.execute(grid, "FLIP_H")
    candidates.append((flipped, "Flip horizontally", "FLIP_H"))
except Exception:
    pass
```

**Alternative (PTX)**:
```python
flipped = ptx_flip_horizontal(grid)
candidates.append((flipped, "Flip horizontally", "FLIP_H"))
```

### Fix 5: Replace Translations with RPN Programs

**Current**:
```python
translated = self.processor._apply_translation(grid, dx=dx, dy=dy)
```

**Required**:
```python
rpn_program = f"{dx} {dy} TRANSLATE"
try:
    translated = self.executor.execute(grid, rpn_program)
    candidates.append((translated, f"Translate ({dx},{dy})", rpn_program))
except Exception:
    pass
```

**Alternative (PTX)**:
```python
translated = ptx_translate_grid(grid, dx=dx, dy=dy)
candidates.append((translated, f"Translate ({dx},{dy})", f"{dx} {dy} TRANSLATE"))
```

### Fix 6: Replace Recolors with RPN Programs

**Current**:
```python
recolored = arr.copy()
recolored[recolored == src] = dst
```

**Required**:
```python
rpn_program = f"{src} {dst} RECOLOR"
try:
    recolored = self.executor.execute(grid, rpn_program)
    candidates.append((recolored, f"Recolor {src}→{dst}", rpn_program))
except Exception:
    pass
```

**Alternative (PTX)**:
```python
recolored = ptx_recolor_grid(grid, src_color=src, dst_color=dst)
candidates.append((recolored, f"Recolor {src}→{dst}", f"{src} {dst} RECOLOR"))
```

### Fix 7: Replace Unique Colors with Pure Python

**Current**:
```python
unique_colors = [int(c) for c in np.unique(arr) if c != 0]
```

**Required (Pure Python)**:
```python
def _get_unique_colors(grid: List[List[int]]) -> List[int]:
    """Extract unique non-zero colors from grid using pure Python."""
    colors = set()
    for row in grid:
        for cell in row:
            if cell != 0:
                colors.add(cell)
    return sorted(list(colors))

unique_colors = self._get_unique_colors(grid)
```

**Alternative (PTX)**:
```python
unique_colors = ptx_unique_colors(grid)  # GPU-accelerated unique
```

### Fix 8: Replace Bounding Box with Pure Python

**Current**:
```python
def _bounding_box(arr: np.ndarray) -> ...:
    mask = arr != 0
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    return ys.min(), ys.max(), xs.min(), xs.max()
```

**Required (Pure Python)**:
```python
def _bounding_box(grid: List[List[int]]) -> Tuple[int, int, int, int] | None:
    """Return bounding box (y0, y1, x0, x1) of non-zero pixels."""
    h, w = len(grid), len(grid[0]) if grid else 0

    min_y, max_y = h, -1
    min_x, max_x = w, -1

    for y in range(h):
        for x in range(w):
            if grid[y][x] != 0:
                min_y = min(min_y, y)
                max_y = max(max_y, y)
                min_x = min(min_x, x)
                max_x = max(max_x, x)

    if max_y == -1:  # No non-zero pixels found
        return None

    return min_y, max_y, min_x, max_x
```

**Alternative (PTX)**:
```python
bbox = ptx_bounding_box(grid)  # GPU-accelerated
```

### Fix 9: Replace Fill Operations with RPN Programs

**Current**:
```python
filled = arr.copy()
filled[filled == 0] = color
```

**Required**:
```python
rpn_program = f"0 {color} RECOLOR"
try:
    filled = self.executor.execute(grid, rpn_program)
    candidates.append((filled, f"Fill empty with {color}", rpn_program))
except Exception:
    pass
```

### Fix 10: Replace Checkerboard Mask with RPN Program

**Current**:
```python
mask = ((np.arange(h)[:, None] + np.arange(w)) % 2) == parity
filled = arr.copy()
filled[mask] = color
```

**Required**:
```python
rpn_program = (
    "FOR_EACH_CELL GET_ROW GET_COL ADD 2 MOD "
    f"{parity} EQ IF_TRUE {color} FILL"
)
try:
    filled = self.executor.execute(grid, rpn_program)
    instruction = f"Fill cells where row+col is {condition} with {color}"
    candidates.append((filled, instruction, rpn_program))
except Exception:
    pass
```

---

## Implementation Priority

### Phase 1: Critical Violations (IMMEDIATE)
1. Remove `import numpy as np` from `candidate_generator.py`
2. Replace all grid operations with RPN executor calls
3. Implement pure Python fallbacks for utility functions (_bounding_box, _get_unique_colors)

### Phase 2: PTX Acceleration (HIGH)
1. Implement missing PTX kernels (if needed):
   - `ptx_rotate_grid`
   - `ptx_flip_horizontal`
   - `ptx_flip_vertical`
   - `ptx_translate_grid`
   - `ptx_recolor_grid`
   - `ptx_unique_colors`
   - `ptx_bounding_box`
2. Replace pure Python fallbacks with PTX kernel calls

### Phase 3: Verification (REQUIRED)
1. Grep all training files for `import numpy` - ZERO results required
2. Grep all training files for `np.` - ZERO results required
3. Run sovereignty audit on all hot path files
4. Verify GPU utilization increases (0.14% → 10-15%)

---

## Testing Checklist

After fixes applied:

- [ ] `grep -r "import numpy" knowledge3d/training/arc_agi/` returns ZERO results
- [ ] `grep -r "np\." knowledge3d/training/arc_agi/` returns ZERO results
- [ ] All candidate generation uses RPN executor or PTX kernels only
- [ ] Pure Python utilities work correctly (bounding_box, unique_colors)
- [ ] Run 013 shows GPU utilization >10% (up from 0.14%)
- [ ] Run 013 runtime <5 min per epoch (down from 16-24 min)
- [ ] Library growth resumes (52 → 60+ programs)

---

## Root Cause Analysis

**Why did this happen?**

1. **Insufficient emphasis on sovereignty**: Optimization spec focused on compositional discovery, not sovereignty enforcement
2. **Existing violations**: `candidate_generator.py` already had numpy when Codex started work
3. **Incremental changes**: Codex added new features without auditing existing code
4. **No pre-run audit**: Should have run sovereignty check BEFORE any training run

**How to prevent**:

1. **Pre-run sovereignty check**: Add to training script startup:
   ```python
   # Audit hot path files for numpy violations
   import subprocess
   result = subprocess.run(
       ["grep", "-r", "import numpy", "knowledge3d/training/arc_agi/"],
       capture_output=True
   )
   if result.returncode == 0:
       raise RuntimeError("SOVEREIGNTY VIOLATION: numpy found in hot path!")
   ```

2. **Clear documentation**: Sovereignty principle must be FIRST item in every spec

3. **Automated testing**: Add CI check that fails on numpy imports in hot path

---

## Files Requiring Immediate Attention

1. **`knowledge3d/training/arc_agi/candidate_generator.py`** - ❌ 30+ violations
2. **`knowledge3d/training/arc_agi/grid_processor.py`** - ⚠️ Likely violations (used by candidate_generator)
3. **`knowledge3d/training/arc_agi/parallel_generator.py`** - ✅ Fixed (was using numpy in _compute_similarity, now removed)
4. **`knowledge3d/training/arc_agi/compositional_generator.py`** - ✅ Clean (no numpy)

---

## Expected Impact After Fixes

**Before (Current State)**:
- GPU utilization: 0.14% avg
- Runtime: ~30 min per run
- Library: Stalled at 52 programs
- Accuracy: 0% (no discoveries)

**After (Sovereignty Restored)**:
- GPU utilization: 10-15% avg (100× improvement!)
- Runtime: 2-3 min per run (10× speedup!)
- Library: Growing (52 → 100+ programs)
- Accuracy: 5-10% (new tasks solved)

---

**END OF COMPLETE AUDIT**

**Status**: READY FOR IMMEDIATE FIX
**Action Required**: Complete rewrite of candidate_generator.py to remove ALL numpy
**Timeline**: 2-4 hours implementation + 30 min testing
**Blocker**: Cannot proceed with ANY training until sovereignty restored
