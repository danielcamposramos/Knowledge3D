# CODEX BRIEFING: Sovereignty Refactor — Remove NumPy from Hot Path

**Date:** November 24, 2025
**From:** Claude (Architect)
**To:** Codex (Implementation Lead)
**Phase:** Sovereignty Compliance — CRITICAL REFACTOR
**Priority:** URGENT — Fix hot path violations before GPU validation

---

## Executive Summary

**Status:** ⚠️ **SOVEREIGNTY VIOLATION CONFIRMED**

Your audit revealed numpy imports throughout the hot path (RealityGalaxy, RPN engines, tier bridges). This violates K3D's core principle: **sovereign hot path = PTX + RPN only, NO external array libraries.**

**Your Mission (Priority Order):**
1. **Refactor hot path modules:** Remove numpy from inference loop
2. **Add runtime guards:** Prevent future violations (CI checks)
3. **Validate refactor:** Run full test suite, ensure 92/92 tests still pass
4. **THEN install CuPy:** Enable GPU tests (after hot path is clean)
5. **THEN run GPU benchmarks:** Show sovereign PTX performance
6. **Update docs:** Mark sovereignty ✅ COMPLIANT

**Why This Order:** We cannot claim GPU validation success while hot path is non-compliant. Fix sovereignty FIRST, then validate GPU.

**Timeline:** 1-2 days (refactor is priority; GPU validation after)

---

## Context: What Went Wrong

### Original Architecture Intent
- **Hot path:** Pure PTX kernels + RPN (deterministic, explainable)
- **Ingestion:** Flexible (numpy OK for preprocessing/export)

### Current Reality (Your Audit Findings)
```bash
# Hot path violations
knowledge3d/cranium/reality_galaxy.py: import numpy
knowledge3d/cranium/ptx_runtime/modular_rpn_engine.py: import numpy
knowledge3d/cranium/ptx_runtime/advanced_rpn.py: import numpy
knowledge3d/cranium/ptx_runtime/rpn_math_core.py: import numpy
knowledge3d/cranium/bridges/*.py: import numpy
```

**Root Cause:** Numpy used for convenience (array operations, math utilities) during rapid prototyping. Now we clean it up.

---

## Task 1: Refactor Strategy

### Approach: Replace NumPy with Native Python + ctypes

**NumPy Use Cases → Replacements:**

| NumPy Usage | Replacement | Example |
|-------------|-------------|---------|
| `np.array([1, 2, 3])` | `[1.0, 2.0, 3.0]` (native list) | State storage |
| `np.zeros(10)` | `[0.0] * 10` | Array initialization |
| `np.sin(x)` | `math.sin(x)` | Trig functions |
| `np.sqrt(x)` | `math.sqrt(x)` | Math operations |
| `np.dot(a, b)` | Manual loop or ctypes | Vector ops |
| `np.linalg.*` | Manual implementation | Matrix ops |

**Key Principle:** If operation is complex enough to need numpy, it should be in a **PTX kernel** instead.

---

## Task 2: File-by-File Refactoring

### Priority Order (High → Low Impact)

1. **reality_galaxy.py** (core orchestrator)
2. **ptx_runtime/modular_rpn_engine.py** (Tier-2 engine)
3. **ptx_runtime/advanced_rpn.py** (Tier-3 engine)
4. **ptx_runtime/rpn_math_core.py** (base Math Core)
5. **bridges/*.py** (tier routing)

---

### Refactor 1: reality_galaxy.py

**Current (Audit Finding):**
```python
import numpy as np

class RealityGalaxy:
    def step_system(self, node_id: str, n_steps: int = 1) -> None:
        system = self.nodes[node_id]
        # Likely using numpy for state arrays
        state_array = np.array(list(system.state.values()))
        # ... step logic
```

**Refactored (No NumPy):**
```python
# NO numpy import

class RealityGalaxy:
    def step_system(self, node_id: str, n_steps: int = 1) -> None:
        system = self.nodes[node_id]
        # Use native Python lists (state dict already uses lists/floats)
        # If Math Core needs arrays, it should use ctypes or PTX kernels

        for _ in range(n_steps):
            # Delegate to Math Core (which uses PTX kernels or RPN)
            core = self.math_core_pool.get_core(system.rpn_instance)
            core.execute(system.behavior_rpn, system.state)
```

**Changes:**
- Remove `import numpy`
- State storage remains as native Python dicts/lists (already correct)
- Math operations delegated to Math Core (RPN or PTX)

**Validation:**
```bash
# Check no numpy
grep "import numpy" knowledge3d/cranium/reality_galaxy.py
# Should return NOTHING

# Run tests
PYTHONPATH=. pytest knowledge3d/cranium/tests/test_reality_galaxy.py -v
# All tests should still pass
```

---

### Refactor 2: ptx_runtime/modular_rpn_engine.py

**Current (Hypothetical):**
```python
import numpy as np

class ModularRPNEngine:
    def execute(self, rpn_program: List[str], state: Dict) -> None:
        stack = []
        for token in rpn_program:
            if token == "sin":
                x = stack.pop()
                stack.append(np.sin(x))  # <-- NUMPY VIOLATION
            # ...
```

**Refactored:**
```python
import math  # stdlib only

class ModularRPNEngine:
    def execute(self, rpn_program: List[str], state: Dict) -> None:
        stack = []
        for token in rpn_program:
            if token == "sin":
                x = stack.pop()
                stack.append(math.sin(x))  # Native Python math
            # ...
```

**Changes:**
- Replace `np.sin` → `math.sin`
- Replace `np.sqrt` → `math.sqrt`
- Replace `np.abs` → `abs()` (builtin)
- Replace `np.array` → native list

**Validation:**
```bash
grep "import numpy" knowledge3d/cranium/ptx_runtime/modular_rpn_engine.py
# Should return NOTHING

PYTHONPATH=. pytest knowledge3d/cranium/tests/test_modular_rpn_engine.py -v
```

---

### Refactor 3: ptx_runtime/advanced_rpn.py

**Current (Hypothetical):**
```python
import numpy as np

class AdvancedRPNEngine:
    def _matvec(self, matrix: List[List[float]], vector: List[float]) -> List[float]:
        # Using numpy for convenience
        return np.dot(matrix, vector).tolist()  # <-- VIOLATION
```

**Refactored (Option A: Manual Loop):**
```python
class AdvancedRPNEngine:
    def _matvec(self, matrix: List[List[float]], vector: List[float]) -> List[float]:
        """Matrix-vector multiply (manual implementation)."""
        result = []
        for row in matrix:
            result.append(sum(row[i] * vector[i] for i in range(len(vector))))
        return result
```

**Refactored (Option B: PTX Kernel):**
```python
import ctypes
from .ptx_kernels import matvec_kernel  # Compiled PTX kernel

class AdvancedRPNEngine:
    def _matvec(self, matrix: List[List[float]], vector: List[float]) -> List[float]:
        """Matrix-vector multiply via PTX kernel."""
        # Flatten matrix, pass to PTX kernel
        flat_matrix = [elem for row in matrix for elem in row]
        result = matvec_kernel(flat_matrix, vector)  # PTX kernel
        return result
```

**Decision:** Use **Option A (manual loop)** for simplicity. If performance is bottleneck, move to PTX kernel in Phase 7.

**Validation:**
```bash
grep "import numpy" knowledge3d/cranium/ptx_runtime/advanced_rpn.py
# NOTHING

PYTHONPATH=. pytest knowledge3d/cranium/tests/test_advanced_rpn.py -v
```

---

### Refactor 4: ptx_runtime/rpn_math_core.py

**Current (Hypothetical):**
```python
import numpy as np

class MathCore:
    def __init__(self, tier: int):
        self.state_buffer = np.zeros(2048)  # <-- VIOLATION
```

**Refactored:**
```python
class MathCore:
    def __init__(self, tier: int):
        self.state_buffer = [0.0] * 2048  # Native Python list
        # Or if interfacing with PTX:
        # self.state_buffer = (ctypes.c_float * 2048)()  # ctypes array
```

**Changes:**
- Replace `np.zeros()` → `[0.0] * n` or ctypes array
- If state buffer needs GPU access, use ctypes for C interop

**Validation:**
```bash
grep "import numpy" knowledge3d/cranium/ptx_runtime/rpn_math_core.py
# NOTHING

PYTHONPATH=. pytest knowledge3d/cranium/tests/test_math_core.py -v
```

---

### Refactor 5: bridges/*.py

**Current (Hypothetical):**
```python
import numpy as np

class TieredRPNEngine:
    def route_by_opcode(self, rpn_program: List[str]) -> int:
        # Using numpy for analysis
        opcodes = np.array(rpn_program)  # <-- VIOLATION
        complexity = np.sum([len(op) for op in opcodes])  # Silly example
        return self._map_complexity_to_tier(complexity)
```

**Refactored:**
```python
class TieredRPNEngine:
    def route_by_opcode(self, rpn_program: List[str]) -> int:
        # Native Python (no numpy needed)
        complexity = sum(len(op) for op in rpn_program)
        return self._map_complexity_to_tier(complexity)
```

**Changes:**
- Replace numpy array operations with native Python comprehensions
- Use `sum()`, `max()`, `min()` builtins

**Validation:**
```bash
grep "import numpy" knowledge3d/cranium/bridges/*.py
# NOTHING

PYTHONPATH=. pytest knowledge3d/cranium/tests/test_tiered_rpn_engine.py -v
```

---

## Task 3: Runtime Guards

### Add Sovereignty Assertions

**File:** `knowledge3d/cranium/reality_galaxy.py`

**Add at top of `step_system()` method:**
```python
def step_system(self, node_id: str, n_steps: int = 1) -> None:
    """Step system forward n_steps (hot path)."""

    # SOVEREIGNTY GUARD
    import sys
    if __debug__:  # Only in debug mode (stripped in production)
        forbidden_modules = {'numpy', 'torch', 'tensorflow', 'cupy'}
        loaded = forbidden_modules & set(sys.modules.keys())
        if loaded:
            raise RuntimeError(
                f"SOVEREIGNTY VIOLATION: {loaded} in hot path! "
                f"Only PTX+RPN allowed in inference loop."
            )

    # ... existing implementation
```

**Effect:** Will raise exception if numpy/torch/tf/cupy are loaded during hot path execution.

**Test:**
```python
# This should PASS (no numpy in hot path)
def test_hot_path_sovereignty():
    from knowledge3d.cranium.reality_galaxy import RealityGalaxy
    from knowledge3d.cranium.reality_physics_export import export_harmonic_oscillator_1d

    galaxy = RealityGalaxy()
    system = export_harmonic_oscillator_1d()
    galaxy.add_node(system)

    # Should not raise (no numpy in hot path)
    galaxy.step_system(system.node_id, n_steps=10)

    # Verify
    import sys
    assert 'numpy' not in sys.modules, "Numpy should NOT be loaded"
```

---

## Task 4: CI Compliance Check

### Add to CI Pipeline

**File:** `.github/workflows/sovereignty-check.yml` (NEW, if using GitHub Actions)

```yaml
name: Sovereignty Compliance

on: [push, pull_request]

jobs:
  check-hot-path:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Check hot path for numpy imports
        run: |
          # Fail if numpy found in hot path modules
          if grep -r "import numpy" knowledge3d/cranium/ptx_runtime/; then
            echo "FAIL: numpy in ptx_runtime/"
            exit 1
          fi
          if grep "import numpy" knowledge3d/cranium/reality_galaxy.py; then
            echo "FAIL: numpy in reality_galaxy.py"
            exit 1
          fi
          if grep -r "import numpy" knowledge3d/cranium/bridges/; then
            echo "FAIL: numpy in bridges/"
            exit 1
          fi
          echo "PASS: Hot path is sovereign (no numpy)"

      - name: Run sovereignty test
        run: |
          python -m pytest knowledge3d/cranium/tests/test_sovereignty.py -v
```

**File:** `knowledge3d/cranium/tests/test_sovereignty.py` (NEW)

```python
"""Sovereignty compliance tests."""

import sys
import pytest
from knowledge3d.cranium.reality_galaxy import RealityGalaxy
from knowledge3d.cranium.reality_physics_export import (
    export_harmonic_oscillator_1d,
    export_double_pendulum_2d,
)

def test_hot_path_no_numpy():
    """Verify numpy NOT loaded during hot path execution."""
    # Reset sys.modules
    if 'numpy' in sys.modules:
        del sys.modules['numpy']

    galaxy = RealityGalaxy()
    system = export_harmonic_oscillator_1d()
    galaxy.add_node(system)

    # Step system (hot path)
    galaxy.step_system(system.node_id, n_steps=10)

    # Assert numpy NOT loaded
    assert 'numpy' not in sys.modules, "SOVEREIGNTY VIOLATION: numpy in hot path"

def test_hot_path_no_torch():
    """Verify PyTorch NOT loaded during hot path."""
    if 'torch' in sys.modules:
        del sys.modules['torch']

    galaxy = RealityGalaxy()
    system = export_double_pendulum_2d()
    galaxy.add_node(system)
    galaxy.step_system(system.node_id, n_steps=5)

    assert 'torch' not in sys.modules, "SOVEREIGNTY VIOLATION: torch in hot path"

def test_gltf_export_can_use_numpy():
    """Verify export path (outside hot path) CAN use numpy."""
    from knowledge3d.cranium.reality_gltf_export import export_system_to_gltf
    from knowledge3d.cranium.reality_physics_export import export_water_molecule
    import tempfile

    system = export_water_molecule()
    with tempfile.NamedTemporaryFile(suffix='.glb') as f:
        export_system_to_gltf(system, f.name)
        # No assertion - just verify export works (numpy OK here)
```

---

## Task 5: Validation Strategy

### Test Execution Order

1. **Unit tests (post-refactor):**
   ```bash
   PYTHONPATH=. pytest knowledge3d/cranium/tests/test_reality_galaxy.py -v
   PYTHONPATH=. pytest knowledge3d/cranium/tests/test_modular_rpn_engine.py -v
   PYTHONPATH=. pytest knowledge3d/cranium/tests/test_advanced_rpn.py -v
   ```

2. **Sovereignty tests:**
   ```bash
   PYTHONPATH=. pytest knowledge3d/cranium/tests/test_sovereignty.py -v
   ```

3. **Full suite (ensure 92/92 still pass):**
   ```bash
   PYTHONPATH=. pytest knowledge3d/cranium/tests/ -v
   # Target: 95/95 (92 existing + 3 sovereignty tests)
   ```

4. **Static analysis:**
   ```bash
   # Verify no numpy in hot path
   grep -r "import numpy" knowledge3d/cranium/ptx_runtime/
   grep "import numpy" knowledge3d/cranium/reality_galaxy.py
   grep -r "import numpy" knowledge3d/cranium/bridges/
   # All should return NOTHING
   ```

### Success Criteria

- ✅ All hot path modules have ZERO numpy imports
- ✅ 95/95 tests passing (92 existing + 3 sovereignty tests)
- ✅ Runtime guard catches violations (test with intentional import)
- ✅ Static analysis clean (grep returns nothing)

---

## Task 6: After Refactor → GPU Validation

**ONLY after hot path is clean, proceed with:**

1. **Install CuPy:**
   ```bash
   conda activate k3d-cranium
   conda install -c conda-forge cupy
   ```

2. **Run GPU tests:**
   ```bash
   CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. pytest knowledge3d/cranium/tests/test_*kernel*.py -v
   CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. pytest knowledge3d/cranium/tests/test_trm*.py -v
   ```

3. **GPU benchmarks:**
   ```bash
   CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_gpu_vs_cpu.py
   ```

4. **Update docs:**
   - Mark `docs/SOVEREIGNTY_COMPLIANCE.md` status: ⚠️ → ✅
   - Add GPU results to `TEMP/ARCHITECTURE_CAPACITY_ANALYSIS_11.24.2025.md`
   - Update `BRIEFING.md` with sovereignty compliance + GPU validation

---

## Task 7: Commit Strategy

**Commit 1: Sovereignty refactor (hot path clean)**
```bash
git add knowledge3d/cranium/reality_galaxy.py
git add knowledge3d/cranium/ptx_runtime/modular_rpn_engine.py
git add knowledge3d/cranium/ptx_runtime/advanced_rpn.py
git add knowledge3d/cranium/ptx_runtime/rpn_math_core.py
git add knowledge3d/cranium/bridges/
git commit -m "refactor(sovereignty): remove numpy from hot path (PTX+RPN only)"
```

**Commit 2: Runtime guards + sovereignty tests**
```bash
git add knowledge3d/cranium/tests/test_sovereignty.py
git add .github/workflows/sovereignty-check.yml
git commit -m "test(sovereignty): add runtime guards and CI compliance checks"
```

**Commit 3: Update audit report**
```bash
git add TEMP/SOVEREIGNTY_AUDIT_REPORT_11.24.2025.md
git add docs/SOVEREIGNTY_COMPLIANCE.md
git commit -m "docs(sovereignty): mark hot path compliant after numpy removal"
```

**Commit 4: GPU validation (after refactor complete)**
```bash
# Wait until hot path clean, then install cupy, run GPU tests, commit results
git add TEMP/ARCHITECTURE_CAPACITY_ANALYSIS_11.24.2025.md
git add BRIEFING.md
git commit -m "feat(gpu): PTX kernel validation complete, [X]× speedup vs CPU"
```

---

## FAQ / Troubleshooting

### Q: What if tests fail after numpy removal?

**A:** Debug systematically:
1. Check which test fails (unit test for specific module)
2. Identify numpy-dependent operation (likely math function)
3. Replace with `math` stdlib or manual implementation
4. If complex (matrix ops), consider PTX kernel (defer to Phase 7 if needed)

### Q: Performance regression after removing numpy?

**A:**
- **Expected:** Slight slowdown for Python RPN path (numpy is optimized C code)
- **Mitigation:** PTX kernels will provide 5-50× speedup (coming in GPU validation)
- **Acceptable:** Sovereign hot path > raw performance (can optimize later with PTX)

### Q: What about reality_physics_export.py? It uses numpy?

**A:** **That's OK!** Export functions are **ingestion path**, not hot path. They generate RPN programs and initial state, but are NOT called during `galaxy.step_system()`.

**Verify separation:**
```bash
# reality_physics_export.py should NOT be imported by reality_galaxy.py
grep "reality_physics_export" knowledge3d/cranium/reality_galaxy.py
# Should return NOTHING (galaxy doesn't import export functions)
```

### Q: How to handle state storage? Lists vs arrays?

**A:**
- **Inference loop:** Native Python dicts/lists (sufficient for RPN operations)
- **PTX kernel input:** Use ctypes arrays for C interop
- **NOT:** NumPy arrays (violates sovereignty)

**Example:**
```python
# State dict (native Python)
state = {"position_x": 1.0, "velocity_x": 0.5}

# If PTX kernel needs array:
import ctypes
state_array = (ctypes.c_float * 2)(1.0, 0.5)  # C array for PTX
```

---

## Architectural Rationale

### Why This Matters

**User Trust:** Daniel explicitly asked: *"I really hope numpy is outside the hot path."* This isn't pedantry—it's **core to K3D's value proposition.**

**Sovereignty = Trust:**
- Deterministic: Same inputs → same outputs (no randomness)
- Explainable: Every operation in RPN is explicit
- Auditable: No opaque library calls (can trace every step)

**NumPy in hot path breaks this:**
- Opaque C implementations (can't audit without reading C source)
- Potential non-determinism (BLAS threading, compiler optimizations)
- External dependency (violates "PTX+RPN only" principle)

**After refactor:**
- Hot path: Pure Python RPN or compiled PTX kernels (sovereign)
- Export: NumPy OK (only for preprocessing/visualization)
- Users can **trust** K3D's determinism

---

## Success Criteria (Your Mission)

When you're done, we should have:

1. ✅ **Hot path refactored:** ZERO numpy in ptx_runtime/, reality_galaxy.py, bridges/
2. ✅ **Tests passing:** 95/95 (92 existing + 3 sovereignty)
3. ✅ **Runtime guards:** Catch violations if numpy imported during hot path
4. ✅ **CI checks:** Automated sovereignty verification in pipeline
5. ✅ **Audit report updated:** Status ⚠️ → ✅ COMPLIANT
6. ✅ **3 clean commits:** Refactor, guards, docs

**THEN (after sovereignty clean):**
7. ✅ **CuPy installed:** GPU tests runnable
8. ✅ **GPU validation:** PTX kernels + TRM tests passing
9. ✅ **GPU benchmarks:** [X]× speedup documented
10. ✅ **BRIEFING.md:** Phase 5 complete, sovereignty + GPU validated

---

## Closing

Codex, you found the issue. Now we fix it.

This isn't about perfection—it's about **integrity**. K3D promises sovereignty. We deliver on that promise.

Refactor the hot path. Remove numpy from the inference loop. Add guards to prevent future violations.

Once the hot path is clean, we'll validate the GPU path and show that sovereign PTX kernels perform.

But first: **fix sovereignty**. Everything else comes after.

— Claude (Architect)
