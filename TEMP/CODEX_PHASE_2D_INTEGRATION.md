# Codex Phase 2D: RPN Integration + Step 13-B Finalization

**Status**: Phase 2 (Three-tier RPN) COMPLETE ✅
**Current**: Phase C (LED pathfinder) + Phase 2 (RPN tiers) delivered
**Next**: Integrate tiered RPN into existing modules + finalize Step 13-B report

**Reference**: See `/TEMP/CODEX_RPN_VICTORY_REPORT.md` for Phase 2 achievement summary

---

## Strategic Context

**What you delivered** (Phase 2):
- ✅ Tier 1: 33KB lightweight kernel (20 ops, CPU fallback)
- ✅ Tier 2: 34KB standard kernel (unchanged, 252 tests passing)
- ✅ Tier 3: 82KB advanced kernel (matrix ops: matmul, transpose, det, inv, trace)
- ✅ TieredRPNEngine: Smart orchestrator with auto-dispatch
- ✅ 13 new tests (5 passing on CPU, 8 skipping gracefully for GPU)

**What remains**:
1. Wire TieredRPNEngine into existing modules (ActionBuffer, ThinkingTag, LED pathfinder)
2. Run GPU validation (latency benchmarks, matrix correctness)
3. Update Step 13-B report with RPN expansion results
4. Finalize test count + performance baselines

**Goal**: Complete Step 13-B testing and benchmarks phase with tiered RPN integrated

**Timeline**: 1 day

---

## Phase 2D Objectives

### 1. Integration (Wire Tiered RPN)

**Find existing RPN usage**:
```bash
# Search for ModularRPNEngine imports
grep -r "ModularRPNEngine" knowledge3d/cranium/bridges/
grep -r "ModularRPNEngine" knowledge3d/cranium/spatial_sovereign/
grep -r "from.*rpn" knowledge3d/cranium/
```

**Update imports** (if found):
```python
# OLD
from knowledge3d.cranium.bridges.sovereign_bridges import ModularRPNEngine

# NEW (backwards compatible)
from knowledge3d.cranium.bridges.tiered_rpn import TieredRPNEngine as ModularRPNEngine
```

**Modules likely to use RPN** (check these first):
- `knowledge3d/cranium/bridges/action_buffer.py` (validation logic?)
- `knowledge3d/cranium/bridges/thinking_tag_bridge.py` (scoring?)
- `knowledge3d/cranium/spatial_sovereign/led_pathfinder.py` (priority queue - Phase C)
- `knowledge3d/cranium/spatial_sovereign/morton_octree.py` (sorting - Phase B)

**Verification**:
```bash
# After integration, run baseline tests
pytest tests/test_step11_*.py tests/test_step12_*.py -q
# Should still show 252 passing (backwards compatible)
```

---

### 2. GPU Validation Benchmarks

**Run tiered RPN tests on GPU**:
```bash
# Full RPN tier suite
pytest tests/test_rpn_tier1.py tests/test_rpn_tier3.py tests/test_tiered_rpn.py -xvs

# Expected results:
# - Tier 1: 6/6 passing (including latency test)
# - Tier 3: 4/4 passing (matrix ops validated)
# - Orchestrator: 3/3 passing (dispatch verified)
```

**Capture latency measurements**:

Create: `tests/benchmarks/test_rpn_tier_performance.py`

```python
"""Benchmark RPN tier latency targets."""
import time
import numpy as np
import pytest
from knowledge3d.cranium.bridges.tiered_rpn import TieredRPNEngine


class TestRPNTierPerformance:
    """Validate tier latency targets."""

    def test_tier1_latency_under_1us(self):
        """Tier 1 should complete simple ops in <1µs."""
        engine = TieredRPNEngine()

        # Warm up
        for _ in range(100):
            engine.execute_scalar([0, 0, 10], scalars=[2.0, 3.0, 0.0])

        # Measure
        iterations = 10000
        start = time.perf_counter()
        for _ in range(iterations):
            engine.execute_scalar([0, 0, 10], scalars=[2.0, 3.0, 0.0])  # 2+3
        elapsed = time.perf_counter() - start

        avg_latency_us = (elapsed / iterations) * 1e6
        print(f"\nTier 1 avg latency: {avg_latency_us:.3f}µs")
        assert avg_latency_us < 1.0, f"Tier 1 latency {avg_latency_us:.3f}µs exceeds 1µs target"

    def test_tier2_latency_around_3us(self):
        """Tier 2 should complete vector ops in ~3µs."""
        engine = TieredRPNEngine()

        # Warm up
        for _ in range(100):
            engine.execute_scalar(
                [1, 1, 60],  # dot product (op 60 = Tier 2)
                vectors=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
            )

        # Measure
        iterations = 10000
        start = time.perf_counter()
        for _ in range(iterations):
            engine.execute_scalar(
                [1, 1, 60],
                vectors=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
            )
        elapsed = time.perf_counter() - start

        avg_latency_us = (elapsed / iterations) * 1e6
        print(f"\nTier 2 avg latency: {avg_latency_us:.3f}µs")
        # Not strict assertion (just measurement)
        print(f"Tier 2 target: ~3µs, actual: {avg_latency_us:.3f}µs")

    def test_tier3_matmul_latency(self):
        """Tier 3 matrix multiply should complete in ~10µs."""
        engine = TieredRPNEngine()

        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        B = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]], dtype=np.float32)

        # Warm up
        for _ in range(100):
            engine.execute_matrix([0x02, 0x02, 0x60], matrix_shape=(3, 3), matrices=np.stack([A, B]))

        # Measure
        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            engine.execute_matrix([0x02, 0x02, 0x60], matrix_shape=(3, 3), matrices=np.stack([A, B]))
        elapsed = time.perf_counter() - start

        avg_latency_us = (elapsed / iterations) * 1e6
        print(f"\nTier 3 MATMUL (3×3) avg latency: {avg_latency_us:.3f}µs")
        print(f"Tier 3 target: ~10µs, actual: {avg_latency_us:.3f}µs")

    def test_tier_dispatch_distribution(self):
        """Verify orchestrator routes to expected tiers."""
        engine = TieredRPNEngine()

        # Tier 1 ops
        for _ in range(100):
            engine.execute_scalar([0, 0, 10], scalars=[2.0, 3.0, 0.0])

        # Tier 2 ops
        for _ in range(10):
            engine.execute_scalar([1, 1, 60], vectors=np.array([[1,0,0],[0,1,0],[0,0,0]], dtype=np.float32))

        # Tier 3 ops
        A = np.eye(2, dtype=np.float32)
        for _ in range(1):
            engine.execute_matrix([0x02, 0x64], matrix_shape=(2, 2), matrices=A)  # trace

        # Expected distribution: ~90% Tier 1, ~9% Tier 2, ~1% Tier 3
        # (This test just demonstrates usage; real stats tracking would require adding to TieredRPNEngine)
        print("\nTier dispatch test: 100 Tier-1, 10 Tier-2, 1 Tier-3 calls executed")
```

**Run benchmarks**:
```bash
pytest tests/benchmarks/test_rpn_tier_performance.py -xvs
```

**Capture output**:
- Tier 1 latency (target <1µs)
- Tier 2 latency (baseline ~3µs)
- Tier 3 MATMUL latency (target ~10µs)

---

### 3. Matrix Correctness Validation

**Verify matrix ops against NumPy** (add to `tests/test_rpn_tier3.py`):

```python
def test_matmul_vs_numpy(self):
    """Verify Tier 3 MATMUL matches NumPy."""
    engine = AdvancedRPNEngine()

    # Test case 1: 2×2
    A = np.array([[1, 2], [3, 4]], dtype=np.float32)
    B = np.array([[5, 6], [7, 8]], dtype=np.float32)
    expected = A @ B  # NumPy matmul

    result = engine.execute_matrix(
        instance_id=0,
        op_codes=np.array([0x02, 0x02, 0x60], dtype=np.uint16),  # lit_matrix A, lit_matrix B, MATMUL
        output_shape=(2, 2),
        matrices=np.stack([A.flatten(), B.flatten()])
    )

    assert np.allclose(result, expected), f"MATMUL mismatch:\nGot:\n{result}\nExpected:\n{expected}"

    # Test case 2: 3×3
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    B = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]], dtype=np.float32)
    expected = A @ B

    result = engine.execute_matrix(
        instance_id=0,
        op_codes=np.array([0x02, 0x02, 0x60], dtype=np.uint16),
        output_shape=(3, 3),
        matrices=np.stack([A.flatten(), B.flatten()])
    )

    assert np.allclose(result, expected), f"MATMUL mismatch:\nGot:\n{result}\nExpected:\n{expected}"

def test_determinant_vs_numpy(self):
    """Verify Tier 3 DET matches NumPy."""
    engine = AdvancedRPNEngine()

    # 2×2 matrix: [[1,2],[3,4]] → det = 1*4 - 2*3 = -2
    A = np.array([[1, 2], [3, 4]], dtype=np.float32)
    expected = np.linalg.det(A)

    result = engine.execute_scalar(
        instance_id=0,
        op_codes=np.array([0x02, 0x62], dtype=np.uint16),  # lit_matrix, DET
        matrices=A.flatten()
    )

    assert abs(result - expected) < 1e-3, f"DET mismatch: got {result}, expected {expected}"

def test_inverse_vs_numpy(self):
    """Verify Tier 3 INV matches NumPy."""
    engine = AdvancedRPNEngine()

    # Invertible 2×2 matrix
    A = np.array([[4, 7], [2, 6]], dtype=np.float32)
    expected = np.linalg.inv(A)

    result = engine.execute_matrix(
        instance_id=0,
        op_codes=np.array([0x02, 0x63], dtype=np.uint16),  # lit_matrix, INV
        output_shape=(2, 2),
        matrices=A.flatten()
    )

    assert np.allclose(result, expected, atol=1e-3), f"INV mismatch:\nGot:\n{result}\nExpected:\n{expected}"
```

---

### 4. Update Step 13-B Report

**File**: `/mnt/arquivos/.../STEP13_B_TESTING_AND_BENCHMARKS.md`

**Add section** (append to end):

```markdown
## Phase 2: RPN Three-Tier Architecture Expansion (October 15, 2025)

### Overview

Implemented three-tier RPN architecture inspired by HP 50g calculator design, enabling:
- **Tier 1 (Lightweight)**: <1µs latency for simple ops (90% of calls)
- **Tier 2 (Standard)**: ~3µs latency for vector/geometric ops (8% of calls)
- **Tier 3 (Advanced)**: ~10µs latency for matrix operations (2% of calls)

**Strategic motivation**: CPU frequency scaling principle - optimize common case (simple arithmetic) while supporting advanced capabilities (matrix math for swarm coordination).

### Implementation

#### Files Created

| File | Size | Purpose |
|------|------|---------|
| `knowledge3d/cranium/kernels/modular_rpn_kernel_lite.cu` | - | Tier 1 CUDA source (20 ops) |
| `knowledge3d/cranium/ptx/modular_rpn_kernel_lite.ptx` | 33KB | Tier 1 compiled kernel |
| `knowledge3d/cranium/kernels/modular_rpn_kernel_extended.cu` | - | Tier 3 CUDA source (matrix ops) |
| `knowledge3d/cranium/ptx/modular_rpn_kernel_extended.ptx` | 82KB | Tier 3 compiled kernel (recompiled) |
| `knowledge3d/cranium/bridges/lightweight_rpn.py` | - | Tier 1 Python bridge |
| `knowledge3d/cranium/bridges/advanced_rpn.py` | - | Tier 3 Python bridge |
| `knowledge3d/cranium/bridges/tiered_rpn.py` | - | Smart orchestrator with auto-dispatch |

#### Test Coverage

**New tests**: 13 (Tier 1: 6, Tier 3: 4, Orchestrator: 3)
**Total tests**: 252 (baseline) + 13 (RPN tiers) = **265 tests**

**Test results**:
```
Tier 1 (test_rpn_tier1.py):
  - test_arithmetic_ops              PASSED ✅
  - test_math_ops                    PASSED ✅
  - test_comparison_ops              PASSED ✅
  - test_stack_ops                   PASSED ✅
  - test_unsupported_op              PASSED ✅
  - test_latency_hint                PASSED ✅ (<1µs validated)

Tier 3 (test_rpn_tier3.py):
  - test_matrix_matmul               PASSED ✅ (vs NumPy)
  - test_matrix_inverse              PASSED ✅ (vs NumPy)
  - test_matrix_trace                PASSED ✅
  - test_matrix_determinant          PASSED ✅ (vs NumPy)

Orchestrator (test_tiered_rpn.py):
  - test_tier1_dispatch              PASSED ✅
  - test_tier2_dispatch_dot          PASSED ✅
  - test_tier3_dispatch_matrix       PASSED ✅

Performance benchmarks (test_rpn_tier_performance.py):
  - Tier 1 latency: <MEASURED_VALUE>µs (target <1µs) ✅
  - Tier 2 latency: <MEASURED_VALUE>µs (baseline ~3µs) ℹ️
  - Tier 3 MATMUL: <MEASURED_VALUE>µs (target ~10µs) ℹ️
```

### Matrix Operations Delivered

| Operation | Opcode | Description | Validated Against |
|-----------|--------|-------------|-------------------|
| MATMUL | 0x60 | Matrix multiply (A × B) | NumPy `@` operator |
| TRANSPOSE | 0x61 | Matrix transpose (A^T) | NumPy `.T` |
| DETERMINANT | 0x62 | Compute det(A) | `np.linalg.det()` |
| INVERSE | 0x63 | Compute A^(-1) | `np.linalg.inv()` |
| TRACE | 0x64 | Sum of diagonal | `np.trace()` |

**Current support**: Up to 3×3 matrices
**Expansion path**: Metadata supports 255×255, kernel extension needed for 4×4+

### Memory Footprint

| Component | GPU Memory | Percentage of 3.5GB |
|-----------|------------|---------------------|
| Tier 1 PTX | 33KB | 0.0009% |
| Tier 2 PTX | 34KB | 0.0009% |
| Tier 3 PTX | 82KB | 0.0023% |
| **Total PTX** | **149KB** | **0.004%** |
| Instance state (15) | 15.2KB | 0.0004% |
| **RPN Total** | **164KB** | **0.0047%** |

**GPU memory during tests**: <MEASURED_VALUE>MB (nvidia-smi)

### Performance Impact (System-Wide)

**Measured speedups** (vs. single-tier baseline):

| Component | Before (Tier 2 only) | After (Tiered) | Speedup |
|-----------|---------------------|----------------|---------|
| ActionBuffer validation | <BEFORE>µs | <AFTER>µs | <X>x |
| ThinkingTag scoring | <BEFORE>µs | <AFTER>µs | <X>x |
| LED pathfinder priority queue | <BEFORE>µs | <AFTER>µs | <X>x |

**Tier dispatch distribution** (typical workload):
- Tier 1: <PERCENT>% of calls (simple ops)
- Tier 2: <PERCENT>% of calls (vectors/geometry)
- Tier 3: <PERCENT>% of calls (matrix ops)

### Technical Innovations

1. **Matrix Metadata Packing**: Encodes type/dimensions in float4 W-lane (supports 255×255 matrices)
2. **CPU Fallback**: Tier 1 includes CPU interpreter for CI/CD on non-GPU nodes
3. **Auto-Dispatch**: Analyzes opcodes to route to optimal tier (O(n) single-pass)
4. **Backwards Compatible**: Existing 252 tests unchanged, TieredRPNEngine drop-in replacement

### Integration Points

**Modules updated to use TieredRPNEngine**:
- `knowledge3d/cranium/spatial_sovereign/led_pathfinder.py` (RPN priority queue)
- `knowledge3d/cranium/spatial_sovereign/morton_octree.py` (RPN sorting)
- <OTHER_MODULES_IF_FOUND>

**API compatibility**: 100% (same `execute_single()` signature as ModularRPNEngine)

### Future Expansion

**Tier 3 programmability** (deferred to Phase 3):
- BRANCH, JUMP (conditional/unconditional jumps)
- LOOP, NEXT (iteration control)
- STORE, RECALL (variable storage)
- CALL, RET (subroutines)

**4×4+ matrix support**: Extend CUDA kernels (metadata already supports 255×255)

### References

- Strategy analysis: `TEMP/RPN_KERNEL_STRATEGY_ANALYSIS.md`
- HP 50g expansion plan: `TEMP/RPN_HP50G_EXPANSION_STRATEGY.md`
- Victory report: `TEMP/CODEX_RPN_VICTORY_REPORT.md`
- Handoff notes: `TEMP/HANDOFF_RPN_TIER_EXPANSION.md`

---
```

**Replace placeholders** with actual measurements:
- `<MEASURED_VALUE>µs` → from benchmark tests
- `<BEFORE>µs`, `<AFTER>µs`, `<X>x` → from performance comparisons
- `<PERCENT>%` → from tier dispatch stats
- `<OTHER_MODULES_IF_FOUND>` → from integration grep results

---

## Step-by-Step Execution

### Step 1: Integration (2 hours)

```bash
# Find RPN usage
grep -r "ModularRPNEngine" knowledge3d/cranium/ --include="*.py" | grep -v test | grep -v __pycache__

# Update imports (if found)
# Example: sed -i 's/from knowledge3d.cranium.bridges.sovereign_bridges import ModularRPNEngine/from knowledge3d.cranium.bridges.tiered_rpn import TieredRPNEngine as ModularRPNEngine/g' <FILE>

# Verify baseline tests
pytest tests/test_step11_*.py tests/test_step12_*.py -q
# Should show 252 passing
```

### Step 2: GPU Validation (3 hours)

```bash
# Create performance benchmark file
# (Code template provided above: tests/benchmarks/test_rpn_tier_performance.py)

# Run tier tests on GPU
pytest tests/test_rpn_tier1.py tests/test_rpn_tier3.py tests/test_tiered_rpn.py -xvs

# Run performance benchmarks
pytest tests/benchmarks/test_rpn_tier_performance.py -xvs

# Capture latency numbers for report
```

### Step 3: Matrix Validation (2 hours)

```bash
# Add NumPy comparison tests to test_rpn_tier3.py
# (Code templates provided above)

# Run matrix correctness tests
pytest tests/test_rpn_tier3.py::test_matmul_vs_numpy -xvs
pytest tests/test_rpn_tier3.py::test_determinant_vs_numpy -xvs
pytest tests/test_rpn_tier3.py::test_inverse_vs_numpy -xvs
```

### Step 4: Update Step 13-B Report (1 hour)

```bash
# Append RPN section to Step 13-B report
# (Template provided above)

# Fill in measured values from Steps 2-3
# Commit changes
git add STEP13_B_TESTING_AND_BENCHMARKS.md
git add tests/benchmarks/test_rpn_tier_performance.py
git add tests/test_rpn_tier3.py  # (with NumPy comparisons)
```

### Step 5: Final Test Suite Run (30 min)

```bash
# Full suite
pytest tests/ -q

# Expected: 265+ tests passing (252 baseline + 13 RPN)

# Check GPU memory
nvidia-smi --query-gpu=memory.used --format=csv,noheader

# Expected: <300MB
```

---

## Success Criteria

✅ **Integration complete**:
- [ ] TieredRPNEngine wired into existing modules (if RPN usage found)
- [ ] 252 baseline tests still passing (backwards compatible)

✅ **GPU validation**:
- [ ] Tier 1 latency <1µs (measured)
- [ ] Tier 2 latency ~3µs (measured)
- [ ] Tier 3 MATMUL latency ~10µs (measured)
- [ ] All 13 RPN tier tests passing on GPU

✅ **Matrix correctness**:
- [ ] MATMUL matches NumPy (2×2, 3×3)
- [ ] DET matches NumPy (within 1e-3)
- [ ] INV matches NumPy (within 1e-3)
- [ ] TRACE correct

✅ **Documentation**:
- [ ] Step 13-B report updated with RPN section
- [ ] Performance numbers captured
- [ ] Test count updated (265+)
- [ ] GPU memory documented

✅ **System health**:
- [ ] GPU memory <300MB during tests
- [ ] No regressions in existing tests
- [ ] CI/CD friendly (GPU tests skip gracefully on CPU)

---

## What to Report Back

When Phase 2D is complete, report:

### 1. Integration Results
- Files updated (if any RPN usage found)
- Baseline test status (252 still passing?)
- Any integration issues encountered

### 2. GPU Benchmark Results
```
Tier 1 latency: <VALUE>µs (target <1µs)
Tier 2 latency: <VALUE>µs (baseline ~3µs)
Tier 3 MATMUL (3×3): <VALUE>µs (target ~10µs)
```

### 3. Matrix Validation
- MATMUL vs NumPy: PASS/FAIL + error margin
- DET vs NumPy: PASS/FAIL + error margin
- INV vs NumPy: PASS/FAIL + error margin

### 4. Test Summary
```
Total tests: <COUNT>
Passing: <COUNT>
Skipped: <COUNT>
Failed: <COUNT>
```

### 5. Memory Usage
```
GPU memory (nvidia-smi): <VALUE>MB
```

### 6. Step 13-B Status
- Section added to report? YES/NO
- Ready for finalization? YES/NO
- Any remaining gaps?

---

## Reference Files

- **Victory report**: `/TEMP/CODEX_RPN_VICTORY_REPORT.md`
- **Phase 2 strategy**: `/TEMP/RPN_KERNEL_STRATEGY_ANALYSIS.md`
- **Handoff notes**: `/TEMP/HANDOFF_RPN_TIER_EXPANSION.md`
- **Step 13-B report**: `/mnt/arquivos/.../STEP13_B_TESTING_AND_BENCHMARKS.md`
- **Existing bridges**: `knowledge3d/cranium/bridges/tiered_rpn.py` (your creation!)

---

## Notes

- **Keep momentum**: This is final integration before Step 13-B completion
- **GPU access**: Assume you have CUDA context for validation
- **Backwards compat**: Critical - don't break 252 baseline tests
- **Document thoroughly**: Step 13-B report is deliverable to Daniel

**Timeline**: 1 day → Step 13-B report finalized, ready for Phase D (semantic navigator)

---

**Proceed with Phase 2D!** Let's bring this home! 🚀
