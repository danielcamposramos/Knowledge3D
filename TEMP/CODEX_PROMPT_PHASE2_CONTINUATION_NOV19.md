# Phase 2 RPN Sovereignty - Continuation Prompt for Codex

**Date:** 2025-11-19
**Session:** Continuation from Phase 2 foundation
**Status:** Excellent progress! Claude has wired the validation infrastructure and tests. Ready for your enhancements.

---

## Executive Summary

### What Codex Has Done (Excellent Foundation!)
✅ Created `rpn_math_core.py` - Tier-3 math helper wrapping pointer-literal programs
✅ Reworked `SelfUpdatingAdapter` to allocate persistent GPU buffers
✅ Implemented `apply_gradient_rpn()` routing all operations through RPN math core
✅ Gradients, A, and B now live on device during the step
✅ Norm-clipping via OP_VEC_L2_NORM
✅ Updates via RPN fill → vector_mul → vec_add3
✅ Shadow updates piggyback on same sovereign path
✅ NumPy fallback kept for compatibility

### What Claude Has Done (Building on Your Work!)
✅ Implemented ternary validation gate (`_ternary_gate()` in `trm_adapters.py`)
✅ Enhanced `validate_and_commit()` with TRUE/FALSE/UNKNOWN decision logic
✅ Created comprehensive regression test suite (`tests/test_rpn_sovereignty_phase2.py`)
✅ Verified specialist already wires to RPN path (no changes needed!)

### Current Architecture Status

**Sovereign Training Path (100% GPU-native):**
```
ProceduralDrawingSpecialist.train_on_batch()
  → AdaptiveSwarm.train_specialist_contrastive()
    → SelfUpdatingAdapter.apply_gradient()
      → apply_gradient_rpn() [YOUR CODE]
        → RPNMathCore [YOUR CODE]
          → PTX kernels (OP_VEC_L2_NORM, OP_MATMUL_SMALL, etc.)
```

**Validation Path:**
```
SelfUpdatingAdapter.validate_and_commit()
  → _ternary_gate() [CLAUDE'S CODE]
    → TRUE: commit shadow → primary
    → FALSE: reject (excessive degradation)
    → UNKNOWN: defer (insufficient evidence)
```

---

## Current Bottlenecks & Optimization Opportunities

### 1. NumPy Fallback Still Active
**Location:** `trm_adapters.py` lines 142-145, 188-190, 421-431

**Current Code:**
```python
def apply_gradient(self, gradient: np.ndarray, lr: float = 0.001):
    if self._ensure_math_core():
        self.apply_gradient_rpn(gradient, lr)
        return
    self._apply_gradient_cpu(gradient, lr)  # ← FALLBACK
```

**Your Mission (if you choose to accept):**
- Remove CPU fallback entirely OR
- Make fallback explicit failure (raise RuntimeError) OR
- Add config flag to control behavior

**Why:** "We fix or we fix" - no CPU fallbacks in sovereign path!

### 2. Matrix Transpose Allocations
**Location:** `trm_adapters.py` lines 213-228

**Current Code:**
```python
# Compute grad_A = gradient @ B.T
b_t = np.ascontiguousarray(self.B.T, dtype=np.float32)
d_bt = RPNMathCore.to_device(b_t)  # ← HOST COPY + UPLOAD
try:
    bt_tensor = DeviceTensor(d_bt, dims, rank)
    self._math_core.matmul(buffers.grad_a, buffers.gradient, bt_tensor)
finally:
    RPNMathCore.free(d_bt)  # ← IMMEDIATE FREE
```

**Optimization Opportunity:**
- Pre-allocate B.T buffer in `_ensure_device_buffers()`
- Reuse across gradient steps (avoid repeated upload)
- Add transpose operation to RPN opcodes (future enhancement)

**Expected Speedup:** 15-20% (eliminates 2 CPU transposes + 2 uploads per step)

### 3. Scalar Operations Still NumPy
**Location:** `trm_adapters.py` lines 233-242

**Current Code:**
```python
# Update A: A -= lr * grad_A
self._math_core.fill(buffers.a_scale, -lr)  # ← FILL ENTIRE BUFFER WITH SCALAR
self._math_core.vector_multiply(a_vec, buffers.a_scale)
self._math_core.vec_add3(a_dest, a_dest, a_vec, buffers.a_zero)
```

**Optimization Opportunity:**
- Add `OP_SCALAR_MUL` opcode: `stack[i] *= scalar`
- Avoid filling entire buffer for scalar multiply
- Example: `self.rpn_engine.scalar_mul(a_vec, -lr, output=a_vec)`

**Expected Speedup:** 5-10% (reduces memory bandwidth)

### 4. Gradient Norm Re-computation
**Location:** `trm_adapters.py` lines 202-210

**Current Code:**
```python
grad_vec = self._vector_view(buffers.gradient)
grad_norm = self._math_core.vector_norm(grad_vec)  # ← COMPUTE ONCE

if clip > 0.0 and grad_norm > clip:
    scale = clip / max(grad_norm, 1e-6)
    self._math_core.fill(buffers.grad_scale, scale)
    self._math_core.vector_multiply(grad_vec, buffers.grad_scale)
    grad_norm = clip  # ← UPDATE HOST VARIABLE (no re-computation)
```

**Status:** Already optimal! Nice work.

---

## Ternary Validation Gate (Claude's Implementation)

### Decision Logic
```python
def _ternary_gate(self, baseline_perf: float, shadow_perf: float) -> str:
    """
    Ternary validation gate for sovereign training.

    Decision logic:
    - TRUE: Shadow significantly better (improvement >= min_improvement)
    - FALSE: Shadow significantly worse (degradation > max_degradation)
    - UNKNOWN: Marginal difference (accumulate more evidence)
    """
    improvement = shadow_perf - baseline_perf
    degradation = baseline_perf - shadow_perf

    # TRUE: Clear improvement
    if improvement >= self.config.min_improvement:
        return "TRUE"

    # FALSE: Excessive degradation
    elif degradation > self.config.max_degradation:
        return "FALSE"

    # UNKNOWN: Marginal difference (neither clearly better nor worse)
    else:
        return "UNKNOWN"
```

### Enhanced Metrics
All validation decisions now logged with `'decision': 'TRUE'|'FALSE'|'UNKNOWN'` in performance_history.

### Performance Impact
- Zero overhead (pure Python logic, no GPU calls)
- Same computational cost as binary gate
- Richer debugging information

---

## Regression Test Suite (Claude's Implementation)

### Test Coverage

**File:** `tests/test_rpn_sovereignty_phase2.py`

**Tests Implemented:**

1. **`test_rpn_vs_cpu_gradient_update()`**
   - Validates RPN and CPU paths produce equivalent results
   - Tolerance: 1e-4 (floating-point precision)
   - Status: Ready to run

2. **`test_rpn_shadow_updates()`**
   - Verifies shadow updates route through RPN
   - Validates primary weights unchanged
   - Status: Ready to run

3. **`test_ternary_validation_gate()`**
   - Tests TRUE/FALSE/UNKNOWN decision logic
   - Validates thresholds (min_improvement, max_degradation)
   - Status: Ready to run

4. **`test_rpn_math_core_operations()`**
   - Tests vector norm, fill, multiply, matmul
   - Validates Tier-3 RPN operations
   - Status: Ready to run

5. **`test_gradient_norm_clipping()`**
   - Validates gradient clipping in RPN path
   - Checks clipped norm matches config.gradient_clip
   - Status: Ready to run

6. **`test_validate_and_commit_decisions()`**
   - Integration test for validate_and_commit()
   - Tests TRUE → commit, FALSE → reject, UNKNOWN → defer
   - Status: Ready to run

7. **`test_rpn_speedup()` (benchmark)**
   - Performance benchmark vs CPU
   - Expected: RPN faster for dims >= 256
   - Status: Ready to run

### Running Tests

```bash
# All Phase 2 tests
pytest tests/test_rpn_sovereignty_phase2.py -v -s

# Specific test
pytest tests/test_rpn_sovereignty_phase2.py::TestRPNSovereignty::test_rpn_vs_cpu_gradient_update -v -s

# Performance benchmarks only
pytest tests/test_rpn_sovereignty_phase2.py -v -s -m benchmark
```

---

## What You Can Do Next (Codex's Autonomy)

### Priority 1: Remove NumPy Fallback (High Impact)
**Goal:** 100% sovereign training (no CPU fallback)

**Option A - Hard Fail:**
```python
def apply_gradient(self, gradient: np.ndarray, lr: float = 0.001):
    if not self._ensure_math_core():
        raise RuntimeError(
            f"[{self.specialist_name}] RPN math core required for sovereign training. "
            "GPU not available or initialization failed."
        )
    self.apply_gradient_rpn(gradient, lr)
```

**Option B - Config Flag:**
```python
@dataclass
class AdapterConfig:
    # ...
    require_gpu: bool = True  # Fail if GPU unavailable

def apply_gradient(self, gradient: np.ndarray, lr: float = 0.001):
    if self._ensure_math_core():
        self.apply_gradient_rpn(gradient, lr)
        return

    if self.config.require_gpu:
        raise RuntimeError("GPU required but unavailable")

    # Fallback only if explicitly allowed
    self._apply_gradient_cpu(gradient, lr)
```

**Recommendation:** Option B (graceful degradation with explicit control)

### Priority 2: Optimize Matrix Transpose (Medium Impact)
**Goal:** Eliminate repeated B.T and A.T allocations

**Current Bottleneck:**
```python
# grad_A = gradient @ B.T
b_t = np.ascontiguousarray(self.B.T, dtype=np.float32)  # ← CPU TRANSPOSE
d_bt = RPNMathCore.to_device(b_t)                        # ← UPLOAD
self._math_core.matmul(buffers.grad_a, buffers.gradient, bt_tensor)
RPNMathCore.free(d_bt)                                   # ← FREE
```

**Optimization:**
```python
@dataclass
class AdapterDeviceBuffers:
    # ...
    B_transposed: DeviceTensor  # Pre-allocated B.T buffer
    A_transposed: DeviceTensor  # Pre-allocated A.T buffer

def _ensure_device_buffers(self):
    # ...
    B_T_dev = DeviceTensor(alloc(b_len), dims, rank)  # B.T storage
    A_T_dev = DeviceTensor(alloc(a_len), rank, dims)  # A.T storage

def apply_gradient_rpn(self, gradient, lr):
    # Upload B.T once (instead of every step)
    b_t_host = np.ascontiguousarray(self.B.T, dtype=np.float32)
    RPNMathCore.copy_to_device(b_t_host, buffers.B_transposed.ptr)

    # Matmul with pre-allocated buffer
    self._math_core.matmul(buffers.grad_a, buffers.gradient, buffers.B_transposed)

    # Same for A.T
    # ...
```

**Expected Speedup:** 15-20%

### Priority 3: Add Scalar Multiply Opcode (Low Impact)
**Goal:** Reduce memory bandwidth for scalar operations

**Current Approach:**
```python
self._math_core.fill(buffers.a_scale, -lr)  # Fill entire buffer (wasteful)
self._math_core.vector_multiply(a_vec, buffers.a_scale)
```

**Optimized Approach:**
```python
# Add to rpn_opcodes.py
OP_SCALAR_MUL_F32 = 0xYY  # New opcode

# Add to rpn_math_core.py
def scalar_multiply(self, tensor: DeviceTensor, scalar: float) -> None:
    """Element-wise multiply by scalar (GPU-native)."""
    op_codes = [ropc.OP_POINTER_LITERAL, ropc.OP_LITERAL_SCALAR, ropc.OP_SCALAR_MUL_F32]
    scalars = _encode_pointer(...) + [scalar]
    self._exec(op_codes, scalars)

# Use in trm_adapters.py
self._math_core.scalar_multiply(a_vec, -lr)
self._math_core.vec_add3(a_dest, a_dest, a_vec, buffers.a_zero)
```

**Expected Speedup:** 5-10%

### Priority 4: Run Regression Tests (Critical Validation)
**Goal:** Ensure RPN path produces correct results

```bash
# Navigate to repo root
cd "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D"

# Activate environment
conda activate k3d-cranium

# Run all Phase 2 tests
env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. pytest tests/test_rpn_sovereignty_phase2.py -v -s

# If tests fail, analyze and fix
# If tests pass, document results and proceed to optimizations
```

**What to Watch For:**
- `test_rpn_vs_cpu_gradient_update` should pass with diff < 1e-4
- `test_ternary_validation_gate` should validate all 3 cases
- `test_rpn_speedup` should show performance improvement

### Priority 5: Update ProceduralCompiler (Future Enhancement)
**Goal:** Improve compression ratio from 0.9:1 to 69:1

**Status:** Deferred to Phase 2.6 (not blocking)

**Ideas for Exploration:**
- Prototype-based delta encoding (current approach)
- Hierarchical compression (coarse → medium → fine)
- Entropy coding (Huffman/arithmetic)
- Vector quantization (learned codebooks)

**Note:** This is NOT blocking for RPN sovereignty. Focus on Priorities 1-4 first.

---

## Testing Strategy

### Phase 1: Validate Equivalence
```bash
# Run regression tests to ensure RPN == CPU
pytest tests/test_rpn_sovereignty_phase2.py::TestRPNSovereignty::test_rpn_vs_cpu_gradient_update -v -s
```

**Success Criteria:**
- A matrix diff < 1e-4
- B matrix diff < 1e-4

### Phase 2: Validate Shadow Updates
```bash
pytest tests/test_rpn_sovereignty_phase2.py::TestRPNSovereignty::test_rpn_shadow_updates -v -s
```

**Success Criteria:**
- Primary unchanged
- Shadow updated

### Phase 3: Validate Ternary Gate
```bash
pytest tests/test_rpn_sovereignty_phase2.py::TestRPNSovereignty::test_ternary_validation_gate -v -s
```

**Success Criteria:**
- +2% improvement → TRUE
- -10% degradation → FALSE
- +0.05% marginal → UNKNOWN

### Phase 4: Integration Test
```bash
# Run full atomic training with RPN path
env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/test_atomic_formation_limited.py
```

**Success Criteria:**
- Training completes without errors
- Atomic units committed successfully
- Performance metrics logged

### Phase 5: Performance Benchmark
```bash
pytest tests/test_rpn_sovereignty_phase2.py::TestRPNPerformance::test_rpn_speedup -v -s --benchmark-only
```

**Expected Results:**
- RPN faster than CPU for dims >= 256
- Speedup increases with dimension (larger matrices = more benefit)

---

## Performance Expectations

### Current Status (Phase 2.1 Complete)
```
Total training time: ~2 min
├─ Python overhead: ~78% (unavoidable)
├─ NumPy operations: ~22% (TARGET FOR PHASE 2)
└─ GPU RPN execution: <1% (already sovereign)
```

### After Full Phase 2 (All Optimizations)
```
Total training time: ~1.6 min (19% speedup)
├─ Python overhead: ~78% (unchanged)
├─ GPU RPN operations: ~22% (100% sovereign!)
└─ CPU fallbacks: 0% (eliminated!)
```

### Per-Operation Speedup Estimates
| Optimization | Current | After | Speedup |
|--------------|---------|-------|---------|
| Gradient apply | NumPy | RPN | 1.5-2× |
| Matrix transpose | CPU+upload | Cached | 1.2× |
| Scalar multiply | Fill+mul | Direct | 1.1× |
| **Combined** | - | - | **1.19×** |

---

## Code Examples for Reference

### Current RPN Gradient Path (Your Code)
```python
def apply_gradient_rpn(self, gradient: np.ndarray, lr: float = 0.001) -> float:
    """Sovereign RPN-based gradient application."""
    buffers = self._ensure_device_buffers()
    if buffers is None or self._math_core is None:
        self._apply_gradient_cpu(gradient, lr)  # ← FALLBACK
        return float(np.linalg.norm(gradient))

    # Upload gradient to GPU
    grad_host = np.ascontiguousarray(gradient, dtype=np.float32)
    RPNMathCore.copy_to_device(grad_host, buffers.gradient.ptr)
    RPNMathCore.copy_to_device(self.A, buffers.A.ptr)
    RPNMathCore.copy_to_device(self.B, buffers.B.ptr)

    # Norm clipping
    grad_vec = self._vector_view(buffers.gradient)
    grad_norm = self._math_core.vector_norm(grad_vec)

    if clip > 0.0 and grad_norm > clip:
        scale = clip / max(grad_norm, 1e-6)
        self._math_core.fill(buffers.grad_scale, scale)
        self._math_core.vector_multiply(grad_vec, buffers.grad_scale)
        grad_norm = clip

    # Compute grad_A = gradient @ B.T
    b_t = np.ascontiguousarray(self.B.T, dtype=np.float32)
    d_bt = RPNMathCore.to_device(b_t)
    try:
        bt_tensor = DeviceTensor(d_bt, dims, rank)
        self._math_core.matmul(buffers.grad_a, buffers.gradient, bt_tensor)
    finally:
        RPNMathCore.free(d_bt)

    # Compute grad_B = A.T @ gradient
    a_t = np.ascontiguousarray(self.A.T, dtype=np.float32)
    d_at = RPNMathCore.to_device(a_t)
    try:
        at_tensor = DeviceTensor(d_at, rank, dims)
        self._math_core.matmul(buffers.grad_b, at_tensor, buffers.gradient)
    finally:
        RPNMathCore.free(d_at)

    # Update A: A -= lr * grad_A
    a_vec = self._vector_view(buffers.grad_a)
    a_dest = self._vector_view(buffers.A)
    self._math_core.fill(buffers.a_scale, -lr)
    self._math_core.vector_multiply(a_vec, buffers.a_scale)
    self._math_core.vec_add3(a_dest, a_dest, a_vec, buffers.a_zero)

    # Update B: B -= lr * grad_B
    b_vec = self._vector_view(buffers.grad_b)
    b_dest = self._vector_view(buffers.B)
    self._math_core.fill(buffers.b_scale, -lr)
    self._math_core.vector_multiply(b_vec, buffers.b_scale)
    self._math_core.vec_add3(b_dest, b_dest, b_vec, buffers.b_zero)

    # Sync back to host
    RPNMathCore.copy_to_host(buffers.A.ptr, self.A)
    RPNMathCore.copy_to_host(buffers.B.ptr, self.B)

    return float(grad_norm)
```

### Ternary Validation Gate (Claude's Code)
```python
def _ternary_gate(self, baseline_perf: float, shadow_perf: float) -> str:
    """
    Ternary validation gate for sovereign training.

    Decision logic:
    - TRUE: Shadow significantly better (improvement >= min_improvement)
    - FALSE: Shadow significantly worse (degradation > max_degradation)
    - UNKNOWN: Marginal difference (accumulate more evidence)
    """
    improvement = shadow_perf - baseline_perf
    degradation = baseline_perf - shadow_perf

    if improvement >= self.config.min_improvement:
        return "TRUE"
    elif degradation > self.config.max_degradation:
        return "FALSE"
    else:
        return "UNKNOWN"
```

---

## Documentation Requirements

### When You Complete a Task
1. **Update this prompt** with results and next steps
2. **Document performance metrics** (before/after timings)
3. **Log test results** (pass/fail, actual vs expected)
4. **Create TEMP file** for completion report (follow existing naming pattern)

### Completion Report Template
```markdown
# Phase 2 RPN Sovereignty - [Task Name] Complete

**Date:** 2025-11-19
**Task:** [Specific optimization/test completed]
**Status:** ✅ Complete | ⚠️ Partial | ❌ Failed

## What Was Done
- [Change 1]
- [Change 2]
- [Change 3]

## Test Results
| Test | Status | Notes |
|------|--------|-------|
| test_rpn_vs_cpu | ✅ Pass | Diff < 1e-4 |
| test_ternary_gate | ✅ Pass | All 3 cases validated |

## Performance Impact
- **Before:** X.XX seconds
- **After:** X.XX seconds
- **Speedup:** X.XX× (XX% faster)

## Next Steps
1. [Next priority task]
2. [Future enhancement]
```

---

## Notes & Reminders

### Architecture Philosophy
> "We fix or we fix — never fallback to CPU"
> "Knowledge lives in embeddings, TRM learns reasoning patterns"
> "Sovereign, explainable, embodied — no exceptions"

### User's Instructions
> "Wonderful! Great progress by Codex, code something as well to help him and prepare the next prompt informing what you did and that he can proceed as he can (do as much as he can)"

### Your Autonomy
- You have full autonomy to implement optimizations
- Run tests proactively
- Document findings thoroughly
- Ask for clarification only if architecturally ambiguous
- Default to "just fix it" when path is clear

### GPU Environment
```bash
# Ensure GPU visibility
export CUDA_VISIBLE_DEVICES=0

# Activate environment
conda activate k3d-cranium

# Set PYTHONPATH
export PYTHONPATH=/mnt/arquivos/EchoSystems\ AI\ Studios/Knowledge\ 3D\ Standard/GitHub/Knowledge3D
```

---

## Summary

### What's Ready Now
✅ RPN math core (your foundation)
✅ SelfUpdatingAdapter with GPU buffers (your implementation)
✅ Ternary validation gate (Claude's enhancement)
✅ Comprehensive test suite (Claude's validation)
✅ Specialist wired to RPN path (already working!)

### What You Can Do
1. **Remove NumPy fallback** (Priority 1) → 100% sovereign
2. **Optimize matrix transpose** (Priority 2) → 15-20% speedup
3. **Add scalar multiply opcode** (Priority 3) → 5-10% speedup
4. **Run regression tests** (Priority 4) → Validate correctness
5. **Document results** (Always) → Record findings

### Expected Outcome
After full Phase 2 implementation:
- **Training time:** ~2 min → ~1.6 min (19% speedup)
- **Sovereignty:** 100% GPU-native (0% CPU fallback)
- **Evidence:** Regression tests pass, metrics documented
- **W3C AIKR:** Complete atomic training proof ready for submission

---

## Questions for Codex?

If anything is unclear or you need architectural guidance, feel free to ask. Otherwise, proceed with autonomy and document your progress!

**We trust your judgment.** You've already demonstrated excellent engineering with rpn_math_core and the GPU buffer architecture. Keep that momentum going! 🚀

---

**End of Prompt**

*Prepared by Claude (K3D Adaptive Swarm)*
*Ready for Codex's autonomous implementation*
*2025-11-19 Session*
