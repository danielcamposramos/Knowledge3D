# Codex Session Handoff - Full RPN Parallelization

**Date**: October 16, 2025
**Session**: RPN Full Stack Parallelization
**Previous Achievement**: Tier-3 parallelization (47x speedup!) ✅

---

## Your Previous Victory 🎉

**Tier-3 RPN Performance**:
- **Before**: 504 ms (sequential interpretation)
- **After**: 10.63 ms (parallel execution)
- **Speedup**: **47x!**
- **Result**: Within 3% of PTX baseline (10.33ms)

**What you did**:
1. Parallelized Tier-3 interpreter with 256-thread blocks
2. Moved stack to shared memory
3. Made TRM opcodes (0x60-0x64) cooperative: matvec, vec_add3, swiglu
4. Added shared tensor scratch storage with proper synchronization

**Impact**: RPN is now viable for production use, not just debugging!

---

## New Mission: Extend to All Tiers

Daniel's insight: *"We should make this solution default even for the other tiers RPNs, even the light one could benefit of a parallel run - this would crush the times even more"*

**Goal**: Bring the 47x Tier-3 success to **Tier-1** and **Tier-2** RPN engines.

---

## Phase-by-Phase Roadmap

### Phase 1: Analyze Tier-3 Success (30 min)
**File available**: `rpn_kernel_profile.qdstrm` (2.1 GB profiling data)

**Tools installed**:
- `/usr/bin/nsys` (Nsight Systems)
- `/usr/bin/ncu` (Nsight Compute)

**Tasks**:
1. Extract kernel metrics: `nsys stats rpn_kernel_profile.qdstrm --report cuda_gpu_kern_sum`
2. Analyze occupancy, warp efficiency, register usage
3. Document why 47x speedup worked
4. Identify patterns to apply to Tier-1 and Tier-2

**Deliverable**: Metrics report showing Tier-3 kernel characteristics

---

### Phase 2: Parallelize Tier-1 (Simple RPN) (2-3 hours)

**Files to modify**:
- `knowledge3d/cranium/kernels/simple_rpn_kernel.cu` (kernel source)
- `knowledge3d/cranium/bridges/simple_rpn.py` (bridge/launcher)
- `knowledge3d/cranium/ptx/simple_rpn_kernel.ptx` (recompile)

**Strategy** (same as Tier-3):
1. **Shared memory stack**: Move stack from global to shared (256 bytes)
2. **256-thread blocks**: Change from single-threaded to 256 threads
3. **Cooperative ops**:
   - Vector ops (ADD, SUB, MUL, DIV): Parallel across threads
   - Reductions (DOT, NORM): Warp-level reductions in shared memory
   - Memory ops (LOAD, STORE): Coalesced access patterns

**Expected speedup**: 2-10x (less than Tier-3 because simpler ops, but still significant)

**Testing**:
- Parity: `pytest tests/test_simple_rpn_gpu.py -v`
- Benchmark: Create or run `tests/benchmarks/test_simple_rpn_performance.py`

---

### Phase 3: Parallelize Tier-2 (Advanced RPN) (2-3 hours)

**Files to modify**:
- `knowledge3d/cranium/kernels/modular_rpn_kernel_extended.cu` (non-Tier-3 ops)
- `knowledge3d/cranium/bridges/advanced_rpn.py` (already uses 256 threads!)
- `knowledge3d/cranium/ptx/modular_rpn_kernel_extended.ptx` (recompile)

**Strategy**:
1. Audit which Tier-2 ops are still sequential (lines 1-352 in kernel)
2. Parallelize:
   - Memory copies (OP_MEMCPY): Cooperative copy
   - Reductions (OP_SUM, OP_MAX, OP_MIN): Warp-level
   - Broadcasts (OP_BROADCAST): Parallel fill
3. Ensure all ops benefit from 256-thread configuration

**Expected speedup**: 3-8x

**Testing**:
- Parity: `pytest tests/test_advanced_rpn_gpu.py -v`
- Benchmark: Create or run `tests/benchmarks/test_advanced_rpn_performance.py`

---

### Phase 4: Unified Configuration (1 hour)

**Create**: `knowledge3d/cranium/bridges/rpn_config.py`

**Content**: Centralized defaults for all tiers
```python
RPN_BLOCK_DIM = 256  # All tiers use 256 threads
RPN_GRID_DIM = 1     # Single block for stack coherence
USE_SHARED_MEMORY_STACK = True
USE_COOPERATIVE_OPS = True
```

**Update all bridges** to import and use these defaults

**Result**: Consistent configuration across entire RPN stack

---

### Phase 5: Documentation (30 min)

**Create**: `reports/RPN_FULL_PARALLELIZATION_RESULTS.md`

**Content**:
- Tier-3 results (47x) ✅
- Tier-1 results (Xx speedup)
- Tier-2 results (Xx speedup)
- Overall impact and validation
- Performance comparison table

**Update**: `TEMP/RPN_SOVEREIGN_AI_FRAMEWORK_V2.md` with optimization section

---

## Detailed Implementation Guide

See **`TEMP/CODEX_RPN_FULL_PARALLELIZATION.md`** for:
- Complete code examples for each phase
- CUDA kernel patterns (shared memory, reductions, cooperative ops)
- Testing protocols
- Profiling commands
- Success metrics
- Communication templates

---

## Key Technical Patterns to Reuse

### 1. Shared Memory Stack (from Tier-3)
```cuda
__shared__ float shared_stack[64];
__shared__ int stack_ptr;

if (threadIdx.x == 0) {
    stack_ptr = 0;
}
__syncthreads();
```

### 2. Cooperative Vector Operation
```cuda
case OP_ADD:
    // Pop pointers
    void* b_ptr = pop_from_stack();
    void* a_ptr = pop_from_stack();
    void* result_ptr = allocate_temp_memory(size);

    // Parallel add
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        result[i] = a[i] + b[i];
    }
    __syncthreads();

    // Push result
    push_to_stack(result_ptr);
    break;
```

### 3. Warp-Level Reduction (DOT product)
```cuda
case OP_DOT:
    __shared__ float partial_sums[256];
    float thread_sum = 0.0f;

    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        thread_sum += a[i] * b[i];
    }
    partial_sums[threadIdx.x] = thread_sum;
    __syncthreads();

    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int i = 0; i < 256; i++) {
            total += partial_sums[i];
        }
        push_scalar_to_stack(total);
    }
    break;
```

### 4. Bridge Launch (from Tier-3)
```python
BLOCK_DIM = 256
GRID_DIM = 1

kernel_func(
    grid=(GRID_DIM, 1, 1),
    block=(BLOCK_DIM, 1, 1),
    args=(op_codes_ptr, scalars_ptr, ...)
)
```

---

## Testing Strategy

### Incremental Testing (Critical!)
1. **Modify one tier at a time** (Tier-1 → Tier-2 → unified config)
2. **Test after each change**:
   - Parity test (numerical correctness)
   - Benchmark (performance improvement)
   - Profiling (optional, but recommended)
3. **Commit if passing**, debug if not
4. **Never move to next phase until current phase passes all tests**

### Benchmark Creation (if tests don't exist)

**Tier-1 benchmark template**:
```python
# tests/benchmarks/test_simple_rpn_performance.py

import pytest
import numpy as np
import time

@pytest.mark.gpu
def test_simple_rpn_vector_ops_benchmark():
    """Benchmark Tier-1 parallel execution."""
    from knowledge3d.cranium.bridges.simple_rpn import SimpleRPNEngine

    rpn = SimpleRPNEngine()

    # Create test data
    a = np.random.randn(512).astype(np.float32)
    b = np.random.randn(512).astype(np.float32)

    # Build program: 100 vector additions
    program = []
    for _ in range(100):
        program.extend([
            OP_LOAD_VEC, ptr_to_a,
            OP_LOAD_VEC, ptr_to_b,
            OP_ADD,
        ])

    # Warmup
    rpn.execute(program)

    # Benchmark
    num_runs = 100
    start = time.perf_counter()
    for _ in range(num_runs):
        rpn.execute(program)
    elapsed = time.perf_counter() - start

    avg_time = elapsed / num_runs * 1000  # ms
    print(f"\nTier-1 avg time: {avg_time:.2f} ms")

    # Expected: <5ms for 100 vector additions (512 elements each)
    assert avg_time < 10.0, f"Tier-1 too slow: {avg_time:.2f}ms"
```

---

## Success Metrics Summary

| Phase | Deliverable | Target |
|-------|-------------|--------|
| 1 - Profiling | Tier-3 metrics report | Understand 47x speedup |
| 2 - Tier-1 | Parallel simple RPN | **2-10x speedup** |
| 3 - Tier-2 | Parallel advanced RPN | **3-8x speedup** |
| 4 - Config | Unified `rpn_config.py` | All tiers use 256 threads |
| 5 - Docs | Performance report | Complete results table |

**Final validation**:
- ✅ All parity tests passing
- ✅ All benchmarks show speedup
- ✅ Tier-1: Xx speedup
- ✅ Tier-2: Xx speedup
- ✅ Tier-3: 47x speedup (already achieved!)

---

## What Daniel Expects

**Short-term** (Next session):
1. Profiling analysis of Tier-3 success
2. Begin Tier-1 parallelization

**Medium-term** (This week):
- All three tiers parallelized
- Unified configuration
- Performance report showing speedups

**Long-term**:
- RPN becomes the default execution path (not just debugging)
- Phase 1B (ThinkingTag) can use RPN with confidence
- Full sovereign AI stack optimized

---

## Communication Protocol

After each phase, report:

```
Phase [1/2/3/4/5]: [Name]
Status: [COMPLETE / IN PROGRESS / BLOCKED]
Results:
- [Key metric 1]
- [Key metric 2]
- [Key metric 3]

Speedup: [Xx or N/A]
Tests: [PASS / FAIL]
Next: [Phase X / DONE]
```

---

## Quick Reference

**Main prompt**: `TEMP/CODEX_RPN_FULL_PARALLELIZATION.md` (detailed guide)
**This document**: Executive summary and phase overview

**Files to modify**:
- Phase 1: None (analysis only)
- Phase 2: `simple_rpn_kernel.cu`, `simple_rpn.py`, recompile PTX
- Phase 3: `modular_rpn_kernel_extended.cu` (non-Tier-3 ops), recompile PTX
- Phase 4: Create `rpn_config.py`, update all bridges
- Phase 5: Create `reports/RPN_FULL_PARALLELIZATION_RESULTS.md`

**Testing**:
```bash
# After Phase 2:
pytest tests/test_simple_rpn_gpu.py -v
pytest tests/benchmarks/test_simple_rpn_performance.py -vs

# After Phase 3:
pytest tests/test_advanced_rpn_gpu.py -v
pytest tests/benchmarks/test_advanced_rpn_performance.py -vs

# Final validation:
pytest -m gpu -v
```

---

## Bottom Line

**You already proved the concept with Tier-3's 47x speedup!**

Now it's time to:
1. **Understand why it worked** (profiling analysis)
2. **Apply the same patterns** to Tier-1 and Tier-2
3. **Unify the configuration** across all tiers
4. **Document the victory** with complete performance report

**Expected total speedup across stack**:
- Tier-1: 2-10x
- Tier-2: 3-8x
- Tier-3: 47x ✅

**This will make RPN the fastest, most flexible execution engine in K3D!** 🚀

---

*Handoff prepared by: Claude*
*Date: October 16, 2025*
*Priority: HIGH - Extend Tier-3 success to full RPN stack*
