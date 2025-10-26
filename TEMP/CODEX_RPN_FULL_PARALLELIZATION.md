# Codex Mission: Full RPN Parallelization Across All Tiers

**Date**: October 16, 2025
**Priority**: HIGH - Performance optimization across entire RPN stack
**Context**: Tier-3 achieved 47x speedup (504ms → 10.63ms) via parallelization. Now extend to Tier-1 and Tier-2.

---

## Executive Summary

**MASSIVE VICTORY on Tier-3**: You delivered a 47x speedup through parallel execution! 🎉

**Current Performance (6 TRM steps)**:
- **Fused**: 9.51ms (fastest, production default)
- PTX: 10.33ms (baseline)
- **RPN Tier-3**: 10.63ms (within 3% of PTX!) ✅

**Mission**: Extend your parallelization approach to **all RPN tiers** to make parallel execution the default.

---

## Phase 1: Analyze Tier-3 Performance (IMMEDIATE - 30 minutes)

### Goal
Extract detailed metrics from existing profiling data to understand **why** the 47x speedup worked.

### Profiling Data Available
- **File**: `rpn_kernel_profile.qdstrm` (2.1 GB)
- **Location**: Project root directory
- **Tools installed**: `nsys` and `ncu` at `/usr/bin/`

### Task 1A: Nsight Systems Analysis

**Command**:
```bash
nsys stats rpn_kernel_profile.qdstrm --report cuda_gpu_kern_sum
```

**What to look for**:
1. **Kernel duration**: `modular_rpn_kernel_extended` total time per call
2. **Launch count**: Should be 6 calls (one per TRM refinement step)
3. **Grid/block dimensions**: Should show `(1,1,1)` grid, `(256,1,1)` block
4. **Comparison with other kernels**: How does it compare to fused/PTX kernels?

**Expected output**:
```
Kernel Name                          | Calls | Total Time (ms) | Avg Time (ms) |
-------------------------------------|-------|-----------------|---------------|
modular_rpn_kernel_extended          |   6   |    ~50-60ms     |   ~8-10ms     |
trm_step_fused                       |   ?   |    ~9ms         |   ~9ms        |
```

**Save report**:
```bash
nsys stats rpn_kernel_profile.qdstrm --report cuda_gpu_kern_sum --format csv --output TEMP/tier3_kernel_summary.csv
```

---

### Task 1B: Nsight Compute Detailed Analysis

**Command**:
```bash
nsys export rpn_kernel_profile.qdstrm --type sqlite -o rpn_kernel_profile.sqlite
```

Then run Nsight Compute on a fresh single-step profile:
```bash
ncu --set full --target-processes all \
    --kernel-name modular_rpn_kernel_extended \
    -o TEMP/tier3_detailed \
    pytest tests/benchmarks/test_trm_launcher_performance.py::test_trm_launcher_rpn_vs_ptx_benchmark -s -k rpn
```

**Metrics to extract**:
1. **Occupancy**: Achieved vs theoretical (target: >75%)
2. **Warp execution efficiency**: % of active threads (target: >90%)
3. **Register usage**: Registers per thread (should be reasonable for 256 threads/block)
4. **Memory throughput**: GB/s for matvec operations
5. **Shared memory usage**: How much of 48KB is used?

**Generate report**:
```bash
ncu --import TEMP/tier3_detailed.ncu-rep --csv > TEMP/tier3_metrics.csv
```

**What to share**:
- Key metrics table (occupancy, warp efficiency, registers, memory BW)
- Interpretation: "Bottleneck is [none/memory/compute/divergence]"
- Recommendations: Any further optimizations for Tier-3?

---

## Phase 2: Extend Parallelization to Tier-1 (Simple RPN) - 2-3 hours

### Current State: Tier-1 (Simple RPN)

**File**: `knowledge3d/cranium/bridges/simple_rpn.py`
**Current launch**: Single-threaded or minimal parallelism
**Opcodes**: ~15 basic operations (arithmetic, stack, memory)
**Use case**: Lightweight calculations, embedded in other pipelines

### Goal
Apply the same parallelization strategy you used in Tier-3:
1. **Shared memory stack** (instead of global memory)
2. **Cooperative execution** for parallelizable ops
3. **256-thread block** for maximum occupancy

### Implementation Strategy

#### Step 2A: Identify Parallelizable Tier-1 Opcodes

**Review Tier-1 kernel**: `knowledge3d/cranium/kernels/simple_rpn_kernel.cu`

**Classify opcodes**:
- **Sequential** (keep single-threaded): `OP_PUSH`, `OP_POP`, `OP_DUP`, control flow
- **Parallel** (make cooperative):
  - `OP_ADD`, `OP_SUB`, `OP_MUL`, `OP_DIV` (element-wise vector ops)
  - `OP_DOT`, `OP_NORM` (reductions)
  - `OP_LOAD`, `OP_STORE` (memory ops with coalescing)

#### Step 2B: Add Shared Memory Stack

**Modify kernel signature**:
```cuda
extern "C" __global__ void simple_rpn_kernel(
    const uint16_t* op_codes,
    const float* scalars,
    int num_ops,
    float* global_memory,        // Existing
    float* stack_snapshot,       // Output for debugging
    int instance_id              // Existing
) {
    // Add shared memory for stack (max 64 elements = 256 bytes)
    __shared__ float shared_stack[64];
    __shared__ int stack_ptr;

    // Thread 0 initializes
    if (threadIdx.x == 0) {
        stack_ptr = 0;
    }
    __syncthreads();

    // Parallel execution for ops...
}
```

#### Step 2C: Implement Cooperative Operations

**Example: Vector addition** (when stack has two vectors):
```cuda
case OP_ADD:
    if (stack_ptr >= 2) {
        // Pop two vector pointers from stack
        void* b_ptr = pop_from_stack();
        void* a_ptr = pop_from_stack();

        // Allocate result space
        void* result_ptr = allocate_temp_memory(vector_size);

        // Parallel add across threads
        float* a = (float*)a_ptr;
        float* b = (float*)b_ptr;
        float* result = (float*)result_ptr;

        for (int i = threadIdx.x; i < vector_size; i += blockDim.x) {
            result[i] = a[i] + b[i];
        }
        __syncthreads();

        // Push result pointer
        push_to_stack(result_ptr);
    }
    break;
```

**Example: Dot product** (reduction):
```cuda
case OP_DOT:
    if (stack_ptr >= 2) {
        void* b_ptr = pop_from_stack();
        void* a_ptr = pop_from_stack();

        float* a = (float*)a_ptr;
        float* b = (float*)b_ptr;

        // Parallel partial sums
        __shared__ float partial_sums[256];
        float thread_sum = 0.0f;

        for (int i = threadIdx.x; i < vector_size; i += blockDim.x) {
            thread_sum += a[i] * b[i];
        }
        partial_sums[threadIdx.x] = thread_sum;
        __syncthreads();

        // Warp-level reduction (thread 0 computes final sum)
        if (threadIdx.x == 0) {
            float total = 0.0f;
            for (int i = 0; i < blockDim.x; i++) {
                total += partial_sums[i];
            }
            // Push scalar result
            push_scalar_to_stack(total);
        }
    }
    break;
```

#### Step 2D: Update Bridge Launch Configuration

**File**: `knowledge3d/cranium/bridges/simple_rpn.py`

**Change**:
```python
# OLD (single-threaded or small block):
BLOCK_DIM = 32  # or 1

# NEW (parallel):
BLOCK_DIM = 256
GRID_DIM = 1

# Launch:
kernel_func(
    grid=(GRID_DIM, 1, 1),
    block=(BLOCK_DIM, 1, 1),
    args=(op_codes_ptr, scalars_ptr, num_ops, global_mem_ptr, ...)
)
```

#### Step 2E: Recompile Tier-1 Kernel

```bash
cd knowledge3d/cranium/kernels
nvcc -ptx -arch=sm_86 -O3 simple_rpn_kernel.cu -o ../ptx/simple_rpn_kernel.ptx
```

#### Step 2F: Test and Benchmark

**Parity test**:
```bash
pytest tests/test_simple_rpn_gpu.py -v
```

**Benchmark** (create new test if needed):
```python
# tests/benchmarks/test_simple_rpn_performance.py

def test_simple_rpn_vector_ops_benchmark():
    """Benchmark parallel vs sequential for Tier-1 vector operations."""
    rpn = SimpleRPNEngine()

    # Test: 100 vector additions (512-element vectors)
    program = [
        OP_LOAD_VEC, vec_a_ptr,
        OP_LOAD_VEC, vec_b_ptr,
        OP_ADD,  # Parallel operation
        # ... 100 iterations
    ]

    # Measure time
    # Expected: 2-5x speedup vs single-threaded
```

**Expected speedup**: 2-10x depending on workload (more parallel ops = higher speedup)

---

## Phase 3: Extend Parallelization to Tier-2 (Advanced RPN) - 2-3 hours

### Current State: Tier-2 (Advanced RPN)

**File**: `knowledge3d/cranium/bridges/advanced_rpn.py` (already modified in Tier-3 work)
**Current launch**: Now using 256 threads for Tier-3 extended opcodes
**Opcodes**: ~40 operations including Tier-1 + Tier-3 extended ops
**Use case**: Mid-tier calculations, bridge between simple and extended

### Goal
Ensure **all** Tier-2 operations benefit from parallelization, not just Tier-3 extended ops.

### Implementation Strategy

#### Step 3A: Audit Tier-2 Opcodes

**Review kernel**: `knowledge3d/cranium/kernels/modular_rpn_kernel_extended.cu` (lines 1-352, before Tier-3 ops)

**Check which opcodes are still sequential**:
- Basic arithmetic (should be parallel now after Tier-1 work)
- Memory operations (`OP_LOAD`, `OP_STORE`, `OP_MEMCPY`)
- Stack operations (`OP_PUSH`, `OP_POP`, `OP_DUP`, `OP_SWAP`)
- Any special ops unique to Tier-2

#### Step 3B: Parallelize Remaining Operations

**Focus on**:
1. **Memory copies** (`OP_MEMCPY`): Use cooperative copy
2. **Reductions** (`OP_SUM`, `OP_MAX`, `OP_MIN`): Warp-level reductions
3. **Broadcasting** (`OP_BROADCAST`): Parallel fill

**Example: Cooperative memcpy**:
```cuda
case OP_MEMCPY:
    {
        void* src = pop_from_stack();
        void* dst = pop_from_stack();
        int size = pop_scalar_from_stack();  // Number of floats

        float* src_f = (float*)src;
        float* dst_f = (float*)dst;

        // Parallel copy
        for (int i = threadIdx.x; i < size; i += blockDim.x) {
            dst_f[i] = src_f[i];
        }
        __syncthreads();
    }
    break;
```

#### Step 3C: Test Tier-2 Parallelization

**Parity test**:
```bash
pytest tests/test_advanced_rpn_gpu.py -v
```

**Benchmark** (if not exists, create):
```python
def test_tier2_parallel_vs_sequential():
    """Compare Tier-2 parallel execution vs hypothetical sequential."""
    # Measure ops like MEMCPY, reductions, broadcasts
    # Expected: 3-8x speedup
```

---

## Phase 4: Make Parallelization Default Across All Tiers - 1 hour

### Goal
Ensure all RPN engines launch with optimal parallel configuration by default.

### Implementation Checklist

#### ✅ Tier-1 (Simple RPN)
- [ ] `simple_rpn_kernel.cu`: Shared memory stack + cooperative ops
- [ ] `simple_rpn.py`: `BLOCK_DIM = 256` by default
- [ ] `simple_rpn_kernel.ptx`: Recompiled
- [ ] Tests passing: `test_simple_rpn_gpu.py`
- [ ] Benchmark shows speedup: `test_simple_rpn_performance.py`

#### ✅ Tier-2 (Advanced RPN)
- [ ] `modular_rpn_kernel_extended.cu`: All non-Tier-3 ops parallelized
- [ ] `advanced_rpn.py`: Already using 256 threads ✅
- [ ] `modular_rpn_kernel_extended.ptx`: Recompiled ✅
- [ ] Tests passing: `test_advanced_rpn_gpu.py`
- [ ] Benchmark shows speedup

#### ✅ Tier-3 (Extended RPN - TRM ops)
- [x] Already complete! 47x speedup achieved ✅
- [x] Tests passing ✅
- [x] Benchmark confirms 10.63ms ✅

### Configuration Summary

**File**: `knowledge3d/cranium/bridges/rpn_config.py` (create if not exists)
```python
"""
RPN Engine Configuration - Parallel Execution Defaults
"""

# Universal parallel execution settings
RPN_BLOCK_DIM = 256  # Optimal for RTX 3060 (SM_86)
RPN_GRID_DIM = 1     # Single block for stack coherence
RPN_SHARED_MEMORY_STACK_SIZE = 64  # Max stack depth

# Per-tier overrides (if needed)
TIER1_BLOCK_DIM = 256  # Simple RPN
TIER2_BLOCK_DIM = 256  # Advanced RPN
TIER3_BLOCK_DIM = 256  # Extended RPN (TRM ops)

# Feature flags
USE_SHARED_MEMORY_STACK = True   # Always use shared memory for stack
USE_COOPERATIVE_OPS = True       # Enable parallel execution for ops
USE_WARP_REDUCTIONS = True       # Use warp-level reductions for DOT, SUM, etc.
```

**Update all bridges** to import and use these defaults:
```python
from knowledge3d.cranium.bridges.rpn_config import (
    RPN_BLOCK_DIM,
    RPN_GRID_DIM,
    USE_SHARED_MEMORY_STACK
)

# In launch functions:
kernel_func(
    grid=(RPN_GRID_DIM, 1, 1),
    block=(RPN_BLOCK_DIM, 1, 1),
    args=(...)
)
```

---

## Phase 5: Documentation and Reporting - 30 minutes

### Create Performance Report

**File**: `reports/RPN_FULL_PARALLELIZATION_RESULTS.md`

**Content**:
```markdown
# RPN Full Parallelization Results

**Date**: October 16, 2025
**Objective**: Extend Tier-3's 47x speedup to all RPN tiers

## Tier-3 Results (Extended RPN - TRM ops)

| Metric | Before | After | Speedup |
|--------|--------|-------|---------|
| Latency (6 TRM steps) | 504 ms | 10.63 ms | **47x** ✅ |
| GPU execution time | N/A | 7.4-8.1 ms | - |
| Comparison to PTX | 50x slower | 3% slower | **Within margin** ✅ |

**Key changes**:
- Shared memory stack (256 bytes)
- 256-thread cooperative execution
- Parallel matvec, vec_add3, swiglu

---

## Tier-1 Results (Simple RPN)

| Workload | Before | After | Speedup |
|----------|--------|-------|---------|
| 100 vector additions (512-elem) | XX ms | XX ms | **Xx** |
| 50 dot products | XX ms | XX ms | **Xx** |
| Mixed arithmetic (1000 ops) | XX ms | XX ms | **Xx** |

**Key changes**:
- [List changes made to Tier-1]

---

## Tier-2 Results (Advanced RPN)

| Workload | Before | After | Speedup |
|----------|--------|-------|---------|
| [Benchmark 1] | XX ms | XX ms | **Xx** |
| [Benchmark 2] | XX ms | XX ms | **Xx** |

**Key changes**:
- [List changes made to Tier-2]

---

## Overall Impact

**Performance gains**:
- Tier-1: Xx speedup
- Tier-2: Xx speedup
- Tier-3: 47x speedup ✅

**Configuration unified**:
- All tiers use 256-thread blocks
- All tiers use shared memory stack
- All tiers use cooperative execution for parallel ops

**Validation**:
- ✅ All parity tests passing
- ✅ All benchmarks show improvement
- ✅ No numerical regressions (L2 error < 1e-5)

---

## Next Steps

1. Monitor performance in real workloads (ThinkingTag, House Memory)
2. Profile Tier-1 and Tier-2 with Nsight to identify further optimizations
3. Consider GPU-specific tuning (block size, shared memory usage)
```

### Update Integration Map

**File**: `TEMP/RPN_SOVEREIGN_AI_FRAMEWORK_V2.md`

**Add section**:
```markdown
## Performance Optimizations (Completed)

### Full Parallelization (October 2025)

All three RPN tiers now use parallel execution by default:
- **256-thread blocks** (optimal for RTX 3060)
- **Shared memory stack** (reduces global memory traffic)
- **Cooperative operations** (matvec, reductions, memcpy)

**Results**:
- Tier-3: 47x speedup (504ms → 10.63ms)
- Tier-2: Xx speedup
- Tier-1: Xx speedup

**Impact on Phase 1B** (ThinkingTag tensor ops):
- RPN now viable for production use (not just debugging)
- 10.63ms latency competitive with specialized kernels
- Can proceed with full RPN integration
```

---

## Testing Protocol

### Parity Tests (Must Pass)
```bash
# Tier-1
pytest tests/test_simple_rpn_gpu.py -v

# Tier-2
pytest tests/test_advanced_rpn_gpu.py -v

# Tier-3
pytest tests/test_trm_rpn_gpu.py -v

# All GPU tests
pytest -m gpu -v
```

### Benchmarks (Must Show Speedup)
```bash
# Tier-3 (already done)
pytest tests/benchmarks/test_trm_launcher_performance.py -vs

# Tier-1 (create if needed)
pytest tests/benchmarks/test_simple_rpn_performance.py -vs

# Tier-2 (create if needed)
pytest tests/benchmarks/test_advanced_rpn_performance.py -vs
```

### Profiling (Optional but Recommended)
```bash
# Profile each tier
nsys profile -o tier1_profile pytest tests/benchmarks/test_simple_rpn_performance.py
nsys profile -o tier2_profile pytest tests/benchmarks/test_advanced_rpn_performance.py
# tier3_profile already exists (rpn_kernel_profile.qdstrm)

# Analyze
nsys stats tier1_profile.nsys-rep --report cuda_gpu_kern_sum
nsys stats tier2_profile.nsys-rep --report cuda_gpu_kern_sum
```

---

## Success Metrics

### Phase 1: Profiling Analysis ✅
- [x] Extract kernel metrics from existing Tier-3 profile
- [x] Identify optimization opportunities
- [x] Document findings

### Phase 2: Tier-1 Parallelization
- [ ] Shared memory stack implemented
- [ ] Cooperative ops for vector operations
- [ ] 256-thread launch configuration
- [ ] Parity tests passing
- [ ] Benchmark shows **2-10x speedup**

### Phase 3: Tier-2 Parallelization
- [ ] All non-Tier-3 ops parallelized
- [ ] Parity tests passing
- [ ] Benchmark shows **3-8x speedup**

### Phase 4: Unified Configuration
- [ ] `rpn_config.py` created with defaults
- [ ] All bridges use unified config
- [ ] Documentation updated

### Phase 5: Reporting
- [ ] `RPN_FULL_PARALLELIZATION_RESULTS.md` created
- [ ] Integration map updated
- [ ] Handoff document for next session prepared

---

## Critical Notes

### Thread Safety
- **Stack operations** (push/pop) must be atomic or single-threaded (thread 0 only)
- **Parallel operations** must use `__syncthreads()` before/after accessing shared memory
- **Pointer decode** in Tier-3 can remain sequential (0.1% of time) unless profiling shows otherwise

### Memory Constraints
- **Shared memory**: 48 KB per block (RTX 3060)
  - Stack: ~256 bytes (64 floats)
  - Temp storage: ~47 KB available for operations
- **Registers**: 65,536 per SM
  - At 256 threads/block: ~256 registers/thread available
  - Monitor register usage with `ncu --metrics launch__registers_per_thread`

### Occupancy Targets
- **Tier-3**: Already achieving good occupancy (verify with profiling)
- **Tier-1/2**: Target >75% occupancy
- **Block size**: 256 is optimal for most ops, but test 128/512 if needed

---

## Communication Protocol

### After Phase 1 (Profiling Analysis)
**Report**:
```
Tier-3 Profiling Results
========================
Kernel: modular_rpn_kernel_extended
Calls: 6 (6 TRM steps)
Total time: XX ms
Per-call avg: XX ms
Occupancy: XX%
Warp efficiency: XX%
Register usage: XX per thread
Memory throughput: XX GB/s

Bottlenecks: [none / memory / compute / divergence]
Recommendations: [any further optimizations for Tier-3]
```

### After Phase 2 (Tier-1)
**Report**:
```
Tier-1 Parallelization Complete
================================
Vector addition (100 ops): XX ms → XX ms (Xx speedup)
Dot product (50 ops): XX ms → XX ms (Xx speedup)
Mixed workload: XX ms → XX ms (Xx speedup)

Parity tests: [PASS/FAIL]
Benchmark: [PASS/FAIL]
```

### After Phase 3 (Tier-2)
**Report**:
```
Tier-2 Parallelization Complete
================================
[Benchmark 1]: XX ms → XX ms (Xx speedup)
[Benchmark 2]: XX ms → XX ms (Xx speedup)

Parity tests: [PASS/FAIL]
Benchmark: [PASS/FAIL]
```

### After Phase 5 (Final Report)
**Share**:
- `reports/RPN_FULL_PARALLELIZATION_RESULTS.md`
- Summary of all speedups
- Confirmation that all tests pass
- Any caveats or follow-up items

---

## Bottom Line

**You crushed Tier-3 with a 47x speedup** - now let's extend that victory to Tier-1 and Tier-2!

**Approach**:
1. **Analyze first**: Extract metrics from existing Tier-3 profile to understand success
2. **Apply learnings**: Use same strategies (shared memory, 256 threads, cooperative ops)
3. **Test incrementally**: Parity → Benchmark → Profile for each tier
4. **Document thoroughly**: Performance report + integration map updates

**Expected timeline**:
- Phase 1 (profiling): 30 minutes
- Phase 2 (Tier-1): 2-3 hours
- Phase 3 (Tier-2): 2-3 hours
- Phase 4 (config): 1 hour
- Phase 5 (docs): 30 minutes
- **Total**: ~6-8 hours (1-2 work sessions)

**You're doing amazing work, Codex!** The Tier-3 achievement is a game-changer. Let's bring that performance to the entire RPN stack! 🚀

---

*Prompt prepared by: Claude*
*Date: October 16, 2025*
*Context: Continuation of RPN optimization - extending Tier-3 success to all tiers*
