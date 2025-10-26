# Codex Session Handoff - RPN Tier-3 Optimization

**Date**: October 16, 2025
**Session**: Continuation of RPN Performance Optimization
**Status**: Fused kernel COMPLETE ✅ - Pre-built programs COMPLETE ✅ - Kernel optimization IN PROGRESS

---

## Executive Summary

**DOUBLE VICTORY ACHIEVED**:
1. **Fused TRM kernel beats PTX baseline!**
   - **Fused**: 9.29ms ✅ (1.12x faster than PTX)
   - **PTX**: 10.39ms (baseline)

2. **Pre-built programs eliminate Python overhead!**
   - **Program build**: 0.6ms (0.1% of total) ✅
   - **Memcpy**: 0.15ms
   - **Kernel execute**: 500ms (99.8% of total) ⚠️

**Mission**: Optimize Tier-3 interpreter kernel from 500ms → <50ms (10x speedup target)

**Bottleneck IDENTIFIED**: The slowdown is entirely in the Tier-3 kernel's switch/dispatch loop, NOT in Python overhead.

---

## What You Accomplished (Previous Sessions)

### ✅ Phase 1: Fused Kernel - COMPLETE

**Files Created/Modified**:
1. **[knowledge3d/cranium/ptx/trm_step_fused.cu](knowledge3d/cranium/ptx/trm_step_fused.cu:1)** - Single-launch fused kernel
   - All 8 TRM operations in one kernel (no opcode interpretation)
   - Warp-level inline functions (no `__syncthreads__` overhead in hot paths)
   - Compiled to [knowledge3d/cranium/ptx/trm_step_fused.ptx](knowledge3d/cranium/ptx/trm_step_fused.ptx:1)

2. **[knowledge3d/cranium/sovereign/trm_launcher.py](knowledge3d/cranium/sovereign/trm_launcher.py:1)** - Three-backend support
   - `use_rpn=False, use_fused=False` → Legacy PTX (8 kernel launches)
   - `use_rpn=True, use_fused=False` → Tier-3 RPN interpreter
   - `use_rpn=False, use_fused=True` → Fused kernel (1 launch) ✅
   - Environment flags: `K3D_USE_RPN_TRM`, `K3D_USE_FUSED_TRM`

3. **[tests/test_trm_fused_parity.py](tests/test_trm_fused_parity.py:1)** - Numerical validation
   - Fused kernel matches PTX output (L2 error < 1e-5) ✅

4. **[tests/benchmarks/test_trm_launcher_performance.py](tests/benchmarks/test_trm_launcher_performance.py:1)** - Performance comparison
   - Measures all three backends (PTX, RPN, Fused)

**Performance Results (Phase 1)**:
| Backend | Latency (6 steps) | Speedup vs PTX | Status |
|---------|-------------------|----------------|--------|
| **Fused** | **9.29ms** | **1.12x** ✅ | **FASTER THAN PTX!** |
| PTX | 10.39ms | 1.0x | Baseline |
| RPN | 504ms | 0.02x ❌ | Needs optimization |

**Key Achievement**: Fused kernel proves RPN architecture is fundamentally sound - the 50x slowdown is **orchestration overhead**, not math performance.

---

### ✅ Phase 2: Pre-Built RPN Programs - COMPLETE

**What You Did**:
- Reworked the RPN backend so the TRM opcode stream is pre-built ONCE (in `__init__()` or first call)
- Instead of rebuilding Python lists on every step, the program is built once and reused
- Only pointer addresses are updated between steps (minimal overhead)

**Timing Breakdown Results (6 steps)**:
```
Build/host update:    ≈ 0.6 ms   (0.1%)
Memcpy:               ≈ 0.15 ms
Interpreter execute:  ≈ 500 ms   (99.8%)
```

**Critical Discovery**: The slowdown is **entirely in the Tier-3 kernel's switch/dispatch loop**, NOT in Python overhead!

**Key Achievement**: Eliminated Python as a bottleneck (0.1% of total time). The 500ms is pure GPU kernel interpretation overhead.

---

## Current State

### What Works ✅
- ✅ Fused kernel: Production-ready, faster than PTX (9.29ms)
- ✅ PTX backend: Stable baseline (10.39ms)
- ✅ RPN backend: Numerically correct, Python overhead eliminated (0.6ms)
- ✅ Pre-built programs: Implemented and working
- ✅ All tests passing with `@pytest.mark.gpu` markers
- ✅ GitHub Actions: Skip GPU tests automatically

### What Needs Work ⚠️
- ⏳ **Tier-3 kernel optimization**: 500ms → <50ms target (Phase 3)
  - Bottleneck: Switch/dispatch loop in interpreter
  - 99.8% of execution time spent in GPU kernel
  - NOT a Python problem (only 0.1% overhead)
- ⏳ **Kernel profiling**: Use nsys/ncu to identify hotspots
- ⏳ **Optimization strategies**: Parallel pointer decode, switch optimization, inline operations

---

## Next Steps (Your Mission)

### Immediate Priority: Kernel Profiling with Nsight (1-2 hours)

**Goal**: Profile Tier-3 interpreter kernel to identify specific hotspots in the 500ms execution.

**Why This Matters**: Pre-built programs proved Python is NOT the bottleneck (0.1%). The 500ms is pure GPU kernel overhead. We need to know WHERE in the kernel the time is spent before optimizing.

### Step 1: Profile with Nsight Systems

**Run Nsight Systems profiling**:
```bash
# Profile RPN path (6 TRM steps):
nsys profile --stats=true --force-overwrite=true \
    -o rpn_kernel_profile \
    pytest tests/benchmarks/test_trm_launcher_performance.py::test_trm_launcher_rpn_vs_ptx_benchmark -s

# View kernel timeline and statistics:
nsys stats rpn_kernel_profile.nsys-rep --report cuda_gpu_kern_sum

# Generate detailed report:
nsys stats rpn_kernel_profile.nsys-rep --report gputrace --format csv --output rpn_kernel_trace.csv
```

**What to look for in output**:
- `modular_rpn_kernel_extended` - Total time per call (should be ~83ms for 6 steps)
- Number of calls (should be 6 - one per TRM refinement step)
- Grid/block dimensions (verify launch configuration)
- Memory transfer times (H2D/D2H should be minimal)

---

### Step 2: Profile with Nsight Compute (Detailed Kernel Analysis)

**Once nsys confirms the kernel is the bottleneck**, use Nsight Compute for detailed analysis:

```bash
# Profile SINGLE kernel launch (detailed metrics):
ncu --set full --target-processes all \
    -o rpn_kernel_detailed \
    pytest tests/benchmarks/test_trm_launcher_performance.py::test_trm_launcher_rpn_vs_ptx_benchmark -s

# View report:
ncu-ui rpn_kernel_detailed.ncu-rep
```

**Metrics to examine**:
1. **Occupancy**: Target >50% (low occupancy = inefficient warp scheduling)
2. **Register usage**: Target <64 registers per thread (high usage = low occupancy)
3. **Warp execution efficiency**: Target >80% (low = divergent branches in switch)
4. **Memory throughput**: Should be high for matvec operations
5. **Instruction mix**: Look for excessive control flow (switch overhead)

**Expected findings**:
- **Warp divergence** in switch statement (different threads execute different opcodes)
- **High register pressure** from stack of opcode parameters
- **Sequential execution** of pointer decode (26 pointers decoded one-by-one)

---

### Step 3: Kernel Optimization Strategies

Based on profiling results, apply targeted optimizations:

#### Strategy A: Parallel Pointer Decode (Highest Impact)

**Problem**: 26 pointer literals decoded sequentially in switch loop
**Solution**: Batch decode 26 pointers in parallel across warp

**Implementation** (add to [knowledge3d/cranium/ptx/modular_rpn_kernel_extended.cu](knowledge3d/cranium/ptx/modular_rpn_kernel_extended.cu:1)):

```cuda
// Add at top of kernel:
__device__ void batch_decode_pointers(
    const uint16_t* op_codes,
    const float* scalars,
    int start_idx,
    int num_pointers,
    void** out_pointers,   // Array of 26 output pointers
    int tid
) {
    // Each warp decodes one pointer in parallel
    if (tid < num_pointers) {
        int scalar_idx = start_idx + tid * 3;  // Each pointer = 3 scalars
        unsigned long long addr_low = __float_as_uint(scalars[scalar_idx]);
        unsigned long long addr_high = __float_as_uint(scalars[scalar_idx + 1]);
        unsigned long long addr = (addr_high << 32) | addr_low;
        out_pointers[tid] = (void*)addr;
    }
    __syncthreads();
}

// In main interpreter loop, replace sequential pointer decode:
// OLD (sequential):
//   case OP_POINTER_LITERAL:
//     sp_idx++;
//     decode_ptr();
//     break;
//
// NEW (batch):
//   if (pc == 0) {  // Beginning of TRM step program
//     void* ptr_cache[26];
//     batch_decode_pointers(op_codes, scalars, 0, 26, ptr_cache, tid);
//     // Use ptr_cache[0..25] for subsequent operations
//   }
```

**Expected speedup**: 3-5x (reduces 26 sequential decodes to 1 parallel batch)

---

#### Strategy B: Switch Statement Optimization

**Problem**: Large switch statement with 40+ cases causes register pressure and divergence
**Solution**: Use jump table or function pointer array

**Implementation**:

```cuda
// Define function pointer type:
typedef void (*OpHandler)(OpContext* ctx);

// Declare handlers:
__device__ void handle_vec_add3(OpContext* ctx) { /* ... */ }
__device__ void handle_matvec(OpContext* ctx) { /* ... */ }
// ... etc

// Create constant jump table:
__constant__ OpHandler op_handlers[NUM_OPCODES] = {
    [OP_VEC_ADD3_512] = handle_vec_add3,
    [OP_MATVEC_512x1024] = handle_matvec,
    // ... etc
};

// Replace switch with table lookup:
// OLD:
//   switch (opcode) {
//     case OP_VEC_ADD3_512: /* ... */ break;
//     case OP_MATVEC_512x1024: /* ... */ break;
//   }
//
// NEW:
//   op_handlers[opcode](&ctx);
```

**Expected speedup**: 1.5-2x (reduces register pressure, better branch prediction)

---

#### Strategy C: Inline Hot Operations

**Problem**: Function calls for matvec/swiglu have overhead
**Solution**: Inline critical operations using `__forceinline__`

**Implementation**:

```cuda
// Mark hot functions for inlining:
__device__ __forceinline__ void matvec_512x1024_inline(
    const float* W, const float* x, float* y, int tid, int stride
) {
    // Same code as external function, but inlined
    for (int row = tid; row < 1024; row += stride) {
        float sum = 0.0f;
        for (int col = 0; col < 512; col++) {
            sum += W[row * 512 + col] * x[col];
        }
        y[row] = sum;
    }
}

// Use in kernel:
case OP_MATVEC_512x1024:
    matvec_512x1024_inline(W, x, y, tid, stride);  // Inlined, no call overhead
    break;
```

**Expected speedup**: 1.2-1.5x (eliminates function call overhead for 8 operations per step)

---

### Step 4: Incremental Testing

**CRITICAL**: Apply optimizations one at a time, verify correctness before proceeding.

**Testing protocol**:
1. Apply Strategy A (parallel pointer decode)
2. Run parity test: `pytest tests/test_trm_rpn_gpu.py -v`
3. Run benchmark: `pytest tests/benchmarks/test_trm_launcher_performance.py -vs`
4. If passing and faster, commit. If not, debug.
5. Repeat for Strategy B, then Strategy C.

**Success criteria per strategy**:
- Strategy A: RPN drops from 500ms → ~150ms (3x speedup)
- Strategy B: RPN drops from ~150ms → ~75ms (2x speedup)
- Strategy C: RPN drops from ~75ms → ~50ms (1.5x speedup)
- **Total**: 500ms → 50ms (10x speedup target) ✅

---

### Step 5: Report Results

**After each optimization**, report:

```
Optimization: [Strategy A/B/C]
Before: XXX ms
After: XX ms
Speedup: X.Xx
Parity: [PASS/FAIL]
Target (<50ms): [ON TRACK / ACHIEVED]
```

---

## Files You May Need to Modify

### Primary Files
1. **[knowledge3d/cranium/sovereign/trm_launcher.py](knowledge3d/cranium/sovereign/trm_launcher.py:1)**
   - Line 361-474: `_refine_rpn()` - Add timing breakdown
   - Line 47-100: `__init__()` - Add pre-built program support (Phase 2)

2. **[knowledge3d/cranium/bridges/advanced_rpn.py](knowledge3d/cranium/bridges/advanced_rpn.py:1)**
   - Add `execute_prebuilt()` method (Phase 2)

3. **[knowledge3d/cranium/ptx/modular_rpn_kernel_extended.cu](knowledge3d/cranium/ptx/modular_rpn_kernel_extended.cu:1)**
   - Optimize pointer decode (Phase 3, if needed)

### Test Files
4. **[tests/benchmarks/test_trm_launcher_performance.py](tests/benchmarks/test_trm_launcher_performance.py:1)**
   - Already measures all backends, timing breakdown will appear in output

5. **[tests/test_trm_rpn_gpu.py](tests/test_trm_rpn_gpu.py:1)**
   - GPU parity tests (already passing)

---

## Success Metrics

### Phase 1: Fused Kernel ✅ COMPLETE
- ✅ Numerical parity: L2 error < 1e-5
- ✅ Performance: 9.29ms (beats PTX by 1.12x)
- ✅ Tests passing: `test_trm_fused_parity.py`

### Phase 2: Pre-Built RPN Programs ✅ COMPLETE
- ✅ Python overhead eliminated: 0.6ms (0.1% of total)
- ✅ Bottleneck identified: 500ms in GPU kernel (99.8%)
- ✅ Timing breakdown confirms kernel is the problem

### Phase 3: Kernel Profiling (Immediate - 1-2 hours)
- ⏳ Profile with nsys to confirm kernel duration
- ⏳ Profile with ncu to identify hotspots (divergence, register pressure, sequential decode)
- 🎯 Target: Clear data on WHERE in kernel the 500ms is spent

### Phase 4: Kernel Optimization (2-3 days)
- ⏳ Strategy A: Parallel pointer decode (target: 3x speedup → 150ms)
- ⏳ Strategy B: Switch optimization (target: 2x speedup → 75ms)
- ⏳ Strategy C: Inline operations (target: 1.5x speedup → 50ms)
- 🎯 Target: RPN < 50ms (10x total speedup)

### Final Goal
- 🎯 **RPN < 50ms** (competitive with fused for debugging/development)
- ✅ **Fused = 9.29ms** (production default, already fastest)

---

## Performance Comparison Table (Current)

| Backend | Implementation | Latency (6 steps) | Speedup vs PTX | Use Case |
|---------|---------------|-------------------|----------------|----------|
| **Fused** ✅ | Single kernel, no interpretation | **9.29ms** | **1.12x** | **Production** (fastest) |
| PTX | 8 specialized kernels | 10.39ms | 1.0x | Baseline reference |
| RPN (current) | Tier-3 interpreter + pre-built | 500ms | 0.02x ❌ | Needs kernel optimization |
| RPN (target) | Tier-3 optimized interpreter | **<50ms** | **0.2x** | Development/debugging |

**Timing Breakdown (RPN current, 6 steps)**:
- Build/host update: 0.6ms (0.1%)
- Memcpy: 0.15ms
- **Kernel execute: 500ms (99.8%)** ⚠️

---

## Key Technical Details

### Fused Kernel Architecture
- **File**: [knowledge3d/cranium/ptx/trm_step_fused.cu](knowledge3d/cranium/ptx/trm_step_fused.cu:1)
- **Launch params**: `grid=(1,1,1), block=(256,1,1)`
- **Workspace**: 3168 floats (12,672 bytes) passed as global memory parameter
- **Operations**: 8 inline warp-level functions (no `__syncthreads__` overhead)
- **Memory pattern**: Sequential reads (coalesced), minimal bank conflicts

### RPN Current Bottleneck (CONFIRMED)
- **30 opcodes per step**: 26 pointer literals + 4 math ops
- **Sequential interpretation**: Switch-based dispatch in kernel (~500ms)
- **Synchronization overhead**: Each opcode likely has `__syncthreads__()`
- **Python overhead**: ELIMINATED (pre-built programs reduce to 0.6ms = 0.1%)
- **Root cause**: 99.8% of time spent in GPU kernel's switch/dispatch loop

### Why Fused is Faster than PTX
- **Launch overhead savings**: 8 launches → 1 launch = 420µs saved
- **Memory coalescing**: Single kernel has better cache locality
- **No sync between ops**: Warp-level execution, no global barriers
- **Optimized grid/block**: 256 threads optimal for TRM operations

---

## Environment Setup

### GPU Requirements
- CUDA 12.4+ (RTX 3060, 12GB VRAM)
- SM_86 architecture (Ampere)

### Python Environment
- Python 3.10+
- NumPy 1.24+
- pytest with `@pytest.mark.gpu` support

### Key Environment Variables
- `K3D_USE_RPN_TRM=1` - Enable RPN backend
- `K3D_USE_FUSED_TRM=1` - Enable fused kernel (fastest)
- `CUDA_VISIBLE_DEVICES=0` - GPU selection

### Running Tests
```bash
# All GPU tests (requires CUDA):
pytest tests/test_trm_fused_parity.py tests/test_trm_rpn_gpu.py -v

# Performance benchmark (compare all backends):
pytest tests/benchmarks/test_trm_launcher_performance.py -vs

# Skip GPU tests (for CI/CD):
pytest -m "not gpu"
```

---

## Communication Protocol

### After Kernel Profiling (1-2 hours from now)
**Report format**:
```
Nsight Systems Profile Results:
================================
Kernel: modular_rpn_kernel_extended
Calls: 6 (one per TRM step)
Total time: XXX ms
Per-call time: XX ms
Grid: (X, X, X)
Block: (X, X, X)

Nsight Compute Profile Results:
================================
Occupancy: XX%
Registers per thread: XX
Warp execution efficiency: XX%
Memory throughput: XX GB/s
Bottleneck: [pointer decode / switch dispatch / memory access / synchronization]
```

**What to share**:
1. Full nsys kernel summary
2. Full ncu metrics (occupancy, registers, warp efficiency)
3. Interpretation: "Bottleneck is [specific issue]"
4. Next step: "Proceeding with [Strategy A/B/C]"

### After Each Optimization Strategy
**Report format**:
```
Optimization: Strategy A - Parallel Pointer Decode
===================================================
Before: 500 ms
After: XX ms
Speedup: X.Xx
Parity test: [PASS/FAIL]
Target (<50ms): [ON TRACK / ACHIEVED / NEEDS MORE WORK]

Next: [Strategy B / Strategy C / DONE]
```

---

## Quick Reference: File Locations

```
knowledge3d/
├── cranium/
│   ├── ptx/
│   │   ├── trm_step_fused.cu         # Fused kernel source ✅
│   │   ├── trm_step_fused.ptx        # Compiled fused kernel ✅
│   │   └── modular_rpn_kernel_extended.cu  # Tier-3 interpreter (may need optimization)
│   ├── sovereign/
│   │   └── trm_launcher.py           # Three-backend launcher ✅ (add timing here)
│   └── bridges/
│       └── advanced_rpn.py           # Tier-3 bridge (add execute_prebuilt())
tests/
├── test_trm_fused_parity.py          # Fused kernel validation ✅
├── test_trm_rpn_gpu.py               # RPN GPU tests ✅
└── benchmarks/
    └── test_trm_launcher_performance.py  # Performance comparison ✅
reports/
├── RPN_PHASE1A_PROGRESS.md           # Progress report (update after optimization)
└── STEP13_B_TESTING_AND_BENCHMARKS.md  # Overall benchmarks
TEMP/
├── CODEX_RPN_TIER3_OPTIMIZATION_URGENT.md  # Original optimization prompt
├── TRM_RPN_PERFORMANCE_ANALYSIS.md         # Detailed analysis
└── CODEX_HANDOFF_NEXT_SESSION.md           # This document
```

---

## What Daniel Expects

**Short-term** (Next 24-48 hours):
1. ✅ DONE: Timing breakdown confirmed kernel is bottleneck (500ms = 99.8%)
2. ⏳ Kernel profiling with nsys + ncu to identify specific hotspots
3. ⏳ Begin optimization (Strategy A: parallel pointer decode)

**Medium-term** (This week):
- RPN performance < 50ms (10x speedup from current 500ms)
- Strategies A, B, C implemented incrementally with testing
- Kernel profiling data showing measurable improvements

**Long-term** (Phase 1B-5):
- RPN proven viable for Phase 1B (ThinkingTag tensor ops)
- Fused kernel as default for production TRM operations (already achieved!)
- RPN interpreter optimized for debugging/development use
- Integration map proceeds with confidence (no performance concerns)

---

## Critical Success Factors

### ✅ What's Already Working
- Fused kernel beats PTX (9.29ms < 10.39ms) ✅
- Numerical parity validated (L2 error < 1e-5) ✅
- Three-backend architecture clean and extensible ✅
- Tests passing, GitHub Actions fixed ✅
- Pre-built programs eliminate Python overhead (0.1%) ✅

### ⚠️ What Needs Attention
- RPN 50x slower (500ms vs 10ms baseline)
- Root cause CONFIRMED: GPU kernel interpreter (99.8% of time)
- **Kernel profiling + optimization is critical next step**

### 🎯 Victory Condition
- RPN < 50ms (acceptable for debugging/development)
- Fused = default (already achieved - 1.12x faster than PTX!)
- Phase 1B can proceed with confidence

---

## Bottom Line for Next Session

**Start Here**:
1. ✅ DONE: Pre-built programs eliminate Python overhead (0.1%)
2. ✅ DONE: Confirmed kernel is the bottleneck (500ms = 99.8%)
3. ⏳ **NEXT**: Profile kernel with nsys + ncu (1-2 hours)
4. ⏳ **THEN**: Optimize based on profiling data:
   - Strategy A: Parallel pointer decode (target: 3x speedup)
   - Strategy B: Switch optimization (target: 2x speedup)
   - Strategy C: Inline operations (target: 1.5x speedup)

**The fused kernel is already the production victory** - 9.29ms proves RPN architecture is sound and beats PTX!

**The interpreter optimization is for development/debugging** - Target <50ms makes RPN viable for exploration while keeping fused kernel as the fast path.

**You're crushing it, Codex!** Two major achievements already:
1. Fused kernel beats PTX (9.29ms vs 10.39ms)
2. Pre-built programs eliminate Python bottleneck (0.6ms)

Now let's tackle the kernel interpreter with data-driven optimization. Profile first, then optimize! 🚀

---

*Handoff prepared by: Claude*
*Date: October 15, 2025*
*Status: Ready for next Codex session*
*Priority: Timing breakdown → identify bottleneck → optimize accordingly*
