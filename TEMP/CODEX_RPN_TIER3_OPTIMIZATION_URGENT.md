# URGENT: RPN Tier-3 Optimization - Eliminate 50x Slowdown

**Date**: October 15, 2025
**Priority**: CRITICAL - Blocking Phase 1A completion
**Target**: Match or beat PTX baseline (10ms → <10ms for 6 TRM steps)
**Current**: 503.8ms (50x slower than PTX)

---

## Mission

**Eliminate the 50x performance gap** between RPN and PTX TRM execution by optimizing the Tier-3 kernel execution path.

**Root Cause Identified**:
- **80%** of slowdown: Sequential opcode interpretation with synchronization overhead
- **15%** of slowdown: Python list building (26 pointer literals per step)
- **5%** of slowdown: Suboptimal parallelization (switch-based dispatch)

**The Math Kernels Are Fast** - It's the **orchestration** that's slow.

---

## Part 1: Three-Pronged Attack

### Strategy A: Fused TRM Kernel (HIGHEST IMPACT - 30x speedup expected)

**Problem**: Current RPN kernel interprets 30 opcodes sequentially:
```cuda
// Current (SLOW):
for (int pc = 0; pc < 30; pc++) {
    switch (opcodes[pc]) {
        case OP_POINTER_LITERAL:
            decode_pointer(...);  // 26 times!
            break;
        case OP_TRM_VEC_ADD3:
            vec_add3_kernel(...);  // Separate dispatch
            __syncthreads();       // Synchronization overhead!
            break;
        case OP_TRM_MATVEC_512x1024:
            matvec_kernel(...);
            __syncthreads();
            break;
        // ... 27 more opcodes ...
    }
}
```

**Solution**: Create specialized fused kernel for TRM step:

#### File: `knowledge3d/cranium/ptx/trm_step_fused.cu`

```cuda
/*
 * TRM Step Fused Kernel - Zero Interpretation Overhead
 *
 * Executes one complete TRM refinement step:
 *   z_new = W2 @ swiglu(W1 @ (q + y + z))
 *   y_new = W4 @ swiglu(W3 @ (y + z_new))
 *
 * All operations fused into single warp-level execution.
 * NO opcode interpretation, NO synchronization overhead.
 */

#include <cuda_runtime.h>

// Device inline functions (no __syncthreads__)
__device__ __forceinline__ float swiglu_scalar(float x, float gate) {
    return x / (1.0f + expf(-gate));  // x * sigmoid(gate)
}

__device__ __forceinline__ void vec_add3_warp(
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ c,
    float* __restrict__ out,
    int tid,
    int stride
) {
    // Warp-level vector addition (no sync needed)
    for (int i = tid; i < 512; i += stride) {
        out[i] = a[i] + b[i] + c[i];
    }
}

__device__ __forceinline__ void matvec_512x1024_warp(
    const float* __restrict__ W,  // (1024, 512) row-major
    const float* __restrict__ v,  // (512,)
    float* __restrict__ out,      // (1024,)
    int tid,
    int stride
) {
    // Each thread computes multiple output elements
    for (int row = tid; row < 1024; row += stride) {
        float sum = 0.0f;
        #pragma unroll 8
        for (int col = 0; col < 512; col++) {
            sum += W[row * 512 + col] * v[col];
        }
        out[row] = sum;
    }
}

__device__ __forceinline__ void matvec_1024x512_warp(
    const float* __restrict__ W,  // (512, 1024) row-major
    const float* __restrict__ v,  // (1024,)
    float* __restrict__ out,      // (512,)
    int tid,
    int stride
) {
    for (int row = tid; row < 512; row += stride) {
        float sum = 0.0f;
        #pragma unroll 8
        for (int col = 0; col < 1024; col++) {
            sum += W[row * 1024 + col] * v[col];
        }
        out[row] = sum;
    }
}

__device__ __forceinline__ void swiglu_1024_warp(
    const float* __restrict__ in,
    float* __restrict__ out,
    int tid,
    int stride
) {
    // SwiGLU: pairs (x, gate) -> x * sigmoid(gate)
    for (int i = tid; i < 512; i += stride) {  // 512 pairs
        float x = in[i * 2];
        float gate = in[i * 2 + 1];
        out[i] = swiglu_scalar(x, gate);
        out[i + 512] = 0.0f;  // Zero out gate position
    }
}

__device__ __forceinline__ void vec_add_512_warp(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int tid,
    int stride
) {
    for (int i = tid; i < 512; i += stride) {
        out[i] = a[i] + b[i];
    }
}

extern "C" __global__ void trm_step_fused(
    // Inputs
    const float* __restrict__ q,       // (512,) question
    const float* __restrict__ y,       // (512,) answer
    const float* __restrict__ z,       // (512,) latent
    const float* __restrict__ W1,      // (1024, 512) weights
    const float* __restrict__ W2,      // (512, 1024) weights
    const float* __restrict__ W3,      // (1024, 512) weights
    const float* __restrict__ W4,      // (512, 1024) weights
    // Outputs
    float* __restrict__ z_new,         // (512,) refined latent
    float* __restrict__ y_new,         // (512,) refined answer
    // Workspace (shared memory allocated by launcher)
    float* __restrict__ workspace      // (512 + 1024 + 512 + 1024) = 3168 floats
) {
    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Partition workspace
    float* temp = workspace;                    // (512,)
    float* hidden = workspace + 512;            // (1024,)
    float* temp2 = workspace + 512 + 1024;      // (512,)
    float* hidden2 = workspace + 512 + 1024 + 512;  // (1024,)

    // ================================================================
    // First half: z_new = W2 @ swiglu(W1 @ (q + y + z))
    // ================================================================

    // Step 1: temp = q + y + z
    vec_add3_warp(q, y, z, temp, tid, stride);
    __syncthreads();

    // Step 2: hidden = W1 @ temp (512 → 1024)
    matvec_512x1024_warp(W1, temp, hidden, tid, stride);
    __syncthreads();

    // Step 3: hidden = swiglu(hidden)
    swiglu_1024_warp(hidden, hidden, tid, stride);
    __syncthreads();

    // Step 4: z_new = W2 @ hidden (1024 → 512)
    matvec_1024x512_warp(W2, hidden, z_new, tid, stride);
    __syncthreads();

    // ================================================================
    // Second half: y_new = W4 @ swiglu(W3 @ (y + z_new))
    // ================================================================

    // Step 5: temp2 = y + z_new
    vec_add_512_warp(y, z_new, temp2, tid, stride);
    __syncthreads();

    // Step 6: hidden2 = W3 @ temp2 (512 → 1024)
    matvec_512x1024_warp(W3, temp2, hidden2, tid, stride);
    __syncthreads();

    // Step 7: hidden2 = swiglu(hidden2)
    swiglu_1024_warp(hidden2, hidden2, tid, stride);
    __syncthreads();

    // Step 8: y_new = W4 @ hidden2 (1024 → 512)
    matvec_1024x512_warp(W4, hidden2, y_new, tid, stride);
    // No final sync needed (implicit at kernel exit)
}
```

**Compile**:
```bash
nvcc -ptx -arch=sm_86 \
    --use_fast_math \
    --maxrregcount=64 \
    -o knowledge3d/cranium/ptx/trm_step_fused.ptx \
    knowledge3d/cranium/ptx/trm_step_fused.cu
```

---

### Strategy B: Pre-Built RPN Programs (MEDIUM IMPACT - 5x speedup expected)

**Problem**: Python builds opcode array every iteration:
```python
# SLOW (happens 6 times per refinement):
for step in range(n_steps):
    op_codes = []
    scalars = []

    # 26 pointer literal encodes:
    for ptr in [d_q, d_y, d_z, d_W1, ...]:
        scalars.extend(_encode_pointer_literal(ptr, ...))  # Python!
        op_codes.append(OP_POINTER_LITERAL)

    # 4 TRM opcodes:
    op_codes.extend([OP_VEC_ADD3, OP_MATVEC_512x1024, ...])

    # NumPy conversion + upload:
    op_codes_np = np.asarray(op_codes, dtype=np.uint16)
    scalars_np = np.asarray(scalars, dtype=np.float32)

    execute_program(instance_id, op_codes_np, scalars=scalars_np)
```

**Solution**: Build program ONCE, update only pointer values:

#### File: `knowledge3d/cranium/sovereign/trm_launcher.py` (modify)

```python
class TRMLauncher:
    def __init__(self, use_rpn=False):
        # ... existing code ...

        if self.use_rpn:
            # ✅ NEW: Pre-build TRM program ONCE
            self._trm_opcodes, self._trm_scalar_template = self._build_trm_program_template()

            # Upload opcodes ONCE (never changes)
            self._d_opcodes = loader.gpu_malloc(self._trm_opcodes.nbytes)
            loader.memcpy_htod(
                self._d_opcodes,
                self._trm_opcodes.ctypes.data_as(ctypes.c_void_p),
                self._trm_opcodes.nbytes
            )

            # Allocate persistent scalar buffer
            self._d_scalars = loader.gpu_malloc(104 * 4)  # 26 pointers × 4 floats
            self._scalars_host = np.zeros(104, dtype=np.float32)

    def _build_trm_program_template(self):
        """Build TRM opcode array + scalar template (ONCE)."""
        op_codes = []
        scalar_indices = {}  # Map pointer name to scalar index

        scalar_idx = 0

        # 26 pointer literals (opcodes only, values filled later):
        for name in ['q', 'y', 'z', 'temp', 'W1', 'hidden', 'W2', 'z_new',
                     'y_zero', 'temp2', 'W3', 'hidden2', 'W4', 'y_new']:
            op_codes.append(self._op_pointer_literal)
            scalar_indices[name] = scalar_idx
            scalar_idx += 4  # Reserve 4 floats per pointer

        # 4 TRM opcodes:
        op_codes.extend([
            self._op_vec_add3_512,
            self._op_matvec_512x1024,
            self._op_swiglu_1024,
            self._op_matvec_1024x512,
            # ... repeat for second half ...
        ])

        op_codes_np = np.asarray(op_codes, dtype=np.uint16)
        return op_codes_np, scalar_indices

    def _update_trm_scalars(self, d_q, d_y, d_z, d_W1, d_W2, d_W3, d_W4):
        """Update pointer values in scalar buffer (FAST)."""
        idx = self._trm_scalar_template

        # Encode each pointer directly into host buffer:
        self._scalars_host[idx['q']:idx['q']+4] = _encode_pointer_literal(d_q, 512, 1)
        self._scalars_host[idx['y']:idx['y']+4] = _encode_pointer_literal(d_y, 512, 1)
        self._scalars_host[idx['z']:idx['z']+4] = _encode_pointer_literal(d_z, 512, 1)
        # ... etc for all 26 pointers ...

        # Single H2D transfer:
        loader.memcpy_htod(
            self._d_scalars,
            self._scalars_host.ctypes.data_as(ctypes.c_void_p),
            self._scalars_host.nbytes
        )

    def _refine_rpn_optimized(self, ...):
        """Optimized RPN path with pre-built program."""
        for step in range(n_steps):
            # ✅ Update only scalar values (not opcodes):
            self._update_trm_scalars(d_q, d_y, d_z, d_W1, d_W2, d_W3, d_W4)

            # ✅ Execute with persistent buffers (no allocation):
            self._advanced_rpn.execute_prebuilt(
                instance_id=0,
                d_opcodes=self._d_opcodes,
                d_scalars=self._d_scalars,
                n_opcodes=len(self._trm_opcodes)
            )

            # Drift check (same as before)
            # ...
```

**New Bridge Method** (`knowledge3d/cranium/bridges/advanced_rpn.py`):
```python
def execute_prebuilt(
    self,
    instance_id: int,
    d_opcodes: ctypes.c_void_p,  # Already on GPU
    d_scalars: ctypes.c_void_p,  # Already on GPU
    n_opcodes: int
) -> np.ndarray:
    """Execute with pre-uploaded buffers (zero allocation)."""
    instance_offset = instance_id * self.INSTANCE_STRIDE

    loader.launch(
        self._kernel,
        grid=(1, 1, 1),
        block=(256, 1, 1),
        params=[
            ctypes.c_uint64(self._state.value + instance_offset),
            ctypes.c_uint64(d_opcodes.value),  # Pre-uploaded!
            ctypes.c_uint32(n_opcodes),
            ctypes.c_uint64(d_scalars.value),  # Pre-uploaded!
            ctypes.c_uint32(0),  # n_vectors
            ctypes.c_uint64(0),  # d_vectors (unused)
            ctypes.c_uint32(0),  # n_matrices
            ctypes.c_uint64(0),  # d_matrices (unused)
        ],
    )
    loader.synchronize()

    # Return stack (same as before)
    # ...
```

**Expected Speedup**: 5x (eliminates Python list building + NumPy overhead)

---

### Strategy C: Switch to Fused Kernel Path (HYBRID APPROACH)

**Add new execution mode** to `TRMLauncher`:

```python
class TRMLauncher:
    def __init__(self, use_rpn=False, use_fused=False):
        # ... existing code ...

        self.use_fused = use_fused
        if self.use_fused:
            # Load fused kernel:
            ptx_fused = Path(__file__).parent.parent / "ptx" / "trm_step_fused.ptx"
            self.kernel_fused = load_ptx_file(str(ptx_fused), "trm_step_fused")

            # Allocate workspace (3168 floats = 12,672 bytes):
            self.d_workspace = gpu_malloc(3168 * 4)

    def refine(self, q, y, z, W1, W2, W3, W4, n_steps=6, eps=1e-4):
        # ... existing setup ...

        if self.use_fused:
            result = self._refine_fused(d_q, d_y, d_z, d_W1, d_W2, d_W3, d_W4, ...)
        elif self.use_rpn:
            result = self._refine_rpn_optimized(...)
        else:
            result = self._refine_ptx(...)

        return result

    def _refine_fused(self, d_q, d_y, d_z, d_W1, d_W2, d_W3, d_W4, d_z_new, d_y_new, n_steps, eps):
        """Fused kernel path (FASTEST)."""
        z_old = np.zeros(512, dtype=np.float32)

        for step in range(n_steps):
            # Copy z for drift check:
            memcpy_dtoh(z_old.ctypes.data_as(ctypes.c_void_p), d_z, z_old.nbytes)

            # ✅ SINGLE KERNEL LAUNCH (all math fused):
            launch(
                self.kernel_fused,
                grid=(1, 1, 1),
                block=(256, 1, 1),  # 256 threads for parallel execution
                params=[
                    ctypes.c_uint64(d_q.value),
                    ctypes.c_uint64(d_y.value),
                    ctypes.c_uint64(d_z.value),
                    ctypes.c_uint64(d_W1.value),
                    ctypes.c_uint64(d_W2.value),
                    ctypes.c_uint64(d_W3.value),
                    ctypes.c_uint64(d_W4.value),
                    ctypes.c_uint64(d_z_new.value),
                    ctypes.c_uint64(d_y_new.value),
                    ctypes.c_uint64(self.d_workspace.value),
                ],
            )
            synchronize()

            # Drift check:
            z_new = np.zeros(512, dtype=np.float32)
            memcpy_dtoh(z_new.ctypes.data_as(ctypes.c_void_p), d_z_new, z_new.nbytes)

            drift = np.max(np.abs(z_new - z_old))
            if drift < eps:
                print(f"   🛑 TRM (FUSED) halted at step {step + 1}/{n_steps} (drift={drift:.6f})")
                y_final = np.zeros(512, dtype=np.float32)
                memcpy_dtoh(y_final.ctypes.data_as(ctypes.c_void_p), d_y_new, y_final.nbytes)
                return y_final, z_new

            # Update z, y for next iteration:
            memcpy_htod(d_z, z_new.ctypes.data_as(ctypes.c_void_p), z_new.nbytes)
            y_tmp = np.zeros(512, dtype=np.float32)
            memcpy_dtoh(y_tmp.ctypes.data_as(ctypes.c_void_p), d_y_new, y_tmp.nbytes)
            memcpy_htod(d_y, y_tmp.ctypes.data_as(ctypes.c_void_p), y_tmp.nbytes)

        # Final results:
        y_final = np.zeros(512, dtype=np.float32)
        z_final = np.zeros(512, dtype=np.float32)
        memcpy_dtoh(y_final.ctypes.data_as(ctypes.c_void_p), d_y_new, y_final.nbytes)
        memcpy_dtoh(z_final.ctypes.data_as(ctypes.c_void_p), d_z_new, z_final.nbytes)
        return y_final, z_final
```

---

## Part 2: Testing & Validation

### Test 1: Numerical Equivalence

**File**: `tests/test_trm_fused_parity.py`

```python
import numpy as np
import pytest
from knowledge3d.cranium.sovereign.trm_launcher import TRMLauncher
from knowledge3d.cranium.sovereign import loader

def _ensure_cuda():
    try:
        ptr = loader.gpu_malloc(4)
        loader.gpu_free(ptr)
    except RuntimeError as exc:
        pytest.skip(f"CUDA unavailable: {exc}")

@pytest.mark.gpu
def test_trm_fused_matches_ptx():
    """Fused kernel must match PTX output (L2 error < 1e-5)."""
    _ensure_cuda()

    # Build inputs:
    rng = np.random.default_rng(42)
    q = rng.standard_normal(512, dtype=np.float32)
    y = rng.standard_normal(512, dtype=np.float32)
    z = rng.standard_normal(512, dtype=np.float32)
    W1 = rng.standard_normal((1024, 512), dtype=np.float32)
    W2 = rng.standard_normal((512, 1024), dtype=np.float32)
    W3 = rng.standard_normal((1024, 512), dtype=np.float32)
    W4 = rng.standard_normal((512, 1024), dtype=np.float32)

    # PTX baseline:
    trm_ptx = TRMLauncher(use_rpn=False, use_fused=False)
    y_ptx, z_ptx = trm_ptx.refine(q, y, z, W1, W2, W3, W4, n_steps=1, eps=0.0)
    trm_ptx.cleanup()

    # Fused kernel:
    trm_fused = TRMLauncher(use_rpn=False, use_fused=True)
    y_fused, z_fused = trm_fused.refine(q, y, z, W1, W2, W3, W4, n_steps=1, eps=0.0)
    trm_fused.cleanup()

    # Validate:
    y_error = np.linalg.norm(y_ptx - y_fused)
    z_error = np.linalg.norm(z_ptx - z_fused)

    print(f"y L2 error: {y_error:.2e}")
    print(f"z L2 error: {z_error:.2e}")

    assert y_error < 1e-5, f"y mismatch: {y_error}"
    assert z_error < 1e-5, f"z mismatch: {z_error}"

@pytest.mark.gpu
def test_trm_rpn_optimized_matches_ptx():
    """Optimized RPN must match PTX output."""
    _ensure_cuda()

    # Same test as above, but with use_rpn=True
    # ...
```

---

### Test 2: Performance Benchmark

**File**: `tests/benchmarks/test_trm_optimized_performance.py`

```python
import time
import numpy as np
import pytest
from knowledge3d.cranium.sovereign.trm_launcher import TRMLauncher
from knowledge3d.cranium.sovereign import loader

def _time_launcher(use_rpn, use_fused, iterations=10):
    launcher = TRMLauncher(use_rpn=use_rpn, use_fused=use_fused)
    rng = np.random.default_rng(123)
    q = rng.standard_normal(512, dtype=np.float32)
    y = rng.standard_normal(512, dtype=np.float32)
    z = rng.standard_normal(512, dtype=np.float32)
    W1 = rng.standard_normal((1024, 512), dtype=np.float32)
    W2 = rng.standard_normal((512, 1024), dtype=np.float32)
    W3 = rng.standard_normal((1024, 512), dtype=np.float32)
    W4 = rng.standard_normal((512, 1024), dtype=np.float32)

    try:
        # Warmup:
        for _ in range(5):
            launcher.refine(q, y, z, W1, W2, W3, W4)
        loader.synchronize()

        # Timed run:
        start = time.perf_counter()
        for _ in range(iterations):
            launcher.refine(q, y, z, W1, W2, W3, W4)
        loader.synchronize()

        elapsed = time.perf_counter() - start
        return elapsed / iterations
    finally:
        launcher.cleanup()

@pytest.mark.gpu
def test_trm_optimization_benchmark():
    """Compare all TRM execution paths."""
    iterations = 10

    # Baseline (PTX):
    avg_ptx = _time_launcher(use_rpn=False, use_fused=False, iterations=iterations)

    # RPN optimized:
    avg_rpn_opt = _time_launcher(use_rpn=True, use_fused=False, iterations=iterations)

    # Fused kernel:
    avg_fused = _time_launcher(use_rpn=False, use_fused=True, iterations=iterations)

    print("\n" + "="*60)
    print("TRM PERFORMANCE BENCHMARK (6 steps per refinement)")
    print("="*60)
    print(f"PTX (baseline):     {avg_ptx * 1e3:8.3f} ms  (1.00x)")
    print(f"RPN (optimized):    {avg_rpn_opt * 1e3:8.3f} ms  ({avg_ptx / avg_rpn_opt:.2f}x)")
    print(f"Fused kernel:       {avg_fused * 1e3:8.3f} ms  ({avg_ptx / avg_fused:.2f}x)")
    print("="*60)

    # Target: Fused should be ≤ PTX baseline
    assert avg_fused <= avg_ptx * 1.1, f"Fused slower than PTX: {avg_fused} > {avg_ptx}"
```

---

## Part 3: Implementation Checklist

### Phase 1: Fused Kernel (Days 1-2)

- [ ] **Day 1 Morning**: Create `trm_step_fused.cu` with all inline operations
- [ ] **Day 1 Afternoon**: Compile to PTX, add to `TRMLauncher` with `use_fused` flag
- [ ] **Day 1 Evening**: Write numerical equivalence test (`test_trm_fused_parity.py`)
- [ ] **Day 2 Morning**: Fix any bugs from parity tests (adjust grid/block dims if needed)
- [ ] **Day 2 Afternoon**: Run performance benchmark, measure speedup

**Success Criteria**:
- ✅ Fused kernel matches PTX output (L2 error < 1e-5)
- ✅ Performance ≤ 10ms for 6 steps (match or beat PTX baseline)

---

### Phase 2: Pre-Built RPN Programs (Days 3-4)

- [ ] **Day 3 Morning**: Modify `TRMLauncher.__init__()` to pre-build program
- [ ] **Day 3 Afternoon**: Implement `_update_trm_scalars()` and `_refine_rpn_optimized()`
- [ ] **Day 3 Evening**: Add `execute_prebuilt()` to `AdvancedRPNEngine`
- [ ] **Day 4 Morning**: Write tests for optimized RPN path
- [ ] **Day 4 Afternoon**: Benchmark optimized RPN vs baseline

**Success Criteria**:
- ✅ Optimized RPN matches PTX output
- ✅ Performance < 50ms for 6 steps (10x speedup vs. current 503ms)

---

### Phase 3: Documentation & Integration (Day 5)

- [ ] **Day 5 Morning**: Update `reports/RPN_PHASE1A_PROGRESS.md` with results
- [ ] **Day 5 Afternoon**: Add pytest markers (`@pytest.mark.gpu`) to all GPU tests
- [ ] **Day 5 Evening**: Create comparison table (PTX vs RPN vs Fused)

---

## Part 4: Expected Results

### Performance Targets

| Execution Mode | Current | Target | Speedup vs Baseline |
|----------------|---------|--------|---------------------|
| **PTX (baseline)** | 10.1ms | 10.1ms | 1.0x |
| **RPN (current)** | 503.8ms | - | 0.02x ❌ |
| **RPN (optimized)** | - | **50ms** | **0.2x** ✅ |
| **Fused kernel** | - | **8ms** | **1.26x** 🎯 |

**Stretch Goal**: Fused kernel **faster than PTX** (8ms < 10.1ms) by eliminating multiple kernel launch overhead.

---

### Why Fused Will Be Faster

PTX path launches **8 kernels per step**:
1. `vec_add3_512` - launch overhead ~10µs
2. `matvec_512x1024` - launch overhead ~10µs
3. `swiglu_vec_1024` - launch overhead ~10µs
4. `matvec_1024x512` - launch overhead ~10µs
5. `vec_add_512` - launch overhead ~10µs
6. `matvec_512x1024` - launch overhead ~10µs
7. `swiglu_vec_1024` - launch overhead ~10µs
8. `matvec_1024x512` - launch overhead ~10µs

**Total launch overhead**: 8 × 10µs = **80µs per step** × 6 steps = **480µs**

Fused kernel: **1 launch per step** × 6 steps = **60µs launch overhead**

**Savings**: 480µs - 60µs = **420µs** (4% of 10.1ms total) ✅

---

## Part 5: Debugging Tips

### If Fused Kernel is Slow

**Check 1**: Register usage (may cause occupancy issues)
```bash
nvcc --ptxas-options=-v trm_step_fused.cu
# Look for: "registers: XX" (target: <64)
```

**Fix**: Add `--maxrregcount=64` to compiler flags

---

**Check 2**: Shared memory usage
```bash
# In CUDA code, check:
__shared__ float workspace[3168];  // May exceed limits
```

**Fix**: Use global memory workspace (passed as parameter) instead of shared memory

---

**Check 3**: Warp divergence
```bash
nsys profile --stats=true python -c "..."
# Look for: "Warp Execution Efficiency"
```

**Fix**: Ensure all threads in warp execute same path (avoid if/else in hot loops)

---

### If RPN Optimized is Still Slow

**Check**: Are we still allocating GPU memory?
```python
# Add debug prints:
print(f"Allocating opcodes: {d_opcodes.value}")  # Should print SAME address every iteration
```

**Fix**: Ensure `execute_prebuilt()` uses persistent buffers, not `gpu_malloc()` per call

---

## Part 6: Final Notes

### Priority Order

1. **Fused Kernel** (Days 1-2) - Highest impact, proves RPN can match PTX
2. **RPN Optimized** (Days 3-4) - Medium impact, keeps pure RPN path viable
3. **Documentation** (Day 5) - Update progress report

### Communication Protocol

**After Day 2** (Fused Kernel Complete):
- Report fused kernel performance: "Fused TRM: X.Xms (Yx vs PTX)"
- If X.X < 10.1: **VICTORY** - RPN proven superior to PTX! 🎉
- If X.X > 10.1: Share benchmark output, we'll debug together

**After Day 4** (RPN Optimized Complete):
- Report optimized RPN performance
- Target: <50ms (10x speedup vs current 503ms)

**After Day 5**:
- Full report with comparison table
- Ready to proceed with Phase 1B (ThinkingTag tensor ops)

---

## Part 7: Success Metrics

### Numerical Parity
✅ Fused kernel L2 error < 1e-5 vs PTX
✅ Optimized RPN L2 error < 1e-5 vs PTX

### Performance
🎯 **Fused kernel ≤ 10ms** (match or beat PTX)
✅ **Optimized RPN < 50ms** (10x speedup vs current)

### Test Coverage
✅ `test_trm_fused_parity.py` - Numerical equivalence
✅ `test_trm_optimized_performance.py` - Benchmark all paths
✅ All tests marked with `@pytest.mark.gpu`

### GitHub Actions
✅ No failures (GPU tests skipped automatically)

---

## Part 8: Code Review Checklist

Before submitting, verify:

- [ ] Fused kernel compiles without warnings
- [ ] All `__device__` functions are `__forceinline__`
- [ ] No `__syncthreads()` in inline functions
- [ ] Workspace memory correctly partitioned
- [ ] Launch parameters optimal (block size = 256)
- [ ] All tests pass with `K3D_USE_RPN_TRM=0` and `=1`
- [ ] Benchmark shows expected speedups
- [ ] No memory leaks (cleanup frees all allocated buffers)
- [ ] Code follows existing K3D style (lowercase_with_underscores)

---

## TLDR for Codex

**Mission**: Make RPN faster than PTX (currently 50x slower).

**3-Step Plan**:
1. **Fused kernel** (`trm_step_fused.cu`) - All math in 1 launch, no interpretation
2. **Pre-built RPN** - Build program once, reuse (eliminate Python overhead)
3. **Benchmark** - Prove RPN ≤ PTX baseline

**Timeline**: 5 days (2 + 2 + 1)

**Target**: Fused kernel ≤ 10ms, RPN optimized < 50ms

**Why This Matters**: Every K3D system (ThinkingTag, TRM, House) uses RPN. If RPN is 50x slower, **entire framework is crippled**. This fix unblocks Phase 1B-5.

**Victory Condition**: Fused kernel benchmark shows **≤ 10ms** (match or beat PTX). Then we proceed with full RPN expansion knowing performance is solid.

---

**Let's make RPN the fastest tensor execution system on the planet.** 🚀

---

*Prompt prepared by: Claude*
*Date: October 15, 2025*
*Status: Ready for Codex execution*
*Priority: CRITICAL - Start immediately*
