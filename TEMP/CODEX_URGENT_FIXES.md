# Codex Urgent Fixes - Profiling & Tier-1 Bug

**Date**: October 16, 2025
**Priority**: CRITICAL - Unblock profiling and fix Tier-1 regressions
**Context**: Mid-session snapshot shows two blockers preventing progress

---

## Issue 1: Profiling Blocked (ERR_NVGPUCTRPERM) - IMMEDIATE FIX

### Problem
- `nsys stats` rejects `.qdstrm` (missing importer)
- `ncu` fails with `ERR_NVGPUCTRPERM` (GPU counter permissions)

### Root Cause
NVIDIA GPU profiling counters are restricted by default for security. Need elevated permissions.

### Solution A: Temporary Permission Fix (Quick - 2 minutes)

**Ask Daniel to run** (requires sudo):
```bash
# Enable GPU profiling counters (temporary, lost on reboot)
sudo sh -c 'echo 1 >/proc/sys/kernel/perf_event_paranoid'

# Enable NVIDIA profiling
sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0
```

**Verify it works**:
```bash
# Check perf_event_paranoid
cat /proc/sys/kernel/perf_event_paranoid  # Should show: 1 or -1

# Test ncu
ncu --query-metrics
```

**Now re-run profiling**:
```bash
# Fresh profile with proper export
nsys profile --stats=true --force-overwrite=true \
    --export sqlite \
    -o tier3_fresh_profile \
    pytest tests/benchmarks/test_trm_launcher_performance.py::test_trm_launcher_rpn_vs_ptx_benchmark -s -k rpn

# Analyze
nsys stats tier3_fresh_profile.sqlite --report cuda_gpu_kern_sum

# Detailed metrics with ncu (now should work!)
ncu --set full --target-processes all \
    --kernel-name modular_rpn_kernel_extended \
    -o TEMP/tier3_ncu \
    pytest tests/benchmarks/test_trm_launcher_performance.py::test_trm_launcher_rpn_vs_ptx_benchmark -s -k rpn
```

---

### Solution B: Analyze Existing .qdstrm (Workaround)

If permissions can't be changed immediately, try converting existing profile:

```bash
# Try sqlite export from existing file
nsys export rpn_kernel_profile.qdstrm --type sqlite -o rpn_kernel_profile.sqlite

# If that works, analyze
nsys stats rpn_kernel_profile.sqlite --report cuda_gpu_kern_sum

# Alternative: Use nsight-systems GUI (if X11 available)
nsight-sys rpn_kernel_profile.qdstrm
```

---

### Solution C: Extract Metrics from Launcher Output (Immediate)

**You already have timing data from the benchmark!** Parse it:

```bash
# Run benchmark with verbose output
pytest tests/benchmarks/test_trm_launcher_performance.py -vs 2>&1 | tee TEMP/tier3_timing.txt

# Extract key metrics:
grep -E "(RPN|PTX|Fused|GPU execution|ms)" TEMP/tier3_timing.txt
```

**Expected output**:
```
Fused TRM (6 steps): 9.51 ms
PTX TRM (6 steps): 10.33 ms
RPN TRM (6 steps): 10.63 ms
  └─ GPU execution: 7.4-8.1 ms
  └─ Build/host: 0.6 ms
  └─ Memcpy: 0.15 ms
```

**This is enough to proceed with Phase 2/3** while waiting for detailed profiling!

---

## Issue 2: Tier-1 Literal Indexing Bug - CRITICAL FIX

### Problem
> "literal indexing is incorrect (all scalars/vectors resolve to slot i)"

**Root cause**: Shared memory stack indices not properly managed when threads are parallel.

### Current Broken Code Pattern
```cuda
// BROKEN: Each thread has its own i, race condition!
for (int i = 0; i < num_ops; i++) {
    switch (op_codes[i]) {
        case OP_LITERAL:
            // Thread 0 and thread 1 both try to use i
            // Results in wrong scalar being loaded
            float val = scalars[i];  // ❌ Wrong index!
            break;
    }
}
```

### Fix: Thread-0-Only Scalar Indexing

**File**: `knowledge3d/cranium/kernels/simple_rpn_kernel.cu`

**Pattern to implement**:
```cuda
extern "C" __global__ void simple_rpn_kernel(
    const uint16_t* op_codes,
    const float* scalars,
    int num_ops,
    float* global_memory,
    float* stack_snapshot,
    int instance_id
) {
    // Shared memory for stack
    __shared__ float shared_stack[64];
    __shared__ int stack_ptr;

    // ✅ ADD: Shared scalar/vector indices (thread 0 manages these)
    __shared__ int scalar_index;
    __shared__ int vector_index;

    // Thread 0 initializes
    if (threadIdx.x == 0) {
        stack_ptr = 0;
        scalar_index = 0;  // ✅ Initialize
        vector_index = 0;  // ✅ Initialize
    }
    __syncthreads();

    // Opcode loop (thread 0 only for sequential control)
    if (threadIdx.x == 0) {
        for (int pc = 0; pc < num_ops; pc++) {
            uint16_t opcode = op_codes[pc];

            switch (opcode) {
                case OP_PUSH_LITERAL:
                    // ✅ Use shared scalar_index, increment after
                    shared_stack[stack_ptr++] = scalars[scalar_index++];
                    break;

                case OP_LOAD_VEC:
                    {
                        // ✅ Use shared vector_index
                        void* vec_ptr = /* decode from scalars[vector_index...] */;
                        shared_stack[stack_ptr++] = (float)(uintptr_t)vec_ptr;
                        vector_index += 3;  // ✅ Pointer = 3 floats
                    }
                    break;

                case OP_ADD:
                    // Parallel operation - ALL threads participate
                    {
                        __syncthreads();  // Ensure thread 0 finished setup

                        if (stack_ptr >= 2) {
                            // Thread 0 pops pointers
                            if (threadIdx.x == 0) {
                                /* pop logic */
                            }
                            __syncthreads();

                            // ALL threads do parallel add
                            for (int i = threadIdx.x; i < 512; i += blockDim.x) {
                                result[i] = a[i] + b[i];
                            }
                            __syncthreads();

                            // Thread 0 pushes result
                            if (threadIdx.x == 0) {
                                /* push logic */
                            }
                        }
                        __syncthreads();
                    }
                    break;
            }
        }
    } else {
        // Other threads wait for parallel operations
        for (int pc = 0; pc < num_ops; pc++) {
            __syncthreads();  // Sync before each op

            uint16_t opcode = op_codes[pc];

            // Only participate in parallel ops
            if (opcode == OP_ADD || opcode == OP_MUL || opcode == OP_DOT) {
                // Parallel work here
            }

            __syncthreads();  // Sync after each op
        }
    }
}
```

### Key Changes

1. **Add shared indices**:
   ```cuda
   __shared__ int scalar_index;
   __shared__ int vector_index;
   ```

2. **Thread 0 manages sequential ops**:
   - Literal pushes use `scalars[scalar_index++]`
   - No race conditions

3. **All threads participate in parallel ops**:
   - ADD, MUL, DOT, etc. use `for (int i = threadIdx.x; ...)`
   - Proper `__syncthreads()` before/after

4. **Non-zero threads wait**:
   - Sync at each opcode boundary
   - Only execute when parallel op detected

---

### Specific Fix for Your Code

**Locate this pattern** in `simple_rpn_kernel.cu`:
```cuda
// FIND:
for (int i = 0; i < num_ops; i++) {
    switch (op_codes[i]) {
        case OP_LITERAL:
            float val = scalars[i];  // ❌ BUG HERE
```

**Replace with**:
```cuda
// Thread 0 initializes
if (threadIdx.x == 0) {
    scalar_idx = 0;
}
__syncthreads();

// Thread 0 executes sequential ops
if (threadIdx.x == 0) {
    for (int i = 0; i < num_ops; i++) {
        switch (op_codes[i]) {
            case OP_LITERAL:
                float val = scalars[scalar_idx++];  // ✅ FIXED
                shared_stack[stack_ptr++] = val;
                break;
```

---

### Testing the Fix

**After rebuilding PTX**:
```bash
cd knowledge3d/cranium/kernels
nvcc -ptx -arch=sm_86 -O3 simple_rpn_kernel.cu -o ../ptx/simple_rpn_kernel.ptx
```

**Run tests**:
```bash
# Should pass now
pytest tests/test_rpn_tier1.py -v
pytest tests/test_sovereign_rpn.py -v -k tier1
```

**Expected**: All literal push/pop operations work correctly, no more zero fallbacks.

---

## Issue 3: Tier-2 Kernel Coverage (MEDIUM PRIORITY)

### Problem
> "new cooperative CUDA implementation (10 arithmetic/vector ops) compiled to PTX but unvalidated"

The new Tier-2 kernel only has 10 ops, but `ModularRPNEngine.OPCODES` expects full coverage.

### Solution: Complete Opcode Implementation

**File**: `knowledge3d/cranium/kernels/modular_rpn_kernel_extended.cu`

**Find the new parallel kernel section** (likely around lines 1-200)

**Ensure ALL these opcodes are implemented**:

```cuda
// Basic arithmetic (already done?)
case OP_ADD:      // ✅
case OP_SUB:      // ✅
case OP_MUL:      // ✅
case OP_DIV:      // ✅

// Stack operations (sequential, thread 0 only)
case OP_PUSH:     // Need to add
case OP_POP:      // Need to add
case OP_DUP:      // Need to add
case OP_SWAP:     // Need to add
case OP_OVER:     // Need to add
case OP_ROT:      // Need to add

// Memory operations (parallel)
case OP_LOAD:     // Need to add (coalesced load)
case OP_STORE:    // Need to add (coalesced store)
case OP_MEMCPY:   // Need to add (cooperative copy)

// Reductions (parallel with shared memory)
case OP_DOT:      // ✅ (if already done)
case OP_SUM:      // Need to add
case OP_MAX:      // Need to add
case OP_MIN:      // Need to add
case OP_NORM:     // Need to add

// Vector operations (parallel)
case OP_BROADCAST: // Need to add
case OP_SCALE:     // Need to add
case OP_NEGATE:    // Need to add

// Transforms (if in Tier-2)
case OP_ROTATE:    // Need to add
case OP_TRANSLATE: // Need to add
```

**Pattern for each opcode**:

**Stack ops** (thread 0 only):
```cuda
case OP_DUP:
    if (threadIdx.x == 0) {
        if (stack_ptr > 0) {
            shared_stack[stack_ptr] = shared_stack[stack_ptr - 1];
            stack_ptr++;
        }
    }
    __syncthreads();
    break;
```

**Parallel reduction** (all threads):
```cuda
case OP_SUM:
    {
        if (threadIdx.x == 0) {
            void* vec_ptr = pop_from_stack();
        }
        __syncthreads();

        __shared__ float partial_sums[256];
        float thread_sum = 0.0f;

        for (int i = threadIdx.x; i < size; i += blockDim.x) {
            thread_sum += vec[i];
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
        __syncthreads();
    }
    break;
```

**Rebuild and test**:
```bash
nvcc -ptx -arch=sm_86 -O3 modular_rpn_kernel_extended.cu -o ../ptx/modular_rpn_kernel_extended.ptx
pytest tests/test_sovereign_rpn.py -v -k tier2
```

---

## Immediate Action Plan (Next 2 Hours)

### Step 1: Fix Profiling (15 min)
**Daniel**: Run permission fix commands (see Solution A above)

**Codex**: Re-run profiling:
```bash
nsys profile --export sqlite -o tier3_fresh pytest tests/benchmarks/test_trm_launcher_performance.py -s -k rpn
nsys stats tier3_fresh.sqlite --report cuda_gpu_kern_sum
```

**OR** (if permissions blocked): Use Solution C (parse benchmark output)

---

### Step 2: Fix Tier-1 Bug (30 min)

**Critical changes**:
1. Add `__shared__ int scalar_index; __shared__ int vector_index;`
2. Initialize in thread 0: `scalar_index = 0; vector_index = 0;`
3. Replace all `scalars[i]` with `scalars[scalar_index++]` in literal ops
4. Wrap sequential ops in `if (threadIdx.x == 0) { ... }`
5. Wrap parallel ops with proper `__syncthreads()`

**Rebuild PTX**:
```bash
cd knowledge3d/cranium/kernels
nvcc -ptx -arch=sm_86 -O3 simple_rpn_kernel.cu -o ../ptx/simple_rpn_kernel.ptx
```

**Test**:
```bash
pytest tests/test_rpn_tier1.py -v
pytest tests/test_sovereign_rpn.py -v -k tier1
```

**Success criteria**: All tests pass, no zero fallbacks

---

### Step 3: Complete Tier-2 Opcodes (1 hour)

1. List all missing opcodes in kernel
2. Implement using patterns above (stack ops, parallel reductions, etc.)
3. Rebuild PTX
4. Test with `pytest tests/test_sovereign_rpn.py -v -k tier2`

---

### Step 4: Report Progress (15 min)

**Format**:
```
URGENT FIXES STATUS
===================

✅ Profiling: [FIXED / WORKAROUND / BLOCKED]
   - Method: [Permissions / SQLite export / Benchmark parsing]
   - Tier-3 metrics: [kernel time, occupancy, etc.]

✅ Tier-1 literal bug: [FIXED / IN PROGRESS]
   - Tests: pytest tests/test_rpn_tier1.py → [PASS/FAIL]
   - Regressions resolved: [YES/NO]

⏳ Tier-2 coverage: [COMPLETE / IN PROGRESS]
   - Opcodes implemented: [X/Y]
   - Tests: pytest tests/test_sovereign_rpn.py -k tier2 → [PASS/FAIL]

Next: [Continue with benchmarks / Need more help]
```

---

## Code Reference: Complete Tier-1 Kernel Template

**File**: `knowledge3d/cranium/kernels/simple_rpn_kernel.cu`

```cuda
extern "C" __global__ void simple_rpn_kernel(
    const uint16_t* op_codes,
    const float* scalars,
    int num_ops,
    float* global_memory,
    float* stack_snapshot,
    int instance_id
) {
    // Shared memory
    __shared__ float shared_stack[64];
    __shared__ int stack_ptr;
    __shared__ int scalar_index;
    __shared__ int vector_index;

    // Thread 0 initializes
    if (threadIdx.x == 0) {
        stack_ptr = 0;
        scalar_index = 0;
        vector_index = 0;
    }
    __syncthreads();

    // Main execution
    for (int pc = 0; pc < num_ops; pc++) {
        uint16_t opcode = op_codes[pc];

        // Sequential operations (thread 0 only)
        if (threadIdx.x == 0) {
            switch (opcode) {
                case OP_PUSH_LITERAL:
                    shared_stack[stack_ptr++] = scalars[scalar_index++];
                    break;

                case OP_POP:
                    if (stack_ptr > 0) stack_ptr--;
                    break;

                case OP_DUP:
                    if (stack_ptr > 0) {
                        shared_stack[stack_ptr] = shared_stack[stack_ptr - 1];
                        stack_ptr++;
                    }
                    break;
            }
        }
        __syncthreads();

        // Parallel operations (all threads)
        switch (opcode) {
            case OP_ADD:
                {
                    // Thread 0 pops operands
                    float* a_ptr;
                    float* b_ptr;
                    float* result_ptr;

                    if (threadIdx.x == 0) {
                        // Pop and allocate
                    }
                    __syncthreads();

                    // All threads add
                    for (int i = threadIdx.x; i < 512; i += blockDim.x) {
                        result_ptr[i] = a_ptr[i] + b_ptr[i];
                    }
                    __syncthreads();

                    // Thread 0 pushes result
                    if (threadIdx.x == 0) {
                        // Push result pointer
                    }
                    __syncthreads();
                }
                break;

            case OP_DOT:
                {
                    __shared__ float partial_sums[256];

                    // Similar pattern...
                }
                break;
        }
        __syncthreads();
    }
}
```

---

## Bottom Line

**Two critical fixes needed before proceeding**:

1. **Profiling**: Either get permissions OR use benchmark output as workaround
2. **Tier-1 bug**: Add shared indices, thread-0 control for sequential ops

**Once fixed**, you can proceed with:
- Phase 1: Analyze Tier-3 metrics (even from benchmark output)
- Phase 2: Complete Tier-2 opcode coverage
- Phase 3: Run benchmarks and document speedups

**You're 80% there!** Just need these two fixes to unblock the rest of the work. 🚀

---

*Prepared by: Claude*
*Date: October 16, 2025*
*Priority: URGENT - Unblock Codex's progress*
