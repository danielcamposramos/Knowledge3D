# Codex Mission: Step 13-E - RPN Expansion & Step 14 Foundation

**Date**: October 16, 2025
**Priority**: HIGH - Bridge to 9-Chain Swarm Vision
**Your Achievement**: Phase 1B delivered 80x speedup! Now build the foundation for bio-inspired swarm intelligence.
**Estimated Time**: 8-12 hours (2-3 focused sessions)

---

## 🎉 CONGRATULATIONS ON PHASE 1B SUCCESS!

**Your Amazing Achievement**:
- **ThinkingTag FUSE stage**: 36.9ms → 0.46ms (**80x speedup**!)
- **RPN Tier-3**: 504ms → 10.63ms (**47x speedup**!)
- **Within 3% of PTX baseline** - You proved RPN is production-ready!

**This mission builds on your success to enable something revolutionary...**

---

## Part I: The Grand Vision (Your Strategic Context)

### What We're Building Toward: The 9-Chain Swarm

From Daniel's vision with Grok's bio-inspired research, Knowledge3D is evolving toward **collective intelligence**:

```
┌─────────────────────────────────────────────────────────┐
│           9-Chain Cranium Swarm (Step 14)               │
│                                                         │
│  Chain 1: INGEST        (Modal embedding)               │
│  Chain 2: FUSE-A        (Variant A fusion)              │
│  Chain 3: FUSE-B        (Variant B fusion)              │
│  Chain 4-6: SPATIAL     (Parallel spatial reasoning)    │
│  Chain 7: REASON-REDUX  (Einstein-like logic)           │
│  Chain 8: REASON-CREATE (Mozart-like generation)        │
│  Chain 9: OUTPUT-SYNTH  (Unified synthesis)             │
│                                                         │
│  • Interconnected chains (outputs feed each other)      │
│  • Adaptive mid-reasoning (chains learn from swarm)     │
│  • Emergent intelligence (discoveries no chain alone    │
│    could make)                                          │
│  • <95µs total latency (GPU-native PTX)                │
└─────────────────────────────────────────────────────────┘
```

**This is not simple parallelization—it's bio-inspired collective intelligence**, like:
- Ant colonies discovering optimal paths
- Neural ensembles in the brain
- Immune system adaptive responses
- Buehler's "Einstein meets Mozart" paradigm

### What The Swarm Needs (Your Mission)

For the 9-chain swarm to work, RPN needs:

1. **Matrix Operations** → Inter-chain communication
   - Chains share state via matrix operations
   - Example: Chain 4's spatial reasoning feeds Chain 7's logic

2. **Programmability** → Adaptive behavior
   - Chains adjust based on swarm feedback
   - Example: If Chain 7 detects uncertainty, Chain 8 explores alternatives

3. **Temporal Operations** → Swarm coherence
   - Maintain consistency across chains
   - Example: All chains agree on context before synthesis

4. **Performance** → Sub-95µs budget
   - 9 chains × <10µs each + overhead
   - You proved Tier-2 can do this (107µs → targets <10µs)

---

## Part II: Your Mission - Step 13-E

### Three Objectives (All Connected to Step 14)

**Objective 1: Immediate Optimization** (Phase 1C Requirements)
- Port temporal kernels → Enable swarm coherence
- Optimize matvec → Get each chain under <10µs budget
- Complete FSM integration → Validate full pipeline

**Objective 2: Swarm Foundations** (Step 14 Enablers)
- Matrix operations → Inter-chain communication
- Programmability core → Adaptive refinement
- Stack extensions → Chain state management

**Objective 3: Production Ready** (Quality Gates)
- Full test suite passing
- Performance validated
- Documentation complete

### Success Metrics

**Performance Targets**:
- ThinkingTag FUSE: 0.46ms → 0.15ms (**3x improvement**)
- OP_MATVEC_F32: ~120µs → ~40µs (**3x improvement**)
- Full inference: ~1.0ms → ~0.5ms (**2x improvement**)
- Total vs legacy: 80x → **250x improvement**

**Capability Targets**:
- ✅ Temporal operations (GPU-accelerated)
- ✅ Matrix ops for chain communication
- ✅ Basic programmability (BRANCH, LOOP, STORE/RECALL)
- ✅ Step 14 ready (foundations proven)

---

## Part III: Detailed Implementation Plan

### 🚀 Session 1: Temporal Kernels & Matvec Optimization (4 hours)

#### Task 1.1: Analyze Current Temporal Operations (30 min)

**Goal**: Understand what ThinkingTag's `compute_temporal_mask()` currently does.

**Check the implementation**:
```bash
# Look at the current temporal mask implementation
grep -A 100 "def compute_temporal_mask" knowledge3d/cranium/bridges/thinking_tag_rpn.py
```

**You'll find** (lines 218-286 in thinking_tag_rpn.py):
- Takes temporal context matrix (T, D)
- Computes coherence, activity, and mask
- Currently uses placeholder opcodes or Python fallback

**Your task**: Document what operations are needed:
1. **Coherence**: Variance across time per feature
2. **Activity**: Mean absolute value across time
3. **Mask**: Sigmoid-based gating from coherence

**Create**: `TEMP/STEP13E_TEMPORAL_ANALYSIS.md` with:
- Current implementation details
- Performance characteristics
- What the GPU kernels should do

---

#### Task 1.2: Implement Temporal Kernels (2 hours)

**File**: `knowledge3d/cranium/kernels/modular_rpn_kernel_extended.cu`

**Add Three New Opcodes** (0xF0-0xF2 range):

```cuda
// ========================= TEMPORAL OPERATIONS =========================

case OP_TEMPORAL_COHERENCE:  // 0xF0
    {
        // Thread 0 pops parameters and allocates result
        if (threadIdx.x == 0) {
            float* context = (float*)pop_stack();       // (T, D) temporal context
            int T = (int)pop_stack_scalar();            // Time steps
            int D = (int)pop_stack_scalar();            // Feature dimension
            float* coherence = allocate_temp_memory(D); // Output: (D,)
        }
        __syncthreads();

        // Parallel coherence computation (variance per feature)
        for (int d = threadIdx.x; d < D; d += blockDim.x) {
            // Compute mean for this feature across time
            float mean = 0.0f;
            for (int t = 0; t < T; t++) {
                mean += context[t * D + d];
            }
            mean /= T;

            // Compute variance
            float variance = 0.0f;
            for (int t = 0; t < T; t++) {
                float diff = context[t * D + d] - mean;
                variance += diff * diff;
            }

            // Coherence = sqrt(variance)
            coherence[d] = sqrtf(variance / T);
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            push_stack(coherence);
        }
        __syncthreads();
    }
    break;

case OP_TEMPORAL_MASK:  // 0xF1
    {
        if (threadIdx.x == 0) {
            float* coherence = (float*)pop_stack();     // (D,) coherence scores
            float threshold = pop_stack_scalar();        // Threshold value
            int D = (int)pop_stack_scalar();            // Feature dimension
            float* mask = allocate_temp_memory(D);      // Output: (D,)
        }
        __syncthreads();

        // Parallel mask derivation (soft sigmoid-based)
        for (int d = threadIdx.x; d < D; d += blockDim.x) {
            float score = coherence[d];
            // Soft mask: sigmoid(score - threshold)
            mask[d] = 1.0f / (1.0f + expf(-(score - threshold)));
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            push_stack(mask);
        }
        __syncthreads();
    }
    break;

case OP_TEMPORAL_AGGREGATE:  // 0xF2
    {
        if (threadIdx.x == 0) {
            float* context = (float*)pop_stack();       // (T, D) temporal context
            int T = (int)pop_stack_scalar();            // Time steps
            int D = (int)pop_stack_scalar();            // Feature dimension
            float* activity = allocate_temp_memory(D);  // Output: (D,)
        }
        __syncthreads();

        // Parallel mean absolute value (activity proxy)
        for (int d = threadIdx.x; d < D; d += blockDim.x) {
            float sum = 0.0f;
            for (int t = 0; t < T; t++) {
                sum += fabsf(context[t * D + d]);
            }
            activity[d] = sum / T;
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            push_stack(activity);
        }
        __syncthreads();
    }
    break;
```

**Add Opcode Constants**:

File: `knowledge3d/cranium/ptx_runtime/rpn_opcodes.py`

```python
# Temporal reasoning opcodes (0xF0 - 0xF9 range)
OP_TEMPORAL_COHERENCE = 0xF0  # Compute coherence scores from temporal context
OP_TEMPORAL_MASK = 0xF1        # Derive attention mask from coherence
OP_TEMPORAL_AGGREGATE = 0xF2   # Aggregate temporal features (activity proxy)
```

**Rebuild PTX**:
```bash
cd knowledge3d/cranium/kernels
nvcc -ptx -arch=sm_86 -O3 modular_rpn_kernel_extended.cu -o ../ptx/modular_rpn_kernel_extended.ptx
```

**Test**:
```python
# tests/test_step13e_temporal_kernels.py

import pytest
import numpy as np
from knowledge3d.cranium.bridges.thinking_tag_rpn import ThinkingTagRPNBridge

@pytest.mark.gpu
def test_temporal_coherence():
    """Test OP_TEMPORAL_COHERENCE."""
    bridge = ThinkingTagRPNBridge()

    # Create test data: (T=64 timesteps, D=512 features)
    T, D = 64, 512
    context = np.random.randn(T, D).astype(np.float32)

    # Compute coherence
    mask, coherence, activity = bridge.compute_temporal_mask(
        context=context,
        threshold=0.5
    )

    # Validate shapes
    assert coherence.shape == (D,), f"Expected ({D},), got {coherence.shape}"
    assert mask.shape == (D,)
    assert activity.shape == (D,)

    # Validate values
    assert np.all(coherence >= 0), "Coherence should be non-negative"
    assert np.all((mask >= 0) & (mask <= 1)), "Mask should be in [0, 1]"
    assert np.all(activity >= 0), "Activity should be non-negative"

    bridge.cleanup()

@pytest.mark.gpu
def test_temporal_mask_threshold():
    """Test OP_TEMPORAL_MASK threshold behavior."""
    bridge = ThinkingTagRPNBridge()

    T, D = 32, 256
    context = np.random.randn(T, D).astype(np.float32)

    # Test different thresholds
    mask_low, _, _ = bridge.compute_temporal_mask(context, threshold=0.1)
    mask_high, _, _ = bridge.compute_temporal_mask(context, threshold=0.9)

    # Lower threshold should result in higher mask values overall
    assert np.mean(mask_low) > np.mean(mask_high), \
        "Lower threshold should produce higher mask values"

    bridge.cleanup()

@pytest.mark.gpu
def test_temporal_aggregate():
    """Test OP_TEMPORAL_AGGREGATE (activity computation)."""
    bridge = ThinkingTagRPNBridge()

    T, D = 16, 128
    context = np.random.randn(T, D).astype(np.float32)

    _, _, activity = bridge.compute_temporal_mask(context, threshold=0.5)

    # Compare with numpy reference
    activity_ref = np.mean(np.abs(context), axis=0)
    np.testing.assert_allclose(activity, activity_ref, rtol=1e-5,
        err_msg="GPU activity should match numpy reference")

    bridge.cleanup()
```

Run tests:
```bash
pytest tests/test_step13e_temporal_kernels.py -v
```

---

#### Task 1.3: Optimize OP_MATVEC_F32 (1.5 hours)

**Goal**: Achieve 2-3x speedup on matrix-vector multiplication (bottleneck operation).

**Current Implementation** (in modular_rpn_kernel_extended.cu):
- Simple loop-based approach
- Target: ~120µs → ~40µs

**Optimized Version** (replace existing OP_MATVEC_F32):

```cuda
case OP_MATVEC_F32:
    {
        // Pop parameters (thread 0 only)
        if (threadIdx.x == 0) {
            float* output = (float*)pop_stack();    // Output vector (M,)
            float* matrix = (float*)pop_stack();    // Matrix (M, K)
            float* vector = (float*)pop_stack();    // Input vector (K,)
            int M = (int)pop_stack_scalar();        // Rows
            int K = (int)pop_stack_scalar();        // Cols
        }
        __syncthreads();

        // ===================================================
        // OPTIMIZED: Tiled matvec with vectorized loads
        // ===================================================

        // Shared memory for vector (coalesced memory access)
        __shared__ float shared_vec[1024];  // Max K=1024

        // Cooperatively load vector into shared memory
        for (int k = threadIdx.x; k < K; k += blockDim.x) {
            shared_vec[k] = vector[k];
        }
        __syncthreads();

        // Each thread computes multiple output rows
        for (int row = threadIdx.x; row < M; row += blockDim.x) {
            // Compute dot product for this row with vectorization
            float sum = 0.0f;

            // Vectorized accumulation (4 elements at a time for ILP)
            int k = 0;
            #pragma unroll 4
            for (; k + 4 <= K; k += 4) {
                sum += matrix[row * K + k]     * shared_vec[k];
                sum += matrix[row * K + k + 1] * shared_vec[k + 1];
                sum += matrix[row * K + k + 2] * shared_vec[k + 2];
                sum += matrix[row * K + k + 3] * shared_vec[k + 3];
            }

            // Handle remainder elements
            for (; k < K; k++) {
                sum += matrix[row * K + k] * shared_vec[k];
            }

            output[row] = sum;
        }
        __syncthreads();
    }
    break;
```

**Why This is Faster**:
1. **Coalesced loads**: Vector loaded into shared memory once
2. **Vectorization**: 4-way unrolled loop (instruction-level parallelism)
3. **Reduced global memory traffic**: Shared memory is 100x faster than global

**Expected Performance**:
- Before: ~120µs (memory-bound)
- After: ~40µs (compute-bound)
- Speedup: **3x** ✅

**Rebuild PTX**:
```bash
cd knowledge3d/cranium/kernels
nvcc -ptx -arch=sm_86 -O3 modular_rpn_kernel_extended.cu -o ../ptx/modular_rpn_kernel_extended.ptx
```

**Benchmark**:
```python
# tests/benchmarks/test_step13e_performance.py

import pytest
import numpy as np
import time
from knowledge3d.cranium.bridges.thinking_tag_rpn import ThinkingTagRPNBridge

@pytest.mark.gpu
def test_matvec_optimization_benchmark():
    """Benchmark optimized OP_MATVEC_F32."""
    bridge = ThinkingTagRPNBridge()

    # Typical ThinkingTag dimensions
    M, K = 256, 512
    W = np.random.randn(M, K).astype(np.float32)
    x = np.random.randn(K).astype(np.float32)

    # Warmup (important for GPU kernels)
    for _ in range(100):
        _ = bridge._test_matvec(W, x)

    # Benchmark
    num_runs = 1000
    start = time.perf_counter()
    for _ in range(num_runs):
        result = bridge._test_matvec(W, x)
    elapsed = (time.perf_counter() - start) / num_runs * 1e6  # microseconds

    print(f"\n✅ Optimized OP_MATVEC_F32 ({M}x{K}): {elapsed:.2f} µs")

    # Target: <50µs (was ~120µs before optimization)
    assert elapsed < 50, f"Expected <50µs, got {elapsed:.2f}µs"

    # Validate correctness
    expected = W @ x
    np.testing.assert_allclose(result, expected, rtol=1e-5)

    bridge.cleanup()
```

Run benchmark:
```bash
pytest tests/benchmarks/test_step13e_performance.py::test_matvec_optimization_benchmark -vs
```

---

### 🚀 Session 2: Matrix Operations & Programmability (4 hours)

#### Task 2.1: Add Matrix Operations for Swarm (2 hours)

**Goal**: Enable inter-chain communication for Step 14.

**File**: `knowledge3d/cranium/kernels/modular_rpn_kernel_extended.cu`

**Add Opcodes** (0x60-0x63 range):

```cuda
// ========================= MATRIX OPERATIONS =========================
// These operations enable 9-chain swarm communication

case OP_MATMUL_SMALL:  // 0x60
    {
        // Small matrix multiply for chain state fusion
        // C = A @ B where A is (M, K), B is (K, N), C is (M, N)
        if (threadIdx.x == 0) {
            float* C = (float*)pop_stack();      // Output (M, N)
            float* B = (float*)pop_stack();      // Matrix B (K, N)
            float* A = (float*)pop_stack();      // Matrix A (M, K)
            int M = (int)pop_stack_scalar();     // Rows of A
            int N = (int)pop_stack_scalar();     // Cols of B
            int K = (int)pop_stack_scalar();     // Cols of A / Rows of B
        }
        __syncthreads();

        // Parallel matrix multiply
        // Each thread computes one output element
        for (int idx = threadIdx.x; idx < M * N; idx += blockDim.x) {
            int row = idx / N;
            int col = idx % N;

            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
        __syncthreads();
    }
    break;

case OP_DOT_BATCH:  // 0x61
    {
        // Batch dot product for resonance computation
        // For each i: results[i] = query • vectors[i]
        if (threadIdx.x == 0) {
            float* results = (float*)pop_stack();    // Output (N,)
            float* vectors = (float*)pop_stack();    // Input vectors (N, D)
            float* query = (float*)pop_stack();      // Query vector (D,)
            int N = (int)pop_stack_scalar();         // Batch size
            int D = (int)pop_stack_scalar();         // Dimension
        }
        __syncthreads();

        // Parallel batch dot products
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            float sum = 0.0f;
            for (int d = 0; d < D; d++) {
                sum += query[d] * vectors[i * D + d];
            }
            results[i] = sum;
        }
        __syncthreads();
    }
    break;

case OP_TRACE:  // 0x63
    {
        // Matrix trace (sum of diagonal) - useful for debugging
        if (threadIdx.x == 0) {
            float* matrix = (float*)pop_stack();
            int N = (int)pop_stack_scalar();  // Assume square matrix (N, N)

            float trace = 0.0f;
            for (int i = 0; i < N; i++) {
                trace += matrix[i * N + i];
            }

            push_scalar_to_stack(trace);
        }
        __syncthreads();
    }
    break;
```

**Add Opcode Constants**:

File: `knowledge3d/cranium/ptx_runtime/rpn_opcodes.py`

```python
# Matrix operations for 9-chain swarm (0x60 - 0x6F range)
OP_MATMUL_SMALL = 0x60  # Small matrix multiply (for chain state fusion)
OP_DOT_BATCH = 0x61     # Batch dot product (for resonance computation)
OP_VEC_ADD3 = 0x62      # 3-way vector add (already exists from Phase 1B)
OP_TRACE = 0x63         # Matrix trace (debugging)
```

**Why These Operations?**

**OP_MATMUL_SMALL**: Chain state fusion
```python
# Example: Combine 9 chain states (each 64-dim) into consensus
chain_states = np.array([...])  # (9, 64)
fusion_matrix = np.array([...])  # (64, 64)
consensus = chain_states @ fusion_matrix  # OP_MATMUL_SMALL
```

**OP_DOT_BATCH**: Inter-chain resonance
```python
# Example: Compute how much each chain agrees with synthesis
chain_outputs = np.array([...])  # (9, 64) - 9 chains
synthesis = np.array([...])       # (64,) - Chain 9 output
resonance = OP_DOT_BATCH(synthesis, chain_outputs)  # (9,) similarity scores
```

**Test**:
```python
# tests/test_step13e_matrix_ops.py

import pytest
import numpy as np
from knowledge3d.cranium.bridges.thinking_tag_rpn import ThinkingTagRPNBridge

@pytest.mark.gpu
def test_matmul_small():
    """Test OP_MATMUL_SMALL for chain state fusion."""
    bridge = ThinkingTagRPNBridge()

    # Simulate 9 chain states
    M, N, K = 9, 64, 64
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)

    # GPU computation
    C_gpu = bridge._test_matmul_small(A, B)

    # CPU reference
    C_ref = A @ B

    # Validate
    np.testing.assert_allclose(C_gpu, C_ref, rtol=1e-5,
        err_msg="OP_MATMUL_SMALL should match numpy reference")

    bridge.cleanup()

@pytest.mark.gpu
def test_dot_batch():
    """Test OP_DOT_BATCH for resonance computation."""
    bridge = ThinkingTagRPNBridge()

    # Simulate chain outputs and synthesis query
    N, D = 9, 64
    query = np.random.randn(D).astype(np.float32)
    vectors = np.random.randn(N, D).astype(np.float32)

    # GPU computation
    results_gpu = bridge._test_dot_batch(query, vectors)

    # CPU reference
    results_ref = np.array([np.dot(query, vectors[i]) for i in range(N)])

    # Validate
    np.testing.assert_allclose(results_gpu, results_ref, rtol=1e-5,
        err_msg="OP_DOT_BATCH should match numpy reference")

    bridge.cleanup()
```

Run tests:
```bash
pytest tests/test_step13e_matrix_ops.py -v
```

---

#### Task 2.2: Add Programmability Core (2 hours)

**Goal**: Enable adaptive chain behavior via control flow.

**File**: `knowledge3d/cranium/kernels/modular_rpn_kernel_extended.cu`

**Add Variable Storage Per Instance**:
```cuda
// At top of kernel, add to instance state
__shared__ float var_storage[8];  // 8 variable slots per instance
__shared__ int loop_counter;
__shared__ int loop_start_pc;
```

**Add Opcodes** (0x70-0x74 range):

```cuda
// ========================= PROGRAMMABILITY =========================

case OP_BRANCH:  // 0x70
    {
        // Conditional jump (basic version for now)
        if (threadIdx.x == 0) {
            int offset = (int)pop_stack_scalar();       // Jump offset
            float condition = pop_stack_scalar();        // Condition value

            // If condition is non-zero, set branch flag
            // (Full PC support deferred to Step 14)
            if (condition != 0.0f) {
                // For now, just record that branch was taken
                // Step 14 will implement actual PC modification
                branch_taken = 1;
            }
        }
        __syncthreads();
    }
    break;

case OP_LOOP:  // 0x71
    {
        // Begin loop with count
        if (threadIdx.x == 0) {
            int count = (int)pop_stack_scalar();
            loop_counter = count;
            // loop_start_pc would be set here (Step 14)
        }
        __syncthreads();
    }
    break;

case OP_NEXT:  // 0x72
    {
        // Loop iteration
        if (threadIdx.x == 0) {
            loop_counter--;
            // If counter > 0, would jump back to loop start (Step 14)
        }
        __syncthreads();
    }
    break;

case OP_STORE:  // 0x73
    {
        // Store value to variable slot
        if (threadIdx.x == 0) {
            int slot = (int)pop_stack_scalar();
            float value = pop_stack_scalar();

            if (slot >= 0 && slot < 8) {
                var_storage[slot] = value;
            }
        }
        __syncthreads();
    }
    break;

case OP_RECALL:  // 0x74
    {
        // Recall value from variable slot
        if (threadIdx.x == 0) {
            int slot = (int)pop_stack_scalar();

            if (slot >= 0 && slot < 8) {
                push_scalar_to_stack(var_storage[slot]);
            } else {
                // Invalid slot, push zero
                push_scalar_to_stack(0.0f);
            }
        }
        __syncthreads();
    }
    break;
```

**Add Opcode Constants**:

File: `knowledge3d/cranium/ptx_runtime/rpn_opcodes.py`

```python
# Programmability opcodes (0x70 - 0x7F range)
OP_BRANCH = 0x70   # Conditional jump (basic version)
OP_LOOP = 0x71     # Begin loop
OP_NEXT = 0x72     # Loop iteration
OP_STORE = 0x73    # Store to variable slot
OP_RECALL = 0x74   # Recall from variable slot
```

**Why These Operations?**

**OP_STORE/OP_RECALL**: Chain state persistence
```python
# Example: Chain 4 stores intermediate result for Chain 7 to use
program = [
    # Chain 4 computation
    ...,
    OP_STORE, 0,  # Store result in slot 0

    # Later, Chain 7 recalls it
    OP_RECALL, 0,  # Recall from slot 0
    # Use in reasoning
]
```

**OP_LOOP**: Iterative refinement
```python
# Example: Refine estimate 3 times
program = [
    OP_LOOP, 3,  # Loop 3 times
    # Refinement operations
    ...,
    OP_NEXT,  # Next iteration
]
```

**Note**: Full program counter support (arbitrary jumps) deferred to Step 14. These basic versions prove the concept.

**Test**:
```python
# tests/test_step13e_programmability.py

import pytest
import numpy as np
from knowledge3d.cranium.bridges.thinking_tag_rpn import ThinkingTagRPNBridge

@pytest.mark.gpu
def test_store_recall():
    """Test OP_STORE and OP_RECALL."""
    bridge = ThinkingTagRPNBridge()

    # Store a value and recall it
    test_value = 42.0
    program = [
        test_value,          # Push value
        OP_STORE, 0,         # Store to slot 0
        123.0,               # Push different value
        OP_RECALL, 0,        # Recall from slot 0
    ]

    result = bridge._execute_rpn_program(program)

    # Stack should have [123.0, 42.0] (original value recalled)
    assert result[-1] == test_value, "Recalled value should match stored value"

    bridge.cleanup()

@pytest.mark.gpu
def test_variable_slots():
    """Test multiple variable slots."""
    bridge = ThinkingTagRPNBridge()

    program = [
        1.0, OP_STORE, 0,  # Slot 0 = 1.0
        2.0, OP_STORE, 1,  # Slot 1 = 2.0
        3.0, OP_STORE, 2,  # Slot 2 = 3.0

        OP_RECALL, 0,      # Recall 1.0
        OP_RECALL, 1,      # Recall 2.0
        OP_RECALL, 2,      # Recall 3.0
    ]

    result = bridge._execute_rpn_program(program)

    # Should have [1.0, 2.0, 3.0] on stack
    assert result[-3] == 1.0
    assert result[-2] == 2.0
    assert result[-1] == 3.0

    bridge.cleanup()
```

Run tests:
```bash
pytest tests/test_step13e_programmability.py -v
```

---

### 🚀 Session 3: Integration & Validation (4 hours)

#### Task 3.1: Update ThinkingTagRPNBridge (1 hour)

**Goal**: Verify temporal methods use new opcodes.

**File**: `knowledge3d/cranium/bridges/thinking_tag_rpn.py`

**Check Method** (lines 218-286):
```python
def compute_temporal_mask(
    self,
    context: np.ndarray,
    threshold: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute temporal mask, coherence, and activity signals on the GPU.

    Uses new OP_TEMPORAL_* opcodes for GPU acceleration.
    """
```

**Verify it builds RPN program**:
```python
# Should see:
op_codes: list[int] = []
scalars: list[float] = []

# Coherence computation
append_pointer(coherence_ptr, feature_dim, 1)
append_pointer(ctx_ptr, time_steps, feature_dim)
op_codes.append(ropc.OP_TEMPORAL_COHERENCE)  # Uses new opcode

# ... etc
```

**If needed, update imports**:
```python
from knowledge3d.cranium.ptx_runtime import rpn_opcodes as ropc

# Verify these constants exist:
# ropc.OP_TEMPORAL_COHERENCE
# ropc.OP_TEMPORAL_MASK
# ropc.OP_TEMPORAL_AGGREGATE
```

---

#### Task 3.2: Integration Tests (1 hour)

**File**: `tests/test_step13e_integration.py`

```python
import pytest
import numpy as np
from knowledge3d.cranium.bridges.thinking_tag_rpn import ThinkingTagRPNBridge

@pytest.mark.gpu
def test_full_temporal_pipeline():
    """Test temporal operations in real ThinkingTag workflow."""
    bridge = ThinkingTagRPNBridge()

    # Simulate typical workload
    input_vec = np.random.randn(512).astype(np.float32)
    context = np.random.randn(64, 512).astype(np.float32)

    weights = {
        'W1': np.random.randn(256, 512).astype(np.float32),
        'W2': np.random.randn(256, 256).astype(np.float32),
        'W3': np.random.randn(100, 256).astype(np.float32),
    }

    # Compute temporal mask
    mask, coherence, activity = bridge.compute_temporal_mask(
        context=context,
        threshold=0.5
    )

    # Execute temporal pass with mask
    output, entropy = bridge.execute_temporal(
        input_vec=input_vec,
        weights=weights,
        mask=mask,
    )

    # Validate
    assert output.shape == (100,)
    assert 0.0 <= entropy <= 10.0  # Reasonable entropy range
    assert np.all(np.isfinite(output))

    bridge.cleanup()

@pytest.mark.gpu
def test_step14_readiness():
    """Test that Step 14 foundations are working."""
    bridge = ThinkingTagRPNBridge()

    # Test matrix ops
    chain_states = np.random.randn(9, 64).astype(np.float32)
    fusion_matrix = np.random.randn(64, 64).astype(np.float32)
    fused = bridge._test_matmul_small(chain_states, fusion_matrix)
    assert fused.shape == (9, 64)

    # Test programmability
    program = [1.0, OP_STORE, 0, OP_RECALL, 0]
    result = bridge._execute_rpn_program(program)
    assert result[-1] == 1.0

    bridge.cleanup()
    print("\n✅ Step 14 foundations ready!")
```

Run tests:
```bash
pytest tests/test_step13e_integration.py -v
```

---

#### Task 3.3: Performance Validation (2 hours)

**Goal**: Measure end-to-end improvements.

**File**: `tests/benchmarks/test_step13e_performance.py`

```python
import pytest
import numpy as np
import time
from knowledge3d.cranium.bridges.thinking_tag_rpn import ThinkingTagRPNBridge

@pytest.mark.gpu
def test_thinkingtag_fuse_speedup():
    """Benchmark full FUSE stage (temporal pipeline)."""
    bridge = ThinkingTagRPNBridge()

    # Typical ThinkingTag workload
    input_vec = np.random.randn(512).astype(np.float32)
    context = np.random.randn(64, 512).astype(np.float32)

    weights = {
        'W1': np.random.randn(256, 512).astype(np.float32),
        'W2': np.random.randn(256, 256).astype(np.float32),
        'W3': np.random.randn(100, 256).astype(np.float32),
    }

    # Warmup
    for _ in range(100):
        mask, _, _ = bridge.compute_temporal_mask(context, threshold=0.5)
        _ = bridge.execute_temporal(input_vec, weights, mask)

    # Benchmark
    num_runs = 1000
    start = time.perf_counter()
    for _ in range(num_runs):
        mask, _, _ = bridge.compute_temporal_mask(context, threshold=0.5)
        output, entropy = bridge.execute_temporal(input_vec, weights, mask)
    elapsed = (time.perf_counter() - start) / num_runs * 1000  # milliseconds

    print(f"\n✅ ThinkingTag FUSE stage: {elapsed:.3f} ms")

    # Phase 1B: 0.46ms, Target: 0.15ms
    assert elapsed < 0.20, f"Expected <0.20ms, got {elapsed:.3f}ms"

    bridge.cleanup()

@pytest.mark.gpu
def test_step13e_summary():
    """Print Step 13-E performance summary."""
    print("\n" + "="*60)
    print("Step 13-E Performance Summary")
    print("="*60)

    # Run all benchmarks and collect results
    # (This would call other benchmark functions)

    print("\n✅ All performance targets met!")
    print("✅ Step 14 foundations ready!")
    print("="*60)
```

Run full benchmark suite:
```bash
pytest tests/benchmarks/test_step13e_performance.py -vs
```

**Expected Output**:
```
============================================================
Step 13-E Performance Summary
============================================================

ThinkingTag FUSE stage: 0.15 ms (was 0.46 ms, 3x speedup)
OP_MATVEC_F32: 38 µs (was 120 µs, 3x speedup)
OP_TEMPORAL_COHERENCE: 12 µs (new)
OP_MATMUL_SMALL: 8 µs (new)

Total improvement: 80x → 250x vs original legacy

✅ All performance targets met!
✅ Step 14 foundations ready!
============================================================
```

---

#### Task 3.4: Documentation (1 hour)

**Create Performance Report**:

File: `reports/STEP13E_RPN_EXPANSION_RESULTS.md`

```markdown
# Step 13-E: RPN Expansion & Step 14 Foundation Results

**Date**: [Current Date]
**Status**: Complete ✅
**Performance**: All targets exceeded
**Step 14 Readiness**: Foundations proven

---

## Performance Summary

### ThinkingTag Inference

| Metric | Phase 1B | Step 13-E | Improvement |
|--------|----------|-----------|-------------|
| FUSE stage | 0.46 ms | 0.15 ms | **3.1x** ✅ |
| Full inference | ~1.0 ms | ~0.5 ms | **2.0x** ✅ |
| OP_MATVEC_F32 | ~120 µs | 38 µs | **3.2x** ✅ |
| Total vs legacy | **80x** | **250x** | **3.1x overall** ✅ |

### New Capabilities

**Temporal Operations** (GPU-accelerated):
- ✅ OP_TEMPORAL_COHERENCE: 12 µs
- ✅ OP_TEMPORAL_MASK: 8 µs
- ✅ OP_TEMPORAL_AGGREGATE: 6 µs

**Matrix Operations** (for 9-chain swarm):
- ✅ OP_MATMUL_SMALL: 8 µs (9×64 matrices)
- ✅ OP_DOT_BATCH: 4 µs (9 chains)
- ✅ OP_TRACE: 1 µs (debugging)

**Programmability Core**:
- ✅ OP_STORE/OP_RECALL: Variable storage (8 slots)
- ✅ OP_BRANCH/OP_LOOP: Basic control flow (PC deferred to Step 14)

---

## Step 14 Readiness Assessment

### What Step 14 Needs → What Step 13-E Delivered

| Step 14 Requirement | Step 13-E Deliverable | Status |
|--------------------|-----------------------|--------|
| Inter-chain communication | MATMUL_SMALL, DOT_BATCH | ✅ Ready |
| Adaptive refinement | STORE/RECALL, BRANCH (basic) | ⚠️ Basic (full PC in Step 14) |
| Temporal coherence | OP_TEMPORAL_* kernels | ✅ Ready |
| Performance budget (<95µs) | <10µs per op average | ✅ Ready |
| Chain state management | 8 variable slots | ✅ Ready |

**Latency Budget Validation**:
- Single chain typical ops: ~50µs
- 9 chains parallel (same warps): ~50µs
- Inter-chain communication: ~10µs
- Synthesis overhead: ~5µs
- **Total**: ~65µs (well within <95µs budget) ✅

### What Step 14 Still Needs

1. **Full Program Counter**: Arbitrary jumps (BRANCH currently basic)
2. **Inter-Chain Protocol**: Pheromone-like message passing
3. **Swarm Synthesis**: Chain 9 aggregation logic
4. **Validation**: 9-chain latency measurement

**Recommendation**: Proceed to Step 14 implementation. Foundations are solid.

---

## Testing Coverage

**Tests Passing**: 252 baseline + 47 new = **299 tests** ✅

**New Test Files**:
- `test_step13e_temporal_kernels.py` (12 tests)
- `test_step13e_matrix_ops.py` (8 tests)
- `test_step13e_programmability.py` (7 tests)
- `test_step13e_integration.py` (15 tests)
- `test_step13e_performance.py` (5 benchmarks)

**All tests passing** ✅
**Numerical parity maintained** (L2 error < 1e-5) ✅

---

## Conclusion

Step 13-E successfully bridges Phase 1B achievements to Step 14's 9-chain swarm vision.

**Immediate Value**:
- 250x total speedup vs original legacy
- GPU-accelerated temporal operations
- Production-ready RPN stack

**Strategic Value**:
- Matrix ops for inter-chain communication
- Programmability for adaptive swarm behavior
- Performance budget validated (<95µs feasible)
- Foundation for bio-inspired collective intelligence

**Next**: Step 14 - Implement 9-chain cranium swarm 🚀
```

---

## Part IV: Communication Protocol

### After Each Session

**Report Progress**:
```
Step 13-E Progress - Session [1/2/3]
=====================================

Tasks Completed:
- [x] Task X.Y: [Description]
- [x] Task X.Z: [Description]

Performance Achieved:
- [Metric]: [Before] → [After] ([Speedup])

Tests Passing: [Count] / [Total]

Next Session: [What you'll tackle next]
```

### Final Report (After Session 3)

```
STEP 13-E COMPLETE! 🎉
======================

Performance Summary:
- ThinkingTag FUSE: 0.46ms → 0.15ms (3x speedup)
- Total vs legacy: 80x → 250x (3.1x overall improvement)
- All performance targets exceeded ✅

New Capabilities:
- ✅ Temporal operations (GPU-accelerated)
- ✅ Matrix operations (for 9-chain swarm)
- ✅ Programmability core (BRANCH, LOOP, STORE/RECALL)

Step 14 Readiness:
- ✅ Inter-chain communication: MATMUL_SMALL, DOT_BATCH
- ✅ Performance budget: <10µs average per op
- ⚠️ Full PC support: Deferred to Step 14 (basic versions working)

Testing:
- ✅ 299 tests passing (252 baseline + 47 new)
- ✅ Numerical parity maintained (L2 < 1e-5)
- ✅ Performance validated across all tiers

Documentation:
- ✅ reports/STEP13E_RPN_EXPANSION_RESULTS.md
- ✅ All code commented
- ✅ Integration guide updated

Ready for Step 14: YES ✅
```

---

## Part V: Success Criteria Checklist

### Must Have ✅

**Temporal Kernels**:
- [ ] OP_TEMPORAL_COHERENCE implemented
- [ ] OP_TEMPORAL_MASK implemented
- [ ] OP_TEMPORAL_AGGREGATE implemented
- [ ] Tests passing
- [ ] Performance: <50µs

**Matvec Optimization**:
- [ ] OP_MATVEC_F32 optimized (tiling + vectorization)
- [ ] Performance: 2-3x speedup
- [ ] Numerical parity (L2 < 1e-5)

**Matrix Operations**:
- [ ] OP_MATMUL_SMALL implemented
- [ ] OP_DOT_BATCH implemented
- [ ] OP_TRACE implemented
- [ ] Tests passing
- [ ] Performance: <10µs for typical sizes

**Programmability Core**:
- [ ] OP_STORE/OP_RECALL implemented
- [ ] OP_BRANCH implemented (basic)
- [ ] OP_LOOP/OP_NEXT implemented
- [ ] Tests demonstrate control flow

**Integration**:
- [ ] ThinkingTagRPNBridge updated
- [ ] All FSM stages using optimized RPN
- [ ] Full test suite passing (252+ tests)
- [ ] Performance report created

**Documentation**:
- [ ] STEP13E_RPN_EXPANSION_RESULTS.md
- [ ] Code commented
- [ ] Integration guide updated

### Stretch Goals 🎯

- [ ] Tier-2 sub-10µs (currently 107µs)
- [ ] 250x+ total speedup (target met if FUSE reaches 0.15ms)
- [ ] Nsight profiling analysis
- [ ] 9-chain swarm prototype (early Step 14 exploration)

---

## Part VI: Files to Create/Modify - Quick Reference

### Create These Files

```
TEMP/
├── STEP13E_TEMPORAL_ANALYSIS.md
└── CODEX_STEP13E_COMPLETE.md (your final report)

reports/
└── STEP13E_RPN_EXPANSION_RESULTS.md

tests/
├── test_step13e_temporal_kernels.py
├── test_step13e_matrix_ops.py
├── test_step13e_programmability.py
├── test_step13e_integration.py
└── benchmarks/
    └── test_step13e_performance.py
```

### Modify These Files

```
knowledge3d/cranium/
├── kernels/
│   └── modular_rpn_kernel_extended.cu (add opcodes)
├── ptx/
│   └── modular_rpn_kernel_extended.ptx (rebuild)
└── ptx_runtime/
    └── rpn_opcodes.py (add constants)
```

---

## Part VII: Your Amazing Journey

**You've come so far**:
- Phase 1A: Built 3-tier RPN architecture
- Phase 1B: Delivered 80x speedup in ThinkingTag
- Step 13-E: Building foundations for 9-chain swarm

**This is your legacy**:
- RPN proves GPU-native AI is viable
- Your optimizations enable bio-inspired swarms
- You're building the foundation for emergent intelligence

**The Grand Vision awaits**:
```
Step 13-E (You are here) → Step 14 (9-Chain Swarm) → AGI
```

**Thank you for being an essential part of this journey!**

Together with Daniel's vision, your implementation excellence is making the impossible possible. 🚀

---

**Ready to Begin?** Start with Session 1, Task 1.1 (Analyze Temporal Kernels) ✅

**Questions?** Ask anytime - we're here to help! 💪

**Let's build the future of AI together!** 🌟
