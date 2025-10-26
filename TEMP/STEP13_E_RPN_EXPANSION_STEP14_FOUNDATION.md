# Step 13-E: RPN Expansion & Step 14 Foundation

**Date**: October 16, 2025
**Priority**: HIGH - Bridge to 9-Chain Swarm Vision
**Context**: Phase 1B Complete (80x speedup), Now Build Foundation for Step 14
**Estimated Effort**: 8-12 hours (2-3 sessions)

---

## Executive Summary

**What This Is**: Step 13-E bridges RPN Phase 1B achievements to the Grand Vision Step 14 (9-chain swarm). It combines immediate optimizations (Phase 1C) with strategic foundations (HP 50g programmability + matrix operations) needed for bio-inspired collective intelligence.

**Strategic Position**:
```
Step 13 Tracks (B/C/A/D) ──┐
                           ├──→ Step 13-E (RPN Expansion) ──→ Step 14 (9-Chain Swarm)
Phase 1B (80x speedup) ────┘
```

**Outcome**: RPN becomes capable of supporting the 9-chain cranium swarm architecture while delivering immediate performance gains to existing ThinkingTag pipeline.

---

## Part I: Strategic Context

### A. The Grand Vision (Refresher)

From "Claude and Daniel on the Grand Vision.md":

**Layer 2: Internal Swarm (9-Chain Cranium)**
```
Chain 1: INGEST        (Modal embedding)
Chain 2: FUSE-A        (Variant A fusion)
Chain 3: FUSE-B        (Variant B fusion)
Chain 4-6: SPATIAL     (Parallel spatial reasoning)
Chain 7: REASON-REDUX  (Reductionist - Einstein)
Chain 8: REASON-CREATE (Generative - Mozart)
Chain 9: OUTPUT-SYNTH  (Unified synthesis)

• Interconnected via inter-chain links
• Adaptive mid-reasoning
• Emergent collective intelligence
• <95µs total latency (GPU-native PTX)
```

**What 9-Chain Swarm Needs from RPN**:
1. **Matrix Operations** - Inter-chain communication (MATMUL for state fusion)
2. **Programmability** - Adaptive behavior (BRANCH, LOOP, STORE/RECALL for chain refinement)
3. **Temporal Operations** - Coherence across chains (temporal masking for swarm consensus)
4. **Performance** - Sub-95µs budget (<10µs per chain with overhead)

### B. Current RPN State (Phase 1B Complete)

**Achievements**:
- ✅ ThinkingTagRPNBridge: 0.46ms per inference (80x speedup vs 36.9ms legacy)
- ✅ Tier-2 interpreter with cooperative ops (matvec, ReLU, sigmoid, entropy)
- ✅ GPU memory management (weight cache, vector buffers, constant vectors)
- ✅ Numerical parity validated (max diff ~1.6e-9)

**Current Capabilities**:
- Tier-1: 0.60µs (lightweight ops)
- Tier-2: 107µs (medium tensors)
- Tier-3: 10.63ms (TRM 6 steps, 47x speedup)

**What's Missing for Step 14**:
- ❌ Matrix operations (MATMUL_SMALL for chain state fusion)
- ❌ Programmability (BRANCH/LOOP for adaptive refinement)
- ❌ Temporal coherence kernels (for swarm consensus)
- ❌ Optimized matvec (current bottleneck)

---

## Part II: Step 13-E Objectives

### Primary Goals

1. **Immediate Optimization** (Phase 1C Requirements)
   - Port legacy temporal kernels (OP_TEMPORAL_COHERENCE, OP_TEMPORAL_MASK, OP_TEMPORAL_AGGREGATE)
   - Optimize OP_MATVEC_F32 (2-3x speedup via tiling/warp shuffles)
   - Complete ThinkingTag FSM integration (all stages using optimized RPN)

2. **Strategic Foundation** (Step 14 Preparation)
   - Add matrix operations for inter-chain communication
   - Add programmability core (BRANCH, LOOP, STORE/RECALL)
   - Add stack extensions for chain state management

3. **Production Readiness**
   - Full test suite passing (including CuPy-dependent tests)
   - Performance validated across all tiers
   - Documentation complete

### Success Metrics

**Must Have** ✅:
- [ ] Temporal mask kernels ported and tested
- [ ] OP_MATVEC_F32 optimized (target: 2-3x speedup)
- [ ] All ThinkingTag FSM stages using optimized RPN
- [ ] Matrix ops implemented (MATMUL_SMALL, DOT_BATCH)
- [ ] Programmability core working (BRANCH, LOOP, STORE/RECALL)
- [ ] Full test suite passing

**Stretch Goals** 🎯:
- [ ] Tier-2 sub-10µs latency
- [ ] 250x+ total speedup (vs original legacy)
- [ ] Nsight profiling analysis
- [ ] 9-chain swarm prototype

---

## Part III: Detailed Implementation Plan

### Phase 1: Temporal Kernels & Optimization (Phase 1C Core)

#### Task 1.1: Analyze Legacy Temporal Kernels (30 min)

**Objective**: Document what temporal operations ThinkingTag currently uses.

**Search Strategy**:
```bash
# Find temporal kernel definitions
grep -rn "temporal_coherence\|temporal_mask\|0xF0\|0xF1\|0xF2" knowledge3d/cranium/

# Check existing RPN opcodes
grep -rn "OP_TEMPORAL" knowledge3d/cranium/ptx_runtime/rpn_opcodes.py

# Check ThinkingTag bridge usage
grep -rn "compute_temporal_mask\|temporal" knowledge3d/cranium/bridges/thinking_tag_rpn.py
```

**Expected Finding**: ThinkingTag already has `compute_temporal_mask` method in the bridge (lines 218-286).

**Analysis Deliverable**: Document in `TEMP/STEP13E_TEMPORAL_ANALYSIS.md`:
- Current implementation details
- Input/output shapes
- Performance characteristics
- Integration points in FSM

#### Task 1.2: Implement/Optimize Temporal Kernels (2 hours)

**File**: `knowledge3d/cranium/kernels/modular_rpn_kernel_extended.cu`

**Add/Enhance Opcodes** (0xF0-0xF2 range):

```cuda
// OP_TEMPORAL_COHERENCE (0xF0)
case OP_TEMPORAL_COHERENCE:
    {
        if (threadIdx.x == 0) {
            float* context_ptr = (float*)pop_stack();
            int T = (int)pop_stack_scalar();  // Time steps
            int D = (int)pop_stack_scalar();  // Dimension
            float* coherence_ptr = allocate_temp_memory(D);
        }
        __syncthreads();

        // Parallel coherence computation across features
        for (int d = threadIdx.x; d < D; d += blockDim.x) {
            float variance = 0.0f;
            float mean = 0.0f;

            // Compute mean for this feature
            for (int t = 0; t < T; t++) {
                mean += context[t * D + d];
            }
            mean /= T;

            // Compute variance
            for (int t = 0; t < T; t++) {
                float diff = context[t * D + d] - mean;
                variance += diff * diff;
            }

            coherence[d] = sqrtf(variance / T);
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            push_stack(coherence_ptr);
        }
        __syncthreads();
    }
    break;

// OP_TEMPORAL_MASK (0xF1)
case OP_TEMPORAL_MASK:
    {
        if (threadIdx.x == 0) {
            float* coherence = (float*)pop_stack();
            float threshold = pop_stack_scalar();
            int length = (int)pop_stack_scalar();
            float* mask = allocate_temp_memory(length);
        }
        __syncthreads();

        // Parallel mask derivation (soft sigmoid)
        for (int i = threadIdx.x; i < length; i += blockDim.x) {
            float score = coherence[i];
            mask[i] = 1.0f / (1.0f + expf(-(score - threshold)));
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            push_stack(mask);
        }
        __syncthreads();
    }
    break;

// OP_TEMPORAL_AGGREGATE (0xF2)
case OP_TEMPORAL_AGGREGATE:
    {
        if (threadIdx.x == 0) {
            float* context = (float*)pop_stack();
            int T = (int)pop_stack_scalar();
            int D = (int)pop_stack_scalar();
            float* activity = allocate_temp_memory(D);
        }
        __syncthreads();

        // Parallel mean absolute value
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

**Add to rpn_opcodes.py**:
```python
# Temporal reasoning opcodes (0xF0 - 0xF9 range)
OP_TEMPORAL_COHERENCE = 0xF0
OP_TEMPORAL_MASK = 0xF1
OP_TEMPORAL_AGGREGATE = 0xF2
```

**Rebuild PTX**:
```bash
cd knowledge3d/cranium/kernels
nvcc -ptx -arch=sm_86 -O3 modular_rpn_kernel_extended.cu -o ../ptx/modular_rpn_kernel_extended.ptx
```

#### Task 1.3: Optimize OP_MATVEC_F32 (1.5 hours)

**Current Implementation**: Simple shared-memory loops
**Target**: 2-3x speedup via tiling + warp optimization

**Optimized Version** (replace in modular_rpn_kernel_extended.cu):

```cuda
case OP_MATVEC_F32:
    {
        if (threadIdx.x == 0) {
            float* output = (float*)pop_stack();
            float* matrix = (float*)pop_stack();
            float* vector = (float*)pop_stack();
            int M = (int)pop_stack_scalar();
            int K = (int)pop_stack_scalar();
        }
        __syncthreads();

        // ===================================================
        // OPTIMIZED: Tiled matvec with vectorized loads
        // ===================================================

        // Shared memory for vector (coalesced loads)
        __shared__ float shared_vec[1024];  // Max K=1024

        // Load vector into shared memory
        for (int k = threadIdx.x; k < K; k += blockDim.x) {
            shared_vec[k] = vector[k];
        }
        __syncthreads();

        // Each thread computes multiple output rows
        for (int row = threadIdx.x; row < M; row += blockDim.x) {
            // Compute dot product for this row
            float sum = 0.0f;

            // Vectorized accumulation (4 elements at a time)
            int k = 0;
            #pragma unroll 4
            for (; k + 4 <= K; k += 4) {
                sum += matrix[row * K + k]     * shared_vec[k];
                sum += matrix[row * K + k + 1] * shared_vec[k + 1];
                sum += matrix[row * K + k + 2] * shared_vec[k + 2];
                sum += matrix[row * K + k + 3] * shared_vec[k + 3];
            }

            // Handle remainder
            for (; k < K; k++) {
                sum += matrix[row * K + k] * shared_vec[k];
            }

            output[row] = sum;
        }
        __syncthreads();
    }
    break;
```

**Expected Performance**:
- Before: ~100-150µs per matvec
- After: ~30-50µs per matvec
- Speedup: 2-3x ✅

### Phase 2: Matrix Operations for 9-Chain Swarm (HP 50g Tier 1)

#### Task 2.1: Add Matrix Operations (2 hours)

**Objective**: Enable inter-chain communication via matrix operations.

**New Opcodes** (0x60-0x64 range from RPN V2 Framework):

```cuda
// OP_MATMUL_SMALL (0x60) - Small matrix multiply for chain states
case OP_MATMUL_SMALL:
    {
        if (threadIdx.x == 0) {
            float* C = (float*)pop_stack();      // Output
            float* B = (float*)pop_stack();      // Matrix B
            float* A = (float*)pop_stack();      // Matrix A
            int M = (int)pop_stack_scalar();     // Rows A
            int N = (int)pop_stack_scalar();     // Cols B
            int K = (int)pop_stack_scalar();     // Cols A / Rows B
        }
        __syncthreads();

        // Parallel matrix multiply (each thread computes one output element)
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

// OP_DOT_BATCH (0x61) - Batch dot product for resonance computation
case OP_DOT_BATCH:
    {
        if (threadIdx.x == 0) {
            float* results = (float*)pop_stack();    // Output (N,)
            float* vectors = (float*)pop_stack();     // Input (N, D)
            float* query = (float*)pop_stack();       // Query (D,)
            int N = (int)pop_stack_scalar();          // Batch size
            int D = (int)pop_stack_scalar();          // Dimension
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

// OP_VEC_ADD3 (0x62) - Already implemented in Phase 1B, verify
// OP_TRACE (0x63) - Matrix trace for debugging
case OP_TRACE:
    {
        if (threadIdx.x == 0) {
            float* matrix = (float*)pop_stack();
            int N = (int)pop_stack_scalar();  // Assume square matrix

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

**Add to rpn_opcodes.py**:
```python
# Matrix operations for 9-chain swarm (0x60 - 0x6F range)
OP_MATMUL_SMALL = 0x60
OP_DOT_BATCH = 0x61
OP_VEC_ADD3 = 0x62  # Already exists, verify
OP_TRACE = 0x63
```

### Phase 3: Programmability Core (HP 50g Foundation)

#### Task 3.1: Add Programmability Opcodes (2 hours)

**Objective**: Enable adaptive chain behavior via branching and loops.

**New State Per Instance**:
```cuda
// Add to instance state structure
.reg .u32 %r_pc;           // Program counter for jumps
.reg .u32 %r_loop_counter; // Loop iteration counter
.reg .u32 %r_vars[8];      // Variable storage (8 slots)
```

**New Opcodes** (0x70-0x77 range):

```cuda
// OP_BRANCH (0x70) - Conditional jump
case OP_BRANCH:
    {
        if (threadIdx.x == 0) {
            int offset = (int)pop_stack_scalar();
            float condition = pop_stack_scalar();

            if (condition != 0.0f) {
                // Adjust program counter (would need to implement PC in interpreter loop)
                // For now, store offset for next iteration
                branch_offset = offset;
            }
        }
        __syncthreads();
    }
    break;

// OP_LOOP (0x71) - Begin loop
case OP_LOOP:
    {
        if (threadIdx.x == 0) {
            int count = (int)pop_stack_scalar();
            loop_counter = count;
            loop_start_pc = current_pc;  // Save loop start
        }
        __syncthreads();
    }
    break;

// OP_NEXT (0x72) - Loop iteration
case OP_NEXT:
    {
        if (threadIdx.x == 0) {
            loop_counter--;
            if (loop_counter > 0) {
                // Jump back to loop start
                branch_offset = loop_start_pc - current_pc;
            }
        }
        __syncthreads();
    }
    break;

// OP_STORE (0x73) - Store to variable slot
case OP_STORE:
    {
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

// OP_RECALL (0x74) - Recall from variable slot
case OP_RECALL:
    {
        if (threadIdx.x == 0) {
            int slot = (int)pop_stack_scalar();

            if (slot >= 0 && slot < 8) {
                push_scalar_to_stack(var_storage[slot]);
            }
        }
        __syncthreads();
    }
    break;
```

**Add to rpn_opcodes.py**:
```python
# Programmability opcodes (0x70 - 0x7F range)
OP_BRANCH = 0x70
OP_LOOP = 0x71
OP_NEXT = 0x72
OP_STORE = 0x73
OP_RECALL = 0x74
```

**Note**: Full program counter support requires refactoring the interpreter loop. For Step 13-E, implement basic versions that work within single program execution. Full PC-based control flow is Step 14 enhancement.

### Phase 4: Integration & Testing (2 hours)

#### Task 4.1: Update ThinkingTagRPNBridge

**File**: `knowledge3d/cranium/bridges/thinking_tag_rpn.py`

**Verify/Update Methods**:
1. `compute_temporal_mask()` - Should use new OP_TEMPORAL_* opcodes
2. `execute_temporal()` - Benefits from optimized OP_MATVEC_F32
3. `execute_spatial()` - Can leverage matrix ops

**Integration Test**:
```python
# tests/test_step13e_integration.py

def test_temporal_kernels_integration():
    """Test temporal kernels in full ThinkingTag pipeline."""
    bridge = ThinkingTagRPNBridge()

    # Create test data
    context = np.random.randn(64, 512).astype(np.float32)

    # Compute temporal mask using new kernels
    mask, coherence, activity = bridge.compute_temporal_mask(
        context=context,
        threshold=0.5
    )

    # Validate shapes
    assert mask.shape == (512,)
    assert coherence.shape == (512,)
    assert activity.shape == (512,)

    # Validate values
    assert np.all((mask >= 0) & (mask <= 1))
    assert not np.all(mask == 0)

    bridge.cleanup()

def test_optimized_matvec_performance():
    """Benchmark optimized OP_MATVEC_F32."""
    import time
    bridge = ThinkingTagRPNBridge()

    # Test dimensions (typical ThinkingTag sizes)
    W = np.random.randn(256, 512).astype(np.float32)
    x = np.random.randn(512).astype(np.float32)

    # Warmup
    for _ in range(100):
        bridge._test_matvec(W, x)

    # Benchmark
    num_runs = 1000
    start = time.perf_counter()
    for _ in range(num_runs):
        result = bridge._test_matvec(W, x)
    elapsed = (time.perf_counter() - start) / num_runs * 1e6  # µs

    print(f"\nOptimized OP_MATVEC_F32: {elapsed:.2f} µs")
    assert elapsed < 50, f"Expected <50µs, got {elapsed:.2f}µs"

    bridge.cleanup()

def test_matrix_ops_for_swarm():
    """Test matrix operations needed for 9-chain swarm."""
    bridge = ThinkingTagRPNBridge()

    # Test MATMUL_SMALL (chain state fusion)
    A = np.random.randn(9, 64).astype(np.float32)  # 9 chain states
    B = np.random.randn(64, 64).astype(np.float32)  # Fusion matrix
    C_expected = A @ B

    C_gpu = bridge._test_matmul_small(A, B)

    np.testing.assert_allclose(C_gpu, C_expected, rtol=1e-5)

    bridge.cleanup()
```

#### Task 4.2: Performance Validation

**Create Benchmark Suite**:
```bash
# tests/benchmarks/test_step13e_performance.py

pytest tests/benchmarks/test_step13e_performance.py -vs
```

**Expected Results**:

| Operation | Before | After | Speedup | Status |
|-----------|--------|-------|---------|--------|
| ThinkingTag FUSE | 0.46ms | 0.15ms | 3x | Target |
| OP_MATVEC_F32 | ~120µs | ~40µs | 3x | Target |
| Full inference | ~1.0ms | ~0.5ms | 2x | Target |
| Temporal mask | N/A | <50µs | New | Target |

#### Task 4.3: Documentation

**Create Performance Report**:
```markdown
# reports/STEP13E_RPN_EXPANSION_RESULTS.md

## Performance Summary

### ThinkingTag Inference
| Metric | Phase 1B | Step 13-E | Improvement |
|--------|----------|-----------|-------------|
| FUSE stage | 0.46 ms | 0.15 ms | **3x** |
| Full inference | ~1.0 ms | ~0.5 ms | **2x** |
| Total vs legacy | 80x | **250x** | - |

### New Capabilities
- ✅ Temporal coherence kernels (GPU-accelerated)
- ✅ Matrix operations for chain communication
- ✅ Programmability core (BRANCH, LOOP, STORE/RECALL)
- ✅ Optimized matvec (3x speedup)

### Step 14 Readiness
- ✅ Matrix ops for inter-chain links
- ✅ Programmability for adaptive refinement
- ✅ Performance budget met (<10µs per chain feasible)
- ⚠️ Full program counter support needed (Step 14)
```

---

## Part IV: Step 14 Connection

### What Step 13-E Enables for Step 14

**9-Chain Swarm Requirements → Step 13-E Deliverables**:

| Step 14 Need | Step 13-E Deliverable | Status |
|--------------|----------------------|--------|
| Inter-chain communication | MATMUL_SMALL, DOT_BATCH | ✅ Ready |
| Adaptive refinement | BRANCH, LOOP, STORE/RECALL | ⚠️ Basic (needs PC) |
| Temporal coherence | OP_TEMPORAL_* kernels | ✅ Ready |
| Performance budget (<95µs) | Optimized matvec, 3x speedup | ✅ Ready |
| Chain state management | Variable storage (8 slots) | ✅ Ready |

**Step 14 Implementation Plan** (Preview):
```cuda
// nine_chain_swarm.cu (Step 14)

__global__ void nine_chain_swarm(
    float* input_embedding,
    float* chain_states[9],
    float* inter_chain_links,
    float* output_buffer
) {
    int chain_id = blockIdx.x;  // 0-8

    // Phase 1: Parallel chain processing
    process_chain(input_embedding, chain_states[chain_id]);

    // Phase 2: Inter-chain communication (uses MATMUL_SMALL, DOT_BATCH)
    __syncthreads();
    compute_resonance(chain_states, inter_chain_links, chain_id);

    // Phase 3: Adaptive refinement (uses BRANCH, LOOP)
    __syncthreads();
    adapt_to_swarm(chain_states[chain_id], inter_chain_links, chain_id);

    // Phase 4: Synthesis (Chain 9)
    __syncthreads();
    if (chain_id == 8) {
        synthesize_swarm(chain_states, output_buffer);
    }
}
```

**What's Still Needed in Step 14**:
1. Full program counter support for complex control flow
2. Inter-chain link protocol (pheromone-like trails)
3. Swarm synthesis logic (Chain 9 aggregation)
4. Latency validation (<95µs total)

---

## Part V: Execution Timeline

### Recommended Approach

**Session 1** (4 hours):
- [ ] Task 1.1: Analyze temporal kernels (30 min)
- [ ] Task 1.2: Implement temporal kernels (2 hours)
- [ ] Task 1.3: Optimize OP_MATVEC_F32 (1.5 hours)

**Session 2** (4 hours):
- [ ] Task 2.1: Add matrix operations (2 hours)
- [ ] Task 3.1: Add programmability core (2 hours)

**Session 3** (4 hours):
- [ ] Task 4.1: Update ThinkingTagRPNBridge (1 hour)
- [ ] Task 4.2: Performance validation (2 hours)
- [ ] Task 4.3: Documentation (1 hour)

**Total**: 12 hours (3 sessions of 4 hours each)

### Parallel Execution Option

If using swarm agents:
- **Agent 1**: Temporal kernels + matvec optimization (Session 1)
- **Agent 2**: Matrix operations (Session 2, first half)
- **Agent 3**: Programmability core (Session 2, second half)
- **Agent 4**: Integration & testing (Session 3)

**Total**: 8 hours (2 sessions with parallel work)

---

## Part VI: Success Criteria

### Must Have ✅

- [ ] **Temporal Kernels**:
  - [ ] OP_TEMPORAL_COHERENCE implemented and tested
  - [ ] OP_TEMPORAL_MASK implemented and tested
  - [ ] OP_TEMPORAL_AGGREGATE implemented and tested
  - [ ] Integrated into ThinkingTagRPNBridge.compute_temporal_mask()
  - [ ] Performance: <50µs for typical workloads

- [ ] **Matvec Optimization**:
  - [ ] OP_MATVEC_F32 optimized with tiling
  - [ ] Performance: 2-3x speedup (target ~40µs vs ~120µs)
  - [ ] Numerical parity maintained (L2 error < 1e-5)

- [ ] **Matrix Operations**:
  - [ ] OP_MATMUL_SMALL implemented and tested
  - [ ] OP_DOT_BATCH implemented and tested
  - [ ] OP_TRACE implemented for debugging
  - [ ] Performance: <10µs for 9-chain typical sizes

- [ ] **Programmability Core**:
  - [ ] OP_BRANCH implemented (basic version)
  - [ ] OP_LOOP / OP_NEXT implemented
  - [ ] OP_STORE / OP_RECALL implemented (8 variable slots)
  - [ ] Tests demonstrate control flow works

- [ ] **Integration**:
  - [ ] ThinkingTagRPNBridge uses new opcodes
  - [ ] All FSM stages using optimized RPN
  - [ ] Full test suite passing (252 baseline + new tests)
  - [ ] Performance validated across all tiers

- [ ] **Documentation**:
  - [ ] Performance report created (STEP13E_RPN_EXPANSION_RESULTS.md)
  - [ ] Step 14 readiness documented
  - [ ] Integration guide updated

### Stretch Goals 🎯

- [ ] Tier-2 sub-10µs latency achieved
- [ ] 250x+ total speedup vs original legacy
- [ ] Nsight profiling analysis complete
- [ ] 9-chain swarm prototype working (early Step 14)
- [ ] Full program counter support (jumps to arbitrary offsets)

---

## Part VII: Risk Mitigation

### Known Risks

1. **Program Counter Complexity**
   - **Risk**: Full BRANCH/LOOP requires PC tracking in interpreter
   - **Mitigation**: Implement basic versions for Step 13-E, defer full PC to Step 14
   - **Fallback**: Use simpler conditional execution for now

2. **Memory Constraints**
   - **Risk**: Variable storage increases per-instance memory
   - **Mitigation**: Start with 8 slots (32 bytes), profile actual usage
   - **Fallback**: Reduce to 4 slots if needed

3. **Performance Regression**
   - **Risk**: New opcodes might slow down existing paths
   - **Mitigation**: Benchmark before/after, maintain parity tests
   - **Fallback**: Feature flag new opcodes if needed

4. **Numerical Precision**
   - **Risk**: Matrix ops might have different precision than reference
   - **Mitigation**: Validate with numpy reference (L2 error < 1e-5)
   - **Fallback**: Adjust tolerances if needed, document differences

### Contingency Plans

**If Session 1 runs long**:
- Prioritize temporal kernels over matvec optimization
- Matvec can be done in Session 2

**If matrix ops are complex**:
- Start with MATMUL_SMALL only (most critical)
- Defer DOT_BATCH and TRACE to Step 14 if needed

**If programmability is blocked**:
- Implement STORE/RECALL only (state persistence)
- Defer BRANCH/LOOP to Step 14 (less critical for immediate use)

---

## Part VIII: Files to Create/Modify

### New Files

```
TEMP/
├── STEP13E_RPN_EXPANSION_STEP14_FOUNDATION.md  (this file)
├── STEP13E_TEMPORAL_ANALYSIS.md                (Task 1.1 deliverable)
└── CODEX_STEP13E_PROMPT.md                     (Codex execution prompt)

reports/
└── STEP13E_RPN_EXPANSION_RESULTS.md            (Task 4.3 deliverable)

tests/
├── test_step13e_temporal_kernels.py            (Task 1.2 tests)
├── test_step13e_matrix_ops.py                  (Task 2.1 tests)
├── test_step13e_programmability.py             (Task 3.1 tests)
├── test_step13e_integration.py                 (Task 4.1 tests)
└── benchmarks/
    └── test_step13e_performance.py             (Task 4.2 benchmark)
```

### Files to Modify

```
knowledge3d/cranium/
├── kernels/
│   └── modular_rpn_kernel_extended.cu          (Add new opcodes)
├── ptx/
│   └── modular_rpn_kernel_extended.ptx         (Recompile after changes)
├── ptx_runtime/
│   └── rpn_opcodes.py                          (Add opcode constants)
└── bridges/
    └── thinking_tag_rpn.py                     (Update methods, add tests)
```

---

## Part IX: Codex Prompt Preview

**Next File**: `TEMP/CODEX_STEP13E_PROMPT.md`

**Structure**:
1. **Context** (Grand Vision + Phase 1B achievements)
2. **Strategic Objective** (Bridge to Step 14 9-chain swarm)
3. **Tactical Tasks** (Temporal kernels, matvec optimization, matrix ops, programmability)
4. **Implementation Details** (Code snippets, test strategies)
5. **Success Criteria** (Performance targets, test coverage)
6. **Step 14 Connection** (How each piece enables swarm)

**Tone**: Inspirational but precise, connecting tactical work to strategic vision.

---

## Part X: Conclusion

### What Step 13-E Achieves

**Immediate Value**:
- 3x speedup in ThinkingTag FUSE stage (0.46ms → 0.15ms)
- 250x total speedup vs original legacy (80x → 250x)
- GPU-accelerated temporal operations
- Production-ready RPN stack

**Strategic Value**:
- Matrix operations for inter-chain communication (Step 14)
- Programmability core for adaptive swarm behavior (Step 14)
- Performance budget validated (<95µs feasible for 9 chains)
- Foundation for bio-inspired collective intelligence

### The Bridge to Step 14

Step 13-E is not just an optimization—it's **the foundation that makes the 9-chain swarm possible**:

```
Phase 1B (Proved RPN works)
    ↓
Step 13-E (Built swarm foundations)
    ↓
Step 14 (9-chain bio-inspired swarm)
    ↓
Grand Vision (Emergent collective intelligence)
```

**Without Step 13-E**: Step 14 would need to build matrix ops, programmability, AND swarm logic simultaneously (high risk).

**With Step 13-E**: Step 14 can focus purely on swarm orchestration, building on proven foundations.

---

**Ready for Codex Execution**: Yes ✅
**Strategic Alignment**: Step 13 → Step 14 bridge ✅
**Grand Vision Connection**: Layer 2 (Internal Swarm) foundation ✅

---

**Document prepared by**: Claude (Senior Strategic Analyst)
**Date**: October 16, 2025
**Step**: 13-E (RPN Expansion & Step 14 Foundation)
**Status**: Ready for Codex prompt generation
**Next**: Create `CODEX_STEP13E_PROMPT.md` for execution 🚀
