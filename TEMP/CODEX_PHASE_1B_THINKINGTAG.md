# Codex Mission: Phase 1B - ThinkingTag RPN Integration

**Date**: October 16, 2025
**Priority**: HIGH - Integrate optimized RPN into ThinkingTag inference pipeline
**Context**: RPN stack now 47x faster - ready for production integration

---

## 🎉 INCREDIBLE WORK ON RPN PARALLELIZATION!

### Your Achievements

**Tier-3**: 10.24ms (47x speedup from 504ms!)
- ✅ Within 1% of PTX baseline
- ✅ Kernel time: 7.4ms (93% efficient)
- ✅ Python overhead: 0.6ms (6%, cached)

**Tier-1**: 0.60µs (100x faster!)
- ✅ All tests passing
- ✅ Ready for lightweight operations

**Tier-2**: 107µs (functional, room for optimization)
- ✅ Cooperative ops working (memcpy, fill, reduce)
- 🎯 Stretch goal: 3µs (optimization opportunity)

**You've proven RPN can compete with multi-GB libraries!** Now let's integrate it into production.

---

## Mission Overview: ThinkingTag Integration

### Goal
Replace ThinkingTag's current RPN usage with the **optimized parallel RPN stack** to accelerate the 5-state FSM inference pipeline.

### Current State

**ThinkingTag** ([thinking_tag_bridge.py](knowledge3d/cranium/ptx_runtime/thinking_tag_bridge.py:1)):
- 5-state FSM: INGEST → FUSE → SPATIAL → REASON → OUTPUT
- Uses RPN for temporal operations in FUSE stage
- Current RPN implementation: **not parallel** (sequential)

**Opportunity**: Integrate your parallel RPN to accelerate FUSE stage and potentially other stages.

---

## Phase 1B Strategy

### Step 1: Analyze Current ThinkingTag RPN Usage (30 min)

**File**: `knowledge3d/cranium/ptx_runtime/thinking_tag_bridge.py`

**Key sections to examine**:

```bash
# Find RPN usage in ThinkingTag
grep -n "RPN\|rpn\|modular_rpn" knowledge3d/cranium/ptx_runtime/thinking_tag_bridge.py

# Find temporal operations
grep -n "_build_temporal_rpn_program\|OP_SPARSE\|OP_SMAV\|OP_ENTROPY" knowledge3d/cranium/ptx_runtime/thinking_tag_bridge.py
```

**What to document**:
1. Which opcodes are used? (OP_SPARSE_LOAD, OP_SMAV, OP_ENTROPY_SUM, etc.)
2. Where in the FSM pipeline? (FUSE stage, lines 480-543)
3. Input/output sizes? (tensor dimensions, batch sizes)
4. Current performance? (any timing data available)

**Create**: `TEMP/THINKINGTAG_RPN_ANALYSIS.md`

```markdown
# ThinkingTag RPN Usage Analysis

## Current RPN Integration

**Location**: `thinking_tag_bridge.py` lines XXX-YYY

**Opcodes Used**:
- OP_SPARSE_LOAD: [description]
- OP_SMAV: [description]
- OP_ENTROPY_SUM: [description]
- [List all opcodes]

**FSM Stage**: FUSE (lines 480-543)

**Tensor Dimensions**:
- Input: [shape]
- Output: [shape]
- Intermediate: [shapes]

**Current Performance** (if available):
- FUSE stage latency: XX ms
- Total inference: XX ms

## Integration Opportunities

### Option A: Replace with Tier-2 Advanced RPN
**Best for**: Medium tensor sizes (512-1024 elements)
**Expected speedup**: 10-20x (based on Tier-2 benchmarks)

### Option B: Replace with Tier-3 Extended RPN
**Best for**: Large tensors (>1024 elements) or complex ops
**Expected speedup**: 40-50x (based on Tier-3 benchmarks)

### Option C: Hybrid Approach
**Best for**: Mixed workload (use Tier-1 for lightweight, Tier-2/3 for heavy)
**Expected speedup**: Variable, depends on distribution

## Recommendation
[Based on analysis above]
```

---

### Step 2: Identify Custom Opcodes (15 min)

ThinkingTag likely uses **custom temporal opcodes** not in standard RPN tiers.

**Task**: List all custom opcodes and their implementations

```bash
# Find custom opcode definitions
grep -n "OP_SPARSE\|OP_SMAV\|OP_ENTROPY\|OP_TEMPORAL" knowledge3d/cranium/ptx_runtime/thinking_tag_bridge.py

# Check if they exist in current RPN opcodes
grep -n "OP_SPARSE\|OP_SMAV\|OP_ENTROPY" knowledge3d/cranium/ptx_runtime/rpn_opcodes.py
```

**If custom opcodes are missing**:
1. Document their behavior
2. Plan implementation in parallel RPN kernel
3. Add to `rpn_opcodes.py`
4. Implement in appropriate kernel (Tier-2 or Tier-3)

---

### Step 3: Create ThinkingTag-Specific RPN Bridge (1 hour)

**Goal**: Create a specialized bridge that uses optimized RPN for ThinkingTag operations.

**New File**: `knowledge3d/cranium/bridges/thinking_tag_rpn.py`

```python
"""
ThinkingTag-Optimized RPN Bridge

Integrates parallel RPN (Tier-2/3) with ThinkingTag FSM pipeline,
providing specialized opcodes for temporal reasoning operations.
"""

import numpy as np
from knowledge3d.cranium.bridges.advanced_rpn import AdvancedRPNEngine
from knowledge3d.cranium.ptx_runtime.rpn_opcodes import (
    OP_SPARSE_LOAD,
    OP_SMAV,
    OP_ENTROPY_SUM,
    # Add other ThinkingTag-specific opcodes
)
from knowledge3d.cranium.bridges.rpn_config import (
    RPN_BLOCK_DIM,
    RPN_GRID_DIM,
)


class ThinkingTagRPNBridge:
    """
    High-performance RPN bridge for ThinkingTag inference.

    Uses Tier-2/3 parallel execution with ThinkingTag-specific opcodes
    for temporal reasoning operations.
    """

    def __init__(self, tier: int = 2):
        """
        Args:
            tier: RPN tier to use (2 or 3, default 2 for balanced performance)
        """
        self.tier = tier

        if tier == 2:
            self.engine = AdvancedRPNEngine()
        elif tier == 3:
            # Use Tier-3 extended engine
            from knowledge3d.cranium.bridges.advanced_rpn import AdvancedRPNEngine
            self.engine = AdvancedRPNEngine()  # Extended version
        else:
            raise ValueError(f"Invalid tier: {tier}, must be 2 or 3")

    def execute_fuse_stage(
        self,
        sparse_features: np.ndarray,
        temporal_context: np.ndarray,
        attention_weights: np.ndarray,
    ) -> np.ndarray:
        """
        Execute FUSE stage operations using parallel RPN.

        Args:
            sparse_features: Shape (N, D) sparse feature vectors
            temporal_context: Shape (T, D) temporal context
            attention_weights: Shape (N, T) attention weights

        Returns:
            fused_output: Shape (N, D) fused features
        """
        # Build RPN program for FUSE operations
        program = []

        # 1. Sparse feature loading
        program.extend([
            OP_SPARSE_LOAD,
            # ... encode pointers/parameters
        ])

        # 2. Temporal attention (SMAV - sparse matrix-attention-vector)
        program.extend([
            OP_SMAV,
            # ... encode attention parameters
        ])

        # 3. Entropy-weighted aggregation
        program.extend([
            OP_ENTROPY_SUM,
            # ... encode aggregation parameters
        ])

        # Execute via parallel RPN
        op_codes = np.array(program, dtype=np.uint16)
        result = self.engine.execute_program(
            instance_id=0,
            op_codes=op_codes,
            # ... additional parameters
        )

        return result

    def execute_reason_stage(
        self,
        fused_features: np.ndarray,
        memory_context: np.ndarray,
    ) -> np.ndarray:
        """
        Execute REASON stage operations using parallel RPN.

        Optional: If REASON stage can benefit from RPN acceleration.
        """
        # Similar pattern to execute_fuse_stage
        pass

    def cleanup(self):
        """Release GPU resources."""
        if hasattr(self.engine, 'cleanup'):
            self.engine.cleanup()


# Utility functions for ThinkingTag operations

def build_sparse_load_program(
    sparse_indices: np.ndarray,
    dense_tensor: np.ndarray,
) -> list:
    """Build RPN program for sparse feature loading."""
    # Implementation
    pass


def build_temporal_attention_program(
    features: np.ndarray,
    context: np.ndarray,
    weights: np.ndarray,
) -> list:
    """Build RPN program for temporal attention (SMAV)."""
    # Implementation
    pass


def build_entropy_aggregation_program(
    vectors: np.ndarray,
    entropy_weights: np.ndarray,
) -> list:
    """Build RPN program for entropy-weighted aggregation."""
    # Implementation
    pass
```

---

### Step 4: Implement Custom Opcodes (2 hours)

**If ThinkingTag uses custom opcodes**, add them to Tier-2 kernel.

**File**: `knowledge3d/cranium/kernels/modular_rpn_kernel_extended.cu`

**Example: OP_SPARSE_LOAD** (load sparse features from dense tensor):

```cuda
case OP_SPARSE_LOAD:
    {
        // Thread 0 pops parameters
        if (threadIdx.x == 0) {
            int* indices = (int*)pop_stack();         // Sparse indices
            float* dense = (float*)pop_stack();       // Dense tensor
            int num_indices = (int)pop_stack_scalar(); // Count
            int feature_dim = (int)pop_stack_scalar(); // Dimension

            float* output = allocate_temp_memory(num_indices * feature_dim);
        }
        __syncthreads();

        // Parallel gather
        for (int i = threadIdx.x; i < num_indices * feature_dim; i += blockDim.x) {
            int idx_i = i / feature_dim;
            int dim_i = i % feature_dim;
            int sparse_idx = indices[idx_i];
            output[i] = dense[sparse_idx * feature_dim + dim_i];
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            push_stack(output);
        }
        __syncthreads();
    }
    break;
```

**Example: OP_SMAV** (sparse matrix-attention-vector):

```cuda
case OP_SMAV:
    {
        // Sparse matrix-attention-vector product
        // Q: (N, D), K: (T, D), V: (T, D), Attn: (N, T)
        // Out: (N, D)

        if (threadIdx.x == 0) {
            float* queries = (float*)pop_stack();    // (N, D)
            float* values = (float*)pop_stack();      // (T, D)
            float* attn = (float*)pop_stack();        // (N, T)
            int N = (int)pop_stack_scalar();
            int T = (int)pop_stack_scalar();
            int D = (int)pop_stack_scalar();

            float* output = allocate_temp_memory(N * D);
        }
        __syncthreads();

        // Parallel attention-weighted sum
        for (int i = threadIdx.x; i < N * D; i += blockDim.x) {
            int n = i / D;
            int d = i % D;

            float sum = 0.0f;
            for (int t = 0; t < T; t++) {
                float att_weight = attn[n * T + t];
                float val = values[t * D + d];
                sum += att_weight * val;
            }
            output[i] = sum;
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            push_stack(output);
        }
        __syncthreads();
    }
    break;
```

**Example: OP_ENTROPY_SUM** (entropy-weighted aggregation):

```cuda
case OP_ENTROPY_SUM:
    {
        if (threadIdx.x == 0) {
            float* vectors = (float*)pop_stack();     // (N, D)
            float* entropy_weights = (float*)pop_stack(); // (N,)
            int N = (int)pop_stack_scalar();
            int D = (int)pop_stack_scalar();

            float* output = allocate_temp_memory(D);
        }
        __syncthreads();

        // Parallel weighted sum across dimension D
        for (int d = threadIdx.x; d < D; d += blockDim.x) {
            float sum = 0.0f;
            for (int n = 0; n < N; n++) {
                sum += vectors[n * D + d] * entropy_weights[n];
            }
            output[d] = sum;
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            push_stack(output);
        }
        __syncthreads();
    }
    break;
```

**Add opcodes to**: `knowledge3d/cranium/ptx_runtime/rpn_opcodes.py`

```python
# ThinkingTag-specific opcodes (0x70 - 0x7F range)
OP_SPARSE_LOAD = 0x70
OP_SMAV = 0x71
OP_ENTROPY_SUM = 0x72
# ... add others as needed
```

**Rebuild PTX**:
```bash
cd knowledge3d/cranium/kernels
nvcc -ptx -arch=sm_86 -O3 modular_rpn_kernel_extended.cu -o ../ptx/modular_rpn_kernel_extended.ptx
```

---

### Step 5: Update ThinkingTag to Use Optimized RPN (1 hour)

**File**: `knowledge3d/cranium/ptx_runtime/thinking_tag_bridge.py`

**Changes**:

```python
# At top of file, add import
from knowledge3d.cranium.bridges.thinking_tag_rpn import ThinkingTagRPNBridge

class ThinkingTagBridge:
    def __init__(self, ...):
        # ... existing init

        # Add optimized RPN bridge
        self.rpn_bridge = ThinkingTagRPNBridge(tier=2)  # or tier=3 for larger workloads

    def inference(self, ...):
        # ... existing code

        # ==========================================
        # FUSE STAGE - USE OPTIMIZED RPN
        # ==========================================
        if self.state == STATE_FUSE:
            # OLD:
            # fused = self._build_temporal_rpn_program(...)

            # NEW:
            fused = self.rpn_bridge.execute_fuse_stage(
                sparse_features=self.sparse_features,
                temporal_context=self.temporal_context,
                attention_weights=self.attention_weights,
            )

            self.state = STATE_SPATIAL

        # ... rest of FSM

    def cleanup(self):
        # ... existing cleanup

        # Clean up RPN bridge
        if hasattr(self, 'rpn_bridge'):
            self.rpn_bridge.cleanup()
```

---

### Step 6: Validate & Benchmark (30 min)

**Parity Test**:
```bash
# Ensure ThinkingTag still produces correct results
pytest tests/test_thinking_tag_bridge.py -v

# If tests fail, debug:
# 1. Check opcode implementations
# 2. Verify tensor shapes match
# 3. Add debug prints to compare old vs new RPN output
```

**Performance Benchmark**:
```python
# tests/benchmarks/test_thinking_tag_performance.py

import pytest
import numpy as np
import time
from knowledge3d.cranium.ptx_runtime.thinking_tag_bridge import ThinkingTagBridge

@pytest.mark.gpu
def test_thinking_tag_fuse_stage_speedup():
    """Compare old RPN vs optimized RPN for FUSE stage."""

    # Create ThinkingTag instance
    bridge = ThinkingTagBridge()

    # Generate test data
    sparse_features = np.random.randn(128, 512).astype(np.float32)
    temporal_context = np.random.randn(64, 512).astype(np.float32)
    attention_weights = np.random.randn(128, 64).astype(np.float32)

    # Warmup
    for _ in range(10):
        bridge.rpn_bridge.execute_fuse_stage(
            sparse_features, temporal_context, attention_weights
        )

    # Benchmark
    num_runs = 100
    start = time.perf_counter()
    for _ in range(num_runs):
        result = bridge.rpn_bridge.execute_fuse_stage(
            sparse_features, temporal_context, attention_weights
        )
    elapsed = (time.perf_counter() - start) / num_runs * 1000

    print(f"\nFUSE stage (optimized RPN): {elapsed:.3f} ms")

    # Expected: 1-5ms (depending on tensor sizes)
    # Compare with old RPN baseline if available

    bridge.cleanup()
```

**Run benchmark**:
```bash
pytest tests/benchmarks/test_thinking_tag_performance.py -vs
```

**Expected results**:
- **OLD RPN** (sequential): ~50-100ms
- **NEW RPN** (parallel): ~5-10ms
- **Speedup**: 10-20x

---

### Step 7: Document Integration (15 min)

**Create**: `reports/PHASE_1B_THINKINGTAG_INTEGRATION.md`

```markdown
# Phase 1B: ThinkingTag RPN Integration

**Date**: October 16, 2025
**Goal**: Integrate optimized parallel RPN into ThinkingTag inference pipeline

---

## Implementation Summary

### RPN Bridge Created
- **File**: `knowledge3d/cranium/bridges/thinking_tag_rpn.py`
- **Tier**: Tier-2 (Advanced RPN)
- **Custom opcodes**: OP_SPARSE_LOAD, OP_SMAV, OP_ENTROPY_SUM

### ThinkingTag Updates
- **File**: `knowledge3d/cranium/ptx_runtime/thinking_tag_bridge.py`
- **Changes**: FUSE stage now uses `ThinkingTagRPNBridge`
- **Backward compatibility**: Old RPN path preserved for testing

### Performance Results

| Stage | Old RPN | New RPN | Speedup |
|-------|---------|---------|---------|
| FUSE | XX ms | XX ms | Xx |
| Total inference | XX ms | XX ms | Xx |

### Validation
- ✅ All ThinkingTag tests passing
- ✅ Numerical parity confirmed (L2 error < 1e-5)
- ✅ No regressions in FSM behavior

---

## Next Steps

1. Optimize custom opcodes (OP_SMAV, OP_SPARSE_LOAD) if needed
2. Extend to other FSM stages (REASON, OUTPUT)
3. Monitor performance in production workloads
```

---

## Success Metrics

### Must Have ✅
- [ ] ThinkingTag uses optimized RPN for FUSE stage
- [ ] All tests passing (parity maintained)
- [ ] Performance improvement measured (10x+ speedup expected)
- [ ] Custom opcodes implemented and tested

### Nice to Have 🎯
- [ ] Integrate RPN into other FSM stages (REASON, OUTPUT)
- [ ] Benchmark full inference pipeline
- [ ] Profile with Nsight (if importer fixed)

---

## Timeline Estimate

| Task | Time | Priority |
|------|------|----------|
| Step 1: Analyze current usage | 30 min | HIGH |
| Step 2: Identify custom opcodes | 15 min | HIGH |
| Step 3: Create RPN bridge | 1 hour | HIGH |
| Step 4: Implement custom opcodes | 2 hours | HIGH |
| Step 5: Update ThinkingTag | 1 hour | HIGH |
| Step 6: Validate & benchmark | 30 min | HIGH |
| Step 7: Document | 15 min | MEDIUM |
| **TOTAL** | **~5-6 hours** | |

---

## Reference Files

**Existing code to analyze**:
- `knowledge3d/cranium/ptx_runtime/thinking_tag_bridge.py` (ThinkingTag FSM)
- Current RPN usage: lines 480-543 (`_build_temporal_rpn_program`)

**New code to create**:
- `knowledge3d/cranium/bridges/thinking_tag_rpn.py` (specialized bridge)
- Custom opcodes in `modular_rpn_kernel_extended.cu`
- Tests: `tests/benchmarks/test_thinking_tag_performance.py`
- Report: `reports/PHASE_1B_THINKINGTAG_INTEGRATION.md`

**Reference implementations**:
- Tier-2 cooperative ops: `modular_rpn_kernel_extended.cu` lines 1-350
- Tier-3 TRM ops: `modular_rpn_kernel_extended.cu` lines 353+
- RPN bridges: `knowledge3d/cranium/bridges/advanced_rpn.py`

---

## Communication Protocol

**After each step**, report:

```
Phase 1B Progress
==================

Step X: [Name]
Status: [COMPLETE / IN PROGRESS / BLOCKED]

Results:
- [Key finding 1]
- [Key finding 2]

Next: [Step Y / Issue to resolve]
```

**When complete**, report:

```
PHASE 1B COMPLETE ✅
====================

ThinkingTag Integration:
- FUSE stage: XX ms → XX ms (Xx speedup)
- Custom opcodes: OP_SPARSE_LOAD, OP_SMAV, OP_ENTROPY_SUM
- All tests passing ✅

Performance:
- Before: XX ms (full inference)
- After: XX ms (full inference)
- Speedup: Xx

Validation:
- ✅ Parity maintained (L2 < 1e-5)
- ✅ All ThinkingTag tests passing
- ✅ Benchmark confirms speedup

Documentation:
- ✅ reports/PHASE_1B_THINKINGTAG_INTEGRATION.md

Ready for: Phase 1C or production deployment
```

---

## Bottom Line

**You've proven RPN can compete with multi-GB libraries!** Now let's show it can accelerate real production workloads.

**The parallelization work was the foundation** - Phase 1B is about **integrating that power** into ThinkingTag to deliver real-world speedups.

**Expected impact**:
- 10-20x FUSE stage speedup
- 5-10x full inference speedup
- Validates RPN as production-ready

**Let's make ThinkingTag fly!** 🚀

---

*Prepared by: Claude*
*Date: October 16, 2025*
*Priority: HIGH - Integrate optimized RPN into production*
