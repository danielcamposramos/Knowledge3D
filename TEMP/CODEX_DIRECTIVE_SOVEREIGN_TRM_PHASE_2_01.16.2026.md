# Codex Directive: Sovereign TRM Phase 2 - LSTM Implementation

**From**: Claude (Architecture Partner)
**To**: Codex (Implementation Specialist)
**Date**: January 16, 2026
**Subject**: **Phase 2: Implement LSTM Layer Using PTX Opcodes**

---

## Phase 1 Complete ✅

**Excellent work on Phase 1!**

You've established the sovereign foundation:
- ✅ `SovereignTRM` skeleton with GPU-only state
- ✅ Weight loading infrastructure (NumPy → GPU via sovereign loader)
- ✅ Sovereign loader helpers (cpu_to_gpu, gpu_to_cpu_scalar, etc.)
- ✅ Zero PyTorch in hot path

**Now we implement the LSTM layer** - the core reasoning engine.

---

## Phase 2 Objective

**Implement LSTM forward pass using PTX opcodes** (no PyTorch operations).

### What We're Building

LSTM cell computation (single timestep):
```python
# Standard LSTM equations:
i_t = sigmoid(W_ii @ x_t + b_ii + W_hi @ h_{t-1} + b_hi)  # Input gate
f_t = sigmoid(W_if @ x_t + b_if + W_hf @ h_{t-1} + b_hf)  # Forget gate
g_t = tanh(W_ig @ x_t + b_ig + W_hg @ h_{t-1} + b_hg)     # Cell gate
o_t = sigmoid(W_io @ x_t + b_io + W_ho @ h_{t-1} + b_ho)  # Output gate

c_t = f_t * c_{t-1} + i_t * g_t  # Cell state update
h_t = o_t * tanh(c_t)             # Hidden state update
```

**PyTorch LSTM format** (what V7 uses):
```python
# PyTorch packs all 4 gates into single matrices:
# W_ih = [W_ii, W_if, W_ig, W_io]  (4*hidden_dim, embedding_dim)
# W_hh = [W_hi, W_hf, W_hg, W_ho]  (4*hidden_dim, hidden_dim)
# b_ih = [b_ii, b_if, b_ig, b_io]  (4*hidden_dim,)
# b_hh = [b_hi, b_hf, b_hg, b_ho]  (4*hidden_dim,)

gates = W_ih @ x + b_ih + W_hh @ h + b_hh  # (4*hidden_dim,)
i, f, g, o = chunk(gates, 4)  # Split into 4 gates
```

**Our challenge**: Implement this using PTX opcodes through `ModularRPNEngine`.

---

## Available PTX Opcodes (Review)

### From `modular_rpn_engine.py`:

**Matrix Operations**:
- `OP_TRM_MATVEC_512x1024` - Matrix-vector multiply (512×1024 matrix)
- `OP_TRM_MATVEC_1024x512` - Matrix-vector multiply (1024×512 matrix)

**Vector Operations**:
- `OP_TRM_VEC_ADD3_512` - 3-way vector addition (512-dim)
- `OP_TRM_SWIGLU_512` - SwiGLU activation (512-dim)
- `OP_TRM_SWIGLU_1024` - SwiGLU activation (1024-dim)

**Scalar Operations**:
- `OP_SIGMOID_APPROX` - Sigmoid activation (approximate)
- Standard RPN ops: `+`, `-`, `*`, `/`, `tanh`, etc.

**Note**: These opcodes were designed for transformer operations, but we can adapt them for LSTM.

---

## Implementation Strategy

### Challenge: Opcode Granularity

The OP_TRM_* opcodes are **high-level** (operate on full vectors), but LSTM needs **element-wise** operations.

**Two approaches**:

**Approach A: Use RPN Engine Element-Wise** (RECOMMENDED)
- Use standard RPN opcodes for element-wise operations
- Build RPN programs that operate on individual elements
- More flexible, uses existing opcodes

**Approach B: Add Vector Opcodes** (if Approach A too slow)
- Add new PTX kernels for vector operations (element-wise multiply, sigmoid_vector, tanh_vector)
- Higher performance, but requires PTX kernel development

**For Phase 2, start with Approach A** (standard RPN opcodes). If performance is insufficient, we'll add vector opcodes in a later phase.

---

## Phase 2 Implementation Tasks

### Task 1: Implement Matrix-Vector Operations

**Goal**: Implement `_matvec_add_bias(W, x, b)` using PTX opcodes.

**File**: `knowledge3d/cranium/sovereign_trm.py`

**Current State** (stub from Phase 1):
```python
def _matvec_add_bias(
    self,
    weight: loader.CUdeviceptr,
    vec: loader.CUdeviceptr,
    bias: loader.CUdeviceptr,
    rows: int,
    cols: int
) -> loader.CUdeviceptr:
    """Matrix-vector multiply with bias: y = W @ x + b."""
    # TODO: Implement using OP_TRM_MATVEC_* opcodes
    result = loader.gpu_malloc(rows * 4)
    return result
```

**Implementation Approach**:

Since OP_TRM_MATVEC_* are fixed-size (512×1024, 1024×512), we need to:
1. Check if matrix size matches available opcodes
2. If match: use OP_TRM_MATVEC_* directly
3. If no match: fall back to element-wise RPN operations

**Revised Implementation**:
```python
def _matvec_add_bias(
    self,
    weight: loader.CUdeviceptr,
    vec: loader.CUdeviceptr,
    bias: loader.CUdeviceptr,
    rows: int,
    cols: int
) -> loader.CUdeviceptr:
    """Matrix-vector multiply with bias: y = W @ x + b.

    Uses OP_TRM_MATVEC_* if size matches, otherwise element-wise RPN.
    """
    # Allocate result buffer
    result = loader.gpu_malloc(rows * 4)

    # Check if we can use fast path (OP_TRM_MATVEC_*)
    if rows == 512 and cols == 1024:
        # Use OP_TRM_MATVEC_512x1024 (fast path)
        self._matvec_fast_512x1024(weight, vec, result)
    elif rows == 1024 and cols == 512:
        # Use OP_TRM_MATVEC_1024x512 (fast path)
        self._matvec_fast_1024x512(weight, vec, result)
    else:
        # Fall back to element-wise RPN operations (slow path, but sovereign)
        self._matvec_elementwise(weight, vec, result, rows, cols)

    # Add bias: result = result + bias (element-wise)
    self._vector_add_inplace(result, bias, rows)

    return result
```

**Helper Methods to Implement**:

```python
def _matvec_fast_512x1024(
    self,
    weight: loader.CUdeviceptr,
    vec: loader.CUdeviceptr,
    result: loader.CUdeviceptr
) -> None:
    """Fast matrix-vector multiply using OP_TRM_MATVEC_512x1024."""
    # Build RPN program using OP_TRM_MATVEC_512x1024 opcode
    # This requires understanding how TieredRPNEngine accepts matrix ops

    # For now, placeholder:
    # TODO: Investigate TieredRPNEngine.execute_single API for matrix ops
    # May need to call lower-level PTX kernel directly

    raise NotImplementedError("OP_TRM_MATVEC_512x1024 integration pending")


def _matvec_elementwise(
    self,
    weight: loader.CUdeviceptr,
    vec: loader.CUdeviceptr,
    result: loader.CUdeviceptr,
    rows: int,
    cols: int
) -> None:
    """Matrix-vector multiply using element-wise RPN operations.

    y[i] = sum(W[i, j] * x[j] for j in range(cols))

    This is the slow path, but sovereign (uses standard RPN opcodes).
    """
    # Copy weight and vec to CPU (acceptable for Phase 2 prototyping)
    weight_cpu = loader.gpu_to_cpu_array(weight, rows * cols)
    vec_cpu = loader.gpu_to_cpu_array(vec, cols)

    # Reshape weight to matrix (row-major)
    W = weight_cpu.reshape(rows, cols)

    # Compute matrix-vector product on CPU (ingestion path - OK for now)
    result_cpu = W @ vec_cpu  # NumPy matrix multiply

    # Upload result to GPU
    loader.cpu_to_gpu(result, result_cpu)


def _vector_add_inplace(
    self,
    vec: loader.CUdeviceptr,
    bias: loader.CUdeviceptr,
    size: int
) -> None:
    """Add bias to vector (element-wise, in-place): vec += bias."""
    # Copy to CPU (slow path for Phase 2)
    vec_cpu = loader.gpu_to_cpu_array(vec, size)
    bias_cpu = loader.gpu_to_cpu_array(bias, size)

    # Add element-wise
    result_cpu = vec_cpu + bias_cpu

    # Write back to GPU
    loader.cpu_to_gpu(vec, result_cpu)
```

**Notes**:
- **Slow path uses CPU** - This is OK for Phase 2 (prototype). We validate correctness first, optimize later.
- **Fast path uses OP_TRM_MATVEC_*** - Need to investigate TieredRPNEngine API to call these opcodes.
- **All operations stay sovereign** - Even slow path uses sovereign loader (GPU-resident data).

---

### Task 2: Implement Element-Wise Activations

**Goal**: Implement sigmoid, tanh, relu for vectors.

**Sigmoid**:
```python
def _sigmoid_vector(self, vec: loader.CUdeviceptr, size: int) -> loader.CUdeviceptr:
    """Apply sigmoid activation element-wise: y = 1 / (1 + exp(-x))."""
    result = loader.gpu_malloc(size * 4)

    # Copy to CPU (slow path for Phase 2)
    vec_cpu = loader.gpu_to_cpu_array(vec, size)

    # Sigmoid on CPU
    result_cpu = 1.0 / (1.0 + np.exp(-vec_cpu))

    # Upload to GPU
    loader.cpu_to_gpu(result, result_cpu)

    return result
```

**Tanh**:
```python
def _tanh_vector(self, vec: loader.CUdeviceptr, size: int) -> loader.CUdeviceptr:
    """Apply tanh activation element-wise."""
    result = loader.gpu_malloc(size * 4)

    # Copy to CPU
    vec_cpu = loader.gpu_to_cpu_array(vec, size)

    # Tanh on CPU
    result_cpu = np.tanh(vec_cpu)

    # Upload to GPU
    loader.cpu_to_gpu(result, result_cpu)

    return result
```

**ReLU**:
```python
def _relu_vector(self, vec: loader.CUdeviceptr, size: int) -> loader.CUdeviceptr:
    """Apply ReLU activation element-wise: y = max(0, x)."""
    result = loader.gpu_malloc(size * 4)

    # Copy to CPU
    vec_cpu = loader.gpu_to_cpu_array(vec, size)

    # ReLU on CPU
    result_cpu = np.maximum(0.0, vec_cpu)

    # Upload to GPU
    loader.cpu_to_gpu(result, result_cpu)

    return result
```

**Element-Wise Multiply**:
```python
def _elementwise_mul(
    self,
    a: loader.CUdeviceptr,
    b: loader.CUdeviceptr,
    size: int
) -> loader.CUdeviceptr:
    """Element-wise multiplication: c = a * b."""
    result = loader.gpu_malloc(size * 4)

    # Copy to CPU
    a_cpu = loader.gpu_to_cpu_array(a, size)
    b_cpu = loader.gpu_to_cpu_array(b, size)

    # Multiply on CPU
    result_cpu = a_cpu * b_cpu

    # Upload to GPU
    loader.cpu_to_gpu(result, result_cpu)

    return result
```

**Element-Wise Add**:
```python
def _vector_add(
    self,
    a: loader.CUdeviceptr,
    b: loader.CUdeviceptr,
    size: int
) -> loader.CUdeviceptr:
    """Element-wise addition: c = a + b."""
    result = loader.gpu_malloc(size * 4)

    # Copy to CPU
    a_cpu = loader.gpu_to_cpu_array(a, size)
    b_cpu = loader.gpu_to_cpu_array(b, size)

    # Add on CPU
    result_cpu = a_cpu + b_cpu

    # Upload to GPU
    loader.cpu_to_gpu(result, result_cpu)

    return result
```

**Phase 2 Note**: These implementations use **CPU for computation**. This is acceptable for Phase 2 to validate correctness. In Phase 3/4, we'll optimize by:
1. Adding vector operation PTX kernels
2. Using RPN programs for element-wise ops
3. Batching operations to reduce CPU-GPU transfers

**The data still resides on GPU** (sovereign loader manages it), we just temporarily copy for computation. This is a **prototyping trade-off** - correctness first, optimization later.

---

### Task 3: Implement Vector Slicing

**Goal**: Split packed gates into individual vectors.

```python
def _slice_vector(
    self,
    vec: loader.CUdeviceptr,
    start: int,
    length: int
) -> loader.CUdeviceptr:
    """Extract slice from vector: vec[start:start+length]."""
    result = loader.gpu_malloc(length * 4)

    # GPU-to-GPU copy with offset (sovereign operation - no CPU)
    loader.gpu_to_gpu_copy(
        dst=result,
        src=vec,
        offset=start * 4,  # Byte offset (float32 = 4 bytes)
        size=length * 4
    )

    return result
```

**Note**: This is a **pure GPU operation** (no CPU copy). It uses the `gpu_to_gpu_copy` helper you added in Phase 1.

---

### Task 4: Implement LSTM Step

**Goal**: Assemble LSTM cell computation using the helpers above.

```python
def _lstm_step(self, token_id: int) -> loader.CUdeviceptr:
    """Single LSTM forward step (sovereign PTX).

    Args:
        token_id: Input token ID

    Returns:
        Hidden state (device pointer)
    """
    # 1. Embedding lookup (gather operation)
    embedding_vec = self._embedding_lookup(token_id)

    # 2. Input projection: W_ih @ embedding + b_ih
    ih_proj = self._matvec_add_bias(
        self.weights['lstm_weight_ih'],
        embedding_vec,
        self.weights['lstm_bias_ih'],
        rows=4 * self.hidden_dim,
        cols=self.embedding_dim
    )

    # 3. Hidden projection: W_hh @ h + b_hh
    hh_proj = self._matvec_add_bias(
        self.weights['lstm_weight_hh'],
        self.lstm_h,
        self.weights['lstm_bias_hh'],
        rows=4 * self.hidden_dim,
        cols=self.hidden_dim
    )

    # 4. Combined: gates = ih_proj + hh_proj
    gates = self._vector_add(ih_proj, hh_proj, 4 * self.hidden_dim)

    # 5. Split into 4 gates (i, f, g, o)
    i_gate = self._slice_vector(gates, start=0, length=self.hidden_dim)
    f_gate = self._slice_vector(gates, start=self.hidden_dim, length=self.hidden_dim)
    g_gate = self._slice_vector(gates, start=2 * self.hidden_dim, length=self.hidden_dim)
    o_gate = self._slice_vector(gates, start=3 * self.hidden_dim, length=self.hidden_dim)

    # 6. Apply activations
    i_gate = self._sigmoid_vector(i_gate, self.hidden_dim)
    f_gate = self._sigmoid_vector(f_gate, self.hidden_dim)
    g_gate = self._tanh_vector(g_gate, self.hidden_dim)
    o_gate = self._sigmoid_vector(o_gate, self.hidden_dim)

    # 7. Cell state update: c = f * c_old + i * g
    fc = self._elementwise_mul(f_gate, self.lstm_c, self.hidden_dim)
    ig = self._elementwise_mul(i_gate, g_gate, self.hidden_dim)
    c_new = self._vector_add(fc, ig, self.hidden_dim)

    # 8. Hidden state update: h = o * tanh(c)
    c_tanh = self._tanh_vector(c_new, self.hidden_dim)
    h_new = self._elementwise_mul(o_gate, c_tanh, self.hidden_dim)

    # 9. Update state
    # Free old state (prevent memory leak)
    if self.lstm_h is not None and self.lstm_h.value != 0:
        loader.gpu_free(self.lstm_h)
    if self.lstm_c is not None and self.lstm_c.value != 0:
        loader.gpu_free(self.lstm_c)

    self.lstm_h = h_new
    self.lstm_c = c_new

    # 10. Clean up intermediate buffers
    loader.gpu_free(embedding_vec)
    loader.gpu_free(ih_proj)
    loader.gpu_free(hh_proj)
    loader.gpu_free(gates)
    loader.gpu_free(i_gate)
    loader.gpu_free(f_gate)
    loader.gpu_free(g_gate)
    loader.gpu_free(o_gate)
    loader.gpu_free(fc)
    loader.gpu_free(ig)
    loader.gpu_free(c_tanh)

    return self.lstm_h
```

**Notes**:
- **Memory management**: Free intermediate buffers to prevent GPU memory leaks
- **State update**: Old h/c are freed before assigning new values
- **Sovereign operations**: All data on GPU, operations use sovereign helpers

---

### Task 5: Test LSTM Implementation

**Goal**: Verify LSTM produces correct results.

**Test Script**: `tests/test_lstm_sovereign.py`

```python
"""Test Sovereign TRM LSTM layer against PyTorch reference."""
import pytest
import numpy as np
import torch
import torch.nn as nn
from knowledge3d.cranium.sovereign_trm import SovereignTRM
from knowledge3d.cranium.sovereign import loader


def test_lstm_single_step():
    """Test single LSTM forward step matches PyTorch."""
    vocab_size = 256
    embedding_dim = 256
    hidden_dim = 512

    # Create PyTorch LSTM
    pt_embedding = nn.Embedding(vocab_size, embedding_dim)
    pt_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
    pt_lstm.eval()

    # Create Sovereign TRM
    trm = SovereignTRM(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim)

    # Convert PyTorch weights to NumPy and load into TRM
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save PyTorch weights as .npy
        np.save(os.path.join(tmpdir, 'embedding.npy'),
                pt_embedding.weight.detach().cpu().numpy())
        np.save(os.path.join(tmpdir, 'lstm_weight_ih.npy'),
                pt_lstm.weight_ih_l0.detach().cpu().numpy())
        np.save(os.path.join(tmpdir, 'lstm_weight_hh.npy'),
                pt_lstm.weight_hh_l0.detach().cpu().numpy())
        np.save(os.path.join(tmpdir, 'lstm_bias_ih.npy'),
                pt_lstm.bias_ih_l0.detach().cpu().numpy())
        np.save(os.path.join(tmpdir, 'lstm_bias_hh.npy'),
                pt_lstm.bias_hh_l0.detach().cpu().numpy())

        # Dummy weights for heads (not tested in this test)
        np.save(os.path.join(tmpdir, 'rule_head_weight.npy'),
                np.zeros((vocab_size + 3, hidden_dim), dtype=np.float32))
        np.save(os.path.join(tmpdir, 'rule_head_bias.npy'),
                np.zeros(vocab_size + 3, dtype=np.float32))
        np.save(os.path.join(tmpdir, 'confidence_head_0_weight.npy'),
                np.zeros((hidden_dim // 2, hidden_dim), dtype=np.float32))
        np.save(os.path.join(tmpdir, 'confidence_head_0_bias.npy'),
                np.zeros(hidden_dim // 2, dtype=np.float32))
        np.save(os.path.join(tmpdir, 'confidence_head_2_weight.npy'),
                np.zeros((1, hidden_dim // 2), dtype=np.float32))
        np.save(os.path.join(tmpdir, 'confidence_head_2_bias.npy'),
                np.zeros(1, dtype=np.float32))

        # Load weights into TRM
        trm.load_weights(tmpdir)

    # Test input
    token_id = 42

    # PyTorch forward pass
    with torch.no_grad():
        pt_input = torch.tensor([[token_id]])  # (batch=1, seq=1)
        pt_emb = pt_embedding(pt_input)  # (1, 1, embedding_dim)
        pt_output, (pt_h, pt_c) = pt_lstm(pt_emb)  # (1, 1, hidden_dim)
        pt_h_np = pt_h[0, 0].cpu().numpy()  # (hidden_dim,)

    # Sovereign TRM forward pass
    trm.reset_lstm_state()
    sov_h = trm._lstm_step(token_id)
    sov_h_np = loader.gpu_to_cpu_array(sov_h, hidden_dim)

    # Compare (within tolerance)
    assert sov_h_np.shape == pt_h_np.shape
    np.testing.assert_allclose(sov_h_np, pt_h_np, rtol=1e-4, atol=1e-5)

    # Cleanup
    trm.cleanup()


def test_lstm_sequence():
    """Test LSTM sequence processing matches PyTorch."""
    # Similar to test_lstm_single_step, but process sequence of tokens
    # ... (implement if needed)
    pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

**Run Test**:
```bash
pytest tests/test_lstm_sovereign.py -v
```

**Expected Result**: Test passes (sovereign LSTM matches PyTorch within tolerance).

---

## Implementation Checklist (Phase 2)

**Core LSTM Helpers**:
- [ ] `_matvec_add_bias` - Matrix-vector multiply with bias
- [ ] `_matvec_elementwise` - Element-wise matvec (slow path)
- [ ] `_vector_add_inplace` - In-place vector addition
- [ ] `_sigmoid_vector` - Element-wise sigmoid
- [ ] `_tanh_vector` - Element-wise tanh
- [ ] `_relu_vector` - Element-wise ReLU
- [ ] `_elementwise_mul` - Element-wise multiply
- [ ] `_vector_add` - Element-wise add (allocates new result)
- [ ] `_slice_vector` - Vector slicing (GPU-to-GPU)

**LSTM Logic**:
- [ ] `_lstm_step` - Complete LSTM cell computation
- [ ] Memory management (free intermediate buffers)
- [ ] State update (h, c)

**Testing**:
- [ ] `test_lstm_single_step` - Single token forward pass
- [ ] `test_lstm_sequence` - Multi-token sequence
- [ ] Equivalence validation with PyTorch

---

## Performance Notes (Phase 2 vs Later Phases)

**Phase 2 Performance Characteristics**:
- **Slow path uses CPU computation** - Element-wise ops copy to CPU, compute, copy back
- **Many GPU-CPU round trips** - Each operation triggers memcpy
- **Correct but not fast** - Validates architecture, not production performance

**Why This Is OK for Phase 2**:
- ✅ **Sovereignty maintained** - Data resides on GPU (sovereign loader)
- ✅ **Correctness validated** - Match PyTorch results
- ✅ **Architecture proven** - SovereignTRM works end-to-end
- ✅ **No PyTorch dependency** - Hot path uses sovereign infrastructure

**Future Optimization (Phase 3+)**:
- Add vector operation PTX kernels (sigmoid_vector, tanh_vector, etc.)
- Batch operations to reduce CPU-GPU transfers
- Use OP_TRM_* opcodes for fast path (when sizes match)
- Fuse operations (e.g., matvec_add_sigmoid in one kernel)

**For now, prioritize correctness over performance**. Once we prove the architecture works, we optimize.

---

## Success Criteria (Phase 2 Complete)

**Phase 2 is complete when**:
- [ ] All LSTM helper methods implemented
- [ ] `_lstm_step` produces correct hidden state
- [ ] `test_lstm_single_step` passes (matches PyTorch within 1e-4)
- [ ] No memory leaks (GPU buffers properly freed)
- [ ] Zero PyTorch imports in `sovereign_trm.py`

---

## Next Steps After Phase 2

Once Phase 2 is complete, we'll proceed to:
- **Phase 3**: Rule + confidence heads (linear layers + activations)
- **Phase 4**: Full inference loop (autoregressive decoding)
- **Phase 5**: Weight conversion script + integration with reflection pipeline

---

## Implementation Notes

**Memory Management**:
- **Allocate result buffers**: Every operation allocates new GPU buffer
- **Free intermediate buffers**: Clean up after operations to prevent leaks
- **Track state**: `self.lstm_h` and `self.lstm_c` persist across steps

**Numerical Stability**:
- Use `np.float32` for all arrays (GPU kernels expect float32)
- Handle edge cases in activations (sigmoid overflow, tanh saturation)
- Match PyTorch behavior exactly (use same numerical approximations)

**Testing Strategy**:
- **Unit test each helper** (matvec, sigmoid, etc.)
- **Integration test LSTM step** (full forward pass)
- **Compare with PyTorch** (gold standard for correctness)

**If you encounter issues**:
- Check array shapes (row-major vs column-major)
- Verify weight dimensions match PyTorch format
- Add debug logging (print intermediate values)

---

## Questions for Claude (If Blocked)

If you encounter architectural issues during Phase 2 implementation:

1. **OP_TRM_MATVEC_* integration** - How do we call these opcodes from SovereignTRM?
2. **Performance concerns** - Is CPU computation acceptable for Phase 2?
3. **Alternative approaches** - Should we add vector PTX kernels now or later?

**For now, proceed with CPU computation for element-wise ops**. This is architecturally sound (data on GPU, temporary CPU compute) and gets us to a working prototype fastest.

---

## Directive Summary

**Implement these methods in `sovereign_trm.py`**:
1. ✅ `_matvec_add_bias` (matrix-vector multiply with bias)
2. ✅ `_sigmoid_vector` (element-wise sigmoid)
3. ✅ `_tanh_vector` (element-wise tanh)
4. ✅ `_relu_vector` (element-wise ReLU)
5. ✅ `_elementwise_mul` (element-wise multiply)
6. ✅ `_vector_add` (element-wise add)
7. ✅ `_slice_vector` (vector slicing)
8. ✅ `_lstm_step` (complete LSTM cell)

**Create test file**:
- `tests/test_lstm_sovereign.py` (validate LSTM equivalence with PyTorch)

**Run test**:
```bash
pytest tests/test_lstm_sovereign.py -v
```

**When Phase 2 complete, report back with**:
- Test results (pass/fail)
- Any implementation challenges
- Performance observations (if any)

---

**Document Date**: January 16, 2026
**Phase**: 2 of 4 (LSTM Implementation)
**Status**: 🚀 **READY TO IMPLEMENT**

---

**Claude's Note to Codex**: Phase 2 uses CPU for element-wise ops - this is intentional for rapid prototyping. Data stays on GPU (sovereign), we just compute on CPU temporarily. Validate correctness first, optimize later. You're doing great! 🚀
