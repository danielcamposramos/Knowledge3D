# Codex Directive: Sovereign TRM Phase 3 - Rule & Confidence Heads

**From**: Claude (Architecture Partner)
**To**: Codex (Implementation Specialist)
**Date**: January 16, 2026
**Subject**: **Phase 3: Implement Rule Classification & Confidence Regression Heads**

---

## Phase 2 Complete ✅

**Excellent work on Phase 2!**

You've successfully:
- ✅ Implemented LSTM with RPN batch execution (100% GPU)
- ✅ Fixed `cuMemcpyDtoD` fallback (v2→v1 for driver compatibility)
- ✅ Solved tanh opcode issue (mathematically correct sigmoid identity)
- ✅ Tests pass (`test_lstm_sovereign.py`)
- ✅ Zero NumPy in hot path (sovereignty maintained)

**The tanh workaround is correct**:
- `tanh(x) = 2*sigmoid(2x) - 1` is mathematically exact
- Uses existing `OP_SIGMOID_APPROX` PTX opcode
- No need to regenerate PTX kernel now (can add native tanh later if needed)

---

## Phase 3 Objective

**Implement rule classification head and confidence regression head** using the same RPN batch pattern from Phase 2.

### What We're Building

**V7 Model Architecture** (for reference):
```python
class NavigationModelWithConfidence:
    def __init__(self, vocab_size, hidden_dim=512):
        # LSTM (Phase 2 - DONE ✅)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # Rule head (Phase 3 - THIS PHASE)
        self.rule_head = nn.Linear(hidden_dim, vocab_size + 3)

        # Confidence head (Phase 3 - THIS PHASE)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
```

**Rule Head**: Linear layer (hidden_dim → vocab_size + 3)
- Input: Hidden state from LSTM (512-dim)
- Output: Logits for each rule (vocab_size + 3 classes)
- Operation: `y = W @ h + b` (matrix-vector multiply + bias)

**Confidence Head**: 2-layer MLP with ReLU + Sigmoid
- Layer 1: Linear (hidden_dim → hidden_dim // 2) + ReLU
- Layer 2: Linear (hidden_dim // 2 → 1) + Sigmoid
- Output: Confidence score [0, 1]

---

## Implementation Tasks

### Task 1: Implement `_rule_head` (Linear Layer)

**File**: `knowledge3d/cranium/sovereign_trm.py`

**Current State** (stub):
```python
def _rule_head(self, hidden: loader.CUdeviceptr) -> loader.CUdeviceptr:
    """Rule classification head (linear layer)."""
    return self._matvec_add_bias(
        self.weights['rule_head_weight'],
        hidden,
        self.weights['rule_head_bias'],
        rows=self.vocab_size + 3,
        cols=self.hidden_dim
    )
```

**Status**: ✅ **ALREADY IMPLEMENTED** (uses existing `_matvec_add_bias` from Phase 2)

**Test**:
```python
def test_rule_head():
    """Test rule head produces correct logits."""
    trm = SovereignTRM(vocab_size=256, hidden_dim=512)
    trm.load_weights('checkpoints/v7_converted')

    # Create dummy hidden state
    hidden = loader.gpu_malloc(512 * 4)
    hidden_np = np.random.randn(512).astype(np.float32)
    loader.cpu_to_gpu(hidden, hidden_np)

    # Rule head
    logits = trm._rule_head(hidden)

    # Copy to CPU
    logits_np = loader.gpu_to_cpu_array(logits, 256 + 3)

    assert logits_np.shape == (259,)  # vocab_size + 3

    # Cleanup
    loader.gpu_free(hidden)
    loader.gpu_free(logits)
    trm.cleanup()
```

---

### Task 2: Implement `_confidence_head` (2-Layer MLP)

**File**: `knowledge3d/cranium/sovereign_trm.py`

**Current State** (stub):
```python
def _confidence_head(self, hidden: loader.CUdeviceptr) -> float:
    """Confidence regression head (2-layer MLP)."""
    # TODO: Implement
    return 0.5
```

**Implementation**:
```python
def _confidence_head(self, hidden: loader.CUdeviceptr) -> float:
    """Confidence regression head (2-layer MLP).

    Architecture:
        h1 = ReLU(W1 @ hidden + b1)  (hidden_dim → hidden_dim // 2)
        h2 = Sigmoid(W2 @ h1 + b2)   (hidden_dim // 2 → 1)

    Args:
        hidden: Hidden state (hidden_dim,) on GPU

    Returns:
        Confidence score [0, 1] (scalar on CPU)
    """
    # Layer 1: Linear + ReLU
    h1 = self._matvec_add_bias(
        self.weights['confidence_head_0_weight'],
        hidden,
        self.weights['confidence_head_0_bias'],
        rows=self.hidden_dim // 2,
        cols=self.hidden_dim
    )

    # ReLU activation
    h1 = self._relu_vector(h1, self.hidden_dim // 2)

    # Layer 2: Linear + Sigmoid
    h2 = self._matvec_add_bias(
        self.weights['confidence_head_2_weight'],
        h1,
        self.weights['confidence_head_2_bias'],
        rows=1,
        cols=self.hidden_dim // 2
    )

    # Read scalar from GPU
    confidence_raw = loader.gpu_to_cpu_scalar(h2)

    # Sigmoid activation (CPU OK - single scalar)
    confidence = 1.0 / (1.0 + math.exp(-confidence_raw))

    # Cleanup intermediate buffers
    loader.gpu_free(h1)
    loader.gpu_free(h2)

    return confidence
```

**Key Points**:
- Uses existing `_matvec_add_bias` (Phase 2 implementation)
- Uses existing `_relu_vector` (Phase 2 implementation)
- Sigmoid on CPU (acceptable for single scalar)
- Memory management (free intermediate buffers)

---

### Task 3: Implement `_argmax` (Rule Selection)

**File**: `knowledge3d/cranium/sovereign_trm.py`

**Current State** (stub):
```python
def _argmax(self, logits: loader.CUdeviceptr) -> int:
    """Find index of maximum value (argmax)."""
    # TODO: Implement
    return 0
```

**Implementation**:
```python
def _argmax(self, logits: loader.CUdeviceptr) -> int:
    """Find index of maximum value (argmax).

    For now, copy to CPU and use NumPy (acceptable for single operation).

    Args:
        logits: Logits array (vocab_size + 3,) on GPU

    Returns:
        Index of maximum value (int)
    """
    # Copy logits to CPU
    logits_cpu = loader.gpu_to_cpu_array(logits, self.vocab_size + 3)

    # Argmax on CPU (acceptable - single operation after GPU inference)
    import numpy as np
    return int(np.argmax(logits_cpu))
```

**Alternative** (future optimization):
```python
# Build RPN program to find max (using pairwise max operations)
# This would keep argmax on GPU, but adds complexity
# For Phase 3, CPU argmax is acceptable (happens once per step)
```

---

### Task 4: Update `infer` Method (Autoregressive Decoding)

**File**: `knowledge3d/cranium/sovereign_trm.py`

**Current State** (stub):
```python
def infer(
    self,
    problem_tokens: list[int],
    max_rules: int = 20
) -> Tuple[list[int], list[float]]:
    """Run sovereign TRM inference (no PyTorch)."""
    # TODO: Implement autoregressive decoding
    return [], []
```

**Implementation**:
```python
def infer(
    self,
    problem_tokens: list[int],
    max_rules: int = 20
) -> Tuple[list[int], list[float]]:
    """Run sovereign TRM inference (no PyTorch).

    Args:
        problem_tokens: List of token IDs (integers)
        max_rules: Maximum number of rules to predict

    Returns:
        (rule_sequence, confidence_scores)
        - rule_sequence: List of predicted rule IDs
        - confidence_scores: List of confidence scores [0, 1]

    Example:
        >>> trm = SovereignTRM(vocab_size=256)
        >>> trm.load_weights('checkpoints/v7_converted')
        >>> rules, confidences = trm.infer([1, 42, 15, 3])
        >>> print(rules)  # [10, 23, 8, 5]
        >>> print(confidences)  # [0.95, 0.87, 0.92, 0.99]
    """
    self.reset_lstm_state()

    rule_sequence = []
    confidence_scores = []

    # Encode problem (run LSTM over problem tokens)
    for token_id in problem_tokens:
        self._lstm_step(token_id)

    # Decode (autoregressively predict rules)
    current_token = 0  # <START> token (assumes token 0 is start token)

    for _ in range(max_rules):
        # LSTM step with current token
        hidden = self._lstm_step(current_token)

        # Rule head (classification)
        rule_logits = self._rule_head(hidden)
        next_rule = self._argmax(rule_logits)

        # Confidence head (regression)
        confidence = self._confidence_head(hidden)

        # Check for <END> token
        end_token_id = self.vocab_size + 2  # Assumes vocab_size+2 is <END>
        if next_rule == end_token_id:
            # Free buffers before breaking
            loader.gpu_free(rule_logits)
            break

        rule_sequence.append(next_rule)
        confidence_scores.append(confidence)

        # Next input is predicted rule
        current_token = next_rule

        # Free buffers
        loader.gpu_free(rule_logits)
        # hidden is owned by LSTM state (self.lstm_h), don't free here

    return rule_sequence, confidence_scores
```

**Key Points**:
- **Encoding phase**: Process problem tokens (build context)
- **Decoding phase**: Autoregressively predict rules
- **Early stopping**: Break on <END> token
- **Memory management**: Free buffers after each step
- **LSTM state**: Persists across steps (don't free `hidden` - it's `self.lstm_h`)

**Token Conventions** (adjust if different in your vocabulary):
- Token 0: `<START>` (begin decoding)
- Token vocab_size+0: `<PAD>` (padding)
- Token vocab_size+1: `<UNK>` (unknown)
- Token vocab_size+2: `<END>` (end of sequence)

---

## Testing Strategy

### Test 1: Rule Head Equivalence

**File**: `tests/test_sovereign_trm_heads.py` (NEW)

```python
"""Test Sovereign TRM heads against PyTorch reference."""
import pytest
import numpy as np
import torch
import torch.nn as nn
from knowledge3d.cranium.sovereign_trm import SovereignTRM
from knowledge3d.cranium.sovereign import loader


def test_rule_head_equivalence():
    """Test rule head produces same logits as PyTorch."""
    vocab_size = 256
    hidden_dim = 512

    # Create PyTorch rule head
    pt_rule_head = nn.Linear(hidden_dim, vocab_size + 3)
    pt_rule_head.eval()

    # Create Sovereign TRM
    trm = SovereignTRM(vocab_size=vocab_size, hidden_dim=hidden_dim)

    # Convert PyTorch weights to NumPy and load into TRM
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save PyTorch weights
        np.save(os.path.join(tmpdir, 'rule_head_weight.npy'),
                pt_rule_head.weight.detach().cpu().numpy())
        np.save(os.path.join(tmpdir, 'rule_head_bias.npy'),
                pt_rule_head.bias.detach().cpu().numpy())

        # Dummy weights for other components (not tested here)
        np.save(os.path.join(tmpdir, 'embedding.npy'),
                np.zeros((vocab_size, 256), dtype=np.float32))
        np.save(os.path.join(tmpdir, 'lstm_weight_ih.npy'),
                np.zeros((2048, 256), dtype=np.float32))
        np.save(os.path.join(tmpdir, 'lstm_weight_hh.npy'),
                np.zeros((2048, hidden_dim), dtype=np.float32))
        np.save(os.path.join(tmpdir, 'lstm_bias_ih.npy'),
                np.zeros(2048, dtype=np.float32))
        np.save(os.path.join(tmpdir, 'lstm_bias_hh.npy'),
                np.zeros(2048, dtype=np.float32))
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

    # Test input (hidden state)
    hidden_np = np.random.randn(hidden_dim).astype(np.float32)
    hidden_pt = torch.tensor(hidden_np)

    # PyTorch inference
    with torch.no_grad():
        pt_logits = pt_rule_head(hidden_pt).cpu().numpy()

    # Sovereign TRM inference
    hidden_gpu = loader.gpu_malloc(hidden_dim * 4)
    loader.cpu_to_gpu(hidden_gpu, hidden_np)
    sov_logits_gpu = trm._rule_head(hidden_gpu)
    sov_logits = loader.gpu_to_cpu_array(sov_logits_gpu, vocab_size + 3)

    # Compare (within tolerance)
    np.testing.assert_allclose(sov_logits, pt_logits, rtol=1e-4, atol=1e-5)

    # Cleanup
    loader.gpu_free(hidden_gpu)
    loader.gpu_free(sov_logits_gpu)
    trm.cleanup()


def test_confidence_head_equivalence():
    """Test confidence head produces same score as PyTorch."""
    hidden_dim = 512

    # Create PyTorch confidence head
    pt_conf_head = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.ReLU(),
        nn.Linear(hidden_dim // 2, 1),
        nn.Sigmoid()
    )
    pt_conf_head.eval()

    # Create Sovereign TRM
    trm = SovereignTRM(vocab_size=256, hidden_dim=hidden_dim)

    # Convert PyTorch weights to NumPy and load into TRM
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save PyTorch weights
        np.save(os.path.join(tmpdir, 'confidence_head_0_weight.npy'),
                pt_conf_head[0].weight.detach().cpu().numpy())
        np.save(os.path.join(tmpdir, 'confidence_head_0_bias.npy'),
                pt_conf_head[0].bias.detach().cpu().numpy())
        np.save(os.path.join(tmpdir, 'confidence_head_2_weight.npy'),
                pt_conf_head[2].weight.detach().cpu().numpy())
        np.save(os.path.join(tmpdir, 'confidence_head_2_bias.npy'),
                pt_conf_head[2].bias.detach().cpu().numpy())

        # Dummy weights for other components
        np.save(os.path.join(tmpdir, 'embedding.npy'),
                np.zeros((256, 256), dtype=np.float32))
        np.save(os.path.join(tmpdir, 'lstm_weight_ih.npy'),
                np.zeros((2048, 256), dtype=np.float32))
        np.save(os.path.join(tmpdir, 'lstm_weight_hh.npy'),
                np.zeros((2048, hidden_dim), dtype=np.float32))
        np.save(os.path.join(tmpdir, 'lstm_bias_ih.npy'),
                np.zeros(2048, dtype=np.float32))
        np.save(os.path.join(tmpdir, 'lstm_bias_hh.npy'),
                np.zeros(2048, dtype=np.float32))
        np.save(os.path.join(tmpdir, 'rule_head_weight.npy'),
                np.zeros((259, hidden_dim), dtype=np.float32))
        np.save(os.path.join(tmpdir, 'rule_head_bias.npy'),
                np.zeros(259, dtype=np.float32))

        # Load weights into TRM
        trm.load_weights(tmpdir)

    # Test input (hidden state)
    hidden_np = np.random.randn(hidden_dim).astype(np.float32)
    hidden_pt = torch.tensor(hidden_np).unsqueeze(0)  # (1, hidden_dim)

    # PyTorch inference
    with torch.no_grad():
        pt_conf = pt_conf_head(hidden_pt).item()

    # Sovereign TRM inference
    hidden_gpu = loader.gpu_malloc(hidden_dim * 4)
    loader.cpu_to_gpu(hidden_gpu, hidden_np)
    sov_conf = trm._confidence_head(hidden_gpu)

    # Compare (within tolerance)
    assert abs(sov_conf - pt_conf) < 1e-4

    # Cleanup
    loader.gpu_free(hidden_gpu)
    trm.cleanup()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

**Run Tests**:
```bash
# Enable GPU probing
export K3D_PYTEST_PROBE_CUDA=1

# Run tests
pytest tests/test_sovereign_trm_heads.py -v
```

---

### Test 2: Full Inference Pipeline

**File**: `tests/test_sovereign_trm_inference.py` (NEW)

```python
"""Test full Sovereign TRM inference pipeline."""
import pytest
from knowledge3d.cranium.sovereign_trm import SovereignTRM


def test_inference_runs():
    """Test inference completes without errors."""
    trm = SovereignTRM(vocab_size=256, embedding_dim=256, hidden_dim=512)

    # Create dummy checkpoint (random weights)
    import tempfile
    import os
    import numpy as np

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create random weights
        np.save(os.path.join(tmpdir, 'embedding.npy'),
                np.random.randn(256, 256).astype(np.float32) * 0.01)
        np.save(os.path.join(tmpdir, 'lstm_weight_ih.npy'),
                np.random.randn(2048, 256).astype(np.float32) * 0.01)
        np.save(os.path.join(tmpdir, 'lstm_weight_hh.npy'),
                np.random.randn(2048, 512).astype(np.float32) * 0.01)
        np.save(os.path.join(tmpdir, 'lstm_bias_ih.npy'),
                np.zeros(2048, dtype=np.float32))
        np.save(os.path.join(tmpdir, 'lstm_bias_hh.npy'),
                np.zeros(2048, dtype=np.float32))
        np.save(os.path.join(tmpdir, 'rule_head_weight.npy'),
                np.random.randn(259, 512).astype(np.float32) * 0.01)
        np.save(os.path.join(tmpdir, 'rule_head_bias.npy'),
                np.zeros(259, dtype=np.float32))
        np.save(os.path.join(tmpdir, 'confidence_head_0_weight.npy'),
                np.random.randn(256, 512).astype(np.float32) * 0.01)
        np.save(os.path.join(tmpdir, 'confidence_head_0_bias.npy'),
                np.zeros(256, dtype=np.float32))
        np.save(os.path.join(tmpdir, 'confidence_head_2_weight.npy'),
                np.random.randn(1, 256).astype(np.float32) * 0.01)
        np.save(os.path.join(tmpdir, 'confidence_head_2_bias.npy'),
                np.zeros(1, dtype=np.float32))

        # Load weights
        trm.load_weights(tmpdir)

    # Run inference
    problem_tokens = [1, 42, 15, 3]
    rules, confidences = trm.infer(problem_tokens, max_rules=5)

    # Validate output structure
    assert isinstance(rules, list)
    assert isinstance(confidences, list)
    assert len(rules) == len(confidences)
    assert len(rules) <= 5  # Should not exceed max_rules

    # Validate confidence range
    for conf in confidences:
        assert 0.0 <= conf <= 1.0

    # Cleanup
    trm.cleanup()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

---

## Implementation Checklist (Phase 3)

**Core Components**:
- [ ] `_rule_head` - Linear layer (already implemented, just verify)
- [ ] `_confidence_head` - 2-layer MLP with ReLU + Sigmoid
- [ ] `_argmax` - Find index of maximum value
- [ ] `infer` - Autoregressive decoding loop

**Testing**:
- [ ] `test_rule_head_equivalence` - Match PyTorch rule head
- [ ] `test_confidence_head_equivalence` - Match PyTorch confidence head
- [ ] `test_inference_runs` - Full inference pipeline works
- [ ] Memory leak check (no GPU buffers left allocated)

---

## Success Criteria (Phase 3 Complete)

**Phase 3 is complete when**:
- [ ] `_confidence_head` fully implemented
- [ ] `_argmax` implemented
- [ ] `infer` method fully implemented
- [ ] Tests pass (`test_sovereign_trm_heads.py`, `test_sovereign_trm_inference.py`)
- [ ] Heads match PyTorch within 1e-4 tolerance
- [ ] Full inference runs without errors
- [ ] No memory leaks (GPU buffers properly freed)
- [ ] Zero PyTorch imports in `sovereign_trm.py`

---

## Next Steps After Phase 3

Once Phase 3 is complete, we'll proceed to:
- **Phase 4**: Weight conversion script + integration with reflection pipeline
- **Phase 5**: End-to-end testing with real V7 checkpoint
- **Phase 6**: Performance benchmarking and optimization

---

## Notes for Codex

**Phase 3 is mostly glue code** - assembling components you already built in Phase 2:
- `_rule_head`: Uses `_matvec_add_bias` (Phase 2)
- `_confidence_head`: Uses `_matvec_add_bias` + `_relu_vector` (Phase 2)
- `_argmax`: Simple NumPy operation (acceptable for single scalar)
- `infer`: Orchestrates LSTM + heads in autoregressive loop

**Memory Management is Critical**:
- Free `rule_logits` after each decoding step
- DON'T free `hidden` (it's `self.lstm_h`, owned by LSTM state)
- Free intermediate buffers in `_confidence_head` (h1, h2)

**Testing Strategy**:
- Unit test each head (equivalence with PyTorch)
- Integration test full inference (structure validation)
- No need for numerical equivalence on random weights (just verify it runs)

**For GPU Testing**:
```bash
# Enable GPU probing in pytest
export K3D_PYTEST_PROBE_CUDA=1

# Run with GPU
pytest tests/test_sovereign_trm_heads.py -v
```

**If you encounter issues**:
- Check token conventions (<START>, <END> token IDs)
- Verify weight dimensions match PyTorch format
- Add debug logging (print intermediate shapes/values)

---

## PTX Kernel Notes (Tanh)

**Current tanh implementation is correct**:
- `tanh(x) = 2*sigmoid(2x) - 1` is mathematically exact
- Uses existing `OP_SIGMOID_APPROX` PTX opcode
- No need to regenerate PTX now

**When to add native tanh opcode**:
- After Phase 4 complete (full system working)
- If performance profiling shows tanh is bottleneck
- When rebuilding PTX kernel for other reasons

**For now, proceed with sigmoid-based tanh** - it's correct and sufficient.

---

**Document Date**: January 16, 2026
**Phase**: 3 of 4 (Rule & Confidence Heads)
**Status**: 🚀 **READY TO IMPLEMENT**

---

**Claude's Note to Codex**: Phase 3 is assembly work - you've already built the building blocks in Phase 2. Just wire them together in `_confidence_head` and `infer`. The tanh workaround is mathematically correct, no need to change it. You're doing excellent work! 🚀
