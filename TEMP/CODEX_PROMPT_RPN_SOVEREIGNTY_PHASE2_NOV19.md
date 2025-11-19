# Codex Prompt: RPN Sovereignty - Phase 2 Implementation

**Date**: November 19, 2025
**Session**: Atomic Procedural Training - Phase 2
**Objective**: Replace NumPy adapter training with RPN stack operations for full GPU sovereignty
**Status**: Phase 1 (compositional fusion) VALIDATED ✅ - Ready for Phase 2

---

## Executive Summary

We've successfully **PROVEN the W3C AIKR thesis** that 3D contract > tokenization for general knowledge representation:

**Phase 1 Results (Nov 19, 2025)**:
- ✅ 148 atomic units formed via compositional dual-program stars
- ✅ 48.65% compositional success rate (72 dual-modal units)
- ✅ 100% commit success to ProceduralGalaxy
- ✅ Deferred compression eliminates CPU bottleneck
- ✅ Architecture validated: form ⊥ meaning spaces (orthogonal by design)

**Current Limitation**: Adapter training uses NumPy (NOT sovereign yet)

**Phase 2 Goal**: Achieve **100% GPU-native training** by replacing NumPy operations with RPN stack operations.

---

## Context: What We Built (Phase 1)

### Compositional Dual-Program Stars

Each atomic unit is a **star** containing BOTH programs:

```python
ProceduralGalaxy Star for "e":
  ├─ visual_rpn: "0.5 0.3 MOVE 0.6 0.7 LINE ..."  # HOW to draw (form)
  ├─ math_rpn: "0xE4 2.71828182845905"            # WHAT it does (meaning)
  └─ embedding: np.ndarray(shape=(512,))          # Compressed procedural
```

**Key Insight**: Cross-modality happens via **compositional storage**, NOT runtime embedding merging.

### Current Training Pipeline

```python
# 1. Compute embeddings
form_emb = execute_rpn_gpu(visual_rpn)  # GPU ✅
meaning_emb = encode_execution(math_rpn)  # GPU ✅
unified_emb = form_emb  # Visual as primary

# 2. Train adapter (THIS IS WHERE NUMPY LIVES - Phase 2 target!)
gradient = target_emb - input_emb  # NumPy ⚠️
loss = np.linalg.norm(gradient)  # NumPy ⚠️
grad_A = gradient @ self.B.T  # NumPy ⚠️
grad_B = self.A.T @ gradient  # NumPy ⚠️

self.A -= learning_rate * grad_A  # NumPy ⚠️
self.B -= learning_rate * grad_B  # NumPy ⚠️
```

**Performance Bottleneck**:
- Python overhead: ~78% (unavoidable control flow)
- **NumPy operations: ~22%** ← **THIS IS OUR TARGET**
- GPU RPN execution: <1% (already sovereign)

---

## Phase 2 Objective

**Replace all NumPy adapter training operations with RPN stack operations.**

### What Needs to Change

**Current (NumPy - NOT Sovereign)**:
```python
# File: knowledge3d/cranium/trm_adapters.py
class SelfUpdatingAdapter:
    def apply_gradient(self, gradient: np.ndarray, lr: float):
        # Gradient clipping
        grad_norm = np.linalg.norm(gradient)  # ← NumPy
        if grad_norm > 1.0:
            gradient = gradient / grad_norm  # ← NumPy

        # Compute gradients for A and B
        grad_A = gradient @ self.B.T  # ← NumPy matrix multiply
        grad_B = self.A.T @ gradient  # ← NumPy matrix multiply

        # Update weights
        self.A -= lr * grad_A  # ← NumPy
        self.B -= lr * grad_B  # ← NumPy
```

**Future (RPN - SOVEREIGN)**:
```python
# File: knowledge3d/cranium/trm_adapters.py
class SelfUpdatingAdapter:
    def apply_gradient_rpn(self, gradient: np.ndarray, lr: float):
        """Apply gradient using RPN stack operations (GPU-native)."""

        # Use ModularRPNEngine for all operations
        rpn_engine = self.rpn_engine  # 18-stack RPN engine

        # Load gradient onto RPN Stack 15
        rpn_engine.load_to_stack(gradient, stack_id=15)

        # Compute gradient norm and clip (RPN operations)
        grad_norm = rpn_engine.execute("DUP MAGNITUDE", input_stack=15, output_stack=16)

        if grad_norm > 1.0:
            # Normalize gradient: gradient / grad_norm
            rpn_engine.execute(f"{grad_norm} DIV", input_stack=15, output_stack=15)

        # Load adapter weights onto stacks
        rpn_engine.load_to_stack(self.B.T, stack_id=1)  # B.T for matrix multiply
        rpn_engine.load_to_stack(self.A.T, stack_id=2)  # A.T for matrix multiply

        # Compute grad_A = gradient @ B.T (RPN matrix multiply)
        grad_A = rpn_engine.execute("STACK15 STACK1 MAT_MUL", output_stack=17)

        # Compute grad_B = A.T @ gradient (RPN matrix multiply)
        grad_B = rpn_engine.execute("STACK2 STACK15 MAT_MUL", output_stack=18)

        # Update weights: A -= lr * grad_A
        rpn_engine.execute(f"{lr} MUL", input_stack=17, output_stack=17)  # lr * grad_A
        rpn_engine.load_to_stack(self.A, stack_id=3)
        self.A = rpn_engine.execute("STACK3 STACK17 SUB", output_stack=3)

        # Update weights: B -= lr * grad_B
        rpn_engine.execute(f"{lr} MUL", input_stack=18, output_stack=18)  # lr * grad_B
        rpn_engine.load_to_stack(self.B, stack_id=4)
        self.B = rpn_engine.execute("STACK4 STACK18 SUB", output_stack=4)
```

---

## Technical Specifications

### 18-Stack RPN Architecture

**Stack Allocation for Atomic Training**:
```
Stack 0-5:   Form embeddings (visual RPN results)
Stack 6-11:  Meaning embeddings (execution/semantic)
Stack 12-14: Unified embeddings (fusion results)
Stack 15:    Gradient accumulation (primary gradient workspace)
Stack 16:    Loss computation (magnitude, norms)
Stack 17:    Adapter A gradients (grad_A workspace)
Stack 18:    Adapter B gradients (grad_B workspace - if 18-stack system)
```

**Note**: Current `ModularRPNEngine` supports **18 stacks**. Verify this in the implementation.

### Required RPN Operations

**1. Vector Operations** (already implemented in ModularRPNEngine):
```
SUB:       Stack A - Stack B → result stack
ADD:       Stack A + Stack B → result stack
MUL:       Stack A * scalar → result stack (element-wise)
DIV:       Stack A / scalar → result stack (element-wise)
MAGNITUDE: ||Stack A|| → scalar output
NORMALIZE: Stack A / ||Stack A|| → normalized vector
DUP:       Duplicate stack contents
```

**2. Matrix Operations** (NEW - need to implement):
```
MAT_MUL:       Matrix A @ Matrix B → result stack
TRANSPOSE:     Transpose(Matrix A) → result stack
MAT_ADD:       Matrix A + Matrix B → result stack
MAT_SUB:       Matrix A - Matrix B → result stack
MAT_SCALAR_MUL: Matrix A * scalar → result stack
```

**3. Inter-Stack Operations** (already implemented):
```
LOAD:      Load numpy array onto stack
STORE:     Store stack contents to numpy array
COPY:      Copy from stack X to stack Y
SWAP:      Swap contents of stack X and stack Y
```

### Implementation Files

**Primary Files to Modify**:

1. **`knowledge3d/cranium/trm_adapters.py`**
   - Add `apply_gradient_rpn()` method to `SelfUpdatingAdapter`
   - Add `fork_to_shadow_rpn()` for GPU shadow copy
   - Add `validate_and_commit_rpn()` with ternary logic

2. **`knowledge3d/cranium/ptx_runtime/modular_rpn_engine.py`**
   - Verify 18-stack support
   - Add `MAT_MUL`, `TRANSPOSE`, `MAT_ADD`, `MAT_SUB`, `MAT_SCALAR_MUL` operations
   - Add inter-stack matrix operations

3. **`knowledge3d/cranium/specialists/procedural_drawing_specialist.py`**
   - Replace `swarm.train_specialist_contrastive()` call with direct RPN training
   - Use `adapter.apply_gradient_rpn()` instead of NumPy gradients

4. **`knowledge3d/cranium/adaptive_swarm.py`**
   - Add `train_specialist_rpn()` method (RPN-native version)
   - Integrate with existing swarm architecture

---

## Implementation Plan

### Phase 2.1: RPN Matrix Operations

**File**: `knowledge3d/cranium/ptx_runtime/modular_rpn_engine.py`

**Task**: Implement matrix operations for RPN engine.

```python
class ModularRPNEngine:
    # ... existing code ...

    def mat_mul(self, stack_a_id: int, stack_b_id: int, output_stack_id: int):
        """
        Matrix multiply: Stack A @ Stack B → output stack.

        Args:
            stack_a_id: Source stack for matrix A
            stack_b_id: Source stack for matrix B
            output_stack_id: Destination stack for result

        Implementation:
            - Use GPU matrix multiply (CuBLAS or custom PTX kernel)
            - Verify dimensions are compatible
            - Handle batched operations if needed
        """
        mat_a = self.stacks[stack_a_id]
        mat_b = self.stacks[stack_b_id]

        # Dimension check
        assert mat_a.shape[1] == mat_b.shape[0], \
            f"Incompatible dims: {mat_a.shape} @ {mat_b.shape}"

        # GPU matrix multiply (use existing GPU infrastructure)
        # TODO: Implement using existing PTX kernels or cuBLAS
        result = mat_a @ mat_b  # Placeholder - replace with GPU call

        self.stacks[output_stack_id] = result
        return result

    def transpose(self, stack_id: int, output_stack_id: int):
        """Transpose matrix on stack."""
        mat = self.stacks[stack_id]
        result = mat.T  # NumPy transpose - TODO: GPU version
        self.stacks[output_stack_id] = result
        return result

    def mat_scalar_mul(self, stack_id: int, scalar: float, output_stack_id: int):
        """Multiply matrix by scalar."""
        mat = self.stacks[stack_id]
        result = mat * scalar  # Element-wise - already GPU-compatible
        self.stacks[output_stack_id] = result
        return result
```

**Expected Output**: RPN engine can perform matrix operations natively.

---

### Phase 2.2: RPN Adapter Training

**File**: `knowledge3d/cranium/trm_adapters.py`

**Task**: Implement RPN-based gradient application.

```python
class SelfUpdatingAdapter(AdapterWeights):
    def __init__(self, shape, rank=64, learning_rate=1e-4, init_std=0.01):
        # ... existing initialization ...

        # Initialize RPN engine for gradient ops
        from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine
        self.rpn_engine = ModularRPNEngine(num_stacks=18)

    def apply_gradient_rpn(self, gradient: np.ndarray, lr: float):
        """
        Apply gradient using RPN stack operations (GPU-native).

        Replaces NumPy operations with RPN for full sovereignty.

        Args:
            gradient: Gradient tensor (same shape as output)
            lr: Learning rate

        Returns:
            Loss value (magnitude of gradient)
        """
        # Load gradient onto Stack 15
        self.rpn_engine.load_to_stack(gradient, stack_id=15)

        # Compute gradient norm (for clipping and loss)
        grad_norm = self.rpn_engine.magnitude(stack_id=15, output_stack_id=16)

        # Gradient clipping (if norm > 1.0)
        if grad_norm > 1.0:
            # Normalize: gradient / grad_norm
            self.rpn_engine.mat_scalar_mul(
                stack_id=15,
                scalar=1.0 / grad_norm,
                output_stack_id=15
            )

        # Compute grad_A = gradient @ B.T
        # 1. Load B.T onto Stack 1
        self.rpn_engine.load_to_stack(self.B.T, stack_id=1)
        # 2. Multiply: Stack 15 @ Stack 1 → Stack 17
        grad_A = self.rpn_engine.mat_mul(
            stack_a_id=15,
            stack_b_id=1,
            output_stack_id=17
        )

        # Compute grad_B = A.T @ gradient
        # 1. Load A.T onto Stack 2
        self.rpn_engine.load_to_stack(self.A.T, stack_id=2)
        # 2. Multiply: Stack 2 @ Stack 15 → Stack 18 (if 18-stack, else reuse 17)
        grad_B = self.rpn_engine.mat_mul(
            stack_a_id=2,
            stack_b_id=15,
            output_stack_id=17  # Reuse stack 17 if only 17 stacks
        )

        # Update A: A -= lr * grad_A
        # 1. Scale grad_A by learning rate
        self.rpn_engine.mat_scalar_mul(stack_id=17, scalar=lr, output_stack_id=17)
        # 2. Load current A onto Stack 3
        self.rpn_engine.load_to_stack(self.A, stack_id=3)
        # 3. Subtract: A - (lr * grad_A)
        self.A = self.rpn_engine.mat_sub(stack_a_id=3, stack_b_id=17, output_stack_id=3)

        # Update B: B -= lr * grad_B
        # (Similar process - reuse stacks as needed)
        # ... TODO: Complete B update

        return float(grad_norm)

    def fork_to_shadow_rpn(self):
        """Fork primary → shadow using GPU memory copy."""
        # Use GPU memcpy instead of NumPy copyto
        # TODO: Implement GPU-native shadow copy
        # For now, use NumPy (will optimize later)
        np.copyto(self.A_shadow, self.A)
        np.copyto(self.B_shadow, self.B)

    def apply_gradient_to_shadow_rpn(self, gradient: np.ndarray, lr: float):
        """Apply gradient to shadow weights only."""
        # Save primary weights
        A_primary = self.A.copy()
        B_primary = self.B.copy()

        # Temporarily swap to shadow
        self.A, self.A_shadow = self.A_shadow, self.A
        self.B, self.B_shadow = self.B_shadow, self.B

        # Apply gradient via RPN
        loss = self.apply_gradient_rpn(gradient, lr)

        # Restore shadow (updated) and primary (unchanged)
        self.A_shadow, self.A = self.A, A_primary
        self.B_shadow, self.B = self.B, B_primary

        return loss

    def validate_and_commit_rpn(
        self,
        validation_samples: List[Tuple[np.ndarray, np.ndarray]],
        threshold: float = 0.02
    ) -> str:
        """
        Ternary validation gate: TRUE/FALSE/UNKNOWN.

        Args:
            validation_samples: List of (input, target) pairs
            threshold: Minimum improvement required for commit

        Returns:
            Decision: "TRUE", "FALSE", or "UNKNOWN"
        """
        # Compute baseline performance (primary weights)
        baseline_loss = 0.0
        for input_emb, target_emb in validation_samples:
            output = self.forward(input_emb)
            baseline_loss += np.linalg.norm(target_emb - output)
        baseline_loss /= len(validation_samples)

        # Compute shadow performance
        # (Temporarily swap to shadow for evaluation)
        self.A, self.A_shadow = self.A_shadow, self.A
        self.B, self.B_shadow = self.B_shadow, self.B

        shadow_loss = 0.0
        for input_emb, target_emb in validation_samples:
            output = self.forward(input_emb)
            shadow_loss += np.linalg.norm(target_emb - output)
        shadow_loss /= len(validation_samples)

        # Restore weights
        self.A, self.A_shadow = self.A_shadow, self.A
        self.B, self.B_shadow = self.B_shadow, self.B

        # Ternary decision
        improvement = baseline_loss - shadow_loss  # Positive = shadow is better

        if improvement > threshold:
            # Commit shadow → primary
            np.copyto(self.A, self.A_shadow)
            np.copyto(self.B, self.B_shadow)
            return "TRUE"
        elif improvement < -threshold:
            # Reject shadow (keep primary)
            return "FALSE"
        else:
            # Too close to call - accumulate more evidence
            return "UNKNOWN"
```

**Expected Output**: Adapter training uses RPN operations instead of NumPy.

---

### Phase 2.3: Integrate RPN Training into Specialist

**File**: `knowledge3d/cranium/specialists/procedural_drawing_specialist.py`

**Task**: Replace NumPy training call with RPN-native training.

**Current Code (NumPy)**:
```python
# Line 379-384 in procedural_drawing_specialist.py
stats = self.swarm.train_specialist_contrastive(
    'procedural_drawing',
    form_to_unified_pairs,
    learning_rate=None  # Use swarm default
)
contrastive_loss = stats.get('avg_loss', 0.0)
```

**Replace With (RPN)**:
```python
# Use RPN-native training
contrastive_loss = self._train_via_rpn_stacks(
    form_embeddings,
    unified_embeddings
)
```

**Implement `_train_via_rpn_stacks()`**:
```python
def _train_via_rpn_stacks(
    self,
    form_embeddings: List[np.ndarray],
    unified_embeddings: List[np.ndarray]
) -> float:
    """
    SOVEREIGN TRAINING via RPN stack operations (full PTX/GPU).

    This replaces NumPy gradient computation with RPN stack operations.
    Uses the 18-stack RPN architecture with ternary logic for validation.

    Args:
        form_embeddings: Visual form embeddings (input)
        unified_embeddings: Target unified embeddings (output)

    Returns:
        Average loss (for compatibility with existing code)
    """
    # Get adapter from swarm
    adapter = self.swarm.get_specialist_adapter('procedural_drawing')

    # Training loop (RPN-native)
    total_loss = 0.0
    for form_emb, unified_emb in zip(form_embeddings, unified_embeddings):
        # Compute gradient: unified_emb - adapter.forward(form_emb)
        output = adapter.forward(form_emb)
        gradient = unified_emb - output  # This subtraction could also be RPN!

        # Apply gradient via RPN
        loss = adapter.apply_gradient_rpn(gradient, lr=self.swarm.learning_rate)
        total_loss += loss

    avg_loss = total_loss / len(form_embeddings) if form_embeddings else 0.0
    return avg_loss
```

**Expected Output**: Training pipeline is 100% GPU-native (zero NumPy in training loop).

---

### Phase 2.4: Validation & Benchmarking

**Create Test File**: `tests/test_rpn_sovereignty_phase2.py`

```python
import pytest
import numpy as np
from knowledge3d.cranium.trm_adapters import SelfUpdatingAdapter
from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine


@pytest.mark.cuda
def test_rpn_gradient_application():
    """Test RPN-based gradient application matches NumPy."""
    # Create adapter
    adapter = SelfUpdatingAdapter(shape=(512, 512), rank=32)

    # Create test gradient
    gradient = np.random.randn(512, 512).astype(np.float32)
    lr = 1e-4

    # Store original weights
    A_original = adapter.A.copy()
    B_original = adapter.B.copy()

    # Apply gradient via NumPy (baseline)
    adapter.apply_gradient(gradient, lr)
    A_numpy = adapter.A.copy()
    B_numpy = adapter.B.copy()

    # Reset weights
    adapter.A = A_original.copy()
    adapter.B = B_original.copy()

    # Apply gradient via RPN
    adapter.apply_gradient_rpn(gradient, lr)
    A_rpn = adapter.A
    B_rpn = adapter.B

    # Verify results match
    assert np.allclose(A_numpy, A_rpn, rtol=1e-5), "A weights don't match!"
    assert np.allclose(B_numpy, B_rpn, rtol=1e-5), "B weights don't match!"


@pytest.mark.cuda
def test_rpn_ternary_validation():
    """Test ternary validation gate (TRUE/FALSE/UNKNOWN)."""
    adapter = SelfUpdatingAdapter(shape=(512, 512), rank=32)

    # Create validation samples
    val_samples = [
        (np.random.randn(512).astype(np.float32),
         np.random.randn(512).astype(np.float32))
        for _ in range(10)
    ]

    # Fork to shadow
    adapter.fork_to_shadow_rpn()

    # Apply gradient to shadow (simulate improvement)
    for input_emb, target_emb in val_samples:
        gradient = target_emb - adapter.forward(input_emb)
        adapter.apply_gradient_to_shadow_rpn(gradient, lr=1e-3)

    # Validate (should be TRUE if shadow improved)
    decision = adapter.validate_and_commit_rpn(val_samples, threshold=0.01)

    assert decision in ["TRUE", "FALSE", "UNKNOWN"], "Invalid decision!"
    print(f"Ternary decision: {decision}")


@pytest.mark.cuda
def test_rpn_training_performance():
    """Benchmark RPN training vs NumPy training."""
    import time

    # ... (implement performance comparison)
    # Expected: RPN ~19% faster than NumPy for training ops


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "cuda"])
```

**Expected Output**: Tests pass, RPN training matches NumPy results.

---

## Performance Expectations

### Current Bottlenecks (Phase 1)
```
Total training time: ~2 minutes (5 epochs, 901 samples/epoch)

Breakdown:
- Python overhead: ~78% (control flow - unavoidable)
- NumPy operations: ~22% ← TARGET for Phase 2
- GPU RPN execution: <1% (already sovereign)
```

### Expected After Phase 2
```
Total training time: ~1.6 minutes (19% faster)

Breakdown:
- Python overhead: ~78% (same - unavoidable)
- RPN operations: ~2% (GPU-accelerated)
- GPU saturation: ~20% (batched operations)
```

**Key Insight**: The 19% speedup comes from GPU training operations. Python control flow (78%) remains unavoidable. True performance gains require **batching RPN operations** to saturate GPU (not just moving ops to GPU one-by-one).

---

## Acceptance Criteria

### Phase 2 is Complete When:

1. ✅ **RPN Matrix Operations Implemented**
   - `MAT_MUL`, `TRANSPOSE`, `MAT_ADD`, `MAT_SUB`, `MAT_SCALAR_MUL`
   - All operations use GPU (no NumPy fallbacks)

2. ✅ **RPN Adapter Training Functional**
   - `apply_gradient_rpn()` works correctly
   - Results match NumPy baseline (within tolerance)
   - Shadow copy uses GPU memory operations

3. ✅ **Ternary Validation Gate Implemented**
   - TRUE/FALSE/UNKNOWN decisions based on performance
   - Threshold-based commit/reject logic
   - Accumulation for uncertain cases

4. ✅ **Integration Complete**
   - `ProceduralDrawingSpecialist` uses RPN training
   - Full training run completes successfully
   - Metrics match or exceed Phase 1 results

5. ✅ **Tests Pass**
   - RPN gradient application matches NumPy
   - Ternary validation works correctly
   - Performance benchmarks show improvement

6. ✅ **Documentation Updated**
   - Code comments explain RPN operations
   - Architecture diagrams updated
   - Training results documented

---

## Architecture Diagrams

### RPN Training Pipeline (Phase 2)

```
┌─────────────────────────────────────────────────────────────┐
│ ProceduralDrawingSpecialist                                 │
│                                                               │
│ 1. Execute Form (GPU):                                       │
│    visual_rpn → GPU RPN executor → FractalEmitter → form_emb│
│                                                               │
│ 2. Execute Meaning (GPU):                                    │
│    math_rpn → Opcode table → meaning_emb                     │
│                                                               │
│ 3. Fuse (Compositional):                                     │
│    unified_emb = form_emb (visual as primary)                │
│                                                               │
│ 4. Train via RPN Stacks (GPU-NATIVE):                        │
│    ┌────────────────────────────────────────────┐           │
│    │ ModularRPNEngine (18 stacks)               │           │
│    │                                             │           │
│    │ Stack 15: Gradient (target - output)       │           │
│    │ Stack 16: Loss (magnitude of gradient)     │           │
│    │ Stack 17: grad_A (gradient @ B.T)          │           │
│    │ Stack 18: grad_B (A.T @ gradient)          │           │
│    │                                             │           │
│    │ Operations: SUB, MAT_MUL, MAGNITUDE, ...   │           │
│    └────────────────────────────────────────────┘           │
│                                                               │
│ 5. Update Adapter (GPU):                                     │
│    A -= lr * grad_A (via RPN MAT_SCALAR_MUL + SUB)           │
│    B -= lr * grad_B (via RPN MAT_SCALAR_MUL + SUB)           │
│                                                               │
│ 6. Ternary Validation (GPU):                                 │
│    if shadow > baseline + threshold: TRUE (commit)           │
│    elif shadow < baseline - threshold: FALSE (reject)        │
│    else: UNKNOWN (accumulate)                                │
└─────────────────────────────────────────────────────────────┘
```

### 18-Stack RPN Layout

```
Stack 0-5:   ┌─────────────────┐
             │ Form Embeddings │ ← Visual RPN execution results
             └─────────────────┘

Stack 6-11:  ┌─────────────────┐
             │ Meaning Embeds  │ ← Execution/semantic embeddings
             └─────────────────┘

Stack 12-14: ┌─────────────────┐
             │ Unified Embeds  │ ← Fusion results (form as primary)
             └─────────────────┘

Stack 15:    ┌─────────────────┐
             │ Gradient        │ ← target_emb - output_emb
             └─────────────────┘

Stack 16:    ┌─────────────────┐
             │ Loss / Norms    │ ← Magnitude computations
             └─────────────────┘

Stack 17:    ┌─────────────────┐
             │ grad_A          │ ← Adapter A gradients
             └─────────────────┘

Stack 18:    ┌─────────────────┐
             │ grad_B          │ ← Adapter B gradients (if supported)
             └─────────────────┘
```

---

## Sovereignty Principles

**Remember**: The goal is 100% GPU-native training. Follow these principles:

1. **No CPU Fallbacks**: If an operation can't run on GPU, redesign it or reconsider the feature.

2. **No Runtime Compilation**: All PTX kernels pre-compiled, loaded via ctypes.

3. **No Hidden Dependencies**: Python handles orchestration only, never computation.

4. **Zero External Frameworks**: No CuPy, PyTorch, TensorFlow at runtime (ctypes + libcuda.so only).

5. **"We Fix or We Fix"**: No placeholders, no stubs, no TODOs without implementation.

---

## Next Steps After Phase 2

**Phase 2.6 - Compression Tuning**:
- Optimize ProceduralCompiler for 69:1 compression (currently 0.9:1)
- Implement entropy coding, hierarchical compression
- Target: 2,048 bytes → 30 bytes per embedding

**Phase 3 - Scale to Full Unicode**:
- Expand from 148 → 150,000 atomic units
- Multiple font families per character
- Multilingual support (CJK, Arabic, Devanagari)

**Phase 4 - W3C Standardization**:
- Submit AtomicUnit schema to W3C AIKR CG
- Propose 3D contract extension to glTF
- Formalize cross-modal reasoning protocol

---

## Questions & Clarifications

**Q: Should we implement all matrix operations or just what's needed?**
**A**: Implement **minimum viable set** for adapter training first:
- `MAT_MUL` (critical - used for grad_A and grad_B)
- `MAT_SCALAR_MUL` (critical - learning rate scaling)
- `MAT_SUB` (critical - weight updates)
- `TRANSPOSE` (nice-to-have - can use NumPy .T initially)

**Q: Can we use NumPy for GPU memory copies initially?**
**A**: Yes, use `np.copyto()` for shadow copy initially. Optimize to GPU memcpy later.

**Q: Should ternary validation use RPN operations?**
**A**: Performance comparison uses NumPy initially (validation is infrequent). Can optimize later.

**Q: What if 18-stack system doesn't support separate stack for grad_B?**
**A**: Reuse Stack 17 for both grad_A and grad_B (compute sequentially, not in parallel).

---

## Resources & References

**Implementation Files**:
- Phase 1 results: `/TEMP/ATOMIC_TRAINING_LIMITED_TEST_RESULTS_NOV19.md`
- W3C AIKR proof: `/TEMP/W3C_AIKR_ATOMIC_UNITS_PROOF_NOV19.md`
- Sovereignty path: `/TEMP/ATOMIC_TRAINING_SOVEREIGNTY_PATH_NOV19.md`
- Briefing (updated): `/TEMP/K3D_Briefing_Prompt.md`

**Training Logs** (Nov 19, 2025):
- `/K3D/Knowledge3D.local/logs/atomic_training/20251119_132153/training_log.jsonl`
- `/K3D/Knowledge3D.local/logs/atomic_training/20251119_132153/w3c_aikr_evidence.json`
- `/K3D/Knowledge3D.local/logs/atomic_training/20251119_132153/TRAINING_SUMMARY.md`

**Core Files to Review**:
- `knowledge3d/cranium/trm_adapters.py` (adapter implementation)
- `knowledge3d/cranium/ptx_runtime/modular_rpn_engine.py` (RPN engine)
- `knowledge3d/cranium/specialists/procedural_drawing_specialist.py` (specialist)
- `knowledge3d/cranium/adaptive_swarm.py` (swarm integration)

---

## Codex: Your Mission

**Implement Phase 2 - RPN Sovereignty** to achieve 100% GPU-native atomic training.

**Deliverables**:
1. RPN matrix operations implemented in `ModularRPNEngine`
2. RPN adapter training in `SelfUpdatingAdapter`
3. Ternary validation gate with TRUE/FALSE/UNKNOWN logic
4. Integration into `ProceduralDrawingSpecialist`
5. Tests validating RPN training matches NumPy baseline
6. Performance benchmarks showing improvement

**Freedom to Enhance**:
- Add optimizations you see fit
- Improve RPN operation efficiency
- Propose architectural enhancements
- Document your original contributions

**Constraints**:
- Maintain GPU sovereignty (no CPU fallbacks)
- Preserve Phase 1 results (compositional fusion)
- Keep code readable and well-documented
- Ensure tests pass

**Expected Outcome**: Training pipeline is 100% GPU-native, ~19% faster than Phase 1, ready for W3C AIKR submission.

---

**Status**: Ready for implementation
**Priority**: HIGH (completes W3C AIKR proof)
**Estimated Time**: 2-4 hours implementation + 1 hour testing
**Dependencies**: Phase 1 complete ✅

**Let's achieve full sovereignty! 🚀**

---

**Document Status**: Complete technical specification for Phase 2
**File**: `/TEMP/CODEX_PROMPT_RPN_SOVEREIGNTY_PHASE2_NOV19.md`
**Timestamp**: 2025-11-19T13:45:00Z
