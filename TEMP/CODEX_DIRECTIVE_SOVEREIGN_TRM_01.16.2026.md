# Codex Directive: Implement Sovereign TRM (Remove PyTorch Dependency)

**From**: Claude (Architecture Partner)
**To**: Codex (Implementation Specialist)
**Date**: January 16, 2026
**Subject**: **REPLACE PyTorch V7 with Sovereign TRM Implementation**

---

## Executive Summary

**Objective**: Remove PyTorch dependency from hot path by implementing Sovereign TRM using existing OP_TRM_* PTX opcodes.

**Why**: The current PyTorch V7 implementation violates sovereignty (external ML framework in hot path) and causes CUDA context conflicts with sovereign loader.

**Solution**: Implement `SovereignTRM` class that:
- Uses OP_TRM_MATVEC_*, OP_TRM_SWIGLU_* PTX opcodes for inference
- Loads weights from Galaxy Universe (not PyTorch .pt files)
- Zero PyTorch dependency in inference (hot path sovereign)
- Eliminates CUDA context conflicts (single GPU framework)

**This is NOT a band-aid fix** - this is the correct architectural implementation. The PyTorch V7 was a prototype to validate concepts. Now we implement it correctly using sovereign infrastructure.

---

## Current Problem (PyTorch V7 Sovereignty Violation)

### What V7 Currently Does (WRONG)

**File**: `knowledge3d/training/math_benchmarks/navigation_model_with_confidence.py`

```python
class NavigationModelWithConfidence(pl.LightningModule):  # ❌ PyTorch Lightning
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512):
        super().__init__()
        # PyTorch layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # ❌ PyTorch
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)  # ❌ PyTorch

        self.rule_head = nn.Linear(hidden_dim, vocab_size + 3)  # ❌ PyTorch
        self.confidence_head = nn.Sequential(  # ❌ PyTorch
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, problem_tokens):
        emb = self.embedding(problem_tokens)  # ❌ PyTorch operations
        lstm_out, _ = self.lstm(emb)
        # ... PyTorch tensor operations
```

**Usage in Reflection**:
```python
# solve_with_reflection.py
model = NavigationModelWithConfidence.load_from_checkpoint('checkpoints/v7.pt')  # ❌ PyTorch
model = model.to('cuda')  # ❌ PyTorch CUDA context (conflicts with sovereign loader)

# Inference
rule_logits, confidence = model(problem_tokens)  # ❌ PyTorch forward pass
```

**Why This Is Wrong**:
- ❌ PyTorch is external ML framework (not sovereign)
- ❌ PyTorch CUDA context conflicts with sovereign loader's context
- ❌ Hot path uses PyTorch operations (violates sovereignty)
- ❌ `.pt` checkpoint files (not Galaxy Universe)
- ❌ Multi-framework complexity (PyTorch + PTX)

---

## Correct Solution: Sovereign TRM

### What We Already Have (Infrastructure Ready)

**PTX Opcodes for TRM** (from `modular_rpn_engine.py`):
```python
OP_TRM_MATVEC_1024x512   # Matrix-vector multiply (1024×512)
OP_TRM_MATVEC_512x1024   # Matrix-vector multiply (512×1024)
OP_TRM_SWIGLU_1024       # SwiGLU activation (1024-dim)
OP_TRM_SWIGLU_512        # SwiGLU activation (512-dim)
OP_TRM_VEC_ADD3_512      # 3-way vector addition (512-dim)
OP_SIGMOID_APPROX        # Sigmoid activation (approximate)
```

**These opcodes ALREADY EXIST** - they were designed for sovereign TRM from the start.

**Sovereign Infrastructure**:
- ✅ Modular RPN Engine (PTX execution)
- ✅ Galaxy Universe (VRAM-resident memory)
- ✅ Math Core Pool (GPU allocation)
- ✅ Sovereign Loader (CUDA context management)
- ✅ TRM PTX opcodes (matrix ops, activations)

**What's Missing**: High-level `SovereignTRM` class that orchestrates these opcodes.

---

## Implementation Specification

### Phase 1: Sovereign TRM Class (Core Implementation)

**File**: `knowledge3d/cranium/sovereign_trm.py` (NEW)

**Purpose**: Sovereign TRM inference using PTX opcodes (no PyTorch).

**Architecture**:
```python
from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine
from knowledge3d.cranium.sovereign import loader
from typing import Tuple, Optional
import numpy as np

class SovereignTRM:
    """Sovereign TRM using PTX kernels (zero PyTorch dependency).

    This class implements TRM (Tiny Reasoning Model) inference using only
    sovereign PTX opcodes. All weights reside in Galaxy Universe (VRAM).
    All computation happens via hand-authored PTX kernels.

    Architecture:
        - Embedding layer (vocab_size → embedding_dim)
        - LSTM layer (embedding_dim → hidden_dim)
        - Rule head (hidden_dim → vocab_size + 3)
        - Confidence head (hidden_dim → 1)

    Example:
        trm = SovereignTRM(vocab_size=256, embedding_dim=256, hidden_dim=512)
        trm.load_weights('checkpoints/v7_converted')
        rule_logits, confidence = trm.infer(problem_tokens)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
    ):
        """Initialize Sovereign TRM.

        Args:
            vocab_size: Vocabulary size (number of distinct tokens)
            embedding_dim: Embedding dimension
            hidden_dim: LSTM hidden dimension
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Initialize sovereign RPN engine (PTX execution)
        self.rpn_engine = ModularRPNEngine()

        # Weights stored in Galaxy Universe (VRAM-resident)
        self.weights = {}

        # LSTM state (stored on GPU)
        self.lstm_h = None  # Hidden state (hidden_dim,)
        self.lstm_c = None  # Cell state (hidden_dim,)

    def load_weights(self, checkpoint_dir: str) -> None:
        """Load TRM weights from Galaxy Universe checkpoint.

        Args:
            checkpoint_dir: Directory containing weight arrays (.npy files)

        Expected files:
            embedding.npy          (vocab_size, embedding_dim)
            lstm_weight_ih.npy     (4*hidden_dim, embedding_dim)
            lstm_weight_hh.npy     (4*hidden_dim, hidden_dim)
            lstm_bias_ih.npy       (4*hidden_dim,)
            lstm_bias_hh.npy       (4*hidden_dim,)
            rule_head_weight.npy   (vocab_size+3, hidden_dim)
            rule_head_bias.npy     (vocab_size+3,)
            confidence_head_0_weight.npy  (hidden_dim//2, hidden_dim)
            confidence_head_0_bias.npy    (hidden_dim//2,)
            confidence_head_2_weight.npy  (1, hidden_dim//2)
            confidence_head_2_bias.npy    (1,)
        """
        import os

        # Load weights from .npy files (NumPy format - ingestion path OK)
        weight_files = [
            'embedding',
            'lstm_weight_ih', 'lstm_weight_hh',
            'lstm_bias_ih', 'lstm_bias_hh',
            'rule_head_weight', 'rule_head_bias',
            'confidence_head_0_weight', 'confidence_head_0_bias',
            'confidence_head_2_weight', 'confidence_head_2_bias',
        ]

        for name in weight_files:
            path = os.path.join(checkpoint_dir, f"{name}.npy")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Weight file not found: {path}")

            # Load to CPU (ingestion path - OK to use NumPy)
            weight_array = np.load(path)

            # Upload to GPU (sovereign loader handles allocation)
            self.weights[name] = self._upload_to_gpu(weight_array)

    def _upload_to_gpu(self, array: np.ndarray) -> loader.CUdeviceptr:
        """Upload NumPy array to GPU memory.

        Args:
            array: NumPy array (any shape)

        Returns:
            GPU device pointer
        """
        # Flatten to 1D (RPN kernels work with flat buffers)
        flat = array.astype(np.float32).flatten()
        size_bytes = flat.nbytes

        # Allocate GPU memory (sovereign loader)
        device_ptr = loader.gpu_malloc(size_bytes)

        # Copy data to GPU
        loader.cpu_to_gpu(device_ptr, flat)

        return device_ptr

    def reset_lstm_state(self) -> None:
        """Reset LSTM hidden/cell state to zeros."""
        # Allocate zero-initialized state on GPU
        state_size = self.hidden_dim * 4  # float32

        self.lstm_h = loader.gpu_malloc(state_size)
        self.lstm_c = loader.gpu_malloc(state_size)

        # Zero-fill (sovereign operation)
        zeros = np.zeros(self.hidden_dim, dtype=np.float32)
        loader.cpu_to_gpu(self.lstm_h, zeros)
        loader.cpu_to_gpu(self.lstm_c, zeros)

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
        current_token = 0  # <START> token

        for _ in range(max_rules):
            # LSTM step with current token
            hidden = self._lstm_step(current_token)

            # Rule head (classification)
            rule_logits = self._rule_head(hidden)
            next_rule = self._argmax(rule_logits)

            # Confidence head (regression)
            confidence = self._confidence_head(hidden)

            # Check for <END> token
            if next_rule == self.vocab_size + 2:  # <END> token
                break

            rule_sequence.append(next_rule)
            confidence_scores.append(confidence)

            current_token = next_rule

        return rule_sequence, confidence_scores

    def _lstm_step(self, token_id: int) -> loader.CUdeviceptr:
        """Single LSTM forward step (sovereign PTX).

        Args:
            token_id: Input token ID

        Returns:
            Hidden state (device pointer)
        """
        # 1. Embedding lookup (gather operation)
        embedding_vec = self._embedding_lookup(token_id)

        # 2. LSTM cell computation (using OP_TRM_MATVEC_* opcodes)
        # PyTorch LSTM: i, f, g, o = chunk(W_ih @ x + b_ih + W_hh @ h + b_hh, 4)

        # Input projection: W_ih @ embedding + b_ih
        ih_proj = self._matvec_add_bias(
            self.weights['lstm_weight_ih'],
            embedding_vec,
            self.weights['lstm_bias_ih'],
            rows=4 * self.hidden_dim,
            cols=self.embedding_dim
        )

        # Hidden projection: W_hh @ h + b_hh
        hh_proj = self._matvec_add_bias(
            self.weights['lstm_weight_hh'],
            self.lstm_h,
            self.weights['lstm_bias_hh'],
            rows=4 * self.hidden_dim,
            cols=self.hidden_dim
        )

        # Combined: ih_proj + hh_proj
        gates = self._vector_add(ih_proj, hh_proj, size=4 * self.hidden_dim)

        # Split into 4 gates: i, f, g, o (each hidden_dim)
        # NOTE: This requires implementing a "split" helper or doing 4 separate operations
        i_gate = self._slice_vector(gates, start=0, length=self.hidden_dim)
        f_gate = self._slice_vector(gates, start=self.hidden_dim, length=self.hidden_dim)
        g_gate = self._slice_vector(gates, start=2*self.hidden_dim, length=self.hidden_dim)
        o_gate = self._slice_vector(gates, start=3*self.hidden_dim, length=self.hidden_dim)

        # Apply activations: sigmoid(i), sigmoid(f), tanh(g), sigmoid(o)
        i_gate = self._sigmoid_vector(i_gate, self.hidden_dim)
        f_gate = self._sigmoid_vector(f_gate, self.hidden_dim)
        g_gate = self._tanh_vector(g_gate, self.hidden_dim)
        o_gate = self._sigmoid_vector(o_gate, self.hidden_dim)

        # Cell state update: c = f * c_old + i * g
        c_new = self._lstm_cell_update(f_gate, self.lstm_c, i_gate, g_gate)

        # Hidden state update: h = o * tanh(c)
        h_new = self._elementwise_mul(o_gate, self._tanh_vector(c_new, self.hidden_dim), self.hidden_dim)

        # Update state
        self.lstm_h = h_new
        self.lstm_c = c_new

        return self.lstm_h

    def _rule_head(self, hidden: loader.CUdeviceptr) -> loader.CUdeviceptr:
        """Rule classification head (linear layer).

        Args:
            hidden: Hidden state (hidden_dim,)

        Returns:
            Rule logits (vocab_size + 3,)
        """
        return self._matvec_add_bias(
            self.weights['rule_head_weight'],
            hidden,
            self.weights['rule_head_bias'],
            rows=self.vocab_size + 3,
            cols=self.hidden_dim
        )

    def _confidence_head(self, hidden: loader.CUdeviceptr) -> float:
        """Confidence regression head (2-layer MLP).

        Args:
            hidden: Hidden state (hidden_dim,)

        Returns:
            Confidence score [0, 1]
        """
        # Layer 1: Linear + ReLU
        h1 = self._matvec_add_bias(
            self.weights['confidence_head_0_weight'],
            hidden,
            self.weights['confidence_head_0_bias'],
            rows=self.hidden_dim // 2,
            cols=self.hidden_dim
        )
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
        confidence = loader.gpu_to_cpu_scalar(h2)

        # Sigmoid activation (CPU OK - single scalar)
        return 1.0 / (1.0 + np.exp(-confidence))

    # ------------------------------------------------------------------ #
    # PTX opcode helpers (these orchestrate OP_TRM_* opcodes)
    # ------------------------------------------------------------------ #

    def _embedding_lookup(self, token_id: int) -> loader.CUdeviceptr:
        """Look up embedding vector for token.

        Args:
            token_id: Token ID (integer)

        Returns:
            Embedding vector (embedding_dim,) on GPU
        """
        # This is a gather operation: embedding[token_id, :]
        # For now, use CPU fallback (extract row from GPU matrix)

        # Allocate result buffer
        result = loader.gpu_malloc(self.embedding_dim * 4)  # float32

        # Copy row from embedding matrix (offset = token_id * embedding_dim * 4)
        offset = token_id * self.embedding_dim * 4
        loader.gpu_to_gpu_copy(
            dst=result,
            src=self.weights['embedding'],
            offset=offset,
            size=self.embedding_dim * 4
        )

        return result

    def _matvec_add_bias(
        self,
        weight: loader.CUdeviceptr,
        vec: loader.CUdeviceptr,
        bias: loader.CUdeviceptr,
        rows: int,
        cols: int
    ) -> loader.CUdeviceptr:
        """Matrix-vector multiply with bias: y = W @ x + b.

        Uses OP_TRM_MATVEC_* opcodes.
        """
        # Allocate result buffer
        result = loader.gpu_malloc(rows * 4)  # float32

        # Build RPN program using OP_TRM_MATVEC_*
        # NOTE: This is pseudocode - actual implementation depends on PTX kernel interface

        # For 512x1024 matrix:
        if rows == 512 and cols == 1024:
            opcode = 'OP_TRM_MATVEC_512x1024'
        elif rows == 1024 and cols == 512:
            opcode = 'OP_TRM_MATVEC_1024x512'
        else:
            # Generic case - may need to implement additional opcodes
            raise NotImplementedError(f"Matrix shape {rows}x{cols} not supported")

        # Execute PTX kernel (this is the hot path - all PTX)
        # NOTE: Actual implementation needs to build RPN program and execute via rpn_engine
        # This is a placeholder showing the architecture

        return result

    def _vector_add(
        self,
        a: loader.CUdeviceptr,
        b: loader.CUdeviceptr,
        size: int
    ) -> loader.CUdeviceptr:
        """Element-wise vector addition: c = a + b.

        Uses OP_TRM_VEC_ADD3_512 or similar.
        """
        result = loader.gpu_malloc(size * 4)

        # Use OP_TRM_VEC_ADD3_512 if size matches
        # Otherwise, use generic addition opcode

        return result

    def _sigmoid_vector(self, vec: loader.CUdeviceptr, size: int) -> loader.CUdeviceptr:
        """Apply sigmoid activation element-wise.

        Uses OP_SIGMOID_APPROX opcode.
        """
        result = loader.gpu_malloc(size * 4)

        # Build RPN program: [vec] OP_SIGMOID_APPROX
        # Execute via rpn_engine

        return result

    def _tanh_vector(self, vec: loader.CUdeviceptr, size: int) -> loader.CUdeviceptr:
        """Apply tanh activation element-wise.

        Uses tanh opcode (may need to add if not present).
        """
        result = loader.gpu_malloc(size * 4)

        # Build RPN program: [vec] tanh

        return result

    def _relu_vector(self, vec: loader.CUdeviceptr, size: int) -> loader.CUdeviceptr:
        """Apply ReLU activation element-wise.

        ReLU(x) = max(0, x)
        """
        result = loader.gpu_malloc(size * 4)

        # Build RPN program: [vec] 0 max (element-wise)

        return result

    def _elementwise_mul(
        self,
        a: loader.CUdeviceptr,
        b: loader.CUdeviceptr,
        size: int
    ) -> loader.CUdeviceptr:
        """Element-wise multiplication: c = a * b."""
        result = loader.gpu_malloc(size * 4)

        # Build RPN program: [a] [b] * (element-wise)

        return result

    def _lstm_cell_update(
        self,
        f_gate: loader.CUdeviceptr,
        c_old: loader.CUdeviceptr,
        i_gate: loader.CUdeviceptr,
        g_gate: loader.CUdeviceptr
    ) -> loader.CUdeviceptr:
        """LSTM cell update: c = f * c_old + i * g."""
        # c = f * c_old + i * g (element-wise)

        fc = self._elementwise_mul(f_gate, c_old, self.hidden_dim)
        ig = self._elementwise_mul(i_gate, g_gate, self.hidden_dim)
        c_new = self._vector_add(fc, ig, self.hidden_dim)

        return c_new

    def _argmax(self, logits: loader.CUdeviceptr) -> int:
        """Find index of maximum value (argmax).

        For now, copy to CPU and use NumPy (acceptable for single scalar).
        """
        # Copy logits to CPU
        logits_cpu = loader.gpu_to_cpu_array(logits, self.vocab_size + 3)

        # Argmax on CPU (acceptable - single operation after GPU inference)
        return int(np.argmax(logits_cpu))

    def _slice_vector(
        self,
        vec: loader.CUdeviceptr,
        start: int,
        length: int
    ) -> loader.CUdeviceptr:
        """Extract slice from vector: vec[start:start+length]."""
        result = loader.gpu_malloc(length * 4)

        # Copy slice (GPU-to-GPU copy with offset)
        loader.gpu_to_gpu_copy(
            dst=result,
            src=vec,
            offset=start * 4,
            size=length * 4
        )

        return result

    def cleanup(self) -> None:
        """Free GPU resources."""
        # Free all weight buffers
        for ptr in self.weights.values():
            loader.gpu_free(ptr)

        # Free LSTM state
        if self.lstm_h is not None:
            loader.gpu_free(self.lstm_h)
        if self.lstm_c is not None:
            loader.gpu_free(self.lstm_c)

        # Close RPN engine
        self.rpn_engine.close()
```

**Key Architecture Points**:
1. ✅ **Zero PyTorch dependency** - all inference uses PTX opcodes
2. ✅ **Weights in GPU memory** - uploaded via sovereign loader
3. ✅ **LSTM implemented with PTX opcodes** - matrix ops + activations
4. ✅ **Single CUDA context** - sovereign loader owns context, no PyTorch conflict
5. ✅ **Hot path sovereign** - all computation via hand-authored PTX kernels

---

### Phase 2: Weight Conversion (PyTorch → NumPy)

**File**: `scripts/convert_v7_to_sovereign.py` (NEW)

**Purpose**: Convert PyTorch V7 checkpoint to NumPy arrays for Sovereign TRM.

```python
"""Convert PyTorch V7 checkpoint to Sovereign TRM format.

This script converts a PyTorch Lightning checkpoint (.pt file) to NumPy
arrays that can be loaded by SovereignTRM.

Usage:
    python3 scripts/convert_v7_to_sovereign.py \\
        --input checkpoints/v7.pt \\
        --output checkpoints/v7_converted/
"""
import argparse
import os
import numpy as np
import torch


def convert_checkpoint(input_path: str, output_dir: str) -> None:
    """Convert PyTorch checkpoint to NumPy arrays.

    Args:
        input_path: Path to PyTorch .pt checkpoint
        output_dir: Directory to save .npy weight files
    """
    # Load PyTorch checkpoint
    print(f"Loading PyTorch checkpoint: {input_path}")
    checkpoint = torch.load(input_path, map_location='cpu')

    # Extract state dict
    state_dict = checkpoint.get('state_dict', checkpoint)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Weight mapping (PyTorch layer name → output file name)
    weight_map = {
        'embedding.weight': 'embedding.npy',
        'lstm.weight_ih_l0': 'lstm_weight_ih.npy',
        'lstm.weight_hh_l0': 'lstm_weight_hh.npy',
        'lstm.bias_ih_l0': 'lstm_bias_ih.npy',
        'lstm.bias_hh_l0': 'lstm_bias_hh.npy',
        'rule_head.weight': 'rule_head_weight.npy',
        'rule_head.bias': 'rule_head_bias.npy',
        'confidence_head.0.weight': 'confidence_head_0_weight.npy',
        'confidence_head.0.bias': 'confidence_head_0_bias.npy',
        'confidence_head.2.weight': 'confidence_head_2_weight.npy',
        'confidence_head.2.bias': 'confidence_head_2_bias.npy',
    }

    # Convert each weight
    for pt_name, npy_name in weight_map.items():
        if pt_name not in state_dict:
            print(f"WARNING: {pt_name} not found in checkpoint")
            continue

        # Get PyTorch tensor
        tensor = state_dict[pt_name]

        # Convert to NumPy (CPU)
        array = tensor.detach().cpu().numpy()

        # Save as .npy
        output_path = os.path.join(output_dir, npy_name)
        np.save(output_path, array)

        print(f"  {pt_name:30s} → {npy_name:30s} (shape: {array.shape})")

    print(f"\nConversion complete! Weights saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch V7 to Sovereign TRM")
    parser.add_argument('--input', required=True, help='Input PyTorch checkpoint (.pt)')
    parser.add_argument('--output', required=True, help='Output directory for .npy files')
    args = parser.parse_args()

    convert_checkpoint(args.input, args.output)


if __name__ == '__main__':
    main()
```

**Usage**:
```bash
# Convert existing V7 checkpoint
python3 scripts/convert_v7_to_sovereign.py \
    --input checkpoints/v7.pt \
    --output checkpoints/v7_converted/

# Output:
#   checkpoints/v7_converted/
#       embedding.npy
#       lstm_weight_ih.npy
#       lstm_weight_hh.npy
#       lstm_bias_ih.npy
#       lstm_bias_hh.npy
#       rule_head_weight.npy
#       rule_head_bias.npy
#       confidence_head_0_weight.npy
#       confidence_head_0_bias.npy
#       confidence_head_2_weight.npy
#       confidence_head_2_bias.npy
```

---

### Phase 3: Sovereign Loader Extensions

**File**: `knowledge3d/cranium/sovereign/loader.py`

**Add Missing Helper Functions**:

```python
def cpu_to_gpu(device_ptr: CUdeviceptr, host_array: np.ndarray) -> None:
    """Copy data from CPU (NumPy array) to GPU.

    Args:
        device_ptr: GPU device pointer (destination)
        host_array: NumPy array (source, must be float32)
    """
    _ensure_init()

    # Ensure contiguous + correct dtype
    if not host_array.flags['C_CONTIGUOUS']:
        host_array = np.ascontiguousarray(host_array)
    if host_array.dtype != np.float32:
        host_array = host_array.astype(np.float32)

    # Copy to GPU
    size_bytes = host_array.nbytes
    ck(nvcuda.cuMemcpyHtoD(
        device_ptr,
        host_array.ctypes.data,
        size_bytes
    ))


def gpu_to_cpu_scalar(device_ptr: CUdeviceptr) -> float:
    """Copy single float from GPU to CPU.

    Args:
        device_ptr: GPU device pointer (source)

    Returns:
        Float value
    """
    _ensure_init()

    # Allocate CPU buffer (single float)
    result = np.zeros(1, dtype=np.float32)

    # Copy from GPU
    ck(nvcuda.cuMemcpyDtoH(
        result.ctypes.data,
        device_ptr,
        4  # sizeof(float)
    ))

    return float(result[0])


def gpu_to_cpu_array(device_ptr: CUdeviceptr, count: int) -> np.ndarray:
    """Copy array from GPU to CPU.

    Args:
        device_ptr: GPU device pointer (source)
        count: Number of float32 elements

    Returns:
        NumPy array (float32)
    """
    _ensure_init()

    # Allocate CPU buffer
    result = np.zeros(count, dtype=np.float32)

    # Copy from GPU
    ck(nvcuda.cuMemcpyDtoH(
        result.ctypes.data,
        device_ptr,
        count * 4  # sizeof(float) * count
    ))

    return result


def gpu_to_gpu_copy(
    dst: CUdeviceptr,
    src: CUdeviceptr,
    offset: int,
    size: int
) -> None:
    """Copy data from GPU to GPU (with source offset).

    Args:
        dst: Destination device pointer
        src: Source device pointer
        offset: Byte offset in source
        size: Number of bytes to copy
    """
    _ensure_init()

    # Add offset to source pointer
    src_offset = CUdeviceptr(src.value + offset)

    # Copy GPU-to-GPU
    ck(nvcuda.cuMemcpyDtoD(
        dst,
        src_offset,
        size
    ))


def gpu_free(device_ptr: CUdeviceptr) -> None:
    """Free GPU memory.

    Args:
        device_ptr: Device pointer to free
    """
    _ensure_init()

    if device_ptr.value == 0:
        return  # NULL pointer

    nvcuda.cuMemFree(device_ptr)
```

---

### Phase 4: Update Reflection Script (Use Sovereign TRM)

**File**: `solve_with_reflection.py`

**Replace PyTorch V7 with Sovereign TRM**:

```python
# REMOVE PyTorch imports
# import torch  # ❌ DELETE
# from knowledge3d.training.navigation_model import NavigationModelWithConfidence  # ❌ DELETE

# ADD Sovereign TRM import
from knowledge3d.cranium.sovereign_trm import SovereignTRM  # ✅ NEW

class ReflectiveSolver:
    def __init__(self, checkpoint_path: str, vocab_size: int = 256):
        """Initialize reflective solver with Sovereign TRM.

        Args:
            checkpoint_path: Path to converted checkpoint directory (not .pt file!)
            vocab_size: Vocabulary size
        """
        # Initialize Sovereign TRM (no PyTorch)
        self.trm = SovereignTRM(
            vocab_size=vocab_size,
            embedding_dim=256,
            hidden_dim=512
        )

        # Load weights (NumPy arrays → GPU via sovereign loader)
        self.trm.load_weights(checkpoint_path)

        # No device management needed (sovereign loader handles context)

    def solve_with_reflection(self, problem: str) -> dict:
        """Solve problem using Sovereign TRM + symbolic verification.

        Args:
            problem: Problem string

        Returns:
            Result dict with solution, confidence, verification
        """
        # Tokenize problem (ingestion path - Python OK)
        problem_tokens = self.tokenize_problem(problem)

        # TRM inference (HOT PATH - sovereign PTX only)
        rule_sequence, confidences = self.trm.infer(problem_tokens, max_rules=20)

        # Convert rules to RPN program
        rpn_program = self.rules_to_rpn(rule_sequence)

        # Execute RPN program (HOT PATH - sovereign PTX)
        from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine
        rpn_engine = ModularRPNEngine()

        try:
            result = rpn_engine.evaluate(rpn_program)

            # Symbolic verification (RecursiveSolver)
            verified = self.verify_solution(problem, result)

            return {
                'problem': problem,
                'rule_sequence': rule_sequence,
                'confidences': confidences,
                'rpn_program': rpn_program,
                'result': result,
                'verified': verified,
                'avg_confidence': sum(confidences) / len(confidences) if confidences else 0.0
            }
        except Exception as e:
            return {
                'problem': problem,
                'rule_sequence': rule_sequence,
                'confidences': confidences,
                'error': str(e),
                'verified': False
            }
        finally:
            rpn_engine.close()
```

**Key Changes**:
- ❌ Remove `torch.load()` → ✅ Use `SovereignTRM.load_weights()`
- ❌ Remove `model.to('cuda')` → ✅ Sovereign loader handles GPU
- ❌ Remove PyTorch forward pass → ✅ Use `trm.infer()` (PTX opcodes)
- ✅ Zero PyTorch dependency in hot path
- ✅ No CUDA context conflicts

---

## Implementation Phases

### Phase 1: Core Sovereign TRM (Week 1)

**Tasks**:
1. ✅ Implement `SovereignTRM` class skeleton (`knowledge3d/cranium/sovereign_trm.py`)
2. ✅ Add sovereign loader helpers (`cpu_to_gpu`, `gpu_to_cpu_scalar`, etc.)
3. ✅ Implement weight loading (`load_weights` from .npy files)
4. ✅ Implement embedding lookup
5. ✅ Implement basic PTX opcode helpers (`_matvec_add_bias`, `_vector_add`, etc.)

**Success Criteria**:
- [ ] `SovereignTRM` class exists and can be instantiated
- [ ] Can load converted weights from NumPy arrays
- [ ] Can allocate GPU buffers via sovereign loader
- [ ] Basic PTX operations work (matvec, vector add)

---

### Phase 2: LSTM Implementation (Week 1-2)

**Tasks**:
1. ✅ Implement `_lstm_step` using PTX opcodes
2. ✅ Implement gate activations (sigmoid, tanh)
3. ✅ Implement cell state update
4. ✅ Test LSTM forward pass (compare with PyTorch)

**Success Criteria**:
- [ ] LSTM forward pass produces same results as PyTorch V7 (within tolerance)
- [ ] All operations use PTX opcodes (no PyTorch)
- [ ] LSTM state management works correctly

---

### Phase 3: Rule + Confidence Heads (Week 2)

**Tasks**:
1. ✅ Implement `_rule_head` (linear layer)
2. ✅ Implement `_confidence_head` (2-layer MLP)
3. ✅ Implement `_argmax` for rule selection
4. ✅ Test heads (compare with PyTorch)

**Success Criteria**:
- [ ] Rule head produces same logits as PyTorch V7
- [ ] Confidence head produces same scores as PyTorch V7
- [ ] Argmax correctly selects max rule

---

### Phase 4: Full Inference + Integration (Week 2)

**Tasks**:
1. ✅ Implement `infer` (autoregressive decoding)
2. ✅ Create weight conversion script (`convert_v7_to_sovereign.py`)
3. ✅ Update `solve_with_reflection.py` to use Sovereign TRM
4. ✅ Run benchmarks and validate equivalence

**Success Criteria**:
- [ ] Sovereign TRM produces same results as PyTorch V7 (within tolerance)
- [ ] Benchmarks run without CUDA context errors
- [ ] Zero PyTorch imports in hot path
- [ ] Performance comparable or better than PyTorch V7

---

## Testing Strategy

### Unit Tests

**File**: `tests/test_sovereign_trm.py`

```python
import pytest
import numpy as np
from knowledge3d.cranium.sovereign_trm import SovereignTRM


def test_sovereign_trm_initialization():
    """Test SovereignTRM can be instantiated."""
    trm = SovereignTRM(vocab_size=256, embedding_dim=256, hidden_dim=512)
    assert trm.vocab_size == 256
    assert trm.embedding_dim == 256
    assert trm.hidden_dim == 512
    trm.cleanup()


def test_weight_loading():
    """Test loading weights from NumPy arrays."""
    # Create dummy checkpoint
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy weights
        np.save(os.path.join(tmpdir, 'embedding.npy'), np.random.randn(256, 256).astype(np.float32))
        np.save(os.path.join(tmpdir, 'lstm_weight_ih.npy'), np.random.randn(2048, 256).astype(np.float32))
        # ... (create all required weight files)

        trm = SovereignTRM(vocab_size=256)
        trm.load_weights(tmpdir)

        assert 'embedding' in trm.weights
        assert 'lstm_weight_ih' in trm.weights

        trm.cleanup()


def test_inference_equivalence():
    """Test Sovereign TRM produces same results as PyTorch V7."""
    # Load both models
    import torch
    from knowledge3d.training.navigation_model import NavigationModelWithConfidence

    # PyTorch V7
    pt_model = NavigationModelWithConfidence.load_from_checkpoint(
        'checkpoints/v7.pt',
        map_location='cpu'
    )
    pt_model.eval()

    # Sovereign TRM
    trm = SovereignTRM(vocab_size=256)
    trm.load_weights('checkpoints/v7_converted')

    # Test problem
    problem_tokens = [1, 42, 15, 3]

    # PyTorch inference
    with torch.no_grad():
        pt_input = torch.tensor([problem_tokens])
        pt_rules, pt_conf = pt_model(pt_input)
        pt_rules_cpu = pt_rules.argmax(dim=-1).cpu().numpy()
        pt_conf_cpu = pt_conf.cpu().numpy()

    # Sovereign inference
    sov_rules, sov_conf = trm.infer(problem_tokens)

    # Compare (within tolerance)
    assert len(sov_rules) == len(pt_rules_cpu[0])
    for i in range(len(sov_rules)):
        assert sov_rules[i] == pt_rules_cpu[0][i]
        assert abs(sov_conf[i] - pt_conf_cpu[0][i]) < 1e-3

    trm.cleanup()
```

---

### Integration Tests

**File**: `tests/test_reflection_sovereign.py`

```python
def test_reflection_pipeline_sovereign():
    """Test full reflection pipeline with Sovereign TRM."""
    from solve_with_reflection import ReflectiveSolver

    solver = ReflectiveSolver(checkpoint_path='checkpoints/v7_converted')

    # Test problem
    problem = "What is the derivative of x^2?"

    # Solve
    result = solver.solve_with_reflection(problem)

    # Validate structure
    assert 'problem' in result
    assert 'rule_sequence' in result
    assert 'confidences' in result
    assert 'rpn_program' in result

    # Validate no PyTorch dependency
    import sys
    assert 'torch' not in sys.modules  # PyTorch should not be imported
```

---

## Success Criteria (Final)

**Sovereign TRM Implementation Complete When**:
- [ ] `SovereignTRM` class fully implemented
- [ ] All inference uses PTX opcodes (no PyTorch)
- [ ] Weights loaded from Galaxy Universe (NumPy arrays → GPU)
- [ ] LSTM, rule head, confidence head implemented with PTX
- [ ] Weight conversion script works (`convert_v7_to_sovereign.py`)
- [ ] `solve_with_reflection.py` updated to use Sovereign TRM
- [ ] Benchmarks run without CUDA context errors
- [ ] Results match PyTorch V7 (within 1e-3 tolerance)
- [ ] Zero PyTorch imports in hot path (`knowledge3d/cranium/`)
- [ ] Unit tests pass (test_sovereign_trm.py)
- [ ] Integration tests pass (test_reflection_sovereign.py)

---

## Migration Plan

### Step 1: Verify Current PyTorch V7 Baseline
```bash
# Run PyTorch V7 benchmarks (for comparison)
python3 scripts/run_sovereign_math_benchmarks.py \
    --datasets calculus \
    --max-problems 10 \
    --use-reflection

# Save results for comparison
```

### Step 2: Implement Sovereign TRM Core
```bash
# Create sovereign_trm.py
touch knowledge3d/cranium/sovereign_trm.py

# Implement core class (skeleton)
# ... (Codex implements)
```

### Step 3: Convert V7 Weights
```bash
# Convert PyTorch checkpoint to NumPy
python3 scripts/convert_v7_to_sovereign.py \
    --input checkpoints/v7.pt \
    --output checkpoints/v7_converted/

# Verify conversion
ls checkpoints/v7_converted/*.npy
```

### Step 4: Implement LSTM + Heads
```bash
# Implement _lstm_step, _rule_head, _confidence_head
# ... (Codex implements)

# Unit test
pytest tests/test_sovereign_trm.py -v
```

### Step 5: Update Reflection Script
```bash
# Update solve_with_reflection.py to use SovereignTRM
# ... (Codex implements)

# Integration test
pytest tests/test_reflection_sovereign.py -v
```

### Step 6: Run Sovereign Benchmarks
```bash
# Run benchmarks with Sovereign TRM (no PyTorch)
python3 scripts/run_sovereign_math_benchmarks.py \
    --datasets calculus \
    --max-problems 10 \
    --use-reflection

# Compare results with PyTorch V7 baseline
```

### Step 7: Validate Equivalence
```bash
# Run equivalence tests (Sovereign vs PyTorch)
pytest tests/test_inference_equivalence.py -v

# Expected: All tests pass (within 1e-3 tolerance)
```

### Step 8: Remove PyTorch Dependency
```bash
# Remove PyTorch imports from hot path
# knowledge3d/cranium/ should have zero PyTorch dependency

# Verify no PyTorch in hot path
grep -r "import torch" knowledge3d/cranium/
# Expected: No matches
```

---

## Why This Is The Correct Fix

### Current Problem (PyTorch V7)
- ❌ PyTorch in hot path (sovereignty violation)
- ❌ CUDA context conflicts (PyTorch vs sovereign loader)
- ❌ Multi-framework complexity (PyTorch + PTX)
- ❌ External dependency (PyTorch Lightning)
- ❌ Band-aid fixes needed (primary context sharing)

### Sovereign TRM Solution
- ✅ Zero PyTorch dependency (hot path sovereign)
- ✅ Single CUDA context (sovereign loader owns it)
- ✅ All inference via PTX opcodes (hand-authored kernels)
- ✅ Weights in Galaxy Universe (VRAM-resident)
- ✅ No CUDA context conflicts (no multi-framework issues)

**The OP_TRM_* opcodes were designed for this** - sovereign TRM was always the intended architecture. PyTorch V7 was a prototype to validate concepts. Now we implement it correctly.

---

## Notes for Codex

**This is NOT a quick fix** - this is implementing the correct architecture. Budget 1-2 weeks for full implementation.

**PyTorch V7 served its purpose** - it validated confidence scoring, calibration loss, and verification loop. But it was never meant to be production code.

**The infrastructure is ready**:
- ✅ Sovereign loader (CUDA context management)
- ✅ PTX opcodes (OP_TRM_MATVEC_*, OP_TRM_SWIGLU_*)
- ✅ Math core pool (GPU allocation)
- ✅ Galaxy Universe (memory architecture)

**What's missing**: High-level `SovereignTRM` class that orchestrates PTX opcodes.

**Start with Phase 1** (core implementation), then incrementally add LSTM, heads, and inference. Test equivalence with PyTorch V7 at each phase.

**When complete**: Zero PyTorch dependency in hot path, no CUDA context conflicts, full sovereignty restored.

---

**Document Date**: January 16, 2026
**Status**: 🚀 **READY FOR IMPLEMENTATION**
**Estimated Time**: 1-2 weeks (phased implementation)
**Priority**: **CRITICAL** - Restores sovereignty, eliminates CUDA conflicts

---

**Claude's Directive**: Implement Sovereign TRM using existing OP_TRM_* PTX opcodes. Remove PyTorch from hot path. This is the correct architectural fix, not a workaround. The infrastructure is ready - we just need to wire it together. Start with Phase 1 (core class), test incrementally, validate equivalence with PyTorch V7. When complete, sovereignty restored. 🚀
