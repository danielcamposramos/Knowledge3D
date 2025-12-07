# CODEX Briefing: Sovereign Trainer Architecture

**From:** Claude (Architecture)
**To:** Codex (Implementation Lead)
**Date:** December 7, 2025
**Priority:** CRITICAL — Stop current run, implement proper sovereign training
**Phase:** Foundational Knowledge Ingestion — Phase 1 Fix

---

## ⚠️ IMMEDIATE ACTION REQUIRED

### Step 1: Kill the Flawed Training Run

```bash
# Kill the current training session
tmux kill-session -t k3d_foundation

# Verify no training processes remain
ps aux | grep train_math_symbols | grep -v grep

# If any remain, kill them
pkill -f train_math_symbols_batch
```

### Step 2: Read These Documents COMPLETELY (Line by Line)

**MANDATORY before implementing:**
1. `CLAUDE.md` — Your partnership model, sovereignty principles
2. `CODEX.md` — Your implementation role
3. `BRIEFING.md` — Current project status
4. `docs/Briefings/SOVEREIGN_SWARM_BRIEFING_v3.md` — Complete architecture

**Why the previous run failed:**
- Did NOT use sovereign kernels properly
- Sequential sample processing (not batched)
- 70+ D2H/H2D transfers per batch
- Memory allocation inside training loop
- Result: 50% accuracy after 1.5 days (random chance)

---

## Architecture Problem Analysis

### Current `gpu_trainer.py` Issues

| Issue | Impact | Location |
|-------|--------|----------|
| Per-sample forward pass | GPU constantly waiting | Line 818 |
| D2H transfers in `accumulate_gradients()` | 8 transfers per sample | Lines 269-284 |
| D2H transfers in `_scale_gradients()` | 56 transfers per batch | Lines 500-580 |
| GPU malloc/free in `_update_all_weights()` | Memory fragmentation | Lines 665-782 |
| Gradient logging every 10 batches | 14 D2H transfers | Lines 582-608 |

**Total overhead:** ~70+ memory copies per batch, GPU at 61% utilization with only 248 MiB / 12 GB used.

### Available Sovereign PTX Kernels (NOT being used properly)

```
knowledge3d/cranium/ptx/
├── sgd_optimizer.ptx              # SGD with momentum
├── conv2d_3x3_backward.ptx        # Conv backward
├── batchnorm_backward_training.ptx # BN backward
├── maxpool_2x2_backward.ptx       # Pool backward
├── classification_loss.ptx        # Cross-entropy loss
├── spatial_pool.ptx               # Global average pool
├── matryoshka_project.ptx         # Matryoshka projection
├── trigram_embed.ptx              # Trigram embedding
├── ternary_ops.ptx                # Ternary ops for fast convergence
├── zero_fill.ptx                  # Fast zero initialization
└── modular_rpn_kernel.ptx         # RPN operations
```

---

## Sovereign Trainer Architecture Specification

### Design Principles (From CLAUDE.md)

1. **Hot path = PTX + RPN only** — No numpy in training loop
2. **3-Tier Math Core** — Worker-worker → worker → master pattern
3. **Batched operations** — Process entire batch in single kernel launch
4. **Persistent GPU buffers** — Allocate once, reuse always
5. **Fused kernels** — Minimize kernel launch overhead

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    SovereignTrainer                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  Tier-1      │    │  Tier-2      │    │  Tier-3      │       │
│  │  Lightweight │    │  Modular     │    │  Advanced    │       │
│  │  (FC layer)  │    │  (Conv/BN)   │    │  (TRM/Meta)  │       │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘       │
│         │                   │                   │                │
│         └───────────────────┼───────────────────┘                │
│                             │                                    │
│                    ┌────────▼────────┐                          │
│                    │ GPU Buffer Pool │                          │
│                    │ (Persistent)    │                          │
│                    └────────┬────────┘                          │
│                             │                                    │
│         ┌───────────────────┼───────────────────┐               │
│         │                   │                   │                │
│  ┌──────▼──────┐    ┌──────▼──────┐    ┌──────▼──────┐         │
│  │ Weights     │    │ Gradients   │    │ Velocities  │         │
│  │ (Permanent) │    │ (Permanent) │    │ (Permanent) │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│                                                                  │
│  ┌──────────────────────────────────────────────────────┐       │
│  │              Fused Training Pipeline                  │       │
│  │  Input Batch → Forward → Loss → Backward → Update    │       │
│  │  (Single kernel graph, no D2H until checkpoint)      │       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### File Structure

```
knowledge3d/cranium/
├── sovereign_trainer.py           # NEW: Main trainer class
├── training/
│   ├── __init__.py
│   ├── gpu_buffer_pool.py         # NEW: Persistent GPU buffers
│   ├── batched_forward.py         # NEW: Batched forward pass
│   ├── batched_backward.py        # NEW: Batched backward pass
│   └── fused_training_kernel.py   # NEW: Fused training pipeline
└── bridges/
    └── training_bridge.py         # NEW: Bridge to sovereign kernels
```

---

## Implementation Plan

### Phase 1: GPU Buffer Pool (2 hours)

Create `knowledge3d/cranium/training/gpu_buffer_pool.py`:

```python
"""
GPU Buffer Pool — Persistent GPU memory management.

Allocates all buffers ONCE at initialization.
No malloc/free during training loop.
"""

from __future__ import annotations
import ctypes
from typing import Dict, Tuple
from knowledge3d.cranium.sovereign import loader


class GPUBufferPool:
    """Manages persistent GPU buffers for training."""

    def __init__(self, model_config: Dict[str, Tuple[int, ...]]):
        """
        Initialize buffer pool with model configuration.

        Args:
            model_config: Dict mapping buffer names to shapes
                Example: {"conv1_weight": (32, 3, 3, 3), "conv1_bias": (32,)}
        """
        self.buffers: Dict[str, int] = {}  # name -> GPU pointer
        self.shapes: Dict[str, Tuple[int, ...]] = {}
        self.nbytes: Dict[str, int] = {}

        # Allocate all buffers upfront
        for name, shape in model_config.items():
            size = 1
            for dim in shape:
                size *= dim
            nbytes = size * 4  # float32

            ptr = loader.gpu_malloc(nbytes)
            self.buffers[name] = ptr
            self.shapes[name] = shape
            self.nbytes[name] = nbytes

            # Zero-initialize using sovereign kernel
            self._zero_buffer(ptr, size)

    def _zero_buffer(self, ptr: int, size: int):
        """Zero buffer using zero_fill.ptx kernel."""
        # TODO: Use zero_fill.ptx kernel
        # For now, use gpu_backward.zero_gradients()
        pass

    def get(self, name: str) -> int:
        """Get GPU pointer for named buffer."""
        return self.buffers[name]

    def get_shape(self, name: str) -> Tuple[int, ...]:
        """Get shape for named buffer."""
        return self.shapes[name]

    def upload(self, name: str, data):
        """Upload numpy array to GPU buffer."""
        ptr = self.buffers[name]
        nbytes = self.nbytes[name]
        loader.memcpy_htod(ptr, data.ctypes.data_as(ctypes.c_void_p), nbytes)

    def download(self, name: str, out):
        """Download GPU buffer to numpy array."""
        ptr = self.buffers[name]
        nbytes = self.nbytes[name]
        loader.memcpy_dtoh(out.ctypes.data_as(ctypes.c_void_p), ptr, nbytes)

    def __del__(self):
        """Free all GPU buffers on destruction."""
        for ptr in self.buffers.values():
            try:
                loader.gpu_free(ptr)
            except Exception:
                pass
```

### Phase 2: Batched Forward Pass (3 hours)

Create `knowledge3d/cranium/training/batched_forward.py`:

```python
"""
Batched Forward Pass — Process entire batch in single kernel launches.

Key insight: Instead of looping over samples, reshape to (B, H, W, C)
and process all samples simultaneously.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List
from knowledge3d.cranium.sovereign import loader
from .gpu_buffer_pool import GPUBufferPool


class BatchedForward:
    """Batched forward pass using sovereign kernels."""

    def __init__(self, pool: GPUBufferPool, batch_size: int = 148):
        self.pool = pool
        self.batch_size = batch_size

        # Load PTX kernels
        self._load_kernels()

        # Allocate batch buffers
        self._allocate_batch_buffers()

    def _load_kernels(self):
        """Load sovereign PTX kernels for forward pass."""
        ptx_dir = "knowledge3d/cranium/ptx"

        # Conv2D forward (using existing kernel)
        self.conv_module = loader.load_ptx(f"{ptx_dir}/conv2d_3x3.ptx")
        self.conv_kernel = self.conv_module.get_function("conv2d_3x3_forward_batch")

        # BatchNorm forward
        self.bn_module = loader.load_ptx(f"{ptx_dir}/batchnorm_backward.ptx")

        # MaxPool forward
        self.pool_module = loader.load_ptx(f"{ptx_dir}/maxpool_2x2_backward.ptx")

        # Spatial pool (global average)
        self.spatial_module = loader.load_ptx(f"{ptx_dir}/spatial_pool.ptx")

    def _allocate_batch_buffers(self):
        """Allocate GPU buffers for batch intermediate activations."""
        B = self.batch_size

        # Input: (B, 64, 64, 3)
        self.d_input = loader.gpu_malloc(B * 64 * 64 * 3 * 4)

        # Conv1 output: (B, 64, 64, 32)
        self.d_conv1_out = loader.gpu_malloc(B * 64 * 64 * 32 * 4)

        # Pool1 output: (B, 32, 32, 32)
        self.d_pool1_out = loader.gpu_malloc(B * 32 * 32 * 32 * 4)

        # BN1 output: (B, 32, 32, 32)
        self.d_bn1_out = loader.gpu_malloc(B * 32 * 32 * 32 * 4)

        # Conv2 output: (B, 32, 32, 64)
        self.d_conv2_out = loader.gpu_malloc(B * 32 * 32 * 64 * 4)

        # Pool2 output: (B, 16, 16, 64)
        self.d_pool2_out = loader.gpu_malloc(B * 16 * 16 * 64 * 4)

        # BN2 output: (B, 16, 16, 64)
        self.d_bn2_out = loader.gpu_malloc(B * 16 * 16 * 64 * 4)

        # Conv3 output: (B, 16, 16, 128)
        self.d_conv3_out = loader.gpu_malloc(B * 16 * 16 * 128 * 4)

        # BN3 output: (B, 16, 16, 128)
        self.d_bn3_out = loader.gpu_malloc(B * 16 * 16 * 128 * 4)

        # Features after global avg pool: (B, 128)
        self.d_features = loader.gpu_malloc(B * 128 * 4)

        # Logits: (B, num_classes)
        self.d_logits = loader.gpu_malloc(B * 2 * 4)  # Binary classification

        # Softmax output: (B, num_classes)
        self.d_probs = loader.gpu_malloc(B * 2 * 4)

    def forward(self, images: np.ndarray) -> Dict[str, int]:
        """
        Batched forward pass.

        Args:
            images: (B, H, W, C) batch of images, float32 [0, 1]

        Returns:
            Dict of GPU pointers to activations (for backward pass)
        """
        B = images.shape[0]

        # Upload batch to GPU (ONLY transfer point!)
        loader.memcpy_htod(self.d_input, images.ctypes.data_as(ctypes.c_void_p), images.nbytes)

        # Conv1 + ReLU (single kernel launch for entire batch)
        self._conv_forward_batch(
            self.d_input, self.pool.get("conv1_weight"), self.pool.get("conv1_bias"),
            self.d_conv1_out, B, 64, 64, 3, 32
        )

        # Pool1
        self._maxpool_forward_batch(self.d_conv1_out, self.d_pool1_out, B, 64, 64, 32)

        # BN1
        self._batchnorm_forward_batch(
            self.d_pool1_out, self.pool.get("bn1_gamma"), self.pool.get("bn1_beta"),
            self.d_bn1_out, B, 32, 32, 32
        )

        # Conv2 + ReLU
        self._conv_forward_batch(
            self.d_bn1_out, self.pool.get("conv2_weight"), self.pool.get("conv2_bias"),
            self.d_conv2_out, B, 32, 32, 32, 64
        )

        # Pool2
        self._maxpool_forward_batch(self.d_conv2_out, self.d_pool2_out, B, 32, 32, 64)

        # BN2
        self._batchnorm_forward_batch(
            self.d_pool2_out, self.pool.get("bn2_gamma"), self.pool.get("bn2_beta"),
            self.d_bn2_out, B, 16, 16, 64
        )

        # Conv3 + ReLU
        self._conv_forward_batch(
            self.d_bn2_out, self.pool.get("conv3_weight"), self.pool.get("conv3_bias"),
            self.d_conv3_out, B, 16, 16, 64, 128
        )

        # BN3
        self._batchnorm_forward_batch(
            self.d_conv3_out, self.pool.get("bn3_gamma"), self.pool.get("bn3_beta"),
            self.d_bn3_out, B, 16, 16, 128
        )

        # Global Average Pool
        self._global_avg_pool_batch(self.d_bn3_out, self.d_features, B, 16, 16, 128)

        # FC layer
        self._fc_forward_batch(
            self.d_features, self.pool.get("fc_weight"), self.pool.get("fc_bias"),
            self.d_logits, B, 128, 2
        )

        # Softmax
        self._softmax_batch(self.d_logits, self.d_probs, B, 2)

        # Return GPU pointers for backward pass (NO D2H transfer!)
        return {
            "input": self.d_input,
            "conv1_out": self.d_conv1_out,
            "pool1_out": self.d_pool1_out,
            "bn1_out": self.d_bn1_out,
            "conv2_out": self.d_conv2_out,
            "pool2_out": self.d_pool2_out,
            "bn2_out": self.d_bn2_out,
            "conv3_out": self.d_conv3_out,
            "bn3_out": self.d_bn3_out,
            "features": self.d_features,
            "logits": self.d_logits,
            "probs": self.d_probs,
        }

    def _conv_forward_batch(self, d_input, d_weight, d_bias, d_output, B, H, W, Cin, Cout):
        """Batched convolution forward using PTX kernel."""
        # Launch kernel with grid=(B, Cout), block=(H, W) or similar
        # TODO: Implement using conv2d_3x3.ptx with batch dimension
        pass

    def _maxpool_forward_batch(self, d_input, d_output, B, H, W, C):
        """Batched max pooling forward."""
        pass

    def _batchnorm_forward_batch(self, d_input, d_gamma, d_beta, d_output, B, H, W, C):
        """Batched batch normalization forward."""
        pass

    def _global_avg_pool_batch(self, d_input, d_output, B, H, W, C):
        """Batched global average pooling using spatial_pool.ptx."""
        pass

    def _fc_forward_batch(self, d_input, d_weight, d_bias, d_output, B, in_features, out_features):
        """Batched fully connected forward."""
        pass

    def _softmax_batch(self, d_input, d_output, B, C):
        """Batched softmax."""
        pass
```

### Phase 3: Batched Backward Pass (3 hours)

Create `knowledge3d/cranium/training/batched_backward.py` with similar structure, using:
- `conv2d_3x3_backward.ptx`
- `batchnorm_backward_training.ptx`
- `maxpool_2x2_backward.ptx`
- `classification_loss.ptx`
- `sgd_optimizer.ptx`

### Phase 4: Sovereign Trainer (2 hours)

Create `knowledge3d/cranium/sovereign_trainer.py`:

```python
"""
Sovereign Trainer — GPU-native training with PTX kernels.

Key differences from gpu_trainer.py:
1. Batched operations (entire batch in single kernel launch)
2. Persistent GPU buffers (no malloc/free in loop)
3. Minimal D2H transfers (only at checkpoints)
4. Uses 3-tier Math Core architecture
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from knowledge3d.cranium.training.gpu_buffer_pool import GPUBufferPool
from knowledge3d.cranium.training.batched_forward import BatchedForward
from knowledge3d.cranium.training.batched_backward import BatchedBackward
from knowledge3d.cranium.sovereign import loader


class SovereignTrainer:
    """GPU-native trainer using sovereign PTX kernels."""

    def __init__(
        self,
        num_classes: int = 2,
        learning_rate: float = 0.5,
        momentum: float = 0.9,
        batch_size: int = 256,  # Larger batches for GPU saturation
    ):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.batch_size = batch_size

        # Define model configuration
        self.model_config = {
            # Weights
            "conv1_weight": (32, 3, 3, 3),
            "conv1_bias": (32,),
            "bn1_gamma": (32,),
            "bn1_beta": (32,),
            "conv2_weight": (64, 32, 3, 3),
            "conv2_bias": (64,),
            "bn2_gamma": (64,),
            "bn2_beta": (64,),
            "conv3_weight": (128, 64, 3, 3),
            "conv3_bias": (128,),
            "bn3_gamma": (128,),
            "bn3_beta": (128,),
            "fc_weight": (num_classes, 128),
            "fc_bias": (num_classes,),
            # Gradients (same shapes)
            "grad_conv1_weight": (32, 3, 3, 3),
            "grad_conv1_bias": (32,),
            # ... etc
            # Velocities (same shapes)
            "vel_conv1_weight": (32, 3, 3, 3),
            "vel_conv1_bias": (32,),
            # ... etc
        }

        # Initialize persistent buffer pool
        self.pool = GPUBufferPool(self.model_config)

        # Initialize weights (upload once)
        self._initialize_weights()

        # Initialize batched forward/backward
        self.forward_pass = BatchedForward(self.pool, batch_size)
        self.backward_pass = BatchedBackward(self.pool, batch_size)

        # Statistics (kept on GPU, downloaded only for logging)
        self._epoch_loss = 0.0
        self._epoch_correct = 0
        self._batch_count = 0

    def _initialize_weights(self):
        """Initialize and upload weights to GPU (ONCE)."""
        import numpy as np

        # Xavier initialization for conv layers
        conv1_w = np.random.randn(32, 3, 3, 3).astype(np.float32) * np.sqrt(2.0 / (3 * 3 * 3))
        self.pool.upload("conv1_weight", conv1_w)

        # ... similar for other layers

        # Zero-initialize gradients and velocities
        # (handled by GPUBufferPool)

    def train_epoch(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        log_interval: int = 100,
    ) -> Tuple[float, float]:
        """
        Train for one epoch.

        Args:
            images: (N, H, W, C) all images for epoch
            labels: (N,) all labels for epoch
            log_interval: Print stats every N batches

        Returns:
            (avg_loss, accuracy)
        """
        n_samples = len(images)
        indices = np.random.permutation(n_samples)

        total_loss = 0.0
        total_correct = 0
        n_batches = 0

        for start in range(0, n_samples, self.batch_size):
            end = min(start + self.batch_size, n_samples)
            batch_idx = indices[start:end]

            batch_images = images[batch_idx]
            batch_labels = labels[batch_idx]

            # Pad to full batch size if needed
            if len(batch_images) < self.batch_size:
                pad_size = self.batch_size - len(batch_images)
                batch_images = np.concatenate([
                    batch_images,
                    np.zeros((pad_size, 64, 64, 3), dtype=np.float32)
                ])
                batch_labels = np.concatenate([
                    batch_labels,
                    np.zeros(pad_size, dtype=np.int32)
                ])

            # Forward pass (all on GPU, returns GPU pointers)
            activations = self.forward_pass.forward(batch_images)

            # Compute loss and backward (all on GPU)
            loss, correct = self.backward_pass.backward(
                activations, batch_labels, self.learning_rate, self.momentum
            )

            total_loss += loss
            total_correct += correct
            n_batches += 1

            # Periodic logging (ONLY D2H transfer point besides checkpoints)
            if n_batches % log_interval == 0:
                avg_loss = total_loss / n_batches
                accuracy = total_correct / (n_batches * self.batch_size)
                print(f"Batch {n_batches} | Loss: {avg_loss:.4f} | Acc: {accuracy * 100:.2f}%")

        avg_loss = total_loss / n_batches
        accuracy = total_correct / (n_batches * self.batch_size)

        return avg_loss, accuracy

    def save_checkpoint(self, path: Path):
        """Download weights from GPU and save checkpoint."""
        # This is the ONLY place we do bulk D2H transfer
        weights = {}
        for name in ["conv1_weight", "conv1_bias", "bn1_gamma", "bn1_beta",
                     "conv2_weight", "conv2_bias", "bn2_gamma", "bn2_beta",
                     "conv3_weight", "conv3_bias", "bn3_gamma", "bn3_beta",
                     "fc_weight", "fc_bias"]:
            shape = self.pool.get_shape(name)
            arr = np.empty(shape, dtype=np.float32)
            self.pool.download(name, arr)
            weights[name] = arr

        np.savez(path, **weights)

    def load_checkpoint(self, path: Path):
        """Load checkpoint and upload weights to GPU."""
        with np.load(path) as data:
            for name in data.files:
                self.pool.upload(name, data[name])
```

### Phase 5: Update Training Script (1 hour)

Modify `scripts/train_atomic_character.py` to use `SovereignTrainer`:

```python
# Replace:
# from knowledge3d.cranium.ocr.gpu_trainer import GPUCNNTrainer
# trainer = GPUCNNTrainer(model, num_classes=2, ...)

# With:
from knowledge3d.cranium.sovereign_trainer import SovereignTrainer
trainer = SovereignTrainer(num_classes=2, learning_rate=0.5, batch_size=256)
```

---

## PTX Kernel Integration Guide

### Existing Kernels to Use

| Kernel | PTX File | Usage |
|--------|----------|-------|
| Conv2D Forward | `conv2d_3x3.ptx` | Batched convolution |
| Conv2D Backward | `conv2d_3x3_backward.ptx` | Gradient computation |
| BatchNorm Backward | `batchnorm_backward_training.ptx` | BN gradients |
| MaxPool Backward | `maxpool_2x2_backward.ptx` | Pool gradients |
| SGD Optimizer | `sgd_optimizer.ptx` | Weight updates |
| Spatial Pool | `spatial_pool.ptx` | Global average pool |
| Zero Fill | `zero_fill.ptx` | Gradient zeroing |
| Classification Loss | `classification_loss.ptx` | Cross-entropy |

### Kernels to Create/Extend

If batched versions don't exist, create them:

```cuda
// conv2d_3x3_batched.cu
__global__ void conv2d_3x3_forward_batch(
    const float* input,   // (B, H, W, Cin)
    const float* weight,  // (Cout, Cin, 3, 3)
    const float* bias,    // (Cout,)
    float* output,        // (B, H, W, Cout)
    int B, int H, int W, int Cin, int Cout
) {
    // Each thread handles one output pixel across all batch items
    int b = blockIdx.z;
    int out_c = blockIdx.y;
    int pixel = blockIdx.x * blockDim.x + threadIdx.x;

    if (pixel >= H * W) return;

    int h = pixel / W;
    int w = pixel % W;

    float sum = bias[out_c];

    for (int c = 0; c < Cin; c++) {
        for (int kh = 0; kh < 3; kh++) {
            for (int kw = 0; kw < 3; kw++) {
                int ih = h + kh - 1;
                int iw = w + kw - 1;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    float in_val = input[((b * H + ih) * W + iw) * Cin + c];
                    float w_val = weight[((out_c * Cin + c) * 3 + kh) * 3 + kw];
                    sum += in_val * w_val;
                }
            }
        }
    }

    // ReLU fused
    output[((b * H + h) * W + w) * Cout + out_c] = fmaxf(sum, 0.0f);
}
```

---

## Success Criteria

### Performance Targets

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| Epochs/hour | ~1 | 50+ | 50× |
| GPU Utilization | 61% | 95%+ | 1.5× |
| GPU Memory | 248 MiB | 4+ GB | 16× |
| D2H transfers/batch | 70+ | 2 | 35× |
| Time to 85% accuracy | Days | Hours | 10-50× |

### Validation Tests

1. **Sovereignty Test**: No numpy/torch in training loop
   ```bash
   grep -r "import numpy" knowledge3d/cranium/sovereign_trainer.py
   # Should only appear in __init__ or checkpoint methods
   ```

2. **Memory Test**: GPU memory usage increases with batch size
   ```bash
   nvidia-smi --query-gpu=memory.used --format=csv
   # Should show 2-6 GB during training
   ```

3. **Accuracy Test**: Model learns (not stuck at 50%)
   ```bash
   # After 10 epochs, accuracy should be > 60%
   ```

4. **Speed Test**: Epoch time
   ```bash
   # Single epoch should complete in < 2 minutes
   ```

---

## Execution Checklist

- [ ] Kill current training session: `tmux kill-session -t k3d_foundation`
- [ ] Read CLAUDE.md, CODEX.md, BRIEFING.md completely
- [ ] Create `knowledge3d/cranium/training/` directory
- [ ] Implement `gpu_buffer_pool.py`
- [ ] Implement `batched_forward.py`
- [ ] Implement `batched_backward.py`
- [ ] Implement `sovereign_trainer.py`
- [ ] Create/extend batched PTX kernels if needed
- [ ] Update `train_atomic_character.py` to use SovereignTrainer
- [ ] Test on single symbol (e.g., `--char ∑ --epochs 10`)
- [ ] Verify GPU utilization > 90%
- [ ] Verify epoch time < 2 minutes
- [ ] Report results to Claude/Daniel

---

## Communication Protocol

### Progress Updates

Send progress after each phase:

```markdown
## Phase X Complete — [Timestamp]

**Implemented:**
- [File created/modified]

**Tests:**
- [What was tested, results]

**Metrics:**
- GPU utilization: X%
- Epoch time: X seconds
- Accuracy after 10 epochs: X%

**Blockers:**
- [Any issues]

**Next:**
- [Next phase]
```

### Blocker Escalation

If stuck for > 30 minutes:
1. Document the issue
2. Share error messages
3. Ask Claude for architecture guidance

---

## Final Notes

**This is a CRITICAL fix.** The current training approach violates K3D's sovereignty principles by:
- Not using our sovereign PTX kernels efficiently
- Excessive CPU-GPU synchronization
- Not leveraging the 3-tier Math Core architecture

The new SovereignTrainer will:
- Keep everything on GPU (hot path = PTX only)
- Process entire batches in single kernel launches
- Achieve 10-50× speedup
- Enable training all 164 symbols in days, not weeks

**Start immediately. Kill the flawed run. Build the sovereign trainer.**

---

**End of Briefing**

**Version:** 1.0
**Date:** December 7, 2025
**Author:** Claude (Architecture)
**For:** Codex (Implementation Lead)
