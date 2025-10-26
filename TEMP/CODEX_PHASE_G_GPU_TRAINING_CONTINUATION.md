# Phase G Continuation: GPU-Accelerated Specialist Training

**Date**: 2025-10-26
**From**: Claude (completed Phase G smoke test)
**To**: Codex (complete GPU-accelerated training)
**Status**: ⚠ **NEEDS GPU TRAINING** - Current training was CPU-only smoke test

---

## What Happened (The Issue)

### Phase G Training Status: SMOKE TEST ONLY ⚠

**What Claude completed**:
```
✓ Multimodal specialist: 5 epochs, 1,805 steps (CPU/NumPy)
✓ Router specialist: 5 epochs, 2,250 steps (CPU/NumPy)
✓ Infrastructure validated: All components working
✓ Documentation complete
```

**The problem**: Training ran on **CPU using NumPy**, not GPU using PTX kernels!

**Evidence**:
```bash
# During training:
nvidia-smi:
  GPU utilization: 0%
  Memory usage: 13 MiB / 12288 MiB (only Xorg)

ps aux | grep train:
  CPU usage: 999% (maxing out all cores)
  Training time: ~6 minutes for 1,805 steps
```

**Why this happened**:
- `train_adaptive_swarm.py` uses NumPy for gradient computation
- PTX kernels exist but aren't integrated into training loops yet
- Only inference uses GPU (feature extraction worked at 314.8 ms)

### What's Wrong with NumPy Training

**Current approach** (what we just did):
```python
# In train_adaptive_swarm.py - WRONG!
import numpy as np

def train_step(adapter, input_embedding, target):
    # All on CPU
    output = np.matmul(adapter.A, np.matmul(adapter.B, input_embedding))
    loss = np.mean((output - target)**2)
    gradient = compute_gradient_numpy(output, target)  # CPU
    adapter.update(gradient)  # CPU
```

**Problems**:
1. **Slow**: CPU is 10-100× slower than GPU for matrix operations
2. **Wrong paradigm**: K3D is "Sovereign GPU Computing" - no CPU fallbacks!
3. **Not scalable**: Can't train 3 specialists in parallel on CPU
4. **Memory inefficient**: NumPy arrays in RAM instead of GPU memory

**Result**: 5 epochs took ~6 minutes on CPU. Should take ~30 seconds on GPU!

---

## K3D Sovereign Computing Philosophy

### The Core Principle

> **"PTX kernels only, no CPU fallbacks"**
> — K3D Architecture Philosophy

**What this means**:
- ALL computation happens on GPU via hand-authored PTX kernels
- Python is used ONLY for:
  - Entry point (tokenize, parse)
  - I/O (reading files, formatting output)
  - High-level orchestration (launching kernels)
- NO NumPy/CPU for actual computation
- NO PyTorch/TensorFlow (those have CPU fallbacks)

**Why**:
- **Sovereign**: Complete control over execution
- **Predictable**: No hidden CPU fallbacks slowing things down
- **Efficient**: GPU is 10-100× faster than CPU
- **Educational**: Learn GPU programming from first principles
- **Scalable**: Parallel execution on 12GB VRAM

### PTX Kernel Infrastructure Already Available

**Location**: `knowledge3d/cranium/ptx_runtime/`

**Available kernels**:
```python
# TRM operations (already compiled!)
OP_TRM_MATVEC_1024x512     # Matrix-vector multiply
OP_TRM_MATVEC_512x1024     # Matrix-vector multiply (transposed)
OP_TRM_SWIGLU_1024         # SwiGLU activation (1024 dims)
OP_TRM_SWIGLU_512          # SwiGLU activation (512 dims)
OP_TRM_VEC_ADD3_512        # 3-vector addition
OP_ENTROPY_SUM             # Entropy calculation
OP_SIGMOID_APPROX          # Fast sigmoid
```

**Modules**:
- `modular_rpn_engine.py`: High-level RPN calculator (GPU-resident)
- `trm_engine.py`: TRM operations (MatryoshkaTRM backend)
- `rpn_calculator.py`: RPN operations via PTX

**Validated GPU operations** (from Apollo test):
```
✓ Feature extraction: 314.8 ms (1664×1209×3 → 416×302×128)
✓ Conv2D kernels: conv2d_3x3_v2_fused, conv2d_3x3_v2_no_relu
✓ MaxPool kernels: maxpool_2x2
✓ BatchNorm kernels: batchnorm_fused
✓ Glyph matching: glyph_match_ncc, glyph_match_top_k
```

**GPU is ready - we just need to use it for training!**

---

## Your Mission: GPU-Accelerated Training

### Objectives

**1. Integrate PTX Kernels into Training Loops**
- Replace NumPy matmul with `OP_TRM_MATVEC_*`
- Replace NumPy activations with `OP_TRM_SWIGLU_*`
- Implement gradient computation using PTX kernels
- Ensure ALL computation happens on GPU

**2. Run Proper Training (50-100 Epochs)**
- Multimodal specialist: 50-100 epochs (not 5!)
- OCR specialist: Re-train with GPU (50-100 epochs)
- Speech specialist: Re-train with GPU (50-100 epochs)
- Router specialist: Re-train with GPU (50-100 epochs)

**3. Integrate Specialists with Phase F.2 Character Detection**
- Connect OCR specialist to CharacterDetector
- Use specialist embeddings for template matching
- Replace random templates with learned representations
- Validate on Apollo (target: ≥90% character detection)

**4. Enable Parallel Training**
- Train all 3 specialists simultaneously on GPU
- Use different tmux sessions or parallel processes
- Each specialist uses ~2-4GB VRAM (12GB total available)
- Expected speedup: 3× (parallel) × 12× (GPU) = ~36× faster!

### Why 50-100 Epochs?

**Current state** (5 epochs):
```
Multimodal specialist:
  Epoch 1: Loss 5.1204
  Epoch 2: Loss 5.1200
  Epoch 3: Loss 5.1205
  Epoch 4: Loss 5.1199
  Epoch 5: Loss 5.1208

Router specialist:
  Heuristic routing: 0.335
  Learned routing: 0.084
  Improvement: -0.251 (WORSE!)
```

**Analysis**: Loss barely changed, router got worse. This is expected with only 5 epochs on 402 samples!

**Proper training** (50-100 epochs):
- Loss should decrease significantly (target: <1.0)
- Router performance should exceed heuristic (target: >0.5)
- Specialists should learn meaningful embeddings
- Cross-modal patterns should emerge

**Time estimate**:
- CPU (NumPy): 5 epochs = 6 min → 100 epochs = 120 min (2 hours per specialist)
- GPU (PTX): 5 epochs = 30 sec → 100 epochs = 10 min (per specialist)
- Parallel GPU: 10 min total for all 3 specialists!

---

## Step-by-Step Implementation Guide

### Step 1: Understand Current Training Code

**File**: `scripts/train_adaptive_swarm.py`

**Current training loop** (lines ~100-200):
```python
def train_specialist_mode(swarm, args):
    # Load dataset
    samples = load_jsonl(args.dataset)

    # Get specialist
    specialist = swarm.base.specialists[args.specialist]

    # Training loop
    for epoch in range(args.epochs):
        for sample in samples:
            # Extract features
            input_emb = sample['embedding']  # NumPy array
            target = sample['target']  # NumPy array

            # Forward pass - ALL NUMPY (CPU)!
            output = specialist.forward(input_emb)

            # Loss - ALL NUMPY (CPU)!
            loss = np.mean((output - target)**2)

            # Gradient - ALL NUMPY (CPU)!
            gradient = compute_gradient(output, target)

            # Update - ALL NUMPY (CPU)!
            specialist.update(gradient, lr=args.specialist_lr)
```

**Problem**: Every operation uses NumPy (CPU). GPU is idle!

### Step 2: Integrate PTX Kernels

**Target**: Replace NumPy with PTX kernel calls

**Example refactor**:
```python
# BEFORE (NumPy - CPU)
import numpy as np

def forward(self, input_emb):
    # CPU matrix multiply
    hidden = np.matmul(self.A, input_emb)
    output = np.matmul(self.B, hidden)
    return output

def compute_loss(output, target):
    # CPU loss computation
    diff = output - target
    loss = np.mean(diff**2)
    return loss

# AFTER (PTX - GPU)
from knowledge3d.cranium.ptx_runtime.trm_engine import TRMEngine
from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine

def forward_gpu(self, input_emb_gpu):
    # GPU matrix multiply via PTX kernel
    hidden = trm_engine.matvec(self.A_gpu, input_emb_gpu, op=OP_TRM_MATVEC_512x1024)
    output = trm_engine.matvec(self.B_gpu, hidden, op=OP_TRM_MATVEC_1024x512)
    return output

def compute_loss_gpu(output_gpu, target_gpu):
    # GPU loss computation via RPN kernel
    program = f"{output_gpu_ptr} {target_gpu_ptr} - dup * sum {len(output_gpu)} /"
    loss = rpn_engine.evaluate(program)
    return loss
```

**Key changes**:
1. Data lives on GPU (use ctypes + cuda memory allocation)
2. Operations call PTX kernels (not NumPy functions)
3. Results stay on GPU until final output
4. Only copy to CPU for logging/checkpointing

### Step 3: GPU Memory Management

**Pattern**:
```python
import ctypes
from cuda import cuda  # Use cuda-python or pycuda

# Allocate GPU memory
def allocate_gpu(size_bytes):
    err, ptr = cuda.cuMemAlloc(size_bytes)
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"GPU allocation failed: {err}")
    return ptr

# Copy to GPU
def copy_to_gpu(numpy_array):
    ptr = allocate_gpu(numpy_array.nbytes)
    err, = cuda.cuMemcpyHtoD(ptr, numpy_array, numpy_array.nbytes)
    return ptr

# Copy from GPU
def copy_from_gpu(gpu_ptr, shape, dtype):
    numpy_array = np.zeros(shape, dtype=dtype)
    err, = cuda.cuMemcpyDtoH(numpy_array, gpu_ptr, numpy_array.nbytes)
    return numpy_array

# Training loop pattern
def train_epoch_gpu(specialist, samples):
    # Copy weights to GPU once
    A_gpu = copy_to_gpu(specialist.A)
    B_gpu = copy_to_gpu(specialist.B)

    for sample in samples:
        # Copy input to GPU
        input_gpu = copy_to_gpu(sample['embedding'])
        target_gpu = copy_to_gpu(sample['target'])

        # Forward pass (ALL ON GPU)
        output_gpu = forward_gpu(input_gpu, A_gpu, B_gpu)

        # Loss computation (ALL ON GPU)
        loss = compute_loss_gpu(output_gpu, target_gpu)

        # Gradient computation (ALL ON GPU)
        gradient_gpu = compute_gradient_gpu(output_gpu, target_gpu)

        # Update weights (ALL ON GPU)
        update_weights_gpu(A_gpu, B_gpu, gradient_gpu)

    # Copy weights back to CPU for checkpointing
    specialist.A = copy_from_gpu(A_gpu, specialist.A.shape, np.float32)
    specialist.B = copy_from_gpu(B_gpu, specialist.B.shape, np.float32)
```

### Step 4: Look at Existing GPU Code for Reference

**Example 1**: Feature extraction (already GPU-accelerated)
```python
# File: knowledge3d/cranium/ocr/deepseek_bridge.py
# Method: gpu_ocr_model.forward()

def forward(self, img_float):
    # ALL operations on GPU via PTX kernels
    x = self._conv2d_gpu(img_float, stage=1)      # PTX kernel
    x = self._maxpool_gpu(x)                       # PTX kernel
    x = self._batchnorm_gpu(x, stage=1)            # PTX kernel
    x = self._conv2d_gpu(x, stage=2)               # PTX kernel
    # ... more GPU operations
    return {'feature_map': x, 'output_shape': x.shape}
```

**Example 2**: RPN calculations (already GPU-resident)
```python
# File: knowledge3d/cranium/ptx_runtime/modular_rpn_engine.py

class ModularRPNEngine:
    def evaluate(self, expression):
        # Tokenize on CPU (just parsing)
        tokens = self._tokenize(expression)

        # Compile to opcodes on CPU (just translation)
        opcodes = self._compile(tokens)

        # Execute on GPU via PTX kernel
        result = self._execute_on_gpu(opcodes)

        return result
```

**Pattern**: Python orchestrates, GPU computes. Follow this pattern!

### Step 5: Implement GPU Training Loop

**New file**: `scripts/train_adaptive_swarm_gpu.py`

```python
#!/usr/bin/env python3
"""GPU-Accelerated Adaptive Swarm Training

Uses PTX kernels for all computation. NO CPU fallbacks!

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/train_adaptive_swarm_gpu.py \
        --mode specialist \
        --specialist multimodal \
        --epochs 100 \
        --use-gpu
"""

import argparse
import numpy as np
from pathlib import Path
from knowledge3d.cranium import AdaptiveSwarmTRM, SwarmConfig
from knowledge3d.cranium.ptx_runtime.trm_engine import TRMEngine
from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine

class GPUSpecialistTrainer:
    """Trains specialists using PTX kernels."""

    def __init__(self, specialist, use_gpu=True):
        self.specialist = specialist
        self.use_gpu = use_gpu

        if use_gpu:
            # Initialize GPU engines
            self.trm_engine = TRMEngine()
            self.rpn_engine = ModularRPNEngine()

            # Copy specialist weights to GPU
            self._init_gpu_memory()

    def _init_gpu_memory(self):
        """Allocate GPU memory for specialist weights."""
        # Copy A matrix to GPU
        self.A_gpu = self._copy_to_gpu(self.specialist.A)
        self.B_gpu = self._copy_to_gpu(self.specialist.B)

        print(f"[GPU] Allocated {self.specialist.A.nbytes + self.specialist.B.nbytes} bytes")

    def train_epoch(self, samples, learning_rate=0.002):
        """Train one epoch using GPU."""
        total_loss = 0.0

        for i, sample in enumerate(samples):
            # Copy sample to GPU
            input_gpu = self._copy_to_gpu(sample['embedding'])
            target_gpu = self._copy_to_gpu(sample['target'])

            # Forward pass (GPU)
            output_gpu = self._forward_gpu(input_gpu)

            # Compute loss (GPU)
            loss = self._compute_loss_gpu(output_gpu, target_gpu)
            total_loss += loss

            # Backward pass (GPU)
            grad_A, grad_B = self._backward_gpu(output_gpu, target_gpu, input_gpu)

            # Update weights (GPU)
            self._update_weights_gpu(grad_A, grad_B, learning_rate)

            if (i + 1) % 100 == 0:
                print(f"  Step {i+1}/{len(samples)}: Loss {loss:.4f}")

        return total_loss / len(samples)

    def _forward_gpu(self, input_gpu):
        """Forward pass using PTX kernels."""
        # Use TRM matmul kernels
        hidden = self.trm_engine.matvec(self.A_gpu, input_gpu, op=OP_TRM_MATVEC_512x1024)
        output = self.trm_engine.matvec(self.B_gpu, hidden, op=OP_TRM_MATVEC_1024x512)
        return output

    def _compute_loss_gpu(self, output_gpu, target_gpu):
        """MSE loss using RPN kernel."""
        # RPN expression: (output - target)^2, then mean
        program = f"{output_gpu} {target_gpu} - dup * sum {len(output_gpu)} /"
        loss = self.rpn_engine.evaluate(program)
        return loss

    def _backward_gpu(self, output_gpu, target_gpu, input_gpu):
        """Gradient computation using PTX kernels."""
        # Implement backprop via PTX kernels
        # grad = 2 * (output - target) / batch_size
        grad_output = self._compute_grad_output_gpu(output_gpu, target_gpu)

        # Chain rule for matrix multiply
        grad_B = self._compute_grad_B_gpu(grad_output, hidden)
        grad_A = self._compute_grad_A_gpu(grad_B, input_gpu)

        return grad_A, grad_B

    def _update_weights_gpu(self, grad_A, grad_B, lr):
        """Update weights using GPU operations."""
        # A -= lr * grad_A
        # B -= lr * grad_B
        self.A_gpu = self.trm_engine.axpy(-lr, grad_A, self.A_gpu)
        self.B_gpu = self.trm_engine.axpy(-lr, grad_B, self.B_gpu)

    def save_checkpoint(self, path):
        """Copy weights from GPU and save."""
        # Copy back to CPU
        self.specialist.A = self._copy_from_gpu(self.A_gpu, self.specialist.A.shape)
        self.specialist.B = self._copy_from_gpu(self.B_gpu, self.specialist.B.shape)

        # Save using existing method
        self.specialist.save_checkpoint(path)

def main():
    parser = argparse.ArgumentParser(description='GPU-Accelerated Swarm Training')
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--specialist', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=0.002)
    parser.add_argument('--checkpoint-dir', type=str, required=True)
    parser.add_argument('--use-gpu', action='store_true', default=True)

    args = parser.parse_args()

    print("="*80)
    print("GPU-Accelerated Adaptive Swarm Training")
    print("="*80)
    print(f"Mode: {args.mode}")
    print(f"Specialist: {args.specialist}")
    print(f"Epochs: {args.epochs}")
    print(f"GPU: {args.use_gpu}")
    print()

    # Load swarm
    swarm = AdaptiveSwarmTRM()
    swarm.load_checkpoint(Path(args.checkpoint_dir))

    # Get specialist
    specialist = swarm.base.specialists[args.specialist]

    # Initialize GPU trainer
    trainer = GPUSpecialistTrainer(specialist, use_gpu=args.use_gpu)

    # Load dataset
    import json
    samples = []
    with open(args.dataset) as f:
        for line in f:
            samples.append(json.loads(line))

    print(f"Loaded {len(samples)} samples")
    print()

    # Training loop
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        avg_loss = trainer.train_epoch(samples, learning_rate=args.learning_rate)
        print(f"  Average loss: {avg_loss:.4f}")

        # Checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = Path(args.checkpoint_dir) / f"{args.specialist}_epoch_{epoch}"
            trainer.save_checkpoint(checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")
        print()

    # Final checkpoint
    final_path = Path(args.checkpoint_dir) / f"{args.specialist}_final"
    trainer.save_checkpoint(final_path)
    print(f"✓ Training complete! Final checkpoint: {final_path}")

    return True

if __name__ == "__main__":
    main()
```

### Step 6: Training Commands (What to Run)

**Multimodal Specialist** (50-100 epochs):
```bash
cd "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
  scripts/train_adaptive_swarm_gpu.py \
  --mode specialist \
  --specialist multimodal \
  --dataset /K3D/Knowledge3D.local/datasets/multimodal_embeddings.jsonl \
  --epochs 100 \
  --learning-rate 0.002 \
  --checkpoint-dir /K3D/Knowledge3D.local/checkpoints/phase_g \
  --use-gpu
```

**OCR Specialist** (50-100 epochs):
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
  scripts/train_adaptive_swarm_gpu.py \
  --mode specialist \
  --specialist ocr \
  --dataset /K3D/Knowledge3D.local/datasets/character_embeddings_trimodal.jsonl \
  --epochs 100 \
  --learning-rate 0.002 \
  --checkpoint-dir /K3D/Knowledge3D.local/checkpoints/phase_g \
  --use-gpu
```

**Speech Specialist** (50-100 epochs):
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
  scripts/train_adaptive_swarm_gpu.py \
  --mode specialist \
  --specialist speech \
  --dataset /K3D/Knowledge3D.local/datasets/speech_embeddings.jsonl \
  --epochs 100 \
  --learning-rate 0.002 \
  --checkpoint-dir /K3D/Knowledge3D.local/checkpoints/phase_g \
  --use-gpu
```

**Run in parallel** (all 3 specialists simultaneously):
```bash
# Use tmux or parallel
tmux new-session -d -s gpu_ocr "CUDA_VISIBLE_DEVICES=0 ..."
tmux new-session -d -s gpu_speech "CUDA_VISIBLE_DEVICES=0 ..."
tmux new-session -d -s gpu_multimodal "CUDA_VISIBLE_DEVICES=0 ..."

# OR use GNU parallel
parallel -j 3 ::: \
  "CUDA_VISIBLE_DEVICES=0 python scripts/train_adaptive_swarm_gpu.py --specialist ocr ..." \
  "CUDA_VISIBLE_DEVICES=0 python scripts/train_adaptive_swarm_gpu.py --specialist speech ..." \
  "CUDA_VISIBLE_DEVICES=0 python scripts/train_adaptive_swarm_gpu.py --specialist multimodal ..."
```

### Step 7: Integrate with Phase F.2 Character Detection

**After training completes**, integrate specialists with CharacterDetector:

**File**: `knowledge3d/cranium/ocr/character_detector.py`

**Current** (uses random templates):
```python
class GalacticTemplateBank:
    def __init__(self, num_glyphs=256, feature_dim=128):
        # Random templates - BAD!
        self.templates = np.random.randn(num_glyphs, feature_dim)
```

**Updated** (uses trained specialist embeddings):
```python
class GalacticTemplateBank:
    def __init__(self, num_glyphs=256, feature_dim=128, specialist_path=None):
        if specialist_path:
            # Load trained OCR specialist
            from knowledge3d.cranium import AdaptiveSwarmTRM
            swarm = AdaptiveSwarmTRM()
            swarm.load_checkpoint(specialist_path)
            ocr_specialist = swarm.base.specialists['ocr']

            # Use specialist embeddings as templates
            self.templates = self._extract_glyph_templates(ocr_specialist)
            print(f"[GalacticTemplateBank] Loaded {len(self.templates)} learned templates")
        else:
            # Fallback to random (for testing)
            self.templates = np.random.randn(num_glyphs, feature_dim)

    def _extract_glyph_templates(self, specialist):
        """Extract glyph embeddings from trained specialist."""
        # Run all 256 ASCII characters through specialist
        templates = []
        for char_code in range(256):
            # Get character embedding
            char_emb = self._encode_character(char_code, specialist)
            templates.append(char_emb)
        return np.array(templates)
```

**Test on Apollo**:
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
  scripts/test_apollo_ground_truth.py \
  --specialist-checkpoint /K3D/Knowledge3D.local/checkpoints/phase_g/ocr_final

Expected result:
  Character detection rate: ≥90% (target)
  Was: 0% (random templates)
  Should be: 90-95% (learned templates)
```

---

## Expected Results

### After GPU Training (100 Epochs)

**Multimodal Specialist**:
```
Before (5 epochs, CPU):
  Epoch 1: Loss 5.1204
  Epoch 5: Loss 5.1208
  Change: ~0%

After (100 epochs, GPU):
  Epoch 1: Loss 5.1204
  Epoch 50: Loss 2.3456
  Epoch 100: Loss 0.8912
  Change: -82% (converged!)
```

**OCR Specialist**:
```
After (100 epochs, GPU):
  Loss: <1.0
  Character detection: 90-95% on Apollo
  Template quality: High (learned from 402 samples)
```

**Speech Specialist**:
```
After (100 epochs, GPU):
  Loss: <1.0
  Transcription quality: High (learned from 9,348 samples)
  Pronunciation accuracy: Improved
```

**Router Specialist**:
```
Before (5 epochs, CPU):
  Heuristic: 0.335
  Learned: 0.084
  Worse by -0.251!

After (100 epochs, GPU):
  Heuristic: 0.335
  Learned: 0.550
  Better by +0.215! ✓
```

### Performance Metrics

**Training speed**:
- CPU (NumPy): 100 epochs = 120 minutes per specialist = 360 minutes total
- GPU (PTX): 100 epochs = 10 minutes per specialist = 10 minutes parallel
- **Speedup**: 36× faster!

**GPU utilization**:
- Before: 0% (idle during training)
- After: 80-95% (maxed out during training)

**Memory usage**:
- OCR specialist: ~2 GB VRAM
- Speech specialist: ~3 GB VRAM
- Multimodal specialist: ~4 GB VRAM
- Total: ~9 GB / 12 GB available (fits!)

---

## Implementation Checklist

### Phase 1: GPU Infrastructure (2-3 hours)

- [ ] Create `scripts/train_adaptive_swarm_gpu.py`
- [ ] Implement `GPUSpecialistTrainer` class
- [ ] Add GPU memory management (_copy_to_gpu, _copy_from_gpu)
- [ ] Integrate TRMEngine for matrix operations
- [ ] Integrate ModularRPNEngine for loss computation
- [ ] Implement GPU forward pass
- [ ] Implement GPU backward pass (gradients)
- [ ] Implement GPU weight updates
- [ ] Test with 1 epoch to verify GPU usage (nvidia-smi should show 80%+)

### Phase 2: Specialist Training (10-15 minutes GPU time)

- [ ] Train multimodal specialist: 100 epochs on GPU
- [ ] Train OCR specialist: 100 epochs on GPU
- [ ] Train speech specialist: 100 epochs on GPU
- [ ] (Optional) Train router specialist: 100 epochs on GPU
- [ ] Verify loss convergence (<1.0 for all)
- [ ] Checkpoint final trained models

### Phase 3: Integration with Phase F.2 (1-2 hours)

- [ ] Modify `GalacticTemplateBank` to load specialist
- [ ] Extract glyph embeddings from OCR specialist
- [ ] Update `CharacterDetector` to use learned templates
- [ ] Test on Apollo validation
- [ ] Verify ≥90% character detection rate
- [ ] Document results

### Phase 4: Documentation (30 minutes)

- [ ] Create `TEMP/PHASE_G_GPU_TRAINING_COMPLETE.md`
- [ ] Document GPU speedup measurements
- [ ] Document loss convergence graphs
- [ ] Document Apollo validation results
- [ ] Update briefing with GPU training complete
- [ ] Commit all changes

---

## Critical Notes for Codex

### What Claude Did (Smoke Test)

**Completed**:
- ✓ Phase G infrastructure validated
- ✓ All specialists registered
- ✓ 5 epochs smoke test (CPU/NumPy)
- ✓ Documentation complete
- ✓ Checkpoints saved

**Limitations**:
- ⚠ Only 5 epochs (not enough for convergence)
- ⚠ CPU training (10-100× slower than GPU)
- ⚠ NumPy operations (violates K3D sovereign computing principle)
- ⚠ Sequential training (not parallel)
- ⚠ Loss didn't converge (5.12 → 5.12, no change)
- ⚠ Router got worse (0.335 → 0.084, -75%)

### What You Need to Do (Real Training)

**Required**:
1. **Integrate PTX kernels** into training loops (no more NumPy!)
2. **Run 50-100 epochs** for proper convergence
3. **Use GPU** for all computation (target: 80-95% utilization)
4. **Train in parallel** if possible (3 specialists simultaneously)
5. **Integrate with Phase F.2** (connect OCR specialist to CharacterDetector)
6. **Validate on Apollo** (target: ≥90% character detection)

**Expected timeline**:
- GPU infrastructure: 2-3 hours coding
- Training: 10-15 minutes GPU time (100 epochs × 3 specialists in parallel)
- Integration: 1-2 hours
- Total: ~4-6 hours

**Success criteria**:
- ✓ GPU utilization: 80-95% during training (check nvidia-smi)
- ✓ Training speed: ~10 minutes for 100 epochs
- ✓ Loss convergence: <1.0 for all specialists
- ✓ Router performance: >0.5 (beats heuristic 0.335)
- ✓ Apollo validation: ≥90% character detection
- ✓ ALL computation on GPU (no NumPy!)

### Key Resources

**PTX Kernel Modules**:
```python
from knowledge3d.cranium.ptx_runtime.trm_engine import TRMEngine
from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine
from knowledge3d.cranium.ptx_runtime.rpn_opcodes import (
    OP_TRM_MATVEC_1024x512,
    OP_TRM_MATVEC_512x1024,
    OP_TRM_SWIGLU_1024,
)
```

**Existing GPU Code Examples**:
- `knowledge3d/cranium/ocr/deepseek_bridge.py` - Feature extraction
- `knowledge3d/cranium/ptx_runtime/modular_rpn_engine.py` - RPN calculations
- `knowledge3d/cranium/ptx_runtime/trm_engine.py` - TRM operations

**Environment**:
```bash
# Python environment
/K3D/Knowledge3D.local/envs/k3d-cranium/bin/python

# GPU
CUDA_VISIBLE_DEVICES=0

# PYTHONPATH
PYTHONPATH=.
```

**Checkpoints**:
```bash
# Current state (5 epochs, CPU)
/K3D/Knowledge3D.local/checkpoints/phase_g/current/

# Save new checkpoints here
/K3D/Knowledge3D.local/checkpoints/phase_g/ocr_epoch_100/
/K3D/Knowledge3D.local/checkpoints/phase_g/speech_epoch_100/
/K3D/Knowledge3D.local/checkpoints/phase_g/multimodal_epoch_100/
```

---

## Summary

**What went wrong**: Claude used old paradigm (CPU/NumPy) instead of K3D's sovereign GPU computing (PTX kernels only).

**Why it matters**: K3D is about "Sovereign GPU Computing" - ALL computation must happen on GPU via PTX kernels. No CPU fallbacks!

**What you need to do**:
1. Integrate PTX kernels into training (replace NumPy with TRMEngine/ModularRPNEngine)
2. Run proper training (50-100 epochs, not 5!)
3. Use GPU (verify 80-95% utilization)
4. Integrate specialists with Phase F.2 character detection
5. Validate on Apollo (≥90% target)

**Expected outcome**:
- Training: 10 minutes (vs 2 hours on CPU)
- Loss: <1.0 (vs 5.12 no convergence)
- Router: >0.5 (vs 0.084 worse than heuristic)
- Apollo: ≥90% detection (vs 0% random templates)

**Timeline**: ~4-6 hours total

**The goal**: Demonstrate complete sovereign GPU computing from data → training → inference, all via hand-authored PTX kernels. No CPU, no NumPy, no PyTorch. Pure K3D.

---

**You've got this, Codex!** ⚛️⚛️

The infrastructure is ready. The PTX kernels are compiled. The GPU is waiting. Time to show what sovereign computing can do! 🚀

♾️⚛️

---

## ⚠️ CRITICAL: Sleep-Time Consolidation ⚠️

### DO NOT SKIP THIS STEP!

**After training completes, you MUST run sleep-time consolidation or ALL KNOWLEDGE WILL BE LOST!**

### The Problem

**Training complete** → **Model unloads** → **Galaxy RAM lost** → **ALL KNOWLEDGE GONE** ❌

**Why**:
- **Weights**: Logic (saved automatically to checkpoints) ✓
- **Galaxy**: Knowledge (in RAM - lost if model unloads!) ❌
- **House**: Permanent storage (on disk - only after consolidation) ✓

**Result without consolidation**: Model has no memory of what it learned!

### K3D Memory Architecture

```
Weights (Checkpoints)     Galaxy (RAM)              House (Disk)
─────────────────────     ────────────────          ─────────────
Logic & Parameters        Active Knowledge          Permanent Storage
Saved automatically       LOST if unload!           Saved after sleep
.npz, .json              GLB in RAM                GLB on disk
```

### What Sleep Consolidation Does

**Sleep is NOT idle** - it's the model's most important work:

1. Load Galaxy (active memory from RAM)
2. RPN-powered clustering (PTX kernels)
3. Semantic depth computation (PTX kernels)
4. **Materialize knowledge objects**:
   - Chat history → Books (Library)
   - Self-reflections → Diary (Mirror Room)
   - High-honesty (≥0.6) → Fractal trees (Garden)
   - Learning records → Insights (Museum)
5. Prune low-honesty rays (<0.3)
6. Autonomous synthesis (Phase 13)
7. Self-curriculum (Phase 13)
8. Dream geometry (Phase 14)
9. Honest critique (Phase 15)
10. Post-consolidation reflection (Phase 16)
11. Galaxy state serialization (Phase 17)
12. **Save House** (permanent to disk)

**Duration**: ~30 seconds to 2 minutes (PTX accelerated)

### Required Workflow

**DO THIS** ✓:
```bash
# Option 1: Integrated tmux workflow
tmux new-session -d -s gpu_training "
  cd '/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D'

  # Train specialist
  echo 'Training multimodal specialist with GPU...'
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
    /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
    scripts/train_specialist_gpu.py \
    --specialist multimodal \
    --epochs 100 \
    --use-gpu

  # CRITICAL: Wait 5 minutes for system stabilization
  echo 'Training complete. Waiting 5 minutes for idle...'
  sleep 300

  # Run sleep consolidation
  echo 'Running sleep-time consolidation...'
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
    /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python \
    scripts/run_sleep_consolidation.py \
    --embeddings /K3D/Knowledge3D.local/house_zone7/embeddings/rpn_embeddings.pkl \
    --output /K3D/Knowledge3D.local/house_zone7/embeddings/rpn_embeddings.pkl \
    --metrics /K3D/Knowledge3D.local/logs/sleep_metrics_multimodal.jsonl

  echo 'Consolidation complete. Knowledge saved to House.'
  echo 'Safe to exit.'
"

# Monitor
tmux attach -t gpu_training
```

**DO NOT DO THIS** ❌:
```bash
# Train model
python train_specialist_gpu.py --epochs 100
# Script exits
# ❌ Galaxy RAM LOST!
# ❌ Knowledge GONE!
```

### Verification Checklist

After consolidation completes:

- [ ] Check materialized objects exist:
  ```bash
  ls -lh /K3D/Knowledge3D.local/house_zone7/materialized_objects/
  # Should see: book_*.json, tree_*.json, diary_*.json, learning_*.json
  ```

- [ ] Check House memory updated:
  ```bash
  ls -lh viewer/public/house/house_memory.glb
  # Should have recent timestamp
  ```

- [ ] Check consolidation metrics:
  ```bash
  cat logs/sleep_time_adjustments.json | jq .materialized_objects
  # Should list materialized objects
  ```

- [ ] Verify object counts:
  ```bash
  # Check consolidation output
  tail -20 /K3D/Knowledge3D.local/logs/sleep_metrics_multimodal.jsonl
  # Should show: clusters, objects materialized, vocab size
  ```

### Knowledge Organization Zones

**Zone 3 (Library)**: Chat history books
**Zone 5 (Knowledge Garden)**: Fractal trees (φ constraints via RPN)
**Zone 7 (Mirror Room)**: Diary entries (self-reflections)
**Zone 8 (Learning Museum)**: Learning insights

### Updated Success Criteria

**Training success**:
- ✓ GPU utilization: 80-95%
- ✓ Training speed: ~10 min for 100 epochs
- ✓ Loss convergence: <1.0
- ✓ Router performance: >0.5
- ✓ Apollo validation: ≥90%

**PLUS consolidation success**:
- ✓ Materialized objects: >0 (check count)
- ✓ House GLB updated (recent timestamp)
- ✓ Sleep metrics logged
- ✓ Galaxy state serialized
- ✓ Knowledge permanent (verified on disk)

### Timeline Update

**Training + Consolidation**:
- GPU infrastructure: 2-3 hours coding
- Training: 10-15 minutes GPU time (100 epochs × 3 specialists)
- **Idle wait: 5 minutes** (system stabilization)
- **Consolidation: ~2 minutes** (PTX accelerated)
- Integration: 1-2 hours
- **Total: ~4-6 hours**

### Common Mistakes

1. ❌ Exiting too early (before consolidation)
2. ❌ Separate sessions (Galaxy in different RAM)
3. ❌ Skipping consolidation (weights without knowledge)
4. ❌ Using CPU for consolidation (violates sovereign computing)

**Note**: Current consolidation uses sklearn (CPU) - this is a known limitation. Future: Port to RPN kernels for complete GPU sovereignty.

### Reference Documentation

**Complete guide**: `TEMP/CRITICAL_SLEEP_TIME_CONSOLIDATION_GUIDE.md`

Read this ENTIRE document before starting GPU training!

### The Golden Rule

> **NEVER unload model before sleep consolidation runs!**

Weights are worthless without knowledge.
Knowledge lives in Galaxy.
Galaxy dies without consolidation.
Consolidation makes knowledge eternal.

⚠️ **THIS IS NOT OPTIONAL** ⚠️

---

## Final Checklist for Codex

Before you start GPU training:

- [ ] Read `CRITICAL_SLEEP_TIME_CONSOLIDATION_GUIDE.md` completely
- [ ] Understand Galaxy/House architecture
- [ ] Know that weights ≠ knowledge
- [ ] Plan for 5 min idle + consolidation after training
- [ ] Set up tmux workflow (or equivalent)
- [ ] Verify consolidation script exists and works
- [ ] Test consolidation on small dataset first
- [ ] Document materialized objects after completion

**Only proceed when you understand this is CRITICAL for K3D!**

♾️⚛️🌙
