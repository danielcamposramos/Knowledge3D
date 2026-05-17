#!/usr/bin/env python3
"""Simple gradient verification script"""

import numpy as np
import ctypes
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knowledge3d.cranium.ocr.deepseek_ocr_model import DeepSeekOCRModel
from knowledge3d.cranium.ocr.gpu_trainer import GPUCNNTrainer
from knowledge3d.cranium.sovereign import loader

# Initialize model and trainer
print("Initializing model and trainer...")
model = DeepSeekOCRModel()
trainer = GPUCNNTrainer(model, learning_rate=0.09, momentum=0.9)

# Create small batch (normalized to [0, 1] like training)
np.random.seed(42)
images = np.random.rand(4, 64, 64, 3).astype(np.float32)  # Already in [0, 1]
labels = np.array([0, 1, 2, 3], dtype=np.int32)

print("\n" + "=" * 80)
print("GRADIENT VERIFICATION")
print("=" * 80)
print(f"Learning rate: 0.09")
print(f"Batch size: {len(images)}")
print()

# Run one training step
print("[Running forward and backward pass...]")
loss, accuracy = trainer.train_batch(images, labels)
print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")
print()

def check_gradient(name, d_grad_ptr, shape):
    """Download gradient from GPU and check statistics"""
    grad = np.empty(shape, dtype=np.float32)
    loader.memcpy_dtoh(grad.ctypes.data_as(ctypes.c_void_p), d_grad_ptr, grad.nbytes)

    grad_flat = grad.flatten()
    has_nan = np.isnan(grad_flat).any()
    has_inf = np.isinf(grad_flat).any()

    if has_nan or has_inf:
        status = "✗"
    else:
        status = "✓"

    grad_abs = np.abs(grad_flat)
    grad_mean = np.mean(grad_abs)
    grad_max = np.max(grad_abs)
    nonzero_pct = (grad_abs > 1e-8).sum() / len(grad_flat) * 100

    print(f"{status} {name:20s} | Mean: {grad_mean:8.2e} | Max: {grad_max:8.2e} | NonZero: {nonzero_pct:5.1f}%")

    if has_nan:
        print(f"  ⚠ WARNING: Contains NaN values!")
    if has_inf:
        print(f"  ⚠ WARNING: Contains inf values!")

    return not (has_nan or has_inf)

print("[Checking gradients...]")
print()

# Check all layer gradients
all_clean = True
all_clean &= check_gradient("Conv1 weights", trainer.d_grad_conv1_w, model.conv1_weight.shape)
all_clean &= check_gradient("Conv1 bias", trainer.d_grad_conv1_b, model.conv1_bias.shape)
all_clean &= check_gradient("BN1 gamma", trainer.d_grad_bn1_gamma, model.bn1_gamma.shape)
all_clean &= check_gradient("BN1 beta", trainer.d_grad_bn1_beta, model.bn1_beta.shape)
print()
all_clean &= check_gradient("Conv2 weights", trainer.d_grad_conv2_w, model.conv2_weight.shape)
all_clean &= check_gradient("Conv2 bias", trainer.d_grad_conv2_b, model.conv2_bias.shape)
all_clean &= check_gradient("BN2 gamma", trainer.d_grad_bn2_gamma, model.bn2_gamma.shape)
all_clean &= check_gradient("BN2 beta", trainer.d_grad_bn2_beta, model.bn2_beta.shape)
print()
all_clean &= check_gradient("Conv3 weights", trainer.d_grad_conv3_w, model.conv3_weight.shape)
all_clean &= check_gradient("Conv3 bias", trainer.d_grad_conv3_b, model.conv3_bias.shape)
all_clean &= check_gradient("BN3 gamma", trainer.d_grad_bn3_gamma, model.bn3_gamma.shape)
all_clean &= check_gradient("BN3 beta", trainer.d_grad_bn3_beta, model.bn3_beta.shape)

print()
if all_clean:
    print("✓ All gradients are clean (no NaN/inf)")
else:
    print("✗ Some gradients contain NaN/inf")

print("=" * 80)
