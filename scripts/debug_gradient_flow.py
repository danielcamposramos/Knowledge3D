#!/usr/bin/env python3
"""Debug gradient flow through backward pass"""

import numpy as np
import ctypes
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knowledge3d.cranium.ocr.deepseek_ocr_model import DeepSeekOCRModel
from knowledge3d.cranium.ocr.gpu_trainer import GPUCNNTrainer

# Initialize
print("Initializing...")
model = DeepSeekOCRModel()
trainer = GPUCNNTrainer(model, learning_rate=0.1, momentum=0.9)

# Small batch
np.random.seed(42)
images = np.random.rand(4, 64, 64, 3).astype(np.float32) * 255.0
labels = np.array([0, 1, 2, 3], dtype=np.int32)

print("\n" + "=" * 80)
print("GRADIENT FLOW DEBUGGING")
print("=" * 80)

# Patch train_batch to capture intermediate gradients
original_backward = trainer._backward_pass

intermediate_grads = {}

def instrumented_backward(cache):
    """Instrumented backward pass that captures intermediate gradients"""
    from knowledge3d.cranium.sovereign import loader

    # Run normal backward
    original_backward(cache)

    # Capture intermediate values by inspecting cache
    print("\n[Forward pass activations]")
    for key in ['conv1_out', 'pool1_out', 'conv2_out', 'pool2_out', 'conv3_out', 'feature_map']:
        if key in cache:
            val = cache[key]
            print(f"{key:20s} | Shape: {str(val.shape):20s} | Range: [{val.min():.2e}, {val.max():.2e}]")

trainer._backward_pass = instrumented_backward

# Run training
print("\n[Running forward pass...]")
loss, acc = trainer.train_batch(images, labels)

print(f"\nLoss: {loss:.4f}, Accuracy: {acc:.2f}%")
print("=" * 80)
