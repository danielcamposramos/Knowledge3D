#!/usr/bin/env python3
"""Inspect CNN weights to check if they're trained"""

import numpy as np
from pathlib import Path

weights_path = Path("/K3D/Knowledge3D.local/checkpoints/phase_g/ocr_gpu_epoch_100/ocr_cnn_weights.npz")

if not weights_path.exists():
    print(f"Weights file not found: {weights_path}")
    exit(1)

print(f"Loading weights from: {weights_path}")
weights = np.load(weights_path)

print("\n" + "=" * 80)
print("CNN WEIGHT INSPECTION")
print("=" * 80)

for key in weights.files:
    arr = weights[key]
    print(f"\n{key}:")
    print(f"  Shape: {arr.shape}")
    print(f"  Dtype: {arr.dtype}")
    print(f"  Mean: {arr.mean():.6f}")
    print(f"  Std: {arr.std():.6f}")
    print(f"  Min: {arr.min():.6f}")
    print(f"  Max: {arr.max():.6f}")

    # Check if weights look random/untrained
    if arr.std() < 0.001:
        print(f"  ⚠️  WARNING: Very low std - might be zeros/constants")
    if abs(arr.mean()) < 0.001 and arr.std() > 0.5:
        print(f"  ✓ Looks like initialized weights (mean~0, std>0)")
    if abs(arr.mean()) > 0.1 or (arr.std() > 0.01 and arr.std() < 1.0):
        print(f"  ✓ Might be trained weights")

print("\n" + "=" * 80)
