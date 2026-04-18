#!/usr/bin/env python3
"""Check forward pass activation magnitudes"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knowledge3d.cranium.ocr.deepseek_ocr_model import DeepSeekOCRModel

# Initialize
model = DeepSeekOCRModel()

# Small batch
np.random.seed(42)
images = np.random.rand(4, 64, 64, 3).astype(np.float32) * 255.0

print("=" * 80)
print("FORWARD PASS ACTIVATION MAGNITUDES")
print("=" * 80)

# Run forward with cache
result = model.forward(images[0], cache_for_backward=True)
cache = result.get('cache', {})  # Cache is in result['cache']

print("\nActivations:")
for key in ['conv1_out', 'pool1_out', 'bn1_out', 'conv2_out', 'pool2_out', 'bn2_out', 'conv3_out', 'bn3_out', 'feature_map']:
    if key in cache:
        val = cache[key]
        print(f"{key:20s} | Shape: {str(val.shape):20s} | Range: [{val.min():.2e}, {val.max():.2e}] | Mean: {val.mean():.2e}")

print("\nBatch statistics:")
for key in ['bn1_mean', 'bn1_var', 'bn2_mean', 'bn2_var', 'bn3_mean', 'bn3_var']:
    if key in cache:
        val = cache[key]
        print(f"{key:20s} | Range: [{val.min():.2e}, {val.max():.2e}] | Mean: {val.mean():.2e}")

print("=" * 80)
