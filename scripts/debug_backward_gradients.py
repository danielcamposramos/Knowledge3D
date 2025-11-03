#!/usr/bin/env python3
"""
Debug Script: Gradient Monitoring for CNN Training

This script monitors gradients during training to detect:
- NaN/Inf values
- Gradient explosion (values > 100)
- Gradient vanishing (values < 1e-6)
- Suspicious gradient patterns

Usage:
    python scripts/debug_backward_gradients.py --checkpoint /path/to/checkpoint
"""

import argparse
import numpy as np
from pathlib import Path
import sys
import ctypes

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge3d.cranium.ocr.deepseek_ocr_model import DeepSeekOCRModel
from knowledge3d.cranium.ocr.gpu_trainer import GPUCNNTrainer
from knowledge3d.cranium.sovereign import loader


def analyze_gradients(trainer: GPUCNNTrainer, name: str, d_grad_ptr: int, shape: tuple):
    """Analyze gradient buffer for anomalies."""
    grad = np.empty(shape, dtype=np.float32)
    loader.memcpy_dtoh(grad.ctypes.data_as(ctypes.c_void_p), d_grad_ptr, grad.nbytes)

    # Compute statistics
    has_nan = np.isnan(grad).any()
    has_inf = np.isinf(grad).any()
    mean = np.nanmean(grad)
    std = np.nanstd(grad)
    min_val = np.nanmin(grad)
    max_val = np.nanmax(grad)
    norm = np.linalg.norm(grad.flatten())

    # Check for anomalies
    anomalies = []
    if has_nan:
        anomalies.append(f"NaN detected ({np.isnan(grad).sum()} values)")
    if has_inf:
        anomalies.append(f"Inf detected ({np.isinf(grad).sum()} values)")
    if max_val > 100:
        anomalies.append(f"Gradient explosion (max={max_val:.2e})")
    if norm < 1e-6:
        anomalies.append(f"Gradient vanishing (norm={norm:.2e})")

    # Print results
    status = "ERROR" if anomalies else "OK"
    print(f"  {name:20s} | mean={mean:+.2e} std={std:.2e} min={min_val:+.2e} max={max_val:+.2e} norm={norm:.2e} | {status}")

    if anomalies:
        for anomaly in anomalies:
            print(f"    WARNING: {anomaly}")

    return len(anomalies) == 0


def main():
    parser = argparse.ArgumentParser(description="Debug CNN gradient flow")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint to load (optional)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for testing")
    args = parser.parse_args()

    print("=" * 80)
    print("CNN Gradient Monitoring Tool")
    print("=" * 80)

    # Initialize model and trainer
    print("\nInitializing model...")
    model = DeepSeekOCRModel(num_glyphs=62)
    trainer = GPUCNNTrainer(model, num_classes=62, learning_rate=0.01, momentum=0.9)

    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = np.load(args.checkpoint, allow_pickle=True)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'].item())

    # Create dummy batch
    print(f"\nCreating dummy batch (size={args.batch_size})...")
    batch_size = args.batch_size
    images = [np.random.rand(64, 64, 3).astype(np.float32) for _ in range(batch_size)]
    labels = [np.random.randint(0, 62) for _ in range(batch_size)]

    # Zero gradients
    print("\nZeroing gradients...")
    trainer._zero_gradients()

    # Run one training step
    print("\nRunning forward + backward pass...")
    for i, (img, label) in enumerate(zip(images, labels)):
        logits, probs, cache = trainer.forward(img)
        loss = trainer.accumulate_gradients(img, label, cache)
        print(f"  Sample {i+1}/{batch_size}: loss={loss:.4f}")

    # Scale gradients
    print(f"\nScaling gradients by 1/{batch_size}...")
    trainer._scale_gradients(batch_size)

    # Analyze all gradient buffers
    print("\nAnalyzing gradients:")
    print("-" * 80)

    all_ok = True

    # Conv1 gradients
    all_ok &= analyze_gradients(trainer, "conv1_weight", trainer.d_grad_conv1_w, model.conv1_weight.shape)
    all_ok &= analyze_gradients(trainer, "conv1_bias", trainer.d_grad_conv1_b, model.conv1_bias.shape)

    # BN1 gradients
    all_ok &= analyze_gradients(trainer, "bn1_gamma", trainer.d_grad_bn1_gamma, model.bn1_gamma.shape)
    all_ok &= analyze_gradients(trainer, "bn1_beta", trainer.d_grad_bn1_beta, model.bn1_beta.shape)

    # Conv2 gradients
    all_ok &= analyze_gradients(trainer, "conv2_weight", trainer.d_grad_conv2_w, model.conv2_weight.shape)
    all_ok &= analyze_gradients(trainer, "conv2_bias", trainer.d_grad_conv2_b, model.conv2_bias.shape)

    # BN2 gradients
    all_ok &= analyze_gradients(trainer, "bn2_gamma", trainer.d_grad_bn2_gamma, model.bn2_gamma.shape)
    all_ok &= analyze_gradients(trainer, "bn2_beta", trainer.d_grad_bn2_beta, model.bn2_beta.shape)

    # Conv3 gradients
    all_ok &= analyze_gradients(trainer, "conv3_weight", trainer.d_grad_conv3_w, model.conv3_weight.shape)
    all_ok &= analyze_gradients(trainer, "conv3_bias", trainer.d_grad_conv3_b, model.conv3_bias.shape)

    # BN3 gradients
    all_ok &= analyze_gradients(trainer, "bn3_gamma", trainer.d_grad_bn3_gamma, model.bn3_gamma.shape)
    all_ok &= analyze_gradients(trainer, "bn3_beta", trainer.d_grad_bn3_beta, model.bn3_beta.shape)

    # FC gradients
    all_ok &= analyze_gradients(trainer, "fc_weight", trainer.d_grad_fc_weight, trainer.fc_weight.shape)
    all_ok &= analyze_gradients(trainer, "fc_bias", trainer.d_grad_fc_bias, trainer.fc_bias.shape)

    print("-" * 80)

    if all_ok:
        print("\nSUCCESS: All gradients are healthy!")
        return 0
    else:
        print("\nWARNING: Some gradients have anomalies. Check output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
