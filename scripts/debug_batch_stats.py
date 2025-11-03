#!/usr/bin/env python3
"""
Debug Script: BatchNorm Statistics Monitoring

This script monitors BatchNorm statistics during training to detect:
- Near-zero variance (< 1e-3)
- Exploding variance (> 100)
- Mean drift
- Running statistics drift

Usage:
    python scripts/debug_batch_stats.py --checkpoint /path/to/checkpoint
"""

import argparse
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge3d.cranium.ocr.deepseek_ocr_model import DeepSeekOCRModel


def analyze_bn_stats(model: DeepSeekOCRModel, layer_name: str, gamma, beta, running_mean, running_var):
    """Analyze BatchNorm statistics for anomalies."""
    # Compute statistics
    var_min = np.min(running_var)
    var_max = np.max(running_var)
    var_mean = np.mean(running_var)
    mean_abs = np.mean(np.abs(running_mean))

    # Check for anomalies
    anomalies = []
    if var_min < 1e-3:
        anomalies.append(f"Near-zero variance detected (min={var_min:.2e})")
    if var_max > 100:
        anomalies.append(f"Exploding variance detected (max={var_max:.2e})")
    if mean_abs > 10:
        anomalies.append(f"Large mean drift (mean_abs={mean_abs:.2e})")

    # Gamma/beta checks
    gamma_min, gamma_max = np.min(gamma), np.max(gamma)
    beta_abs = np.mean(np.abs(beta))

    if gamma_min < 0.1:
        anomalies.append(f"Small gamma detected (min={gamma_min:.2e})")
    if gamma_max > 2.0:
        anomalies.append(f"Large gamma detected (max={gamma_max:.2e})")

    # Print results
    status = "ERROR" if anomalies else "OK"
    print(f"\n{layer_name}:")
    print(f"  Running mean: min={np.min(running_mean):+.2e} max={np.max(running_mean):+.2e} abs_mean={mean_abs:.2e}")
    print(f"  Running var:  min={var_min:.2e} max={var_max:.2e} mean={var_mean:.2e}")
    print(f"  Gamma:        min={gamma_min:.2e} max={gamma_max:.2e} mean={np.mean(gamma):.2e}")
    print(f"  Beta:         min={np.min(beta):+.2e} max={np.max(beta):+.2e} abs_mean={beta_abs:.2e}")
    print(f"  Status: {status}")

    if anomalies:
        for anomaly in anomalies:
            print(f"    WARNING: {anomaly}")

    return len(anomalies) == 0


def main():
    parser = argparse.ArgumentParser(description="Debug BatchNorm statistics")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint to load (optional)")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of forward passes to run")
    args = parser.parse_args()

    print("=" * 80)
    print("BatchNorm Statistics Monitoring Tool")
    print("=" * 80)

    # Initialize model
    print("\nInitializing model...")
    model = DeepSeekOCRModel(num_glyphs=62)

    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = np.load(args.checkpoint, allow_pickle=True)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'].item())

    # Run forward passes to accumulate statistics
    print(f"\nRunning {args.num_samples} forward passes to update running statistics...")
    for i in range(args.num_samples):
        img = np.random.rand(64, 64, 3).astype(np.float32)
        result = model.forward(img, cache_for_backward=True)
        print(f"  Pass {i+1}/{args.num_samples}: feature_map shape = {result['feature_map'].shape}")

    # Analyze all BatchNorm layers
    print("\n" + "=" * 80)
    print("BatchNorm Statistics Analysis")
    print("=" * 80)

    all_ok = True

    # BN1
    all_ok &= analyze_bn_stats(
        model, "BatchNorm1",
        model.bn1_gamma, model.bn1_beta,
        model.bn1_running_mean, model.bn1_running_var
    )

    # BN2
    all_ok &= analyze_bn_stats(
        model, "BatchNorm2",
        model.bn2_gamma, model.bn2_beta,
        model.bn2_running_mean, model.bn2_running_var
    )

    # BN3
    all_ok &= analyze_bn_stats(
        model, "BatchNorm3",
        model.bn3_gamma, model.bn3_beta,
        model.bn3_running_mean, model.bn3_running_var
    )

    print("\n" + "=" * 80)

    if all_ok:
        print("\nSUCCESS: All BatchNorm statistics are healthy!")
        return 0
    else:
        print("\nWARNING: Some BatchNorm layers have anomalies. Check output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
