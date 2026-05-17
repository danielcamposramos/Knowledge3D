"""
Batch Optimizer for Procedural Drawing Training.

Dynamically adjusts batch size based on GPU utilization to maximize throughput
while staying within VRAM budget.

Usage:
    optimizer = BatchOptimizer(target_utilization=0.80, max_vram_mb=200)
    batch_size = optimizer.suggest_batch_size(current_size=32, gpu_util=0.07, vram_used=108)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class GPUMetrics:
    """GPU utilization metrics."""
    utilization: float  # 0.0 to 1.0
    vram_used_mb: float
    vram_total_mb: float


class BatchOptimizer:
    """
    Dynamically optimize batch size based on GPU utilization.

    Principles:
    - Target 70-80% GPU utilization for maximum throughput
    - Stay within VRAM budget (<200 MB for K3D)
    - Gradual increases (avoid OOM crashes)
    - Conservative scaling (safety first)
    """

    def __init__(
        self,
        target_utilization: float = 0.75,
        max_vram_mb: float = 11500.0,  # Use full 12GB VRAM (leave 500MB safety margin)
        min_batch_size: int = 8,
        max_batch_size: int = 2048,  # Increased for full VRAM scaling
        scale_factor: float = 1.5  # Conservative scaling
    ):
        """
        Initialize batch optimizer.

        Args:
            target_utilization: Target GPU utilization (0.7-0.8 recommended)
            max_vram_mb: Maximum VRAM usage (MB)
            min_batch_size: Minimum batch size
            max_batch_size: Maximum batch size
            scale_factor: Scaling factor for batch size adjustments
        """
        self.target_utilization = target_utilization
        self.max_vram_mb = max_vram_mb
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.scale_factor = scale_factor

        # History tracking
        self.history = []
        self.last_suggestion = None

    def suggest_batch_size(
        self,
        current_batch_size: int,
        gpu_utilization: float,
        vram_used_mb: float
    ) -> int:
        """
        Suggest optimal batch size based on current GPU metrics.

        Args:
            current_batch_size: Current batch size
            gpu_utilization: Current GPU utilization (0.0 to 1.0)
            vram_used_mb: Current VRAM usage (MB)

        Returns:
            Suggested batch size
        """
        # Record metrics
        self.history.append({
            'batch_size': current_batch_size,
            'utilization': gpu_utilization,
            'vram_mb': vram_used_mb
        })

        # Check VRAM headroom
        vram_headroom = self.max_vram_mb - vram_used_mb
        vram_ratio = vram_used_mb / self.max_vram_mb

        # Decision logic
        if vram_ratio > 0.9:
            # Approaching VRAM limit - decrease batch size
            new_batch = max(self.min_batch_size, int(current_batch_size / self.scale_factor))
            reason = "VRAM limit approaching"

        elif gpu_utilization < 0.3:
            # Very low GPU usage - increase aggressively
            # Conservative: 7% → 32% still safe with current 108 MB
            if vram_ratio < 0.6:  # 108 MB / 180 MB = 0.6
                # Plenty of VRAM headroom
                new_batch = min(self.max_batch_size, int(current_batch_size * self.scale_factor))
                reason = "Low GPU utilization + VRAM headroom"
            else:
                new_batch = min(self.max_batch_size, int(current_batch_size * 1.2))
                reason = "Low GPU utilization (conservative)"

        elif gpu_utilization < self.target_utilization - 0.1:
            # Below target - increase moderately
            new_batch = min(self.max_batch_size, int(current_batch_size * 1.2))
            reason = "Below target utilization"

        elif gpu_utilization > self.target_utilization + 0.1:
            # Above target - decrease moderately
            new_batch = max(self.min_batch_size, int(current_batch_size / 1.2))
            reason = "Above target utilization"

        else:
            # Within target range - maintain
            new_batch = current_batch_size
            reason = "Optimal range"

        # Ensure power-of-2 or multiple of 8 (GPU-friendly)
        new_batch = (new_batch // 8) * 8
        new_batch = max(self.min_batch_size, min(self.max_batch_size, new_batch))

        self.last_suggestion = {
            'batch_size': new_batch,
            'reason': reason,
            'vram_headroom_mb': vram_headroom,
            'utilization': gpu_utilization
        }

        return new_batch

    def get_optimization_report(self) -> str:
        """Generate optimization report based on history."""
        if not self.history:
            return "No history available"

        latest = self.history[-1]
        suggestion = self.last_suggestion

        report = []
        report.append("="*60)
        report.append("GPU Batch Optimization Report")
        report.append("="*60)
        report.append(f"\nCurrent State:")
        report.append(f"  Batch size: {latest['batch_size']}")
        report.append(f"  GPU utilization: {latest['utilization']*100:.1f}%")
        report.append(f"  VRAM usage: {latest['vram_mb']:.1f} MB / {self.max_vram_mb:.1f} MB")
        report.append(f"  VRAM headroom: {self.max_vram_mb - latest['vram_mb']:.1f} MB")

        if suggestion:
            report.append(f"\nSuggestion:")
            report.append(f"  New batch size: {suggestion['batch_size']}")
            report.append(f"  Reason: {suggestion['reason']}")
            report.append(f"  Expected VRAM headroom: {suggestion['vram_headroom_mb']:.1f} MB")

        report.append(f"\nOptimization Potential:")
        util_gap = max(0, self.target_utilization - latest['utilization'])
        vram_free_pct = ((self.max_vram_mb - latest['vram_mb']) / self.max_vram_mb) * 100

        if util_gap > 0.3 and vram_free_pct > 40:
            report.append(f"  ⚠️  HIGH: GPU underutilized ({latest['utilization']*100:.1f}%), VRAM available ({vram_free_pct:.1f}%)")
            report.append(f"  → Recommend batch size increase to {suggestion['batch_size']}")
        elif util_gap > 0.1:
            report.append(f"  ⚡ MODERATE: Some GPU headroom available")
        else:
            report.append(f"  ✓ GOOD: Operating near target utilization")

        report.append(f"\nHistory ({len(self.history)} samples):")
        for i, entry in enumerate(self.history[-5:]):  # Last 5 entries
            report.append(f"  {i+1}. Batch={entry['batch_size']}, "
                         f"GPU={entry['utilization']*100:.1f}%, "
                         f"VRAM={entry['vram_mb']:.1f}MB")

        report.append("="*60)
        return "\n".join(report)


def estimate_optimal_batch_size(
    current_batch: int,
    gpu_util: float,
    vram_used_mb: float,
    target_vram_mb: float = 180.0
) -> int:
    """
    Quick estimate of optimal batch size (without optimizer state).

    Args:
        current_batch: Current batch size
        gpu_util: Current GPU utilization (0.0 to 1.0)
        vram_used_mb: Current VRAM usage (MB)
        target_vram_mb: Target maximum VRAM (MB)

    Returns:
        Estimated optimal batch size
    """
    # Simple heuristic: scale linearly with inverse of utilization
    # If GPU at 7% and we want 75%, we can theoretically 10x the load
    # But be conservative due to VRAM constraints

    vram_ratio = vram_used_mb / target_vram_mb  # e.g., 108/180 = 0.6

    if gpu_util < 0.1 and vram_ratio < 0.7:
        # Lots of headroom - can increase significantly
        # Example: 7% GPU, 60% VRAM → can go 3-4x
        scale = min(4.0, 0.7 / (gpu_util + 0.01))  # Avoid div by zero
        scale = min(scale, (1.0 - vram_ratio) / 0.1)  # VRAM constraint
    elif gpu_util < 0.5 and vram_ratio < 0.8:
        # Moderate headroom
        scale = min(2.0, 0.7 / (gpu_util + 0.01))
        scale = min(scale, (1.0 - vram_ratio) / 0.15)
    else:
        # Near limits
        scale = 1.0

    new_batch = int(current_batch * scale)
    # Round to multiple of 8 (GPU-friendly)
    new_batch = (new_batch // 8) * 8
    return max(8, min(256, new_batch))


__all__ = ['BatchOptimizer', 'GPUMetrics', 'estimate_optimal_batch_size']
