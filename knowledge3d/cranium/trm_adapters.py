"""
TRM Adapters: Self-Updating Low-Rank Specialist Modules

Implements LoRA-style adapters with independent self-updating capability.

Key Features:
- Low-rank decomposition: ΔW = A @ B (memory efficient)
- Shadow weights: Safe testing before committing
- Validation gating: Only accept improvements
- Independent evolution: Each specialist updates separately

Architecture:
    Base Model (W_base) + Adapter (A @ B) = Active Weights

Memory Efficiency:
    Full specialist: 2048×2048 = 16.8M params
    Adapter (rank-64): 2×(2048×64) = 262K params
    Reduction: 64× smaller!

Usage:
    # Create adapter
    adapter = SelfUpdatingAdapter(
        shape=(512, 512),
        rank=64,
        specialist_name='ocr'
    )

    # Train with self-updating
    adapter.fork_to_shadow()
    adapter.apply_gradient_to_shadow(gradient)
    success = adapter.validate_and_commit(base_weights, eval_fn)
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Any
import json
from dataclasses import dataclass


def _to_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to plain Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


@dataclass
class AdapterConfig:
    """Adapter configuration."""
    rank: int = 64                    # Bottleneck dimension
    alpha: float = 1.0                # Scaling factor
    learning_rate: float = 0.001      # Update learning rate
    gradient_clip: float = 1.0        # Gradient clipping threshold
    min_improvement: float = 0.001    # Minimum improvement to commit (0.1%)
    max_degradation: float = 0.05     # Maximum allowed degradation (5%)


class AdapterWeights:
    """
    Low-rank adapter using LoRA-style decomposition.

    Instead of storing full ΔW [D×D], decompose as:
        ΔW = α × (A @ B)
        where A [D×r], B [r×D], r << D

    Memory savings: O(D²) → O(2Dr)

    Example:
        Full: 2048×2048 = 4.2M params (16.8 MB)
        Rank-64: 2×(2048×64) = 262K params (1.05 MB)
        Reduction: 16× smaller
    """

    def __init__(self, shape: Tuple[int, int], rank: int = 64,
                 alpha: float = 1.0, init_std: float = 0.01):
        """
        Initialize low-rank adapter.

        Args:
            shape: Full weight shape [D, D]
            rank: Bottleneck dimension (r)
            alpha: Scaling factor for adapter strength
            init_std: Initialization standard deviation
        """
        if shape[0] != shape[1]:
            raise ValueError(f"Adapter requires square shape, got {shape}")

        self.shape = shape
        self.rank = min(rank, shape[0])  # Ensure rank <= dimension
        self.alpha = alpha

        # Low-rank decomposition: ΔW = A @ B
        self.A = np.random.randn(shape[0], self.rank).astype(np.float32) * init_std
        self.B = np.random.randn(self.rank, shape[1]).astype(np.float32) * init_std

        # Zero-initialize B for stable training (LoRA best practice)
        self.B.fill(0.0)

    def get_delta(self) -> np.ndarray:
        """
        Reconstruct full adapter delta.

        Returns: ΔW = α × (A @ B)  [D×D]
        """
        return self.alpha * (self.A @ self.B)

    def apply_gradient(self, gradient: np.ndarray, lr: float = 0.001):
        """
        Update adapter weights given gradient for full ΔW.

        Uses chain rule to compute gradients for A and B:
            ∂L/∂A = ∂L/∂ΔW @ B.T
            ∂L/∂B = A.T @ ∂L/∂ΔW

        Args:
            gradient: Gradient w.r.t. full ΔW [D×D]
            lr: Learning rate
        """
        if gradient.shape != self.shape:
            raise ValueError(f"Gradient shape {gradient.shape} != adapter shape {self.shape}")

        # Gradient clipping (prevent instability)
        grad_norm = np.linalg.norm(gradient)
        if grad_norm > 1.0:
            gradient = gradient / grad_norm

        # Compute gradients for A and B using chain rule
        grad_A = gradient @ self.B.T  # [D×D] @ [D×r] = [D×r]
        grad_B = self.A.T @ gradient  # [r×D] @ [D×D] = [r×D]

        # Gradient descent
        self.A -= lr * grad_A
        self.B -= lr * grad_B

    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return self.A.size + self.B.size

    def get_memory_mb(self) -> float:
        """Get memory footprint in MB (fp32)."""
        return self.get_num_params() * 4 / (1024**2)

    def save(self, path: Path):
        """Save adapter to disk."""
        np.savez_compressed(
            path,
            A=self.A,
            B=self.B,
            alpha=self.alpha,
            rank=self.rank,
            shape=self.shape
        )

    def load(self, path: Path):
        """Load adapter from disk."""
        data = np.load(path)
        self.A = data['A']
        self.B = data['B']
        self.alpha = float(data['alpha'])
        self.rank = int(data['rank'])
        self.shape = tuple(data['shape'])


class SelfUpdatingAdapter(AdapterWeights):
    """
    Adapter with shadow weights and validation gating.

    Enables safe self-updating:
    1. Fork primary → shadow
    2. Apply gradient to shadow
    3. Validate shadow on holdout set
    4. Commit if improved, reject otherwise

    Prevents catastrophic forgetting through validation gate.
    """

    def __init__(self, shape: Tuple[int, int], rank: int = 64,
                 specialist_name: str = "specialist",
                 config: Optional[AdapterConfig] = None):
        """
        Initialize self-updating adapter.

        Args:
            shape: Weight shape [D, D]
            rank: Bottleneck dimension
            specialist_name: Identifier for this specialist
            config: Adapter configuration
        """
        super().__init__(shape, rank)

        self.specialist_name = specialist_name
        self.config = config or AdapterConfig()

        # Shadow weights (candidate updates)
        self.A_shadow = np.zeros_like(self.A)
        self.B_shadow = np.zeros_like(self.B)

        # Validation tracking
        self.validation_samples = []
        self.baseline_performance = 0.0

        # Update statistics
        self.update_count = 0
        self.accepted_count = 0
        self.rejected_count = 0
        self.performance_history = []

        print(f"[{specialist_name}] Self-updating adapter initialized")
        print(f"  Shape: {shape}, Rank: {rank}")
        print(f"  Parameters: {self.get_num_params()/1e3:.1f}K ({self.get_memory_mb():.2f} MB)")

    def set_validation_samples(self, samples: List[Dict]):
        """Set specialist-specific validation set."""
        self.validation_samples = samples
        print(f"[{self.specialist_name}] Validation set: {len(samples)} samples")

    def fork_to_shadow(self):
        """Copy primary weights → shadow for testing."""
        np.copyto(self.A_shadow, self.A)
        np.copyto(self.B_shadow, self.B)

    def get_delta_shadow(self) -> np.ndarray:
        """Get shadow delta: ΔW_shadow = α × (A_shadow @ B_shadow)"""
        return self.alpha * (self.A_shadow @ self.B_shadow)

    def apply_gradient_to_shadow(self, gradient: np.ndarray,
                                 lr: Optional[float] = None):
        """
        Apply gradient to shadow weights.

        Primary weights unchanged - testing update safely.
        """
        lr = lr or self.config.learning_rate

        # Gradient clipping
        grad_norm = np.linalg.norm(gradient)
        if grad_norm > self.config.gradient_clip:
            gradient = gradient / grad_norm * self.config.gradient_clip

        # Apply to shadow
        grad_A = gradient @ self.B_shadow.T
        grad_B = self.A_shadow.T @ gradient

        self.A_shadow -= lr * grad_A
        self.B_shadow -= lr * grad_B

    def validate_and_commit(self, base_weights: np.ndarray,
                           eval_fn: Callable[[np.ndarray, List], float]) -> Tuple[bool, float, float]:
        """
        Validate shadow adapter and commit if performance improves.

        Args:
            base_weights: Current base model weights [D×D]
            eval_fn: Function that evaluates (weights, samples) → performance

        Returns:
            (success, baseline_perf, shadow_perf)
        """
        if len(self.validation_samples) == 0:
            print(f"[{self.specialist_name}] Warning: No validation samples, skipping validation")
            return False, 0.0, 0.0

        # Evaluate baseline (primary adapter + base)
        W_baseline = base_weights + self.get_delta()
        baseline_perf = eval_fn(W_baseline, self.validation_samples)

        # Evaluate shadow (shadow adapter + base)
        W_shadow = base_weights + self.get_delta_shadow()
        shadow_perf = eval_fn(W_shadow, self.validation_samples)

        # Decision criteria
        improvement = shadow_perf - baseline_perf
        degradation = baseline_perf - shadow_perf

        # Check for improvement
        if improvement >= self.config.min_improvement:
            # Performance improved → commit shadow → primary
            np.copyto(self.A, self.A_shadow)
            np.copyto(self.B, self.B_shadow)

            self.baseline_performance = shadow_perf
            self.accepted_count += 1

            # Record success
            self.performance_history.append({
                'step': self.update_count,
                'baseline': baseline_perf,
                'shadow': shadow_perf,
                'improvement': improvement,
                'accepted': True
            })

            print(f"[{self.specialist_name}] ✓ Update accepted: "
                  f"{baseline_perf:.4f} → {shadow_perf:.4f} (+{improvement:.4f})")

            self.update_count += 1
            return True, baseline_perf, shadow_perf

        elif degradation > self.config.max_degradation:
            # Excessive degradation → reject
            self.rejected_count += 1

            self.performance_history.append({
                'step': self.update_count,
                'baseline': baseline_perf,
                'shadow': shadow_perf,
                'degradation': degradation,
                'accepted': False,
                'reason': 'excessive_degradation'
            })

            print(f"[{self.specialist_name}] ✗ Update rejected: "
                  f"{baseline_perf:.4f} → {shadow_perf:.4f} (-{degradation:.4f}) "
                  f"- Excessive degradation")

            self.update_count += 1
            return False, baseline_perf, shadow_perf

        else:
            # Insufficient improvement → reject
            self.rejected_count += 1

            self.performance_history.append({
                'step': self.update_count,
                'baseline': baseline_perf,
                'shadow': shadow_perf,
                'improvement': improvement,
                'accepted': False,
                'reason': 'insufficient_improvement'
            })

            print(f"[{self.specialist_name}] ✗ Update rejected: "
                  f"{baseline_perf:.4f} → {shadow_perf:.4f} (+{improvement:.4f}) "
                  f"- Insufficient improvement")

            self.update_count += 1
            return False, baseline_perf, shadow_perf

    def get_stats(self) -> Dict[str, Any]:
        """Get adapter statistics."""
        acceptance_rate = self.accepted_count / max(self.update_count, 1)

        return {
            'specialist_name': self.specialist_name,
            'shape': self.shape,
            'rank': self.rank,
            'params': self.get_num_params(),
            'memory_mb': self.get_memory_mb(),
            'update_count': self.update_count,
            'accepted': self.accepted_count,
            'rejected': self.rejected_count,
            'acceptance_rate': acceptance_rate,
            'baseline_performance': self.baseline_performance,
            'recent_history': self.performance_history[-10:]
        }

    def save_checkpoint(self, checkpoint_dir: Path):
        """Save adapter checkpoint with metadata."""
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save weights
        self.save(checkpoint_dir / f'{self.specialist_name}_adapter.npz')

        # Save metadata
        metadata = {
            'config': {
                'rank': self.config.rank,
                'alpha': self.config.alpha,
                'learning_rate': self.config.learning_rate
            },
            'stats': self.get_stats(),
            'performance_history': self.performance_history
        }

        serializable_metadata = _to_serializable(metadata)

        with open(checkpoint_dir / f'{self.specialist_name}_metadata.json', 'w') as f:
            json.dump(serializable_metadata, f, indent=2)

        print(f"[{self.specialist_name}] Checkpoint saved to {checkpoint_dir}")

    def load_checkpoint(self, checkpoint_dir: Path):
        """Load adapter checkpoint."""
        # Load weights
        self.load(checkpoint_dir / f'{self.specialist_name}_adapter.npz')

        # Load metadata
        with open(checkpoint_dir / f'{self.specialist_name}_metadata.json', 'r') as f:
            metadata = json.load(f)

        self.performance_history = metadata.get('performance_history', [])
        stats = metadata.get('stats', {})
        self.baseline_performance = stats.get('baseline_performance', 0.0)

        print(f"[{self.specialist_name}] Checkpoint loaded from {checkpoint_dir}")
        print(f"  Baseline performance: {self.baseline_performance:.4f}")
        print(f"  Acceptance rate: {stats.get('acceptance_rate', 0)*100:.1f}%")
